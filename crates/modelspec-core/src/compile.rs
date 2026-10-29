//! The compiler driver: `.mspec` source to Typed HIR, to verified Tensor IR and to a verified
//! training step (DESIGN.md § 3, § 3.1, § 3.3 passes 1–5).
//!
//! [`compile_hir`] parses, resolves and checks; [`compile_tensor`] then lowers, removes dead code
//! and verifies; [`compile_training`] then differentiates the loss, removes every `stop_gradient`,
//! attaches SGD and verifies the step, legal effects included. A source fault is reported as the
//! diagnostics of the stage that found it. A program the verifier refuses after lowering, a
//! lowering failure on checked HIR, or an autodiff or SGD failure on verified Tensor IR is a
//! compiler bug and is reported under [`codes::COMPILER_BUG`], never as a user code (DESIGN.md
//! § 8.1).

use crate::artifact::{Backend, CompiledArtifact, E_ARTIFACT_005};
use crate::autodiff::differentiate;
use crate::device::DeviceProfile;
use crate::diag::Diagnostic;
use crate::hir::{DataRef, DataRole, Hir, HirData, HirExpr, HirExprKind, HirTrain};
use crate::ir::kernel::MatMulLimits;
use crate::ir::tensor::{TensorOp, TensorProgram, TensorType, ValueId};
use crate::ir::training::{OptimizerRule, TrainConfig, TrainingStep, attach};
use crate::ir::verify::{verify, verify_plan_for, verify_step};
use crate::lower::{self, LoweredProgram};
use crate::passes::{dce, masked_rows, stop_gradient};
use crate::resolve::OPERATORS;
use crate::schedule::{self, ScheduleError};

/// Compiler-bug diagnostic codes.
pub mod codes {
    use crate::diag::Code;

    /// A compiler verification failure in the tensor or training pipeline, not a user error
    /// (DESIGN.md § 8.1): lowering refused checked HIR, the program its passes produced does not
    /// verify, or autodiff, optimizer attachment or the step's verification refused verified
    /// Tensor IR.
    /// No source file is meant to reach it.
    pub const COMPILER_BUG: Code = Code::new("E-INTERNAL-002");

    /// A tensor of the training step is too large for the Vulkan target of M1: one dispatch over
    /// it needs more than 65,535 workgroups in each of three dimensions or more than 2^32
    /// invocations, its buffer holds more than 4,294,967,295 bytes (the `u32` range of a storage
    /// buffer binding), or its byte count does not fit in 64 bits (DESIGN.md § 8.1). The file
    /// checks; reduce the shape.
    pub const TOO_LARGE_FOR_TARGET: Code = Code::new("E-TARGET-010");

    /// An operator of the training step has no Vulkan kernel for its operands, such as an
    /// `attention` over more than 512 keys per sequence, or the optimizer `muon`, whose
    /// step runs on the CPU interpreter only and is reported at the `train` block; and, refused by
    /// `infer --backend vulkan` rather than the schedule, `dropout`, whose inference runs the
    /// step plan with its dropouts (DESIGN.md § 8.1). The file checks; use `--backend cpu`.
    pub const NO_KERNEL: Code = Code::new("E-TARGET-011");
}

/// A `.mspec` file compiled for training.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledTraining {
    /// The verified training step: forward, loss, backward and the SGD subgraph, with its effect
    /// tokens and state versions.
    pub training: TrainingStep,
    /// The Tensor IR before autodiff, with its `prediction` value, every dropout the identity
    /// ([`lower::without_dropout`]) and every `stop_gradient` removed as dropout is, each use
    /// reading its operand ([`stop_gradient::eliminate`]): what the data loader, the final loss
    /// line, evaluation and inference run. Its node ids are those of the training step.
    pub forward: LoweredProgram,
    /// The `train` block's learning rate, steps, seed and data declarations; its optimizer rule
    /// is the step's.
    pub config: TrainConfig,
}

/// [`compile_tensor`], then [`differentiate`] of its loss, the removal of every `StopGradient`
/// ([`stop_gradient::eliminate_training`]), [`attach`] of the `train` block's optimizer, `sgd`,
/// `adamw` or `muon`, and [`verify_step`]. The step keeps every dropout; the forward program is
/// the same Tensor IR with each dropout the identity and no `StopGradient`, verified.
pub fn compile_training(path: &str, source: &str) -> Result<CompiledTraining, Vec<Diagnostic>> {
    compile_training_with(path, source, None)
}

/// [`compile_training`] whose step, with `masked_rows` a capacity, computes its weighted
/// `cross_entropy` loss on that many gathered rows ([`masked_rows::gather`], DESIGN.md § 3.3):
/// the step reads the inputs of [`masked_rows::site`], which the host fills at every step. The
/// forward program is [`compile_training`]'s. A model whose loss has no such site, or a
/// capacity of at least its row count, gives [`compile_training`]'s step.
pub fn compile_training_with(
    path: &str,
    source: &str,
    masked_rows: Option<u64>,
) -> Result<CompiledTraining, Vec<Diagnostic>> {
    let hir = compile_hir(path, source)?;
    let config = TrainConfig::from_hir(&hir.train);
    let lowered = tensor_stage(&hir, dce::eliminate)?;
    let differentiated = match masked_rows {
        Some(capacity) => masked_rows::gather(lowered.program.clone(), capacity),
        None => lowered.program.clone(),
    };
    if let Err(error) = verify(&differentiated) {
        return Err(vec![compiler_bug(
            "the Tensor IR with the masked-row gather does not verify",
            &error,
        )]);
    }
    let training = training_stage(&differentiated, config.lr, hir.train.optimizer)?;
    let forward = without_stop_gradient(lower::without_dropout(lowered));
    if let Err(error) = verify(&forward.program) {
        return Err(vec![compiler_bug(
            "the Tensor IR without dropout does not verify",
            &error,
        )]);
    }
    Ok(CompiledTraining {
        training,
        forward,
        config,
    })
}

/// [`compile_training`], then the fixed safe schedule ([`schedule::schedule`]) and
/// [`verify_plan`]: Kernel IR, the step plan and the forward plan, in an artifact for `backend`.
/// A Cpu artifact keeps the views and has no kernels and empty plans: it takes the views of
/// [`schedule::schedule_views`], so an op with no Vulkan kernel does not refuse it. For Vulkan,
/// such an op is [`codes::NO_KERNEL`] at the op's source span, and a `muon` step is
/// [`codes::NO_KERNEL`] at the `train` block before anything is scheduled.
pub fn compile_artifact(
    path: &str,
    source: &str,
    backend: Backend,
) -> Result<CompiledArtifact, Vec<Diagnostic>> {
    compile_artifact_for(path, source, backend, None)
}

/// [`compile_artifact`] for the device `profile` describes: the plan is scheduled within its
/// limits ([`schedule::schedule_for`] with [`MatMulLimits::of`] the profile), verified against
/// it ([`verify_plan_for`]), and the artifact records those limits
/// ([`CompiledArtifact::matmul_limits`]). A profile whose limits are below the Vulkan minimum
/// the smallest tile needs ([`MatMulLimits::checked`]) describes a device no plan runs on: it
/// is refused with [`E_ARTIFACT_005`] naming the field, its bound and the reported value, so no
/// artifact records such limits. Without a profile it is [`compile_artifact`].
pub fn compile_artifact_for(
    path: &str,
    source: &str,
    backend: Backend,
    profile: Option<&DeviceProfile>,
) -> Result<CompiledArtifact, Vec<Diagnostic>> {
    compile_artifact_with(path, source, backend, profile, None)
}

/// [`compile_artifact_for`] of [`compile_training_with`] at `masked_rows`.
pub fn compile_artifact_with(
    path: &str,
    source: &str,
    backend: Backend,
    profile: Option<&DeviceProfile>,
    masked_rows: Option<u64>,
) -> Result<CompiledArtifact, Vec<Diagnostic>> {
    let compiled = compile_training_with(path, source, masked_rows)?;
    artifact_of(&compiled, source, backend, profile)
}

/// Why [`compile_function`] compiled no function.
#[derive(Clone, Debug, PartialEq)]
pub enum FunctionError {
    /// The file does not compile: its diagnostics, as [`compile_hir`] reports them.
    Diagnostics(Vec<Diagnostic>),
    /// The model declares no function `name`; `functions` names the ones it declares, in source
    /// order.
    Unknown {
        name: String,
        functions: Vec<String>,
    },
}

/// The model function `name` of the file compiled as an inference entry
/// (story:decider-infer-on-data): a [`CompiledTraining`] whose `train` block reads one `input`
/// per argument of the function, named and typed as the argument and at its span, whose loss
/// is the function's result itself when it is rank 0 and its `mean` otherwise, and whose
/// prediction ([`LoweredProgram::prediction`]) is the function's result. The function's body
/// is lowered as the training loss inlines it, with every dropout the identity
/// ([`lower::without_dropout`]), into the same parameter table as the file's `train` block, so
/// a checkpoint of the file supplies its parameters; the step differentiates that loss and
/// attaches `sgd` at the block's learning rate whatever the block's optimizer, so a Vulkan plan
/// can be scheduled for it, and nothing ever runs its update. The configuration keeps the
/// block's seed and steps. The function need not be `forward` and need not return a loss.
pub fn compile_function(
    path: &str,
    source: &str,
    name: &str,
) -> Result<CompiledTraining, FunctionError> {
    let hir = compile_hir(path, source).map_err(FunctionError::Diagnostics)?;
    let Some(entry) = function_entry(&hir, name) else {
        return Err(FunctionError::Unknown {
            name: name.to_owned(),
            functions: hir.model.functions.iter().map(|f| f.name.clone()).collect(),
        });
    };
    let config = TrainConfig::from_hir(&entry.train);
    let lowered = lower::lower_predicting(&entry, name)
        .map_err(|error| vec![compiler_bug("lowering refused checked HIR", &error)])
        .and_then(|lowered| {
            let forward = lower::without_dropout(lowered);
            match verify(&forward.program) {
                Ok(()) => Ok(forward),
                Err(error) => Err(vec![compiler_bug(
                    "the Tensor IR of the function does not verify",
                    &error,
                )]),
            }
        })
        .map_err(FunctionError::Diagnostics)?;
    if lowered.program.node(lowered.prediction).is_none() {
        return Err(FunctionError::Diagnostics(vec![compiler_bug(
            "the lowered Tensor IR lost its prediction",
            &format!("prediction {} is not defined", lowered.prediction),
        )]));
    }
    let training = training_stage(&lowered.program, config.lr, OptimizerRule::Sgd)
        .map_err(FunctionError::Diagnostics)?;
    Ok(CompiledTraining {
        training,
        forward: lowered,
        config,
    })
}

/// [`compile_function`], then the artifact of its step for `backend`, as
/// [`compile_artifact`] makes one for the file's `train` block.
pub fn compile_function_artifact(
    path: &str,
    source: &str,
    name: &str,
    backend: Backend,
) -> Result<CompiledArtifact, FunctionError> {
    let compiled = compile_function(path, source, name)?;
    artifact_of(&compiled, source, backend, None).map_err(FunctionError::Diagnostics)
}

/// The checked `hir` with its `train` block replaced by the inference entry of the model
/// function `name` ([`compile_function`]); `None` when the model declares no such function.
fn function_entry(hir: &Hir, name: &str) -> Option<Hir> {
    let (reference, function) = hir.model.function(name)?;
    let data: Vec<HirData> = function
        .args
        .iter()
        .map(|arg| HirData {
            role: DataRole::Input,
            name: arg.name.clone(),
            ty: arg.ty.clone(),
            span: arg.span.clone(),
        })
        .collect();
    let call = HirExpr {
        kind: HirExprKind::Call {
            function: reference,
            args: data
                .iter()
                .enumerate()
                .map(|(at, decl)| HirExpr {
                    kind: HirExprKind::Data(DataRef(at)),
                    ty: decl.ty.clone(),
                    span: decl.span.clone(),
                })
                .collect(),
        },
        ty: function.output.clone(),
        span: function.span.clone(),
    };
    let loss = if function.output.rank() == 0 {
        call
    } else {
        HirExpr {
            kind: HirExprKind::Op {
                op: TensorOp::ReduceMean,
                args: vec![call],
            },
            ty: TensorType::f32(&[]),
            span: function.span.clone(),
        }
    };
    let mut entry = hir.clone();
    entry.train = HirTrain {
        data,
        loss,
        optimizer: OptimizerRule::Sgd,
        span: function.span.clone(),
        ..hir.train.clone()
    };
    Some(entry)
}

/// The artifact of `compiled`, compiled from `source`, for `backend`, scheduled within the
/// limits of `profile` when there is one ([`compile_artifact_for`]).
fn artifact_of(
    compiled: &CompiledTraining,
    source: &str,
    backend: Backend,
    profile: Option<&DeviceProfile>,
) -> Result<CompiledArtifact, Vec<Diagnostic>> {
    let unsupported = match compiled.training.optimizer.rule {
        OptimizerRule::Sgd | OptimizerRule::AdamW(_) => None,
        OptimizerRule::Muon(_) => Some("muon"),
    };
    if backend == Backend::Vulkan
        && let Some(name) = unsupported
    {
        return Err(vec![
            Diagnostic::new(
                codes::NO_KERNEL,
                format!("optimizer `{name}` has no Vulkan kernel yet"),
            )
            .note("compile and train it with `--backend cpu`")
            .at(compiled.config.span.clone()),
        ]);
    }
    let limits = profile
        .map(|profile| MatMulLimits::of(profile).checked())
        .transpose()
        .map_err(|below| {
            vec![
                Diagnostic::new(
                    E_ARTIFACT_005,
                    "the device profile reports less than the Vulkan minimum the smallest matmul \
                     tile needs, so no plan runs on it",
                )
                .with("field", below.field)
                .with("bound", below.bound.to_string())
                .with("reported", below.value.to_string()),
            ]
        })?;
    let plan = match backend {
        Backend::Cpu => schedule::schedule_views(
            &compiled.training,
            compiled.forward.prediction,
            &compiled.config,
        ),
        Backend::Vulkan => schedule::schedule_for(
            &compiled.training,
            compiled.forward.prediction,
            &compiled.config,
            limits,
        ),
    }
    .map_err(|error| {
        if error.is_target_limit() {
            vec![
                Diagnostic::new(
                    codes::TOO_LARGE_FOR_TARGET,
                    "a tensor is too large for the vulkan target",
                )
                .with("error", error.to_string()),
            ]
        } else if let ScheduleError::NoKernel { value, op } = error {
            vec![no_kernel(&compiled.training.training.program, value, op)]
        } else {
            vec![compiler_bug(
                "the schedule refused the training step",
                &error,
            )]
        }
    })?;
    verify_plan_for(&plan, profile)
        .map_err(|error| vec![compiler_bug("the execution plan does not verify", &error)])?;
    let mut artifact = CompiledArtifact::new(backend, source, &compiled.training, plan);
    artifact.matmul_limits = limits;
    Ok(artifact)
}

/// `lowered` with every `StopGradient` replaced by its operand ([`stop_gradient::eliminate`]),
/// the prediction too, so the forward program names the values the training step names.
fn without_stop_gradient(lowered: LoweredProgram) -> LoweredProgram {
    let (program, replaced) = stop_gradient::eliminate(lowered.program);
    LoweredProgram {
        program,
        prediction: replaced.of(lowered.prediction),
        declared_inputs: lowered.declared_inputs,
    }
}

/// Differentiates the single output of `forward`, its loss, removes every `StopGradient`
/// ([`stop_gradient::eliminate_training`]), attaches the optimizer `rule` at `lr` and verifies
/// the step.
fn training_stage(
    forward: &TensorProgram,
    lr: f32,
    rule: OptimizerRule,
) -> Result<TrainingStep, Vec<Diagnostic>> {
    let [loss] = forward.outputs[..] else {
        return Err(vec![compiler_bug(
            "the lowered Tensor IR does not have the loss as its one output",
            &format!("outputs: {}", forward.outputs.len()),
        )]);
    };
    let training = differentiate(forward, loss)
        .map_err(|error| vec![compiler_bug("autodiff refused verified Tensor IR", &error)])?;
    let training = stop_gradient::eliminate_training(training);
    let step = attach(training, lr, rule).map_err(|error| {
        vec![compiler_bug(
            "optimizer attachment refused the training program",
            &error,
        )]
    })?;
    verify_step(&step)
        .map_err(|error| vec![compiler_bug("the training step does not verify", &error)])?;
    Ok(step)
}

/// Parses, resolves and checks one source file; `path` names it in diagnostic spans. Returns
/// the Typed HIR, or every diagnostic in source order.
pub fn compile_hir(path: &str, source: &str) -> Result<Hir, Vec<Diagnostic>> {
    crate::types::check_source(path, source)
}

/// [`compile_hir`], then lowering ([`lower::lower`]), dead-code elimination ([`dce::eliminate`])
/// and verification ([`verify`]).
pub fn compile_tensor(path: &str, source: &str) -> Result<LoweredProgram, Vec<Diagnostic>> {
    let hir = compile_hir(path, source)?;
    tensor_stage(&hir, dce::eliminate)
}

/// Lowers `hir`, runs `passes` over the program, then verifies the result and that the
/// prediction is still defined.
fn tensor_stage(
    hir: &Hir,
    passes: fn(TensorProgram) -> TensorProgram,
) -> Result<LoweredProgram, Vec<Diagnostic>> {
    let lowered = lower::lower(hir)
        .map_err(|error| vec![compiler_bug("lowering refused checked HIR", &error)])?;
    let program = passes(lowered.program);
    if let Err(error) = verify(&program) {
        return Err(vec![compiler_bug(
            "the lowered Tensor IR does not verify",
            &error,
        )]);
    }
    if program.node(lowered.prediction).is_none() {
        return Err(vec![compiler_bug(
            "the lowered Tensor IR lost its prediction",
            &format!("prediction {} is not defined", lowered.prediction),
        )]);
    }
    Ok(LoweredProgram {
        program,
        prediction: lowered.prediction,
        declared_inputs: lowered.declared_inputs,
    })
}

/// [`codes::NO_KERNEL`] for the node `value` of `program`, whose op `op` has no Vulkan kernel:
/// named by its source operator, at the node's source span when it has one.
fn no_kernel(program: &TensorProgram, value: ValueId, op: &str) -> Diagnostic {
    let operator = OPERATORS
        .iter()
        .find(|(_, registered)| registered.name() == op)
        .map_or_else(
            || format!("the internal op `{op}`"),
            |(name, _)| format!("operator `{name}`"),
        );
    let diagnostic = Diagnostic::new(
        codes::NO_KERNEL,
        format!("{operator} has no Vulkan kernel yet"),
    )
    .note("compile and train it with `--backend cpu`");
    match program
        .node(value)
        .and_then(|node| node.source_span.clone())
    {
        Some(span) => diagnostic.at(span),
        None => diagnostic,
    }
}

fn compiler_bug(what: &str, error: &dyn std::fmt::Display) -> Diagnostic {
    Diagnostic::new(codes::COMPILER_BUG, format!("compiler bug: {what}"))
        .with("error", error.to_string())
        .note("this is a fault in modelspec, not in the specification")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::tensor::{Dim, TensorOp, ValueId};

    const EXAMPLE: &str = include_str!("../../../examples/linear_regression.mspec");

    fn example_hir() -> Hir {
        compile_hir("linear_regression.mspec", EXAMPLE).expect("the example checks")
    }

    fn the_compiler_bug(result: Result<LoweredProgram, Vec<Diagnostic>>) -> Diagnostic {
        let diagnostics = result.expect_err("the stage refuses the program");
        assert_eq!(diagnostics.len(), 1, "{diagnostics:#?}");
        let diagnostic = diagnostics.into_iter().next().unwrap();
        assert_eq!(diagnostic.code, codes::COMPILER_BUG, "{diagnostic}");
        assert!(
            !crate::types::codes::ALL.contains(&diagnostic.code),
            "a compiler bug is not a user code"
        );
        assert!(
            diagnostic.message.starts_with("compiler bug: "),
            "{diagnostic}"
        );
        diagnostic
    }

    fn the_training_bug(result: Result<TrainingStep, Vec<Diagnostic>>) -> Diagnostic {
        the_compiler_bug(result.map(|_| example_lowered()))
    }

    fn example_lowered() -> LoweredProgram {
        tensor_stage(&example_hir(), crate::passes::dce::eliminate).expect("the example lowers")
    }

    #[test]
    fn compile_training_returns_the_verified_step_the_forward_program_and_the_config() {
        let compiled = compile_training("linear_regression.mspec", EXAMPLE).expect("it compiles");
        verify_step(&compiled.training).expect("the step verifies");
        assert_eq!(compiled.forward, example_lowered());
        assert_eq!(compiled.config, TrainConfig::from_hir(&example_hir().train));
        assert_eq!(compiled.training.optimizer.lr, 0.1);
        assert_eq!(compiled.training.optimizer.updates.len(), 1);
        assert_eq!(
            compiled.training.training.loss,
            compiled.forward.program.outputs[0]
        );
    }

    #[test]
    fn compile_training_reports_source_faults_as_the_checker_does() {
        let broken = EXAMPLE.replace("matmul(x, w)", "matmul(w, x)");
        let checked = compile_hir("f.mspec", &broken).expect_err("the checker refuses it");
        let trained = compile_training("f.mspec", &broken).expect_err("so does training");
        assert_eq!(trained, checked);
    }

    #[test]
    fn a_forward_program_without_a_single_output_is_a_compiler_bug() {
        let mut program = example_lowered().program;
        program.outputs.push(program.outputs[0]);
        let diagnostic = the_training_bug(training_stage(&program, 0.1, OptimizerRule::Sgd));
        assert!(
            diagnostic.to_string().contains("outputs: 2"),
            "{diagnostic}"
        );
    }

    #[test]
    fn an_autodiff_failure_is_a_compiler_bug() {
        let mut program = example_lowered().program;
        program.outputs = vec![program.inputs[0]];
        let diagnostic = the_training_bug(training_stage(&program, 0.1, OptimizerRule::Sgd));
        assert!(
            diagnostic.message.contains("autodiff refused"),
            "{diagnostic}"
        );
    }

    #[test]
    fn an_sgd_attachment_failure_is_a_compiler_bug() {
        let program = example_lowered().program;
        let diagnostic = the_training_bug(training_stage(&program, -0.1, OptimizerRule::Sgd));
        assert!(
            diagnostic.to_string().contains("learning rate -0.1"),
            "{diagnostic}"
        );
    }

    /// The example with `adamw(...)`: the step carries the rule, the moments and the step
    /// inputs, and its Vulkan artifact dispatches the update as kernels (coordinator decision
    /// D73).
    #[test]
    fn an_adamw_file_compiles_its_step_for_the_cpu_and_for_vulkan() {
        let source = EXAMPLE.replace(
            "sgd(lr = 0.1)",
            "adamw(lr = 0.1, weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, \
             warmup_steps = 10)",
        );
        let compiled = compile_training("adamw.mspec", &source).expect("it compiles");
        let OptimizerRule::AdamW(adamw) = compiled.training.optimizer.rule else {
            panic!("{:?}", compiled.training.optimizer.rule);
        };
        assert_eq!(adamw.warmup_steps, 10);
        assert_eq!(compiled.training.optimizer.moments.len(), 1);
        assert!(compiled.training.optimizer.step_inputs.is_some());
        let artifact = compile_artifact("adamw.mspec", &source, Backend::Vulkan)
            .expect("the AdamW step has Vulkan kernels");
        let ops: Vec<&str> = artifact
            .execution
            .kernels
            .iter()
            .map(|kernel| kernel.op)
            .collect();
        assert!(ops.contains(&"Sqrt") && ops.contains(&"Div"), "{ops:?}");
    }

    #[test]
    fn the_tensor_stage_with_dce_accepts_the_example() {
        let lowered = tensor_stage(&example_hir(), crate::passes::dce::eliminate)
            .expect("the example lowers");
        assert_eq!(lowered.program.nodes.len(), 5);
    }

    #[test]
    fn a_program_that_fails_verification_after_its_passes_is_reported_as_a_compiler_bug() {
        let diagnostic = the_compiler_bug(tensor_stage(&example_hir(), |mut program| {
            program.outputs.push(ValueId(99));
            program
        }));
        assert!(
            diagnostic.to_string().contains("output %99 is not defined"),
            "{diagnostic}"
        );
    }

    #[test]
    fn a_pass_that_removes_the_prediction_is_reported_as_a_compiler_bug() {
        let diagnostic = the_compiler_bug(tensor_stage(&example_hir(), |mut program| {
            program
                .nodes
                .retain(|n| !matches!(n.op, TensorOp::MatMul | TensorOp::Mse));
            program.outputs = vec![program.inputs[1]];
            program
        }));
        assert!(
            diagnostic.to_string().contains("prediction"),
            "{diagnostic}"
        );
    }

    #[test]
    fn a_lowering_failure_on_checked_hir_is_reported_as_a_compiler_bug() {
        let mut hir = example_hir();
        hir.model.params[0].ty.shape[0] = Dim::Static(3);
        let diagnostic = the_compiler_bug(tensor_stage(&hir, crate::passes::dce::eliminate));
        assert!(
            diagnostic
                .to_string()
                .contains("matmul contracting dimension mismatch"),
            "{diagnostic}"
        );
    }

    #[test]
    fn checked_hir_whose_loss_does_not_call_forward_is_a_compiler_bug() {
        let mut hir = example_hir();
        let crate::hir::HirExprKind::Op { args, .. } = &mut hir.train.loss.kind else {
            panic!("the example's loss is mse(...)");
        };
        args[0] = args[1].clone();
        let diagnostic = the_compiler_bug(tensor_stage(&hir, crate::passes::dce::eliminate));
        assert!(
            diagnostic
                .to_string()
                .contains("calls `model.forward` 0 time(s), not once"),
            "{diagnostic}"
        );
    }

    #[test]
    fn checked_hir_whose_loss_calls_forward_twice_is_a_compiler_bug() {
        let mut hir = example_hir();
        let once = hir.train.loss.clone();
        hir.train.loss = crate::hir::HirExpr {
            kind: crate::hir::HirExprKind::Op {
                op: TensorOp::Add,
                args: vec![once.clone(), once.clone()],
            },
            ty: once.ty.clone(),
            span: once.span.clone(),
        };
        let diagnostic = the_compiler_bug(tensor_stage(&hir, crate::passes::dce::eliminate));
        assert!(
            diagnostic
                .to_string()
                .contains("calls `model.forward` 2 time(s), not once"),
            "{diagnostic}"
        );
    }

    /// `forward` returns the loss; `scores` and `bias` read the parameters, `plain` none, and
    /// `noisy` has a dropout.
    const FUNCTIONS: &str = "model Two {
    param w: Tensor<f32, [3, 2]> = xavier_uniform();
    param b: Tensor<f32, [2]> = zeros();

    fn forward(x: Tensor<f32, [4, 3]>, y: Tensor<f32, []>) -> Tensor<f32, []> {
        let e = sub(add_rows(matmul(x, w), b), y);
        return mean(mul(e, e));
    }

    fn scores(x: Tensor<f32, [4, 3]>) -> Tensor<f32, [4, 2]> {
        return matmul(x, w);
    }

    fn plain(x: Tensor<f32, [4, 3]>, ids: Tensor<i32, [4]>) -> Tensor<f32, [4, 3]> {
        return embedding(x, ids);
    }

    fn noisy(x: Tensor<f32, [4, 3]>) -> Tensor<f32, [4, 2]> {
        return dropout(matmul(x, w), p = 0.5);
    }
}

train Fit {
    model = Two;
    input x: Tensor<f32, [4, 3]>;
    target y: Tensor<f32, []>;
    loss = model.forward(x, y);
    optimizer = adamw(lr = 0.1, weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, warmup_steps = 1);
    steps = 3;
    seed = 5;
    device = cpu;
}
";

    fn input_names(program: &TensorProgram) -> Vec<&str> {
        program
            .nodes
            .iter()
            .filter_map(|n| match &n.op {
                TensorOp::Input { name } => Some(name.as_str()),
                _ => None,
            })
            .collect()
    }

    /// A function other than `forward` compiles as an entry: one input per argument, the
    /// function's result the prediction, its mean the loss, `sgd` whatever the block's
    /// optimizer, and the file's parameter table.
    #[test]
    fn a_named_function_compiles_with_its_arguments_as_inputs_and_its_result_predicted() {
        let file = compile_training("two.mspec", FUNCTIONS).expect("the file compiles");
        let compiled = compile_function("two.mspec", FUNCTIONS, "scores").expect("it compiles");
        let forward = &compiled.forward;
        assert_eq!(input_names(&forward.program), ["x"]);
        let prediction = forward.program.node(forward.prediction).unwrap();
        assert_eq!(prediction.op, TensorOp::MatMul);
        assert_eq!(prediction.ty, TensorType::f32(&[4, 2]));
        let loss = forward.program.node(forward.program.outputs[0]).unwrap();
        assert_eq!(loss.op, TensorOp::ReduceMean);
        assert_eq!(loss.operands, [forward.prediction]);
        assert_eq!(compiled.training.optimizer.rule, OptimizerRule::Sgd);
        assert_eq!(compiled.config.inputs.len(), 1);
        assert!(compiled.config.targets.is_empty());
        assert_eq!(compiled.config.seed, file.config.seed);
        assert_eq!(forward.program.params, file.forward.program.params);
        compile_function_artifact("two.mspec", FUNCTIONS, "scores", Backend::Vulkan)
            .expect("its Vulkan artifact compiles");
    }

    /// `forward` itself is an entry too, its rank-0 result the loss and the prediction.
    #[test]
    fn forward_compiles_as_an_entry_with_its_result_as_the_loss() {
        let compiled = compile_function("two.mspec", FUNCTIONS, "forward").expect("it compiles");
        let forward = &compiled.forward;
        assert_eq!(input_names(&forward.program), ["x", "y"]);
        assert_eq!(forward.program.outputs, [forward.prediction]);
    }

    /// A function that reads no parameter, with an `i32` argument, compiles; a dropout in the
    /// function is the identity.
    #[test]
    fn a_function_without_parameters_or_with_dropout_compiles() {
        let plain = compile_function("two.mspec", FUNCTIONS, "plain").expect("it compiles");
        assert_eq!(input_names(&plain.forward.program), ["x", "ids"]);
        let noisy = compile_function("two.mspec", FUNCTIONS, "noisy").expect("it compiles");
        assert!(
            noisy
                .training
                .training
                .program
                .nodes
                .iter()
                .all(|n| !matches!(n.op, TensorOp::Dropout { .. }))
        );
        assert_eq!(input_names(&noisy.forward.program), ["x"]);
    }

    /// A name that is no function of the model names the ones it declares.
    #[test]
    fn an_unknown_function_names_the_functions_of_the_model() {
        let error = compile_function("two.mspec", FUNCTIONS, "logits").unwrap_err();
        assert_eq!(
            error,
            FunctionError::Unknown {
                name: "logits".into(),
                functions: vec![
                    "forward".into(),
                    "scores".into(),
                    "plain".into(),
                    "noisy".into()
                ],
            }
        );
        let broken = FUNCTIONS.replace("matmul(x, w);", "matmul(w, x);");
        assert!(matches!(
            compile_function("two.mspec", &broken, "scores"),
            Err(FunctionError::Diagnostics(_))
        ));
    }
}
