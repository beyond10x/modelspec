//! The compiler driver: `.mspec` source to Typed HIR, to verified Tensor IR and to a verified
//! training step (DESIGN.md § 3, § 3.1, § 3.3 passes 1–5).
//!
//! [`compile_hir`] parses, resolves and checks; [`compile_tensor`] then lowers, removes dead code
//! and verifies; [`compile_training`] then differentiates the loss, attaches SGD and verifies the
//! step, legal effects included. A source fault is reported as the diagnostics of the stage that
//! found it. A program the verifier refuses after lowering, a lowering failure on checked HIR, or
//! an autodiff or SGD failure on verified Tensor IR is a compiler bug and is reported under
//! [`codes::COMPILER_BUG`], never as a user code (DESIGN.md § 8.1).

use crate::artifact::{Backend, CompiledArtifact, E_ARTIFACT_005};
use crate::autodiff::differentiate;
use crate::device::DeviceProfile;
use crate::diag::Diagnostic;
use crate::hir::Hir;
use crate::ir::kernel::MatMulLimits;
use crate::ir::tensor::{TensorProgram, ValueId};
use crate::ir::training::{OptimizerRule, TrainConfig, TrainingStep, attach};
use crate::ir::verify::{verify, verify_plan_for, verify_step};
use crate::lower::{self, LoweredProgram};
use crate::passes::dce;
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

    /// An operator of the training step has no Vulkan kernel yet, or the optimizer `muon`, whose
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
    /// The Tensor IR before autodiff, with its `prediction` value and every dropout the identity
    /// ([`lower::without_dropout`]): what the data loader, the final loss line, evaluation and
    /// inference run. Its node ids are those of the training step.
    pub forward: LoweredProgram,
    /// The `train` block's learning rate, steps, seed and data declarations; its optimizer rule
    /// is the step's.
    pub config: TrainConfig,
}

/// [`compile_tensor`], then [`differentiate`] of its loss, [`attach`] of the `train` block's
/// optimizer, `sgd`, `adamw` or `muon`, and [`verify_step`]. The step keeps every dropout; the
/// forward program is the same Tensor IR with each dropout the identity, verified.
pub fn compile_training(path: &str, source: &str) -> Result<CompiledTraining, Vec<Diagnostic>> {
    let hir = compile_hir(path, source)?;
    let config = TrainConfig::from_hir(&hir.train);
    let lowered = tensor_stage(&hir, dce::eliminate)?;
    let training = training_stage(&lowered.program, config.lr, hir.train.optimizer)?;
    let forward = lower::without_dropout(lowered);
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
    let compiled = compile_training(path, source)?;
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

/// Differentiates the single output of `forward`, its loss, attaches the optimizer `rule` at
/// `lr` and verifies the step.
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
}
