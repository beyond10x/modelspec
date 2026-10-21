//! The SGD step of Training IR and its legal-effects check (DESIGN.md § 3.1, § 3.2, § 3.3 pass 5).
//!
//! Each training step here is builder-made: a forward program built with
//! [`TensorProgramBuilder`], differentiated, then given its SGD step by [`attach_sgd`]. Each
//! legal-effects rule has its own case, which asserts the rule's own error, so removing one rule
//! turns its case red even where a later rule would still refuse the program.

use modelspec_core::autodiff::differentiate;
use modelspec_core::ir::print::{print_step, print_training};
use modelspec_core::ir::tensor::{
    Initializer, ParamId, TensorOp, TensorProgram, TensorProgramBuilder, TensorType, ValueId,
};
use modelspec_core::ir::training::{
    Access, AttachError, DataDecl, Effect, EffectToken, StateVersion, Subgraph, TrainConfig,
    TrainingStep, attach_sgd,
};
use modelspec_core::ir::verify::{TrainingVerifyError, verify_step, verify_training};
use modelspec_core::types::check_source;

const LR: f32 = 0.1;

fn ty(shape: &[u64]) -> TensorType {
    TensorType::f32(shape)
}

/// A builder-made forward program, its loss and the values the cases name.
struct Forward {
    program: TensorProgram,
    loss: ValueId,
    /// The `ParamRead` nodes of `w`, in node order.
    reads: Vec<ValueId>,
}

/// `mse(matmul(x, w), y)` with the shapes of `examples/linear_regression.mspec`; with `unused`,
/// a second parameter `LinearReg.u` of the same type that nothing reads.
fn linear(unused: bool) -> Forward {
    let mut b = TensorProgramBuilder::new();
    let w = b
        .declare_param("LinearReg.w", ty(&[4, 1]), Initializer::XavierUniform)
        .unwrap();
    if unused {
        b.declare_param("LinearReg.u", ty(&[4, 1]), Initializer::XavierUniform)
            .unwrap();
    }
    let r = b.param_read(w).unwrap();
    let x = b.input("x", ty(&[8, 4])).unwrap();
    let y = b.input("y", ty(&[8, 1])).unwrap();
    let h = b.matmul(x, r).unwrap();
    let loss = b.mse(h, y).unwrap();
    Forward {
        program: b.finish(),
        loss,
        reads: vec![r],
    }
}

/// `mse(matmul(x, w) + matmul(x, w), y)` with `w` read by two `ParamRead` nodes.
fn read_twice() -> Forward {
    let mut b = TensorProgramBuilder::new();
    let w = b
        .declare_param("LinearReg.w", ty(&[4, 1]), Initializer::XavierUniform)
        .unwrap();
    let x = b.input("x", ty(&[8, 4])).unwrap();
    let y = b.input("y", ty(&[8, 1])).unwrap();
    let r1 = b.param_read(w).unwrap();
    let h1 = b.matmul(x, r1).unwrap();
    let r2 = b.param_read(w).unwrap();
    let h2 = b.matmul(x, r2).unwrap();
    let s = b.add(h1, h2).unwrap();
    let loss = b.mse(s, y).unwrap();
    Forward {
        program: b.finish(),
        loss,
        reads: vec![r1, r2],
    }
}

fn step_of(forward: &Forward) -> TrainingStep {
    let training = differentiate(&forward.program, forward.loss).unwrap();
    attach_sgd(training, LR).unwrap()
}

const W: ParamId = ParamId(0);
const U: ParamId = ParamId(1);

/// The update of `w` and the `Mul lr, grad` it subtracts.
fn update_and_scaled(step: &TrainingStep) -> (ValueId, ValueId) {
    let update = step.optimizer.updates[&W];
    let node = step.training.program.node(update).unwrap();
    (update, node.operands[1])
}

fn read(token: u32, version: u64) -> Effect {
    Effect {
        token: EffectToken(token),
        param: W,
        access: Access::Read,
        version: StateVersion(version),
    }
}

fn write(token: u32, version: u64) -> Effect {
    Effect {
        token: EffectToken(token),
        param: W,
        access: Access::Write,
        version: StateVersion(version),
    }
}

// The SGD attachment.

#[test]
fn sgd_appends_p_minus_lr_times_grad_after_the_gradients() {
    let forward = linear(false);
    let step = step_of(&forward);
    let program = &step.training.program;
    let grad = step.training.gradients[&W];
    let (update, scaled) = update_and_scaled(&step);

    let update_node = program.node(update).unwrap();
    assert_eq!(update_node.op, TensorOp::Sub);
    assert_eq!(update_node.operands, vec![forward.reads[0], scaled]);
    assert_eq!(update_node.ty, ty(&[4, 1]));
    let scaled_node = program.node(scaled).unwrap();
    assert_eq!(scaled_node.op, TensorOp::Mul);
    let lr = scaled_node.operands[0];
    assert_eq!(scaled_node.operands, vec![lr, grad]);
    assert_eq!(program.node(lr).unwrap().op, TensorOp::Scalar(LR));
    for value in [lr, scaled, update] {
        assert_eq!(
            step.training.subgraphs[&value],
            Subgraph::Optimizer,
            "{value}"
        );
    }
    assert_eq!(program.outputs, vec![forward.loss, grad, update]);
    assert_eq!(step.optimizer.lr, LR);
    verify_step(&step).unwrap();
}

#[test]
fn sgd_versions_every_parameter_at_zero_and_writes_version_one() {
    let forward = linear(false);
    let step = step_of(&forward);
    let (update, _) = update_and_scaled(&step);
    assert_eq!(
        step.optimizer.versions.iter().collect::<Vec<_>>(),
        vec![(&W, &StateVersion(0))]
    );
    assert_eq!(
        step.optimizer.effects.iter().collect::<Vec<_>>(),
        vec![(&forward.reads[0], &read(0, 0)), (&update, &write(1, 1))]
    );
}

#[test]
fn sgd_orders_every_read_before_every_write() {
    let forward = read_twice();
    let step = step_of(&forward);
    let (update, _) = update_and_scaled(&step);
    assert_eq!(
        step.optimizer.effects_in_order(),
        vec![
            (forward.reads[0], read(0, 0)),
            (forward.reads[1], read(1, 0)),
            (update, write(2, 1)),
        ]
    );
    verify_step(&step).unwrap();
}

/// A parameter the loss does not read gets no gradient, no update and no write: it keeps its
/// version, and the step still verifies.
#[test]
fn sgd_gives_an_unused_parameter_no_update_and_no_write() {
    let forward = linear(true);
    let step = step_of(&forward);
    assert_eq!(step.optimizer.updates.keys().collect::<Vec<_>>(), vec![&W]);
    assert!(
        step.optimizer.effects.values().all(|e| e.param != U),
        "{:?}",
        step.optimizer.effects
    );
    assert_eq!(step.optimizer.versions[&U], StateVersion(0));
    assert_eq!(step.training.program.outputs.len(), 3);
    verify_step(&step).unwrap();
}

#[test]
fn sgd_refuses_a_learning_rate_that_is_not_positive_and_finite() {
    for lr in [0.0, -0.0, -0.1, f32::NAN, f32::INFINITY] {
        let forward = linear(false);
        let training = differentiate(&forward.program, forward.loss).unwrap();
        assert_eq!(
            attach_sgd(training, lr),
            Err(AttachError::LearningRate {
                lr_bits: lr.to_bits()
            }),
            "lr {lr}"
        );
    }
}

/// `verify_training` admits no optimizer node: a step is checked by `verify_step`.
#[test]
fn verify_training_refuses_an_optimizer_tag() {
    let step = step_of(&linear(false));
    let (_, scaled) = update_and_scaled(&step);
    let lr = step.training.program.node(scaled).unwrap().operands[0];
    assert_eq!(
        verify_training(&step.training),
        Err(TrainingVerifyError::OptimizerTagWithoutOptimizer { node: lr })
    );
}

// The optimizer subgraph of a step.

#[test]
fn a_forward_backward_or_loss_node_using_an_optimizer_value_is_refused() {
    let mut step = step_of(&linear(false));
    let (update, scaled) = update_and_scaled(&step);
    step.training.subgraphs.insert(update, Subgraph::Gradient);
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::OptimizerValueUsed {
            node: update,
            tag: Subgraph::Gradient,
            operand: scaled,
        })
    );
}

#[test]
fn an_update_for_a_parameter_without_gradient_is_refused() {
    let mut step = step_of(&linear(true));
    let (update, _) = update_and_scaled(&step);
    step.optimizer.updates.insert(U, update);
    step.training.program.outputs.push(update);
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::UpdateWithoutGradient { param: U })
    );
}

#[test]
fn a_parameter_with_a_gradient_but_no_update_is_refused() {
    let mut step = step_of(&linear(false));
    step.optimizer.updates.clear();
    step.training.program.outputs.pop();
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::MissingUpdate { param: W })
    );
}

#[test]
fn an_update_of_another_type_is_refused() {
    let mut step = step_of(&linear(false));
    let (_, scaled) = update_and_scaled(&step);
    let lr = step.training.program.node(scaled).unwrap().operands[0];
    step.optimizer.updates.insert(W, lr);
    *step.training.program.outputs.last_mut().unwrap() = lr;
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::UpdateType {
            param: W,
            expected: ty(&[4, 1]),
            got: ty(&[]),
        })
    );
}

#[test]
fn an_update_outside_the_optimizer_subgraph_is_refused() {
    let forward = linear(false);
    let mut step = step_of(&forward);
    let r = forward.reads[0];
    step.optimizer.updates.insert(W, r);
    *step.training.program.outputs.last_mut().unwrap() = r;
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::UpdateNotInOptimizer {
            param: W,
            value: r,
            tag: Subgraph::Forward,
        })
    );
}

#[test]
fn step_outputs_out_of_layout_are_refused() {
    let mut step = step_of(&linear(false));
    step.training.program.outputs.swap(1, 2);
    assert!(
        matches!(
            verify_step(&step),
            Err(TrainingVerifyError::OutputLayout { .. })
        ),
        "{:?}",
        verify_step(&step)
    );
}

// The legal effects of DESIGN.md § 3.2.

/// The story's named case: the write of `w` is ordered after its first read and before its
/// last read of the old version.
#[test]
fn a_write_ordered_before_the_last_read_of_the_old_version_is_refused() {
    let forward = read_twice();
    let mut step = step_of(&forward);
    let (update, _) = update_and_scaled(&step);
    step.optimizer.effects.insert(forward.reads[1], read(2, 0));
    step.optimizer.effects.insert(update, write(1, 1));
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::WriteBeforeRead {
            param: W,
            write: EffectToken(1),
            read: EffectToken(2),
            read_value: forward.reads[1],
        })
    );
}

#[test]
fn a_second_write_of_a_parameter_in_one_step_is_refused() {
    let mut step = step_of(&linear(false));
    let (update, scaled) = update_and_scaled(&step);
    step.optimizer.effects.insert(scaled, write(2, 1));
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::SecondWrite {
            param: W,
            first: update,
            second: scaled,
        })
    );
}

#[test]
fn a_write_that_skips_a_version_is_refused() {
    let mut step = step_of(&linear(false));
    let (update, _) = update_and_scaled(&step);
    step.optimizer.effects.insert(update, write(1, 2));
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::WrongNextVersion {
            param: W,
            current: StateVersion(0),
            got: StateVersion(2),
        })
    );
}

#[test]
fn a_write_that_keeps_the_version_is_refused() {
    let mut step = step_of(&linear(false));
    let (update, _) = update_and_scaled(&step);
    step.optimizer.effects.insert(update, write(1, 0));
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::WrongNextVersion {
            param: W,
            current: StateVersion(0),
            got: StateVersion(0),
        })
    );
}

/// At the last version there is no next one: refused, not a panic or a wrap to 0.
#[test]
fn a_write_past_the_last_version_is_refused() {
    let forward = linear(false);
    let mut step = step_of(&forward);
    let (update, _) = update_and_scaled(&step);
    step.optimizer.versions.insert(W, StateVersion(u64::MAX));
    step.optimizer
        .effects
        .insert(forward.reads[0], read(0, u64::MAX));
    step.optimizer.effects.insert(update, write(1, 0));
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::WrongNextVersion {
            param: W,
            current: StateVersion(u64::MAX),
            got: StateVersion(0),
        })
    );
}

#[test]
fn an_update_without_a_write_is_refused() {
    let mut step = step_of(&linear(false));
    let (update, _) = update_and_scaled(&step);
    step.optimizer.effects.remove(&update);
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::UpdateWithoutWrite {
            param: W,
            value: update
        })
    );
}

#[test]
fn a_write_carried_by_a_value_other_than_the_update_is_refused() {
    let mut step = step_of(&linear(false));
    let (update, scaled) = update_and_scaled(&step);
    let effect = step.optimizer.effects.remove(&update).unwrap();
    step.optimizer.effects.insert(scaled, effect);
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::WriteNotUpdate {
            value: scaled,
            param: W
        })
    );
}

#[test]
fn a_param_read_without_a_read_effect_is_refused() {
    let forward = linear(false);
    let mut step = step_of(&forward);
    step.optimizer.effects.remove(&forward.reads[0]);
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::ReadWithoutEffect {
            node: forward.reads[0]
        })
    );
}

#[test]
fn a_read_carried_by_a_node_that_is_not_a_param_read_is_refused() {
    let mut step = step_of(&linear(false));
    let (_, scaled) = update_and_scaled(&step);
    step.optimizer.effects.insert(scaled, read(5, 0));
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::ReadEffectMismatch {
            value: scaled,
            param: W
        })
    );
}

#[test]
fn a_read_of_another_version_is_refused() {
    let forward = linear(false);
    let mut step = step_of(&forward);
    step.optimizer.effects.insert(forward.reads[0], read(0, 1));
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::ReadOfOtherVersion {
            value: forward.reads[0],
            param: W,
            expected: StateVersion(0),
            got: StateVersion(1),
        })
    );
}

#[test]
fn two_effects_with_one_token_are_refused() {
    let forward = linear(false);
    let mut step = step_of(&forward);
    let (update, _) = update_and_scaled(&step);
    step.optimizer.effects.insert(update, write(0, 1));
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::DuplicateEffectToken {
            token: EffectToken(0),
            first: forward.reads[0],
            second: update,
        })
    );
}

#[test]
fn an_effect_carried_by_an_undefined_value_is_refused() {
    let mut step = step_of(&linear(false));
    step.optimizer.effects.insert(ValueId(999), read(7, 0));
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::EffectWithoutNode {
            value: ValueId(999)
        })
    );
}

#[test]
fn a_parameter_without_a_state_version_is_refused() {
    let mut step = step_of(&linear(true));
    step.optimizer.versions.remove(&U);
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::MissingVersion { param: U })
    );
}

#[test]
fn a_state_version_for_an_undeclared_parameter_is_refused() {
    let mut step = step_of(&linear(false));
    step.optimizer.versions.insert(ParamId(9), StateVersion(0));
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::VersionForUndeclaredParam { param: ParamId(9) })
    );
}

// Printing.

const LINEAR_STEP: &str = "\
param @LinearReg.w: Tensor<f32, [4, 1]> = xavier_uniform {v0}
optimizer sgd lr = 0.1
forward %0: Tensor<f32, [4, 1]> = ParamRead @LinearReg.w {!0 read @LinearReg.w v0}
forward %1: Tensor<f32, [8, 4]> = Input \"x\"
forward %2: Tensor<f32, [8, 1]> = Input \"y\"
forward %3: Tensor<f32, [8, 1]> = MatMul %1, %0
loss %4: Tensor<f32, []> = Mse %3, %2
loss %5: Tensor<f32, []> = Scalar 1.0
forward %6: Tensor<f32, [8, 1]> = Sub %3, %2
forward %7: Tensor<f32, []> = Scalar 0.25
gradient %8: Tensor<f32, []> = Mul %5, %7
gradient %9: Tensor<f32, [8, 1]> = Mul %6, %8
forward %10: Tensor<f32, [4, 8]> = Transpose %1
gradient %11: Tensor<f32, [4, 1]> = MatMul %10, %9
optimizer %12: Tensor<f32, []> = Scalar 0.1
optimizer %13: Tensor<f32, [4, 1]> = Mul %12, %11
optimizer %14: Tensor<f32, [4, 1]> = Sub %0, %13 {!1 write @LinearReg.w v1}
output %4 loss
output %11 gradient @LinearReg.w
output %14 update @LinearReg.w
";

#[test]
fn a_step_prints_its_subgraphs_effects_and_versions() {
    assert_eq!(print_step(&step_of(&linear(false))), LINEAR_STEP);
}

#[test]
fn a_training_program_prints_its_subgraphs_without_effects() {
    let forward = linear(false);
    let training = differentiate(&forward.program, forward.loss).unwrap();
    let expected: String = LINEAR_STEP
        .lines()
        .filter(|l| !l.starts_with("optimizer") && !l.contains("update"))
        .map(|l| match l.find(" {") {
            Some(at) => format!("{}\n", &l[..at]),
            None => format!("{l}\n"),
        })
        .collect();
    assert_eq!(print_training(&training), expected);
}

#[test]
fn an_unused_parameter_prints_with_its_version_and_no_update() {
    let printed = print_step(&step_of(&linear(true)));
    assert!(
        printed.contains("param @LinearReg.u: Tensor<f32, [4, 1]> = xavier_uniform {v0}\n"),
        "{printed}"
    );
    assert!(!printed.contains("update @LinearReg.u"), "{printed}");
    assert!(!printed.contains("write @LinearReg.u"), "{printed}");
}

// The configuration of the train block.

#[test]
fn train_config_carries_the_sgd_configuration_of_the_train_block() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/linear_regression.mspec"
    );
    let source = std::fs::read_to_string(path).unwrap();
    let hir = check_source("linear_regression.mspec", &source).unwrap();
    let config = TrainConfig::from_hir(&hir.train);
    assert_eq!(config.steps, 500);
    assert_eq!(config.lr, 0.1);
    assert_eq!(config.seed, 42);
    let decls = |d: &[DataDecl]| -> Vec<(String, TensorType)> {
        d.iter().map(|d| (d.name.clone(), d.ty.clone())).collect()
    };
    assert_eq!(decls(&config.inputs), vec![("x".to_string(), ty(&[8, 4]))]);
    assert_eq!(decls(&config.targets), vec![("y".to_string(), ty(&[8, 1]))]);
    assert_eq!(config.span, hir.train.span);
}
