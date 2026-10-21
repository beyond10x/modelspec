//! Adversary cases for the SGD step of Training IR and `verify_step`: steps the verifier
//! accepts against its own documentation, and rules no case in `training_verify.rs` pins.

use modelspec_core::autodiff::differentiate;
use modelspec_core::ir::tensor::{
    Initializer, ParamId, TensorProgram, TensorProgramBuilder, TensorType, ValueId,
};
use modelspec_core::ir::training::{
    Access, Effect, EffectToken, StateVersion, Subgraph, TrainingStep, attach_sgd,
};
use modelspec_core::ir::verify::{TrainingVerifyError, verify_step};

const W: ParamId = ParamId(0);
const U: ParamId = ParamId(1);

fn ty(shape: &[u64]) -> TensorType {
    TensorType::f32(shape)
}

/// `mse(matmul(x, w) + matmul(x, w), y)` with `w` read by two `ParamRead` nodes, and a second
/// parameter `LinearReg.u` nothing reads.
fn read_twice_with_unused() -> (TensorProgram, ValueId, [ValueId; 2]) {
    let mut b = TensorProgramBuilder::new();
    let w = b
        .declare_param("LinearReg.w", ty(&[4, 1]), Initializer::XavierUniform)
        .unwrap();
    b.declare_param("LinearReg.u", ty(&[4, 1]), Initializer::XavierUniform)
        .unwrap();
    let x = b.input("x", ty(&[8, 4])).unwrap();
    let y = b.input("y", ty(&[8, 1])).unwrap();
    let r1 = b.param_read(w).unwrap();
    let h1 = b.matmul(x, r1).unwrap();
    let r2 = b.param_read(w).unwrap();
    let h2 = b.matmul(x, r2).unwrap();
    let s = b.add(h1, h2).unwrap();
    let loss = b.mse(s, y).unwrap();
    (b.finish(), loss, [r1, r2])
}

/// `mse(matmul(x, w), y)`.
fn linear() -> (TensorProgram, ValueId) {
    let mut b = TensorProgramBuilder::new();
    let w = b
        .declare_param("LinearReg.w", ty(&[4, 1]), Initializer::XavierUniform)
        .unwrap();
    let r = b.param_read(w).unwrap();
    let x = b.input("x", ty(&[8, 4])).unwrap();
    let y = b.input("y", ty(&[8, 1])).unwrap();
    let h = b.matmul(x, r).unwrap();
    let loss = b.mse(h, y).unwrap();
    (b.finish(), loss)
}

fn step(program: &TensorProgram, loss: ValueId) -> TrainingStep {
    attach_sgd(differentiate(program, loss).unwrap(), 0.1).unwrap()
}

fn effect(token: u32, param: ParamId, access: Access, version: u64) -> Effect {
    Effect {
        token: EffectToken(token),
        param,
        access,
        version: StateVersion(version),
    }
}

/// The write of `w` sits between its two reads; the second read hides behind a read effect
/// that names `u`. `ReadEffectMismatch` is the only rule that sees it: weaken it to "any
/// `ParamRead`" and the step verifies with a write ordered before the last read of `w`.
#[test]
fn a_param_read_whose_read_effect_names_another_parameter_is_refused() {
    let (program, loss, [r1, r2]) = read_twice_with_unused();
    let mut step = step(&program, loss);
    let update = step.optimizer.updates[&W];
    step.optimizer
        .effects
        .insert(r1, effect(0, W, Access::Read, 0));
    step.optimizer
        .effects
        .insert(update, effect(1, W, Access::Write, 1));
    step.optimizer
        .effects
        .insert(r2, effect(2, U, Access::Read, 0));
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::ReadEffectMismatch {
            value: r2,
            param: U
        })
    );
}

/// `UndefinedUpdate` has no case in `training_verify.rs`: an update that names a value no node
/// defines, with the outputs left as they were.
#[test]
fn an_update_no_node_defines_is_refused_by_its_own_rule() {
    let (program, loss) = linear();
    let mut step = step(&program, loss);
    step.optimizer.updates.insert(W, ValueId(999));
    assert_eq!(
        verify_step(&step),
        Err(TrainingVerifyError::UndefinedUpdate {
            param: W,
            value: ValueId(999)
        })
    );
}

/// `Subgraph::Gradient` holds the parameter gradients, `Subgraph::Optimizer` the learning rate
/// and the next values, and `verify_training` tags every seed-dependent node `Gradient`.
/// `verify_step` lets any seed-dependent node be tagged `Optimizer`, so the gradient output of
/// `w` itself can be.
#[test]
fn a_gradient_tagged_optimizer_is_refused() {
    let (program, loss) = linear();
    let mut step = step(&program, loss);
    let grad = step.training.gradients[&W];
    step.training.subgraphs.insert(grad, Subgraph::Optimizer);
    let result = verify_step(&step);
    assert!(
        result.is_err(),
        "the gradient {grad} of @LinearReg.w is tagged optimizer and the step verifies: {result:?}"
    );
}

/// `Optimizer::lr` is documented "positive and finite", and `verify_step` "checks every
/// invariant of a training step"; the printer shows it as the step's learning rate.
#[test]
fn a_step_whose_learning_rate_is_not_positive_and_finite_is_refused() {
    let (program, loss) = linear();
    for lr in [f32::NAN, 0.0, -0.1, f32::INFINITY] {
        let mut step = step(&program, loss);
        step.optimizer.lr = lr;
        let result = verify_step(&step);
        assert!(result.is_err(), "optimizer lr {lr} verifies: {result:?}");
    }
}

/// The documented order of `verify_step` puts the rules of `verify_training`, the output layout
/// of a step among them, before "no node outside the optimizer subgraph using an optimizer
/// value". A step breaking both reports the later rule.
#[test]
fn verify_step_reports_the_output_layout_before_an_optimizer_value_used() {
    let (program, loss) = linear();
    let mut step = step(&program, loss);
    let update = step.optimizer.updates[&W];
    step.training.subgraphs.insert(update, Subgraph::Gradient);
    step.training.program.outputs.swap(1, 2);
    let result = verify_step(&step);
    assert!(
        matches!(result, Err(TrainingVerifyError::OutputLayout { .. })),
        "{result:?}"
    );
}
