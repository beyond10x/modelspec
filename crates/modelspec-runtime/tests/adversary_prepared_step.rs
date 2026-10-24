//! Adversary cases for story:cpu-prepared-training-step: a training program in which two
//! parameters share one gradient value, so the step's outputs list that value twice.
//!
//! Autodiff passes the adjoint of a same-shape `Add` to both operands unchanged
//! (`modelspec_core::ops::gradient`, row `Add`), so `add(b1, b2)` gives `b1` and `b2` the same
//! gradient `ValueId`, and the training program's outputs are `[loss, g, g, update, update]`.

use std::collections::BTreeMap;

use modelspec_core::autodiff::differentiate;
use modelspec_core::compile::compile_training;
use modelspec_core::ir::tensor::{Initializer, TensorProgram, TensorProgramBuilder, TensorType};
use modelspec_core::ir::training::{TrainingStep, attach_sgd};
use modelspec_runtime::cpu::{HostTensor, evaluate};
use modelspec_runtime::train::data::synthetic;
use modelspec_runtime::train::init::{Stream, initialize};
use modelspec_runtime::train::optimizer::{ParamState, PreparedStep, sgd_step};
use modelspec_runtime::train::runner;

/// `loss = mse(b1 + b2, t)` over `[3]`, with `b1 = [1, 2, 3]`, `b2 = 0`, `t = 0`: the loss is
/// `(1 + 4 + 9) / 3` and the gradient of each parameter is `(b1 + b2 - t) * 2 / 3`.
fn shared_gradient_step() -> (
    TensorProgram,
    TrainingStep,
    ParamState,
    BTreeMap<String, HostTensor<f32>>,
) {
    let mut b = TensorProgramBuilder::new();
    let ty = TensorType::f32(&[3]);
    let p1 = b
        .declare_param("M.b1", ty.clone(), Initializer::XavierUniform)
        .unwrap();
    let p2 = b
        .declare_param("M.b2", ty.clone(), Initializer::XavierUniform)
        .unwrap();
    let t = b.input("t", ty.clone()).unwrap();
    let r1 = b.param_read(p1).unwrap();
    let r2 = b.param_read(p2).unwrap();
    let sum = b.add(r1, r2).unwrap();
    let loss = b.mse(sum, t).unwrap();
    b.output(loss).unwrap();
    let forward = b.finish();
    let step = attach_sgd(differentiate(&forward, loss).unwrap(), 0.1).unwrap();
    assert_eq!(
        step.training.gradients[&p1], step.training.gradients[&p2],
        "precondition: both parameters share one gradient value"
    );
    let values = BTreeMap::from([
        (p1, HostTensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap()),
        (p2, HostTensor::new(vec![3], vec![0.0, 0.0, 0.0]).unwrap()),
    ]);
    let state = ParamState::new(&step, values);
    let inputs = BTreeMap::from([(
        "t".to_string(),
        HostTensor::new(vec![3], vec![0.0; 3]).unwrap(),
    )]);
    (forward, step, state, inputs)
}

/// The interpreter's outputs of that training program are right: output 0, which the base's
/// `evaluate_step` read as the loss, is the forward loss, and the gradient listed twice is the
/// same tensor at both positions. So the defect is in the step's sorting of its outputs.
#[test]
fn the_interpreter_outputs_of_a_shared_gradient_are_right() {
    let (forward, step, state, inputs) = shared_gradient_step();
    let expected_loss = evaluate(&forward, &inputs, &state.values).unwrap()[0].data()[0];
    let outputs = evaluate(&step.training.program, &inputs, &state.values).unwrap();
    assert_eq!(outputs.len(), 5);
    assert_eq!(outputs[0].data()[0].to_bits(), expected_loss.to_bits());
    assert_eq!(outputs[1], outputs[2]);
    assert_eq!(outputs[1].data(), &[2.0f32 / 3.0, 4.0 / 3.0, 2.0]);
}

/// same-bits: `sgd_step` on a step whose two parameters share a gradient returns the forward
/// loss, and a gradient for each parameter, as the base's `evaluate_step` did (it cloned an
/// output listed twice).
#[test]
fn a_shared_gradient_keeps_the_loss_and_both_gradients() {
    let (forward, step, mut state, inputs) = shared_gradient_step();
    let expected_loss = evaluate(&forward, &inputs, &state.values).unwrap()[0].data()[0];
    assert_eq!(expected_loss, 14.0f32 / 3.0);
    let out = sgd_step(&step, &inputs, &mut state).unwrap();
    assert_eq!(
        out.loss.to_bits(),
        expected_loss.to_bits(),
        "loss {} against the forward loss {expected_loss}",
        out.loss
    );
    assert_eq!(
        out.gradients.len(),
        2,
        "a gradient per parameter: {:?}",
        out.gradients.keys()
    );
}

/// The same through a prepared step, twice: the loss of each step is the forward loss of the
/// state it ran on.
#[test]
fn a_prepared_step_with_a_shared_gradient_keeps_the_loss() {
    let (forward, step, mut state, inputs) = shared_gradient_step();
    let mut prepared = PreparedStep::new(&step).unwrap();
    for k in 0..2 {
        let expected = evaluate(&forward, &inputs, &state.values).unwrap()[0].data()[0];
        let pending = prepared.evaluate(&inputs, &state).unwrap();
        assert_eq!(
            pending.loss().to_bits(),
            expected.to_bits(),
            "step {k}: loss {} against the forward loss {expected}",
            pending.loss()
        );
        assert_eq!(
            pending.gradients().len(),
            2,
            "step {k}: a gradient per parameter"
        );
        pending.apply(&mut state);
    }
}

/// The same from `.mspec` source through the training loop: a model whose two parameters are
/// added reports, as the loss of step 0, the forward loss of its initial parameters.
#[test]
fn a_run_with_a_shared_gradient_reports_the_forward_loss() {
    let source = r#"
model Shared {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();
    param b1: Tensor<f32, [8, 1]> = xavier_uniform();
    param b2: Tensor<f32, [8, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>)
        -> Tensor<f32, [8, 1]> {
        return add(matmul(x, w), add(b1, b2));
    }
}

train Fit {
    model = Shared;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 42;
    device = cpu;
}
"#;
    let compiled = compile_training("shared.mspec", source).expect("the source compiles");
    let gradients: Vec<_> = compiled.training.training.gradients.values().collect();
    assert_eq!(gradients.len(), 3);
    assert!(
        gradients[1] == gradients[2],
        "precondition: b1 and b2 share a gradient value: {gradients:?}"
    );
    let forward = &compiled.forward.program;
    let inputs = synthetic(&compiled.config, &compiled.forward)
        .unwrap()
        .inputs;
    let initial = initialize(forward, compiled.config.seed, Stream::Init).unwrap();
    let expected = evaluate(forward, &inputs, &initial).unwrap()[0].data()[0];
    let run = runner::run(&compiled).expect("the run finishes");
    assert_eq!(
        run.losses[0].to_bits(),
        expected.to_bits(),
        "loss of step 0 {} against the forward loss {expected}",
        run.losses[0]
    );
}
