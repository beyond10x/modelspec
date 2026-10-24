//! Adversary pass 2 on story:cpu-prepared-training-step: the output positions of a prepared step
//! when one gradient value is listed more than twice, a parameter the loss does not read, the
//! non-finite check and a resumed run on a shared-gradient program.
//!
//! `add(add(b1, b2), b3)` gives all three parameters the same gradient `ValueId` (autodiff passes
//! the adjoint of a same-shape `add` to both operands unchanged), so the training program lists
//! that value three times. The global allocator of this binary counts the allocations of the
//! calling thread.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::BTreeMap;

use modelspec_core::autodiff::differentiate;
use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::tensor::{Initializer, TensorProgramBuilder, TensorType, ValueId};
use modelspec_core::ir::training::{StateVersion, attach_sgd};
use modelspec_runtime::cpu::{HostTensor, evaluate, evaluate_all};
use modelspec_runtime::train::data::synthetic;
use modelspec_runtime::train::init::{Stream, initialize};
use modelspec_runtime::train::optimizer::{ParamState, PreparedStep, sgd_step};
use modelspec_runtime::train::runner::{self, RunError};

thread_local! {
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

/// The system allocator, counting each allocation of the calling thread.
struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocations() -> u64 {
    ALLOCATIONS.with(Cell::get)
}

/// A model whose three biases are added: `b1`, `b2` and `b3` share one gradient value.
const THREE_BIASES: &str = r#"
model Shared {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();
    param b1: Tensor<f32, [8, 1]> = xavier_uniform();
    param b2: Tensor<f32, [8, 1]> = xavier_uniform();
    param b3: Tensor<f32, [8, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>)
        -> Tensor<f32, [8, 1]> {
        return add(matmul(x, w), add(add(b1, b2), b3));
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

fn three_biases(steps: u64) -> CompiledTraining {
    let mut compiled = compile_training("shared.mspec", THREE_BIASES).expect("the source compiles");
    compiled.config.steps = steps;
    let gradients: Vec<ValueId> = compiled
        .training
        .training
        .gradients
        .values()
        .copied()
        .collect();
    assert_eq!(gradients.len(), 4);
    let shared = gradients.iter().filter(|&&g| g == gradients[1]).count();
    assert_eq!(
        shared, 3,
        "precondition: b1, b2 and b3 share one gradient value: {gradients:?}"
    );
    compiled
}

/// storage-reused on a program that lists one gradient three times, from `.mspec` source: the
/// second of two steps allocates no more than the number of outputs the step returns: 4 times
/// against 9 outputs, the two copies of the shared gradient with their shapes (10 before each
/// moved output kept its shape with its storage).
#[test]
fn a_second_step_with_a_gradient_listed_three_times_allocates_at_most_one_per_output() {
    let compiled = three_biases(2);
    let step = &compiled.training;
    let outputs = step.training.program.outputs.len() as u64;
    assert_eq!(outputs, 9, "loss, four gradients, four updates");
    let inputs = synthetic(&compiled.config, &compiled.forward)
        .unwrap()
        .inputs;
    let initial = initialize(
        &compiled.forward.program,
        compiled.config.seed,
        Stream::Init,
    )
    .unwrap();
    let mut state = ParamState::new(step, initial);
    let mut prepared = PreparedStep::new(step).unwrap();
    prepared.step(&inputs, &mut state).unwrap();
    let before = allocations();
    let evaluated = prepared.evaluate(&inputs, &state).unwrap();
    let loss = evaluated.loss();
    let gradients = evaluated.gradients().len();
    evaluated.apply(&mut state);
    let second = allocations() - before;
    assert!(loss.is_finite());
    assert_eq!(gradients, 4);
    assert!(
        second <= outputs,
        "the second step allocated {second} times; it returns {outputs} outputs"
    );
}

/// A gradient listed three times: at every step of a prepared step the loss is the forward
/// loss, each of the three parameters has the gradient `evaluate_all` computes for that value,
/// and each update is the `evaluate_all` value of its update node, bit for bit.
#[test]
fn a_gradient_listed_three_times_keeps_the_loss_gradients_and_updates() {
    let mut b = TensorProgramBuilder::new();
    let ty = TensorType::f32(&[3]);
    let params: Vec<_> = ["M.b1", "M.b2", "M.b3"]
        .iter()
        .map(|path| {
            b.declare_param(*path, ty.clone(), Initializer::XavierUniform)
                .unwrap()
        })
        .collect();
    let t = b.input("t", ty.clone()).unwrap();
    let reads: Vec<_> = params.iter().map(|&p| b.param_read(p).unwrap()).collect();
    let inner = b.add(reads[0], reads[1]).unwrap();
    let sum = b.add(inner, reads[2]).unwrap();
    let loss = b.mse(sum, t).unwrap();
    b.output(loss).unwrap();
    let forward = b.finish();
    let step = attach_sgd(differentiate(&forward, loss).unwrap(), 0.1).unwrap();
    let g = step.training.gradients[&params[0]];
    assert!(
        params.iter().all(|p| step.training.gradients[p] == g),
        "precondition: one gradient value for all three parameters"
    );
    let mut state = ParamState::new(
        &step,
        BTreeMap::from([
            (
                params[0],
                HostTensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap(),
            ),
            (
                params[1],
                HostTensor::new(vec![3], vec![0.5, -1.0, 0.25]).unwrap(),
            ),
            (
                params[2],
                HostTensor::new(vec![3], vec![-2.0, 0.0, 7.0]).unwrap(),
            ),
        ]),
    );
    let inputs = BTreeMap::from([(
        "t".to_string(),
        HostTensor::new(vec![3], vec![0.1, 0.2, 0.3]).unwrap(),
    )]);
    let mut prepared = PreparedStep::new(&step).unwrap();
    for k in 0..3 {
        let expected_loss = evaluate(&forward, &inputs, &state.values).unwrap()[0].data()[0];
        let all = evaluate_all(&step.training.program, &inputs, &state.values).unwrap();
        let pending = prepared.evaluate(&inputs, &state).unwrap();
        assert_eq!(
            pending.loss().to_bits(),
            expected_loss.to_bits(),
            "step {k}: loss {} against {expected_loss}",
            pending.loss()
        );
        assert_eq!(pending.gradients().len(), 3, "step {k}");
        for p in &params {
            assert_eq!(
                pending.gradients()[p],
                all.values[&g],
                "step {k}: gradient of {p}"
            );
        }
        pending.apply(&mut state);
        for p in &params {
            assert_eq!(
                state.values[p], all.values[&step.optimizer.updates[p]],
                "step {k}: update of {p}"
            );
            assert_eq!(state.versions[p], StateVersion(k + 1));
        }
    }
}

/// A parameter the loss does not read has no gradient and no update; beside a shared gradient
/// it keeps its value and version, and the other two parameters step as `sgd_step` steps them.
#[test]
fn a_parameter_the_loss_does_not_read_keeps_its_value_beside_a_shared_gradient() {
    let mut b = TensorProgramBuilder::new();
    let ty = TensorType::f32(&[3]);
    let unread = b
        .declare_param("M.a", ty.clone(), Initializer::XavierUniform)
        .unwrap();
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
    assert!(!step.training.gradients.contains_key(&unread));
    let values = BTreeMap::from([
        (
            unread,
            HostTensor::new(vec![3], vec![9.0, 8.0, 7.0]).unwrap(),
        ),
        (p1, HostTensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap()),
        (p2, HostTensor::new(vec![3], vec![0.0; 3]).unwrap()),
    ]);
    let mut state = ParamState::new(&step, values.clone());
    let mut unprepared = state.clone();
    let inputs = BTreeMap::from([(
        "t".to_string(),
        HostTensor::new(vec![3], vec![0.0; 3]).unwrap(),
    )]);
    let mut prepared = PreparedStep::new(&step).unwrap();
    for k in 0..3 {
        let expected_loss = evaluate(&forward, &inputs, &state.values).unwrap()[0].data()[0];
        let loss = prepared.step(&inputs, &mut state).unwrap();
        assert_eq!(loss.to_bits(), expected_loss.to_bits(), "step {k}");
        sgd_step(&step, &inputs, &mut unprepared).unwrap();
        assert_eq!(state, unprepared, "step {k}");
    }
    assert_eq!(state.values[&unread], values[&unread]);
    assert_eq!(state.versions[&unread], StateVersion(0));
    assert_eq!(state.versions[&p1], StateVersion(3));
}

/// The non-finite check reads the loss at position 0: with every bias at `1e20` the loss
/// overflows to `inf` while the gradient, listed three times, stays finite, so the run stops at
/// step 0 on the loss. Read from any gradient position the loss would be finite and the updated
/// parameters finite, and the run would not stop. The observer sees the infinite loss and a
/// gradient for each of the four parameters, the three biases' bit-identical.
#[test]
fn a_run_with_a_shared_gradient_stops_on_an_infinite_loss_at_step_0() {
    let compiled = three_biases(3);
    let forward = &compiled.forward.program;
    let initial = initialize(forward, compiled.config.seed, Stream::Init).unwrap();
    let mut init = BTreeMap::new();
    for (&id, decl) in &forward.params {
        if decl.path.contains(".b") {
            init.insert(id, HostTensor::new(vec![8, 1], vec![1.0e20f32; 8]).unwrap());
        }
    }
    assert_eq!(init.len(), 3);
    let biases: Vec<_> = init.keys().copied().collect();
    let mut seen = Vec::new();
    let result = runner::run_with(&compiled, None, &init, |o| {
        let gradients = o.gradients.expect("step 0 makes an update");
        assert_eq!(gradients.len(), 4);
        for gradient in gradients.values() {
            assert!(gradient.data().iter().all(|v| v.is_finite()));
        }
        assert_eq!(gradients[&biases[0]], gradients[&biases[1]]);
        assert_eq!(gradients[&biases[0]], gradients[&biases[2]]);
        seen.push((o.step, o.loss));
    });
    let Err(RunError::NonFinite(diverged)) = result else {
        panic!("the run did not stop on the infinite loss: {result:?}");
    };
    let mut values = initial;
    values.extend(init);
    let inputs = synthetic(&compiled.config, &compiled.forward)
        .unwrap()
        .inputs;
    let expected = evaluate(forward, &inputs, &values).unwrap()[0].data()[0];
    assert_eq!(expected, f32::INFINITY, "precondition: the loss overflows");
    assert_eq!(seen, [(0, f32::INFINITY)]);
    assert_eq!(diverged.run.losses, [f32::INFINITY]);
    let text = diverged.diagnostic.to_string();
    assert!(
        text.contains("the loss of step 0 is not finite"),
        "diagnostic: {text}"
    );
}

/// A resumed run on a program that lists one gradient three times: 2 steps, then 3 resumed from
/// the state after them, give the losses of 5 steps that never stopped, bit for bit, and the
/// same final state.
#[test]
fn a_resumed_run_with_a_shared_gradient_equals_one_that_never_stopped() {
    let whole = three_biases(5);
    let uninterrupted = runner::run(&whole).unwrap();
    let half = runner::run(&three_biases(2)).unwrap();
    let rest = three_biases(3);
    let resumed =
        runner::run_from(&rest, None, &half.state.values, 2, |_| {}, |_, _| Ok(())).unwrap();
    let bits = |losses: &[f32]| losses.iter().map(|l| l.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&half.losses[..2]), bits(&uninterrupted.losses[..2]));
    assert_eq!(bits(&resumed.losses), bits(&uninterrupted.losses[2..]));
    assert_eq!(resumed.state, uninterrupted.state);
}
