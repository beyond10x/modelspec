//! story:cpu-prepared-training-step: a CPU run verifies its training step once, builds its
//! tables once, frees each intermediate after its last consumer, reuses per-op output storage
//! and moves its outputs out, on the MNIST training program of `examples/mnist_mlp.mspec`.
//!
//! The counters read here are per thread, so the cases of this binary do not see each other's
//! calls. The global allocator of this binary counts the allocations of the calling thread.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::tensor::{TensorOp, TensorProgram, ValueId};
use modelspec_core::ir::verify;
use modelspec_runtime::cpu::interpreter::{self, Prepared};
use modelspec_runtime::cpu::{HostTensor, evaluate_all};
use modelspec_runtime::train::data::synthetic;
use modelspec_runtime::train::init::{Stream, initialize};
use modelspec_runtime::train::optimizer::{self, ParamState, PreparedStep, sgd_step};

const MNIST_MLP: &str = "examples/mnist_mlp.mspec";

/// The blake3 of the bits of the loss and of every gradient and parameter of three MNIST steps
/// through `sgd_step` ([`digest_of_three_steps`]), recorded on the tree before this story and
/// re-recorded once when DESIGN.md § 12 ADR-009 fused each later product into its sum.
const THREE_STEPS_BLAKE3: &str = "432a1c821f71330fc17110d9fd7338090750d71ce0eee3e06e12883b4b9555a1";

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

struct Mnist {
    compiled: CompiledTraining,
    inputs: BTreeMap<String, HostTensor<f32>>,
}

impl Mnist {
    fn load() -> Self {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(MNIST_MLP);
        let source = std::fs::read_to_string(&path).expect("the example reads");
        let compiled = compile_training(MNIST_MLP, &source).expect("the example compiles");
        let inputs = synthetic(&compiled.config, &compiled.forward)
            .expect("the synthetic data is drawn")
            .inputs;
        Self { compiled, inputs }
    }

    fn state(&self) -> ParamState {
        let initial = initialize(
            &self.compiled.forward.program,
            self.compiled.config.seed,
            Stream::Init,
        )
        .expect("the parameters are drawn");
        ParamState::new(&self.compiled.training, initial)
    }

    fn program(&self) -> &TensorProgram {
        &self.compiled.training.training.program
    }
}

fn hash_tensor(digest: &mut blake3::Hasher, tensor: &HostTensor<f32>) {
    for x in tensor.data() {
        digest.update(&x.to_bits().to_le_bytes());
    }
}

/// The blake3 of three steps through `sgd_step`: per step the loss bits and each gradient's,
/// then every parameter's after the three updates.
fn digest_of_three_steps(mnist: &Mnist) -> String {
    let mut state = mnist.state();
    let mut digest = blake3::Hasher::new();
    for _ in 0..3 {
        let out = sgd_step(&mnist.compiled.training, &mnist.inputs, &mut state).unwrap();
        digest.update(&out.loss.to_bits().to_le_bytes());
        for gradient in out.gradients.values() {
            hash_tensor(&mut digest, gradient);
        }
    }
    for value in state.values.values() {
        hash_tensor(&mut digest, value);
    }
    digest.finalize().to_hex().to_string()
}

/// `count` steps of a prepared step: the verifier and table counters it moves.
fn prepared_run_counts(mnist: &Mnist, count: usize) -> [u64; 5] {
    let read = || {
        [
            verify::counters::verify(),
            verify::counters::verify_step(),
            verify::counters::effects_in_order(),
            optimizer::counters::writes_table(),
            interpreter::counters::in_place_transposes(),
        ]
    };
    let before = read();
    let mut state = mnist.state();
    let mut prepared = PreparedStep::new(&mnist.compiled.training).expect("the step verifies");
    for _ in 0..count {
        prepared.step(&mnist.inputs, &mut state).unwrap();
    }
    let after = read();
    std::array::from_fn(|i| after[i] - before[i])
}

/// verify-once-per-run: each verifier is called once per run, whatever its number of steps.
#[test]
fn a_prepared_run_calls_each_verifier_once() {
    let mnist = Mnist::load();
    let [verify, verify_step, ..] = prepared_run_counts(&mnist, 5);
    assert_eq!(verify_step, 1, "verify_step calls in a run of 5 steps");
    assert_eq!(verify, 1, "verify calls in a run of 5 steps");
    let [verify, verify_step, ..] = prepared_run_counts(&mnist, 1);
    assert_eq!((verify, verify_step), (1, 1), "a run of 1 step");
}

/// A step of `sgd_step` is a run of one step: it verifies the step, and the step's program once,
/// not again in the interpreter (before this story `verify` ran twice per step).
#[test]
fn each_unprepared_step_verifies_again() {
    let mnist = Mnist::load();
    let mut state = mnist.state();
    let (verify, verify_step) = (verify::counters::verify(), verify::counters::verify_step());
    for _ in 0..3 {
        sgd_step(&mnist.compiled.training, &mnist.inputs, &mut state).unwrap();
    }
    assert_eq!(verify::counters::verify_step() - verify_step, 3);
    assert_eq!(verify::counters::verify() - verify, 3);
}

/// tables-once-per-run: `effects_in_order`, the `writes` table and `in_place_transposes` are
/// built once per run, whatever its number of steps.
#[test]
fn a_prepared_run_builds_each_table_once() {
    let mnist = Mnist::load();
    for steps in [1, 5] {
        let [_, _, effects, writes, in_place] = prepared_run_counts(&mnist, steps);
        assert_eq!(
            effects, 1,
            "effects_in_order builds in a run of {steps} steps"
        );
        assert_eq!(writes, 1, "writes table builds in a run of {steps} steps");
        assert_eq!(
            in_place, 1,
            "in_place_transposes builds in a run of {steps} steps"
        );
    }
}

/// The largest number of values live at once in `program` by last-consumer analysis: a value is
/// live from its node through its last consumer, an output to the end. A `transpose` whose every
/// reader is a `matmul` reading it as its left operand, and which is not an output, has no value
/// (the `matmul` reads its operand in place), so its readers are its operand's consumers.
fn peak_by_last_consumer(program: &TensorProgram) -> usize {
    let n = program.nodes.len();
    let index: BTreeMap<ValueId, usize> = program
        .nodes
        .iter()
        .enumerate()
        .map(|(i, node)| (node.id, i))
        .collect();
    let mut read_elsewhere: BTreeSet<ValueId> = program.outputs.iter().copied().collect();
    for node in &program.nodes {
        for (k, &operand) in node.operands.iter().enumerate() {
            if !(node.op == TensorOp::MatMul && k == 0) {
                read_elsewhere.insert(operand);
            }
        }
    }
    let in_place: BTreeMap<usize, usize> = program
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.op == TensorOp::Transpose && !read_elsewhere.contains(&node.id))
        .map(|(i, node)| (i, index[&node.operands[0]]))
        .collect();
    let mut last: Vec<usize> = (0..n).collect();
    for (j, node) in program.nodes.iter().enumerate() {
        if in_place.contains_key(&j) {
            continue;
        }
        for operand in &node.operands {
            let i = index[operand];
            let i = in_place.get(&i).copied().unwrap_or(i);
            last[i] = last[i].max(j);
        }
    }
    for output in &program.outputs {
        last[index[output]] = usize::MAX;
    }
    (0..n)
        .filter(|i| !in_place.contains_key(i))
        .map(|i| {
            (0..=i)
                .filter(|v| !in_place.contains_key(v) && last[*v] >= i)
                .count()
        })
        .max()
        .unwrap_or(0)
}

/// intermediates-freed: during a step of the MNIST training program the peak of live values is
/// the last-consumer peak, below the node count; `evaluate_all` still keeps every value.
#[test]
fn a_step_frees_each_intermediate_after_its_last_consumer() {
    let mnist = Mnist::load();
    let program = mnist.program();
    let expected = peak_by_last_consumer(program);
    let nodes = program.nodes.len();
    assert!(
        expected < nodes,
        "the analysis peak {expected} is below the node count {nodes}"
    );
    let mut state = mnist.state();
    let mut prepared = PreparedStep::new(&mnist.compiled.training).unwrap();
    for _ in 0..2 {
        prepared.step(&mnist.inputs, &mut state).unwrap();
        assert_eq!(
            interpreter::counters::peak_live_values(),
            expected,
            "peak live values of a step; the program has {nodes} nodes"
        );
    }

    let evaluation = evaluate_all(program, &mnist.inputs, &state.values).unwrap();
    assert_eq!(interpreter::counters::peak_live_values(), nodes);
    assert_eq!(
        evaluation.values.len(),
        nodes,
        "evaluate_all keeps every value"
    );
}

/// storage-reused: the second of two consecutive steps allocates no more than the number of
/// outputs the step returns. On MNIST it allocates nothing against 5 outputs (4 before each
/// moved output kept its shape with its storage).
#[test]
fn a_second_step_allocates_at_most_one_per_output() {
    let mnist = Mnist::load();
    let outputs = mnist.program().outputs.len() as u64;
    let mut state = mnist.state();
    let mut prepared = PreparedStep::new(&mnist.compiled.training).unwrap();
    prepared.step(&mnist.inputs, &mut state).unwrap();
    let before = allocations();
    let evaluated = prepared.evaluate(&mnist.inputs, &state).unwrap();
    let loss = evaluated.loss();
    let gradients = evaluated.gradients().len();
    evaluated.apply(&mut state);
    let second = allocations() - before;
    assert!(loss.is_finite());
    assert_eq!(gradients, 2);
    assert!(
        second <= outputs,
        "the second step allocated {second} times (0 when every output is moved); it returns \
         {outputs} outputs"
    );
}

/// Allocations of each of `lens` elements, held while a case runs, so that an address freed
/// before them is not handed out again afterwards: a buffer address that recurs after them was
/// never freed.
fn blockers(lens: impl IntoIterator<Item = usize>) -> Vec<Vec<f32>> {
    lens.into_iter().map(|n| vec![1.0f32; n]).collect()
}

/// outputs-moved: an output leaves the interpreter as the buffer the interpreter wrote, not a
/// copy of it: a buffer handed back is the one the next run returns.
#[test]
fn outputs_leave_the_interpreter_without_a_copy() {
    let mnist = Mnist::load();
    let state = mnist.state();
    let mut prepared = Prepared::new(mnist.program()).expect("the program verifies");
    let first = prepared.run(&mnist.inputs, &state.values).unwrap();
    let buffers: Vec<*const f32> = first.iter().map(|t| t.data().as_ptr()).collect();
    let lens: Vec<usize> = first.iter().map(HostTensor::len).collect();
    for (position, output) in first.into_iter().enumerate() {
        prepared.recycle(position, output);
    }
    let _held = blockers(lens);
    let second = prepared.run(&mnist.inputs, &state.values).unwrap();
    let again: Vec<*const f32> = second.iter().map(|t| t.data().as_ptr()).collect();
    assert_eq!(again, buffers);
}

/// outputs-moved, through a step: each update becomes its parameter's value in the buffer the
/// interpreter wrote, and the value it replaces is the storage of the next update, so the buffers
/// of a parameter alternate between two allocations.
#[test]
fn an_update_becomes_the_parameter_without_a_copy() {
    let mnist = Mnist::load();
    let mut state = mnist.state();
    let mut prepared = PreparedStep::new(&mnist.compiled.training).unwrap();
    let lens: Vec<usize> = state.values.values().map(HostTensor::len).collect();
    let mut held = Vec::new();
    let mut buffers = Vec::new();
    for _ in 0..4 {
        prepared.step(&mnist.inputs, &mut state).unwrap();
        held.push(blockers(lens.iter().copied()));
        let at: Vec<*const f32> = state.values.values().map(|v| v.data().as_ptr()).collect();
        buffers.push(at);
    }
    assert_eq!(buffers[2], buffers[0]);
    assert_eq!(buffers[3], buffers[1]);
    assert_ne!(buffers[1], buffers[0]);
}

/// same-bits: three steps through `sgd_step` give the bits recorded before this story, and a
/// prepared step gives the same loss, gradients and parameters bit for bit at every step.
#[test]
fn a_prepared_step_keeps_the_bits() {
    let mnist = Mnist::load();
    assert_eq!(digest_of_three_steps(&mnist), THREE_STEPS_BLAKE3);

    let mut unprepared = mnist.state();
    let mut state = mnist.state();
    let mut prepared = PreparedStep::new(&mnist.compiled.training).unwrap();
    for _ in 0..3 {
        let expected = sgd_step(&mnist.compiled.training, &mnist.inputs, &mut unprepared).unwrap();
        let evaluated = prepared.evaluate(&mnist.inputs, &state).unwrap();
        assert_eq!(evaluated.loss().to_bits(), expected.loss.to_bits());
        assert_eq!(evaluated.gradients(), &expected.gradients);
        evaluated.apply(&mut state);
        assert_eq!(state, unprepared);
    }
}

/// storage-reused with a shared gradient: two parameters whose gradient is one value (the adjoint
/// of a same-shape `add` reaches both operands unchanged) give outputs `[loss, g, g, u1, u2]`.
/// The first listing of `g` is a copy and the only output that allocates, once for its data and
/// once for its shape: the second step allocates 2 times (5 before each moved output kept its
/// shape with its storage), no more than the outputs it returns, and both gradients and the loss
/// stay right.
#[test]
fn a_second_step_with_a_shared_gradient_allocates_at_most_one_per_output() {
    use modelspec_core::autodiff::differentiate;
    use modelspec_core::ir::tensor::{Initializer, TensorProgramBuilder, TensorType};
    use modelspec_core::ir::training::attach_sgd;

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
    let step = attach_sgd(differentiate(&b.finish(), loss).unwrap(), 0.1).unwrap();
    let outputs = &step.training.program.outputs;
    assert_eq!(
        outputs[1], outputs[2],
        "precondition: one gradient listed twice"
    );
    let mut state = ParamState::new(
        &step,
        BTreeMap::from([
            (p1, HostTensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap()),
            (p2, HostTensor::new(vec![3], vec![0.0; 3]).unwrap()),
        ]),
    );
    let inputs = BTreeMap::from([(
        "t".to_string(),
        HostTensor::new(vec![3], vec![0.0; 3]).unwrap(),
    )]);
    let mut unprepared = state.clone();
    let mut prepared = PreparedStep::new(&step).unwrap();
    prepared.step(&inputs, &mut state).unwrap();
    sgd_step(&step, &inputs, &mut unprepared).unwrap();
    let expected = sgd_step(&step, &inputs, &mut unprepared.clone()).unwrap();
    let before = allocations();
    let evaluated = prepared.evaluate(&inputs, &state).unwrap();
    let loss = evaluated.loss();
    let after = allocations();
    let gradients = evaluated.gradients().clone();
    let applying = allocations();
    evaluated.apply(&mut state);
    let second = (after - before) + (allocations() - applying);
    assert_eq!(loss.to_bits(), expected.loss.to_bits());
    assert_eq!(gradients, expected.gradients);
    assert_eq!(gradients.len(), 2);
    sgd_step(&step, &inputs, &mut unprepared).unwrap();
    assert_eq!(
        state, unprepared,
        "both parameters updated as sgd_step updates them"
    );
    assert!(
        second <= outputs.len() as u64,
        "the second step allocated {second} times (2 for the copy of g, data and shape); it \
         returns {} outputs",
        outputs.len()
    );
}
