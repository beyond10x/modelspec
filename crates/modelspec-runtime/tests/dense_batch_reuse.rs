//! story:dense-step-reuses-batch-buffers: a dense training step fills buffers allocated once per
//! run.
//!
//! - **no-per-step-allocation**: after the first fill, [`DenseData::fill_step`] and
//!   [`DenseData::fill_eval_batch`] allocate no `x`, `y` (shape or data) or key `String`. The
//!   global allocator of this test binary logs every pointer it hands out (`alloc`,
//!   `alloc_zeroed`, `realloc`) while the calling thread records; after a fill, the shape and
//!   data pointers of `x` and `y` and the pointers of both keys must not be among the pointers
//!   handed out during that fill, and a runner step allocates no more than a synthetic one.
//!   Buffer identity alone would not do: a fresh buffer freed at the end of one step is often
//!   handed back by the allocator for the next.
//! - **same-loss-bits**: [`run_with`] and [`evaluate_dense`] give the bits of a loop over fresh
//!   [`DenseData::step`] and [`DenseData::eval_batch`] maps, the loop the runner ran before.
//!
//! The fixture is six samples of a 3-feature, 2-class model with `B = 2`, written under the test
//! temp dir.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_runtime::cpu::{HostTensor, evaluate, evaluate_all};
use modelspec_runtime::train::data::{DenseBatch, DenseData, dense, dense_eval};
use modelspec_runtime::train::init::{Stream, initialize};
use modelspec_runtime::train::optimizer::{ParamState, sgd_step};
use modelspec_runtime::train::runner::{evaluate_dense, run_with};

/// How many pointers one recording window can hold.
const LOG: usize = 1 << 14;

static HANDED_OUT: [AtomicUsize; LOG] = [const { AtomicUsize::new(0) }; LOG];
static COUNT: AtomicUsize = AtomicUsize::new(0);
/// One recording window at a time: the log is shared by every thread of the binary.
static WINDOW: Mutex<()> = Mutex::new(());

thread_local! {
    static RECORDING: Cell<bool> = const { Cell::new(false) };
}

struct Logging;

fn note(ptr: *mut u8) {
    if RECORDING.try_with(Cell::get).unwrap_or(false) {
        let i = COUNT.fetch_add(1, Ordering::Relaxed);
        if i < LOG {
            HANDED_OUT[i].store(ptr as usize, Ordering::Relaxed);
        }
    }
}

// SAFETY: every call is forwarded to `System` unchanged; `note` only stores the pointer.
unsafe impl GlobalAlloc for Logging {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        note(ptr);
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        note(ptr);
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let ptr = unsafe { System.realloc(ptr, layout, new_size) };
        note(ptr);
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Logging = Logging;

/// Runs `f` while this thread records, and returns the pointers handed out meanwhile.
fn handed_out(f: impl FnOnce()) -> BTreeSet<usize> {
    let _window = WINDOW
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    COUNT.store(0, Ordering::Relaxed);
    RECORDING.with(|r| r.set(true));
    f();
    RECORDING.with(|r| r.set(false));
    let count = COUNT.load(Ordering::Relaxed);
    assert!(
        count <= LOG,
        "{count} allocations overflow the log of {LOG}"
    );
    (0..count)
        .map(|i| HANDED_OUT[i].load(Ordering::Relaxed))
        .collect()
}

/// Runs `f` while this thread records, and returns how many allocations it made.
fn allocations(f: impl FnOnce()) -> usize {
    let _window = WINDOW
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    COUNT.store(0, Ordering::Relaxed);
    RECORDING.with(|r| r.set(true));
    f();
    RECORDING.with(|r| r.set(false));
    COUNT.load(Ordering::Relaxed)
}

const MODEL: &str = "\
model Lin {
    param w: Tensor<f32, [3, 2]> = xavier_uniform();

    fn forward(pixels: Tensor<f32, [2, 3]>)
        -> Tensor<f32, [2, 2]> {
        return matmul(pixels, w);
    }
}

train Fit {
    model = Lin;
    input pixels: Tensor<f32, [2, 3]>;
    target label: Tensor<f32, [2, 2]>;
    loss = mse(model.forward(pixels), label);
    optimizer = sgd(lr = 0.1);
    steps = 7;
    seed = 42;
    device = cpu;
}
";

const SAMPLES: usize = 6;
const X: [f32; SAMPLES * 3] = [
    1.0, 0.0, 0.5, 0.0, 1.0, -0.5, 1.0, 1.0, 0.0, 2.0, 0.0, 1.0, -1.0, 0.5, 0.25, 0.0, -2.0, 1.5,
];
const Y: [f32; SAMPLES * 2] = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 3.0, 2.0, 0.0, 0.5, 0.5];
/// Fourteen steps' worth of indices plus the loss after the last update.
const ORDER: [u32; 16] = [5, 2, 0, 3, 4, 1, 1, 1, 3, 0, 2, 5, 4, 4, 0, 2];

fn compiled() -> CompiledTraining {
    compile_training("dense_batch_reuse.mspec", MODEL).expect("the fixture compiles")
}

fn write_f32(path: &Path, values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).expect("the fixture is written");
}

/// A fresh directory `name` holding the fixture, with `order.u32` when `order` is given.
fn fixture(name: &str, order: Option<&[u32]>) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("dense_batch_reuse")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("the old fixture is removed");
    }
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    write_f32(&dir.join("pixels.f32"), &X);
    write_f32(&dir.join("label.f32"), &Y);
    if let Some(order) = order {
        let bytes: Vec<u8> = order.iter().flat_map(|i| i.to_le_bytes()).collect();
        std::fs::write(dir.join("order.u32"), bytes).expect("the order is written");
    }
    dir
}

fn train_data(name: &str, order: Option<&[u32]>) -> (CompiledTraining, DenseData) {
    let compiled = compiled();
    let data = dense(&fixture(name, order), &compiled.config, &compiled.forward)
        .expect("the fixture is dense data");
    (compiled, data)
}

/// The pointers of the buffers a step must not allocate: each key, and each tensor's shape and
/// data.
fn buffers(inputs: &BTreeMap<String, HostTensor<f32>>) -> Vec<(String, usize)> {
    inputs
        .iter()
        .flat_map(|(name, tensor)| {
            [
                (format!("key {name}"), name.as_ptr() as usize),
                (format!("shape of {name}"), tensor.shape().as_ptr() as usize),
                (format!("data of {name}"), tensor.data().as_ptr() as usize),
            ]
        })
        .collect()
}

/// Fills `batch` with `fill` once, then `rounds` more times, and asserts after each later fill
/// that it allocated no key and no data buffer and holds what `fresh` makes for that round.
fn assert_reused(
    rounds: u64,
    mut fill: impl FnMut(u64, &mut DenseBatch),
    fresh: impl Fn(u64) -> BTreeMap<String, HostTensor<f32>>,
) {
    let mut batch = DenseBatch::default();
    fill(0, &mut batch);
    assert_eq!(batch.inputs(), &fresh(0), "round 0 holds its samples");
    for k in 1..=rounds {
        let allocated = handed_out(|| fill(k, &mut batch));
        for (what, ptr) in buffers(batch.inputs()) {
            assert!(
                !allocated.contains(&ptr),
                "round {k} allocated the {what} ({} allocations in the round)",
                allocated.len()
            );
        }
        assert_eq!(batch.inputs(), &fresh(k), "round {k} holds its samples");
    }
}

#[test]
fn no_per_step_allocation_without_an_order() {
    let (_, data) = train_data("no_order", None);
    assert_reused(9, |k, b| data.fill_step(k, b), |k| data.step(k));
}

#[test]
fn no_per_step_allocation_with_an_order() {
    let (_, data) = train_data("order", Some(&ORDER));
    assert_reused(7, |k, b| data.fill_step(k, b), |k| data.step(k));
}

#[test]
fn no_per_eval_batch_allocation() {
    let compiled = compiled();
    let data = dense_eval(&fixture("eval", None), &compiled.config, &compiled.forward)
        .expect("the fixture is eval data");
    let batches = (data.samples() / data.batch()) as u64;
    assert_reused(
        batches - 1,
        |b, batch| data.fill_eval_batch(b as usize, batch),
        |b| data.eval_batch(b as usize),
    );
}

/// The allocations one step of [`run_with`] makes, from the difference between a run of
/// `long` and a run of `short` steps, so what a run makes once cancels.
fn per_step(compiled: &CompiledTraining, data: Option<&DenseData>) -> f64 {
    let (short, long) = (4u64, 12u64);
    let count = |steps: u64| {
        let mut compiled = compiled.clone();
        compiled.config.steps = steps;
        allocations(|| {
            run_with(&compiled, data, &BTreeMap::new(), |_| {}).expect("the run");
        })
    };
    (count(long) as f64 - count(short) as f64) / (long - short) as f64
}

/// The runner feeds each dense step from one [`DenseBatch`]: a dense step allocates no more than
/// a synthetic step of the same program, whose inputs are borrowed and never rebuilt. The keys,
/// the data buffers, the shapes and the map are not rebuilt (measured: dense 104.125, synthetic
/// 104.125 allocations per step; 111.125 before the story, 106.125 with a shape per fill).
#[test]
fn the_runner_refills_one_batch_per_run() {
    let (compiled, data) = train_data("runner", None);
    let dense = per_step(&compiled, Some(&data));
    let synthetic = per_step(&compiled, None);
    assert!(
        dense - synthetic <= 0.0,
        "a dense step allocates {dense} times, a synthetic step {synthetic}"
    );
}

/// Step 3 without an order, `B = 2`, `N = 6`: positions 6 and 7 wrap to samples 0 and 1, written
/// out here rather than taken from [`DenseData::step`].
#[test]
fn step_3_without_an_order_holds_samples_0_and_1_literally() {
    let (_, data) = train_data("literal", None);
    let mut batch = DenseBatch::default();
    data.fill_step(3, &mut batch);
    let x = HostTensor::new(vec![2, 3], vec![1.0, 0.0, 0.5, 0.0, 1.0, -0.5]).expect("x");
    let y = HostTensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).expect("y");
    let expected = BTreeMap::from([("pixels".to_string(), x), ("label".to_string(), y)]);
    assert_eq!(batch.inputs(), &expected);
}

/// The losses of the loop the runner ran before this story: a fresh map per step.
fn fresh_losses(compiled: &CompiledTraining, data: &DenseData) -> Vec<f32> {
    let forward = &compiled.forward.program;
    let initial = initialize(forward, compiled.config.seed, Stream::Init).expect("initialized");
    let mut state = ParamState::new(&compiled.training, initial);
    let steps = compiled.config.steps;
    let mut losses: Vec<f32> = (0..steps)
        .map(|k| {
            sgd_step(&compiled.training, &data.step(k), &mut state)
                .expect("the step runs")
                .loss
        })
        .collect();
    let last = evaluate(forward, &data.step(steps), &state.values).expect("the forward runs");
    losses.push(last[0].data()[0]);
    losses
}

fn bits(losses: &[f32]) -> Vec<u32> {
    losses.iter().map(|l| l.to_bits()).collect()
}

#[test]
fn same_loss_bits_as_a_fresh_map_per_step() {
    for (name, order) in [("bits_no_order", None), ("bits_order", Some(&ORDER[..]))] {
        let (compiled, data) = train_data(name, order);
        let run = run_with(&compiled, Some(&data), &BTreeMap::new(), |_| {}).expect("the run");
        assert_eq!(
            bits(&run.losses),
            bits(&fresh_losses(&compiled, &data)),
            "{name}"
        );
    }
}

#[test]
fn same_eval_bits_as_a_fresh_map_per_batch() {
    let (compiled, data) = train_data("eval_bits", Some(&ORDER));
    let run = run_with(&compiled, Some(&data), &BTreeMap::new(), |_| {}).expect("the run");
    let eval = evaluate_dense(&compiled, &data, &run.state.values).expect("the eval");
    let mut loss = 0.0f64;
    for b in 0..data.samples() / data.batch() {
        let values = evaluate_all(
            &compiled.forward.program,
            &data.eval_batch(b),
            &run.state.values,
        )
        .expect("the forward runs");
        loss += f64::from(values.outputs[0].data()[0]);
    }
    let loss = loss / (data.samples() / data.batch()) as f64;
    assert_eq!(eval.loss.to_bits(), loss.to_bits());
}
