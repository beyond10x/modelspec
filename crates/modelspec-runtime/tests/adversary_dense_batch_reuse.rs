//! Adversary cases for story:dense-step-reuses-batch-buffers.
//!
//! - A refilled batch must not allocate any part of `x` or `y`, the shape included.
//! - A batch refilled by another [`DenseData`] keeps the previous entries, as documented.
//! - Literal sample rows across an epoch boundary, past the end of an order, after a switch from
//!   a train batch to an eval batch, and after a shrink of `B`: no stale value survives.
//! - The per-step allocation count of the runner is the same on every repetition and in either
//!   order of the dense and the synthetic run.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_runtime::cpu::HostTensor;
use modelspec_runtime::train::data::{DenseBatch, DenseData, dense, dense_eval};
use modelspec_runtime::train::runner::run_with;

const LOG: usize = 1 << 14;

static HANDED_OUT: [AtomicUsize; LOG] = [const { AtomicUsize::new(0) }; LOG];
static COUNT: AtomicUsize = AtomicUsize::new(0);
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

fn handed_out(f: impl FnOnce()) -> BTreeSet<usize> {
    let _window = WINDOW.lock().unwrap_or_else(|p| p.into_inner());
    COUNT.store(0, Ordering::Relaxed);
    RECORDING.with(|r| r.set(true));
    f();
    RECORDING.with(|r| r.set(false));
    let count = COUNT.load(Ordering::Relaxed);
    assert!(count <= LOG);
    (0..count)
        .map(|i| HANDED_OUT[i].load(Ordering::Relaxed))
        .collect()
}

fn allocations(f: impl FnOnce()) -> usize {
    let _window = WINDOW.lock().unwrap_or_else(|p| p.into_inner());
    COUNT.store(0, Ordering::Relaxed);
    RECORDING.with(|r| r.set(true));
    f();
    RECORDING.with(|r| r.set(false));
    COUNT.load(Ordering::Relaxed)
}

/// A linear model with batch `b` and declarations `input`, `target`.
fn model(b: usize, input: &str, target: &str) -> String {
    format!(
        "\
model Lin {{
    param w: Tensor<f32, [3, 2]> = xavier_uniform();

    fn forward({input}: Tensor<f32, [{b}, 3]>)
        -> Tensor<f32, [{b}, 2]> {{
        return matmul({input}, w);
    }}
}}

train Fit {{
    model = Lin;
    input {input}: Tensor<f32, [{b}, 3]>;
    target {target}: Tensor<f32, [{b}, 2]>;
    loss = mse(model.forward({input}), {target});
    optimizer = sgd(lr = 0.1);
    steps = 7;
    seed = 42;
    device = cpu;
}}
"
    )
}

const ROWS_X: [[f32; 3]; 6] = [
    [1.0, 0.0, 0.5],
    [0.0, 1.0, -0.5],
    [1.0, 1.0, 0.0],
    [2.0, 0.0, 1.0],
    [-1.0, 0.5, 0.25],
    [0.0, -2.0, 1.5],
];
const ROWS_Y: [[f32; 2]; 6] = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.0, 3.0],
    [2.0, 0.0],
    [0.5, 0.5],
];

fn compiled(b: usize, input: &str, target: &str) -> CompiledTraining {
    compile_training(
        "adversary_dense_batch_reuse.mspec",
        &model(b, input, target),
    )
    .expect("the fixture compiles")
}

fn write_f32(path: &Path, values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).expect("written");
}

/// A fresh directory holding the samples `rows` of `input`/`target`, and `order` if given.
fn fixture(
    name: &str,
    input: &str,
    target: &str,
    rows: &[usize],
    order: Option<&[u32]>,
) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary_dense_batch_reuse")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("removed");
    }
    std::fs::create_dir_all(&dir).expect("created");
    let x: Vec<f32> = rows.iter().flat_map(|&r| ROWS_X[r]).collect();
    let y: Vec<f32> = rows.iter().flat_map(|&r| ROWS_Y[r]).collect();
    write_f32(&dir.join(format!("{input}.f32")), &x);
    write_f32(&dir.join(format!("{target}.f32")), &y);
    if let Some(order) = order {
        let bytes: Vec<u8> = order.iter().flat_map(|i| i.to_le_bytes()).collect();
        std::fs::write(dir.join("order.u32"), bytes).expect("written");
    }
    dir
}

const ALL: [usize; 6] = [0, 1, 2, 3, 4, 5];

fn train(name: &str, b: usize, input: &str, target: &str, order: Option<&[u32]>) -> DenseData {
    let c = compiled(b, input, target);
    dense(
        &fixture(name, input, target, &ALL, order),
        &c.config,
        &c.forward,
    )
    .expect("dense")
}

/// The expected inputs of the fixture rows `rows`, written from the row tables.
fn expected(input: &str, target: &str, rows: &[usize]) -> BTreeMap<String, HostTensor<f32>> {
    let b = rows.len() as u64;
    let x: Vec<f32> = rows.iter().flat_map(|&r| ROWS_X[r]).collect();
    let y: Vec<f32> = rows.iter().flat_map(|&r| ROWS_Y[r]).collect();
    BTreeMap::from([
        (input.to_string(), HostTensor::new(vec![b, 3], x).unwrap()),
        (target.to_string(), HostTensor::new(vec![b, 2], y).unwrap()),
    ])
}

/// Acceptance no-per-step-allocation: after the first fill, a fill allocates no part of `x` or
/// `y`. The shape `Vec` of each tensor is part of that tensor.
#[test]
fn a_refill_allocates_no_shape_of_x_or_y() {
    let data = train("shape", 2, "pixels", "label", None);
    let mut batch = DenseBatch::default();
    data.fill_step(0, &mut batch);
    for k in 1..=4 {
        let allocated = handed_out(|| data.fill_step(k, &mut batch));
        for (name, tensor) in batch.inputs() {
            assert!(
                !allocated.contains(&(tensor.shape().as_ptr() as usize)),
                "step {k} allocated the shape of {name} ({} allocations in the step)",
                allocated.len()
            );
        }
    }
}

/// The documented contract of [`DenseBatch`]: "A batch is filled by one [`DenseData`] only"
/// (`crates/modelspec-runtime/src/train/data.rs`, the `DenseBatch` doc comment). A batch last
/// filled by another data keeps that data's entries beside the new ones; this case pins that
/// behaviour, so a change to the contract is made on purpose (correction round 1, finding 2).
#[test]
fn a_batch_refilled_by_other_data_keeps_the_previous_entries() {
    let first = train("other_a", 2, "pixels", "label", None);
    let second = train("other_b", 2, "x", "y", None);
    let mut batch = DenseBatch::default();
    first.fill_step(0, &mut batch);
    second.fill_step(1, &mut batch);
    let mut expected = first.step(0);
    expected.extend(second.step(1));
    assert_eq!(
        batch.inputs(),
        &expected,
        "a DenseBatch is filled by one DenseData only (the DenseBatch doc comment in data.rs); \
         refilled by another, it keeps the previous entries"
    );
}

/// N = 6, B = 4, no order: step 1 straddles the epoch boundary, rows 4, 5, 0, 1.
#[test]
fn a_step_across_the_epoch_boundary_holds_the_wrapped_rows() {
    let data = train("epoch", 4, "pixels", "label", None);
    let mut batch = DenseBatch::default();
    data.fill_step(0, &mut batch);
    assert_eq!(batch.inputs(), &expected("pixels", "label", &[0, 1, 2, 3]));
    data.fill_step(1, &mut batch);
    assert_eq!(batch.inputs(), &expected("pixels", "label", &[4, 5, 0, 1]));
    data.fill_step(2, &mut batch);
    assert_eq!(batch.inputs(), &expected("pixels", "label", &[2, 3, 4, 5]));
}

/// An order of 15 indices for 7 steps of B = 2: step 7 reads positions 14 and 15 → 0.
#[test]
fn a_step_past_the_end_of_the_order_wraps() {
    let order: [u32; 15] = [5, 2, 0, 3, 4, 1, 1, 1, 3, 0, 2, 5, 4, 4, 3];
    let data = train("order_wrap", 2, "pixels", "label", Some(&order));
    let mut batch = DenseBatch::default();
    for k in 0..7u64 {
        data.fill_step(k, &mut batch);
        let rows = [
            order[2 * k as usize] as usize,
            order[2 * k as usize + 1] as usize,
        ];
        assert_eq!(
            batch.inputs(),
            &expected("pixels", "label", &rows),
            "step {k}"
        );
    }
    data.fill_step(7, &mut batch);
    assert_eq!(batch.inputs(), &expected("pixels", "label", &[3, 5]));
}

/// A train batch and then an eval batch of a different sample set, into one `DenseBatch`.
#[test]
fn a_train_batch_then_an_eval_batch_holds_the_eval_rows() {
    let c = compiled(2, "pixels", "label");
    let train_data = train(
        "switch_train",
        2,
        "pixels",
        "label",
        Some(&[5u32, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 5, 4]),
    );
    let eval_rows = [3, 1, 4, 0];
    let eval = dense_eval(
        &fixture("switch_eval", "pixels", "label", &eval_rows, None),
        &c.config,
        &c.forward,
    )
    .expect("eval");
    let mut batch = DenseBatch::default();
    train_data.fill_step(1, &mut batch);
    assert_eq!(batch.inputs(), &expected("pixels", "label", &[3, 2]));
    eval.fill_eval_batch(1, &mut batch);
    assert_eq!(batch.inputs(), &expected("pixels", "label", &[4, 0]));
    eval.fill_eval_batch(0, &mut batch);
    assert_eq!(batch.inputs(), &expected("pixels", "label", &[3, 1]));
}

/// B = 4 then B = 2 with the same declaration names: no value of the longer fill survives.
#[test]
fn a_shorter_fill_leaves_no_stale_value() {
    let wide = train("wide", 4, "pixels", "label", None);
    let narrow = train("narrow", 2, "pixels", "label", None);
    let mut batch = DenseBatch::default();
    wide.fill_step(1, &mut batch);
    narrow.fill_step(2, &mut batch);
    assert_eq!(batch.inputs(), &expected("pixels", "label", &[4, 5]));
    wide.fill_step(0, &mut batch);
    assert_eq!(batch.inputs(), &expected("pixels", "label", &[0, 1, 2, 3]));
}

/// The public default: an empty batch, equal to another default.
#[test]
fn the_default_batch_is_empty() {
    assert!(DenseBatch::default().inputs().is_empty());
    assert_eq!(DenseBatch::default(), DenseBatch::default());
}

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

/// The figure the unit's runner test bounds is the same on every repetition, either order.
#[test]
fn the_per_step_allocation_count_is_deterministic() {
    let c = compiled(2, "pixels", "label");
    let data = dense(
        &fixture("determinism", "pixels", "label", &ALL, None),
        &c.config,
        &c.forward,
    )
    .expect("dense");
    let synthetic_first = per_step(&c, None);
    let dense_first: Vec<f64> = (0..5).map(|_| per_step(&c, Some(&data))).collect();
    let synthetic: Vec<f64> = (0..5).map(|_| per_step(&c, None)).collect();
    eprintln!("dense {dense_first:?} synthetic {synthetic:?} synthetic-first {synthetic_first}");
    assert!(
        dense_first.iter().all(|&d| d == dense_first[0]),
        "{dense_first:?}"
    );
    assert!(
        synthetic.iter().all(|&s| s == synthetic_first),
        "{synthetic:?}"
    );
    assert_eq!(
        dense_first[0] - synthetic_first,
        0.0,
        "a refill allocates no shape (correction round 1, finding 1)"
    );
}
