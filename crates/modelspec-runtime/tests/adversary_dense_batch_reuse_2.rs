//! Adversary pass 2 for story:dense-step-reuses-batch-buffers, against the refill correction.
//!
//! - After the first fill, a fill allocates nothing at all: no key, shape, data buffer or map
//!   node, across the epoch wrap, past the end of an order, and for eval batches.
//! - A refill to a shape of the same element count in other dimensions takes the new shape and
//!   the new rows, and the old shape comes back on the next fill.
//! - A resumed dense run, whose batch is first filled at step `start`, gives the losses the full
//!   run gives from that step on.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_runtime::cpu::HostTensor;
use modelspec_runtime::train::data::{DenseBatch, DenseData, dense, dense_eval};
use modelspec_runtime::train::runner::{run_from, run_with};

static COUNT: AtomicUsize = AtomicUsize::new(0);
static WINDOW: Mutex<()> = Mutex::new(());

thread_local! {
    static RECORDING: Cell<bool> = const { Cell::new(false) };
}

struct Counting;

fn note() {
    if RECORDING.try_with(Cell::get).unwrap_or(false) {
        COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: every call is forwarded to `System` unchanged; `note` only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note();
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocations(f: impl FnOnce()) -> usize {
    let _window = WINDOW.lock().unwrap_or_else(|p| p.into_inner());
    COUNT.store(0, Ordering::Relaxed);
    RECORDING.with(|r| r.set(true));
    f();
    RECORDING.with(|r| r.set(false));
    COUNT.load(Ordering::Relaxed)
}

/// A linear model `[b, f] x [f, o]` with declarations `pixels`, `label`.
fn model(b: usize, f: usize, o: usize, steps: u64) -> String {
    format!(
        "\
model Lin {{
    param w: Tensor<f32, [{f}, {o}]> = xavier_uniform();

    fn forward(pixels: Tensor<f32, [{b}, {f}]>)
        -> Tensor<f32, [{b}, {o}]> {{
        return matmul(pixels, w);
    }}
}}

train Fit {{
    model = Lin;
    input pixels: Tensor<f32, [{b}, {f}]>;
    target label: Tensor<f32, [{b}, {o}]>;
    loss = mse(model.forward(pixels), label);
    optimizer = sgd(lr = 0.1);
    steps = {steps};
    seed = 42;
    device = cpu;
}}
"
    )
}

fn compiled(b: usize, f: usize, o: usize, steps: u64) -> CompiledTraining {
    compile_training(
        "adversary_dense_batch_reuse_2.mspec",
        &model(b, f, o, steps),
    )
    .expect("the fixture compiles")
}

fn write_f32(path: &Path, values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).expect("written");
}

fn fixture(name: &str, x: &[f32], y: &[f32], order: Option<&[u32]>) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary_dense_batch_reuse_2")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("removed");
    }
    std::fs::create_dir_all(&dir).expect("created");
    write_f32(&dir.join("pixels.f32"), x);
    write_f32(&dir.join("label.f32"), y);
    if let Some(order) = order {
        let bytes: Vec<u8> = order.iter().flat_map(|i| i.to_le_bytes()).collect();
        std::fs::write(dir.join("order.u32"), bytes).expect("written");
    }
    dir
}

/// Five samples of 3 features and 2 outputs: N = 5 is not a multiple of B = 2.
const X5: [f32; 15] = [
    1.0, 0.0, 0.5, 0.0, 1.0, -0.5, 1.0, 1.0, 0.0, 2.0, 0.0, 1.0, -1.0, 0.5, 0.25,
];
const Y5: [f32; 10] = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 3.0, 2.0, 0.0];

fn data_2x3(name: &str, steps: u64, order: Option<&[u32]>) -> (CompiledTraining, DenseData) {
    let c = compiled(2, 3, 2, steps);
    let d = dense(&fixture(name, &X5, &Y5, order), &c.config, &c.forward).expect("dense");
    (c, d)
}

fn tensor(shape: &[u64], data: &[f32]) -> HostTensor<f32> {
    HostTensor::new(shape.to_vec(), data.to_vec()).expect("a tensor")
}

fn inputs(x: HostTensor<f32>, y: HostTensor<f32>) -> BTreeMap<String, HostTensor<f32>> {
    BTreeMap::from([("pixels".to_string(), x), ("label".to_string(), y)])
}

/// Acceptance no-per-step-allocation, read as strictly as it can be: after the first fill a
/// fill makes no allocation of any kind, including at the epoch wrap of N = 5, B = 2 and past
/// the end of an order.
#[test]
fn a_refill_after_the_first_allocates_nothing() {
    let (_, plain) = data_2x3("count_plain", 7, None);
    let order: [u32; 14] = [4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 4, 3, 2, 1];
    let (_, ordered) = data_2x3("count_order", 7, Some(&order));
    for (what, data) in [("no order", &plain), ("order", &ordered)] {
        let mut batch = DenseBatch::default();
        let first = allocations(|| data.fill_step(0, &mut batch));
        assert!(
            first >= 4,
            "{what}: the first fill made only {first} allocation(s)"
        );
        for k in 1..=9u64 {
            let n = allocations(|| data.fill_step(k, &mut batch));
            assert_eq!(n, 0, "{what}: step {k} made {n} allocation(s)");
        }
    }
    let c = compiled(2, 3, 2, 7);
    let eval = dense_eval(
        &fixture("count_eval", &X5[..12], &Y5[..8], None),
        &c.config,
        &c.forward,
    )
    .expect("eval");
    let mut batch = DenseBatch::default();
    eval.fill_eval_batch(0, &mut batch);
    for b in [1usize, 0, 1] {
        let n = allocations(|| eval.fill_eval_batch(b, &mut batch));
        assert_eq!(n, 0, "eval batch {b} made {n} allocation(s)");
    }
}

/// N = 5, B = 2, no order: step 2 reads samples 4 and 0 (positions 4, 5 mod 5), written out.
#[test]
fn the_last_step_of_an_uneven_epoch_wraps_to_sample_0() {
    let (_, data) = data_2x3("uneven", 7, None);
    let mut batch = DenseBatch::default();
    for k in 0..=2u64 {
        data.fill_step(k, &mut batch);
    }
    let expected = inputs(
        tensor(&[2, 3], &[-1.0, 0.5, 0.25, 1.0, 0.0, 0.5]),
        tensor(&[2, 2], &[2.0, 0.0, 1.0, 0.0]),
    );
    assert_eq!(batch.inputs(), &expected);
}

/// `x` of shape [2, 3] and then [3, 2], six elements either way, under the same name: the second
/// fill must carry the shape [3, 2] (and `y` its [3, 2] of six elements after four), and the
/// third fill the shape [2, 3] again.
#[test]
fn a_refill_to_the_same_count_in_other_dims_takes_the_new_shape() {
    let (_, wide) = data_2x3("dims_wide", 7, None);
    let c = compiled(3, 2, 2, 7);
    let tall_x = [0.5, -1.0, 2.0, 0.0, 1.5, 3.0];
    let tall_y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tall = dense(
        &fixture("dims_tall", &tall_x, &tall_y, None),
        &c.config,
        &c.forward,
    )
    .expect("dense");
    let mut batch = DenseBatch::default();
    wide.fill_step(0, &mut batch);
    tall.fill_step(0, &mut batch);
    assert_eq!(
        batch.inputs(),
        &inputs(tensor(&[3, 2], &tall_x), tensor(&[3, 2], &tall_y))
    );
    wide.fill_step(1, &mut batch);
    assert_eq!(
        batch.inputs(),
        &inputs(
            tensor(&[2, 3], &[1.0, 1.0, 0.0, 2.0, 0.0, 1.0]),
            tensor(&[2, 2], &[1.0, 1.0, 0.0, 3.0]),
        )
    );
}

/// A resumed run fills its batch first at step `start`; from there its losses are the full
/// run's, bit for bit, with and without an order.
#[test]
fn a_resumed_dense_run_gives_the_full_runs_losses() {
    let order: [u32; 14] = [4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 4, 3, 2, 1];
    for (name, order) in [("resume_plain", None), ("resume_order", Some(&order[..]))] {
        let (c, data) = data_2x3(name, 7, order);
        let full = run_with(&c, Some(&data), &BTreeMap::new(), |_| {}).expect("the full run");
        let mut at3 = None;
        let mut first = c.clone();
        first.config.steps = 3;
        run_from(
            &first,
            Some(&data),
            &BTreeMap::new(),
            0,
            |_| {},
            |k, state| {
                if k == 3 {
                    at3 = Some(state.values.clone());
                }
                Ok(())
            },
        )
        .expect("the first part");
        let mut rest = c.clone();
        rest.config.steps = 4;
        let resumed = run_from(
            &rest,
            Some(&data),
            &at3.expect("a state after 3 updates"),
            3,
            |_| {},
            |_, _| Ok(()),
        )
        .expect("the resumed part");
        let bits = |l: &[f32]| l.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&resumed.losses), bits(&full.losses[3..]), "{name}");
    }
}
