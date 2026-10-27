//! Adversary pass 1 of story:transformer-block-benchmark: the bench's result line, its
//! matmul-only FLOP rule recounted product by product from `examples/transformer_block.mspec`
//! independently of the closed form in `benches/transformer_block/block.rs`, and the synthetic
//! runner's key-mask rule (`train::data::synthetic`) on programs the unit's own cases do not
//! build.

#[allow(dead_code)]
#[path = "../benches/mlp_sweep/sweep.rs"]
mod sweep;

#[allow(dead_code)]
#[path = "../benches/transformer_block/block.rs"]
mod block;

use std::collections::BTreeMap;

use block::{BlockResult, Outcome, Point};
use modelspec_core::compile::compile_training;
use modelspec_core::ir::tensor::{TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::{HostTensor, evaluate, evaluate_all};
use modelspec_runtime::train::data::{key_masks, synthetic};
use modelspec_runtime::train::init::{Stream, initialize, standard_normal};
use sweep::TimedRun;

/// DESIGN.md § 9 gate 5, the run tolerance the story's first-step loss check is held to:
/// `|g − c| ≤ 1e-4 + 1e-3·|c|`.
fn within_run_tolerance(g: f64, c: f64) -> bool {
    (g - c).abs() <= 1e-4 + 1e-3 * c.abs()
}

/// The mean of the squared differences of two `f32` tensors, summed in `f64`.
fn mse_f64(a: &[f32], b: &[f32]) -> f64 {
    let sum: f64 = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| {
            let d = f64::from(a) - f64::from(b);
            d * d
        })
        .sum();
    sum / a.len() as f64
}

/// ADV-TB-5, at the example's batch 128: the CPU interpreter's step-0 loss of the example, on
/// its synthetic batch and initial parameters, against the same mean of the same `f32`
/// prediction and target summed in `f64`. The PyTorch opponent's loss on this checkpoint and
/// batch is the `f64` value to 8 digits, so this is the first-step loss check of the story at
/// the batch the bench runs largest. Slow in a debug build (two forward passes over 16,384 rows).
#[test]
fn adv_tb_5_the_cpu_step_0_loss_at_batch_128_is_within_the_run_tolerance_of_its_f64_value() {
    let point = Point { batch: 128 };
    let source = point.source().expect("a source");
    let compiled = compile_training(&point.file_name(), &source).expect("compiles");
    let data = synthetic(&compiled.config, &compiled.forward).expect("synthetic data");
    let forward = &compiled.forward.program;
    let params = initialize(forward, compiled.config.seed, Stream::Init).expect("initial values");
    let evaluation = evaluate_all(forward, &data.inputs, &params).expect("evaluates");
    let loss = f64::from(evaluation.outputs[0].data()[0]);
    let prediction = &evaluation.values[&compiled.forward.prediction];
    let exact = mse_f64(prediction.data(), data.inputs["y"].data());
    assert!(
        within_run_tolerance(loss, exact),
        "CPU loss {loss} against {exact} summed in f64: |diff| {:.3e} > bound {:.3e}",
        (loss - exact).abs(),
        1e-4 + 1e-3 * exact.abs()
    );
}

/// ADV-TB-5, the operator alone: `mse` of a `[16384, 128]` tensor of standard normal draws
/// against zeros, the element count of the example's prediction at batch 128, against the same
/// mean summed in `f64`.
#[test]
fn adv_tb_5_the_cpu_mse_of_2_million_elements_is_within_the_run_tolerance_of_its_f64_value() {
    let ty = TensorType::f32(&[16_384, 128]);
    let mut b = TensorProgramBuilder::new();
    let a = b.input("a", ty.clone()).unwrap();
    let z = b.input("z", ty.clone()).unwrap();
    let loss = b.mse(a, z).unwrap();
    b.output(loss).unwrap();
    let program = b.finish();
    let draws = standard_normal(42, Stream::Data, "a", &ty).expect("draws");
    let zeros = HostTensor::new(vec![16_384, 128], vec![0.0f32; 16_384 * 128]).unwrap();
    let inputs = BTreeMap::from([
        ("a".to_owned(), draws.clone()),
        ("z".to_owned(), zeros.clone()),
    ]);
    let outputs = evaluate(&program, &inputs, &BTreeMap::new()).expect("evaluates");
    let loss = f64::from(outputs[0].data()[0]);
    let exact = mse_f64(draws.data(), zeros.data());
    assert!(
        within_run_tolerance(loss, exact),
        "CPU mse {loss} against {exact} summed in f64: |diff| {:.3e} > bound {:.3e}",
        (loss - exact).abs(),
        1e-4 + 1e-3 * exact.abs()
    );
}

fn timed(us_per_step: f64) -> TimedRun {
    TimedRun {
        us_per_step,
        seconds: 5.0,
        span_unix_ms: [0, 5000],
    }
}

/// ADV-TB-1. The unit's `tokens_per_second_and_gflops_are_over_the_median_step_time` gives three
/// equal runs, whose mean, median, first and last are one number, so it stays green when the
/// figures are taken over any of them. Three distinct runs out of order: median 2000, mean 3000,
/// first 6000, last 2000 only by coincidence of the middle.
#[test]
fn adv_tb_1_tokens_per_second_and_gflops_are_over_the_median_of_distinct_runs() {
    for runs in [
        vec![timed(6000.0), timed(1000.0), timed(2000.0)],
        vec![timed(2000.0), timed(6000.0), timed(1000.0)],
    ] {
        let result =
            BlockResult::measured("modelspec-vulkan", Point { batch: 8 }, 1000, runs, None);
        let Outcome::Measured {
            median_us_per_step,
            tokens_per_s,
            gflops,
            ..
        } = result.outcome
        else {
            panic!("measured");
        };
        assert_eq!(median_us_per_step, 2000.0);
        assert_eq!(tokens_per_s, 512_000.0);
        assert_eq!(gflops, 2_717_908_992.0 / 2_000_000.0);
    }
}

/// One matrix product `[m, k] x [k, n]`, `2·m·k·n` FLOP.
fn product(m: u64, k: u64, n: u64) -> u64 {
    2 * m * k * n
}

/// The block's step recounted as a list of products, read off the `.mspec` file and the
/// gradient rules of a matmul (`dA = dC·Bᵀ`, `dB = Aᵀ·dC`) and of attention
/// (`S = Q·Kᵀ`, `O = P·V`: `dP = dO·Vᵀ`, `dV = Pᵀ·dO`, `dQ = dS·K`, `dK = dSᵀ·Q`), per head
/// and sequence. Nothing differentiates the batch `x`, so layer 0's Q, K and V projections
/// have no input adjoint.
fn recounted(batch: u64) -> u64 {
    let (s, h, f, heads) = (128u64, 128u64, 512u64, 2u64);
    let d = h / heads;
    let n = batch * s;
    let mut total = 0;
    for layer in 0..2 {
        for projection in ["q", "k", "v", "o"] {
            total += product(n, h, h);
            total += product(h, n, h);
            let input_is_x = layer == 0 && projection != "o";
            if !input_is_x {
                total += product(n, h, h);
            }
        }
        for (k, out) in [(h, f), (f, h)] {
            total += product(n, k, out);
            total += product(k, n, out);
            total += product(n, out, k);
        }
        let per_head_and_sequence = [
            product(s, d, s),
            product(s, s, d),
            product(s, d, s),
            product(s, s, d),
            product(s, s, d),
            product(s, s, d),
        ];
        total += batch * heads * per_head_and_sequence.iter().sum::<u64>();
    }
    total
}

/// The closed form `N·(42·H² + 24·H·F + 24·S·H)` against the product-by-product recount, at the
/// bench's batches and at batches that are not a multiple of 8.
#[test]
fn the_closed_form_is_the_product_by_product_recount_at_every_batch() {
    for batch in [1, 3, 8, 32, 128, 1000] {
        assert_eq!(Point { batch }.flop_per_step(), recounted(batch), "{batch}");
    }
}

/// A program reading two inputs only as the key masks of two attentions.
const TWO_MASKS: &str = "model M {
    param wq: Tensor<f32, [4, 4]> = xavier_uniform();
    fn forward(x: Tensor<f32, [6, 4]>, m1: Tensor<f32, [2, 3]>, m2: Tensor<f32, [2, 3]>) -> Tensor<f32, [6, 4]> {
        let q = matmul(x, wq);
        let a = attention(q, q, x, m1, heads = 2);
        return attention(a, a, a, m2, heads = 2);
    }
}
train T {
    model = M;
    input x: Tensor<f32, [6, 4]>;
    input m1: Tensor<f32, [2, 3]>;
    input m2: Tensor<f32, [2, 3]>;
    target y: Tensor<f32, [6, 4]>;
    loss = mse(model.forward(x, m1, m2), y);
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 7;
    device = cpu;
}
";

/// Two inputs each read only as a key mask are both all ones, and the one other input is drawn.
#[test]
fn two_key_mask_inputs_are_both_all_ones() {
    let compiled = compile_training("m.mspec", TWO_MASKS).expect("compiles");
    let masks = key_masks(&compiled.forward);
    assert_eq!(masks.iter().collect::<Vec<_>>(), ["m1", "m2"]);
    let data = synthetic(&compiled.config, &compiled.forward).expect("synthetic data");
    assert_eq!(
        data.inputs.keys().collect::<Vec<_>>(),
        ["m1", "m2", "x", "y"]
    );
    for mask in ["m1", "m2"] {
        assert_eq!(data.inputs[mask].data(), &[1.0; 6], "{mask}");
    }
    assert!(data.inputs["x"].data().iter().any(|&v| v != 1.0));
}

/// The example's key mask is the only key mask, and its batch is drawn: the rule finds the
/// example's `key_mask` and nothing else, at every batch the bench substitutes.
#[test]
fn the_example_s_key_mask_is_its_only_key_mask_at_every_batch() {
    for point in block::points() {
        let source = point.source().expect("a source");
        let compiled = compile_training(&point.file_name(), &source).expect("compiles");
        let masks = key_masks(&compiled.forward);
        assert_eq!(
            masks.iter().collect::<Vec<_>>(),
            ["key_mask"],
            "{}",
            point.batch
        );
    }
}
