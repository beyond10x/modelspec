//! Layer norm and softmax over the last axis on the CPU interpreter (DESIGN.md § 2.2, § 4,
//! § 9 gate 3; story:bert-layer-norm-softmax).
//!
//! `layer_norm(x, gamma, beta)` normalizes each row of the `[N, H]` `x` by its mean and biased
//! variance, `eps = 1e-12`, then scales by `gamma` and shifts by `beta`, both `[H]`. Its gradients
//! are the internal ops `LayerNormGradX(x, gamma, G)` and `LayerNormGradGamma(x, G)`, and
//! `SumRows(G)` for `beta`. `softmax(x)` is `exp(x - max) / sum(exp(x - max))` per row, and its
//! gradient the internal op `SoftmaxGrad(y, G)` of its own result `y`. The finite-difference cases
//! use the harness of `tests/common/finite_difference.rs`.

use std::collections::BTreeMap;

use modelspec_core::ir::tensor::{TensorOp, TensorProgram, TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::{Element, HostTensor, evaluate};

#[allow(dead_code)]
#[path = "common/finite_difference.rs"]
mod finite_difference;

use finite_difference::{Rng, gradient_check, input, mse_to_target, param, seed_of};

/// Rows of the normalized tensor.
const N: u64 = 4;
/// Width of the layer-norm row.
const H: u64 = 5;
/// Width of the softmax row.
const K: u64 = 6;
/// Width of the input of the `matmul` before each operator.
const D: u64 = 3;

/// `mse(layer_norm(matmul(x, w), gamma, beta), t)` with `N = 4` and `H = 5`: the gradients of
/// `w`, `gamma` and `beta` are within `TOL_F64` of central differences in `f64` and `TOL_F32` in
/// `f32`; the gradient of `gamma` is a `LayerNormGradGamma`, that of `beta` a `SumRows`, and the
/// adjoint of `matmul(x, w)` a `LayerNormGradX`.
#[test]
fn layer_norm_gradients_of_w_gamma_and_beta_match_central_differences() {
    let training = gradient_check(
        "layer_norm_gradients_of_w_gamma_and_beta_match_central_differences",
        |b| {
            let x = input(b, "x", &[N, D]);
            let w = param(b, "w", &[D, H]);
            let gamma = param(b, "gamma", &[H]);
            let beta = param(b, "beta", &[H]);
            let xw = b.matmul(x, w).unwrap();
            let y = b.layer_norm(xw, gamma, beta).unwrap();
            assert_eq!(b.type_of(y), Some(&TensorType::f32(&[N, H])));
            mse_to_target(b, y)
        },
    );
    let program = &training.program;
    let gradients: Vec<_> = training.gradients.values().copied().collect();
    assert_eq!(gradients.len(), 3, "a gradient for w, gamma and beta");
    let dw = program.node(gradients[0]).unwrap();
    assert_eq!(dw.op, TensorOp::MatMul, "dW = transpose(x) @ dZ");
    let dz = program.node(dw.operands[1]).unwrap();
    assert_eq!(dz.op, TensorOp::LayerNormGradX, "{dz:?}");
    assert_eq!(dz.ty, TensorType::f32(&[N, H]));
    let dgamma = program.node(gradients[1]).unwrap();
    assert_eq!(dgamma.op, TensorOp::LayerNormGradGamma, "{dgamma:?}");
    assert_eq!(dgamma.ty, TensorType::f32(&[H]));
    let dbeta = program.node(gradients[2]).unwrap();
    assert_eq!(dbeta.op, TensorOp::SumRows, "{dbeta:?}");
}

/// `x` itself trainable: `mse(layer_norm(x, gamma, beta), t)` with all three parameters; the
/// gradient of `x` is the `LayerNormGradX` of the adjoint.
#[test]
fn layer_norm_gradient_of_x_itself_matches_central_differences() {
    gradient_check(
        "layer_norm_gradient_of_x_itself_matches_central_differences",
        |b| {
            let x = param(b, "x", &[N, H]);
            let gamma = param(b, "gamma", &[H]);
            let beta = param(b, "beta", &[H]);
            let y = b.layer_norm(x, gamma, beta).unwrap();
            mse_to_target(b, y)
        },
    );
}

/// `mse(softmax(matmul(x, w)), t)` with `K = 6`: the gradient of `w` is within `TOL_F64` of
/// central differences in `f64` and `TOL_F32` in `f32`, and the adjoint of `matmul(x, w)` is a
/// `SoftmaxGrad` reading the `softmax` result.
#[test]
fn softmax_gradient_of_w_matches_central_differences() {
    let training = gradient_check("softmax_gradient_of_w_matches_central_differences", |b| {
        let x = input(b, "x", &[N, D]);
        let w = param(b, "w", &[D, K]);
        let xw = b.matmul(x, w).unwrap();
        let y = b.softmax(xw).unwrap();
        assert_eq!(b.type_of(y), Some(&TensorType::f32(&[N, K])));
        mse_to_target(b, y)
    });
    let program = &training.program;
    let dw = program
        .node(*training.gradients.values().next().unwrap())
        .unwrap();
    assert_eq!(dw.op, TensorOp::MatMul);
    let dz = program.node(dw.operands[1]).unwrap();
    assert_eq!(dz.op, TensorOp::SoftmaxGrad, "{dz:?}");
    let y = program
        .nodes
        .iter()
        .find(|n| n.op == TensorOp::Softmax)
        .unwrap();
    assert_eq!(dz.operands[0], y.id, "SoftmaxGrad reads the forward result");
}

/// A program of one operator over inputs of the given shapes, its result the one output.
fn program_of(
    shapes: &[(&str, &[u64])],
    build: impl FnOnce(
        &mut TensorProgramBuilder,
        &[modelspec_core::ir::tensor::ValueId],
    ) -> modelspec_core::ir::tensor::ValueId,
) -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let values: Vec<_> = shapes
        .iter()
        .map(|&(name, shape)| b.input(name, TensorType::f32(shape)).unwrap())
        .collect();
    let y = build(&mut b, &values);
    b.output(y).unwrap();
    b.finish()
}

/// Runs `program` in the element type `T` on `f32` inputs widened by `convert`.
fn run_in<T: Element>(
    program: &TensorProgram,
    inputs: &[(&str, &[u64], &[f32])],
    convert: impl Fn(f32) -> T,
) -> HostTensor<T> {
    let inputs: BTreeMap<String, HostTensor<T>> = inputs
        .iter()
        .map(|(name, shape, data)| {
            let data = data.iter().map(|&v| convert(v)).collect();
            (
                (*name).to_owned(),
                HostTensor::new(shape.to_vec(), data).unwrap(),
            )
        })
        .collect();
    evaluate(program, &inputs, &BTreeMap::new())
        .unwrap()
        .remove(0)
}

/// The program `softmax(x)` over an input `x` of shape `shape`.
fn softmax_program(shape: &[u64]) -> TensorProgram {
    program_of(&[("x", shape)], |b, v| b.softmax(v[0]).unwrap())
}

/// `softmax` of the row `[1000, 999, 0]` in `f32` is finite, sums to 1 within `1e-6`, and equals
/// the `f64` result within `1e-6`: the row maximum is subtracted before `exp`, so nothing
/// overflows.
#[test]
fn softmax_large_logits_are_finite_sum_to_1_and_match_f64() {
    let program = softmax_program(&[1, 3]);
    let row = [1000.0f32, 999.0, 0.0];
    let in_f32 = run_in(&program, &[("x", &[1, 3], &row)], |v| v);
    let in_f64 = run_in(&program, &[("x", &[1, 3], &row)], f64::from);
    let got = in_f32.data();
    assert!(got.iter().all(|v| v.is_finite()), "{got:?}");
    let total: f64 = got.iter().map(|&v| f64::from(v)).sum();
    assert!((total - 1.0).abs() <= 1e-6, "sum {total}: {got:?}");
    for (i, (&g, &w)) in got.iter().zip(in_f64.data()).enumerate() {
        assert!(
            (f64::from(g) - w).abs() <= 1e-6,
            "element {i}: f32 {g:e}, f64 {w:e}"
        );
    }
}

/// The program `layer_norm(x, gamma, beta)` over inputs of an `[n, h]` `x` and `[h]` rows.
fn layer_norm_program(n: u64, h: u64) -> TensorProgram {
    program_of(
        &[("x", &[n, h]), ("gamma", &[h]), ("beta", &[h])],
        |b, v| b.layer_norm(v[0], v[1], v[2]).unwrap(),
    )
}

/// Every output row of `layer_norm` with `gamma = 1` and `beta = 0` has mean within `1e-6` of 0
/// and biased variance within `1e-5` of 1, in `f32` and in `f64`. The rows have different
/// offsets and scales, each offset at most 1.5 times the row's scale: in `f32` the bound is a
/// property of the row's conditioning, not of the operator (see
/// [`layer_norm_f32_mean_error_grows_with_the_offset_over_the_spread`]).
#[test]
fn layer_norm_rows_have_mean_0_and_variance_1() {
    let (n, h) = (N, H);
    let mut rng = Rng::new(seed_of("layer_norm_rows_have_mean_0_and_variance_1"));
    let offsets = [0.0f32, 3.0, -7.5, 0.01];
    let scales = [1.0f32, 10.0, 5.0, 0.05];
    let x: Vec<f32> = (0..n as usize)
        .flat_map(|r| {
            let row: Vec<f32> = (0..h)
                .map(|_| offsets[r] + scales[r] * rng.value())
                .collect();
            row
        })
        .collect();
    let gamma = vec![1.0f32; h as usize];
    let beta = vec![0.0f32; h as usize];
    let program = layer_norm_program(n, h);
    let inputs: [(&str, &[u64], &[f32]); 3] = [
        ("x", &[n, h], &x),
        ("gamma", &[h], &gamma),
        ("beta", &[h], &beta),
    ];
    let in_f32: Vec<f64> = run_in(&program, &inputs, |v| v)
        .data()
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let in_f64 = run_in(&program, &inputs, f64::from).data().to_vec();
    for (label, out) in [("f32", in_f32), ("f64", in_f64)] {
        for (r, row) in out.chunks_exact(h as usize).enumerate() {
            let mean = row.iter().sum::<f64>() / h as f64;
            let variance = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / h as f64;
            assert!(
                mean.abs() <= 1e-6,
                "{label} row {r}: mean {mean:e}: {row:?}"
            );
            assert!(
                (variance - 1.0).abs() <= 1e-5,
                "{label} row {r}: variance {variance}: {row:?}"
            );
        }
    }
}

/// Rows whose offset is far above their spread, `-7.5 + 0.05 v` for `v` in `[-1, -0.1] ∪
/// [0.1, 1]`, lose the `1e-6` mean bound in `f32`, as any computation that rounds the row's
/// elements and mean to `f32` does: each is off by up to half an ulp of 7.5, `2.4e-7`, which the
/// row's standard deviation `s` scales up. Over 1000 such rows of `H = 5` the largest mean error
/// of the `f32` evaluation is above `1e-6`, and every row's is within `4 u max|x| / s`, `u` the
/// unit roundoff `2^-24` and `s` evaluated in `f64` (DESIGN.md § 4 records the measured values);
/// the `f64` evaluation stays within `1e-12`.
#[test]
fn layer_norm_f32_mean_error_grows_with_the_offset_over_the_spread() {
    let (n, h) = (1000u64, H);
    let mut rng = Rng::new(seed_of(
        "layer_norm_f32_mean_error_grows_with_the_offset_over_the_spread",
    ));
    let x: Vec<f32> = (0..n * h).map(|_| -7.5 + 0.05 * rng.value()).collect();
    let gamma = vec![1.0f32; h as usize];
    let beta = vec![0.0f32; h as usize];
    let program = layer_norm_program(n, h);
    let inputs: [(&str, &[u64], &[f32]); 3] = [
        ("x", &[n, h], &x),
        ("gamma", &[h], &gamma),
        ("beta", &[h], &beta),
    ];
    let means = |out: Vec<f64>| -> Vec<f64> {
        out.chunks_exact(h as usize)
            .map(|row| (row.iter().sum::<f64>() / h as f64).abs())
            .collect()
    };
    let in_f32 = means(
        run_in(&program, &inputs, |v| v)
            .data()
            .iter()
            .map(|&v| f64::from(v))
            .collect(),
    );
    let in_f64 = means(run_in(&program, &inputs, f64::from).data().to_vec());
    let unit_roundoff = f64::powi(2.0, -24);
    let ratios: Vec<f64> = x
        .chunks_exact(h as usize)
        .zip(&in_f32)
        .map(|(row, &error)| {
            let row: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
            let mean = row.iter().sum::<f64>() / h as f64;
            let s = (row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / h as f64).sqrt();
            let largest = row.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            error / (unit_roundoff * largest / s)
        })
        .collect();
    let largest = |v: &[f64]| v.iter().copied().fold(0.0f64, f64::max);
    let (worst32, worst64, worst_ratio) = (largest(&in_f32), largest(&in_f64), largest(&ratios));
    println!(
        "largest mean of a normalized row: f32 {worst32:e}, f64 {worst64:e}; \
         largest error over u max|x| / s: {worst_ratio}"
    );
    assert!(worst32 > 1e-6, "f32 largest mean {worst32:e}");
    assert!(
        worst_ratio <= 4.0,
        "f32 error over u max|x| / s: {worst_ratio}"
    );
    assert!(worst64 <= 1e-12, "f64 largest mean {worst64:e}");
}

/// `layer_norm` in `f32` is bit for bit the two-pass computation of coordinator decision D32
/// (DESIGN.md § 4): the row mean summed in ascending column order from the first column and
/// divided by `H`; the biased variance the mean of the squared deviations, the first square
/// rounded and each later one fused into the sum (`mul_add`); `1 / sqrt(var + 1e-12)` with
/// `f32::sqrt` and a division; then `(x - mean) * rstd * gamma + beta`, each step rounded.
#[test]
fn layer_norm_f32_bits_follow_the_two_pass_order() {
    let (n, h) = (3u64, 7u64);
    let mut rng = Rng::new(seed_of("layer_norm_f32_bits_follow_the_two_pass_order"));
    let x: Vec<f32> = (0..n * h)
        .map(|i| rng.value() * (1 + i % 4) as f32)
        .collect();
    let gamma: Vec<f32> = (0..h).map(|_| rng.value()).collect();
    let beta: Vec<f32> = (0..h).map(|_| rng.value()).collect();
    let got = run_in(
        &layer_norm_program(n, h),
        &[
            ("x", &[n, h], &x),
            ("gamma", &[h], &gamma),
            ("beta", &[h], &beta),
        ],
        |v| v,
    );
    let mut want = Vec::new();
    for row in x.chunks_exact(h as usize) {
        let mean = row[1..].iter().fold(row[0], |acc, &v| acc + v) / h as f32;
        let first = row[0] - mean;
        let variance = row[1..]
            .iter()
            .fold(first * first, |acc, &v| (v - mean).mul_add(v - mean, acc))
            / h as f32;
        let rstd = 1.0f32 / (variance + 1e-12f32).sqrt();
        for (j, &v) in row.iter().enumerate() {
            want.push((v - mean) * rstd * gamma[j] + beta[j]);
        }
    }
    let got: Vec<u32> = got.data().iter().map(|v| v.to_bits()).collect();
    let want: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, want);
}

/// `softmax` in `f32` is bit for bit the order of coordinator decision D32 (DESIGN.md § 4): the
/// row maximum, `f32::exp` of each element minus it, the exponentials summed in ascending column
/// order from the first column, and each exponential divided by that sum.
#[test]
fn softmax_f32_bits_follow_max_exp_sum_divide() {
    let (n, k) = (3u64, 9u64);
    let mut rng = Rng::new(seed_of("softmax_f32_bits_follow_max_exp_sum_divide"));
    let x: Vec<f32> = (0..n * k).map(|_| rng.value() * 8.0).collect();
    let got = run_in(&softmax_program(&[n, k]), &[("x", &[n, k], &x)], |v| v);
    let mut want = Vec::new();
    for row in x.chunks_exact(k as usize) {
        let max = row[1..]
            .iter()
            .fold(row[0], |m, &v| if v > m { v } else { m });
        let e: Vec<f32> = row.iter().map(|&v| (v - max).exp()).collect();
        let total = e[1..].iter().fold(e[0], |acc, &v| acc + v);
        want.extend(e.iter().map(|&v| v / total));
    }
    let got: Vec<u32> = got.data().iter().map(|v| v.to_bits()).collect();
    let want: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, want);
}

/// `SoftmaxGrad(y, g)` is `y * (g - rowsum(g * y))` per row; on a row whose `y` sums to 1 and
/// whose `g` is constant the gradient is 0, because `softmax` is invariant to adding a constant
/// to a row. The values are exact in binary, so the `f64` evaluation is exact.
#[test]
fn softmax_grad_of_a_constant_adjoint_is_zero() {
    let mut b = TensorProgramBuilder::new();
    let y = b.input("y", TensorType::f32(&[2, 4])).unwrap();
    let g = b.input("g", TensorType::f32(&[2, 4])).unwrap();
    let dx = b.softmax_grad(y, g).unwrap();
    b.output(dx).unwrap();
    let program = b.finish();
    let ys = [0.125f32, 0.125, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25];
    let gs = [2.0f32, 2.0, 2.0, 2.0, 1.0, -1.0, 0.5, 0.0];
    let out = run_in(
        &program,
        &[("y", &[2, 4], &ys), ("g", &[2, 4], &gs)],
        f64::from,
    );
    let out = out.data();
    for &v in &out[..4] {
        assert!(v.abs() <= 1e-15, "{out:?}");
    }
    let d = 0.25 * (1.0 - 1.0 + 0.5 + 0.0);
    for (j, &g) in gs[4..].iter().enumerate() {
        let want = 0.25 * (f64::from(g) - d);
        assert!((out[4 + j] - want).abs() <= 1e-15, "{out:?}");
    }
}
