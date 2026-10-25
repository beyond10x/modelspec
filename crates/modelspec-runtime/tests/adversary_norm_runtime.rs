//! Adversary pass 1 on story:bert-layer-norm-softmax (DESIGN.md § 2.2, § 4), on the CPU
//! interpreter: the norm and softmax ops at the edges of their domain, the summation order of
//! every row reduction inside the gradient ops, the two-pass variance, the autodiff wiring of a
//! softmax result read twice, and the `f32` mean bound DESIGN.md § 4 states, at BERT's width.

use std::collections::BTreeMap;

use modelspec_core::autodiff::differentiate;
use modelspec_core::ir::tensor::{TensorProgram, TensorProgramBuilder, TensorType, ValueId};
use modelspec_runtime::cpu::{Element, HostTensor, evaluate, evaluate_all_with_ids};

#[allow(dead_code)]
#[path = "common/finite_difference.rs"]
mod finite_difference;

use finite_difference::{Point, Rng, gradient_check, input, mse_to_target, param, seed_of};

/// `eps` of `layer_norm`, as DESIGN.md § 4 states it: `1e-12`, an `f32` constant.
const EPS: f32 = 1e-12;

/// A program over `f32` inputs of the given shapes whose one output is `build`'s value.
fn program_of(
    shapes: &[(&str, &[u64])],
    build: impl FnOnce(&mut TensorProgramBuilder, &[ValueId]) -> ValueId,
) -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let values: Vec<ValueId> = shapes
        .iter()
        .map(|&(name, shape)| b.input(name, TensorType::f32(shape)).unwrap())
        .collect();
    let y = build(&mut b, &values);
    b.output(y).unwrap();
    b.finish()
}

/// The one output of `program` evaluated in `T`.
fn run<T: Element>(program: &TensorProgram, inputs: &[(&str, &[u64], Vec<T>)]) -> Vec<T> {
    let inputs: BTreeMap<String, HostTensor<T>> = inputs
        .iter()
        .map(|(name, shape, data)| {
            (
                (*name).to_owned(),
                HostTensor::new(shape.to_vec(), data.clone()).unwrap(),
            )
        })
        .collect();
    evaluate(program, &inputs, &BTreeMap::new())
        .unwrap()
        .remove(0)
        .data()
        .to_vec()
}

fn layer_norm_program(n: u64, h: u64) -> TensorProgram {
    program_of(
        &[("x", &[n, h]), ("gamma", &[h]), ("beta", &[h])],
        |b, v| b.layer_norm(v[0], v[1], v[2]).unwrap(),
    )
}

fn grad_x_program(n: u64, h: u64) -> TensorProgram {
    program_of(
        &[("x", &[n, h]), ("gamma", &[h]), ("g", &[n, h])],
        |b, v| b.layer_norm_grad_x(v[0], v[1], v[2]).unwrap(),
    )
}

fn grad_gamma_program(n: u64, h: u64) -> TensorProgram {
    program_of(&[("x", &[n, h]), ("g", &[n, h])], |b, v| {
        b.layer_norm_grad_gamma(v[0], v[1]).unwrap()
    })
}

fn softmax_program(n: u64, k: u64) -> TensorProgram {
    program_of(&[("x", &[n, k])], |b, v| b.softmax(v[0]).unwrap())
}

fn softmax_grad_program(n: u64, k: u64) -> TensorProgram {
    program_of(&[("y", &[n, k]), ("g", &[n, k])], |b, v| {
        b.softmax_grad(v[0], v[1]).unwrap()
    })
}

fn sum_rows_program(n: u64, h: u64) -> TensorProgram {
    program_of(&[("g", &[n, h])], |b, v| b.sum_rows(v[0]).unwrap())
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

// ---------------------------------------------------------------------------------------------
// Hand-written `f32` references: every reduction over the indices of `order`, a plain sum from
// its first term, a sum of products rounding its first product and fusing each later one
// (`mul_add`, ADR-009). `rev` reverses the order, to show that a test is sensitive to it.
// ---------------------------------------------------------------------------------------------

fn order(len: usize, rev: bool) -> Vec<usize> {
    if rev {
        (0..len).rev().collect()
    } else {
        (0..len).collect()
    }
}

fn sum_in(v: &[f32], rev: bool) -> f32 {
    let idx = order(v.len(), rev);
    idx[1..].iter().fold(v[idx[0]], |acc, &j| acc + v[j])
}

fn fused_in(a: &[f32], b: &[f32], rev: bool) -> f32 {
    let idx = order(a.len(), rev);
    idx[1..]
        .iter()
        .fold(a[idx[0]] * b[idx[0]], |acc, &j| a[j].mul_add(b[j], acc))
}

fn moments_in(row: &[f32], rev: bool) -> (f32, f32) {
    let h = row.len() as f32;
    let mean = sum_in(row, rev) / h;
    let d: Vec<f32> = row.iter().map(|&x| x - mean).collect();
    let var = fused_in(&d, &d, rev) / h;
    (mean, 1.0f32 / (var + EPS).sqrt())
}

fn ref_layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], rev: bool) -> Vec<f32> {
    let h = gamma.len();
    let mut out = Vec::new();
    for row in x.chunks_exact(h) {
        let (mean, rstd) = moments_in(row, rev);
        for j in 0..h {
            out.push((row[j] - mean) * rstd * gamma[j] + beta[j]);
        }
    }
    out
}

fn ref_grad_x(x: &[f32], gamma: &[f32], g: &[f32], rev: bool) -> Vec<f32> {
    let h = gamma.len();
    let width = h as f32;
    let mut out = Vec::new();
    for (row, g_row) in x.chunks_exact(h).zip(g.chunks_exact(h)) {
        let (mean, rstd) = moments_in(row, rev);
        let xhat: Vec<f32> = row.iter().map(|&v| (v - mean) * rstd).collect();
        let gx: Vec<f32> = (0..h).map(|j| g_row[j] * gamma[j]).collect();
        let a = sum_in(&gx, rev) / width;
        let b = fused_in(&gx, &xhat, rev) / width;
        out.extend((0..h).map(|j| rstd * ((gx[j] - a) - xhat[j] * b)));
    }
    out
}

fn ref_grad_gamma(x: &[f32], g: &[f32], h: usize, rev: bool) -> Vec<f32> {
    let rows: Vec<(&[f32], &[f32])> = x.chunks_exact(h).zip(g.chunks_exact(h)).collect();
    let mut out: Vec<f32> = Vec::new();
    for (k, &r) in order(rows.len(), rev).iter().enumerate() {
        let (row, g_row) = rows[r];
        let (mean, rstd) = moments_in(row, rev);
        for j in 0..h {
            let xhat = (row[j] - mean) * rstd;
            if k == 0 {
                out.push(g_row[j] * xhat);
            } else {
                out[j] = g_row[j].mul_add(xhat, out[j]);
            }
        }
    }
    out
}

fn ref_softmax_grad(y: &[f32], g: &[f32], k: usize, rev: bool) -> Vec<f32> {
    let mut out = Vec::new();
    for (y_row, g_row) in y.chunks_exact(k).zip(g.chunks_exact(k)) {
        let d = fused_in(g_row, y_row, rev);
        out.extend(y_row.iter().zip(g_row).map(|(&y, &g)| y * (g - d)));
    }
    out
}

fn ref_sum_rows(g: &[f32], h: usize, rev: bool) -> Vec<f32> {
    let rows: Vec<&[f32]> = g.chunks_exact(h).collect();
    (0..h)
        .map(|j| {
            let column: Vec<f32> = rows.iter().map(|r| r[j]).collect();
            sum_in(&column, rev)
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Attack 1: the edges.
// ---------------------------------------------------------------------------------------------

/// Attack 1. A constant row has variance 0, so `rstd = 1 / sqrt(eps)`, about `1e6`.
/// `LayerNormGradX` there is `rstd * (gx - mean(gx))`; it is finite and matches central
/// differences in `f64` taken with a step far below `sqrt(eps)` (the harness step `1e-5` is
/// above it, where the function is not locally linear). The `f32` evaluation is within `1e-3`
/// of the `f64` one.
#[test]
fn adversary_norm_constant_row_grad_x_matches_f64_differences_below_sqrt_eps() {
    let h = 5u64;
    let gamma = [0.7f32, -1.3, 0.2, 1.1, -0.4];
    let g = [0.3f32, -0.9, 1.7, 0.25, -0.6];
    let ln = layer_norm_program(1, h);
    let gx = grad_x_program(1, h);
    for c in [0.0f32, 0.5] {
        let x = vec![c; h as usize];
        let grad64 = run(
            &gx,
            &[
                ("x", &[1, h], widen(&x)),
                ("gamma", &[h], widen(&gamma)),
                ("g", &[1, h], widen(&g)),
            ],
        );
        let loss = |x: &[f64]| -> f64 {
            run(
                &ln,
                &[
                    ("x", &[1, h], x.to_vec()),
                    ("gamma", &[h], widen(&gamma)),
                    ("beta", &[h], vec![0.0; h as usize]),
                ],
            )
            .iter()
            .zip(&g)
            .map(|(y, &g)| y * f64::from(g))
            .sum()
        };
        let step = 1e-9;
        let diffs: Vec<f64> = (0..h as usize)
            .map(|i| {
                let (mut p, mut m) = (widen(&x), widen(&x));
                p[i] += step;
                m[i] -= step;
                (loss(&p) - loss(&m)) / (p[i] - m[i])
            })
            .collect();
        let scale = diffs.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        assert!(scale > 1e5, "c = {c}: differences {diffs:?}");
        let err = grad64
            .iter()
            .zip(&diffs)
            .fold(0.0f64, |a, (g, d)| a.max((g - d).abs()))
            / scale;
        assert!(
            err <= 1e-5,
            "c = {c}: f64 error {err:e}: {grad64:?} vs {diffs:?}"
        );
        let grad32 = run(
            &gx,
            &[
                ("x", &[1, h], x.clone()),
                ("gamma", &[h], gamma.to_vec()),
                ("g", &[1, h], g.to_vec()),
            ],
        );
        assert!(grad32.iter().all(|v| v.is_finite()), "{grad32:?}");
        let err32 = grad32
            .iter()
            .zip(&grad64)
            .fold(0.0f64, |a, (&g, w)| a.max((f64::from(g) - w).abs()))
            / scale;
        assert!(err32 <= 1e-3, "c = {c}: f32 error {err32:e}");
    }
}

/// Attack 1, finding N2. A constant row of `H = 10` copies of a value whose ascending `f32` sum
/// is inexact (`0.1`, `3.3`, `77.7`, `123.456`, `5000.3`; `1000.1` happens to sum exactly): the
/// mean is not the value itself, every deviation is the same nonzero `c - mean`, and once its
/// square is above `eps` each normalized element is `+-1`, so `layer_norm` returns
/// `beta +- gamma`, not `beta` (measured: `-0.9915` for `77.7`, `-0.999998` for `5000.3`). The
/// `f64` evaluation returns `beta` within `1e-6`, and so would an `f32` evaluation whose mean is
/// correctly rounded, since `c` is an `f32`.
#[test]
#[ignore = "N2: f32 layer_norm of a constant row of an inexact value is beta +- gamma, not beta"]
fn adversary_norm_constant_row_of_an_inexact_value_normalizes_to_beta() {
    let h = 10u64;
    let ln = layer_norm_program(1, h);
    let mut wrong = Vec::new();
    for c in [0.1f32, 1000.1, 3.3, 77.7, 123.456, 5000.3] {
        let x = vec![c; h as usize];
        let gamma = vec![1.0f32; h as usize];
        let beta = vec![0.0f32; h as usize];
        let out64 = run(
            &ln,
            &[
                ("x", &[1, h], widen(&x)),
                ("gamma", &[h], widen(&gamma)),
                ("beta", &[h], widen(&beta)),
            ],
        );
        assert!(out64.iter().all(|v| v.abs() <= 1e-6), "f64 {c}: {out64:?}");
        let out32 = run(
            &ln,
            &[
                ("x", &[1, h], x),
                ("gamma", &[h], gamma),
                ("beta", &[h], beta),
            ],
        );
        if out32.iter().any(|v| v.abs() > 1e-3) {
            wrong.push(format!(
                "f32 layer_norm of a constant row of {c}: {out32:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Attack 1. `H = 1`: the row is its own mean, so the output is exactly `beta` and both
/// gradients exactly 0. `gamma = 0`: `LayerNormGradX` is exactly 0. `H = 2`: the normalized row is
/// `+-1` whatever the values, so the derivative is of order `eps`; the `f32` `LayerNormGradX`
/// stays within `1e-6` of 0.
#[test]
fn adversary_norm_width_1_width_2_and_zero_gamma_are_exact_or_near_zero() {
    let (n, h) = (3u64, 1u64);
    let x = vec![0.3f32, -7.0, 1e20];
    let gamma = vec![-1.5f32];
    let beta = vec![0.25f32];
    let g = vec![2.0f32, -3.0, 0.5];
    let out = run(
        &layer_norm_program(n, h),
        &[
            ("x", &[n, h], x.clone()),
            ("gamma", &[h], gamma.clone()),
            ("beta", &[h], beta.clone()),
        ],
    );
    assert_eq!(out, vec![0.25f32; 3]);
    let dx = run(
        &grad_x_program(n, h),
        &[
            ("x", &[n, h], x.clone()),
            ("gamma", &[h], gamma),
            ("g", &[n, h], g.clone()),
        ],
    );
    assert!(dx.iter().all(|&v| v == 0.0), "{dx:?}");
    let dgamma = run(
        &grad_gamma_program(n, h),
        &[("x", &[n, h], x), ("g", &[n, h], g)],
    );
    assert!(dgamma.iter().all(|&v| v == 0.0), "{dgamma:?}");

    let (n, h) = (2u64, 4u64);
    let x = vec![0.3f32, -0.7, 2.5, 1.0, 4.0, 4.5, -1.0, 0.0];
    let g = vec![1.0f32, -2.0, 0.5, 3.0, -1.0, 0.25, 2.0, -0.5];
    let dx = run(
        &grad_x_program(n, h),
        &[
            ("x", &[n, h], x),
            ("gamma", &[h], vec![0.0f32; 4]),
            ("g", &[n, h], g),
        ],
    );
    assert!(dx.iter().all(|&v| v == 0.0), "{dx:?}");

    let (n, h) = (2u64, 2u64);
    let x = vec![1.0f32, 1.5, -3.0, 7.25];
    let g = vec![0.7f32, -1.2, 2.0, 0.3];
    let dx = run(
        &grad_x_program(n, h),
        &[
            ("x", &[n, h], x),
            ("gamma", &[h], vec![1.3f32, -0.6]),
            ("g", &[n, h], g),
        ],
    );
    assert!(dx.iter().all(|v| v.abs() <= 1e-6), "{dx:?}");

    // Two values closer than `sqrt(eps)`: the variance is below `eps`, the row is in the linear
    // regime of `1 / sqrt(var + eps)`, and the gradient is about `1e6 (gx - mean(gx))`, not
    // small. The two values are 13 ulps apart, so the rounded `f32` mean moves each deviation by
    // several percent; the `f32` evaluation is within `1e-2` of the `f64` one (measured 1.5e-3).
    let x = vec![0.1f32, 0.1000001];
    let g = vec![-0.4f32, 1.1];
    let gamma = vec![1.3f32, -0.6];
    let program = grad_x_program(1, h);
    let dx32 = run(
        &program,
        &[
            ("x", &[1, h], x.clone()),
            ("gamma", &[h], gamma.clone()),
            ("g", &[1, h], g.clone()),
        ],
    );
    let dx64 = run(
        &program,
        &[
            ("x", &[1, h], widen(&x)),
            ("gamma", &[h], widen(&gamma)),
            ("g", &[1, h], widen(&g)),
        ],
    );
    let scale = dx64.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    assert!(scale > 1e4, "{dx64:?}");
    for (a, b) in dx32.iter().zip(&dx64) {
        assert!(
            (f64::from(*a) - b).abs() <= 1e-2 * scale,
            "{dx32:?} vs {dx64:?}"
        );
    }
}

/// Attack 1, finding N3. An element above about `2e19` overflows the squared deviation in `f32`:
/// the variance is `inf`, `rstd` is 0, and the output row is exactly `beta`, finite and silently
/// wrong, where the `f64` evaluation gives the normalized row `[2, -0.5, -0.5, -0.5, -0.5]`.
/// This test pins the behaviour; the finding is that nothing reports it.
#[test]
fn adversary_norm_a_huge_element_overflows_the_variance_and_returns_beta() {
    let h = 5u64;
    let x = vec![1e30f32, 0.0, 0.0, 0.0, 0.0];
    let gamma = vec![1.0f32; 5];
    let beta = vec![0.5f32, -0.5, 0.25, 0.0, 1.0];
    let ln = layer_norm_program(1, h);
    let out32 = run(
        &ln,
        &[
            ("x", &[1, h], x.clone()),
            ("gamma", &[h], gamma.clone()),
            ("beta", &[h], beta.clone()),
        ],
    );
    assert_eq!(out32, beta, "f32 output");
    let out64 = run(
        &ln,
        &[
            ("x", &[1, h], widen(&x)),
            ("gamma", &[h], widen(&gamma)),
            ("beta", &[h], widen(&beta)),
        ],
    );
    let want = [2.5f64, -1.0, -0.25, -0.5, 0.5];
    for (got, want) in out64.iter().zip(want) {
        assert!((got - want).abs() <= 1e-12, "f64 {out64:?}");
    }
}

/// Attack 1. `K = 1`: `softmax` is exactly 1 and `SoftmaxGrad` exactly 0. Equal logits give
/// `1 / K` each. Logits of `+-1e30` are finite; `-inf` beside a finite logit is 0; `-0.0` logits
/// give `1 / K`. `+inf`, and a row whose maximum is `-inf`, give NaN (`inf - inf`): the story
/// says nothing about infinite logits, so that is pinned here and is no finding.
#[test]
fn adversary_norm_softmax_at_its_edges() {
    let k1 = run(
        &softmax_program(3, 1),
        &[("x", &[3, 1], vec![5.0f32, -1e30, 0.0])],
    );
    assert_eq!(k1, vec![1.0f32; 3]);
    let dk1 = run(
        &softmax_grad_program(3, 1),
        &[
            ("y", &[3, 1], k1),
            ("g", &[3, 1], vec![2.5f32, -1e10, 0.125]),
        ],
    );
    assert!(dk1.iter().all(|&v| v == 0.0), "{dk1:?}");

    let rows: [[f32; 4]; 6] = [
        [3.0, 3.0, 3.0, 3.0],
        [1e30, -1e30, 0.0, -1e30],
        [f32::NEG_INFINITY, 0.0, 0.0, f32::NEG_INFINITY],
        [-0.0, -0.0, -0.0, -0.0],
        [f32::INFINITY, 0.0, 1.0, 2.0],
        [f32::NEG_INFINITY; 4],
    ];
    let x: Vec<f32> = rows.iter().flatten().copied().collect();
    let y = run(&softmax_program(6, 4), &[("x", &[6, 4], x)]);
    assert_eq!(&y[0..4], &[0.25f32; 4]);
    assert_eq!(&y[4..8], &[1.0f32, 0.0, 0.0, 0.0]);
    assert_eq!(&y[8..12], &[0.0f32, 0.5, 0.5, 0.0]);
    assert_eq!(&y[12..16], &[0.25f32; 4]);
    assert!(y[16..24].iter().all(|v| v.is_nan()), "{:?}", &y[16..24]);
}

/// Attack 1. A logit 200 above the rest underflows their exponentials to 0: `softmax` is exactly
/// `[1, 0, 0]` and `SoftmaxGrad` exactly 0 in every column, the survivor's `1 * (g - g)`
/// included. At 100 above, the others are subnormal (`exp(-100)` is about `3.7e-44`) and
/// nonzero; the survivor's gradient is still exactly 0, the others `y * (g - g0)`.
#[test]
fn adversary_norm_softmax_underflow_gives_an_exact_zero_gradient() {
    let x = vec![200.0f32, 0.0, 0.0, 100.0, 0.0, 0.0];
    let y = run(&softmax_program(2, 3), &[("x", &[2, 3], x)]);
    assert_eq!(&y[0..3], &[1.0f32, 0.0, 0.0]);
    assert_eq!(y[3], 1.0f32);
    assert!(
        y[4] > 0.0 && y[4] < f32::MIN_POSITIVE,
        "subnormal: {}",
        y[4]
    );
    let g = vec![0.75f32, -2.0, 3.0, 0.75, -2.0, 3.0];
    let dx = run(
        &softmax_grad_program(2, 3),
        &[("y", &[2, 3], y.clone()), ("g", &[2, 3], g.clone())],
    );
    assert!(dx[0..4].iter().all(|&v| v == 0.0), "{dx:?}");
    assert_eq!(dx[4], y[4] * (g[4] - g[3]), "{dx:?}");
    assert_eq!(dx[5], y[5] * (g[5] - g[3]), "{dx:?}");
}

// ---------------------------------------------------------------------------------------------
// Attack 2: bit order inside the gradient ops.
// ---------------------------------------------------------------------------------------------

/// An op, its interpreter result, and its reference in ascending and in descending order.
type OrderCase = (&'static str, Vec<f32>, Vec<f32>, Vec<f32>);

/// Values whose sums depend on the order: each column has its own magnitude.
fn mixed(rng: &mut Rng, n: usize, scales: &[f32]) -> Vec<f32> {
    (0..n * scales.len())
        .map(|i| rng.value() * scales[i % scales.len()])
        .collect()
}

/// Attack 2. `layer_norm`, `LayerNormGradX`, `LayerNormGradGamma`, `SoftmaxGrad` and `SumRows` in
/// `f32` are bit for bit the hand-written ascending order with ADR-009 fusion, on rows whose
/// columns span eight decades. Each reference evaluated in descending order differs in at least
/// one bit, so the assertion separates the two orders.
#[test]
fn adversary_norm_gradient_ops_follow_the_ascending_fused_order_bit_for_bit() {
    let (n, h) = (4usize, 7usize);
    let (nu, hu) = (n as u64, h as u64);
    let mut rng = Rng::new(seed_of(
        "adversary_norm_gradient_ops_follow_the_ascending_fused_order_bit_for_bit",
    ));
    let x = mixed(&mut rng, n, &[1e4, 1.0, 3e3, 1e-3, 7.0, 2e4, 0.5]);
    let g = mixed(&mut rng, n, &[1.0, 1e4, 1e-2, 3e3, 0.3, 1e4, 9.0]);
    let gamma = mixed(&mut rng, 1, &[1.0, 1e3, 0.2, 1e-2, 5.0, 1e3, 1.0]);
    let beta = mixed(&mut rng, 1, &[1.0; 7]);
    let mut y = mixed(&mut rng, n, &[1.0, 1e-4, 0.5, 1e-3, 1.0, 1e-4, 0.25]);
    for v in &mut y {
        *v = v.abs();
    }

    let cases: Vec<OrderCase> = vec![
        (
            "LayerNorm",
            run(
                &layer_norm_program(nu, hu),
                &[
                    ("x", &[nu, hu], x.clone()),
                    ("gamma", &[hu], gamma.clone()),
                    ("beta", &[hu], beta.clone()),
                ],
            ),
            ref_layer_norm(&x, &gamma, &beta, false),
            ref_layer_norm(&x, &gamma, &beta, true),
        ),
        (
            "LayerNormGradX",
            run(
                &grad_x_program(nu, hu),
                &[
                    ("x", &[nu, hu], x.clone()),
                    ("gamma", &[hu], gamma.clone()),
                    ("g", &[nu, hu], g.clone()),
                ],
            ),
            ref_grad_x(&x, &gamma, &g, false),
            ref_grad_x(&x, &gamma, &g, true),
        ),
        (
            "LayerNormGradGamma",
            run(
                &grad_gamma_program(nu, hu),
                &[("x", &[nu, hu], x.clone()), ("g", &[nu, hu], g.clone())],
            ),
            ref_grad_gamma(&x, &g, h, false),
            ref_grad_gamma(&x, &g, h, true),
        ),
        (
            "SoftmaxGrad",
            run(
                &softmax_grad_program(nu, hu),
                &[("y", &[nu, hu], y.clone()), ("g", &[nu, hu], g.clone())],
            ),
            ref_softmax_grad(&y, &g, h, false),
            ref_softmax_grad(&y, &g, h, true),
        ),
        (
            "SumRows",
            run(&sum_rows_program(nu, hu), &[("g", &[nu, hu], g.clone())]),
            ref_sum_rows(&g, h, false),
            ref_sum_rows(&g, h, true),
        ),
    ];
    for (op, got, ascending, descending) in cases {
        assert_eq!(
            bits(&got),
            bits(&ascending),
            "{op}: {got:?} vs {ascending:?}"
        );
        assert_ne!(
            bits(&ascending),
            bits(&descending),
            "{op}: the data does not separate the two orders"
        );
    }
}

/// The sum of the rounded products `a[j] * b[j]` in ascending order, without fusion.
fn plain_in(a: &[f32], b: &[f32]) -> f32 {
    let products: Vec<f32> = a.iter().zip(b).map(|(&a, &b)| a * b).collect();
    sum_in(&products, false)
}

/// `softmax` by hand: the row maximum, `exp`, the sum in the order `rev` gives, the division.
fn ref_softmax(x: &[f32], k: usize, rev: bool) -> Vec<f32> {
    let mut out = Vec::new();
    for row in x.chunks_exact(k) {
        let max = row[1..]
            .iter()
            .fold(row[0], |m, &v| if v > m { v } else { m });
        let e: Vec<f32> = row.iter().map(|&v| (v - max).exp()).collect();
        let total = sum_in(&e, rev);
        out.extend(e.iter().map(|&v| v / total));
    }
    out
}

/// Attack 9 (test quality). The fusion of each sum of products and the order of the `softmax`
/// sum are pinned on data that separates them: `layer_norm`'s variance, the `b` of
/// `LayerNormGradX` and the `d` of `SoftmaxGrad` each differ from the same computation with
/// plain rounded products, and the `softmax` sum differs in descending order; the interpreter
/// equals the fused, ascending reference bit for bit in every case. Moderate values (`[-1, -0.1]
/// ∪ [0.1, 1]`) over wide rows, where a last-bit difference in a reduction reaches the output.
#[test]
fn adversary_norm_fusion_and_the_softmax_sum_order_are_pinned_bit_for_bit() {
    let (n, h) = (8usize, 64usize);
    let (nu, hu) = (n as u64, h as u64);
    let mut rng = Rng::new(seed_of(
        "adversary_norm_fusion_and_the_softmax_sum_order_are_pinned_bit_for_bit",
    ));
    let mut draw = |len: usize| -> Vec<f32> { (0..len).map(|_| rng.value()).collect() };
    let (x, g, gamma, beta) = (draw(n * h), draw(n * h), draw(h), draw(h));
    let logits: Vec<f32> = draw(n * h).iter().map(|v| v * 6.0).collect();
    let y = ref_softmax(&logits, h, false);

    let unfused_layer_norm: Vec<f32> = x
        .chunks_exact(h)
        .flat_map(|row| {
            let mean = sum_in(row, false) / h as f32;
            let d: Vec<f32> = row.iter().map(|&v| v - mean).collect();
            let rstd = 1.0f32 / (plain_in(&d, &d) / h as f32 + EPS).sqrt();
            (0..h)
                .map(|j| d[j] * rstd * gamma[j] + beta[j])
                .collect::<Vec<_>>()
        })
        .collect();
    let unfused_grad_x: Vec<f32> = x
        .chunks_exact(h)
        .zip(g.chunks_exact(h))
        .flat_map(|(row, g_row)| {
            let (mean, rstd) = moments_in(row, false);
            let xhat: Vec<f32> = row.iter().map(|&v| (v - mean) * rstd).collect();
            let gx: Vec<f32> = (0..h).map(|j| g_row[j] * gamma[j]).collect();
            let a = sum_in(&gx, false) / h as f32;
            let b = plain_in(&gx, &xhat) / h as f32;
            (0..h)
                .map(|j| rstd * ((gx[j] - a) - xhat[j] * b))
                .collect::<Vec<_>>()
        })
        .collect();
    let unfused_softmax_grad: Vec<f32> = y
        .chunks_exact(h)
        .zip(g.chunks_exact(h))
        .flat_map(|(y_row, g_row)| {
            let d = plain_in(g_row, y_row);
            y_row
                .iter()
                .zip(g_row)
                .map(|(&y, &g)| y * (g - d))
                .collect::<Vec<_>>()
        })
        .collect();

    let cases: Vec<OrderCase> = vec![
        (
            "LayerNorm variance fused",
            run(
                &layer_norm_program(nu, hu),
                &[
                    ("x", &[nu, hu], x.clone()),
                    ("gamma", &[hu], gamma.clone()),
                    ("beta", &[hu], beta.clone()),
                ],
            ),
            ref_layer_norm(&x, &gamma, &beta, false),
            unfused_layer_norm,
        ),
        (
            "LayerNormGradX b fused",
            run(
                &grad_x_program(nu, hu),
                &[
                    ("x", &[nu, hu], x.clone()),
                    ("gamma", &[hu], gamma.clone()),
                    ("g", &[nu, hu], g.clone()),
                ],
            ),
            ref_grad_x(&x, &gamma, &g, false),
            unfused_grad_x,
        ),
        (
            "SoftmaxGrad d fused",
            run(
                &softmax_grad_program(nu, hu),
                &[("y", &[nu, hu], y.clone()), ("g", &[nu, hu], g.clone())],
            ),
            ref_softmax_grad(&y, &g, h, false),
            unfused_softmax_grad,
        ),
        (
            "Softmax sum ascending",
            run(
                &softmax_program(nu, hu),
                &[("x", &[nu, hu], logits.clone())],
            ),
            y.clone(),
            ref_softmax(&logits, h, true),
        ),
    ];
    for (what, got, pinned, other) in cases {
        assert_eq!(bits(&got), bits(&pinned), "{what}: {got:?} vs {pinned:?}");
        assert_ne!(
            bits(&pinned),
            bits(&other),
            "{what}: the data does not separate the two computations"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Attack 3: two passes, not one.
// ---------------------------------------------------------------------------------------------

/// Attack 3. Rows `1000 + v`, `v` in `[-1, -0.1] ∪ [0.1, 1]`, `H = 16`: the two-pass variance
/// keeps every `f32` output row's biased variance within `1e-2` of 1; `E[x^2] - E[x]^2` in `f32`
/// loses every significant bit of the variance there (`E[x^2]` is about `1e6`, its ulp `0.06`).
#[test]
fn adversary_norm_offset_1e3_scale_1_rows_keep_unit_variance() {
    let (n, h) = (64u64, 16u64);
    let mut rng = Rng::new(seed_of(
        "adversary_norm_offset_1e3_scale_1_rows_keep_unit_variance",
    ));
    let x: Vec<f32> = (0..n * h).map(|_| 1000.0 + rng.value()).collect();
    let out = run(
        &layer_norm_program(n, h),
        &[
            ("x", &[n, h], x),
            ("gamma", &[h], vec![1.0; h as usize]),
            ("beta", &[h], vec![0.0; h as usize]),
        ],
    );
    for (r, row) in widen(&out).chunks_exact(h as usize).enumerate() {
        let mean = row.iter().sum::<f64>() / h as f64;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / h as f64;
        assert!(
            (var - 1.0).abs() <= 1e-2,
            "row {r}: variance {var}: {row:?}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Attack 4: autodiff wiring.
// ---------------------------------------------------------------------------------------------

/// `y = softmax(x W)` read twice, directly and through `layer_norm(y, gamma, beta)`:
/// `mse(y + layer_norm(y, gamma, beta), t)`.
fn softmax_read_twice(b: &mut TensorProgramBuilder) -> ValueId {
    let x = input(b, "x", &[4, 3]);
    let w = param(b, "w", &[3, 6]);
    let gamma = param(b, "gamma", &[6]);
    let beta = param(b, "beta", &[6]);
    let xw = b.matmul(x, w).unwrap();
    let y = b.softmax(xw).unwrap();
    let z = b.layer_norm(y, gamma, beta).unwrap();
    let sum = b.add(y, z).unwrap();
    mse_to_target(b, sum)
}

/// Attack 4. The adjoint of a `softmax` result read twice, once through `layer_norm`, is the sum
/// of both contributions before `SoftmaxGrad`: every gradient matches central differences.
#[test]
fn adversary_norm_softmax_read_twice_once_through_layer_norm_matches_differences() {
    gradient_check(
        "adversary_norm_softmax_read_twice_once_through_layer_norm_matches_differences",
        softmax_read_twice,
    );
}

/// Attack 4. One parameter as both `gamma` and `beta`: its gradient is the sum of
/// `LayerNormGradGamma` and `SumRows`.
#[test]
fn adversary_norm_one_parameter_as_gamma_and_beta_matches_differences() {
    gradient_check(
        "adversary_norm_one_parameter_as_gamma_and_beta_matches_differences",
        |b| {
            let x = input(b, "x", &[4, 3]);
            let w = param(b, "w", &[3, 5]);
            let p = param(b, "p", &[5]);
            let xw = b.matmul(x, w).unwrap();
            let y = b.layer_norm(xw, p, p).unwrap();
            mse_to_target(b, y)
        },
    );
}

/// Attack 4. `softmax` of a parameter itself, and that `softmax` the loss's operand.
#[test]
fn adversary_norm_softmax_of_a_parameter_matches_differences() {
    gradient_check(
        "adversary_norm_softmax_of_a_parameter_matches_differences",
        |b| {
            let p = param(b, "p", &[4, 6]);
            let y = b.softmax(p).unwrap();
            mse_to_target(b, y)
        },
    );
}

/// Attack 4. The training program of [`softmax_read_twice`] gives the same `f32` bits when the
/// interpreter frees and reuses storage slots (`evaluate`, what `train` runs) as when it keeps
/// every value (`evaluate_all`): `SoftmaxGrad` reads the `softmax` result, never a slot that a
/// later value overwrote.
#[test]
fn adversary_norm_slot_reuse_leaves_the_gradient_bits_unchanged() {
    let mut b = TensorProgramBuilder::new();
    let loss = softmax_read_twice(&mut b);
    let forward = b.finish();
    let training = differentiate(&forward, loss).unwrap();
    let mut rng = Rng::new(seed_of(
        "adversary_norm_slot_reuse_leaves_the_gradient_bits_unchanged",
    ));
    let point = Point::draw(&forward, &BTreeMap::new(), &mut rng);
    let reused = finite_difference::run(&training, &point, |v| v);
    let kept = evaluate_all_with_ids(
        &training.program,
        &finite_difference::tensors(&point.inputs, |v| v),
        &point.id_tensors(),
        &finite_difference::tensors(&point.params, |v| v),
    )
    .unwrap()
    .outputs;
    assert_eq!(reused.len(), kept.len());
    for (i, (a, b)) in reused.iter().zip(&kept).enumerate() {
        assert_eq!(bits(a.data()), bits(b.data()), "output {i}");
    }
}

// ---------------------------------------------------------------------------------------------
// Attack 8: the bound of DESIGN.md § 4 at BERT's width.
// ---------------------------------------------------------------------------------------------

/// For each row: the mean of the `f32` output row, over `u max|x| / s` (`u = 2^-24`, `s` the
/// row's standard deviation in `f64`).
fn ratios(x: &[f32], out: &[f32], h: usize) -> Vec<f64> {
    let u = f64::powi(2.0, -24);
    x.chunks_exact(h)
        .zip(out.chunks_exact(h))
        .map(|(row, out)| {
            let row = widen(row);
            let mean = row.iter().sum::<f64>() / h as f64;
            let s = (row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / h as f64).sqrt();
            let largest = row.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let error = (widen(out).iter().sum::<f64>() / h as f64).abs();
            error / (u * largest / s)
        })
        .collect()
}

/// Attack 8, finding N1. DESIGN.md § 4 states that the `f32` mean error of the output rows is a
/// property of any evaluation that rounds the row and its mean to `f32`, every row within
/// `2.34 u max|x| / s`; coordinator decision D36 calls that the general `f32` limit. It was
/// measured at `H = 5`. At BERT's `H = 768`, on the same row distribution
/// (`-7.5 + 0.05 v`), the operator's worst ratio is far above 2.34 (and above the unit test's 4),
/// while the same computation with the mean correctly rounded to `f32` (summed in `f64`) stays
/// within 1.5: the excess is the ascending `f32` sum, which grows with `H`, not the type.
#[test]
fn adversary_norm_the_design_mean_bound_does_not_hold_at_bert_width() {
    let (n, h) = (200usize, 768usize);
    let mut rng = Rng::new(seed_of(
        "adversary_norm_the_design_mean_bound_does_not_hold_at_bert_width",
    ));
    let x: Vec<f32> = (0..n * h).map(|_| -7.5 + 0.05 * rng.value()).collect();
    let (nu, hu) = (n as u64, h as u64);
    let out = run(
        &layer_norm_program(nu, hu),
        &[
            ("x", &[nu, hu], x.clone()),
            ("gamma", &[hu], vec![1.0; h]),
            ("beta", &[hu], vec![0.0; h]),
        ],
    );
    let rounded_mean: Vec<f32> = x
        .chunks_exact(h)
        .flat_map(|row| {
            let mean = (widen(row).iter().sum::<f64>() / h as f64) as f32;
            let d: Vec<f32> = row.iter().map(|&v| v - mean).collect();
            let var = fused_in(&d, &d, false) / h as f32;
            let rstd = 1.0f32 / (var + EPS).sqrt();
            d.into_iter().map(move |d| d * rstd).collect::<Vec<_>>()
        })
        .collect();
    let worst = |v: Vec<f64>| v.into_iter().fold(0.0f64, f64::max);
    let operator = worst(ratios(&x, &out, h));
    let reference = worst(ratios(&x, &rounded_mean, h));
    println!("H = {h}: worst ratio, operator {operator}, correctly rounded mean {reference}");
    assert!(reference <= 1.5, "correctly rounded mean: {reference}");
    assert!(
        operator > 4.0,
        "operator worst ratio {operator} is within the stated bound"
    );
}
