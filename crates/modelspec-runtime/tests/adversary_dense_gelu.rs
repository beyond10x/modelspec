//! Adversary pass 1 on story:bert-dense-gelu: the values and gradients of `gelu`, the order and
//! boundaries of `SumRows`, and the bias gradient through more than one use.
//!
//! The reference for `gelu` here is computed in `f64` from the same tanh form, written as
//! `x / (1 + exp(-2u))` where the left tail makes `1 + tanh(u)` lose its digits, and the `f32`
//! result is compared with that reference rounded to `f32`, in units in the last place.

use std::collections::BTreeMap;

use modelspec_core::ir::tensor::{TensorProgram, TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::{HostTensor, evaluate};

#[allow(dead_code)]
#[path = "common/finite_difference.rs"]
mod finite_difference;

use finite_difference::{gradient_check, input, mse_to_target, param};

const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
const CUBIC: f64 = 0.044_715;

/// The tanh form in `f64`, as the unit's own test writes it.
fn gelu_tanh_f64(x: f64) -> f64 {
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + CUBIC * x * x * x)).tanh())
}

/// The same function in `f64` without the cancellation of `1 + tanh(u)` in the left tail.
fn gelu_exp_f64(x: f64) -> f64 {
    let u = SQRT_2_OVER_PI * (x + CUBIC * x * x * x);
    x / (1.0 + (-2.0 * u).exp())
}

/// The derivative of the tanh form in `f64`, written without cancellation.
fn gelu_slope_f64(x: f64) -> f64 {
    let u = SQRT_2_OVER_PI * (x + CUBIC * x * x * x);
    let s = 1.0 / (1.0 + (-2.0 * u).exp());
    let r = 1.0 / (1.0 + (2.0 * u).exp());
    let du = SQRT_2_OVER_PI * (1.0 + 3.0 * CUBIC * x * x);
    s + 2.0 * x * s * r * du
}

/// The program `[gelu(x), gelu_grad(x, g)]` over inputs of shape `[n]`.
fn gelu_and_grad_program(n: u64) -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", TensorType::f32(&[n])).unwrap();
    let g = b.input("g", TensorType::f32(&[n])).unwrap();
    let y = b.gelu(x).unwrap();
    let dx = b.gelu_grad(x, g).unwrap();
    b.output(y).unwrap();
    b.output(dx).unwrap();
    b.finish()
}

/// `gelu(x)` and `gelu_grad(x, 1)` of the interpreter in `f32`, element by element.
fn gelu_f32(points: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = points.len() as u64;
    let inputs = BTreeMap::from([
        (
            "x".to_owned(),
            HostTensor::new(vec![n], points.to_vec()).unwrap(),
        ),
        (
            "g".to_owned(),
            HostTensor::new(vec![n], vec![1.0f32; points.len()]).unwrap(),
        ),
    ]);
    let out = evaluate(&gelu_and_grad_program(n), &inputs, &BTreeMap::new()).unwrap();
    (out[0].data().to_vec(), out[1].data().to_vec())
}

/// The same in `f64`.
fn gelu_f64(points: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = points.len() as u64;
    let inputs = BTreeMap::from([
        (
            "x".to_owned(),
            HostTensor::new(vec![n], points.to_vec()).unwrap(),
        ),
        (
            "g".to_owned(),
            HostTensor::new(vec![n], vec![1.0f64; points.len()]).unwrap(),
        ),
    ]);
    let out = evaluate(&gelu_and_grad_program(n), &inputs, &BTreeMap::new()).unwrap();
    (out[0].data().to_vec(), out[1].data().to_vec())
}

/// The distance in `f32` units in the last place between two finite `f32` of one sign.
fn ulps(a: f32, b: f32) -> u32 {
    let key = |v: f32| {
        let bits = v.to_bits() as i64;
        if bits < 0x8000_0000 {
            bits
        } else {
            0x8000_0000 - bits
        }
    };
    (key(a) - key(b)).unsigned_abs() as u32
}

/// Attack 1: Scenario 4's bound at every multiple of `1/1024` in `[-3, 3]` (6145 points), the
/// worst relative error printed.
#[test]
fn adversary_gelu_f32_within_1e_6_relative_at_every_1_over_1024_in_minus_3_to_3() {
    let points: Vec<f32> = (-3072..=3072).map(|k| k as f32 / 1024.0).collect();
    let (values, _) = gelu_f32(&points);
    let mut worst = (0.0f64, 0.0f32);
    for (&x, &got) in points.iter().zip(&values) {
        let want = gelu_tanh_f64(f64::from(x));
        if want == 0.0 {
            assert_eq!(got, 0.0, "gelu({x})");
            continue;
        }
        let e = ((f64::from(got) - want) / want).abs();
        if e > worst.0 {
            worst = (e, x);
        }
    }
    println!("worst relative error over [-3, 3] step 1/1024: {worst:?}");
    assert!(worst.0 <= 1e-6, "gelu: {worst:?}");
}

/// Attack 1: the same bound at every `f32` in `[-3, -1]` (12,582,913 points), the left tail where
/// the error is largest; the worst relative error is printed.
#[test]
#[ignore = "F3: 7 f32 points in [-3, -1] exceed 1e-6 relative, worst 1.067e-6 at x = -2.9703214"]
fn adversary_gelu_f32_within_1e_6_relative_at_every_f32_in_minus_3_to_minus_1() {
    let (low, high) = ((-1.0f32).to_bits(), (-3.0f32).to_bits());
    let points: Vec<f32> = (low..=high).map(f32::from_bits).collect();
    let mut worst = (0.0f64, 0.0f32);
    let mut over = 0usize;
    for chunk in points.chunks(1 << 20) {
        let (values, _) = gelu_f32(chunk);
        for (&x, &got) in chunk.iter().zip(&values) {
            let want = gelu_tanh_f64(f64::from(x));
            let e = ((f64::from(got) - want) / want).abs();
            if e > 1e-6 {
                over += 1;
            }
            if e > worst.0 {
                worst = (e, x);
            }
        }
    }
    println!(
        "every f32 in [-3, -1]: worst relative error {worst:?}, {over} of {} points over 1e-6",
        points.len()
    );
    assert!(worst.0 <= 1e-6, "gelu: {worst:?}, {over} points over 1e-6");
}

/// Attack 1: points the unit's tests do not use. Each finite `f32` value is within
/// `4 + 3 ceil(k)` ulps of the `f64` form rounded to `f32`, `k = |x gelu'(x) / gelu(x)|` the
/// condition number (about 230 at `x = -10`, where no `f32` evaluation of `exp(-2u)` can do
/// better); `-0` stays `-0`; `NaN` is `NaN` and `+inf` is `+inf`.
#[test]
fn adversary_gelu_f32_off_grid_and_extreme_points_match_the_f64_form() {
    let mut points: Vec<f32> = vec![
        0.5 / 64.0,
        -0.5 / 64.0,
        2.5 / 64.0 - 3.0,
        3.0 - 0.5 / 64.0,
        -3.0 + 0.5 / 64.0,
        10.0,
        -10.0,
        20.0,
        -20.0,
        88.0,
        -88.0,
        1e3,
        -1e3,
        1e-40,
        -1e-40,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        5.0,
        -5.0,
        -4.0,
    ];
    points.extend((0..64).map(|k| -3.0 - k as f32 / 16.0));
    let (values, _) = gelu_f32(&points);
    let mut report = Vec::new();
    for (&x, &got) in points.iter().zip(&values) {
        let wide = gelu_exp_f64(f64::from(x));
        let want = wide as f32;
        let d = ulps(got, want);
        let condition = if wide == 0.0 {
            0.0
        } else {
            (f64::from(x) * gelu_slope_f64(f64::from(x)) / wide).abs()
        };
        let allowed = 4 + 3 * condition.ceil() as u32;
        report.push(format!(
            "x={x:e} got={got:e} want={want:e} ulps={d} allowed={allowed}"
        ));
        assert!(got.is_finite(), "gelu({x}) = {got}");
        assert!(
            d <= allowed,
            "gelu({x}): got {got:e}, f64 form {want:e}, {d} ulps, allowed {allowed}"
        );
    }
    println!("{}", report.join("\n"));
    let (special, _) = gelu_f32(&[-0.0, 0.0, f32::NAN, f32::INFINITY]);
    assert_eq!(special[0].to_bits(), (-0.0f32).to_bits(), "gelu(-0) is -0");
    assert_eq!(special[1].to_bits(), 0.0f32.to_bits(), "gelu(+0) is +0");
    assert!(special[2].is_nan(), "gelu(NaN)");
    assert_eq!(special[3], f32::INFINITY, "gelu(+inf)");
}

/// Attack 1: the `f64` tanh form itself, the unit's reference, is 0 at `x = -10` while the
/// function is about `-1.2e-37` there: a relative bound against it means nothing below about
/// `x = -5`. This pins what the reference does, so the report can say where the bound holds.
#[test]
fn adversary_the_f64_tanh_reference_underflows_to_zero_in_the_left_tail() {
    assert_eq!(gelu_tanh_f64(-10.0), 0.0);
    let exact = gelu_exp_f64(-10.0);
    assert!(exact < -1e-38 && exact > -1e-36, "{exact:e}");
    let (in_f32, _) = gelu_f32(&[-10.0]);
    assert!(
        (f64::from(in_f32[0]) - exact).abs() <= 1e-6 * exact.abs(),
        "the interpreter keeps the tail: {:e} against {exact:e}",
        in_f32[0]
    );
}

/// Attack 2: `GeluGrad` at `0` is exactly `0.5`; `gelu'(x) + gelu'(-x) = 1` within `f32` over
/// `[-3, 3]` at every `1/64`; far out it is finite and near its limits 0 and 1.
#[test]
fn adversary_gelu_grad_is_half_at_0_symmetric_and_finite_in_the_tails() {
    let (_, slope) = gelu_f32(&[0.0, -0.0]);
    assert_eq!(slope[0], 0.5);
    assert_eq!(slope[1], 0.5);

    let points: Vec<f32> = (-192..=192).map(|k| k as f32 / 64.0).collect();
    let (_, slope) = gelu_f32(&points);
    let n = points.len();
    let mut worst = 0.0f64;
    for i in 0..n {
        let sum = f64::from(slope[i]) + f64::from(slope[n - 1 - i]);
        worst = worst.max((sum - 1.0).abs());
    }
    println!("worst |gelu'(x) + gelu'(-x) - 1| over [-3, 3]: {worst:e}");
    assert!(worst <= 1e-6, "{worst:e}");

    let tails = [
        10.0f32, -10.0, 20.0, -20.0, 88.0, -88.0, 1e3, -1e3, 1e10, -1e10, 1e15, -1e15,
    ];
    let (_, slope) = gelu_f32(&tails);
    for (&x, &d) in tails.iter().zip(&slope) {
        let want = if x > 0.0 { 1.0 } else { 0.0 };
        assert!(d.is_finite(), "gelu'({x}) = {d}");
        assert!((d - want).abs() <= 1e-6, "gelu'({x}) = {d:e}, limit {want}");
    }
}

/// Attack 10, finding F5: in the right tail `s` rounds towards 1 and `1 - s` keeps few digits, so
/// DESIGN.md § 4 computes `r` from its own exponential. Over `[3, 8]` at every `1/16` the `f32`
/// `GeluGrad(x, 1)` is within `2.5e-7` (about 2 ulps at 1) of the derivative in `f64`; with
/// `r = 1 - s` it is off by up to about `1.5e-6` near `x = 5`, and the unit's suite stays green.
#[test]
fn adversary_gelu_grad_right_tail_keeps_r_from_its_own_exponential() {
    let points: Vec<f32> = (48..=128).map(|k| k as f32 / 16.0).collect();
    let (_, slope) = gelu_f32(&points);
    let mut worst = (0.0f64, 0.0f32);
    for (&x, &d) in points.iter().zip(&slope) {
        let e = (f64::from(d) - gelu_slope_f64(f64::from(x))).abs();
        if e > worst.0 {
            worst = (e, x);
        }
    }
    println!("worst |GeluGrad - gelu'| over [3, 8]: {worst:?}");
    assert!(worst.0 <= 2.5e-7, "{worst:?}");
}

/// Attack 2, finding F4: past `|x|` of about `5e19` the `f32` `GeluGrad` is `NaN`, although the
/// input is finite and the derivative tends to 1 (right) and 0 (left): `r` rounds to 0 and the
/// slope `u'(x)` overflows to infinity, and `0 * inf` is `NaN`.
#[test]
#[ignore = "F4: GeluGrad is NaN for finite |x| >= ~5e19 in f32 (0 * inf)"]
fn adversary_gelu_grad_is_finite_for_every_finite_input() {
    let points = [1e20f32, -1e20, f32::MAX, f32::MIN];
    let (_, slope) = gelu_f32(&points);
    for (&x, &d) in points.iter().zip(&slope) {
        assert!(d.is_finite(), "gelu'({x:e}) = {d}");
    }
}

/// Attack 2: central differences of the interpreter's `f64` `gelu` at `±3`, `±6`, `±10` agree
/// with its `f64` `GeluGrad` within `1e-7` relative, and both with the `f64` derivative.
#[test]
fn adversary_gelu_grad_matches_central_differences_at_3_6_10() {
    let h = 1e-5;
    for x in [3.0f64, -3.0, 6.0, -6.0, 10.0, -10.0] {
        let (values, slope) = gelu_f64(&[x + h, x - h, x]);
        let difference = (values[0] - values[1]) / (2.0 * h);
        let analytic = slope[2];
        let want = gelu_slope_f64(x);
        assert!(
            (difference - analytic).abs() <= 1e-7 * analytic.abs().max(1e-300),
            "gelu'({x}): central difference {difference:e}, GeluGrad {analytic:e}"
        );
        assert!(
            (analytic - want).abs() <= 1e-12 * want.abs().max(1e-300),
            "gelu'({x}): GeluGrad {analytic:e}, derivative {want:e}"
        );
    }
}

/// The program `sum_rows(g)` over an input of shape `[rows, cols]`.
fn sum_rows_of(rows: u64, cols: u64, g: &[f32]) -> Vec<f32> {
    let mut b = TensorProgramBuilder::new();
    let input = b.input("g", TensorType::f32(&[rows, cols])).unwrap();
    let sum = b.sum_rows(input).unwrap();
    b.output(sum).unwrap();
    let program = b.finish();
    let inputs = BTreeMap::from([(
        "g".to_owned(),
        HostTensor::new(vec![rows, cols], g.to_vec()).unwrap(),
    )]);
    evaluate(&program, &inputs, &BTreeMap::new()).unwrap()[0]
        .data()
        .to_vec()
}

/// Attack 4: four rows, so that pairwise summation `(r0 + r1) + (r2 + r3)` and the ascending
/// left fold `((r0 + r1) + r2) + r3` differ: `1e8, 1, -1e8, 1` folds to 1, pairs to 0, and the
/// descending fold `((r3 + r2) + r1) + r0` to 0.
#[test]
fn adversary_sum_rows_is_the_ascending_left_fold_not_pairwise() {
    assert_eq!(sum_rows_of(4, 1, &[1e8, 1.0, -1e8, 1.0]), [1.0]);
}

/// Attack 4: the bits of `SumRows` over 257 rows of mixed magnitudes equal a plain ascending
/// left fold in `f32` computed here, twice over; `N = 1` returns `G`'s one row bit for bit.
#[test]
fn adversary_sum_rows_bits_equal_an_ascending_fold_and_repeat() {
    let (rows, cols) = (257u64, 7u64);
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let g: Vec<f32> = (0..rows * cols)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mantissa = (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            let exponent = ((state >> 8) % 24) as i32 - 12;
            mantissa * 2f32.powi(exponent)
        })
        .collect();
    let mut want = g[..cols as usize].to_vec();
    for row in g.chunks_exact(cols as usize).skip(1) {
        for (w, &v) in want.iter_mut().zip(row) {
            *w += v;
        }
    }
    let first = sum_rows_of(rows, cols, &g);
    let second = sum_rows_of(rows, cols, &g);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&first), bits(&want));
    assert_eq!(bits(&first), bits(&second));

    let one = [-0.0f32, 1e-40, f32::MAX, -3.5];
    assert_eq!(bits(&sum_rows_of(1, 4, &one)), bits(&one));
}

/// Attack 3: `N = 0` and `H = 0` never reach the interpreter's row loop (`chunks_exact(0)`
/// panics): the builder or the verifier refuses a zero dimension.
#[test]
fn adversary_add_rows_with_a_zero_dimension_is_refused_not_a_panic() {
    for (x_shape, b_shape) in [([0u64, 3u64], [3u64]), ([2, 0], [0])] {
        let mut b = TensorProgramBuilder::new();
        let Ok(x) = b.input("x", TensorType::f32(&x_shape)) else {
            continue;
        };
        let Ok(row) = b.input("b", TensorType::f32(&b_shape)) else {
            continue;
        };
        let Ok(y) = b.add_rows(x, row) else {
            continue;
        };
        let Ok(s) = b.sum_rows(y) else {
            continue;
        };
        b.output(y).unwrap();
        b.output(s).unwrap();
        let program = b.finish();
        let count = |s: &[u64]| s.iter().product::<u64>() as usize;
        let inputs = BTreeMap::from([
            (
                "x".to_owned(),
                HostTensor::new(x_shape.to_vec(), vec![0.0f32; count(&x_shape)]).unwrap(),
            ),
            (
                "b".to_owned(),
                HostTensor::new(b_shape.to_vec(), vec![0.0f32; count(&b_shape)]).unwrap(),
            ),
        ]);
        assert!(
            evaluate(&program, &inputs, &BTreeMap::new()).is_err(),
            "x {x_shape:?}, b {b_shape:?} evaluated"
        );
    }
}

/// Attack 3: `N = H`, the shape where `b[n]` and `b[h]` are both in bounds: every element of
/// row `n` column `h` is `x[n][h] + b[h]`.
#[test]
fn adversary_add_rows_with_n_equal_h_adds_b_by_column() {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", TensorType::f32(&[3, 3])).unwrap();
    let row = b.input("b", TensorType::f32(&[3])).unwrap();
    let y = b.add_rows(x, row).unwrap();
    b.output(y).unwrap();
    let program = b.finish();
    let inputs = BTreeMap::from([
        (
            "x".to_owned(),
            HostTensor::new(vec![3, 3], vec![0.0f32; 9]).unwrap(),
        ),
        (
            "b".to_owned(),
            HostTensor::new(vec![3], vec![1.0f32, 2.0, 3.0]).unwrap(),
        ),
    ]);
    let out = evaluate(&program, &inputs, &BTreeMap::new()).unwrap();
    assert_eq!(
        out[0].data(),
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
}

/// Attack 5: the bias read by two `add_rows` and the result through `gelu`: its gradient is the
/// sum of two `SumRows`, and it passes the harness with `N = 1` rows (where `SumRows` is the
/// identity) and with `N = 6`.
#[test]
fn adversary_a_bias_read_twice_through_gelu_passes_the_gradient_check() {
    for rows in [1u64, 6] {
        let name = format!("adversary_bias_read_twice_{rows}");
        gradient_check(&name, |b| {
            let x = input(b, "x", &[rows, 4]);
            let w = param(b, "w", &[4, 3]);
            let bias = param(b, "b", &[3]);
            let xw = b.matmul(x, w).unwrap();
            let z = b.add_rows(xw, bias).unwrap();
            let y = b.gelu(z).unwrap();
            let y = b.add_rows(y, bias).unwrap();
            mse_to_target(b, y)
        });
    }
}
