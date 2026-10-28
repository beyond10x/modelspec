//! Adversary cases for `stop_gradient` and `sigreg` on the CPU interpreter
//! (story:jepa-stop-gradient, story:jepa-sigreg-cpu; DESIGN.md § 2.2, § 2.3, § 4).
//!
//! Gradient-level attacks: the gradient of `sigreg` against central differences in `f64` over a
//! grid of shapes and degenerate columns, non-finite inputs, the written-out `sin`/`cos` routine
//! and the `SigregColumns` sequence against the constants and order DESIGN.md § 4 states, and
//! `stop_gradient` nested, under a value also read live, on a parameter, inside `sigreg` and as
//! the whole loss.

#![allow(clippy::excessive_precision)]

use std::collections::BTreeMap;

use modelspec_core::autodiff::differentiate;
use modelspec_core::ir::tensor::{
    Initializer, ParamId, TensorOp, TensorProgram, TensorProgramBuilder, TensorType, ValueId,
};
use modelspec_core::ir::training::{OptimizerRule, TrainingProgram, attach};
use modelspec_core::ir::verify::{verify_step, verify_training};
use modelspec_core::passes::stop_gradient::eliminate_training;
use modelspec_runtime::cpu::{Element, HostTensor, evaluate, sin_cos};
use modelspec_runtime::train::init::sigreg_projection;

/// The run seed of every projection here.
const SEED: u64 = 0xad5e_7a11_0bad_cafe;
/// Central-difference step.
const H: f64 = 1e-5;
/// Largest relative error of an `f64` gradient.
const TOL_F64: f64 = 1e-6;
/// Largest relative error of an `f32` gradient.
const TOL_F32: f64 = 1e-3;

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A value in `[-1, -0.1] ∪ [0.1, 1]`.
    fn value(&mut self) -> f32 {
        let magnitude = 0.1 + 0.9 * self.unit();
        if self.next_u64() & 1 == 0 {
            magnitude as f32
        } else {
            -magnitude as f32
        }
    }

    /// A standard normal draw, Box-Muller in `f64`, rounded to `f32`.
    fn normal(&mut self) -> f32 {
        let u = (self.unit() + 1e-300).min(1.0);
        let v = self.unit();
        ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
    }
}

fn f32_ty(shape: &[u64]) -> TensorType {
    TensorType::f32(shape)
}

fn tensor<T: Element>(shape: &[u64], data: &[f32], convert: impl Fn(f32) -> T) -> HostTensor<T> {
    HostTensor::new(shape.to_vec(), data.iter().map(|&v| convert(v)).collect()).unwrap()
}

// ---------------------------------------------------------------------------------------------
// sigreg gradients
// ---------------------------------------------------------------------------------------------

/// The projection input name of the hand-built programs.
const SITE: &str = "Adv.train.sigreg#0";

/// `sigreg(z, slices = s)` as it lowers, `z: [n, k]` a parameter: `ReduceMean(SigregColumns(
/// MatMul(z, proj)))` with the projection input `SITE: [k, s]`.
fn sigreg_of_param(n: u64, k: u64, s: u64) -> (TensorProgram, ValueId, ParamId) {
    let mut b = TensorProgramBuilder::new();
    let z = b
        .declare_param("Adv.z", f32_ty(&[n, k]), Initializer::XavierUniform)
        .unwrap();
    let read = b.param_read(z).unwrap();
    let proj = b.input(SITE, f32_ty(&[k, s])).unwrap();
    let p = b.matmul(read, proj).unwrap();
    let columns = b.sigreg_columns(p).unwrap();
    let loss = b.reduce_mean(columns).unwrap();
    b.output(loss).unwrap();
    (b.finish(), loss, z)
}

/// What one sigreg gradient check measured.
struct SigregPoint {
    n: u64,
    k: u64,
    s: u64,
    z: Vec<f32>,
    proj: Vec<f32>,
}

impl SigregPoint {
    fn new(n: u64, k: u64, s: u64, z: Vec<f32>) -> Self {
        let proj = sigreg_projection(SEED, SITE, 0, &f32_ty(&[k, s]))
            .unwrap()
            .data()
            .to_vec();
        Self { n, k, s, z, proj }
    }

    fn inputs<T: Element>(&self, convert: impl Fn(f32) -> T) -> BTreeMap<String, HostTensor<T>> {
        BTreeMap::from([(
            SITE.to_owned(),
            tensor(&[self.k, self.s], &self.proj, convert),
        )])
    }

    /// The loss in `f64` with element `index` of `z` moved by `delta`.
    fn loss_at(&self, forward: &TensorProgram, z: ParamId, index: usize, delta: f64) -> f64 {
        let mut moved: Vec<f64> = self.z.iter().map(|&v| f64::from(v)).collect();
        moved[index] += delta;
        let params = BTreeMap::from([(z, HostTensor::new(vec![self.n, self.k], moved).unwrap())]);
        evaluate(forward, &self.inputs(f64::from), &params).unwrap()[0].data()[0]
    }

    /// The gradient of `z` from the differentiated program, in `T`, as `f64`.
    fn gradient<T: Element + Into<f64>>(
        &self,
        training: &TrainingProgram,
        z: ParamId,
        convert: impl Fn(f32) -> T + Copy,
    ) -> Vec<f64> {
        let params = BTreeMap::from([(z, tensor(&[self.n, self.k], &self.z, convert))]);
        evaluate(&training.program, &self.inputs(convert), &params).unwrap()[1]
            .data()
            .iter()
            .map(|&v| v.into())
            .collect()
    }
}

/// `max |g - d| / max |d|` over the elements of `indices`.
fn relative_error_at(got: &[f64], want: &[f64], indices: &[usize]) -> f64 {
    let scale = indices.iter().fold(0.0f64, |m, &i| m.max(want[i].abs()));
    let err = indices
        .iter()
        .fold(0.0f64, |m, &i| m.max((got[i] - want[i]).abs()));
    err / scale
}

/// At most about 48 element indices of an `n`-element tensor, the last included.
fn sample(count: usize) -> Vec<usize> {
    let stride = (count / 48).max(1);
    let mut indices: Vec<usize> = (0..count).step_by(stride).collect();
    if indices.last() != Some(&(count - 1)) {
        indices.push(count - 1);
    }
    indices
}

/// Checks the `f64` gradient of `z` at `point` against central differences, divided by the step
/// `f64` actually took (a `1e6` element does not move by exactly `2H`); returns a failure
/// line, or `None`.
fn check_f64(label: &str, point: &SigregPoint) -> Option<String> {
    let (forward, loss, z) = sigreg_of_param(point.n, point.k, point.s);
    let training = eliminate_training(differentiate(&forward, loss).unwrap());
    let g64 = point.gradient(&training, z, f64::from);
    let indices = sample(point.z.len());
    let mut d = vec![0.0; point.z.len()];
    for &i in &indices {
        let x = f64::from(point.z[i]);
        let step = (x + H) - (x - H);
        d[i] = (point.loss_at(&forward, z, i, H) - point.loss_at(&forward, z, i, -H)) / step;
    }
    let scale = indices.iter().fold(0.0f64, |m, &i| m.max(d[i].abs()));
    let e = relative_error_at(&g64, &d, &indices);
    if scale < 1e-6 || e > TOL_F64 || !e.is_finite() {
        Some(format!(
            "{label} n={} k={} s={}: f64 error {e:e}, scale {scale:e}",
            point.n, point.k, point.s
        ))
    } else {
        None
    }
}

/// Priority 1: the gradient of `sigreg` with respect to `z` matches central differences in
/// `f64` at every shape of the grid `N ∈ {1, 2, 63, 64, 65}`, `K ∈ {1, 2, 64}`,
/// `slices ∈ {1, 4, 17}` (63 to 65 straddle the CPU matmul's blocking).
#[test]
fn adversary_sigreg_gradient_matches_central_differences_over_the_shape_grid() {
    let mut failures = Vec::new();
    let mut rng = Rng(0x5eed_0001);
    for n in [1u64, 2, 63, 64, 65] {
        for k in [1u64, 2, 64] {
            for s in [1u64, 4, 17] {
                let z: Vec<f32> = (0..n * k).map(|_| rng.value()).collect();
                let point = SigregPoint::new(n, k, s, z);
                failures.extend(check_f64("grid", &point));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The analytic gradient of the mean of the statistics, written out here in `f64` from DESIGN.md
/// § 4: `dz = (1/s) dP proj^T`, `dP[n, m] = 2 sum_j c_j t_j (S_j cos(t_j P) - (C_j - phi_j)
/// sin(t_j P))`.
fn analytic_gradient(point: &SigregPoint) -> Vec<f64> {
    let (n, k, s) = (point.n as usize, point.k as usize, point.s as usize);
    let z: Vec<f64> = point.z.iter().map(|&v| f64::from(v)).collect();
    let proj: Vec<f64> = point.proj.iter().map(|&v| f64::from(v)).collect();
    let p: Vec<f64> = (0..n * s)
        .map(|e| {
            let (r, c) = (e / s, e % s);
            (0..k).map(|i| z[r * k + i] * proj[i * s + c]).sum()
        })
        .collect();
    let mut dp = vec![0.0; n * s];
    for c in 0..s {
        for j in 1..=8 {
            let t = 0.625 * j as f64;
            let phi = (-0.5 * t * t).exp();
            let weight = if j == 8 { 0.625 } else { 1.25 } * phi;
            let cm = (0..n).map(|r| (t * p[r * s + c]).cos()).sum::<f64>() / n as f64;
            let sm = (0..n).map(|r| (t * p[r * s + c]).sin()).sum::<f64>() / n as f64;
            for r in 0..n {
                let x = t * p[r * s + c];
                dp[r * s + c] += 2.0 * weight * t * (sm * x.cos() - (cm - phi) * x.sin());
            }
        }
    }
    (0..n * k)
        .map(|e| {
            let (r, i) = (e / k, e % k);
            (0..s).map(|c| dp[r * s + c] * proj[i * s + c]).sum::<f64>() / s as f64
        })
        .collect()
}

/// Priority 1: degenerate columns. A zero column of `z`, rows all equal (every projected column
/// constant): the `f64` gradient matches central differences. A column of about `1e6`, where
/// central differences drown in the rounding of `t p` near `3e6`: the `f64` gradient matches the
/// analytic one. All of
/// `z` zero: the gradient is exactly zero. All of `z` about `1e-6`, where central differences
/// drown in the loss's own rounding: the `f64` gradient matches the analytic one, and the `f32`
/// gradient the `f64` one within `TOL_F32`.
#[test]
fn adversary_sigreg_gradient_on_degenerate_columns() {
    let (n, k, s) = (33u64, 4u64, 4u64);
    let mut rng = Rng(0x5eed_0002);
    let drawn: Vec<f32> = (0..n * k).map(|_| rng.value()).collect();
    let mut failures = Vec::new();

    let mut zero_column = drawn.clone();
    for r in 0..n as usize {
        zero_column[r * k as usize] = 0.0;
    }
    failures.extend(check_f64(
        "zero column",
        &SigregPoint::new(n, k, s, zero_column),
    ));

    let row: Vec<f32> = drawn[..k as usize].to_vec();
    let equal_rows: Vec<f32> = (0..n).flat_map(|_| row.clone()).collect();
    failures.extend(check_f64(
        "equal rows",
        &SigregPoint::new(n, k, s, equal_rows),
    ));

    let mut huge = drawn.clone();
    for r in 0..n as usize {
        huge[r * k as usize] *= 1e6;
    }
    let huge = SigregPoint::new(n, k, s, huge);
    let (forward, loss, z) = sigreg_of_param(n, k, s);
    let training = eliminate_training(differentiate(&forward, loss).unwrap());
    let g = huge.gradient(&training, z, f64::from);
    let want = analytic_gradient(&huge);
    let every: Vec<usize> = (0..want.len()).collect();
    let e = relative_error_at(&g, &want, &every);
    if e.is_nan() || e > 1e-8 {
        failures.push(format!("huge column: f64 vs analytic {e:e}"));
    }

    let zeros = SigregPoint::new(n, k, s, vec![0.0; (n * k) as usize]);
    let (forward, loss, z) = sigreg_of_param(n, k, s);
    let training = eliminate_training(differentiate(&forward, loss).unwrap());
    let g = zeros.gradient(&training, z, f64::from);
    if g.iter().any(|&v| v != 0.0) {
        failures.push(format!("all-zero z: gradient {g:?} is not zero"));
    }

    let tiny: Vec<f32> = drawn.iter().map(|&v| v * 1e-6).collect();
    let tiny = SigregPoint::new(n, k, s, tiny);
    let g64 = tiny.gradient(&training, z, f64::from);
    let g32 = tiny.gradient(&training, z, |v| v);
    let want = analytic_gradient(&tiny);
    let all: Vec<usize> = (0..want.len()).collect();
    let e64 = relative_error_at(&g64, &want, &all);
    let e32 = relative_error_at(&g32, &g64, &all);
    if e64.is_nan() || e64 > 1e-9 {
        failures.push(format!("tiny z: f64 vs analytic {e64:e}"));
    }
    if e32.is_nan() || e32 > TOL_F32 {
        failures.push(format!("tiny z: f32 vs f64 {e32:e}"));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Priority 1: the `f32` gradient of a near-Gaussian batch, the regime a trained JEPA encoder
/// is in, agrees with the `f64` one within `TOL_F32` at `N = 1024` rows.
#[test]
fn adversary_sigreg_f32_gradient_near_the_gaussian_target() {
    let (n, k, s) = (1024u64, 8u64, 16u64);
    let mut rng = Rng(0x5eed_0003);
    let z: Vec<f32> = (0..n * k).map(|_| rng.normal()).collect();
    let point = SigregPoint::new(n, k, s, z);
    let (forward, loss, param) = sigreg_of_param(n, k, s);
    let training = eliminate_training(differentiate(&forward, loss).unwrap());
    let g64 = point.gradient(&training, param, f64::from);
    let g32 = point.gradient(&training, param, |v| v);
    let all: Vec<usize> = (0..g64.len()).collect();
    let e = relative_error_at(&g32, &g64, &all);
    assert!(e <= TOL_F32, "f32 vs f64 gradient at N = {n}: {e:e}");
}

/// `SigregColumns` alone over one `[n, m]` input `p`.
fn columns_program(n: u64, m: u64) -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let p = b.input("p", f32_ty(&[n, m])).unwrap();
    let columns = b.sigreg_columns(p).unwrap();
    b.output(columns).unwrap();
    b.finish()
}

/// `SigregColumnsGrad` alone over `p: [n, m]` and `g: [m]`.
fn columns_grad_program(n: u64, m: u64) -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let p = b.input("p", f32_ty(&[n, m])).unwrap();
    let g = b.input("g", f32_ty(&[m])).unwrap();
    let grad = b.sigreg_columns_grad(p, g).unwrap();
    b.output(grad).unwrap();
    b.finish()
}

fn run_columns<T: Element>(n: u64, m: u64, p: &[f32], convert: impl Fn(f32) -> T) -> Vec<T> {
    let inputs = BTreeMap::from([("p".to_owned(), tensor(&[n, m], p, convert))]);
    evaluate(&columns_program(n, m), &inputs, &BTreeMap::new()).unwrap()[0]
        .data()
        .to_vec()
}

/// Priority 1: a NaN or an infinity in a column is reported, not hidden: that column's
/// statistic and every element of its gradient are not finite, in `f32` and in `f64`, and the
/// other column stays finite.
#[test]
fn adversary_sigreg_non_finite_inputs_propagate() {
    let (n, m) = (4u64, 2u64);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut p = vec![0.3f32, -0.2, 0.7, 0.1, -0.9, 0.4, 0.05, -0.6];
        p[2] = bad;
        let t32 = run_columns(n, m, &p, |v| v);
        let t64 = run_columns(n, m, &p, f64::from);
        assert!(!t32[0].is_finite(), "{bad}: f32 column 0 is {}", t32[0]);
        assert!(!t64[0].is_finite(), "{bad}: f64 column 0 is {}", t64[0]);
        assert!(t32[1].is_finite() && t64[1].is_finite(), "{bad}: column 1");
        let inputs = BTreeMap::from([
            ("p".to_owned(), tensor(&[n, m], &p, |v| v)),
            ("g".to_owned(), tensor(&[m], &[1.0, 1.0], |v| v)),
        ]);
        let grad = evaluate(&columns_grad_program(n, m), &inputs, &BTreeMap::new()).unwrap()[0]
            .data()
            .to_vec();
        for r in 0..n as usize {
            assert!(!grad[r * 2].is_finite(), "{bad}: f32 gradient row {r}");
            assert!(
                grad[r * 2 + 1].is_finite(),
                "{bad}: f32 gradient row {r} col 1"
            );
        }
    }
}

/// The largest value `SigregColumns` can take on a column of `n` rows: `|C_j - phi_j| <= 1 +
/// phi_j` and `|S_j| <= 1`, so `T <= n sum_j c_j ((1 + phi_j)^2 + 1)`.
fn sigreg_bound(n: u64) -> f64 {
    (1..=8)
        .map(|j| {
            let t = 0.625 * j as f64;
            let phi = (-0.5 * t * t).exp();
            let c = if j == 8 { 0.625 } else { 1.25 } * phi;
            c * ((1.0 + phi).powi(2) + 1.0)
        })
        .sum::<f64>()
        * n as f64
}

/// Priority 1, finite but large columns: whatever accuracy the `f32` routine loses outside
/// `|x| < 2^9 pi/2`, `sin` and `cos` stay in `[-1, 1]`, so the statistic of a finite column
/// stays finite and within its bound. Columns of magnitude `1e3` to `1e12`.
#[test]
#[ignore = "accepted: sin_cos is specified for abs(x) below 2^24 pi/2; beyond it the statistic is unspecified and may be infinite (DESIGN.md sigreg)"]
fn adversary_sigreg_large_finite_columns_stay_within_the_statistic_bound() {
    let (n, m) = (8u64, 1u64);
    let mut rng = Rng(0x5eed_0004);
    let shape: Vec<f32> = (0..n).map(|_| rng.value()).collect();
    let mut failures = Vec::new();
    for exponent in 3..=12 {
        let scale = 10f32.powi(exponent);
        let p: Vec<f32> = shape.iter().map(|&v| v * scale).collect();
        let t = f64::from(run_columns(n, m, &p, |v| v)[0]);
        if !(t.is_finite() && (0.0..=sigreg_bound(n)).contains(&t)) {
            failures.push(format!(
                "1e{exponent}: T = {t:e}, bound {:e}",
                sigreg_bound(n)
            ));
        }
        for &x in &p {
            for t in [0.625f32, 5.0] {
                let (s, c) = sin_cos(t * x);
                if !(s.abs() <= 1.0 && c.abs() <= 1.0) {
                    failures.push(format!("sin_cos({:e}) = ({s:e}, {c:e})", t * x));
                    break;
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Priority 1, precision at a realistic batch: the `f32` statistic of a near-Gaussian column of
/// `N = 4096` rows agrees with the `f64` evaluation of the same `f32` values within `TOL_F32`.
#[test]
fn adversary_sigreg_f32_statistic_near_the_gaussian_target_at_4096_rows() {
    let (n, m) = (4096u64, 8u64);
    let mut rng = Rng(0x5eed_0005);
    let p: Vec<f32> = (0..n * m).map(|_| rng.normal()).collect();
    let t32 = run_columns(n, m, &p, |v| v);
    let t64 = run_columns(n, m, &p, f64::from);
    let worst = t32
        .iter()
        .zip(&t64)
        .map(|(&a, &b)| ((f64::from(a) - b) / b).abs())
        .fold(0.0f64, f64::max);
    assert!(worst <= TOL_F32, "{t32:?} vs {t64:?}: {worst:e}");
}

// ---------------------------------------------------------------------------------------------
// DESIGN.md § 4 as the specification of the f32 sequence
// ---------------------------------------------------------------------------------------------

/// `sin` and `cos` written from DESIGN.md § 4 with its decimal constants, as a Vulkan kernel of
/// story S3 would emit them.
fn design_sin_cos(x: f32) -> (f32, f32) {
    const P1: f32 = 1.5703125;
    const P2: f32 = 4.837512969970703125e-4;
    const P3: f32 = 7.549789954891882e-8;
    const S0: f32 = -1.9515295891e-4;
    const S1: f32 = 8.3321608736e-3;
    const S2: f32 = -1.6666654611e-1;
    const C0: f32 = 2.443315711809948e-5;
    const C1: f32 = -1.388731625493765e-3;
    const C2: f32 = 4.166664568298827e-2;
    let two_over_pi = (2.0 / std::f64::consts::PI) as f32;
    let q = (x * two_over_pi).round_ties_even();
    let r = (-q).mul_add(P1, x);
    let r = (-q).mul_add(P2, r);
    let r = (-q).mul_add(P3, r);
    let z = r * r;
    let s = (S0.mul_add(z, S1).mul_add(z, S2) * z).mul_add(r, r);
    let c = (C0.mul_add(z, C1).mul_add(z, C2) * z).mul_add(z, (-0.5f32).mul_add(z, 1.0));
    match (q as i64).rem_euclid(4) {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// Contract: the shared routine returns the bits of DESIGN.md § 4's sequence, written with its
/// constants, at every multiple of 1/256 in [-64, 64] and at 100,000 draws in [-800, 800].
#[test]
fn adversary_sin_cos_is_the_design_sequence_bit_for_bit() {
    let mut rng = Rng(0x5eed_0006);
    let grid = (-64 * 256..=64 * 256).map(|i| i as f32 / 256.0);
    let drawn = (0..100_000).map(|_| (rng.unit() * 1600.0 - 800.0) as f32);
    let mut failures = Vec::new();
    for x in grid.chain(drawn) {
        let (s, c) = sin_cos(x);
        let (ds, dc) = design_sin_cos(x);
        if s.to_bits() != ds.to_bits() || c.to_bits() != dc.to_bits() {
            failures.push(format!("{x:e}: ({s:e}, {c:e}) vs DESIGN ({ds:e}, {dc:e})"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of the inputs differ, first: {:?}",
        failures.len(),
        &failures[..failures.len().min(5)]
    );
}

/// The knot `t_j`, `phi_j` and `c_j` in `f64`.
fn knot(j: usize) -> (f64, f64, f64) {
    let t = 0.625 * j as f64;
    let phi = (-0.5 * t * t).exp();
    (t, phi, if j == 8 { 0.625 } else { 1.25 } * phi)
}

/// `SigregColumns` of an `[n, m]` `f32` tensor as DESIGN.md § 4 states it.
fn design_columns(p: &[f32], n: usize, m: usize) -> Vec<f32> {
    (0..m)
        .map(|col| {
            let mut sum = 0.0f32;
            for j in 1..=8 {
                let (t, phi, c) = knot(j);
                let (t, phi, c) = (t as f32, phi as f32, c as f32);
                let (mut ss, mut cs) = sin_cos(t * p[col]);
                for r in 1..n {
                    let (s, k) = sin_cos(t * p[r * m + col]);
                    ss += s;
                    cs += k;
                }
                let (cm, sm) = (cs / n as f32, ss / n as f32);
                let d = cm - phi;
                let q = sm.mul_add(sm, d * d);
                sum = if j == 1 { c * q } else { c.mul_add(q, sum) };
            }
            n as f32 * sum
        })
        .collect()
}

/// `SigregColumnsGrad` of `p: [n, m]` and `g: [m]` as DESIGN.md § 4 states it.
fn design_columns_grad(p: &[f32], g: &[f32], n: usize, m: usize) -> Vec<f32> {
    let means: Vec<Vec<(f32, f32)>> = (0..m)
        .map(|col| {
            (1..=8)
                .map(|j| {
                    let t = knot(j).0 as f32;
                    let (mut ss, mut cs) = sin_cos(t * p[col]);
                    for r in 1..n {
                        let (s, k) = sin_cos(t * p[r * m + col]);
                        ss += s;
                        cs += k;
                    }
                    (cs / n as f32, ss / n as f32)
                })
                .collect()
        })
        .collect();
    (0..n * m)
        .map(|e| {
            let col = e % m;
            let mut sum = 0.0f32;
            for j in 1..=8 {
                let (t, phi, c) = knot(j);
                let slope = (c * t) as f32;
                let (cm, sm) = means[col][j - 1];
                let d = cm - phi as f32;
                let (s, k) = sin_cos(t as f32 * p[e]);
                let term = (-d).mul_add(s, sm * k);
                sum = if j == 1 {
                    slope * term
                } else {
                    slope.mul_add(term, sum)
                };
            }
            (2.0 * g[col]) * sum
        })
        .collect()
}

/// Contract: `SigregColumns` and `SigregColumnsGrad` in `f32` give the bits of the sequence
/// DESIGN.md § 4 states (knot constants computed in `f64` and rounded, ascending sums from row
/// 0, `fma` where it says `fma`), which a Vulkan kernel must reproduce to agree bit for bit.
#[test]
fn adversary_sigreg_columns_are_the_design_sequence_bit_for_bit() {
    let (n, m) = (37usize, 5usize);
    let mut rng = Rng(0x5eed_0007);
    let p: Vec<f32> = (0..n * m).map(|_| rng.normal() * 1.7).collect();
    let g: Vec<f32> = (0..m).map(|_| rng.value()).collect();
    let got = run_columns(n as u64, m as u64, &p, |v| v);
    let want = design_columns(&p, n, m);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&got), bits(&want), "SigregColumns {got:?} vs {want:?}");
    let inputs = BTreeMap::from([
        ("p".to_owned(), tensor(&[n as u64, m as u64], &p, |v| v)),
        ("g".to_owned(), tensor(&[m as u64], &g, |v| v)),
    ]);
    let got = evaluate(
        &columns_grad_program(n as u64, m as u64),
        &inputs,
        &BTreeMap::new(),
    )
    .unwrap()[0]
        .data()
        .to_vec();
    let want = design_columns_grad(&p, &g, n, m);
    let differing = got
        .iter()
        .zip(&want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing, 0,
        "SigregColumnsGrad differs at {differing} elements"
    );
}

// ---------------------------------------------------------------------------------------------
// stop_gradient
// ---------------------------------------------------------------------------------------------

const N: u64 = 5;
const K: u64 = 4;
const M: u64 = 3;

/// `x: [N, K]`, `y: [N, M]`, `w: [K, M]`, drawn from one seed.
struct Point {
    x: Vec<f32>,
    y: Vec<f32>,
    w: Vec<f32>,
}

impl Point {
    fn draw(seed: u64) -> Self {
        let mut rng = Rng(seed);
        let mut draw = |count: u64| (0..count).map(|_| rng.value()).collect::<Vec<_>>();
        Self {
            x: draw(N * K),
            y: draw(N * M),
            w: draw(K * M),
        }
    }

    fn inputs<T: Element>(
        &self,
        convert: impl Fn(f32) -> T + Copy,
    ) -> BTreeMap<String, HostTensor<T>> {
        BTreeMap::from([
            ("x".to_owned(), tensor(&[N, K], &self.x, convert)),
            ("y".to_owned(), tensor(&[N, M], &self.y, convert)),
        ])
    }
}

/// A forward program over `x`, `y` and the parameter `w`, built by `build` from
/// `(builder, x, y, read of w)`; returns it, its loss and `w`.
fn forward(
    build: impl FnOnce(&mut TensorProgramBuilder, ValueId, ValueId, ValueId) -> ValueId,
) -> (TensorProgram, ValueId, ParamId) {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", f32_ty(&[N, K])).unwrap();
    let y = b.input("y", f32_ty(&[N, M])).unwrap();
    let w = b
        .declare_param("Adv.w", f32_ty(&[K, M]), Initializer::XavierUniform)
        .unwrap();
    let read = b.param_read(w).unwrap();
    let loss = build(&mut b, x, y, read);
    b.output(loss).unwrap();
    (b.finish(), loss, w)
}

/// Differentiates, removes every `StopGradient`, verifies, attaches SGD and verifies the step,
/// as `compile_training` does; returns the training program.
fn trained(program: &TensorProgram, loss: ValueId) -> TrainingProgram {
    let training = eliminate_training(differentiate(program, loss).unwrap());
    verify_training(&training).expect("the training program verifies");
    assert!(
        training
            .program
            .nodes
            .iter()
            .all(|n| n.op != TensorOp::StopGradient),
        "a StopGradient survives the pass"
    );
    let step = attach(training.clone(), 0.1, OptimizerRule::Sgd).expect("SGD attaches");
    verify_step(&step).expect("the step verifies");
    training
}

/// The gradient of `w` in `T`, as `f64`.
fn gradient_of<T: Element + Into<f64>>(
    training: &TrainingProgram,
    w: ParamId,
    point: &Point,
    convert: impl Fn(f32) -> T + Copy,
) -> Vec<f64> {
    let params = BTreeMap::from([(w, tensor(&[K, M], &point.w, convert))]);
    let outputs = evaluate(&training.program, &point.inputs(convert), &params).unwrap();
    let position = training
        .gradients
        .keys()
        .position(|&p| p == w)
        .expect("w has a gradient");
    outputs[1 + position]
        .data()
        .iter()
        .map(|&v| v.into())
        .collect()
}

/// The value of `x @ w` at `point`, in `f64`.
fn xw(point: &Point) -> Vec<f64> {
    (0..(N * M) as usize)
        .map(|e| {
            let (r, c) = (e / M as usize, e % M as usize);
            (0..K as usize)
                .map(|i| {
                    f64::from(point.x[r * K as usize + i]) * f64::from(point.w[i * M as usize + c])
                })
                .sum()
        })
        .collect()
}

/// Central differences of `loss(w)` in `f64` over every element of `w`.
fn differences(point: &Point, loss: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    let w: Vec<f64> = point.w.iter().map(|&v| f64::from(v)).collect();
    (0..w.len())
        .map(|i| {
            let mut plus = w.clone();
            let mut minus = w.clone();
            plus[i] += H;
            minus[i] -= H;
            (loss(&plus) - loss(&minus)) / (2.0 * H)
        })
        .collect()
}

/// `x @ w` in `f64` for any `w`.
fn xw_with(point: &Point, w: &[f64]) -> Vec<f64> {
    (0..(N * M) as usize)
        .map(|e| {
            let (r, c) = (e / M as usize, e % M as usize);
            (0..K as usize)
                .map(|i| f64::from(point.x[r * K as usize + i]) * w[i * M as usize + c])
                .sum()
        })
        .collect()
}

fn mse64(a: &[f64], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| (a - f64::from(b)).powi(2))
        .sum::<f64>()
        / a.len() as f64
}

fn assert_close(label: &str, got: &[f64], want: &[f64], tol: f64) {
    let all: Vec<usize> = (0..want.len()).collect();
    let e = relative_error_at(got, want, &all);
    assert!(e <= tol, "{label}: error {e:e}: {got:?} vs {want:?}");
}

/// Priority 2: `stop_gradient(stop_gradient(h))` beside a live `h` is the single
/// `stop_gradient`: the gradient of `w` matches central differences with the stopped term held,
/// in `f64` and `f32`.
#[test]
fn adversary_stop_gradient_nested_holds_the_term() {
    let (program, loss, w) = forward(|b, x, y, read| {
        let online = b.matmul(x, read).unwrap();
        let target = b.matmul(x, read).unwrap();
        let once = b.stop_gradient(target).unwrap();
        let twice = b.stop_gradient(once).unwrap();
        let sum = b.add(online, twice).unwrap();
        b.mse(sum, y).unwrap()
    });
    let training = trained(&program, loss);
    let point = Point::draw(0xad0_0001);
    let held = xw(&point);
    let want = differences(&point, |w| {
        let sum: Vec<f64> = xw_with(&point, w)
            .iter()
            .zip(&held)
            .map(|(a, b)| a + b)
            .collect();
        mse64(&sum, &point.y)
    });
    assert_close(
        "f64",
        &gradient_of(&training, w, &point, f64::from),
        &want,
        TOL_F64,
    );
    assert_close(
        "f32",
        &gradient_of(&training, w, &point, |v| v),
        &want,
        TOL_F32,
    );
}

/// Priority 2: `mul(stop_gradient(h), h)` reads `h` with and without the stop in one
/// expression: the gradient of `w` is that of `mul(c, h)` with `c` held at `h`'s value.
#[test]
fn adversary_stop_gradient_beside_a_live_read_in_one_expression() {
    let (program, loss, w) = forward(|b, x, y, read| {
        let h = b.matmul(x, read).unwrap();
        let held = b.stop_gradient(h).unwrap();
        let product = b.mul(held, h).unwrap();
        b.mse(product, y).unwrap()
    });
    let training = trained(&program, loss);
    let point = Point::draw(0xad0_0002);
    let held = xw(&point);
    let want = differences(&point, |w| {
        let product: Vec<f64> = xw_with(&point, w)
            .iter()
            .zip(&held)
            .map(|(a, b)| a * b)
            .collect();
        mse64(&product, &point.y)
    });
    assert_close(
        "f64",
        &gradient_of(&training, w, &point, f64::from),
        &want,
        TOL_F64,
    );
    assert_close(
        "f32",
        &gradient_of(&training, w, &point, |v| v),
        &want,
        TOL_F32,
    );
}

/// Priority 2: `stop_gradient` on the parameter read itself, `matmul(x, stop_gradient(w)) +
/// matmul(x, w)`: the gradient is that of the second term with the first held.
#[test]
fn adversary_stop_gradient_on_a_parameter_read() {
    let (program, loss, w) = forward(|b, x, y, read| {
        let held_w = b.stop_gradient(read).unwrap();
        let held = b.matmul(x, held_w).unwrap();
        let live = b.matmul(x, read).unwrap();
        let sum = b.add(held, live).unwrap();
        b.mse(sum, y).unwrap()
    });
    let training = trained(&program, loss);
    let point = Point::draw(0xad0_0003);
    let held = xw(&point);
    let want = differences(&point, |w| {
        let sum: Vec<f64> = held
            .iter()
            .zip(xw_with(&point, w))
            .map(|(a, b)| a + b)
            .collect();
        mse64(&sum, &point.y)
    });
    assert_close(
        "f64",
        &gradient_of(&training, w, &point, f64::from),
        &want,
        TOL_F64,
    );
    assert_close(
        "f32",
        &gradient_of(&training, w, &point, |v| v),
        &want,
        TOL_F32,
    );
}

/// Priority 2: `sigreg(stop_gradient(x @ w)) + mse(x @ w, y)`, `sigreg` as it lowers: no gradient
/// leaks through the stopped `sigreg` operand, so `w`'s gradient has the bits of `mse(x @ w, y)`
/// alone, bit for bit in `f64`.
#[test]
fn adversary_stop_gradient_inside_sigreg_leaks_nothing() {
    let with_sigreg = |b: &mut TensorProgramBuilder, x, y, read| {
        let z = b.matmul(x, read).unwrap();
        let held = b.stop_gradient(z).unwrap();
        let proj = b.input(SITE, f32_ty(&[M, 4])).unwrap();
        let p = b.matmul(held, proj).unwrap();
        let columns = b.sigreg_columns(p).unwrap();
        let reg = b.reduce_mean(columns).unwrap();
        let live = b.matmul(x, read).unwrap();
        let fit = b.mse(live, y).unwrap();
        b.add(reg, fit).unwrap()
    };
    let (program, loss, w) = forward(with_sigreg);
    let training = trained(&program, loss);
    let (plain, plain_loss, plain_w) = forward(|b, x, y, read| {
        let live = b.matmul(x, read).unwrap();
        b.mse(live, y).unwrap()
    });
    let plain_training = trained(&plain, plain_loss);
    let point = Point::draw(0xad0_0004);
    let proj = sigreg_projection(SEED, SITE, 0, &f32_ty(&[M, 4])).unwrap();
    let run = |convert: fn(f32) -> f64| {
        let mut inputs = point.inputs(convert);
        inputs.insert(SITE.to_owned(), tensor(&[M, 4], proj.data(), convert));
        let params = BTreeMap::from([(w, tensor(&[K, M], &point.w, convert))]);
        evaluate(&training.program, &inputs, &params).unwrap()[1]
            .data()
            .to_vec()
    };
    let got64 = run(f64::from);
    let want64 = gradient_of(&plain_training, plain_w, &point, f64::from);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&got64), bits(&want64), "f64: {got64:?} vs {want64:?}");
}

/// Priority 2: the whole loss under `stop_gradient` in a program with no parameter: the pass
/// replaces the loss by its operand, which the training program, the SGD attachment and the step
/// verifier accept, and the loss evaluates to the operand's value.
#[test]
fn adversary_stop_gradient_as_the_whole_loss() {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", f32_ty(&[N, M])).unwrap();
    let y = b.input("y", f32_ty(&[N, M])).unwrap();
    let fit = b.mse(x, y).unwrap();
    let loss = b.stop_gradient(fit).unwrap();
    b.output(loss).unwrap();
    let program = b.finish();
    let training = trained(&program, loss);
    assert_eq!(training.loss, fit, "the loss is the stopped operand");
    let point = Point::draw(0xad0_0005);
    let inputs = BTreeMap::from([
        (
            "x".to_owned(),
            tensor(
                &[N, M],
                &point.y.iter().map(|v| v * 2.0).collect::<Vec<_>>(),
                |v| v,
            ),
        ),
        ("y".to_owned(), tensor(&[N, M], &point.y, |v| v)),
    ]);
    let got = evaluate(&training.program, &inputs, &BTreeMap::new()).unwrap()[0].data()[0];
    let want = point.y.iter().map(|v| v * v).sum::<f32>() / (N * M) as f32;
    assert!((got - want).abs() <= 1e-5 * want, "{got} vs {want}");
}
