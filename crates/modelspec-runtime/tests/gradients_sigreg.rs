//! `sigreg` on the CPU interpreter (DESIGN.md § 2.2, § 2.3, § 4; story:jepa-sigreg-cpu).
//!
//! `sigreg(z, slices = M)` with `z: f32[N, K]` lowers to
//! `ReduceMean(SigregColumns(MatMul(z, proj)))`, `proj: f32[K, M]` the internal input of its site,
//! drawn by the training host each step on Philox stream tag 6. It is SIGReg with the
//! Epps–Pulley statistic of LeJEPA (Balestriero and LeCun, arXiv 2511.08544):
//!
//! - § 4.2, Definition 2, eq. (5): SIGReg is the mean over the directions `a` of the statistic
//!   `T` of the projected samples `a^T z_n`; here the directions are the `M` columns of `proj`.
//! - § 4.2.3, (Epps–Pulley) and its PyTorch listing ("SIGReg with Epps-Pulley statistic with DDP
//!   support"): `A = randn(K, M)`, `A /= A.norm(p=2, dim=0)`, `t = linspace(-5, 5, 17)`,
//!   `exp_f = exp(-0.5 t^2)`, `err = |ecf - exp_f|^2 * exp_f`, `T = trapz(err, t) * N`. The listing
//!   uses `exp(-t^2/2)` both as the target characteristic function and as the window `w(t)`,
//!   while the prose of § 4.2.3 writes `w(t) = e^{-t^2/sigma^2}`; ModelSpec follows the listing.
//! - § 5.1: the trapezoidal rule with 17 knots is enough, and the symmetry of the integrand
//!   doubles the knots for free; § 6.1 recommends 17 knots on [-5, 5].
//!
//! The independent evaluation below is the listing's 17-knot trapezoid over `[-5, 5]`, written
//! out in `f64`. The interpreter's 8 positive knots `t_j = 0.625 j` with weights `1.25 phi_j`
//! (`j < 8`) and `0.625 phi_8` are that trapezoid folded by symmetry (the knot at 0 contributes
//! `|1 - 1|^2 = 0`), so the two agree only if the folding is right.

use std::collections::BTreeMap;

use modelspec_core::autodiff::differentiate;
use modelspec_core::ir::tensor::{TensorOp, TensorProgram, TensorProgramBuilder, TensorType};
use modelspec_core::ir::training::sigreg_sites;
use modelspec_runtime::cpu::{HostTensor, evaluate};
use modelspec_runtime::train::init::{Stream, philox4x32_10, sigreg_projection};

#[allow(dead_code)]
#[path = "common/finite_difference.rs"]
mod finite_difference;

use finite_difference::{
    H as STEP, Ids, MIN_SCALE, Point, Rng, TOL_F32, TOL_F64, assert_structure, input, loss_at,
    param, relative_error, run, seed_of,
};

/// The run seed of every case.
const SEED: u64 = 0x5eed_5160_1234_abcd;
/// The site of the hand-built program.
const SITE: &str = "Reg.forward.sigreg#0";
/// Rows of `z`.
const N: u64 = 33;
/// Columns of `z`.
const K: u64 = 8;
/// Inner dimension of `matmul(x, w)`.
const IN: u64 = 4;
/// Projections.
const M: u64 = 16;

/// The draw id of `site` at `step`, derived from `blake3` here.
fn id_of(site: &str, step: u64) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(site.as_bytes());
    hasher.update(&step.to_le_bytes());
    let hash = hasher.finalize();
    let mut first = [0u8; 8];
    first.copy_from_slice(&hash.as_bytes()[..8]);
    u64::from_le_bytes(first)
}

/// The projection of `site` at `step`, `[k, m]`, derived from Philox here: element `i = r m + c`
/// is the Box-Muller normal of words 0 and 1 of `philox4x32_10([i, 6, id low, id high], key)`,
/// each column divided by its `f64` L2 norm (squares summed in ascending row order from the first
/// row) and rounded to `f32` once.
fn projection_of(site: &str, step: u64, k: u64, m: u64) -> Vec<f32> {
    let id = id_of(site, step);
    let uniform = |w: u32| (f64::from(w) + 0.5) / 4_294_967_296.0;
    let normal: Vec<f64> = (0..k * m)
        .map(|i| {
            let b = philox4x32_10(
                [i as u32, 6, id as u32, (id >> 32) as u32],
                [SEED as u32, (SEED >> 32) as u32],
            );
            (-2.0 * uniform(b[0]).ln()).sqrt() * (std::f64::consts::TAU * uniform(b[1])).cos()
        })
        .collect();
    let (k, m) = (k as usize, m as usize);
    let norms: Vec<f64> = (0..m)
        .map(|c| {
            let mut squares = (0..k).map(|r| normal[r * m + c] * normal[r * m + c]);
            let first = squares.next().unwrap();
            squares.fold(first, |acc, s| acc + s).sqrt()
        })
        .collect();
    (0..k * m)
        .map(|i| (normal[i] / norms[i % m]) as f32)
        .collect()
}

/// `sigreg(matmul(x, w), slices = M)` as it lowers, with `x: [N, IN]`, `w: [IN, K]` and the
/// projection input `SITE: [K, M]`.
fn sigreg_program() -> (TensorProgram, modelspec_core::ir::tensor::ValueId) {
    let mut b = TensorProgramBuilder::new();
    let x = input(&mut b, "x", &[N, IN]);
    let w = param(&mut b, "Reg.w", &[IN, K]);
    let proj = input(&mut b, SITE, &[K, M]);
    let z = b.matmul(x, w).unwrap();
    let p = b.matmul(z, proj).unwrap();
    let columns = b.sigreg_columns(p).unwrap();
    let loss = b.reduce_mean(columns).unwrap();
    b.output(loss).unwrap();
    (b.finish(), loss)
}

/// A point of the program: `x` and `w` drawn by the harness, the projection the one of `SITE`
/// at step 0.
fn point(forward: &TensorProgram) -> Point {
    let mut rng = Rng::new(seed_of("gradients_sigreg"));
    let mut point = Point::draw(forward, &Ids::new(), &mut rng);
    let projection = sigreg_projection(SEED, SITE, 0, &TensorType::f32(&[K, M])).unwrap();
    point.inputs.insert(
        SITE.to_owned(),
        (projection.shape().to_vec(), projection.data().to_vec()),
    );
    point
}

/// LeJEPA's statistic, written out from the § 4.2.3 listing in `f64`: for each column of
/// `p = z proj`, `err(t) = |mean_n exp(i t p_n) - exp(-t^2/2)|^2 exp(-t^2/2)` at the 17 knots of
/// `linspace(-5, 5, 17)`, integrated by the trapezoidal rule and times `N`; then the mean over
/// the columns (Definition 2, eq. (5)).
fn sigreg_f64(z: &[f64], proj: &[f64], n: usize, k: usize, m: usize) -> f64 {
    let knots: Vec<f64> = (0..17).map(|j| -5.0 + 10.0 * j as f64 / 16.0).collect();
    let mut total = 0.0;
    for c in 0..m {
        let column: Vec<f64> = (0..n)
            .map(|r| (0..k).map(|i| z[r * k + i] * proj[i * m + c]).sum())
            .collect();
        let err: Vec<f64> = knots
            .iter()
            .map(|&t| {
                let window = (-0.5 * t * t).exp();
                let re = column.iter().map(|&v| (t * v).cos()).sum::<f64>() / n as f64;
                let im = column.iter().map(|&v| (t * v).sin()).sum::<f64>() / n as f64;
                ((re - window).powi(2) + im.powi(2)) * window
            })
            .collect();
        let trapezoid: f64 = knots
            .windows(2)
            .zip(err.windows(2))
            .map(|(t, e)| (t[1] - t[0]) * (e[0] + e[1]) / 2.0)
            .sum();
        total += trapezoid * n as f64;
    }
    total / m as f64
}

/// Acceptance. The gradient of `sigreg(matmul(x, w), slices = 16)` with respect to `w`
/// (`N = 33`, `K = 8`), the projection fixed to step 0 of a named site, is within `TOL_F64` in
/// `f64` and `TOL_F32` in `f32` of central differences of the loss in `f64`. The training program
/// holds one `SigregColumnsGrad` reading the forward's projected matrix, and the projection input
/// takes no gradient.
#[test]
fn gradients_sigreg_match_central_differences_at_a_fixed_projection() {
    let (forward, loss) = sigreg_program();
    let training = differentiate(&forward, loss).unwrap();
    assert_structure(&forward, loss, &training);
    let grads: Vec<_> = training
        .program
        .nodes
        .iter()
        .filter(|n| n.op == TensorOp::SigregColumnsGrad)
        .collect();
    assert_eq!(grads.len(), 1, "one SigregColumnsGrad");
    let columns = forward
        .nodes
        .iter()
        .find(|n| n.op == TensorOp::SigregColumns)
        .unwrap();
    assert_eq!(
        grads[0].operands[0], columns.operands[0],
        "SigregColumnsGrad reads the forward's projected matrix"
    );
    assert_eq!(sigreg_sites(&forward), vec![SITE.to_owned()]);

    let point = point(&forward);
    let w = training.gradients.keys().copied().next().unwrap();
    let g64 = run(&training, &point, f64::from)[1].data().to_vec();
    let g32: Vec<f64> = run(&training, &point, |v| v)[1]
        .data()
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let n = point.params[&w].1.len();
    let differences: Vec<f64> = (0..n)
        .map(|i| {
            (loss_at(&forward, &point, w, i, STEP) - loss_at(&forward, &point, w, i, -STEP))
                / (2.0 * STEP)
        })
        .collect();
    let scale = differences.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(scale >= MIN_SCALE, "differences {differences:?} too small");
    let e64 = relative_error(&g64, &differences);
    let e32 = relative_error(&g32, &differences);
    assert!(
        e64 <= TOL_F64,
        "f64 error {e64:e}: {g64:?} vs {differences:?}"
    );
    assert!(
        e32 <= TOL_F32,
        "f32 error {e32:e}: {g32:?} vs {differences:?}"
    );
}

/// Acceptance. The value of the loss equals the independent `f64` evaluation of LeJEPA's
/// 17-knot trapezoid within `1e-5` relative, in the interpreter's `f64` reference and in `f32`
/// within `TOL_F32`.
#[test]
fn sigreg_value_is_the_lejepa_epps_pulley_trapezoid() {
    let (forward, _) = sigreg_program();
    let point = point(&forward);
    let wide =
        |name: &str| -> Vec<f64> { point.inputs[name].1.iter().map(|&v| f64::from(v)).collect() };
    let (x, proj) = (wide("x"), wide(SITE));
    let w: Vec<f64> = point
        .params
        .values()
        .next()
        .unwrap()
        .1
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let (n, k, inner, m) = (N as usize, K as usize, IN as usize, M as usize);
    let z: Vec<f64> = (0..n * k)
        .map(|e| {
            let (r, c) = (e / k, e % k);
            (0..inner).map(|i| x[r * inner + i] * w[i * k + c]).sum()
        })
        .collect();
    let want = sigreg_f64(&z, &proj, n, k, m);
    assert!(
        want > 0.0,
        "the statistic of drawn data is positive: {want}"
    );

    let tensors64 = finite_difference::tensors(&point.inputs, f64::from);
    let params64 = finite_difference::tensors(&point.params, f64::from);
    let got64 = evaluate(&forward, &tensors64, &params64).unwrap()[0].data()[0];
    let e64 = ((got64 - want) / want).abs();
    assert!(e64 <= 1e-5, "f64 {got64} vs {want}: {e64:e}");

    let tensors32 = finite_difference::tensors(&point.inputs, |v| v);
    let params32 = finite_difference::tensors(&point.params, |v| v);
    let got32 = f64::from(evaluate(&forward, &tensors32, &params32).unwrap()[0].data()[0]);
    let e32 = ((got32 - want) / want).abs();
    assert!(e32 <= TOL_F32, "f32 {got32} vs {want}: {e32:e}");
}

/// A sample exactly the standard normal's characteristic function cannot be built from finitely
/// many rows, but a column of all zeros has `C_j = 1` and `S_j = 0` at every knot, so its
/// statistic is `N sum_j c_j (1 - phi_j)^2`, the same as LeJEPA's trapezoid of it.
#[test]
fn sigreg_of_a_zero_column_is_the_trapezoid_of_a_point_mass() {
    let mut b = TensorProgramBuilder::new();
    let p = input(&mut b, "p", &[5, 1]);
    let columns = b.sigreg_columns(p).unwrap();
    b.output(columns).unwrap();
    let program = b.finish();
    let inputs = BTreeMap::from([(
        "p".to_owned(),
        HostTensor::new(vec![5, 1], vec![0.0f64; 5]).unwrap(),
    )]);
    let got = evaluate(&program, &inputs, &BTreeMap::new()).unwrap()[0].data()[0];
    let want = sigreg_f64(&[0.0; 5], &[1.0], 5, 1, 1);
    assert!(((got - want) / want).abs() <= 1e-12, "{got} vs {want}");
}

/// The projection of a site at a step is drawn on Philox stream tag 6 from the site's draw id at
/// the step, Box-Muller normal per element, each column of unit `f64` L2 norm rounded to `f32`
/// once; the runtime's helper agrees bit for bit with the derivation here, and a later step
/// draws a different matrix.
#[test]
fn sigreg_projection_follows_philox_stream_tag_6() {
    assert_eq!(Stream::Projection.tag(), 6);
    let ty = TensorType::f32(&[K, M]);
    for step in [0, 1, 7] {
        let got = sigreg_projection(SEED, SITE, step, &ty).unwrap();
        assert_eq!(got.shape(), &[K, M]);
        let want = projection_of(SITE, step, K, M);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(got.data()), bits(&want), "step {step}");
    }
    let zero = sigreg_projection(SEED, SITE, 0, &ty).unwrap();
    let one = sigreg_projection(SEED, SITE, 1, &ty).unwrap();
    assert_ne!(zero.data(), one.data(), "steps draw different projections");
    let other = sigreg_projection(SEED, "Reg.forward.sigreg#1", 0, &ty).unwrap();
    assert_ne!(
        zero.data(),
        other.data(),
        "sites draw different projections"
    );
    for c in 0..M as usize {
        let norm: f64 = (0..K as usize)
            .map(|r| f64::from(zero.data()[r * M as usize + c]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() <= 1e-6, "column {c}: norm {norm}");
    }
}
