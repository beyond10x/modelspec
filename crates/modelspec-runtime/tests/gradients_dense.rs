//! Dense layers with a bias row and the tanh-form GELU on the CPU interpreter (DESIGN.md § 2.2,
//! § 4, § 9 gate 3; story:bert-dense-gelu).
//!
//! `add_rows(x, b)` adds the `[H]` row `b` to every row of the `[N, H]` `x`. Its gradient with
//! respect to `x` is the adjoint itself and with respect to `b` the internal op `SumRows`, the
//! rows of the adjoint summed in ascending order. `gelu(x)` is
//! `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))` element by element, and its gradient the
//! internal op `GeluGrad(x, G)`, the derivative of that form times `G`. The finite-difference
//! cases use the harness of `tests/common/finite_difference.rs`.

use std::collections::BTreeMap;

use modelspec_core::autodiff::differentiate;
use modelspec_core::ir::tensor::{TensorOp, TensorProgram, TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::{HostTensor, evaluate, evaluate_all};

#[allow(dead_code)]
#[path = "common/finite_difference.rs"]
mod finite_difference;

use finite_difference::{
    Ids, MIN_SCALE, Point, Rng, TOL_F32, TOL_F64, assert_structure, gradient_check, input, loss_at,
    mse_to_target, param, relative_error, run, seed_of, tensors,
};

/// Rows of the dense layer's input.
const N: u64 = 5;
/// Width of the dense layer's output, and of its bias row.
const H: u64 = 3;
/// Width of the dense layer's input.
const K: u64 = 4;

/// `sqrt(2 / pi)`, in `f64`.
const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
/// The cubic coefficient of the tanh form.
const CUBIC: f64 = 0.044_715;

/// The tanh form of GELU, in `f64`.
fn gelu_f64(x: f64) -> f64 {
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + CUBIC * x * x * x)).tanh())
}

/// The derivative of [`gelu_f64`], in `f64`.
fn gelu_derivative_f64(x: f64) -> f64 {
    let t = (SQRT_2_OVER_PI * (x + CUBIC * x * x * x)).tanh();
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * SQRT_2_OVER_PI * (1.0 + 3.0 * CUBIC * x * x)
}

/// `mse(add_rows(matmul(x, w), b), t)` with `N = 5` and `H = 3`: the gradients of `w` and `b` are
/// within `TOL_F64` of central differences in `f64` and `TOL_F32` in `f32`, and the gradient of
/// `b` is a `SumRows` of the adjoint.
#[test]
fn add_rows_gradients_of_w_and_b_match_central_differences() {
    let training = gradient_check(
        "add_rows_gradients_of_w_and_b_match_central_differences",
        |b| {
            let x = input(b, "x", &[N, K]);
            let w = param(b, "w", &[K, H]);
            let bias = param(b, "b", &[H]);
            let xw = b.matmul(x, w).unwrap();
            let y = b.add_rows(xw, bias).unwrap();
            assert_eq!(b.type_of(y), Some(&TensorType::f32(&[N, H])));
            mse_to_target(b, y)
        },
    );
    assert_eq!(training.gradients.len(), 2, "a gradient for w and for b");
    let bias = *training.gradients.values().last().unwrap();
    let node = training.program.node(bias).unwrap();
    assert_eq!(node.op, TensorOp::SumRows, "the gradient of b: {node:?}");
    assert_eq!(node.ty, TensorType::f32(&[H]));
}

/// `x` itself trainable: `mse(add_rows(x, b), t)` with `x` and `b` both parameters; the adjoint of
/// `x` is the adjoint of the result.
#[test]
fn add_rows_gradient_of_x_is_the_adjoint() {
    gradient_check("add_rows_gradient_of_x_is_the_adjoint", |b| {
        let x = param(b, "x", &[N, H]);
        let bias = param(b, "b", &[H]);
        let y = b.add_rows(x, bias).unwrap();
        mse_to_target(b, y)
    });
}

/// The program `add_rows(x, b)` over inputs `x: [3, 2]` and `b: [2]`.
fn add_rows_program() -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", TensorType::f32(&[3, 2])).unwrap();
    let bias = b.input("b", TensorType::f32(&[2])).unwrap();
    let y = b.add_rows(x, bias).unwrap();
    b.output(y).unwrap();
    b.finish()
}

/// `b` is added to every row of `x`, in `f32` and in `f64`.
#[test]
fn add_rows_adds_b_to_every_row() {
    let program = add_rows_program();
    let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [10.0f32, 20.0];
    let inputs = BTreeMap::from([
        (
            "x".to_owned(),
            HostTensor::new(vec![3, 2], x.to_vec()).unwrap(),
        ),
        (
            "b".to_owned(),
            HostTensor::new(vec![2], b.to_vec()).unwrap(),
        ),
    ]);
    let out = evaluate(&program, &inputs, &BTreeMap::new()).unwrap();
    assert_eq!(out[0].shape(), &[3, 2]);
    assert_eq!(out[0].data(), &[11.0, 22.0, 13.0, 24.0, 15.0, 26.0]);
    let inputs: BTreeMap<String, HostTensor<f64>> = inputs
        .into_iter()
        .map(|(k, v)| {
            let data = v.data().iter().map(|&e| f64::from(e)).collect();
            (k, HostTensor::new(v.shape().to_vec(), data).unwrap())
        })
        .collect();
    let out = evaluate(&program, &inputs, &BTreeMap::new()).unwrap();
    assert_eq!(out[0].data(), &[11.0, 22.0, 13.0, 24.0, 15.0, 26.0]);
}

/// The program `sum_rows(g)` over an input `g` of shape `shape`.
fn sum_rows_program(shape: &[u64]) -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let g = b.input("g", TensorType::f32(shape)).unwrap();
    let sum = b.sum_rows(g).unwrap();
    assert_eq!(
        b.type_of(sum),
        Some(&TensorType::f32(&shape[1..])),
        "[N, H] -> [H]"
    );
    b.output(sum).unwrap();
    b.finish()
}

fn sum_rows_of(shape: &[u64], g: &[f32]) -> Vec<f32> {
    let inputs = BTreeMap::from([(
        "g".to_owned(),
        HostTensor::new(shape.to_vec(), g.to_vec()).unwrap(),
    )]);
    evaluate(&sum_rows_program(shape), &inputs, &BTreeMap::new()).unwrap()[0]
        .data()
        .to_vec()
}

/// `SumRows` adds the rows in ascending order in `f32` (coordinator decision D28): with the
/// column `1, 1e8, -1e8`, `(1 + 1e8) - 1e8` rounds to 0 where the descending order gives 1; the
/// second column checks every element of a row lands in its own column.
#[test]
fn add_rows_bias_gradient_sums_rows_in_ascending_order() {
    let got = sum_rows_of(&[3, 2], &[1.0, 2.0, 1e8, 3.0, -1e8, 4.0]);
    assert_eq!(got, [0.0, 9.0]);
}

/// The sum starts from its first row, not from `+0`: a column of negative zeros sums to `-0`, as
/// the other sums of the interpreter do (DESIGN.md § 2.2).
#[test]
fn add_rows_bias_gradient_starts_from_the_first_row() {
    let got = sum_rows_of(&[2, 1], &[-0.0, -0.0]);
    assert_eq!(got, [0.0]);
    assert!(got[0].is_sign_negative(), "-0 + -0 is -0; got {got:?}");
}

/// The points at which the story pins the value of `gelu`.
const GELU_POINTS: [f32; 7] = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0];

/// The program `gelu(x)` over an input `x` of shape `[7]`.
fn gelu_program() -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", TensorType::f32(&[7])).unwrap();
    let y = b.gelu(x).unwrap();
    assert_eq!(b.type_of(y), Some(&TensorType::f32(&[7])), "same shape");
    b.output(y).unwrap();
    b.finish()
}

/// The interpreter's `f32` value of `gelu` at `-3, -1, -0.5, 0, 0.5, 1, 3` equals the tanh form
/// evaluated in `f64` within `1e-6` relative (exactly 0 at 0), and its `f64` value within
/// `1e-12` relative.
#[test]
fn gelu_values_match_the_tanh_form_in_f64() {
    let program = gelu_program();
    let inputs = BTreeMap::from([(
        "x".to_owned(),
        HostTensor::new(vec![7], GELU_POINTS.to_vec()).unwrap(),
    )]);
    let in_f32 = evaluate(&program, &inputs, &BTreeMap::new()).unwrap();
    let inputs64 = tensors(
        &BTreeMap::from([("x".to_owned(), (vec![7], GELU_POINTS.to_vec()))]),
        f64::from,
    );
    let in_f64 = evaluate(&program, &inputs64, &BTreeMap::new()).unwrap();
    for (i, &x) in GELU_POINTS.iter().enumerate() {
        let want = gelu_f64(f64::from(x));
        let got32 = f64::from(in_f32[0].data()[i]);
        let got64 = in_f64[0].data()[i];
        assert!(
            (got32 - want).abs() <= 1e-6 * want.abs(),
            "gelu({x}) in f32: {got32:e}, the tanh form in f64: {want:e}, relative error {:e}",
            ((got32 - want) / want).abs()
        );
        assert!(
            (got64 - want).abs() <= 1e-12 * want.abs(),
            "gelu({x}) in f64: {got64:e}, the tanh form: {want:e}"
        );
    }
}

/// Between the pinned points too: at every multiple of `1/64` in `[-3, 3]` (385 points, each an
/// exact `f32`), the `f32` value of `gelu` is within `1e-6` relative of the tanh form in `f64`,
/// and `GeluGrad(x, 1)` within `1e-6` absolute of its derivative (absolute, because the
/// derivative crosses 0 near `x = -0.75`, where no relative bound holds; it lies in
/// `[-0.13, 1.13]` on this range).
#[test]
fn gelu_values_between_minus_3_and_3_stay_within_1e_6_relative() {
    let points: Vec<f32> = (-192..=192).map(|k| k as f32 / 64.0).collect();
    let n = points.len() as u64;
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", TensorType::f32(&[n])).unwrap();
    let g = b.input("g", TensorType::f32(&[n])).unwrap();
    let y = b.gelu(x).unwrap();
    let dx = b.gelu_grad(x, g).unwrap();
    b.output(y).unwrap();
    b.output(dx).unwrap();
    let program = b.finish();
    let inputs = BTreeMap::from([
        (
            "x".to_owned(),
            HostTensor::new(vec![n], points.clone()).unwrap(),
        ),
        (
            "g".to_owned(),
            HostTensor::new(vec![n], vec![1.0f32; points.len()]).unwrap(),
        ),
    ]);
    let out = evaluate(&program, &inputs, &BTreeMap::new()).unwrap();
    /// The larger of two `(relative error, x)` pairs by error.
    fn worse(a: (f64, f32), b: (f64, f32)) -> (f64, f32) {
        if b.0 > a.0 { b } else { a }
    }
    let (mut worst_value, mut worst_slope) = ((0.0f64, 0.0f32), (0.0f64, 0.0f32));
    for (i, &x) in points.iter().enumerate() {
        let want = gelu_f64(f64::from(x));
        let got = f64::from(out[0].data()[i]);
        if want != 0.0 {
            worst_value = worse(worst_value, (((got - want) / want).abs(), x));
        } else {
            assert_eq!(got, 0.0, "gelu(0)");
        }
        let want = gelu_derivative_f64(f64::from(x));
        let got = f64::from(out[1].data()[i]);
        worst_slope = worse(worst_slope, ((got - want).abs(), x));
    }
    println!("largest error: value {worst_value:?} relative, slope {worst_slope:?} absolute");
    assert!(worst_value.0 <= 1e-6, "gelu: {worst_value:?}");
    assert!(worst_slope.0 <= 1e-6, "GeluGrad: {worst_slope:?}");
}

/// `GeluGrad(x, g)` is the derivative of the tanh form at `x` times `g`, in `f32` within `1e-5`
/// relative of the derivative evaluated in `f64`, at the points of [`GELU_POINTS`].
#[test]
fn gelu_grad_values_are_the_derivative_of_the_tanh_form_times_g() {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", TensorType::f32(&[7])).unwrap();
    let g = b.input("g", TensorType::f32(&[7])).unwrap();
    let dx = b.gelu_grad(x, g).unwrap();
    b.output(dx).unwrap();
    let program = b.finish();
    let gs = [0.5f32, -2.0, 1.0, 3.0, -1.0, 0.25, 2.0];
    let inputs = BTreeMap::from([
        (
            "x".to_owned(),
            HostTensor::new(vec![7], GELU_POINTS.to_vec()).unwrap(),
        ),
        (
            "g".to_owned(),
            HostTensor::new(vec![7], gs.to_vec()).unwrap(),
        ),
    ]);
    let out = evaluate(&program, &inputs, &BTreeMap::new()).unwrap();
    for (i, (&x, &g)) in GELU_POINTS.iter().zip(&gs).enumerate() {
        let want = gelu_derivative_f64(f64::from(x)) * f64::from(g);
        let got = f64::from(out[0].data()[i]);
        assert!(
            (got - want).abs() <= 1e-5 * want.abs(),
            "GeluGrad({x}, {g}): {got:e}, want {want:e}"
        );
    }
}

/// The name of the GELU gradient case, its seed.
const GELU_CASE: &str = "gelu_gradient_of_w_matches_central_differences_across_minus_3_to_3";

/// Pre-activations `-3, -2, ..., 3`: both tails and the region near 0.
const PRE_ACTIVATIONS: [f64; 7] = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];

/// `mse(gelu(matmul(x, w)), t)` with `x: [7, 1]` and `w: [1, 3]`: the gradient of `w` is within
/// `TOL_F64` of central differences in `f64` and `TOL_F32` in `f32`. `w` and `t` are drawn by the
/// harness; `x` is set so that the column of `w` with the largest magnitude takes the
/// pre-activations `-3, -2, ..., 3` and every other column a scaled copy inside `[-3, 3]`.
#[test]
fn gelu_gradient_of_w_matches_central_differences_across_minus_3_to_3() {
    let rows = PRE_ACTIVATIONS.len() as u64;
    let mut b = TensorProgramBuilder::new();
    let x = input(&mut b, "x", &[rows, 1]);
    let w = param(&mut b, "w", &[1, H]);
    let z = b.matmul(x, w).unwrap();
    let y = b.gelu(z).unwrap();
    let loss = mse_to_target(&mut b, y);
    let forward = b.finish();
    let training = differentiate(&forward, loss).unwrap();
    assert_structure(&forward, loss, &training);
    let gelu_grads = training
        .program
        .nodes
        .iter()
        .filter(|n| n.op == TensorOp::GeluGrad)
        .count();
    assert_eq!(gelu_grads, 1, "one GeluGrad for one gelu");

    let mut rng = Rng::new(seed_of(GELU_CASE));
    let mut point = Point::draw(&forward, &Ids::new(), &mut rng);
    let (_, ws) = point.params.values().next().unwrap().clone();
    let largest = ws.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
    let xs: Vec<f32> = PRE_ACTIVATIONS
        .iter()
        .map(|&p| (p / largest) as f32)
        .collect();
    point.inputs.insert("x".to_owned(), (vec![rows, 1], xs));

    let evaluation = evaluate_all(
        &forward,
        &tensors(&point.inputs, f64::from),
        &tensors(&point.params, f64::from),
    )
    .unwrap();
    let zs = evaluation.values[&z].data();
    let lowest = zs.iter().copied().fold(f64::INFINITY, f64::min);
    let highest = zs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let nearest = zs.iter().fold(f64::INFINITY, |m, v| m.min(v.abs()));
    assert!(
        (lowest + 3.0).abs() < 1e-5 && (highest - 3.0).abs() < 1e-5 && nearest == 0.0,
        "pre-activations {zs:?} do not span [-3, 3] through 0"
    );

    let mut in_f64 = run(&training, &point, f64::from);
    let mut in_f32 = run(&training, &point, |v| v);
    let in_f64 = in_f64.remove(1);
    let in_f32 = in_f32.remove(1);
    let param = *forward.params.keys().next().unwrap();
    let n = point.params[&param].1.len();
    let step = finite_difference::H;
    let differences: Vec<f64> = (0..n)
        .map(|i| {
            (loss_at(&training.program, &point, param, i, step)
                - loss_at(&training.program, &point, param, i, -step))
                / (2.0 * step)
        })
        .collect();
    let scale = differences.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(scale >= MIN_SCALE, "differences {differences:?} too small");
    let g32: Vec<f64> = in_f32.data().iter().map(|&v| f64::from(v)).collect();
    let e64 = relative_error(in_f64.data(), &differences);
    let e32 = relative_error(&g32, &differences);
    assert!(
        e64 <= TOL_F64,
        "f64 error {e64:e}; gradient {:?}, differences {differences:?}",
        in_f64.data()
    );
    assert!(
        e32 <= TOL_F32,
        "f32 error {e32:e}; gradient {g32:?}, differences {differences:?}"
    );
}

/// `gelu` after a dense layer with a bias row, the BERT feed-forward shape: the gradients of
/// `w` and `b` through `gelu(add_rows(matmul(x, w), b))` pass the harness's check.
#[test]
fn gelu_of_a_dense_layer_with_a_bias_row_passes_the_gradient_check() {
    gradient_check(
        "gelu_of_a_dense_layer_with_a_bias_row_passes_the_gradient_check",
        |b| {
            let x = input(b, "x", &[N, K]);
            let w = param(b, "w", &[K, H]);
            let bias = param(b, "b", &[H]);
            let xw = b.matmul(x, w).unwrap();
            let z = b.add_rows(xw, bias).unwrap();
            let y = b.gelu(z).unwrap();
            mse_to_target(b, y)
        },
    );
}
