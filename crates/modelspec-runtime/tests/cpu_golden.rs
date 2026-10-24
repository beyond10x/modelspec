//! CPU golden values for every registry op (DESIGN.md § 9 gate 2) and the reduction order policy
//! of DESIGN.md § 2.2.
//!
//! Every expected value is hand-computed and exactly representable in the element type, so each
//! result is compared bit for bit, in `f32` and in `f64`. Programs are typed `f32` (the only M0
//! dtype); the interpreter's element type decides the arithmetic.

use std::collections::BTreeMap;

use modelspec_core::ir::tensor::{
    Dim, Initializer, ParamId, TensorProgram, TensorProgramBuilder, TensorType, ValueId,
};
use modelspec_core::ir::verify::VerifyError;
use modelspec_runtime::cpu::{
    Element, Evaluation, HostTensor, RuntimeError, TensorError, evaluate, evaluate_all,
};

/// `2^53`: adding 1 to it rounds back to it in both `f32` and `f64`.
const BIG: f64 = 9_007_199_254_740_992.0;

/// An element type the cases run in, with bit-exact comparison.
trait Real: Element {
    /// `v` in this type; panics unless the conversion is exact.
    fn of(v: f64) -> Self;
    fn bits(self) -> u64;
    /// The expected value for this type when `f32` and `f64` differ.
    fn pick(in_f32: f64, in_f64: f64) -> f64;
}

impl Real for f32 {
    fn of(v: f64) -> Self {
        let x = v as f32;
        assert_eq!(
            f64::from(x).to_bits(),
            v.to_bits(),
            "{v} is not exact in f32"
        );
        x
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn pick(in_f32: f64, _: f64) -> f64 {
        in_f32
    }
}

impl Real for f64 {
    fn of(v: f64) -> Self {
        v
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn pick(_: f64, in_f64: f64) -> f64 {
        in_f64
    }
}

fn ty(shape: &[u64]) -> TensorType {
    TensorType::f32(shape)
}

fn t<T: Real>(shape: &[u64], data: &[f64]) -> HostTensor<T> {
    HostTensor::new(shape.to_vec(), data.iter().map(|&v| T::of(v)).collect()).unwrap()
}

fn assert_bits<T: Real>(got: &HostTensor<T>, shape: &[u64], expected: &[f64]) {
    assert_eq!(got.shape(), shape, "shape");
    let got_bits: Vec<u64> = got.data().iter().map(|&v| v.bits()).collect();
    let want_bits: Vec<u64> = expected.iter().map(|&v| T::of(v).bits()).collect();
    assert_eq!(
        got_bits,
        want_bits,
        "values: got {:?}, expected {:?}",
        got.data(),
        expected
    );
}

type Inputs<T> = BTreeMap<String, HostTensor<T>>;
type Params<T> = BTreeMap<ParamId, HostTensor<T>>;

fn inputs<T: Real>(named: Vec<(&str, HostTensor<T>)>) -> Inputs<T> {
    named.into_iter().map(|(n, v)| (n.to_string(), v)).collect()
}

fn no_params<T: Real>() -> Params<T> {
    BTreeMap::new()
}

/// Runs `program` on named inputs and no parameters; returns its outputs.
fn run<T: Real>(program: &TensorProgram, named: Vec<(&str, HostTensor<T>)>) -> Vec<HostTensor<T>> {
    evaluate(program, &inputs(named), &no_params()).unwrap()
}

/// `op(x)` over one input `x` of shape `shape`.
fn unary(
    shape: &[u64],
    op: impl FnOnce(&mut TensorProgramBuilder, ValueId) -> ValueId,
) -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", ty(shape)).unwrap();
    let y = op(&mut b, x);
    b.output(y).unwrap();
    b.finish()
}

/// `op(a, b)` over inputs `a` and `b` of the given shapes.
fn binary(
    a_shape: &[u64],
    b_shape: &[u64],
    op: impl FnOnce(&mut TensorProgramBuilder, ValueId, ValueId) -> ValueId,
) -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("a", ty(a_shape)).unwrap();
    let y = b.input("b", ty(b_shape)).unwrap();
    let z = op(&mut b, x, y);
    b.output(z).unwrap();
    b.finish()
}

fn eval_unary<T: Real>(
    shape: &[u64],
    data: &[f64],
    op: impl FnOnce(&mut TensorProgramBuilder, ValueId) -> ValueId,
) -> HostTensor<T> {
    let program = unary(shape, op);
    run(&program, vec![("x", t::<T>(shape, data))]).remove(0)
}

fn eval_binary<T: Real>(
    (a_shape, a): (&[u64], &[f64]),
    (b_shape, bv): (&[u64], &[f64]),
    op: impl FnOnce(&mut TensorProgramBuilder, ValueId, ValueId) -> ValueId,
) -> HostTensor<T> {
    let program = binary(a_shape, b_shape, op);
    run(
        &program,
        vec![("a", t::<T>(a_shape, a)), ("b", t::<T>(b_shape, bv))],
    )
    .remove(0)
}

// ---- leaves ----

fn input_passes_its_value_through<T: Real>() {
    let got = eval_unary::<T>(&[2, 2], &[1.0, -2.0, 0.5, 8.0], |_, x| x);
    assert_bits(&got, &[2, 2], &[1.0, -2.0, 0.5, 8.0]);
}

fn param_read_passes_the_parameter_value_through<T: Real>() {
    let mut b = TensorProgramBuilder::new();
    let w = b
        .declare_param("M.w", ty(&[3, 1]), Initializer::XavierUniform)
        .unwrap();
    let v = b.param_read(w).unwrap();
    b.output(v).unwrap();
    let program = b.finish();
    let params = BTreeMap::from([(w, t::<T>(&[3, 1], &[0.25, -1.0, 3.0]))]);
    let out = evaluate(&program, &inputs::<T>(vec![]), &params).unwrap();
    assert_bits(&out[0], &[3, 1], &[0.25, -1.0, 3.0]);
}

fn scalar_constant_is_its_f32_value_widened<T: Real>() {
    let mut b = TensorProgramBuilder::new();
    let c = b.scalar(2.5);
    let tenth = b.scalar(0.1);
    b.output(c).unwrap();
    b.output(tenth).unwrap();
    let out = run::<T>(&b.finish(), vec![]);
    assert_bits(&out[0], &[], &[2.5]);
    assert_bits(&out[1], &[], &[f64::from(0.1f32)]);
}

// ---- elementwise ----

const A4: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
const B4: [f64; 4] = [10.0, 20.0, 30.0, 40.0];

fn add_same_shape<T: Real>() {
    let got = eval_binary::<T>((&[2, 2], &A4), (&[2, 2], &B4), |b, x, y| {
        b.add(x, y).unwrap()
    });
    assert_bits(&got, &[2, 2], &[11.0, 22.0, 33.0, 44.0]);
}

fn add_tensor_and_scalar<T: Real>() {
    let got = eval_binary::<T>((&[2, 2], &A4), (&[], &[0.5]), |b, x, y| {
        b.add(x, y).unwrap()
    });
    assert_bits(&got, &[2, 2], &[1.5, 2.5, 3.5, 4.5]);
}

fn add_scalar_and_tensor<T: Real>() {
    let got = eval_binary::<T>((&[], &[0.5]), (&[2, 2], &A4), |b, x, y| {
        b.add(x, y).unwrap()
    });
    assert_bits(&got, &[2, 2], &[1.5, 2.5, 3.5, 4.5]);
}

fn add_two_scalars<T: Real>() {
    let got = eval_binary::<T>((&[], &[0.5]), (&[], &[-2.0]), |b, x, y| {
        b.add(x, y).unwrap()
    });
    assert_bits(&got, &[], &[-1.5]);
}

fn sub_same_shape<T: Real>() {
    let got = eval_binary::<T>((&[2, 2], &A4), (&[2, 2], &B4), |b, x, y| {
        b.sub(x, y).unwrap()
    });
    assert_bits(&got, &[2, 2], &[-9.0, -18.0, -27.0, -36.0]);
}

fn sub_tensor_minus_scalar<T: Real>() {
    let got = eval_binary::<T>((&[2, 2], &A4), (&[], &[0.5]), |b, x, y| {
        b.sub(x, y).unwrap()
    });
    assert_bits(&got, &[2, 2], &[0.5, 1.5, 2.5, 3.5]);
}

fn sub_scalar_minus_tensor<T: Real>() {
    let got = eval_binary::<T>((&[], &[0.5]), (&[2, 2], &A4), |b, x, y| {
        b.sub(x, y).unwrap()
    });
    assert_bits(&got, &[2, 2], &[-0.5, -1.5, -2.5, -3.5]);
}

fn mul_same_shape<T: Real>() {
    let got = eval_binary::<T>((&[2, 2], &A4), (&[2, 2], &B4), |b, x, y| {
        b.mul(x, y).unwrap()
    });
    assert_bits(&got, &[2, 2], &[10.0, 40.0, 90.0, 160.0]);
}

fn mul_tensor_and_scalar<T: Real>() {
    let got = eval_binary::<T>((&[2, 2], &A4), (&[], &[0.5]), |b, x, y| {
        b.mul(x, y).unwrap()
    });
    assert_bits(&got, &[2, 2], &[0.5, 1.0, 1.5, 2.0]);
}

fn mul_scalar_and_tensor<T: Real>() {
    let got = eval_binary::<T>((&[], &[-3.0]), (&[2, 2], &A4), |b, x, y| {
        b.mul(x, y).unwrap()
    });
    assert_bits(&got, &[2, 2], &[-3.0, -6.0, -9.0, -12.0]);
}

fn arithmetic_runs_in_the_element_type<T: Real>() {
    let got = eval_binary::<T>((&[1], &[16_777_216.0]), (&[1], &[1.0]), |b, x, y| {
        b.add(x, y).unwrap()
    });
    assert_bits(&got, &[1], &[T::pick(16_777_216.0, 16_777_217.0)]);
}

fn relu_keeps_positives_and_zeroes_the_rest<T: Real>() {
    let got = eval_unary::<T>(&[2, 3], &[-2.0, -0.0, 0.0, 0.5, 3.0, -7.0], |b, x| {
        b.relu(x).unwrap()
    });
    assert_bits(&got, &[2, 3], &[0.0, 0.0, 0.0, 0.5, 3.0, 0.0]);
}

fn relu_grad_passes_g_where_a_is_positive<T: Real>() {
    let got = eval_binary::<T>(
        (&[2, 3], &[-1.0, 0.0, 2.0, -0.0, 3.0, -4.0]),
        (&[2, 3], &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        |b, a, g| b.relu_grad(a, g).unwrap(),
    );
    assert_bits(&got, &[2, 3], &[0.0, 0.0, 7.0, 0.0, 9.0, 0.0]);
}

fn broadcast_scalar_fills_the_shape<T: Real>() {
    let got = eval_unary::<T>(&[], &[1.5], |b, x| {
        b.broadcast_scalar(x, vec![Dim::Static(2), Dim::Static(3)])
            .unwrap()
    });
    assert_bits(&got, &[2, 3], &[1.5; 6]);
}

fn transpose_swaps_the_two_axes<T: Real>() {
    let got = eval_unary::<T>(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], |b, x| {
        b.transpose(x).unwrap()
    });
    assert_bits(&got, &[3, 2], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

// ---- matmul ----

fn matmul_non_square_2x3_by_3x4<T: Real>() {
    let got = eval_binary::<T>(
        (&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        (
            &[3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        ),
        |b, x, y| b.matmul(x, y).unwrap(),
    );
    assert_bits(
        &got,
        &[2, 4],
        &[38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0],
    );
}

/// Row 0's products are `[2^53, 1, -2^53, 1]` in index order: ascending accumulation gives 1;
/// descending or pairwise gives 0. Row 1, `[1, 2^53, 1, -2^53]`, gives 0 ascending and 1
/// descending.
fn matmul_accumulates_over_the_contracting_index_ascending<T: Real>() {
    let got = eval_binary::<T>(
        (&[2, 4], &[BIG, 1.0, -BIG, 1.0, 1.0, BIG, 1.0, -BIG]),
        (&[4, 1], &[1.0, 1.0, 1.0, 1.0]),
        |b, x, y| b.matmul(x, y).unwrap(),
    );
    assert_bits(&got, &[2, 1], &[1.0, 0.0]);
}

/// The accumulator starts at the first product, not at `+0`: `-0 + -0` is `-0`.
fn matmul_accumulation_starts_from_the_first_product<T: Real>() {
    let got = eval_binary::<T>(
        (&[1, 2], &[-1.0, -1.0]),
        (&[2, 1], &[0.0, 0.0]),
        |b, x, y| b.matmul(x, y).unwrap(),
    );
    assert_bits(&got, &[1, 1], &[-0.0]);
}

/// DESIGN.md § 12 ADR-009: the second product `(1 + 2^-12)^2 = 1 + 2^-11 + 2^-24` is fused into
/// the sum `-(1 + 2^-11)`, one rounding, so the sum is `2^-24` in both types; rounding the product
/// first would give 0 in `f32`. In `f64` the product is exact either way.
fn matmul_fuses_each_later_product_into_its_sum<T: Real>() {
    let e = 1.0 + 2f64.powi(-12);
    let got = eval_binary::<T>(
        (&[1, 2], &[-(1.0 + 2f64.powi(-11)), e]),
        (&[2, 1], &[1.0, e]),
        |b, x, y| b.matmul(x, y).unwrap(),
    );
    assert_bits(&got, &[1, 1], &[2f64.powi(-24)]);
}

// ---- reductions ----

fn reduce_sum_of_every_element<T: Real>() {
    let got = eval_unary::<T>(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], |b, x| {
        b.reduce_sum(x).unwrap()
    });
    assert_bits(&got, &[], &[21.0]);
}

/// `[[2^53, 1], [-2^53, 1]]`: sequential row-major gives 1; column-major gives 2; pairwise and
/// reverse give 0.
fn reduce_sum_accumulates_sequentially_in_row_major_order<T: Real>() {
    let got = eval_unary::<T>(&[2, 2], &[BIG, 1.0, -BIG, 1.0], |b, x| {
        b.reduce_sum(x).unwrap()
    });
    assert_bits(&got, &[], &[1.0]);
}

fn reduce_sum_starts_from_the_first_term<T: Real>() {
    let got = eval_unary::<T>(&[2], &[-0.0, -0.0], |b, x| b.reduce_sum(x).unwrap());
    assert_bits(&got, &[], &[-0.0]);
}

fn reduce_mean_divides_by_the_element_count<T: Real>() {
    let three = eval_unary::<T>(&[3], &[1.0, 2.0, 3.0], |b, x| b.reduce_mean(x).unwrap());
    assert_bits(&three, &[], &[2.0]);
    let six = eval_unary::<T>(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 3.0], |b, x| {
        b.reduce_mean(x).unwrap()
    });
    assert_bits(&six, &[], &[3.0]);
}

fn reduce_mean_is_the_ordered_sum_over_the_count<T: Real>() {
    let got = eval_unary::<T>(&[2, 2], &[BIG, 1.0, -BIG, 1.0], |b, x| {
        b.reduce_mean(x).unwrap()
    });
    assert_bits(&got, &[], &[0.25]);
}

/// `d = a - b = [1, 2, 2, -1, -2, 2]`, squares sum to 18, over 6 elements.
fn mse_of_two_tensors<T: Real>() {
    let got = eval_binary::<T>(
        (&[2, 3], &[3.0, 5.0, 4.0, 0.0, 1.0, 7.0]),
        (&[2, 3], &[2.0, 3.0, 2.0, 1.0, 3.0, 5.0]),
        |b, x, y| b.mse(x, y).unwrap(),
    );
    assert_bits(&got, &[], &[3.0]);
}

/// Squares `[s, L, s, s]` with `s` below half an ulp of `L` and `2s` above it: summed in
/// row-major order each `s` is absorbed and the result is `L / 4`; column-major, reverse or
/// pairwise order adds `2s` first and lands one ulp higher. The first case decides `f64`
/// (`L = 2^54`, `s = 1.5625`), the second `f32` (`L = 2^24`, `s = 0.765625`).
fn mse_sums_squared_differences_in_row_major_order<T: Real>() {
    let wide = eval_binary::<T>(
        (&[2, 2], &[1.5, 134_217_728.0, 1.5, 1.5]),
        (&[2, 2], &[0.25, 0.0, 0.25, 0.25]),
        |b, x, y| b.mse(x, y).unwrap(),
    );
    assert_bits(&wide, &[], &[4_503_599_627_370_496.0]);
    let narrow = eval_binary::<T>(
        (&[2, 2], &[1.0, 4096.0, 1.0, 1.0]),
        (&[2, 2], &[0.125, 0.0, 0.125, 0.125]),
        |b, x, y| b.mse(x, y).unwrap(),
    );
    assert_bits(&narrow, &[], &[T::pick(4_194_304.0, 4_194_304.574_218_75)]);
}

// ---- whole programs ----

/// `loss = mse(x @ w, y)`, outputs `[loss, prediction, loss]`.
fn linear_regression() -> (TensorProgram, ParamId, [ValueId; 5]) {
    let mut b = TensorProgramBuilder::new();
    let w = b
        .declare_param("LinearReg.w", ty(&[2, 1]), Initializer::XavierUniform)
        .unwrap();
    let x = b.input("x", ty(&[2, 2])).unwrap();
    let wv = b.param_read(w).unwrap();
    let prediction = b.matmul(x, wv).unwrap();
    let y = b.input("y", ty(&[2, 1])).unwrap();
    let loss = b.mse(prediction, y).unwrap();
    b.output(loss).unwrap();
    b.output(prediction).unwrap();
    b.output(loss).unwrap();
    (b.finish(), w, [x, wv, prediction, y, loss])
}

fn linear_regression_values<T: Real>() -> (Inputs<T>, Params<T>, ParamId) {
    let (_, w, _) = linear_regression();
    let named = inputs(vec![
        ("x", t::<T>(&[2, 2], &[1.0, 2.0, 3.0, 4.0])),
        ("y", t::<T>(&[2, 1], &[4.0, 12.0])),
    ]);
    let params = BTreeMap::from([(w, t::<T>(&[2, 1], &[1.0, 2.0]))]);
    (named, params, w)
}

fn outputs_come_in_program_output_order<T: Real>() {
    let (program, _, _) = linear_regression();
    let (named, params, _) = linear_regression_values::<T>();
    let out = evaluate(&program, &named, &params).unwrap();
    assert_eq!(out.len(), 3);
    assert_bits(&out[0], &[], &[1.0]);
    assert_bits(&out[1], &[2, 1], &[5.0, 11.0]);
    assert_bits(&out[2], &[], &[1.0]);
}

fn evaluate_all_returns_every_value<T: Real>() {
    let (program, _, [x, wv, prediction, y, loss]) = linear_regression();
    let (named, params, _) = linear_regression_values::<T>();
    let Evaluation { outputs, values } = evaluate_all(&program, &named, &params).unwrap();
    assert_eq!(outputs.len(), 3);
    assert_bits(&outputs[1], &[2, 1], &[5.0, 11.0]);
    assert_eq!(
        values.keys().copied().collect::<Vec<_>>(),
        program.nodes.iter().map(|n| n.id).collect::<Vec<_>>()
    );
    assert_bits(&values[&x], &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    assert_bits(&values[&wv], &[2, 1], &[1.0, 2.0]);
    assert_bits(&values[&prediction], &[2, 1], &[5.0, 11.0]);
    assert_bits(&values[&y], &[2, 1], &[4.0, 12.0]);
    assert_bits(&values[&loss], &[], &[1.0]);
}

macro_rules! in_both_types {
    ($($case:ident),* $(,)?) => {
        mod in_f32 {
            $( #[test] fn $case() { super::$case::<f32>() } )*
        }
        mod in_f64 {
            $( #[test] fn $case() { super::$case::<f64>() } )*
        }
    };
}

in_both_types!(
    input_passes_its_value_through,
    param_read_passes_the_parameter_value_through,
    scalar_constant_is_its_f32_value_widened,
    add_same_shape,
    add_tensor_and_scalar,
    add_scalar_and_tensor,
    add_two_scalars,
    sub_same_shape,
    sub_tensor_minus_scalar,
    sub_scalar_minus_tensor,
    mul_same_shape,
    mul_tensor_and_scalar,
    mul_scalar_and_tensor,
    arithmetic_runs_in_the_element_type,
    relu_keeps_positives_and_zeroes_the_rest,
    relu_grad_passes_g_where_a_is_positive,
    broadcast_scalar_fills_the_shape,
    transpose_swaps_the_two_axes,
    matmul_non_square_2x3_by_3x4,
    matmul_accumulates_over_the_contracting_index_ascending,
    matmul_accumulation_starts_from_the_first_product,
    matmul_fuses_each_later_product_into_its_sum,
    reduce_sum_of_every_element,
    reduce_sum_accumulates_sequentially_in_row_major_order,
    reduce_sum_starts_from_the_first_term,
    reduce_mean_divides_by_the_element_count,
    reduce_mean_is_the_ordered_sum_over_the_count,
    mse_of_two_tensors,
    mse_sums_squared_differences_in_row_major_order,
    outputs_come_in_program_output_order,
    evaluate_all_returns_every_value,
);

// ---- runtime errors ----

#[test]
fn the_verifier_runs_before_anything_else() {
    let (mut program, _, [_, _, prediction, _, _]) = linear_regression();
    let node = program
        .nodes
        .iter_mut()
        .find(|n| n.id == prediction)
        .unwrap();
    node.ty = ty(&[1, 1]);
    let err = evaluate::<f32>(&program, &BTreeMap::new(), &BTreeMap::new()).unwrap_err();
    assert_eq!(
        err,
        RuntimeError::Verify(VerifyError::TypeMismatch {
            node: prediction,
            declared: ty(&[1, 1]),
            inferred: ty(&[2, 1]),
        })
    );
}

#[test]
fn a_missing_input_is_refused() {
    let (program, _, _) = linear_regression();
    let (mut named, params, _) = linear_regression_values::<f32>();
    named.remove("y");
    assert_eq!(
        evaluate(&program, &named, &params).unwrap_err(),
        RuntimeError::MissingInput { name: "y".into() }
    );
}

#[test]
fn a_value_for_no_input_is_refused() {
    let (program, _, _) = linear_regression();
    let (mut named, params, _) = linear_regression_values::<f32>();
    named.insert("z".into(), t::<f32>(&[], &[1.0]));
    assert_eq!(
        evaluate(&program, &named, &params).unwrap_err(),
        RuntimeError::UnexpectedInput { name: "z".into() }
    );
}

#[test]
fn an_input_of_the_wrong_shape_but_the_right_count_is_refused() {
    let (program, _, _) = linear_regression();
    let (mut named, params, _) = linear_regression_values::<f32>();
    named.insert("y".into(), t::<f32>(&[1, 2], &[4.0, 12.0]));
    assert_eq!(
        evaluate(&program, &named, &params).unwrap_err(),
        RuntimeError::InputShape {
            name: "y".into(),
            declared: ty(&[2, 1]),
            got: vec![1, 2],
        }
    );
}

#[test]
fn an_input_of_the_wrong_rank_is_refused() {
    let (program, _, _) = linear_regression();
    let (mut named, params, _) = linear_regression_values::<f64>();
    named.insert("x".into(), t::<f64>(&[4], &[1.0, 2.0, 3.0, 4.0]));
    assert_eq!(
        evaluate(&program, &named, &params).unwrap_err(),
        RuntimeError::InputShape {
            name: "x".into(),
            declared: ty(&[2, 2]),
            got: vec![4],
        }
    );
}

/// `[2]` matches every dimension of `[2, 1]` it has; only the rank tells them apart.
#[test]
fn an_input_whose_shape_is_a_prefix_of_the_declared_shape_is_refused() {
    let (program, _, _) = linear_regression();
    let (mut named, params, _) = linear_regression_values::<f32>();
    named.insert("y".into(), t::<f32>(&[2], &[4.0, 12.0]));
    assert_eq!(
        evaluate(&program, &named, &params).unwrap_err(),
        RuntimeError::InputShape {
            name: "y".into(),
            declared: ty(&[2, 1]),
            got: vec![2],
        }
    );
}

#[test]
fn a_parameter_whose_shape_is_a_prefix_of_the_declared_shape_is_refused() {
    let (program, _, _) = linear_regression();
    let (named, mut params, w) = linear_regression_values::<f32>();
    params.insert(w, t::<f32>(&[2], &[1.0, 2.0]));
    assert_eq!(
        evaluate(&program, &named, &params).unwrap_err(),
        RuntimeError::ParamShape {
            param: w,
            path: "LinearReg.w".into(),
            declared: ty(&[2, 1]),
            got: vec![2],
        }
    );
}

#[test]
fn a_missing_parameter_is_refused() {
    let (program, _, _) = linear_regression();
    let (named, mut params, w) = linear_regression_values::<f32>();
    params.remove(&w);
    assert_eq!(
        evaluate(&program, &named, &params).unwrap_err(),
        RuntimeError::MissingParam {
            param: w,
            path: "LinearReg.w".into(),
        }
    );
}

#[test]
fn a_declared_parameter_no_node_reads_is_still_required() {
    let mut b = TensorProgramBuilder::new();
    let w = b
        .declare_param("M.unused", ty(&[1]), Initializer::XavierUniform)
        .unwrap();
    let c = b.scalar(1.0);
    b.output(c).unwrap();
    let program = b.finish();
    assert_eq!(
        evaluate::<f32>(&program, &BTreeMap::new(), &BTreeMap::new()).unwrap_err(),
        RuntimeError::MissingParam {
            param: w,
            path: "M.unused".into(),
        }
    );
}

#[test]
fn a_value_for_an_undeclared_parameter_is_refused() {
    let (program, _, _) = linear_regression();
    let (named, mut params, _) = linear_regression_values::<f32>();
    params.insert(ParamId(7), t::<f32>(&[], &[1.0]));
    assert_eq!(
        evaluate(&program, &named, &params).unwrap_err(),
        RuntimeError::UndeclaredParam { param: ParamId(7) }
    );
}

#[test]
fn a_parameter_of_the_wrong_shape_is_refused() {
    let (program, _, _) = linear_regression();
    let (named, mut params, w) = linear_regression_values::<f32>();
    params.insert(w, t::<f32>(&[1, 2], &[1.0, 2.0]));
    assert_eq!(
        evaluate(&program, &named, &params).unwrap_err(),
        RuntimeError::ParamShape {
            param: w,
            path: "LinearReg.w".into(),
            declared: ty(&[2, 1]),
            got: vec![1, 2],
        }
    );
}

#[test]
fn evaluate_all_refuses_what_evaluate_refuses() {
    let (program, _, _) = linear_regression();
    let (mut named, params, _) = linear_regression_values::<f32>();
    named.remove("x");
    assert_eq!(
        evaluate_all(&program, &named, &params).unwrap_err(),
        RuntimeError::MissingInput { name: "x".into() }
    );
}

// ---- allocation (DESIGN.md § 2.2 "allocations over backend limits", § 8.1) ----

/// `[2^24, 1] x [1, 2^24]` in `f32` verifies, but its result needs 2^50 bytes (1 PiB), beyond the
/// host's address space, so the allocation is refused at once and nothing is written. The two
/// operands are 64 MiB each. (A result over `isize::MAX` bytes would need an 8 GiB operand.)
#[test]
fn a_matmul_result_beyond_host_memory_is_an_allocation_error() {
    let n = 1u64 << 24;
    let program = binary(&[n, 1], &[1, n], |b, x, y| b.matmul(x, y).unwrap());
    let ones = vec![1.0f32; n as usize];
    let named = inputs(vec![
        ("a", HostTensor::new(vec![n, 1], ones.clone()).unwrap()),
        ("b", HostTensor::new(vec![1, n], ones).unwrap()),
    ]);
    let err = evaluate(&program, &named, &no_params()).unwrap_err();
    assert_eq!(
        err,
        RuntimeError::Allocation {
            value: ValueId(2),
            bytes: 1 << 50,
        }
    );
    assert_eq!(
        err.to_string(),
        "cannot allocate 1125899906842624 bytes for %2"
    );
}

/// `broadcast_scalar` to `[2^61]` verifies (2^63 bytes in the program's `f32`), but in `f64` the
/// value needs 2^64 bytes, which `u64` cannot count; the error still names the exact byte count.
#[test]
fn a_value_whose_f64_byte_count_overflows_u64_is_an_allocation_error() {
    let program = unary(&[], |b, x| {
        b.broadcast_scalar(x, vec![Dim::Static(1 << 61)]).unwrap()
    });
    let named = inputs(vec![("x", HostTensor::scalar(1.0f64))]);
    assert_eq!(
        evaluate(&program, &named, &no_params()).unwrap_err(),
        RuntimeError::Allocation {
            value: ValueId(1),
            bytes: 1 << 64,
        }
    );
}

// ---- host tensors ----

#[test]
fn host_tensor_holds_shape_and_row_major_data() {
    let tensor = HostTensor::new(vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(tensor.len(), 6);
    assert_eq!(tensor.rank(), 2);
    assert_eq!(tensor.into_data(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn host_tensor_scalar_is_rank_zero_with_one_element() {
    let s = HostTensor::scalar(2.5f64);
    assert_eq!(s.shape(), &[] as &[u64]);
    assert_eq!(s.data(), &[2.5]);
    assert_eq!(s.rank(), 0);
}

#[test]
fn host_tensor_refuses_data_of_the_wrong_length() {
    assert_eq!(
        HostTensor::new(vec![2, 3], vec![0.0f32; 5]).unwrap_err(),
        TensorError::LengthMismatch {
            shape: vec![2, 3],
            expected: 6,
            got: 5,
        }
    );
}

#[test]
fn host_tensor_refuses_an_element_count_that_overflows() {
    assert_eq!(
        HostTensor::new(vec![u64::MAX, 2], Vec::<f64>::new()).unwrap_err(),
        TensorError::CountOverflow {
            shape: vec![u64::MAX, 2],
        }
    );
}
