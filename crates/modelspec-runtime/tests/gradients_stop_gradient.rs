//! `stop_gradient` on the CPU interpreter (DESIGN.md § 2.2, § 3.3, § 4, § 9 gate 3;
//! story:jepa-stop-gradient).
//!
//! `stop_gradient(x)` is `x` in the forward pass and passes no adjoint back to `x`. Autodiff does
//! not walk through it, and the pass `passes::stop_gradient` replaces each use of it by its
//! operand after autodiff, so the training program the interpreter runs holds no `StopGradient`.
//!
//! The case is the loss `mse(add(matmul(x, w), stop_gradient(matmul(x, w))), y)`. Its gradient
//! of `w` is compared, in `f64` and in `f32`, with central differences of the same function with
//! the stopped term held as a constant `c`, the value `matmul(x, w)` takes at the point:
//! `mse(add(matmul(x, w), c), y)`. That gradient is half of the gradient of the loss without
//! `stop_gradient`, `mse(add(matmul(x, w), matmul(x, w)), y)`.

use std::collections::BTreeMap;

use modelspec_core::autodiff::differentiate;
use modelspec_core::ir::tensor::{
    Initializer, ParamId, TensorOp, TensorProgram, TensorProgramBuilder, TensorType, ValueId,
};
use modelspec_core::ir::training::TrainingProgram;
use modelspec_core::ir::verify::verify_training;
use modelspec_core::passes::stop_gradient;
use modelspec_runtime::cpu::{Element, HostTensor, evaluate};

#[allow(dead_code)]
#[path = "common/finite_difference.rs"]
mod finite_difference;

use finite_difference::{H, MIN_SCALE, Rng, TOL_F32, TOL_F64, relative_error, seed_of};

/// Rows of `x`.
const N: u64 = 5;
/// Columns of `x`, rows of `w`.
const K: u64 = 4;
/// Columns of `w`.
const M: u64 = 3;

/// The forward program `mse(add(matmul(x, w), stopped(matmul(x, w))), y)` over the inputs `x`
/// and `y` and the parameter `w`, where `stopped` wraps the second term or not. Returns the
/// program, its loss, `w`'s id and the second term's value (the `StopGradient` when `stop`).
fn program(stop: bool) -> (TensorProgram, ValueId, ParamId, ValueId) {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", TensorType::f32(&[N, K])).unwrap();
    let w = b
        .declare_param(
            "Target.w",
            TensorType::f32(&[K, M]),
            Initializer::XavierUniform,
        )
        .unwrap();
    let read = b.param_read(w).unwrap();
    let online = b.matmul(x, read).unwrap();
    let second = b.matmul(x, read).unwrap();
    let second = if stop {
        let stopped = b.stop_gradient(second).unwrap();
        assert_eq!(
            b.type_of(stopped),
            Some(&TensorType::f32(&[N, M])),
            "x's type"
        );
        stopped
    } else {
        second
    };
    let sum = b.add(online, second).unwrap();
    let y = b.input("y", TensorType::f32(&[N, M])).unwrap();
    let loss = b.mse(sum, y).unwrap();
    b.output(loss).unwrap();
    (b.finish(), loss, w, second)
}

/// The point: `x`, `y` and `w`, drawn from the case's fixed seed.
struct Point {
    x: Vec<f32>,
    y: Vec<f32>,
    w: Vec<f32>,
}

impl Point {
    fn draw(name: &str) -> Self {
        let mut rng = Rng::new(seed_of(name));
        let mut values = |n: u64| (0..n).map(|_| rng.value()).collect::<Vec<f32>>();
        Self {
            x: values(N * K),
            y: values(N * M),
            w: values(K * M),
        }
    }

    fn inputs<T: Element>(&self, convert: impl Fn(f32) -> T) -> BTreeMap<String, HostTensor<T>> {
        let tensor = |shape: &[u64], data: &[f32]| {
            let data = data.iter().map(|&v| convert(v)).collect();
            HostTensor::new(shape.to_vec(), data).unwrap()
        };
        BTreeMap::from([
            ("x".to_owned(), tensor(&[N, K], &self.x)),
            ("y".to_owned(), tensor(&[N, M], &self.y)),
        ])
    }

    fn params<T: Element>(
        &self,
        w: ParamId,
        convert: impl Fn(f32) -> T,
    ) -> BTreeMap<ParamId, HostTensor<T>> {
        let data = self.w.iter().map(|&v| convert(v)).collect();
        BTreeMap::from([(w, HostTensor::new(vec![K, M], data).unwrap())])
    }
}

/// Differentiates `forward` at its loss and runs the stop-gradient pass; the result verifies and
/// gives `w` a gradient.
fn trained(forward: &TensorProgram, loss: ValueId, w: ParamId) -> TrainingProgram {
    let training = differentiate(forward, loss).unwrap();
    let training = stop_gradient::eliminate_training(training);
    verify_training(&training).unwrap();
    assert_eq!(
        training.gradients.keys().copied().collect::<Vec<_>>(),
        [w],
        "a gradient for w"
    );
    training
}

/// The gradient of `w`, the second output, evaluated in `T`.
fn gradient<T: Element>(
    training: &TrainingProgram,
    w: ParamId,
    point: &Point,
    convert: impl Fn(f32) -> T + Copy,
) -> Vec<T> {
    let out = evaluate(
        &training.program,
        &point.inputs(convert),
        &point.params(w, convert),
    )
    .unwrap();
    out[1].data().to_vec()
}

/// The held function `mse(add(matmul(x, w), c), y)`, `c` an input.
fn held_program() -> (TensorProgram, ParamId) {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", TensorType::f32(&[N, K])).unwrap();
    let w = b
        .declare_param(
            "Target.w",
            TensorType::f32(&[K, M]),
            Initializer::XavierUniform,
        )
        .unwrap();
    let read = b.param_read(w).unwrap();
    let online = b.matmul(x, read).unwrap();
    let c = b.input("c", TensorType::f32(&[N, M])).unwrap();
    let sum = b.add(online, c).unwrap();
    let y = b.input("y", TensorType::f32(&[N, M])).unwrap();
    let loss = b.mse(sum, y).unwrap();
    b.output(loss).unwrap();
    (b.finish(), w)
}

/// `matmul(x, w)` at the point, in `f64`: the stopped term.
fn stopped_term(point: &Point) -> Vec<f64> {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", TensorType::f32(&[N, K])).unwrap();
    let w = b.input("w", TensorType::f32(&[K, M])).unwrap();
    let xw = b.matmul(x, w).unwrap();
    b.output(xw).unwrap();
    let w_data = point.w.iter().map(|&v| f64::from(v)).collect();
    let mut inputs = point.inputs(f64::from);
    inputs.remove("y");
    inputs.insert("w".to_owned(), HostTensor::new(vec![K, M], w_data).unwrap());
    evaluate(&b.finish(), &inputs, &BTreeMap::new()).unwrap()[0]
        .data()
        .to_vec()
}

/// Central differences in `f64` of the held function at the point, `c` the stopped term there.
fn held_differences(point: &Point) -> Vec<f64> {
    let (held, w) = held_program();
    let mut inputs = point.inputs(f64::from);
    inputs.insert(
        "c".to_owned(),
        HostTensor::new(vec![N, M], stopped_term(point)).unwrap(),
    );
    let loss_at = |index: usize, delta: f64| {
        let mut data: Vec<f64> = point.w.iter().map(|&v| f64::from(v)).collect();
        data[index] += delta;
        let params = BTreeMap::from([(w, HostTensor::new(vec![K, M], data).unwrap())]);
        evaluate(&held, &inputs, &params).unwrap()[0].data()[0]
    };
    (0..point.w.len())
        .map(|i| (loss_at(i, H) - loss_at(i, -H)) / (2.0 * H))
        .collect()
}

/// The acceptance case: the gradient of `w` matches central differences of the function with the
/// stopped term held constant, within `TOL_F64` in `f64` and `TOL_F32` in `f32`.
#[test]
fn stop_gradient_gradient_of_w_matches_the_function_with_the_stopped_term_held() {
    let name = "stop_gradient_gradient_of_w_matches_the_function_with_the_stopped_term_held";
    let (forward, loss, w, stopped) = program(true);
    assert_eq!(forward.node(stopped).unwrap().op, TensorOp::StopGradient);
    let training = trained(&forward, loss, w);
    let point = Point::draw(name);
    let differences = held_differences(&point);
    let scale = differences.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        scale >= MIN_SCALE,
        "differences {differences:?} too small to compare"
    );

    let g64 = gradient(&training, w, &point, f64::from);
    let g32: Vec<f64> = gradient(&training, w, &point, |v| v)
        .into_iter()
        .map(f64::from)
        .collect();
    let e64 = relative_error(&g64, &differences);
    let e32 = relative_error(&g32, &differences);
    assert!(
        e64 <= TOL_F64,
        "f64 error {e64:e}; gradient {g64:?}, differences {differences:?}"
    );
    assert!(
        e32 <= TOL_F32,
        "f32 error {e32:e}; gradient {g32:?}, differences {differences:?}"
    );
}

/// The gradient with `stop_gradient` is exactly half of the gradient without it, in `f64` and in
/// `f32`: the second term's adjoint is dropped, not scaled.
#[test]
fn stop_gradient_gradient_is_half_of_the_gradient_without_it() {
    let name = "stop_gradient_gradient_is_half_of_the_gradient_without_it";
    let point = Point::draw(name);
    let (stopped, stopped_loss, w, _) = program(true);
    let (full, full_loss, full_w, _) = program(false);
    let stopped = trained(&stopped, stopped_loss, w);
    let full = trained(&full, full_loss, full_w);

    let half64 = gradient(&stopped, w, &point, f64::from);
    let whole64 = gradient(&full, full_w, &point, f64::from);
    let doubled: Vec<f64> = half64.iter().map(|g| 2.0 * g).collect();
    assert_eq!(doubled, whole64, "f64");

    let half32 = gradient(&stopped, w, &point, |v| v);
    let whole32 = gradient(&full, full_w, &point, |v| v);
    let doubled: Vec<f32> = half32.iter().map(|g| 2.0 * g).collect();
    assert_eq!(doubled, whole32, "f32");
    assert!(
        half64.iter().any(|g| *g != 0.0),
        "a gradient that is not zero"
    );
}

/// The forward value of `stop_gradient(x)` is `x`, bit for bit: the losses with and without it
/// are equal at the point.
#[test]
fn stop_gradient_forward_value_is_its_operand() {
    let point = Point::draw("stop_gradient_forward_value_is_its_operand");
    let (stopped, _, w, _) = program(true);
    let (full, _, full_w, _) = program(false);
    let with = evaluate(&stopped, &point.inputs(|v| v), &point.params(w, |v| v)).unwrap();
    let without = evaluate(&full, &point.inputs(|v| v), &point.params(full_w, |v| v)).unwrap();
    assert_eq!(with[0].data(), without[0].data());
}

/// The pass leaves no `StopGradient` in the training program, and every use of the stopped value
/// reads its operand instead; autodiff built no adjoint for the stopped term.
#[test]
fn the_training_program_holds_no_stop_gradient() {
    let (forward, loss, w, stopped) = program(true);
    let operand = forward.node(stopped).unwrap().operands[0];
    let differentiated = differentiate(&forward, loss).unwrap();
    assert!(
        differentiated
            .program
            .nodes
            .iter()
            .any(|n| n.op == TensorOp::StopGradient),
        "autodiff keeps the forward nodes"
    );
    let training = trained(&forward, loss, w);
    let program = &training.program;
    assert!(
        program.nodes.iter().all(|n| n.op != TensorOp::StopGradient),
        "{program:?}"
    );
    assert!(program.node(stopped).is_none());
    assert!(!training.subgraphs.contains_key(&stopped));
    assert!(
        program.nodes.iter().all(|n| !n.operands.contains(&stopped)),
        "no node reads the removed value"
    );
    assert!(
        program.nodes.iter().any(|n| n.operands.contains(&operand)),
        "the add reads the stopped operand directly"
    );
    let matmuls_of_x_transposed = program
        .nodes
        .iter()
        .filter(|n| n.op == TensorOp::MatMul)
        .count();
    assert_eq!(
        matmuls_of_x_transposed, 3,
        "two forward matmuls and one gradient matmul: no adjoint for the stopped term"
    );
}
