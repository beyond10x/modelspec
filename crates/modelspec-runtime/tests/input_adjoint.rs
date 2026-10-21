//! Finite-difference checks of the input adjoint (DESIGN.md § 4; scenarios
//! `observatory.adjoint.finite-difference` and `observatory.adjoint.unread-input` of
//! `story:observatory-input-adjoint`).
//!
//! Each case asks [`differentiate_inputs`] for the adjoint of named inputs of a rank-0 target and
//! runs the result on the CPU interpreter, in `f64` and in `f32`. Each adjoint is compared with
//! central finite differences of the target, evaluated in `f64` on the program the target was
//! taken from, not on the adjoint program. The constants and the error measure are those of
//! `tests/gradients.rs`: `max_i |g_i - d_i| / max_i |d_i|`, at most [`TOL_F64`] in `f64` and
//! [`TOL_F32`] in `f32`, at a point drawn from a fixed test-local seed with every relu operand at
//! least [`KINK_MARGIN`] from the kink.

use std::collections::BTreeMap;

use modelspec_core::autodiff::{
    InputAdjoint, TARGET_MASK, differentiate_inputs, output_element_target,
};
use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::tensor::{
    Dim, Initializer, ParamId, TensorOp, TensorProgram, TensorProgramBuilder, TensorType, ValueId,
};
use modelspec_core::ir::verify::verify;
use modelspec_runtime::cpu::{Element, HostTensor, evaluate, evaluate_all};

/// Central-difference step, in `f64`.
const H: f64 = 1e-5;
/// Largest error of an adjoint evaluated in `f64`.
const TOL_F64: f64 = 1e-6;
/// Largest error of an adjoint evaluated in `f32`.
const TOL_F32: f64 = 1e-3;
/// Smallest distance of a relu operand from 0.
const KINK_MARGIN: f64 = 1e-2;
/// Smallest largest-difference magnitude for the relative error to mean something.
const MIN_SCALE: f64 = 1e-3;

const LINEAR: &str = include_str!("../../../examples/linear_regression.mspec");
const TWO_LAYER: &str = include_str!("../../modelspec-docs/models/two_layer_relu.mspec");

/// SplitMix64: the fixed test-local generator.
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
        let v = if self.next_u64() & 1 == 0 {
            magnitude
        } else {
            -magnitude
        };
        v as f32
    }
}

/// FNV-1a of a case name: each case its own fixed seed.
fn seed_of(name: &str) -> u64 {
    name.bytes().fold(0xCBF2_9CE4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
    })
}

fn shape_of(ty: &TensorType) -> Vec<u64> {
    ty.shape
        .iter()
        .map(|d| match *d {
            Dim::Static(n) => n,
            Dim::Symbol(s) => panic!("unresolved dimension ?{s}"),
        })
        .collect()
}

type Values<K> = BTreeMap<K, (Vec<u64>, Vec<f32>)>;

/// Values for every input and parameter of a program, in `f32`.
#[derive(Clone)]
struct Point {
    inputs: Values<String>,
    params: Values<ParamId>,
}

impl Point {
    /// Draws every input and parameter of `program`, except the inputs in `fixed`, which take
    /// the values given.
    fn draw(program: &TensorProgram, fixed: &Values<String>, rng: &mut Rng) -> Self {
        let mut draw = |t: &TensorType| {
            let shape = shape_of(t);
            let n = shape.iter().product::<u64>() as usize;
            (shape, (0..n).map(|_| rng.value()).collect::<Vec<_>>())
        };
        let mut inputs = BTreeMap::new();
        for &v in &program.inputs {
            let node = program.node(v).unwrap();
            let TensorOp::Input { name } = &node.op else {
                panic!("{v} is not an input")
            };
            let value = match fixed.get(name) {
                Some(value) => value.clone(),
                None => draw(&node.ty),
            };
            inputs.insert(name.clone(), value);
        }
        let params = program
            .params
            .iter()
            .map(|(&id, decl)| (id, draw(&decl.ty)))
            .collect();
        Self { inputs, params }
    }

    /// This point with element `index` of input `name` moved by `delta`, in `f64`.
    fn moved(&self, name: &str, index: usize, delta: f64) -> (Inputs<f64>, Params<f64>) {
        let mut inputs = tensors(&self.inputs, f64::from);
        let (shape, data) = &self.inputs[name];
        let mut moved: Vec<f64> = data.iter().map(|&v| f64::from(v)).collect();
        moved[index] += delta;
        inputs.insert(
            name.to_owned(),
            HostTensor::new(shape.clone(), moved).unwrap(),
        );
        (inputs, tensors(&self.params, f64::from))
    }
}

type Inputs<T> = BTreeMap<String, HostTensor<T>>;
type Params<T> = BTreeMap<ParamId, HostTensor<T>>;

fn tensors<K: Ord + Clone, T: Element>(
    values: &Values<K>,
    convert: impl Fn(f32) -> T,
) -> BTreeMap<K, HostTensor<T>> {
    values
        .iter()
        .map(|(k, (shape, data))| {
            let data = data.iter().map(|&v| convert(v)).collect();
            (k.clone(), HostTensor::new(shape.clone(), data).unwrap())
        })
        .collect()
}

/// The smallest distance from 0 of any relu operand of `program` at `point`; infinite when it
/// has none.
fn kink_distance(program: &TensorProgram, point: &Point) -> f64 {
    let evaluation = evaluate_all(
        program,
        &tensors(&point.inputs, f64::from),
        &tensors(&point.params, f64::from),
    )
    .unwrap();
    program
        .nodes
        .iter()
        .filter(|n| n.op == TensorOp::Relu)
        .flat_map(|n| evaluation.values[&n.operands[0]].data().to_vec())
        .map(f64::abs)
        .fold(f64::INFINITY, f64::min)
}

/// `max_i |got_i - want_i| / max_i |want_i|`.
fn relative_error(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
    err / scale
}

/// The value of `target` in `program` at `inputs`, `params`, in `f64`.
fn target_at(
    program: &TensorProgram,
    target: ValueId,
    (inputs, params): (Inputs<f64>, Params<f64>),
) -> f64 {
    evaluate_all(program, &inputs, &params).unwrap().values[&target].data()[0]
}

/// The adjoint of each input of `checked` in `adjoint`, evaluated in `f64` and in `f32`, against
/// central differences of `target` in `program`. Returns the point and the `f64` outputs.
fn adjoint_check(
    name: &str,
    program: &TensorProgram,
    target: ValueId,
    adjoint: &InputAdjoint,
    checked: &[&str],
    fixed: &Values<String>,
) -> (Point, Vec<HostTensor<f64>>) {
    verify(&adjoint.program).expect("the adjoint program verifies");
    let mut rng = Rng(seed_of(name));
    let point = (0..1000)
        .map(|_| Point::draw(program, fixed, &mut rng))
        .find(|p| kink_distance(program, p) >= KINK_MARGIN)
        .expect("a point away from every relu kink");

    let in_f64 = evaluate(
        &adjoint.program,
        &tensors(&point.inputs, f64::from),
        &tensors(&point.params, f64::from),
    )
    .unwrap();
    let in_f32 = evaluate(
        &adjoint.program,
        &tensors(&point.inputs, |v| v),
        &tensors(&point.params, |v| v),
    )
    .unwrap();
    let base = target_at(program, target, point.moved(checked[0], 0, 0.0));
    assert_eq!(
        in_f64[0].data(),
        &[base],
        "{name}: the first output is the target"
    );

    for input in checked {
        let k = adjoint
            .adjoints
            .iter()
            .position(|(n, _)| n == input)
            .unwrap_or_else(|| panic!("{name}: no adjoint of `{input}`"));
        let n = point.inputs[*input].1.len();
        let differences: Vec<f64> = (0..n)
            .map(|i| {
                (target_at(program, target, point.moved(input, i, H))
                    - target_at(program, target, point.moved(input, i, -H)))
                    / (2.0 * H)
            })
            .collect();
        let scale = differences.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            scale >= MIN_SCALE,
            "{name}: {input}: differences {differences:?} too small to compare"
        );
        let g64 = in_f64[1 + k].data();
        let g32: Vec<f64> = in_f32[1 + k].data().iter().map(|&v| f64::from(v)).collect();
        assert_eq!(in_f64[1 + k].shape(), &point.inputs[*input].0[..]);
        let e64 = relative_error(g64, &differences);
        let e32 = relative_error(&g32, &differences);
        assert!(
            e64 <= TOL_F64,
            "{name}: {input}: f64 error {e64:e}; adjoint {g64:?}, differences {differences:?}"
        );
        assert!(
            e32 <= TOL_F32,
            "{name}: {input}: f32 error {e32:e}; adjoint {g32:?}, differences {differences:?}"
        );
    }
    (point, in_f64)
}

fn compiled(name: &str, source: &str) -> CompiledTraining {
    compile_training(name, source).expect("the model compiles")
}

/// The loss of `compiled` as the target; the adjoint of `x`.
fn loss_case(name: &str, source: &str) {
    let compiled = compiled(name, source);
    let forward = &compiled.forward.program;
    let loss = compiled.training.training.loss;
    let adjoint = differentiate_inputs(forward, loss, &["x"]).unwrap();
    adjoint_check(name, forward, loss, &adjoint, &["x"], &BTreeMap::new());
}

/// Element `index` of the prediction of `compiled` as the target; the adjoint of `x`.
fn element_case(name: &str, source: &str, index: &[u64]) {
    let compiled = compiled(name, source);
    let forward = &compiled.forward.program;
    let element = output_element_target(forward, compiled.forward.prediction, index).unwrap();
    let prediction_ty = &forward.node(compiled.forward.prediction).unwrap().ty;
    let shape = shape_of(prediction_ty);
    let mut mask = vec![0.0f32; shape.iter().product::<u64>() as usize];
    mask[element.offset as usize] = 1.0;
    let fixed = BTreeMap::from([(TARGET_MASK.to_owned(), (shape, mask))]);
    let adjoint = differentiate_inputs(&element.program, element.target, &["x"]).unwrap();
    let (point, in_f64) = adjoint_check(
        name,
        &element.program,
        element.target,
        &adjoint,
        &["x"],
        &fixed,
    );

    let inputs = tensors(&point.inputs, f64::from);
    let params = tensors(&point.params, f64::from);
    let prediction = evaluate_all(forward, &inputs_without_mask(&inputs), &params)
        .unwrap()
        .values[&compiled.forward.prediction]
        .data()[element.offset as usize];
    assert_eq!(
        in_f64[0].data(),
        &[prediction],
        "{name}: the target is element {index:?} of the prediction"
    );
}

fn inputs_without_mask(inputs: &Inputs<f64>) -> Inputs<f64> {
    let mut inputs = inputs.clone();
    inputs.remove(TARGET_MASK);
    inputs
}

// observatory.adjoint.finite-difference

#[test]
fn observatory_adjoint_finite_difference_linear_regression_loss() {
    loss_case("linear_regression.mspec", LINEAR);
}

#[test]
fn observatory_adjoint_finite_difference_linear_regression_output_element() {
    element_case("linear_regression.mspec", LINEAR, &[0, 0]);
}

#[test]
fn observatory_adjoint_finite_difference_two_layer_relu_loss() {
    loss_case("two_layer_relu.mspec", TWO_LAYER);
}

#[test]
fn observatory_adjoint_finite_difference_two_layer_relu_output_element() {
    element_case("two_layer_relu.mspec", TWO_LAYER, &[0, 0]);
}

/// Open point 3: asked for the `mse` target too, the rule's `-dA` for operand `B` is its adjoint.
#[test]
fn observatory_adjoint_finite_difference_the_mse_target_when_asked() {
    for (name, source) in [
        ("linear_regression.mspec", LINEAR),
        ("two_layer_relu.mspec", TWO_LAYER),
    ] {
        let compiled = compiled(name, source);
        let forward = &compiled.forward.program;
        let loss = compiled.training.training.loss;
        let adjoint = differentiate_inputs(forward, loss, &["y", "x"]).unwrap();
        let case = format!("{name} y and x");
        adjoint_check(
            &case,
            forward,
            loss,
            &adjoint,
            &["y", "x"],
            &BTreeMap::new(),
        );
    }
}

// observatory.adjoint.unread-input

#[test]
fn observatory_adjoint_unread_input() {
    let mut b = TensorProgramBuilder::new();
    let w = b
        .declare_param("w", TensorType::f32(&[4, 2]), Initializer::XavierUniform)
        .unwrap();
    let a = b.input("a", TensorType::f32(&[3, 4])).unwrap();
    b.input("unread", TensorType::f32(&[2, 5])).unwrap();
    let r = b.param_read(w).unwrap();
    let h = b.matmul(a, r).unwrap();
    let hh = b.mul(h, h).unwrap();
    let target = b.reduce_mean(hh).unwrap();
    b.output(target).unwrap();
    let forward = b.finish();

    let adjoint = differentiate_inputs(&forward, target, &["unread", "a"])
        .expect("an unread input is not refused");
    let (point, in_f64) = adjoint_check(
        "unread_input",
        &forward,
        target,
        &adjoint,
        &["a"],
        &BTreeMap::new(),
    );
    assert_eq!(adjoint.adjoints[0].0, "unread");
    assert_eq!(in_f64[1].shape(), &[2, 5]);
    assert_eq!(in_f64[1].data(), &[0.0; 10], "f64: zeros of its shape");
    let in_f32 = evaluate(
        &adjoint.program,
        &tensors(&point.inputs, |v| v),
        &tensors(&point.params, |v| v),
    )
    .unwrap();
    assert_eq!(in_f32[1].shape(), &[2, 5]);
    assert_eq!(in_f32[1].data(), &[0.0f32; 10], "f32: zeros of its shape");
}
