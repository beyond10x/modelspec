//! Adversary finite-difference cases for `story:observatory-input-adjoint`: shared uses of an
//! input, a rank-0 input broadcast through `Mul`, `Add` and `Sub` on either side, an input as
//! both operands of `MatMul`, an `mse` whose two operands both depend on the asked input, a
//! repeated name evaluated on the CPU, and output elements other than `[0, 0]`.
//!
//! Error measure and tolerances as `tests/gradients.rs`: `max_i |g_i - d_i| / max_i |d_i|`,
//! `TOL_F64` = 1e-6 evaluated in `f64`, `TOL_F32` = 1e-3 evaluated in `f32`; differences are
//! central, in `f64`, on the program the target was taken from.

use std::collections::BTreeMap;

use modelspec_core::autodiff::{TARGET_MASK, differentiate_inputs, output_element_target};
use modelspec_core::compile::compile_training;
use modelspec_core::ir::tensor::{
    Dim, Initializer, ParamId, TensorOp, TensorProgram, TensorProgramBuilder, TensorType, ValueId,
};
use modelspec_runtime::cpu::{HostTensor, evaluate, evaluate_all};

const H: f64 = 1e-5;
const TOL_F64: f64 = 1e-6;
const TOL_F32: f64 = 1e-3;
const KINK_MARGIN: f64 = 1e-2;

const LINEAR: &str = include_str!("../../../examples/linear_regression.mspec");
const TWO_LAYER: &str = include_str!("../../modelspec-docs/models/two_layer_relu.mspec");

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let unit = (z >> 11) as f64 / (1u64 << 53) as f64;
        let magnitude = 0.2 + 0.8 * unit;
        (if z & 1 == 0 { magnitude } else { -magnitude }) as f32
    }
}

fn shape_of(ty: &TensorType) -> Vec<u64> {
    ty.shape
        .iter()
        .map(|d| match *d {
            Dim::Static(n) => n,
            Dim::Symbol(s) => panic!("?{s}"),
        })
        .collect()
}

#[derive(Clone)]
struct Point {
    inputs: BTreeMap<String, (Vec<u64>, Vec<f32>)>,
    params: BTreeMap<ParamId, (Vec<u64>, Vec<f32>)>,
}

impl Point {
    fn draw(
        program: &TensorProgram,
        fixed: &BTreeMap<String, (Vec<u64>, Vec<f32>)>,
        rng: &mut Rng,
    ) -> Self {
        let mut draw = |t: &TensorType| {
            let shape = shape_of(t);
            let n = shape.iter().product::<u64>() as usize;
            (shape, (0..n).map(|_| rng.next()).collect::<Vec<_>>())
        };
        let mut inputs = BTreeMap::new();
        for &v in &program.inputs {
            let node = program.node(v).unwrap();
            let TensorOp::Input { name } = &node.op else {
                panic!()
            };
            let value = fixed.get(name).cloned().unwrap_or_else(|| draw(&node.ty));
            inputs.insert(name.clone(), value);
        }
        let params = program
            .params
            .iter()
            .map(|(&id, d)| (id, draw(&d.ty)))
            .collect();
        Self { inputs, params }
    }

    fn f64s(
        &self,
    ) -> (
        BTreeMap<String, HostTensor<f64>>,
        BTreeMap<ParamId, HostTensor<f64>>,
    ) {
        (
            self.inputs
                .iter()
                .map(|(k, (s, d))| {
                    let d = d.iter().map(|&v| f64::from(v)).collect();
                    (k.clone(), HostTensor::new(s.clone(), d).unwrap())
                })
                .collect(),
            self.params
                .iter()
                .map(|(k, (s, d))| {
                    let d = d.iter().map(|&v| f64::from(v)).collect();
                    (*k, HostTensor::new(s.clone(), d).unwrap())
                })
                .collect(),
        )
    }

    fn f32s(
        &self,
    ) -> (
        BTreeMap<String, HostTensor<f32>>,
        BTreeMap<ParamId, HostTensor<f32>>,
    ) {
        (
            self.inputs
                .iter()
                .map(|(k, (s, d))| (k.clone(), HostTensor::new(s.clone(), d.clone()).unwrap()))
                .collect(),
            self.params
                .iter()
                .map(|(k, (s, d))| (*k, HostTensor::new(s.clone(), d.clone()).unwrap()))
                .collect(),
        )
    }
}

fn kink_distance(program: &TensorProgram, point: &Point) -> f64 {
    let (i, p) = point.f64s();
    let ev = evaluate_all(program, &i, &p).unwrap();
    program
        .nodes
        .iter()
        .filter(|n| n.op == TensorOp::Relu)
        .flat_map(|n| ev.values[&n.operands[0]].data().to_vec())
        .map(f64::abs)
        .fold(f64::INFINITY, f64::min)
}

fn target_at(
    program: &TensorProgram,
    target: ValueId,
    point: &Point,
    name: &str,
    k: usize,
    delta: f64,
) -> f64 {
    let (mut i, p) = point.f64s();
    let t = i.remove(name).unwrap();
    let mut data = t.data().to_vec();
    data[k] += delta;
    i.insert(
        name.to_owned(),
        HostTensor::new(t.shape().to_vec(), data).unwrap(),
    );
    evaluate_all(program, &i, &p).unwrap().values[&target].data()[0]
}

fn relative_error(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
    err / scale
}

/// Asks for `names` (repeats allowed), then checks every distinct name against central
/// differences, in `f64` and in `f32`; each output position of a repeated name too.
fn check(
    case: &str,
    program: &TensorProgram,
    target: ValueId,
    names: &[&str],
    fixed: &BTreeMap<String, (Vec<u64>, Vec<f32>)>,
    seed: u64,
) {
    let adjoint =
        differentiate_inputs(program, target, names).unwrap_or_else(|e| panic!("{case}: {e}"));
    let mut rng = Rng(seed);
    let point = (0..1000)
        .map(|_| Point::draw(program, fixed, &mut rng))
        .find(|p| kink_distance(program, p) >= KINK_MARGIN)
        .expect("a point off every kink");
    let (i64s, p64s) = point.f64s();
    let (i32s, p32s) = point.f32s();
    let out64 = evaluate(&adjoint.program, &i64s, &p64s).unwrap();
    let out32 = evaluate(&adjoint.program, &i32s, &p32s).unwrap();
    assert_eq!(
        out64.len(),
        1 + names.len(),
        "{case}: one output per name asked"
    );
    for (pos, name) in names.iter().enumerate() {
        let n = point.inputs[*name].1.len();
        let diffs: Vec<f64> = (0..n)
            .map(|k| {
                (target_at(program, target, &point, name, k, H)
                    - target_at(program, target, &point, name, k, -H))
                    / (2.0 * H)
            })
            .collect();
        let scale = diffs.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            scale >= 1e-3,
            "{case}: {name}: differences too small {diffs:?}"
        );
        let g64 = out64[1 + pos].data();
        let g32: Vec<f64> = out32[1 + pos]
            .data()
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        assert_eq!(
            out64[1 + pos].shape(),
            &point.inputs[*name].0[..],
            "{case}: {name}: shape"
        );
        let e64 = relative_error(g64, &diffs);
        let e32 = relative_error(&g32, &diffs);
        assert!(
            e64 <= TOL_F64,
            "{case}: {name}@{pos}: f64 error {e64:e}: {g64:?} vs {diffs:?}"
        );
        assert!(
            e32 <= TOL_F32,
            "{case}: {name}@{pos}: f32 error {e32:e}: {g32:?} vs {diffs:?}"
        );
    }
}

/// An input `x` read twice by one `Mul`, once by `MatMul` on each side, once by `Mse`'s B and
/// through `Relu` and `Transpose`; a rank-0 input `s` broadcast on either side of `Mul`, `Add`
/// and `Sub`.
#[test]
fn adversary_shared_uses_and_scalar_broadcasts_match_finite_differences() {
    let mut b = TensorProgramBuilder::new();
    let w = b
        .declare_param("w", TensorType::f32(&[3, 3]), Initializer::XavierUniform)
        .unwrap();
    let x = b.input("x", TensorType::f32(&[3, 3])).unwrap();
    let s = b.input("s", TensorType::f32_scalar()).unwrap();
    let rw = b.param_read(w).unwrap();
    let xx = b.mul(x, x).unwrap();
    let xm = b.matmul(x, x).unwrap();
    let xw = b.matmul(rw, x).unwrap();
    let a = b.add(xx, xm).unwrap();
    let a = b.add(a, xw).unwrap();
    let sa = b.mul(s, a).unwrap();
    let as_ = b.mul(sa, s).unwrap();
    let t = b.transpose(as_).unwrap();
    let r = b.relu(t).unwrap();
    let d = b.sub(s, r).unwrap();
    let e = b.sub(d, s).unwrap();
    let f = b.add(s, e).unwrap();
    let g = b.add(f, s).unwrap();
    let m = b.mse(g, x).unwrap();
    let mean = b.reduce_mean(x).unwrap();
    let ms = b.mul(mean, s).unwrap();
    let target = b.add(m, ms).unwrap();
    b.output(target).unwrap();
    let forward = b.finish();
    let none = BTreeMap::new();
    check("shared x, s", &forward, target, &["x", "s"], &none, 11);
    check(
        "shared s, x, x",
        &forward,
        target,
        &["s", "x", "x"],
        &none,
        12,
    );
}

/// The repeated name on the CPU: every output position matches, in `f64` and `f32`.
#[test]
fn adversary_a_repeated_name_evaluates_at_every_position() {
    let compiled = compile_training("two_layer_relu.mspec", TWO_LAYER).unwrap();
    let forward = &compiled.forward.program;
    let loss = compiled.training.training.loss;
    check(
        "two-layer x, y, x",
        forward,
        loss,
        &["x", "y", "x"],
        &BTreeMap::new(),
        13,
    );
}

/// Output elements other than `[0, 0]`: the last row, a middle row, both models.
#[test]
fn adversary_output_elements_past_the_first_match_finite_differences() {
    for (name, source, rows) in [
        ("linear_regression.mspec", LINEAR, [3u64, 7]),
        ("two_layer_relu.mspec", TWO_LAYER, [9, 15]),
    ] {
        let compiled = compile_training(name, source).unwrap();
        let forward = &compiled.forward.program;
        let prediction = compiled.forward.prediction;
        let shape = shape_of(&forward.node(prediction).unwrap().ty);
        for row in rows {
            let element = output_element_target(forward, prediction, &[row, 0]).unwrap();
            assert_eq!(element.offset, row, "{name}: [{row}, 0]");
            let mut mask = vec![0.0f32; shape.iter().product::<u64>() as usize];
            mask[element.offset as usize] = 1.0;
            let fixed = BTreeMap::from([(TARGET_MASK.to_owned(), (shape.clone(), mask))]);
            let case = format!("{name} [{row}, 0]");
            check(
                &case,
                &element.program,
                element.target,
                &["x"],
                &fixed,
                20 + row,
            );

            let adjoint = differentiate_inputs(&element.program, element.target, &["x"]).unwrap();
            let mut rng = Rng(99 + row);
            let point = Point::draw(&element.program, &fixed, &mut rng);
            let (i, p) = point.f64s();
            let g = evaluate(&adjoint.program, &i, &p).unwrap();
            let x_cols = point.inputs["x"].0[1] as usize;
            let nonzero_rows: Vec<usize> = g[1]
                .data()
                .iter()
                .enumerate()
                .filter(|(_, v)| **v != 0.0)
                .map(|(k, _)| k / x_cols)
                .collect();
            assert!(
                nonzero_rows.iter().all(|&r| r as u64 == row),
                "{case}: only row {row} of x reaches element [{row}, 0]: {nonzero_rows:?}"
            );
        }
    }
}
