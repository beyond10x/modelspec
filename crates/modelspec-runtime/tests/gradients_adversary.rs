//! Adversary finite-difference cases: shared uses beyond two (DESIGN.md § 4 "accumulate every
//! incoming adjoint exactly once", § 9 gate 3), checked in `f64` (relative error at most 1e-6)
//! and `f32` (at most 1e-3) against central differences in `f64`.

use std::collections::BTreeMap;

use modelspec_core::autodiff::differentiate;
use modelspec_core::ir::tensor::{
    Dim, Initializer, ParamId, TensorOp, TensorProgramBuilder, TensorType, ValueId,
};
use modelspec_core::ir::training::TrainingProgram;
use modelspec_runtime::cpu::{Element, HostTensor, evaluate};

const H: f64 = 1e-5;
const TOL_F64: f64 = 1e-6;
const TOL_F32: f64 = 1e-3;

/// SplitMix64, values in `[-1, -0.1] ∪ [0.1, 1]`.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn value(&mut self) -> f32 {
        let unit = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let magnitude = 0.1 + 0.9 * unit;
        (if self.next_u64() & 1 == 0 {
            magnitude
        } else {
            -magnitude
        }) as f32
    }
}

fn ty(shape: &[u64]) -> TensorType {
    TensorType::f32(shape)
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

type Point = (
    BTreeMap<String, (Vec<u64>, Vec<f32>)>,
    BTreeMap<ParamId, (Vec<u64>, Vec<f32>)>,
);

fn tensors<K: Ord + Clone, T: Element>(
    values: &BTreeMap<K, (Vec<u64>, Vec<f32>)>,
    convert: impl Fn(f32) -> T,
) -> BTreeMap<K, HostTensor<T>> {
    values
        .iter()
        .map(|(k, (s, d))| {
            let data = d.iter().map(|&v| convert(v)).collect();
            (k.clone(), HostTensor::new(s.clone(), data).unwrap())
        })
        .collect()
}

fn loss_moved(t: &TrainingProgram, point: &Point, p: ParamId, i: usize, delta: f64) -> f64 {
    let inputs = tensors(&point.0, f64::from);
    let mut params = tensors(&point.1, f64::from);
    let (shape, data) = &point.1[&p];
    let mut moved: Vec<f64> = data.iter().map(|&v| f64::from(v)).collect();
    moved[i] += delta;
    params.insert(p, HostTensor::new(shape.clone(), moved).unwrap());
    evaluate(&t.program, &inputs, &params).unwrap()[0].data()[0]
}

fn check(name: &str, seed: u64, build: impl FnOnce(&mut TensorProgramBuilder) -> ValueId) {
    let mut b = TensorProgramBuilder::new();
    let loss = build(&mut b);
    let forward = b.finish();
    let training = differentiate(&forward, loss).unwrap();
    assert_eq!(
        training.gradients.keys().collect::<Vec<_>>(),
        forward.params.keys().collect::<Vec<_>>(),
        "{name}: a gradient per parameter"
    );
    let mut rng = Rng(seed);
    let mut draw = |t: &TensorType| {
        let s = shape_of(t);
        let n = s.iter().product::<u64>() as usize;
        (s, (0..n).map(|_| rng.value()).collect::<Vec<f32>>())
    };
    let mut inputs = BTreeMap::new();
    for &v in &forward.inputs {
        let node = forward.node(v).unwrap();
        let TensorOp::Input { name } = &node.op else {
            unreachable!()
        };
        inputs.insert(name.clone(), draw(&node.ty));
    }
    let params = forward
        .params
        .iter()
        .map(|(&id, d)| (id, draw(&d.ty)))
        .collect();
    let point: Point = (inputs, params);
    let g64 = evaluate(
        &training.program,
        &tensors(&point.0, f64::from),
        &tensors(&point.1, f64::from),
    )
    .unwrap();
    let g32 = evaluate(
        &training.program,
        &tensors(&point.0, |v| v),
        &tensors(&point.1, |v| v),
    )
    .unwrap();
    for (k, &param) in training.gradients.keys().enumerate() {
        let n = point.1[&param].1.len();
        let d: Vec<f64> = (0..n)
            .map(|i| {
                (loss_moved(&training, &point, param, i, H)
                    - loss_moved(&training, &point, param, i, -H))
                    / (2.0 * H)
            })
            .collect();
        let scale = d.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(scale >= 1e-3, "{name}: differences too small: {d:?}");
        let e = |got: Vec<f64>| {
            got.iter()
                .zip(&d)
                .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()))
                / scale
        };
        let got64 = g64[1 + k].data().to_vec();
        let got32: Vec<f64> = g32[1 + k].data().iter().map(|&v| f64::from(v)).collect();
        let (e64, e32) = (e(got64.clone()), e(got32));
        assert!(
            e64 <= TOL_F64,
            "{name}: {param}: f64 error {e64:e}; gradient {got64:?}, differences {d:?}"
        );
        assert!(e32 <= TOL_F32, "{name}: {param}: f32 error {e32:e}");
    }
}

fn param(b: &mut TensorProgramBuilder, path: &str, shape: &[u64]) -> ValueId {
    let id = b
        .declare_param(path, ty(shape), Initializer::XavierUniform)
        .unwrap();
    b.param_read(id).unwrap()
}

fn mse_to_target(b: &mut TensorProgramBuilder, y: ValueId) -> ValueId {
    let shape = shape_of(b.type_of(y).unwrap());
    let t = b.input("t", ty(&shape)).unwrap();
    b.mse(y, t).unwrap()
}

/// `h` used three times: `(h + h) + h`; its adjoint is the sum of three contributions.
#[test]
fn shared_value_used_three_times() {
    check("shared_value_used_three_times", 0x5EED_0003, |b| {
        let w = param(b, "w", &[4, 3]);
        let x = b.input("x", ty(&[5, 4])).unwrap();
        let h = b.matmul(x, w).unwrap();
        let hh = b.add(h, h).unwrap();
        let y = b.add(hh, h).unwrap();
        mse_to_target(b, y)
    });
}

/// One parameter read by three `ParamRead` nodes: `(r1 * r2) * r3`.
#[test]
fn parameter_read_three_times() {
    check("parameter_read_three_times", 0x5EED_0013, |b| {
        let id = b
            .declare_param("p", ty(&[3, 4]), Initializer::XavierUniform)
            .unwrap();
        let r1 = b.param_read(id).unwrap();
        let r2 = b.param_read(id).unwrap();
        let r3 = b.param_read(id).unwrap();
        let r12 = b.mul(r1, r2).unwrap();
        let y = b.mul(r12, r3).unwrap();
        mse_to_target(b, y)
    });
}
