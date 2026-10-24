//! The finite-difference gradient harness of DESIGN.md § 4 and § 9 gate 3, shared by the
//! gradient suites (`tests/gradients.rs`, `tests/gradients_embedding.rs`).
//!
//! A case builds a forward program, differentiates it with
//! [`modelspec_core::autodiff::differentiate`] and runs the resulting training program on the CPU
//! interpreter. Its gradients, evaluated in `f64` and in `f32`, are compared with central finite
//! differences of the loss evaluated in `f64`. `f32` inputs and parameters are drawn from a fixed
//! test-local seed, at least 0.1 away from zero, and every relu operand is kept at least
//! [`KINK_MARGIN`] away from the kink. `i32` inputs (token ids) are not drawn: the case gives
//! them, and they are held fixed while the parameters are perturbed.
//!
//! The error of a gradient is `max_i |g_i - d_i| / max_i |d_i|` over its elements, `d` the
//! differences. A wrong rule (a dropped transpose, a missing factor 2 in `mse`, a missing `1/n` in
//! `mean`, an adjoint counted once instead of per use) is off by a factor of order one, far above
//! either tolerance.
//!
//! Not every test binary that includes this module uses all of it.

use std::collections::BTreeMap;

use modelspec_core::autodiff::differentiate;
use modelspec_core::ir::tensor::{
    DType, Dim, Initializer, ParamId, TensorOp, TensorProgram, TensorProgramBuilder, TensorType,
    ValueId,
};
use modelspec_core::ir::training::{Subgraph, TrainingProgram};
use modelspec_core::ir::verify::verify_training;
use modelspec_runtime::cpu::{Element, HostTensor, evaluate_all_with_ids, evaluate_with_ids};

/// Central-difference step, in `f64`.
pub const H: f64 = 1e-5;
/// Largest error of a gradient evaluated in `f64`.
pub const TOL_F64: f64 = 1e-6;
/// Largest error of a gradient evaluated in `f32`.
pub const TOL_F32: f64 = 1e-3;
/// Smallest distance of a relu operand from 0.
pub const KINK_MARGIN: f64 = 1e-2;
/// Smallest largest-difference magnitude for the relative error to mean something.
pub const MIN_SCALE: f64 = 1e-3;

/// SplitMix64: the fixed test-local generator.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A value in `[-1, -0.1] ∪ [0.1, 1]`.
    pub fn value(&mut self) -> f32 {
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
pub fn seed_of(name: &str) -> u64 {
    name.bytes().fold(0xCBF2_9CE4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
    })
}

pub fn ty(shape: &[u64]) -> TensorType {
    TensorType::f32(shape)
}

pub fn shape_of(ty: &TensorType) -> Vec<u64> {
    ty.shape
        .iter()
        .map(|d| match *d {
            Dim::Static(n) => n,
            Dim::Symbol(s) => panic!("unresolved dimension ?{s}"),
        })
        .collect()
}

pub type Values<K> = BTreeMap<K, (Vec<u64>, Vec<f32>)>;

/// The `i32` inputs of a case, by name: the shape and the ids.
pub type Ids = BTreeMap<String, (Vec<u64>, Vec<i32>)>;

/// Values for every input and parameter of a program: `f32` inputs and parameters, and the
/// `i32` inputs the case gives.
pub struct Point {
    pub inputs: Values<String>,
    pub ids: Ids,
    pub params: Values<ParamId>,
}

impl Point {
    /// Draws every `f32` input and parameter of `program`; its `i32` inputs are `ids`, which must
    /// name each of them.
    pub fn draw(program: &TensorProgram, ids: &Ids, rng: &mut Rng) -> Self {
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
            match node.ty.dtype {
                DType::F32 => {
                    inputs.insert(name.clone(), draw(&node.ty));
                }
                DType::I32 => assert!(ids.contains_key(name), "no ids for input `{name}`"),
            }
        }
        let params = program
            .params
            .iter()
            .map(|(&id, decl)| (id, draw(&decl.ty)))
            .collect();
        Self {
            inputs,
            ids: ids.clone(),
            params,
        }
    }

    /// The `i32` inputs as host tensors.
    pub fn id_tensors(&self) -> BTreeMap<String, HostTensor<i32>> {
        self.ids
            .iter()
            .map(|(k, (shape, data))| {
                (
                    k.clone(),
                    HostTensor::new(shape.clone(), data.clone()).unwrap(),
                )
            })
            .collect()
    }
}

pub fn tensors<K: Ord + Clone, T: Element>(
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

pub fn run<T: Element>(
    training: &TrainingProgram,
    point: &Point,
    convert: impl Fn(f32) -> T + Copy,
) -> Vec<HostTensor<T>> {
    evaluate_with_ids(
        &training.program,
        &tensors(&point.inputs, convert),
        &point.id_tensors(),
        &tensors(&point.params, convert),
    )
    .unwrap()
}

/// The loss in `f64`, the first output of `program`, with element `index` of `param` moved by
/// `delta`; the `i32` inputs stay as they are.
pub fn loss_at(
    program: &TensorProgram,
    point: &Point,
    param: ParamId,
    index: usize,
    delta: f64,
) -> f64 {
    let inputs = tensors(&point.inputs, f64::from);
    let mut params = tensors(&point.params, f64::from);
    let (shape, data) = &point.params[&param];
    let mut moved: Vec<f64> = data.iter().map(|&v| f64::from(v)).collect();
    moved[index] += delta;
    params.insert(param, HostTensor::new(shape.clone(), moved).unwrap());
    evaluate_with_ids(program, &inputs, &point.id_tensors(), &params).unwrap()[0].data()[0]
}

/// The smallest distance from 0 of any relu operand of `forward` at `point`.
pub fn kink_distance(forward: &TensorProgram, training: &TrainingProgram, point: &Point) -> f64 {
    let evaluation = evaluate_all_with_ids(
        &training.program,
        &tensors(&point.inputs, f64::from),
        &point.id_tensors(),
        &tensors(&point.params, f64::from),
    )
    .unwrap();
    forward
        .nodes
        .iter()
        .filter(|n| n.op == TensorOp::Relu)
        .flat_map(|n| evaluation.values[&n.operands[0]].data().to_vec())
        .map(f64::abs)
        .fold(f64::INFINITY, f64::min)
}

/// `max_i |got_i - want_i| / max_i |want_i|`.
pub fn relative_error(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
    err / scale
}

/// The structure every differentiated program has: the forward nodes kept as they were, tagged
/// `Forward` (the loss `Loss`); exactly one more `Loss` node, the seed, when there is a gradient;
/// every other new node `Gradient`; outputs the loss then one gradient per parameter in `ParamId`
/// order, each of its parameter's type.
pub fn assert_structure(forward: &TensorProgram, loss: ValueId, training: &TrainingProgram) {
    assert_eq!(training.loss, loss, "loss value");
    let program = &training.program;
    assert_eq!(program.params, forward.params, "parameter table");
    assert_eq!(program.inputs, forward.inputs, "inputs");
    assert_eq!(
        &program.nodes[..forward.nodes.len()],
        &forward.nodes[..],
        "forward nodes are kept"
    );
    for node in &forward.nodes {
        let want = if node.id == loss {
            Subgraph::Loss
        } else {
            Subgraph::Forward
        };
        assert_eq!(training.subgraphs.get(&node.id), Some(&want), "{}", node.id);
    }
    let new = &program.nodes[forward.nodes.len()..];
    let seeds: Vec<_> = new
        .iter()
        .filter(|n| training.subgraphs.get(&n.id) == Some(&Subgraph::Loss))
        .collect();
    let expected_seeds = usize::from(!training.gradients.is_empty());
    assert_eq!(seeds.len(), expected_seeds, "seed nodes");
    for seed in &seeds {
        assert_eq!(seed.op, TensorOp::Scalar(1.0), "the seed is dL/dL = 1");
    }
    let mut on_seed: Vec<ValueId> = seeds.iter().map(|s| s.id).collect();
    for node in new {
        let tag = training.subgraphs.get(&node.id).copied();
        if tag == Some(Subgraph::Loss) {
            continue;
        }
        let depends = node.operands.iter().any(|o| on_seed.contains(o));
        let want = if depends {
            on_seed.push(node.id);
            Subgraph::Gradient
        } else {
            Subgraph::Forward
        };
        assert_eq!(tag, Some(want), "{}: {:?}", node.id, node.op);
    }
    let mut outputs = vec![loss];
    outputs.extend(training.gradients.values());
    assert_eq!(program.outputs, outputs, "outputs");
    for (param, &g) in &training.gradients {
        assert_eq!(
            program.node(g).unwrap().ty,
            program.params[param].ty,
            "gradient type of {param}"
        );
    }
    verify_training(training).unwrap();
}

/// What a gradient check computed, for assertions beyond the tolerances.
pub struct Checked {
    pub training: TrainingProgram,
    /// The gradient of each parameter in `ParamId` order, evaluated in `f64`.
    pub in_f64: Vec<HostTensor<f64>>,
    /// The same, evaluated in `f32`.
    pub in_f32: Vec<HostTensor<f32>>,
}

/// Builds the forward graph, differentiates it and checks every parameter's gradient against
/// finite differences in `f64` and `f32`. Every parameter must receive a gradient.
pub fn gradient_check(
    name: &str,
    build: impl FnOnce(&mut TensorProgramBuilder) -> ValueId,
) -> TrainingProgram {
    gradient_check_with_ids(name, &Ids::new(), build).training
}

/// [`gradient_check`] of a forward graph whose `i32` inputs are `ids`, held fixed while the
/// parameters are perturbed. Returns the training program and both evaluations of each gradient.
pub fn gradient_check_with_ids(
    name: &str,
    ids: &Ids,
    build: impl FnOnce(&mut TensorProgramBuilder) -> ValueId,
) -> Checked {
    let mut b = TensorProgramBuilder::new();
    let loss = build(&mut b);
    let forward = b.finish();
    let training = differentiate(&forward, loss).unwrap();
    assert_structure(&forward, loss, &training);
    assert_eq!(
        training.gradients.keys().collect::<Vec<_>>(),
        forward.params.keys().collect::<Vec<_>>(),
        "{name}: a gradient per parameter"
    );

    let mut rng = Rng::new(seed_of(name));
    let point = (0..1000)
        .map(|_| Point::draw(&forward, ids, &mut rng))
        .find(|p| kink_distance(&forward, &training, p) >= KINK_MARGIN)
        .expect("a point away from every relu kink");

    let mut in_f64 = run(&training, &point, f64::from);
    let mut in_f32 = run(&training, &point, |v| v);
    for (k, (&param, _)) in training.gradients.iter().enumerate() {
        let path = &forward.params[&param].path;
        let n = point.params[&param].1.len();
        let differences: Vec<f64> = (0..n)
            .map(|i| {
                (loss_at(&training.program, &point, param, i, H)
                    - loss_at(&training.program, &point, param, i, -H))
                    / (2.0 * H)
            })
            .collect();
        let scale = differences.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            scale >= MIN_SCALE,
            "{name}: {path}: differences {differences:?} too small to compare"
        );
        let g64 = in_f64[1 + k].data();
        let g32: Vec<f64> = in_f32[1 + k].data().iter().map(|&v| f64::from(v)).collect();
        let e64 = relative_error(g64, &differences);
        let e32 = relative_error(&g32, &differences);
        assert!(
            e64 <= TOL_F64,
            "{name}: {path}: f64 error {e64:e}; gradient {g64:?}, differences {differences:?}"
        );
        assert!(
            e32 <= TOL_F32,
            "{name}: {path}: f32 error {e32:e}; gradient {g32:?}, differences {differences:?}"
        );
    }
    in_f64.remove(0);
    in_f32.remove(0);
    Checked {
        training,
        in_f64,
        in_f32,
    }
}

pub fn param(b: &mut TensorProgramBuilder, path: &str, shape: &[u64]) -> ValueId {
    let id = b
        .declare_param(path, ty(shape), Initializer::XavierUniform)
        .unwrap();
    b.param_read(id).unwrap()
}

pub fn input(b: &mut TensorProgramBuilder, name: &str, shape: &[u64]) -> ValueId {
    b.input(name, ty(shape)).unwrap()
}

/// An `i32` input of shape `shape`.
pub fn id_input(b: &mut TensorProgramBuilder, name: &str, shape: &[u64]) -> ValueId {
    b.input(name, TensorType::i32(shape)).unwrap()
}

/// `mse(y, t)` against a fresh target `t` of `y`'s shape.
pub fn mse_to_target(b: &mut TensorProgramBuilder, y: ValueId) -> ValueId {
    let shape = shape_of(b.type_of(y).unwrap());
    let t = input(b, "t", &shape);
    b.mse(y, t).unwrap()
}
