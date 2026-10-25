//! Dropout on the CPU interpreter, drawn from Philox through an explicit per-step RNG token
//! (DESIGN.md § 2.3, § 4, § 9 gate 3; story:bert-dropout).
//!
//! `dropout(x, p = <float literal>)` reads, beside `x`, its site's RNG token: an internal `i32`
//! input of shape `[2]` holding the draw id `[id low, id high]` of the site at the step. Element
//! `i` (row-major) takes `philox4x32_10([i, 4, id low, id high], key)` under the run's key, word
//! 0 mapped to `u_i = (w + 0.5) / 2^32`, and is kept when `u_i >= p`; a kept element is
//! `x * (1 / (1 - p))`, a dropped one `0.0` (coordinator decision D47, 2026-10-25,
//! `.engineering/waves/bert-1.md`; stream tag 4 by D30). The draw id of a site at step `s` is the
//! first 8 bytes of `blake3(site name, then s as 8 little-endian bytes)`, little-endian. The
//! gradient `DropoutGrad(token, g)` derives the same mask from the same token.
//!
//! The ids and draws are derived here from `blake3` and `philox4x32_10` directly, not through
//! the runtime's helpers, so a wrong tag, word or byte order in the runtime shows.

use std::collections::BTreeMap;

use modelspec_core::autodiff::differentiate;
use modelspec_core::compile::compile_tensor;
use modelspec_core::ir::tensor::{
    ParamId, TensorOp, TensorProgram, TensorProgramBuilder, TensorType, ValueId,
};
use modelspec_core::ir::training::{TrainingProgram, dropout_sites};
use modelspec_runtime::cpu::{HostTensor, RuntimeError, evaluate_seeded, evaluate_with_ids};
use modelspec_runtime::train::init::{Stream, dropout_id, dropout_token, philox4x32_10};

#[allow(dead_code)]
#[path = "common/finite_difference.rs"]
mod finite_difference;

use finite_difference::{
    H as STEP, Ids, MIN_SCALE, Point, Rng, TOL_F32, TOL_F64, assert_structure, input, param,
    relative_error, seed_of,
};

/// The run seed of every case.
const SEED: u64 = 0x5eed_d0d0_1234_abcd;
/// The site of the hand-built programs.
const SITE: &str = "Reg.forward.dropout#0";

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

/// The token of `site` at `step`: `[id low, id high]` as `i32` bit patterns.
fn token_of(site: &str, step: u64) -> (Vec<u64>, Vec<i32>) {
    let id = id_of(site, step);
    (vec![2], vec![id as u32 as i32, (id >> 32) as u32 as i32])
}

/// Whether element `index` of the draw `id` is kept at rate `p`, derived from Philox here.
fn kept(id: u64, index: u32, p: f32) -> bool {
    let block = philox4x32_10(
        [index, 4, id as u32, (id >> 32) as u32],
        [SEED as u32, (SEED >> 32) as u32],
    );
    let u = (f64::from(block[0]) + 0.5) / 4_294_967_296.0;
    u >= f64::from(p)
}

fn id_tensors(ids: &Ids) -> BTreeMap<String, HostTensor<i32>> {
    ids.iter()
        .map(|(name, (shape, data))| {
            (
                name.clone(),
                HostTensor::new(shape.clone(), data.clone()).unwrap(),
            )
        })
        .collect()
}

fn tensors<T: modelspec_runtime::cpu::Element>(
    values: &BTreeMap<String, (Vec<u64>, Vec<f32>)>,
    convert: impl Fn(f32) -> T,
) -> BTreeMap<String, HostTensor<T>> {
    values
        .iter()
        .map(|(k, (shape, data))| {
            let data = data.iter().map(|&v| convert(v)).collect();
            (k.clone(), HostTensor::new(shape.clone(), data).unwrap())
        })
        .collect()
}

fn params<T: modelspec_runtime::cpu::Element>(
    values: &BTreeMap<ParamId, (Vec<u64>, Vec<f32>)>,
    convert: impl Fn(f32) -> T,
) -> BTreeMap<ParamId, HostTensor<T>> {
    values
        .iter()
        .map(|(k, (shape, data))| {
            let data = data.iter().map(|&v| convert(v)).collect();
            (*k, HostTensor::new(shape.clone(), data).unwrap())
        })
        .collect()
}

/// `mse(dropout(matmul(x, w), p), t)` with `x: [6, 4]`, `w: [4, 3]` and the token input `SITE`.
fn dropout_mse(b: &mut TensorProgramBuilder, p: f32) -> ValueId {
    let x = input(b, "x", &[6, 4]);
    let w = param(b, "Reg.w", &[4, 3]);
    let token = b.input(SITE, TensorType::i32(&[2])).unwrap();
    let h = b.matmul(x, w).unwrap();
    let dropped = b.dropout(h, token, p).unwrap();
    let t = input(b, "t", &[6, 3]);
    b.mse(dropped, t).unwrap()
}

/// The loss in `f64` of `program` at `point`, element `index` of `param` moved by `delta`.
fn loss_at(
    program: &TensorProgram,
    point: &Point,
    param: ParamId,
    index: usize,
    delta: f64,
) -> f64 {
    let mut values = params(&point.params, f64::from);
    let (shape, data) = &point.params[&param];
    let mut moved: Vec<f64> = data.iter().map(|&v| f64::from(v)).collect();
    moved[index] += delta;
    values.insert(param, HostTensor::new(shape.clone(), moved).unwrap());
    evaluate_seeded(
        program,
        &tensors(&point.inputs, f64::from),
        &id_tensors(&point.ids),
        &values,
        SEED,
    )
    .unwrap()[0]
        .data()[0]
}

/// Scenario 2. At a fixed token and `p = 0.3`, the gradient of `mse(dropout(matmul(x, w)), t)`
/// with respect to `w` is within `TOL_F64` in `f64` and `TOL_F32` in `f32` of central
/// differences of the loss in `f64`; the mask drops some and keeps some elements, and the
/// training program holds `DropoutGrad` reading the forward's token.
#[test]
fn gradients_dropout_match_central_differences_at_a_fixed_token() {
    let mut b = TensorProgramBuilder::new();
    let loss = dropout_mse(&mut b, 0.3);
    b.output(loss).unwrap();
    let forward = b.finish();
    let training: TrainingProgram = differentiate(&forward, loss).unwrap();
    assert_structure(&forward, loss, &training);
    let token = forward.inputs[1];
    let grads: Vec<_> = training
        .program
        .nodes
        .iter()
        .filter(|n| matches!(n.op, TensorOp::DropoutGrad { .. }))
        .collect();
    assert_eq!(grads.len(), 1, "one DropoutGrad");
    assert_eq!(grads[0].op, TensorOp::DropoutGrad { p: 0.3 });
    assert_eq!(
        grads[0].operands[0], token,
        "DropoutGrad reads the forward's token"
    );

    let given: Ids = BTreeMap::from([(SITE.to_owned(), token_of(SITE, 3))]);
    let mut rng = Rng::new(seed_of("gradients_dropout"));
    let point = Point::draw(&forward, &given, &mut rng);
    let id = id_of(SITE, 3);
    let kept_count = (0..18).filter(|&i| kept(id, i, 0.3)).count();
    assert!(
        (1..18).contains(&kept_count),
        "the mask keeps {kept_count} of 18: the case must drop some and keep some"
    );

    let w = ParamId(0);
    let run = |convert_f64: bool| -> Vec<f64> {
        let ids = id_tensors(&point.ids);
        if convert_f64 {
            let out = evaluate_seeded(
                &training.program,
                &tensors(&point.inputs, f64::from),
                &ids,
                &params(&point.params, f64::from),
                SEED,
            )
            .unwrap();
            out[1].data().to_vec()
        } else {
            let out = evaluate_seeded(
                &training.program,
                &tensors(&point.inputs, |v| v),
                &ids,
                &params(&point.params, |v| v),
                SEED,
            )
            .unwrap();
            out[1].data().iter().map(|&v| f64::from(v)).collect()
        }
    };
    let g64 = run(true);
    let g32 = run(false);
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

/// The gradient through a dropped element is exactly `+0.0`, and through a kept one `g * scale`,
/// `scale = 1 / (1 - p)` in `f32`: `DropoutGrad` of `sum(dropout(x))` is the mask times the
/// scale.
#[test]
fn dropout_grad_is_the_scaled_mask_of_the_forward() {
    let p = 0.3f32;
    let mut b = TensorProgramBuilder::new();
    let x = param(&mut b, "Reg.x", &[4, 5]);
    let token = b.input(SITE, TensorType::i32(&[2])).unwrap();
    let dropped = b.dropout(x, token, p).unwrap();
    let loss = b.reduce_sum(dropped).unwrap();
    let forward = b.finish();
    let training = differentiate(&forward, loss).unwrap();
    let values: BTreeMap<ParamId, HostTensor<f32>> = BTreeMap::from([(
        ParamId(0),
        HostTensor::new(vec![4, 5], (0..20).map(|i| i as f32 - 7.5).collect()).unwrap(),
    )]);
    let (shape, data) = token_of(SITE, 11);
    let ids = BTreeMap::from([(SITE.to_owned(), HostTensor::new(shape, data).unwrap())]);
    let out = evaluate_seeded(&training.program, &BTreeMap::new(), &ids, &values, SEED).unwrap();
    let scale = 1.0f32 / (1.0f32 - p);
    let id = id_of(SITE, 11);
    for (i, &g) in out[1].data().iter().enumerate() {
        let want = if kept(id, i as u32, p) { scale } else { 0.0 };
        assert_eq!(g.to_bits(), want.to_bits(), "element {i}");
    }
}

/// Scenario 3. For a fixed seed, site and step, the kept elements of `dropout` are exactly those
/// derived from `philox4x32_10([i, 4, id low, id high], key)`, each kept value `x * scale` with
/// `scale = 1.0f32 / (1.0f32 - p)` and each dropped value `+0.0`; over `10_000` elements at
/// `p = 0.1` the kept share lies within `0.9 +- 0.01`. The runtime's id and token helpers agree
/// with the derivation here, and the stream tag is 4.
#[test]
fn dropout_mask_derivation_follows_philox_stream_tag_4() {
    assert_eq!(Stream::Dropout.tag(), 4);
    assert_eq!(dropout_id(SITE, 7), id_of(SITE, 7));
    let (shape, data) = token_of(SITE, 7);
    assert_eq!(
        dropout_token(SITE, 7),
        HostTensor::new(shape, data).unwrap()
    );

    let p = 0.1f32;
    let n = 10_000u64;
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", TensorType::f32(&[100, 100])).unwrap();
    let token = b.input(SITE, TensorType::i32(&[2])).unwrap();
    let dropped = b.dropout(x, token, p).unwrap();
    b.output(dropped).unwrap();
    let program = b.finish();
    let xs: Vec<f32> = (0..n).map(|i| 0.25 + (i % 97) as f32 * 0.125).collect();
    let inputs = BTreeMap::from([(
        "x".to_owned(),
        HostTensor::new(vec![100, 100], xs.clone()).unwrap(),
    )]);
    let ids = BTreeMap::from([(SITE.to_owned(), dropout_token(SITE, 7))]);
    let out = evaluate_seeded(&program, &inputs, &ids, &BTreeMap::new(), SEED).unwrap();
    let scale = 1.0f32 / (1.0f32 - p);
    let id = id_of(SITE, 7);
    let mut kept_count = 0u64;
    for (i, (&y, &x)) in out[0].data().iter().zip(&xs).enumerate() {
        if kept(id, i as u32, p) {
            kept_count += 1;
            assert_eq!(y.to_bits(), (x * scale).to_bits(), "kept element {i}");
        } else {
            assert_eq!(y.to_bits(), 0.0f32.to_bits(), "dropped element {i}");
        }
    }
    let share = kept_count as f64 / n as f64;
    assert!((share - 0.9).abs() <= 0.01, "kept share {share}");

    let in_f64 = evaluate_seeded(
        &program,
        &BTreeMap::from([(
            "x".to_owned(),
            HostTensor::new(vec![100, 100], xs.iter().map(|&v| f64::from(v)).collect()).unwrap(),
        )]),
        &ids,
        &BTreeMap::new(),
        SEED,
    )
    .unwrap();
    for (i, &y) in in_f64[0].data().iter().enumerate() {
        assert_eq!(y != 0.0, kept(id, i as u32, p), "f64 mask at {i}");
    }
}

/// Another seed draws another mask from the same token: the key is the run's seed.
#[test]
fn dropout_mask_derivation_is_keyed_by_the_seed() {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", TensorType::f32(&[64])).unwrap();
    let token = b.input(SITE, TensorType::i32(&[2])).unwrap();
    let dropped = b.dropout(x, token, 0.5).unwrap();
    b.output(dropped).unwrap();
    let program = b.finish();
    let inputs = BTreeMap::from([(
        "x".to_owned(),
        HostTensor::new(vec![64], vec![1.0f32; 64]).unwrap(),
    )]);
    let ids = BTreeMap::from([(SITE.to_owned(), dropout_token(SITE, 0))]);
    let a = evaluate_seeded(&program, &inputs, &ids, &BTreeMap::new(), 1).unwrap();
    let b = evaluate_seeded(&program, &inputs, &ids, &BTreeMap::new(), 2).unwrap();
    assert_ne!(a[0], b[0]);
}

/// A program with a dropout node run without a seed is refused before any node runs: no key,
/// no draw.
#[test]
fn dropout_without_a_seed_is_refused() {
    let mut b = TensorProgramBuilder::new();
    let x = b.input("x", TensorType::f32(&[4])).unwrap();
    let token = b.input(SITE, TensorType::i32(&[2])).unwrap();
    let dropped = b.dropout(x, token, 0.5).unwrap();
    b.output(dropped).unwrap();
    let program = b.finish();
    let inputs = BTreeMap::from([(
        "x".to_owned(),
        HostTensor::new(vec![4], vec![1.0f32; 4]).unwrap(),
    )]);
    let ids = BTreeMap::from([(SITE.to_owned(), dropout_token(SITE, 0))]);
    let refused = evaluate_with_ids(&program, &inputs, &ids, &BTreeMap::new());
    assert!(
        matches!(refused, Err(RuntimeError::NoSeed { .. })),
        "{refused:?}"
    );
}

const TWO_SITES: &str = r#"model Reg {
    param w: Tensor<f32, [4, 3]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {
        let h = matmul(x, w);
        let a = dropout(h, p = 0.5);
        let b = dropout(h, p = 0.5);
        return add(a, b);
    }
}

train Fit {
    model = Reg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 3]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
"#;

/// The value of the `Dropout` node `node` of `program` with every element of `x` 1.0, at `step`.
fn mask_at(program: &TensorProgram, node: ValueId, step: u64) -> Vec<bool> {
    let mut outputs = program.clone();
    outputs.outputs = vec![node];
    let tokens: BTreeMap<String, HostTensor<i32>> = dropout_sites(program)
        .into_iter()
        .map(|site| {
            let token = dropout_token(&site, step);
            (site, token)
        })
        .collect();
    let inputs = BTreeMap::from([
        (
            "x".to_owned(),
            HostTensor::new(vec![8, 4], vec![1.0f32; 32]).unwrap(),
        ),
        (
            "y".to_owned(),
            HostTensor::new(vec![8, 3], vec![0.0f32; 24]).unwrap(),
        ),
    ]);
    let params = BTreeMap::from([(
        ParamId(0),
        HostTensor::new(vec![4, 3], vec![0.5f32; 12]).unwrap(),
    )]);
    let out = evaluate_seeded(&outputs, &inputs, &tokens, &params, SEED).unwrap();
    out[0].data().iter().map(|&v| v != 0.0).collect()
}

/// Scenario 4. Two `dropout` calls on the same value in one function are two sites,
/// `Reg.forward.dropout#0` and `#1`, each with its own token input; both `Dropout` nodes stay in
/// the Tensor IR after dead-code elimination, their masks differ, and one site's masks differ
/// between steps 0 and 1.
#[test]
fn dropout_sites_and_steps_differ() {
    let lowered = compile_tensor("two_sites.mspec", TWO_SITES).expect("the source compiles");
    let program = &lowered.program;
    let nodes: Vec<&modelspec_core::ir::tensor::TensorNode> = program
        .nodes
        .iter()
        .filter(|n| matches!(n.op, TensorOp::Dropout { .. }))
        .collect();
    assert_eq!(nodes.len(), 2, "both Dropout nodes survive DCE");
    assert_eq!(nodes[0].operands[0], nodes[1].operands[0], "one value");
    assert_ne!(nodes[0].operands[1], nodes[1].operands[1], "two tokens");
    assert_eq!(
        dropout_sites(program),
        ["Reg.forward.dropout#0", "Reg.forward.dropout#1"]
    );
    let first = mask_at(program, nodes[0].id, 0);
    let second = mask_at(program, nodes[1].id, 0);
    assert_ne!(first, second, "two sites, two masks");
    assert_ne!(
        first,
        mask_at(program, nodes[0].id, 1),
        "two steps, two masks"
    );
    assert_eq!(
        first,
        mask_at(program, nodes[0].id, 0),
        "one site and step, one mask"
    );
}
