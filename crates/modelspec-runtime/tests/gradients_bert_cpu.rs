//! The whole-model finite-difference check of the hidden-32 twin of BERT-tiny,
//! `crates/modelspec-cli/tests/fixtures/bert_cpu.mspec` (story:bert-tiny-cpu-model; DESIGN.md
//! § 4, § 9 gate 3).
//!
//! The file is compiled to Tensor IR with its dropouts in place and differentiated. At one
//! fixture batch (two sequences of 16, the second with four padded keys, six selected rows) and
//! the RNG tokens of every dropout site fixed at step 3, as `gradients_dropout.rs` fixes one, the
//! training program's gradient of every parameter, evaluated by the CPU interpreter in `f64`, is
//! compared with central differences of the forward program's loss in `f64` on 16 elements of
//! the parameter (all of a parameter with fewer) chosen by a fixed test seed. The error of a
//! parameter is the harness's: `max_i |g_i - d_i| / max_i |d_i|` over the checked elements, at
//! most `TOL_F64`. The two attention key biases are the exception: their gradient is zero (see
//! [`is_key_bias`]), so their gradient and differences are held to an absolute bound instead.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use modelspec_core::autodiff::differentiate;
use modelspec_core::compile::compile_tensor;
use modelspec_core::ir::tensor::{ParamId, TensorOp, TensorProgram};
use modelspec_core::ir::training::dropout_sites;
use modelspec_runtime::cpu::{HostTensor, evaluate_seeded};

#[allow(dead_code)]
#[path = "common/finite_difference.rs"]
mod finite_difference;

use finite_difference::{
    H as STEP, Ids, MIN_SCALE, Point, Rng, TOL_F64, Values, assert_structure, relative_error,
    seed_of,
};

const TWIN: &str = include_str!("../../modelspec-cli/tests/fixtures/bert_cpu.mspec");
/// The run seed the dropout masks are keyed with.
const SEED: u64 = 0x5eed_be27_0032_0016;
/// The step whose dropout tokens are fixed.
const TOKEN_STEP: u64 = 3;
/// Elements checked per parameter.
const CHECKED: usize = 16;
const B: usize = 2;
const S: usize = 16;
const V: i32 = 300;

/// The draw id of `site` at `step`: the first 8 bytes of `blake3(site, step as 8 LE bytes)`.
fn id_of(site: &str, step: u64) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(site.as_bytes());
    hasher.update(&step.to_le_bytes());
    let mut first = [0u8; 8];
    first.copy_from_slice(&hasher.finalize().as_bytes()[..8]);
    u64::from_le_bytes(first)
}

/// The token of `site` at `step`: `[id low, id high]` as `i32` bit patterns.
fn token_of(site: &str, step: u64) -> (Vec<u64>, Vec<i32>) {
    let id = id_of(site, step);
    (vec![2], vec![id as u32 as i32, (id >> 32) as u32 as i32])
}

/// The fixture batch: `ids`, `positions` and `labels` as `i32` inputs beside the dropout tokens,
/// and `key_mask` and `weights` as `f32`.
fn fixture_batch(rng: &mut Rng) -> (Ids, Values<String>) {
    let n = B * S;
    let token = |rng: &mut Rng| 5 + (rng.next_u64() % (V as u64 - 5)) as i32;
    let ids: Vec<i32> = (0..n).map(|_| token(rng)).collect();
    let positions: Vec<i32> = (0..n).map(|i| (i % S) as i32).collect();
    let labels: Vec<i32> = (0..n).map(|_| token(rng)).collect();
    let key_mask: Vec<f32> = (0..n)
        .map(|i| if i >= S && i % S >= 12 { 0.0 } else { 1.0 })
        .collect();
    let selected = [1usize, 6, 9, 14, 18, 25];
    let weights: Vec<f32> = (0..n)
        .map(|i| if selected.contains(&i) { 1.0 } else { 0.0 })
        .collect();
    let flat = vec![n as u64];
    let ids = BTreeMap::from([
        ("ids".to_owned(), (flat.clone(), ids)),
        ("positions".to_owned(), (flat.clone(), positions)),
        ("labels".to_owned(), (flat.clone(), labels)),
    ]);
    let floats = BTreeMap::from([
        ("key_mask".to_owned(), (vec![B as u64, S as u64], key_mask)),
        ("weights".to_owned(), (flat, weights)),
    ]);
    (ids, floats)
}

fn host<K: Ord + Clone>(
    values: &BTreeMap<K, (Vec<u64>, Vec<f32>)>,
) -> BTreeMap<K, HostTensor<f64>> {
    values
        .iter()
        .map(|(k, (shape, data))| {
            let data = data.iter().map(|&v| f64::from(v)).collect();
            (k.clone(), HostTensor::new(shape.clone(), data).unwrap())
        })
        .collect()
}

fn id_tensors(ids: &Ids) -> BTreeMap<String, HostTensor<i32>> {
    ids.iter()
        .map(|(k, (shape, data))| {
            (
                k.clone(),
                HostTensor::new(shape.clone(), data.clone()).unwrap(),
            )
        })
        .collect()
}

/// The loss of `program` in `f64` at `point`, element `index` of `param` moved by `delta`.
fn loss_at(
    program: &TensorProgram,
    point: &Point,
    param: ParamId,
    index: usize,
    delta: f64,
) -> f64 {
    let mut params = host(&point.params);
    let (shape, data) = &point.params[&param];
    let mut moved: Vec<f64> = data.iter().map(|&v| f64::from(v)).collect();
    moved[index] += delta;
    params.insert(param, HostTensor::new(shape.clone(), moved).unwrap());
    evaluate_seeded(
        program,
        &host(&point.inputs),
        &id_tensors(&point.ids),
        &params,
        SEED,
    )
    .unwrap()[0]
        .data()[0]
}

/// The bound of a gradient that is zero: `TOL_F64` of the smallest scale the relative error is
/// taken at, `MIN_SCALE`.
const ZERO_TOL: f64 = TOL_F64 * MIN_SCALE;

/// Whether `path` is the bias of an attention key projection, `l<k>_bk`. Its gradient is zero:
/// the bias adds `q · b` to every score of a query row, softmax over the keys of the row is
/// unchanged by a constant added to all of them, so the loss does not depend on it. Its
/// differences are rounding noise, so it is held to `ZERO_TOL` absolutely, every element of its
/// gradient as well as the checked differences, instead of to the relative error.
fn is_key_bias(path: &str) -> bool {
    path.rsplit_once('.')
        .is_some_and(|(_, name)| name.starts_with('l') && name.ends_with("_bk"))
}

/// `count` distinct indices below `n` drawn by `rng`, ascending; all of them when `n <= count`.
fn chosen(n: usize, count: usize, rng: &mut Rng) -> Vec<usize> {
    if n <= count {
        return (0..n).collect();
    }
    let mut picked = BTreeSet::new();
    while picked.len() < count {
        picked.insert((rng.next_u64() % n as u64) as usize);
    }
    picked.into_iter().collect()
}

/// Scenario 3: for one fixture batch of the twin and fixed dropout tokens, 16 elements of every
/// parameter (all of a parameter with fewer), chosen by a fixed test seed, have gradients within
/// `TOL_F64` of central differences in `f64`; the tied token table is among them.
#[test]
fn every_parameter_of_the_twin_matches_central_differences_in_f64() {
    let started = Instant::now();
    let lowered = compile_tensor("bert_cpu.mspec", TWIN).unwrap_or_else(|d| panic!("{d:?}"));
    let forward = lowered.program;
    let [loss] = forward.outputs[..] else {
        panic!("the loss is the one output")
    };
    let training = differentiate(&forward, loss).unwrap();
    assert_structure(&forward, loss, &training);
    assert_eq!(
        training.gradients.keys().collect::<Vec<_>>(),
        forward.params.keys().collect::<Vec<_>>(),
        "a gradient per parameter"
    );

    let paths: Vec<&str> = forward.params.values().map(|d| d.path.as_str()).collect();
    let tables: Vec<ParamId> = forward
        .params
        .iter()
        .filter(|(_, d)| d.path.ends_with(".tok_table"))
        .map(|(&id, _)| id)
        .collect();
    let [table] = tables[..] else {
        panic!("one token table among {paths:?}")
    };
    let reads_table = |op: fn(&TensorOp) -> bool| {
        forward.nodes.iter().any(|n| {
            op(&n.op)
                && n.operands.iter().any(|&o| {
                    forward
                        .node(o)
                        .is_some_and(|r| r.op == TensorOp::ParamRead { param: table })
                })
        })
    };
    assert!(
        reads_table(|op| *op == TensorOp::Embedding),
        "the token table is the embedding's table"
    );
    assert!(
        reads_table(|op| *op == TensorOp::Transpose),
        "the decoder is the token table transposed"
    );

    let key_biases: Vec<&str> = paths.iter().copied().filter(|p| is_key_bias(p)).collect();
    assert_eq!(
        key_biases,
        ["BertTiny.l0_bk", "BertTiny.l1_bk"],
        "the key biases are the two of the two layers"
    );

    let sites = dropout_sites(&forward);
    assert_eq!(
        sites.len(),
        5,
        "a dropout after the embeddings and each sublayer: {sites:?}"
    );
    let mut rng = Rng::new(seed_of("gradients_bert_cpu"));
    let (mut ids, floats) = fixture_batch(&mut rng);
    for site in &sites {
        ids.insert(site.clone(), token_of(site, TOKEN_STEP));
    }
    let mut point = Point::draw(&forward, &ids, &mut rng);
    for (name, value) in floats {
        assert!(point.inputs.contains_key(&name), "the twin reads `{name}`");
        point.inputs.insert(name, value);
    }

    let gradients = evaluate_seeded(
        &training.program,
        &host(&point.inputs),
        &id_tensors(&point.ids),
        &host(&point.params),
        SEED,
    )
    .unwrap();

    let mut checked = BTreeSet::new();
    let mut failures = Vec::new();
    println!("parameter | elements checked | max abs error | relative error | tolerance");
    for (k, (&param, _)) in training.gradients.iter().enumerate() {
        let path = &forward.params[&param].path;
        let n = point.params[&param].1.len();
        let indices = chosen(n, CHECKED, &mut rng);
        let g = gradients[1 + k].data();
        let got: Vec<f64> = indices.iter().map(|&i| g[i]).collect();
        let want: Vec<f64> = indices
            .iter()
            .map(|&i| {
                (loss_at(&forward, &point, param, i, STEP)
                    - loss_at(&forward, &point, param, i, -STEP))
                    / (2.0 * STEP)
            })
            .collect();
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let abs = got
            .iter()
            .zip(&want)
            .fold(0.0f64, |m, (g, d)| m.max((g - d).abs()));
        let error = relative_error(&got, &want);
        let largest = g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if is_key_bias(path) {
            println!(
                "{path} | {} of {n} | {abs:.3e} | zero gradient, max |g| {largest:.3e} | \
                 {ZERO_TOL:e} absolute",
                indices.len()
            );
        } else {
            println!(
                "{path} | {} of {n} | {abs:.3e} | {error:.3e} | {TOL_F64:e}",
                indices.len()
            );
        }
        if is_key_bias(path) {
            if scale >= MIN_SCALE || abs > ZERO_TOL || largest > ZERO_TOL {
                failures.push(format!(
                    "{path}: a key bias has a zero gradient, but the differences reach {scale:e}, \
                     the error {abs:e} and the gradient {largest:e}"
                ));
            }
        } else if scale < MIN_SCALE {
            failures.push(format!("{path}: differences {want:?} too small to compare"));
        } else if error > TOL_F64 {
            failures.push(format!(
                "{path}: f64 error {error:e}; gradient {got:?}, differences {want:?}"
            ));
        }
        checked.insert(param);
    }
    println!("wall time {:.2} s", started.elapsed().as_secs_f64());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(checked.contains(&table), "the tied token table is checked");
    assert_eq!(
        checked.len(),
        forward.params.len(),
        "every parameter is checked"
    );
}
