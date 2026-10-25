//! Adversary pass 1 on story:bert-tiny-cpu-model: the program text of
//! `examples/bert_tiny.mspec` and its twin `crates/modelspec-cli/tests/fixtures/bert_cpu.mspec`
//! read as BERT (Devlin et al. 2019) from their Tensor IR, the reach of the twin's gradient check,
//! dropout in that check, and the zero gradient of the key biases.

use std::collections::{BTreeMap, BTreeSet};

use modelspec_core::autodiff::differentiate;
use modelspec_core::compile::compile_tensor;
use modelspec_core::ir::tensor::{ParamId, TensorNode, TensorOp, TensorProgram, ValueId};
use modelspec_core::ir::training::dropout_sites;
use modelspec_runtime::cpu::{HostTensor, evaluate_seeded};

#[allow(dead_code)]
#[path = "common/finite_difference.rs"]
mod finite_difference;

use finite_difference::{H as STEP, Ids, Point, Rng, TOL_F64, Values, relative_error, seed_of};

const EXAMPLE: &str = include_str!("../../../examples/bert_tiny.mspec");
const TWIN: &str = include_str!("../../modelspec-cli/tests/fixtures/bert_cpu.mspec");

/// The constants of `gradients_bert_cpu.rs`, copied so its draws can be replayed.
const SEED: u64 = 0x5eed_be27_0032_0016;
const TOKEN_STEP: u64 = 3;
const CHECKED: usize = 16;
const B: usize = 2;
const S: usize = 16;
const V: i32 = 300;
const H_TWIN: usize = 32;

fn forward_of(name: &str, source: &str) -> TensorProgram {
    compile_tensor(name, source)
        .unwrap_or_else(|d| panic!("{d:?}"))
        .program
}

/// Reads a program's nodes by id and by use.
struct Graph<'a> {
    p: &'a TensorProgram,
}

impl<'a> Graph<'a> {
    fn node(&self, v: ValueId) -> &'a TensorNode {
        self.p.node(v).expect("a defined value")
    }

    /// The parameter name (path without the model) `v` reads, if it is a `ParamRead`.
    fn param(&self, v: ValueId) -> Option<String> {
        match self.node(v).op {
            TensorOp::ParamRead { param } => {
                let path = &self.p.params[&param].path;
                Some(
                    path.rsplit_once('.')
                        .map_or(path.clone(), |(_, n)| n.to_owned()),
                )
            }
            _ => None,
        }
    }

    fn users(&self, v: ValueId, op: impl Fn(&TensorOp) -> bool) -> Vec<&'a TensorNode> {
        self.p
            .nodes
            .iter()
            .filter(|n| op(&n.op) && n.operands.contains(&v))
            .collect()
    }

    fn one_user(
        &self,
        v: ValueId,
        what: &str,
        op: impl Fn(&TensorOp) -> bool,
    ) -> Result<&'a TensorNode, String> {
        match self.users(v, op)[..] {
            [n] => Ok(n),
            ref other => Err(format!("{v}: {} users that are {what}", other.len())),
        }
    }

    /// `x` of `add_rows(matmul(x, <w>), <b>)` at `v`, with `w` and `b` the names given.
    fn dense_input(&self, v: ValueId, w: &str, b: &str) -> Result<ValueId, String> {
        let n = self.node(v);
        if n.op != TensorOp::AddRows {
            return Err(format!("{v} is {:?}, not add_rows of {w}", n.op));
        }
        if self.param(n.operands[1]).as_deref() != Some(b) {
            return Err(format!("{v}: bias is not {b}"));
        }
        let m = self.node(n.operands[0]);
        if m.op != TensorOp::MatMul || self.param(m.operands[1]).as_deref() != Some(w) {
            return Err(format!("{v}: not matmul(_, {w})"));
        }
        Ok(m.operands[0])
    }

    /// The `add_rows(matmul(x, <w>), <b>)` node reading `x`.
    fn dense_of(&self, x: ValueId, w: &str, b: &str) -> Result<ValueId, String> {
        for m in self.users(x, |op| *op == TensorOp::MatMul) {
            if m.operands[0] == x && self.param(m.operands[1]).as_deref() == Some(w) {
                let a = self.one_user(m.id, "add_rows", |op| *op == TensorOp::AddRows)?;
                if self.param(a.operands[1]).as_deref() == Some(b) {
                    return Ok(a.id);
                }
            }
        }
        Err(format!("no add_rows(matmul({x}, {w}), {b})"))
    }

    /// `layer_norm(add(residual, dropout(sub, p = 0.1)), <g>, <b>)` reading `sub`; returns it.
    fn residual_norm(
        &self,
        residual: ValueId,
        sub: ValueId,
        g: &str,
        b: &str,
    ) -> Result<ValueId, String> {
        let d = self.one_user(sub, "dropout", |op| matches!(op, TensorOp::Dropout { .. }))?;
        let TensorOp::Dropout { p } = d.op else {
            unreachable!()
        };
        if p != 0.1 {
            return Err(format!("dropout after {sub} has p = {p}"));
        }
        let add = self.one_user(d.id, "add", |op| *op == TensorOp::Add)?;
        let other: Vec<ValueId> = add
            .operands
            .iter()
            .copied()
            .filter(|&o| o != d.id)
            .collect();
        if other != [residual] {
            return Err(format!(
                "the add after {sub}'s dropout reads {other:?}, not the sublayer input {residual}"
            ));
        }
        let ln = self.one_user(add.id, "layer_norm", |op| *op == TensorOp::LayerNorm)?;
        if ln.operands[0] != add.id
            || self.param(ln.operands[1]).as_deref() != Some(g)
            || self.param(ln.operands[2]).as_deref() != Some(b)
        {
            return Err(format!("layer_norm after {sub} is not ({g}, {b})"));
        }
        Ok(ln.id)
    }
}

/// Walks the Tensor IR of `source` as a post-layer-norm BERT encoder of `layers` layers with
/// `heads` heads and the MLM head with a tied decoder, returning the first deviation.
fn bert_structure(name: &str, source: &str, layers: usize, heads: u64) -> Result<(), String> {
    let p = forward_of(name, source);
    let g = Graph { p: &p };

    // Embeddings: dropout(layer_norm(add(embedding(tok_table, ids), embedding(pos_table, positions)))).
    let embeddings: Vec<&TensorNode> = p
        .nodes
        .iter()
        .filter(|n| n.op == TensorOp::Embedding)
        .collect();
    let input_name = |v: ValueId| match &g.node(v).op {
        TensorOp::Input { name } => Some(name.clone()),
        _ => None,
    };
    let tok = embeddings
        .iter()
        .find(|n| g.param(n.operands[0]).as_deref() == Some("tok_table"))
        .ok_or("no embedding of tok_table")?;
    let pos = embeddings
        .iter()
        .find(|n| g.param(n.operands[0]).as_deref() == Some("pos_table"))
        .ok_or("no embedding of pos_table")?;
    if input_name(tok.operands[1]).as_deref() != Some("ids") {
        return Err("tok_table is not indexed by `ids`".into());
    }
    if input_name(pos.operands[1]).as_deref() != Some("positions") {
        return Err("pos_table is not indexed by `positions`".into());
    }
    let sum = g.one_user(tok.id, "add", |op| *op == TensorOp::Add)?;
    if !sum.operands.contains(&pos.id) {
        return Err("token and position embeddings are not added".into());
    }
    let eln = g.one_user(sum.id, "layer_norm", |op| *op == TensorOp::LayerNorm)?;
    if g.param(eln.operands[1]).as_deref() != Some("emb_ln_gamma") {
        return Err("the embedding layer norm is not emb_ln".into());
    }
    let ed = g.one_user(eln.id, "dropout", |op| {
        matches!(op, TensorOp::Dropout { .. })
    })?;
    if ed.op != (TensorOp::Dropout { p: 0.1 }) {
        return Err(format!("embedding dropout is {:?}", ed.op));
    }

    let mut x = ed.id;
    for l in 0..layers {
        let n = |s: &str| format!("l{l}_{s}");
        let q = g.dense_of(x, &n("wq"), &n("bq"))?;
        let k = g.dense_of(x, &n("wk"), &n("bk"))?;
        let v = g.dense_of(x, &n("wv"), &n("bv"))?;
        let att = g.one_user(q, "attention", |op| {
            matches!(op, TensorOp::Attention { .. })
        })?;
        if att.op != (TensorOp::Attention { heads }) {
            return Err(format!("layer {l}: {:?}, not {heads} heads", att.op));
        }
        if att.operands[..3] != [q, k, v] {
            return Err(format!("layer {l}: attention operands are not (q, k, v)"));
        }
        if input_name(att.operands[3]).as_deref() != Some("key_mask") {
            return Err(format!("layer {l}: the mask is not `key_mask`"));
        }
        let a = g.dense_of(att.id, &n("wo"), &n("bo"))?;
        let h = g.residual_norm(x, a, &n("ln1_gamma"), &n("ln1_beta"))?;
        let f1 = g.dense_of(h, &n("w1"), &n("b1"))?;
        let gelu = g.one_user(f1, "gelu", |op| *op == TensorOp::Gelu)?;
        let f2 = g.dense_of(gelu.id, &n("w2"), &n("b2"))?;
        x = g.residual_norm(h, f2, &n("ln2_gamma"), &n("ln2_beta"))?;
    }

    // MLM head: add_rows(matmul(layer_norm(gelu(dense(x))), transpose(tok_table)), mlm_bias).
    let dense = g.dense_of(x, "mlm_w", "mlm_b")?;
    let gelu = g.one_user(dense, "gelu", |op| *op == TensorOp::Gelu)?;
    let ln = g.one_user(gelu.id, "layer_norm", |op| *op == TensorOp::LayerNorm)?;
    if g.param(ln.operands[1]).as_deref() != Some("mlm_ln_gamma") {
        return Err("the head's layer norm is not mlm_ln".into());
    }
    let dec = g.one_user(ln.id, "matmul", |op| *op == TensorOp::MatMul)?;
    let t = g.node(dec.operands[1]);
    if t.op != TensorOp::Transpose || g.param(t.operands[0]).as_deref() != Some("tok_table") {
        return Err("the decoder is not transpose(tok_table)".into());
    }
    let logits = g.one_user(dec.id, "add_rows", |op| *op == TensorOp::AddRows)?;
    if g.param(logits.operands[1]).as_deref() != Some("mlm_bias") {
        return Err("the decoder bias is not mlm_bias".into());
    }
    let ce = g.one_user(logits.id, "cross_entropy", |op| {
        *op == TensorOp::CrossEntropy
    })?;
    if p.outputs != [ce.id] {
        return Err("the loss is not the cross entropy of the logits".into());
    }

    let dropouts = p
        .nodes
        .iter()
        .filter(|n| matches!(n.op, TensorOp::Dropout { .. }))
        .count();
    if dropouts != 1 + 2 * layers {
        return Err(format!("{dropouts} dropouts, not {}", 1 + 2 * layers));
    }
    let _ = g.dense_input(dense, "mlm_w", "mlm_b")?;
    Ok(())
}

/// H1: both files are, node by node, a two-layer post-layer-norm BERT with two heads, residuals
/// of the sublayer input, dropout 0.1 at five sites and the tied MLM head.
#[test]
fn both_files_are_post_ln_bert_in_the_ir() {
    bert_structure("bert_tiny.mspec", EXAMPLE, 2, 2).unwrap();
    bert_structure("bert_cpu.mspec", TWIN, 2, 2).unwrap();
}

/// The walk above rejects the mutants a digit-masking comparison of the two files (the unit's
/// first guard, deleted under coordinator decision D71) lets through when they are made to both
/// files, or to the twin in its digits only.
#[test]
fn the_structure_walk_rejects_the_mutants() {
    let mutants: [(&str, String); 6] = [
        (
            "layer 1 residual dropped",
            EXAMPLE.replace(
                "layer_norm(add(x0, dropout(a1, p = 0.1)), l1_ln1_gamma",
                "layer_norm(dropout(a1, p = 0.1), l1_ln1_gamma",
            ),
        ),
        (
            "residual of the normalised value",
            EXAMPLE.replace(
                "let x0 = layer_norm(add(h0, dropout(f0",
                "let x0 = layer_norm(add(f0, dropout(f0",
            ),
        ),
        ("one head", TWIN.replace("heads = 2", "heads = 1")),
        (
            "dropout p 0.5 in the twin",
            TWIN.replacen("p = 0.1", "p = 0.5", 1),
        ),
        (
            "layer 1 reads layer 0's query",
            TWIN.replace("matmul(x0, l1_wq), l1_bq", "matmul(x0, l0_wq), l1_bq"),
        ),
        (
            "gelu to relu in the head",
            EXAMPLE.replace(
                "layer_norm(gelu(add_rows(matmul(x1",
                "layer_norm(relu(add_rows(matmul(x1",
            ),
        ),
    ];
    for (what, source) in mutants {
        let base = if source.contains("[4096") {
            EXAMPLE
        } else {
            TWIN
        };
        assert_ne!(source, base, "{what}: the mutation applied");
        let verdict = std::panic::catch_unwind(|| bert_structure(what, &source, 2, 2));
        assert!(
            !matches!(verdict, Ok(Ok(()))),
            "{what}: the structure walk accepts the mutant"
        );
    }
}

/// The fixture batch of `gradients_bert_cpu.rs`, copied verbatim so the test seed's draws replay.
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

fn token_of(site: &str, step: u64) -> (Vec<u64>, Vec<i32>) {
    let mut hasher = blake3::Hasher::new();
    hasher.update(site.as_bytes());
    hasher.update(&step.to_le_bytes());
    let mut first = [0u8; 8];
    first.copy_from_slice(&hasher.finalize().as_bytes()[..8]);
    let id = u64::from_le_bytes(first);
    (vec![2], vec![id as u32 as i32, (id >> 32) as u32 as i32])
}

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

fn host<K: Ord + Clone>(values: &Values<K>) -> BTreeMap<K, HostTensor<f64>> {
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
        .map(|(k, (s, d))| (k.clone(), HostTensor::new(s.clone(), d.clone()).unwrap()))
        .collect()
}

/// The twin's forward program and the fixture point of `gradients_bert_cpu.rs` with its tokens
/// at `step`, and the test RNG in the state it is in after the point was drawn.
fn twin_point(step: u64) -> (TensorProgram, Point, Rng) {
    let forward = forward_of("bert_cpu.mspec", TWIN);
    let mut rng = Rng::new(seed_of("gradients_bert_cpu"));
    let (mut ids, floats) = fixture_batch(&mut rng);
    for site in &dropout_sites(&forward) {
        ids.insert(site.clone(), token_of(site, step));
    }
    let mut point = Point::draw(&forward, &ids, &mut rng);
    for (name, value) in floats {
        point.inputs.insert(name, value);
    }
    (forward, point, rng)
}

fn loss(program: &TensorProgram, point: &Point, params: &Values<ParamId>) -> f64 {
    evaluate_seeded(
        program,
        &host(&point.inputs),
        &id_tensors(&point.ids),
        &host(params),
        SEED,
    )
    .unwrap()[0]
        .data()[0]
}

fn param_id(forward: &TensorProgram, name: &str) -> ParamId {
    *forward
        .params
        .iter()
        .find(|(_, d)| d.path.ends_with(&format!(".{name}")))
        .unwrap()
        .0
}

/// H4: the gradient check runs with dropout active: the same point with the tokens of another
/// step has another loss, so the check's forward and differences see the masks.
#[test]
fn the_gradient_check_point_has_dropout_active() {
    let (forward, at3, _) = twin_point(TOKEN_STEP);
    let (_, at4, _) = twin_point(TOKEN_STEP + 1);
    assert_eq!(at3.params, at4.params);
    let l3 = loss(&forward, &at3, &at3.params);
    let l4 = loss(&forward, &at4, &at4.params);
    assert!(
        (l3 - l4).abs() > 1e-6,
        "loss at step-3 tokens {l3} and step-4 tokens {l4}: dropout is not in the check"
    );
}

/// H4: the key-bias argument at a finite move: every element of `l0_bk` and `l1_bk` moved by
/// +2.0 leaves the loss of the masked fixture batch unchanged to 1e-12 relative, and the same
/// move of `l0_bq` (the control) does not.
#[test]
fn a_finite_key_bias_move_leaves_the_loss() {
    let (forward, point, _) = twin_point(TOKEN_STEP);
    let base = loss(&forward, &point, &point.params);
    let moved = |name: &str| {
        let id = param_id(&forward, name);
        let mut params = point.params.clone();
        for v in &mut params.get_mut(&id).unwrap().1 {
            *v += 2.0;
        }
        loss(&forward, &point, &params)
    };
    for bias in ["l0_bk", "l1_bk"] {
        let l = moved(bias);
        assert!(
            (l - base).abs() <= 1e-12 * base.abs(),
            "{bias} moved by 2.0: loss {l} against {base}"
        );
    }
    let l = moved("l0_bq");
    assert!(
        (l - base).abs() > 1e-6,
        "the control l0_bq moved: {l} against {base}"
    );
}

/// H4, reach: of the 16 `tok_table` elements `gradients_bert_cpu.rs` checks, replayed from its
/// test seed, at least one lies in a row that `ids` reads, so the embedding adjoint of the tied
/// table (and not only the decoder's) is among the checked gradients.
#[test]
fn checked_table_elements_reach_the_embedding_adjoint() {
    let (forward, point, mut rng) = twin_point(TOKEN_STEP);
    let training = differentiate(&forward, forward.outputs[0]).unwrap();
    let table = param_id(&forward, "tok_table");
    let read: BTreeSet<usize> = point.ids["ids"].1.iter().map(|&i| i as usize).collect();
    let mut table_indices = None;
    for &param in training.gradients.keys() {
        let n = point.params[&param].1.len();
        let indices = chosen(n, CHECKED, &mut rng);
        if param == table {
            table_indices = Some(indices);
        }
    }
    let indices = table_indices.expect("the table is checked");
    let rows: Vec<usize> = indices.iter().map(|i| i / H_TWIN).collect();
    assert!(
        indices.iter().any(|&i| i >= CHECKED),
        "the checked elements are the first {CHECKED}: {indices:?}"
    );
    assert!(
        rows.iter().any(|r| read.contains(r)),
        "no checked tok_table element is in a row `ids` reads: rows {rows:?}, ids {read:?}"
    );
}

/// H4, direct: 16 `tok_table` elements in rows that `ids` reads, against central differences in
/// `f64`, within `TOL_F64` relative, at the fixture point with dropout tokens of step 3.
#[test]
fn embedded_rows_of_the_tied_table_match_central_differences() {
    let (forward, point, _) = twin_point(TOKEN_STEP);
    let training = differentiate(&forward, forward.outputs[0]).unwrap();
    let table = param_id(&forward, "tok_table");
    let k = training.gradients.keys().position(|&p| p == table).unwrap();
    let gradients = evaluate_seeded(
        &training.program,
        &host(&point.inputs),
        &id_tensors(&point.ids),
        &host(&point.params),
        SEED,
    )
    .unwrap();
    let g = gradients[1 + k].data();
    let rows: Vec<usize> = point.ids["ids"]
        .1
        .iter()
        .take(8)
        .map(|&i| i as usize)
        .collect();
    let indices: Vec<usize> = rows
        .iter()
        .flat_map(|r| [r * H_TWIN + 3, r * H_TWIN + 29])
        .collect();
    let got: Vec<f64> = indices.iter().map(|&i| g[i]).collect();
    let at = |i: usize, d: f64| {
        let mut moved: Vec<f64> = point.params[&table]
            .1
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        moved[i] += d;
        let shape = point.params[&table].0.clone();
        let mut h = host(&point.params);
        h.insert(table, HostTensor::new(shape, moved).unwrap());
        evaluate_seeded(
            &forward,
            &host(&point.inputs),
            &id_tensors(&point.ids),
            &h,
            SEED,
        )
        .unwrap()[0]
            .data()[0]
    };
    let want: Vec<f64> = indices
        .iter()
        .map(|&i| (at(i, STEP) - at(i, -STEP)) / (2.0 * STEP))
        .collect();
    let error = relative_error(&got, &want);
    assert!(
        error <= TOL_F64,
        "error {error:e}: {got:?} against {want:?}"
    );
}
