//! Multi-head bidirectional attention with a key mask on the CPU interpreter (DESIGN.md § 2.2,
//! § 4, § 9 gate 3; story:bert-attention).
//!
//! `attention(q, k, v, key_mask, heads = A)` takes `q`, `k`, `v` of shape `[N, H]` and
//! `key_mask` of shape `[B, S]` with `N = B * S`; sequence `b` is rows `b * S` to `b * S + S - 1`
//! and head `a` columns `a * H/A` to `(a + 1) * H/A - 1`. Its gradients are the internal ops
//! `AttentionGradQ`, `AttentionGradK` and `AttentionGradV`, each over `q, k, v, key_mask, G`;
//! `key_mask` has none. The key mask is fixed while the parameters are perturbed, so the
//! finite-difference case draws its point with the harness of `tests/common/finite_difference.rs`
//! and then sets the mask.

use std::collections::BTreeMap;

use modelspec_core::autodiff::differentiate;
use modelspec_core::ir::tensor::{ParamId, TensorOp, TensorProgram, TensorProgramBuilder, ValueId};
use modelspec_runtime::cpu::{Element, HostTensor, RuntimeError, evaluate};

#[allow(dead_code)]
#[path = "common/finite_difference.rs"]
mod finite_difference;

use finite_difference::{
    H as STEP, Ids, MIN_SCALE, Point, Rng, TOL_F32, TOL_F64, assert_structure, input, loss_at,
    mse_to_target, param, relative_error, run, seed_of, ty,
};

/// Sequences in the batch.
const B: u64 = 2;
/// Tokens per sequence.
const S: u64 = 3;
/// Rows of `q`, `k` and `v`: `B * S`.
const N: u64 = B * S;
/// Width of `q`, `k` and `v`.
const H: u64 = 4;
/// Heads; each reads `H / HEADS = 2` columns.
const HEADS: u64 = 2;
/// Width of the input of the `matmul`s that make `q`, `k` and `v`.
const D: u64 = 3;

/// The key mask of the scenarios: sequence 0 sees every key, sequence 1 all but its last.
const PARTIAL_MASK: [f32; 6] = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
/// The row of the key the partial mask hides: the last key of sequence 1.
const MASKED_ROW: usize = 5;

/// `mse(attention(matmul(x, w_q), matmul(x, w_k), matmul(x, w_v), key_mask, heads = 2), t)`
/// with `B = 2`, `S = 3`, `H = 4` and the key mask hiding the last key of sequence 1: the
/// gradients of `w_q`, `w_k` and `w_v` are within `TOL_F64` of central differences in `f64` and
/// `TOL_F32` in `f32`, and the adjoints of `q`, `k` and `v` are `AttentionGradQ`,
/// `AttentionGradK` and `AttentionGradV` over the forward operands.
#[test]
fn attention_gradients_of_w_q_w_k_and_w_v_match_central_differences() {
    let name = "attention_gradients_of_w_q_w_k_and_w_v_match_central_differences";
    let mut b = TensorProgramBuilder::new();
    let x = input(&mut b, "x", &[N, D]);
    let mask = input(&mut b, "key_mask", &[B, S]);
    let w_q = param(&mut b, "w_q", &[D, H]);
    let w_k = param(&mut b, "w_k", &[D, H]);
    let w_v = param(&mut b, "w_v", &[D, H]);
    let q = b.matmul(x, w_q).unwrap();
    let k = b.matmul(x, w_k).unwrap();
    let v = b.matmul(x, w_v).unwrap();
    let y = b.attention(q, k, v, mask, HEADS).unwrap();
    assert_eq!(b.type_of(y), Some(&ty(&[N, H])));
    let loss = mse_to_target(&mut b, y);
    let forward = b.finish();
    let training = differentiate(&forward, loss).unwrap();
    assert_structure(&forward, loss, &training);
    assert_eq!(
        training.gradients.len(),
        3,
        "a gradient for w_q, w_k and w_v"
    );

    let program = &training.program;
    for (k_param, grad) in [
        (0, TensorOp::AttentionGradQ { heads: HEADS }),
        (1, TensorOp::AttentionGradK { heads: HEADS }),
        (2, TensorOp::AttentionGradV { heads: HEADS }),
    ] {
        let dw = program.node(training.gradients[&ParamId(k_param)]).unwrap();
        assert_eq!(dw.op, TensorOp::MatMul, "dW = transpose(x) @ dZ");
        let dz = program.node(dw.operands[1]).unwrap();
        assert_eq!(dz.op, grad, "{dz:?}");
        assert_eq!(dz.operands[..4], [q, k, v, mask], "{dz:?}");
    }

    let mut rng = Rng::new(seed_of(name));
    let mut point = Point::draw(&forward, &Ids::new(), &mut rng);
    point
        .inputs
        .insert("key_mask".into(), (vec![B, S], PARTIAL_MASK.to_vec()));
    let in_f64 = run(&training, &point, f64::from);
    let in_f32 = run(&training, &point, |v| v);
    for (k_param, (&param, _)) in training.gradients.iter().enumerate() {
        let path = &forward.params[&param].path;
        let n = point.params[&param].1.len();
        let differences: Vec<f64> = (0..n)
            .map(|i| {
                (loss_at(program, &point, param, i, STEP)
                    - loss_at(program, &point, param, i, -STEP))
                    / (2.0 * STEP)
            })
            .collect();
        let scale = differences.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(scale >= MIN_SCALE, "{path}: differences {differences:?}");
        let g64 = in_f64[1 + k_param].data();
        let g32: Vec<f64> = in_f32[1 + k_param]
            .data()
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let e64 = relative_error(g64, &differences);
        let e32 = relative_error(&g32, &differences);
        assert!(
            e64 <= TOL_F64,
            "{path}: f64 error {e64:e}; {g64:?} vs {differences:?}"
        );
        assert!(
            e32 <= TOL_F32,
            "{path}: f32 error {e32:e}; {g32:?} vs {differences:?}"
        );
    }
}

/// The inputs `q`, `k`, `v` and `key_mask` of `attention` with `heads`, as the program's inputs.
fn attention_program(heads: u64, shape: [u64; 2], mask: [u64; 2]) -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let q = input(&mut b, "q", &shape);
    let k = input(&mut b, "k", &shape);
    let v = input(&mut b, "v", &shape);
    let m = input(&mut b, "key_mask", &mask);
    let y = b.attention(q, k, v, m, heads).unwrap();
    b.output(y).unwrap();
    b.finish()
}

/// `values` as host tensors in `T`, by name.
fn host<T: Element>(
    values: &[(&str, &[u64], &[f32])],
    convert: impl Fn(f32) -> T,
) -> BTreeMap<String, HostTensor<T>> {
    values
        .iter()
        .map(|&(name, shape, data)| {
            let data = data.iter().map(|&v| convert(v)).collect();
            (
                name.to_owned(),
                HostTensor::new(shape.to_vec(), data).unwrap(),
            )
        })
        .collect()
}

/// `n` values drawn from `rng`.
fn draw(rng: &mut Rng, n: u64) -> Vec<f32> {
    (0..n).map(|_| rng.value()).collect()
}

/// Changing `k` and `v` in the row of the masked key leaves the output unchanged bit for bit, in
/// `f32` and in `f64`; with `q`, `k` and `v` trainable, the gradients of `k` and `v` are exactly
/// `+0.0` in that row and not zero in every other row.
#[test]
fn attention_masked_key_changes_nothing_and_gets_zero_gradients() {
    let program = attention_program(HEADS, [N, H], [B, S]);
    let mut rng = Rng::new(seed_of("attention_masked_key"));
    let (q, k, v) = (
        draw(&mut rng, N * H),
        draw(&mut rng, N * H),
        draw(&mut rng, N * H),
    );
    let (mut k2, mut v2) = (k.clone(), v.clone());
    let h = H as usize;
    for c in 0..h {
        k2[MASKED_ROW * h + c] = 1.0e6 * (c as f32 + 1.0);
        v2[MASKED_ROW * h + c] = -3.0e5 * (c as f32 + 2.0);
    }
    let shape: &[u64] = &[N, H];
    let mask_shape: &[u64] = &[B, S];
    let inputs = |k: &[f32], v: &[f32]| -> Vec<(String, Vec<u64>, Vec<f32>)> {
        vec![
            ("q".into(), shape.to_vec(), q.clone()),
            ("k".into(), shape.to_vec(), k.to_vec()),
            ("v".into(), shape.to_vec(), v.to_vec()),
            (
                "key_mask".into(),
                mask_shape.to_vec(),
                PARTIAL_MASK.to_vec(),
            ),
        ]
    };
    let as_refs = |values: &[(String, Vec<u64>, Vec<f32>)]| -> Vec<(String, Vec<u64>, Vec<f32>)> {
        values.to_vec()
    };
    let run32 = |values: Vec<(String, Vec<u64>, Vec<f32>)>| {
        let refs: Vec<(&str, &[u64], &[f32])> = values
            .iter()
            .map(|(n, s, d)| (n.as_str(), s.as_slice(), d.as_slice()))
            .collect();
        let out = evaluate(&program, &host(&refs, |x| x), &BTreeMap::new()).unwrap();
        let out64 = evaluate(&program, &host(&refs, f64::from), &BTreeMap::new()).unwrap();
        (
            out[0]
                .data()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            out64[0]
                .data()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
        )
    };
    let before = run32(as_refs(&inputs(&k, &v)));
    let after = run32(as_refs(&inputs(&k2, &v2)));
    assert_eq!(
        before.0, after.0,
        "f32 output changed with the masked key's row"
    );
    assert_eq!(
        before.1, after.1,
        "f64 output changed with the masked key's row"
    );

    let mut b = TensorProgramBuilder::new();
    let qp = param(&mut b, "q", &[N, H]);
    let kp = param(&mut b, "k", &[N, H]);
    let vp = param(&mut b, "v", &[N, H]);
    let m = input(&mut b, "key_mask", &[B, S]);
    let y = b.attention(qp, kp, vp, m, HEADS).unwrap();
    let loss = mse_to_target(&mut b, y);
    let forward = b.finish();
    let training = differentiate(&forward, loss).unwrap();
    let mut point = Point::draw(&forward, &Ids::new(), &mut rng);
    point
        .inputs
        .insert("key_mask".into(), (vec![B, S], PARTIAL_MASK.to_vec()));
    let in_f32 = run(&training, &point, |x| x);
    let in_f64 = run(&training, &point, f64::from);
    for (position, name) in [(2, "k"), (3, "v")] {
        let g32 = in_f32[position].data();
        let g64 = in_f64[position].data();
        for row in 0..N as usize {
            let r32 = &g32[row * h..row * h + h];
            let r64 = &g64[row * h..row * h + h];
            if row == MASKED_ROW {
                assert!(
                    r32.iter().all(|x| x.to_bits() == 0),
                    "d{name} f32 row {row}: {r32:?}"
                );
                assert!(
                    r64.iter().all(|x| x.to_bits() == 0),
                    "d{name} f64 row {row}: {r64:?}"
                );
            } else {
                assert!(
                    r32.iter().any(|&x| x != 0.0),
                    "d{name} f32 row {row}: {r32:?}"
                );
            }
        }
    }
}

/// `softmax(q_b k_b^T / sqrt(H)) v_b` for one sequence `b`, with the landed operators
/// `matmul`, `transpose`, `mul` and `softmax` over the sequence's `[S, H]` rows.
fn reference_program() -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let q = input(&mut b, "q", &[S, H]);
    let k = input(&mut b, "k", &[S, H]);
    let v = input(&mut b, "v", &[S, H]);
    let kt = b.transpose(k).unwrap();
    let scores = b.matmul(q, kt).unwrap();
    let scale = b.scalar(1.0 / (H as f32).sqrt());
    let scaled = b.mul(scores, scale).unwrap();
    let p = b.softmax(scaled).unwrap();
    let y = b.matmul(p, v).unwrap();
    b.output(y).unwrap();
    b.finish()
}

/// With `heads = 1` and a key mask of ones, each sequence of the output is within `1e-6`
/// (maximum absolute difference) of `softmax(q k^T / sqrt(H)) v` computed per sequence with the
/// operators `softmax` and `matmul`, in `f32`.
#[test]
fn attention_one_head_reference_equals_softmax_matmul_per_sequence() {
    let program = attention_program(1, [N, H], [B, S]);
    let reference = reference_program();
    let mut rng = Rng::new(seed_of("attention_one_head_reference"));
    let (q, k, v) = (
        draw(&mut rng, N * H),
        draw(&mut rng, N * H),
        draw(&mut rng, N * H),
    );
    let ones = vec![1.0f32; (B * S) as usize];
    let got = evaluate(
        &program,
        &host(
            &[
                ("q", &[N, H], &q),
                ("k", &[N, H], &k),
                ("v", &[N, H], &v),
                ("key_mask", &[B, S], &ones),
            ],
            |x| x,
        ),
        &BTreeMap::new(),
    )
    .unwrap()
    .remove(0);
    let rows = (S * H) as usize;
    let mut worst = 0.0f64;
    for seq in 0..B as usize {
        let part = |x: &[f32]| x[seq * rows..seq * rows + rows].to_vec();
        let want = evaluate(
            &reference,
            &host(
                &[
                    ("q", &[S, H], &part(&q)),
                    ("k", &[S, H], &part(&k)),
                    ("v", &[S, H], &part(&v)),
                ],
                |x| x,
            ),
            &BTreeMap::new(),
        )
        .unwrap()
        .remove(0);
        for (g, w) in got.data()[seq * rows..seq * rows + rows]
            .iter()
            .zip(want.data())
        {
            worst = worst.max((f64::from(*g) - f64::from(*w)).abs());
        }
    }
    assert!(worst <= 1e-6, "largest difference {worst:e}");
}

/// The inputs of `attention_program(HEADS, ..)` with `mask` as the key mask.
fn run_with_mask(
    program: &TensorProgram,
    mask: &[f32],
) -> Result<Vec<HostTensor<f32>>, RuntimeError> {
    let mut rng = Rng::new(seed_of("attention_fully_masked"));
    let (q, k, v) = (
        draw(&mut rng, N * H),
        draw(&mut rng, N * H),
        draw(&mut rng, N * H),
    );
    evaluate(
        program,
        &host(
            &[
                ("q", &[N, H], &q),
                ("k", &[N, H], &k),
                ("v", &[N, H], &v),
                ("key_mask", &[B, S], mask),
            ],
            |x| x,
        ),
        &BTreeMap::new(),
    )
}

/// A sequence whose key mask is all 0 is a `RuntimeError` naming that sequence, not NaN, for
/// `attention` and for each of its gradient ops.
#[test]
fn attention_fully_masked_sequence_is_a_runtime_error_not_nan() {
    let program = attention_program(HEADS, [N, H], [B, S]);
    let attention = program.outputs[0];
    let mask = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    assert_eq!(
        run_with_mask(&program, &mask),
        Err(RuntimeError::NoUnmaskedKey {
            value: attention,
            sequence: 1,
        })
    );
    let error = run_with_mask(&program, &mask).unwrap_err().to_string();
    assert!(error.contains("sequence 1"), "{error}");

    for (k, grad) in [
        TensorOp::AttentionGradQ { heads: HEADS },
        TensorOp::AttentionGradK { heads: HEADS },
        TensorOp::AttentionGradV { heads: HEADS },
    ]
    .into_iter()
    .enumerate()
    {
        let mut b = TensorProgramBuilder::new();
        let q = input(&mut b, "q", &[N, H]);
        let kk = input(&mut b, "k", &[N, H]);
        let v = input(&mut b, "v", &[N, H]);
        let m = input(&mut b, "key_mask", &[B, S]);
        let g = b.add(q, v).unwrap();
        let d = b.push(grad.clone(), &[q, kk, v, m, g]).unwrap();
        b.output(d).unwrap();
        let program = b.finish();
        assert_eq!(
            run_with_mask(&program, &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            Err(RuntimeError::NoUnmaskedKey {
                value: d,
                sequence: 0,
            }),
            "{k}: {grad:?}"
        );
    }
}

/// A key mask value other than 0 or 1, `0.5` or NaN, is a `RuntimeError` naming the sequence
/// and the key.
#[test]
fn attention_key_mask_values_other_than_0_or_1_are_runtime_errors() {
    let program = attention_program(HEADS, [N, H], [B, S]);
    let attention: ValueId = program.outputs[0];
    for (mask, sequence, key) in [
        ([1.0, 1.0, 0.5, 1.0, 1.0, 1.0], 0, 2),
        ([1.0, 1.0, 1.0, 1.0, f32::NAN, 1.0], 1, 1),
        ([1.0, 1.0, 1.0, 2.0, 1.0, 1.0], 1, 0),
    ] {
        assert_eq!(
            run_with_mask(&program, &mask),
            Err(RuntimeError::KeyMaskValue {
                value: attention,
                sequence,
                key,
            }),
            "{mask:?}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// The order of every sum and the scale, bit for bit (DESIGN.md § 4, coordinator decision D37).
// ---------------------------------------------------------------------------------------------

/// The heads of the order cases: two, so each head is `H/2` columns wide.
const TWO: usize = 2;

/// Cancelling magnitudes: in `f32` a sum of `1e8`, `1`, `-1e8` and `1` depends on the order of
/// its terms and on whether a product is rounded before it is added, so `v` and `G` scaled by it
/// separate ascending from descending sums and fused from unfused ones.
const CANCEL: [f32; 4] = [1.0e8, 1.0, -1.0e8, 1.0];

/// One attention call at `heads = 2`: `b` sequences of `s` rows of width `h`, the operands and
/// the adjoint `g` of the gradient ops.
struct Order {
    name: String,
    b: usize,
    s: usize,
    h: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    mask: Vec<f32>,
    g: Vec<f32>,
}

/// `a0 * b0` rounded, then each later product fused into the sum with `mul_add`, in the order
/// given (ADR-009); `0.0` for no terms. Written out here, not taken from the interpreter.
fn ascending(terms: impl IntoIterator<Item = (f32, f32)>) -> f32 {
    let mut terms = terms.into_iter();
    let Some((a, b)) = terms.next() else {
        return 0.0;
    };
    let mut acc = a * b;
    for (a, b) in terms {
        acc = a.mul_add(b, acc);
    }
    acc
}

impl Order {
    /// The case `name` with `q` and `k` drawn in `[-2, -0.2] ∪ [0.2, 2]`, and `v` and `G`
    /// drawn and scaled by [`CANCEL`]: `v` by key and column, `G` by query and column.
    fn new(name: &str, b: usize, s: usize, h: usize, mask: &[f32]) -> Self {
        let mut rng = Rng::new(seed_of(name));
        let n = b * s;
        let mut values = |scale: &dyn Fn(usize, usize) -> f32| -> Vec<f32> {
            (0..n * h)
                .map(|e| rng.value() * scale(e / h, e % h))
                .collect()
        };
        let q = values(&|_, _| 2.0);
        let k = values(&|_, _| 2.0);
        let v = values(&|j, c| CANCEL[(j % s + c) % 4]);
        let g = values(&|i, c| CANCEL[(i % s + 3 * c) % 4]);
        Self {
            name: name.to_owned(),
            b,
            s,
            h,
            q,
            k,
            v,
            mask: mask.to_vec(),
            g,
        }
    }

    fn n(&self) -> usize {
        self.b * self.s
    }

    /// `1 / sqrt(H/2)` in `f32`: the square root rounded, then the division rounded.
    fn scale(&self) -> f32 {
        1.0f32 / ((self.h / TWO) as f32).sqrt()
    }

    fn columns(&self, head: usize) -> std::ops::Range<usize> {
        let d = self.h / TWO;
        head * d..(head + 1) * d
    }

    /// The rows of the sequence of row `i`.
    fn sequence(&self, i: usize) -> std::ops::Range<usize> {
        let start = i / self.s * self.s;
        start..start + self.s
    }

    /// The unmasked keys of the sequence of row `i`, ascending.
    fn keys(&self, i: usize) -> Vec<usize> {
        self.sequence(i).filter(|&j| self.mask[j] == 1.0).collect()
    }

    /// The ascending fused dot product of `q[i]` and `k[j]` over the head's columns, then times
    /// the scale.
    fn score(&self, i: usize, j: usize, head: usize) -> f32 {
        let h = self.h;
        ascending(
            self.columns(head)
                .map(|c| (self.q[i * h + c], self.k[j * h + c])),
        ) * self.scale()
    }

    /// The probability of each unmasked key of query `i` in head `head`: the maximum in
    /// ascending key order, the first replaced by each later one that is greater, `exp` of each
    /// score less it, their sum in ascending key order, and each divided by the sum.
    fn probabilities(&self, i: usize, head: usize) -> Vec<(usize, f32)> {
        let keys = self.keys(i);
        let scores: Vec<f32> = keys.iter().map(|&j| self.score(i, j, head)).collect();
        let mut max = scores[0];
        for &x in &scores[1..] {
            if x > max {
                max = x;
            }
        }
        let e: Vec<f32> = scores.iter().map(|&x| (x - max).exp()).collect();
        let mut total = e[0];
        for &x in &e[1..] {
            total += x;
        }
        keys.into_iter()
            .zip(e)
            .map(|(j, e)| (j, e / total))
            .collect()
    }

    /// The probability of key `j` for query `i` in head `head`; `j` is unmasked.
    fn probability(&self, i: usize, j: usize, head: usize) -> f32 {
        let found = self
            .probabilities(i, head)
            .into_iter()
            .find(|&(u, _)| u == j);
        found.expect("an unmasked key").1
    }

    /// `dS[i, j] = P (dP - d)` for each unmasked key `j` of query `i` in head `head`, with
    /// `dP[i, j] = G[i] . v[j]` over the head's columns and `d = sum_j P dP` in ascending key
    /// order, both fused after the first product.
    fn score_grads(&self, i: usize, head: usize) -> Vec<(usize, f32)> {
        let h = self.h;
        let dp = |j: usize| {
            ascending(
                self.columns(head)
                    .map(|c| (self.g[i * h + c], self.v[j * h + c])),
            )
        };
        let p = self.probabilities(i, head);
        let d = ascending(p.iter().map(|&(j, pj)| (pj, dp(j))));
        p.into_iter().map(|(j, pj)| (j, pj * (dp(j) - d))).collect()
    }

    /// The forward: the sum over unmasked keys in ascending order of `P v[j]`, fused after the
    /// first product.
    fn forward(&self) -> Vec<f32> {
        let h = self.h;
        let mut out = vec![0.0f32; self.n() * h];
        for i in 0..self.n() {
            for head in 0..TWO {
                let p = self.probabilities(i, head);
                for c in self.columns(head) {
                    out[i * h + c] = ascending(p.iter().map(|&(j, pj)| (pj, self.v[j * h + c])));
                }
            }
        }
        out
    }

    /// `dQ[i] = scale * sum_j dS[i, j] k[j]` over unmasked keys, ascending, fused after the first.
    fn grad_q(&self) -> Vec<f32> {
        let h = self.h;
        let mut out = vec![0.0f32; self.n() * h];
        for i in 0..self.n() {
            for head in 0..TWO {
                let ds = self.score_grads(i, head);
                for c in self.columns(head) {
                    let sum = ascending(ds.iter().map(|&(j, x)| (x, self.k[j * h + c])));
                    out[i * h + c] = sum * self.scale();
                }
            }
        }
        out
    }

    /// `dK[j] = scale * sum_i dS[i, j] q[i]` over the queries of the sequence, ascending, fused
    /// after the first; `+0` in the row of a masked key.
    fn grad_k(&self) -> Vec<f32> {
        let h = self.h;
        let mut out = vec![0.0f32; self.n() * h];
        for j in (0..self.n()).filter(|&j| self.mask[j] == 1.0) {
            for head in 0..TWO {
                let ds: Vec<f32> = self
                    .sequence(j)
                    .map(|i| {
                        let found = self.score_grads(i, head).into_iter().find(|&(u, _)| u == j);
                        found.expect("an unmasked key").1
                    })
                    .collect();
                for c in self.columns(head) {
                    let terms = self
                        .sequence(j)
                        .zip(&ds)
                        .map(|(i, &x)| (x, self.q[i * h + c]));
                    out[j * h + c] = ascending(terms) * self.scale();
                }
            }
        }
        out
    }

    /// `dV[j] = sum_i P[i, j] G[i]` over the queries of the sequence, ascending, fused after the
    /// first; `+0` in the row of a masked key.
    fn grad_v(&self) -> Vec<f32> {
        let h = self.h;
        let mut out = vec![0.0f32; self.n() * h];
        for j in (0..self.n()).filter(|&j| self.mask[j] == 1.0) {
            for head in 0..TWO {
                for c in self.columns(head) {
                    let terms = self
                        .sequence(j)
                        .map(|i| (self.probability(i, j, head), self.g[i * h + c]));
                    out[j * h + c] = ascending(terms);
                }
            }
        }
        out
    }

    /// The interpreter's `f32` result of `attention` at `heads = 2` (`grad` `None`) or of the
    /// gradient op `grad` over `q, k, v, key_mask, G`.
    fn run(&self, grad: Option<TensorOp>) -> Vec<f32> {
        let (n, h) = (self.n() as u64, self.h as u64);
        let mut b = TensorProgramBuilder::new();
        let q = input(&mut b, "q", &[n, h]);
        let k = input(&mut b, "k", &[n, h]);
        let v = input(&mut b, "v", &[n, h]);
        let m = input(&mut b, "key_mask", &[self.b as u64, self.s as u64]);
        let y = match &grad {
            None => b.attention(q, k, v, m, TWO as u64).unwrap(),
            Some(op) => {
                let g = input(&mut b, "g", &[n, h]);
                b.push(op.clone(), &[q, k, v, m, g]).unwrap()
            }
        };
        b.output(y).unwrap();
        let program = b.finish();
        let shape: &[u64] = &[n, h];
        let mask_shape: &[u64] = &[self.b as u64, self.s as u64];
        let mut values: Vec<(&str, &[u64], &[f32])> = vec![
            ("q", shape, &self.q),
            ("k", shape, &self.k),
            ("v", shape, &self.v),
            ("key_mask", mask_shape, &self.mask),
        ];
        if grad.is_some() {
            values.push(("g", shape, &self.g));
        }
        let inputs = host(&values, |x| x);
        let mut out = evaluate(&program, &inputs, &BTreeMap::new()).unwrap();
        out.remove(0).data().to_vec()
    }
}

/// The order cases, all at `heads = 2`, with masked keys first, in the middle and last in a
/// sequence: head widths 2, 3, 5, 6 and 7. At width 6, `1 / sqrt(H/2)` in `f32` differs from
/// `sqrt(2/H)` and from `1 / sqrt(H/2)` computed in `f64` and rounded, and a scale computed either
/// way changes the bits of the forward and of every gradient op (measured by planting each).
fn order_cases() -> Vec<Order> {
    let one = 1.0f32;
    let cases: [(&str, usize, usize, usize, Vec<f32>); 5] = [
        ("h4", 2, 4, 4, vec![one, one, 0.0, one, one, one, one, one]),
        ("h6", 2, 4, 6, vec![0.0, one, one, one, one, one, one, 0.0]),
        ("h10", 1, 5, 10, vec![one, one, one, 0.0, one]),
        (
            "h12",
            2,
            4,
            12,
            vec![one, 0.0, one, one, one, one, one, one],
        ),
        ("h14", 2, 3, 14, vec![one, one, one, one, 0.0, one]),
    ];
    cases
        .into_iter()
        .map(|(name, b, s, h, mask)| Order::new(&format!("attention_order_{name}"), b, s, h, &mask))
        .collect()
}

/// `got` equals `want` bit for bit; the message names the first element that differs.
fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let differ: Vec<String> = got
        .iter()
        .zip(want)
        .enumerate()
        .filter(|(_, (g, w))| g.to_bits() != w.to_bits())
        .map(|(e, (g, w))| format!("[{e}] {g:e} vs {w:e}"))
        .collect();
    assert!(
        differ.is_empty(),
        "{what}: {} of {} elements differ from the ascending reference: {}",
        differ.len(),
        got.len(),
        differ
            .iter()
            .take(4)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    );
}

/// D37 at `heads = 2`: the forward equals, bit for bit, ascending fused dot products times
/// `1 / sqrt(H/2)`, the maximum and the `exp` sum over the unmasked keys in ascending order, and
/// the output summed over the unmasked keys in ascending order, fused after the first product.
#[test]
fn attention_order_forward_bit_for_bit_at_two_heads() {
    for case in order_cases() {
        assert_bits(&case.name, &case.run(None), &case.forward());
    }
}

/// D37 at `heads = 2`: `AttentionGradQ` equals, bit for bit, the reference whose row sum
/// `sum_j P dP` and `sum_j dS k[j]` run over the unmasked keys in ascending order.
#[test]
fn attention_order_grad_q_bit_for_bit_at_two_heads() {
    for case in order_cases() {
        let got = case.run(Some(TensorOp::AttentionGradQ { heads: TWO as u64 }));
        assert_bits(&case.name, &got, &case.grad_q());
    }
}

/// D37 at `heads = 2`: `AttentionGradK` equals, bit for bit, the reference whose sum over the
/// queries runs in ascending order.
#[test]
fn attention_order_grad_k_bit_for_bit_at_two_heads() {
    for case in order_cases() {
        let got = case.run(Some(TensorOp::AttentionGradK { heads: TWO as u64 }));
        assert_bits(&case.name, &got, &case.grad_k());
    }
}

/// D37 at `heads = 2`: `AttentionGradV` equals, bit for bit, the reference whose sum over the
/// queries runs in ascending order.
#[test]
fn attention_order_grad_v_bit_for_bit_at_two_heads() {
    for case in order_cases() {
        let got = case.run(Some(TensorOp::AttentionGradV { heads: TWO as u64 }));
        assert_bits(&case.name, &got, &case.grad_v());
    }
}
