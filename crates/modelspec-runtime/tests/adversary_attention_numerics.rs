//! Adversary pass 1 on story:bert-attention, on the CPU interpreter: the masked-key arithmetic
//! against a hand-written reference of coordinator decision D37 and DESIGN.md § 4 "Attention.",
//! the gradient ops at their edges against central differences of `sum(Y * G)` in `f64`, the
//! summation order of the forward and of each gradient op bit for bit, the head and sequence
//! slicing, the key mask check, and the autodiff wiring of an attention output used twice and of
//! one value as `q`, `k` and `v`.
//!
//! The reference here is written from the documents, not from the interpreter: every dot product
//! over a head's columns, every sum over keys and every sum over queries runs in ascending index
//! order, its first product rounded and each later one fused (ADR-009); the scale is
//! `1 / sqrt(H / A)` in `f32`, applied after the dot product; the maximum is over the unmasked
//! keys, the first replaced by each later one that is greater.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::collections::BTreeMap;
use std::ops::Range;

use modelspec_core::autodiff::differentiate;
use modelspec_core::ir::tensor::{TensorOp, TensorProgram, TensorProgramBuilder, ValueId};
use modelspec_runtime::cpu::{Element, HostTensor, RuntimeError, evaluate};

#[allow(dead_code)]
#[path = "common/finite_difference.rs"]
mod finite_difference;

use finite_difference::{
    H as STEP, Ids, MIN_SCALE, Point, Rng, TOL_F32, TOL_F64, input, loss_at, mse_to_target, param,
    relative_error, run, seed_of,
};

// ---------------------------------------------------------------------------------------------
// The case, the program and the interpreter.
// ---------------------------------------------------------------------------------------------

/// One attention call: `b` sequences of `s` rows of width `h`, `a` heads, and its operands.
#[derive(Clone, Debug)]
struct Case {
    b: usize,
    s: usize,
    h: usize,
    a: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    mask: Vec<f32>,
}

impl Case {
    /// `q`, `k`, `v` drawn from the seed of `name`, each value in `[-1, -0.1] ∪ [0.1, 1]`.
    fn draw(name: &str, b: usize, s: usize, h: usize, a: usize, mask: &[f32]) -> Self {
        assert_eq!(mask.len(), b * s, "{name}: mask length");
        let mut rng = Rng::new(seed_of(name));
        let len = b * s * h;
        let mut draw = || (0..len).map(|_| rng.value()).collect::<Vec<f32>>();
        let (q, k, v) = (draw(), draw(), draw());
        Self {
            b,
            s,
            h,
            a,
            q,
            k,
            v,
            mask: mask.to_vec(),
        }
    }

    /// As [`Case::draw`], each value also scaled by one of `1/8, 1/2, 1, 2, 8`, so that sums in
    /// another order round differently.
    fn wide(name: &str, b: usize, s: usize, h: usize, a: usize, mask: &[f32]) -> Self {
        let mut case = Self::draw(name, b, s, h, a, mask);
        let mut rng = Rng::new(seed_of(name) ^ 0x5bd1_e995);
        const SCALES: [f32; 5] = [0.125, 0.5, 1.0, 2.0, 8.0];
        for x in case
            .q
            .iter_mut()
            .chain(case.k.iter_mut())
            .chain(case.v.iter_mut())
        {
            *x *= SCALES[(rng.next_u64() % 5) as usize];
        }
        case
    }

    fn n(&self) -> usize {
        self.b * self.s
    }

    fn d(&self) -> usize {
        self.h / self.a
    }

    /// A `G` of `[N, H]` drawn from the seed of `name`.
    fn g(&self, name: &str) -> Vec<f32> {
        let mut rng = Rng::new(seed_of(name) ^ 0x9e37);
        (0..self.n() * self.h).map(|_| rng.value()).collect()
    }

    /// The columns `columns` of every row of `x`, as an `[N, columns.len()]` matrix.
    fn slice(&self, x: &[f32], columns: Range<usize>) -> Vec<f32> {
        (0..self.n())
            .flat_map(|i| x[i * self.h + columns.start..i * self.h + columns.end].to_vec())
            .collect()
    }

    /// The case of head `head` alone: `heads = 1` over its columns.
    fn head(&self, head: usize) -> Case {
        let columns = head * self.d()..(head + 1) * self.d();
        Case {
            b: self.b,
            s: self.s,
            h: self.d(),
            a: 1,
            q: self.slice(&self.q, columns.clone()),
            k: self.slice(&self.k, columns.clone()),
            v: self.slice(&self.v, columns),
            mask: self.mask.clone(),
        }
    }
}

/// `attention` or one of its gradient ops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Forward,
    GradQ,
    GradK,
    GradV,
}

const GRADS: [Op; 3] = [Op::GradQ, Op::GradK, Op::GradV];

impl Op {
    fn tensor_op(self, heads: u64) -> TensorOp {
        match self {
            Op::Forward => TensorOp::Attention { heads },
            Op::GradQ => TensorOp::AttentionGradQ { heads },
            Op::GradK => TensorOp::AttentionGradK { heads },
            Op::GradV => TensorOp::AttentionGradV { heads },
        }
    }
}

/// The program of `op` over the inputs `q`, `k`, `v`, `key_mask` and, for a gradient op, `g`;
/// its one output is the op's result.
fn program(case: &Case, op: Op) -> TensorProgram {
    let (n, h) = (case.n() as u64, case.h as u64);
    let mut b = TensorProgramBuilder::new();
    let q = input(&mut b, "q", &[n, h]);
    let k = input(&mut b, "k", &[n, h]);
    let v = input(&mut b, "v", &[n, h]);
    let m = input(&mut b, "key_mask", &[case.b as u64, case.s as u64]);
    let y = match op {
        Op::Forward => b.attention(q, k, v, m, case.a as u64).unwrap(),
        _ => {
            let g = input(&mut b, "g", &[n, h]);
            b.push(op.tensor_op(case.a as u64), &[q, k, v, m, g])
                .unwrap()
        }
    };
    b.output(y).unwrap();
    b.finish()
}

/// The value the op of [`program`] defines, which a runtime error names.
fn op_value(case: &Case, op: Op) -> ValueId {
    program(case, op).outputs[0]
}

/// The interpreter's result of `op` in `T`.
fn eval<T: Element>(
    case: &Case,
    op: Op,
    [q, k, v]: [Vec<T>; 3],
    mask: Vec<T>,
    g: Vec<T>,
) -> Result<Vec<T>, RuntimeError> {
    let shape = vec![case.n() as u64, case.h as u64];
    let mut inputs = BTreeMap::new();
    for (name, data) in [("q", q), ("k", k), ("v", v)] {
        inputs.insert(
            name.to_owned(),
            HostTensor::new(shape.clone(), data).unwrap(),
        );
    }
    inputs.insert(
        "key_mask".to_owned(),
        HostTensor::new(vec![case.b as u64, case.s as u64], mask).unwrap(),
    );
    if op != Op::Forward {
        inputs.insert("g".to_owned(), HostTensor::new(shape, g).unwrap());
    }
    let out = evaluate(&program(case, op), &inputs, &BTreeMap::new())?;
    Ok(out[0].data().to_vec())
}

fn run32(case: &Case, op: Op, g: &[f32]) -> Result<Vec<f32>, RuntimeError> {
    eval(
        case,
        op,
        [case.q.clone(), case.k.clone(), case.v.clone()],
        case.mask.clone(),
        g.to_vec(),
    )
}

fn widen(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| f64::from(v)).collect()
}

fn run64(case: &Case, op: Op, g: &[f32]) -> Vec<f64> {
    eval(
        case,
        op,
        [widen(&case.q), widen(&case.k), widen(&case.v)],
        widen(&case.mask),
        widen(g),
    )
    .unwrap()
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// `got` equals `want` bit for bit; the message lists the first elements that differ.
fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    let differing: Vec<String> = got
        .iter()
        .zip(want)
        .enumerate()
        .filter(|(_, (g, w))| g.to_bits() != w.to_bits())
        .map(|(i, (g, w))| {
            format!(
                "[{i}] {g:e} ({:#010x}) vs {w:e} ({:#010x})",
                g.to_bits(),
                w.to_bits()
            )
        })
        .collect();
    assert!(
        differing.is_empty(),
        "{what}: {} of {} elements differ: {}",
        differing.len(),
        got.len(),
        differing
            .iter()
            .take(6)
            .cloned()
            .collect::<Vec<_>>()
            .join("; ")
    );
}

// ---------------------------------------------------------------------------------------------
// The reference, from D37 and DESIGN.md § 4.
// ---------------------------------------------------------------------------------------------

/// `a0 * b0`, then each later product fused into the sum, in the order given; `0.0` when empty.
fn dot(terms: impl Iterator<Item = (f32, f32)>) -> f32 {
    let mut acc: Option<f32> = None;
    for (a, b) in terms {
        acc = Some(match acc {
            None => a * b,
            Some(s) => a.mul_add(b, s),
        });
    }
    acc.unwrap_or(0.0)
}

/// The documents' attention over one case. `zero_weight` runs the forward's sum over every key
/// of the sequence with the probability `0.0` at a masked key, as D37 words it; without it the
/// masked keys are left out of the sum.
struct Reference<'c> {
    c: &'c Case,
    scale: f32,
    zero_weight: bool,
}

impl<'c> Reference<'c> {
    fn new(c: &'c Case) -> Self {
        Self {
            c,
            scale: 1.0f32 / ((c.h / c.a) as f32).sqrt(),
            zero_weight: false,
        }
    }

    fn zero_weighted(c: &'c Case) -> Self {
        Self {
            zero_weight: true,
            ..Self::new(c)
        }
    }

    fn cols(&self, head: usize) -> Range<usize> {
        head * self.c.d()..(head + 1) * self.c.d()
    }

    fn seq(&self, i: usize) -> Range<usize> {
        let start = i / self.c.s * self.c.s;
        start..start + self.c.s
    }

    fn unmasked(&self, j: usize) -> bool {
        self.c.mask[j] == 1.0
    }

    fn score(&self, i: usize, j: usize, head: usize) -> f32 {
        let h = self.c.h;
        dot(self
            .cols(head)
            .map(|c| (self.c.q[i * h + c], self.c.k[j * h + c])))
            * self.scale
    }

    /// The probability of every key of the sequence of `i`, `0.0` at a masked key.
    fn probs(&self, i: usize, head: usize) -> Vec<(usize, f32)> {
        let keys: Vec<usize> = self.seq(i).filter(|&j| self.unmasked(j)).collect();
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
        self.seq(i)
            .map(|j| match keys.iter().position(|&u| u == j) {
                Some(p) => (j, e[p] / total),
                None => (j, 0.0),
            })
            .collect()
    }

    /// The probabilities over the unmasked keys only.
    fn kept(&self, i: usize, head: usize) -> Vec<(usize, f32)> {
        self.probs(i, head)
            .into_iter()
            .filter(|&(j, _)| self.unmasked(j))
            .collect()
    }

    fn forward(&self) -> Vec<f32> {
        let (n, h) = (self.c.n(), self.c.h);
        let mut out = vec![0.0f32; n * h];
        for i in 0..n {
            for head in 0..self.c.a {
                let terms = if self.zero_weight {
                    self.probs(i, head)
                } else {
                    self.kept(i, head)
                };
                for c in self.cols(head) {
                    out[i * h + c] = dot(terms.iter().map(|&(j, p)| (p, self.c.v[j * h + c])));
                }
            }
        }
        out
    }

    /// `dP[i, j]`: row `i` of `g` dot row `j` of `v` over the head's columns.
    fn dp(&self, g: &[f32], i: usize, j: usize, head: usize) -> f32 {
        let h = self.c.h;
        dot(self.cols(head).map(|c| (g[i * h + c], self.c.v[j * h + c])))
    }

    /// `dS[i, j] = P (dP - sum_j P dP)` over the unmasked keys of `i`, by key.
    fn ds(&self, g: &[f32], i: usize, head: usize) -> Vec<(usize, f32)> {
        let p = self.kept(i, head);
        let d = dot(p.iter().map(|&(j, pj)| (pj, self.dp(g, i, j, head))));
        p.iter()
            .map(|&(j, pj)| (j, pj * (self.dp(g, i, j, head) - d)))
            .collect()
    }

    fn grad_q(&self, g: &[f32]) -> Vec<f32> {
        let (n, h) = (self.c.n(), self.c.h);
        let mut out = vec![0.0f32; n * h];
        for i in 0..n {
            for head in 0..self.c.a {
                let ds = self.ds(g, i, head);
                for c in self.cols(head) {
                    out[i * h + c] =
                        dot(ds.iter().map(|&(j, x)| (x, self.c.k[j * h + c]))) * self.scale;
                }
            }
        }
        out
    }

    fn grad_k(&self, g: &[f32]) -> Vec<f32> {
        let (n, h, d) = (self.c.n(), self.c.h, self.c.d());
        let ds: Vec<Vec<Vec<(usize, f32)>>> = (0..n)
            .map(|i| (0..self.c.a).map(|head| self.ds(g, i, head)).collect())
            .collect();
        let mut out = vec![0.0f32; n * h];
        for j in (0..n).filter(|&j| self.unmasked(j)) {
            for c in 0..h {
                let head = c / d;
                let at = |i: usize| ds[i][head].iter().find(|&&(u, _)| u == j).unwrap().1;
                out[j * h + c] =
                    dot(self.seq(j).map(|i| (at(i), self.c.q[i * h + c]))) * self.scale;
            }
        }
        out
    }

    fn grad_v(&self, g: &[f32]) -> Vec<f32> {
        let (n, h, d) = (self.c.n(), self.c.h, self.c.d());
        let mut out = vec![0.0f32; n * h];
        for j in (0..n).filter(|&j| self.unmasked(j)) {
            for c in 0..h {
                let head = c / d;
                let at = |i: usize| self.probs(i, head)[j - self.seq(j).start].1;
                out[j * h + c] = dot(self.seq(j).map(|i| (at(i), g[i * h + c])));
            }
        }
        out
    }

    fn of(&self, op: Op, g: &[f32]) -> Vec<f32> {
        match op {
            Op::Forward => self.forward(),
            Op::GradQ => self.grad_q(g),
            Op::GradK => self.grad_k(g),
            Op::GradV => self.grad_v(g),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Attack 1: skipping a masked key against weighting it 0.0.
// ---------------------------------------------------------------------------------------------

/// Attack 1. On finite data with masked keys first, in the middle and last, one masked key whose
/// `k` row would give the row maximum and one whose `v` row is subnormal, the forward equals
/// both the reference that skips masked keys and the D37 wording that weights them `0.0`, bit
/// for bit.
#[test]
fn adversary_attention_a1_skip_equals_zero_weight_on_ordinary_finite_data() {
    let mask = [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
    let mut case = Case::wide("adversary_attention_a1_finite", 3, 3, 4, 2, &mask);
    for c in 0..4 {
        case.k[4 * 4 + c] = 50.0 * case.q[3 * 4 + c].signum();
        case.v[8 * 4 + c] = 1.0e-40;
    }
    let got = run32(&case, Op::Forward, &[]).unwrap();
    assert_bits("skip reference", &got, &Reference::new(&case).forward());
    assert_bits(
        "zero-weight reference",
        &got,
        &Reference::zero_weighted(&case).forward(),
    );
}

/// Attack 1, the DESIGN.md § 4 claim "skipping it adds the same bits for every finite `k` and
/// `v` (`s + 0 = s` and `fma(0, v, s) = s`)". With every unmasked `v` of a column `-0.0` and a
/// positive `v` in the masked row between them, the skipping sum stays `-0.0`, while the sum D37
/// words, `fma(0.0, 1.0, -0.0) = +0.0`, ends `+0.0`. Finding A1.
#[test]
#[ignore = "finding F1: DESIGN.md section 4 says skipping a masked key adds the same bits as weighting it 0.0 for every finite k and v; false for -0.0 sums"]
fn adversary_attention_a1_design_claim_same_bits_holds_for_negative_zero_v() {
    let mut case = Case::draw(
        "adversary_attention_a1_negative_zero",
        1,
        3,
        2,
        1,
        &[1.0, 0.0, 1.0],
    );
    case.v = vec![-0.0, -0.0, 1.0, 1.0, -0.0, -0.0];
    let got = run32(&case, Op::Forward, &[]).unwrap();
    let zero_weighted = Reference::zero_weighted(&case).forward();
    assert!(
        got.iter().zip(&zero_weighted).all(|(g, w)| g == w),
        "the values are equal"
    );
    assert_bits(
        "interpreter against the D37 zero-weighted sum",
        &got,
        &zero_weighted,
    );
}

/// Attack 1, what the interpreter does on the case above: `-0.0`, as the skipping reference.
#[test]
fn adversary_attention_a1_negative_zero_v_gives_negative_zero_as_skipping_does() {
    let mut case = Case::draw(
        "adversary_attention_a1_negative_zero",
        1,
        3,
        2,
        1,
        &[1.0, 0.0, 1.0],
    );
    case.v = vec![-0.0, -0.0, 1.0, 1.0, -0.0, -0.0];
    let got = run32(&case, Op::Forward, &[]).unwrap();
    assert_bits("skip reference", &got, &Reference::new(&case).forward());
    assert!(
        got.iter().all(|x| x.to_bits() == (-0.0f32).to_bits()),
        "{got:?}"
    );
    let zero_weighted = Reference::zero_weighted(&case).forward();
    assert!(
        zero_weighted.iter().all(|x| x.to_bits() == 0),
        "{zero_weighted:?}"
    );
}

/// Attack 1. An infinity or NaN in the masked row of `k` and `v` reaches no result: the forward
/// and every gradient op equal, bit for bit, the run with an ordinary masked row, and the
/// masked rows of `dK` and `dV` stay `+0.0`.
#[test]
fn adversary_attention_a1_inf_and_nan_in_a_masked_row_reach_no_result() {
    let mask = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let case = Case::draw("adversary_attention_a1_nonfinite", 2, 3, 4, 2, &mask);
    let g = case.g("adversary_attention_a1_nonfinite");
    let mut poisoned = case.clone();
    for (row, values) in [
        (2, [f32::INFINITY, f32::NAN, f32::NEG_INFINITY, f32::NAN]),
        (
            4,
            [f32::NAN, f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY],
        ),
    ] {
        for c in 0..4 {
            poisoned.k[row * 4 + c] = values[c];
            poisoned.v[row * 4 + c] = values[(c + 1) % 4];
        }
    }
    for op in [Op::Forward, Op::GradQ, Op::GradK, Op::GradV] {
        let clean = run32(&case, op, &g).unwrap();
        let dirty = run32(&poisoned, op, &g).unwrap();
        assert!(dirty.iter().all(|x| x.is_finite()), "{op:?}: {dirty:?}");
        assert_bits(&format!("{op:?}"), &dirty, &clean);
        if matches!(op, Op::GradK | Op::GradV) {
            for row in [2, 4] {
                assert!(
                    dirty[row * 4..row * 4 + 4].iter().all(|x| x.to_bits() == 0),
                    "{op:?} row {row}"
                );
            }
        }
    }
}

/// Attack 1. A NaN in an unmasked `k` row makes NaN the outputs of its sequence in the head of
/// its column and nothing else; an infinity in an unmasked `v` row reaches the same head of its
/// sequence's outputs and nothing else.
#[test]
fn adversary_attention_a1_nan_and_inf_in_an_unmasked_row_propagate_within_head_and_sequence() {
    let mask = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let case = Case::draw("adversary_attention_a1_unmasked", 2, 3, 4, 2, &mask);
    let mut nan_k = case.clone();
    nan_k.k[4 * 4 + 1] = f32::NAN;
    let got = run32(&nan_k, Op::Forward, &[]).unwrap();
    for i in 0..6 {
        for c in 0..4 {
            let x = got[i * 4 + c];
            let affected = i >= 3 && c < 2;
            assert_eq!(x.is_nan(), affected, "NaN in k: row {i} column {c}: {x}");
        }
    }
    let mut inf_v = case.clone();
    inf_v.v[4 + 3] = f32::INFINITY;
    let got = run32(&inf_v, Op::Forward, &[]).unwrap();
    for i in 0..6 {
        for c in 0..4 {
            let x = got[i * 4 + c];
            let affected = i < 3 && c == 3;
            assert_eq!(
                !x.is_finite(),
                affected,
                "inf in v: row {i} column {c}: {x}"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Attack 2: the gradient ops at their edges, against central differences of sum(Y * G).
// ---------------------------------------------------------------------------------------------

/// `sum_i Y_i G_i` in `f64` for the forward at `q, k, v`.
fn objective(case: &Case, xs: &[Vec<f64>; 3], g: &[f64]) -> f64 {
    let y = eval(case, Op::Forward, xs.clone(), widen(&case.mask), Vec::new()).unwrap();
    y.iter().zip(g).map(|(a, b)| a * b).sum()
}

/// Each gradient op in `f64` is within `TOL_F64` of the central difference of `sum(Y * G)`
/// (relative to the larger of the differences' maximum and 1), and in `f32` and `f64` the rows of
/// `dK` and `dV` at a masked key are `+0.0` bit for bit. Returns the `f32` gradients.
fn edge(name: &str, case: &Case, g: &[f32]) -> [Vec<f32>; 3] {
    let xs = [widen(&case.q), widen(&case.k), widen(&case.v)];
    let g64 = widen(g);
    let mut out: Vec<Vec<f32>> = Vec::new();
    for (which, op) in GRADS.into_iter().enumerate() {
        let analytic = run64(case, op, g);
        let numeric: Vec<f64> = (0..case.n() * case.h)
            .map(|e| {
                let at = |delta: f64| {
                    let mut moved = xs.clone();
                    moved[which][e] += delta;
                    objective(case, &moved, &g64)
                };
                (at(STEP) - at(-STEP)) / (2.0 * STEP)
            })
            .collect();
        let scale = numeric.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        let error = analytic
            .iter()
            .zip(&numeric)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
            / scale;
        assert!(
            error <= TOL_F64,
            "{name} {op:?}: error {error:e}\n{analytic:?}\n{numeric:?}"
        );
        let in_f32 = run32(case, op, g).unwrap();
        if matches!(op, Op::GradK | Op::GradV) {
            for j in (0..case.n()).filter(|&j| case.mask[j] == 0.0) {
                let h = case.h;
                assert!(
                    in_f32[j * h..j * h + h].iter().all(|x| x.to_bits() == 0),
                    "{name} {op:?} f32 masked row {j}: {:?}",
                    &in_f32[j * h..j * h + h]
                );
                assert!(
                    analytic[j * h..j * h + h].iter().all(|x| x.to_bits() == 0),
                    "{name} {op:?} f64 masked row {j}: {:?}",
                    &analytic[j * h..j * h + h]
                );
            }
        }
        out.push(in_f32);
    }
    let [q, k, v] = <[Vec<f32>; 3]>::try_from(out).unwrap();
    [q, k, v]
}

/// Attack 2. `S = 1`: the one key has probability exactly 1, so `dS` is 0, `dQ` and `dK` are 0
/// and `dV` is `G` bit for bit.
#[test]
fn adversary_attention_a2_one_key_per_sequence() {
    let case = Case::draw("adversary_attention_a2_s1", 2, 1, 4, 2, &[1.0, 1.0]);
    let g = case.g("adversary_attention_a2_s1");
    let [dq, dk, dv] = edge("S = 1", &case, &g);
    assert!(dq.iter().chain(&dk).all(|&x| x == 0.0), "{dq:?} {dk:?}");
    assert_eq!(bits(&dv), bits(&g), "dV is G");
}

/// Attack 2. `S = 2` with one key masked in each sequence: the gradients match central
/// differences and the masked rows are `+0.0`.
#[test]
fn adversary_attention_a2_two_keys_one_masked() {
    let case = Case::draw(
        "adversary_attention_a2_s2",
        2,
        2,
        4,
        2,
        &[1.0, 0.0, 0.0, 1.0],
    );
    let g = case.g("adversary_attention_a2_s2");
    let [dq, dk, _] = edge("S = 2", &case, &g);
    assert!(
        dq.iter().chain(&dk).all(|&x| x == 0.0),
        "one key left: dS = 0"
    );
}

/// Attack 2. `heads = H`, each head one column wide.
#[test]
fn adversary_attention_a2_heads_equal_width() {
    let case = Case::draw(
        "adversary_attention_a2_a_eq_h",
        2,
        3,
        4,
        4,
        &[1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    );
    let g = case.g("adversary_attention_a2_a_eq_h");
    edge("A = H", &case, &g);
}

/// Attack 2. `B = 1`.
#[test]
fn adversary_attention_a2_one_sequence() {
    let case = Case::draw(
        "adversary_attention_a2_b1",
        1,
        4,
        4,
        2,
        &[1.0, 1.0, 0.0, 1.0],
    );
    let g = case.g("adversary_attention_a2_b1");
    edge("B = 1", &case, &g);
}

/// Attack 2. One unmasked key among four masked: the masked rows of `dK` and `dV` are `+0.0`,
/// `dQ` and `dK` are 0, and the unmasked row of `dV` is the ascending sum of the rows of `G`.
#[test]
fn adversary_attention_a2_one_unmasked_key_among_many_masked() {
    let case = Case::draw(
        "adversary_attention_a2_lone",
        1,
        5,
        4,
        2,
        &[0.0, 0.0, 1.0, 0.0, 0.0],
    );
    let g = case.g("adversary_attention_a2_lone");
    let [dq, dk, dv] = edge("one unmasked", &case, &g);
    assert!(dq.iter().chain(&dk).all(|&x| x == 0.0));
    for c in 0..4 {
        let want = dot((0..5).map(|i| (1.0, g[i * 4 + c])));
        assert_eq!(dv[2 * 4 + c].to_bits(), want.to_bits(), "column {c}");
    }
}

/// Attack 2. `q` and `k` near 1e3: every `exp` but the winner's underflows, so `dS` is 0 for
/// every key and `dQ` and `dK` are 0, while `dV` still matches central differences in `f64`.
#[test]
fn adversary_attention_a2_saturated_scores() {
    let mut case = Case::draw(
        "adversary_attention_a2_saturated",
        2,
        3,
        4,
        2,
        &[1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    );
    for x in case.q.iter_mut().chain(case.k.iter_mut()) {
        *x *= 1.0e3;
    }
    let g = case.g("adversary_attention_a2_saturated");
    let [dq, dk, _] = edge("saturated", &case, &g);
    assert!(dq.iter().chain(&dk).all(|&x| x == 0.0), "{dq:?} {dk:?}");
}

/// Attack 2. Two keys of a sequence with equal `k` rows tie for the maximum of every query.
#[test]
fn adversary_attention_a2_tied_maximum() {
    let mut case = Case::draw(
        "adversary_attention_a2_tie",
        2,
        3,
        4,
        2,
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    );
    for c in 0..4 {
        case.k[4 + c] = 3.0 * case.q[c].signum();
        case.k[2 * 4 + c] = case.k[4 + c];
    }
    let g = case.g("adversary_attention_a2_tie");
    edge("tie", &case, &g);
}

/// Attack 2. `v = 0`: `dP` is 0, so `dQ` and `dK` are 0.
#[test]
fn adversary_attention_a2_zero_v() {
    let mut case = Case::draw(
        "adversary_attention_a2_v0",
        2,
        3,
        4,
        2,
        &[1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    );
    case.v = vec![0.0; 24];
    let g = case.g("adversary_attention_a2_v0");
    let [dq, dk, _] = edge("v = 0", &case, &g);
    assert!(dq.iter().chain(&dk).all(|&x| x == 0.0));
}

/// Attack 2. `G = 0`: every gradient is 0, and `dV` is `+0.0` everywhere.
#[test]
fn adversary_attention_a2_zero_g() {
    let case = Case::draw(
        "adversary_attention_a2_g0",
        2,
        3,
        4,
        2,
        &[1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    );
    let g = vec![0.0; 24];
    let [dq, dk, dv] = edge("G = 0", &case, &g);
    assert!(dq.iter().chain(&dk).all(|&x| x == 0.0));
    assert!(dv.iter().all(|x| x.to_bits() == 0), "{dv:?}");
}

// ---------------------------------------------------------------------------------------------
// Attack 3: summation order, bit for bit.
// ---------------------------------------------------------------------------------------------

/// The cases of attack 3: head widths 2, 3, 5, 1 and 7, partial masks, and a case whose `v`, `k`
/// and `G` follow the `[1e4, 1, -1e4, 1]` cancellation pattern by key, by query and by column.
fn order_cases() -> Vec<(String, Case, Vec<f32>)> {
    let mut cases = Vec::new();
    for (name, b, s, h, a, mask) in [
        ("w2", 2, 3, 4, 2, vec![1.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        ("w3", 1, 4, 6, 2, vec![1.0, 1.0, 0.0, 1.0]),
        ("w5", 2, 3, 10, 2, vec![0.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        ("w1", 2, 2, 3, 3, vec![1.0, 1.0, 1.0, 0.0]),
        ("w7", 1, 5, 7, 1, vec![1.0, 1.0, 1.0, 0.0, 1.0]),
        (
            "w6",
            2,
            4,
            12,
            2,
            vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        ),
    ] {
        let name = format!("adversary_attention_a3_{name}");
        let case = Case::wide(&name, b, s, h, a, &mask);
        let g = case.g(&name);
        cases.push((name, case, g));
    }
    let name = "adversary_attention_a3_cancel".to_owned();
    let mut case = Case::draw(&name, 2, 4, 4, 2, &[1.0; 8]);
    const PATTERN: [f32; 4] = [1.0e4, 1.0, -1.0e4, 1.0];
    for i in 0..8 {
        for c in 0..4 {
            case.q[i * 4 + c] *= 0.01;
            case.k[i * 4 + c] *= PATTERN[(i + c) % 4] * 0.01;
            case.v[i * 4 + c] *= PATTERN[i % 4];
        }
    }
    let mut g = case.g(&name);
    for i in 0..8 {
        for c in 0..4 {
            g[i * 4 + c] *= PATTERN[(i + 3 * c) % 4];
        }
    }
    cases.push((name, case, g));
    cases
}

/// Attack 3. The forward equals the reference bit for bit: ascending dot products over the
/// head's columns fused after the first, times `1 / sqrt(H/A)` in `f32` after the dot product,
/// the maximum over unmasked keys, the ascending sum of `exp`, and the ascending fused sum over
/// keys.
#[test]
fn adversary_attention_a3_forward_order_bit_for_bit() {
    for (name, case, g) in order_cases() {
        let got = run32(&case, Op::Forward, &g).unwrap();
        assert_bits(&name, &got, &Reference::new(&case).forward());
    }
}

/// Attack 3. `AttentionGradQ` equals the reference bit for bit.
#[test]
fn adversary_attention_a3_grad_q_order_bit_for_bit() {
    for (name, case, g) in order_cases() {
        let got = run32(&case, Op::GradQ, &g).unwrap();
        assert_bits(&name, &got, &Reference::new(&case).of(Op::GradQ, &g));
    }
}

/// Attack 3. `AttentionGradK` equals the reference bit for bit.
#[test]
fn adversary_attention_a3_grad_k_order_bit_for_bit() {
    for (name, case, g) in order_cases() {
        let got = run32(&case, Op::GradK, &g).unwrap();
        assert_bits(&name, &got, &Reference::new(&case).of(Op::GradK, &g));
    }
}

/// Attack 3. `AttentionGradV` equals the reference bit for bit.
#[test]
fn adversary_attention_a3_grad_v_order_bit_for_bit() {
    for (name, case, g) in order_cases() {
        let got = run32(&case, Op::GradV, &g).unwrap();
        assert_bits(&name, &got, &Reference::new(&case).of(Op::GradV, &g));
    }
}

// ---------------------------------------------------------------------------------------------
// Attack 4: heads.
// ---------------------------------------------------------------------------------------------

/// Attack 4. With two heads over different data, the columns of each head of the forward and
/// of every gradient op equal, bit for bit, `heads = 1` over that head's columns of `q`, `k`, `v`
/// and `G`.
#[test]
fn adversary_attention_a4_each_head_is_one_head_attention_over_its_columns() {
    let mask = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let mut case = Case::wide("adversary_attention_a4_heads", 2, 3, 6, 2, &mask);
    for i in 0..6 {
        for c in 3..6 {
            case.v[i * 6 + c] *= 4.0;
            case.k[i * 6 + c] = -case.k[i * 6 + c];
        }
    }
    let g = case.g("adversary_attention_a4_heads");
    for op in [Op::Forward, Op::GradQ, Op::GradK, Op::GradV] {
        let got = run32(&case, op, &g).unwrap();
        for head in 0..2 {
            let columns = head * 3..head * 3 + 3;
            let one = case.head(head);
            let want = run32(&one, op, &case.slice(&g, columns.clone())).unwrap();
            assert_bits(
                &format!("{op:?} head {head}"),
                &case.slice(&got, columns),
                &want,
            );
        }
    }
}

/// `softmax(q k^T * scale) v` over `[S, d]` operands with the landed operators.
fn softmax_matmul(s: u64, d: u64, scale: f32) -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let q = input(&mut b, "q", &[s, d]);
    let k = input(&mut b, "k", &[s, d]);
    let v = input(&mut b, "v", &[s, d]);
    let kt = b.transpose(k).unwrap();
    let scores = b.matmul(q, kt).unwrap();
    let scale = b.scalar(scale);
    let scaled = b.mul(scores, scale).unwrap();
    let p = b.softmax(scaled).unwrap();
    let y = b.matmul(p, v).unwrap();
    b.output(y).unwrap();
    b.finish()
}

/// Attack 4. With `heads = 2` and a mask of ones, every head of every sequence is within `1e-6`
/// of `softmax(q_h k_h^T / sqrt(H/2)) v_h` computed with `softmax` and `matmul`, which the
/// unit's one-head reference cannot say for `A > 1`.
#[test]
fn adversary_attention_a4_two_heads_match_softmax_matmul_per_sequence_and_head() {
    let (b, s, h, a) = (2usize, 3usize, 4usize, 2usize);
    let case = Case::draw("adversary_attention_a4_two_heads", b, s, h, a, &[1.0; 6]);
    let got = run32(&case, Op::Forward, &[]).unwrap();
    let d = h / a;
    let reference = softmax_matmul(s as u64, d as u64, 1.0 / (d as f32).sqrt());
    let mut worst = 0.0f64;
    for seq in 0..b {
        for head in 0..a {
            let part = |x: &[f32]| -> Vec<f32> {
                (seq * s..seq * s + s)
                    .flat_map(|i| x[i * h + head * d..i * h + head * d + d].to_vec())
                    .collect()
            };
            let mut inputs = BTreeMap::new();
            for (name, x) in [("q", &case.q), ("k", &case.k), ("v", &case.v)] {
                inputs.insert(
                    name.to_owned(),
                    HostTensor::new(vec![s as u64, d as u64], part(x)).unwrap(),
                );
            }
            let want = evaluate(&reference, &inputs, &BTreeMap::new()).unwrap();
            for (g, w) in part(&got).iter().zip(want[0].data()) {
                worst = worst.max((f64::from(*g) - f64::from(*w)).abs());
            }
        }
    }
    assert!(worst <= 1e-6, "largest difference {worst:e}");
}

// ---------------------------------------------------------------------------------------------
// Attack 5: sequences.
// ---------------------------------------------------------------------------------------------

/// Attack 5. Row `n` belongs to sequence `n / S`: changing every operand and the mask of one
/// sequence leaves the rows of every other sequence unchanged, bit for bit, in the forward and
/// in each gradient op; for `B = 2, S = 3` and `B = 3, S = 2`.
#[test]
fn adversary_attention_a5_rows_attend_within_their_own_sequence_only() {
    for (b, s, mask, other_mask) in [
        (
            2,
            3,
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        ),
        (
            3,
            2,
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        ),
    ] {
        let name = format!("adversary_attention_a5_b{b}_s{s}");
        let case = Case::draw(&name, b, s, 4, 2, &mask);
        let g = case.g(&name);
        let changed_seq = 1;
        let rows = changed_seq * s..changed_seq * s + s;
        let mut other = case.clone();
        other.mask = other_mask;
        let mut other_g = g.clone();
        for i in rows.clone() {
            for c in 0..4 {
                other.q[i * 4 + c] *= -2.0;
                other.k[i * 4 + c] = 1.5 * other.k[i * 4 + c] + 0.25;
                other.v[i * 4 + c] = 3.0 - other.v[i * 4 + c];
                other_g[i * 4 + c] = -other_g[i * 4 + c];
            }
        }
        for op in [Op::Forward, Op::GradQ, Op::GradK, Op::GradV] {
            let x = run32(&case, op, &g).unwrap();
            let y = run32(&other, op, &other_g).unwrap();
            for i in (0..b * s).filter(|i| !rows.contains(i)) {
                assert_eq!(
                    bits(&x[i * 4..i * 4 + 4]),
                    bits(&y[i * 4..i * 4 + 4]),
                    "B = {b}, S = {s}, {op:?}: row {i} changed with sequence {changed_seq}"
                );
            }
            let changed = rows
                .clone()
                .any(|i| bits(&x[i * 4..i * 4 + 4]) != bits(&y[i * 4..i * 4 + 4]));
            assert!(
                changed,
                "B = {b}, S = {s}, {op:?}: the changed sequence moved"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Attack 9: the key mask check.
// ---------------------------------------------------------------------------------------------

/// Attack 9. `-0.0` in the mask is accepted and masks its key exactly as `0.0` does, for the
/// forward and every gradient op.
#[test]
fn adversary_attention_a9_negative_zero_in_the_mask_masks_like_zero() {
    let case = Case::draw(
        "adversary_attention_a9_neg0",
        2,
        3,
        4,
        2,
        &[1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
    );
    let mut negative = case.clone();
    negative.mask = vec![1.0, -0.0, 1.0, 1.0, 1.0, -0.0];
    let g = case.g("adversary_attention_a9_neg0");
    for op in [Op::Forward, Op::GradQ, Op::GradK, Op::GradV] {
        assert_bits(
            &format!("{op:?}"),
            &run32(&negative, op, &g).unwrap(),
            &run32(&case, op, &g).unwrap(),
        );
    }
}

/// Attack 9. The next `f32` above 1, an infinity of either sign, `-1`, the smallest subnormal and
/// NaN in the mask are each `KeyMaskValue` naming the sequence and the key, for the forward and
/// every gradient op.
#[test]
fn adversary_attention_a9_mask_values_other_than_0_and_1_are_refused_with_sequence_and_key() {
    let base = Case::draw("adversary_attention_a9_values", 2, 3, 4, 2, &[1.0; 6]);
    let g = base.g("adversary_attention_a9_values");
    for (value, at) in [
        (f32::from_bits(1.0f32.to_bits() + 1), 4),
        (f32::INFINITY, 0),
        (f32::NEG_INFINITY, 5),
        (-1.0, 2),
        (f32::from_bits(1), 3),
        (f32::NAN, 1),
    ] {
        let mut case = base.clone();
        case.mask[at] = value;
        for op in [Op::Forward, Op::GradQ, Op::GradK, Op::GradV] {
            assert_eq!(
                run32(&case, op, &g),
                Err(RuntimeError::KeyMaskValue {
                    value: op_value(&case, op),
                    sequence: at / 3,
                    key: at % 3,
                }),
                "{value:e} at {at}, {op:?}"
            );
        }
    }
}

/// Attack 9. The check runs in ascending sequence order and, within a sequence, the value check
/// before the all-masked check: a fully masked sequence 0 is reported before a bad value in
/// sequence 1, and a bad value in an otherwise masked sequence is `KeyMaskValue`.
#[test]
fn adversary_attention_a9_mask_check_order() {
    let base = Case::draw("adversary_attention_a9_order", 2, 3, 4, 2, &[1.0; 6]);
    let mut case = base.clone();
    case.mask = vec![0.0, 0.0, 0.0, 1.0, 0.5, 1.0];
    assert_eq!(
        run32(&case, Op::Forward, &[]),
        Err(RuntimeError::NoUnmaskedKey {
            value: op_value(&case, Op::Forward),
            sequence: 0,
        })
    );
    case.mask = vec![1.0, 1.0, 1.0, 0.0, 0.0, 2.0];
    assert_eq!(
        run32(&case, Op::Forward, &[]),
        Err(RuntimeError::KeyMaskValue {
            value: op_value(&case, Op::Forward),
            sequence: 1,
            key: 2,
        })
    );
}

// ---------------------------------------------------------------------------------------------
// Attack 7: autodiff wiring, against central differences of the loss.
// ---------------------------------------------------------------------------------------------

/// The key mask of the attack 7 programs: sequence 1 hides its middle key.
const MASK_7: [f32; 6] = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0];

/// Every parameter gradient of `mse(out, t)` is within `TOL_F64` in `f64` and `TOL_F32` in
/// `f32` of central differences, with `MASK_7` as the key mask.
fn gradient_check(name: &str, mut b: TensorProgramBuilder, out: ValueId) {
    let loss = mse_to_target(&mut b, out);
    let forward = b.finish();
    let training = differentiate(&forward, loss).unwrap();
    let program = &training.program;
    let mut rng = Rng::new(seed_of(name));
    let mut point = Point::draw(&forward, &Ids::new(), &mut rng);
    point
        .inputs
        .insert("key_mask".into(), (vec![2, 3], MASK_7.to_vec()));
    let in_f64 = run(&training, &point, f64::from);
    let in_f32 = run(&training, &point, |v| v);
    assert_eq!(
        in_f64.len(),
        1 + forward.params.len(),
        "{name}: a gradient per parameter"
    );
    for (k, (&param, _)) in training.gradients.iter().enumerate() {
        let path = &forward.params[&param].path;
        let len = point.params[&param].1.len();
        let differences: Vec<f64> = (0..len)
            .map(|i| {
                (loss_at(program, &point, param, i, STEP)
                    - loss_at(program, &point, param, i, -STEP))
                    / (2.0 * STEP)
            })
            .collect();
        let scale = differences.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            scale >= MIN_SCALE,
            "{name} {path}: differences {differences:?}"
        );
        let g64 = in_f64[1 + k].data();
        let g32: Vec<f64> = in_f32[1 + k].data().iter().map(|&v| f64::from(v)).collect();
        let (e64, e32) = (
            relative_error(g64, &differences),
            relative_error(&g32, &differences),
        );
        assert!(e64 <= TOL_F64, "{name} {path}: f64 error {e64:e}");
        assert!(e32 <= TOL_F32, "{name} {path}: f32 error {e32:e}");
    }
}

/// Attack 7. One value `z = matmul(x, w)` is `q`, `k` and `v` at once, and the attention output
/// is read twice, directly and through `layer_norm`: `mse(add(y, layer_norm(y, gamma, beta)), t)`.
/// The gradients of `w`, `gamma` and `beta` match central differences, so the three gradient
/// ops' contributions to `z` and the two uses of `y` are summed into their adjoints.
#[test]
fn adversary_attention_a7_one_value_as_q_k_v_and_output_read_twice() {
    let mut b = TensorProgramBuilder::new();
    let x = input(&mut b, "x", &[6, 4]);
    let m = input(&mut b, "key_mask", &[2, 3]);
    let w = param(&mut b, "w", &[4, 4]);
    let gamma = param(&mut b, "gamma", &[4]);
    let beta = param(&mut b, "beta", &[4]);
    let z = b.matmul(x, w).unwrap();
    let y = b.attention(z, z, z, m, 2).unwrap();
    let normed = b.layer_norm(y, gamma, beta).unwrap();
    let out = b.add(y, normed).unwrap();
    gradient_check("adversary_attention_a7_shared", b, out);
}

/// Attack 7. `q`, `k` and `v` parameters read directly by `attention` get gradients that match
/// central differences, the masked rows of `k` and `v` included.
#[test]
fn adversary_attention_a7_parameters_as_operands() {
    let mut b = TensorProgramBuilder::new();
    let q = param(&mut b, "q", &[6, 4]);
    let k = param(&mut b, "k", &[6, 4]);
    let v = param(&mut b, "v", &[6, 4]);
    let m = input(&mut b, "key_mask", &[2, 3]);
    let y = b.attention(q, k, v, m, 2).unwrap();
    gradient_check("adversary_attention_a7_params", b, y);
}
