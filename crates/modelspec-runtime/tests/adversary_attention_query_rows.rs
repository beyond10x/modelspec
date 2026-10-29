//! Adversary cases of story:attention-fewer-query-rows: `attention` with `q` of `[B * L, H]`
//! against `k`, `v` of `[B * S, H]` and a `[B, S]` key mask, `1 <= L <= S`.
//!
//! CPU (always run): a hand-built program outside the shape rule is refused by the interpreter,
//! not a panic; the result of any `L` queries per sequence (here the *last* `L`, not the first)
//! equals those rows of the `L = S` result in bits, for `L = 1, 2, S - 1, S`, in `f32` and
//! `f64`; NaN, infinities and `-0.0` in `q`, `k`, `v` and `G` give the same bits at `L = 1` as
//! in those rows of `L = S`, and a NaN in a masked key's `v` reaches no result; the gradients
//! agree with central differences under prefix, non-prefix and one-key masks, an all-masked
//! sequence is refused; and a graph that gathers one row per sequence with `embedding` before
//! attention (the decider's shape) has the gradients central differences give and the forward
//! bits of the full attention's gathered rows.
//!
//! Vulkan (feature `vulkan`): every attention module of `L = 1, 2, 3, S - 1, S` emits within
//! 16 KiB of workgroup memory, and a hand-built kernel outside the rule is an emit error, not a
//! panic. On a device (ignored by default, `MODELSPEC_VK_VALIDATION=1 cargo test -p
//! modelspec-runtime --features vulkan --test adversary_attention_query_rows -- --ignored`):
//! the last `L` rows match the device's own `L = S` rows in bits and the CPU within the GPU
//! tolerance, for `S` that no tile divides, specials, all-masked sequences and enough sequences
//! for a 2-D dispatch; and a timing case prints the `L = S` kernel times of the decider's and
//! BERT-tiny's shapes.

#![allow(clippy::type_complexity)]

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

use modelspec_core::autodiff::differentiate;
use modelspec_core::ir::tensor::{
    TensorNode, TensorOp, TensorProgram, TensorProgramBuilder, TensorType, ValueId,
};
use modelspec_runtime::cpu::{Element, HostTensor, RuntimeError, evaluate, evaluate_with_ids};

#[allow(dead_code)]
#[path = "common/finite_difference.rs"]
mod finite_difference;

#[allow(dead_code)]
mod common;

use finite_difference::{
    H as STEP, Ids, MIN_SCALE, Point, Rng, TOL_F32, TOL_F64, id_input, input, loss_at,
    mse_to_target, param, relative_error, run, seed_of,
};

/// The four attention ops of `heads` heads.
fn ops(heads: u64) -> [TensorOp; 4] {
    [
        TensorOp::Attention { heads },
        TensorOp::AttentionGradQ { heads },
        TensorOp::AttentionGradK { heads },
        TensorOp::AttentionGradV { heads },
    ]
}

fn per_query(op: &TensorOp) -> bool {
    matches!(
        op,
        TensorOp::Attention { .. } | TensorOp::AttentionGradQ { .. }
    )
}

fn arity(op: &TensorOp) -> usize {
    if matches!(op, TensorOp::Attention { .. }) {
        4
    } else {
        5
    }
}

/// The shape of one case: `b` sequences of `s` keys, `h` columns of `heads` heads.
#[derive(Clone, Copy, Debug)]
struct Shape {
    b: u64,
    s: u64,
    h: u64,
    heads: u64,
}

/// The key mask of `shape`: sequence `seq` keeps every key (`seq % 5 == 0`), a prefix of
/// `ceil(s / 2)` keys (1), every other key from key 1 (2, key 0 where `s = 1`), only key `s / 2`
/// (3), or no key (4).
fn mask_of(shape: Shape) -> Vec<f32> {
    let (b, s) = (shape.b as usize, shape.s as usize);
    let mut mask = vec![0.0f32; b * s];
    for seq in 0..b {
        for j in 0..s {
            let keep = match seq % 5 {
                0 => true,
                1 => j < s.div_ceil(2),
                2 => {
                    if s == 1 {
                        true
                    } else {
                        j % 2 == 1
                    }
                }
                3 => j == s / 2,
                _ => false,
            };
            mask[seq * s + j] = if keep { 1.0 } else { 0.0 };
        }
    }
    mask
}

/// The query rows of the `L = S` layout a case of `l` queries per sequence reads: the last `l`
/// of each sequence, in order.
fn selected(shape: Shape, l: u64) -> Vec<usize> {
    (0..shape.b)
        .flat_map(|seq| (shape.s - l..shape.s).map(move |j| (seq * shape.s + j) as usize))
        .collect()
}

/// Rows `rows` of `x`, rows of `h` values.
fn rows_of(x: &[f32], rows: &[usize], h: u64) -> Vec<f32> {
    let h = h as usize;
    rows.iter()
        .flat_map(|&r| x[r * h..(r + 1) * h].to_vec())
        .collect()
}

/// `x` with every row not in `rows` `0.0`.
fn only_rows(x: &[f32], rows: &[usize], h: u64) -> Vec<f32> {
    let h = h as usize;
    let mut out = vec![0.0f32; x.len()];
    for &r in rows {
        out[r * h..(r + 1) * h].copy_from_slice(&x[r * h..(r + 1) * h]);
    }
    out
}

/// The interpreter's result in `T` of `op` over `q` and `g` of `rows` rows, `k`, `v` of
/// `b * s` and the mask; an error or a panic as text.
fn cpu_op<T: Element>(
    op: &TensorOp,
    shape: Shape,
    rows: u64,
    values: [&[f32]; 5],
    convert: impl Fn(f32) -> T,
) -> Result<Vec<T>, String> {
    let Shape { b, s, h, .. } = shape;
    let shapes: [Vec<u64>; 5] = [
        vec![rows, h],
        vec![b * s, h],
        vec![b * s, h],
        vec![b, s],
        vec![rows, h],
    ];
    let names = ["q", "k", "v", "key_mask", "g"];
    let mut builder = TensorProgramBuilder::new();
    let operands: Vec<_> = (0..arity(op))
        .map(|at| input(&mut builder, names[at], &shapes[at]))
        .collect();
    let y = builder
        .push(op.clone(), &operands)
        .map_err(|e| format!("{e:?}"))?;
    builder.output(y).unwrap();
    let program = builder.finish();
    let inputs: BTreeMap<String, HostTensor<T>> = (0..arity(op))
        .map(|at| {
            let data = values[at].iter().map(|&x| convert(x)).collect();
            (
                names[at].to_owned(),
                HostTensor::new(shapes[at].clone(), data).unwrap(),
            )
        })
        .collect();
    match catch_unwind(AssertUnwindSafe(|| {
        evaluate(&program, &inputs, &BTreeMap::new())
    })) {
        Err(_) => Err("the interpreter panicked".to_owned()),
        Ok(Err(e)) => Err(format!("{e}")),
        Ok(Ok(mut out)) => Ok(out.remove(0).data().to_vec()),
    }
}

/// The first differences of `a` against `b`: in bits, or with `zeros_alike` by value; two NaNs
/// are alike either way.
fn diffs<T: Element + Into<f64>>(a: &[T], b: &[T], zeros_alike: bool, label: &str) -> Vec<String> {
    let mut out: Vec<String> = a
        .iter()
        .zip(b)
        .enumerate()
        .filter(|(_, (x, y))| {
            let (x, y): (f64, f64) = ((**x).into(), (**y).into());
            if x.is_nan() && y.is_nan() {
                false
            } else if zeros_alike {
                x != y
            } else {
                x.to_bits() != y.to_bits()
            }
        })
        .take(3)
        .map(|(i, (x, y))| {
            let (x, y): (f64, f64) = ((*x).into(), (*y).into());
            format!("{label} [{i}]: {x:?} against {y:?}")
        })
        .collect();
    if a.len() != b.len() {
        out.push(format!("{label}: {} values against {}", a.len(), b.len()));
    }
    out
}

/// `n` values of the finite-difference generator.
fn draw(rng: &mut Rng, n: u64) -> Vec<f32> {
    (0..n).map(|_| rng.value()).collect()
}

/// The `l`-query results of `shape` in `T` against the `L = S` ones: `Attention` and
/// `AttentionGradQ` equal the selected rows in bits, the key gradients the `L = S` result under
/// a `G` zero outside the selected rows, zeros alike.
#[allow(clippy::too_many_arguments)]
fn compare_cpu<T: Element + Into<f64>>(
    shape: Shape,
    l: u64,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    mask: &[f32],
    convert: impl Fn(f32) -> T + Copy,
    label: &str,
) -> Vec<String> {
    let Shape { b, s, h, heads } = shape;
    let sel = selected(shape, l);
    let (ql, gl, g0) = (
        rows_of(q, &sel, h),
        rows_of(g, &sel, h),
        only_rows(g, &sel, h),
    );
    let mut wrong = Vec::new();
    for op in ops(heads) {
        let at = format!("{label} L {l} {}", op.name());
        let one = match cpu_op(&op, shape, b * l, [&ql, k, v, mask, &gl], convert) {
            Ok(x) => x,
            Err(e) => {
                wrong.push(format!("{at}: {e}"));
                continue;
            }
        };
        let gs: &[f32] = if per_query(&op) { g } else { &g0 };
        let all = match cpu_op(&op, shape, b * s, [q, k, v, mask, gs], convert) {
            Ok(x) => x,
            Err(e) => {
                wrong.push(format!("{at} (L = S): {e}"));
                continue;
            }
        };
        if per_query(&op) {
            let hh = h as usize;
            let want: Vec<T> = sel
                .iter()
                .flat_map(|&r| all[r * hh..(r + 1) * hh].to_vec())
                .collect();
            wrong.extend(diffs(&one, &want, false, &at));
        } else {
            wrong.extend(diffs(&one, &all, true, &at));
        }
    }
    wrong
}

/// The CPU shapes: one key, two keys, 7 keys, 33 keys; each with the mask patterns 0 to 3 (no
/// all-masked sequence, which the CPU refuses).
const CPU_SHAPES: [Shape; 4] = [
    Shape {
        b: 3,
        s: 1,
        h: 4,
        heads: 2,
    },
    Shape {
        b: 4,
        s: 2,
        h: 6,
        heads: 3,
    },
    Shape {
        b: 4,
        s: 7,
        h: 6,
        heads: 2,
    },
    Shape {
        b: 4,
        s: 33,
        h: 8,
        heads: 4,
    },
];

fn lengths(s: u64) -> Vec<u64> {
    let mut ls: Vec<u64> = [1, 2, 3, s.saturating_sub(1), s]
        .into_iter()
        .filter(|&l| l >= 1 && l <= s)
        .collect();
    ls.sort_unstable();
    ls.dedup();
    ls
}

/// The last `L` query rows of every sequence give, for `L = 1, 2, 3, S - 1, S`, the bits of
/// those rows of the `L = S` result, in `f32` and `f64`.
#[test]
fn adv_qrows_cpu_last_rows_equal_those_rows_of_every_row() {
    let mut wrong = Vec::new();
    for shape in CPU_SHAPES {
        let mut rng = Rng::new(seed_of(&format!("adv_qrows_cpu {shape:?}")));
        let n = shape.b * shape.s * shape.h;
        let (q, k, v, g) = (
            draw(&mut rng, n),
            draw(&mut rng, n),
            draw(&mut rng, n),
            draw(&mut rng, n),
        );
        let mask = mask_of(shape);
        for l in lengths(shape.s) {
            let label = format!("{shape:?}");
            wrong.extend(compare_cpu::<f32>(
                shape,
                l,
                &q,
                &k,
                &v,
                &g,
                &mask,
                |x| x,
                &label,
            ));
            wrong.extend(compare_cpu::<f64>(
                shape,
                l,
                &q,
                &k,
                &v,
                &g,
                &mask,
                f64::from,
                &format!("{label} f64"),
            ));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// The special values of [`specials`]: written into the selected rows of `q` and `g`, and into
/// kept and masked keys of `k` and `v` of `shape` (`s = 4`, 4 sequences, mask patterns 0 to 3).
fn specials(shape: Shape, l: u64, q: &mut [f32], k: &mut [f32], v: &mut [f32], g: &mut [f32]) {
    let h = shape.h as usize;
    let s = shape.s as usize;
    let sel = selected(shape, l);
    // Sequence 0 (all keys kept): a NaN in the query, an infinity in the gradient.
    q[sel[0] * h] = f32::NAN;
    g[sel[0] * h + 1] = f32::INFINITY;
    // Sequence 1 (prefix of 2 kept): +inf in a query column, -inf in key 0 of k (kept), and a
    // NaN in v of masked key 3 that must reach nothing.
    let s1 = sel[l as usize];
    q[s1 * h + 2] = f32::INFINITY;
    k[s * h + 1] = f32::NEG_INFINITY;
    v[(s + 3) * h] = f32::NAN;
    v[(s + 3) * h + 3] = f32::INFINITY;
    // Sequence 2 (keys 1 and 3 kept): a query of -0.0 only, a gradient of -0.0 only, a NaN in
    // masked key 0 of k and of v.
    let s2 = sel[2 * l as usize];
    for c in 0..h {
        q[s2 * h + c] = -0.0;
        g[s2 * h + c] = -0.0;
    }
    k[(2 * s) * h + 2] = f32::NAN;
    v[(2 * s) * h + 2] = f32::NAN;
    // Sequence 3 (key 2 kept): +inf in its kept key's v.
    v[(3 * s + 2) * h + 1] = f32::INFINITY;
}

const SPECIAL_SHAPE: Shape = Shape {
    b: 4,
    s: 4,
    h: 4,
    heads: 2,
};

/// With NaN, infinities and `-0.0` in `q`, `k`, `v` and `G`, `L = 1` and `L = 3` give the
/// bits of the selected rows of `L = S` on the CPU.
#[test]
fn adv_qrows_cpu_specials_keep_the_bits_of_every_row() {
    let shape = SPECIAL_SHAPE;
    let mut wrong = Vec::new();
    for l in [1, 3] {
        let mut rng = Rng::new(seed_of("adv_qrows_specials"));
        let n = shape.b * shape.s * shape.h;
        let (mut q, mut k, mut v, mut g) = (
            draw(&mut rng, n),
            draw(&mut rng, n),
            draw(&mut rng, n),
            draw(&mut rng, n),
        );
        specials(shape, l, &mut q, &mut k, &mut v, &mut g);
        let mask = mask_of(shape);
        wrong.extend(compare_cpu::<f32>(
            shape,
            l,
            &q,
            &k,
            &v,
            &g,
            &mask,
            |x| x,
            "specials",
        ));
        wrong.extend(compare_cpu::<f64>(
            shape,
            l,
            &q,
            &k,
            &v,
            &g,
            &mask,
            f64::from,
            "specials f64",
        ));
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// A NaN in the `k` and `v` rows of a masked key reaches no value of the forward result at
/// `L = 1`: sequence 2, whose query is all `-0.0` and whose masked key 0 holds the NaNs, has a
/// finite row.
#[test]
fn adv_qrows_cpu_masked_specials_reach_no_result() {
    let shape = SPECIAL_SHAPE;
    let mut rng = Rng::new(seed_of("adv_qrows_masked_specials"));
    let n = shape.b * shape.s * shape.h;
    let (mut q, mut k, mut v, mut g) = (
        draw(&mut rng, n),
        draw(&mut rng, n),
        draw(&mut rng, n),
        draw(&mut rng, n),
    );
    specials(shape, 1, &mut q, &mut k, &mut v, &mut g);
    let sel = selected(shape, 1);
    let ql = rows_of(&q, &sel, shape.h);
    let mask = mask_of(shape);
    let y = cpu_op::<f32>(
        &TensorOp::Attention { heads: 2 },
        shape,
        shape.b,
        [&ql, &k, &v, &mask, &[]],
        |x| x,
    )
    .expect("the forward runs");
    let h = shape.h as usize;
    let seq2 = &y[2 * h..3 * h];
    assert!(seq2.iter().all(|x| x.is_finite()), "sequence 2: {seq2:?}");
}

/// `mse(attention(xq · w_q, x · w_k, x · w_v, mask, heads = 2), t)` with `L` queries per
/// sequence, its gradients against central differences in `f64` (and `f32`), under the mask
/// `mask` of `[b, s]`.
fn gradient_case(name: &str, b: u64, s: u64, l: u64, mask: Vec<f32>) -> Vec<String> {
    const H: u64 = 4;
    const D: u64 = 3;
    let mut builder = TensorProgramBuilder::new();
    let x_q = input(&mut builder, "x_q", &[b * l, D]);
    let x = input(&mut builder, "x", &[b * s, D]);
    let key_mask = input(&mut builder, "key_mask", &[b, s]);
    let w_q = param(&mut builder, "w_q", &[D, H]);
    let w_k = param(&mut builder, "w_k", &[D, H]);
    let w_v = param(&mut builder, "w_v", &[D, H]);
    let q = builder.matmul(x_q, w_q).unwrap();
    let k = builder.matmul(x, w_k).unwrap();
    let v = builder.matmul(x, w_v).unwrap();
    let y = builder.attention(q, k, v, key_mask, 2).unwrap();
    let loss = mse_to_target(&mut builder, y);
    let forward = builder.finish();
    let training = differentiate(&forward, loss).unwrap();
    let program = &training.program;
    let mut rng = Rng::new(seed_of(name));
    let mut point = Point::draw(&forward, &Ids::new(), &mut rng);
    point.inputs.insert("key_mask".into(), (vec![b, s], mask));
    let mut wrong = Vec::new();
    let in_f64 = match catch_unwind(AssertUnwindSafe(|| run(&training, &point, f64::from))) {
        Ok(x) => x,
        Err(_) => return vec![format!("{name}: the f64 run failed")],
    };
    let in_f32 = run(&training, &point, |v| v);
    for (k_param, (&p, _)) in training.gradients.iter().enumerate() {
        let path = &forward.params[&p].path;
        let n = point.params[&p].1.len();
        let differences: Vec<f64> = (0..n)
            .map(|i| {
                (loss_at(program, &point, p, i, STEP) - loss_at(program, &point, p, i, -STEP))
                    / (2.0 * STEP)
            })
            .collect();
        let scale = differences.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if scale < MIN_SCALE {
            // w_k under one kept key has a zero gradient: softmax of one key is 1.
            let g = in_f64[1 + k_param].data();
            if g.iter().any(|x| x.abs() > 1e-9) {
                wrong.push(format!("{name} {path}: differences ~0, gradient {g:?}"));
            }
            continue;
        }
        let e64 = relative_error(in_f64[1 + k_param].data(), &differences);
        let g32: Vec<f64> = in_f32[1 + k_param]
            .data()
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let e32 = relative_error(&g32, &differences);
        if e64 > TOL_F64 || e32 > TOL_F32 {
            wrong.push(format!(
                "{name} {path}: f64 error {e64:e}, f32 error {e32:e}"
            ));
        }
    }
    wrong
}

/// Central differences agree with the gradients of `L = 1` and `L = S - 1` under a prefix mask,
/// a non-prefix mask, one kept key of four, and one key per sequence (`S = 1`).
#[test]
fn adv_qrows_cpu_gradients_under_every_mask_match_central_differences() {
    let mut wrong = Vec::new();
    let masks: [(&str, u64, Vec<f32>); 4] = [
        ("prefix", 4, vec![1., 1., 0., 0., 1., 1., 1., 0.]),
        ("not prefix", 4, vec![0., 1., 0., 1., 1., 0., 1., 1.]),
        ("one key of four", 4, vec![0., 0., 1., 0., 0., 1., 0., 0.]),
        ("one key", 1, vec![1., 1.]),
    ];
    for (what, s, mask) in masks {
        for l in [1, s.saturating_sub(1).max(1)] {
            wrong.extend(gradient_case(
                &format!("adv_qrows_grad {what} L {l}"),
                2,
                s,
                l,
                mask.clone(),
            ));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// A sequence whose mask keeps no key is refused at `L = 1` as at `L = S`, an error naming the
/// sequence and not a panic, for the forward and each gradient.
#[test]
fn adv_qrows_cpu_all_masked_sequence_is_refused_not_a_panic() {
    let shape = Shape {
        b: 2,
        s: 3,
        h: 4,
        heads: 2,
    };
    let mask = vec![1.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let n = shape.b * shape.s * shape.h;
    let mut rng = Rng::new(7);
    let (q, k, v, g) = (
        draw(&mut rng, n),
        draw(&mut rng, n),
        draw(&mut rng, n),
        draw(&mut rng, n),
    );
    let sel = selected(shape, 1);
    let (ql, gl) = (rows_of(&q, &sel, shape.h), rows_of(&g, &sel, shape.h));
    let mut wrong = Vec::new();
    for op in ops(2) {
        match cpu_op::<f32>(&op, shape, shape.b, [&ql, &k, &v, &mask, &gl], |x| x) {
            Ok(x) => wrong.push(format!("{}: accepted, {x:?}", op.name())),
            Err(e) if e.contains("panicked") => wrong.push(format!("{}: {e}", op.name())),
            Err(e) if !e.contains("sequence 1") && !e.contains('1') => {
                wrong.push(format!("{}: {e} does not name sequence 1", op.name()))
            }
            Err(_) => {}
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// A hand-built program whose attention node breaks the rule (its type written as if it held)
/// is refused by the interpreter as a program that does not verify, never a panic: `q` rows not
/// a multiple of `B`, `L > S`, `k` rows not `B * S`, heads not dividing `H`, a gradient whose
/// result is `q`'s shape for `AttentionGradK`.
#[test]
fn adv_qrows_cpu_hand_built_programs_outside_the_rule_do_not_verify() {
    let f = |shape: &[u64]| TensorType::f32(shape);
    // (label, op, q, k, mask, declared result)
    let cases: Vec<(&str, TensorOp, Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>)> = vec![
        (
            "q rows 5, B 2",
            TensorOp::Attention { heads: 2 },
            vec![5, 4],
            vec![6, 4],
            vec![2, 3],
            vec![5, 4],
        ),
        (
            "L 4 > S 3",
            TensorOp::Attention { heads: 2 },
            vec![8, 4],
            vec![6, 4],
            vec![2, 3],
            vec![8, 4],
        ),
        (
            "k rows 8, B * S 6",
            TensorOp::Attention { heads: 2 },
            vec![2, 4],
            vec![8, 4],
            vec![2, 3],
            vec![2, 4],
        ),
        (
            "heads 3, H 4",
            TensorOp::Attention { heads: 3 },
            vec![2, 4],
            vec![6, 4],
            vec![2, 3],
            vec![2, 4],
        ),
        (
            "GradK of q's shape",
            TensorOp::AttentionGradK { heads: 2 },
            vec![2, 4],
            vec![6, 4],
            vec![2, 3],
            vec![2, 4],
        ),
        (
            "GradQ of k's shape",
            TensorOp::AttentionGradQ { heads: 2 },
            vec![2, 4],
            vec![6, 4],
            vec![2, 3],
            vec![6, 4],
        ),
    ];
    let mut wrong = Vec::new();
    for (label, op, q, k, mask, result) in cases {
        let input = |id: u32, name: &str, shape: &[u64]| TensorNode {
            id: ValueId(id),
            op: TensorOp::Input {
                name: name.to_owned(),
            },
            operands: vec![],
            ty: f(shape),
            source_span: None,
        };
        let mut nodes = vec![
            input(0, "q", &q),
            input(1, "k", &k),
            input(2, "v", &k),
            input(3, "key_mask", &mask),
            input(4, "g", &q),
        ];
        let operands: Vec<ValueId> = (0..arity(&op) as u32).map(ValueId).collect();
        nodes.push(TensorNode {
            id: ValueId(5),
            op: op.clone(),
            operands,
            ty: f(&result),
            source_span: None,
        });
        let program = TensorProgram {
            nodes,
            params: BTreeMap::new(),
            inputs: (0..5).map(ValueId).collect(),
            outputs: vec![ValueId(5)],
        };
        let tensor = |shape: &[u64]| {
            let n = shape.iter().product::<u64>() as usize;
            HostTensor::new(shape.to_vec(), vec![1.0f32; n]).unwrap()
        };
        let inputs: BTreeMap<String, HostTensor<f32>> = [
            ("q", &q),
            ("k", &k),
            ("v", &k),
            ("key_mask", &mask),
            ("g", &q),
        ]
        .into_iter()
        .map(|(name, shape)| (name.to_owned(), tensor(shape)))
        .collect();
        let got = catch_unwind(AssertUnwindSafe(|| {
            evaluate_with_ids(&program, &inputs, &BTreeMap::new(), &BTreeMap::new())
        }));
        match got {
            Err(_) => wrong.push(format!("{label}: the interpreter panicked")),
            Ok(Ok(_)) => wrong.push(format!("{label}: evaluated")),
            Ok(Err(RuntimeError::Verify(_))) => {}
            Ok(Err(other)) => wrong.push(format!("{label}: refused as {other}")),
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// The decider's shape on the CPU: `t3 = x · w_t` of `B * S` rows, `e = embedding(t3, ids)` with
/// `ids = u * S + offset`, then `attention(e · w_q, t3 · w_k, t3 · w_v, mask, heads = 2)` with
/// `L = 1`. The forward equals rows `ids` of the `L = S` attention over `t3 · w_q` in bits, and
/// the gradients of `w_t`, `w_q`, `w_k` and `w_v` agree with central differences, for the first
/// and the last row of each sequence.
#[test]
fn adv_qrows_cpu_gather_before_attention_matches_central_differences() {
    const B: u64 = 3;
    const S: u64 = 4;
    const D: u64 = 3;
    const H: u64 = 4;
    let mask = vec![1., 1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 1.];
    let mut wrong = Vec::new();
    for offset in [0u64, S - 1] {
        let name = format!("adv_qrows_gather offset {offset}");
        let ids: Vec<i32> = (0..B).map(|u| (u * S + offset) as i32).collect();
        let build = |gathered: bool, with_loss: bool| {
            let mut b = TensorProgramBuilder::new();
            let x = input(&mut b, "x", &[B * S, D]);
            let at = id_input(&mut b, "evt_row", &[B]);
            let key_mask = input(&mut b, "key_mask", &[B, S]);
            let w_t = param(&mut b, "w_t", &[D, H]);
            let w_q = param(&mut b, "w_q", &[H, H]);
            let w_k = param(&mut b, "w_k", &[H, H]);
            let w_v = param(&mut b, "w_v", &[H, H]);
            let t3 = b.matmul(x, w_t).unwrap();
            let q = if gathered {
                let e = b.embedding(t3, at).unwrap();
                b.matmul(e, w_q).unwrap()
            } else {
                b.matmul(t3, w_q).unwrap()
            };
            let k = b.matmul(t3, w_k).unwrap();
            let v = b.matmul(t3, w_v).unwrap();
            let y = b.attention(q, k, v, key_mask, 2).unwrap();
            let y = if gathered {
                y
            } else {
                b.embedding(y, at).unwrap()
            };
            if with_loss {
                let loss = mse_to_target(&mut b, y);
                (b.finish(), Some(loss))
            } else {
                b.output(y).unwrap();
                (b.finish(), None)
            }
        };
        let (forward, loss) = build(true, true);
        let loss = loss.unwrap();
        let (gathered, _) = build(true, false);
        let (full, _) = build(false, false);
        let mut id_map = Ids::new();
        id_map.insert("evt_row".into(), (vec![B], ids.clone()));
        let mut rng = Rng::new(seed_of(&name));
        let mut point = Point::draw(&forward, &id_map, &mut rng);
        point
            .inputs
            .insert("key_mask".into(), (vec![B, S], mask.clone()));
        // Forward bits: the gathered program's output against the full one's gathered rows.
        let mut inputs = finite_difference::tensors(&point.inputs, |v| v);
        inputs.remove("t");
        let params = finite_difference::tensors(&point.params, |v| v);
        let id_tensors = point.id_tensors();
        let a = evaluate_with_ids(&gathered, &inputs, &id_tensors, &params).unwrap();
        let c = evaluate_with_ids(&full, &inputs, &id_tensors, &params).unwrap();
        let ya = a[0].data();
        let yc = c[0].data();
        wrong.extend(diffs(ya, yc, false, &format!("{name} forward")));
        let training = differentiate(&forward, loss).unwrap();
        let program = &training.program;
        let in_f64 = run(&training, &point, f64::from);
        for (k_param, (&p, _)) in training.gradients.iter().enumerate() {
            let path = &forward.params[&p].path;
            let n = point.params[&p].1.len();
            let differences: Vec<f64> = (0..n)
                .map(|i| {
                    (loss_at(program, &point, p, i, STEP) - loss_at(program, &point, p, i, -STEP))
                        / (2.0 * STEP)
                })
                .collect();
            let scale = differences.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            if scale < MIN_SCALE {
                wrong.push(format!(
                    "{name} {path}: differences too small {differences:?}"
                ));
                continue;
            }
            let e64 = relative_error(in_f64[1 + k_param].data(), &differences);
            if e64 > TOL_F64 {
                wrong.push(format!("{name} {path}: f64 error {e64:e}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[cfg(feature = "vulkan")]
mod vk {
    //! The Vulkan cases.

    use super::*;

    use modelspec_core::ir::kernel::{Kernel, Operand};
    use modelspec_core::schedule::{attention_loss_kernel, with_softmax_stats};
    use modelspec_runtime::spirv::emit::emit;
    use modelspec_runtime::vulkan::memory::Buffer;
    use modelspec_runtime::vulkan::negotiate::negotiate;
    use modelspec_runtime::vulkan::sync::Barrier;
    use modelspec_runtime::vulkan::{Device, Vulkan};

    use crate::common::{Draw, GPU_ATOL, GPU_RTOL, within};

    /// The kernel of `op` over `q` (and `g`) of `n` rows, `k`, `v` of `m`, a `[b, s]` mask.
    fn kernel_of(op: &TensorOp, n: u64, m: u64, b: u64, s: u64, h: u64) -> Option<Kernel> {
        let (queries, keys, mask) = (
            TensorType::f32(&[n, h]),
            TensorType::f32(&[m, h]),
            TensorType::f32(&[b, s]),
        );
        let types = [&queries, &keys, &keys, &mask, &queries];
        let operands = types[..arity(op)]
            .iter()
            .map(|&t| Operand::Buffer(t.clone()))
            .collect();
        let result = if per_query(op) {
            queries.clone()
        } else {
            keys.clone()
        };
        attention_loss_kernel(op, operands, result)
    }

    // OpTypeFloat, OpTypeVector, OpTypeArray, OpTypePointer, OpConstant, OpVariable, Workgroup.
    const OP_TYPE_FLOAT: u32 = 22;
    const OP_TYPE_VECTOR: u32 = 23;
    const OP_TYPE_ARRAY: u32 = 28;
    const OP_TYPE_POINTER: u32 = 32;
    const OP_CONSTANT: u32 = 43;
    const OP_VARIABLE: u32 = 59;
    const STORAGE_WORKGROUP: u32 = 4;

    /// The bytes of a module's `Workgroup` variables.
    fn workgroup_bytes(words: &[u32]) -> u64 {
        let mut sizes: BTreeMap<u32, u64> = BTreeMap::new();
        let mut constants: BTreeMap<u32, u64> = BTreeMap::new();
        let mut pointees: BTreeMap<u32, u32> = BTreeMap::new();
        let mut total = 0;
        let mut at = 5;
        while at < words.len() {
            let count = (words[at] >> 16) as usize;
            assert!(count > 0, "a zero-length instruction");
            let (op, w) = (words[at] & 0xffff, &words[at + 1..at + count]);
            match op {
                OP_TYPE_FLOAT => {
                    sizes.insert(w[0], u64::from(w[1] / 8));
                }
                OP_TYPE_VECTOR => {
                    let size = sizes.get(&w[1]).copied().unwrap_or(0) * u64::from(w[2]);
                    sizes.insert(w[0], size);
                }
                OP_CONSTANT => {
                    constants.insert(w[1], u64::from(w[2]));
                }
                OP_TYPE_ARRAY => {
                    let size = sizes.get(&w[1]).copied().unwrap_or(0) * constants[&w[2]];
                    sizes.insert(w[0], size);
                }
                OP_TYPE_POINTER if w[1] == STORAGE_WORKGROUP => {
                    pointees.insert(w[0], w[2]);
                }
                OP_VARIABLE if w[2] == STORAGE_WORKGROUP => {
                    total += sizes[&pointees[&w[0]]];
                }
                _ => {}
            }
            at += count;
        }
        total
    }

    /// Every attention module of `L = 1, 2, 3, S - 1, S` over `S` around every tile and halving
    /// boundary, heads of 1, 4 and 64 columns, alone and sharing statistics, emits within the
    /// 16 KiB of workgroup memory every device has.
    #[test]
    fn adv_qrows_every_module_emits_within_16_kib() {
        let mut wrong = Vec::new();
        let keys = [
            1u64, 2, 3, 5, 31, 33, 43, 61, 64, 91, 122, 128, 187, 239, 256, 379, 453, 511, 512,
        ];
        for s in keys {
            for l in lengths(s) {
                for (h, heads) in [(4u64, 4u64), (16, 4), (256, 4)] {
                    for op in ops(heads) {
                        let Some(kernel) = kernel_of(&op, 2 * l, 2 * s, 2, s, h) else {
                            wrong.push(format!("S {s} L {l} h {h} {}: no kernel", op.name()));
                            continue;
                        };
                        for k in [Some(kernel.clone()), with_softmax_stats(kernel)]
                            .into_iter()
                            .flatten()
                        {
                            let label = format!(
                                "S {s} L {l} h {h} {} stats {}",
                                op.name(),
                                k.softmax_stats.is_some()
                            );
                            match catch_unwind(AssertUnwindSafe(|| emit(&k))) {
                                Err(_) => wrong.push(format!("{label}: emit panicked")),
                                Ok(Err(e)) => wrong.push(format!("{label}: {e}")),
                                Ok(Ok(words)) => {
                                    let bytes = workgroup_bytes(&words);
                                    if bytes > 16 * 1024 {
                                        wrong.push(format!("{label}: {bytes} bytes"));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// A hand-built kernel outside the rule (`L > S`, `q` rows not a multiple of `B`, `k` rows
    /// not `B * S`, a result of the wrong rows, statistics of `k`'s rows) is an emit error, not a
    /// panic.
    #[test]
    fn adv_qrows_hand_built_kernels_outside_the_rule_are_emit_errors() {
        let base = kernel_of(&TensorOp::AttentionGradK { heads: 2 }, 2, 6, 2, 3, 4).unwrap();
        let fwd = kernel_of(&TensorOp::Attention { heads: 2 }, 2, 6, 2, 3, 4).unwrap();
        let set = |k: &Kernel, at: usize, shape: &[u64]| {
            let mut k = k.clone();
            k.operands[at] = Operand::Buffer(TensorType::f32(shape));
            k
        };
        let mut wrong_result = fwd.clone();
        wrong_result.result = TensorType::f32(&[6, 4]);
        let mut stats_of_k = with_softmax_stats(fwd.clone()).unwrap();
        stats_of_k.softmax_stats = Some(TensorType::f32(&[6 * 2, 3]));
        let cases: Vec<(&str, Kernel)> = vec![
            ("forward L 4 > S 3", {
                let mut k = set(&fwd, 0, &[8, 4]);
                k.result = TensorType::f32(&[8, 4]);
                k
            }),
            ("forward q rows 5", {
                let mut k = set(&fwd, 0, &[5, 4]);
                k.result = TensorType::f32(&[5, 4]);
                k
            }),
            ("forward k rows 8", set(&set(&fwd, 1, &[8, 4]), 2, &[8, 4])),
            ("forward result of k's rows", wrong_result),
            ("forward statistics of k's rows", stats_of_k),
            ("GradK g of k's rows", set(&base, 4, &[6, 4])),
            ("GradK L 4 > S 3", set(&set(&base, 0, &[8, 4]), 4, &[8, 4])),
        ];
        let mut wrong = Vec::new();
        for (label, k) in cases {
            match catch_unwind(AssertUnwindSafe(|| emit(&k))) {
                Err(_) => wrong.push(format!("{label}: emit panicked")),
                Ok(Ok(_)) => wrong.push(format!("{label}: emitted")),
                Ok(Err(_)) => {}
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Runs `kernels` in one recording on `inputs` (`q`, `k`, `v`, mask, `g`), a statistics
    /// buffer of `stats` values; returns every result (not the statistics).
    fn run_device(
        device: &Device<'_>,
        kernels: &[Kernel],
        inputs: &[Vec<f32>],
        stats: usize,
    ) -> Vec<Vec<f32>> {
        let words: Vec<Vec<u32>> = kernels.iter().map(|k| emit(k).expect("emits")).collect();
        let pipelines: Vec<_> = kernels
            .iter()
            .zip(&words)
            .map(|(k, w)| device.kernel_pipeline(k, w).expect("a pipeline"))
            .collect();
        let mut lengths: Vec<usize> = inputs.iter().map(Vec::len).collect();
        lengths.push(stats.max(1));
        lengths.extend(kernels.iter().map(|k| {
            k.result
                .shape
                .iter()
                .map(|d| match d {
                    modelspec_core::ir::tensor::Dim::Static(n) => *n as usize,
                    _ => 0,
                })
                .product::<usize>()
        }));
        let buffers: Vec<Buffer<'_>> = lengths
            .iter()
            .map(|&n| device.device_local_buffer(4 * n as u64).expect("a buffer"))
            .collect();
        let mut staging: Vec<Buffer<'_>> = lengths
            .iter()
            .map(|&n| {
                device
                    .staging_buffer(4 * n as u64)
                    .expect("a staging buffer")
            })
            .collect();
        for (at, buffer) in staging.iter_mut().enumerate() {
            let values = match inputs.get(at) {
                Some(values) => values.clone(),
                None => vec![f32::from_bits(0x7fc0_dead); lengths[at]],
            };
            buffer.write_f32(&values).expect("written");
        }
        let mut recording = device.record().expect("a recording");
        for (from, to) in staging.iter().zip(&buffers) {
            recording.copy(from, to).expect("an upload");
        }
        recording.barrier(Barrier::TransferToCompute);
        let stats_at = inputs.len();
        for (at, (kernel, pipeline)) in kernels.iter().zip(&pipelines).enumerate() {
            let reads = kernel
                .operands
                .iter()
                .filter(|o| matches!(o, Operand::Buffer(_)))
                .count();
            let mut bound: Vec<&Buffer<'_>> = buffers[..reads].iter().collect();
            if kernel.softmax_stats.is_some() {
                bound.push(&buffers[stats_at]);
            }
            bound.push(&buffers[stats_at + 1 + at]);
            recording
                .dispatch_kernel(pipeline, kernel, &bound)
                .expect("a dispatch");
            recording.barrier(Barrier::ComputeToCompute);
        }
        recording.barrier(Barrier::ComputeToTransfer);
        for at in stats_at + 1..buffers.len() {
            recording
                .copy(&buffers[at], &staging[at])
                .expect("a download");
        }
        recording.barrier(Barrier::TransferToHost);
        let recorded = recording.finish().expect("the recording ends");
        device
            .submit(recorded)
            .expect("submitted")
            .wait()
            .expect("the device finishes");
        staging[stats_at + 1..]
            .iter_mut()
            .map(|b| b.read_f32().expect("read"))
            .collect()
    }

    /// Whether the device's `g` passes against the CPU's `c`: both NaN, both the same infinity,
    /// or within the GPU tolerance.
    fn device_within(g: f32, c: f32) -> bool {
        (g.is_nan() && c.is_nan())
            || (g.is_infinite() && g == c)
            || within(g, c, c.abs(), GPU_ATOL, GPU_RTOL)
    }

    /// One device case: `shape` at `l` queries per sequence (the last `l` of each), its inputs
    /// `q`, `k`, `v`, `g` over every row and the mask. Compares each op alone against the device's
    /// own `L = S` run (bits for the per-query ops, zeros alike for the key gradients), the four
    /// sharing statistics against the alone results in bits, and against the CPU within the GPU
    /// tolerance where every sequence keeps a key and `with_cpu`.
    #[allow(clippy::too_many_arguments)]
    fn device_case(
        vulkan: &Vulkan,
        device: &Device<'_>,
        shape: Shape,
        l: u64,
        all: [&[f32]; 4],
        mask: &[f32],
        with_cpu: bool,
        label: &str,
    ) -> Vec<String> {
        let Shape { b, s, h, heads } = shape;
        let [q, k, v, g] = all;
        let sel = selected(shape, l);
        let (n, m) = (b * l, b * s);
        let inputs = vec![
            rows_of(q, &sel, h),
            k.to_vec(),
            v.to_vec(),
            mask.to_vec(),
            rows_of(g, &sel, h),
        ];
        let g0 = only_rows(g, &sel, h);
        let keeps = mask
            .chunks(s as usize)
            .all(|keys| keys.iter().any(|&x| x != 0.0));
        let mut wrong = Vec::new();
        let mut alone = Vec::new();
        for op in ops(heads) {
            let at = format!("{label} L {l} {}", op.name());
            let Some(kernel) = kernel_of(&op, n, m, b, s, h) else {
                wrong.push(format!("{at}: no kernel"));
                return wrong;
            };
            let out = run_device(device, &[kernel], &inputs, 0).remove(0);
            let full = kernel_of(&op, m, m, b, s, h).expect("the L = S kernel");
            let gs: Vec<f32> = if per_query(&op) {
                g.to_vec()
            } else {
                g0.clone()
            };
            let every = vec![q.to_vec(), k.to_vec(), v.to_vec(), mask.to_vec(), gs];
            let all_out = run_device(device, &[full], &every, 0).remove(0);
            if per_query(&op) {
                let want = rows_of(&all_out, &sel, h);
                wrong.extend(diffs(&out, &want, false, &format!("{at} vs device L = S")));
            } else {
                wrong.extend(diffs(
                    &out,
                    &all_out,
                    true,
                    &format!("{at} vs device L = S"),
                ));
            }
            if keeps && with_cpu {
                match cpu_op::<f32>(
                    &op,
                    shape,
                    n,
                    [&inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4]],
                    |x| x,
                ) {
                    Err(e) => wrong.push(format!("{at}: CPU {e}")),
                    Ok(c) => wrong.extend(
                        out.iter()
                            .zip(&c)
                            .enumerate()
                            .filter(|(_, (g, c))| !device_within(**g, **c))
                            .take(3)
                            .map(|(i, (g, c))| format!("{at} [{i}]: gpu {g:?} cpu {c:?}")),
                    ),
                }
            }
            alone.push(out);
        }
        let shared: Vec<Kernel> = ops(heads)
            .iter()
            .map(|op| with_softmax_stats(kernel_of(op, n, m, b, s, h).unwrap()).expect("shares"))
            .collect();
        let out = run_device(device, &shared, &inputs, (n * heads * 3) as usize);
        for ((op, shared), alone) in ops(heads).iter().zip(&out).zip(&alone) {
            wrong.extend(diffs(
                shared,
                alone,
                false,
                &format!("{label} L {l} {} shared", op.name()),
            ));
        }
        let messages = vulkan.take_validation_messages();
        if !messages.is_empty() {
            wrong.push(format!("{label} L {l}: {messages:?}"));
        }
        wrong
    }

    fn open() -> (Vulkan, usize) {
        let vulkan = Vulkan::load().expect("the Vulkan loader opens");
        (vulkan, 0)
    }

    /// The most `B * L * S * H` the CPU computes in a device case.
    const CPU_WORK: u64 = 1 << 23;

    /// `S` of 1, 3, 37, 100, 129, 511 and 512 keys (none of the last five a multiple of the tile
    /// but 512), at `L = 1, 2, 3, S - 1, S`, the last `L` rows of each sequence, under the mask
    /// patterns 0 to 4 (pattern 4, no key, only past the CPU comparison).
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn adv_qrows_device_last_rows_match_every_row_and_the_cpu() {
        let (vulkan, index) = open();
        let candidates = vulkan.candidates().expect("the devices are probed");
        let negotiated = negotiate(&candidates[index].profile).expect("device 0 negotiates");
        let device = vulkan
            .create_device(index, &negotiated)
            .expect("device 0 opens");
        let shapes = [
            Shape {
                b: 3,
                s: 1,
                h: 8,
                heads: 2,
            },
            Shape {
                b: 4,
                s: 3,
                h: 8,
                heads: 2,
            },
            Shape {
                b: 5,
                s: 37,
                h: 16,
                heads: 2,
            },
            Shape {
                b: 4,
                s: 100,
                h: 64,
                heads: 1,
            },
            Shape {
                b: 4,
                s: 129,
                h: 32,
                heads: 4,
            },
            Shape {
                b: 5,
                s: 511,
                h: 16,
                heads: 2,
            },
            Shape {
                b: 4,
                s: 512,
                h: 16,
                heads: 2,
            },
            Shape {
                b: 4,
                s: 256,
                h: 256,
                heads: 4,
            },
        ];
        let mut wrong = Vec::new();
        for shape in shapes {
            let mut draw = Draw::new(0xadd0_0000 ^ shape.s ^ (shape.h << 20) ^ (shape.b << 40));
            let n = shape.b * shape.s * shape.h;
            let q: Vec<f32> = draw.values(n).into_iter().map(|x| x * 0.25).collect();
            let k: Vec<f32> = draw.values(n).into_iter().map(|x| x * 0.25).collect();
            let v = draw.values(n);
            let g = draw.values(n);
            let mask = mask_of(shape);
            for l in lengths(shape.s) {
                let with_cpu = shape.b * l * shape.s * shape.h <= CPU_WORK;
                wrong.extend(device_case(
                    &vulkan,
                    &device,
                    shape,
                    l,
                    [&q, &k, &v, &g],
                    &mask,
                    with_cpu,
                    &format!("{shape:?}"),
                ));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// NaN, infinities and `-0.0` in `q`, `k`, `v` and `G` (the CPU case's [`specials`]) on the
    /// device at `L = 1` and `L = 3`: the bits of its own `L = S` rows, and the CPU's NaNs,
    /// infinities and values within the GPU tolerance.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn adv_qrows_device_specials_match_every_row_and_the_cpu() {
        let (vulkan, index) = open();
        let candidates = vulkan.candidates().expect("the devices are probed");
        let negotiated = negotiate(&candidates[index].profile).expect("device 0 negotiates");
        let device = vulkan
            .create_device(index, &negotiated)
            .expect("device 0 opens");
        let shape = SPECIAL_SHAPE;
        let mut wrong = Vec::new();
        for l in [1, 3] {
            let mut rng = Rng::new(seed_of("adv_qrows_specials"));
            let n = shape.b * shape.s * shape.h;
            let (mut q, mut k, mut v, mut g) = (
                draw(&mut rng, n),
                draw(&mut rng, n),
                draw(&mut rng, n),
                draw(&mut rng, n),
            );
            specials(shape, l, &mut q, &mut k, &mut v, &mut g);
            let mask = mask_of(shape);
            wrong.extend(device_case(
                &vulkan,
                &device,
                shape,
                l,
                [&q, &k, &v, &g],
                &mask,
                true,
                "specials",
            ));
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// 40,000 sequences of 2 keys and 2 heads: 80,000 workgroups at `L = 1`, past the 65,535 of
    /// one dimension, so the dispatch is 2-D. The results match the device's `L = S` rows and the
    /// CPU, with every fifth sequence masked whole.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn adv_qrows_device_many_sequences_dispatch_in_two_dimensions() {
        let (vulkan, index) = open();
        let candidates = vulkan.candidates().expect("the devices are probed");
        let negotiated = negotiate(&candidates[index].profile).expect("device 0 negotiates");
        let device = vulkan
            .create_device(index, &negotiated)
            .expect("device 0 opens");
        let mut wrong = Vec::new();
        for (shape, cpu) in [
            (
                Shape {
                    b: 40_000,
                    s: 2,
                    h: 4,
                    heads: 2,
                },
                false,
            ),
            (
                Shape {
                    b: 40_004,
                    s: 2,
                    h: 4,
                    heads: 2,
                },
                false,
            ),
        ] {
            let kernel = kernel_of(
                &TensorOp::Attention { heads: 2 },
                shape.b,
                shape.b * shape.s,
                shape.b,
                shape.s,
                shape.h,
            )
            .unwrap();
            if kernel.workgroups[1] < 2 && kernel.workgroups[2] < 2 {
                wrong.push(format!(
                    "{shape:?}: workgroups {:?} are 1-D",
                    kernel.workgroups
                ));
            }
            let mut draw = Draw::new(0x2d2d ^ shape.b);
            let n = shape.b * shape.s * shape.h;
            let q: Vec<f32> = draw.values(n).into_iter().map(|x| x * 0.25).collect();
            let k: Vec<f32> = draw.values(n).into_iter().map(|x| x * 0.25).collect();
            let v = draw.values(n);
            let g = draw.values(n);
            let mask = mask_of(shape);
            for l in [1, 2] {
                wrong.extend(device_case(
                    &vulkan,
                    &device,
                    shape,
                    l,
                    [&q, &k, &v, &g],
                    &mask,
                    cpu,
                    &format!("{shape:?}"),
                ));
            }
            // The same sequences with every one keeping a key, against the CPU.
            let kept: Vec<f32> = mask
                .chunks(shape.s as usize)
                .flat_map(|keys| {
                    if keys.iter().all(|&x| x == 0.0) {
                        vec![1.0; keys.len()]
                    } else {
                        keys.to_vec()
                    }
                })
                .collect();
            wrong.extend(device_case(
                &vulkan,
                &device,
                shape,
                1,
                [&q, &k, &v, &g],
                &kept,
                true,
                &format!("{shape:?} kept"),
            ));
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    // ADV-TIMING-BEGIN
    /// The device time of each `L = S` attention kernel, sharing statistics as a training step
    /// does, of the decider's event encoder (256 sequences of 256 keys, 256 columns, 4 heads) and
    /// BERT-tiny (32 sequences of 128 keys, 128 columns, 2 heads): the median of 41 recordings of
    /// the four dispatches, timestamps between them. Prints one line per shape and op.
    #[test]
    #[ignore = "needs a Vulkan device; a timing, run under the GPU lock"]
    fn adv_qrows_timing_l_equals_s_kernels() {
        let vulkan = Vulkan::load().expect("the Vulkan loader opens");
        let candidates = vulkan.candidates().expect("the devices are probed");
        let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
        let device = vulkan
            .create_device(0, &negotiated)
            .expect("device 0 opens");
        let clock = device.clock().expect("the queue writes timestamps");
        let tag = std::env::var("ADV_TAG").unwrap_or_default();
        for (name, b, s, h, heads) in [
            ("decider", 256u64, 256u64, 256u64, 4u64),
            ("bert_tiny", 32, 128, 128, 2),
        ] {
            let n = b * s;
            let (rows, keys, mask_ty) = (
                TensorType::f32(&[n, h]),
                TensorType::f32(&[n, h]),
                TensorType::f32(&[b, s]),
            );
            let names = [
                "Attention",
                "AttentionGradQ",
                "AttentionGradK",
                "AttentionGradV",
            ];
            let ops_of = [
                TensorOp::Attention { heads },
                TensorOp::AttentionGradQ { heads },
                TensorOp::AttentionGradK { heads },
                TensorOp::AttentionGradV { heads },
            ];
            let kernels: Vec<Kernel> = ops_of
                .iter()
                .map(|op| {
                    let ar = if matches!(op, TensorOp::Attention { .. }) {
                        4
                    } else {
                        5
                    };
                    let types = [&rows, &keys, &keys, &mask_ty, &rows];
                    let operands = types[..ar]
                        .iter()
                        .map(|&t| Operand::Buffer(t.clone()))
                        .collect();
                    let k = attention_loss_kernel(op, operands, rows.clone()).expect("a kernel");
                    with_softmax_stats(k).expect("shares")
                })
                .collect();
            let words: Vec<Vec<u32>> = kernels.iter().map(|k| emit(k).expect("emits")).collect();
            let pipelines: Vec<_> = kernels
                .iter()
                .zip(&words)
                .map(|(k, w)| device.kernel_pipeline(k, w).expect("a pipeline"))
                .collect();
            let mut draw = Draw::new(0x7173 ^ s);
            let mut mask = vec![0.0f32; (b * s) as usize];
            for seq in 0..b as usize {
                let len = [
                    s as usize,
                    s as usize * 3 / 4,
                    s as usize / 2,
                    s as usize - 1,
                ][seq % 4]
                    .max(1);
                for j in 0..len {
                    mask[seq * s as usize + j] = 1.0;
                }
            }
            let inputs = vec![
                draw.values(n * h)
                    .into_iter()
                    .map(|x| x * 0.25)
                    .collect::<Vec<f32>>(),
                draw.values(n * h).into_iter().map(|x| x * 0.25).collect(),
                draw.values(n * h),
                mask,
                draw.values(n * h),
            ];
            let mut lengths: Vec<usize> = inputs.iter().map(Vec::len).collect();
            lengths.push((n * heads * 3) as usize);
            lengths.extend([(n * h) as usize; 4]);
            let buffers: Vec<Buffer<'_>> = lengths
                .iter()
                .map(|&c| device.device_local_buffer(4 * c as u64).expect("a buffer"))
                .collect();
            let mut staging: Vec<Buffer<'_>> = inputs
                .iter()
                .map(|x| {
                    device
                        .staging_buffer(4 * x.len() as u64)
                        .expect("a staging buffer")
                })
                .collect();
            for (buffer, x) in staging.iter_mut().zip(&inputs) {
                buffer.write_f32(x).expect("written");
            }
            let mut recording = device.record().expect("a recording");
            for (from, to) in staging.iter().zip(&buffers) {
                recording.copy(from, to).expect("an upload");
            }
            recording.barrier(Barrier::TransferToCompute);
            let recorded = recording.finish().expect("ends");
            device
                .submit(recorded)
                .expect("submitted")
                .wait()
                .expect("uploads");
            let pool = device.timestamp_pool(5).expect("a pool");
            let mut times: Vec<[f64; 4]> = Vec::new();
            for round in 0..46 {
                let mut recording = device.record().expect("a recording");
                recording.reset_timestamps(&pool);
                recording.timestamp(&pool, 0, true).expect("a stamp");
                for (at, (kernel, pipeline)) in kernels.iter().zip(&pipelines).enumerate() {
                    let reads = if at == 0 { 4 } else { 5 };
                    let mut bound: Vec<&Buffer<'_>> = buffers[..reads].iter().collect();
                    bound.push(&buffers[5]);
                    bound.push(&buffers[6 + at]);
                    recording
                        .dispatch_kernel(pipeline, kernel, &bound)
                        .expect("a dispatch");
                    recording.barrier(Barrier::ComputeToCompute);
                    recording
                        .timestamp(&pool, 1 + at as u32, true)
                        .expect("a stamp");
                }
                let recorded = recording.finish().expect("ends");
                device
                    .submit(recorded)
                    .expect("submitted")
                    .wait()
                    .expect("runs");
                let stamps = pool.read().expect("stamps");
                if round >= 5 {
                    let mut t = [0.0; 4];
                    for i in 0..4 {
                        let ticks = stamps[i + 1].wrapping_sub(stamps[i]);
                        t[i] = ticks as f64 * f64::from(clock.period_ns) / 1000.0;
                    }
                    times.push(t);
                }
            }
            for (i, op) in names.iter().enumerate() {
                let mut col: Vec<f64> = times.iter().map(|t| t[i]).collect();
                col.sort_by(|a, b| a.partial_cmp(b).unwrap());
                println!(
                    "ADV-TIME {tag} {name} {op} median_us {:.1}",
                    col[col.len() / 2]
                );
            }
            let mut total: Vec<f64> = times.iter().map(|t| t.iter().sum()).collect();
            total.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "ADV-TIME {tag} {name} all4 median_us {:.1}",
                total[total.len() / 2]
            );
        }
    }
    // ADV-TIMING-END
}
