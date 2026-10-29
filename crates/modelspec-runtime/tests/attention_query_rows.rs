//! `attention` with fewer query rows per sequence than key rows on the CPU interpreter
//! (DESIGN.md § 2.2, § 4; story:attention-fewer-query-rows).
//!
//! `q` has `B * L` rows, `k` and `v` have `B * S` and the key mask is `[B, S]`, `1 <= L <= S`:
//! query `i` of sequence `b` is row `b * L + i` and attends to the unmasked keys of rows `b * S`
//! to `b * S + S - 1`. With `L = 1` the result for each sequence is row 0 of the `L = S` result
//! bit for bit, forward and the three gradients (`AttentionGradK` and `AttentionGradV` of `L = S`
//! with `G` zero in every row but the first of each sequence, as when the loss reads only that
//! row), and the gradients of the new form agree with central differences.

use std::collections::BTreeMap;

use modelspec_core::autodiff::differentiate;
use modelspec_core::ir::tensor::{TensorOp, TensorProgramBuilder};
use modelspec_runtime::cpu::{Element, HostTensor, evaluate};

#[allow(dead_code)]
#[path = "common/finite_difference.rs"]
mod finite_difference;

use finite_difference::{
    H as STEP, Ids, MIN_SCALE, Point, Rng, TOL_F32, TOL_F64, input, loss_at, mse_to_target, param,
    relative_error, run, seed_of, ty,
};

/// Sequences, keys per sequence, width and heads of the bit cases; each head reads 3 columns.
const B: usize = 3;
const S: usize = 5;
const H: usize = 6;
const HEADS: u64 = 2;

/// The key mask of the bit cases: sequence 0 keeps every key, sequence 1 masks its first key,
/// sequence 2 keeps only its keys 1 and 3.
const MASK: [f32; B * S] = [
    1.0, 1.0, 1.0, 1.0, 1.0, //
    0.0, 1.0, 1.0, 1.0, 1.0, //
    0.0, 1.0, 0.0, 1.0, 0.0,
];

/// `n` values drawn from `rng`.
fn draw(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.value()).collect()
}

/// The interpreter's result in `T` of `op` (`Attention` over `q, k, v, key_mask`, a gradient op
/// over those and `g`), `q` and `g` of `rows` rows, `k` and `v` of `B * S`.
fn evaluate_op<T: Element>(
    op: &TensorOp,
    rows: usize,
    values: [&[f32]; 5],
    convert: impl Fn(f32) -> T,
) -> Vec<T> {
    let (h, keys) = (H as u64, (B * S) as u64);
    let shapes: [Vec<u64>; 5] = [
        vec![rows as u64, h],
        vec![keys, h],
        vec![keys, h],
        vec![B as u64, S as u64],
        vec![rows as u64, h],
    ];
    let arity = if matches!(op, TensorOp::Attention { .. }) {
        4
    } else {
        5
    };
    let names = ["q", "k", "v", "key_mask", "g"];
    let mut b = TensorProgramBuilder::new();
    let operands: Vec<_> = (0..arity)
        .map(|at| input(&mut b, names[at], &shapes[at]))
        .collect();
    let y = b.push(op.clone(), &operands).unwrap();
    b.output(y).unwrap();
    let program = b.finish();
    let inputs: BTreeMap<String, HostTensor<T>> = (0..arity)
        .map(|at| {
            let data = values[at].iter().map(|&x| convert(x)).collect();
            let tensor = HostTensor::new(shapes[at].clone(), data).unwrap();
            (names[at].to_owned(), tensor)
        })
        .collect();
    let mut out = evaluate(&program, &inputs, &BTreeMap::new()).unwrap();
    out.remove(0).data().to_vec()
}

/// Rows `b * S` of `x`, one per sequence.
fn first_rows(x: &[f32]) -> Vec<f32> {
    (0..B)
        .flat_map(|b| x[b * S * H..b * S * H + H].to_vec())
        .collect()
}

/// `x` with every row but the first of each sequence `0.0`.
fn only_first_rows(x: &[f32]) -> Vec<f32> {
    x.chunks(H)
        .enumerate()
        .flat_map(|(row, values)| {
            if row % S == 0 {
                values.to_vec()
            } else {
                vec![0.0; H]
            }
        })
        .collect()
}

/// The bits of `values`, each widened to 64 bits.
fn bits<T: Element + Into<f64>>(values: &[T]) -> Vec<u64> {
    values
        .iter()
        .map(|&v| Into::<f64>::into(v).to_bits())
        .collect()
}

/// The `L = 1` results of `Attention`, `AttentionGradQ`, `AttentionGradK` and `AttentionGradV`
/// in `T` against the `L = S` ones: the first two equal row 0 of each sequence of the `L = S`
/// result, the last two the whole `L = S` result under a `G` that is zero but in row 0 of each
/// sequence; each in bits.
fn one_query_row_is_row_0_of_every_row<T: Element + Into<f64>>(
    what: &str,
    convert: impl Fn(f32) -> T + Copy,
) {
    let mut rng = Rng::new(seed_of(what));
    let n = B * S * H;
    let (q, k, v, g) = (
        draw(&mut rng, n),
        draw(&mut rng, n),
        draw(&mut rng, n),
        draw(&mut rng, n),
    );
    let (q1, g1, g0) = (first_rows(&q), first_rows(&g), only_first_rows(&g));
    let ops = [
        TensorOp::Attention { heads: HEADS },
        TensorOp::AttentionGradQ { heads: HEADS },
        TensorOp::AttentionGradK { heads: HEADS },
        TensorOp::AttentionGradV { heads: HEADS },
    ];
    for op in ops {
        let one = evaluate_op(&op, B, [&q1, &k, &v, &MASK, &g1], convert);
        let per_query_row = matches!(
            op,
            TensorOp::Attention { .. } | TensorOp::AttentionGradQ { .. }
        );
        let (every, want) = if per_query_row {
            let every = evaluate_op(&op, B * S, [&q, &k, &v, &MASK, &g], convert);
            let rows: Vec<T> = (0..B)
                .flat_map(|b| every[b * S * H..b * S * H + H].to_vec())
                .collect();
            (every.len(), rows)
        } else {
            let every = evaluate_op(&op, B * S, [&q, &k, &v, &MASK, &g0], convert);
            (every.len(), every)
        };
        assert_eq!(
            one.len(),
            if per_query_row { B * H } else { every },
            "{what} {}",
            op.name()
        );
        assert_eq!(bits(&one), bits(&want), "{what} {}", op.name());
        assert!(
            one.iter().any(|&x| Into::<f64>::into(x) != 0.0),
            "{what} {}: all zero",
            op.name()
        );
    }
}

#[test]
fn one_query_row_per_sequence_is_row_0_of_every_row_in_f32() {
    one_query_row_is_row_0_of_every_row::<f32>("one_query_row_f32", |x| x);
}

#[test]
fn one_query_row_per_sequence_is_row_0_of_every_row_in_f64() {
    one_query_row_is_row_0_of_every_row::<f64>("one_query_row_f64", f64::from);
}

/// `mse(attention(matmul(x_q, w_q), matmul(x, w_k), matmul(x, w_v), key_mask, heads = 2), t)`
/// with `L` query rows per sequence, `B = 2`, `S = 3`, `H = 4` and the key mask hiding the last
/// key of sequence 1: the adjoint of `q` has `q`'s `B * L` rows and those of `k` and `v` the
/// `B * S` rows of `k` and `v`, and the gradients of `w_q`, `w_k` and `w_v` are within `TOL_F64`
/// of central differences in `f64` and `TOL_F32` in `f32`, for `L = 1` and `L = 2`.
#[test]
fn fewer_query_rows_gradients_match_central_differences() {
    const B: u64 = 2;
    const S: u64 = 3;
    const H: u64 = 4;
    const D: u64 = 3;
    for l in [1, 2] {
        let name = format!("fewer_query_rows_gradients_{l}");
        let mut b = TensorProgramBuilder::new();
        let x_q = input(&mut b, "x_q", &[B * l, D]);
        let x = input(&mut b, "x", &[B * S, D]);
        let mask = input(&mut b, "key_mask", &[B, S]);
        let w_q = param(&mut b, "w_q", &[D, H]);
        let w_k = param(&mut b, "w_k", &[D, H]);
        let w_v = param(&mut b, "w_v", &[D, H]);
        let q = b.matmul(x_q, w_q).unwrap();
        let k = b.matmul(x, w_k).unwrap();
        let v = b.matmul(x, w_v).unwrap();
        let y = b.attention(q, k, v, mask, 2).unwrap();
        assert_eq!(b.type_of(y), Some(&ty(&[B * l, H])));
        let loss = mse_to_target(&mut b, y);
        let forward = b.finish();
        let training = differentiate(&forward, loss).unwrap();
        let program = &training.program;
        for (grad, rows) in [
            (TensorOp::AttentionGradQ { heads: 2 }, B * l),
            (TensorOp::AttentionGradK { heads: 2 }, B * S),
            (TensorOp::AttentionGradV { heads: 2 }, B * S),
        ] {
            let node = program
                .nodes
                .iter()
                .find(|n| n.op == grad)
                .unwrap_or_else(|| panic!("{name}: no {grad:?}"));
            assert_eq!(node.ty, ty(&[rows, H]), "{name}: {grad:?}");
        }

        let mut rng = Rng::new(seed_of(&name));
        let mut point = Point::draw(&forward, &Ids::new(), &mut rng);
        point.inputs.insert(
            "key_mask".into(),
            (vec![B, S], vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        );
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
            assert!(scale >= MIN_SCALE, "{name} {path}: {differences:?}");
            let g64 = in_f64[1 + k_param].data();
            let g32: Vec<f64> = in_f32[1 + k_param]
                .data()
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            let e64 = relative_error(g64, &differences);
            let e32 = relative_error(&g32, &differences);
            assert!(e64 <= TOL_F64, "{name} {path}: f64 error {e64:e}");
            assert!(e32 <= TOL_F32, "{name} {path}: f32 error {e32:e}");
        }
    }
}
