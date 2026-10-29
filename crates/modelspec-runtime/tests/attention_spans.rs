//! `attention` over `(start, length)` spans on the CPU interpreter (DESIGN.md § 2.2, § 4;
//! story:attention-event-spans).
//!
//! The fourth operand is an `i32 [U, 2]` tensor of spans in place of the `f32 [B, S]` key mask:
//! span `u` is rows `start .. start + length` of `k` and `v`, one sequence laid out back to back,
//! the spans in ascending row order without overlap. With `q` of the rows of `k` query row `i`
//! attends to the keys of the span that holds it, and a row outside every span gives `+0`; with
//! `q` of `U` rows query `u` attends to the keys of span `u` (one query per span).
//!
//! Every span row of the result is, bit for bit, the row of today's prefix-masked attention on the
//! padded layout that holds the same token (sequence `u` of `S` rows, the first `length` kept),
//! for `Attention` and `AttentionGradQ`; `AttentionGradK` and `AttentionGradV` equal it as values
//! under a `G` that is zero in the padded layout's masked rows (a zero term may turn a `-0.0` sum
//! into `+0.0`). Rows outside every span are `+0` whatever `q`, `k`, `v` and `G` hold there. One
//! query per span gives the bits of the span's first row. A negative, empty, out-of-range,
//! overlapping or out-of-order span is refused by name before the op runs, and the gradients of
//! both forms agree with central differences.

use std::collections::BTreeMap;

use modelspec_core::ir::tensor::{TensorOp, TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::{Element, HostTensor, evaluate_with_ids};

#[allow(dead_code)]
#[path = "common/finite_difference.rs"]
mod finite_difference;

use finite_difference::{
    Ids, Rng, gradient_check_with_ids, id_input, input, mse_to_target, param, seed_of,
};

/// Keys per sequence of the padded layout, width and heads; each head reads 3 columns.
const S: usize = 5;
const H: usize = 6;
const HEADS: u64 = 2;

/// The spans of the bit cases, `(start, length)`, in 17 rows: one row before the first, two
/// between the first and the second, none between the second and the third, one before the last
/// and two after it.
const SPANS: [(usize, usize); 4] = [(1, 5), (8, 1), (9, 3), (13, 2)];
const ROWS: usize = 17;
const U: usize = SPANS.len();

/// `n` values drawn from `rng`.
fn draw(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.value()).collect()
}

/// The spans as the `i32` tensor's data.
fn span_ids(spans: &[(usize, usize)]) -> Vec<i32> {
    spans
        .iter()
        .flat_map(|&(start, length)| [start as i32, length as i32])
        .collect()
}

/// The prefix key mask of the padded layout: sequence `u` keeps its first `length` keys.
fn padded_mask() -> Vec<f32> {
    SPANS
        .iter()
        .flat_map(|&(_, length)| (0..S).map(move |j| if j < length { 1.0 } else { 0.0 }))
        .collect()
}

/// The span row that holds padded row `r`, if `r` is kept.
fn span_row(r: usize) -> Option<usize> {
    let (u, j) = (r / S, r % S);
    let (start, length) = SPANS[u];
    (j < length).then_some(start + j)
}

/// Whether row `i` of the span layout lies in a span.
fn in_span(i: usize) -> bool {
    SPANS
        .iter()
        .any(|&(start, length)| (start..start + length).contains(&i))
}

/// The span layout of the padded `x` (rows of `H`): each kept padded row at its span row, every
/// other row from `fill`.
fn compact(x: &[f32], fill: &[f32]) -> Vec<f32> {
    let mut out = fill.to_vec();
    for r in 0..U * S {
        if let Some(i) = span_row(r) {
            out[i * H..(i + 1) * H].copy_from_slice(&x[r * H..(r + 1) * H]);
        }
    }
    out
}

/// `x` with every masked padded row `0.0`.
fn zero_masked(x: &[f32]) -> Vec<f32> {
    x.chunks(H)
        .enumerate()
        .flat_map(|(r, row)| {
            if span_row(r).is_some() {
                row.to_vec()
            } else {
                vec![0.0; H]
            }
        })
        .collect()
}

/// The interpreter's result in `T` of `op` over `q` of `q_rows` rows, `k`, `v` of `k_rows`, the
/// fourth operand (`f32` mask data, or with `spans` the `i32` spans) of `fourth` shape, and `g`
/// of `q`'s shape for a gradient op.
#[allow(clippy::too_many_arguments)]
fn evaluate_op<T: Element>(
    op: &TensorOp,
    q_rows: usize,
    k_rows: usize,
    fourth: &[u64],
    values: [&[f32]; 4],
    spans: Option<&[i32]>,
    convert: impl Fn(f32) -> T,
) -> Result<Vec<T>, String> {
    let shapes: [Vec<u64>; 5] = [
        vec![q_rows as u64, H as u64],
        vec![k_rows as u64, H as u64],
        vec![k_rows as u64, H as u64],
        fourth.to_vec(),
        vec![q_rows as u64, H as u64],
    ];
    let arity = if matches!(op, TensorOp::Attention { .. }) {
        4
    } else {
        5
    };
    let names = ["q", "k", "v", "keys", "g"];
    let mut b = TensorProgramBuilder::new();
    let operands: Vec<_> = (0..arity)
        .map(|at| {
            let ty = if at == 3 && spans.is_some() {
                TensorType::i32(&shapes[at])
            } else {
                TensorType::f32(&shapes[at])
            };
            b.input(names[at], ty).unwrap()
        })
        .collect();
    let y = b.push(op.clone(), &operands).map_err(|e| e.to_string())?;
    b.output(y).unwrap();
    let program = b.finish();
    let floats = [values[0], values[1], values[2], &[], values[3]];
    let mut inputs: BTreeMap<String, HostTensor<T>> = BTreeMap::new();
    let mut ids: BTreeMap<String, HostTensor<i32>> = BTreeMap::new();
    for at in 0..arity {
        if at == 3 {
            if let Some(spans) = spans {
                ids.insert(
                    names[at].to_owned(),
                    HostTensor::new(shapes[at].clone(), spans.to_vec()).unwrap(),
                );
            }
            continue;
        }
        let data = floats[at].iter().map(|&x| convert(x)).collect();
        inputs.insert(
            names[at].to_owned(),
            HostTensor::new(shapes[at].clone(), data).unwrap(),
        );
    }
    if spans.is_none() {
        let mask = padded_mask().into_iter().map(&convert).collect();
        inputs.insert(
            "keys".to_owned(),
            HostTensor::new(shapes[3].clone(), mask).unwrap(),
        );
    }
    let mut out = evaluate_with_ids(&program, &inputs, &ids, &BTreeMap::new())
        .map_err(|e| format!("{e:?}"))?;
    Ok(out.remove(0).data().to_vec())
}

/// The bits of `values`, each widened to 64 bits.
fn bits<T: Element + Into<f64>>(values: &[T]) -> Vec<u64> {
    values
        .iter()
        .map(|&v| Into::<f64>::into(v).to_bits())
        .collect()
}

const OPS: [TensorOp; 4] = [
    TensorOp::Attention { heads: HEADS },
    TensorOp::AttentionGradQ { heads: HEADS },
    TensorOp::AttentionGradK { heads: HEADS },
    TensorOp::AttentionGradV { heads: HEADS },
];

/// Whether `op` writes one row per query.
fn per_query(op: &TensorOp) -> bool {
    matches!(
        op,
        TensorOp::Attention { .. } | TensorOp::AttentionGradQ { .. }
    )
}

/// The span form of every op in `T` against the padded prefix-masked form: span rows in bits
/// (`Attention`, `AttentionGradQ`) or as values (`AttentionGradK`, `AttentionGradV` under a `G`
/// zero in the masked padded rows), rows outside every span `+0` in bits.
fn span_rows_are_the_padded_rows<T: Element + Into<f64>>(
    what: &str,
    convert: impl Fn(f32) -> T + Copy,
) {
    let mut rng = Rng::new(seed_of(what));
    let n = U * S * H;
    let (q, k, v, g) = (
        draw(&mut rng, n),
        draw(&mut rng, n),
        draw(&mut rng, n),
        draw(&mut rng, n),
    );
    let fill: Vec<Vec<f32>> = (0..4).map(|_| draw(&mut rng, ROWS * H)).collect();
    let (cq, ck, cv, cg) = (
        compact(&q, &fill[0]),
        compact(&k, &fill[1]),
        compact(&v, &fill[2]),
        compact(&g, &fill[3]),
    );
    let spans = span_ids(&SPANS);
    let mask_shape = [U as u64, S as u64];
    let span_shape = [U as u64, 2];
    let g0 = zero_masked(&g);
    for op in OPS {
        let label = format!("{what} {}", op.name());
        let span = evaluate_op(
            &op,
            ROWS,
            ROWS,
            &span_shape,
            [&cq, &ck, &cv, &cg],
            Some(&spans),
            convert,
        )
        .unwrap_or_else(|e| panic!("{label}: {e}"));
        let g_padded = if per_query(&op) { &g } else { &g0 };
        let padded = evaluate_op(
            &op,
            U * S,
            U * S,
            &mask_shape,
            [&q, &k, &v, g_padded],
            None,
            convert,
        )
        .unwrap();
        assert_eq!(span.len(), ROWS * H, "{label}");
        for r in 0..U * S {
            let Some(i) = span_row(r) else { continue };
            let (got, want) = (&span[i * H..(i + 1) * H], &padded[r * H..(r + 1) * H]);
            if per_query(&op) {
                assert_eq!(
                    bits(got),
                    bits(want),
                    "{label}: span row {i}, padded row {r}"
                );
            } else {
                let as_values =
                    |x: &[T]| x.iter().map(|&v| Into::<f64>::into(v)).collect::<Vec<_>>();
                assert_eq!(
                    as_values(got),
                    as_values(want),
                    "{label}: span row {i}, padded row {r}"
                );
            }
        }
        for i in (0..ROWS).filter(|&i| !in_span(i)) {
            assert_eq!(
                bits(&span[i * H..(i + 1) * H]),
                vec![0u64; H],
                "{label}: row {i} lies in no span"
            );
        }
        assert!(
            span.iter().any(|&x| Into::<f64>::into(x) != 0.0),
            "{label}: all zero"
        );
    }
}

#[test]
fn span_rows_are_the_bits_of_the_padded_prefix_mask_in_f32() {
    span_rows_are_the_padded_rows::<f32>("span_rows_f32", |x| x);
}

#[test]
fn span_rows_are_the_bits_of_the_padded_prefix_mask_in_f64() {
    span_rows_are_the_padded_rows::<f64>("span_rows_f64", f64::from);
}

/// One query per span (`q` of `U` rows) against the span form over every row: `Attention` and
/// `AttentionGradQ` give, in bits, the row of each span's first row; `AttentionGradK` and
/// `AttentionGradV` equal as values the result over every row under a `G` that is zero but in
/// each span's first row; rows outside every span stay `+0`.
#[test]
fn one_query_per_span_is_the_first_row_of_each_span() {
    let mut rng = Rng::new(seed_of("one_query_per_span"));
    let n = ROWS * H;
    let (q, k, v, g) = (
        draw(&mut rng, n),
        draw(&mut rng, n),
        draw(&mut rng, n),
        draw(&mut rng, n),
    );
    let first_rows = |x: &[f32]| -> Vec<f32> {
        SPANS
            .iter()
            .flat_map(|&(start, _)| x[start * H..(start + 1) * H].to_vec())
            .collect()
    };
    let only_first = |x: &[f32]| -> Vec<f32> {
        x.chunks(H)
            .enumerate()
            .flat_map(|(i, row)| {
                if SPANS.iter().any(|&(start, _)| start == i) {
                    row.to_vec()
                } else {
                    vec![0.0; H]
                }
            })
            .collect()
    };
    let (q1, g1, g0) = (first_rows(&q), first_rows(&g), only_first(&g));
    let spans = span_ids(&SPANS);
    let shape = [U as u64, 2];
    for op in OPS {
        let label = op.name();
        let one = evaluate_op(
            &op,
            U,
            ROWS,
            &shape,
            [&q1, &k, &v, &g1],
            Some(&spans),
            |x| x,
        )
        .unwrap_or_else(|e| panic!("{label}: {e}"));
        if per_query(&op) {
            let every = evaluate_op(
                &op,
                ROWS,
                ROWS,
                &shape,
                [&q, &k, &v, &g],
                Some(&spans),
                |x| x,
            )
            .unwrap();
            assert_eq!(bits(&one), bits(&first_rows(&every)), "{label}");
        } else {
            let every = evaluate_op(
                &op,
                ROWS,
                ROWS,
                &shape,
                [&q, &k, &v, &g0],
                Some(&spans),
                |x| x,
            )
            .unwrap();
            assert_eq!(one, every, "{label}");
            for i in (0..ROWS).filter(|&i| !in_span(i)) {
                assert_eq!(
                    bits(&one[i * H..(i + 1) * H]),
                    vec![0u64; H],
                    "{label}: row {i}"
                );
            }
        }
        assert!(one.iter().any(|&x| x != 0.0), "{label}: all zero");
    }
}

/// The name of the error `spans` give the forward op over 17 rows: the `Debug` form's variant.
fn refusal(spans: &[i32]) -> String {
    let u = spans.len() / 2;
    let mut rng = Rng::new(seed_of("refusal"));
    let (q, k, v) = (
        draw(&mut rng, ROWS * H),
        draw(&mut rng, ROWS * H),
        draw(&mut rng, ROWS * H),
    );
    let op = TensorOp::Attention { heads: HEADS };
    match evaluate_op(
        &op,
        ROWS,
        ROWS,
        &[u as u64, 2],
        [&q, &k, &v, &[]],
        Some(spans),
        |x| x,
    ) {
        Ok(_) => "accepted".to_owned(),
        Err(e) => e,
    }
}

/// A negative start or length, an empty span, a span past the rows, one that starts inside the
/// span before it and one that starts before it are each refused before the op runs, by name and
/// span; the bit cases' spans are accepted.
#[test]
fn bad_spans_are_refused_by_name() {
    let cases: [(&str, Vec<i32>, &str, &str); 7] = [
        (
            "negative start",
            vec![1, 5, -1, 2],
            "SpanNegative",
            "span: 1",
        ),
        (
            "negative length",
            vec![1, 5, 8, -1],
            "SpanNegative",
            "span: 1",
        ),
        ("empty", vec![1, 5, 8, 0], "SpanEmpty", "span: 1"),
        (
            "past the rows",
            vec![1, 5, 16, 2],
            "SpanOutOfRange",
            "span: 1",
        ),
        ("overlapping", vec![1, 5, 5, 2], "SpanOverlap", "span: 1"),
        ("out of order", vec![8, 2, 1, 5], "SpanOverlap", "span: 1"),
        (
            "first past the rows",
            vec![17, 1],
            "SpanOutOfRange",
            "span: 0",
        ),
    ];
    for (what, spans, name, at) in cases {
        let got = refusal(&spans);
        assert!(got.starts_with(name), "{what}: {got}");
        assert!(got.contains(at), "{what}: {got}");
    }
    assert_eq!(refusal(&span_ids(&SPANS)), "accepted");
}

/// The span operand must be `i32 [U, 2]` with `q` of the rows of `k` or of `U` rows; any other
/// `q` or span shape is a shape mismatch, and an `f32 [U, 2]` operand is the mask form.
#[test]
fn span_shapes_outside_the_rule_are_refused() {
    let typed = |q_rows: u64, spans: &[u64]| {
        let mut b = TensorProgramBuilder::new();
        let q = b.input("q", TensorType::f32(&[q_rows, H as u64])).unwrap();
        let k = b
            .input("k", TensorType::f32(&[ROWS as u64, H as u64]))
            .unwrap();
        let s = b.input("s", TensorType::i32(spans)).unwrap();
        b.attention(q, k, k, s, HEADS)
            .map(|y| b.type_of(y).cloned())
    };
    let (n, u) = (ROWS as u64, U as u64);
    assert_eq!(
        typed(n, &[u, 2]).unwrap(),
        Some(TensorType::f32(&[n, H as u64]))
    );
    assert_eq!(
        typed(u, &[u, 2]).unwrap(),
        Some(TensorType::f32(&[u, H as u64]))
    );
    for (q_rows, spans) in [
        (u + 1, vec![u, 2]),
        (n, vec![u, 3]),
        (n, vec![u * 2]),
        (n, vec![0, 2]),
    ] {
        let err = typed(q_rows, &spans).unwrap_err().to_string();
        assert!(err.contains("Attention"), "{q_rows} {spans:?}: {err}");
    }
}

/// `mse(attention(matmul(x_q, w_q), matmul(x, w_k), matmul(x, w_v), spans, heads = 2), t)` over
/// the bit cases' spans, `x_q` the rows of `x` (every row a query) or one row per span: the
/// gradients of `w_q`, `w_k` and `w_v` are within tolerance of central differences in `f64` and
/// `f32`.
#[test]
fn span_gradients_match_central_differences() {
    const D: u64 = 3;
    let (n, u) = (ROWS as u64, U as u64);
    for q_rows in [n, u] {
        let name = format!("span_gradients_{q_rows}");
        let ids: Ids = BTreeMap::from([("spans".to_owned(), (vec![u, 2], span_ids(&SPANS)))]);
        gradient_check_with_ids(&name, &ids, |b| {
            let x_q = input(b, "x_q", &[q_rows, D]);
            let x = input(b, "x", &[n, D]);
            let spans = id_input(b, "spans", &[u, 2]);
            let w_q = param(b, "w_q", &[D, H as u64]);
            let w_k = param(b, "w_k", &[D, H as u64]);
            let w_v = param(b, "w_v", &[D, H as u64]);
            let q = b.matmul(x_q, w_q).unwrap();
            let k = b.matmul(x, w_k).unwrap();
            let v = b.matmul(x, w_v).unwrap();
            let y = b.attention(q, k, v, spans, HEADS).unwrap();
            mse_to_target(b, y)
        });
    }
}
