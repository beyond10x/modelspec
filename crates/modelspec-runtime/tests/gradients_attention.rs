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
