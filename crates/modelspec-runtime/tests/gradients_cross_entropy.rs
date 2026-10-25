//! Weighted cross-entropy over integer labels on the CPU interpreter (DESIGN.md § 2.2, § 4, § 9
//! gate 3; story:bert-mlm-cross-entropy).
//!
//! `cross_entropy(logits, labels, weights)` takes `logits: f32[N, V]`, `labels: i32[N]` and
//! `weights: f32[N]` and is `sum_n weights[n] * loss_n / sum_n weights[n]`, `loss_n` the
//! max-subtracted log-sum-exp of row `n` less its logit at `labels[n]` (coordinator decision
//! D43, 2026-10-25, `.engineering/waves/bert-1.md`). A weight sum of `0.0` gives the loss `0.0`
//! and a zero gradient (D30). Its gradient with respect to `logits` is the internal
//! `CrossEntropyGrad(logits, labels, weights, g)`; `labels` and `weights` have none. The weights
//! are fixed while the parameters are perturbed, so each finite-difference case draws its point
//! with the harness of `tests/common/finite_difference.rs` and then sets the weights.

use std::collections::BTreeMap;

use modelspec_core::autodiff::differentiate;
use modelspec_core::ir::tensor::{
    ParamId, TensorOp, TensorProgram, TensorProgramBuilder, TensorType, ValueId,
};
use modelspec_core::ir::training::TrainingProgram;
use modelspec_runtime::cpu::{Element, HostTensor, RuntimeError, evaluate_with_ids};

#[allow(dead_code)]
#[path = "common/finite_difference.rs"]
mod finite_difference;

use finite_difference::{
    H as STEP, Ids, MIN_SCALE, Point, Rng, TOL_F32, TOL_F64, assert_structure, id_input, input,
    loss_at, param, relative_error, run, seed_of,
};

/// Rows: the selected and unselected positions of a batch.
const N: u64 = 6;
/// Classes: the vocabulary.
const V: u64 = 5;
/// Width of the input of the `matmul` that makes the logits.
const D: u64 = 3;
/// Width of the embedding table of the tied decoder.
const H: u64 = 4;

/// The weights of scenarios 2 and 3: rows 1 and 4 are not selected.
const WEIGHTS: [f32; 6] = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
/// A label per row, every class among them.
const LABELS: [i32; 6] = [0, 4, 2, 1, 3, 2];
/// The token ids of the tied decoder.
const IDS: [i32; 6] = [1, 3, 0, 4, 2, 3];

/// The `i32` inputs of a case, by name.
fn ids(entries: &[(&str, &[i32])]) -> Ids {
    entries
        .iter()
        .map(|&(name, data)| (name.to_owned(), (vec![data.len() as u64], data.to_vec())))
        .collect()
}

/// Draws a point of `forward` with the `i32` inputs `given` and sets the input `weights` to
/// `weights`.
fn point_with_weights(forward: &TensorProgram, given: &Ids, name: &str, weights: &[f32]) -> Point {
    let mut rng = Rng::new(seed_of(name));
    let mut point = Point::draw(forward, given, &mut rng);
    point
        .inputs
        .insert("weights".into(), (vec![N], weights.to_vec()));
    point
}

/// Every parameter's gradient of `training`, in `f64` and in `f32`, is within `TOL_F64` and
/// `TOL_F32` of central differences of the loss in `f64` at `point`.
fn assert_gradients_match_differences(
    forward: &TensorProgram,
    training: &TrainingProgram,
    point: &Point,
) {
    let program = &training.program;
    let in_f64 = run(training, point, f64::from);
    let in_f32 = run(training, point, |v| v);
    for (k, (&param, _)) in training.gradients.iter().enumerate() {
        let path = &forward.params[&param].path;
        let n = point.params[&param].1.len();
        let differences: Vec<f64> = (0..n)
            .map(|i| {
                (loss_at(program, point, param, i, STEP) - loss_at(program, point, param, i, -STEP))
                    / (2.0 * STEP)
            })
            .collect();
        let scale = differences.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(scale >= MIN_SCALE, "{path}: differences {differences:?}");
        let g64 = in_f64[1 + k].data();
        let g32: Vec<f64> = in_f32[1 + k].data().iter().map(|&v| f64::from(v)).collect();
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

/// Scenario 2: `cross_entropy(matmul(x, w), labels, weights)` with `N = 6`, `V = 5` and the
/// weights `[1, 0, 1, 1, 0, 1]`: the gradient of `w` is within `TOL_F64` of central differences
/// in `f64` and `TOL_F32` in `f32`, and the adjoint of the logits is one `CrossEntropyGrad` over
/// the forward operands and the seed.
#[test]
fn cross_entropy_gradient_of_w_matches_central_differences() {
    let name = "cross_entropy_gradient_of_w_matches_central_differences";
    let mut b = TensorProgramBuilder::new();
    let x = input(&mut b, "x", &[N, D]);
    let labels = id_input(&mut b, "labels", &[N]);
    let weights = input(&mut b, "weights", &[N]);
    let w = param(&mut b, "w", &[D, V]);
    let logits = b.matmul(x, w).unwrap();
    let loss = b.cross_entropy(logits, labels, weights).unwrap();
    assert_eq!(b.type_of(loss), Some(&TensorType::f32_scalar()));
    let forward = b.finish();
    let training = differentiate(&forward, loss).unwrap();
    assert_structure(&forward, loss, &training);
    assert_eq!(training.gradients.len(), 1, "a gradient for w alone");

    let program = &training.program;
    let dw = program.node(training.gradients[&ParamId(0)]).unwrap();
    assert_eq!(dw.op, TensorOp::MatMul, "dW = transpose(x) @ dZ");
    let dz = program.node(dw.operands[1]).unwrap();
    assert_eq!(dz.op, TensorOp::CrossEntropyGrad, "{dz:?}");
    assert_eq!(dz.operands[..3], [logits, labels, weights], "{dz:?}");
    assert_eq!(
        program.node(dz.operands[3]).unwrap().op,
        TensorOp::Scalar(1.0),
        "the adjoint of the loss is the seed"
    );

    let given = ids(&[("labels", &LABELS)]);
    let point = point_with_weights(&forward, &given, name, &WEIGHTS);
    assert_gradients_match_differences(&forward, &training, &point);
}

/// Scenario 3: the tied decoder
/// `cross_entropy(matmul(embedding(table, ids), transpose(table)), labels, weights)` reads the
/// one parameter `table` twice; its gradient is the sum of the adjoints of both reads, an
/// `EmbeddingGrad` and a `Transpose`, and passes the same check.
#[test]
fn tied_decoder_gradient_of_the_table_sums_both_reads() {
    let name = "tied_decoder_gradient_of_the_table_sums_both_reads";
    let mut b = TensorProgramBuilder::new();
    let tokens = id_input(&mut b, "ids", &[N]);
    let labels = id_input(&mut b, "labels", &[N]);
    let weights = input(&mut b, "weights", &[N]);
    let table = param(&mut b, "table", &[V, H]);
    let hidden = b.embedding(table, tokens).unwrap();
    let decoder = b.transpose(table).unwrap();
    let logits = b.matmul(hidden, decoder).unwrap();
    let loss = b.cross_entropy(logits, labels, weights).unwrap();
    let forward = b.finish();
    let training = differentiate(&forward, loss).unwrap();
    assert_structure(&forward, loss, &training);

    let program = &training.program;
    let dtable = program.node(training.gradients[&ParamId(0)]).unwrap();
    assert_eq!(
        dtable.op,
        TensorOp::Add,
        "the adjoints of the two reads add up"
    );
    let mut parts: Vec<&str> = dtable
        .operands
        .iter()
        .map(|&o| program.node(o).unwrap().op.name())
        .collect();
    parts.sort_unstable();
    assert_eq!(parts, ["EmbeddingGrad", "Transpose"], "{dtable:?}");

    let given = ids(&[("ids", &IDS), ("labels", &LABELS)]);
    let point = point_with_weights(&forward, &given, name, &WEIGHTS);
    assert_gradients_match_differences(&forward, &training, &point);
}

/// The inputs `logits: [N, V]`, `labels` and `weights` of `cross_entropy`, its value the output.
fn loss_program() -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let logits = input(&mut b, "logits", &[N, V]);
    let labels = id_input(&mut b, "labels", &[N]);
    let weights = input(&mut b, "weights", &[N]);
    let loss = b.cross_entropy(logits, labels, weights).unwrap();
    b.output(loss).unwrap();
    b.finish()
}

/// `CrossEntropyGrad(logits, labels, weights, g)` over the inputs of [`loss_program`] and a
/// rank-0 input `g`, its value the output.
fn grad_program() -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let logits = input(&mut b, "logits", &[N, V]);
    let labels = id_input(&mut b, "labels", &[N]);
    let weights = input(&mut b, "weights", &[N]);
    let g = input(&mut b, "g", &[]);
    let dz = b
        .push(TensorOp::CrossEntropyGrad, &[logits, labels, weights, g])
        .unwrap();
    b.output(dz).unwrap();
    b.finish()
}

/// Runs `program` in `T` with the `f32` inputs `floats` and the labels `labels`.
fn run_with<T: Element>(
    program: &TensorProgram,
    floats: &[(&str, &[u64], &[f32])],
    labels: &[i32],
    convert: impl Fn(f32) -> T,
) -> Result<Vec<HostTensor<T>>, RuntimeError> {
    let inputs: BTreeMap<String, HostTensor<T>> = floats
        .iter()
        .map(|&(name, shape, data)| {
            let data = data.iter().map(|&v| convert(v)).collect();
            (
                name.to_owned(),
                HostTensor::new(shape.to_vec(), data).unwrap(),
            )
        })
        .collect();
    let ids = BTreeMap::from([(
        "labels".to_owned(),
        HostTensor::new(vec![N], labels.to_vec()).unwrap(),
    )]);
    evaluate_with_ids(program, &inputs, &ids, &BTreeMap::new())
}

/// The loss of [`loss_program`] in `T`.
fn loss_of<T: Element>(
    logits: &[f32],
    labels: &[i32],
    weights: &[f32],
    convert: impl Fn(f32) -> T,
) -> Result<T, RuntimeError> {
    let floats: [(&str, &[u64], &[f32]); 2] =
        [("logits", &[N, V], logits), ("weights", &[N], weights)];
    run_with(&loss_program(), &floats, labels, convert).map(|out| out[0].data()[0])
}

/// The gradient of [`grad_program`] in `f32`.
fn grad_of(
    logits: &[f32],
    labels: &[i32],
    weights: &[f32],
    g: f32,
) -> Result<Vec<f32>, RuntimeError> {
    let floats: [(&str, &[u64], &[f32]); 3] = [
        ("logits", &[N, V], logits),
        ("weights", &[N], weights),
        ("g", &[], &[g]),
    ];
    run_with(&grad_program(), &floats, labels, |v| v).map(|out| out[0].data().to_vec())
}

/// `n` values drawn from `rng`.
fn draw(rng: &mut Rng, n: u64) -> Vec<f32> {
    (0..n).map(|_| rng.value()).collect()
}

/// Per row of `logits`: its maximum in ascending order and the sum of `exp(x - max)` in
/// ascending order from the first term, in `f32` (D43).
fn row_softmax(row: &[f32]) -> (f32, f32) {
    let max = row[1..]
        .iter()
        .fold(row[0], |m, &x| if x > m { x } else { m });
    let mut terms = row.iter().map(|&x| (x - max).exp());
    let first = terms.next().unwrap();
    (max, terms.fold(first, |acc, e| acc + e))
}

/// D43 written out in `f32`: per row `max + ln(sum) - logits[n, labels[n]]`; the numerator
/// `sum_n weights[n] * loss_n`, its first product rounded and each later one fused, and the
/// denominator `sum_n weights[n]`, both in ascending `n`; their quotient, `0.0` when the
/// denominator is `0.0`.
fn reference_loss(logits: &[f32], labels: &[i32], weights: &[f32]) -> f32 {
    let (mut numerator, mut denominator) = (0.0f32, 0.0f32);
    for (n, row) in logits.as_chunks::<{ V as usize }>().0.iter().enumerate() {
        let (max, total) = row_softmax(row);
        let loss = max + total.ln() - row[labels[n] as usize];
        if n == 0 {
            numerator = weights[0] * loss;
            denominator = weights[0];
        } else {
            numerator = weights[n].mul_add(loss, numerator);
            denominator += weights[n];
        }
    }
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// D43's gradient written out in `f32`: `g * weights[n] / sum(weights) * (p[n, v] - [v ==
/// labels[n]])`, left to right, `p` recomputed as `exp(x - max) / sum`; all `0.0` when the
/// weight sum is `0.0`.
fn reference_grad(logits: &[f32], labels: &[i32], weights: &[f32], g: f32) -> Vec<f32> {
    let mut terms = weights.iter().copied();
    let first = terms.next().unwrap();
    let total_weight = terms.fold(first, |acc, w| acc + w);
    let mut out = Vec::new();
    for (n, row) in logits.as_chunks::<{ V as usize }>().0.iter().enumerate() {
        let (max, total) = row_softmax(row);
        for (v, &x) in row.iter().enumerate() {
            if total_weight == 0.0 {
                out.push(0.0);
                continue;
            }
            let p = (x - max).exp() / total;
            let y = if v == labels[n] as usize { 1.0 } else { 0.0 };
            out.push(g * weights[n] / total_weight * (p - y));
        }
    }
    out
}

/// Scenario 4: uniform logits give `ln V` within `1e-6`, in `f32` and in `f64`.
#[test]
fn cross_entropy_values_uniform_logits_give_ln_v() {
    let logits = vec![0.7f32; (N * V) as usize];
    let ln_v = (V as f64).ln();
    let in_f32 = loss_of(&logits, &LABELS, &WEIGHTS, |v| v).unwrap();
    let in_f64 = loss_of(&logits, &LABELS, &WEIGHTS, f64::from).unwrap();
    assert!(
        (f64::from(in_f32) - ln_v).abs() <= 1e-6,
        "{in_f32} vs {ln_v}"
    );
    assert!((in_f64 - ln_v).abs() <= 1e-6, "{in_f64} vs {ln_v}");
}

/// Scenario 4: changing the logits of a row whose weight is 0 leaves the loss unchanged, bit for
/// bit, in `f32` and in `f64`.
#[test]
fn cross_entropy_values_ignore_the_logits_of_a_weight_0_row() {
    let mut rng = Rng::new(seed_of("cross_entropy_values_ignore"));
    let logits = draw(&mut rng, N * V);
    let mut changed = logits.clone();
    let v = V as usize;
    for (c, x) in changed[v..2 * v].iter_mut().enumerate() {
        *x = 40.0 * (c as f32 + 1.0) - 90.0;
    }
    for c in 0..v {
        changed[4 * v + c] = -7.5e2 + c as f32;
    }
    let before = loss_of(&logits, &LABELS, &WEIGHTS, |v| v).unwrap();
    let after = loss_of(&changed, &LABELS, &WEIGHTS, |v| v).unwrap();
    assert_eq!(before.to_bits(), after.to_bits(), "{before} vs {after}");
    let before = loss_of(&logits, &LABELS, &WEIGHTS, f64::from).unwrap();
    let after = loss_of(&changed, &LABELS, &WEIGHTS, f64::from).unwrap();
    assert_eq!(before.to_bits(), after.to_bits(), "{before} vs {after}");
    let gradient = grad_of(&changed, &LABELS, &WEIGHTS, 1.0).unwrap();
    for row in [1, 4] {
        assert!(
            gradient[row * v..(row + 1) * v].iter().all(|&x| x == 0.0),
            "row {row} of the gradient: {gradient:?}"
        );
    }
}

/// Scenario 4: logits of magnitude `1e3` give a finite loss, within `1e-5` relative of the
/// `f64` evaluation, and a finite gradient.
#[test]
fn cross_entropy_values_of_logits_of_magnitude_1e3_are_finite() {
    let logits: Vec<f32> = (0..N * V)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            sign * (1.0e3 - i as f32 * 0.75)
        })
        .collect();
    let in_f32 = loss_of(&logits, &LABELS, &WEIGHTS, |v| v).unwrap();
    let in_f64 = loss_of(&logits, &LABELS, &WEIGHTS, f64::from).unwrap();
    assert!(
        in_f32.is_finite() && in_f64.is_finite(),
        "{in_f32}, {in_f64}"
    );
    assert!(
        (f64::from(in_f32) - in_f64).abs() <= 1e-5 * in_f64.abs(),
        "{in_f32} vs {in_f64}"
    );
    let gradient = grad_of(&logits, &LABELS, &WEIGHTS, 1.0).unwrap();
    assert!(gradient.iter().all(|x| x.is_finite()), "{gradient:?}");
}

/// D43: the loss and the gradient are the bits of the order written out in
/// [`reference_loss`] and [`reference_grad`], over 50 draws of logits, labels, weights in
/// `[0, 2)` and an adjoint `g`.
#[test]
fn cross_entropy_values_follow_the_ascending_fused_order_bit_for_bit() {
    let mut rng = Rng::new(seed_of("cross_entropy_values_bits"));
    for draw_index in 0..50 {
        let logits: Vec<f32> = draw(&mut rng, N * V).iter().map(|&x| 4.0 * x).collect();
        let labels: Vec<i32> = (0..N).map(|_| (rng.next_u64() % V) as i32).collect();
        let weights: Vec<f32> = (0..N).map(|_| (2.0 * rng.unit()) as f32).collect();
        let g = rng.value();
        let loss = loss_of(&logits, &labels, &weights, |v| v).unwrap();
        let want = reference_loss(&logits, &labels, &weights);
        assert_eq!(
            loss.to_bits(),
            want.to_bits(),
            "draw {draw_index}: {loss} vs {want}"
        );
        let gradient = grad_of(&logits, &labels, &weights, g).unwrap();
        let want = reference_grad(&logits, &labels, &weights, g);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&gradient), bits(&want), "draw {draw_index}");
    }
}

/// Scenario 5, coordinator decision D30 (D43 corrects the story's list of refusals): all weights
/// 0 is no error; the loss is `+0.0` and the gradient of every logit `+0.0`, in the loss op, its
/// gradient op and a whole training step.
#[test]
fn cross_entropy_refusals_do_not_include_all_weights_0() {
    let mut rng = Rng::new(seed_of("cross_entropy_refusals_zero"));
    let logits = draw(&mut rng, N * V);
    let zeros = [0.0f32; N as usize];
    let loss = loss_of(&logits, &LABELS, &zeros, |v| v).unwrap();
    assert_eq!(loss.to_bits(), 0.0f32.to_bits(), "{loss}");
    let gradient = grad_of(&logits, &LABELS, &zeros, 1.0).unwrap();
    assert!(
        gradient.iter().all(|x| x.to_bits() == 0.0f32.to_bits()),
        "{gradient:?}"
    );

    let mut b = TensorProgramBuilder::new();
    let x = input(&mut b, "x", &[N, D]);
    let labels = id_input(&mut b, "labels", &[N]);
    let weights = input(&mut b, "weights", &[N]);
    let w = param(&mut b, "w", &[D, V]);
    let z = b.matmul(x, w).unwrap();
    let loss = b.cross_entropy(z, labels, weights).unwrap();
    let forward = b.finish();
    let training = differentiate(&forward, loss).unwrap();
    let given = ids(&[("labels", &LABELS)]);
    let point = point_with_weights(&forward, &given, "zero_step", &zeros);
    let outputs = run(&training, &point, |v| v);
    assert_eq!(outputs[0].data(), &[0.0f32], "loss");
    assert!(
        outputs[1].data().iter().all(|&x| x == 0.0),
        "dW: {:?}",
        outputs[1]
    );
}

/// The value the `CrossEntropy` node of [`loss_program`] defines, and the `CrossEntropyGrad`
/// node of [`grad_program`].
fn op_value(program: &TensorProgram) -> ValueId {
    program
        .nodes
        .iter()
        .find(|n| matches!(n.op, TensorOp::CrossEntropy | TensorOp::CrossEntropyGrad))
        .unwrap()
        .id
}

/// Scenario 5: a weight of `-1`, a NaN weight and an infinite weight are each a
/// `RuntimeError::LossWeight` naming the row, in the loss op and in its gradient op, not NaN or a
/// panic.
#[test]
fn cross_entropy_refusals_of_negative_and_non_finite_weights_name_the_row() {
    let mut rng = Rng::new(seed_of("cross_entropy_refusals_weights"));
    let logits = draw(&mut rng, N * V);
    for (row, bad) in [
        (2usize, -1.0f32),
        (0, f32::NAN),
        (5, f32::INFINITY),
        (3, f32::NEG_INFINITY),
    ] {
        let mut weights = WEIGHTS;
        weights[row] = bad;
        let error = loss_of(&logits, &LABELS, &weights, |v| v).unwrap_err();
        assert_eq!(
            error,
            RuntimeError::LossWeight {
                value: op_value(&loss_program()),
                row,
            },
            "{bad}"
        );
        assert!(error.to_string().contains(&format!("row {row}")), "{error}");
        assert_eq!(
            loss_of(&logits, &LABELS, &weights, f64::from).unwrap_err(),
            error,
            "{bad} in f64"
        );
        assert_eq!(
            grad_of(&logits, &LABELS, &weights, 1.0).unwrap_err(),
            RuntimeError::LossWeight {
                value: op_value(&grad_program()),
                row,
            },
            "{bad} in the gradient op"
        );
    }
    let mut weights = WEIGHTS;
    weights[1] = -0.0;
    assert!(
        loss_of(&logits, &LABELS, &weights, |v| v).is_ok(),
        "-0.0 is not negative"
    );
}

/// Scenario 5: a label of `V` or of `-1` is a `RuntimeError::LabelOutOfRange` naming the input,
/// the row, the label and the classes, in the loss op and in its gradient op.
#[test]
fn cross_entropy_refusals_of_labels_outside_the_classes_name_the_row() {
    let mut rng = Rng::new(seed_of("cross_entropy_refusals_labels"));
    let logits = draw(&mut rng, N * V);
    for (row, bad) in [(3usize, V as i32), (0, -1), (5, i32::MAX)] {
        let mut labels = LABELS;
        labels[row] = bad;
        let error = loss_of(&logits, &labels, &WEIGHTS, |v| v).unwrap_err();
        assert_eq!(
            error,
            RuntimeError::LabelOutOfRange {
                name: "labels".into(),
                value: op_value(&loss_program()),
                row,
                label: bad,
                classes: V,
            },
            "{bad}"
        );
        let text = error.to_string();
        assert!(
            text.contains(&format!("row {row}")) && text.contains("`labels`"),
            "{text}"
        );
        assert_eq!(
            grad_of(&logits, &labels, &WEIGHTS, 1.0).unwrap_err(),
            RuntimeError::LabelOutOfRange {
                name: "labels".into(),
                value: op_value(&grad_program()),
                row,
                label: bad,
                classes: V,
            },
            "{bad} in the gradient op"
        );
    }
}

/// The rows are checked in ascending order, the label of a row before its weight: the first bad
/// row is named.
#[test]
fn cross_entropy_refusals_name_the_first_bad_row() {
    let mut rng = Rng::new(seed_of("cross_entropy_refusals_first"));
    let logits = draw(&mut rng, N * V);
    let mut labels = LABELS;
    labels[4] = V as i32;
    let mut weights = WEIGHTS;
    weights[2] = -1.0;
    weights[4] = -1.0;
    assert!(matches!(
        loss_of(&logits, &labels, &weights, |v| v),
        Err(RuntimeError::LossWeight { row: 2, .. })
    ));
    weights[2] = 1.0;
    assert!(matches!(
        loss_of(&logits, &labels, &weights, |v| v),
        Err(RuntimeError::LabelOutOfRange { row: 4, .. })
    ));
}
