//! Adversary pass 1 of story:bert-mlm-cross-entropy: the numerics of `cross_entropy` and
//! `CrossEntropyGrad` on the CPU interpreter against an independent reference of coordinator
//! decision D43 (DESIGN.md § 4, "Cross-entropy"), on data chosen so that another summation order
//! gives other bits, and at the edges of the weights and logits.
//!
//! A case that fails on the unit as written is `#[ignore]`d with its finding id in its name.

use std::collections::BTreeMap;

use modelspec_core::autodiff::differentiate;
use modelspec_core::ir::tensor::{ParamId, TensorOp, TensorProgram, TensorProgramBuilder};
use modelspec_runtime::cpu::{Element, HostTensor, RuntimeError, evaluate_with_ids};

#[allow(dead_code)]
#[path = "common/finite_difference.rs"]
mod finite_difference;

use finite_difference::{Point, Rng, id_input, input, param, seed_of, tensors};

/// `u = 2^-24`, half an ulp of `1.0` in `f32`.
const U: f32 = 1.0 / 16_777_216.0;

/// The logit whose `exp` is about `0.7 u`: below half an ulp of 1 alone, above it twice.
fn small_logit() -> f32 {
    (0.7f32).ln() + U.ln()
}

/// `cross_entropy(logits: [n, v], labels, weights)`, its value the output.
fn loss_program(n: u64, v: u64) -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let logits = input(&mut b, "logits", &[n, v]);
    let labels = id_input(&mut b, "labels", &[n]);
    let weights = input(&mut b, "weights", &[n]);
    let loss = b.cross_entropy(logits, labels, weights).unwrap();
    b.output(loss).unwrap();
    b.finish()
}

/// `CrossEntropyGrad(logits, labels, weights, g)`, its value the output.
fn grad_program(n: u64, v: u64) -> TensorProgram {
    let mut b = TensorProgramBuilder::new();
    let logits = input(&mut b, "logits", &[n, v]);
    let labels = id_input(&mut b, "labels", &[n]);
    let weights = input(&mut b, "weights", &[n]);
    let g = input(&mut b, "g", &[]);
    let dz = b
        .push(TensorOp::CrossEntropyGrad, &[logits, labels, weights, g])
        .unwrap();
    b.output(dz).unwrap();
    b.finish()
}

struct Case<'a> {
    v: usize,
    logits: &'a [f32],
    labels: &'a [i32],
    weights: &'a [f32],
}

impl Case<'_> {
    fn n(&self) -> usize {
        self.labels.len()
    }

    fn run<T: Element>(
        &self,
        program: &TensorProgram,
        g: Option<f32>,
    ) -> Result<Vec<T>, RuntimeError> {
        let (n, v) = (self.n() as u64, self.v as u64);
        let convert = |d: &[f32]| d.iter().map(|&x| T::from_f32(x)).collect::<Vec<T>>();
        let mut inputs = BTreeMap::from([
            (
                "logits".to_owned(),
                HostTensor::new(vec![n, v], convert(self.logits)).unwrap(),
            ),
            (
                "weights".to_owned(),
                HostTensor::new(vec![n], convert(self.weights)).unwrap(),
            ),
        ]);
        if let Some(g) = g {
            inputs.insert(
                "g".to_owned(),
                HostTensor::new(vec![], vec![T::from_f32(g)]).unwrap(),
            );
        }
        let ids = BTreeMap::from([(
            "labels".to_owned(),
            HostTensor::new(vec![n], self.labels.to_vec()).unwrap(),
        )]);
        evaluate_with_ids(program, &inputs, &ids, &BTreeMap::new())
            .map(|out| out[0].data().to_vec())
    }

    fn loss<T: Element>(&self) -> Result<T, RuntimeError> {
        self.run(&loss_program(self.n() as u64, self.v as u64), None)
            .map(|d| d[0])
    }

    fn grad<T: Element>(&self, g: f32) -> Result<Vec<T>, RuntimeError> {
        self.run(&grad_program(self.n() as u64, self.v as u64), Some(g))
    }

    /// Row `n`'s maximum (ascending, replaced only by a greater value) and its sum of
    /// `exp(x - max)` in ascending index order from the first term, written as index loops.
    fn row<T: Element>(&self, n: usize) -> (T, T) {
        let at = |c: usize| T::from_f32(self.logits[n * self.v + c]);
        let mut max = at(0);
        for c in 1..self.v {
            if at(c) > max {
                max = at(c);
            }
        }
        let mut total = (at(0) - max).exp();
        for c in 1..self.v {
            total = total + (at(c) - max).exp();
        }
        (max, total)
    }

    /// D43 written as index loops in `T`.
    fn reference_loss<T: Element>(&self) -> T {
        let w = |n: usize| T::from_f32(self.weights[n]);
        let mut numerator = T::ZERO;
        let mut denominator = T::ZERO;
        for n in 0..self.n() {
            let (max, total) = self.row::<T>(n);
            let label = T::from_f32(self.logits[n * self.v + self.labels[n] as usize]);
            let loss = max + total.ln() - label;
            if n == 0 {
                numerator = w(0) * loss;
                denominator = w(0);
            } else {
                numerator = w(n).mul_add(loss, numerator);
                denominator = denominator + w(n);
            }
        }
        if denominator == T::ZERO {
            T::ZERO
        } else {
            numerator / denominator
        }
    }

    /// DESIGN.md § 4's gradient written as index loops in `T`.
    fn reference_grad<T: Element>(&self, g: f32) -> Vec<T> {
        let g = T::from_f32(g);
        let mut denominator = T::from_f32(self.weights[0]);
        for n in 1..self.n() {
            denominator = denominator + T::from_f32(self.weights[n]);
        }
        let mut out = Vec::new();
        for n in 0..self.n() {
            let (max, total) = self.row::<T>(n);
            for c in 0..self.v {
                if denominator == T::ZERO {
                    out.push(T::ZERO);
                    continue;
                }
                let x = T::from_f32(self.logits[n * self.v + c]);
                let target = if c == self.labels[n] as usize {
                    T::from_f32(1.0)
                } else {
                    T::ZERO
                };
                let scale = g * T::from_f32(self.weights[n]) / denominator;
                out.push(scale * ((x - max).exp() / total - target));
            }
        }
        out
    }
}

fn bits32(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn bits64(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Rows whose exp terms are `[0.7u, 0.7u, 1, 0.7u]` (ascending sum `1 + 2u`, descending `1`)
/// and weights `[0.7u, 0.7u, 1]` (ascending sum `1 + 2u`, descending `1`).
fn order_separating() -> (Vec<f32>, Vec<i32>, Vec<f32>) {
    let e = small_logit();
    let logits = vec![e, e, 0.0, e, e, e, 0.0, e, 0.0, 1.5, -2.0, 0.25];
    let labels = vec![0, 2, 1];
    let weights = vec![0.7 * U, 0.7 * U, 1.0];
    (logits, labels, weights)
}

/// The data really separates the orders: an ascending and a descending sum differ in `f32`.
#[test]
fn adv_order_separating_data_separates_the_orders() {
    let t = small_logit().exp();
    assert_ne!((t + t) + 1.0f32, (1.0f32 + t) + t);
    let w = 0.7 * U;
    assert_ne!((w + w) + 1.0f32, (1.0f32 + w) + w);
}

/// H1: the loss and every element of the gradient are the bits of the ascending reference on
/// order-separating rows and weights, in `f32` and in `f64`.
#[test]
fn adv_order_separating_rows_match_the_ascending_reference_bit_for_bit() {
    let (logits, labels, weights) = order_separating();
    let case = Case {
        v: 4,
        logits: &logits,
        labels: &labels,
        weights: &weights,
    };
    let loss = case.loss::<f32>().unwrap();
    assert_eq!(
        loss.to_bits(),
        case.reference_loss::<f32>().to_bits(),
        "{loss}"
    );
    let loss = case.loss::<f64>().unwrap();
    assert_eq!(
        loss.to_bits(),
        case.reference_loss::<f64>().to_bits(),
        "{loss}"
    );
    for g in [1.0f32, -0.375, 3.0e-3] {
        let grad = case.grad::<f32>(g).unwrap();
        assert_eq!(
            bits32(&grad),
            bits32(&case.reference_grad::<f32>(g)),
            "g = {g}"
        );
        let grad = case.grad::<f64>(g).unwrap();
        assert_eq!(
            bits64(&grad),
            bits64(&case.reference_grad::<f64>(g)),
            "g = {g}"
        );
    }
}

/// H1: 200 draws of `N = 7`, `V = 9` with logits of magnitude up to 20, weights among 0, 1 and
/// `[0, 3)`, and labels in range: the loss and the gradient are the reference's bits in `f64`
/// (the unit's own bit test runs `f32` only).
#[test]
fn adv_random_draws_match_the_reference_bit_for_bit_in_f64_and_f32() {
    let mut rng = Rng::new(seed_of("adv_random_draws_cross_entropy"));
    let (n, v) = (7usize, 9usize);
    for draw in 0..200 {
        let logits: Vec<f32> = (0..n * v).map(|_| 20.0 * rng.value()).collect();
        let labels: Vec<i32> = (0..n).map(|_| (rng.next_u64() % v as u64) as i32).collect();
        let weights: Vec<f32> = (0..n)
            .map(|_| match rng.next_u64() % 3 {
                0 => 0.0,
                1 => 1.0,
                _ => (3.0 * rng.unit()) as f32,
            })
            .collect();
        let g = rng.value();
        let case = Case {
            v,
            logits: &logits,
            labels: &labels,
            weights: &weights,
        };
        let (l32, l64) = (case.loss::<f32>().unwrap(), case.loss::<f64>().unwrap());
        assert_eq!(
            l32.to_bits(),
            case.reference_loss::<f32>().to_bits(),
            "draw {draw}"
        );
        assert_eq!(
            l64.to_bits(),
            case.reference_loss::<f64>().to_bits(),
            "draw {draw}"
        );
        let (g32, g64) = (case.grad::<f32>(g).unwrap(), case.grad::<f64>(g).unwrap());
        assert_eq!(
            bits32(&g32),
            bits32(&case.reference_grad::<f32>(g)),
            "draw {draw}"
        );
        assert_eq!(
            bits64(&g64),
            bits64(&case.reference_grad::<f64>(g)),
            "draw {draw}"
        );
    }
}

/// H3: weights of `-0.0` are accepted; a weight sum of `-0.0` is a sum of 0, so the loss is `+0`
/// and every gradient element `+0`, never `-0`.
#[test]
fn adv_negative_zero_weights_give_a_positive_zero_loss_and_gradient() {
    let logits = [0.5f32, -1.0, 2.0, 0.25, 3.0, -0.5];
    let labels = [1, 2];
    let weights = [-0.0f32, -0.0];
    let case = Case {
        v: 3,
        logits: &logits,
        labels: &labels,
        weights: &weights,
    };
    assert_eq!(case.loss::<f32>().unwrap().to_bits(), 0.0f32.to_bits());
    assert_eq!(case.loss::<f64>().unwrap().to_bits(), 0.0f64.to_bits());
    for g in [1.0f32, -1.0] {
        let grad = case.grad::<f32>(g).unwrap();
        assert!(grad.iter().all(|x| x.to_bits() == 0), "{grad:?}");
    }
}

/// H3: a label outside the classes in a row of weight 0 is refused, not skipped.
#[test]
fn adv_zero_weight_row_with_a_label_out_of_range_is_refused() {
    let logits = [0.5f32, -1.0, 2.0, 0.25, 3.0, -0.5];
    let labels = [1, 3];
    let weights = [1.0f32, 0.0];
    let case = Case {
        v: 3,
        logits: &logits,
        labels: &labels,
        weights: &weights,
    };
    assert!(matches!(
        case.loss::<f32>(),
        Err(RuntimeError::LabelOutOfRange {
            row: 1,
            label: 3,
            classes: 3,
            ..
        })
    ));
    assert!(matches!(
        case.grad::<f32>(1.0),
        Err(RuntimeError::LabelOutOfRange { row: 1, .. })
    ));
}

/// H3: rows of weight 0 are not skipped (DESIGN.md § 4): an infinite logit in one makes the
/// loss NaN, in `f32` and `f64`.
#[test]
fn adv_zero_weight_row_with_an_infinite_logit_makes_the_loss_nan() {
    let logits = [0.5f32, -1.0, 2.0, f32::INFINITY, 3.0, -0.5];
    let labels = [1, 2];
    let weights = [1.0f32, 0.0];
    let case = Case {
        v: 3,
        logits: &logits,
        labels: &labels,
        weights: &weights,
    };
    assert!(case.loss::<f32>().unwrap().is_nan());
    assert!(case.loss::<f64>().unwrap().is_nan());
}

/// F1, as corrected by coordinator decision D46 (DESIGN.md § 4): a row of weight 0 contributes
/// `0 * loss_n`, so finite logits that keep its row loss finite leave the loss's bits unchanged
/// (`1e38` and `-1e38`: a row loss of `2e38`), and finite logits `3e38` and `-3e38` give an
/// infinite `f32` row loss (`max - logit` overflows), so the loss is NaN (`0 * inf`).
#[test]
fn adv_f1_zero_weight_row_keeps_the_loss_bits_only_while_its_row_loss_is_finite() {
    let labels = [1, 1];
    let weights = [1.0f32, 0.0];
    let before = [0.5f32, -1.0, 2.0, 0.25, 3.0, -0.5];
    let finite = [0.5f32, -1.0, 2.0, 1.0e38, -1.0e38, 0.0];
    let overflowing = [0.5f32, -1.0, 2.0, 3.0e38, -3.0e38, 0.0];
    let loss = |logits: &[f32]| {
        Case {
            v: 3,
            logits,
            labels: &labels,
            weights: &weights,
        }
        .loss::<f32>()
        .unwrap()
    };
    let (b, a) = (loss(&before), loss(&finite));
    assert!(b.is_finite(), "before {b}");
    assert_eq!(b.to_bits(), a.to_bits(), "before {b}, after {a}");
    let a = loss(&overflowing);
    assert!(a.is_nan(), "before {b}, after {a}");
}

/// F2: weights that are each finite and accepted but whose sum overflows `f32`: the weighted
/// mean of finite row losses lies between their minimum and maximum, and the gradient is not
/// zero. The numerator `sum w * loss` overflows first.
#[test]
#[ignore = "F2: finite accepted weights whose sum or products overflow give a NaN loss and a zero gradient"]
fn adv_f2_weights_whose_sum_overflows_give_a_finite_loss_and_gradient() {
    let logits = [0.0f32, 2.0, 0.0, 2.0];
    let labels = [0, 0];
    let weights = [3.0e38f32, 3.0e38];
    let case = Case {
        v: 2,
        logits: &logits,
        labels: &labels,
        weights: &weights,
    };
    let want = case.loss::<f64>().unwrap();
    let loss = case.loss::<f32>().unwrap();
    let grad = case.grad::<f32>(1.0).unwrap();
    assert!(
        (f64::from(loss) - want).abs() <= 1e-5 * want,
        "loss {loss}, f64 {want}; gradient {grad:?}"
    );
    assert!(grad.iter().any(|&x| x != 0.0), "gradient {grad:?}");
}

/// F2: a single weight whose product with a row loss overflows: the loss is the row loss, about
/// 2.13, since the weights normalize it.
#[test]
#[ignore = "F2: a finite accepted weight whose product with the row loss overflows gives an infinite loss"]
fn adv_f2_one_large_weight_gives_the_row_loss() {
    let logits = [0.0f32, 2.0, 0.0, 2.0];
    let labels = [0, 0];
    let weights = [3.0e38f32, 0.0];
    let case = Case {
        v: 2,
        logits: &logits,
        labels: &labels,
        weights: &weights,
    };
    let want = case.loss::<f64>().unwrap();
    let loss = case.loss::<f32>().unwrap();
    assert!(
        (f64::from(loss) - want).abs() <= 1e-5 * want,
        "loss {loss}, f64 {want}"
    );
}

/// F2: a single subnormal weight: the loss is the row loss, since the weights normalize it, but
/// the unnormalized product `w * loss` rounds to a multiple of the smallest subnormal.
#[test]
#[ignore = "F2: a subnormal weight quantizes the loss"]
fn adv_f2_one_subnormal_weight_gives_the_row_loss() {
    let logits = [0.0f32, 2.0, 0.0, 2.0];
    let labels = [0, 0];
    let weights = [f32::from_bits(1), 0.0];
    let case = Case {
        v: 2,
        logits: &logits,
        labels: &labels,
        weights: &weights,
    };
    let want = case.loss::<f64>().unwrap();
    let loss = case.loss::<f32>().unwrap();
    assert!(
        (f64::from(loss) - want).abs() <= 1e-5 * want,
        "loss {loss}, f64 {want}"
    );
}

/// H4: `V = 1`: every row loss is 0, so the loss is `+0` and the gradient every element 0.
#[test]
fn adv_one_class_gives_a_zero_loss_and_gradient() {
    let logits = [3.5f32, -2.0, 1.0e30];
    let labels = [0, 0, 0];
    let weights = [1.0f32, 2.0, 0.5];
    let case = Case {
        v: 1,
        logits: &logits,
        labels: &labels,
        weights: &weights,
    };
    assert_eq!(case.loss::<f32>().unwrap().to_bits(), 0.0f32.to_bits());
    assert!(case.grad::<f32>(1.0).unwrap().iter().all(|&x| x == 0.0));
}

/// H4: a label logit of `-inf` in a weighted row gives an infinite loss and a finite gradient
/// (`p - 1 = -1` at the label).
#[test]
fn adv_minus_infinite_label_logit_gives_an_infinite_loss_and_a_finite_gradient() {
    let logits = [f32::NEG_INFINITY, 1.0, 2.0];
    let labels = [0];
    let weights = [1.0f32];
    let case = Case {
        v: 3,
        logits: &logits,
        labels: &labels,
        weights: &weights,
    };
    assert_eq!(case.loss::<f32>().unwrap(), f32::INFINITY);
    let grad = case.grad::<f32>(1.0).unwrap();
    assert!(grad.iter().all(|x| x.is_finite()), "{grad:?}");
    assert_eq!(grad[0], -1.0);
}

/// H4: a weighted row of `-inf` logits (max `-inf`, `-inf - -inf` NaN) and a NaN logit give a
/// NaN loss, not a panic or a refusal.
#[test]
fn adv_minus_infinite_row_and_nan_logit_give_nan_not_a_panic() {
    for logits in [[f32::NEG_INFINITY; 3], [0.0, f32::NAN, 1.0]] {
        let case = Case {
            v: 3,
            logits: &logits,
            labels: &[1],
            weights: &[1.0],
        };
        assert!(case.loss::<f32>().unwrap().is_nan(), "{logits:?}");
        assert!(
            case.grad::<f32>(1.0).unwrap().iter().all(|x| x.is_nan()),
            "{logits:?}"
        );
    }
}

/// H6: the tied decoder's table gradient is, bit for bit, the sum of the gradients of an untied
/// model whose embedding table and decoder table hold the same values, in `f32` and `f64`: both
/// adjoints are present and nothing else is added.
#[test]
fn adv_tied_decoder_gradient_is_the_bitwise_sum_of_the_untied_adjoints() {
    let (n, v, h) = (6u64, 5u64, 4u64);
    let labels_data = [0, 4, 2, 1, 3, 2];
    let ids_data = [1, 3, 0, 4, 2, 3];
    let weights_data = vec![1.0f32, 0.0, 1.0, 2.5, 0.0, 1.0];

    let mut b = TensorProgramBuilder::new();
    let tokens = id_input(&mut b, "ids", &[n]);
    let labels = id_input(&mut b, "labels", &[n]);
    let weights = input(&mut b, "weights", &[n]);
    let table = param(&mut b, "table", &[v, h]);
    let hidden = b.embedding(table, tokens).unwrap();
    let decoder = b.transpose(table).unwrap();
    let logits = b.matmul(hidden, decoder).unwrap();
    let loss = b.cross_entropy(logits, labels, weights).unwrap();
    let tied = b.finish();
    let tied_training = differentiate(&tied, loss).unwrap();

    let mut b = TensorProgramBuilder::new();
    let tokens = id_input(&mut b, "ids", &[n]);
    let labels = id_input(&mut b, "labels", &[n]);
    let weights = input(&mut b, "weights", &[n]);
    let enc = param(&mut b, "enc", &[v, h]);
    let dec = param(&mut b, "dec", &[v, h]);
    let hidden = b.embedding(enc, tokens).unwrap();
    let decoder = b.transpose(dec).unwrap();
    let logits = b.matmul(hidden, decoder).unwrap();
    let loss = b.cross_entropy(logits, labels, weights).unwrap();
    let untied = b.finish();
    let untied_training = differentiate(&untied, loss).unwrap();
    assert_eq!(untied_training.gradients.len(), 2);

    let given = BTreeMap::from([
        ("ids".to_owned(), (vec![n], ids_data.to_vec())),
        ("labels".to_owned(), (vec![n], labels_data.to_vec())),
    ]);
    let mut rng = Rng::new(seed_of("adv_tied_decoder_bitwise"));
    let mut point = Point::draw(&tied, &given, &mut rng);
    for x in point.params.get_mut(&ParamId(0)).unwrap().1.iter_mut() {
        *x *= 3.0;
    }
    point
        .inputs
        .insert("weights".into(), (vec![n], weights_data.clone()));
    let table_values = point.params[&ParamId(0)].clone();
    let mut untied_params = BTreeMap::new();
    untied_params.insert(ParamId(0), table_values.clone());
    untied_params.insert(ParamId(1), table_values);

    fn check<T: Element + std::fmt::Debug>(
        tied: &modelspec_core::ir::training::TrainingProgram,
        untied: &modelspec_core::ir::training::TrainingProgram,
        point: &Point,
        untied_params: &BTreeMap<ParamId, (Vec<u64>, Vec<f32>)>,
        convert: fn(f32) -> T,
    ) {
        let ids = point.id_tensors();
        let inputs = tensors(&point.inputs, convert);
        let t = evaluate_with_ids(
            &tied.program,
            &inputs,
            &ids,
            &tensors(&point.params, convert),
        )
        .unwrap();
        let u = evaluate_with_ids(
            &untied.program,
            &inputs,
            &ids,
            &tensors(untied_params, convert),
        )
        .unwrap();
        assert_eq!(
            format!("{:?}", t[0].data()),
            format!("{:?}", u[0].data()),
            "losses"
        );
        let sum: Vec<T> = u[1]
            .data()
            .iter()
            .zip(u[2].data())
            .map(|(&a, &b)| a + b)
            .collect();
        assert_eq!(format!("{:?}", t[1].data()), format!("{sum:?}"));
        let (a, b) = (u[1].data(), u[2].data());
        assert!(a.iter().any(|&x| x != T::ZERO) && b.iter().any(|&x| x != T::ZERO));
    }
    check::<f32>(
        &tied_training,
        &untied_training,
        &point,
        &untied_params,
        |x| x,
    );
    check::<f64>(
        &tied_training,
        &untied_training,
        &point,
        &untied_params,
        f64::from,
    );
}
