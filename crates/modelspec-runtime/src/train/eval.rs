//! Held-out masked-language-model evaluation during training (story:bert-heldout-evaluation,
//! coordinator decision D55; `spec/domains/data.yaml`, `SplitRole::HeldOut`, "MLM loss and
//! accuracy are evaluated on the held-out split").
//!
//! [`HeldOut::new`] draws the batches `b = 0 .. ⌊n / B⌋` of the `HeldOut` split of a packed
//! `--data` directory once, each [`mlm_batch`](crate::dataset::mlm::mlm_batch)`(packed, HeldOut,
//! policy, seed, b, B)` with the run seed and the batch size `B` of the training batches
//! ([`PackedData::draw`]): `HeldOut` batch `b` holds the sequences `b·B .. (b + 1)·B` in stored
//! order, its masking drawn on [`Stream::Masking`](super::init::Stream::Masking) with the id
//! `b + 2^63`, so every evaluation of a run reads the same batches and two runs with one seed
//! read the same. A split of fewer than `B` sequences is [`EvalError::TooFewSequences`], refused
//! before the run starts.
//!
//! [`HeldOut::evaluate`] runs the forward program, the one without dropout
//! ([`CompiledTraining::forward`]), on the CPU interpreter over every batch with the batch's
//! tensors bound by declaration name as the training step binds them, and reads the logits as
//! the intermediate value `prediction`, as the data loader reads it ([`super::data`]). The
//! program is verified and planned once per run, in [`HeldOut::new`], as a [`Prepared`] program
//! that holds `prediction` after each run; every batch of every evaluation runs that prepared
//! program, each intermediate freed after its last consumer and its storage refilled by the next
//! batch, so no batch's values outlive the next batch. Over every row `n` of every batch, in
//! batch order and then row order:
//!
//! - the loss is `Σ weight_n · nll_n / Σ weight_n`, `nll_n` the row's `max + ln(Σ
//!   exp(x - max)) - x[label_n]` in `f32` as `cross_entropy` computes it (DESIGN.md § 2.2),
//!   both sums whole-split sums in `f64` (coordinator decision D84), the numerator's first
//!   product rounded and each later one fused into the sum, the denominator summed from its
//!   first term, the quotient rounded to `f32` once, and `0` when the weight sum is 0, so over
//!   one batch it is the bits of that batch's `cross_entropy`;
//! - the accuracy is `correct / selected` in `f32` over the rows of positive weight, a row
//!   correct when the first largest logit of its row, the lowest id among ties, is its label,
//!   and never when its row holds a NaN; `0` when no row is selected.
//!
//! A Vulkan run computes the logits of each batch on the device, from the tensors
//! [`HeldOut::batch`] gives, and [`HeldOut::metrics_of`] reduces them as above
//! (coordinator decision D73), or [`HeldOut::tallying`] one batch at a time in batch order, to
//! the same bits (coordinator decision D76).

use std::cell::RefCell;
use std::collections::BTreeMap;

use modelspec_core::compile::CompiledTraining;
use modelspec_core::ir::tensor::{DType, Dim, ParamId, ValueId};
use thiserror::Error;

use super::data::{DataError, PackedBatch, PackedData};
use super::init::{sigreg_projections, with_projections};
use super::runner::{RunError, argmax};
use crate::cpu::interpreter::Prepared;
use crate::cpu::{HostTensor, RuntimeError};
use crate::dataset::SplitRole;

/// The name of the held-out loss among the metric samples of a step
/// (`modelspec.observatory.MetricSample`).
pub const HELDOUT_MLM_LOSS: &str = "heldout_mlm_loss";
/// The name of the held-out accuracy among the metric samples of a step.
pub const HELDOUT_MLM_ACCURACY: &str = "heldout_mlm_accuracy";

/// Why the held-out split cannot be evaluated.
#[derive(Clone, Debug, PartialEq, Error)]
pub enum EvalError {
    /// The split holds fewer sequences than one batch.
    #[error(
        "the HeldOut split of {dir} holds {sequences} sequences, fewer than the batch of \
         {batch} that --eval-every evaluates"
    )]
    TooFewSequences {
        dir: String,
        sequences: usize,
        batch: usize,
    },
    /// The model's prediction is not `f32` logits of one row per batch position.
    #[error(
        "--eval-every reads the prediction as f32 logits [{rows}, V], one row per position of \
         the batch; the model's prediction is {prediction}"
    )]
    Prediction { rows: usize, prediction: String },
    /// The forward program failed on a held-out batch.
    #[error("held-out batch {batch}: {source}")]
    Runtime { batch: usize, source: RuntimeError },
    /// Device logits that are not one `[B·S, V]` tensor per batch ([`HeldOut::metrics_of`]).
    #[error("{given} logits do not fill the {batches} held-out batches of [B·S, V] each")]
    Logits { batches: usize, given: usize },
    /// A label outside the classes of the logits.
    #[error("held-out batch {batch}, row {row}: label {label} is not below the {classes} classes")]
    Label {
        batch: usize,
        row: usize,
        label: i32,
        classes: usize,
    },
}

/// The held-out loss and accuracy of one evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeldOutMetrics {
    pub loss: f32,
    pub accuracy: f32,
}

/// One drawn `HeldOut` batch: the tensors the forward program reads, and the labels and weights
/// of its rows, which the reduction reads whether or not the program declares them.
struct Batch {
    tensors: PackedBatch,
    labels: Vec<i32>,
    weights: Vec<f32>,
}

/// The forward program, verified and planned to hold `prediction` after each run, and the
/// outputs of its last run, given back to it before the next one.
struct Forward<'p> {
    prepared: Prepared<'p, f32>,
    outputs: Vec<HostTensor<f32>>,
}

impl<'p> Forward<'p> {
    /// Verifies and plans `compiled`'s forward program; a program the verifier rejects is the
    /// error of batch 0, the first batch that would have run it.
    fn new(compiled: &'p CompiledTraining) -> Result<Self, EvalError> {
        let forward = &compiled.forward;
        let prepared = Prepared::keeping(&forward.program, &[forward.prediction])
            .map_err(|source| EvalError::Runtime { batch: 0, source })?;
        Ok(Forward {
            prepared,
            outputs: Vec::new(),
        })
    }
}

/// The `HeldOut` batches of a run, drawn once, and the forward program they run on, prepared
/// once.
pub struct HeldOut<'p> {
    batches: Vec<Batch>,
    /// The classes `V` of the logits.
    classes: usize,
    /// The forward program of the `compiled` that [`HeldOut::new`] was given.
    forward: RefCell<Forward<'p>>,
}

/// The classes `V` of `compiled`'s prediction when it is `f32` logits `[rows, V]`.
fn classes(compiled: &CompiledTraining, rows: usize) -> Result<usize, EvalError> {
    let prediction = compiled
        .forward
        .program
        .node(compiled.forward.prediction)
        .map(|n| &n.ty);
    let classes = match prediction {
        Some(ty) if ty.dtype == DType::F32 => match ty.shape.as_slice() {
            &[Dim::Static(n), Dim::Static(v)] if n as usize == rows && v > 0 => Some(v as usize),
            _ => None,
        },
        _ => None,
    };
    classes.ok_or_else(|| EvalError::Prediction {
        rows,
        prediction: prediction.map_or_else(|| "undefined".to_owned(), |t| t.to_string()),
    })
}

impl<'p> HeldOut<'p> {
    /// Draws the `HeldOut` batches of `data` for `compiled` and prepares its forward program,
    /// refusing a split of fewer than `B` sequences and a prediction that is not `f32` logits
    /// `[B·S, V]`.
    pub fn new(compiled: &'p CompiledTraining, data: &PackedData) -> Result<Self, RunError> {
        let batch = data.batch();
        let sequences = data.sequences(SplitRole::HeldOut);
        if sequences < batch {
            return Err(EvalError::TooFewSequences {
                dir: data.dir().display().to_string(),
                sequences,
                batch,
            }
            .into());
        }
        let classes = classes(compiled, batch * data.binding().seq_len as usize)?;
        let forward = RefCell::new(Forward::new(compiled)?);
        let batches = (0..sequences / batch)
            .map(|b| {
                let drawn = data.draw(SplitRole::HeldOut, b as u64)?;
                let (labels, weights) = (drawn.labels.clone(), drawn.weights.clone());
                let mut tensors = PackedBatch::default();
                data.fill(drawn, &mut tensors);
                Ok(Batch {
                    tensors,
                    labels,
                    weights,
                })
            })
            .collect::<Result<Vec<_>, DataError>>()?;
        Ok(HeldOut {
            batches,
            classes,
            forward,
        })
    }

    /// The number of batches each evaluation reads.
    pub fn batches(&self) -> usize {
        self.batches.len()
    }

    /// The held-out loss and accuracy of `compiled`'s forward program with `params`.
    ///
    /// For the `compiled` that [`HeldOut::new`] was given, every batch runs the forward program
    /// prepared there. Another `compiled` is checked as `new` checks it, and its forward program
    /// verified and planned for this evaluation alone. A `sigreg` site reads its projection of
    /// step 0; [`HeldOut::evaluate_at`] names the step.
    pub fn evaluate(
        &self,
        compiled: &CompiledTraining,
        params: &BTreeMap<ParamId, HostTensor<f32>>,
    ) -> Result<HeldOutMetrics, RunError> {
        self.evaluate_at(compiled, params, 0)
    }

    /// The held-out loss and accuracy as [`HeldOut::evaluate`] measures them, every `sigreg` site
    /// reading its projection of step `step`, the step whose parameters `params` are
    /// (story:jepa-sigreg-cpu).
    pub fn evaluate_at(
        &self,
        compiled: &CompiledTraining,
        params: &BTreeMap<ParamId, HostTensor<f32>>,
        step: u64,
    ) -> Result<HeldOutMetrics, RunError> {
        let prediction = compiled.forward.prediction;
        let projections =
            sigreg_projections(&compiled.forward.program, compiled.config.seed, step)?;
        let mut forward = self.forward.borrow_mut();
        if std::ptr::eq(forward.prepared.program(), &compiled.forward.program) {
            return self.measure(&mut forward, prediction, self.classes, params, &projections);
        }
        let rows = self.batches.first().map_or(0, |batch| batch.labels.len());
        let classes = classes(compiled, rows)?;
        self.measure(
            &mut Forward::new(compiled)?,
            prediction,
            classes,
            params,
            &projections,
        )
    }

    /// The reduction of the module documentation over every batch, each run on `forward` with
    /// `params` and the `sigreg` `projections` beside the batch's inputs.
    fn measure(
        &self,
        forward: &mut Forward<'_>,
        prediction: ValueId,
        v: usize,
        params: &BTreeMap<ParamId, HostTensor<f32>>,
        projections: &BTreeMap<String, HostTensor<f32>>,
    ) -> Result<HeldOutMetrics, RunError> {
        let Forward { prepared, outputs } = forward;
        let mut tally = Tally::default();
        for (b, batch) in self.batches.iter().enumerate() {
            let inputs = with_projections(batch.tensors.inputs(), projections);
            prepared
                .run_into(&inputs, batch.tensors.ids(), params, outputs)
                .map_err(|source| EvalError::Runtime { batch: b, source })?;
            for (position, output) in outputs.drain(..).enumerate() {
                prepared.recycle(position, output);
            }
            let logits = prepared
                .kept(prediction)
                .ok_or(DataError::UndefinedPrediction(prediction))?;
            tally.add(b, batch, logits.data(), v)?;
        }
        Ok(tally.metrics())
    }

    /// The classes `V` of the logits each batch's prediction holds.
    pub fn classes(&self) -> usize {
        self.classes
    }

    /// The tensors of batch `b`, `b` below [`HeldOut::batches`], by declaration name, as
    /// [`HeldOut::evaluate`] binds them: what a device run uploads into the data views of its
    /// plan before it evaluates the batch.
    pub fn batch(&self, b: usize) -> Option<&PackedBatch> {
        self.batches.get(b).map(|batch| &batch.tensors)
    }

    /// The held-out loss and accuracy of the logits `logits[b]` of every batch `b`, each
    /// `[B·S, V]` row-major, reduced as [`HeldOut::evaluate`] reduces the logits of the CPU
    /// interpreter: what a device run that computed the logits reports.
    pub fn metrics_of(&self, logits: &[Vec<f32>]) -> Result<HeldOutMetrics, RunError> {
        if logits.len() != self.batches.len() {
            return Err(EvalError::Logits {
                batches: self.batches.len(),
                given: logits.len(),
            }
            .into());
        }
        let mut tallying = self.tallying();
        for logits in logits {
            tallying.add(logits)?;
        }
        tallying.metrics()
    }

    /// The reduction of [`HeldOut::metrics_of`] fed one batch's logits at a time, in batch
    /// order: what a device run that reads back one batch's logits at a time reports, the same
    /// bits as `metrics_of` on all of them at once.
    pub fn tallying(&self) -> Tallying<'_, 'p> {
        Tallying {
            held_out: self,
            tally: Tally::default(),
            next: 0,
        }
    }
}

/// The held-out loss and accuracy of logits given one batch at a time ([`HeldOut::tallying`]).
pub struct Tallying<'h, 'p> {
    held_out: &'h HeldOut<'p>,
    tally: Tally,
    /// The batch the next logits are of.
    next: usize,
}

impl Tallying<'_, '_> {
    /// Adds `logits`, `[B·S, V]` row-major, as the logits of the next batch.
    pub fn add(&mut self, logits: &[f32]) -> Result<(), RunError> {
        let held_out = self.held_out;
        let batches = held_out.batches.len();
        let Some(batch) = held_out.batches.get(self.next) else {
            return Err(EvalError::Logits {
                batches,
                given: self.next + 1,
            }
            .into());
        };
        if logits.len() != batch.labels.len() * held_out.classes {
            return Err(EvalError::Logits {
                batches,
                given: logits.len(),
            }
            .into());
        }
        self.tally.add(self.next, batch, logits, held_out.classes)?;
        self.next += 1;
        Ok(())
    }

    /// The held-out loss and accuracy, once the logits of every batch were added.
    pub fn metrics(&self) -> Result<HeldOutMetrics, RunError> {
        let batches = self.held_out.batches.len();
        if self.next != batches {
            return Err(EvalError::Logits {
                batches,
                given: self.next,
            }
            .into());
        }
        Ok(self.tally.metrics())
    }
}

/// The running sums of the reduction of the module documentation.
#[derive(Default)]
struct Tally {
    numerator: Option<f64>,
    denominator: Option<f64>,
    correct: u64,
    selected: u64,
    /// The rows whose `row_nll` the tally computed.
    softmaxes: u64,
}

impl Tally {
    /// Adds the rows of batch `b`, whose logits of `v` classes per row are `logits`.
    fn add(&mut self, b: usize, batch: &Batch, logits: &[f32], v: usize) -> Result<(), RunError> {
        let rows = logits.chunks_exact(v);
        for (n, ((row, &label), &weight)) in rows.zip(&batch.labels).zip(&batch.weights).enumerate()
        {
            let Some(target) = usize::try_from(label).ok().filter(|&l| l < v) else {
                return Err(EvalError::Label {
                    batch: b,
                    row: n,
                    label,
                    classes: v,
                }
                .into());
            };
            if weight.to_bits() == 0 && nll_is_finite(row) {
                // `+0 · nll` with `nll` finite and not negative is `+0`, so the fused term
                // leaves each sum `s` at `s + 0`, the first one at `+0`: no softmax needed.
                self.numerator = Some(self.numerator.map_or(0.0, |sum| sum + 0.0));
                self.denominator = Some(self.denominator.map_or(0.0, |sum| sum + 0.0));
                continue;
            }
            self.softmaxes += 1;
            let (wide, nll) = (f64::from(weight), f64::from(row_nll(row, target)));
            self.numerator = Some(match self.numerator {
                None => wide * nll,
                Some(sum) => wide.mul_add(nll, sum),
            });
            self.denominator = Some(self.denominator.map_or(wide, |sum| sum + wide));
            if weight > 0.0 {
                self.selected += 1;
                if argmax(row) == Some(target) {
                    self.correct += 1;
                }
            }
        }
        Ok(())
    }

    fn metrics(&self) -> HeldOutMetrics {
        let numerator = self.numerator.unwrap_or(0.0);
        let denominator = self.denominator.unwrap_or(0.0);
        HeldOutMetrics {
            loss: if denominator == 0.0 {
                0.0
            } else {
                (numerator / denominator) as f32
            },
            accuracy: if self.selected == 0 {
                0.0
            } else {
                self.correct as f32 / self.selected as f32
            },
        }
    }
}

/// Whether [`row_nll`] of `row` is finite and not negative for every label, known without its
/// softmax: every logit's magnitude is below `5e37` (false for a NaN or an infinity), so the
/// largest exceeds the label's by less than `1e38`. Then the sum of `exp(x - max)` holds the
/// term `exp(0) = 1` and at most `row.len()` terms of at most 1, so its logarithm is at least
/// `+0` and small, `max + ln(sum)` rounds to at least `max`, and the difference from
/// `row[label] <= max` is at least `+0` and below `f32::MAX`. Any other row may have a NaN or
/// infinite nll, which the reduction must see. One pass without a branch, so it vectorizes.
fn nll_is_finite(row: &[f32]) -> bool {
    row.iter()
        .fold(true, |small, &x| small & (x.abs() < 5.0e37))
}

/// `max + ln(Σ exp(x - max)) - row[label]`: the maximum the first element replaced by each later
/// greater one, the sum in ascending order from its first term, as `cross_entropy` computes a
/// row (DESIGN.md § 2.2, coordinator decision D43).
fn row_nll(row: &[f32], label: usize) -> f32 {
    let max = row[1..]
        .iter()
        .fold(row[0], |m, &x| if x > m { x } else { m });
    let total = row[1..]
        .iter()
        .fold((row[0] - max).exp(), |sum, &x| sum + (x - max).exp());
    max + total.ln() - row[label]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_of_equal_logits_has_the_nll_ln_v() {
        let row = [0.0f32; 4];
        assert_eq!(row_nll(&row, 2), 4.0f32.ln());
    }

    /// The reduction of the module documentation over every row, each row's `row_nll` computed:
    /// the tally as it was before it skipped rows (story:bert-tiny-vs-pytorch-cuda).
    fn every_row(batches: &[(Vec<f32>, Vec<i32>, Vec<f32>)], v: usize) -> HeldOutMetrics {
        let (mut numerator, mut denominator) = (None::<f64>, None::<f64>);
        let (mut correct, mut selected) = (0u64, 0u64);
        for (logits, labels, weights) in batches {
            for ((row, &label), &weight) in logits.chunks_exact(v).zip(labels).zip(weights) {
                let target = label as usize;
                let (wide, nll) = (f64::from(weight), f64::from(row_nll(row, target)));
                numerator = Some(match numerator {
                    None => wide * nll,
                    Some(sum) => wide.mul_add(nll, sum),
                });
                denominator = Some(denominator.map_or(wide, |sum| sum + wide));
                if weight > 0.0 {
                    selected += 1;
                    if argmax(row) == Some(target) {
                        correct += 1;
                    }
                }
            }
        }
        let (numerator, denominator) = (numerator.unwrap_or(0.0), denominator.unwrap_or(0.0));
        HeldOutMetrics {
            loss: if denominator == 0.0 {
                0.0
            } else {
                (numerator / denominator) as f32
            },
            accuracy: if selected == 0 {
                0.0
            } else {
                correct as f32 / selected as f32
            },
        }
    }

    fn tallied(batches: &[(Vec<f32>, Vec<i32>, Vec<f32>)], v: usize) -> (HeldOutMetrics, u64) {
        let mut tally = Tally::default();
        for (b, (logits, labels, weights)) in batches.iter().enumerate() {
            let batch = Batch {
                tensors: PackedBatch::default(),
                labels: labels.clone(),
                weights: weights.clone(),
            };
            tally.add(b, &batch, logits, v).expect("valid labels");
        }
        (tally.metrics(), tally.softmaxes)
    }

    /// A batch of `weights.len()` rows of `v` logits, row `n`'s logits `logits(n, c)`, label
    /// `n % v`.
    fn batch(
        v: usize,
        weights: &[f32],
        logits: impl Fn(usize, usize) -> f32,
    ) -> (Vec<f32>, Vec<i32>, Vec<f32>) {
        let rows = weights.len();
        let values = (0..rows * v).map(|i| logits(i / v, i % v)).collect();
        let labels = (0..rows).map(|n| (n % v) as i32).collect();
        (values, labels, weights.to_vec())
    }

    #[test]
    fn the_tally_takes_the_softmax_of_only_the_rows_that_can_change_its_sums() {
        let v = 8;
        let smooth = |n: usize, c: usize| ((n * 31 + c * 17) % 23) as f32 * 0.37 - 4.0;
        // Rows of weight 0 with finite logits add `+0 · nll`, which changes neither sum: their
        // softmax is skipped. Rows of positive weight, and rows of weight 0 whose nll may not be
        // finite (a NaN or an infinity among the logits, or logits so far apart that the nll may
        // overflow), are computed as before.
        let mut odd = batch(v, &[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], smooth);
        odd.0[2 * v + 3] = f32::NAN;
        odd.0[3 * v + 1] = f32::INFINITY;
        odd.0[5 * v] = 3.0e38;
        odd.0[5 * v + 5] = -3.0e38;
        let batches = vec![
            batch(v, &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], smooth),
            odd,
            batch(v, &[0.0; 8], smooth),
        ];
        let (metrics, softmaxes) = tallied(&batches, v);
        assert_eq!(softmaxes, 2 + 2 + 3);
        let expected = every_row(&batches, v);
        assert_eq!(metrics.loss.to_bits(), expected.loss.to_bits());
        assert_eq!(metrics.accuracy.to_bits(), expected.accuracy.to_bits());
    }

    #[test]
    fn skipping_a_row_keeps_the_bits_of_every_sum() {
        let v = 8;
        let smooth = |n: usize, c: usize| ((n * 13 + c * 7) % 19) as f32 * 0.81 - 6.0;
        // A first row of weight 0; a sum of -0 (a weight of -1 on a row whose nll is +0) before
        // a row of weight 0; a batch of no selected row; and random weights.
        let peaked = |_: usize, c: usize| if c == 0 { 0.0 } else { -1000.0 };
        let cases = vec![
            vec![batch(v, &[0.0, 1.0, 0.0, 1.0], smooth)],
            vec![batch(v, &[-1.0, 0.0, 0.0], peaked)],
            vec![batch(v, &[0.0, 0.0, 0.0], smooth)],
            vec![
                batch(v, &[0.5, 0.0, 2.0, 0.0, 0.0, 1.0], smooth),
                batch(v, &[0.0, 0.0, 1.0, 0.0, 3.0, 0.0], peaked),
            ],
        ];
        for batches in cases {
            let (metrics, _) = tallied(&batches, v);
            let expected = every_row(&batches, v);
            assert_eq!(
                metrics.loss.to_bits(),
                expected.loss.to_bits(),
                "{batches:?}"
            );
            assert_eq!(metrics.accuracy.to_bits(), expected.accuracy.to_bits());
        }
    }

    #[test]
    fn a_tie_goes_to_the_lowest_id() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), Some(1));
        assert_eq!(argmax(&[1.0, f32::NAN]), None);
    }
}
