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
//! - the loss is `Σ weight_n · nll_n / Σ weight_n` in `f32`, `nll_n` the row's `max + ln(Σ
//!   exp(x - max)) - x[label_n]` as `cross_entropy` computes it (DESIGN.md § 2.2), the
//!   numerator's first product rounded and each later one fused into the sum, the denominator
//!   summed from its first term, and `0` when the weight sum is 0, so over one batch it is the
//!   bits of that batch's `cross_entropy`;
//! - the accuracy is `correct / selected` in `f32` over the rows of positive weight, a row
//!   correct when the first largest logit of its row, the lowest id among ties, is its label,
//!   and never when its row holds a NaN; `0` when no row is selected.
//!
//! A Vulkan run computes the logits of each batch on the device, from the tensors
//! [`HeldOut::batch`] gives, and [`HeldOut::metrics_of`] reduces them as above
//! (coordinator decision D73).

use std::cell::RefCell;
use std::collections::BTreeMap;

use modelspec_core::compile::CompiledTraining;
use modelspec_core::ir::tensor::{DType, Dim, ParamId, ValueId};
use thiserror::Error;

use super::data::{DataError, PackedBatch, PackedData};
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
    /// verified and planned for this evaluation alone.
    pub fn evaluate(
        &self,
        compiled: &CompiledTraining,
        params: &BTreeMap<ParamId, HostTensor<f32>>,
    ) -> Result<HeldOutMetrics, RunError> {
        let prediction = compiled.forward.prediction;
        let mut forward = self.forward.borrow_mut();
        if std::ptr::eq(forward.prepared.program(), &compiled.forward.program) {
            return self.measure(&mut forward, prediction, self.classes, params);
        }
        let rows = self.batches.first().map_or(0, |batch| batch.labels.len());
        let classes = classes(compiled, rows)?;
        self.measure(&mut Forward::new(compiled)?, prediction, classes, params)
    }

    /// The reduction of the module documentation over every batch, each run on `forward`.
    fn measure(
        &self,
        forward: &mut Forward<'_>,
        prediction: ValueId,
        v: usize,
        params: &BTreeMap<ParamId, HostTensor<f32>>,
    ) -> Result<HeldOutMetrics, RunError> {
        let Forward { prepared, outputs } = forward;
        let mut tally = Tally::default();
        for (b, batch) in self.batches.iter().enumerate() {
            prepared
                .run_into(batch.tensors.inputs(), batch.tensors.ids(), params, outputs)
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
        let mut tally = Tally::default();
        for (b, (batch, logits)) in self.batches.iter().zip(logits).enumerate() {
            if logits.len() != batch.labels.len() * self.classes {
                return Err(EvalError::Logits {
                    batches: self.batches.len(),
                    given: logits.len(),
                }
                .into());
            }
            tally.add(b, batch, logits, self.classes)?;
        }
        Ok(tally.metrics())
    }
}

/// The running sums of the reduction of the module documentation.
#[derive(Default)]
struct Tally {
    numerator: Option<f32>,
    denominator: Option<f32>,
    correct: u64,
    selected: u64,
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
            let nll = row_nll(row, target);
            self.numerator = Some(match self.numerator {
                None => weight * nll,
                Some(sum) => weight.mul_add(nll, sum),
            });
            self.denominator = Some(self.denominator.map_or(weight, |sum| sum + weight));
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
                numerator / denominator
            },
            accuracy: if self.selected == 0 {
                0.0
            } else {
                self.correct as f32 / self.selected as f32
            },
        }
    }
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

    #[test]
    fn a_tie_goes_to_the_lowest_id() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), Some(1));
        assert_eq!(argmax(&[1.0, f32::NAN]), None);
    }
}
