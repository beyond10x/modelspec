//! The training loop (DESIGN.md § 6): state initialization from the seed, the step counter and
//! loss telemetry.
//!
//! [`run`] draws the initial parameters on [`Stream::Init`] and the synthetic data of
//! [`super::data`], then runs `config.steps` SGD steps of the compiled training step. The loss of
//! step `k` is the loss of the parameters after `k` updates: step `k < steps` reports the loss its
//! own evaluation computes before its update, and step `steps` is the forward program evaluated on
//! the final parameters. A run therefore records `steps + 1` losses.
//!
//! [`run_with`] trains on dense data read from files instead, optionally from parameter values read
//! from files, and [`evaluate_dense`] measures the trained parameters on an eval directory.
//! [`run_from`] resumes at a checkpoint's step and reports the state after each update, for
//! `--checkpoint-every`. [`run_on`] is [`run_from`] on either kind of `--data`, dense files or a
//! packed masked-language-model directory ([`Data`]), whose step `k` feeds the batch of step `k`
//! in `f32` and `i32` tensors (story:bert-train-data-binding). A run depends on nothing but the
//! compiled program, its seed, its start step, and the files of `--data`, `--init` and
//! `--resume`.
//!
//! [`run_with`] and [`run_from`] show each step to their observer as an [`Observed`]: its loss,
//! the parameter values that loss is of and, for a step that makes an update, the gradients the
//! update applies (story:observatory-parameter-statistics).

use std::collections::BTreeMap;

use modelspec_core::compile::CompiledTraining;
use modelspec_core::diag::{Code, Diagnostic};
use modelspec_core::ir::tensor::ParamId;
use thiserror::Error;

use super::checkpoint::CheckpointError;
use super::data::{
    DataError, DenseBatch, DenseData, PackedBatch, PackedData, SyntheticData, synthetic,
};
use super::init::{InitError, Stream, initialize};
use super::optimizer::{ParamState, PreparedStep, StepError};
use crate::cpu::{HostTensor, RuntimeError, evaluate_all, evaluate_with_ids};

/// No `i32` tensors, the step inputs of synthetic and dense data.
static NO_IDS: BTreeMap<String, HostTensor<i32>> = BTreeMap::new();

/// The loss of a training step, or a parameter after the update of a step, is NaN or ±inf: the
/// run stops at the first such step and prints no final parameters.
pub const E_TRAIN_001: Code = Code::new("E-TRAIN-001");

/// A finished run.
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    /// The step the run started at: 0, or the step of the checkpoint it resumed from.
    pub start_step: u64,
    /// Each declared parameter's value before the first step.
    pub initial: BTreeMap<ParamId, HostTensor<f32>>,
    /// The loss after `start_step + i` updates at index `i`, for `i` in `0..=steps`.
    pub losses: Vec<f32>,
    /// The parameters' values and versions after the last step.
    pub state: ParamState,
    /// The teacher parameters of the synthetic data: the known solution.
    pub teacher: BTreeMap<ParamId, HostTensor<f32>>,
}

/// Why a run stopped.
#[derive(Clone, Debug, PartialEq, Error)]
pub enum RunError {
    /// No data: [`DataError::Refused`] is a user error, the other variants are not.
    #[error(transparent)]
    Data(#[from] DataError),
    #[error("cannot initialize the parameters: {0}")]
    Init(#[from] InitError),
    /// Step `step` (counting from 0) failed.
    #[error("step {step}: {source}")]
    Step { step: u64, source: StepError },
    /// The forward program failed on the final parameters.
    #[error("final loss: {0}")]
    Runtime(#[from] RuntimeError),
    /// A checkpoint could not be written: [`CheckpointError::Refused`] does not occur here.
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// `start_step + steps` is past `u64::MAX`.
    #[error(
        "a run from step {start} for {steps} steps ends past step {}",
        u64::MAX
    )]
    StepCount { start: u64, steps: u64 },
    /// The loss or a parameter stopped being finite ([`E_TRAIN_001`]).
    #[error("{}", .0.diagnostic)]
    NonFinite(Box<Diverged>),
}

/// A run stopped by [`E_TRAIN_001`]: its diagnostic, and the run up to the step that failed.
#[derive(Clone, Debug, PartialEq)]
pub struct Diverged {
    pub diagnostic: Diagnostic,
    /// The run so far: `losses` ends with the failing step's loss and `state` is the state that
    /// step reached.
    pub run: Run,
}

/// The [`E_TRAIN_001`] diagnostic for step `step` with loss `loss` and the parameter values
/// `values`, or `None` when the loss and every parameter are finite. A non-finite loss is named
/// before a parameter; parameters are named by canonical path, in path order. The Vulkan runner
/// calls it on what it reads back ([`crate::vulkan::runner`]).
pub(crate) fn non_finite(
    compiled: &CompiledTraining,
    step: u64,
    loss: f32,
    values: &BTreeMap<ParamId, HostTensor<f32>>,
) -> Option<Diagnostic> {
    if !loss.is_finite() {
        return Some(
            Diagnostic::new(
                E_TRAIN_001,
                format!("the loss of step {step} is not finite"),
            )
            .with("step", step.to_string())
            .with("loss", loss.to_string()),
        );
    }
    let mut paths: Vec<&str> = values
        .iter()
        .filter(|(_, value)| value.data().iter().any(|v| !v.is_finite()))
        .filter_map(|(id, _)| compiled.forward.program.params.get(id))
        .map(|decl| decl.path.as_str())
        .collect();
    if paths.is_empty() {
        return None;
    }
    paths.sort_unstable();
    let named: Vec<String> = paths.iter().map(|path| format!("@{path}")).collect();
    Some(
        Diagnostic::new(
            E_TRAIN_001,
            format!("the update of step {step} leaves a parameter not finite"),
        )
        .with("step", step.to_string())
        .with("loss", loss.to_string())
        .with("parameter", named.join(", ")),
    )
}

/// Runs `compiled.config.steps` SGD steps of `compiled.training` on synthetic data from
/// `compiled.config.seed`.
pub fn run(compiled: &CompiledTraining) -> Result<Run, RunError> {
    run_observed(compiled, |_, _| {})
}

/// Runs as [`run`] does and calls `observe(k, loss)` with each loss as it is recorded, while the
/// run goes on: `k` runs through `0..=steps` in order, and `loss` is `Run::losses[k]`. A run that
/// stops early has observed the losses it recorded before it stopped; a run refused before its
/// first step observes none.
pub fn run_observed(
    compiled: &CompiledTraining,
    mut observe: impl FnMut(u64, f32),
) -> Result<Run, RunError> {
    run_with(compiled, None, &BTreeMap::new(), |o| {
        observe(o.step, o.loss)
    })
}

/// One step as the observer of [`run_with`] and [`run_from`] sees it, before its update.
#[derive(Clone, Copy, Debug)]
pub struct Observed<'a> {
    /// The step, `k` of `Run::losses[k - start]`.
    pub step: u64,
    /// The loss of [`Self::values`].
    pub loss: f32,
    /// Every declared parameter's value after `step` updates, before this step's own.
    pub values: &'a BTreeMap<ParamId, HostTensor<f32>>,
    /// The gradient of the loss at [`Self::values`] for each parameter the loss reads, the one
    /// this step's update applies; `None` for the last step, which makes no update.
    pub gradients: Option<&'a BTreeMap<ParamId, HostTensor<f32>>>,
}

/// Runs as [`run_observed`] does, on `data` when it is given instead of synthetic data, and with
/// each parameter in `init` starting from that value instead of its draw, and shows `observe`
/// each step as an [`Observed`]. Step `k` feeds the samples of [`DenseData::step`]`(k)`, filled
/// into one [`DenseBatch`] per run ([`DenseData::fill_step`]); the loss after the
/// last update is the forward program on `step(steps)`. A dense run has no teacher:
/// [`Run::teacher`] is empty.
pub fn run_with(
    compiled: &CompiledTraining,
    data: Option<&DenseData>,
    init: &BTreeMap<ParamId, HostTensor<f32>>,
    observe: impl FnMut(&Observed<'_>),
) -> Result<Run, RunError> {
    run_from(compiled, data, init, 0, observe, |_, _| Ok(()))
}

/// Runs as [`run_with`] does, from step `start` instead of 0: the steps are numbered
/// `start..=start + steps`, step `k` feeds [`DenseData::step`]`(k)`, and each parameter with an
/// update starts at version `start`. A resumed run passes the checkpoint's step and parameters as
/// `start` and `init`. After each update `after(k, state)` is called with the number of updates
/// `k` so far and the state they reached; an error from it stops the run.
pub fn run_from(
    compiled: &CompiledTraining,
    data: Option<&DenseData>,
    init: &BTreeMap<ParamId, HostTensor<f32>>,
    start: u64,
    observe: impl FnMut(&Observed<'_>),
    after: impl FnMut(u64, &ParamState) -> Result<(), CheckpointError>,
) -> Result<Run, RunError> {
    run_on(compiled, data.map(Data::Dense), init, start, observe, after)
}

/// The data of `--data`: dense files, or a packed masked-language-model directory.
#[derive(Clone, Copy)]
pub enum Data<'a> {
    Dense(&'a DenseData),
    Packed(&'a PackedData),
}

/// Runs as [`run_from`] does, on `data` of either kind. With packed data step `k` feeds
/// [`PackedData::step`]`(k)`, the batch of step `k` of the run that started at 0, so a resumed run
/// feeds the batches of the run that never stopped; the loss after the last update is the
/// forward program on the batch of step `start + steps`. A packed run has no teacher.
pub fn run_on(
    compiled: &CompiledTraining,
    data: Option<Data<'_>>,
    init: &BTreeMap<ParamId, HostTensor<f32>>,
    start: u64,
    mut observe: impl FnMut(&Observed<'_>),
    mut after: impl FnMut(u64, &ParamState) -> Result<(), CheckpointError>,
) -> Result<Run, RunError> {
    let config = &compiled.config;
    let end = start.checked_add(config.steps).ok_or(RunError::StepCount {
        start,
        steps: config.steps,
    })?;
    let forward = &compiled.forward.program;
    let mut initial = initialize(forward, config.seed, Stream::Init)?;
    initial.extend(init.iter().map(|(&p, v)| (p, v.clone())));
    let mut feed = match data {
        Some(Data::Dense(dense)) => Feed::Dense(dense, DenseBatch::default()),
        Some(Data::Packed(packed)) => Feed::Packed(packed, PackedBatch::default()),
        None => Feed::Synthetic(synthetic(config, &compiled.forward)?),
    };
    let mut state = ParamState::new(&compiled.training, initial.clone());
    for param in compiled.training.optimizer.updates.keys() {
        if let Some(version) = state.versions.get_mut(param) {
            version.0 = version.0.saturating_add(start);
        }
    }
    // Prepared at the first step that makes an update, so a run of no steps verifies nothing more
    // than before; every later step reuses it (story:cpu-prepared-training-step).
    let mut prepared: Option<PreparedStep<'_>> = None;
    let mut losses = Vec::new();
    for step in start..=end {
        let loss = if step < end {
            if prepared.is_none() {
                let ready = PreparedStep::new(&compiled.training)
                    .map_err(|source| RunError::Step { step, source })?;
                prepared = Some(ready);
            }
            let prepared = prepared.as_mut().expect("prepared above");
            let (inputs, ids) = feed.inputs(step)?;
            let evaluated = prepared
                .evaluate_with_ids(inputs, ids, &state)
                .map_err(|source| RunError::Step { step, source })?;
            observe(&Observed {
                step,
                loss: evaluated.loss(),
                values: &state.values,
                gradients: Some(evaluated.gradients()),
            });
            evaluated.apply(&mut state)
        } else {
            let (inputs, ids) = feed.inputs(end)?;
            let loss = evaluate_with_ids(forward, inputs, ids, &state.values)?[0].data()[0];
            observe(&Observed {
                step,
                loss,
                values: &state.values,
                gradients: None,
            });
            loss
        };
        losses.push(loss);
        if let Some(diagnostic) = non_finite(compiled, step, loss, &state.values) {
            return Err(RunError::NonFinite(Box::new(Diverged {
                diagnostic,
                run: Run {
                    start_step: start,
                    initial,
                    losses,
                    state,
                    teacher: feed.teacher(),
                },
            })));
        }
        if step < end {
            after(step + 1, &state)?;
        }
    }
    Ok(Run {
        start_step: start,
        initial,
        losses,
        state,
        teacher: feed.teacher(),
    })
}

/// The `f32` and the `i32` inputs of one step, by name.
type StepInputs<'f> = (
    &'f BTreeMap<String, HostTensor<f32>>,
    &'f BTreeMap<String, HostTensor<i32>>,
);

/// Where the inputs of a step come from: synthetic data, the same every step, or dense or packed
/// data and the one batch every step of the run is filled into.
enum Feed<'a> {
    Synthetic(SyntheticData),
    Dense(&'a DenseData, DenseBatch),
    Packed(&'a PackedData, PackedBatch),
}

impl Feed<'_> {
    /// The inputs of step `step`: `f32` tensors, and the `i32` tensors only packed data has.
    fn inputs(&mut self, step: u64) -> Result<StepInputs<'_>, DataError> {
        match self {
            Feed::Synthetic(s) => Ok((&s.inputs, &NO_IDS)),
            Feed::Dense(data, batch) => {
                data.fill_step(step, batch);
                Ok((batch.inputs(), &NO_IDS))
            }
            Feed::Packed(data, batch) => {
                data.fill_step(step, batch)?;
                Ok((batch.inputs(), batch.ids()))
            }
        }
    }

    /// The teacher parameters of synthetic data; dense and packed data have none.
    fn teacher(self) -> BTreeMap<ParamId, HostTensor<f32>> {
        match self {
            Feed::Synthetic(s) => s.teacher,
            Feed::Dense(..) | Feed::Packed(..) => BTreeMap::new(),
        }
    }
}

/// The loss and accuracy of the forward program over every sample of an eval directory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eval {
    /// The mean of the per-batch losses, summed in `f64`.
    pub loss: f64,
    /// The fraction of samples whose prediction row has its first maximum at the index of the
    /// first maximum of the target row; a prediction row holding a NaN is never correct.
    pub accuracy: f64,
}

/// Runs the forward program of `compiled` with `params` over `data` in batches of `B`
/// ([`DenseData::eval_batch`]); `data` holds a whole number of batches ([`super::data::dense_eval`]).
pub fn evaluate_dense(
    compiled: &CompiledTraining,
    data: &DenseData,
    params: &BTreeMap<ParamId, HostTensor<f32>>,
) -> Result<Eval, RunError> {
    let prediction = compiled.forward.prediction;
    let batches = data.samples() / data.batch();
    let mut loss = 0.0f64;
    let mut correct = 0usize;
    let mut batch = DenseBatch::default();
    for b in 0..batches {
        data.fill_eval_batch(b, &mut batch);
        let inputs = batch.inputs();
        let mut values = evaluate_all(&compiled.forward.program, inputs, params)?;
        loss += f64::from(values.outputs[0].data()[0]);
        let predicted = values
            .values
            .remove(&prediction)
            .ok_or(DataError::UndefinedPrediction(prediction))?;
        let target = inputs[data.target()].data();
        let width = target.len() / data.batch();
        correct += predicted
            .data()
            .chunks_exact(width)
            .zip(target.chunks_exact(width))
            .filter(|(p, t)| argmax(p).is_some_and(|i| Some(i) == argmax(t)))
            .count();
    }
    Ok(Eval {
        loss: loss / batches as f64,
        accuracy: correct as f64 / data.samples() as f64,
    })
}

/// The index of the first maximum of `row`, or `None` when `row` holds a NaN.
fn argmax(row: &[f32]) -> Option<usize> {
    if row.iter().any(|v| v.is_nan()) {
        return None;
    }
    Some((1..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use modelspec_core::compile::compile_training;
    use modelspec_core::ir::training::StateVersion;

    use crate::cpu::evaluate;
    use crate::train::data::E_DATA_003;

    const EXAMPLE: &str = include_str!("../../../../examples/linear_regression.mspec");
    const W: ParamId = ParamId(0);

    fn compiled(source: &str, steps: u64) -> CompiledTraining {
        let mut compiled = compile_training("fit.mspec", source).expect("the source compiles");
        compiled.config.steps = steps;
        compiled
    }

    #[test]
    fn a_run_records_one_loss_per_step_and_the_final_loss() {
        let run = run(&compiled(EXAMPLE, 3)).unwrap();
        assert_eq!(run.losses.len(), 4);
        assert_eq!(run.state.versions[&W], StateVersion(3));
    }

    #[test]
    fn the_step_0_loss_is_the_forward_loss_of_the_initial_parameters() {
        let c = compiled(EXAMPLE, 2);
        let run = run(&c).unwrap();
        let data = synthetic(&c.config, &c.forward).unwrap();
        let at_init = evaluate(&c.forward.program, &data.inputs, &run.initial).unwrap();
        assert_eq!(run.losses[0], at_init[0].data()[0]);
        let at_end = evaluate(&c.forward.program, &data.inputs, &run.state.values).unwrap();
        assert_eq!(run.losses[2], at_end[0].data()[0]);
    }

    #[test]
    fn the_loss_of_step_k_is_the_loss_after_k_updates() {
        let one = run(&compiled(EXAMPLE, 1)).unwrap();
        let three = run(&compiled(EXAMPLE, 3)).unwrap();
        assert_eq!(three.losses[..2], one.losses[..]);
    }

    #[test]
    fn the_initial_parameters_are_the_init_stream_draw_and_the_teacher_the_teacher_stream() {
        let c = compiled(EXAMPLE, 1);
        let run = run(&c).unwrap();
        assert_eq!(
            run.initial,
            initialize(&c.forward.program, 42, Stream::Init).unwrap()
        );
        assert_eq!(
            run.teacher,
            initialize(&c.forward.program, 42, Stream::Teacher).unwrap()
        );
    }

    #[test]
    fn a_run_is_deterministic() {
        let a = run(&compiled(EXAMPLE, 20)).unwrap();
        let b = run(&compiled(EXAMPLE, 20)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn another_seed_starts_elsewhere() {
        let a = run(&compiled(EXAMPLE, 1)).unwrap();
        let other = EXAMPLE.replace("seed = 42;", "seed = 43;");
        let b = run(&compiled(&other, 1)).unwrap();
        assert_ne!(a.initial, b.initial);
    }

    #[test]
    fn the_loss_falls() {
        let run = run(&compiled(EXAMPLE, 50)).unwrap();
        assert!(run.losses[50] < run.losses[0], "{:?}", run.losses);
    }

    #[test]
    fn a_parameter_the_loss_does_not_read_keeps_its_value_and_version() {
        let source = EXAMPLE.replace(
            "    fn forward",
            "    param u: Tensor<f32, [2, 2]> = xavier_uniform();\n\n    fn forward",
        );
        let c = compiled(&source, 3);
        let unused = c
            .forward
            .program
            .params
            .iter()
            .find(|(_, decl)| decl.path == "LinearReg.u")
            .map(|(&p, _)| p)
            .expect("`u` is declared");
        let run = run(&c).unwrap();
        assert_eq!(run.state.values[&unused], run.initial[&unused]);
        assert_eq!(run.state.versions[&unused], StateVersion(0));
        assert_eq!(run.state.versions[&W], StateVersion(3));
    }

    #[test]
    fn the_observer_sees_every_loss_in_step_order() {
        let mut seen = Vec::new();
        let run =
            run_observed(&compiled(EXAMPLE, 3), |step, loss| seen.push((step, loss))).unwrap();
        let expected: Vec<(u64, f32)> = (0..).zip(run.losses.iter().copied()).collect();
        assert_eq!(seen, expected);
        assert_eq!(seen.len(), 4);
    }

    #[test]
    fn an_observed_run_is_the_run() {
        let c = compiled(EXAMPLE, 5);
        assert_eq!(run_observed(&c, |_, _| {}).unwrap(), run(&c).unwrap());
    }

    /// The observer of step `k` sees the values after `k` updates, the values step `k`'s loss is
    /// of, and for `k < steps` the gradient step `k`'s update applies to them; the last step
    /// makes no update and has no gradient.
    #[test]
    fn the_observer_sees_the_values_of_each_step_and_the_gradient_its_update_applies() {
        let c = compiled(EXAMPLE, 3);
        let data = synthetic(&c.config, &c.forward).unwrap();
        let training = &c.training.training;
        let position = training
            .program
            .outputs
            .iter()
            .position(|&o| o == training.gradients[&W])
            .unwrap();
        let mut seen = Vec::new();
        let whole = run_with(&c, None, &BTreeMap::new(), |o| {
            seen.push((o.step, o.loss, o.values.clone(), o.gradients.cloned()))
        })
        .unwrap();
        assert_eq!(seen.len(), 4);
        for (k, (step, loss, values, gradients)) in seen.iter().enumerate() {
            assert_eq!(*step, k as u64);
            assert_eq!(*loss, whole.losses[k], "step {k}");
            let after_k = run(&compiled(EXAMPLE, k as u64)).unwrap();
            assert_eq!(values, &after_k.state.values, "step {k}");
            if k < 3 {
                let outputs = evaluate(&training.program, &data.inputs, values).unwrap();
                let gradients = gradients.as_ref().expect("a step with an update");
                assert_eq!(gradients[&W], outputs[position], "step {k}");
            } else {
                assert_eq!(*gradients, None, "the last step makes no update");
            }
        }
        assert_eq!(seen[3].2, whole.state.values);
    }

    /// A parameter the loss does not read has values at every step and a gradient at none.
    #[test]
    fn a_parameter_the_loss_does_not_read_is_observed_without_a_gradient() {
        let source = EXAMPLE.replace(
            "    fn forward",
            "    param u: Tensor<f32, [2, 2]> = xavier_uniform();\n\n    fn forward",
        );
        let c = compiled(&source, 2);
        let mut seen = Vec::new();
        run_with(&c, None, &BTreeMap::new(), |o| {
            seen.push((
                o.values.keys().copied().collect::<Vec<_>>(),
                o.gradients.map(|g| g.keys().copied().collect::<Vec<_>>()),
            ))
        })
        .unwrap();
        let declared: Vec<ParamId> = c.forward.program.params.keys().copied().collect();
        assert_eq!(declared.len(), 2);
        assert_eq!(
            seen,
            vec![
                (declared.clone(), Some(vec![W])),
                (declared.clone(), Some(vec![W])),
                (declared, None),
            ]
        );
    }

    #[test]
    fn a_refused_run_observes_no_loss() {
        let source = EXAMPLE
            .replace("return matmul(x, w);", "return w;")
            .replace(
                "param w: Tensor<f32, [4, 1]>",
                "param w: Tensor<f32, [8, 1]>",
            );
        let mut seen = 0;
        assert!(run_observed(&compiled(&source, 1), |_, _| seen += 1).is_err());
        assert_eq!(seen, 0);
    }

    #[test]
    fn a_divergent_run_stops_at_its_first_non_finite_loss() {
        let source = EXAMPLE.replace("sgd(lr = 0.1)", "sgd(lr = 100.0)");
        let mut seen = Vec::new();
        let error = run_observed(&compiled(&source, 500), |step, loss| {
            seen.push((step, loss))
        })
        .unwrap_err();
        let RunError::NonFinite(diverged) = error else {
            panic!("not E-TRAIN-001: {error:?}");
        };
        assert_eq!(diverged.diagnostic.code, E_TRAIN_001);
        let losses = &diverged.run.losses;
        let (&last, before) = losses.split_last().unwrap();
        assert!(!last.is_finite(), "{losses:?}");
        assert!(before.iter().all(|l| l.is_finite()), "{losses:?}");
        let expected: Vec<(u64, f32)> = (0..).zip(losses.iter().copied()).collect();
        assert_eq!(format!("{seen:?}"), format!("{expected:?}"));
    }

    #[test]
    fn a_data_refusal_stops_the_run_before_any_step() {
        let source = EXAMPLE
            .replace("model LinearReg", "model Const")
            .replace("= LinearReg;", "= Const;")
            .replace(
                "param w: Tensor<f32, [4, 1]>",
                "param w: Tensor<f32, [8, 1]>",
            )
            .replace("return matmul(x, w);", "return w;");
        match run(&compiled(&source, 1)) {
            Err(RunError::Data(DataError::Refused(d))) => assert_eq!(d.code, E_DATA_003, "{d}"),
            other => panic!("not refused: {other:?}"),
        }
    }
}
