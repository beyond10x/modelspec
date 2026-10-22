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
//! `--checkpoint-every`. A run depends on nothing but the compiled program, its seed, its start
//! step, and the files of `--data`, `--init` and `--resume`.

use std::borrow::Cow;
use std::collections::BTreeMap;

use modelspec_core::compile::CompiledTraining;
use modelspec_core::ir::tensor::ParamId;
use thiserror::Error;

use super::checkpoint::CheckpointError;
use super::data::{DataError, DenseData, SyntheticData, synthetic};
use super::init::{InitError, Stream, initialize};
use super::optimizer::{ParamState, StepError, sgd_step};
use crate::cpu::{HostTensor, RuntimeError, evaluate, evaluate_all};

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
    observe: impl FnMut(u64, f32),
) -> Result<Run, RunError> {
    run_with(compiled, None, &BTreeMap::new(), observe)
}

/// Runs as [`run_observed`] does, on `data` when it is given instead of synthetic data, and with
/// each parameter in `init` starting from that value instead of its draw. Step `k` feeds
/// [`DenseData::step`]`(k)`; the loss after the last update is the forward program on
/// `step(steps)`. A dense run has no teacher: [`Run::teacher`] is empty.
pub fn run_with(
    compiled: &CompiledTraining,
    data: Option<&DenseData>,
    init: &BTreeMap<ParamId, HostTensor<f32>>,
    observe: impl FnMut(u64, f32),
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
    mut observe: impl FnMut(u64, f32),
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
    let feed = match data {
        Some(dense) => Feed::Dense(dense),
        None => Feed::Synthetic(synthetic(config, &compiled.forward)?),
    };
    let inputs = |step: u64| match &feed {
        Feed::Synthetic(s) => Cow::Borrowed(&s.inputs),
        Feed::Dense(d) => Cow::Owned(d.step(step)),
    };
    let mut state = ParamState::new(&compiled.training, initial.clone());
    for param in compiled.training.optimizer.updates.keys() {
        if let Some(version) = state.versions.get_mut(param) {
            version.0 = version.0.saturating_add(start);
        }
    }
    let mut losses = Vec::new();
    for step in start..end {
        let loss = sgd_step(&compiled.training, &inputs(step), &mut state)
            .map_err(|source| RunError::Step { step, source })?;
        observe(step, loss);
        losses.push(loss);
        after(step + 1, &state)?;
    }
    let last = evaluate(forward, &inputs(end), &state.values)?[0].data()[0];
    observe(end, last);
    losses.push(last);
    Ok(Run {
        start_step: start,
        initial,
        losses,
        state,
        teacher: match feed {
            Feed::Synthetic(s) => s.teacher,
            Feed::Dense(_) => BTreeMap::new(),
        },
    })
}

/// Where the inputs of a step come from.
enum Feed<'a> {
    Synthetic(SyntheticData),
    Dense(&'a DenseData),
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
    for b in 0..batches {
        let inputs = data.eval_batch(b);
        let mut values = evaluate_all(&compiled.forward.program, &inputs, params)?;
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
