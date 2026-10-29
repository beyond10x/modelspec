//! A training run on one Vulkan device that the host feeds before every step (coordinator decision
//! D73, story:bert-vulkan-kernels-attention-loss-adamw), for a step plan that reads more than its
//! parameters and one batch: the steps are recorded in segments of at most `cap` steps each, one
//! command buffer and one host wait per segment (story:vulkan-stepwise-multi-step-submit), one
//! per evaluation the device reduces, and one per held-out batch an evaluation reduces on the host.
//!
//! [`super::runner::train`] records up to [`super::runner::CHUNK_STEPS`] steps into one command
//! buffer and uploads once, so it runs a step plan whose only inputs are the parameters and the
//! synthetic batch. [`train_stepwise`] runs the plans it refuses:
//!
//!
//! - a step with `dropout`: each `Dropout` and `DropoutGrad` pipeline is created under the run
//!   key, the run's seed ([`super::Device::kernel_pipeline_seeded`]), and the RNG token of every
//!   dropout site at step `k` ([`crate::train::init::dropout_token`]) is uploaded before step `k`,
//!   so the device draws the CPU run's masks;
//! - an `adamw(...)` step: its update runs as the kernels of the `Optimizer` subgraph, the
//!   scheduled rate and both bias corrections of step `k` ([`AdamW::scalars`]`(lr, k + 1, end)`)
//!   are uploaded before it, and each moment `m` and `v` lives in a device buffer of its own,
//!   zero at the start, which the step plan updates in place, so the moments never leave the
//!   device until the run reads them back;
//! - a step with `sigreg` (story:jepa-sigreg-vulkan): the projection of every `sigreg` site at
//!   step `k` ([`crate::train::init::sigreg_projection`]) is uploaded before step `k`, the
//!   final loss and every dense eval batch reading the projections of the last step and a
//!   held-out batch those of the step it is evaluated at, as the CPU run draws them;
//! - a step that gathers the rows of non-zero weight (DESIGN.md § 3.3, `--masked-rows`): the
//!   gathered row ids, labels and weights of step `k` ([`crate::train::masked::gather`]) are
//!   uploaded before it, and the step records only the plan steps the training step keeps
//!   ([`modelspec_core::ir::kernel::ExecutionPlan::training_steps`]), not the full head that
//!   only the prediction reads;
//! - dense or packed `--data` ([`Data`]): the batch of step `k` is uploaded before it, as the CPU
//!   run feeds it ([`crate::train::runner::run_checked`]);
//! - held-out evaluation ([`Evaluation`], `--eval-every`): at every multiple of `every` and the last
//!   step, each `HeldOut` batch is copied into the data views and runs the steps of the plan the
//!   prediction is computed from, without the loss sum
//!   ([`modelspec_core::ir::kernel::ExecutionPlan::prediction_steps`]), with dropout off. With
//!   [`HeldOutReduction::Device`] (story:vulkan-eval-metrics-on-device) every batch is recorded
//!   in one command buffer before the evaluated step's segment, followed by the kernels of
//!   [`crate::spirv::emit::held_out_kernels`], which add its row losses and correct rows to a
//!   device state of [`crate::spirv::emit::HELD_OUT_STATE`] floats; the state comes down at its
//!   end, and after its one host wait [`HeldOut::metrics_of_sums`] divides it as the CPU
//!   evaluation divides its sums. With [`HeldOutReduction::Host`] (coordinator decision D76,
//!   story:vulkan-eval-download-per-batch), and for a plan whose prediction does not start its
//!   buffer, each batch is a command buffer of its own before the step's, which copies the
//!   prediction into a download buffer of one prediction ([`download_bytes`]), and after its
//!   host wait the host adds the logits to the reduction of the CPU evaluation, in batch order
//!   ([`HeldOut::tallying`]), before the next batch overwrites them;
//! - dense evaluation ([`StepwiseOptions::dense_eval`], `--eval`; story:decider-dense-named-data):
//!   at the last step, before its command buffer, each batch of the eval directory is a command
//!   buffer of its own, as a held-out batch is, which copies the loss down, and the prediction
//!   too when the accuracy is measured ([`DenseData::compared_target`]); the host sums the
//!   losses in `f64` in batch order and counts the correct target rows as
//!   [`crate::train::runner::evaluate_dense`] does, and shows the result at the last step
//!   ([`StepSeen::dense_eval`]); with [`StepwiseOptions::dense_predictions`] each batch's
//!   prediction comes down too and is kept in batch order, which `infer --data --function` runs
//!   with 0 steps on the step of a model function (story:decider-infer-on-data).
//!
//! Dropout is off where the CPU run has none: in the evaluation of the loss after the last
//! update and in held-out evaluation, which the CPU run computes with the forward program. There
//! the loss steps run with each `Dropout` pipeline replaced by the one of the same kernel at
//! `p = 0`, which keeps every element and scales it by 1, so it copies its input.
//!
//! The steps run in segments (story:vulkan-stepwise-multi-step-submit). A segment is the steps
//! `k..=through` one command buffer records: at most `cap` of them,
//! [`StepwiseOptions::submit_steps`] or the runner's choice [`steps_per_submit`], ending at a
//! step after which a checkpoint is due and before an evaluated step and the last step, which is
//! a segment of its own. Before it is recorded the host draws each step's batch, tokens and
//! scalars as the one-step run draws them for that step (its dropout tokens, its batch and its
//! AdamW step count `k + 1`) into the step's region of an upload buffer. The command buffer
//! copies each step's region in, runs its plan steps (the loss steps without dropout at the last
//! step) and copies its loss down to its slot, with a compute-to-compute and a
//! compute-to-transfer barrier between two steps, the barriers that open each later command
//! buffer; the parameters and moments come down after the last step and at each due checkpoint.
//! Since every step reads what the one-step run's step reads, in the same order, the losses and
//! parameters are its bits.
//!
//! Two upload buffers, two loss downloads and two timestamp pools take turns: the host submits a
//! segment behind the one in flight, then waits for that one, shows its steps to the observer and
//! draws and records the next segment while the device runs. An evaluated step, the last step and
//! a segment whose state the host reads are not queued behind another: the host waits for the
//! segment in flight first, and for a segment whose state it reads at once.
//!
//! The first command buffer of the run, a held-out batch's or the first segment's, copies the
//! parameters and the zero moments up. An evaluated step first runs its evaluation: each batch an
//! evaluation reduces on the host in a command buffer of its own, or every batch and the held-out
//! kernels of one the device reduces in one command buffer, whose state comes down after its host
//! wait. After a segment's host wait the host checks each step's loss, and the parameters and
//! moments where it read them, with E-TRAIN-001 as the CPU run does, and shows the steps to the
//! observer ([`StepSeen`]) in order. A step whose loss is not finite stops the run at that step,
//! after the host waited for the segment in flight, with one more host wait, which reads the
//! parameters and moments after that step, as one step per command buffer leaves them: each
//! segment first copies the parameters and moments it starts from into a device buffer of its
//! own, one per upload buffer, and that command buffer copies them back, runs the segment's
//! steps through the failing one again from the segment's uploads and copies the result down.
//!
//! The run keeps a training clock (story:bert-tiny-vs-pytorch-cuda, coordinator decision D87):
//! [`RunStats::training`] sums the spans in which segments are in flight, from recording the
//! first segment after a pause to the host wait that leaves none in flight, without the last
//! step's; a span holds the host's work on later segments while the device runs (drawing their
//! batches, recording them and showing the steps before them to the observer). The evaluations
//! are timed apart into [`RunStats::evaluation`], from recording the first command buffer of an
//! evaluation to the host's metrics. Neither holds the plan build, the SPIR-V emission and pipeline
//! creation, the batches the host draws before a span opens, the final loss evaluation and the
//! dense eval batches before it (the CPU run measures `--eval` after its loop), or the checkpoint
//! files, which are written after a pause. [`StepSeen::trained`] is the clock through the spans
//! closed before its segment was recorded: at an evaluated step, which starts a segment after a
//! pause, the clock through the updates before the step.
//!
//! A profiled run ([`train_stepwise_profiled`], `train --profile`) writes a timestamp before and
//! after each dispatch of a segment's command buffer into its timestamp pool, reset at the start
//! of the command buffer and read after its host wait, so it adds no host wait; the evaluations
//! and dense eval batches are not timed. The stamps of every step, then those of the final loss
//! evaluation, make the [`Profile`] of the chunked run ([`super::profile::timings`]); its steady
//! time per step also holds the device's idle time between two segments and any evaluation
//! between them.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use modelspec_core::artifact::{CompiledArtifact, check_device};
use modelspec_core::compile::CompiledTraining;
use modelspec_core::compile::codes;
use modelspec_core::device::DeviceProfile;
use modelspec_core::diag::Diagnostic;
use modelspec_core::ir::kernel::{BufferView, Kernel, Operand, PlanOp};
use modelspec_core::ir::tensor::{DType, ParamId, TensorType};
use modelspec_core::ir::training::{
    AdamW, C1_INPUT, C2_INPUT, LR_T_INPUT, Moment, OptimizerRule, dropout_sites, moment_input,
    sigreg_sites,
};
use modelspec_core::passes::masked_rows;
use modelspec_core::schedule::{MAX_SPAN_KEYS, host_inputs};

use super::Device;
use super::dispatch::{Bound, DescriptorArena, Recording, Submitted};
use super::memory::{Buffer, MemorySpan};
use super::negotiate::Negotiated;
use super::pipeline::{Pipeline, PipelineRequest, PipelineStatistics, reads_seed};
use super::pipeline_cache::make_pipelines;
use super::profile::{PlanName, Profile, Site, timings};
use super::runner::{
    Action, Checkpoints, DeviceRunError, DivergedOnDevice, RunStats, Stamps, VulkanRunError,
    action_positions, actions, buffer, count_u32, descriptors, numerics_class, on_device,
    parameters, record, state_of,
};
use super::sync::{Barrier, recorded_barriers};
use crate::cpu::HostTensor;
use crate::spirv::emit::{
    HELD_OUT_STATE, HeldOutKernels, emit_kernels, emit_module_with, held_out_kernels, held_out_sums,
};
use crate::train::data::{DataError, DenseBatch, DenseData, PackedBatch, synthetic};
use crate::train::eval::{HeldOut, HeldOutMetrics};
use crate::train::init::{Stream, dropout_token, initialize, sigreg_projection};
use crate::train::masked;
use crate::train::runner::{Data, Diverged, Eval, Feed, Run, RunError, correct_rows, non_finite};

/// The held-out evaluation of a stepwise run: `held_out` at every multiple of `every` among the
/// run's steps and at its last step, as `--eval-every` asks.
pub struct Evaluation<'o, 'h> {
    pub held_out: &'o HeldOut<'h>,
    /// The steps between two evaluations: positive.
    pub every: u64,
    /// Where each batch's logits are reduced.
    pub reduction: HeldOutReduction,
}

/// Where a stepwise run reduces the logits of each evaluated `HeldOut` batch to the loss and
/// accuracy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeldOutReduction {
    /// On the device (story:vulkan-eval-metrics-on-device): the kernels of
    /// [`crate::spirv::emit::held_out_kernels`] after each batch's prediction, the whole
    /// evaluation in one command buffer before the evaluated step's segment,
    /// [`crate::spirv::emit::HELD_OUT_STATE`] floats read back after its one host wait and
    /// finished by [`HeldOut::metrics_of_sums`]. A plan whose prediction view does not start its
    /// buffer is reduced on the host. The accuracy is the host reduction's bit for bit and the
    /// loss within 4 ulps of it (coordinator decision D141): the device's `Exp` and `Log` are not
    /// the host libm's.
    #[default]
    Device,
    /// On the host (story:vulkan-eval-download-per-batch): each batch's logits read back after
    /// a command buffer and host wait of its own and added to [`HeldOut::tallying`], the oracle
    /// of the device reduction.
    Host,
}

/// Where a stepwise run starts, what it trains on and what it does on the way; the default is a
/// run from step 0, the initializers and the synthetic data, that evaluates and writes nothing.
#[derive(Default)]
pub struct StepwiseOptions<'o, 'h> {
    /// The step the run starts at; an AdamW run starts at 0, as the CPU run does.
    pub start: u64,
    /// The values that replace the initializer draws of their parameters.
    pub init: BTreeMap<ParamId, HostTensor<f32>>,
    /// The dense or packed data to train on instead of the synthetic data.
    pub data: Option<Data<'o>>,
    /// The held-out evaluation, on packed data.
    pub eval: Option<Evaluation<'o, 'h>>,
    /// The dense eval data of `--eval`, measured with the parameters after the last update as
    /// [`crate::train::runner::evaluate_dense`] measures it on the CPU.
    pub dense_eval: Option<&'o DenseData>,
    /// With [`Self::dense_eval`], where the prediction of every dense eval batch goes, appended
    /// in batch order: what `infer --data --function` writes (story:decider-infer-on-data).
    pub dense_predictions: Option<&'o mut Vec<f32>>,
    /// The checkpoints to write on the way.
    pub checkpoints: Option<Checkpoints<'o>>,
    /// The most steps one command buffer records; `None` lets the runner choose
    /// ([`steps_per_submit`]).
    pub submit_steps: Option<u64>,
}

/// One step of a stepwise run as its observer sees it, after the host waited for it.
#[derive(Clone, Copy, Debug)]
pub struct StepSeen<'a> {
    /// The step, `k` of `Run::losses[k - start]`.
    pub step: u64,
    /// The loss after `step` updates.
    pub loss: f32,
    /// The held-out loss and accuracy of the parameters after `step` updates, at an evaluated
    /// step.
    pub held_out: Option<HeldOutMetrics>,
    /// The loss, and where it is defined the accuracy, of the dense eval data of `--eval` with
    /// the parameters after the last update: at the last step of a run given
    /// [`StepwiseOptions::dense_eval`].
    pub dense_eval: Option<Eval>,
    /// The parameters after `step` updates where the host holds them without another read-back:
    /// at the first step, the values it uploaded, and at the last, the values it read back.
    pub values: Option<&'a BTreeMap<ParamId, HostTensor<f32>>>,
    /// The training clock through the updates before `step` ([`RunStats::training`]): 0 at the
    /// first step.
    pub trained: Duration,
}

/// The observer of a stepwise run: an error stops the run at that step and is its error.
pub type Observe<'f> = dyn FnMut(&StepSeen<'_>) -> Result<(), Box<RunError>> + 'f;

/// Whether `artifact`'s step plan needs the stepwise run: it has a dropout kernel, the step
/// reads optimizer inputs (AdamW's rate, bias corrections and moments), or it has a `sigreg`
/// site, whose projection is drawn anew for every step, or it gathers the rows of non-zero
/// weight, which the host selects from every batch.
pub fn needed(compiled: &CompiledTraining, artifact: &CompiledArtifact) -> bool {
    artifact.execution.kernels.iter().any(reads_seed)
        || !compiled.training.optimizer.host_inputs().is_empty()
        || !sigreg_sites(&compiled.training.training.program).is_empty()
        || masked_rows::site(&compiled.training.training.program).is_some()
}

/// `E-TARGET-011` at the `attention` that reads it, when the dense data `data` or the dense eval
/// data `dense_eval` of a run of `compiled` hold a span longer than [`MAX_SPAN_KEYS`] keys
/// (story:attention-event-spans): the four attention kernels read at most that many keys of a
/// span, so the device would compute another result than the CPU interpreter, which reads the
/// span whole. Checked over every span of the files, before the first step; `None` when every
/// span is at most [`MAX_SPAN_KEYS`] keys long or the run reads no spans.
pub fn long_span_refused(
    compiled: &CompiledTraining,
    data: Option<Data<'_>>,
    dense_eval: Option<&DenseData>,
) -> Option<Diagnostic> {
    let dense = match data {
        Some(Data::Dense(dense)) => Some(dense),
        _ => None,
    };
    let long = [dense, dense_eval]
        .into_iter()
        .flatten()
        .find_map(|d| d.span_longer_than(&compiled.forward, MAX_SPAN_KEYS))?;
    let refused = Diagnostic::new(
        codes::NO_KERNEL,
        format!(
            "operator `attention` has no Vulkan kernel for a span of {} keys: its kernels read at \
             most {MAX_SPAN_KEYS} keys of a span",
            long.length
        ),
    )
    .with("input", long.input)
    .with("position", long.position.to_string())
    .with("length", long.length.to_string())
    .with("bound", format!("length <= {MAX_SPAN_KEYS}"))
    .note("split the span, or train and infer it with `--backend cpu`");
    Some(match long.read_at {
        Some(span) => refused.at(span),
        None => refused,
    })
}

/// Loads Vulkan, selects device `device` (0 when `None`), negotiates and opens it, with the
/// pipeline executable statistics when `profiled` and the device offers them, and runs
/// [`train_stepwise_profiled`] on it. A span longer than [`MAX_SPAN_KEYS`] keys in its dense
/// data is refused before a device is opened ([`long_span_refused`]).
pub fn train_stepwise_on_device(
    device: Option<usize>,
    profiled: bool,
    compiled: &CompiledTraining,
    artifact: &CompiledArtifact,
    options: StepwiseOptions<'_, '_>,
    observe: &mut Observe<'_>,
) -> Result<(Run, RunStats, Option<Profile>), DeviceRunError> {
    if let Some(refused) = long_span_refused(compiled, options.data, options.dense_eval) {
        return Err(DeviceRunError::Refused(refused));
    }
    on_device(device, profiled, artifact, |opened, profile, negotiated| {
        Ok(train_stepwise_profiled(
            opened, profile, negotiated, compiled, artifact, profiled, options, observe,
        )?)
    })
}

/// What one host input of the step is: the token of a dropout site, the projection of a
/// `sigreg` site of this type, a scalar of an AdamW step, a moment of a parameter, or an input of
/// the masked-row gather.
#[derive(Clone, Debug)]
enum Hosted {
    Token(String),
    Projection(String, TensorType),
    LrT,
    C1,
    C2,
    Moment(ParamId, Moment),
    /// One of the three inputs of the masked-row gather.
    Masked(MaskedInput),
}

/// Which input of the masked-row gather a host input is.
#[derive(Clone, Copy, Debug)]
enum MaskedInput {
    Rows,
    Labels,
    Weights,
}

/// The most steps one command buffer of a stepwise run records when the caller names none
/// ([`StepwiseOptions::submit_steps`]): past it a longer segment saves no measurable host time
/// (story:vulkan-stepwise-multi-step-submit).
pub const SUBMIT_STEPS: u64 = 8;

/// The most bytes the per-step uploads of the two segments a run keeps take together, unless
/// one step's upload alone is more.
pub const SUBMIT_UPLOAD_BYTES: u64 = 64 << 20;

/// The steps one command buffer of a stepwise run records at most when the caller names none:
/// [`SUBMIT_STEPS`], fewer when the uploads of two segments of steps of `step_bytes` each would
/// pass [`SUBMIT_UPLOAD_BYTES`], and at least 1. The evaluation and checkpoint schedule ends a
/// segment earlier: at a due checkpoint, before an evaluated step and before the last step.
pub fn steps_per_submit(step_bytes: u64) -> u64 {
    (SUBMIT_UPLOAD_BYTES / (2 * step_bytes).max(1)).clamp(1, SUBMIT_STEPS)
}

/// A segment of a stepwise run, steps `first..=through` recorded into one command buffer, and
/// what its steps show the observer once the host waited for it.
struct Segment {
    first: u64,
    through: u64,
    /// Which of the two upload, loss and timestamp buffers it uses.
    parity: usize,
    /// The held-out metrics of step `first`, evaluated before the segment.
    metrics: Option<HeldOutMetrics>,
    /// The `--eval` result, at the last step.
    dense_eval: Option<Eval>,
    /// The training clock when the segment was recorded.
    trained: Duration,
    /// The timestamps it wrote.
    stamps: u32,
}

/// What one word range of the per-step upload holds.
enum Fed {
    /// The data declaration of this name, `i32` ones as their bits.
    Data {
        name: String,
        ids: bool,
    },
    Hosted(Hosted),
}

/// Trains `compiled` on `device`, whose profile is `profile` and which was opened with
/// `negotiated`, by the execution plan of `artifact`, one command buffer and one host wait per
/// segment of steps and per evaluation (the module documentation), and calls `observe` for each
/// step after its segment's wait.
pub fn train_stepwise(
    device: &Device<'_>,
    profile: &DeviceProfile,
    negotiated: &Negotiated,
    compiled: &CompiledTraining,
    artifact: &CompiledArtifact,
    options: StepwiseOptions<'_, '_>,
    observe: &mut Observe<'_>,
) -> Result<(Run, RunStats), VulkanRunError> {
    let (run, stats, _) = train_stepwise_profiled(
        device, profile, negotiated, compiled, artifact, false, options, observe,
    )?;
    Ok((run, stats))
}

/// [`train_stepwise`], and when `profiled` with a timestamp before and after every dispatch of
/// each segment's command buffer on a device with timestamp queries, read back after the
/// segment's host wait (the module documentation): the run's commands, results and
/// [`RunStats`] are those of [`train_stepwise`], and the [`Profile`] is that of the chunked run,
/// `None` when not `profiled`. A run that diverges returns the profile of the steps it recorded,
/// without the final loss evaluation, in its [`DivergedOnDevice`].
#[allow(clippy::too_many_arguments)]
pub fn train_stepwise_profiled(
    device: &Device<'_>,
    profile: &DeviceProfile,
    negotiated: &Negotiated,
    compiled: &CompiledTraining,
    artifact: &CompiledArtifact,
    profiled: bool,
    options: StepwiseOptions<'_, '_>,
    observe: &mut Observe<'_>,
) -> Result<(Run, RunStats, Option<Profile>), VulkanRunError> {
    check_device(artifact, profile).map_err(|refused| VulkanRunError::Plan(refused.to_string()))?;
    let plan = &artifact.execution;
    let config = &compiled.config;
    let forward = &compiled.forward;
    let refused = |error: RunError| VulkanRunError::Run(Box::new(error));
    let StepwiseOptions {
        start,
        init,
        data,
        eval,
        dense_eval,
        mut dense_predictions,
        checkpoints,
        submit_steps,
    } = options;
    if let Some(long) = long_span_refused(compiled, data, dense_eval) {
        return Err(refused(RunError::Data(DataError::Refused(long))));
    }
    let end = start.checked_add(config.steps).ok_or_else(|| {
        refused(RunError::StepCount {
            start,
            steps: config.steps,
        })
    })?;
    let adamw: Option<AdamW> = match compiled.training.optimizer.rule {
        OptimizerRule::Sgd => None,
        OptimizerRule::AdamW(adamw) => Some(adamw),
        OptimizerRule::Muon(_) => {
            return Err(VulkanRunError::Plan(
                "the muon optimizer has no Vulkan run".into(),
            ));
        }
    };
    if let Some(adamw) = adamw {
        if start > 0 {
            return Err(refused(RunError::MomentsNotCheckpointed { start }));
        }
        if config.steps > 0 && adamw.warmup_steps >= end {
            return Err(refused(RunError::Schedule {
                warmup_steps: adamw.warmup_steps,
                steps: end,
            }));
        }
    }
    let lr = compiled.training.optimizer.lr;
    let mut initial = initialize(&forward.program, config.seed, Stream::Init)
        .map_err(|error| refused(error.into()))?;
    initial.extend(init);
    let mut feed = match data {
        Some(Data::Dense(dense)) => Feed::Dense(dense, DenseBatch::default()),
        Some(Data::Packed(packed)) => Feed::Packed(packed, PackedBatch::default()),
        None => Feed::Synthetic(synthetic(config, forward).map_err(|error| refused(error.into()))?),
    };
    let (every, mut write) = match checkpoints {
        Some(Checkpoints { every, write }) if every > 0 => (every, Some(write)),
        _ => (0, None),
    };
    if let Some(eval) = &eval
        && eval.every == 0
    {
        return Err(VulkanRunError::Plan(
            "a held-out evaluation every 0 steps".into(),
        ));
    }

    // The host inputs of the step, each with the view the plan reads it from.
    let sites = dropout_sites(&compiled.training.training.program);
    let projected = sigreg_sites(&compiled.training.training.program);
    let masked_site = masked_rows::site(&compiled.training.training.program);
    let mut moment_of = BTreeMap::new();
    for parameter in &plan.parameters {
        for moment in [Moment::First, Moment::Second] {
            moment_of.insert(
                moment_input(moment, &parameter.path),
                (parameter.param, moment),
            );
        }
    }
    let hosted = host_inputs(&compiled.training, forward.prediction, config)
        .map_err(|error| VulkanRunError::Plan(format!("the host inputs: {error}")))?;
    let mut fed: Vec<(BufferView, Fed)> = plan
        .data
        .iter()
        .map(|declaration| {
            (
                declaration.view,
                Fed::Data {
                    name: declaration.name.clone(),
                    ids: declaration.ty.dtype == DType::I32,
                },
            )
        })
        .collect();
    let mut moments: Vec<(BufferView, ParamId, Moment)> = Vec::new();
    for input in hosted {
        if input.view.buffer as usize >= plan.buffers.len() {
            return Err(VulkanRunError::Plan(format!(
                "host input {} is read from {}, which names no plan buffer",
                input.name, input.view
            )));
        }
        let what = if sites.contains(&input.name) {
            Hosted::Token(input.name.clone())
        } else if projected.contains(&input.name) {
            Hosted::Projection(input.name.clone(), input.ty.clone())
        } else if input.name == LR_T_INPUT && adamw.is_some() {
            Hosted::LrT
        } else if input.name == C1_INPUT && adamw.is_some() {
            Hosted::C1
        } else if input.name == C2_INPUT && adamw.is_some() {
            Hosted::C2
        } else if let Some(&(param, moment)) = moment_of.get(&input.name) {
            Hosted::Moment(param, moment)
        } else if masked_site.is_some() && input.name == masked_rows::ROWS_INPUT {
            Hosted::Masked(MaskedInput::Rows)
        } else if let Some(site) = &masked_site
            && input.name == site.labels_input()
        {
            Hosted::Masked(MaskedInput::Labels)
        } else if let Some(site) = &masked_site
            && input.name == site.weights_input()
        {
            Hosted::Masked(MaskedInput::Weights)
        } else {
            return Err(VulkanRunError::Plan(format!(
                "host input {} has no Vulkan run",
                input.name
            )));
        };
        match what {
            Hosted::Moment(param, moment) => moments.push((input.view, param, moment)),
            other => fed.push((input.view, Fed::Hosted(other))),
        }
    }

    // The device memory of the run: every buffer it makes is made after this.
    let memory = MemorySpan::start();
    let emitting = Instant::now();
    let modules = emit_kernels(artifact, profile)?;
    let emitted = emitting.elapsed();
    // Each `Dropout` at `p = 0`: the pipelines the loss steps run with dropout off.
    let mut kept_kernels: Vec<(usize, Kernel)> = Vec::new();
    for (at, kernel) in plan.kernels.iter().enumerate() {
        if kernel.op == "Dropout" {
            kept_kernels.push((at, without_dropout(kernel)?));
        }
    }
    let kept_modules = kept_kernels
        .iter()
        .map(|(_, kept)| emit_module_with(kept, profile, artifact.fast_math))
        .collect::<Result<Vec<_>, _>>()?;
    // Every kernel's pipeline, then the identities, one per distinct module, the dropout ones
    // under the run key (story:vulkan-pipeline-startup).
    let requests: Vec<PipelineRequest<'_>> = plan
        .kernels
        .iter()
        .zip(&modules)
        .map(|(kernel, module)| PipelineRequest {
            kernel,
            words: &module.spirv,
            seed: reads_seed(kernel).then_some(config.seed),
        })
        .chain(
            kept_kernels
                .iter()
                .zip(&kept_modules)
                .map(|((_, kept), module)| PipelineRequest {
                    kernel: kept,
                    words: &module.spirv,
                    seed: Some(config.seed),
                }),
        )
        .collect();
    let (made, pipeline_creation) = make_pipelines(device, profile, &requests)?;
    let each = made.each();
    let (pipelines, kept_pipelines) = each.split_at(plan.kernels.len());
    let pipelines: Vec<&Pipeline<'_>> = pipelines.to_vec();
    let identities: BTreeMap<usize, &Pipeline<'_>> = kept_kernels
        .iter()
        .map(|(at, _)| *at)
        .zip(kept_pipelines.iter().copied())
        .collect();
    // The statistics of every kernel's pipeline, read once when profiled on a device opened with
    // the pipeline executable extension, as the chunked run reads them.
    let pipeline_statistics = if profiled && device.pipeline_statistics_enabled() {
        PipelineStatistics::collect(
            plan.kernels
                .iter()
                .zip(&pipelines)
                .map(|(kernel, pipeline)| (kernel.id, pipeline.statistics())),
        )
    } else {
        PipelineStatistics::NotCaptured
    };

    let scratch: BTreeMap<u32, BufferView> = plan
        .step
        .iter()
        .filter(|step| step.op == PlanOp::Allocate)
        .flat_map(|step| step.writes.iter().map(|view| (view.buffer, *view)))
        .collect();
    let mut buffers = Vec::with_capacity(plan.buffers.len());
    for (index, &bytes) in plan.buffers.iter().enumerate() {
        buffers.push(match scratch.get(&(index as u32)) {
            Some(view) => device.scratch_buffer(view)?,
            None => device.device_local_buffer(bytes)?,
        });
    }

    // What goes up once: every parameter's initial value, then every moment, zero.
    let mut first_words: Vec<f32> = Vec::new();
    let mut first_views: Vec<BufferView> = Vec::new();
    for parameter in &plan.parameters {
        let view = parameter.views[0];
        let value = initial.get(&parameter.param).ok_or_else(|| {
            VulkanRunError::Plan(format!("parameter {} has no initial value", parameter.path))
        })?;
        check_count(&view, value.len())?;
        first_words.extend_from_slice(value.data());
        first_views.push(view);
    }
    for (view, param, _) in &moments {
        let value = initial.get(param).ok_or_else(|| {
            VulkanRunError::Plan(format!("the moment of {param:?} has no parameter value"))
        })?;
        check_count(view, value.len())?;
        first_words.extend(std::iter::repeat_n(0.0f32, value.len()));
        first_views.push(*view);
    }
    let mut first_upload = staging(device, 4 * first_words.len() as u64)?;
    if let Some(upload) = &mut first_upload {
        upload.write_f32(&first_words)?;
    }

    // What goes up before every step: the batch, the tokens and the scalars, one region per step
    // of a segment, in two buffers: the segment in flight's and the one recorded behind it.
    // A segment holds no more steps than the run has, so a named `submit_steps` past them sizes
    // the buffers for the run's steps (finding MS-3), and one whose buffers would pass `u64`
    // bytes is refused.
    let step_bytes: u64 = fed.iter().map(|(view, _)| view.size_bytes).sum();
    let cap = submit_steps
        .unwrap_or_else(|| steps_per_submit(step_bytes))
        .min((end - start).saturating_add(1))
        .max(1);
    let too_large = || {
        VulkanRunError::Plan(format!(
            "{cap} steps per command buffer of {step_bytes} upload bytes each exceed the sizes \
             a stepwise run addresses"
        ))
    };
    let upload_bytes = step_bytes.checked_mul(cap).ok_or_else(too_large)?;
    let uploads = [
        staging(device, upload_bytes)?,
        staging(device, upload_bytes)?,
    ];
    // Where each projection lies in the per-step upload: a held-out batch reads the projections
    // of the step it is evaluated at, as the CPU evaluation does.
    let mut projection_regions: Vec<(u64, BufferView)> = Vec::new();
    let mut at = 0;
    for (view, what) in &fed {
        if matches!(what, Fed::Hosted(Hosted::Projection(..))) {
            projection_regions.push((at, *view));
        }
        at += view.size_bytes;
    }

    // The held-out batches, laid out as the data views, uploaded once.
    let prediction = plan.prediction_view;
    let held_out = eval.as_ref().map(|eval| eval.held_out);
    let eval_batches = held_out.map_or(0, HeldOut::batches);
    let data_bytes: u64 = plan.data.iter().map(|d| d.view.size_bytes).sum();
    let mut eval_words: Vec<f32> = Vec::new();
    if let Some(held_out) = held_out {
        for b in 0..eval_batches {
            let batch = held_out
                .batch(b)
                .ok_or_else(|| VulkanRunError::Plan(format!("held-out batch {b} is missing")))?;
            for declaration in &plan.data {
                let ids = declaration.ty.dtype == DType::I32;
                push_data(
                    &mut eval_words,
                    &declaration.view,
                    &declaration.name,
                    ids,
                    batch.inputs(),
                    batch.ids(),
                )?;
            }
        }
    }
    let mut eval_upload = staging(device, 4 * eval_words.len() as u64)?;
    if let Some(upload) = &mut eval_upload {
        upload.write_f32(&eval_words)?;
    }
    // The held-out reduction on the device, unless the evaluation asks for the host's or the
    // prediction does not start its buffer, which a module binds whole.
    let device_tally = match &eval {
        Some(eval) if eval.reduction == HeldOutReduction::Device && prediction.offset == 0 => Some(
            DeviceTally::new(device, profile, eval.held_out, &prediction)?,
        ),
        _ => None,
    };

    // The dense eval batches of `--eval`, laid out as the data views, uploaded once.
    let dense_batches = dense_eval.map_or(0, |dense| dense.samples() / dense.batch());
    let dense_compared = dense_eval.and_then(DenseData::compared_target);
    let mut dense_batch = DenseBatch::default();
    let mut dense_words: Vec<f32> = Vec::new();
    if let Some(dense) = dense_eval {
        for b in 0..dense_batches {
            dense.fill_eval_batch(b, &mut dense_batch);
            for declaration in &plan.data {
                push_data(
                    &mut dense_words,
                    &declaration.view,
                    &declaration.name,
                    declaration.ty.dtype == DType::I32,
                    dense_batch.inputs(),
                    dense_batch.ids(),
                )?;
            }
        }
    }
    let mut dense_upload = staging(device, 4 * dense_words.len() as u64)?;
    if let Some(upload) = &mut dense_upload {
        upload.write_f32(&dense_words)?;
    }

    // What comes down: the loss every step and every dense eval batch, the prediction of one
    // batch after each held-out batch reduced on the host and each dense eval batch whose
    // accuracy is measured or whose prediction is kept, the state of the device reduction after
    // each evaluated step, the parameters and moments at the end and at each checkpoint.
    let dense_predicted = dense_compared.is_some() || dense_predictions.is_some();
    let host_tallied = if device_tally.is_some() {
        0
    } else {
        eval_batches
    };
    let predicted_batches = host_tallied + if dense_predicted { dense_batches } else { 0 };
    let download = download_bytes(&prediction, predicted_batches);
    // One loss slot per step of a segment, in two buffers as the uploads are.
    let loss_bytes = download.loss.checked_mul(cap).ok_or_else(too_large)?;
    let loss_downloads = [
        device.download_buffer(loss_bytes)?,
        device.download_buffer(loss_bytes)?,
    ];
    let mut prediction_download = download_buffer(device, download.prediction)?;
    let param_bytes: u64 = plan.parameters.iter().map(|p| p.views[0].size_bytes).sum();
    let moment_bytes: u64 = moments.iter().map(|(view, _, _)| view.size_bytes).sum();
    let state_download = device.download_buffer(param_bytes + moment_bytes)?;
    // The parameters and moments at the start of a segment, copied on the device as it starts,
    // one buffer per upload buffer: a run that diverges inside a segment the device ran past
    // restores them and runs the segment's steps again through the failing one (finding MS-2).
    let backups = [
        device.device_local_buffer(param_bytes + moment_bytes)?,
        device.device_local_buffer(param_bytes + moment_bytes)?,
    ];

    let dispatch_steps = plan
        .step
        .iter()
        .filter(|step| step.op == PlanOp::Dispatch)
        .count();
    let binds = if identities.is_empty() { 1 } else { 2 };
    let (tally_sets, tally_descriptors) = match device_tally {
        Some(_) => (DeviceTally::SETS, DeviceTally::DESCRIPTORS),
        None => (0, 0),
    };
    let arena = device.descriptor_arena(
        count_u32(binds * dispatch_steps + tally_sets)?,
        count_u32(binds * descriptors(plan) + tally_descriptors)?,
    )?;
    let stepping: Vec<&Pipeline<'_>> = pipelines.clone();
    let (step_actions, loss_at) = actions(plan, &arena, &stepping, &buffers)?;
    let kept: Vec<&Pipeline<'_>> = pipelines
        .iter()
        .enumerate()
        .map(|(at, pipeline)| identities.get(&at).copied().unwrap_or(*pipeline))
        .collect();
    let kept_actions = if identities.is_empty() {
        None
    } else {
        Some(actions(plan, &arena, &kept, &buffers)?.0)
    };
    let loss_actions: Vec<&Action<'_>> = loss_at
        .iter()
        .map(|&at| &kept_actions.as_ref().unwrap_or(&step_actions)[at])
        .collect();
    // A held-out batch reads the prediction, not the loss: the loss steps without the loss sum.
    let eval_actions: Vec<&Action<'_>> = action_positions(plan, &plan.prediction_steps())?
        .into_iter()
        .map(|at| &kept_actions.as_ref().unwrap_or(&step_actions)[at])
        .collect();
    // The actions of a training step: those of the plan steps it keeps, without the full head
    // only the prediction reads; every action of a plan whose loss reads the prediction.
    let train_actions: Vec<&Action<'_>> = action_positions(plan, &plan.training_steps())?
        .into_iter()
        .map(|at| &step_actions[at])
        .collect();
    let loss = buffer(&buffers, &plan.loss_view)?;
    let tally_bound = match &device_tally {
        Some(tally) => Some(tally.bind(&arena, buffer(&buffers, &prediction)?)?),
        None => None,
    };

    // The dispatch sites a profile times, and the two timestamp pools, one per upload buffer, that
    // a segment's command buffer resets and writes: two queries per dispatch of each training step
    // of a segment or of the final loss evaluation, whichever records more. A training step records
    // `train_actions`, so its sites are theirs (story:vulkan-mlm-masked-row-gather).
    let site = |plan_name: PlanName, action: &Action<'_>| match action {
        Action::Dispatch(_, at) => {
            let id = plan.step[*at].kernel_module_id?;
            let entry = plan
                .kernels
                .iter()
                .zip(&modules)
                .find(|(kernel, _)| kernel.id == id)
                .map(|(_, module)| module.entry_point.clone())?;
            Site::new(plan, plan_name, *at, entry)
        }
        _ => None,
    };
    let step_sites: Vec<Site> = train_actions
        .iter()
        .filter_map(|action| site(PlanName::Step, action))
        .collect();
    let final_sites: Vec<Site> = loss_actions
        .iter()
        .filter_map(|action| site(PlanName::Final, action))
        .collect();
    let clock = device
        .clock()
        .filter(|_| profiled && profile.timestamp_queries);
    let segment_sites = usize::try_from(cap)
        .ok()
        .and_then(|cap| step_sites.len().checked_mul(cap))
        .and_then(|sites| sites.max(final_sites.len()).checked_mul(2))
        .ok_or_else(too_large)?;
    let queries = count_u32(segment_sites)?;
    let pools = match clock {
        Some(_) => [
            Some(device.timestamp_pool(queries)?),
            Some(device.timestamp_pool(queries)?),
        ],
        None => [None, None],
    };
    // The stamps read back so far: each recorded step's, then the final loss evaluation's.
    let mut stamped: Vec<u64> = Vec::new();
    // The profile of the steps whose stamps `stamped` holds, with the final loss evaluation's
    // when `finished`.
    let profile_of = |stamped: &[u64], finished: bool| -> Result<Option<Profile>, VulkanRunError> {
        if !profiled {
            return Ok(None);
        }
        let timings = match clock {
            Some(clock) => {
                let finals: &[Site] = if finished { &final_sites } else { &[] };
                let per_step = 2 * step_sites.len();
                let steps = match per_step {
                    0 => 0,
                    _ => stamped.len().saturating_sub(2 * finals.len()) / per_step,
                };
                Some(
                    timings(&step_sites, finals, steps as u64, stamped, clock)
                        .map_err(VulkanRunError::Plan)?,
                )
            }
            None => None,
        };
        Ok(Some(Profile {
            timings,
            pipeline_creation,
            compile: emitted,
            pipeline_statistics: pipeline_statistics.clone(),
        }))
    };

    let class = numerics_class(negotiated, artifact);
    let waits_before = device.host_waits();
    let submits_before = device.submits();
    let dispatches_before = device.dispatches();
    let uploaded_before = device.bytes_uploaded();
    let downloaded_before = device.bytes_downloaded();
    let mut dispatches_per_step = 0;
    let mut barriers_per_step = 0;
    let mut first_submit: Option<Instant> = None;
    // The training clock and the held-out evaluations' (the module documentation).
    let mut training = Duration::ZERO;
    let mut evaluation = Duration::ZERO;
    let count = |ran: u64,
                 dispatches_per_step: u64,
                 barriers_per_step: u64,
                 first_submit: Option<Instant>,
                 training: Duration,
                 evaluation: Duration| RunStats {
        steps: ran,
        host_waits: device.host_waits() - waits_before,
        submits: device.submits() - submits_before,
        dispatches: device.dispatches() - dispatches_before,
        dispatches_per_step,
        barriers_per_step,
        bytes_uploaded: device.bytes_uploaded() - uploaded_before,
        bytes_downloaded: device.bytes_downloaded() - downloaded_before,
        device_allocations: memory.allocations(),
        peak_device_bytes: memory.peak_bytes(),
        class,
        elapsed: first_submit.map_or_else(Default::default, |at| at.elapsed()),
        training,
        evaluation,
        pipelines: made.distinct() as u64,
        pipeline_creation,
    };
    // The parameters, then the moments, as the state download holds them.
    let state_views: Vec<BufferView> = plan
        .parameters
        .iter()
        .map(|p| p.views[0])
        .chain(moments.iter().map(|(view, _, _)| *view))
        .collect();
    // The parameters and moments of a state read-back.
    let read_state = |read: &[f32]| -> Result<_, VulkanRunError> {
        let values = parameters(plan, &initial, read)?;
        let mut at = (param_bytes / 4) as usize;
        let mut read_moments = BTreeMap::new();
        for (view, param, moment) in &moments {
            let count = view.element_count as usize;
            let shape = initial[param].shape().to_vec();
            let value = HostTensor::new(shape, read[at..at + count].to_vec())
                .map_err(|error| VulkanRunError::Plan(format!("a moment: {error}")))?;
            read_moments.insert((*param, *moment), value);
            at += count;
        }
        Ok((values, read_moments))
    };

    let mut losses: Vec<f32> = Vec::new();
    let mut words: Vec<f32> =
        Vec::with_capacity(usize::try_from(upload_bytes / 4).map_err(|_| too_large())?);
    let mut finished = None;
    // Whether a command buffer of the run was recorded ([`open`]).
    let mut opened = false;
    // Whether step `s` is evaluated, whether a checkpoint is due after its update, and whether
    // its command buffer copies the parameters and moments down.
    let evaluated = |s: u64| {
        eval.as_ref()
            .is_some_and(|eval| s.is_multiple_of(eval.every) || s == end)
    };
    let writes = write.is_some();
    let due = |s: u64| s < end && writes && (s + 1).is_multiple_of(every) && s + 1 < end;
    let reads_state = |s: u64| s == end || due(s);
    // The segment in flight, and the segments waited for and not yet shown to the observer.
    let mut in_flight: Option<(Submitted<'_>, Segment)> = None;
    let mut waited: Vec<Segment> = Vec::new();
    // Since when segments have been in flight without a pause: the open span of the training
    // clock.
    let mut busy_since: Option<Instant> = None;
    // The upload, loss and timestamp buffers of the next segment: 0 or 1, in turn.
    let mut parity = 0usize;
    let mut next = Some(start);
    loop {
        for segment in waited.drain(..) {
            let Segment {
                first,
                through,
                parity: was,
                metrics,
                dense_eval: dense_measured,
                trained,
                stamps,
            } = segment;
            if let Some(pool) = &pools[was] {
                stamped.extend(pool.read_first(stamps)?);
            }
            let segment_losses: Vec<f32> =
                loss_downloads[was].mapped_f32()?[..(through - first + 1) as usize].to_vec();
            let mut state = if reads_state(through) {
                Some(read_state(state_download.mapped_f32()?)?)
            } else {
                None
            };
            for s in first..=through {
                let last = s == end;
                let loss_s = segment_losses[(s - first) as usize];
                let state_s = if s == through { state.take() } else { None };
                let values = if s == start {
                    Some(&initial)
                } else if last {
                    state_s.as_ref().map(|(values, _)| values)
                } else {
                    None
                };
                observe(&StepSeen {
                    step: s,
                    loss: loss_s,
                    held_out: if s == first { metrics } else { None },
                    dense_eval: if s == first { dense_measured } else { None },
                    values,
                    trained,
                })
                .map_err(VulkanRunError::Run)?;
                losses.push(loss_s);
                let unread = BTreeMap::new();
                let diagnostic = match &state_s {
                    Some((values, read_moments)) => {
                        non_finite(compiled, s, loss_s, values, read_moments)
                    }
                    None => non_finite(compiled, s, loss_s, &unread, &BTreeMap::new()),
                };
                if let Some(diagnostic) = diagnostic {
                    // The device may also have run the rest of the segment and the segment in
                    // flight, past step `s`.
                    if let Some((submitted, _)) = in_flight.take() {
                        submitted.wait()?;
                    }
                    if let Some(since) = busy_since.take() {
                        training += since.elapsed();
                    }
                    let (values, read_moments) = match state_s {
                        Some(state) => state,
                        None => {
                            // One more host wait: the parameters and moments after step `s`, as
                            // one step per command buffer leaves them (finding MS-2). The
                            // segment's start state comes back from its backup and its steps
                            // through `s` run again from its uploads, which no later segment
                            // has overwritten, and the result comes down.
                            let mut recording = device.record()?;
                            recording.barrier(Barrier::ComputeToCompute);
                            recording.barrier(Barrier::ComputeToTransferWrite);
                            recording.barrier(Barrier::TransferToTransfer);
                            restore_state(&mut recording, &state_views, &buffers, &backups[was])?;
                            for r in (first..=s).filter(|r| *r != end) {
                                let slot = r - first;
                                if slot > 0 {
                                    recording.barrier(Barrier::ComputeToCompute);
                                    recording.barrier(Barrier::ComputeToTransfer);
                                    recording.barrier(Barrier::TransferToTransfer);
                                }
                                if let Some(upload) = &uploads[was] {
                                    let mut at = slot * step_bytes;
                                    for (view, _) in &fed {
                                        recording.copy_region(
                                            upload,
                                            at,
                                            buffer(&buffers, view)?,
                                            view.offset,
                                            view.size_bytes,
                                        )?;
                                        at += view.size_bytes;
                                    }
                                }
                                recording.barrier(Barrier::TransferToCompute);
                                record(
                                    &mut recording,
                                    train_actions.iter().copied(),
                                    &buffers,
                                    None,
                                )?;
                            }
                            recording.barrier(Barrier::ComputeToTransfer);
                            recording.barrier(Barrier::TransferToTransfer);
                            copy_state(&mut recording, &state_views, &buffers, &state_download)?;
                            recording.barrier(Barrier::TransferToHost);
                            let recorded = recording.finish()?;
                            device.submit(recorded)?.wait()?;
                            read_state(state_download.mapped_f32()?)?
                        }
                    };
                    let version = if last { s } else { s + 1 };
                    let mut state = state_of(compiled, values, version);
                    state.moments.extend(read_moments);
                    let run = Run {
                        start_step: start,
                        initial: initial.clone(),
                        losses,
                        state,
                        teacher: feed.teacher(),
                    };
                    let ran = s - start;
                    return Err(VulkanRunError::NonFinite(Box::new(DivergedOnDevice {
                        diverged: Diverged { diagnostic, run },
                        stats: count(
                            ran,
                            dispatches_per_step,
                            barriers_per_step,
                            first_submit,
                            training,
                            evaluation,
                        ),
                        profile: profile_of(&stamped, last)?,
                    })));
                }
                if let Some((values, read_moments)) = state_s {
                    if due(s)
                        && let Some(write) = write.as_mut()
                    {
                        write(s + 1, &values).map_err(|error| refused(error.into()))?;
                    }
                    if last {
                        finished = Some((values, read_moments));
                    }
                }
            }
        }
        let Some(k) = next else {
            break;
        };
        // The segment of steps `k..=through` one command buffer records: at most `cap` steps,
        // ending at a due checkpoint and before an evaluated step and the last step, which is a
        // segment of its own.
        let through = if k == end {
            end
        } else {
            let mut s = k;
            while s + 1 - k < cap && !due(s) && s + 1 < end && !evaluated(s + 1) {
                s += 1;
            }
            s
        };
        // A segment that starts with an evaluation or ends the run waits for the one in flight
        // first: its evaluation reads the parameters after step `k - 1`, and the run's last
        // command buffer is timed apart from the training clock.
        if (k == end || evaluated(k))
            && let Some((submitted, segment)) = in_flight.take()
        {
            submitted.wait()?;
            waited.push(segment);
            if let Some(since) = busy_since.take() {
                training += since.elapsed();
            }
            continue;
        }
        if let Some(upload) = &uploads[parity] {
            words.clear();
            for s in k..=through {
                let (inputs, ids) = feed.inputs(s).map_err(|error| refused(error.into()))?;
                let scalars = adamw.map(|adamw| adamw.scalars(lr, s + 1, end));
                let gathered = match &masked_site {
                    Some(site) => Some(
                        masked::gather(site, inputs, ids, s)
                            .map_err(|error| refused(error.into()))?,
                    ),
                    None => None,
                };
                for (view, what) in &fed {
                    match what {
                        Fed::Data { name, ids: is_ids } => {
                            push_data(&mut words, view, name, *is_ids, inputs, ids)?;
                        }
                        Fed::Hosted(Hosted::Token(site)) => {
                            let token = dropout_token(site, s);
                            check_count(view, token.len())?;
                            words.extend(token.data().iter().map(|&w| f32::from_bits(w as u32)));
                        }
                        Fed::Hosted(Hosted::Projection(site, ty)) => {
                            let projection = sigreg_projection(config.seed, site, s, ty)
                                .map_err(|error| refused(error.into()))?;
                            check_count(view, projection.len())?;
                            words.extend_from_slice(projection.data());
                        }
                        Fed::Hosted(Hosted::Masked(input)) => {
                            let gathered = gathered.as_ref().ok_or_else(|| {
                                VulkanRunError::Plan("a masked-row input without its site".into())
                            })?;
                            let as_bits = |values: &[i32]| -> Vec<f32> {
                                values.iter().map(|&v| f32::from_bits(v as u32)).collect()
                            };
                            let values = match input {
                                MaskedInput::Rows => as_bits(&gathered.rows),
                                MaskedInput::Labels => as_bits(&gathered.labels),
                                MaskedInput::Weights => gathered.weights.clone(),
                            };
                            check_count(view, values.len())?;
                            words.extend_from_slice(&values);
                        }
                        Fed::Hosted(scalar) => {
                            let scalars = scalars.ok_or_else(|| {
                                VulkanRunError::Plan("an optimizer scalar without adamw".into())
                            })?;
                            check_count(view, 1)?;
                            words.push(match scalar {
                                Hosted::LrT => scalars.lr_t,
                                Hosted::C1 => scalars.c1,
                                _ => scalars.c2,
                            });
                        }
                    }
                }
            }
            // SAFETY: the last submission that read this buffer, the segment two before this
            // one, was waited for when the segment after it was submitted: at most one segment
            // is in flight, and it reads the other buffer. Nothing takes a slice of an upload's
            // mapping.
            unsafe { upload.write_f32_shared(&words)? };
        }
        // An evaluated step reduced on the host: before the segment's command buffer, each
        // held-out batch in one of its own, read back and tallied before the next batch
        // overwrites the download. The projections it reads are step `k`'s, the first of the
        // segment's upload.
        let metrics = match held_out {
            Some(held_out) if evaluated(k) && tally_bound.is_none() => {
                let download = prediction_download.as_mut().ok_or_else(|| {
                    VulkanRunError::Plan("a held-out evaluation without its download".into())
                })?;
                let mut tally = held_out.tallying();
                let evaluating = Instant::now();
                for b in 0..eval_batches {
                    {
                        let mut recording = device.record()?;
                        open(
                            &mut recording,
                            &mut opened,
                            first_upload.as_ref(),
                            &first_views,
                            &buffers,
                        )?;
                        if let Some(upload) = &eval_upload {
                            let mut at = b as u64 * data_bytes;
                            for declaration in &plan.data {
                                let view = declaration.view;
                                recording.copy_region(
                                    upload,
                                    at,
                                    buffer(&buffers, &view)?,
                                    view.offset,
                                    view.size_bytes,
                                )?;
                                at += view.size_bytes;
                            }
                        }
                        if let Some(upload) = &uploads[parity] {
                            for (at, view) in &projection_regions {
                                recording.copy_region(
                                    upload,
                                    *at,
                                    buffer(&buffers, view)?,
                                    view.offset,
                                    view.size_bytes,
                                )?;
                            }
                        }
                        recording.barrier(Barrier::TransferToCompute);
                        record(&mut recording, eval_actions.iter().copied(), &buffers, None)?;
                        recording.barrier(Barrier::ComputeToTransfer);
                        recording.copy_region(
                            buffer(&buffers, &prediction)?,
                            prediction.offset,
                            download,
                            0,
                            prediction.size_bytes,
                        )?;
                        recording.barrier(Barrier::TransferToHost);
                        let recorded = recording.finish()?;
                        first_submit.get_or_insert_with(Instant::now);
                        device.submit(recorded)?.wait()?;
                    }
                    tally.add(download.mapped_f32()?).map_err(refused)?;
                }
                let metrics = tally.metrics().map_err(refused)?;
                evaluation += evaluating.elapsed();
                Some(metrics)
            }
            _ => None,
        };
        // An evaluated step reduced on the device: every held-out batch and the reduction in one
        // command buffer before the segment's, and the reduction's state read back after its
        // host wait, timed into the evaluation clock.
        let metrics = match (&device_tally, &tally_bound, held_out) {
            (Some(tally), Some(bound), Some(held_out)) if evaluated(k) => {
                let evaluating = Instant::now();
                {
                    let mut recording = device.record()?;
                    open(
                        &mut recording,
                        &mut opened,
                        first_upload.as_ref(),
                        &first_views,
                        &buffers,
                    )?;
                    for b in 0..eval_batches {
                        if b > 0 {
                            recording.barrier(Barrier::ComputeToCompute);
                            recording.barrier(Barrier::ComputeToTransfer);
                            recording.barrier(Barrier::TransferToTransfer);
                        }
                        if let Some(upload) = &eval_upload {
                            let mut at = b as u64 * data_bytes;
                            for declaration in &plan.data {
                                let view = declaration.view;
                                recording.copy_region(
                                    upload,
                                    at,
                                    buffer(&buffers, &view)?,
                                    view.offset,
                                    view.size_bytes,
                                )?;
                                at += view.size_bytes;
                            }
                        }
                        if let Some(upload) = &uploads[parity] {
                            for (at, view) in &projection_regions {
                                recording.copy_region(
                                    upload,
                                    *at,
                                    buffer(&buffers, view)?,
                                    view.offset,
                                    view.size_bytes,
                                )?;
                            }
                        }
                        tally.copy_targets(&mut recording, b)?;
                        recording.barrier(Barrier::TransferToCompute);
                        record(&mut recording, eval_actions.iter().copied(), &buffers, None)?;
                        recording.barrier(Barrier::ComputeToCompute);
                        tally.reduce(&mut recording, bound, b == 0);
                    }
                    recording.barrier(Barrier::ComputeToCompute);
                    recording.barrier(Barrier::ComputeToTransfer);
                    recording.barrier(Barrier::TransferToTransfer);
                    tally.copy_state(&mut recording)?;
                    recording.barrier(Barrier::TransferToHost);
                    let recorded = recording.finish()?;
                    first_submit.get_or_insert_with(Instant::now);
                    device.submit(recorded)?.wait()?;
                }
                let (numerator, correct) = tally.sums()?;
                let metrics = held_out
                    .metrics_of_sums(numerator, correct)
                    .map_err(refused)?;
                evaluation += evaluating.elapsed();
                Some(metrics)
            }
            _ => metrics,
        };

        // The last step of a run with `--eval`: before the step's segment, each dense eval
        // batch in one of its own, its loss, and its prediction when the accuracy is measured,
        // read back before the next batch overwrites them; the losses are summed in `f64` in
        // batch order, as the CPU evaluation sums them.
        let dense_measured = match dense_eval {
            Some(dense) if k == end => {
                let mut sum = 0.0f64;
                let mut correct = 0usize;
                let mut rows = 0usize;
                for b in 0..dense_batches {
                    {
                        let mut recording = device.record()?;
                        open(
                            &mut recording,
                            &mut opened,
                            first_upload.as_ref(),
                            &first_views,
                            &buffers,
                        )?;
                        if let Some(upload) = &dense_upload {
                            let mut at = b as u64 * data_bytes;
                            for declaration in &plan.data {
                                let view = declaration.view;
                                recording.copy_region(
                                    upload,
                                    at,
                                    buffer(&buffers, &view)?,
                                    view.offset,
                                    view.size_bytes,
                                )?;
                                at += view.size_bytes;
                            }
                        }
                        if let Some(upload) = &uploads[parity] {
                            for (at, view) in &projection_regions {
                                recording.copy_region(
                                    upload,
                                    *at,
                                    buffer(&buffers, view)?,
                                    view.offset,
                                    view.size_bytes,
                                )?;
                            }
                        }
                        recording.barrier(Barrier::TransferToCompute);
                        record(&mut recording, loss_actions.iter().copied(), &buffers, None)?;
                        recording.barrier(Barrier::ComputeToTransfer);
                        recording.copy_region(
                            loss,
                            plan.loss_view.offset,
                            &loss_downloads[parity],
                            0,
                            4,
                        )?;
                        if dense_predicted {
                            let download = prediction_download.as_ref().ok_or_else(|| {
                                VulkanRunError::Plan("a dense eval without its download".into())
                            })?;
                            recording.copy_region(
                                buffer(&buffers, &prediction)?,
                                prediction.offset,
                                download,
                                0,
                                prediction.size_bytes,
                            )?;
                        }
                        recording.barrier(Barrier::TransferToHost);
                        let recorded = recording.finish()?;
                        first_submit.get_or_insert_with(Instant::now);
                        device.submit(recorded)?.wait()?;
                    }
                    sum += f64::from(loss_downloads[parity].mapped_f32()?[0]);
                    let predicted = match prediction_download.as_mut() {
                        Some(download) if dense_predicted => Some(download.read_f32()?),
                        _ => None,
                    };
                    if let (Some(target), Some(predicted)) = (dense_compared, &predicted) {
                        dense.fill_eval_batch(b, &mut dense_batch);
                        let target = dense_batch.inputs().get(target).ok_or_else(|| {
                            VulkanRunError::Plan(format!(
                                "the dense eval target {target} is missing"
                            ))
                        })?;
                        let (batch_correct, batch_rows) = correct_rows(predicted, target);
                        correct += batch_correct;
                        rows += batch_rows;
                    }
                    if let (Some(kept), Some(predicted)) =
                        (dense_predictions.as_deref_mut(), predicted)
                    {
                        kept.extend_from_slice(&predicted);
                    }
                }
                Some(Eval {
                    loss: sum / dense_batches as f64,
                    accuracy: dense_compared.map(|_| correct as f64 / rows as f64),
                })
            }
            _ => None,
        };
        // The segment's command buffer: per step, its upload copied in from its region, its plan
        // steps (the loss steps without dropout at the last step), and its loss copied down to
        // its slot; the state after the segment's last step where it is read. It is submitted
        // behind the segment in flight, which the host then waits for, so the device runs one
        // segment while the host shows the one before to the observer and records the next.
        let trained = training;
        if busy_since.is_none() {
            busy_since = Some(Instant::now());
        }
        let mut recording = device.record()?;
        let mut stamps = pools[parity].as_ref().map(|pool| Stamps { pool, next: 0 });
        if let Some(stamps) = &stamps {
            recording.reset_timestamps(stamps.pool);
        }
        open(
            &mut recording,
            &mut opened,
            first_upload.as_ref(),
            &first_views,
            &buffers,
        )?;
        copy_state(&mut recording, &state_views, &buffers, &backups[parity])?;
        for s in k..=through {
            let slot = s - k;
            if slot > 0 {
                // The barriers that open a command buffer, and one from the copies of the step
                // before, so that this step's copies write its views after them.
                recording.barrier(Barrier::ComputeToCompute);
                recording.barrier(Barrier::ComputeToTransfer);
                recording.barrier(Barrier::TransferToTransfer);
            }
            let barriers_before = recorded_barriers();
            if let Some(upload) = &uploads[parity] {
                let mut at = slot * step_bytes;
                for (view, _) in &fed {
                    recording.copy_region(
                        upload,
                        at,
                        buffer(&buffers, view)?,
                        view.offset,
                        view.size_bytes,
                    )?;
                    at += view.size_bytes;
                }
            }
            recording.barrier(Barrier::TransferToCompute);
            let before = recording.dispatches();
            if s == end {
                record(
                    &mut recording,
                    loss_actions.iter().copied(),
                    &buffers,
                    stamps.as_mut(),
                )?;
            } else {
                record(
                    &mut recording,
                    train_actions.iter().copied(),
                    &buffers,
                    stamps.as_mut(),
                )?;
                dispatches_per_step = recording.dispatches() - before;
            }
            recording.barrier(Barrier::ComputeToTransfer);
            recording.copy_region(
                loss,
                plan.loss_view.offset,
                &loss_downloads[parity],
                4 * slot,
                4,
            )?;
            if s != end {
                barriers_per_step = recorded_barriers() - barriers_before;
            }
        }
        if reads_state(through) {
            copy_state(&mut recording, &state_views, &buffers, &state_download)?;
        }
        recording.barrier(Barrier::TransferToHost);
        let recorded = recording.finish()?;
        first_submit.get_or_insert_with(Instant::now);
        let submitted = device.submit(recorded)?;
        let segment = Segment {
            first: k,
            through,
            parity,
            metrics,
            dense_eval: dense_measured,
            trained,
            stamps: stamps.map_or(0, |stamps| stamps.next),
        };
        if let Some((behind, segment)) = in_flight.replace((submitted, segment)) {
            behind.wait()?;
            waited.push(segment);
        }
        // A segment whose state the host reads, and the last, is waited for at once.
        if reads_state(through)
            && let Some((submitted, segment)) = in_flight.take()
        {
            submitted.wait()?;
            waited.push(segment);
            if let Some(since) = busy_since.take()
                && through != end
            {
                training += since.elapsed();
            }
        }
        parity = 1 - parity;
        next = (through != end).then(|| through + 1);
    }
    let stats = count(
        end - start,
        dispatches_per_step,
        barriers_per_step,
        first_submit,
        training,
        evaluation,
    );
    let profiled = profile_of(&stamped, true)?;
    let (values, read_moments) = finished.expect("the last step reads the state back");
    let mut state = state_of(compiled, values, end);
    state.moments.extend(read_moments);
    Ok((
        Run {
            start_step: start,
            initial,
            losses,
            state,
            teacher: feed.teacher(),
        },
        stats,
        profiled,
    ))
}

/// The bytes of the host-visible buffers a stepwise run reads back through before its host
/// waits, apart from the parameters and moments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DownloadBytes {
    /// The loss of a step, one `f32`.
    pub loss: u64,
    /// One batch's prediction: the held-out logits, or the dense eval prediction whose accuracy
    /// is measured; 0 when no batch's prediction is read.
    pub prediction: u64,
}

impl DownloadBytes {
    /// Both buffers together.
    pub fn total(&self) -> u64 {
        self.loss + self.prediction
    }
}

/// The download buffers of a stepwise run whose plan predicts into `prediction` and which
/// reads the prediction of `eval_batches` held-out or dense eval batches: the loss, and one
/// batch's prediction whatever the number of batches, since each batch is evaluated in a command
/// buffer of its own and read back before the next (story:vulkan-eval-download-per-batch). The
/// held-out batches of an evaluation the device reduces ([`HeldOutReduction::Device`]) are not
/// among them: their state comes down through a buffer of its own.
pub fn download_bytes(prediction: &BufferView, eval_batches: usize) -> DownloadBytes {
    DownloadBytes {
        loss: 4,
        prediction: if eval_batches == 0 {
            0
        } else {
            prediction.size_bytes
        },
    }
}

/// The held-out reduction on the device ([`HeldOutReduction::Device`]): the kernels of
/// [`held_out_kernels`] for the split's batches, their pipelines, each batch's labels and
/// weights uploaded once, the device buffers the kernels bind beside the prediction, and the
/// download of the state.
struct DeviceTally<'d> {
    kernels: HeldOutKernels,
    /// The pipelines of `HeldOutRows`, the first `HeldOutSum` and the next ones.
    pipelines: [Pipeline<'d>; 3],
    /// Per batch, its labels' bits then its weights, `rows` words each.
    upload: Buffer<'d>,
    labels: Buffer<'d>,
    weights: Buffer<'d>,
    rows: Buffer<'d>,
    state: Buffer<'d>,
    download: Buffer<'d>,
    /// The bytes of one batch's labels, and of its weights.
    row_bytes: u64,
}

impl<'d> DeviceTally<'d> {
    /// The descriptor sets [`DeviceTally::bind`] takes from the run's arena, and their
    /// descriptors.
    const SETS: usize = 3;
    const DESCRIPTORS: usize = 9;

    /// The reduction of `held_out` on `device`, whose plan predicts into `prediction`: one
    /// batch's logits, `rows` rows of the split's classes.
    fn new(
        device: &'d Device<'_>,
        profile: &DeviceProfile,
        held_out: &HeldOut<'_>,
        prediction: &BufferView,
    ) -> Result<Self, VulkanRunError> {
        let classes = held_out.classes() as u64;
        let rows = held_out.targets(0).map_or(0, |(labels, _)| labels.len()) as u64;
        if rows.checked_mul(classes) != Some(prediction.element_count) {
            return Err(VulkanRunError::Plan(format!(
                "the prediction {prediction} is not {rows} held-out rows of {classes} logits"
            )));
        }
        let kernels = held_out_kernels(rows, classes)?;
        let pipeline = |kernel: &Kernel| -> Result<Pipeline<'d>, VulkanRunError> {
            let module = emit_module_with(kernel, profile, false)?;
            Ok(device.kernel_pipeline(kernel, &module.spirv)?)
        };
        let pipelines = [
            pipeline(&kernels.rows)?,
            pipeline(&kernels.first)?,
            pipeline(&kernels.next)?,
        ];
        let mut words = Vec::with_capacity(2 * rows as usize * held_out.batches());
        for b in 0..held_out.batches() {
            let (labels, weights) = held_out
                .targets(b)
                .filter(|(labels, weights)| {
                    labels.len() as u64 == rows && weights.len() as u64 == rows
                })
                .ok_or_else(|| {
                    VulkanRunError::Plan(format!("held-out batch {b} has not {rows} rows"))
                })?;
            words.extend(labels.iter().map(|&label| f32::from_bits(label as u32)));
            words.extend_from_slice(weights);
        }
        let mut upload = device.staging_buffer(4 * words.len() as u64)?;
        upload.write_f32(&words)?;
        let row_bytes = 4 * rows;
        Ok(DeviceTally {
            kernels,
            pipelines,
            upload,
            labels: device.device_local_buffer(row_bytes)?,
            weights: device.device_local_buffer(row_bytes)?,
            rows: device.device_local_buffer(2 * row_bytes)?,
            state: device.device_local_buffer(4 * HELD_OUT_STATE)?,
            download: device.download_buffer(4 * HELD_OUT_STATE)?,
            row_bytes,
        })
    }

    /// The three dispatches, bound in `arena`: `HeldOutRows` over `prediction`, the buffer of
    /// the plan's prediction view, then the first and the next `HeldOutSum`.
    fn bind<'a>(
        &'a self,
        arena: &'a DescriptorArena<'_>,
        prediction: &'a Buffer<'_>,
    ) -> Result<[Bound<'a>; 3], VulkanRunError> {
        let [rows, first, next] = &self.pipelines;
        let sum = [&self.rows, &self.weights, &self.state];
        Ok([
            arena.bind(
                rows,
                &self.kernels.rows,
                &[prediction, &self.labels, &self.rows],
            )?,
            arena.bind(first, &self.kernels.first, &sum)?,
            arena.bind(next, &self.kernels.next, &sum)?,
        ])
    }

    /// Copies batch `b`'s labels and weights into the buffers the kernels read.
    fn copy_targets<'r>(
        &'r self,
        recording: &mut Recording<'r>,
        b: usize,
    ) -> Result<(), VulkanRunError> {
        let at = 2 * b as u64 * self.row_bytes;
        recording.copy_region(&self.upload, at, &self.labels, 0, self.row_bytes)?;
        recording.copy_region(
            &self.upload,
            at + self.row_bytes,
            &self.weights,
            0,
            self.row_bytes,
        )?;
        Ok(())
    }

    /// Records the reduction of the batch whose prediction and targets are in place: its rows,
    /// a compute-to-compute barrier, and its sum, the evaluation's first when `first`.
    fn reduce<'r>(&self, recording: &mut Recording<'r>, bound: &'r [Bound<'_>; 3], first: bool) {
        recording.dispatch_bound(&bound[0]);
        recording.barrier(Barrier::ComputeToCompute);
        recording.dispatch_bound(&bound[if first { 1 } else { 2 }]);
    }

    /// Copies the state down, after the barrier that orders it after the last sum.
    fn copy_state<'r>(&'r self, recording: &mut Recording<'r>) -> Result<(), VulkanRunError> {
        recording.copy_region(&self.state, 0, &self.download, 0, 4 * HELD_OUT_STATE)?;
        Ok(())
    }

    /// The numerator and correct count of the state copied down, after the host wait.
    fn sums(&self) -> Result<(f64, u64), VulkanRunError> {
        let state: [f32; 4] = self.download.mapped_f32()?[..4]
            .try_into()
            .map_err(|_| VulkanRunError::Plan("the held-out state is not 4 floats".into()))?;
        Ok(held_out_sums(state))
    }
}

/// Starts a command buffer of a stepwise run. The first of the run copies the parameters'
/// initial values and the zero moments up from `first`, the views `views` one after the other;
/// every later one starts after the one before it, whose dispatches wrote what this one's read
/// and read the views its copies overwrite, and whose copies wrote the views and buffers this
/// one's copies overwrite (story:vulkan-stepwise-multi-step-submit, finding MS-1: a segment
/// submitted behind the one in flight runs after it with no host wait between them). Either way
/// the copies this one records next read and write after those before them.
fn open<'r>(
    recording: &mut Recording<'r>,
    opened: &mut bool,
    first: Option<&'r Buffer<'_>>,
    views: &[BufferView],
    buffers: &'r [Buffer<'_>],
) -> Result<(), VulkanRunError> {
    if *opened {
        recording.barrier(Barrier::ComputeToCompute);
        recording.barrier(Barrier::ComputeToTransfer);
        recording.barrier(Barrier::TransferToTransfer);
    } else if let Some(upload) = first {
        let mut at = 0;
        for view in views {
            recording.copy_region(
                upload,
                at,
                buffer(buffers, view)?,
                view.offset,
                view.size_bytes,
            )?;
            at += view.size_bytes;
        }
        recording.barrier(Barrier::TransferToTransfer);
    }
    *opened = true;
    Ok(())
}

/// Copies the views `views` of `buffers`, one after the other, to the start of `download`.
fn copy_state<'r>(
    recording: &mut Recording<'r>,
    views: &[BufferView],
    buffers: &'r [Buffer<'_>],
    download: &'r Buffer<'_>,
) -> Result<(), VulkanRunError> {
    let mut at = 0;
    for view in views {
        recording.copy_region(
            buffer(buffers, view)?,
            view.offset,
            download,
            at,
            view.size_bytes,
        )?;
        at += view.size_bytes;
    }
    Ok(())
}

/// Copies `from`, laid out as [`copy_state`] lays out its download, back into the views `views`
/// of `buffers`.
fn restore_state<'r>(
    recording: &mut Recording<'r>,
    views: &[BufferView],
    buffers: &'r [Buffer<'_>],
    from: &'r Buffer<'_>,
) -> Result<(), VulkanRunError> {
    let mut at = 0;
    for view in views {
        recording.copy_region(
            from,
            at,
            buffer(buffers, view)?,
            view.offset,
            view.size_bytes,
        )?;
        at += view.size_bytes;
    }
    Ok(())
}

/// `kernel`, a `Dropout`, at drop probability 0: it keeps every element and scales it by 1.
fn without_dropout(kernel: &Kernel) -> Result<Kernel, VulkanRunError> {
    let mut kept = kernel.clone();
    match kept.operands.last_mut() {
        Some(Operand::Immediate(p)) => *p = 0.0,
        _ => {
            return Err(VulkanRunError::Plan(format!(
                "dropout kernel {} carries no drop probability",
                kernel.id
            )));
        }
    }
    kept.id = Kernel::module_id(&kept.signature());
    Ok(kept)
}

/// A staging buffer of `bytes`, or none for 0.
fn staging<'d>(device: &'d Device<'_>, bytes: u64) -> Result<Option<Buffer<'d>>, VulkanRunError> {
    if bytes == 0 {
        return Ok(None);
    }
    Ok(Some(device.staging_buffer(bytes)?))
}

/// A download buffer of `bytes`, or none for 0.
fn download_buffer<'d>(
    device: &'d Device<'_>,
    bytes: u64,
) -> Result<Option<Buffer<'d>>, VulkanRunError> {
    if bytes == 0 {
        return Ok(None);
    }
    Ok(Some(device.download_buffer(bytes)?))
}

fn check_count(view: &BufferView, count: usize) -> Result<(), VulkanRunError> {
    if count as u64 != view.element_count {
        return Err(VulkanRunError::Plan(format!(
            "{view} holds {} elements, the host value {count}",
            view.element_count
        )));
    }
    Ok(())
}

/// Appends the words of the data declaration `name`, read from `view`: the `f32` tensor of that
/// name, or with `ids` the `i32` one as its bits.
fn push_data(
    words: &mut Vec<f32>,
    view: &BufferView,
    name: &str,
    ids: bool,
    inputs: &BTreeMap<String, HostTensor<f32>>,
    id_inputs: &BTreeMap<String, HostTensor<i32>>,
) -> Result<(), VulkanRunError> {
    let missing = || VulkanRunError::Plan(format!("data {name} has no batch tensor"));
    if ids {
        let value = id_inputs.get(name).ok_or_else(missing)?;
        check_count(view, value.len())?;
        words.extend(value.data().iter().map(|&w| f32::from_bits(w as u32)));
    } else {
        let value = inputs.get(name).ok_or_else(missing)?;
        check_count(view, value.len())?;
        words.extend_from_slice(value.data());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use modelspec_core::compile::compile_training;
    use modelspec_core::ir::tensor::TensorType;

    #[test]
    fn a_dropout_without_its_probability_keeps_every_element() {
        let kernel = modelspec_core::schedule::rowwise_kernel(
            &modelspec_core::ir::tensor::TensorOp::Dropout { p: 0.1 },
            vec![
                Operand::Buffer(TensorType::f32(&[4, 8])),
                Operand::Buffer(TensorType::i32(&[2])),
            ],
            TensorType::f32(&[4, 8]),
        )
        .expect("a dropout kernel");
        let kept = without_dropout(&kernel).expect("a drop probability");
        assert_eq!(kept.operands.last(), Some(&Operand::Immediate(0.0)));
        assert_ne!(kept.id, kernel.id);
        assert_eq!(kept.id, Kernel::module_id(&kept.signature()));
    }

    #[test]
    fn an_sgd_step_without_dropout_runs_chunked() {
        let source = include_str!("../../../../examples/linear_regression.mspec");
        let compiled = compile_training("fit.mspec", source).expect("the source compiles");
        let artifact = modelspec_core::compile::compile_artifact(
            "fit.mspec",
            source,
            modelspec_core::artifact::Backend::Vulkan,
        )
        .expect("the artifact compiles");
        assert!(!needed(&compiled, &artifact));
    }

    #[test]
    fn a_segment_holds_eight_steps_unless_two_segments_of_uploads_pass_the_budget() {
        assert_eq!(steps_per_submit(0), SUBMIT_STEPS);
        assert_eq!(steps_per_submit(64 << 10), SUBMIT_STEPS);
        assert_eq!(
            steps_per_submit(SUBMIT_UPLOAD_BYTES / (2 * SUBMIT_STEPS)),
            SUBMIT_STEPS
        );
        assert_eq!(
            steps_per_submit(SUBMIT_UPLOAD_BYTES / (2 * SUBMIT_STEPS) + 1),
            7
        );
        assert_eq!(steps_per_submit(SUBMIT_UPLOAD_BYTES / 4), 2);
        assert_eq!(steps_per_submit(SUBMIT_UPLOAD_BYTES), 1);
        assert_eq!(steps_per_submit(u64::MAX / 2), 1);
    }
}
