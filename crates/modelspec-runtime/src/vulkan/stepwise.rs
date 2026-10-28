//! A training run on one Vulkan device that the host feeds before every step (coordinator decision
//! D73, story:bert-vulkan-kernels-attention-loss-adamw): one command buffer and one host wait per
//! step, and one per evaluated held-out batch, for a step plan that reads more than its
//! parameters and one batch.
//!
//! [`super::runner::train`] records up to [`super::runner::CHUNK_STEPS`] steps into one command
//! buffer and uploads once, so it runs a step plan whose only inputs are the parameters and the
//! synthetic batch. [`train_stepwise`] runs the plans it refuses:
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
//!   final loss reading the projections of the last step and a held-out batch those of the step
//!   it is evaluated at, as the CPU run draws them;
//! - dense or packed `--data` ([`Data`]): the batch of step `k` is uploaded before it, as the CPU
//!   run feeds it ([`crate::train::runner::run_checked`]);
//! - held-out evaluation ([`Evaluation`], `--eval-every`): at every multiple of `every` and the last
//!   step, before the step's command buffer, each `HeldOut` batch is a command buffer of its own
//!   (coordinator decision D76, story:vulkan-eval-download-per-batch): it copies the batch into
//!   the data views, runs the steps of the plan the loss is computed from
//!   ([`modelspec_core::ir::kernel::ExecutionPlan::loss_steps`]) with dropout off and copies the
//!   prediction into a download buffer of one prediction ([`download_bytes`]), and after its
//!   host wait the host adds the logits to the reduction of the CPU evaluation, in batch order
//!   ([`HeldOut::tallying`]), before the next batch overwrites them.
//!
//! Dropout is off where the CPU run has none: in the evaluation of the loss after the last
//! update and in held-out evaluation, which the CPU run computes with the forward program. There
//! the loss steps run with each `Dropout` pipeline replaced by the one of the same kernel at
//! `p = 0`, which keeps every element and scales it by 1, so it copies its input.
//!
//! The first command buffer of the run, a held-out batch's or step `start`'s, copies the
//! parameters and the zero moments up. Step `k`'s command buffer, for `k` from the start to the
//! end, after the held-out batches of an evaluated step: the step's batch, tokens and scalars are
//! copied up, the step plan runs (the loss steps without dropout at the end), and the loss is
//! copied down, with the parameters and moments after the last step and at each due checkpoint.
//! The host waits once per step and once per held-out batch of an evaluated step, checks the
//! loss, and the parameters and moments where it read them, with E-TRAIN-001 as the CPU run
//! does, and shows the step to the observer ([`StepSeen`]). A step whose loss is not
//! finite stops the run with one more host wait, which reads the parameters and moments the
//! diverged run holds.

use std::collections::BTreeMap;
use std::time::Instant;

use modelspec_core::artifact::{CompiledArtifact, check_device};
use modelspec_core::compile::CompiledTraining;
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{BufferView, Kernel, Operand, PlanOp};
use modelspec_core::ir::tensor::{DType, ParamId, TensorType};
use modelspec_core::ir::training::{
    AdamW, C1_INPUT, C2_INPUT, LR_T_INPUT, Moment, OptimizerRule, dropout_sites, moment_input,
    sigreg_sites,
};
use modelspec_core::schedule::host_inputs;

use super::Device;
use super::dispatch::Recording;
use super::memory::{Buffer, MemorySpan};
use super::negotiate::Negotiated;
use super::pipeline::{Pipeline, reads_seed};
use super::runner::{
    Action, Checkpoints, DeviceRunError, DivergedOnDevice, RunStats, VulkanRunError, actions,
    buffer, count_u32, descriptors, numerics_class, on_device, parameters, record, state_of,
};
use super::sync::{Barrier, recorded_barriers};
use crate::cpu::HostTensor;
use crate::spirv::emit::{emit_kernels, emit_module_with};
use crate::train::data::{DenseBatch, PackedBatch, synthetic};
use crate::train::eval::{HeldOut, HeldOutMetrics};
use crate::train::init::{Stream, dropout_token, initialize, sigreg_projection};
use crate::train::runner::{Data, Diverged, Feed, Run, RunError, non_finite};

/// The held-out evaluation of a stepwise run: `held_out` at every multiple of `every` among the
/// run's steps and at its last step, as `--eval-every` asks.
pub struct Evaluation<'o, 'h> {
    pub held_out: &'o HeldOut<'h>,
    /// The steps between two evaluations: positive.
    pub every: u64,
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
    /// The checkpoints to write on the way.
    pub checkpoints: Option<Checkpoints<'o>>,
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
    /// The parameters after `step` updates where the host holds them without another read-back:
    /// at the first step, the values it uploaded, and at the last, the values it read back.
    pub values: Option<&'a BTreeMap<ParamId, HostTensor<f32>>>,
}

/// Why a stepwise run is not profiled: `--profile` times the chunked run's dispatch sites.
pub const NOT_PROFILED: &str = "--profile times the chunked device run; a run with dropout, \
                                sigreg, AdamW, --data or --eval-every records one command \
                                buffer per step and has no profile yet";

/// The observer of a stepwise run: an error stops the run at that step and is its error.
pub type Observe<'f> = dyn FnMut(&StepSeen<'_>) -> Result<(), Box<RunError>> + 'f;

/// Whether `artifact`'s step plan needs the stepwise run: it has a dropout kernel, the step
/// reads optimizer inputs (AdamW's rate, bias corrections and moments), or it has a `sigreg`
/// site, whose projection is drawn anew for every step.
pub fn needed(compiled: &CompiledTraining, artifact: &CompiledArtifact) -> bool {
    artifact.execution.kernels.iter().any(reads_seed)
        || !compiled.training.optimizer.host_inputs().is_empty()
        || !sigreg_sites(&compiled.training.training.program).is_empty()
}

/// Loads Vulkan, selects device `device` (0 when `None`), negotiates and opens it, and runs
/// [`train_stepwise`] on it.
pub fn train_stepwise_on_device(
    device: Option<usize>,
    compiled: &CompiledTraining,
    artifact: &CompiledArtifact,
    options: StepwiseOptions<'_, '_>,
    observe: &mut Observe<'_>,
) -> Result<(Run, RunStats), DeviceRunError> {
    on_device(device, false, artifact, |opened, profile, negotiated| {
        Ok(train_stepwise(
            opened, profile, negotiated, compiled, artifact, options, observe,
        )?)
    })
}

/// What one host input of the step is: the token of a dropout site, the projection of a
/// `sigreg` site of this type, a scalar of an AdamW step, or a moment of a parameter.
#[derive(Clone, Debug)]
enum Hosted {
    Token(String),
    Projection(String, TensorType),
    LrT,
    C1,
    C2,
    Moment(ParamId, Moment),
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
/// step and per evaluated held-out batch (the module documentation), and calls `observe` after
/// each step's wait.
pub fn train_stepwise(
    device: &Device<'_>,
    profile: &DeviceProfile,
    negotiated: &Negotiated,
    compiled: &CompiledTraining,
    artifact: &CompiledArtifact,
    options: StepwiseOptions<'_, '_>,
    observe: &mut Observe<'_>,
) -> Result<(Run, RunStats), VulkanRunError> {
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
        checkpoints,
    } = options;
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
    let modules = emit_kernels(artifact, profile)?;
    let mut pipelines = Vec::with_capacity(plan.kernels.len());
    for (kernel, module) in plan.kernels.iter().zip(&modules) {
        pipelines.push(if reads_seed(kernel) {
            device.kernel_pipeline_seeded(kernel, &module.spirv, config.seed)?
        } else {
            device.kernel_pipeline(kernel, &module.spirv)?
        });
    }
    // Each `Dropout` at `p = 0`: the pipelines the loss steps run with dropout off.
    let mut identities: BTreeMap<usize, Pipeline<'_>> = BTreeMap::new();
    for (at, kernel) in plan.kernels.iter().enumerate() {
        if kernel.op == "Dropout" {
            let kept = without_dropout(kernel)?;
            let module = emit_module_with(&kept, profile, artifact.fast_math)?;
            identities.insert(
                at,
                device.kernel_pipeline_seeded(&kept, &module.spirv, config.seed)?,
            );
        }
    }

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

    // What goes up before every step: the batch, the tokens and the scalars.
    let step_bytes: u64 = fed.iter().map(|(view, _)| view.size_bytes).sum();
    let mut step_upload = staging(device, step_bytes)?;
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

    // What comes down: the loss every step, the logits of one held-out batch after each
    // evaluated batch, the parameters and moments at the end and at each checkpoint.
    let download = download_bytes(&prediction, eval_batches);
    let mut loss_download = device.download_buffer(download.loss)?;
    let mut prediction_download = download_buffer(device, download.prediction)?;
    let param_bytes: u64 = plan.parameters.iter().map(|p| p.views[0].size_bytes).sum();
    let moment_bytes: u64 = moments.iter().map(|(view, _, _)| view.size_bytes).sum();
    let mut state_download = device.download_buffer(param_bytes + moment_bytes)?;

    let dispatch_steps = plan
        .step
        .iter()
        .filter(|step| step.op == PlanOp::Dispatch)
        .count();
    let binds = if identities.is_empty() { 1 } else { 2 };
    let arena = device.descriptor_arena(
        count_u32(binds * dispatch_steps)?,
        count_u32(binds * descriptors(plan))?,
    )?;
    let stepping: Vec<&Pipeline<'_>> = pipelines.iter().collect();
    let (step_actions, loss_at) = actions(plan, &arena, &stepping, &buffers)?;
    let kept: Vec<&Pipeline<'_>> = pipelines
        .iter()
        .enumerate()
        .map(|(at, pipeline)| identities.get(&at).unwrap_or(pipeline))
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
    let loss = buffer(&buffers, &plan.loss_view)?;

    let class = numerics_class(negotiated, artifact);
    let waits_before = device.host_waits();
    let submits_before = device.submits();
    let dispatches_before = device.dispatches();
    let uploaded_before = device.bytes_uploaded();
    let downloaded_before = device.bytes_downloaded();
    let mut dispatches_per_step = 0;
    let mut barriers_per_step = 0;
    let mut first_submit: Option<Instant> = None;
    let count = |ran: u64,
                 dispatches_per_step: u64,
                 barriers_per_step: u64,
                 first_submit: Option<Instant>| RunStats {
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
    let mut words: Vec<f32> = Vec::with_capacity((step_bytes / 4) as usize);
    let mut finished = None;
    // Whether a command buffer of the run was recorded ([`open`]).
    let mut opened = false;
    for k in start..=end {
        let last = k == end;
        let evaluated = eval
            .as_ref()
            .is_some_and(|eval| k.is_multiple_of(eval.every) || last);
        let due = !last && write.is_some() && (k + 1).is_multiple_of(every) && k + 1 < end;
        let reads_state = last || due;
        if let Some(upload) = &mut step_upload {
            let (inputs, ids) = feed.inputs(k).map_err(|error| refused(error.into()))?;
            let scalars = adamw.map(|adamw| adamw.scalars(lr, k + 1, end));
            words.clear();
            for (view, what) in &fed {
                match what {
                    Fed::Data { name, ids: is_ids } => {
                        push_data(&mut words, view, name, *is_ids, inputs, ids)?;
                    }
                    Fed::Hosted(Hosted::Token(site)) => {
                        let token = dropout_token(site, k);
                        check_count(view, token.len())?;
                        words.extend(token.data().iter().map(|&w| f32::from_bits(w as u32)));
                    }
                    Fed::Hosted(Hosted::Projection(site, ty)) => {
                        let projection = sigreg_projection(config.seed, site, k, ty)
                            .map_err(|error| refused(error.into()))?;
                        check_count(view, projection.len())?;
                        words.extend_from_slice(projection.data());
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
            upload.write_f32(&words)?;
        }
        // An evaluated step: before the step's command buffer, each held-out batch in one of its
        // own, read back and tallied before the next batch overwrites the download.
        let metrics = match held_out {
            Some(held_out) if evaluated => {
                let download = prediction_download.as_mut().ok_or_else(|| {
                    VulkanRunError::Plan("a held-out evaluation without its download".into())
                })?;
                let mut tally = held_out.tallying();
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
                        if let Some(upload) = &step_upload {
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
                    tally.add(&download.read_f32()?).map_err(refused)?;
                }
                Some(tally.metrics().map_err(refused)?)
            }
            _ => None,
        };
        {
            let mut recording = device.record()?;
            open(
                &mut recording,
                &mut opened,
                first_upload.as_ref(),
                &first_views,
                &buffers,
            )?;
            let barriers_before = recorded_barriers();
            if let Some(upload) = &step_upload {
                let mut at = 0;
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
            if last {
                record(&mut recording, loss_actions.iter().copied(), &buffers, None)?;
            } else {
                record(&mut recording, &step_actions, &buffers, None)?;
                dispatches_per_step = recording.dispatches() - before;
            }
            recording.barrier(Barrier::ComputeToTransfer);
            recording.copy_region(loss, plan.loss_view.offset, &loss_download, 0, 4)?;
            if !last {
                barriers_per_step = recorded_barriers() - barriers_before;
            }
            if reads_state {
                copy_state(&mut recording, &state_views, &buffers, &state_download)?;
            }
            recording.barrier(Barrier::TransferToHost);
            let recorded = recording.finish()?;
            first_submit.get_or_insert_with(Instant::now);
            device.submit(recorded)?.wait()?;
        }
        let loss_k = loss_download.read_f32()?[0];
        let state = if reads_state {
            Some(read_state(&state_download.read_f32()?)?)
        } else {
            None
        };
        let values = if k == start {
            Some(&initial)
        } else if last {
            state.as_ref().map(|(values, _)| values)
        } else {
            None
        };
        observe(&StepSeen {
            step: k,
            loss: loss_k,
            held_out: metrics,
            values,
        })
        .map_err(VulkanRunError::Run)?;
        losses.push(loss_k);
        let unread = BTreeMap::new();
        let diagnostic = match &state {
            Some((values, read_moments)) => non_finite(compiled, k, loss_k, values, read_moments),
            None => non_finite(compiled, k, loss_k, &unread, &BTreeMap::new()),
        };
        if let Some(diagnostic) = diagnostic {
            let (values, read_moments) = match state {
                Some(state) => state,
                None => {
                    // One more host wait: the parameters and moments the diverged run holds.
                    let mut recording = device.record()?;
                    recording.barrier(Barrier::ComputeToTransfer);
                    copy_state(&mut recording, &state_views, &buffers, &state_download)?;
                    recording.barrier(Barrier::TransferToHost);
                    let recorded = recording.finish()?;
                    device.submit(recorded)?.wait()?;
                    read_state(&state_download.read_f32()?)?
                }
            };
            let version = if last { k } else { k + 1 };
            let mut state = state_of(compiled, values, version);
            state.moments.extend(read_moments);
            let run = Run {
                start_step: start,
                initial: initial.clone(),
                losses,
                state,
                teacher: feed.teacher(),
            };
            let ran = k - start;
            return Err(VulkanRunError::NonFinite(Box::new(DivergedOnDevice {
                diverged: Diverged { diagnostic, run },
                stats: count(ran, dispatches_per_step, barriers_per_step, first_submit),
                profile: None,
            })));
        }
        if let Some((values, read_moments)) = state {
            if due && let Some(write) = write.as_mut() {
                write(k + 1, &values).map_err(|error| refused(error.into()))?;
            }
            if last {
                finished = Some((values, read_moments));
            }
        }
    }
    let stats = count(
        end - start,
        dispatches_per_step,
        barriers_per_step,
        first_submit,
    );
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
    ))
}

/// The bytes of the host-visible buffers a stepwise run reads back through before its host
/// waits, apart from the parameters and moments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DownloadBytes {
    /// The loss of a step, one `f32`.
    pub loss: u64,
    /// The held-out logits, 0 without held-out evaluation.
    pub prediction: u64,
}

impl DownloadBytes {
    /// Both buffers together.
    pub fn total(&self) -> u64 {
        self.loss + self.prediction
    }
}

/// The download buffers of a stepwise run whose plan predicts into `prediction` and which
/// evaluates `eval_batches` held-out batches: the loss, and one batch's prediction whatever the
/// number of batches, since each batch is evaluated in a command buffer of its own and read
/// back before the next (story:vulkan-eval-download-per-batch).
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

/// Starts a command buffer of a stepwise run. The first of the run copies the parameters'
/// initial values and the zero moments up from `first`, the views `views` one after the other;
/// every later one starts after the one before it, whose dispatches wrote what this one's read
/// and read the views its copies overwrite.
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
}
