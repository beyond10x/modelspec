//! A whole training run on one Vulkan device (DESIGN.md § 6): the artifact's step plan, recorded
//! `steps` times into chained command buffers, with one host wait at the end and one at each
//! checkpoint the run is asked for.
//!
//! [`train`] draws the initial parameters and the synthetic batch from the seed as the CPU run
//! does ([`crate::train::runner::run`]), makes one device-local buffer per plan buffer (the
//! plan's `Allocate` steps run here, once), copies the parameters and the batch up once, and
//! binds every dispatch of the step plan once. Step `k` records the step plan, then copies the
//! loss view into slot `k` of a device-local loss buffer; after the last step the steps of the
//! step plan up to the loss evaluate the loss of the final parameters into slot `steps`. Up to
//! [`CHUNK_STEPS`] steps go into one command buffer; the command buffers are submitted one after
//! the other without waiting, the last behind a fence, and the host waits once, then reads the
//! parameters and the `steps + 1` losses back. The result is a [`Run`] the CPU run can be
//! compared with, and the [`RunStats`] of the device.
//!
//! [`train_with`] starts the run at a checkpoint's step and parameters ([`RunOptions::start`],
//! [`RunOptions::init`]) and writes checkpoints on the way ([`Checkpoints`], DESIGN.md § 6.2): the
//! step a checkpoint is due after ends a command buffer, which also copies the parameters down;
//! the host waits for it, reads them and calls [`Checkpoints::write`], and recording goes on in a
//! new command buffer. Each checkpoint therefore costs one host wait and one parameter read-back,
//! and no other step does.
//!
//! A training run checks what it reads back as the CPU run checks each step (E-TRAIN-001,
//! [`crate::train::runner::E_TRAIN_001`]), without a read-back or host wait of its own: after the
//! last wait every loss and the final parameters, and at each checkpoint the losses since the
//! last stop, copied into the checkpoint's read-back, and its parameters, before the checkpoint is
//! written. The first non-finite value stops the run with [`VulkanRunError::NonFinite`] and the
//! CPU run's diagnostic, the losses kept through the failing step's, and the [`RunStats`] and
//! [`Profile`] of the steps the device ran ([`DivergedOnDevice`]). A parameter that goes non-finite
//! between two read-backs while its step's loss is finite is named at the first later loss it
//! makes non-finite, or at the read-back, not at its own step ([`diverged`]). [`infer`] does not
//! check.
//!
//! [`infer`] runs the steps of the step plan up to the loss once on a checkpoint's parameters and
//! the synthetic batch, with one host wait, and reads the prediction and the loss back.
//!
//! [`train_profiled`] runs the same recording with a timestamp before and after each dispatch,
//! written into one query pool per command buffer and read after the one wait, and returns a
//! [`Profile`] of every dispatch site (DESIGN.md § 8, `--profile`).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Range;
use std::time::{Duration, Instant};

use modelspec_core::artifact::CompiledArtifact;
use modelspec_core::compile::CompiledTraining;
use modelspec_core::device::DeviceProfile;
use modelspec_core::diag::Diagnostic;
use modelspec_core::ir::kernel::{BufferView, ExecutionPlan, KernelKind, PlanOp, PlanStep};
use modelspec_core::ir::tensor::{Dim, ParamId};
use thiserror::Error;

use super::dispatch::{Bound, DescriptorArena, Recording, TimestampPool};
use super::memory::Buffer;
use super::negotiate::{Negotiated, NumericsClass, negotiate};
use super::pipeline::Pipeline;
use super::profile::{PlanName, Profile, Site, timings};
use super::select::select_device;
use super::sync::Barrier;
use super::{Device, ExecError, Vulkan};
use crate::cpu::HostTensor;
use crate::spirv::emit::{EmitError, emit_kernels};
use crate::train::checkpoint::CheckpointError;
use crate::train::data::synthetic;
use crate::train::init::{Stream, initialize};
use crate::train::optimizer::ParamState;
use crate::train::runner::{Diverged, Run, RunError, non_finite};

/// The most training steps one command buffer records.
pub const CHUNK_STEPS: u64 = 256;

/// What a device run did, counted on the device while it ran.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunStats {
    /// The training steps run.
    pub steps: u64,
    /// The times the host waited for the device.
    pub host_waits: u64,
    /// The command buffers submitted.
    pub submits: u64,
    /// The dispatches recorded in all of them.
    pub dispatches: u64,
    /// The dispatches one training step records.
    pub dispatches_per_step: u64,
    /// The bytes copied from staging buffers to device-local ones.
    pub bytes_uploaded: u64,
    /// The bytes copied from device-local buffers to staging ones.
    pub bytes_downloaded: u64,
    /// The comparison with the CPU run the results get (DESIGN.md § 5.4).
    pub class: NumericsClass,
    /// From the first submit to the end of the last wait, the checkpoints written on the way
    /// included.
    pub elapsed: Duration,
}

impl RunStats {
    /// The steady-state time of one step in microseconds: [`RunStats::elapsed`] over the steps.
    pub fn us_per_step(&self) -> f64 {
        if self.steps == 0 {
            return 0.0;
        }
        self.elapsed.as_secs_f64() * 1e6 / self.steps as f64
    }
}

impl fmt::Display for RunStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "steps: {}", self.steps)?;
        writeln!(f, "host waits: {}", self.host_waits)?;
        writeln!(f, "submits: {}", self.submits)?;
        writeln!(f, "dispatches: {}", self.dispatches)?;
        writeln!(f, "dispatches per step: {}", self.dispatches_per_step)?;
        writeln!(f, "bytes uploaded: {}", self.bytes_uploaded)?;
        writeln!(f, "bytes downloaded: {}", self.bytes_downloaded)?;
        writeln!(f, "numerics class: {}", self.class)?;
        writeln!(f, "us per step: {:.3}", self.us_per_step())
    }
}

/// Why a device run stopped.
#[derive(Debug, Error)]
pub enum VulkanRunError {
    /// The initial parameters or the synthetic data could not be drawn, as the CPU run reports,
    /// the run would end past step `u64::MAX`, or a checkpoint could not be written.
    #[error(transparent)]
    Run(Box<RunError>),
    /// A loss or a parameter the device read back is not finite (E-TRAIN-001).
    #[error("{}", .0.diverged.diagnostic)]
    NonFinite(Box<DivergedOnDevice>),
    /// A kernel of the plan has no SPIR-V module.
    #[error(transparent)]
    Emit(#[from] EmitError),
    /// The device refused a buffer, pipeline, recording or submission.
    #[error(transparent)]
    Exec(#[from] ExecError),
    /// The plan is not one this runner executes.
    #[error("the execution plan cannot run on the device: {0}")]
    Plan(String),
}

/// A device run that E-TRAIN-001 stopped: the CPU run's diagnostic and the run through the
/// failing step, with the counters of what the device ran up to the read-back that showed it and,
/// when profiled, the profile of those steps, as a finished run returns them (DESIGN.md § 8).
#[derive(Debug)]
pub struct DivergedOnDevice {
    /// The diagnostic and the run through the failing step.
    pub diverged: Diverged,
    /// What the device ran: the steps through the stop whose read-back showed the value.
    pub stats: RunStats,
    /// The profile of those steps; without a final loss evaluation when a checkpoint stop
    /// showed the value.
    pub profile: Option<Profile>,
}

/// Why `modelspec train --backend vulkan` or `infer --backend vulkan` did not run.
#[derive(Debug, Error)]
pub enum DeviceRunError {
    /// No loader, no device, or a device below the requirements: E-TARGET-002 to E-TARGET-004.
    #[error("{0}")]
    Refused(Diagnostic),
    /// The run started and failed.
    #[error(transparent)]
    Failed(#[from] VulkanRunError),
}

/// Writes the checkpoint after `k` updates, given each parameter's value by id.
pub type WriteCheckpoint<'c> =
    dyn FnMut(u64, &BTreeMap<ParamId, HostTensor<f32>>) -> Result<(), CheckpointError> + 'c;

/// The checkpoints a device run writes on the way: one after every step `k` after the start and
/// before the end that is a multiple of `every`, as `--checkpoint-every` asks.
pub struct Checkpoints<'c> {
    /// The steps between two checkpoints; 0 asks for none.
    pub every: u64,
    /// Called with `k` and the parameters after `k` updates; an error stops the run.
    pub write: &'c mut WriteCheckpoint<'c>,
}

/// Where a device run starts and what it writes on the way; the default is a run from step 0
/// and the initializers that writes nothing.
#[derive(Default)]
pub struct RunOptions<'c> {
    /// The step the run starts at: 0, or the step of the checkpoint it resumes from. The run
    /// ends at `start + steps`.
    pub start: u64,
    /// The values that replace the initializer draws of their parameters: a checkpoint's.
    pub init: BTreeMap<ParamId, HostTensor<f32>>,
    /// The checkpoints to write on the way.
    pub checkpoints: Option<Checkpoints<'c>>,
}

/// The forward program on the device ([`infer`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Inference {
    /// The model output.
    pub prediction: HostTensor<f32>,
    /// The loss of the synthetic batch.
    pub loss: f32,
}

/// Loads Vulkan, selects device `device` (0 when `None`), negotiates and opens it, and runs
/// [`train`] on it.
pub fn train_on_device(
    device: Option<usize>,
    compiled: &CompiledTraining,
    artifact: &CompiledArtifact,
) -> Result<(Run, RunStats), DeviceRunError> {
    let (run, stats, _) = train_on_device_with(device, compiled, artifact, false)?;
    Ok((run, stats))
}

/// [`train_on_device`], running [`train_profiled`] instead of [`train`] when `profiled`.
pub fn train_on_device_with(
    device: Option<usize>,
    compiled: &CompiledTraining,
    artifact: &CompiledArtifact,
    profiled: bool,
) -> Result<(Run, RunStats, Option<Profile>), DeviceRunError> {
    train_on_device_from(device, compiled, artifact, profiled, RunOptions::default())
}

/// [`train_on_device_with`], from the start and with the checkpoints of `options`
/// ([`train_with`]).
pub fn train_on_device_from(
    device: Option<usize>,
    compiled: &CompiledTraining,
    artifact: &CompiledArtifact,
    profiled: bool,
    options: RunOptions<'_>,
) -> Result<(Run, RunStats, Option<Profile>), DeviceRunError> {
    on_device(device, |opened, profile, negotiated| {
        let job = Job {
            steps: compiled.config.steps,
            profiled,
            predict: false,
        };
        let done = execute(
            opened, profile, negotiated, compiled, artifact, job, options,
        )?;
        Ok((done.run, done.stats, done.profile))
    })
}

/// Loads Vulkan, selects device `device` (0 when `None`), negotiates and opens it, and runs
/// [`infer`] on it with `params`.
pub fn infer_on_device(
    device: Option<usize>,
    compiled: &CompiledTraining,
    artifact: &CompiledArtifact,
    params: &BTreeMap<ParamId, HostTensor<f32>>,
) -> Result<(Inference, RunStats), DeviceRunError> {
    on_device(device, |opened, profile, negotiated| {
        Ok(infer(
            opened, profile, negotiated, compiled, artifact, params,
        )?)
    })
}

/// Loads Vulkan, selects, negotiates and opens device `device` (0 when `None`) and calls `body`
/// with it.
fn on_device<T>(
    device: Option<usize>,
    body: impl FnOnce(&Device<'_>, &DeviceProfile, &Negotiated) -> Result<T, DeviceRunError>,
) -> Result<T, DeviceRunError> {
    let vulkan = Vulkan::load().map_err(DeviceRunError::Refused)?;
    let candidates = vulkan.candidates().map_err(DeviceRunError::Refused)?;
    let index =
        select_device(&candidates, Some(device.unwrap_or(0))).map_err(DeviceRunError::Refused)?[0];
    let profile = &candidates[index].profile;
    let negotiated = negotiate(profile).map_err(DeviceRunError::Refused)?;
    let opened = vulkan
        .create_device(index, &negotiated)
        .map_err(DeviceRunError::Refused)?;
    body(&opened, profile, &negotiated)
}

/// One recorded command of the step plan.
enum Action<'a> {
    /// A dispatch and the index of its step in the step plan.
    Dispatch(Bound<'a>, usize),
    Barrier,
    Copy {
        from: BufferView,
        to: BufferView,
    },
}

/// Trains `compiled` on `device`, whose profile is `profile` and which was opened with
/// `negotiated`, by the execution plan of `artifact`, compiled from the same source for Vulkan.
pub fn train(
    device: &Device<'_>,
    profile: &DeviceProfile,
    negotiated: &Negotiated,
    compiled: &CompiledTraining,
    artifact: &CompiledArtifact,
) -> Result<(Run, RunStats), VulkanRunError> {
    train_with(
        device,
        profile,
        negotiated,
        compiled,
        artifact,
        RunOptions::default(),
    )
}

/// [`train`] from `options.start` with the parameters of `options.init` in place of their
/// draws, writing the checkpoints of `options.checkpoints` on the way: one host wait and one
/// parameter read-back each. The losses are those of steps `start..=start + steps`, and each
/// updated parameter ends at version `start + steps`, as a resumed CPU run's
/// ([`crate::train::runner::run_from`]).
pub fn train_with(
    device: &Device<'_>,
    profile: &DeviceProfile,
    negotiated: &Negotiated,
    compiled: &CompiledTraining,
    artifact: &CompiledArtifact,
    options: RunOptions<'_>,
) -> Result<(Run, RunStats), VulkanRunError> {
    let job = Job {
        steps: compiled.config.steps,
        profiled: false,
        predict: false,
    };
    let done = execute(
        device, profile, negotiated, compiled, artifact, job, options,
    )?;
    Ok((done.run, done.stats))
}

/// The forward program of `compiled` on `device` with the parameter values `params` (a
/// checkpoint's, [`crate::train::checkpoint::Checkpoint::parameters_for`]) and the synthetic batch
/// of the file's `train` block: the steps of the step plan up to the loss, recorded once, with one
/// host wait, and the prediction and the loss read back. A parameter `params` does not name keeps
/// its initializer draw. The [`RunStats`] count 0 steps.
pub fn infer(
    device: &Device<'_>,
    profile: &DeviceProfile,
    negotiated: &Negotiated,
    compiled: &CompiledTraining,
    artifact: &CompiledArtifact,
    params: &BTreeMap<ParamId, HostTensor<f32>>,
) -> Result<(Inference, RunStats), VulkanRunError> {
    let job = Job {
        steps: 0,
        profiled: false,
        predict: true,
    };
    let options = RunOptions {
        init: params.clone(),
        ..RunOptions::default()
    };
    let done = execute(
        device, profile, negotiated, compiled, artifact, job, options,
    )?;
    let prediction = done
        .prediction
        .expect("a predicting run returns its prediction");
    let loss = done.run.losses[0];
    Ok((Inference { prediction, loss }, done.stats))
}

/// [`train`], with a timestamp before and after every dispatch it records when the device has
/// timestamp queries: the timestamps go into one query pool per command buffer and are read
/// back after the run's one host wait, without a wait of their own, and summed per dispatch
/// site ([`profile::timings`]). The run's commands, results and [`RunStats`] are those of
/// [`train`]; the [`Profile`] adds the time to emit the SPIR-V and create the pipelines.
pub fn train_profiled(
    device: &Device<'_>,
    profile: &DeviceProfile,
    negotiated: &Negotiated,
    compiled: &CompiledTraining,
    artifact: &CompiledArtifact,
) -> Result<(Run, RunStats, Profile), VulkanRunError> {
    let job = Job {
        steps: compiled.config.steps,
        profiled: true,
        predict: false,
    };
    let done = execute(
        device,
        profile,
        negotiated,
        compiled,
        artifact,
        job,
        RunOptions::default(),
    )?;
    let profiled = done.profile.expect("a profiled run returns its profile");
    Ok((done.run, done.stats, profiled))
}

/// What [`execute`] runs: `steps` steps, with timestamps when `profiled`, and the prediction
/// read back after the final loss when `predict`.
#[derive(Clone, Copy)]
struct Job {
    steps: u64,
    profiled: bool,
    predict: bool,
}

/// What [`execute`] returns.
struct Done {
    run: Run,
    stats: RunStats,
    profile: Option<Profile>,
    prediction: Option<HostTensor<f32>>,
}

/// The timestamps of one command buffer: its pool and the next query to write.
struct Stamps<'p> {
    pool: &'p TimestampPool<'p>,
    next: u32,
}

/// [`train`] for `job` from the start and with the checkpoints of `options`.
fn execute(
    device: &Device<'_>,
    profile: &DeviceProfile,
    negotiated: &Negotiated,
    compiled: &CompiledTraining,
    artifact: &CompiledArtifact,
    job: Job,
    options: RunOptions<'_>,
) -> Result<Done, VulkanRunError> {
    let plan = &artifact.execution;
    let config = &compiled.config;
    let Job {
        steps,
        profiled,
        predict,
    } = job;
    let forward = &compiled.forward;
    let refused = |error: RunError| VulkanRunError::Run(Box::new(error));
    let RunOptions {
        start,
        init,
        checkpoints,
    } = options;
    let end = start
        .checked_add(steps)
        .ok_or_else(|| refused(RunError::StepCount { start, steps }))?;
    let mut initial = initialize(&forward.program, config.seed, Stream::Init)
        .map_err(|error| refused(error.into()))?;
    initial.extend(init);
    let data = synthetic(config, forward).map_err(|error| refused(error.into()))?;
    let (every, mut write) = match checkpoints {
        Some(Checkpoints { every, write }) if every > 0 => (every, Some(write)),
        _ => (0, None),
    };
    // The steps after which the host waits, counted from the start: each due checkpoint, then
    // the end.
    let mut stops: Vec<u64> = if write.is_some() {
        (1..steps)
            .filter(|k| (start + k).is_multiple_of(every))
            .collect()
    } else {
        Vec::new()
    };
    stops.push(steps);

    let emitting = Instant::now();
    let modules = emit_kernels(artifact, profile)?;
    let emitted = emitting.elapsed();
    let creating = Instant::now();
    let mut pipelines = Vec::with_capacity(plan.kernels.len());
    for (kernel, module) in plan.kernels.iter().zip(&modules) {
        pipelines.push(device.kernel_pipeline(kernel, &module.spirv)?);
    }
    let pipeline_creation = creating.elapsed();

    // The plan's `Allocate` steps: the scratch buffers, made once for the whole run.
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

    // What goes up: every parameter's initial value, then every data declaration's batch.
    let mut uploads: Vec<(BufferView, &[f32])> = Vec::new();
    for parameter in &plan.parameters {
        let value = initial.get(&parameter.param).ok_or_else(|| {
            VulkanRunError::Plan(format!("parameter {} has no initial value", parameter.path))
        })?;
        uploads.push((parameter.views[0], value.data()));
    }
    for declaration in &plan.data {
        let value = data.inputs.get(&declaration.name).ok_or_else(|| {
            VulkanRunError::Plan(format!("data {} has no synthetic batch", declaration.name))
        })?;
        uploads.push((declaration.view, value.data()));
    }
    for (view, values) in &uploads {
        if values.len() as u64 != view.element_count {
            return Err(VulkanRunError::Plan(format!(
                "{view} holds {} elements, the host value {}",
                view.element_count,
                values.len()
            )));
        }
    }
    let upload_bytes: u64 = uploads.iter().map(|(view, _)| view.size_bytes).sum();
    let mut upload = device.staging_buffer(upload_bytes)?;
    upload.write_f32(
        &uploads
            .iter()
            .flat_map(|(_, v)| v.iter().copied())
            .collect::<Vec<_>>(),
    )?;

    let loss_slots = steps
        .checked_add(1)
        .ok_or_else(|| VulkanRunError::Plan(format!("{steps} steps have no final loss")))?;
    let losses = device.device_local_buffer(4 * loss_slots)?;
    let param_bytes: u64 = plan.parameters.iter().map(|p| p.views[0].size_bytes).sum();
    let prediction_bytes = if predict {
        plan.prediction_view.size_bytes
    } else {
        0
    };
    let mut download = device.staging_buffer(param_bytes + 4 * loss_slots + prediction_bytes)?;

    let dispatch_steps = plan
        .step
        .iter()
        .filter(|step| step.op == PlanOp::Dispatch)
        .count();
    let arena =
        device.descriptor_arena(count_u32(dispatch_steps)?, count_u32(descriptors(plan))?)?;
    let (actions, loss_end) = actions(plan, &arena, &pipelines, &buffers)?;

    // The dispatch sites a profile times, and one timestamp pool per chunk of `CHUNK_STEPS`
    // steps: two queries per dispatch it records.
    let site = |plan_name: PlanName, action: &Action<'_>| match action {
        Action::Dispatch(_, at) => {
            let id = plan.step[*at].kernel_module_id?;
            let entry = plan
                .kernels
                .iter()
                .zip(&modules)
                .find(|(kernel, _)| kernel.id == id)
                .map(|(_, module)| module.entry_point.clone())?;
            Some(Site {
                plan: plan_name,
                position: *at,
                kernel_module_id: id,
                entry,
            })
        }
        _ => None,
    };
    let step_sites: Vec<Site> = actions
        .iter()
        .filter_map(|action| site(PlanName::Step, action))
        .collect();
    let final_sites: Vec<Site> = actions[..loss_end]
        .iter()
        .filter_map(|action| site(PlanName::Final, action))
        .collect();
    let clock = device
        .clock()
        .filter(|_| profiled && profile.timestamp_queries);
    let mut pools = Vec::new();
    if clock.is_some() {
        let per_step = 2 * step_sites.len() as u64;
        let chunks = steps.div_ceil(CHUNK_STEPS).max(1);
        for chunk in 0..chunks {
            let chunk_steps = (steps - chunk * CHUNK_STEPS).min(CHUNK_STEPS);
            let mut queries = per_step * chunk_steps;
            if chunk + 1 == chunks {
                queries += 2 * final_sites.len() as u64;
            }
            let queries = u32::try_from(queries).map_err(|_| {
                VulkanRunError::Plan(format!("{queries} timestamp queries do not fit u32"))
            })?;
            pools.push(device.timestamp_pool(queries)?);
        }
    }
    let stamps_of = |chunk: usize, next: u32| pools.get(chunk).map(|pool| Stamps { pool, next });

    let class = numerics_class(negotiated, plan);
    let waits_before = device.host_waits();
    let submits_before = device.submits();
    let dispatches_before = device.dispatches();
    let uploaded_before = device.bytes_uploaded();
    let downloaded_before = device.bytes_downloaded();
    let mut dispatches_per_step = 0;
    let mut first_submit: Option<Instant> = None;
    let loss = buffer(&buffers, &plan.loss_view)?;
    // The counters of the first `ran` steps, from the first submit to the last wait.
    let count = |ran: u64, dispatches_per_step: u64, first_submit: Option<Instant>| RunStats {
        steps: ran,
        host_waits: device.host_waits() - waits_before,
        submits: device.submits() - submits_before,
        dispatches: device.dispatches() - dispatches_before,
        dispatches_per_step,
        bytes_uploaded: device.bytes_uploaded() - uploaded_before,
        bytes_downloaded: device.bytes_downloaded() - downloaded_before,
        class,
        elapsed: first_submit
            .expect("a run submits at least one command buffer")
            .elapsed(),
    };
    // The profile of the first `ran` steps, when `profiled`. `stopped` is `None` for the whole
    // run, every pool written and the final loss evaluation last; for a run stopped at a
    // checkpoint it is the chunk the last step recorded into and the queries written in its pool,
    // the pools after it unwritten and no final loss evaluation recorded.
    let profile_of = |ran: u64, stopped: Option<(usize, u32)>| {
        if !profiled {
            return Ok(None);
        }
        let timings = match clock {
            Some(clock) => {
                let mut stamps = Vec::new();
                let full = stopped.map_or(pools.len(), |(chunk, _)| chunk);
                for pool in &pools[..full] {
                    stamps.extend(pool.read()?);
                }
                if let Some((chunk, written)) = stopped
                    && let Some(pool) = pools.get(chunk)
                {
                    stamps.extend(pool.read_first(written)?);
                }
                let finals: &[Site] = if stopped.is_none() { &final_sites } else { &[] };
                Some(
                    timings(&step_sites, finals, ran, &stamps, clock)
                        .map_err(VulkanRunError::Plan)?,
                )
            }
            None => None,
        };
        Ok::<_, VulkanRunError>(Some(Profile {
            timings,
            pipeline_creation,
            compile: emitted,
        }))
    };
    // The chunk the next step records into and its next timestamp query, across the command
    // buffers a checkpoint ends.
    let mut chunk = 0;
    let mut next_query = 0;
    let mut from = 0;
    // The first step of the last stop's window, which the check after the last wait reads.
    let mut last_from = 0;
    let param_floats = (param_bytes / 4) as usize;
    for &stop in &stops {
        let last = stop == steps;
        {
            let mut chain = device.chain();
            let mut recording = device.record()?;
            let mut stamps = stamps_of(chunk, next_query);
            if from == 0 {
                if let Some(stamps) = &stamps {
                    recording.reset_timestamps(stamps.pool);
                }
                let mut at = 0;
                for (view, _) in &uploads {
                    recording.copy_region(
                        &upload,
                        at,
                        buffer(&buffers, view)?,
                        view.offset,
                        view.size_bytes,
                    )?;
                    at += view.size_bytes;
                }
                recording.barrier(Barrier::TransferToCompute);
            }
            for k in from..stop {
                if k > 0 && k % CHUNK_STEPS == 0 {
                    if k != from {
                        let recorded = recording.finish()?;
                        first_submit.get_or_insert_with(Instant::now);
                        chain.submit(recorded)?;
                        recording = device.record()?;
                    }
                    chunk += 1;
                    stamps = stamps_of(chunk, 0);
                    if let Some(stamps) = &stamps {
                        recording.reset_timestamps(stamps.pool);
                    }
                }
                let before = recording.dispatches();
                record(&mut recording, &actions, &buffers, stamps.as_mut())?;
                dispatches_per_step = recording.dispatches() - before;
                recording.barrier(Barrier::ComputeToTransfer);
                recording.copy_region(loss, plan.loss_view.offset, &losses, 4 * k, 4)?;
                recording.barrier(Barrier::TransferToCompute);
            }
            if last {
                record(
                    &mut recording,
                    &actions[..loss_end],
                    &buffers,
                    stamps.as_mut(),
                )?;
                recording.barrier(Barrier::ComputeToTransfer);
                recording.copy_region(loss, plan.loss_view.offset, &losses, 4 * steps, 4)?;
            }
            let mut at = 0;
            for parameter in &plan.parameters {
                let view = parameter.views[0];
                recording.copy_region(
                    buffer(&buffers, &view)?,
                    view.offset,
                    &download,
                    at,
                    view.size_bytes,
                )?;
                at += view.size_bytes;
            }
            if last {
                if predict {
                    let view = plan.prediction_view;
                    recording.copy_region(
                        buffer(&buffers, &view)?,
                        view.offset,
                        &download,
                        at + 4 * loss_slots,
                        view.size_bytes,
                    )?;
                }
                recording.barrier(Barrier::TransferToTransfer);
                recording.copy_region(&losses, 0, &download, at, 4 * loss_slots)?;
            } else if stop > from {
                // A checkpoint's read-back also carries the losses of the steps since the last
                // stop, so the check before the checkpoint is written sees them.
                recording.barrier(Barrier::TransferToTransfer);
                recording.copy_region(
                    &losses,
                    4 * from,
                    &download,
                    at + 4 * from,
                    4 * (stop - from),
                )?;
            }
            recording.barrier(Barrier::TransferToHost);
            let recorded = recording.finish()?;
            first_submit.get_or_insert_with(Instant::now);
            chain.finish(recorded)?.wait()?;
            next_query = stamps.map_or(0, |stamps| stamps.next);
        }
        if !last && let Some(write) = write.as_mut() {
            let read = download.read_f32()?;
            let values = parameters(plan, &initial, &read)?;
            let read_losses = &read[param_floats..param_floats + stop as usize];
            if let Some((kept, diagnostic)) =
                diverged(compiled, start, from..stop, false, read_losses, &values)
            {
                let run = Run {
                    start_step: start,
                    initial,
                    losses: read_losses[..kept as usize].to_vec(),
                    state: state_of(compiled, values, start + stop),
                    teacher: data.teacher,
                };
                return Err(VulkanRunError::NonFinite(Box::new(DivergedOnDevice {
                    diverged: Diverged { diagnostic, run },
                    stats: count(stop, dispatches_per_step, first_submit),
                    profile: profile_of(stop, Some((chunk, next_query)))?,
                })));
            }
            write(start + stop, &values).map_err(|error| refused(error.into()))?;
        }
        last_from = from;
        from = stop;
    }
    let stats = count(steps, dispatches_per_step, first_submit);
    let profiled = profile_of(steps, None)?;

    let read = download.read_f32()?;
    let values = parameters(plan, &initial, &read)?;
    let at = param_floats;
    let losses = read[at..at + loss_slots as usize].to_vec();
    if !predict
        && let Some((kept, diagnostic)) =
            diverged(compiled, start, last_from..steps, true, &losses, &values)
    {
        let run = Run {
            start_step: start,
            initial,
            losses: losses[..kept as usize].to_vec(),
            state: state_of(compiled, values, end),
            teacher: data.teacher,
        };
        return Err(VulkanRunError::NonFinite(Box::new(DivergedOnDevice {
            diverged: Diverged { diagnostic, run },
            stats,
            profile: profiled,
        })));
    }
    let prediction = if predict {
        let at = at + loss_slots as usize;
        let count = plan.prediction_view.element_count as usize;
        let node = forward.program.node(forward.prediction).ok_or_else(|| {
            VulkanRunError::Plan("the forward program defines no prediction value".into())
        })?;
        let shape = node
            .ty
            .shape
            .iter()
            .map(|dim| match dim {
                Dim::Static(n) => Ok(*n),
                Dim::Symbol(_) => Err(VulkanRunError::Plan(format!(
                    "the prediction {} is not static",
                    node.ty
                ))),
            })
            .collect::<Result<Vec<u64>, _>>()?;
        Some(
            HostTensor::new(shape, read[at..at + count].to_vec())
                .map_err(|error| VulkanRunError::Plan(format!("the prediction: {error}")))?,
        )
    } else {
        None
    };
    let run = Run {
        start_step: start,
        initial,
        losses,
        state: state_of(compiled, values, end),
        teacher: data.teacher,
    };
    Ok(Done {
        run,
        stats,
        profile: profiled,
        prediction,
    })
}

/// The state of `values` after `version` updates of each parameter the optimizer updates.
fn state_of(
    compiled: &CompiledTraining,
    values: BTreeMap<ParamId, HostTensor<f32>>,
    version: u64,
) -> ParamState {
    let mut state = ParamState::new(&compiled.training, values);
    for param in compiled.training.optimizer.updates.keys() {
        if let Some(at) = state.versions.get_mut(param) {
            at.0 = at.0.saturating_add(version);
        }
    }
    state
}

/// What the CPU run refuses with E-TRAIN-001 among the values one read-back of the device holds
/// ([`crate::train::runner::run_from`]), with the CPU run's diagnostic, or `None`. `losses[i]` is
/// the loss after `i` updates from `start`, and is read for every `i` in `window`, the steps since
/// the last read-back; `values` are the parameters after `window.end` updates, the only ones the
/// device reads back. The check runs in the CPU run's order as far as the read-back shows it:
/// each loss of the window, then the parameters as the update of the window's last step, then,
/// when `last`, the loss of the final parameters `losses[window.end]`. A parameter that went
/// non-finite at an update before the window's last is not seen there; with plain SGD it stays
/// non-finite (`p - lr * g` of a NaN or ±inf `p` is NaN or ±inf), so the first loss it makes
/// non-finite, or the read-back of the parameters, names it at a later step than the CPU run.
/// Returns how many losses the refused run keeps, through the failing step's.
fn diverged(
    compiled: &CompiledTraining,
    start: u64,
    window: Range<u64>,
    last: bool,
    losses: &[f32],
    values: &BTreeMap<ParamId, HostTensor<f32>>,
) -> Option<(u64, Diagnostic)> {
    let unread = BTreeMap::new();
    for i in window.clone() {
        let loss = losses[i as usize];
        if !loss.is_finite() {
            return non_finite(compiled, start + i, loss, &unread).map(|found| (i + 1, found));
        }
    }
    let checks = window.end.checked_sub(1).filter(|_| !window.is_empty());
    let checks = checks.into_iter().chain(last.then_some(window.end));
    for i in checks {
        if let Some(found) = non_finite(compiled, start + i, losses[i as usize], values) {
            return Some((i + 1, found));
        }
    }
    None
}

/// Each parameter's value by id from the first floats of a read-back, in plan order, shaped as
/// its initial value.
fn parameters(
    plan: &ExecutionPlan,
    initial: &BTreeMap<ParamId, HostTensor<f32>>,
    read: &[f32],
) -> Result<BTreeMap<ParamId, HostTensor<f32>>, VulkanRunError> {
    let mut values = BTreeMap::new();
    let mut at = 0usize;
    for parameter in &plan.parameters {
        let count = parameter.views[0].element_count as usize;
        let shape = initial[&parameter.param].shape().to_vec();
        let value = HostTensor::new(shape, read[at..at + count].to_vec())
            .map_err(|error| VulkanRunError::Plan(format!("{}: {error}", parameter.path)))?;
        values.insert(parameter.param, value);
        at += count;
    }
    Ok(values)
}

/// The plan buffer `view` lies in.
fn buffer<'b, 'd>(
    buffers: &'b [Buffer<'d>],
    view: &BufferView,
) -> Result<&'b Buffer<'d>, VulkanRunError> {
    buffers
        .get(view.buffer as usize)
        .ok_or_else(|| VulkanRunError::Plan(format!("{view} names no plan buffer")))
}

/// The storage-buffer descriptors the dispatches of the step plan bind, one set each: every view
/// a dispatch reads and every view it writes ([`actions`]).
fn descriptors(plan: &ExecutionPlan) -> usize {
    plan.step
        .iter()
        .filter(|step| step.op == PlanOp::Dispatch)
        .map(|step| step.reads.len() + step.writes.len())
        .sum()
}

fn count_u32(count: usize) -> Result<u32, VulkanRunError> {
    u32::try_from(count)
        .map_err(|_| VulkanRunError::Plan(format!("{count} descriptors do not fit u32")))
}

/// The recorded commands of the step plan, every dispatch bound once in `arena`, and how many
/// of them evaluate the loss: the steps up to the last that writes the loss view, none of
/// which writes a parameter.
fn actions<'a>(
    plan: &ExecutionPlan,
    arena: &'a DescriptorArena<'_>,
    pipelines: &'a [Pipeline<'_>],
    buffers: &'a [Buffer<'_>],
) -> Result<(Vec<Action<'a>>, usize), VulkanRunError> {
    let parameters: BTreeSet<u32> = plan
        .parameters
        .iter()
        .flat_map(|p| p.views.iter().map(|view| view.buffer))
        .collect();
    let mut actions = Vec::new();
    let mut loss_end = None;
    let mut writes_parameter = None;
    for (at, step) in plan.step.iter().enumerate() {
        match step.op {
            PlanOp::Allocate => continue,
            PlanOp::Barrier => actions.push(Action::Barrier),
            PlanOp::Copy => match (&step.reads[..], &step.writes[..]) {
                ([from], [to]) => actions.push(Action::Copy {
                    from: *from,
                    to: *to,
                }),
                _ => return Err(unexpected(at, step, "a copy reads one view and writes one")),
            },
            PlanOp::Dispatch => {
                let id = step
                    .kernel_module_id
                    .ok_or_else(|| unexpected(at, step, "a dispatch names no kernel"))?;
                let index = plan
                    .kernels
                    .iter()
                    .position(|kernel| kernel.id == id)
                    .ok_or_else(|| unexpected(at, step, "the plan has no such kernel"))?;
                if step.writes.len() != plan.kernels[index].written() {
                    return Err(unexpected(
                        at,
                        step,
                        "a dispatch writes the views its kernel writes",
                    ));
                }
                let mut bound = Vec::with_capacity(step.reads.len() + step.writes.len());
                for view in step.reads.iter().chain(&step.writes) {
                    if view.offset != 0 {
                        return Err(unexpected(at, step, "a dispatch binds a view at offset 0"));
                    }
                    bound.push(buffer(buffers, view)?);
                }
                actions.push(Action::Dispatch(
                    arena.bind(&pipelines[index], &plan.kernels[index], &bound)?,
                    at,
                ));
            }
            PlanOp::Complete => {
                return Err(unexpected(at, step, "a runner places completion points"));
            }
        }
        if writes_parameter.is_none()
            && step
                .writes
                .iter()
                .any(|view| parameters.contains(&view.buffer))
        {
            writes_parameter = Some(at);
        }
        if step
            .writes
            .iter()
            .any(|view| view.buffer == plan.loss_view.buffer)
        {
            loss_end = Some((at, actions.len()));
        }
    }
    let (loss_at, loss_end) =
        loss_end.ok_or_else(|| VulkanRunError::Plan("no step writes the loss".into()))?;
    if let Some(at) = writes_parameter.filter(|&at| at < loss_at) {
        return Err(unexpected(
            at,
            &plan.step[at],
            "a parameter is written before the loss",
        ));
    }
    Ok((actions, loss_end))
}

fn unexpected(at: usize, step: &PlanStep, reason: &str) -> VulkanRunError {
    VulkanRunError::Plan(format!("step {at} ({}): {reason}", step.op))
}

/// Records `actions`: a plan barrier as a compute-to-compute barrier, a plan copy between the
/// barriers that order it after the dispatches before it and before the ones after it. With
/// `stamps`, a timestamp before and after each dispatch into its next two queries.
fn record<'r>(
    recording: &mut Recording<'r>,
    actions: &'r [Action<'_>],
    buffers: &'r [Buffer<'_>],
    mut stamps: Option<&mut Stamps<'r>>,
) -> Result<(), VulkanRunError> {
    for action in actions {
        match action {
            Action::Dispatch(bound, _) => match stamps.as_deref_mut() {
                Some(stamps) => {
                    recording.timestamp(stamps.pool, stamps.next, false)?;
                    recording.dispatch_bound(bound);
                    recording.timestamp(stamps.pool, stamps.next + 1, true)?;
                    stamps.next += 2;
                }
                None => recording.dispatch_bound(bound),
            },
            Action::Barrier => recording.barrier(Barrier::ComputeToCompute),
            Action::Copy { from, to } => {
                recording.barrier(Barrier::ComputeToTransfer);
                recording.copy_region(
                    buffer(buffers, from)?,
                    from.offset,
                    buffer(buffers, to)?,
                    to.offset,
                    from.size_bytes.min(to.size_bytes),
                )?;
                recording.barrier(Barrier::TransferToCompute);
            }
        }
    }
    Ok(())
}

/// The class of a training run's results (DESIGN.md § 5.4): bit-for-bit only on a device
/// negotiated bit-for-bit whose every result is in the M1 bit-for-bit set, which holds the matmul
/// outputs and the `mse` loss sum before its division. A run reports its losses after the
/// division and updates its parameters by SGD, so every M1 training run is in the tolerance class.
fn numerics_class(negotiated: &Negotiated, plan: &ExecutionPlan) -> NumericsClass {
    let in_set = plan
        .kernels
        .iter()
        .all(|kernel| kernel.kind == KernelKind::MatMul);
    match negotiated.numerics {
        NumericsClass::BitForBit if in_set && !plan.kernels.is_empty() => NumericsClass::BitForBit,
        _ => NumericsClass::Tolerance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use modelspec_core::artifact::Backend;
    use modelspec_core::compile::{compile_artifact, compile_training};

    const EXAMPLE: &str = include_str!("../../../../examples/linear_regression.mspec");
    const W: ParamId = ParamId(0);

    fn compiled() -> CompiledTraining {
        compile_training("fit.mspec", EXAMPLE).expect("the example compiles")
    }

    fn w(values: [f32; 4]) -> BTreeMap<ParamId, HostTensor<f32>> {
        BTreeMap::from([(W, HostTensor::new(vec![4, 1], values.to_vec()).unwrap())])
    }

    const FINITE: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

    /// The step, the losses kept and the diagnostic's text of a refused read-back.
    fn refused(found: Option<(u64, Diagnostic)>) -> (u64, String) {
        let (kept, diagnostic) = found.expect("the read-back is refused");
        (kept, diagnostic.to_string())
    }

    fn vulkan_plan(file: &str, source: &str) -> ExecutionPlan {
        compile_artifact(file, source, Backend::Vulkan)
            .expect("the example compiles for Vulkan")
            .execution
    }

    /// The pool holds one descriptor per view each set binds: 14 dispatches of `mnist_mlp`
    /// bind 39, the fused matmul and ReLU of step 0 two reads and two writes
    /// (`crates/modelspec-cli/tests/golden/mnist_mlp.kernel.ir`).
    #[test]
    fn the_pool_counts_every_read_and_write_of_the_mnist_plan() {
        let plan = vulkan_plan(
            "mnist_mlp.mspec",
            include_str!("../../../../examples/mnist_mlp.mspec"),
        );
        let first = &plan.step[0];
        assert_eq!(first.op, PlanOp::Dispatch);
        assert_eq!((first.reads.len(), first.writes.len()), (2, 2));
        assert_eq!(descriptors(&plan), 39);
    }

    /// The sets of the MNIST plan's dispatches, taken in plan order from a pool's room as the
    /// arena binds them: the pool of 38 the runner made before refuses the last set by name,
    /// the pool of [`descriptors`] holds them all.
    #[test]
    fn a_pool_short_of_the_mnist_plan_refuses_its_last_set() {
        use crate::vulkan::ExecError;
        use crate::vulkan::dispatch::PoolRoom;

        let plan = vulkan_plan(
            "mnist_mlp.mspec",
            include_str!("../../../../examples/mnist_mlp.mspec"),
        );
        let sets: Vec<usize> = plan
            .step
            .iter()
            .filter(|step| step.op == PlanOp::Dispatch)
            .map(|step| step.reads.len() + step.writes.len())
            .collect();
        assert_eq!(sets.len(), 14);
        let take_all = |room: &PoolRoom| sets.iter().try_for_each(|&needed| room.take(needed));

        let short = PoolRoom::new(14, 38);
        match take_all(&short) {
            Err(ExecError::DescriptorPoolFull {
                sets: 14,
                descriptors: 38,
                bound_sets: 13,
                bound: 36,
                needed: 3,
            }) => {}
            other => panic!("the pool of 38 refuses the 14th set of 3, got {other:?}"),
        }
        let sized = PoolRoom::new(14, count_u32(descriptors(&plan)).unwrap());
        take_all(&sized).expect("the pool of 39 holds every set");
    }

    /// The 7 dispatches of `linear_regression` bind 19 views, each writing one; a second write on
    /// one dispatch takes one more descriptor, and a copy takes none.
    #[test]
    fn a_two_write_step_takes_a_descriptor_per_write() {
        let mut plan = vulkan_plan("fit.mspec", EXAMPLE);
        assert_eq!(descriptors(&plan), 19);
        let at = plan
            .step
            .iter()
            .position(|step| step.op == PlanOp::Dispatch && step.writes.len() == 1)
            .expect("a dispatch writes one view");
        let written = plan.step[at].writes[0];
        plan.step[at].writes.push(written);
        assert_eq!(descriptors(&plan), 20);
        let copy = PlanStep {
            op: PlanOp::Copy,
            kernel_module_id: None,
            workgroups: None,
            reads: vec![written],
            writes: vec![written],
            depends_on: Vec::new(),
            value: None,
        };
        plan.step.push(copy);
        assert_eq!(descriptors(&plan), 20);
    }

    #[test]
    fn a_finite_read_back_passes() {
        let losses = [3.0, 2.0, 1.0, 0.5, 0.25];
        let c = compiled();
        assert!(diverged(&c, 0, 0..4, true, &losses, &w(FINITE)).is_none());
        assert!(diverged(&c, 0, 1..3, false, &losses, &w(FINITE)).is_none());
    }

    #[test]
    fn the_first_non_finite_loss_of_the_window_is_named_with_the_cpu_text() {
        let losses = [3.0, f32::INFINITY, f32::NAN, 0.5, 0.25];
        let c = compiled();
        assert_eq!(
            refused(diverged(&c, 0, 0..4, true, &losses, &w([f32::NAN; 4]))),
            (
                2,
                "E-TRAIN-001: the loss of step 1 is not finite\n  step: 1\n  loss: inf".into()
            )
        );
    }

    #[test]
    fn a_non_finite_parameter_read_back_is_the_update_of_the_windows_last_step() {
        let losses = [3.0, 2.0, 1.0, 0.5, 0.25];
        let c = compiled();
        let values = w([1.0, 2.0, f32::NEG_INFINITY, 4.0]);
        let expected = (
            4,
            "E-TRAIN-001: the update of step 13 leaves a parameter not finite\n  step: 13\n  \
             loss: 0.5\n  parameter: @LinearReg.w"
                .to_owned(),
        );
        assert_eq!(
            refused(diverged(&c, 10, 2..4, false, &losses, &values)),
            expected
        );
        assert_eq!(
            refused(diverged(&c, 10, 0..4, true, &losses, &values)),
            expected
        );
    }

    #[test]
    fn the_final_loss_is_checked_after_the_last_update() {
        let losses = [3.0, 2.0, 1.0, 0.5, f32::NAN];
        let c = compiled();
        assert_eq!(
            refused(diverged(&c, 0, 0..4, true, &losses, &w(FINITE))),
            (
                5,
                "E-TRAIN-001: the loss of step 4 is not finite\n  step: 4\n  loss: NaN".into()
            )
        );
        assert!(diverged(&c, 0, 0..4, false, &losses, &w(FINITE)).is_none());
    }

    #[test]
    fn a_run_of_no_steps_checks_its_one_loss_then_its_parameters() {
        let c = compiled();
        assert_eq!(
            refused(diverged(&c, 7, 0..0, true, &[f32::NAN], &w([f32::NAN; 4]))),
            (
                1,
                "E-TRAIN-001: the loss of step 7 is not finite\n  step: 7\n  loss: NaN".into()
            )
        );
        assert_eq!(
            refused(diverged(&c, 7, 0..0, true, &[1.5], &w([f32::NAN; 4]))).1,
            "E-TRAIN-001: the update of step 7 leaves a parameter not finite\n  step: 7\n  \
             loss: 1.5\n  parameter: @LinearReg.w"
        );
    }
}
