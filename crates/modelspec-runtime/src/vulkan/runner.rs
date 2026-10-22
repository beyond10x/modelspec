//! A whole training run on one Vulkan device (DESIGN.md § 6): the artifact's step plan, recorded
//! `steps` times into chained command buffers, with one host wait at the end.
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
//! [`train_profiled`] runs the same recording with a timestamp before and after each dispatch,
//! written into one query pool per command buffer and read after the one wait, and returns a
//! [`Profile`] of every dispatch site (DESIGN.md § 8, `--profile`).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::time::{Duration, Instant};

use modelspec_core::artifact::CompiledArtifact;
use modelspec_core::compile::CompiledTraining;
use modelspec_core::device::DeviceProfile;
use modelspec_core::diag::Diagnostic;
use modelspec_core::ir::kernel::{BufferView, ExecutionPlan, KernelKind, PlanOp, PlanStep};
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
use crate::train::data::synthetic;
use crate::train::init::{Stream, initialize};
use crate::train::optimizer::ParamState;
use crate::train::runner::{Run, RunError};

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
    /// From the first submit to the end of the one wait.
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
    /// The initial parameters or the synthetic data could not be drawn, as the CPU run reports.
    #[error(transparent)]
    Run(Box<RunError>),
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

/// Why `modelspec train --backend vulkan` did not run.
#[derive(Debug, Error)]
pub enum DeviceRunError {
    /// No loader, no device, or a device below the requirements: E-TARGET-002 to E-TARGET-004.
    #[error("{0}")]
    Refused(Diagnostic),
    /// The run started and failed.
    #[error(transparent)]
    Failed(#[from] VulkanRunError),
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
    let vulkan = Vulkan::load().map_err(DeviceRunError::Refused)?;
    let candidates = vulkan.candidates().map_err(DeviceRunError::Refused)?;
    let index =
        select_device(&candidates, Some(device.unwrap_or(0))).map_err(DeviceRunError::Refused)?[0];
    let profile = &candidates[index].profile;
    let negotiated = negotiate(profile).map_err(DeviceRunError::Refused)?;
    let opened = vulkan
        .create_device(index, &negotiated)
        .map_err(DeviceRunError::Refused)?;
    Ok(execute(
        &opened,
        profile,
        &negotiated,
        compiled,
        artifact,
        profiled,
    )?)
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
    let (run, stats, _) = execute(device, profile, negotiated, compiled, artifact, false)?;
    Ok((run, stats))
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
    let (run, stats, profiled) = execute(device, profile, negotiated, compiled, artifact, true)?;
    let profiled = profiled.expect("a profiled run returns its profile");
    Ok((run, stats, profiled))
}

/// The timestamps of one command buffer: its pool and the next query to write.
struct Stamps<'p> {
    pool: &'p TimestampPool<'p>,
    next: u32,
}

/// [`train`], profiled when `profiled`.
fn execute(
    device: &Device<'_>,
    profile: &DeviceProfile,
    negotiated: &Negotiated,
    compiled: &CompiledTraining,
    artifact: &CompiledArtifact,
    profiled: bool,
) -> Result<(Run, RunStats, Option<Profile>), VulkanRunError> {
    let plan = &artifact.execution;
    let config = &compiled.config;
    let steps = config.steps;
    let forward = &compiled.forward;
    let refused = |error: RunError| VulkanRunError::Run(Box::new(error));
    let initial = initialize(&forward.program, config.seed, Stream::Init)
        .map_err(|error| refused(error.into()))?;
    let data = synthetic(config, forward).map_err(|error| refused(error.into()))?;

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
    let mut download = device.staging_buffer(param_bytes + 4 * loss_slots)?;

    let dispatch_steps = plan
        .step
        .iter()
        .filter(|step| step.op == PlanOp::Dispatch)
        .count();
    let descriptors: usize = plan
        .step
        .iter()
        .filter(|step| step.op == PlanOp::Dispatch)
        .map(|step| step.reads.len() + 1)
        .sum();
    let arena = device.descriptor_arena(count_u32(dispatch_steps)?, count_u32(descriptors)?)?;
    let (actions, loss_end) = actions(plan, &arena, &pipelines, &buffers)?;

    // The dispatch sites a profile times, and one timestamp pool per command buffer: two
    // queries per dispatch it records.
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
    let forward_sites: Vec<Site> = actions[..loss_end]
        .iter()
        .filter_map(|action| site(PlanName::Forward, action))
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
                queries += 2 * forward_sites.len() as u64;
            }
            let queries = u32::try_from(queries).map_err(|_| {
                VulkanRunError::Plan(format!("{queries} timestamp queries do not fit u32"))
            })?;
            pools.push(device.timestamp_pool(queries)?);
        }
    }
    let stamps_of = |chunk: usize| pools.get(chunk).map(|pool| Stamps { pool, next: 0 });

    let class = numerics_class(negotiated, plan);
    let waits_before = device.host_waits();
    let submits_before = device.submits();
    let dispatches_before = device.dispatches();
    let uploaded_before = device.bytes_uploaded();
    let downloaded_before = device.bytes_downloaded();
    let mut dispatches_per_step = 0;
    let mut first_submit: Option<Instant> = None;
    let started = {
        let loss = buffer(&buffers, &plan.loss_view)?;
        let mut chain = device.chain();
        let mut recording = device.record()?;
        let mut chunk = 0;
        let mut stamps = stamps_of(chunk);
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
        for k in 0..steps {
            if k > 0 && k % CHUNK_STEPS == 0 {
                let recorded = recording.finish()?;
                first_submit.get_or_insert_with(Instant::now);
                chain.submit(recorded)?;
                recording = device.record()?;
                chunk += 1;
                stamps = stamps_of(chunk);
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
        record(
            &mut recording,
            &actions[..loss_end],
            &buffers,
            stamps.as_mut(),
        )?;
        recording.barrier(Barrier::ComputeToTransfer);
        recording.copy_region(loss, plan.loss_view.offset, &losses, 4 * steps, 4)?;
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
        recording.barrier(Barrier::TransferToTransfer);
        recording.copy_region(&losses, 0, &download, at, 4 * loss_slots)?;
        recording.barrier(Barrier::TransferToHost);
        let recorded = recording.finish()?;
        let started = *first_submit.get_or_insert_with(Instant::now);
        chain.finish(recorded)?.wait()?;
        started
    };
    let elapsed = started.elapsed();
    let stats = RunStats {
        steps,
        host_waits: device.host_waits() - waits_before,
        submits: device.submits() - submits_before,
        dispatches: device.dispatches() - dispatches_before,
        dispatches_per_step,
        bytes_uploaded: device.bytes_uploaded() - uploaded_before,
        bytes_downloaded: device.bytes_downloaded() - downloaded_before,
        class,
        elapsed,
    };

    let profiled = if profiled {
        let timings = match clock {
            Some(clock) => {
                let mut stamps = Vec::new();
                for pool in &pools {
                    stamps.extend(pool.read()?);
                }
                Some(
                    timings(&step_sites, &forward_sites, steps, &stamps, clock)
                        .map_err(VulkanRunError::Plan)?,
                )
            }
            None => None,
        };
        Some(Profile {
            timings,
            pipeline_creation,
            compile: emitted,
        })
    } else {
        None
    };

    let read = download.read_f32()?;
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
    let losses = read[at..at + loss_slots as usize].to_vec();
    let mut state = ParamState::new(&compiled.training, values);
    for param in compiled.training.optimizer.updates.keys() {
        if let Some(version) = state.versions.get_mut(param) {
            version.0 = version.0.saturating_add(steps);
        }
    }
    let run = Run {
        start_step: 0,
        initial,
        losses,
        state,
        teacher: data.teacher,
    };
    Ok((run, stats, profiled))
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
                let [written] = step.writes[..] else {
                    return Err(unexpected(at, step, "a dispatch writes one view"));
                };
                let mut bound = Vec::with_capacity(step.reads.len() + 1);
                for view in step.reads.iter().chain([&written]) {
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
