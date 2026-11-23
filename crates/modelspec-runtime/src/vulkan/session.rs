//! A training worker owns the device, bindings and recordings for their full borrowed
//! lifetimes. A fresh minibatch overwrites only its staging range after the preceding fence.

use std::collections::BTreeMap;
use std::sync::mpsc;
use std::thread::JoinHandle;

use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{BufferView, MatMulLimits, PlanOp};
use modelspec_core::ir::tensor::{DType, ParamId, ValueId};
use modelspec_core::ir::training::{C1_INPUT, C2_INPUT, LR_T_INPUT, Moment, TrainingStep};
use modelspec_core::ir::verify::verify_tensor_plan_for;
use modelspec_core::schedule::schedule_training_session;

use super::execution::{checked, error, pipelines, with_device};
use super::execution_validation::Checks;
use super::pipeline_cache::PipelineCache;
use super::runner::{actions, count_u32, descriptors, record};
use super::stepwise::{copy_state, restore_state};
use super::sync::Barrier;
use super::{Device, Vulkan};
use crate::cpu::HostTensor;
use crate::cpu::interpreter::static_shape;
use crate::execution::{ExecutionError, Output};
use crate::spirv::emit::emit_module;
use crate::train::session::{Floats, Ids, TrainingDeviceStats, TrainingSnapshot, invalid};

type Result<T> = std::result::Result<T, ExecutionError>;
type Reply<T> = mpsc::Sender<Result<T>>;

enum Binding {
    Float(String),
    Integer(String),
    Scalar(usize),
}

struct Layout {
    bindings: Vec<Binding>,
    words: usize,
}

enum Command {
    Update(Vec<f32>, Reply<(f32, Vec<Output>)>),
    Snapshot(u64, Reply<TrainingSnapshot>),
    Stats(Reply<TrainingDeviceStats>),
}

pub(crate) struct DeviceSession {
    commands: Option<mpsc::Sender<Command>>,
    worker: Option<JoinHandle<()>>,
    controls: Checks,
    layout: Layout,
    seed: u64,
    failed: bool,
}

impl DeviceSession {
    pub(crate) fn new(
        index: usize,
        step: TrainingStep,
        initial: TrainingSnapshot,
        seed: u64,
    ) -> Result<Self> {
        let mut observed = step.training.program.clone();
        observed.outputs.clear();
        let controls = Checks::prepare_training(&mut observed);
        let outputs = observed.outputs;
        let (ready, started) = mpsc::channel();
        let (commands, receive) = mpsc::channel();
        let worker = std::thread::Builder::new()
            .name(format!("modelspec-adamw-{index}"))
            .spawn(move || {
                let result = with_device(index, |loader, device, profile, cache| {
                    serve(
                        loader, device, profile, cache, step, initial, seed, &outputs, &ready,
                        receive,
                    )
                });
                if let Err(e) = result {
                    let _ = ready.send(Err(e));
                }
            })
            .map_err(error)?;
        match started.recv().map_err(error).and_then(|r| r) {
            Ok(layout) => Ok(Self {
                commands: Some(commands),
                worker: Some(worker),
                controls,
                layout,
                seed,
                failed: false,
            }),
            Err(e) => {
                drop(commands);
                let _ = worker.join();
                Err(e)
            }
        }
    }

    fn request<T>(&self, make: impl FnOnce(Reply<T>) -> Command) -> Result<T> {
        if self.failed {
            return Err(invalid("resident update failed; restore a checkpoint"));
        }
        let (send, recv) = mpsc::channel();
        self.commands
            .as_ref()
            .expect("active session")
            .send(make(send))
            .map_err(error)?;
        recv.recv().map_err(error)?
    }

    pub(crate) fn update(&mut self, floats: &Floats, ids: &Ids, scalars: [f32; 3]) -> Result<f32> {
        self.controls
            .before(floats, ids, &BTreeMap::new(), Some(self.seed))?;
        let mut data = Vec::with_capacity(self.layout.words);
        for key in &self.layout.bindings {
            match key {
                Binding::Float(k) => data.extend_from_slice(floats[k].data()),
                Binding::Integer(k) => {
                    data.extend(ids[k].data().iter().map(|&n| f32::from_bits(n as u32)))
                }
                Binding::Scalar(i) => data.push(scalars[*i]),
            }
        }
        let result =
            self.request(|reply| Command::Update(data, reply))
                .and_then(|(loss, controls)| {
                    self.controls.after(&controls)?;
                    if !loss.is_finite() {
                        return Err(invalid("non-finite training loss"));
                    }
                    Ok(loss)
                });
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    pub(crate) fn snapshot(&mut self, step: u64) -> Result<TrainingSnapshot> {
        self.request(|reply| Command::Snapshot(step, reply))
    }

    pub(crate) fn stats(&self) -> Result<TrainingDeviceStats> {
        self.request(Command::Stats)
    }
}

impl Drop for DeviceSession {
    fn drop(&mut self) {
        drop(self.commands.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[derive(Clone, Copy)]
enum StateKey {
    Parameter(ParamId),
    Slot(ParamId, Moment),
}

fn saved(s: &TrainingSnapshot, key: StateKey) -> &HostTensor<f32> {
    match key {
        StateKey::Parameter(p) => &s.parameters[&p],
        StateKey::Slot(p, m) => &s.moments[&(p, m)],
    }
}

fn bytes(views: &[BufferView]) -> u64 {
    views.iter().map(|v| v.size_bytes).sum::<u64>().max(4)
}

#[allow(clippy::too_many_arguments)]
fn serve(
    loader: &Vulkan,
    device: &Device<'_>,
    profile: &DeviceProfile,
    cache: &PipelineCache<'_>,
    step: TrainingStep,
    initial: TrainingSnapshot,
    seed: u64,
    outputs: &[ValueId],
    ready: &Reply<Layout>,
    commands: mpsc::Receiver<Command>,
) -> Result<()> {
    let limits = MatMulLimits::of(profile).checked().map_err(error)?;
    let scheduled = schedule_training_session(&step, outputs, Some(limits)).map_err(error)?;
    let plan = scheduled.plan;
    verify_tensor_plan_for(&plan, &scheduled.outputs, profile).map_err(error)?;
    if plan
        .buffers
        .iter()
        .any(|&n| n > u64::from(profile.max_storage_buffer_range))
    {
        return Err(error(
            "plan exceeds the device's maximum storage buffer range",
        ));
    }
    let modules = plan
        .kernels
        .iter()
        .map(|k| emit_module(k, profile))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(error)?;
    let compiled = pipelines(device, cache, &plan, &modules, seed)?;
    let buffers = plan
        .buffers
        .iter()
        .map(|&n| device.device_local_buffer(n))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(error)?;
    let ds = plan
        .step
        .iter()
        .filter(|s| s.op == PlanOp::Dispatch)
        .count();
    let arena = device
        .descriptor_arena(
            count_u32(ds).map_err(error)?,
            count_u32(descriptors(&plan)).map_err(error)?,
        )
        .map_err(error)?;
    let kernels = compiled.each();
    let (actions, _) = actions(&plan, &arena, &kernels, &buffers).map_err(error)?;

    let mut states: Vec<_> = plan
        .parameters
        .iter()
        .map(|p| (StateKey::Parameter(p.param), p.views[0]))
        .collect();
    let slots: BTreeMap<_, _> = step
        .optimizer
        .state_slots()
        .into_iter()
        .map(|s| (s.input, StateKey::Slot(s.param, s.moment)))
        .collect();
    let binding = |name: String, dtype: DType| {
        if dtype == DType::I32 {
            Binding::Integer(name)
        } else {
            Binding::Float(name)
        }
    };
    let mut batch: Vec<_> = plan
        .data
        .iter()
        .map(|p| (binding(p.name.clone(), p.ty.dtype), p.view))
        .collect();
    for h in scheduled.host {
        if let Some(key) = slots.get(&h.value) {
            states.push((*key, h.view));
        } else {
            let role = [LR_T_INPUT, C1_INPUT, C2_INPUT]
                .iter()
                .position(|k| *k == h.name)
                .map_or_else(|| binding(h.name, h.ty.dtype), Binding::Scalar);
            batch.push((role, h.view));
        }
    }
    let (keys, state_views): (Vec<_>, Vec<_>) = states.into_iter().unzip();
    let (bindings, batch_views): (Vec<_>, Vec<_>) = batch.into_iter().unzip();
    let shapes: Vec<_> = keys
        .iter()
        .map(|&k| saved(&initial, k).shape().to_vec())
        .collect();
    // The initial upload is used once and released before the first minibatch.
    {
        let mut init_upload = device.staging_buffer(bytes(&state_views)).map_err(error)?;
        let words: Vec<_> = keys
            .iter()
            .flat_map(|&k| saved(&initial, k).data().iter().copied())
            .collect();
        init_upload.write_f32(&words).map_err(error)?;
        let mut rec = device.record().map_err(error)?;
        restore_state(&mut rec, &state_views, &buffers, &init_upload).map_err(error)?;
        rec.barrier(Barrier::TransferToCompute);
        device
            .submit(rec.finish().map_err(error)?)
            .map_err(error)?
            .wait()
            .map_err(error)?;
    }
    drop(initial);
    let upload = device.staging_buffer(bytes(&batch_views)).map_err(error)?;
    let returned: Vec<_> = std::iter::once(plan.loss_view)
        .chain(scheduled.outputs)
        .collect();
    let result_shapes: Vec<_> = outputs
        .iter()
        .map(|id| {
            static_shape(
                &step
                    .training
                    .program
                    .node(*id)
                    .expect("verified control")
                    .ty,
            )
        })
        .collect();
    let result_buffer = device.download_buffer(bytes(&returned)).map_err(error)?;
    let mut rec = device.record_reusable().map_err(error)?;
    rec.barrier(Barrier::ComputeToCompute);
    rec.barrier(Barrier::ComputeToTransfer);
    rec.barrier(Barrier::TransferToTransfer);
    restore_state(&mut rec, &batch_views, &buffers, &upload).map_err(error)?;
    rec.barrier(Barrier::TransferToCompute);
    record(&mut rec, &actions, &buffers, None).map_err(error)?;
    rec.barrier(Barrier::ComputeToTransfer);
    rec.barrier(Barrier::TransferToTransfer);
    copy_state(&mut rec, &returned, &buffers, &result_buffer).map_err(error)?;
    rec.barrier(Barrier::TransferToHost);
    let update = rec.finish().map_err(error)?.reusable().map_err(error)?;
    let snapshot_buffer = device.download_buffer(bytes(&state_views)).map_err(error)?;
    let mut rec = device.record_reusable().map_err(error)?;
    rec.barrier(Barrier::ComputeToTransfer);
    rec.barrier(Barrier::TransferToTransfer);
    copy_state(&mut rec, &state_views, &buffers, &snapshot_buffer).map_err(error)?;
    rec.barrier(Barrier::TransferToHost);
    let snapshot = rec.finish().map_err(error)?.reusable().map_err(error)?;
    let words = batch_views.iter().map(|v| v.element_count as usize).sum();
    ready
        .send(checked(loader, Ok(Layout { bindings, words })))
        .map_err(error)?;
    for command in commands {
        match command {
            Command::Stats(reply) => {
                let _ = reply.send(Ok(TrainingDeviceStats {
                    recordings: device.recordings(),
                    submits: device.submits(),
                    host_waits: device.host_waits(),
                    dispatches: device.dispatches(),
                    bytes_uploaded: device.bytes_uploaded(),
                    bytes_downloaded: device.bytes_downloaded(),
                }));
            }
            Command::Update(batch, reply) => {
                let result = (|| {
                    // SAFETY: the sole owning worker waits for each previous submission
                    // before accepting the next message, and never maps an upload for reading.
                    unsafe { upload.write_f32_shared(&batch) }.map_err(error)?;
                    device
                        .submit_again(&update)
                        .map_err(error)?
                        .wait()
                        .map_err(error)?;
                    let data = result_buffer.mapped_f32().map_err(error)?;
                    let loss = data[0];
                    let mut offset = 1;
                    let observed = result_shapes
                        .iter()
                        .zip(returned.iter().skip(1))
                        .map(|(shape, v)| {
                            let end = offset + v.element_count as usize;
                            let tensor =
                                HostTensor::from_parts(shape.clone(), data[offset..end].to_vec());
                            offset = end;
                            Output::F32(tensor)
                        })
                        .collect();
                    Ok((loss, observed))
                })();
                let _ = reply.send(checked(loader, result));
            }
            Command::Snapshot(step, reply) => {
                let result = (|| {
                    device
                        .submit_again(&snapshot)
                        .map_err(error)?
                        .wait()
                        .map_err(error)?;
                    let data = snapshot_buffer.mapped_f32().map_err(error)?;
                    let mut state = TrainingSnapshot {
                        step,
                        parameters: BTreeMap::new(),
                        moments: BTreeMap::new(),
                    };
                    let mut offset = 0;
                    for ((key, shape), view) in keys.iter().zip(&shapes).zip(&state_views) {
                        let end = offset + view.element_count as usize;
                        let t = HostTensor::from_parts(shape.clone(), data[offset..end].to_vec());
                        match *key {
                            StateKey::Parameter(p) => {
                                state.parameters.insert(p, t);
                            }
                            StateKey::Slot(p, m) => {
                                state.moments.insert((p, m), t);
                            }
                        }
                        offset = end;
                    }
                    Ok(state)
                })();
                let _ = reply.send(checked(loader, result));
            }
        }
    }
    Ok(())
}
