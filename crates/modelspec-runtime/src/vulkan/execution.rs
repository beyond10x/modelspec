//! An in-process Vulkan queue for native clients. Its worker owns the Vulkan loader, device,
//! and compiled plans, so their lifetimes follow the driver's ownership without leaked handles
//! or self-referential application state. Commands from different plans are serialized on the
//! compute queue. A plan's device-local parameters stay there until their host values change.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, Weak, mpsc};

use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{BufferView, ExecutionPlan, MatMulLimits, PlanOp};
use modelspec_core::ir::tensor::{DType, ParamId, TensorProgram};
use modelspec_core::ir::verify::verify_tensor_plan_for;
use modelspec_core::schedule::schedule_tensor;

use super::execution_validation::Checks;
use super::memory::Buffer;
use super::negotiate::negotiate;
use super::pipeline::{KernelPipelines, PipelineRequest, reads_seed};
use super::pipeline_cache::PipelineCache;
use super::runner::{actions, buffer, count_u32, descriptors, record};
use super::select::select_device;
use super::sync::Barrier;
use super::{Device, Vulkan};
use crate::cpu::HostTensor;
use crate::cpu::interpreter::{check_inputs, check_params, static_shape};
use crate::execution::{ExecutionError, Output};
use crate::spirv::emit::{KernelModule, emit_module};

type GpuResult<T> = Result<T, ExecutionError>;
type Params = BTreeMap<ParamId, HostTensor<f32>>;
type Reply<T> = mpsc::Sender<GpuResult<T>>;

fn error(reason: impl std::fmt::Display) -> ExecutionError {
    ExecutionError::Vulkan(reason.to_string())
}

fn checked<T>(loader: &Vulkan, result: GpuResult<T>) -> GpuResult<T> {
    let messages = loader.take_validation_messages();
    if !messages.is_empty() {
        return Err(error(messages.join("\n")));
    }
    result
}

struct Call {
    floats: BTreeMap<String, HostTensor<f32>>,
    ids: BTreeMap<String, HostTensor<i32>>,
    changed: Params,
    seed: Option<u64>,
}

enum Command {
    Prepare(Arc<TensorProgram>, Reply<u64>),
    Run(u64, Call, Reply<Vec<Output>>),
    Remove(u64),
}

struct Queue(mpsc::Sender<Command>);
static QUEUES: Mutex<BTreeMap<usize, Weak<Queue>>> = Mutex::new(BTreeMap::new());

impl Queue {
    fn shared(index: usize) -> GpuResult<Arc<Self>> {
        let mut old = QUEUES.lock().map_err(error)?;
        if let Some(queue) = old.get(&index).and_then(Weak::upgrade) {
            return Ok(queue);
        }
        let (send, commands) = mpsc::channel();
        let (ready, started) = mpsc::channel();
        std::thread::Builder::new()
            .name(format!("modelspec-vulkan-{index}"))
            .spawn(move || {
                if let Err(cause) = serve(index, &ready, commands) {
                    let _ = ready.send(Err(cause));
                }
            })
            .map_err(error)?;
        started.recv().map_err(error)??;
        let queue = Arc::new(Self(send));
        old.insert(index, Arc::downgrade(&queue));
        Ok(queue)
    }

    fn request<T>(&self, command: impl FnOnce(Reply<T>) -> Command) -> GpuResult<T> {
        let (send, result) = mpsc::channel();
        self.0.send(command(send)).map_err(error)?;
        result.recv().map_err(error)?
    }
}

fn serve(index: usize, ready: &Reply<()>, commands: mpsc::Receiver<Command>) -> GpuResult<()> {
    let loader = Vulkan::load().map_err(ExecutionError::Target)?;
    let candidates = loader.candidates().map_err(ExecutionError::Target)?;
    let index = select_device(&candidates, Some(index)).map_err(ExecutionError::Target)?[0];
    let profile = &candidates[index].profile;
    let negotiated = negotiate(profile).map_err(ExecutionError::Target)?;
    let device = loader
        .create_device(index, &negotiated)
        .map_err(ExecutionError::Target)?;
    let cache = device.pipeline_cache(profile, None).map_err(error)?;
    let mut plans = BTreeMap::new();
    let mut next = 0;
    ready.send(Ok(())).map_err(error)?;
    for command in commands {
        match command {
            Command::Prepare(program, reply) => {
                let compiled = Compiled::new(&device, profile, &cache, &program);
                let result = checked(&loader, compiled).map(|plan| {
                    let id = next;
                    next += 1;
                    plans.insert(id, plan);
                    id
                });
                let _ = reply.send(result);
            }
            Command::Run(id, call, reply) => {
                let result = plans
                    .get_mut(&id)
                    .ok_or_else(|| error("plan was released"))
                    .and_then(|plan| plan.run(&device, &cache, call));
                let _ = reply.send(checked(&loader, result));
            }
            Command::Remove(id) => {
                plans.remove(&id);
            }
        }
    }
    Ok(())
}

pub(crate) struct DevicePlan {
    queue: Arc<Queue>,
    id: u64,
    program: Arc<TensorProgram>,
    uploaded: Params,
    checks: Checks,
    output_count: usize,
}

impl DevicePlan {
    pub(crate) fn new(mut program: TensorProgram, index: usize) -> GpuResult<Self> {
        let output_count = program.outputs.len();
        let checks = Checks::prepare(&mut program);
        let queue = Queue::shared(index)?;
        let program = Arc::new(program);
        let id = queue.request(|reply| Command::Prepare(Arc::clone(&program), reply))?;
        Ok(Self {
            queue,
            id,
            program,
            uploaded: Params::new(),
            checks,
            output_count,
        })
    }

    pub(crate) fn run(
        &mut self,
        floats: &BTreeMap<String, HostTensor<f32>>,
        ids: &BTreeMap<String, HostTensor<i32>>,
        params: &Params,
        seed: Option<u64>,
    ) -> GpuResult<Vec<Output>> {
        check_inputs(&self.program, floats, ids)?;
        check_params(&self.program, params)?;
        self.checks.before(floats, ids, params, seed)?;
        let changed: Params = params
            .iter()
            .filter(|(id, value)| {
                !self.uploaded.get(id).is_some_and(|old| {
                    old.data()
                        .iter()
                        .zip(value.data())
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                })
            })
            .map(|(id, value)| (*id, value.clone()))
            .collect();
        let dirty: Vec<_> = changed.keys().copied().collect();
        let call = Call {
            floats: floats.clone(),
            ids: ids.clone(),
            changed,
            seed,
        };
        let result = self
            .queue
            .request(|reply| Command::Run(self.id, call, reply));
        if result.is_ok() {
            for id in dirty {
                self.uploaded.insert(id, params[&id].clone());
            }
        } else {
            // A failed submission may already have copied some of its inputs.
            self.uploaded.clear();
        }
        let mut outputs = result?;
        self.checks.after(&outputs)?;
        outputs.truncate(self.output_count);
        Ok(outputs)
    }
}

impl Drop for DevicePlan {
    fn drop(&mut self) {
        let _ = self.queue.0.send(Command::Remove(self.id));
    }
}

struct Compiled<'d> {
    plan: ExecutionPlan,
    buffers: Vec<Buffer<'d>>,
    modules: Vec<KernelModule>,
    pipelines: KernelPipelines<'d>,
    needs_seed: bool,
    seed: u64,
    upload: Buffer<'d>,
    download: Buffer<'d>,
    outputs: Vec<(BufferView, Vec<u64>)>,
}

fn pipelines<'d>(
    device: &'d Device<'_>,
    cache: &PipelineCache<'_>,
    plan: &ExecutionPlan,
    modules: &[KernelModule],
    seed: u64,
) -> GpuResult<KernelPipelines<'d>> {
    let requests = plan
        .kernels
        .iter()
        .zip(modules)
        .map(|(kernel, module)| PipelineRequest {
            kernel,
            words: &module.spirv,
            seed: reads_seed(kernel).then_some(seed),
        })
        .collect::<Vec<_>>();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    device
        .kernel_pipelines(&requests, Some(cache), threads)
        .map_err(error)
}

impl<'d> Compiled<'d> {
    fn new(
        device: &'d Device<'_>,
        profile: &DeviceProfile,
        cache: &PipelineCache<'_>,
        program: &TensorProgram,
    ) -> GpuResult<Self> {
        let limits = MatMulLimits::of(profile).checked().map_err(error)?;
        let (plan, views) = schedule_tensor(program, Some(limits)).map_err(error)?;
        verify_tensor_plan_for(&plan, &views, profile).map_err(error)?;
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
            .map(|kernel| emit_module(kernel, profile))
            .collect::<Result<Vec<_>, _>>()
            .map_err(error)?;
        let made = pipelines(device, cache, &plan, &modules, 0)?;
        let buffers = plan
            .buffers
            .iter()
            .map(|&size| device.device_local_buffer(size))
            .collect::<Result<_, _>>()
            .map_err(error)?;
        let upload_bytes = plan
            .data
            .iter()
            .map(|d| d.view.size_bytes)
            .chain(plan.parameters.iter().map(|p| p.views[0].size_bytes))
            .sum::<u64>();
        let upload = device.staging_buffer(upload_bytes.max(4)).map_err(error)?;
        let download = device
            .download_buffer(views.iter().map(|v| v.size_bytes).sum())
            .map_err(error)?;
        let outputs = views
            .into_iter()
            .zip(&program.outputs)
            .map(|(view, id)| {
                (
                    view,
                    static_shape(&program.node(*id).expect("verified output").ty),
                )
            })
            .collect();
        let needs_seed = plan.kernels.iter().any(reads_seed);
        Ok(Self {
            plan,
            buffers,
            modules,
            pipelines: made,
            needs_seed,
            seed: 0,
            upload,
            download,
            outputs,
        })
    }

    fn run(
        &mut self,
        device: &'d Device<'_>,
        cache: &PipelineCache<'_>,
        call: Call,
    ) -> GpuResult<Vec<Output>> {
        if let Some(seed) = call.seed
            && self.needs_seed
            && seed != self.seed
        {
            self.pipelines = pipelines(device, cache, &self.plan, &self.modules, seed)?;
            self.seed = seed;
        }
        let mut words = Vec::new();
        let mut copies = Vec::new();
        for parameter in &self.plan.parameters {
            if let Some(value) = call.changed.get(&parameter.param) {
                copies.push((words.len() as u64 * 4, parameter.views[0]));
                words.extend(value.data().iter().map(|x| x.to_bits() as i32));
            }
        }
        for input in &self.plan.data {
            copies.push((words.len() as u64 * 4, input.view));
            match input.ty.dtype {
                DType::F32 => words.extend(
                    call.floats[&input.name]
                        .data()
                        .iter()
                        .map(|x| x.to_bits() as i32),
                ),
                DType::I32 => words.extend_from_slice(call.ids[&input.name].data()),
            }
        }
        self.upload.write_i32(&words).map_err(error)?;
        let dispatches = self
            .plan
            .step
            .iter()
            .filter(|s| s.op == PlanOp::Dispatch)
            .count();
        let arena = device
            .descriptor_arena(
                count_u32(dispatches).map_err(error)?,
                count_u32(descriptors(&self.plan)).map_err(error)?,
            )
            .map_err(error)?;
        let (actions, _) =
            actions(&self.plan, &arena, &self.pipelines.each(), &self.buffers).map_err(error)?;
        let mut rec = device.record().map_err(error)?;
        for (offset, view) in copies {
            rec.copy_region(
                &self.upload,
                offset,
                buffer(&self.buffers, &view).map_err(error)?,
                view.offset,
                view.size_bytes,
            )
            .map_err(error)?;
        }
        rec.barrier(Barrier::TransferToTransfer);
        rec.barrier(Barrier::TransferToCompute);
        record(&mut rec, &actions, &self.buffers, None).map_err(error)?;
        rec.barrier(Barrier::TransferToTransfer);
        rec.barrier(Barrier::ComputeToTransfer);
        let mut output_offset = 0;
        for (view, _) in &self.outputs {
            rec.copy_region(
                buffer(&self.buffers, view).map_err(error)?,
                view.offset,
                &self.download,
                output_offset,
                view.size_bytes,
            )
            .map_err(error)?;
            output_offset += view.size_bytes;
        }
        rec.barrier(Barrier::TransferToHost);
        device
            .submit(rec.finish().map_err(error)?)
            .map_err(error)?
            .wait()
            .map_err(error)?;
        let floats = self.download.mapped_f32().map_err(error)?;
        let integers = self.download.mapped_i32().map_err(error)?;
        let mut offset = 0;
        Ok(self
            .outputs
            .iter()
            .map(|(view, shape)| {
                let range = offset..offset + view.element_count as usize;
                offset = range.end;
                match view.dtype {
                    DType::F32 => Output::F32(HostTensor::from_parts(
                        shape.clone(),
                        floats[range].to_vec(),
                    )),
                    DType::I32 => Output::I32(HostTensor::from_parts(
                        shape.clone(),
                        integers[range].to_vec(),
                    )),
                }
            })
            .collect())
    }
}
