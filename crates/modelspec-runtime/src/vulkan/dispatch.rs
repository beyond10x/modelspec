//! Recording, submitting and waiting.
//!
//! A [`Recording`] is one primary command buffer: whole-buffer copies, [`Barrier`]s and
//! dispatches, in the order they are recorded. [`Recording::finish`] ends it,
//! [`Device::submit`] submits it to the device's one queue with a fence, and
//! [`Submitted::wait`] waits on that fence: one host wait, counted by [`Device::host_waits`].
//! Every buffer and pipeline a recording uses stays borrowed until its submission is waited on.

use ash::vk;
use modelspec_core::ir::kernel::{Kernel, KernelKind, Operand};
use modelspec_core::ir::tensor::{Dim, TensorType};

use super::memory::Buffer;
use super::pipeline::{PUSH_CONSTANT_BYTES, Pipeline};
use super::sync::Barrier;
use super::{Device, ExecError};

/// The command pool, command buffer and descriptor pools of one recording.
struct Commands<'r> {
    device: &'r Device<'r>,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    descriptor_pools: Vec<vk::DescriptorPool>,
}

impl Drop for Commands<'_> {
    fn drop(&mut self) {
        let device = &self.device.device;
        // SAFETY: the command buffer is not pending: it was never submitted, or its
        // `Submitted` waited on its fence before dropping it.
        unsafe {
            for &pool in &self.descriptor_pools {
                device.destroy_descriptor_pool(pool, None);
            }
            device.destroy_command_pool(self.pool, None);
        }
    }
}

/// A command buffer being recorded.
pub struct Recording<'r> {
    commands: Commands<'r>,
}

/// A command buffer recorded and ended, ready for [`Device::submit`].
pub struct Recorded<'r> {
    commands: Commands<'r>,
}

/// A submitted command buffer and the fence it signals.
pub struct Submitted<'r> {
    commands: Commands<'r>,
    fence: vk::Fence,
    waited: bool,
}

impl<'d> Device<'d> {
    /// Starts a recording on a new command pool of the device's queue family.
    pub fn record<'r>(&'r self) -> Result<Recording<'r>, ExecError>
    where
        'd: 'r,
    {
        let info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(self.queue_family);
        // SAFETY: `info` outlives the call; the pool is destroyed by `Commands`' drop.
        let pool = unsafe { self.device.create_command_pool(&info, None) }
            .map_err(ExecError::call("vkCreateCommandPool"))?;
        let mut commands = Commands {
            device: self,
            pool,
            command: vk::CommandBuffer::null(),
            descriptor_pools: Vec::new(),
        };
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: `pool` belongs to this device; the buffer is freed with it.
        commands.command = unsafe { self.device.allocate_command_buffers(&info) }
            .map_err(ExecError::call("vkAllocateCommandBuffers"))?[0];
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer is new, in the initial state.
        unsafe { self.device.begin_command_buffer(commands.command, &begin) }
            .map_err(ExecError::call("vkBeginCommandBuffer"))?;
        Ok(Recording { commands })
    }

    /// Submits `recorded` to the device's queue with a new fence.
    pub fn submit<'r>(&'r self, recorded: Recorded<'r>) -> Result<Submitted<'r>, ExecError> {
        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: `fence_info` outlives the call; the fence is destroyed by `Submitted`'s drop.
        let fence = unsafe { self.device.create_fence(&fence_info, None) }
            .map_err(ExecError::call("vkCreateFence"))?;
        let mut submitted = Submitted {
            commands: recorded.commands,
            fence,
            waited: true,
        };
        // SAFETY: the device was created with one queue of `queue_family`.
        let queue = unsafe { self.device.get_device_queue(self.queue_family, 0) };
        let command_buffers = [submitted.commands.command];
        let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        // SAFETY: the command buffer is ended and not pending; `submits` outlives the call.
        unsafe { self.device.queue_submit(queue, &submits, fence) }
            .map_err(ExecError::call("vkQueueSubmit"))?;
        submitted.waited = false;
        Ok(submitted)
    }
}

impl<'r> Recording<'r> {
    fn device(&self) -> &'r ash::Device {
        &self.commands.device.device
    }

    /// Copies all of `from` to the start of `to`.
    pub fn copy(&mut self, from: &'r Buffer<'_>, to: &'r Buffer<'_>) -> Result<(), ExecError> {
        if from.bytes() > to.bytes() {
            return Err(ExecError::TooShort {
                needed: from.bytes(),
                size: to.bytes(),
            });
        }
        let regions = [vk::BufferCopy::default().size(from.bytes())];
        // SAFETY: both buffers belong to this device and outlive the submission (`'r`).
        unsafe {
            self.device()
                .cmd_copy_buffer(self.commands.command, from.buffer, to.buffer, &regions)
        };
        Ok(())
    }

    /// Records `barrier`.
    pub fn barrier(&mut self, barrier: Barrier) {
        barrier.record(self.device(), self.commands.command);
    }

    /// Dispatches `pipeline` over `count` elements in `workgroups`, with `buffers` bound in
    /// order to bindings 0, 1, … of descriptor set 0 and `count` as the push constant. Each
    /// binding must hold `count` elements, or one if the pipeline marks it rank-0.
    pub fn dispatch(
        &mut self,
        pipeline: &'r Pipeline<'_>,
        buffers: &[&'r Buffer<'_>],
        count: u32,
        workgroups: [u32; 3],
    ) -> Result<(), ExecError> {
        // A rank-0 broadcast binding is read at index 0 by every invocation; every other binding
        // is indexed by invocation up to `count`.
        let needed: Vec<u64> = pipeline
            .broadcast
            .iter()
            .map(|&broadcast| if broadcast { 1 } else { u64::from(count) })
            .collect();
        self.checked_dispatch(
            pipeline,
            buffers,
            &needed,
            count,
            u64::from(count),
            workgroups,
        )
    }

    /// Dispatches the module of `kernel`, built by the emitter, with `buffers` bound to its
    /// operands then its result: the count, the elements each binding must hold and the
    /// invocations its workgroups must cover come from the kernel ([`kernel_bindings`]).
    pub fn dispatch_kernel(
        &mut self,
        pipeline: &'r Pipeline<'_>,
        kernel: &Kernel,
        buffers: &[&'r Buffer<'_>],
    ) -> Result<(), ExecError> {
        let bindings = kernel_bindings(kernel)?;
        self.checked_dispatch(
            pipeline,
            buffers,
            &bindings.elements,
            bindings.count,
            bindings.invocations,
            kernel.workgroups,
        )
    }

    /// Records a two-stage reduction: stage 1 from `input` into `partials`, a compute-to-compute
    /// barrier, then stage 2 from `partials` into `output`.
    pub fn reduce(
        &mut self,
        stage1: (&'r Pipeline<'_>, &Kernel),
        stage2: (&'r Pipeline<'_>, &Kernel),
        input: &'r Buffer<'_>,
        partials: &'r Buffer<'_>,
        output: &'r Buffer<'_>,
    ) -> Result<(), ExecError> {
        for (kernel, kind) in [
            (stage1.1, KernelKind::ReduceStage1),
            (stage2.1, KernelKind::ReduceStage2),
        ] {
            if kernel.kind != kind {
                return Err(ExecError::KernelShape {
                    kernel: kernel.signature(),
                    reason: format!("the reduction stage is not a {kind} kernel"),
                });
            }
        }
        self.dispatch_kernel(stage1.0, stage1.1, &[input, partials])?;
        self.barrier(Barrier::ComputeToCompute);
        self.dispatch_kernel(stage2.0, stage2.1, &[partials, output])
    }

    fn checked_dispatch(
        &mut self,
        pipeline: &'r Pipeline<'_>,
        buffers: &[&'r Buffer<'_>],
        needed: &[u64],
        count: u32,
        invocations: u64,
        workgroups: [u32; 3],
    ) -> Result<(), ExecError> {
        if buffers.len() != pipeline.bindings as usize || needed.len() != buffers.len() {
            return Err(ExecError::Bindings {
                expected: pipeline.bindings,
                got: buffers.len(),
            });
        }
        for (binding, (buffer, &needed)) in buffers.iter().zip(needed).enumerate() {
            let elements = buffer.bytes() / 4;
            if elements < needed {
                return Err(ExecError::CountPastBuffer {
                    count,
                    binding,
                    needed,
                    elements,
                });
            }
        }
        let size = pipeline
            .local_size
            .iter()
            .map(|&n| u64::from(n))
            .product::<u64>();
        let covered = workgroups.iter().map(|&n| u64::from(n)).product::<u64>() * size;
        if covered < invocations {
            return Err(ExecError::Workgroups {
                workgroups,
                size: size as u32,
                covered,
                count,
            });
        }
        let device = self.device();
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(pipeline.bindings.max(1))];
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sizes);
        // SAFETY: `info` outlives the call; the pool is destroyed by `Commands`' drop.
        let pool = unsafe { device.create_descriptor_pool(&info, None) }
            .map_err(ExecError::call("vkCreateDescriptorPool"))?;
        self.commands.descriptor_pools.push(pool);
        let layouts = [pipeline.set_layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        // SAFETY: `pool` has room for one set of this layout.
        let set = unsafe { device.allocate_descriptor_sets(&info) }
            .map_err(ExecError::call("vkAllocateDescriptorSets"))?[0];
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
            .iter()
            .map(|buffer| {
                [vk::DescriptorBufferInfo::default()
                    .buffer(buffer.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)]
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet<'_>> = infos
            .iter()
            .enumerate()
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        let command = self.commands.command;
        let push = count.to_ne_bytes();
        debug_assert_eq!(push.len() as u32, PUSH_CONSTANT_BYTES);
        // SAFETY: the set is new and not in use; `command` is recording; the pipeline, its
        // layout and the buffers outlive the submission (`'r`).
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                &[set],
                &[],
            );
            device.cmd_push_constants(
                command,
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push,
            );
            device.cmd_dispatch(command, workgroups[0], workgroups[1], workgroups[2]);
        }
        Ok(())
    }

    /// Ends the recording.
    pub fn finish(self) -> Result<Recorded<'r>, ExecError> {
        // SAFETY: the command buffer is recording.
        unsafe { self.device().end_command_buffer(self.commands.command) }
            .map_err(ExecError::call("vkEndCommandBuffer"))?;
        Ok(Recorded {
            commands: self.commands,
        })
    }
}

/// What a dispatch of one kernel's module binds and covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelBindings {
    /// The push constant: the output elements of an elementwise, broadcast, SGD, transpose or
    /// matmul kernel; the elements a reduction stage or a loss sum reads.
    pub count: u32,
    /// The elements each binding must hold, operands then result.
    pub elements: Vec<u64>,
    /// The invocations the workgroups must cover.
    pub invocations: u64,
}

fn elements_of(kernel: &Kernel, ty: &TensorType) -> Result<u64, ExecError> {
    ty.shape
        .iter()
        .map(|dim| match dim {
            Dim::Static(n) => Ok(*n),
            Dim::Symbol(_) => Err(ExecError::KernelShape {
                kernel: kernel.signature(),
                reason: format!("{ty} is not static"),
            }),
        })
        .product()
}

/// The count, binding sizes and invocations of a dispatch of `kernel`'s module.
pub fn kernel_bindings(kernel: &Kernel) -> Result<KernelBindings, ExecError> {
    let mut operands = Vec::new();
    for operand in &kernel.operands {
        if let Operand::Buffer(ty) = operand {
            operands.push((ty.rank() == 0, elements_of(kernel, ty)?));
        }
    }
    let result = elements_of(kernel, &kernel.result)?;
    let first = operands.first().map_or(0, |&(_, n)| n);
    let (count, invocations) = match kernel.kind {
        KernelKind::ReduceStage1 => (first, first),
        KernelKind::ReduceStage2 | KernelKind::LossSum => (first, 1),
        KernelKind::Elementwise
        | KernelKind::Broadcast
        | KernelKind::SgdUpdate
        | KernelKind::Transpose
        | KernelKind::MatMul => (result, result),
    };
    let elements = match kernel.kind {
        KernelKind::Elementwise | KernelKind::Broadcast | KernelKind::SgdUpdate => operands
            .iter()
            .map(|&(scalar, _)| if scalar { 1 } else { count })
            .chain([count])
            .collect(),
        _ => operands.iter().map(|&(_, n)| n).chain([result]).collect(),
    };
    let count = u32::try_from(count).map_err(|_| ExecError::KernelShape {
        kernel: kernel.signature(),
        reason: format!("a count of {count} does not fit u32"),
    })?;
    Ok(KernelBindings {
        count,
        elements,
        invocations,
    })
}

impl Submitted<'_> {
    /// Waits on the submission's fence: one host wait.
    pub fn wait(mut self) -> Result<(), ExecError> {
        self.block()
    }

    fn block(&mut self) -> Result<(), ExecError> {
        if self.waited {
            return Ok(());
        }
        let device = self.commands.device;
        // SAFETY: the fence belongs to this device and was submitted.
        let waited = unsafe { device.device.wait_for_fences(&[self.fence], true, u64::MAX) }
            .map_err(ExecError::call("vkWaitForFences"));
        device.host_waits.set(device.host_waits.get() + 1);
        if waited.is_ok() {
            self.waited = true;
        }
        waited
    }
}

impl Drop for Submitted<'_> {
    fn drop(&mut self) {
        if self.block().is_err() {
            // The fence never signalled: the device is lost, and waiting on all of it is the
            // only way left to free the command buffer safely.
            // SAFETY: the device is alive for the call.
            let _ = unsafe { self.commands.device.device.device_wait_idle() };
        }
        // SAFETY: the fence is signalled or the device idle, so nothing waits on it.
        unsafe { self.commands.device.device.destroy_fence(self.fence, None) };
    }
}
