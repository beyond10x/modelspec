//! Recording, submitting and waiting.
//!
//! A [`Recording`] is one primary command buffer: whole-buffer copies, [`Barrier`]s and
//! dispatches, in the order they are recorded. [`Recording::finish`] ends it,
//! [`Device::submit`] submits it to the device's one queue with a fence, and
//! [`Submitted::wait`] waits on that fence: one host wait, counted by [`Device::host_waits`].
//! Every buffer and pipeline a recording uses stays borrowed until its submission is waited on.
//!
//! A [`Chain`] submits several recordings to the queue one after the other without a fence and
//! fences only the last ([`Chain::finish`]): a fence signals once every earlier submission to the
//! queue has finished, so the one wait covers the whole chain. A [`DescriptorArena`] binds a
//! dispatch's buffers once ([`DescriptorArena::bind`]) for a [`Bound`] dispatch that many
//! recordings record ([`Recording::dispatch_bound`]), and refuses a set past the sets or
//! descriptors of its pool ([`ExecError::DescriptorPoolFull`]). Each submission adds what it
//! recorded to the device's counters: one submit, its dispatches, and the bytes of its copies
//! from a staging buffer to a device-local one (uploaded) and back (downloaded).

use std::cell::Cell;
use std::marker::PhantomData;

use ash::vk;
use modelspec_core::ir::kernel::{Kernel, KernelKind, MatMulTile, Operand};
use modelspec_core::ir::tensor::{Dim, TensorType};

use super::memory::Buffer;
use super::pipeline::{PUSH_CONSTANT_BYTES, Pipeline};
use super::pool::{Admitted, PoolRoom, admit, pool_info, pool_sizes};
use super::sync::Barrier;
use super::{Device, ExecError};

/// What one recording recorded, added to the device's counters when it is submitted.
#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    dispatches: u64,
    uploaded: u64,
    downloaded: u64,
}

/// The command pool, command buffer and descriptor pools of one recording.
struct Commands<'r> {
    device: &'r Device<'r>,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    descriptor_pools: Vec<vk::DescriptorPool>,
    counts: Counts,
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

/// A submitted command buffer and the fence it signals, with the command buffers of a
/// [`Chain`] submitted before it, which the fence also covers.
pub struct Submitted<'r> {
    commands: Commands<'r>,
    fence: vk::Fence,
    waited: bool,
    earlier: Vec<Commands<'r>>,
}

/// Command buffers submitted one after the other without a fence; [`Chain::finish`] fences the
/// last. Dropped before then, it waits for the device to go idle: a host wait.
pub struct Chain<'r> {
    device: &'r Device<'r>,
    pending: Vec<Commands<'r>>,
}

impl Device<'_> {
    /// Adds the counts of a submitted recording to the device's counters.
    fn count(&self, counts: Counts) {
        let add = |cell: &Cell<u64>, n: u64| cell.set(cell.get() + n);
        add(&self.submits, 1);
        add(&self.dispatches, counts.dispatches);
        add(&self.bytes_uploaded, counts.uploaded);
        add(&self.bytes_downloaded, counts.downloaded);
    }

    /// Submits `command` to the device's one queue, signalling `fence` (null for none).
    fn queue_submit(&self, command: vk::CommandBuffer, fence: vk::Fence) -> Result<(), ExecError> {
        // SAFETY: the device was created with one queue of `queue_family`.
        let queue = unsafe { self.device.get_device_queue(self.queue_family, 0) };
        let command_buffers = [command];
        let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        // SAFETY: the command buffer is ended and not pending; `submits` outlives the call.
        unsafe { self.device.queue_submit(queue, &submits, fence) }
            .map_err(ExecError::call("vkQueueSubmit"))
    }

    /// Starts a chain of submissions behind one fence.
    pub fn chain<'r>(&'r self) -> Chain<'r> {
        Chain {
            device: self,
            pending: Vec::new(),
        }
    }
}

impl<'r> Chain<'r> {
    /// Submits `recorded` without a fence and keeps it until the chain's fence is waited on.
    pub fn submit(&mut self, recorded: Recorded<'r>) -> Result<(), ExecError> {
        self.device
            .queue_submit(recorded.commands.command, vk::Fence::null())?;
        self.device.count(recorded.commands.counts);
        self.pending.push(recorded.commands);
        Ok(())
    }

    /// Submits `recorded` with a new fence, which signals once it and every recording the chain
    /// submitted before it have finished.
    pub fn finish(mut self, recorded: Recorded<'r>) -> Result<Submitted<'r>, ExecError> {
        let mut submitted = self.device.submit(recorded)?;
        submitted.earlier = std::mem::take(&mut self.pending);
        Ok(submitted)
    }
}

impl Drop for Chain<'_> {
    fn drop(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        // The chain was not finished: nothing fences its submissions, and waiting for the
        // device to go idle is the only way to free their command buffers safely.
        // SAFETY: the device is alive for the call.
        let _ = unsafe { self.device.device.device_wait_idle() };
        self.device.host_waits.set(self.device.host_waits.get() + 1);
    }
}

/// Descriptor sets made once and bound by many dispatches: one pool, destroyed with the arena.
pub struct DescriptorArena<'d> {
    device: &'d Device<'d>,
    pool: vk::DescriptorPool,
    room: PoolRoom,
}

impl Drop for DescriptorArena<'_> {
    fn drop(&mut self) {
        // SAFETY: every `Bound` borrows the arena and every recording that records one borrows
        // the `Bound`, so no pending command buffer uses a set of the pool.
        unsafe { self.device.device.destroy_descriptor_pool(self.pool, None) };
    }
}

/// A dispatch whose buffers are bound in a descriptor set of a [`DescriptorArena`], checked as
/// [`Recording::dispatch_kernel`] checks them, ready to record any number of times.
pub struct Bound<'a> {
    pipeline: &'a Pipeline<'a>,
    set: vk::DescriptorSet,
    count: u32,
    workgroups: [u32; 3],
    _buffers: PhantomData<(&'a DescriptorArena<'a>, &'a Buffer<'a>)>,
}

impl Device<'_> {
    /// An arena with room for `sets` descriptor sets of `descriptors` storage buffers in all.
    pub fn descriptor_arena<'d>(
        &'d self,
        sets: u32,
        descriptors: u32,
    ) -> Result<DescriptorArena<'d>, ExecError> {
        let room = PoolRoom::new(sets, descriptors);
        let pool = create_pool(&self.device, &room)?;
        Ok(DescriptorArena {
            device: self,
            pool,
            room,
        })
    }
}

impl<'d> DescriptorArena<'d> {
    /// Binds `buffers` to the module of `kernel`, its buffer operands then each binding it
    /// writes, as [`Recording::dispatch_kernel`] would, in a new descriptor set of the arena.
    pub fn bind<'a>(
        &'a self,
        pipeline: &'a Pipeline<'_>,
        kernel: &Kernel,
        buffers: &[&'a Buffer<'_>],
    ) -> Result<Bound<'a>, ExecError>
    where
        'd: 'a,
    {
        let bindings = kernel_bindings(kernel)?;
        check_dispatch(
            pipeline,
            buffers,
            &bindings.elements,
            bindings.count,
            bindings.invocations,
            kernel.workgroups,
        )?;
        let admitted = admit(&self.room, buffers.len())?;
        let set = allocate_set(&self.device.device, self.pool, admitted, pipeline, buffers)?;
        Ok(Bound {
            pipeline,
            set,
            count: bindings.count,
            workgroups: kernel.workgroups,
            _buffers: PhantomData,
        })
    }
}

/// Checks a dispatch of `pipeline` over `buffers` as [`Recording::dispatch`] documents.
fn check_dispatch(
    pipeline: &Pipeline<'_>,
    buffers: &[&Buffer<'_>],
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
    Ok(())
}

/// A descriptor pool made with exactly the sets and descriptors of `room` ([`pool_info`]).
fn create_pool(device: &ash::Device, room: &PoolRoom) -> Result<vk::DescriptorPool, ExecError> {
    let sizes = pool_sizes(room);
    let info = pool_info(room, &sizes);
    // SAFETY: `info` and `sizes` outlive the call; the caller destroys the pool.
    unsafe { device.create_descriptor_pool(&info, None) }
        .map_err(ExecError::call("vkCreateDescriptorPool"))
}

/// A descriptor set of `pipeline`'s layout from `pool`, which `admitted` admitted into the
/// pool's room, with `buffers`, whole, at bindings 0, 1, ….
fn allocate_set(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    admitted: Admitted,
    pipeline: &Pipeline<'_>,
    buffers: &[&Buffer<'_>],
) -> Result<vk::DescriptorSet, ExecError> {
    if admitted.bindings() != buffers.len() {
        return Err(ExecError::Bindings {
            expected: pipeline.bindings,
            got: admitted.bindings(),
        });
    }
    let layouts = [pipeline.set_layout];
    let info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    // SAFETY: `pool` belongs to this device; a pool without room fails the call.
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
    // SAFETY: the set is new and not in use by any command buffer.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
    Ok(set)
}

/// A pool of timestamp queries, written by [`Recording::timestamp`] and read back once the
/// recordings that write it have been waited on ([`TimestampPool::read`]).
pub struct TimestampPool<'d> {
    device: &'d Device<'d>,
    pool: vk::QueryPool,
    count: u32,
}

impl Drop for TimestampPool<'_> {
    fn drop(&mut self) {
        // SAFETY: every recording that writes the pool borrows it, so none is pending here.
        unsafe { self.device.device.destroy_query_pool(self.pool, None) };
    }
}

impl<'d> Device<'d> {
    /// A pool of `count` timestamp queries, which a recording resets before it writes them
    /// ([`Recording::reset_timestamps`]).
    pub fn timestamp_pool(&'d self, count: u32) -> Result<TimestampPool<'d>, ExecError> {
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(count.max(1));
        // SAFETY: `info` outlives the call; the pool is destroyed by the `TimestampPool`'s drop.
        let pool = unsafe { self.device.create_query_pool(&info, None) }
            .map_err(ExecError::call("vkCreateQueryPool"))?;
        Ok(TimestampPool {
            device: self,
            pool,
            count,
        })
    }
}

impl TimestampPool<'_> {
    /// The queries of the pool.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// The timestamps of the pool's queries, in query order. The recordings that wrote them
    /// must have been waited on: this does not wait, and a query without its result fails.
    pub fn read(&self) -> Result<Vec<u64>, ExecError> {
        self.read_first(self.count)
    }

    /// The timestamps of the pool's first `count` queries (at most the pool's), in query order:
    /// those of a run that stopped before it wrote the rest. As [`TimestampPool::read`], it does
    /// not wait.
    pub fn read_first(&self, count: u32) -> Result<Vec<u64>, ExecError> {
        let count = count.min(self.count);
        let mut stamps = vec![0u64; count as usize];
        if count == 0 {
            return Ok(stamps);
        }
        // SAFETY: the pool belongs to this device and `stamps` holds one u64 per query read, at
        // most the pool's.
        unsafe {
            self.device.device.get_query_pool_results(
                self.pool,
                0,
                &mut stamps,
                vk::QueryResultFlags::TYPE_64,
            )
        }
        .map_err(ExecError::call("vkGetQueryPoolResults"))?;
        Ok(stamps)
    }
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
            counts: Counts::default(),
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
            earlier: Vec::new(),
        };
        self.queue_submit(submitted.commands.command, fence)?;
        self.count(submitted.commands.counts);
        submitted.waited = false;
        Ok(submitted)
    }
}

impl<'r> Recording<'r> {
    fn device(&self) -> &'r ash::Device {
        &self.commands.device.device
    }

    /// The dispatches recorded so far.
    pub fn dispatches(&self) -> u64 {
        self.commands.counts.dispatches
    }

    /// Copies all of `from` to the start of `to`.
    pub fn copy(&mut self, from: &'r Buffer<'_>, to: &'r Buffer<'_>) -> Result<(), ExecError> {
        self.copy_region(from, 0, to, 0, from.bytes())
    }

    /// Copies `size` bytes of `from` at `from_offset` to `to` at `to_offset`. A copy from a
    /// staging buffer to a device-local one counts as uploaded bytes, the reverse as downloaded.
    pub fn copy_region(
        &mut self,
        from: &'r Buffer<'_>,
        from_offset: u64,
        to: &'r Buffer<'_>,
        to_offset: u64,
        size: u64,
    ) -> Result<(), ExecError> {
        for (offset, buffer) in [(from_offset, from), (to_offset, to)] {
            if offset + size > buffer.bytes() {
                return Err(ExecError::TooShort {
                    needed: offset + size,
                    size: buffer.bytes(),
                });
            }
        }
        let regions = [vk::BufferCopy::default()
            .src_offset(from_offset)
            .dst_offset(to_offset)
            .size(size)];
        // SAFETY: both buffers belong to this device, the region lies inside both, and they
        // outlive the submission (`'r`).
        unsafe {
            self.device()
                .cmd_copy_buffer(self.commands.command, from.buffer, to.buffer, &regions)
        };
        match (from.is_staging(), to.is_staging()) {
            (true, false) => self.commands.counts.uploaded += size,
            (false, true) => self.commands.counts.downloaded += size,
            _ => {}
        }
        Ok(())
    }

    /// Records a dispatch bound once in a [`DescriptorArena`].
    pub fn dispatch_bound(&mut self, bound: &'r Bound<'_>) {
        self.record_dispatch(bound.pipeline, bound.set, bound.count, bound.workgroups);
    }

    fn record_dispatch(
        &mut self,
        pipeline: &Pipeline<'_>,
        set: vk::DescriptorSet,
        count: u32,
        workgroups: [u32; 3],
    ) {
        let device = self.device();
        let command = self.commands.command;
        let push = count.to_ne_bytes();
        debug_assert_eq!(push.len() as u32, PUSH_CONSTANT_BYTES);
        // SAFETY: `command` is recording; the set is written and its pool, the pipeline, its
        // layout and the buffers outlive the submission (`'r`).
        unsafe {
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
        self.commands.counts.dispatches += 1;
    }

    /// Resets every query of `pool`, before the recording writes any of them.
    pub fn reset_timestamps(&mut self, pool: &'r TimestampPool<'_>) {
        // SAFETY: `command` is recording and the pool outlives the submission (`'r`).
        unsafe {
            self.device()
                .cmd_reset_query_pool(self.commands.command, pool.pool, 0, pool.count)
        };
    }

    /// Writes a timestamp into query `query` of `pool`: once every command recorded before has
    /// started, at `end` false (top of pipe), or finished, at `end` true (bottom of pipe).
    pub fn timestamp(
        &mut self,
        pool: &'r TimestampPool<'_>,
        query: u32,
        end: bool,
    ) -> Result<(), ExecError> {
        if query >= pool.count {
            return Err(ExecError::QueryPastPool {
                query,
                count: pool.count,
            });
        }
        let stage = if end {
            vk::PipelineStageFlags::BOTTOM_OF_PIPE
        } else {
            vk::PipelineStageFlags::TOP_OF_PIPE
        };
        // SAFETY: `command` is recording, the query lies in the pool, and the pool outlives the
        // submission (`'r`).
        unsafe {
            self.device()
                .cmd_write_timestamp(self.commands.command, stage, pool.pool, query)
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
    /// buffer operands then each binding it writes ([`Kernel::written`]): the count, the
    /// elements each binding must hold and the invocations its workgroups must cover come from
    /// the kernel ([`kernel_bindings`]).
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
        check_dispatch(pipeline, buffers, needed, count, invocations, workgroups)?;
        let device = self.device();
        let room = PoolRoom::new(1, pipeline.bindings);
        // The pool is destroyed by `Commands`' drop.
        let pool = create_pool(device, &room)?;
        self.commands.descriptor_pools.push(pool);
        let admitted = admit(&room, buffers.len())?;
        let set = allocate_set(device, pool, admitted, pipeline, buffers)?;
        self.record_dispatch(pipeline, set, count, workgroups);
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
    /// The push constant: the output elements of an elementwise, broadcast, SGD, transpose,
    /// matmul or per-element kernel; the elements a reduction stage or a loss sum reads; the
    /// output rows of a per-row kernel and the output columns of a per-column one.
    pub count: u32,
    /// The elements each binding must hold, operands then each written binding
    /// ([`Kernel::written`]).
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

/// The invocations of a matmul kernel's tile ([`MatMulTile`]): its workgroups times their size.
fn matmul_invocations(kernel: &Kernel) -> Result<u64, ExecError> {
    let (m, k, n) =
        Kernel::matmul_shape(&kernel.operands).ok_or_else(|| ExecError::KernelShape {
            kernel: kernel.signature(),
            reason: "a matmul reads static [m, k] and [k, n]".to_owned(),
        })?;
    let tile = MatMulTile::of(m, k, n);
    Ok(tile.workgroups(m, n) * u64::from(tile.workgroup_size))
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
        | KernelKind::Transpose => (result, result),
        KernelKind::MatMul => (result, matmul_invocations(kernel)?),
        KernelKind::PerElement => (result, result),
        // One invocation per column of the result: its last dimension, which is all of it for
        // a column reduction and the row width of `SigregColumnsGrad`.
        KernelKind::PerColumn => {
            let columns = match kernel.result.shape.last() {
                Some(Dim::Static(columns)) => *columns,
                _ => {
                    return Err(ExecError::KernelShape {
                        kernel: kernel.signature(),
                        reason: "a per-column kernel writes static columns".to_owned(),
                    });
                }
            };
            (columns, columns)
        }
        KernelKind::PerRow => {
            let rows = match kernel.result.shape.first() {
                Some(Dim::Static(rows)) => *rows,
                _ => {
                    return Err(ExecError::KernelShape {
                        kernel: kernel.signature(),
                        reason: "a per-row kernel writes static rows".to_owned(),
                    });
                }
            };
            let rows = rows * kernel.heads().unwrap_or(1);
            (rows, rows)
        }
    };
    let elements = match kernel.kind {
        KernelKind::Elementwise | KernelKind::Broadcast | KernelKind::SgdUpdate => operands
            .iter()
            .map(|&(scalar, _)| if scalar { 1 } else { count })
            .chain([count])
            .collect(),
        _ => operands
            .iter()
            .map(|&(_, n)| n)
            .chain(std::iter::repeat_n(result, kernel.written()))
            .collect(),
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
