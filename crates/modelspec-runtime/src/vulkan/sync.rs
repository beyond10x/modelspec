//! The memory barriers between the steps of a recording.
//!
//! Each barrier makes the writes of one kind of step visible to the next: an upload to a
//! dispatch, a dispatch to the next dispatch, a dispatch to a download, a copy to a later copy
//! that reads what it wrote, a download to the host.
//!
//! Every barrier is recorded by [`Barrier::record`], which counts it for the recording thread
//! ([`recorded_barriers`]).

use std::cell::Cell;

use ash::vk;

thread_local! {
    static RECORDED: Cell<u64> = const { Cell::new(0) };
}

/// The pipeline barriers this thread has recorded into any command buffer so far. The
/// difference of two readings around a stretch of recording on one thread is that stretch's.
pub fn recorded_barriers() -> u64 {
    RECORDED.get()
}

/// A global memory barrier between two kinds of step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Barrier {
    /// Copies written, before dispatches read or write.
    TransferToCompute,
    /// A dispatch wrote, before the next dispatch reads or writes.
    ComputeToCompute,
    /// Dispatches wrote, before copies read.
    ComputeToTransfer,
    /// Dispatches wrote, before copies overwrite what they wrote.
    ComputeToTransferWrite,
    /// Copies wrote, before later copies read or write.
    TransferToTransfer,
    /// Copies wrote, before the host reads after the fence.
    TransferToHost,
}

impl Barrier {
    /// The source and destination stages and access masks.
    pub fn scopes(
        self,
    ) -> (
        vk::PipelineStageFlags,
        vk::AccessFlags,
        vk::PipelineStageFlags,
        vk::AccessFlags,
    ) {
        use vk::{AccessFlags as A, PipelineStageFlags as S};
        match self {
            Self::TransferToCompute => (
                S::TRANSFER,
                A::TRANSFER_WRITE,
                S::COMPUTE_SHADER,
                A::SHADER_READ | A::SHADER_WRITE,
            ),
            Self::ComputeToCompute => (
                S::COMPUTE_SHADER,
                A::SHADER_WRITE,
                S::COMPUTE_SHADER,
                A::SHADER_READ | A::SHADER_WRITE,
            ),
            Self::ComputeToTransfer => (
                S::COMPUTE_SHADER,
                A::SHADER_WRITE,
                S::TRANSFER,
                A::TRANSFER_READ,
            ),
            Self::ComputeToTransferWrite => (
                S::COMPUTE_SHADER,
                A::SHADER_WRITE,
                S::TRANSFER,
                A::TRANSFER_WRITE,
            ),
            Self::TransferToTransfer => (
                S::TRANSFER,
                A::TRANSFER_WRITE,
                S::TRANSFER,
                A::TRANSFER_READ | A::TRANSFER_WRITE,
            ),
            Self::TransferToHost => (S::TRANSFER, A::TRANSFER_WRITE, S::HOST, A::HOST_READ),
        }
    }

    /// Records the barrier into `command` and counts it ([`recorded_barriers`]).
    pub(crate) fn record(self, device: &ash::Device, command: vk::CommandBuffer) {
        RECORDED.set(RECORDED.get() + 1);
        let (source_stage, source_access, target_stage, target_access) = self.scopes();
        let barriers = [vk::MemoryBarrier::default()
            .src_access_mask(source_access)
            .dst_access_mask(target_access)];
        // SAFETY: `command` is in the recording state, and `barriers` outlives the call.
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                source_stage,
                target_stage,
                vk::DependencyFlags::empty(),
                &barriers,
                &[],
                &[],
            )
        };
    }
}
