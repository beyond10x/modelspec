//! The ModelSpec runtime: executes compiled artifacts and trains them (DESIGN.md § 6).
//!
//! The CPU backend is always available; Vulkan backend code sits behind the `vulkan` feature:
//! device discovery and negotiation, device memory, compute pipelines and dispatch, and the
//! SPIR-V emitter of the elementwise, broadcast and SGD kernels.

pub mod cpu;
pub mod train;

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "vulkan")]
pub mod spirv;
