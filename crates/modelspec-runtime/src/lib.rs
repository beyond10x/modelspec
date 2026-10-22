//! The ModelSpec runtime: executes compiled artifacts and trains them (DESIGN.md § 6).
//!
//! The CPU backend is always available; Vulkan backend code sits behind the `vulkan` feature:
//! device discovery and negotiation, device memory, compute pipelines and dispatch, the SPIR-V
//! emitter of every kernel of a compiled artifact, and the training runner `vulkan::runner`.

pub mod cpu;
pub mod train;

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "vulkan")]
pub mod spirv;
