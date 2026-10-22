//! The ModelSpec runtime: executes compiled artifacts and trains them (DESIGN.md § 6).
//!
//! The CPU backend is always available; Vulkan backend code sits behind the `vulkan` feature:
//! device discovery and negotiation so far.

pub mod cpu;
pub mod train;

#[cfg(feature = "vulkan")]
pub mod vulkan;
