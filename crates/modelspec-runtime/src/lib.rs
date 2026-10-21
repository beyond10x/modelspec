//! The ModelSpec runtime: executes compiled artifacts and trains them (DESIGN.md § 6).
//!
//! The CPU backend is always available; Vulkan backend code sits behind the `vulkan` feature,
//! which is empty in M0.

pub mod cpu;
pub mod train;
