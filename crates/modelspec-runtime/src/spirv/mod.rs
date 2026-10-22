//! SPIR-V for the Vulkan backend (DESIGN.md § 5.1, § 5.2), behind the `vulkan` feature.
//!
//! [`emit`] builds one compute module per Kernel IR kernel with `rspirv`, with no GLSL or HLSL
//! compiler on the path; [`validate`] runs the external `spirv-val` on a module (DESIGN.md § 9
//! gate 6).

pub mod emit;
pub mod validate;
