//! The ModelSpec compiler: `.mspec` source to verified tensor and training IR (DESIGN.md § 3).
//!
//! This crate builds and tests without Vulkan (DESIGN.md § 7).

pub mod artifact;
pub mod autodiff;
pub mod compile;
pub mod diag;
pub mod hir;
pub mod ir;
pub mod lower;
pub mod ops;
pub mod passes;
pub mod resolve;
pub mod schedule;
pub mod syntax;
pub mod types;
