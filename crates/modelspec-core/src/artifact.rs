//! The compiled artifact of a training step (DESIGN.md § 7.1; `spec/domains/artifact.yaml`).
//!
//! A [`CompiledArtifact`] holds the manifest hashes and the [`ExecutionPlan`]: parameter, data,
//! loss and prediction views, kernels, the step plan and the forward plan. A [`Backend::Cpu`]
//! artifact keeps the views and has no kernels and empty plans (DESIGN.md § 7.1). The serialized
//! artifact on disk is not built here.

use std::collections::BTreeMap;
use std::fmt;

use crate::ir::kernel::{BufferView, ExecutionPlan};
use crate::ir::print::print_step;
use crate::ir::tensor::{Dim, ParamDecl, ParamId, TensorType};
use crate::ir::training::TrainingStep;

/// The backend an artifact is compiled for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    Cpu,
    Vulkan,
}

/// A blake3 digest as 64 lowercase hex digits.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ContentHash(pub String);

impl ContentHash {
    fn of(hasher: &blake3::Hasher) -> Self {
        Self(hasher.finalize().to_hex().to_string())
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One parameter the plan reads and updates, by its stable path, with its views: one in M1 (read
/// at version n, written at n + 1), two for ping-pong (DESIGN.md § 6.3 item 3).
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterSpec {
    pub param: ParamId,
    pub path: String,
    /// [`canonical_parameter_id`] of the path.
    pub canonical_parameter_id: u64,
    pub ty: TensorType,
    pub views: Vec<BufferView>,
}

/// Whether a bound tensor is a training input or a target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataRole {
    Input,
    Target,
}

impl fmt::Display for DataRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DataRole::Input => "input",
            DataRole::Target => "target",
        })
    }
}

/// One `input` or `target` of the `train` block and the view a runner writes each batch into.
#[derive(Clone, Debug, PartialEq)]
pub struct DataSpec {
    pub name: String,
    pub role: DataRole,
    pub ty: TensorType,
    pub view: BufferView,
}

/// A training step compiled for a backend.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledArtifact {
    pub backend: Backend,
    /// blake3 of the source text.
    pub source_hash: ContentHash,
    /// blake3 of the printed training step ([`print_step`]); it covers the learning rate.
    pub ir_hash: ContentHash,
    /// [`schema_hash`] of the parameter table.
    pub schema_hash: ContentHash,
    /// Views, kernels and plans; a Cpu artifact has no kernels and empty plans.
    pub execution: ExecutionPlan,
}

impl CompiledArtifact {
    /// The artifact of `step`, compiled from `source`, with its verified `execution` plan.
    pub fn new(
        backend: Backend,
        source: &str,
        step: &TrainingStep,
        mut execution: ExecutionPlan,
    ) -> Self {
        if backend == Backend::Cpu {
            execution.kernels.clear();
            execution.step.clear();
            execution.forward.clear();
        }
        let mut source_hasher = blake3::Hasher::new();
        source_hasher.update(source.as_bytes());
        let mut ir_hasher = blake3::Hasher::new();
        ir_hasher.update(print_step(step).as_bytes());
        Self {
            backend,
            source_hash: ContentHash::of(&source_hasher),
            ir_hash: ContentHash::of(&ir_hasher),
            schema_hash: schema_hash(&step.training.program.params),
            execution,
        }
    }
}

/// blake3 over the parameter paths, dtypes and shapes in `ParamId` order; independent of the
/// backend and the learning rate (DESIGN.md § 6.2).
pub fn schema_hash(params: &BTreeMap<ParamId, ParamDecl>) -> ContentHash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"modelspec-schema/1\n");
    for param in params.values() {
        hasher.update(&(param.path.len() as u64).to_le_bytes());
        hasher.update(param.path.as_bytes());
        hasher.update(param.ty.dtype.to_string().as_bytes());
        hasher.update(&(param.ty.rank() as u64).to_le_bytes());
        for dim in &param.ty.shape {
            match dim {
                Dim::Static(n) => {
                    hasher.update(&[0]);
                    hasher.update(&n.to_le_bytes());
                }
                Dim::Symbol(s) => {
                    hasher.update(&[1]);
                    hasher.update(&u64::from(*s).to_le_bytes());
                }
            }
        }
    }
    ContentHash::of(&hasher)
}

/// The first 8 bytes of blake3 of `path`, little-endian (DESIGN.md § 2.3).
pub fn canonical_parameter_id(path: &str) -> u64 {
    let hash = blake3::hash(path.as_bytes());
    let mut first = [0u8; 8];
    first.copy_from_slice(&hash.as_bytes()[..8]);
    u64::from_le_bytes(first)
}
