//! The compiled artifact of a training step (DESIGN.md § 7.1; `spec/domains/artifact.yaml`).
//!
//! A [`CompiledArtifact`] holds the manifest hashes and the [`ExecutionPlan`]: parameter, data,
//! loss and prediction views, kernels, the step plan and the forward plan. A [`Backend::Cpu`]
//! artifact keeps the views and has no kernels and empty plans (DESIGN.md § 7.1).
//!
//! On disk an artifact is a directory ([`write`], [`read`]): `artifact.json`, the manifest of
//! format version [`FORMAT_VERSION`], and `kernels/<kernel_module_id>.spv`, one SPIR-V module per
//! kernel of a Vulkan artifact; a Cpu artifact has no `kernels/`. The manifest holds the hashes,
//! the compile settings (`deterministic = true`, `fast_math = false`), the parameters, data,
//! buffers, loss and prediction views, the step plan and forward plan, and one entry per kernel:
//! its Kernel IR, entry point, cache key, byte length and the blake3 of its bytes.
//! `artifact_hash` is blake3 over the manifest's JSON with `artifact_hash` and `artifact_id`
//! empty, followed by the bytes of each module in id order; `artifact_id` is its first 16 bytes
//! marked as a version 8 UUID. The JSON is `serde_json`'s pretty form of the manifest, fields in
//! the order of `spec/domains/artifact.yaml`, so two compiles of one file give equal bytes.
//!
//! [`read`] validates an untrusted directory before anything is dispatched, in this order: the
//! manifest is a regular file, readable and of a known format version ([`E_ARTIFACT_001`]);
//! every module file is a regular file no longer than its entry says and `kernels/` holds no
//! other entry ([`E_ARTIFACT_004`]); each module's hash and then `artifact_hash` and
//! `artifact_id` ([`E_ARTIFACT_002`]); then every byte size computed from a shape or element
//! count fits in 64 bits, the invariants of `spec/domains/artifact.yaml`, every Dispatch names a
//! module the artifact owns, every module is SPIR-V, and the plan verifier ([`verify_plan`])
//! ([`E_ARTIFACT_003`]).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::Read as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::diag::{Code, Diagnostic};
use crate::ir::kernel::{
    BufferView, Epilogue, ExecutionPlan, Kernel, KernelKind, KernelModuleId, Operand, PlanOp,
    PlanStep,
};
use crate::ir::print::print_step;
use crate::ir::tensor::{DType, Dim, ParamDecl, ParamId, TensorType, ValueId};
use crate::ir::training::TrainingStep;
use crate::ir::verify::verify_plan;
use crate::schedule::MAX_BUFFER_BYTES;

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

/// The artifact directory cannot be read as an artifact: `artifact.json` is missing, not a
/// regular file, unreadable, larger than [`MAX_MANIFEST_BYTES`] or not a manifest, or its
/// `format_version` is not one this build reads ([`FORMAT_VERSION`]).
pub const E_ARTIFACT_001: Code = Code::new("E-ARTIFACT-001");

/// A kernel module's bytes differ from the blake3 its manifest entry records, or the
/// manifest's `artifact_hash` or `artifact_id` is not the one its content gives: the artifact
/// was changed after it was written.
pub const E_ARTIFACT_002: Code = Code::new("E-ARTIFACT-002");

/// The artifact is sealed but breaks an invariant of `spec/domains/artifact.yaml` or the plan
/// verifier: a byte size that does not fit in 64 bits, a buffer larger than 4294967295 bytes, a
/// kernel module that is not SPIR-V (whole little-endian words, a 5-word header, the magic number
/// 0x07230203), a Dispatch naming a module the artifact does not own, a dependency on a step that
/// is not before it, a view outside its buffer, or any other execution-plan rule.
pub const E_ARTIFACT_003: Code = Code::new("E-ARTIFACT-003");

/// A kernel module file the manifest names is missing, not a regular file (a link, a device or a
/// FIFO), cannot be read, or is longer than the byte length its entry records, or `kernels/`
/// holds an entry no kernel of the manifest names; every file is checked before any module is
/// hashed, and no read goes past the entry's byte length.
pub const E_ARTIFACT_004: Code = Code::new("E-ARTIFACT-004");

/// The manifest format this build writes and reads.
pub const FORMAT_VERSION: u64 = 1;

/// The manifest's file name in an artifact directory.
pub const MANIFEST_FILE: &str = "artifact.json";

/// The directory of an artifact's kernel modules, one `<kernel_module_id>.spv` each.
pub const KERNELS_DIR: &str = "kernels";

/// The largest `artifact.json` [`read`] parses.
pub const MAX_MANIFEST_BYTES: u64 = 64 << 20;

/// `modelspec.artifact.KernelCacheKey`: what a module was built under (DESIGN.md § 5.3).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelCacheKey {
    pub ir_hash: String,
    pub shapes: Vec<Vec<u64>>,
    #[serde(with = "dtype_name")]
    pub dtype: DType,
    pub fast_math: bool,
    pub deterministic: bool,
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver_version: u32,
    pub capabilities: Vec<String>,
}

/// `modelspec.artifact.KernelModule`: one kernel's SPIR-V as an artifact directory holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelModuleFile {
    pub kernel_module_id: KernelModuleId,
    pub entry_point: String,
    pub cache_key: KernelCacheKey,
    /// The module's words as little-endian bytes.
    pub spirv: Vec<u8>,
}

/// An artifact [`read`] accepted.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadedArtifact {
    pub artifact_id: Uuid,
    pub artifact_hash: ContentHash,
    pub compiler_version: String,
    pub fast_math: bool,
    pub deterministic: bool,
    pub artifact: CompiledArtifact,
    /// One module per kernel, in the artifact's kernel order; none for a Cpu artifact.
    pub modules: Vec<KernelModuleFile>,
}

/// Why [`write`] or [`seal`] did not finish.
#[derive(Debug, thiserror::Error)]
pub enum WriteError {
    #[error("{path}: {error}")]
    Io { path: String, error: std::io::Error },
    #[error("the artifact cannot be serialized: {0}")]
    Invalid(String),
}

fn io_error(path: &Path) -> impl FnOnce(std::io::Error) -> WriteError + '_ {
    move |error| WriteError::Io {
        path: path.display().to_string(),
        error,
    }
}

/// Writes `artifact` and its `modules` to `dir` (created with its parents) as format
/// [`FORMAT_VERSION`]: it removes the `artifact.json` and `kernels/` of an earlier artifact in
/// `dir`, writes the modules, then the sealed `artifact.json`, so `dir` holds exactly the new
/// artifact and an interrupted write leaves no manifest. A Vulkan artifact needs one module per
/// kernel; a Cpu artifact takes none and gets no `kernels/`. Returns the artifact hash.
pub fn write(
    dir: &Path,
    artifact: &CompiledArtifact,
    modules: &[KernelModuleFile],
) -> Result<ContentHash, WriteError> {
    let mut manifest = Manifest::of(artifact, modules)?;
    let mut bytes: Vec<(KernelModuleId, &[u8])> = modules
        .iter()
        .map(|m| (m.kernel_module_id, m.spirv.as_slice()))
        .collect();
    bytes.sort_by_key(|(id, _)| *id);
    manifest.seal(&bytes);
    std::fs::create_dir_all(dir).map_err(io_error(dir))?;
    let path = dir.join(MANIFEST_FILE);
    remove_if_present(&path)?;
    let kernels = dir.join(KERNELS_DIR);
    remove_if_present(&kernels)?;
    if !modules.is_empty() {
        std::fs::create_dir(&kernels).map_err(io_error(&kernels))?;
        for module in modules {
            let path = module_path(dir, module.kernel_module_id);
            std::fs::write(&path, &module.spirv).map_err(io_error(&path))?;
        }
    }
    std::fs::write(&path, manifest.to_json()).map_err(io_error(&path))?;
    Ok(ContentHash(manifest.artifact_hash))
}

/// Removes `path` and, when it is a directory and not a link to one, everything in it; a path
/// that does not exist is left as it is.
fn remove_if_present(path: &Path) -> Result<(), WriteError> {
    let removed = match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => Err(error),
        Ok(metadata) if metadata.is_dir() => std::fs::remove_dir_all(path),
        Ok(_) => std::fs::remove_file(path),
    };
    removed.map_err(io_error(path))
}

/// Recomputes `artifact_hash` and `artifact_id` of the manifest in `dir` from its content and
/// the module files it names, and rewrites it; nothing else changes. The hash detects a file
/// changed after it was written; it does not authenticate one (signing is not in M1).
pub fn seal(dir: &Path) -> Result<(), WriteError> {
    let path = dir.join(MANIFEST_FILE);
    let text = std::fs::read_to_string(&path).map_err(io_error(&path))?;
    let mut manifest: Manifest = serde_json::from_str(&text)
        .map_err(|e| WriteError::Invalid(format!("not an artifact manifest: {e}")))?;
    let mut modules = Vec::new();
    for kernel in &manifest.kernels {
        let path = module_path(dir, kernel.kernel_module_id);
        modules.push((
            kernel.kernel_module_id,
            std::fs::read(&path).map_err(io_error(&path))?,
        ));
    }
    modules.sort_by_key(|(id, _)| *id);
    let borrowed: Vec<_> = modules.iter().map(|(id, b)| (*id, b.as_slice())).collect();
    manifest.seal(&borrowed);
    std::fs::write(&path, manifest.to_json()).map_err(io_error(&path))
}

/// The file of module `id` in the artifact directory `dir`.
pub fn module_path(dir: &Path, id: KernelModuleId) -> PathBuf {
    dir.join(KERNELS_DIR).join(format!("{id}.spv"))
}

/// Reads and validates the artifact directory `dir` in the order of the module documentation,
/// and returns the artifact or the diagnostic of the first check it fails.
pub fn read(dir: &Path) -> Result<LoadedArtifact, Diagnostic> {
    let manifest = read_manifest(dir)?;
    let files = open_modules(dir, &manifest)?;
    let mut modules = Vec::new();
    for (kernel, file) in manifest.kernels.iter().zip(files) {
        let path = module_path(dir, kernel.kernel_module_id);
        let refused = |message: &str| {
            Diagnostic::new(E_ARTIFACT_004, message.to_owned())
                .with("module", path.display().to_string())
        };
        let mut spirv = Vec::new();
        file.take(kernel.byte_length.saturating_add(1))
            .read_to_end(&mut spirv)
            .map_err(|e| {
                refused("a kernel module file cannot be read").with("error", e.to_string())
            })?;
        if spirv.len() as u64 > kernel.byte_length {
            return Err(
                refused("a kernel module file is longer than its manifest entry")
                    .with("entry", kernel.byte_length.to_string())
                    .with("read", format!("more than {} bytes", kernel.byte_length)),
            );
        }
        let hash = blake3::hash(&spirv).to_hex().to_string();
        if hash != kernel.blake3 {
            return Err(Diagnostic::new(
                E_ARTIFACT_002,
                "a kernel module differs from the hash its manifest entry records",
            )
            .with("module", path.display().to_string())
            .with("recorded", kernel.blake3.clone())
            .with("found", hash));
        }
        modules.push((kernel.kernel_module_id, spirv));
    }
    let mut sorted: Vec<_> = modules.iter().map(|(id, b)| (*id, b.as_slice())).collect();
    sorted.sort_by_key(|(id, _)| *id);
    let hash = manifest.hash(&sorted);
    if hash.0 != manifest.artifact_hash {
        return Err(Diagnostic::new(
            E_ARTIFACT_002,
            "the artifact differs from the artifact_hash its manifest records",
        )
        .with("artifact", dir.join(MANIFEST_FILE).display().to_string())
        .with("recorded", manifest.artifact_hash.clone())
        .with("found", hash.0));
    }
    let id = artifact_id(&hash);
    if manifest.artifact_id != id.hyphenated().to_string() {
        return Err(Diagnostic::new(
            E_ARTIFACT_002,
            "the artifact_id is not the one its artifact_hash gives",
        )
        .with("recorded", manifest.artifact_id.clone())
        .with("found", id.hyphenated().to_string()));
    }
    let invalid = |message: String| {
        Diagnostic::new(E_ARTIFACT_003, "the artifact breaks an invariant")
            .with("artifact", dir.join(MANIFEST_FILE).display().to_string())
            .with("invariant", message)
    };
    let artifact = manifest.check().map_err(invalid)?;
    for (kernel, (_, spirv)) in manifest.kernels.iter().zip(&modules) {
        if let Err(why) = check_spirv(spirv) {
            return Err(invalid(format!(
                "the module of kernel {} is SPIR-V: {why}",
                kernel.kernel_module_id
            )));
        }
    }
    verify_plan(&artifact.execution).map_err(|e| invalid(e.to_string()))?;
    let modules = manifest
        .kernels
        .iter()
        .zip(modules)
        .map(|(kernel, (_, spirv))| KernelModuleFile {
            kernel_module_id: kernel.kernel_module_id,
            entry_point: kernel.entry_point.clone(),
            cache_key: kernel.cache_key.clone(),
            spirv,
        })
        .collect();
    Ok(LoadedArtifact {
        artifact_id: id,
        artifact_hash: hash,
        compiler_version: manifest.compiler_version,
        fast_math: manifest.fast_math,
        deterministic: manifest.deterministic,
        artifact,
        modules,
    })
}

/// The magic number, the first word of a SPIR-V module.
const SPIRV_MAGIC: u32 = 0x0723_0203;

/// The bytes of a SPIR-V module's header: magic, version, generator, bound and schema words.
const SPIRV_HEADER_BYTES: usize = 20;

/// `Ok` when `bytes` are little-endian SPIR-V words that start with a header and the magic
/// number; otherwise what they lack.
fn check_spirv(bytes: &[u8]) -> Result<(), String> {
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "{} bytes is not a whole number of words",
            bytes.len()
        ));
    }
    if bytes.len() < SPIRV_HEADER_BYTES {
        return Err(format!(
            "{} bytes is shorter than the {SPIRV_HEADER_BYTES}-byte header",
            bytes.len()
        ));
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != SPIRV_MAGIC {
        return Err(format!(
            "the first word is {magic:#010x}, not the magic number {SPIRV_MAGIC:#010x}"
        ));
    }
    Ok(())
}

/// Opens every module file the manifest of `dir` names, in manifest order, after the checks of
/// [`E_ARTIFACT_004`]: each is a regular file, not a link, a device or a FIFO, and no longer than
/// its entry, and `kernels/` holds nothing else (a manifest without kernels has no `kernels/`).
/// No file is read here, so none is hashed before every one has passed.
fn open_modules(dir: &Path, manifest: &Manifest) -> Result<Vec<std::fs::File>, Diagnostic> {
    let mut files = Vec::new();
    for kernel in &manifest.kernels {
        let path = module_path(dir, kernel.kernel_module_id);
        let refused = |message: &str| {
            Diagnostic::new(E_ARTIFACT_004, message.to_owned())
                .with("module", path.display().to_string())
        };
        let regular = |metadata: &std::fs::Metadata| -> Result<(), Diagnostic> {
            if !metadata.is_file() {
                return Err(refused("a kernel module file is not a regular file"));
            }
            if metadata.len() > kernel.byte_length {
                return Err(
                    refused("a kernel module file is longer than its manifest entry")
                        .with("entry", kernel.byte_length.to_string())
                        .with("file", metadata.len().to_string()),
                );
            }
            Ok(())
        };
        let metadata = std::fs::symlink_metadata(&path)
            .map_err(|e| refused("a kernel module file is missing").with("error", e.to_string()))?;
        regular(&metadata)?;
        let file = std::fs::File::open(&path).map_err(|e| {
            refused("a kernel module file cannot be read").with("error", e.to_string())
        })?;
        regular(&file.metadata().map_err(|e| {
            refused("a kernel module file cannot be read").with("error", e.to_string())
        })?)?;
        files.push(file);
    }
    let kernels = dir.join(KERNELS_DIR);
    let stray = |message: &str| {
        Diagnostic::new(E_ARTIFACT_004, message.to_owned())
            .with("kernels", kernels.display().to_string())
    };
    if manifest.kernels.is_empty() {
        return match std::fs::symlink_metadata(&kernels) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(files),
            _ => Err(stray(
                "the artifact has no kernel modules but has a kernels/ entry",
            )),
        };
    }
    let named: BTreeSet<String> = manifest
        .kernels
        .iter()
        .map(|kernel| format!("{}.spv", kernel.kernel_module_id))
        .collect();
    let entries = std::fs::read_dir(&kernels)
        .map_err(|e| stray("kernels/ cannot be listed").with("error", e.to_string()))?;
    for entry in entries {
        let entry =
            entry.map_err(|e| stray("kernels/ cannot be listed").with("error", e.to_string()))?;
        let name = entry.file_name();
        if !name.to_str().is_some_and(|name| named.contains(name)) {
            return Err(
                stray("kernels/ holds a file that no kernel entry of the manifest names")
                    .with("file", name.to_string_lossy().into_owned()),
            );
        }
    }
    Ok(files)
}

/// `artifact.json` of `dir`, of a known format version.
fn read_manifest(dir: &Path) -> Result<Manifest, Diagnostic> {
    let path = dir.join(MANIFEST_FILE);
    let unreadable = |message: &str| {
        Diagnostic::new(E_ARTIFACT_001, message.to_owned())
            .with("artifact", path.display().to_string())
    };
    let metadata = std::fs::symlink_metadata(&path).map_err(|e| {
        unreadable("the artifact manifest cannot be read").with("error", e.to_string())
    })?;
    if !metadata.is_file() {
        return Err(unreadable("the artifact manifest is not a regular file"));
    }
    let file = std::fs::File::open(&path).map_err(|e| {
        unreadable("the artifact manifest cannot be read").with("error", e.to_string())
    })?;
    let mut bytes = Vec::new();
    file.take(MAX_MANIFEST_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| {
            unreadable("the artifact manifest cannot be read").with("error", e.to_string())
        })?;
    if bytes.len() as u64 > MAX_MANIFEST_BYTES {
        return Err(unreadable("the artifact manifest is too large")
            .with("limit", format!("{MAX_MANIFEST_BYTES} bytes")));
    }
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
        unreadable("the artifact manifest is not JSON").with("error", e.to_string())
    })?;
    let version = value.get("format_version");
    if version.and_then(serde_json::Value::as_u64) != Some(FORMAT_VERSION) {
        let found = version.map_or_else(|| "none".to_owned(), ToString::to_string);
        return Err(unreadable("the artifact has an unknown format version")
            .with("format_version", found)
            .with("supported", FORMAT_VERSION.to_string()));
    }
    serde_json::from_value(value).map_err(|e| {
        unreadable("the artifact manifest is not of format version 1").with("error", e.to_string())
    })
}

/// The artifact id of `hash`: its first 16 bytes, marked as a version 8 UUID.
fn artifact_id(hash: &ContentHash) -> Uuid {
    let digest = blake3::Hash::from_hex(&hash.0).expect("a computed hash is 64 hex digits");
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest.as_bytes()[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x80;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid::from_bytes(bytes)
}

/// `f32`, the dtype names of `spec/domains/checkpoint.yaml`.
mod dtype_name {
    use serde::{Deserialize, Deserializer, Serializer};

    use crate::ir::tensor::DType;

    pub fn serialize<S: Serializer>(dtype: &DType, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&dtype.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<DType, D::Error> {
        match String::deserialize(d)?.as_str() {
            "f32" => Ok(DType::F32),
            other => Err(serde::de::Error::custom(format!("unknown dtype `{other}`"))),
        }
    }
}

/// A kernel module id as its hyphenated UUID.
mod module_id {
    use serde::{Deserialize, Deserializer, Serializer};
    use uuid::Uuid;

    use crate::ir::kernel::KernelModuleId;

    pub fn serialize<S: Serializer>(id: &KernelModuleId, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&id.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<KernelModuleId, D::Error> {
        let text = String::deserialize(d)?;
        let uuid = Uuid::try_parse(&text).map_err(serde::de::Error::custom)?;
        if uuid.hyphenated().to_string() != text {
            return Err(serde::de::Error::custom(format!(
                "`{text}` is not a lowercase hyphenated UUID"
            )));
        }
        Ok(KernelModuleId(uuid.into_bytes()))
    }

    pub mod optional {
        use serde::{Deserialize, Deserializer, Serialize, Serializer};

        use crate::ir::kernel::KernelModuleId;

        #[derive(Serialize, Deserialize)]
        struct Wrapped(#[serde(with = "super")] KernelModuleId);

        pub fn serialize<S: Serializer>(
            id: &Option<KernelModuleId>,
            s: S,
        ) -> Result<S::Ok, S::Error> {
            id.map(Wrapped).serialize(s)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(
            d: D,
        ) -> Result<Option<KernelModuleId>, D::Error> {
            Ok(Option::<Wrapped>::deserialize(d)?.map(|Wrapped(id)| id))
        }
    }
}

/// `modelspec.artifact.Backend`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum WireBackend {
    Cpu,
    Vulkan,
}

/// `modelspec.artifact.BufferView`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireView {
    buffer: u32,
    #[serde(with = "dtype_name")]
    dtype: DType,
    element_count: u64,
    offset: u64,
    size_bytes: u64,
    alignment: u64,
}

impl From<&BufferView> for WireView {
    fn from(view: &BufferView) -> Self {
        Self {
            buffer: view.buffer,
            dtype: view.dtype,
            element_count: view.element_count,
            offset: view.offset,
            size_bytes: view.size_bytes,
            alignment: view.alignment,
        }
    }
}

impl From<&WireView> for BufferView {
    fn from(view: &WireView) -> Self {
        Self {
            buffer: view.buffer,
            dtype: view.dtype,
            element_count: view.element_count,
            offset: view.offset,
            size_bytes: view.size_bytes,
            alignment: view.alignment,
        }
    }
}

/// A tensor type: its dtype and static shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireType {
    #[serde(with = "dtype_name")]
    dtype: DType,
    shape: Vec<u64>,
}

/// The static shape of `ty`; a verified program has no symbolic dimension.
fn static_shape(ty: &TensorType) -> Result<Vec<u64>, WriteError> {
    ty.shape
        .iter()
        .map(|dim| match dim {
            Dim::Static(n) => Ok(*n),
            Dim::Symbol(_) => Err(WriteError::Invalid(format!(
                "{ty} has a symbolic dimension"
            ))),
        })
        .collect()
}

impl WireType {
    fn of(ty: &TensorType) -> Result<Self, WriteError> {
        Ok(Self {
            dtype: ty.dtype,
            shape: static_shape(ty)?,
        })
    }

    fn ty(&self) -> TensorType {
        TensorType {
            dtype: self.dtype,
            shape: self.shape.iter().map(|&n| Dim::Static(n)).collect(),
        }
    }

    /// The type's bytes, or `None` when they do not fit in 64 bits.
    fn byte_size(&self) -> Option<u64> {
        self.shape
            .iter()
            .try_fold(self.dtype.size_bytes(), |acc, &n| acc.checked_mul(n))
    }
}

/// `modelspec.artifact.ParameterSpec`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireParameter {
    param: u32,
    path: String,
    canonical_parameter_id: u64,
    #[serde(with = "dtype_name")]
    dtype: DType,
    shape: Vec<u64>,
    views: Vec<WireView>,
}

/// `modelspec.artifact.DataRole`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum WireRole {
    Input,
    Target,
}

/// `modelspec.artifact.DataSpec`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireData {
    name: String,
    role: WireRole,
    #[serde(with = "dtype_name")]
    dtype: DType,
    shape: Vec<u64>,
    view: WireView,
}

/// `modelspec.artifact.PlanOp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum WireOp {
    Allocate,
    Copy,
    Dispatch,
    Barrier,
    Complete,
}

/// `modelspec.artifact.PlanStep`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireStep {
    op: WireOp,
    #[serde(with = "module_id::optional")]
    kernel_module_id: Option<KernelModuleId>,
    workgroups: Option<[u32; 3]>,
    reads: Vec<WireView>,
    writes: Vec<WireView>,
    depends_on: Vec<usize>,
    value: Option<u32>,
}

/// [`KernelKind`], by the name the Kernel IR printer shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum WireKind {
    #[serde(rename = "elementwise")]
    Elementwise,
    #[serde(rename = "broadcast")]
    Broadcast,
    #[serde(rename = "reduce_stage_1")]
    ReduceStage1,
    #[serde(rename = "reduce_stage_2")]
    ReduceStage2,
    #[serde(rename = "loss_sum")]
    LossSum,
    #[serde(rename = "transpose")]
    Transpose,
    #[serde(rename = "matmul")]
    MatMul,
    #[serde(rename = "sgd_update")]
    SgdUpdate,
}

const KINDS: [(WireKind, KernelKind); 8] = [
    (WireKind::Elementwise, KernelKind::Elementwise),
    (WireKind::Broadcast, KernelKind::Broadcast),
    (WireKind::ReduceStage1, KernelKind::ReduceStage1),
    (WireKind::ReduceStage2, KernelKind::ReduceStage2),
    (WireKind::LossSum, KernelKind::LossSum),
    (WireKind::Transpose, KernelKind::Transpose),
    (WireKind::MatMul, KernelKind::MatMul),
    (WireKind::SgdUpdate, KernelKind::SgdUpdate),
];

/// The Tensor IR op names a kernel can implement (`TensorOp::name`).
const OPS: [&str; 14] = [
    "Input",
    "ParamRead",
    "Scalar",
    "Add",
    "Sub",
    "Mul",
    "MatMul",
    "Relu",
    "Transpose",
    "ReduceSum",
    "ReduceMean",
    "Mse",
    "ReluGrad",
    "BroadcastScalar",
];

/// A kernel operand: a buffer of a type, or an `f32` constant by its bits, so that it reads back
/// exactly.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum WireOperand {
    Buffer(WireType),
    ImmediateF32Bits(u32),
}

/// [`Epilogue`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WireEpilogue {
    Relu,
    ReluWritesPreActivation,
}

/// One kernel of the artifact: its module's entry and its Kernel IR.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireKernel {
    #[serde(with = "module_id")]
    kernel_module_id: KernelModuleId,
    entry_point: String,
    cache_key: KernelCacheKey,
    byte_length: u64,
    blake3: String,
    kind: WireKind,
    op: String,
    #[serde(with = "dtype_name")]
    dtype: DType,
    operands: Vec<WireOperand>,
    result: WireType,
    reduces: Option<WireType>,
    epilogue: Option<WireEpilogue>,
    workgroup_size: u32,
    workgroups: [u32; 3],
}

/// `artifact.json`: `modelspec.artifact.CompiledArtifact` and its kernel modules' entries.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format_version: u64,
    artifact_id: String,
    backend: WireBackend,
    artifact_hash: String,
    source_hash: String,
    ir_hash: String,
    fast_math: bool,
    deterministic: bool,
    compiler_version: String,
    schema_hash: String,
    parameters: Vec<WireParameter>,
    data: Vec<WireData>,
    kernels: Vec<WireKernel>,
    plan: Vec<WireStep>,
    forward_plan: Vec<WireStep>,
    buffers: Vec<u64>,
    loss_view: WireView,
    prediction_view: WireView,
}

fn wire_steps(steps: &[PlanStep]) -> Vec<WireStep> {
    steps
        .iter()
        .map(|step| WireStep {
            op: match step.op {
                PlanOp::Allocate => WireOp::Allocate,
                PlanOp::Copy => WireOp::Copy,
                PlanOp::Dispatch => WireOp::Dispatch,
                PlanOp::Barrier => WireOp::Barrier,
                PlanOp::Complete => WireOp::Complete,
            },
            kernel_module_id: step.kernel_module_id,
            workgroups: step.workgroups,
            reads: step.reads.iter().map(WireView::from).collect(),
            writes: step.writes.iter().map(WireView::from).collect(),
            depends_on: step.depends_on.clone(),
            value: step.value.map(|v| v.0),
        })
        .collect()
}

fn plan_steps(steps: &[WireStep]) -> Vec<PlanStep> {
    steps
        .iter()
        .map(|step| PlanStep {
            op: match step.op {
                WireOp::Allocate => PlanOp::Allocate,
                WireOp::Copy => PlanOp::Copy,
                WireOp::Dispatch => PlanOp::Dispatch,
                WireOp::Barrier => PlanOp::Barrier,
                WireOp::Complete => PlanOp::Complete,
            },
            kernel_module_id: step.kernel_module_id,
            workgroups: step.workgroups,
            reads: step.reads.iter().map(BufferView::from).collect(),
            writes: step.writes.iter().map(BufferView::from).collect(),
            depends_on: step.depends_on.clone(),
            value: step.value.map(ValueId),
        })
        .collect()
}

impl WireKernel {
    fn of(kernel: &Kernel, module: &KernelModuleFile) -> Result<Self, WriteError> {
        let kind = KINDS
            .iter()
            .find(|(_, k)| *k == kernel.kind)
            .map(|(w, _)| *w)
            .expect("every kernel kind has a name");
        Ok(Self {
            kernel_module_id: kernel.id,
            entry_point: module.entry_point.clone(),
            cache_key: module.cache_key.clone(),
            byte_length: module.spirv.len() as u64,
            blake3: blake3::hash(&module.spirv).to_hex().to_string(),
            kind,
            op: kernel.op.to_owned(),
            dtype: kernel.dtype,
            operands: kernel
                .operands
                .iter()
                .map(|operand| match operand {
                    Operand::Buffer(ty) => WireType::of(ty).map(WireOperand::Buffer),
                    Operand::Immediate(value) => Ok(WireOperand::ImmediateF32Bits(value.to_bits())),
                })
                .collect::<Result<_, _>>()?,
            result: WireType::of(&kernel.result)?,
            reduces: kernel.reduces.as_ref().map(WireType::of).transpose()?,
            epilogue: kernel.epilogue.map(|epilogue| match epilogue {
                Epilogue::Relu {
                    writes_pre_activation: false,
                } => WireEpilogue::Relu,
                Epilogue::Relu {
                    writes_pre_activation: true,
                } => WireEpilogue::ReluWritesPreActivation,
            }),
            workgroup_size: kernel.workgroup_size,
            workgroups: kernel.workgroups,
        })
    }

    /// The Kernel IR of the entry, or the invariant it breaks.
    fn kernel(&self) -> Result<Kernel, String> {
        let id = self.kernel_module_id;
        let op = OPS
            .iter()
            .find(|op| **op == self.op)
            .ok_or_else(|| format!("kernel {id} names no Tensor IR op (`{}`)", self.op))?;
        let kernel = Kernel {
            id,
            kind: KINDS
                .iter()
                .find(|(w, _)| *w == self.kind)
                .map(|(_, k)| *k)
                .expect("every kind name is a kernel kind"),
            op,
            dtype: self.dtype,
            operands: self
                .operands
                .iter()
                .map(|operand| match operand {
                    WireOperand::Buffer(ty) => Operand::Buffer(ty.ty()),
                    WireOperand::ImmediateF32Bits(bits) => {
                        Operand::Immediate(f32::from_bits(*bits))
                    }
                })
                .collect(),
            result: self.result.ty(),
            reduces: self.reduces.as_ref().map(WireType::ty),
            epilogue: self.epilogue.map(|epilogue| Epilogue::Relu {
                writes_pre_activation: epilogue == WireEpilogue::ReluWritesPreActivation,
            }),
            workgroup_size: self.workgroup_size,
            workgroups: self.workgroups,
        };
        let derived = Kernel::module_id(&kernel.signature());
        if derived != id {
            return Err(format!(
                "kernel {id} is not the id its content gives ({derived})"
            ));
        }
        Ok(kernel)
    }
}

impl Manifest {
    /// The unsealed manifest of `artifact` with `modules`.
    fn of(artifact: &CompiledArtifact, modules: &[KernelModuleFile]) -> Result<Self, WriteError> {
        let execution = &artifact.execution;
        let kernels = match artifact.backend {
            Backend::Cpu if !modules.is_empty() => {
                return Err(WriteError::Invalid(
                    "a Cpu artifact has no kernel modules".to_owned(),
                ));
            }
            Backend::Cpu => Vec::new(),
            Backend::Vulkan => {
                if modules.len() != execution.kernels.len() {
                    return Err(WriteError::Invalid(format!(
                        "{} kernels and {} modules",
                        execution.kernels.len(),
                        modules.len()
                    )));
                }
                execution
                    .kernels
                    .iter()
                    .map(|kernel| {
                        let module = modules
                            .iter()
                            .find(|m| m.kernel_module_id == kernel.id)
                            .ok_or_else(|| {
                                WriteError::Invalid(format!("kernel {} has no module", kernel.id))
                            })?;
                        WireKernel::of(kernel, module)
                    })
                    .collect::<Result<_, _>>()?
            }
        };
        Ok(Self {
            format_version: FORMAT_VERSION,
            artifact_id: String::new(),
            backend: match artifact.backend {
                Backend::Cpu => WireBackend::Cpu,
                Backend::Vulkan => WireBackend::Vulkan,
            },
            artifact_hash: String::new(),
            source_hash: artifact.source_hash.0.clone(),
            ir_hash: artifact.ir_hash.0.clone(),
            fast_math: false,
            deterministic: true,
            compiler_version: env!("CARGO_PKG_VERSION").to_owned(),
            schema_hash: artifact.schema_hash.0.clone(),
            parameters: execution
                .parameters
                .iter()
                .map(|p| {
                    Ok(WireParameter {
                        param: p.param.0,
                        path: p.path.clone(),
                        canonical_parameter_id: p.canonical_parameter_id,
                        dtype: p.ty.dtype,
                        shape: static_shape(&p.ty)?,
                        views: p.views.iter().map(WireView::from).collect(),
                    })
                })
                .collect::<Result<_, WriteError>>()?,
            data: execution
                .data
                .iter()
                .map(|d| {
                    Ok(WireData {
                        name: d.name.clone(),
                        role: match d.role {
                            DataRole::Input => WireRole::Input,
                            DataRole::Target => WireRole::Target,
                        },
                        dtype: d.ty.dtype,
                        shape: static_shape(&d.ty)?,
                        view: WireView::from(&d.view),
                    })
                })
                .collect::<Result<_, WriteError>>()?,
            kernels,
            plan: wire_steps(&execution.step),
            forward_plan: wire_steps(&execution.forward),
            buffers: execution.buffers.clone(),
            loss_view: WireView::from(&execution.loss_view),
            prediction_view: WireView::from(&execution.prediction_view),
        })
    }

    /// The manifest's file content: pretty JSON ending in a newline.
    fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a manifest serializes") + "\n"
    }

    /// blake3 over the JSON of the manifest with `artifact_hash` and `artifact_id` empty, then
    /// over each module's bytes; `modules` are in id order.
    fn hash(&self, modules: &[(KernelModuleId, &[u8])]) -> ContentHash {
        let mut blank = self.clone();
        blank.artifact_hash.clear();
        blank.artifact_id.clear();
        let mut hasher = blake3::Hasher::new();
        hasher.update(blank.to_json().as_bytes());
        for (_, bytes) in modules {
            hasher.update(bytes);
        }
        ContentHash::of(&hasher)
    }

    /// Sets `artifact_hash` and `artifact_id` from the content.
    fn seal(&mut self, modules: &[(KernelModuleId, &[u8])]) {
        let hash = self.hash(modules);
        self.artifact_id = artifact_id(&hash).hyphenated().to_string();
        self.artifact_hash = hash.0;
    }

    /// The artifact, after the byte-size, invariant and module-ownership checks of
    /// [`E_ARTIFACT_003`] that come before the plan verifier.
    fn check(&self) -> Result<CompiledArtifact, String> {
        self.check_byte_sizes()?;
        self.check_invariants()?;
        let kernels = self
            .kernels
            .iter()
            .map(WireKernel::kernel)
            .collect::<Result<Vec<_>, _>>()?;
        let owned: BTreeSet<KernelModuleId> = kernels.iter().map(|k| k.id).collect();
        for (name, steps) in [("plan", &self.plan), ("forward_plan", &self.forward_plan)] {
            for (position, step) in steps.iter().enumerate() {
                if let Some(id) = step.kernel_module_id
                    && !owned.contains(&id)
                {
                    return Err(format!(
                        "{name} step {position} dispatches module {id}, which the artifact does \
                         not own"
                    ));
                }
            }
        }
        let parameters = self
            .parameters
            .iter()
            .map(|p| ParameterSpec {
                param: ParamId(p.param),
                path: p.path.clone(),
                canonical_parameter_id: p.canonical_parameter_id,
                ty: WireType {
                    dtype: p.dtype,
                    shape: p.shape.clone(),
                }
                .ty(),
                views: p.views.iter().map(BufferView::from).collect(),
            })
            .collect();
        let data = self
            .data
            .iter()
            .map(|d| DataSpec {
                name: d.name.clone(),
                role: match d.role {
                    WireRole::Input => DataRole::Input,
                    WireRole::Target => DataRole::Target,
                },
                ty: WireType {
                    dtype: d.dtype,
                    shape: d.shape.clone(),
                }
                .ty(),
                view: BufferView::from(&d.view),
            })
            .collect();
        Ok(CompiledArtifact {
            backend: match self.backend {
                WireBackend::Cpu => Backend::Cpu,
                WireBackend::Vulkan => Backend::Vulkan,
            },
            source_hash: ContentHash(self.source_hash.clone()),
            ir_hash: ContentHash(self.ir_hash.clone()),
            schema_hash: ContentHash(self.schema_hash.clone()),
            execution: ExecutionPlan {
                buffers: self.buffers.clone(),
                parameters,
                data,
                kernels,
                step: plan_steps(&self.plan),
                forward: plan_steps(&self.forward_plan),
                loss_view: BufferView::from(&self.loss_view),
                prediction_view: BufferView::from(&self.prediction_view),
            },
        })
    }

    /// Every byte size computed from a shape or an element count fits in 64 bits.
    fn check_byte_sizes(&self) -> Result<(), String> {
        let mut types: Vec<(String, WireType)> = Vec::new();
        for p in &self.parameters {
            types.push((
                format!("parameter `{}`", p.path),
                WireType {
                    dtype: p.dtype,
                    shape: p.shape.clone(),
                },
            ));
        }
        for d in &self.data {
            types.push((
                format!("data `{}`", d.name),
                WireType {
                    dtype: d.dtype,
                    shape: d.shape.clone(),
                },
            ));
        }
        for k in &self.kernels {
            let id = k.kernel_module_id;
            for operand in &k.operands {
                if let WireOperand::Buffer(ty) = operand {
                    types.push((format!("an operand of kernel {id}"), ty.clone()));
                }
            }
            types.push((format!("the result of kernel {id}"), k.result.clone()));
            if let Some(reduces) = &k.reduces {
                types.push((
                    format!("the reduced tensor of kernel {id}"),
                    reduces.clone(),
                ));
            }
        }
        if let Some((name, _)) = types.iter().find(|(_, ty)| ty.byte_size().is_none()) {
            return Err(format!("the byte size of {name} does not fit in 64 bits"));
        }
        let views = self
            .parameters
            .iter()
            .flat_map(|p| &p.views)
            .chain(self.data.iter().map(|d| &d.view))
            .chain([&self.loss_view, &self.prediction_view])
            .chain(
                self.plan
                    .iter()
                    .chain(&self.forward_plan)
                    .flat_map(|s| s.reads.iter().chain(&s.writes)),
            );
        for view in views {
            if view
                .element_count
                .checked_mul(view.dtype.size_bytes())
                .is_none()
            {
                return Err(format!(
                    "the byte size of a view of buffer b{} with {} elements does not fit in 64 \
                     bits",
                    view.buffer, view.element_count
                ));
            }
        }
        Ok(())
    }

    /// The invariants of `spec/domains/artifact.yaml` the plan verifier does not check.
    fn check_invariants(&self) -> Result<(), String> {
        for (name, hash) in [
            ("source_hash", &self.source_hash),
            ("ir_hash", &self.ir_hash),
            ("schema_hash", &self.schema_hash),
        ] {
            if hash.len() != 64 || !hash.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
                return Err(format!("{name} is 64 lowercase hex digits"));
            }
        }
        if let Some(position) = self.buffers.iter().position(|&size| size == 0) {
            return Err(format!("buffers[{position}] > 0"));
        }
        if let Some(position) = self
            .buffers
            .iter()
            .position(|&size| size > MAX_BUFFER_BYTES)
        {
            return Err(format!("buffers[{position}] <= {MAX_BUFFER_BYTES}"));
        }
        let mut paths = BTreeSet::new();
        let mut canonical = BTreeSet::new();
        for p in &self.parameters {
            if !paths.insert(&p.path) {
                return Err(format!("distinct parameter paths (`{}` repeats)", p.path));
            }
            if !canonical.insert(p.canonical_parameter_id) {
                return Err(format!(
                    "distinct canonical_parameter_id ({} repeats)",
                    p.canonical_parameter_id
                ));
            }
            if p.canonical_parameter_id != canonical_parameter_id(&p.path) {
                return Err(format!(
                    "canonical_parameter_id of `{}` is the first 8 bytes of blake3 of its path",
                    p.path
                ));
            }
        }
        let mut names = BTreeSet::new();
        if let Some(d) = self.data.iter().find(|d| !names.insert(&d.name)) {
            return Err(format!("distinct data names (`{}` repeats)", d.name));
        }
        for (name, steps) in [("plan", &self.plan), ("forward_plan", &self.forward_plan)] {
            for (position, step) in steps.iter().enumerate() {
                let mut seen = BTreeSet::new();
                if let Some(on) = step.depends_on.iter().find(|on| !seen.insert(**on)) {
                    return Err(format!(
                        "{name} step {position}: distinct depends_on ({on} repeats)"
                    ));
                }
                if self.backend == WireBackend::Cpu && step.op == WireOp::Dispatch {
                    return Err(format!(
                        "{name} step {position}: a Cpu artifact has no Dispatch step"
                    ));
                }
            }
        }
        if self.backend == WireBackend::Cpu && !self.kernels.is_empty() {
            return Err("a Cpu artifact has no kernels".to_owned());
        }
        let mut ids = BTreeSet::new();
        for k in &self.kernels {
            if !ids.insert(k.kernel_module_id) {
                return Err(format!(
                    "distinct kernel_module_id ({} repeats)",
                    k.kernel_module_id
                ));
            }
            if k.entry_point.is_empty() {
                return Err(format!("kernel {} has an entry point", k.kernel_module_id));
            }
            let mut capabilities = BTreeSet::new();
            if let Some(c) = k
                .cache_key
                .capabilities
                .iter()
                .find(|c| !capabilities.insert(c.as_str()))
            {
                return Err(format!(
                    "kernel {}: distinct cache_key.capabilities (`{c}` repeats)",
                    k.kernel_module_id
                ));
            }
        }
        Ok(())
    }
}
