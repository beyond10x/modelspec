//! The `.msckpt` checkpoint: writer and validating reader (DESIGN.md § 6.2;
//! `spec/domains/checkpoint.yaml`, `Checkpoint` and `ParameterRecord`).
//!
//! Format version 1 is a 56-byte little-endian header (magic `MSCK`, version, endian marker,
//! a reserved zero word, the manifest size and a blake3 checksum over header and manifest with the
//! checksum bytes counted as zero), a compact JSON manifest, and the payload: each parameter's
//! little-endian `f32` values at a multiple of [`ALIGNMENT`], in path order, zero bytes between,
//! the file ending with the last tensor. The manifest's `payload_hash` is blake3 over the bytes
//! from the payload start to the end of the file. `checkpoint_id` is a version-8 UUID from blake3
//! over the whole manifest with the nil UUID in its place.
//!
//! [`read`] checks, before it allocates a tensor: the header fields ([`E_CKPT_001`]), the manifest
//! size against the file ([`E_CKPT_002`]; it comes before the checksum, which covers the manifest
//! and so needs its size bounded first), the checksum ([`E_CKPT_001`]), the number of records
//! against the file size ([`E_CKPT_002`]), the manifest's JSON, with a UUID, a payload hash and at
//! most [`MAX_RANK`] dimensions per shape, and its invariants ([`E_CKPT_001`]), `file_size`,
//! `alignment`, every offset and length, the end of the last tensor and the zero bytes between
//! tensors ([`E_CKPT_002`]), and the payload hash ([`E_CKPT_004`]). No single allocation it makes
//! is larger than the file plus 4,096 bytes. [`Checkpoint::parameters_for`] and
//! [`Checkpoint::resume_into`] refuse a checkpoint that does not match a compiled file with
//! [`E_CKPT_003`], naming the first difference.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use modelspec_core::artifact::{ContentHash, schema_hash};
use modelspec_core::compile::CompiledTraining;
use modelspec_core::diag::{Code, Diagnostic};
use modelspec_core::ir::tensor::{DType, Dim, ParamId, TensorType};
use serde::de::{DeserializeSeed, Deserializer, IgnoredAny, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use thiserror::Error;

use crate::cpu::HostTensor;

/// Not a readable checkpoint: it cannot be opened, or its magic, format version, endian marker,
/// reserved field, checksum or manifest is wrong.
pub const E_CKPT_001: Code = Code::new("E-CKPT-001");
/// An offset, length or size outside the file or misaligned: the file is shorter than its header
/// or than `file_size`, the manifest size runs past the end, or a tensor is misaligned, overlaps
/// another part of the file, ends past it, or is not its shape's element count times its dtype
/// size, a byte between tensors is not zero, the file does not end at its last tensor, or the
/// manifest has more records than a file of its size can hold.
pub const E_CKPT_002: Code = Code::new("E-CKPT-002");
/// The checkpoint does not match the compiled file: a parameter path, dtype or shape, the schema
/// hash, or for `train --resume` the optimizer differs; the first difference is named.
pub const E_CKPT_003: Code = Code::new("E-CKPT-003");
/// The payload hash differs from the payload.
pub const E_CKPT_004: Code = Code::new("E-CKPT-004");

/// The four magic bytes.
pub const MAGIC: [u8; 4] = *b"MSCK";
/// The format version this module writes and reads.
pub const FORMAT_VERSION: u32 = 1;
/// The endian marker, written little-endian: the bytes `04 03 02 01`.
pub const ENDIAN_MARKER: u32 = 0x0102_0304;
/// The bytes of the header.
pub const HEADER_LEN: usize = 56;
/// The alignment of every tensor offset, in bytes.
pub const ALIGNMENT: u64 = 64;

const CHECKSUM: std::ops::Range<usize> = 24..56;

/// The optimizer of a checkpoint (`modelspec.training.OptimizerKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerKind {
    Sgd,
    AdamW,
}

/// A learning-rate schedule (`modelspec.training.LearningRateSchedule`); M1 writes none.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearningRateSchedule {
    pub warmup_steps: u64,
}

/// Optimizer hyperparameters (`modelspec.training.OptimizerConfig`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizerConfig {
    pub kind: OptimizerKind,
    pub learning_rate: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_decay: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beta1: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beta2: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epsilon: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schedule: Option<LearningRateSchedule>,
}

impl OptimizerConfig {
    /// SGD at the learning rate of `compiled`, with no schedule.
    pub fn of(compiled: &CompiledTraining) -> Self {
        Self {
            kind: OptimizerKind::Sgd,
            learning_rate: f64::from(compiled.config.lr),
            weight_decay: None,
            beta1: None,
            beta2: None,
            epsilon: None,
            schedule: None,
        }
    }

    /// The invariant of `OptimizerConfig`: the AdamW fields are present exactly for AdamW.
    fn is_valid(&self) -> bool {
        let adamw = [self.weight_decay, self.beta1, self.beta2, self.epsilon];
        self.learning_rate.is_finite()
            && match self.kind {
                OptimizerKind::Sgd => adamw.iter().all(Option::is_none),
                OptimizerKind::AdamW => adamw.iter().all(Option::is_some),
            }
    }
}

/// The counter-based PRNG (`modelspec.training.RngAlgorithm`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RngAlgorithm {
    Philox4x32x10,
}

/// The RNG state at the checkpoint's step (`modelspec.training.RngState`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RngState {
    pub algorithm: RngAlgorithm,
    pub seed: u64,
    pub counter: u64,
}

/// One parameter of a checkpoint: its path and value.
#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    pub path: String,
    pub value: HostTensor<f32>,
}

/// A checkpoint in memory: what the manifest and payload hold besides the layout.
#[derive(Clone, Debug, PartialEq)]
pub struct Checkpoint {
    /// The number of updates the parameters have had.
    pub step: u64,
    pub schema_hash: ContentHash,
    pub optimizer: OptimizerConfig,
    pub rng: RngState,
    /// In path order.
    pub parameters: Vec<Parameter>,
}

/// Why a checkpoint was not read, matched or written.
#[derive(Clone, Debug, PartialEq, Error)]
pub enum CheckpointError {
    /// A refusal with an `E-CKPT` code.
    #[error("{0}")]
    Refused(Diagnostic),
    /// The file could not be written.
    #[error("cannot write checkpoint {path}: {message}")]
    Write { path: String, message: String },
}

impl CheckpointError {
    /// The code of a refusal.
    pub fn code(&self) -> Option<Code> {
        match self {
            CheckpointError::Refused(d) => Some(d.code),
            CheckpointError::Write { .. } => None,
        }
    }
}

/// One record of the manifest (`modelspec.checkpoint.ParameterRecord`, `OptimizerSlot`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    slot: Option<String>,
    dtype: ManifestDType,
    #[serde(deserialize_with = "bounded_shape")]
    shape: Vec<u64>,
    offset: u64,
    length: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum ManifestDType {
    F32,
}

/// The JSON manifest as the writer lays it out.
#[derive(Clone, Debug, PartialEq, Serialize)]
struct Manifest {
    checkpoint_id: String,
    file_size: u64,
    alignment: u64,
    schema_hash: String,
    step: u64,
    payload_hash: String,
    optimizer: OptimizerConfig,
    optimizer_slots: Vec<Record>,
    rng: RngState,
    parameters: Vec<Record>,
}

/// The JSON manifest as the reader first parses it: the two record lists stay raw text until
/// [`parse_manifest`] has counted them.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawManifest<'a> {
    checkpoint_id: String,
    file_size: u64,
    alignment: u64,
    schema_hash: String,
    step: u64,
    payload_hash: String,
    optimizer: OptimizerConfig,
    #[serde(borrow)]
    optimizer_slots: &'a RawValue,
    rng: RngState,
    #[serde(borrow)]
    parameters: &'a RawValue,
}

/// The most dimensions a record's shape may have.
pub const MAX_RANK: usize = 8;

/// The bytes beyond the file's size that one allocation of [`read`] may take. A valid file of `n`
/// records has at least `124 * n - 64` bytes (a 64-byte slot and about 60 bytes of manifest
/// each), so the list of its records always fits in its size plus this.
const ALLOCATION_SLACK: usize = 4096;

/// The UUID a manifest carries while its `checkpoint_id` is derived.
const NIL_UUID: &str = "00000000-0000-0000-0000-000000000000";

/// Why a manifest did not parse.
enum ManifestError {
    /// Not JSON of a manifest: [`E_CKPT_001`].
    Json(String),
    /// More records than a file of this size can hold, whose list alone would outgrow the file:
    /// [`E_CKPT_002`].
    TooMany { records: usize, most: usize },
}

/// Deserializes a shape of at most [`MAX_RANK`] dimensions.
fn bounded_shape<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u64>, D::Error> {
    struct Dims;
    impl<'de> Visitor<'de> for Dims {
        type Value = Vec<u64>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "a list of at most {MAX_RANK} dimensions")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<u64>, A::Error> {
            let mut dims = Vec::new();
            while let Some(d) = seq.next_element()? {
                if dims.len() == MAX_RANK {
                    return Err(serde::de::Error::custom(format!(
                        "a shape has more than {MAX_RANK} dimensions"
                    )));
                }
                dims.push(d);
            }
            Ok(dims)
        }
    }
    deserializer.deserialize_seq(Dims)
}

/// The number of elements of a JSON list, read without keeping them.
struct Count(usize);

impl<'de> Deserialize<'de> for Count {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Counting;
        impl<'de> Visitor<'de> for Counting {
            type Value = Count;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a list")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Count, A::Error> {
                let mut n = 0;
                while seq.next_element::<IgnoredAny>()?.is_some() {
                    n += 1;
                }
                Ok(Count(n))
            }
        }
        deserializer.deserialize_seq(Counting)
    }
}

/// A list of exactly the counted number of records, allocated once at that size.
struct Exact(usize);

impl<'de> DeserializeSeed<'de> for Exact {
    type Value = Vec<Record>;
    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Vec<Record>, D::Error> {
        deserializer.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for Exact {
    type Value = Vec<Record>;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "a list of {} records", self.0)
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<Record>, A::Error> {
        let mut records = Vec::with_capacity(self.0);
        while let Some(record) = seq.next_element()? {
            if records.len() == self.0 {
                return Err(serde::de::Error::invalid_length(self.0 + 1, &self));
            }
            records.push(record);
        }
        Ok(records)
    }
}

/// Parses the manifest so that no single allocation outgrows the file: the record lists are
/// counted first and refused when they hold more than `most` records together, and each list is
/// then allocated once at its length.
fn parse_manifest(bytes: &[u8], most: usize) -> Result<Manifest, ManifestError> {
    let json = |error: serde_json::Error| ManifestError::Json(error.to_string());
    let raw: RawManifest = serde_json::from_slice(bytes).map_err(json)?;
    let counted = |list: &RawValue| serde_json::from_str::<Count>(list.get()).map(|c| c.0);
    let parameters = counted(raw.parameters).map_err(json)?;
    let optimizer_slots = counted(raw.optimizer_slots).map_err(json)?;
    let records = parameters.saturating_add(optimizer_slots);
    if records > most {
        return Err(ManifestError::TooMany { records, most });
    }
    let exact = |list: &RawValue, n: usize| {
        let mut de = serde_json::Deserializer::from_str(list.get());
        let records = Exact(n).deserialize(&mut de)?;
        de.end()?;
        Ok(records)
    };
    Ok(Manifest {
        checkpoint_id: raw.checkpoint_id,
        file_size: raw.file_size,
        alignment: raw.alignment,
        schema_hash: raw.schema_hash,
        step: raw.step,
        payload_hash: raw.payload_hash,
        optimizer: raw.optimizer,
        optimizer_slots: exact(raw.optimizer_slots, optimizer_slots).map_err(json)?,
        rng: raw.rng,
        parameters: exact(raw.parameters, parameters).map_err(json)?,
    })
}

/// Whether `s` is a UUID in its hyphenated text form.
fn is_uuid(s: &str) -> bool {
    s.len() == 36
        && s.bytes().enumerate().all(|(i, b)| match i {
            8 | 13 | 18 | 23 => b == b'-',
            _ => b.is_ascii_hexdigit(),
        })
}

impl Checkpoint {
    /// The checkpoint of `compiled` after `step` updates, with each declared parameter's value in
    /// `values`; `rng` is the run's seed at counter 0, as an SGD step draws no number.
    pub fn of_state(
        compiled: &CompiledTraining,
        step: u64,
        values: &BTreeMap<ParamId, HostTensor<f32>>,
    ) -> Self {
        let program = &compiled.forward.program;
        let mut parameters: Vec<Parameter> = program
            .params
            .iter()
            .filter_map(|(id, decl)| {
                Some(Parameter {
                    path: decl.path.clone(),
                    value: values.get(id)?.clone(),
                })
            })
            .collect();
        parameters.sort_by(|a, b| a.path.cmp(&b.path));
        Self {
            step,
            schema_hash: schema_hash(&compiled.training.training.program.params),
            optimizer: OptimizerConfig::of(compiled),
            rng: RngState {
                algorithm: RngAlgorithm::Philox4x32x10,
                seed: compiled.config.seed,
                counter: 0,
            },
            parameters,
        }
    }

    /// The file's bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut payload = Vec::new();
        let mut relative = Vec::new();
        for parameter in &self.parameters {
            let at = (payload.len() as u64).next_multiple_of(ALIGNMENT);
            payload.resize(at as usize, 0);
            for v in parameter.value.data() {
                payload.extend_from_slice(&v.to_le_bytes());
            }
            relative.push((at, payload.len() as u64 - at));
        }
        let payload_hash = blake3::hash(&payload).to_hex().to_string();

        let manifest_at = |start: u64, checkpoint_id: &str| Manifest {
            checkpoint_id: checkpoint_id.to_owned(),
            file_size: start + payload.len() as u64,
            alignment: ALIGNMENT,
            schema_hash: self.schema_hash.0.clone(),
            step: self.step,
            payload_hash: payload_hash.clone(),
            optimizer: self.optimizer,
            optimizer_slots: Vec::new(),
            rng: self.rng,
            parameters: self
                .parameters
                .iter()
                .zip(&relative)
                .map(|(p, &(at, length))| Record {
                    path: p.path.clone(),
                    slot: None,
                    dtype: ManifestDType::F32,
                    shape: p.value.shape().to_vec(),
                    offset: start + at,
                    length,
                })
                .collect(),
        };
        let json = |start: u64, id: &str| {
            serde_json::to_vec(&manifest_at(start, id)).expect("JSON of a manifest")
        };
        let mut start = 0u64;
        let unnamed = loop {
            let manifest = json(start, NIL_UUID);
            let next = (HEADER_LEN as u64 + manifest.len() as u64).next_multiple_of(ALIGNMENT);
            if next == start {
                break manifest;
            }
            start = next;
        };
        let mut id = blake3::Hasher::new();
        id.update(b"modelspec-checkpoint-id/1\n");
        id.update(&unnamed);
        let manifest = json(start, &uuid_of(id.finalize().as_bytes()));
        debug_assert_eq!(manifest.len(), unnamed.len(), "a UUID has one length");

        let mut bytes = Vec::with_capacity(start as usize + payload.len());
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&ENDIAN_MARKER.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&(manifest.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&[0; 32]);
        bytes.extend_from_slice(&manifest);
        let checksum = blake3::hash(&bytes);
        bytes[CHECKSUM].copy_from_slice(checksum.as_bytes());
        bytes.resize(start as usize, 0);
        bytes.extend_from_slice(&payload);
        bytes
    }

    /// The parameter values for `compiled` by id, refused with [`E_CKPT_003`] at the first path,
    /// dtype or shape that differs in path order, then at a differing schema hash.
    pub fn parameters_for(
        &self,
        compiled: &CompiledTraining,
    ) -> Result<BTreeMap<ParamId, HostTensor<f32>>, CheckpointError> {
        let params = &compiled.forward.program.params;
        let mut expected: Vec<(&str, ParamId, &TensorType)> = params
            .iter()
            .map(|(&id, decl)| (decl.path.as_str(), id, &decl.ty))
            .collect();
        expected.sort_by(|a, b| a.0.cmp(b.0));
        let mismatch =
            |message: String| CheckpointError::Refused(Diagnostic::new(E_CKPT_003, message));
        let mut values = BTreeMap::new();
        for i in 0..expected.len().max(self.parameters.len()) {
            let (theirs, ours) = (self.parameters.get(i), expected.get(i));
            let (theirs, &(path, id, ty)) = match (theirs, ours) {
                (Some(t), Some(o)) if t.path == o.0 => (t, o),
                _ => {
                    let name = |p: Option<&str>| p.map_or("none".to_owned(), |p| format!("`{p}`"));
                    return Err(CheckpointError::Refused(
                        Diagnostic::new(
                            E_CKPT_003,
                            "the checkpoint does not match the compiled file: a parameter path differs",
                        )
                        .with("checkpoint", name(theirs.map(|t| t.path.as_str())))
                        .with("compiled file", name(ours.map(|o| o.0))),
                    ));
                }
            };
            if ty.dtype != DType::F32 {
                return Err(mismatch(format!(
                    "the checkpoint does not match the compiled file: parameter `{path}` has dtype f32 in the checkpoint and {} in the compiled file",
                    ty.dtype
                )));
            }
            let shape: Option<Vec<u64>> = ty
                .shape
                .iter()
                .map(|d| match d {
                    Dim::Static(n) => Some(*n),
                    Dim::Symbol(_) => None,
                })
                .collect();
            if shape.as_deref() != Some(theirs.value.shape()) {
                return Err(CheckpointError::Refused(
                    Diagnostic::new(
                        E_CKPT_003,
                        format!(
                            "the checkpoint does not match the compiled file: parameter `{path}` has another shape"
                        ),
                    )
                    .with("checkpoint", format!("{:?}", theirs.value.shape()))
                    .with("compiled file", ty.to_string()),
                ));
            }
            values.insert(id, theirs.value.clone());
        }
        let ours = schema_hash(&compiled.training.training.program.params);
        if ours != self.schema_hash {
            return Err(CheckpointError::Refused(
                Diagnostic::new(
                    E_CKPT_003,
                    "the checkpoint does not match the compiled file: the schema hash differs",
                )
                .with("checkpoint", self.schema_hash.0.clone())
                .with("compiled file", ours.0),
            ));
        }
        Ok(values)
    }

    /// [`Checkpoint::parameters_for`], and refused with [`E_CKPT_003`] unless the optimizer is
    /// that of `compiled`.
    pub fn resume_into(
        &self,
        compiled: &CompiledTraining,
    ) -> Result<BTreeMap<ParamId, HostTensor<f32>>, CheckpointError> {
        let values = self.parameters_for(compiled)?;
        let ours = OptimizerConfig::of(compiled);
        if ours != self.optimizer {
            return Err(CheckpointError::Refused(
                Diagnostic::new(
                    E_CKPT_003,
                    "the checkpoint does not match the compiled file: the optimizer differs",
                )
                .with("checkpoint", describe(&self.optimizer))
                .with("compiled file", describe(&ours)),
            ));
        }
        Ok(values)
    }
}

fn describe(optimizer: &OptimizerConfig) -> String {
    serde_json::to_string(optimizer).expect("JSON of an optimizer")
}

/// The first 16 bytes of `hash` as a version-8 UUID.
fn uuid_of(hash: &[u8; 32]) -> String {
    let mut b = [0u8; 16];
    b.copy_from_slice(&hash[..16]);
    b[6] = (b[6] & 0x0f) | 0x80;
    b[8] = (b[8] & 0x3f) | 0x80;
    let hex: String = b.iter().map(|x| format!("{x:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// Writes `checkpoint` to `path`; a parameter of more than [`MAX_RANK`] dimensions, which
/// [`read`] would refuse, is not written. The bytes go to `<path>.tmp` beside it, are synced and
/// then renamed over `path`, so a write that fails part-way leaves the earlier file at `path`.
pub fn write(path: &Path, checkpoint: &Checkpoint) -> Result<(), CheckpointError> {
    if let Some(p) = checkpoint
        .parameters
        .iter()
        .find(|p| p.value.shape().len() > MAX_RANK)
    {
        return Err(CheckpointError::Write {
            path: path.display().to_string(),
            message: format!(
                "parameter `{}` has {} dimensions, more than the {MAX_RANK} a checkpoint holds",
                p.path,
                p.value.shape().len()
            ),
        });
    }
    let failed = |error: std::io::Error| CheckpointError::Write {
        path: path.display().to_string(),
        message: error.to_string(),
    };
    let mut staged = path.as_os_str().to_owned();
    staged.push(".tmp");
    let staged = std::path::PathBuf::from(staged);
    let written = File::create(&staged)
        .and_then(|mut file| {
            file.write_all(&checkpoint.encode())?;
            file.sync_all()
        })
        .and_then(|()| std::fs::rename(&staged, path));
    if let Err(error) = written {
        let _ = std::fs::remove_file(&staged);
        return Err(failed(error));
    }
    Ok(())
}

/// Reads and validates the checkpoint at `path`, as the module documents.
pub fn read(path: &Path) -> Result<Checkpoint, CheckpointError> {
    let shown = path.display().to_string();
    let unreadable = |error: std::io::Error| {
        CheckpointError::Refused(
            Diagnostic::new(
                E_CKPT_001,
                "not a readable checkpoint: the file cannot be read",
            )
            .with("file", shown.clone())
            .with("error", error.to_string()),
        )
    };
    let mut file = File::open(path).map_err(unreadable)?;
    let size = file.metadata().map_err(unreadable)?.len();
    let outside = |message: String| {
        CheckpointError::Refused(Diagnostic::new(E_CKPT_002, message).with("file", shown.clone()))
    };
    let bad = |message: String| {
        CheckpointError::Refused(
            Diagnostic::new(E_CKPT_001, format!("not a readable checkpoint: {message}"))
                .with("file", shown.clone()),
        )
    };
    if size < HEADER_LEN as u64 {
        return Err(outside(format!(
            "the file has {size} bytes, fewer than the {HEADER_LEN}-byte header"
        )));
    }
    let mut header = [0u8; HEADER_LEN];
    file.read_exact(&mut header).map_err(unreadable)?;
    let word = |at: usize| u32::from_le_bytes(header[at..at + 4].try_into().expect("4 bytes"));
    if header[0..4] != MAGIC {
        return Err(bad("the magic is not MSCK".into()));
    }
    if word(4) != FORMAT_VERSION {
        return Err(bad(format!(
            "format version {} is not {FORMAT_VERSION}",
            word(4)
        )));
    }
    if word(8) != ENDIAN_MARKER {
        return Err(bad(if word(8) == ENDIAN_MARKER.swap_bytes() {
            "the endian marker is big-endian; the payload is little-endian".into()
        } else {
            format!("the endian marker {:#010x} is unknown", word(8))
        }));
    }
    if word(12) != 0 {
        return Err(bad("the reserved header field is not zero".into()));
    }
    let manifest_size = u64::from_le_bytes(header[16..24].try_into().expect("8 bytes"));
    if manifest_size == 0 || manifest_size > size - HEADER_LEN as u64 {
        return Err(outside(format!(
            "the manifest size {manifest_size} does not fit the {size}-byte file after its header"
        )));
    }
    let mut signed = header.to_vec();
    signed[CHECKSUM].fill(0);
    let mut manifest_bytes = vec![0u8; manifest_size as usize];
    file.read_exact(&mut manifest_bytes).map_err(unreadable)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(&signed);
    hasher.update(&manifest_bytes);
    if hasher.finalize().as_bytes() != &header[CHECKSUM] {
        return Err(bad(
            "the checksum differs from the header and manifest".into()
        ));
    }
    let manifest_end = HEADER_LEN as u64 + manifest_size;
    let payload_start = manifest_end.next_multiple_of(ALIGNMENT).min(size);
    let most = usize::try_from(size)
        .unwrap_or(usize::MAX)
        .saturating_add(ALLOCATION_SLACK)
        / std::mem::size_of::<Record>();
    let manifest = parse_manifest(&manifest_bytes, most).map_err(|error| match error {
        ManifestError::Json(error) => bad(format!(
            "the manifest is not a checkpoint manifest: {error}"
        )),
        ManifestError::TooMany { records, most } => outside(format!(
            "the manifest has {records} records, more than the {most} a {size}-byte file can hold"
        )),
    })?;
    drop(manifest_bytes);
    invariants(&manifest).map_err(bad)?;

    if manifest.file_size != size {
        return Err(outside(format!(
            "the manifest's file_size {} is not the file's {size} bytes",
            manifest.file_size
        )));
    }
    if manifest.alignment != ALIGNMENT {
        return Err(outside(format!(
            "the alignment {} is not {ALIGNMENT}",
            manifest.alignment
        )));
    }
    let mut extents =
        Vec::with_capacity(manifest.parameters.len() + manifest.optimizer_slots.len());
    for record in manifest.parameters.iter().chain(&manifest.optimizer_slots) {
        let at = |what: String| outside(format!("parameter `{}`: {what}", record.path));
        if !record.offset.is_multiple_of(ALIGNMENT) {
            return Err(at(format!(
                "offset {} is not a multiple of {ALIGNMENT}",
                record.offset
            )));
        }
        if record.offset < manifest_end {
            return Err(at(format!(
                "offset {} lies inside the header or manifest, which end at {manifest_end}",
                record.offset
            )));
        }
        let end = record.offset.checked_add(record.length);
        if end.is_none_or(|end| end > size) {
            return Err(at(format!(
                "offset {} and length {} end past the {size}-byte file",
                record.offset, record.length
            )));
        }
        let expected = record
            .shape
            .iter()
            .try_fold(DType::F32.size_bytes(), |n, &d| n.checked_mul(d));
        if expected != Some(record.length) {
            return Err(at(format!(
                "length {} is not the shape {:?} times {} bytes",
                record.length,
                record.shape,
                DType::F32.size_bytes()
            )));
        }
        extents.push((record.offset, record.offset + record.length, &record.path));
    }
    extents.sort();
    for pair in extents.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(outside(format!(
                "parameters `{}` and `{}` overlap",
                pair[0].2, pair[1].2
            )));
        }
    }
    let last = extents.iter().map(|e| e.1).max().unwrap_or(manifest_end);
    if last != size {
        return Err(outside(format!(
            "the file has {size} bytes and does not end at the end of its last tensor, byte {last}"
        )));
    }
    let mut rest = vec![0u8; (size - manifest_end) as usize];
    file.read_exact(&mut rest).map_err(unreadable)?;
    let mut cursor = manifest_end;
    for &(from, to, _) in &extents {
        let gap = &rest[(cursor - manifest_end) as usize..(from - manifest_end) as usize];
        if let Some(i) = gap.iter().position(|&b| b != 0) {
            return Err(outside(format!(
                "byte {} lies between tensors and is not zero",
                cursor + i as u64
            )));
        }
        cursor = to;
    }
    let payload = &rest[(payload_start - manifest_end) as usize..];
    if blake3::hash(payload).to_hex().as_str() != manifest.payload_hash {
        return Err(CheckpointError::Refused(
            Diagnostic::new(E_CKPT_004, "the payload hash differs from the payload")
                .with("file", shown.clone()),
        ));
    }
    let mut parameters = Vec::with_capacity(manifest.parameters.len());
    for record in &manifest.parameters {
        let from = (record.offset - payload_start) as usize;
        let bytes = &payload[from..from + record.length as usize];
        let data: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let value = HostTensor::new(record.shape.clone(), data)
            .map_err(|error| bad(format!("parameter `{}`: {error}", record.path)))?;
        parameters.push(Parameter {
            path: record.path.clone(),
            value,
        });
    }
    Ok(Checkpoint {
        step: manifest.step,
        schema_hash: ContentHash(manifest.schema_hash),
        optimizer: manifest.optimizer,
        rng: manifest.rng,
        parameters,
    })
}

/// The invariants of `spec/domains/checkpoint.yaml` the manifest's own fields state.
fn invariants(manifest: &Manifest) -> Result<(), String> {
    if !is_uuid(&manifest.checkpoint_id) {
        return Err(format!(
            "checkpoint_id {:?} is not a UUID",
            manifest.checkpoint_id
        ));
    }
    if !manifest.optimizer.is_valid() {
        return Err("the optimizer's fields do not fit its kind".into());
    }
    if manifest.optimizer.kind == OptimizerKind::Sgd && !manifest.optimizer_slots.is_empty() {
        return Err("an SGD checkpoint has optimizer slots".into());
    }
    if manifest.parameters.is_empty() {
        return Err("the manifest has no parameter".into());
    }
    let mut paths = BTreeSet::new();
    for record in &manifest.parameters {
        if record.slot.is_some() {
            return Err(format!("parameter `{}` has a slot name", record.path));
        }
        if !paths.insert(record.path.as_str()) {
            return Err(format!("parameter `{}` is recorded twice", record.path));
        }
    }
    let mut slots = BTreeSet::new();
    for record in &manifest.optimizer_slots {
        let Some(slot) = &record.slot else {
            return Err(format!(
                "an optimizer slot of `{}` has no name",
                record.path
            ));
        };
        if !slots.insert((record.path.as_str(), slot.as_str())) {
            return Err(format!(
                "slot `{slot}` of `{}` is recorded twice",
                record.path
            ));
        }
    }
    for record in manifest.parameters.iter().chain(&manifest.optimizer_slots) {
        if record.shape.contains(&0) {
            return Err(format!("parameter `{}` has a zero dimension", record.path));
        }
        if record.length == 0 {
            return Err(format!("parameter `{}` has length 0", record.path));
        }
    }
    Ok(())
}
