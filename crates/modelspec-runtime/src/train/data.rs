//! Training data: M0 synthetic teacher data (DESIGN.md § 9 gate 4) and dense `f32` tensors read
//! from a directory (DESIGN.md § 8; `spec/domains/data.yaml`, `DenseDataDirectory`).
//!
//! Synthetic data ([`synthetic`]): the one `input` is a standard normal tensor on
//! [`Stream::Data`], drawn by Box-Muller and named by the input's name; the one `target` is the
//! model's prediction for it under teacher parameters drawn by each parameter's initializer on
//! [`Stream::Teacher`]. The teacher parameters are the known solution training should approach.
//! The prediction is read as the intermediate value `prediction` of the forward program, run on
//! the CPU interpreter with a zero target.
//!
//! Dense data ([`dense`], [`dense_eval`]): the declaration `name: Tensor<f32, [B, d1, …]>` reads
//! `<dir>/<name>.f32`, row-major little-endian `f32` with no header, one sample per `d1 × …`
//! values. Training step `k` feeds the samples `order[k·B + i]` of `<dir>/order.u32` when the
//! file exists, else `(k·B + i) mod N`, for `i` in `0..B`; positions past the end of the order
//! wrap, which only the loss after the last update reads. A file that cannot be read as samples,
//! or holds a NaN or an infinity, is [`E_DATA_007`], an order that does not cover the run is [`E_DATA_008`], and an eval
//! directory that is not a whole number of batches is [`E_DATA_009`].
//!
//! A declaration the loss does not read has no `Input` node after dead-code elimination, so the
//! program could not take a value for it; it is refused at its declaration ([`E_DATA_003`])
//! before anything runs. Neither source supplies `i32` row ids: an `i32` declaration is refused
//! at its declaration too ([`E_DATA_004`]).
//!
//! Packed masked-language-model data ([`source`], [`packed`]; story:bert-train-data-binding,
//! coordinator decision D48): a `--data` directory that holds [`BINDING_FILE`] is a directory
//! `modelspec data pack` wrote. Each `input` and `target` declaration names one tensor of the
//! [`MlmBatch`] (`ids`, `positions`, `key_mask`, `labels`, `weights`), declared `[B, S]` or
//! `[B·S]` with `S` the packed sequence length, and training step `k` feeds the tensors of
//! [`mlm_batch`] of the `Train` split at step `k` with the run seed and the policy of the
//! binding. A declaration that names no batch tensor is [`E_DATA_005`] and one of another dtype
//! or shape [`E_DATA_006`], both before any file but the binding is read; E-DATA-001, E-DATA-002
//! and E-DATA-004 do not apply, and E-DATA-003 does. A directory without [`BINDING_FILE`] stays
//! dense, and when every declaration names a batch tensor its dense refusal follows the line of
//! [`read_as_dense`] (coordinator decision D56).

use std::collections::{BTreeMap, BTreeSet};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use modelspec_core::diag::{Code, Diagnostic};
use modelspec_core::hir::DataRole;
use modelspec_core::ir::tensor::{DType, Dim, ParamId, TensorOp, TensorType, ValueId};
use modelspec_core::ir::training::{DataDecl, TrainConfig};
use modelspec_core::lower::LoweredProgram;
use thiserror::Error;

use super::init::{InitError, Stream, initialize, standard_normal};
use crate::cpu::{HostTensor, RuntimeError, evaluate_all};
use crate::dataset::SplitRole;
use crate::dataset::mlm::{
    self, BINDING_FILE, MlmBatch, Packed, TrainingDataBinding, binding_id, mlm_batch,
};

/// The `train` block does not declare exactly one `input` and one `target`. On a `--data`
/// directory without `binding.json`, read as dense, under declarations that all name batch
/// tensors, `train` prints a line naming the directory and `binding.json` before it.
pub const E_DATA_001: Code = Code::new("E-DATA-001");
/// The `target` is not of the model output type.
pub const E_DATA_002: Code = Code::new("E-DATA-002");
/// A declared `input` or `target` is not read by the loss.
pub const E_DATA_003: Code = Code::new("E-DATA-003");
/// A declared `input` or `target` is `i32`: synthetic data draws `f32` values and dense files
/// hold `f32`, so neither can supply row ids.
pub const E_DATA_004: Code = Code::new("E-DATA-004");
/// A declared `input` or `target` of a run on a packed `--data` directory names no tensor of the
/// masked-language-model batch (`ids`, `positions`, `key_mask`, `labels`, `weights`), or the
/// `train` block of such a run declares none.
pub const E_DATA_005: Code = Code::new("E-DATA-005");
/// A declared `input` or `target` of a run on a packed `--data` directory is not of the type of
/// its batch tensor: `i32` for `ids`, `positions` and `labels`, `f32` for `key_mask` and
/// `weights`, of shape `[B, S]` or `[B·S]` with `S` the packed sequence length and one `B` for
/// every declaration.
pub const E_DATA_006: Code = Code::new("E-DATA-006");
/// A data, order or `--init` file is missing, unreadable, or not a whole number of samples (for
/// `--init`, not the parameter's size), or a data or `--init` file holds a NaN or an infinity, or
/// the data files differ in sample count, or the `input` and `target` do not share a leading
/// batch dimension `B`, or the `--init` directory cannot be read or holds no parameter's file.
pub const E_DATA_007: Code = Code::new("E-DATA-007");
/// `order.u32` holds fewer than `steps · B` indices, or an index not below the sample count.
pub const E_DATA_008: Code = Code::new("E-DATA-008");
/// The `--eval` sample count is not a multiple of the batch dimension `B`.
pub const E_DATA_009: Code = Code::new("E-DATA-009");

/// The data of a synthetic run.
#[derive(Clone, Debug, PartialEq)]
pub struct SyntheticData {
    /// A value for the input and the target, by name: what the training step reads.
    pub inputs: BTreeMap<String, HostTensor<f32>>,
    /// The teacher parameters the target was computed with.
    pub teacher: BTreeMap<ParamId, HostTensor<f32>>,
}

/// Why no data was made.
#[derive(Clone, Debug, PartialEq, Error)]
pub enum DataError {
    /// The `train` block's data declarations do not fit M0 synthetic data: a user error.
    #[error("{0}")]
    Refused(Diagnostic),
    /// `prediction` is not a value of the forward program: a compiler bug.
    #[error("the forward program defines no prediction value {0}")]
    UndefinedPrediction(ValueId),
    #[error(transparent)]
    Init(#[from] InitError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    /// The packed directory `dir` cannot be read or holds no batch of the run: `reason` is the
    /// reader's error, which names the file it concerns.
    #[error("cannot read the packed directory {}: {reason}", dir.display())]
    Packed { dir: PathBuf, reason: String },
}

/// The synthetic data of `config` for the forward program `lowered`, whose value
/// `lowered.prediction` is the model output.
///
/// Refused, in this order, with [`E_DATA_001`] unless the block declares exactly one `input` and
/// one `target`, with [`E_DATA_004`] at the input, else the target, when it is `i32`, with
/// [`E_DATA_002`] unless the target has the type of the prediction, and with
/// [`E_DATA_003`] at the first declaration of `lowered.declared_inputs` the program has no
/// `Input` node for.
pub fn synthetic(
    config: &TrainConfig,
    lowered: &LoweredProgram,
) -> Result<SyntheticData, DataError> {
    let (input, target) = declarations(config, lowered, "M0 synthetic data")?;
    let prediction = lowered.prediction;
    let forward = &lowered.program;
    let teacher = initialize(forward, config.seed, Stream::Teacher)?;
    let x = standard_normal(config.seed, Stream::Data, &input.name, &input.ty)?;
    let zero = zeros(&target.name, &target.ty)?;
    let mut inputs = BTreeMap::from([(input.name.clone(), x), (target.name.clone(), zero)]);
    let mut values = evaluate_all(forward, &inputs, &teacher)?.values;
    let y = values
        .remove(&prediction)
        .ok_or(DataError::UndefinedPrediction(prediction))?;
    inputs.insert(target.name.clone(), y);
    Ok(SyntheticData { inputs, teacher })
}

/// The one `input` and one `target` of `config`, refused as [`synthetic`] documents; `what`
/// names the data in the [`E_DATA_001`] message.
fn declarations<'c>(
    config: &'c TrainConfig,
    lowered: &LoweredProgram,
    what: &str,
) -> Result<(&'c DataDecl, &'c DataDecl), DataError> {
    let prediction = lowered.prediction;
    let forward = &lowered.program;
    let ([input], [target]) = (&config.inputs[..], &config.targets[..]) else {
        return Err(DataError::Refused(
            Diagnostic::new(
                E_DATA_001,
                format!("{what} needs exactly one input and one target"),
            )
            .at(config.span.clone())
            .with("inputs", config.inputs.len().to_string())
            .with("targets", config.targets.len().to_string()),
        ));
    };
    for (decl, role) in [(input, "input"), (target, "target")] {
        if decl.ty.dtype != DType::F32 {
            return Err(DataError::Refused(
                Diagnostic::new(
                    E_DATA_004,
                    format!(
                        "{what} cannot supply the {} {role} `{}`",
                        decl.ty.dtype, decl.name
                    ),
                )
                .at(decl.span.clone())
                .with(role, decl.ty.to_string()),
            ));
        }
    }
    let output = &forward
        .node(prediction)
        .ok_or(DataError::UndefinedPrediction(prediction))?
        .ty;
    if target.ty != *output {
        return Err(DataError::Refused(
            Diagnostic::new(E_DATA_002, "the target is not of the model output type")
                .at(target.span.clone())
                .with("target", target.ty.to_string())
                .with("model output", output.to_string()),
        ));
    }
    refuse_unread(lowered)?;
    Ok((input, target))
}

/// [`E_DATA_003`] at the first declaration of `lowered.declared_inputs` the program has no
/// `Input` node for.
fn refuse_unread(lowered: &LoweredProgram) -> Result<(), DataError> {
    let forward = &lowered.program;
    let read: BTreeSet<&str> = forward
        .inputs
        .iter()
        .filter_map(|&value| match &forward.node(value)?.op {
            TensorOp::Input { name } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    if let Some(unread) = lowered
        .declared_inputs
        .iter()
        .find(|d| !read.contains(d.name.as_str()))
    {
        let role = match unread.role {
            DataRole::Input => "input",
            DataRole::Target => "target",
        };
        return Err(DataError::Refused(
            Diagnostic::new(
                E_DATA_003,
                format!("the {role} `{}` is not read by the loss", unread.name),
            )
            .at(unread.span.clone())
            .with(role, unread.name.clone()),
        ));
    }
    Ok(())
}

/// Dense data read from a directory: a value of every sample of the `input` and the `target`,
/// and for training the optional sample order.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseData {
    /// The batch dimension `B`: the leading dimension of both declarations.
    batch: usize,
    /// The sample count `N` of every file.
    samples: usize,
    /// The input, then the target.
    tensors: [DenseTensor; 2],
    /// The indices of `order.u32`, when the directory has one.
    order: Option<Vec<u32>>,
}

/// The inputs of one training step or eval batch of a [`DenseData`], refilled in place by
/// [`DenseData::fill_step`] and [`DenseData::fill_eval_batch`]: the first fill allocates the key
/// and the data buffer of each declaration, and every later fill by the same data copies into
/// them (story:dense-step-reuses-batch-buffers). A batch is filled by one [`DenseData`] only.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DenseBatch {
    inputs: BTreeMap<String, HostTensor<f32>>,
}

impl DenseBatch {
    /// The value of each declaration, by name.
    pub fn inputs(&self) -> &BTreeMap<String, HostTensor<f32>> {
        &self.inputs
    }
}

/// The samples of one declaration.
#[derive(Clone, Debug, PartialEq)]
struct DenseTensor {
    name: String,
    /// The declared shape `[B, d1, …]`.
    shape: Vec<u64>,
    /// `d1 × …`.
    sample: usize,
    values: Vec<f32>,
}

impl DenseData {
    /// The batch dimension `B`.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// The sample count `N`.
    pub fn samples(&self) -> usize {
        self.samples
    }

    /// The name of the target declaration.
    pub fn target(&self) -> &str {
        &self.tensors[1].name
    }

    /// The inputs of training step `k`: the samples `order[(k·B + i) mod len]` with an order,
    /// `(k·B + i) mod N` without one, for `i` in `0..B`. A new map each call; the training loop
    /// refills one [`DenseBatch`] with [`Self::fill_step`] instead.
    pub fn step(&self, k: u64) -> BTreeMap<String, HostTensor<f32>> {
        let mut batch = DenseBatch::default();
        self.fill_step(k, &mut batch);
        batch.inputs
    }

    /// The inputs of eval batch `b`: the samples `b·B .. (b+1)·B`. A new map each call, as
    /// [`Self::step`].
    pub fn eval_batch(&self, b: usize) -> BTreeMap<String, HostTensor<f32>> {
        let mut batch = DenseBatch::default();
        self.fill_eval_batch(b, &mut batch);
        batch.inputs
    }

    /// Fills `batch` with the inputs of training step `k`, the samples [`Self::step`] names.
    pub fn fill_step(&self, k: u64, batch: &mut DenseBatch) {
        let first = k as u128 * self.batch as u128;
        let position = |i: usize| first + i as u128;
        self.fill(batch, |i| match &self.order {
            Some(order) => order[(position(i) % order.len() as u128) as usize] as usize,
            None => (position(i) % self.samples as u128) as usize,
        });
    }

    /// Fills `batch` with the inputs of eval batch `b`, the samples `b·B .. (b+1)·B`.
    pub fn fill_eval_batch(&self, b: usize, batch: &mut DenseBatch) {
        let first = b * self.batch;
        self.fill(batch, |i| first + i);
    }

    /// Writes the samples `sample(i)`, `i` in `0..B`, of each declaration into its entry of
    /// `batch`. An entry `batch` holds is rewritten in place, its key, shape and data buffer
    /// kept, so only a declaration `batch` does not hold yet allocates (or a fill of more
    /// elements than the entry has held).
    fn fill(&self, batch: &mut DenseBatch, sample: impl Fn(usize) -> usize) {
        for t in &self.tensors {
            let write = |data: &mut Vec<f32>| {
                data.clear();
                for i in 0..self.batch {
                    let s = sample(i);
                    data.extend_from_slice(&t.values[s * t.sample..(s + 1) * t.sample]);
                }
            };
            match batch.inputs.get_mut(t.name.as_str()) {
                Some(tensor) => tensor.refill(&t.shape, write),
                None => {
                    let mut data = Vec::with_capacity(self.batch * t.sample);
                    write(&mut data);
                    let tensor = HostTensor::from_parts(t.shape.clone(), data);
                    batch.inputs.insert(t.name.clone(), tensor);
                }
            }
        }
    }
}

/// The training data of `--data <dir>`: the `input` and `target` files of `config` and the
/// optional `order.u32`, which must hold at least `config.steps · B` indices, each below `N`
/// ([`E_DATA_008`]). Refused first as [`synthetic`] refuses the declarations.
pub fn dense(
    dir: &Path,
    config: &TrainConfig,
    lowered: &LoweredProgram,
) -> Result<DenseData, DataError> {
    let mut data = read_dense(dir, config, lowered)?;
    let path = dir.join("order.u32");
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(data),
        Err(error) => return Err(unreadable("the order file", &path, &error)),
    };
    if bytes.len() % 4 != 0 {
        return Err(DataError::Refused(
            Diagnostic::new(
                E_DATA_007,
                "the order file is not a whole number of u32 indices",
            )
            .with("file", path.display().to_string())
            .with("bytes", bytes.len().to_string()),
        ));
    }
    let order: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    let needed = u128::from(config.steps) * data.batch as u128;
    if (order.len() as u128) < needed || order.is_empty() {
        return Err(DataError::Refused(
            Diagnostic::new(
                E_DATA_008,
                "the order file holds fewer than steps · B indices",
            )
            .with("file", path.display().to_string())
            .with("indices", order.len().to_string())
            .with("needed", needed.max(1).to_string()),
        ));
    }
    if let Some((position, &index)) = order
        .iter()
        .enumerate()
        .find(|&(_, &index)| index as usize >= data.samples)
    {
        return Err(DataError::Refused(
            Diagnostic::new(
                E_DATA_008,
                "the order file names a sample past the end of the data",
            )
            .with("file", path.display().to_string())
            .with("position", position.to_string())
            .with("index", index.to_string())
            .with("samples", data.samples.to_string()),
        ));
    }
    data.order = Some(order);
    Ok(data)
}

/// The eval data of `--eval <dir>`: the `input` and `target` files of `config`, whose sample
/// count must be a multiple of `B` ([`E_DATA_009`]); `order.u32` is not read.
pub fn dense_eval(
    dir: &Path,
    config: &TrainConfig,
    lowered: &LoweredProgram,
) -> Result<DenseData, DataError> {
    let data = read_dense(dir, config, lowered)?;
    if !data.samples.is_multiple_of(data.batch) {
        return Err(DataError::Refused(
            Diagnostic::new(E_DATA_009, "the eval data is not a whole number of batches")
                .with("directory", dir.display().to_string())
                .with("samples", data.samples.to_string())
                .with("batch", data.batch.to_string()),
        ));
    }
    Ok(data)
}

/// The `input` and `target` files of `dir`, without an order.
fn read_dense(
    dir: &Path,
    config: &TrainConfig,
    lowered: &LoweredProgram,
) -> Result<DenseData, DataError> {
    let (input, target) = declarations(config, lowered, "dense data")?;
    let batch = match (input.ty.shape.first(), target.ty.shape.first()) {
        (Some(&Dim::Static(a)), Some(&Dim::Static(b))) if a == b && a > 0 => a,
        _ => {
            return Err(DataError::Refused(
                Diagnostic::new(
                    E_DATA_007,
                    "the input and the target do not share a leading batch dimension",
                )
                .at(target.span.clone())
                .with("input", input.ty.to_string())
                .with("target", target.ty.to_string()),
            ));
        }
    };
    let input = read_tensor(dir, "input", input)?;
    let target = read_tensor(dir, "target", target)?;
    let samples = input.values.len() / input.sample;
    let target_samples = target.values.len() / target.sample;
    if samples != target_samples {
        return Err(DataError::Refused(
            Diagnostic::new(
                E_DATA_007,
                "the data files do not hold the same number of samples",
            )
            .with("directory", dir.display().to_string())
            .with(input.name.clone(), samples.to_string())
            .with(target.name.clone(), target_samples.to_string()),
        ));
    }
    let batch = usize::try_from(batch).map_err(|_| InitError::Shape {
        path: input.name.clone(),
    })?;
    Ok(DenseData {
        batch,
        samples,
        tensors: [input, target],
        order: None,
    })
}

/// The samples of `decl` in `<dir>/<name>.f32`.
fn read_tensor(dir: &Path, role: &str, decl: &DataDecl) -> Result<DenseTensor, DataError> {
    let shape = zeros(&decl.name, &decl.ty)?.shape().to_vec();
    let sample: usize = shape[1..].iter().map(|&n| n as usize).product();
    let path = dir.join(format!("{}.f32", decl.name));
    let values = read_f32(&path).map_err(|error| {
        unreadable(
            &format!("the data file of the {role} `{}`", decl.name),
            &path,
            &error,
        )
    })?;
    if values.is_empty() || !values.len().is_multiple_of(sample) {
        return Err(DataError::Refused(
            Diagnostic::new(
                E_DATA_007,
                format!(
                    "the data file of the {role} `{}` is not a whole number of samples",
                    decl.name
                ),
            )
            .at(decl.span.clone())
            .with("file", path.display().to_string())
            .with("values", values.len().to_string())
            .with("sample", format!("{sample} values")),
        ));
    }
    let what = format!("the data file of the {role} `{}`", decl.name);
    if let Some(refused) = non_finite(&what, &path, &values) {
        return Err(refused);
    }
    Ok(DenseTensor {
        name: decl.name.clone(),
        shape,
        sample,
        values,
    })
}

/// The little-endian `f32` values of the file `path`; a length that is not a whole number of
/// values is [`ErrorKind::InvalidData`].
pub(crate) fn read_f32(path: &Path) -> std::io::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            format!("{} bytes is not a whole number of f32 values", bytes.len()),
        ));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

/// [`E_DATA_007`] when `values`, read as `what` from `path`, hold a NaN or an infinity.
pub(crate) fn non_finite(what: &str, path: &Path, values: &[f32]) -> Option<DataError> {
    let (index, value) = values.iter().enumerate().find(|(_, v)| !v.is_finite())?;
    Some(DataError::Refused(
        Diagnostic::new(
            E_DATA_007,
            format!("{what} holds a value that is not finite"),
        )
        .with("file", path.display().to_string())
        .with("index", index.to_string())
        .with("value", value.to_string()),
    ))
}

/// [`E_DATA_007`]: `what` at `path` cannot be read.
pub(crate) fn unreadable(what: &str, path: &Path, error: &std::io::Error) -> DataError {
    DataError::Refused(
        Diagnostic::new(E_DATA_007, format!("cannot read {what}"))
            .with("file", path.display().to_string())
            .with("error", error.to_string()),
    )
}

/// A tensor of the masked-language-model batch, the field of [`MlmBatch`] a declaration names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BatchTensor {
    Ids,
    Positions,
    KeyMask,
    Labels,
    Weights,
}

impl BatchTensor {
    /// Every batch tensor, in the order `B` is read from the declarations (coordinator decision
    /// D48).
    const ALL: [BatchTensor; 5] = [
        BatchTensor::Ids,
        BatchTensor::Positions,
        BatchTensor::KeyMask,
        BatchTensor::Labels,
        BatchTensor::Weights,
    ];

    fn name(self) -> &'static str {
        match self {
            BatchTensor::Ids => "ids",
            BatchTensor::Positions => "positions",
            BatchTensor::KeyMask => "key_mask",
            BatchTensor::Labels => "labels",
            BatchTensor::Weights => "weights",
        }
    }

    fn dtype(self) -> DType {
        match self {
            BatchTensor::Ids | BatchTensor::Positions | BatchTensor::Labels => DType::I32,
            BatchTensor::KeyMask | BatchTensor::Weights => DType::F32,
        }
    }

    fn of(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.name() == name)
    }
}

/// What the directory of `--data` holds.
#[derive(Clone, Debug, PartialEq)]
pub enum DataSource {
    /// Dense `f32` files ([`dense`]).
    Dense,
    /// A packed masked-language-model directory and its binding, read from [`BINDING_FILE`]
    /// ([`packed`]).
    Packed(TrainingDataBinding),
}

/// The `input` and `target` declarations of `config`, in source order, each with its role.
fn in_source_order(config: &TrainConfig) -> Vec<(&DataDecl, &'static str)> {
    let mut decls: Vec<(&DataDecl, &'static str)> = config
        .inputs
        .iter()
        .map(|d| (d, "input"))
        .chain(config.targets.iter().map(|d| (d, "target")))
        .collect();
    decls.sort_by_key(|(d, _)| (d.span.line, d.span.column));
    decls
}

/// Whether `config` declares an `input` or `target` and every one names a batch tensor.
fn batch_only(config: &TrainConfig) -> bool {
    let decls = in_source_order(config);
    !decls.is_empty()
        && decls
            .iter()
            .all(|(d, _)| BatchTensor::of(&d.name).is_some())
}

/// The line `train` prints before the dense refusal of `dir` when `dir` holds no
/// [`BINDING_FILE`], so it is read as a dense directory, although every `input` and `target`
/// declaration of `config` names a batch tensor: `<dir> holds no binding.json, so it is read as a
/// dense directory` (coordinator decision D56). `None` when `dir` holds the file or a
/// declaration names no batch tensor.
pub fn read_as_dense(dir: &Path, config: &TrainConfig) -> Option<String> {
    let absent = std::fs::metadata(dir.join(BINDING_FILE)).is_err();
    (absent && batch_only(config)).then(|| {
        format!(
            "{} holds no {BINDING_FILE}, so it is read as a dense directory",
            dir.display()
        )
    })
}

/// What `dir` holds: a packed directory when it holds [`BINDING_FILE`], whose binding is read and
/// checked here and is the only file read, and dense files otherwise (coordinator decision D48).
/// A directory that does not exist is read as packed when the `train` block declares batch
/// tensors only, so that its run is refused naming the directory, and as dense otherwise, whose
/// refusals do not change ([`read_as_dense`] is the line `train` prints before them). A binding that cannot be read, or whose `binding_id` is not the id of
/// its fields, is [`DataError::Packed`].
pub fn source(dir: &Path, config: &TrainConfig) -> Result<DataSource, DataError> {
    let file = dir.join(BINDING_FILE);
    let unreadable = |reason: String| DataError::Packed {
        dir: dir.to_path_buf(),
        reason: format!("{}: {reason}", file.display()),
    };
    if std::fs::metadata(&file).is_err() {
        return match std::fs::metadata(dir) {
            Err(error) if error.kind() == ErrorKind::NotFound && batch_only(config) => {
                Err(DataError::Packed {
                    dir: dir.to_path_buf(),
                    reason: error.to_string(),
                })
            }
            _ => Ok(DataSource::Dense),
        };
    }
    let text = std::fs::read_to_string(&file).map_err(|e| unreadable(e.to_string()))?;
    let binding: TrainingDataBinding =
        serde_json::from_str(&text).map_err(|e| unreadable(e.to_string()))?;
    let id = binding_id(
        &binding.dataset_id,
        &binding.tokenizer_id,
        binding.seq_len,
        &binding.masking_policy,
    );
    if binding.seq_len == 0 || id.as_deref() != Some(binding.binding_id.as_str()) {
        return Err(unreadable(format!(
            "binding_id {} is not the id of its dataset, tokenizer, seq_len and masking policy",
            binding.binding_id
        )));
    }
    Ok(DataSource::Packed(binding))
}

/// The training data of a packed `--data` directory: the packed sequences, the binding they are
/// drawn with, and the declaration each batch tensor feeds.
pub struct PackedData {
    dir: PathBuf,
    binding: TrainingDataBinding,
    packed: Packed,
    /// The batch dimension `B`, sequences per step.
    batch: usize,
    seed: u64,
    /// Each declaration's batch tensor, name and shape, in source order.
    bound: Vec<(BatchTensor, String, Vec<u64>)>,
}

/// The inputs of one training step of a [`PackedData`]: its `f32` tensors and its `i32` tensors,
/// by declaration name, as the interpreter takes them ([`crate::cpu::evaluate_with_ids`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PackedBatch {
    inputs: BTreeMap<String, HostTensor<f32>>,
    ids: BTreeMap<String, HostTensor<i32>>,
}

impl PackedBatch {
    /// The `f32` tensors, `key_mask` and `weights`, by name.
    pub fn inputs(&self) -> &BTreeMap<String, HostTensor<f32>> {
        &self.inputs
    }

    /// The `i32` tensors, `ids`, `positions` and `labels`, by name.
    pub fn ids(&self) -> &BTreeMap<String, HostTensor<i32>> {
        &self.ids
    }
}

impl PackedData {
    /// The binding the batches are drawn with.
    pub fn binding(&self) -> &TrainingDataBinding {
        &self.binding
    }

    /// The batch dimension `B`.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// The inputs of training step `k`: the declared tensors of [`mlm_batch`] of the `Train`
    /// split at step `k`, with the run seed and the binding's masking policy. A new batch each
    /// call; the training loop refills one with [`Self::fill_step`].
    pub fn step(&self, k: u64) -> Result<PackedBatch, DataError> {
        let mut batch = PackedBatch::default();
        self.fill_step(k, &mut batch)?;
        Ok(batch)
    }

    /// Replaces the tensors of `batch` with the inputs of training step `k` ([`Self::step`]).
    pub fn fill_step(&self, k: u64, batch: &mut PackedBatch) -> Result<(), DataError> {
        let drawn = self.draw(SplitRole::Train, k)?;
        self.fill(drawn, batch);
        Ok(())
    }

    /// The directory the data was read from.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The number of sequences of the split `role`.
    pub fn sequences(&self, role: SplitRole) -> usize {
        self.packed.sequence_count(role)
    }

    /// [`mlm_batch`] of step `k` of the split `role` with the run seed, the binding's masking
    /// policy and `B` sequences: the training batch of step `k` for `Train`, the held-out batch
    /// `k` for `HeldOut` (story:bert-heldout-evaluation).
    pub fn draw(&self, role: SplitRole, k: u64) -> Result<MlmBatch, DataError> {
        let policy = &self.binding.masking_policy;
        mlm_batch(&self.packed, role, policy, self.seed, k, self.batch).map_err(|e| {
            DataError::Packed {
                dir: self.dir.clone(),
                reason: e.to_string(),
            }
        })
    }

    /// Replaces the tensors of `batch` with the declared tensors of `drawn`, each by its
    /// declaration's name and shape.
    pub fn fill(&self, mut drawn: MlmBatch, batch: &mut PackedBatch) {
        let MlmBatch {
            ids,
            positions,
            key_mask,
            labels,
            weights,
        } = &mut drawn;
        for (tensor, name, shape) in &self.bound {
            let shape = shape.clone();
            match tensor {
                BatchTensor::Ids | BatchTensor::Positions | BatchTensor::Labels => {
                    let data = match tensor {
                        BatchTensor::Ids => std::mem::take(ids),
                        BatchTensor::Positions => std::mem::take(positions),
                        _ => std::mem::take(labels),
                    };
                    batch
                        .ids
                        .insert(name.clone(), HostTensor::from_parts(shape, data));
                }
                BatchTensor::KeyMask | BatchTensor::Weights => {
                    let data = match tensor {
                        BatchTensor::KeyMask => std::mem::take(key_mask),
                        _ => std::mem::take(weights),
                    };
                    batch
                        .inputs
                        .insert(name.clone(), HostTensor::from_parts(shape, data));
                }
            }
        }
    }
}

/// The training data of the packed directory `dir`, whose binding [`source`] read: refused with
/// [`E_DATA_005`], [`E_DATA_006`] and [`E_DATA_003`] as [`bind`] refuses, before any packed file
/// is read; then the packed files are read once ([`mlm::load`]), and a directory whose record
/// does not belong to the binding or holds fewer `Train` sequences than `B` is
/// [`DataError::Packed`].
pub fn packed(
    dir: &Path,
    binding: TrainingDataBinding,
    config: &TrainConfig,
    lowered: &LoweredProgram,
) -> Result<PackedData, DataError> {
    let (batch, bound) = bind(&binding, config, lowered)?;
    let failed = |reason: String| DataError::Packed {
        dir: dir.to_path_buf(),
        reason,
    };
    let packed = mlm::load(dir).map_err(|e| failed(e.to_string()))?;
    if packed.record.binding_id != binding.binding_id || packed.record.seq_len != binding.seq_len {
        return Err(failed(format!(
            "{}: binding_id {} and seq_len {} are not those of {BINDING_FILE}",
            dir.join(mlm::PACKED_FILE).display(),
            packed.record.binding_id,
            packed.record.seq_len
        )));
    }
    let sequences = packed.sequence_count(SplitRole::Train);
    if batch > sequences {
        return Err(failed(format!(
            "a batch of {batch} sequences, more than the {sequences} of the Train split"
        )));
    }
    Ok(PackedData {
        dir: dir.to_path_buf(),
        binding,
        packed,
        batch,
        seed: config.seed,
        bound,
    })
}

/// The batch dimension of `config`'s declarations and the batch tensor each one names, for the
/// sequence length of `binding` (coordinator decision D48). Refused, in this order: with
/// [`E_DATA_005`] at the first declaration in source order that names no batch tensor, or at the
/// block when it declares none; with [`E_DATA_006`] at the first declaration in the order `ids`,
/// `positions`, `key_mask`, `labels`, `weights` when its shape gives no `B` (`[B, S]` gives `B`,
/// `[N]` gives `N / S` when `S` divides `N`); with [`E_DATA_006`] at the first declaration in
/// source order whose dtype is not its tensor's or whose shape is neither `[B, S]` nor `[B·S]`;
/// with [`E_DATA_003`] at a declaration the loss does not read.
#[allow(clippy::type_complexity)]
fn bind(
    binding: &TrainingDataBinding,
    config: &TrainConfig,
    lowered: &LoweredProgram,
) -> Result<(usize, Vec<(BatchTensor, String, Vec<u64>)>), DataError> {
    let seq_len = binding.seq_len;
    let decls = in_source_order(config);
    if decls.is_empty() {
        return Err(DataError::Refused(
            Diagnostic::new(
                E_DATA_005,
                "a run on a packed directory declares no batch tensor",
            )
            .at(config.span.clone())
            .with("batch tensors", batch_tensor_names()),
        ));
    }
    let mut named = Vec::new();
    for &(decl, role) in &decls {
        let Some(tensor) = BatchTensor::of(&decl.name) else {
            return Err(DataError::Refused(
                Diagnostic::new(
                    E_DATA_005,
                    format!(
                        "the {role} `{}` names no tensor of the masked-language-model batch",
                        decl.name
                    ),
                )
                .at(decl.span.clone())
                .with(role, decl.name.clone())
                .with("batch tensors", batch_tensor_names()),
            ));
        };
        named.push((tensor, decl, role));
    }
    let static_shape = |decl: &DataDecl| -> Option<Vec<u64>> {
        decl.ty
            .shape
            .iter()
            .map(|dim| match *dim {
                Dim::Static(n) if n > 0 => Some(n),
                _ => None,
            })
            .collect()
    };
    let (_, first, role) = BatchTensor::ALL
        .iter()
        .find_map(|&t| named.iter().find(|(n, _, _)| *n == t))
        .copied()
        .expect("every declaration names a batch tensor");
    let batch = match static_shape(first).as_deref() {
        Some(&[b, s]) if s == seq_len => Some(b),
        Some(&[n]) if n.is_multiple_of(seq_len) => Some(n / seq_len),
        _ => None,
    };
    let Some(batch) = batch else {
        let dtype = BatchTensor::of(&first.name).expect("named").dtype();
        return Err(mismatch(
            first,
            role,
            format!("Tensor<{dtype}, [B, {seq_len}]> or Tensor<{dtype}, [B·{seq_len}]>"),
        ));
    };
    let rows = batch.checked_mul(seq_len);
    let mut bound = Vec::new();
    for (tensor, decl, role) in named {
        let dtype = tensor.dtype();
        let shape = static_shape(decl);
        let fits = decl.ty.dtype == dtype
            && match shape.as_deref() {
                Some(&[b, s]) => b == batch && s == seq_len,
                Some(&[n]) => Some(n) == rows,
                _ => false,
            };
        let shape = match (fits, shape) {
            (true, Some(shape)) => shape,
            _ => {
                let rows = rows.map_or_else(|| format!("{batch}·{seq_len}"), |n| n.to_string());
                return Err(mismatch(
                    decl,
                    role,
                    format!("Tensor<{dtype}, [{batch}, {seq_len}]> or Tensor<{dtype}, [{rows}]>"),
                ));
            }
        };
        bound.push((tensor, decl.name.clone(), shape));
    }
    refuse_unread(lowered)?;
    let batch = usize::try_from(batch).map_err(|_| InitError::Shape {
        path: first.name.clone(),
    })?;
    Ok((batch, bound))
}

/// The names of the batch tensors, for a diagnostic.
fn batch_tensor_names() -> String {
    BatchTensor::ALL.map(BatchTensor::name).join(", ")
}

/// [`E_DATA_006`] at `decl`, whose batch tensor has the type `expected`.
fn mismatch(decl: &DataDecl, role: &str, expected: String) -> DataError {
    DataError::Refused(
        Diagnostic::new(
            E_DATA_006,
            format!(
                "the {role} `{}` is not of the type of the batch tensor `{}`",
                decl.name, decl.name
            ),
        )
        .at(decl.span.clone())
        .with(role, decl.ty.to_string())
        .with("expected", expected),
    )
}

/// A zero tensor of type `ty`.
fn zeros(name: &str, ty: &TensorType) -> Result<HostTensor<f32>, InitError> {
    let shape = ty
        .shape
        .iter()
        .map(|dim| match *dim {
            Dim::Static(n) if n > 0 => Ok(n),
            _ => Err(InitError::Shape { path: name.into() }),
        })
        .collect::<Result<Vec<u64>, InitError>>()?;
    let count = shape.iter().try_fold(1usize, |acc, &n| {
        usize::try_from(n).ok().and_then(|n| acc.checked_mul(n))
    });
    let count = count.ok_or_else(|| InitError::TooManyElements {
        path: name.into(),
        count: shape
            .iter()
            .fold(1u128, |acc, &n| acc.saturating_mul(u128::from(n))),
    })?;
    Ok(HostTensor::from_parts(shape, vec![0.0; count]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::init::{initialize, standard_normal};
    use modelspec_core::diag::SourceSpan;
    use modelspec_core::hir::HirData;
    use modelspec_core::ir::tensor::{Initializer, TensorProgramBuilder, TensorType};
    use modelspec_core::ir::training::DataDecl;

    fn span(line: u32) -> SourceSpan {
        SourceSpan {
            file: "fit.mspec".into(),
            line,
            column: 5,
        }
    }

    fn decl(name: &str, shape: &[u64], line: u32) -> DataDecl {
        DataDecl {
            name: name.into(),
            ty: TensorType::f32(shape),
            span: span(line),
        }
    }

    fn config(seed: u64, inputs: Vec<DataDecl>, targets: Vec<DataDecl>) -> TrainConfig {
        TrainConfig {
            steps: 500,
            lr: 0.1,
            seed,
            inputs,
            targets,
            span: span(1),
        }
    }

    fn linear_config(seed: u64) -> TrainConfig {
        config(
            seed,
            vec![decl("x", &[8, 4], 3)],
            vec![decl("y", &[8, 1], 4)],
        )
    }

    fn data(role: DataRole, name: &str, shape: &[u64], line: u32) -> HirData {
        HirData {
            role,
            name: name.into(),
            ty: TensorType::f32(shape),
            span: span(line),
        }
    }

    /// The declarations `input x: [8, 4]` (line 3) and `target y: [8, 1]` (line 4).
    fn linear_declarations() -> Vec<HirData> {
        vec![
            data(DataRole::Input, "x", &[8, 4], 3),
            data(DataRole::Target, "y", &[8, 1], 4),
        ]
    }

    /// `mse(matmul(x, w), y)` and its prediction `matmul(x, w)`.
    fn linear() -> LoweredProgram {
        let mut b = TensorProgramBuilder::new();
        let w = b
            .declare_param(
                "LinearReg.w",
                TensorType::f32(&[4, 1]),
                Initializer::XavierUniform,
            )
            .unwrap();
        let x = b.input("x", TensorType::f32(&[8, 4])).unwrap();
        let y = b.input("y", TensorType::f32(&[8, 1])).unwrap();
        let w = b.param_read(w).unwrap();
        let h = b.matmul(x, w).unwrap();
        let loss = b.mse(h, y).unwrap();
        b.output(loss).unwrap();
        LoweredProgram {
            program: b.finish(),
            prediction: h,
            declared_inputs: linear_declarations(),
        }
    }

    /// `mse(b, y)` with prediction `b`: the declared input `x` has no `Input` node, as after
    /// dead-code elimination of a forward that ignores it.
    fn ignores_x() -> LoweredProgram {
        let mut b = TensorProgramBuilder::new();
        let p = b
            .declare_param(
                "Const.b",
                TensorType::f32(&[8, 1]),
                Initializer::XavierUniform,
            )
            .unwrap();
        let y = b.input("y", TensorType::f32(&[8, 1])).unwrap();
        let p = b.param_read(p).unwrap();
        let loss = b.mse(p, y).unwrap();
        b.output(loss).unwrap();
        LoweredProgram {
            program: b.finish(),
            prediction: p,
            declared_inputs: linear_declarations(),
        }
    }

    #[test]
    fn the_target_is_the_teacher_prediction_for_standard_normal_inputs() {
        let forward = linear();
        let data = synthetic(&linear_config(42), &forward).unwrap();
        let x_ty = TensorType::f32(&[8, 4]);
        let x = standard_normal(42, Stream::Data, "x", &x_ty).unwrap();
        let teacher = initialize(&forward.program, 42, Stream::Teacher).unwrap();
        assert_eq!(data.teacher, teacher);
        assert_eq!(data.inputs.keys().collect::<Vec<_>>(), vec!["x", "y"]);
        assert_eq!(data.inputs["x"], x);
        let w = teacher[&ParamId(0)].data();
        let y = data.inputs["y"].data();
        assert_eq!(data.inputs["y"].shape(), &[8, 1]);
        assert_eq!(y.len(), 8);
        for (i, (row, &got)) in x.data().chunks(4).zip(y).enumerate() {
            let expected = row[1..]
                .iter()
                .zip(&w[1..])
                .fold(row[0] * w[0], |acc, (&a, &b)| a.mul_add(b, acc));
            assert_eq!(got, expected, "row {i}");
        }
    }

    #[test]
    fn the_teacher_is_not_the_initial_state() {
        let forward = linear();
        let data = synthetic(&linear_config(42), &forward).unwrap();
        let init = initialize(&forward.program, 42, Stream::Init).unwrap();
        assert_ne!(data.teacher, init);
    }

    #[test]
    fn the_same_seed_makes_the_same_data() {
        let forward = linear();
        let a = synthetic(&linear_config(42), &forward).unwrap();
        let b = synthetic(&linear_config(42), &forward).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn another_seed_makes_other_data() {
        let forward = linear();
        let a = synthetic(&linear_config(42), &forward).unwrap();
        let b = synthetic(&linear_config(43), &forward).unwrap();
        assert_ne!(a.inputs["x"], b.inputs["x"]);
        assert_ne!(a.teacher, b.teacher);
    }

    fn refused(result: Result<SyntheticData, DataError>) -> Diagnostic {
        match result {
            Err(DataError::Refused(d)) => d,
            other => panic!("not refused: {other:?}"),
        }
    }

    #[test]
    fn a_second_input_is_refused() {
        let forward = linear();
        let c = config(
            42,
            vec![decl("x", &[8, 4], 3), decl("z", &[8, 4], 4)],
            vec![decl("y", &[8, 1], 5)],
        );
        let d = refused(synthetic(&c, &forward));
        assert_eq!(d.code, E_DATA_001);
        assert_eq!(d.span, Some(span(1)));
        assert_eq!(
            d.to_string(),
            "E-DATA-001: M0 synthetic data needs exactly one input and one target\n  \
             fit.mspec:1:5\n  \
             inputs: 2\n  \
             targets: 1"
        );
    }

    #[test]
    fn a_missing_target_is_refused() {
        let forward = linear();
        let c = config(42, vec![decl("x", &[8, 4], 3)], vec![]);
        let d = refused(synthetic(&c, &forward));
        assert_eq!(d.code, E_DATA_001);
        assert!(d.to_string().ends_with("inputs: 1\n  targets: 0"), "{d}");
    }

    #[test]
    fn a_missing_input_or_a_second_target_is_refused() {
        let forward = linear();
        for c in [
            config(42, vec![], vec![decl("y", &[8, 1], 4)]),
            config(
                42,
                vec![decl("x", &[8, 4], 3)],
                vec![decl("y", &[8, 1], 4), decl("t", &[8, 1], 5)],
            ),
        ] {
            let d = refused(synthetic(&c, &forward));
            assert_eq!(d.code, E_DATA_001, "{d}");
        }
    }

    #[test]
    fn a_target_not_of_the_model_output_type_is_refused() {
        let forward = linear();
        let c = config(42, vec![decl("x", &[8, 4], 3)], vec![decl("y", &[8, 2], 4)]);
        let d = refused(synthetic(&c, &forward));
        assert_eq!(
            d.to_string(),
            "E-DATA-002: the target is not of the model output type\n  \
             fit.mspec:4:5\n  \
             target: Tensor<f32, [8, 2]>\n  \
             model output: Tensor<f32, [8, 1]>"
        );
    }

    #[test]
    fn an_undefined_prediction_is_an_error_not_a_panic() {
        let mut forward = linear();
        forward.prediction = ValueId(99);
        assert_eq!(
            synthetic(&linear_config(42), &forward),
            Err(DataError::UndefinedPrediction(ValueId(99)))
        );
    }

    #[test]
    fn an_input_the_program_does_not_read_is_refused_at_its_declaration() {
        let d = refused(synthetic(&linear_config(42), &ignores_x()));
        assert_eq!(d.code, E_DATA_003);
        assert_eq!(d.span, Some(span(3)));
        assert_eq!(
            d.to_string(),
            "E-DATA-003: the input `x` is not read by the loss\n  \
             fit.mspec:3:5\n  \
             input: x"
        );
    }

    #[test]
    fn a_target_the_program_does_not_read_is_refused_at_its_declaration() {
        let mut forward = linear();
        let y = forward.program.inputs[1];
        let Some(node) = forward.program.nodes.iter_mut().find(|n| n.id == y) else {
            panic!("y is a node");
        };
        node.op = TensorOp::Input { name: "t".into() };
        let d = refused(synthetic(&linear_config(42), &forward));
        assert_eq!(
            d.to_string(),
            "E-DATA-003: the target `y` is not read by the loss\n  \
             fit.mspec:4:5\n  \
             target: y"
        );
    }

    #[test]
    fn the_count_and_type_refusals_come_before_the_unread_refusal() {
        let forward = ignores_x();
        let two = config(
            42,
            vec![decl("x", &[8, 4], 3), decl("z", &[8, 4], 4)],
            vec![decl("y", &[8, 1], 5)],
        );
        assert_eq!(refused(synthetic(&two, &forward)).code, E_DATA_001);
        let wide = config(42, vec![decl("x", &[8, 4], 3)], vec![decl("y", &[8, 2], 4)]);
        assert_eq!(refused(synthetic(&wide, &forward)).code, E_DATA_002);
    }

    const EMBEDDING: &str = "model M {
    param table: Tensor<f32, [6, 3]> = xavier_uniform();
    fn forward(ids: Tensor<i32, [4]>) -> Tensor<f32, [4, 3]> {
        return embedding(table, ids);
    }
}
train T {
    model = M;
    input ids: Tensor<i32, [4]>;
    target y: Tensor<f32, [4, 3]>;
    loss = mse(model.forward(ids), y);
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 0;
    device = cpu;
}
";

    /// Synthetic inputs are standard normal `f32` draws and dense files hold `f32`: neither
    /// supplies an `i32` declaration, refused at it with E-DATA-004 before any file is read.
    #[test]
    fn an_i32_declaration_is_refused_with_e_data_004() {
        let compiled = modelspec_core::compile::compile_training("e.mspec", EMBEDDING).unwrap();
        let d = refused(synthetic(&compiled.config, &compiled.forward));
        assert_eq!(
            d.to_string(),
            "E-DATA-004: M0 synthetic data cannot supply the i32 input `ids`\n  \
             e.mspec:9:5\n  \
             input: Tensor<i32, [4]>"
        );
        let missing = Path::new("no-such-directory");
        let Err(DataError::Refused(d)) = dense(missing, &compiled.config, &compiled.forward) else {
            panic!("dense data is refused");
        };
        assert_eq!(d.code, E_DATA_004);
        assert!(d.message.starts_with("dense data cannot supply"), "{d}");
    }
}
