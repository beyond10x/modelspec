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
//! before anything runs.

use std::collections::{BTreeMap, BTreeSet};
use std::io::ErrorKind;
use std::path::Path;

use modelspec_core::diag::{Code, Diagnostic};
use modelspec_core::hir::DataRole;
use modelspec_core::ir::tensor::{Dim, ParamId, TensorOp, TensorType, ValueId};
use modelspec_core::ir::training::{DataDecl, TrainConfig};
use modelspec_core::lower::LoweredProgram;
use thiserror::Error;

use super::init::{InitError, Stream, initialize, standard_normal};
use crate::cpu::{HostTensor, RuntimeError, evaluate_all};

/// The `train` block does not declare exactly one `input` and one `target`.
pub const E_DATA_001: Code = Code::new("E-DATA-001");
/// The `target` is not of the model output type.
pub const E_DATA_002: Code = Code::new("E-DATA-002");
/// A declared `input` or `target` is not read by the loss.
pub const E_DATA_003: Code = Code::new("E-DATA-003");
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
}

/// The synthetic data of `config` for the forward program `lowered`, whose value
/// `lowered.prediction` is the model output.
///
/// Refused, in this order, with [`E_DATA_001`] unless the block declares exactly one `input` and
/// one `target`, with [`E_DATA_002`] unless the target has the type of the prediction, and with
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
    Ok((input, target))
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
    /// `(k·B + i) mod N` without one, for `i` in `0..B`.
    pub fn step(&self, k: u64) -> BTreeMap<String, HostTensor<f32>> {
        let first = k as u128 * self.batch as u128;
        let position = |i: usize| first + i as u128;
        let samples: Vec<usize> = (0..self.batch)
            .map(|i| match &self.order {
                Some(order) => order[(position(i) % order.len() as u128) as usize] as usize,
                None => (position(i) % self.samples as u128) as usize,
            })
            .collect();
        self.gather(&samples)
    }

    /// The inputs of eval batch `b`: the samples `b·B .. (b+1)·B`.
    pub fn eval_batch(&self, b: usize) -> BTreeMap<String, HostTensor<f32>> {
        let samples: Vec<usize> = (b * self.batch..(b + 1) * self.batch).collect();
        self.gather(&samples)
    }

    fn gather(&self, samples: &[usize]) -> BTreeMap<String, HostTensor<f32>> {
        self.tensors
            .iter()
            .map(|t| {
                let mut data = Vec::with_capacity(samples.len() * t.sample);
                for &s in samples {
                    data.extend_from_slice(&t.values[s * t.sample..(s + 1) * t.sample]);
                }
                (
                    t.name.clone(),
                    HostTensor::from_parts(t.shape.clone(), data),
                )
            })
            .collect()
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
                .fold(row[0] * w[0], |acc, (&a, &b)| acc + a * b);
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
}
