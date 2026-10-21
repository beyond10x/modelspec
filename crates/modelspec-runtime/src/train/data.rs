//! M0 synthetic teacher data (DESIGN.md § 9 gate 4).
//!
//! M0 has no data file format (DESIGN.md § 12 "Data interface" is open). The one `input` is a
//! standard normal tensor on [`Stream::Data`], drawn by Box-Muller and named by the input's name;
//! the one `target` is the model's prediction for it under teacher parameters drawn by each
//! parameter's initializer on [`Stream::Teacher`]. The teacher parameters are the known solution
//! training should approach. The prediction is read as the intermediate value `prediction` of
//! the forward program, run on the CPU interpreter with a zero target.
//!
//! A declaration the loss does not read has no `Input` node after dead-code elimination, so the
//! program could not take a value for it; it is refused at its declaration ([`E_DATA_003`])
//! before anything runs.

use std::collections::{BTreeMap, BTreeSet};

use modelspec_core::diag::{Code, Diagnostic};
use modelspec_core::hir::DataRole;
use modelspec_core::ir::tensor::{Dim, ParamId, TensorOp, TensorType, ValueId};
use modelspec_core::ir::training::TrainConfig;
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
    let prediction = lowered.prediction;
    let forward = &lowered.program;
    let ([input], [target]) = (&config.inputs[..], &config.targets[..]) else {
        return Err(DataError::Refused(
            Diagnostic::new(
                E_DATA_001,
                "M0 synthetic data needs exactly one input and one target",
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
