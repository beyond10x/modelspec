//! IR verifier: structural and type invariants checked after every pass (DESIGN.md § 3, § 3.2).
//!
//! A verification failure is a compiler bug, not a user error, so it is a [`VerifyError`] and not
//! a [`crate::diag::Diagnostic`] (DESIGN.md § 8.1).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use thiserror::Error;

use crate::ir::tensor::{Dim, ParamId, TensorOp, TensorProgram, TensorType, ValueId};
use crate::ir::training::{Subgraph, TrainingProgram};
use crate::ops::{self, OpError};

/// Where a type invariant failed: a value's type or a parameter table entry's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    Value(ValueId),
    Param(ParamId),
}

impl fmt::Display for Site {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Site::Value(v) => write!(f, "{v}"),
            Site::Param(p) => write!(f, "parameter {p}"),
        }
    }
}

/// A Tensor IR invariant that does not hold.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum VerifyError {
    #[error("{id} is defined more than once")]
    DuplicateValue { id: ValueId },
    #[error("{node} uses {operand} before its definition")]
    UseBeforeDefinition { node: ValueId, operand: ValueId },
    #[error("{node} uses {operand}, which no node defines")]
    UndefinedOperand { node: ValueId, operand: ValueId },
    #[error("{node} reads undeclared parameter {param}")]
    UndeclaredParam { node: ValueId, param: ParamId },
    #[error("{node}: {source}")]
    Op { node: ValueId, source: OpError },
    #[error("{node} is declared {declared} but its operation gives {inferred}")]
    TypeMismatch {
        node: ValueId,
        declared: TensorType,
        inferred: TensorType,
    },
    #[error("{site} has unresolved dimension ?{symbol}")]
    UnresolvedDimension { site: Site, symbol: u32 },
    #[error("{site} has a zero dimension")]
    ZeroDimension { site: Site },
    #[error("{site} element or byte count overflows u64")]
    CountOverflow { site: Site },
    #[error("parameters {first} and {second} share canonical path `{path}`")]
    DuplicateParamPath {
        path: String,
        first: ParamId,
        second: ParamId,
    },
    #[error("Input nodes {first} and {second} share name `{name}`")]
    DuplicateInputName {
        name: String,
        first: ValueId,
        second: ValueId,
    },
    #[error("input {value} is not an Input node")]
    NotAnInput { value: ValueId },
    #[error("input {value} is listed more than once")]
    DuplicateInputEntry { value: ValueId },
    #[error("Input node {value} is not listed in the program inputs")]
    InputNotListed { value: ValueId },
    #[error("program inputs {got:?} are not in node order {expected:?}")]
    InputsOutOfOrder {
        expected: Vec<ValueId>,
        got: Vec<ValueId>,
    },
    #[error("output {output} is not defined")]
    UndefinedOutput { output: ValueId },
}

/// Checks every Tensor IR invariant of `program`, reporting the first failure.
///
/// Every invariant [`crate::ir::tensor::TensorProgramBuilder`] enforces is checked here too, so a
/// program made or changed by other means is held to the same rules.
///
/// Order: the parameter table (concrete types, unique canonical paths), then value uniqueness,
/// then each node in definition order (operands defined earlier, parameter declared, concrete
/// type, registry rule), then the input list (exactly the Input nodes, each once, in node order,
/// names unique) and the output list.
pub fn verify(program: &TensorProgram) -> Result<(), VerifyError> {
    let mut paths: BTreeMap<&str, ParamId> = BTreeMap::new();
    for (&id, param) in &program.params {
        check_concrete(Site::Param(id), &param.ty)?;
        if let Some(&first) = paths.get(param.path.as_str()) {
            return Err(VerifyError::DuplicateParamPath {
                path: param.path.clone(),
                first,
                second: id,
            });
        }
        paths.insert(&param.path, id);
    }

    let mut position = BTreeMap::new();
    for (i, node) in program.nodes.iter().enumerate() {
        if position.insert(node.id, i).is_some() {
            return Err(VerifyError::DuplicateValue { id: node.id });
        }
    }

    for (i, node) in program.nodes.iter().enumerate() {
        let mut operand_types = Vec::with_capacity(node.operands.len());
        for &operand in &node.operands {
            match position.get(&operand) {
                Some(&j) if j < i => operand_types.push(&program.nodes[j].ty),
                Some(_) => {
                    return Err(VerifyError::UseBeforeDefinition {
                        node: node.id,
                        operand,
                    });
                }
                None => {
                    return Err(VerifyError::UndefinedOperand {
                        node: node.id,
                        operand,
                    });
                }
            }
        }
        let declared = match &node.op {
            TensorOp::Input { .. } => &node.ty,
            TensorOp::ParamRead { param } => {
                &program
                    .params
                    .get(param)
                    .ok_or(VerifyError::UndeclaredParam {
                        node: node.id,
                        param: *param,
                    })?
                    .ty
            }
            _ => &node.ty,
        };
        check_concrete(Site::Value(node.id), &node.ty)?;
        let inferred = ops::infer(&node.op, &operand_types, Some(declared)).map_err(|source| {
            VerifyError::Op {
                node: node.id,
                source,
            }
        })?;
        if inferred != node.ty {
            return Err(VerifyError::TypeMismatch {
                node: node.id,
                declared: node.ty.clone(),
                inferred,
            });
        }
    }

    check_inputs(program)?;
    for &output in &program.outputs {
        if !position.contains_key(&output) {
            return Err(VerifyError::UndefinedOutput { output });
        }
    }
    Ok(())
}

/// `program.inputs` lists exactly the Input nodes, each once, in node order; their names are
/// unique.
fn check_inputs(program: &TensorProgram) -> Result<(), VerifyError> {
    let mut names: BTreeMap<&str, ValueId> = BTreeMap::new();
    let mut expected = Vec::new();
    for node in &program.nodes {
        if let TensorOp::Input { name } = &node.op {
            if let Some(&first) = names.get(name.as_str()) {
                return Err(VerifyError::DuplicateInputName {
                    name: name.clone(),
                    first,
                    second: node.id,
                });
            }
            names.insert(name, node.id);
            expected.push(node.id);
        }
    }
    let mut listed = BTreeSet::new();
    for &value in &program.inputs {
        if !expected.contains(&value) {
            return Err(VerifyError::NotAnInput { value });
        }
        if !listed.insert(value) {
            return Err(VerifyError::DuplicateInputEntry { value });
        }
    }
    if let Some(&value) = expected.iter().find(|v| !listed.contains(v)) {
        return Err(VerifyError::InputNotListed { value });
    }
    if program.inputs != expected {
        return Err(VerifyError::InputsOutOfOrder {
            expected,
            got: program.inputs.clone(),
        });
    }
    Ok(())
}

/// No symbolic or zero dimension, and element and byte counts fit `u64` (DESIGN.md § 2.2).
fn check_concrete(site: Site, ty: &TensorType) -> Result<(), VerifyError> {
    let mut elements: u64 = 1;
    for dim in &ty.shape {
        let n = match *dim {
            Dim::Static(n) => n,
            Dim::Symbol(symbol) => return Err(VerifyError::UnresolvedDimension { site, symbol }),
        };
        if n == 0 {
            return Err(VerifyError::ZeroDimension { site });
        }
        elements = elements
            .checked_mul(n)
            .ok_or(VerifyError::CountOverflow { site })?;
    }
    elements
        .checked_mul(ty.dtype.size_bytes())
        .ok_or(VerifyError::CountOverflow { site })?;
    Ok(())
}

/// A Training IR invariant that does not hold (DESIGN.md § 3.1, § 4).
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TrainingVerifyError {
    #[error(transparent)]
    Tensor(#[from] VerifyError),
    #[error("{node} has no subgraph tag")]
    UntaggedNode { node: ValueId },
    #[error("a subgraph tag names {value}, which no node defines")]
    TagWithoutNode { value: ValueId },
    #[error("loss {loss} is not defined")]
    UndefinedLoss { loss: ValueId },
    #[error("loss {loss} is tagged {tag}, not loss")]
    LossNotInLossSubgraph { loss: ValueId, tag: Subgraph },
    #[error("loss {loss} is {ty}, not rank 0")]
    LossNotScalar { loss: ValueId, ty: TensorType },
    #[error("a gradient is given for undeclared parameter {param}")]
    GradientForUndeclaredParam { param: ParamId },
    #[error("the gradient of parameter {param} is {value}, which no node defines")]
    UndefinedGradient { param: ParamId, value: ValueId },
    #[error("the gradient of parameter {param} is {got}; the parameter is {expected}")]
    GradientType {
        param: ParamId,
        expected: TensorType,
        got: TensorType,
    },
    #[error("{node} is tagged loss but is neither the loss nor the one backward seed `1.0`")]
    StrayLossTag { node: ValueId },
    #[error("gradient node {node} does not depend on the backward seed")]
    GradientIndependentOfLoss { node: ValueId },
    #[error("{node} depends on the backward seed but is tagged forward")]
    ForwardDependsOnSeed { node: ValueId },
    #[error(
        "the gradient of parameter {param} is {value}, which does not depend on the backward seed"
    )]
    GradientNotFromSeed { param: ParamId, value: ValueId },
    #[error("parameter {param} is read by the loss but has no gradient")]
    MissingGradient { param: ParamId },
    #[error("a gradient is given for parameter {param}, which the loss does not read")]
    GradientForUnreachableParam { param: ParamId },
    #[error("outputs {got:?} are not the loss then the gradients in parameter order {expected:?}")]
    OutputLayout {
        expected: Vec<ValueId>,
        got: Vec<ValueId>,
    },
}

/// Checks every Training IR invariant of `training`, reporting the first failure.
///
/// Order: the Tensor IR invariants of [`TrainingProgram::program`] ([`verify`]); every node
/// tagged and every tag naming a node; the loss defined, tagged [`Subgraph::Loss`] and rank 0;
/// no node tagged `Loss` but the loss and at most one backward seed, an operand-free
/// `Scalar(1.0)`; every node that depends on the seed through its operands tagged
/// [`Subgraph::Gradient`] and every other `Gradient` node refused; each gradient for a declared
/// parameter, defined, of its parameter's type, and the seed or a value depending on it; a
/// gradient for exactly the parameters the loss reads; the outputs the loss then the gradients
/// in `ParamId` order.
///
/// Dependence is on the seed alone, never on a tag: a node tagged `Loss` by mistake cannot make
/// a node depend on the loss.
pub fn verify_training(training: &TrainingProgram) -> Result<(), TrainingVerifyError> {
    let program = &training.program;
    verify(program)?;

    let defined: BTreeSet<ValueId> = program.nodes.iter().map(|n| n.id).collect();
    for node in &program.nodes {
        if !training.subgraphs.contains_key(&node.id) {
            return Err(TrainingVerifyError::UntaggedNode { node: node.id });
        }
    }
    if let Some(&value) = training.subgraphs.keys().find(|v| !defined.contains(v)) {
        return Err(TrainingVerifyError::TagWithoutNode { value });
    }

    let loss = training.loss;
    let loss_node = program
        .node(loss)
        .ok_or(TrainingVerifyError::UndefinedLoss { loss })?;
    let tag = training.subgraphs[&loss];
    if tag != Subgraph::Loss {
        return Err(TrainingVerifyError::LossNotInLossSubgraph { loss, tag });
    }
    if loss_node.ty.rank() != 0 {
        return Err(TrainingVerifyError::LossNotScalar {
            loss,
            ty: loss_node.ty.clone(),
        });
    }

    let mut seed: Option<ValueId> = None;
    let mut on_seed: BTreeSet<ValueId> = BTreeSet::new();
    for node in &program.nodes {
        let tag = training.subgraphs[&node.id];
        if tag == Subgraph::Loss && node.id != loss {
            let seed_op = node.op == TensorOp::Scalar(1.0) && node.operands.is_empty();
            if !seed_op || seed.is_some() {
                return Err(TrainingVerifyError::StrayLossTag { node: node.id });
            }
            seed = Some(node.id);
            on_seed.insert(node.id);
            continue;
        }
        let depends = node.operands.iter().any(|o| on_seed.contains(o));
        match tag {
            Subgraph::Gradient if !depends => {
                return Err(TrainingVerifyError::GradientIndependentOfLoss { node: node.id });
            }
            Subgraph::Forward if depends => {
                return Err(TrainingVerifyError::ForwardDependsOnSeed { node: node.id });
            }
            _ => {}
        }
        if depends {
            on_seed.insert(node.id);
        }
    }

    for (&param, &value) in &training.gradients {
        let decl = program
            .params
            .get(&param)
            .ok_or(TrainingVerifyError::GradientForUndeclaredParam { param })?;
        let node = program
            .node(value)
            .ok_or(TrainingVerifyError::UndefinedGradient { param, value })?;
        if node.ty != decl.ty {
            return Err(TrainingVerifyError::GradientType {
                param,
                expected: decl.ty.clone(),
                got: node.ty.clone(),
            });
        }
        if !on_seed.contains(&value) {
            return Err(TrainingVerifyError::GradientNotFromSeed { param, value });
        }
    }

    let read_by_loss = params_read_by(program, loss);
    if let Some(&param) = read_by_loss
        .iter()
        .find(|p| !training.gradients.contains_key(p))
    {
        return Err(TrainingVerifyError::MissingGradient { param });
    }
    if let Some(&param) = training
        .gradients
        .keys()
        .find(|p| !read_by_loss.contains(p))
    {
        return Err(TrainingVerifyError::GradientForUnreachableParam { param });
    }

    let mut expected = vec![loss];
    expected.extend(training.gradients.values());
    if program.outputs != expected {
        return Err(TrainingVerifyError::OutputLayout {
            expected,
            got: program.outputs.clone(),
        });
    }
    Ok(())
}

/// The parameters `value` reads, through its operands, in `ParamId` order.
fn params_read_by(program: &TensorProgram, value: ValueId) -> BTreeSet<ParamId> {
    let mut live = BTreeSet::from([value]);
    let mut params = BTreeSet::new();
    for node in program.nodes.iter().rev() {
        if !live.contains(&node.id) {
            continue;
        }
        live.extend(node.operands.iter().copied());
        if let TensorOp::ParamRead { param } = node.op {
            params.insert(param);
        }
    }
    params
}
