//! Tensor IR: the forward computation graph (DESIGN.md § 3.1, § 3.2).
//!
//! A [`TensorProgram`] is a typed SSA graph: each [`TensorNode`] defines one value, and the node
//! list is in definition order. Programs are made with [`TensorProgramBuilder`], which types every
//! result through the operator registry ([`crate::ops`]); [`crate::ir::verify`] checks a program
//! made or changed by any other means.

use std::collections::BTreeMap;
use std::fmt;

use thiserror::Error;

use crate::diag::SourceSpan;
use crate::ops::{self, OpError};

/// The identity of an SSA value, printed `%n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(pub u32);

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// The identity of an entry in a program's parameter table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParamId(pub u32);

impl fmt::Display for ParamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Element type. `f32` is the only M0 dtype (DESIGN.md § 2.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
}

impl DType {
    /// Bytes per element.
    pub const fn size_bytes(self) -> u64 {
        match self {
            DType::F32 => 4,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => f.write_str("f32"),
        }
    }
}

/// One dimension: a compile-time size, or a generic dimension not yet resolved.
///
/// A verified program has no [`Dim::Symbol`] (DESIGN.md § 2.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dim {
    Static(u64),
    Symbol(u32),
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Static(n) => write!(f, "{n}"),
            Dim::Symbol(s) => write!(f, "?{s}"),
        }
    }
}

/// A tensor type, printed `Tensor<f32, [8, 1]>`; rank 0 is a scalar.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorType {
    pub dtype: DType,
    pub shape: Vec<Dim>,
}

impl TensorType {
    /// An `f32` tensor of static shape `shape`.
    pub fn f32(shape: &[u64]) -> Self {
        Self {
            dtype: DType::F32,
            shape: shape.iter().map(|&n| Dim::Static(n)).collect(),
        }
    }

    /// The rank-0 `f32` type.
    pub fn f32_scalar() -> Self {
        Self::f32(&[])
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor<{}, [", self.dtype)?;
        for (i, dim) in self.shape.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{dim}")?;
        }
        f.write_str("]>")
    }
}

/// A Tensor IR operation. Operand counts and typing rules are in [`crate::ops`].
#[derive(Clone, Debug, PartialEq)]
pub enum TensorOp {
    /// A named program input; its type is the node's declared type.
    Input {
        name: String,
    },
    /// Reads a parameter; its type is the parameter table's.
    ParamRead {
        param: ParamId,
    },
    /// An `f32` constant of rank 0.
    Scalar(f32),
    /// Same shapes, or one rank-0 operand.
    Add,
    Sub,
    Mul,
    /// `[M, K] x [K, N] -> [M, N]`.
    MatMul,
    Relu,
    /// `[M, N] -> [N, M]`.
    Transpose,
    /// Any shape to rank 0.
    ReduceSum,
    ReduceMean,
    /// `sum((a - b)^2) / element_count(a)` for identical shapes, rank 0.
    Mse,
    /// Internal gradient op `where(A > 0, G, 0)` over operands `A, G` of one shape (DESIGN.md § 4).
    ReluGrad,
    /// Internal gradient op: a rank-0 operand broadcast to `shape` (DESIGN.md § 4).
    BroadcastScalar {
        shape: Vec<Dim>,
    },
}

impl TensorOp {
    /// The op's name as the printer and the registry errors show it.
    pub fn name(&self) -> &'static str {
        match self {
            TensorOp::Input { .. } => "Input",
            TensorOp::ParamRead { .. } => "ParamRead",
            TensorOp::Scalar(_) => "Scalar",
            TensorOp::Add => "Add",
            TensorOp::Sub => "Sub",
            TensorOp::Mul => "Mul",
            TensorOp::MatMul => "MatMul",
            TensorOp::Relu => "Relu",
            TensorOp::Transpose => "Transpose",
            TensorOp::ReduceSum => "ReduceSum",
            TensorOp::ReduceMean => "ReduceMean",
            TensorOp::Mse => "Mse",
            TensorOp::ReluGrad => "ReluGrad",
            TensorOp::BroadcastScalar { .. } => "BroadcastScalar",
        }
    }
}

/// One SSA definition.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorNode {
    pub id: ValueId,
    pub op: TensorOp,
    pub operands: Vec<ValueId>,
    pub ty: TensorType,
    pub source_span: Option<SourceSpan>,
}

/// How a parameter's initial value is drawn (DESIGN.md § 2.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Initializer {
    XavierUniform,
}

impl fmt::Display for Initializer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Initializer::XavierUniform => f.write_str("xavier_uniform"),
        }
    }
}

/// A parameter table entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamDecl {
    /// Stable canonical path such as `LinearReg.w` (DESIGN.md § 2.3).
    pub path: String,
    pub ty: TensorType,
    pub init: Initializer,
}

/// A Tensor IR program.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TensorProgram {
    /// Definitions in order: an operand is defined by an earlier node.
    pub nodes: Vec<TensorNode>,
    pub params: BTreeMap<ParamId, ParamDecl>,
    /// The program's `Input` nodes, in declaration order; each carries its name.
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
}

impl TensorProgram {
    /// The node defining `id`.
    pub fn node(&self, id: ValueId) -> Option<&TensorNode> {
        self.nodes.iter().find(|n| n.id == id)
    }
}

/// Why the builder refused a node.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BuildError {
    #[error("unknown value {0}")]
    UnknownValue(ValueId),
    #[error("undeclared parameter {0}")]
    UndeclaredParam(ParamId),
    #[error("duplicate input name `{0}`")]
    DuplicateInput(String),
    #[error("duplicate parameter path `{0}`")]
    DuplicateParamPath(String),
    #[error(transparent)]
    Op(#[from] OpError),
}

/// Builds a [`TensorProgram`], typing each result through the operator registry.
#[derive(Debug, Default)]
pub struct TensorProgramBuilder {
    program: TensorProgram,
    types: BTreeMap<ValueId, TensorType>,
}

impl TensorProgramBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a parameter table entry; paths are unique.
    pub fn declare_param(
        &mut self,
        path: impl Into<String>,
        ty: TensorType,
        init: Initializer,
    ) -> Result<ParamId, BuildError> {
        let path = path.into();
        if self.program.params.values().any(|p| p.path == path) {
            return Err(BuildError::DuplicateParamPath(path));
        }
        let id = ParamId(self.program.params.len() as u32);
        self.program.params.insert(id, ParamDecl { path, ty, init });
        Ok(id)
    }

    /// Adds a named input of type `ty`; names are unique.
    pub fn input(
        &mut self,
        name: impl Into<String>,
        ty: TensorType,
    ) -> Result<ValueId, BuildError> {
        self.input_at(name, ty, None)
    }

    /// [`Self::input`] with a source span.
    pub fn input_at(
        &mut self,
        name: impl Into<String>,
        ty: TensorType,
        span: Option<SourceSpan>,
    ) -> Result<ValueId, BuildError> {
        let name = name.into();
        let taken = self.program.inputs.iter().any(|&v| {
            matches!(&self.program.node(v).map(|n| &n.op), Some(TensorOp::Input { name: n }) if *n == name)
        });
        if taken {
            return Err(BuildError::DuplicateInput(name));
        }
        let id = self.define(TensorOp::Input { name }, Vec::new(), ty, span);
        self.program.inputs.push(id);
        Ok(id)
    }

    pub fn param_read(&mut self, param: ParamId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::ParamRead { param }, &[])
    }

    pub fn scalar(&mut self, value: f32) -> ValueId {
        self.define(
            TensorOp::Scalar(value),
            Vec::new(),
            TensorType::f32_scalar(),
            None,
        )
    }

    pub fn add(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::Add, &[a, b])
    }

    pub fn sub(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::Sub, &[a, b])
    }

    pub fn mul(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::Mul, &[a, b])
    }

    pub fn matmul(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::MatMul, &[a, b])
    }

    pub fn relu(&mut self, a: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::Relu, &[a])
    }

    pub fn transpose(&mut self, a: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::Transpose, &[a])
    }

    pub fn reduce_sum(&mut self, a: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::ReduceSum, &[a])
    }

    pub fn reduce_mean(&mut self, a: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::ReduceMean, &[a])
    }

    pub fn mse(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::Mse, &[a, b])
    }

    /// `where(a > 0, g, 0)`.
    pub fn relu_grad(&mut self, a: ValueId, g: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::ReluGrad, &[a, g])
    }

    pub fn broadcast_scalar(&mut self, a: ValueId, shape: Vec<Dim>) -> Result<ValueId, BuildError> {
        self.push(TensorOp::BroadcastScalar { shape }, &[a])
    }

    /// Adds any op except `Input`, typed by the registry.
    pub fn push(&mut self, op: TensorOp, operands: &[ValueId]) -> Result<ValueId, BuildError> {
        self.push_at(op, operands, None)
    }

    /// [`Self::push`] with a source span.
    pub fn push_at(
        &mut self,
        op: TensorOp,
        operands: &[ValueId],
        span: Option<SourceSpan>,
    ) -> Result<ValueId, BuildError> {
        let mut operand_types = Vec::with_capacity(operands.len());
        for &v in operands {
            operand_types.push(self.types.get(&v).ok_or(BuildError::UnknownValue(v))?);
        }
        let declared = match &op {
            TensorOp::ParamRead { param } => Some(
                &self
                    .program
                    .params
                    .get(param)
                    .ok_or(BuildError::UndeclaredParam(*param))?
                    .ty,
            ),
            _ => None,
        };
        let ty = ops::infer(&op, &operand_types, declared)?;
        Ok(self.define(op, operands.to_vec(), ty, span))
    }

    /// Marks `value` as a program output.
    pub fn output(&mut self, value: ValueId) -> Result<(), BuildError> {
        if !self.types.contains_key(&value) {
            return Err(BuildError::UnknownValue(value));
        }
        self.program.outputs.push(value);
        Ok(())
    }

    /// The type of a value defined so far.
    pub fn type_of(&self, value: ValueId) -> Option<&TensorType> {
        self.types.get(&value)
    }

    pub fn finish(self) -> TensorProgram {
        self.program
    }

    fn define(
        &mut self,
        op: TensorOp,
        operands: Vec<ValueId>,
        ty: TensorType,
        source_span: Option<SourceSpan>,
    ) -> ValueId {
        let id = ValueId(self.program.nodes.len() as u32);
        self.types.insert(id, ty.clone());
        self.program.nodes.push(TensorNode {
            id,
            op,
            operands,
            ty,
            source_span,
        });
        id
    }
}
