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

/// Element type (DESIGN.md § 2.2). Every value is `f32` except row ids: an `i32` tensor is only
/// a program input, and only the ids operand of `Embedding` and `EmbeddingGrad` reads one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    I32,
}

impl DType {
    /// Bytes per element.
    pub const fn size_bytes(self) -> u64 {
        match self {
            DType::F32 | DType::I32 => 4,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => f.write_str("f32"),
            DType::I32 => f.write_str("i32"),
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

    /// An `i32` tensor of static shape `shape`: row ids.
    pub fn i32(shape: &[u64]) -> Self {
        Self {
            dtype: DType::I32,
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
    /// `table: f32[V, H]`, `ids: i32[N] -> f32[N, H]`: row `n` of the result is row `ids[n]` of
    /// the table.
    Embedding,
    /// Internal gradient op over operands `ids: i32[N]`, `G: f32[N, H]`: a zero `[rows, H]`
    /// tensor with row `G[n]` added to row `ids[n]`, in ascending `n` (DESIGN.md § 4).
    EmbeddingGrad {
        rows: Dim,
    },
    /// `x: f32[N, H]`, `b: f32[H] -> f32[N, H]`: `b` added to every row of `x`.
    AddRows,
    /// Internal gradient op `f32[N, H] -> f32[H]`: the rows summed in ascending order, starting
    /// from the first row (DESIGN.md § 4).
    SumRows,
    /// The tanh form of GELU, `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`, element by
    /// element on any shape (DESIGN.md § 4).
    Gelu,
    /// Internal gradient op over operands `A, G` of one shape: the derivative of [`Self::Gelu`]
    /// at `A` times `G`, element by element (DESIGN.md § 4).
    GeluGrad,
    /// `x: f32[N, H]`, `gamma: f32[H]`, `beta: f32[H] -> f32[N, H]`: each row of `x` less its mean,
    /// divided by `sqrt(var + 1e-12)` with `var` its biased variance, times `gamma` plus `beta`
    /// (DESIGN.md § 4).
    LayerNorm,
    /// Internal gradient op over operands `x: f32[N, H]`, `gamma: f32[H]`, `G: f32[N, H]`: the
    /// gradient of [`Self::LayerNorm`] with respect to `x` (DESIGN.md § 4).
    LayerNormGradX,
    /// Internal gradient op over operands `x: f32[N, H]`, `G: f32[N, H]` `-> f32[H]`: the gradient
    /// of [`Self::LayerNorm`] with respect to `gamma`, the rows of `G` times the normalized rows
    /// of `x` summed in ascending order (DESIGN.md § 4).
    LayerNormGradGamma,
    /// `x: f32[N, K] -> f32[N, K]`: per row `exp(x - max) / sum(exp(x - max))` (DESIGN.md § 4).
    Softmax,
    /// Internal gradient op over operands `y`, `G` of one `[N, K]` shape, `y` the result of
    /// [`Self::Softmax`]: `y * (G - rowsum(G * y))` (DESIGN.md § 4).
    SoftmaxGrad,
    /// `q: f32[B * L, H]`, `k`, `v: f32[B * S, H]`, `key_mask: f32[B, S]` with `1 <= L <= S`
    /// `-> f32[B * L, H]`: bidirectional multi-head attention, `L = S` the attention of every row
    /// (story:attention-fewer-query-rows). Sequence `b` is query rows `b * L` to `b * L + L - 1`
    /// and key rows `b * S` to `b * S + S - 1`, head `a` columns `a * H/heads` to
    /// `(a + 1) * H/heads - 1`; within each, the masked softmax of `q k^T / sqrt(H/heads)` over
    /// the keys whose mask value is 1, times `v` (DESIGN.md § 4). An `i32 [U, 2]` fourth operand in
    /// place of the mask holds (start, length) spans of key rows, ascending and disjoint, `q` of
    /// the rows of `k` (a query attends to the span that holds it) or of `U` rows (query `u` to
    /// span `u`); a row outside every span is `+0` (story:attention-event-spans).
    Attention {
        heads: u64,
    },
    /// Internal gradient op over operands `q, k, v, key_mask, G`, `G` of `q`'s shape: the
    /// gradient of [`Self::Attention`] with respect to `q` (DESIGN.md § 4).
    AttentionGradQ {
        heads: u64,
    },
    /// Internal gradient op over operands `q, k, v, key_mask, G`: the gradient of
    /// [`Self::Attention`] with respect to `k`, exactly 0 in the row of a masked key.
    AttentionGradK {
        heads: u64,
    },
    /// Internal gradient op over operands `q, k, v, key_mask, G`: the gradient of
    /// [`Self::Attention`] with respect to `v`, exactly 0 in the row of a masked key.
    AttentionGradV {
        heads: u64,
    },
    /// `logits: f32[N, V]`, `labels: i32[N]`, `weights: f32[N]` `-> f32[]`: the weighted mean
    /// `sum_n weights[n] * (logsumexp(logits[n]) - logits[n, labels[n]]) / sum_n weights[n]`, the
    /// log-sum-exp max-subtracted; `0.0` when the weights sum to `0.0` (DESIGN.md § 4).
    CrossEntropy,
    /// Internal gradient op over operands `logits, labels, weights, G`, `G` of rank 0: the gradient
    /// of [`Self::CrossEntropy`] with respect to `logits`,
    /// `G * weights[n] / sum(weights) * (softmax(logits[n]) - onehot(labels[n]))`, all `0.0` when
    /// the weights sum to `0.0` (DESIGN.md § 4).
    CrossEntropyGrad,
    /// Operands `x` of any `f32` shape and the site's RNG token `i32[2]` `-> x`'s type: each
    /// element of `x` times `1 / (1 - p)` where the Philox mask of the token keeps it, `0.0`
    /// where it drops it; `p` in `[0, 1)` (DESIGN.md § 2.3, § 4).
    Dropout {
        p: f32,
    },
    /// Internal gradient op over operands `token: i32[2]`, `G`: `G` times `1 / (1 - p)` where
    /// the mask of [`Self::Dropout`] at the same token keeps the element, `0.0` elsewhere
    /// (DESIGN.md § 4). The mask is recomputed, never stored.
    DropoutGrad {
        p: f32,
    },
    /// `x` of any `f32` shape `-> x`'s type: `x` unchanged, through which no gradient flows
    /// (DESIGN.md § 2.2, § 4). Autodiff does not walk through it, and the pass
    /// [`crate::passes::stop_gradient`] removes it after autodiff, so no training step, CPU
    /// evaluation of a step or schedule holds it.
    StopGradient,
    /// Internal optimizer op: the correctly rounded square root of each element, on any `f32`
    /// shape. Only the `Optimizer` subgraph of an AdamW step has it (DESIGN.md § 2.1, § 8).
    Sqrt,
    /// Internal optimizer op `A / B`, one correctly rounded `f32` division per element: same
    /// shapes, or one rank-0 operand, as `Add`. Only the `Optimizer` subgraph of an AdamW step
    /// has it (DESIGN.md § 2.1, § 8).
    Div,
    /// The source operator `sigreg(z, slices = M)`, `z: f32[N, K]` `-> f32[]`: the mean over `M`
    /// random unit projections of the rows of `z` of the Epps–Pulley statistic of each
    /// (DESIGN.md § 2.2, § 4). It types the call; the lowering expands it to
    /// `ReduceMean(SigregColumns(MatMul(z, proj)))`, `proj: f32[K, M]` the internal projection
    /// input of the call's site, so the verifier refuses it in Tensor IR.
    Sigreg {
        slices: u64,
    },
    /// Internal op `P: f32[N, M] -> f32[M]`: the Epps–Pulley statistic of each column of `P`
    /// against the standard normal, `T_m = N sum_j c_j ((C_j - phi_j)^2 + S_j^2)` over the eight
    /// knots `t_j = 0.625 j`, `C_j` and `S_j` the column means of `cos(t_j P)` and `sin(t_j P)`
    /// (DESIGN.md § 4).
    SigregColumns,
    /// Internal gradient op over operands `P: f32[N, M]`, `G: f32[M]`: the gradient of
    /// [`Self::SigregColumns`] with respect to `P`,
    /// `2 G[m] sum_j c_j t_j (S_j cos(t_j P) - (C_j - phi_j) sin(t_j P))`, `C_j` and `S_j`
    /// recomputed (DESIGN.md § 4).
    SigregColumnsGrad,
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
            TensorOp::Embedding => "Embedding",
            TensorOp::EmbeddingGrad { .. } => "EmbeddingGrad",
            TensorOp::AddRows => "AddRows",
            TensorOp::SumRows => "SumRows",
            TensorOp::Gelu => "Gelu",
            TensorOp::GeluGrad => "GeluGrad",
            TensorOp::LayerNorm => "LayerNorm",
            TensorOp::LayerNormGradX => "LayerNormGradX",
            TensorOp::LayerNormGradGamma => "LayerNormGradGamma",
            TensorOp::Softmax => "Softmax",
            TensorOp::SoftmaxGrad => "SoftmaxGrad",
            TensorOp::Attention { .. } => "Attention",
            TensorOp::AttentionGradQ { .. } => "AttentionGradQ",
            TensorOp::AttentionGradK { .. } => "AttentionGradK",
            TensorOp::AttentionGradV { .. } => "AttentionGradV",
            TensorOp::CrossEntropy => "CrossEntropy",
            TensorOp::CrossEntropyGrad => "CrossEntropyGrad",
            TensorOp::Dropout { .. } => "Dropout",
            TensorOp::DropoutGrad { .. } => "DropoutGrad",
            TensorOp::StopGradient => "StopGradient",
            TensorOp::Sqrt => "Sqrt",
            TensorOp::Div => "Div",
            TensorOp::Sigreg { .. } => "Sigreg",
            TensorOp::SigregColumns => "SigregColumns",
            TensorOp::SigregColumnsGrad => "SigregColumnsGrad",
        }
    }

    /// The `p` of `Dropout` and `DropoutGrad`; `None` for every other op.
    pub fn dropout_rate(&self) -> Option<f32> {
        match self {
            TensorOp::Dropout { p } | TensorOp::DropoutGrad { p } => Some(*p),
            _ => None,
        }
    }

    /// The `heads` of `Attention` and its three gradient ops; `None` for every other op.
    pub fn attention_heads(&self) -> Option<u64> {
        match self {
            TensorOp::Attention { heads }
            | TensorOp::AttentionGradQ { heads }
            | TensorOp::AttentionGradK { heads }
            | TensorOp::AttentionGradV { heads } => Some(*heads),
            _ => None,
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

/// How a parameter's initial value is set (DESIGN.md § 2.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Initializer {
    /// Drawn from the parameter stream; a rank-2 parameter only.
    XavierUniform,
    /// Every element `+0.0`, no draw; a parameter of any rank from 1.
    Zeros,
    /// Every element `1.0`, no draw; a parameter of any rank from 1.
    Ones,
}

impl Initializer {
    /// Every initializer, in the order the language documents them.
    pub const ALL: [Initializer; 3] = [
        Initializer::XavierUniform,
        Initializer::Zeros,
        Initializer::Ones,
    ];

    /// The initializer whose source name, the name it prints with, is `name`.
    pub fn from_name(name: &str) -> Option<Initializer> {
        Self::ALL.into_iter().find(|init| init.name() == name)
    }

    /// Its source name: `xavier_uniform`, `zeros` or `ones`.
    pub const fn name(self) -> &'static str {
        match self {
            Initializer::XavierUniform => "xavier_uniform",
            Initializer::Zeros => "zeros",
            Initializer::Ones => "ones",
        }
    }
}

impl fmt::Display for Initializer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
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

    /// Rows `ids` of `table`.
    pub fn embedding(&mut self, table: ValueId, ids: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::Embedding, &[table, ids])
    }

    /// The gradient of a `[rows, H]` table read at `ids`, from the result's gradient `g`.
    pub fn embedding_grad(
        &mut self,
        ids: ValueId,
        g: ValueId,
        rows: Dim,
    ) -> Result<ValueId, BuildError> {
        self.push(TensorOp::EmbeddingGrad { rows }, &[ids, g])
    }

    /// The `[H]` row `b` added to every row of the `[N, H]` `x`.
    pub fn add_rows(&mut self, x: ValueId, b: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::AddRows, &[x, b])
    }

    /// The rows of the `[N, H]` `g` summed, in ascending order, to an `[H]` row.
    pub fn sum_rows(&mut self, g: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::SumRows, &[g])
    }

    /// The tanh form of GELU of `a`, element by element.
    pub fn gelu(&mut self, a: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::Gelu, &[a])
    }

    /// The derivative of GELU at `a` times `g`, element by element.
    pub fn gelu_grad(&mut self, a: ValueId, g: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::GeluGrad, &[a, g])
    }

    /// Each row of the `[N, H]` `x` normalized, times the `[H]` `gamma` plus the `[H]` `beta`.
    pub fn layer_norm(
        &mut self,
        x: ValueId,
        gamma: ValueId,
        beta: ValueId,
    ) -> Result<ValueId, BuildError> {
        self.push(TensorOp::LayerNorm, &[x, gamma, beta])
    }

    /// The gradient of `layer_norm(x, gamma, _)` with respect to `x`, from the result's `g`.
    pub fn layer_norm_grad_x(
        &mut self,
        x: ValueId,
        gamma: ValueId,
        g: ValueId,
    ) -> Result<ValueId, BuildError> {
        self.push(TensorOp::LayerNormGradX, &[x, gamma, g])
    }

    /// The gradient of `layer_norm(x, _, _)` with respect to `gamma`, from the result's `g`.
    pub fn layer_norm_grad_gamma(&mut self, x: ValueId, g: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::LayerNormGradGamma, &[x, g])
    }

    /// The softmax of each row of the `[N, K]` `x`.
    pub fn softmax(&mut self, x: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::Softmax, &[x])
    }

    /// The gradient of a softmax whose result is `y`, from the result's `g`.
    pub fn softmax_grad(&mut self, y: ValueId, g: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::SoftmaxGrad, &[y, g])
    }

    /// Multi-head attention of the `[B * L, H]` `q` over the `[B * S, H]` `k`, `v` under the
    /// `[B, S]` `key_mask`.
    pub fn attention(
        &mut self,
        q: ValueId,
        k: ValueId,
        v: ValueId,
        key_mask: ValueId,
        heads: u64,
    ) -> Result<ValueId, BuildError> {
        self.push(TensorOp::Attention { heads }, &[q, k, v, key_mask])
    }

    /// The weighted cross-entropy of the `[N, V]` `logits` against the `i32[N]` `labels` under
    /// the `[N]` `weights`, rank 0.
    pub fn cross_entropy(
        &mut self,
        logits: ValueId,
        labels: ValueId,
        weights: ValueId,
    ) -> Result<ValueId, BuildError> {
        self.push(TensorOp::CrossEntropy, &[logits, labels, weights])
    }

    /// `x` under the dropout mask of the `i32[2]` RNG `token` with drop probability `p`.
    pub fn dropout(&mut self, x: ValueId, token: ValueId, p: f32) -> Result<ValueId, BuildError> {
        self.push(TensorOp::Dropout { p }, &[x, token])
    }

    /// The gradient of a dropout at `token` with drop probability `p`, from the result's `g`.
    pub fn dropout_grad(
        &mut self,
        token: ValueId,
        g: ValueId,
        p: f32,
    ) -> Result<ValueId, BuildError> {
        self.push(TensorOp::DropoutGrad { p }, &[token, g])
    }

    /// `x` unchanged, through which no gradient flows.
    pub fn stop_gradient(&mut self, x: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::StopGradient, &[x])
    }

    /// The Epps–Pulley statistic of each column of the projected samples `p: [N, M]`.
    pub fn sigreg_columns(&mut self, p: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::SigregColumns, &[p])
    }

    /// The gradient of [`Self::sigreg_columns`] of `p` from the result's `g: [M]`.
    pub fn sigreg_columns_grad(&mut self, p: ValueId, g: ValueId) -> Result<ValueId, BuildError> {
        self.push(TensorOp::SigregColumnsGrad, &[p, g])
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

#[cfg(test)]
mod tests {
    use super::*;

    /// One of each op. The match below fails to compile when an op is added and not listed.
    fn every_op() -> Vec<TensorOp> {
        let ops = vec![
            TensorOp::Input {
                name: String::new(),
            },
            TensorOp::ParamRead { param: ParamId(0) },
            TensorOp::Scalar(0.0),
            TensorOp::Add,
            TensorOp::Sub,
            TensorOp::Mul,
            TensorOp::MatMul,
            TensorOp::Relu,
            TensorOp::Transpose,
            TensorOp::ReduceSum,
            TensorOp::ReduceMean,
            TensorOp::Mse,
            TensorOp::ReluGrad,
            TensorOp::BroadcastScalar { shape: Vec::new() },
            TensorOp::Embedding,
            TensorOp::EmbeddingGrad {
                rows: Dim::Static(1),
            },
            TensorOp::AddRows,
            TensorOp::SumRows,
            TensorOp::Gelu,
            TensorOp::GeluGrad,
            TensorOp::LayerNorm,
            TensorOp::LayerNormGradX,
            TensorOp::LayerNormGradGamma,
            TensorOp::Softmax,
            TensorOp::SoftmaxGrad,
            TensorOp::Attention { heads: 1 },
            TensorOp::AttentionGradQ { heads: 1 },
            TensorOp::AttentionGradK { heads: 1 },
            TensorOp::AttentionGradV { heads: 1 },
            TensorOp::CrossEntropy,
            TensorOp::CrossEntropyGrad,
            TensorOp::Dropout { p: 0.5 },
            TensorOp::DropoutGrad { p: 0.5 },
            TensorOp::StopGradient,
            TensorOp::Sqrt,
            TensorOp::Div,
            TensorOp::Sigreg { slices: 1 },
            TensorOp::SigregColumns,
            TensorOp::SigregColumnsGrad,
        ];
        for op in &ops {
            match op {
                TensorOp::Input { .. }
                | TensorOp::ParamRead { .. }
                | TensorOp::Scalar(_)
                | TensorOp::Add
                | TensorOp::Sub
                | TensorOp::Mul
                | TensorOp::MatMul
                | TensorOp::Relu
                | TensorOp::Transpose
                | TensorOp::ReduceSum
                | TensorOp::ReduceMean
                | TensorOp::Mse
                | TensorOp::ReluGrad
                | TensorOp::BroadcastScalar { .. }
                | TensorOp::Embedding
                | TensorOp::EmbeddingGrad { .. }
                | TensorOp::AddRows
                | TensorOp::SumRows
                | TensorOp::Gelu
                | TensorOp::GeluGrad
                | TensorOp::LayerNorm
                | TensorOp::LayerNormGradX
                | TensorOp::LayerNormGradGamma
                | TensorOp::Softmax
                | TensorOp::SoftmaxGrad
                | TensorOp::Attention { .. }
                | TensorOp::AttentionGradQ { .. }
                | TensorOp::AttentionGradK { .. }
                | TensorOp::AttentionGradV { .. }
                | TensorOp::CrossEntropy
                | TensorOp::CrossEntropyGrad
                | TensorOp::Dropout { .. }
                | TensorOp::DropoutGrad { .. }
                | TensorOp::StopGradient
                | TensorOp::Sqrt
                | TensorOp::Div
                | TensorOp::Sigreg { .. }
                | TensorOp::SigregColumns
                | TensorOp::SigregColumnsGrad => {}
            }
        }
        ops
    }

    /// The `$defs` entry `name` of a schema file `ess generate` wrote for the observatory.
    fn schema(file: &str, name: &str) -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../modelspec-observatory/schema/types")
            .join(file);
        let text = std::fs::read_to_string(&path).expect("the generated schema exists");
        let document: serde_json::Value = serde_json::from_str(&text).expect("JSON");
        document["$defs"][name].clone()
    }

    fn variants(def: &serde_json::Value) -> Vec<String> {
        def["enum"]
            .as_array()
            .expect("an enum")
            .iter()
            .map(|v| v.as_str().expect("a name").to_owned())
            .collect()
    }

    /// `modelspec.observatory.TensorOperator` names exactly the ops of [`TensorOp`], in order.
    #[test]
    fn every_op_is_a_tensor_operator_of_the_specification() {
        let def = schema(
            "modelspec.observatory.TensorOperator.schema.json",
            "modelspec.observatory.TensorOperator",
        );
        let names: Vec<String> = every_op().iter().map(|op| op.name().to_owned()).collect();
        assert_eq!(variants(&def), names);
    }

    /// `modelspec.checkpoint.DType` names exactly the dtypes of [`DType`], with their sizes.
    #[test]
    fn every_dtype_is_a_dtype_of_the_specification() {
        let def = schema(
            "modelspec.checkpoint.DType.schema.json",
            "modelspec.checkpoint.DType",
        );
        let dtypes = [DType::F32, DType::I32];
        let names: Vec<String> = dtypes
            .iter()
            .map(|d| d.to_string().to_uppercase())
            .collect();
        assert_eq!(variants(&def), names);
        let sizes = &def["x-ess-attributes"][0]["values"];
        for dtype in dtypes {
            let name = dtype.to_string().to_uppercase();
            assert_eq!(sizes[&name].as_u64(), Some(dtype.size_bytes()), "{name}");
        }
    }
}
