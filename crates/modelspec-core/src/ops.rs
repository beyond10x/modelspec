//! The operator registry: per op, its arity and its shape and dtype rule (DESIGN.md § 2.2, "one
//! operator registry").
//!
//! The builder, the verifier and the type checker all type an op through [`infer`], so the three
//! cannot disagree. A rule failure is an [`OpError`] carrying the types involved; the checker maps
//! it to a user diagnostic (a [`OpError::MatMulContracting`] is `E-TYPE-012`). The reverse-mode
//! gradient rule of each op is [`gradient`], used by [`crate::autodiff`].

use thiserror::Error;

use crate::ir::tensor::{DType, Dim, TensorOp, TensorType, ValueId};

/// Why an op's operands do not satisfy its rule.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum OpError {
    #[error("{op} takes {expected} operand(s); got {got}")]
    Arity {
        op: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("{op} needs a declared result type")]
    MissingDeclaredType { op: &'static str },
    #[error("{op} operand {index} must be {expected}; got {got}")]
    OperandDType {
        op: &'static str,
        index: usize,
        expected: DType,
        got: TensorType,
    },
    #[error("a parameter must be f32; got {got}")]
    ParamDType { got: TensorType },
    #[error("{op} operand dtypes differ: lhs {lhs}, rhs {rhs}")]
    DTypeMismatch {
        op: &'static str,
        lhs: TensorType,
        rhs: TensorType,
    },
    #[error("{op} operand shapes differ: lhs {lhs}, rhs {rhs}")]
    ShapeMismatch {
        op: &'static str,
        lhs: TensorType,
        rhs: TensorType,
    },
    #[error("{op} needs a rank-{expected} operand; got {got}")]
    Rank {
        op: &'static str,
        expected: usize,
        got: TensorType,
    },
    #[error(
        "matmul contracting dimension mismatch: lhs {lhs}, rhs {rhs}, \
         expected rhs dimension[0] = {expected}; got {got}"
    )]
    MatMulContracting {
        lhs: TensorType,
        rhs: TensorType,
        expected: Dim,
        got: Dim,
    },
    #[error("{op} needs heads to divide the width {width}; got heads = {heads}")]
    HeadsDivide {
        op: &'static str,
        heads: u64,
        width: Dim,
    },
    /// `p` of `Dropout` or `DropoutGrad` outside `[0, 1)`, held as its `f32` bits so the error
    /// stays `Eq`.
    #[error("{op} needs p in [0, 1); got p = {:?}", f32::from_bits(*bits))]
    DropoutRate { op: &'static str, bits: u32 },
    /// The RNG token operand of `Dropout` or `DropoutGrad` that is not `i32[2]`.
    #[error("{op} operand {index} must be the RNG token i32[2]; got {got}")]
    Token {
        op: &'static str,
        index: usize,
        got: TensorType,
    },
}

/// The kind of value a named compile-time operator argument takes (DESIGN.md § 2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NamedKind {
    /// An integer literal of at least 1.
    PositiveInteger,
    /// A float literal whose `f32` rounding lies in `[0, 1)`.
    Probability,
}

/// A named compile-time argument an operator takes after its operands, such as `heads = 2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NamedParameter {
    pub name: &'static str,
    pub kind: NamedKind,
}

/// The value of a named compile-time argument: a literal of its parameter's kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NamedValue {
    Integer(u64),
    Float(f32),
}

impl std::fmt::Display for NamedValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NamedValue::Integer(n) => write!(f, "{n}"),
            NamedValue::Float(x) => write!(f, "{x:?}"),
        }
    }
}

impl NamedKind {
    /// What a source literal of this kind looks like, for messages: `a positive integer literal`.
    pub const fn describe(self) -> &'static str {
        match self {
            NamedKind::PositiveInteger => "an integer literal of at least 1",
            NamedKind::Probability => "a float literal in [0, 1)",
        }
    }

    /// The value of the integer literal `literal` if this kind admits it.
    pub fn integer(self, literal: u64) -> Option<NamedValue> {
        match self {
            NamedKind::PositiveInteger => (literal >= 1).then_some(NamedValue::Integer(literal)),
            NamedKind::Probability => None,
        }
    }

    /// The value of the float literal `literal` if this kind admits it: for a probability, its
    /// `f32` rounding when that lies in `[0, 1)`.
    pub fn float(self, literal: f64) -> Option<NamedValue> {
        match self {
            NamedKind::PositiveInteger => None,
            NamedKind::Probability => {
                let p = literal as f32;
                is_dropout_rate(p).then_some(NamedValue::Float(p))
            }
        }
    }
}

/// Whether `p` is a drop probability `dropout` takes: in `[0, 1)`, so `1 / (1 - p)` is finite.
/// `NaN` is not.
pub fn is_dropout_rate(p: f32) -> bool {
    (0.0..1.0).contains(&p)
}

/// `heads = <integer>` of `attention`.
const HEADS: NamedParameter = NamedParameter {
    name: "heads",
    kind: NamedKind::PositiveInteger,
};

/// `p = <float>` of `dropout`, the probability that an element is dropped.
const P: NamedParameter = NamedParameter {
    name: "p",
    kind: NamedKind::Probability,
};

/// `slices = <integer>` of `sigreg`, the number of random projections.
const SLICES: NamedParameter = NamedParameter {
    name: "slices",
    kind: NamedKind::PositiveInteger,
};

/// The named compile-time arguments the source operator `op` takes, every one of them required,
/// in the order [`with_named`] takes their values; empty for an operator that takes none.
pub fn named_parameters(op: &TensorOp) -> &'static [NamedParameter] {
    match op {
        TensorOp::Attention { .. } => &[HEADS],
        TensorOp::Dropout { .. } => &[P],
        TensorOp::Sigreg { .. } => &[SLICES],
        _ => &[],
    }
}

/// Whether `op` is a source operator that the lowering expands into other ops, so that Tensor IR
/// never holds it: `Sigreg`, which becomes `ReduceMean(SigregColumns(MatMul(z, proj)))` over the
/// projection input of its site (DESIGN.md § 2.2). The type checker types its call through
/// [`infer`] like any other; the verifier refuses it.
pub fn expanded(op: &TensorOp) -> bool {
    matches!(op, TensorOp::Sigreg { .. })
}

/// The types of the values the lowering derives for the expanded operator `op` ([`expanded`])
/// from `operands` beside its result, each sized by a named argument: for `sigreg(z: f32[N, K],
/// slices = M)` the projection `f32[K, M]`, `P = z @ proj` of `f32[N, M]` and
/// `SigregColumns(P)` of `f32[M]` (DESIGN.md § 2.2). The type checker refuses one whose element
/// or byte count overflows `u64` with E-TYPE-016 at that argument, so lowering never builds it.
/// Empty for an operator that is not expanded or an operand that [`infer`] refuses.
pub fn derived_types(op: &TensorOp, operands: &[&TensorType]) -> Vec<TensorType> {
    match (op, operands) {
        (TensorOp::Sigreg { slices }, [z]) if z.shape.len() == 2 => {
            let m = Dim::Static(*slices);
            let f32_of = |shape: Vec<Dim>| TensorType {
                dtype: DType::F32,
                shape,
            };
            vec![
                f32_of(vec![z.shape[1], m]),
                f32_of(vec![z.shape[0], m]),
                f32_of(vec![m]),
            ]
        }
        _ => Vec::new(),
    }
}

/// `op` with the values of its named arguments, one per entry of [`named_parameters`] and in its
/// order. `op` itself when it takes none.
pub fn with_named(op: &TensorOp, values: &[NamedValue]) -> TensorOp {
    match (op, values) {
        (TensorOp::Attention { .. }, [NamedValue::Integer(heads)]) => {
            TensorOp::Attention { heads: *heads }
        }
        (TensorOp::Dropout { .. }, [NamedValue::Float(p)]) => TensorOp::Dropout { p: *p },
        (TensorOp::Sigreg { .. }, [NamedValue::Integer(slices)]) => {
            TensorOp::Sigreg { slices: *slices }
        }
        _ => op.clone(),
    }
}

/// The index of the RNG token operand of `op`, which no source value supplies: operand 1 of
/// `Dropout`, operand 0 of `DropoutGrad`; `None` for every other op. The lowering gives each
/// dropout site its own token input, `i32[2]` (DESIGN.md § 2.3), the third use of `i32` beside
/// row ids and class labels. A token takes no gradient.
pub fn token_operand(op: &TensorOp) -> Option<usize> {
    match op {
        TensorOp::Dropout { .. } => Some(1),
        TensorOp::DropoutGrad { .. } => Some(0),
        _ => None,
    }
}

/// The type of an RNG token: `i32[2]`, the draw id's low and high 32 bits.
pub fn token_type() -> TensorType {
    TensorType::i32(&[2])
}

/// The number of operands a source call of `op` passes: [`arity`] less the RNG token, which the
/// lowering supplies.
pub fn source_arity(op: &TensorOp) -> usize {
    arity(op) - usize::from(token_operand(op).is_some())
}

/// The operands of the source operator `op` that take no gradient, by index and source name:
/// `key_mask` of `attention`, `labels` and `weights` of `cross_entropy`. A parameter cannot
/// reach one, since training needs a gradient for every parameter the loss reads (DESIGN.md
/// § 2.2).
pub fn constant_operands(op: &TensorOp) -> &'static [(usize, &'static str)] {
    match op {
        TensorOp::Attention { .. } => &[(3, "key_mask")],
        TensorOp::CrossEntropy => &[(1, "labels"), (2, "weights")],
        _ => &[],
    }
}

/// The number of operands `op` takes.
pub fn arity(op: &TensorOp) -> usize {
    match op {
        TensorOp::Input { .. } | TensorOp::ParamRead { .. } | TensorOp::Scalar(_) => 0,
        TensorOp::Relu
        | TensorOp::Transpose
        | TensorOp::ReduceSum
        | TensorOp::ReduceMean
        | TensorOp::BroadcastScalar { .. }
        | TensorOp::SumRows
        | TensorOp::Gelu
        | TensorOp::Softmax
        | TensorOp::StopGradient
        | TensorOp::Sqrt
        | TensorOp::Sigreg { .. }
        | TensorOp::SigregColumns => 1,
        TensorOp::Add
        | TensorOp::Sub
        | TensorOp::Mul
        | TensorOp::MatMul
        | TensorOp::Mse
        | TensorOp::ReluGrad
        | TensorOp::Embedding
        | TensorOp::EmbeddingGrad { .. }
        | TensorOp::AddRows
        | TensorOp::GeluGrad
        | TensorOp::LayerNormGradGamma
        | TensorOp::SoftmaxGrad
        | TensorOp::Dropout { .. }
        | TensorOp::DropoutGrad { .. }
        | TensorOp::Div
        | TensorOp::SigregColumnsGrad => 2,
        TensorOp::LayerNorm | TensorOp::LayerNormGradX | TensorOp::CrossEntropy => 3,
        TensorOp::Attention { .. } | TensorOp::CrossEntropyGrad => 4,
        TensorOp::AttentionGradQ { .. }
        | TensorOp::AttentionGradK { .. }
        | TensorOp::AttentionGradV { .. } => 5,
    }
}

/// The element type operand `index` of `op` must have: `i32` for the ids of `Embedding` and
/// `EmbeddingGrad`, the labels of `CrossEntropy` and `CrossEntropyGrad` and the RNG token of
/// `Dropout` and `DropoutGrad`, `f32` for every other operand (DESIGN.md § 2.2).
pub fn operand_dtype(op: &TensorOp, index: usize) -> DType {
    match (op, index) {
        (TensorOp::Embedding, 1)
        | (TensorOp::EmbeddingGrad { .. }, 0)
        | (TensorOp::CrossEntropy | TensorOp::CrossEntropyGrad, 1)
        | (TensorOp::Dropout { .. }, 1)
        | (TensorOp::DropoutGrad { .. }, 0) => DType::I32,
        _ => DType::F32,
    }
}

/// Whether `operand` is the `i32` spans form of an attention op's fourth operand
/// (story:attention-event-spans), which [`operand_dtype`] does not name: it stays the key mask's
/// `f32`, the form the reference documents.
pub fn attention_spans(op: &TensorOp, operand: &TensorType) -> bool {
    operand.dtype == DType::I32
        && matches!(
            op,
            TensorOp::Attention { .. }
                | TensorOp::AttentionGradQ { .. }
                | TensorOp::AttentionGradK { .. }
                | TensorOp::AttentionGradV { .. }
        )
}

/// The result type of `op` applied to operands of types `operands`.
///
/// `declared` is the type a leaf takes from outside the graph: the node's own type for `Input`,
/// the parameter table entry for `ParamRead`. Other ops ignore it.
pub fn infer(
    op: &TensorOp,
    operands: &[&TensorType],
    declared: Option<&TensorType>,
) -> Result<TensorType, OpError> {
    let name = op.name();
    let expected = arity(op);
    if operands.len() != expected {
        return Err(OpError::Arity {
            op: name,
            expected,
            got: operands.len(),
        });
    }
    for (index, operand) in operands.iter().enumerate() {
        let expected = operand_dtype(op, index);
        if operand.dtype != expected && !(index == 3 && attention_spans(op, operand)) {
            return Err(OpError::OperandDType {
                op: name,
                index,
                expected,
                got: (*operand).clone(),
            });
        }
    }
    match op {
        TensorOp::Input { .. } => declared
            .cloned()
            .ok_or(OpError::MissingDeclaredType { op: name }),
        TensorOp::ParamRead { .. } => {
            let ty = declared.ok_or(OpError::MissingDeclaredType { op: name })?;
            if ty.dtype != DType::F32 {
                return Err(OpError::ParamDType { got: ty.clone() });
            }
            Ok(ty.clone())
        }
        TensorOp::Scalar(_) => Ok(TensorType::f32_scalar()),
        TensorOp::Add | TensorOp::Sub | TensorOp::Mul | TensorOp::Div => {
            let (lhs, rhs) = (operands[0], operands[1]);
            same_dtype(name, lhs, rhs)?;
            if lhs.shape == rhs.shape || rhs.rank() == 0 {
                Ok(lhs.clone())
            } else if lhs.rank() == 0 {
                Ok(rhs.clone())
            } else {
                Err(shape_mismatch(name, lhs, rhs))
            }
        }
        TensorOp::Mse => {
            let (lhs, rhs) = (operands[0], operands[1]);
            same_dtype(name, lhs, rhs)?;
            if lhs.shape != rhs.shape {
                return Err(shape_mismatch(name, lhs, rhs));
            }
            Ok(scalar_of(lhs))
        }
        TensorOp::ReluGrad | TensorOp::GeluGrad => {
            let (a, g) = (operands[0], operands[1]);
            same_dtype(name, a, g)?;
            if a.shape != g.shape {
                return Err(shape_mismatch(name, a, g));
            }
            Ok(g.clone())
        }
        TensorOp::MatMul => {
            let (lhs, rhs) = (operands[0], operands[1]);
            same_dtype(name, lhs, rhs)?;
            rank(name, lhs, 2)?;
            rank(name, rhs, 2)?;
            if lhs.shape[1] != rhs.shape[0] {
                return Err(OpError::MatMulContracting {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    expected: lhs.shape[1],
                    got: rhs.shape[0],
                });
            }
            Ok(TensorType {
                dtype: lhs.dtype,
                shape: vec![lhs.shape[0], rhs.shape[1]],
            })
        }
        TensorOp::Relu | TensorOp::Gelu | TensorOp::StopGradient | TensorOp::Sqrt => {
            Ok(operands[0].clone())
        }
        TensorOp::Transpose => {
            let a = operands[0];
            rank(name, a, 2)?;
            Ok(TensorType {
                dtype: a.dtype,
                shape: vec![a.shape[1], a.shape[0]],
            })
        }
        TensorOp::ReduceSum | TensorOp::ReduceMean => Ok(scalar_of(operands[0])),
        TensorOp::BroadcastScalar { shape } => {
            let a = operands[0];
            rank(name, a, 0)?;
            Ok(TensorType {
                dtype: a.dtype,
                shape: shape.clone(),
            })
        }
        TensorOp::Embedding => {
            let (table, ids) = (operands[0], operands[1]);
            rank(name, table, 2)?;
            rank(name, ids, 1)?;
            Ok(TensorType {
                dtype: table.dtype,
                shape: vec![ids.shape[0], table.shape[1]],
            })
        }
        TensorOp::EmbeddingGrad { rows } => {
            let (ids, g) = (operands[0], operands[1]);
            rank(name, ids, 1)?;
            rank(name, g, 2)?;
            if ids.shape[0] != g.shape[0] {
                return Err(shape_mismatch(name, ids, g));
            }
            Ok(TensorType {
                dtype: g.dtype,
                shape: vec![*rows, g.shape[1]],
            })
        }
        TensorOp::AddRows => {
            let (x, b) = (operands[0], operands[1]);
            same_dtype(name, x, b)?;
            rank(name, x, 2)?;
            rank(name, b, 1)?;
            if b.shape[0] != x.shape[1] {
                return Err(shape_mismatch(name, x, b));
            }
            Ok(x.clone())
        }
        TensorOp::SumRows => {
            let g = operands[0];
            rank(name, g, 2)?;
            Ok(TensorType {
                dtype: g.dtype,
                shape: vec![g.shape[1]],
            })
        }
        TensorOp::LayerNorm => {
            let (x, gamma, beta) = (operands[0], operands[1], operands[2]);
            rows_of(name, x, &[gamma, beta])?;
            Ok(x.clone())
        }
        TensorOp::LayerNormGradX => {
            let (x, gamma, g) = (operands[0], operands[1], operands[2]);
            rows_of(name, x, &[gamma])?;
            same_shape(name, x, g)?;
            Ok(x.clone())
        }
        TensorOp::LayerNormGradGamma => {
            let (x, g) = (operands[0], operands[1]);
            rank(name, x, 2)?;
            same_shape(name, x, g)?;
            Ok(TensorType {
                dtype: x.dtype,
                shape: vec![x.shape[1]],
            })
        }
        TensorOp::Softmax => {
            let x = operands[0];
            rank(name, x, 2)?;
            Ok(x.clone())
        }
        TensorOp::SoftmaxGrad => {
            let (y, g) = (operands[0], operands[1]);
            rank(name, y, 2)?;
            same_shape(name, y, g)?;
            Ok(y.clone())
        }
        TensorOp::Attention { heads } => {
            attention_operands(name, *heads, operands)?;
            Ok(operands[0].clone())
        }
        TensorOp::AttentionGradQ { heads }
        | TensorOp::AttentionGradK { heads }
        | TensorOp::AttentionGradV { heads } => {
            attention_operands(name, *heads, &operands[..4])?;
            same_shape(name, operands[0], operands[4])?;
            let of = match op {
                TensorOp::AttentionGradK { .. } => 1,
                TensorOp::AttentionGradV { .. } => 2,
                _ => 0,
            };
            Ok(operands[of].clone())
        }
        TensorOp::CrossEntropy => {
            cross_entropy_operands(name, operands)?;
            Ok(scalar_of(operands[0]))
        }
        TensorOp::CrossEntropyGrad => {
            cross_entropy_operands(name, &operands[..3])?;
            rank(name, operands[3], 0)?;
            Ok(operands[0].clone())
        }
        TensorOp::Dropout { p } | TensorOp::DropoutGrad { p } => {
            if !is_dropout_rate(*p) {
                return Err(OpError::DropoutRate {
                    op: name,
                    bits: p.to_bits(),
                });
            }
            let index = token_operand(op).unwrap_or(0);
            let token = operands[index];
            if *token != token_type() {
                return Err(OpError::Token {
                    op: name,
                    index,
                    got: token.clone(),
                });
            }
            Ok(operands[1 - index].clone())
        }
        TensorOp::Sigreg { .. } => {
            rank(name, operands[0], 2)?;
            Ok(scalar_of(operands[0]))
        }
        TensorOp::SigregColumns => {
            let p = operands[0];
            rank(name, p, 2)?;
            Ok(TensorType {
                dtype: p.dtype,
                shape: vec![p.shape[1]],
            })
        }
        TensorOp::SigregColumnsGrad => {
            let (p, g) = (operands[0], operands[1]);
            same_dtype(name, p, g)?;
            rank(name, p, 2)?;
            rank(name, g, 1)?;
            if g.shape[0] != p.shape[1] {
                return Err(shape_mismatch(name, p, g));
            }
            Ok(p.clone())
        }
    }
}

/// The operands `logits, labels, weights` of `CrossEntropy` and its gradient op, in this order:
/// `logits` of rank 2, `labels` of rank 1 and `weights` of rank 1, then `labels` and `weights`
/// each as long as the rows of `logits`. A length that differs is a shape mismatch against
/// `logits`.
fn cross_entropy_operands(op: &'static str, operands: &[&TensorType]) -> Result<(), OpError> {
    let (logits, labels, weights) = (operands[0], operands[1], operands[2]);
    rank(op, logits, 2)?;
    rank(op, labels, 1)?;
    rank(op, weights, 1)?;
    for per_row in [labels, weights] {
        if per_row.shape[0] != logits.shape[0] {
            return Err(shape_mismatch(op, logits, per_row));
        }
    }
    Ok(())
}

/// The operands `q, k, v, key_mask` of `Attention` and its gradient ops: `k` and `v` of one
/// shape `[B * S, H]`, `key_mask` of rank 2 and shape `[B, S]`, `q` of rank 2 and shape
/// `[B * L, H]` with `1 <= L <= S` query rows per sequence (`L = S` the self-attention of every
/// row; story:attention-fewer-query-rows), and `heads` dividing `H`. `B` is the rows of the key
/// mask, so `L` is the rows of `q` divided by them. A `k` of another width or rank than `q`, or a
/// `v` of another shape than `k`, is a shape mismatch against `q`; so is a `key_mask` whose
/// `B * S` is not the rows of `k`, or whose `B` does not divide the rows of `q` into at most `S`,
/// or a dimension of `key_mask`, `q` or `k` that is not static; `heads` of 0, or a width that is
/// not static or that `heads` does not divide, is [`OpError::HeadsDivide`].
///
/// In place of the key mask the fourth operand may be `i32 [U, 2]` spans, `U >= 1`
/// (story:attention-event-spans): row `u` is `(start, length)`, keys `start .. start + length`
/// of `k` and `v`. `q` then has the rows of `k`, row `i` the query of the span holding it, or `U`
/// rows, row `u` the query of span `u`; the rows of `k` win where `U` is the rows of `k`. Spans
/// of another rank or width, `U = 0`, or `q` of other rows is a shape mismatch against `q`. That
/// the spans lie in the rows and in ascending order apart is a property of their values, which
/// the CPU interpreter checks (`RuntimeError::SpanOverlap` and its siblings).
fn attention_operands(
    op: &'static str,
    heads: u64,
    operands: &[&TensorType],
) -> Result<(), OpError> {
    let (q, k, v, mask) = (operands[0], operands[1], operands[2], operands[3]);
    rank(op, q, 2)?;
    same_dtype(op, q, k)?;
    if k.rank() != 2 || k.shape[1] != q.shape[1] {
        return Err(shape_mismatch(op, q, k));
    }
    same_dtype(op, q, v)?;
    if v.shape != k.shape {
        return Err(shape_mismatch(op, q, v));
    }
    let rows = if mask.dtype == DType::I32 {
        rank(op, mask, 2)?;
        match (q.shape[0], k.shape[0], mask.shape[0], mask.shape[1]) {
            (Dim::Static(nq), Dim::Static(nk), Dim::Static(u), Dim::Static(2)) => {
                u != 0 && (nq == nk || nq == u)
            }
            _ => false,
        }
    } else {
        same_dtype(op, q, mask)?;
        rank(op, mask, 2)?;
        match (q.shape[0], k.shape[0], mask.shape[0], mask.shape[1]) {
            (Dim::Static(nq), Dim::Static(nk), Dim::Static(b), Dim::Static(s)) => {
                b.checked_mul(s) == Some(nk) && b != 0 && nq % b == 0 && (1..=s).contains(&(nq / b))
            }
            _ => false,
        }
    };
    if !rows {
        return Err(shape_mismatch(op, q, mask));
    }
    let divides = match q.shape[1] {
        Dim::Static(width) => heads != 0 && width % heads == 0,
        Dim::Symbol(_) => false,
    };
    if !divides {
        return Err(OpError::HeadsDivide {
            op,
            heads,
            width: q.shape[1],
        });
    }
    Ok(())
}

/// `x` of rank 2 and each of `rows` of rank 1 and as long as a row of `x`: a row of another rank
/// is a rank error, one of another length a shape mismatch against `x`.
fn rows_of(op: &'static str, x: &TensorType, rows: &[&TensorType]) -> Result<(), OpError> {
    rank(op, x, 2)?;
    for row in rows {
        same_dtype(op, x, row)?;
        rank(op, row, 1)?;
        if row.shape[0] != x.shape[1] {
            return Err(shape_mismatch(op, x, row));
        }
    }
    Ok(())
}

/// `a` and `b` of one dtype and one shape.
fn same_shape(op: &'static str, a: &TensorType, b: &TensorType) -> Result<(), OpError> {
    same_dtype(op, a, b)?;
    if a.shape != b.shape {
        return Err(shape_mismatch(op, a, b));
    }
    Ok(())
}

/// Where a gradient rule puts the nodes it builds; each is typed through [`infer`].
pub trait GradientSink {
    type Error;

    /// Adds `op` over `operands` and returns the value it defines.
    fn emit(&mut self, op: TensorOp, operands: &[ValueId]) -> Result<ValueId, Self::Error>;

    /// The type of a value defined so far.
    fn type_of(&self, value: ValueId) -> &TensorType;
}

/// The reverse-mode gradient rule of `op` (DESIGN.md § 4, the rows of its table).
///
/// `operands` are the node's operands, `result` the value it defines and `g` the adjoint of that
/// value. Returns, in operand order, the contribution to the adjoint of each operand `i` with
/// `wanted[i]`, and `None` for the others; a contribution has its operand's type. The caller sums
/// the contributions of every use of a value. An operand of rank 0 under a same-shape-or-scalar
/// op receives the sum of its broadcast contribution. Returns `Ok(None)` for the internal
/// gradient ops `ReluGrad`, `BroadcastScalar`, `EmbeddingGrad`, `SumRows`, `GeluGrad`,
/// `LayerNormGradX`, `LayerNormGradGamma`, `SoftmaxGrad`, `AttentionGradQ`, `AttentionGradK`,
/// `AttentionGradV`, `CrossEntropyGrad`, `DropoutGrad` and `SigregColumnsGrad`, which have no
/// rule, for the internal optimizer ops `Sqrt` and `Div`, which only the `Optimizer` subgraph of
/// an AdamW step has and through which no gradient is taken, and for the source operator
/// `Sigreg`, which the lowering expands ([`expanded`]) and the verifier refuses in Tensor IR.
///
/// The rules, for `Y = op(A, B, C)` and `G = dL/dY`:
///
/// | op | `dA` | `dB` | `dC` |
/// | --- | --- | --- | --- |
/// | `Add` | `G` | `G` | |
/// | `Sub` | `G` | `-G` | |
/// | `Mul` | `G * B` | `G * A` | |
/// | `MatMul` | `G @ transpose(B)` | `transpose(A) @ G` | |
/// | `Relu` | `where(A > 0, G, 0)` (`ReluGrad`; 0 at `A = 0`) | | |
/// | `Transpose` | `transpose(G)` | | |
/// | `ReduceSum` | `broadcast_scalar(G, shape(A))` | | |
/// | `ReduceMean` | `broadcast_scalar(G * (1 / numel(A)), shape(A))` | | |
/// | `Mse` | `(A - B) * (G * (2 / numel(A)))` | `-dA` | |
/// | `Embedding` | `embedding_grad(B, G, rows(A))` (`EmbeddingGrad`: row `G[n]` added to row `B[n]`) | none: `B` is `i32` ids | |
/// | `AddRows` | `G` | `sum_rows(G)` (`SumRows`: the rows of `G` summed in ascending order) | |
/// | `Gelu` | `gelu_grad(A, G)` (`GeluGrad`: `gelu'(A) * G`) | | |
/// | `LayerNorm` | `layer_norm_grad_x(A, B, G)` (`LayerNormGradX`: `rstd * (G B - mean(G B) - xhat * mean(G B xhat))` per row) | `layer_norm_grad_gamma(A, G)` (`LayerNormGradGamma`: the rows of `G * xhat` summed in ascending order) | `sum_rows(G)` |
/// | `Softmax` | `softmax_grad(Y, G)` (`SoftmaxGrad`: `Y * (G - rowsum(G * Y))`) | | |
/// | `Attention` | `attention_grad_q(A, B, C, D, G)` (`AttentionGradQ`: `scale * dS K` per sequence and head) | `attention_grad_k(A, B, C, D, G)` (`AttentionGradK`: `scale * dS^T Q`, 0 at a masked key) | `attention_grad_v(A, B, C, D, G)` (`AttentionGradV`: `P^T G`, 0 at a masked key) |
/// | `CrossEntropy` | `cross_entropy_grad(A, B, C, G)` (`CrossEntropyGrad`: `G * C[n] / sum(C) * (P[n, v] - [v = B[n]])`) | none: `B` is `i32` labels | none: `C` is constant weights |
/// | `Dropout` | `dropout_grad(B, G)` (`DropoutGrad`: `G * (1 / (1 - p))` where the mask of token `B` keeps the element, 0 elsewhere) | none: `B` is the `i32` RNG token | |
/// | `StopGradient` | none: no gradient flows back through it; the pass `passes::stop_gradient` replaces it by `A` after autodiff | | |
/// | `SigregColumns` | `sigreg_columns_grad(A, G)` (`SigregColumnsGrad`: `2 G[m] sum_j c_j t_j (S_j cos(t_j A) - (C_j - phi_j) sin(t_j A))` in column `m`) | | |
///
/// The constants `1 / numel` and `2 / numel` are `f32` scalars. `Mse`'s `B` is a target in a
/// training program, an input that receives no gradient; a trainable `B` receives `-dA`.
/// `EmbeddingGrad` adds the rows of `G` in ascending `n`, so a row read twice accumulates in a
/// fixed order, and a row never read stays exactly zero. `SumRows` starts from row 0 and adds
/// each later row in ascending order. `gelu'` is the derivative of the tanh form of `Gelu`. For
/// `LayerNorm`, `xhat = (A - mean) * rstd` and `rstd = 1 / sqrt(var + 1e-12)` are recomputed from
/// `A` per row; `Softmax`'s rule reads its own result `Y`. `Attention`'s operand `D` is the key
/// mask, which receives no gradient; each of its three gradient ops recomputes the probabilities
/// `P` of a sequence and head from `A`, `B` and `D`, and with `dP = G V^T` takes
/// `dS = P * (dP - rowsum(P * dP))`; `scale = 1 / sqrt(H / heads)`. `CrossEntropy`'s operands
/// `B` (labels) and `C` (weights) receive no gradient; `CrossEntropyGrad` recomputes the row
/// softmax `P` of `A` through the loss's own code path, and gives zero everywhere when
/// `sum(C) = 0`. `Dropout`'s operand `B` is the site's RNG token, which takes no gradient;
/// `DropoutGrad` recomputes the mask from the same token through the code path of `Dropout`, so
/// no mask is stored, and multiplies by the same `f32` scale `1 / (1 - p)`. `StopGradient` gives
/// no contribution to its operand: autodiff does not walk through it ([`crate::autodiff`]), so a
/// value read only through it receives no adjoint from that read. `SigregColumnsGrad`
/// recomputes the column means `C_j` and `S_j` of `cos(t_j A)` and `sin(t_j A)` through the code
/// path of `SigregColumns`; the projection that `sigreg` multiplies `z` by is an internal input,
/// which takes no gradient, so `MatMul`'s rule gives `dz`.
pub fn gradient<S: GradientSink + ?Sized>(
    sink: &mut S,
    op: &TensorOp,
    operands: &[ValueId],
    result: ValueId,
    g: ValueId,
    wanted: &[bool],
) -> Result<Option<Vec<Option<ValueId>>>, S::Error> {
    let want = |i: usize| wanted.get(i).copied().unwrap_or(false);
    let mut out = vec![None; operands.len()];
    match op {
        TensorOp::Input { .. } | TensorOp::ParamRead { .. } | TensorOp::Scalar(_) => {}
        TensorOp::Add => {
            for (i, &operand) in operands.iter().enumerate() {
                if want(i) {
                    out[i] = Some(unbroadcast(sink, g, operand)?);
                }
            }
        }
        TensorOp::Sub => {
            if want(0) {
                out[0] = Some(unbroadcast(sink, g, operands[0])?);
            }
            if want(1) {
                let summed = unbroadcast(sink, g, operands[1])?;
                out[1] = Some(negate(sink, summed)?);
            }
        }
        TensorOp::Mul => {
            let (a, b) = (operands[0], operands[1]);
            if want(0) {
                let c = sink.emit(TensorOp::Mul, &[g, b])?;
                out[0] = Some(unbroadcast(sink, c, a)?);
            }
            if want(1) {
                let c = sink.emit(TensorOp::Mul, &[g, a])?;
                out[1] = Some(unbroadcast(sink, c, b)?);
            }
        }
        TensorOp::MatMul => {
            let (a, b) = (operands[0], operands[1]);
            if want(0) {
                let bt = sink.emit(TensorOp::Transpose, &[b])?;
                out[0] = Some(sink.emit(TensorOp::MatMul, &[g, bt])?);
            }
            if want(1) {
                let at = sink.emit(TensorOp::Transpose, &[a])?;
                out[1] = Some(sink.emit(TensorOp::MatMul, &[at, g])?);
            }
        }
        TensorOp::Relu => {
            if want(0) {
                out[0] = Some(sink.emit(TensorOp::ReluGrad, &[operands[0], g])?);
            }
        }
        TensorOp::Transpose => {
            if want(0) {
                out[0] = Some(sink.emit(TensorOp::Transpose, &[g])?);
            }
        }
        TensorOp::ReduceSum => {
            if want(0) {
                out[0] = Some(broadcast(sink, g, operands[0])?);
            }
        }
        TensorOp::ReduceMean => {
            if want(0) {
                let a = operands[0];
                let scale = per_element(sink.type_of(a), 1.0);
                let c = sink.emit(TensorOp::Scalar(scale), &[])?;
                let scaled = sink.emit(TensorOp::Mul, &[g, c])?;
                out[0] = Some(broadcast(sink, scaled, a)?);
            }
        }
        TensorOp::Mse => {
            if want(0) || want(1) {
                let (y, t) = (operands[0], operands[1]);
                let scale = per_element(sink.type_of(y), 2.0);
                let d = sink.emit(TensorOp::Sub, &[y, t])?;
                let c = sink.emit(TensorOp::Scalar(scale), &[])?;
                let scaled = sink.emit(TensorOp::Mul, &[g, c])?;
                let dy = sink.emit(TensorOp::Mul, &[d, scaled])?;
                if want(0) {
                    out[0] = Some(dy);
                }
                if want(1) {
                    out[1] = Some(negate(sink, dy)?);
                }
            }
        }
        TensorOp::Embedding => {
            if want(0) {
                let rows = sink.type_of(operands[0]).shape[0];
                out[0] = Some(sink.emit(TensorOp::EmbeddingGrad { rows }, &[operands[1], g])?);
            }
        }
        TensorOp::AddRows => {
            if want(0) {
                out[0] = Some(g);
            }
            if want(1) {
                out[1] = Some(sink.emit(TensorOp::SumRows, &[g])?);
            }
        }
        TensorOp::Gelu => {
            if want(0) {
                out[0] = Some(sink.emit(TensorOp::GeluGrad, &[operands[0], g])?);
            }
        }
        TensorOp::LayerNorm => {
            let (x, gamma) = (operands[0], operands[1]);
            if want(0) {
                out[0] = Some(sink.emit(TensorOp::LayerNormGradX, &[x, gamma, g])?);
            }
            if want(1) {
                out[1] = Some(sink.emit(TensorOp::LayerNormGradGamma, &[x, g])?);
            }
            if want(2) {
                out[2] = Some(sink.emit(TensorOp::SumRows, &[g])?);
            }
        }
        TensorOp::Softmax => {
            if want(0) {
                out[0] = Some(sink.emit(TensorOp::SoftmaxGrad, &[result, g])?);
            }
        }
        TensorOp::Attention { heads } => {
            let heads = *heads;
            let reads = [operands[0], operands[1], operands[2], operands[3], g];
            let rules = [
                TensorOp::AttentionGradQ { heads },
                TensorOp::AttentionGradK { heads },
                TensorOp::AttentionGradV { heads },
            ];
            for (i, rule) in rules.into_iter().enumerate() {
                if want(i) {
                    out[i] = Some(sink.emit(rule, &reads)?);
                }
            }
        }
        TensorOp::CrossEntropy => {
            if want(0) {
                let reads = [operands[0], operands[1], operands[2], g];
                out[0] = Some(sink.emit(TensorOp::CrossEntropyGrad, &reads)?);
            }
        }
        TensorOp::Dropout { p } => {
            if want(0) {
                out[0] = Some(sink.emit(TensorOp::DropoutGrad { p: *p }, &[operands[1], g])?);
            }
        }
        TensorOp::StopGradient => {}
        TensorOp::SigregColumns => {
            if want(0) {
                out[0] = Some(sink.emit(TensorOp::SigregColumnsGrad, &[operands[0], g])?);
            }
        }
        TensorOp::ReluGrad
        | TensorOp::BroadcastScalar { .. }
        | TensorOp::EmbeddingGrad { .. }
        | TensorOp::SumRows
        | TensorOp::GeluGrad
        | TensorOp::LayerNormGradX
        | TensorOp::LayerNormGradGamma
        | TensorOp::SoftmaxGrad
        | TensorOp::AttentionGradQ { .. }
        | TensorOp::AttentionGradK { .. }
        | TensorOp::AttentionGradV { .. }
        | TensorOp::CrossEntropyGrad
        | TensorOp::DropoutGrad { .. }
        | TensorOp::Sqrt
        | TensorOp::Div
        | TensorOp::Sigreg { .. }
        | TensorOp::SigregColumnsGrad => {
            return Ok(None);
        }
    }
    Ok(Some(out))
}

/// `c`, summed to rank 0 when `operand` is rank 0 and `c` is not.
fn unbroadcast<S: GradientSink + ?Sized>(
    sink: &mut S,
    c: ValueId,
    operand: ValueId,
) -> Result<ValueId, S::Error> {
    if sink.type_of(operand).rank() == 0 && sink.type_of(c).rank() > 0 {
        sink.emit(TensorOp::ReduceSum, &[c])
    } else {
        Ok(c)
    }
}

/// The rank-0 `s` broadcast to the shape of `operand`; `s` itself when `operand` is rank 0.
fn broadcast<S: GradientSink + ?Sized>(
    sink: &mut S,
    s: ValueId,
    operand: ValueId,
) -> Result<ValueId, S::Error> {
    let shape = sink.type_of(operand).shape.clone();
    if shape.is_empty() {
        Ok(s)
    } else {
        sink.emit(TensorOp::BroadcastScalar { shape }, &[s])
    }
}

/// `-c`, as `c * -1`.
fn negate<S: GradientSink + ?Sized>(sink: &mut S, c: ValueId) -> Result<ValueId, S::Error> {
    let minus_one = sink.emit(TensorOp::Scalar(-1.0), &[])?;
    sink.emit(TensorOp::Mul, &[c, minus_one])
}

/// `factor / element_count(ty)`, rounded to `f32`.
fn per_element(ty: &TensorType, factor: f64) -> f32 {
    let count: u64 = ty
        .shape
        .iter()
        .map(|dim| match *dim {
            Dim::Static(n) => n,
            Dim::Symbol(_) => unreachable!("gradient rules run on verified programs"),
        })
        .product();
    (factor / count as f64) as f32
}

fn same_dtype(op: &'static str, lhs: &TensorType, rhs: &TensorType) -> Result<(), OpError> {
    if lhs.dtype == rhs.dtype {
        Ok(())
    } else {
        Err(OpError::DTypeMismatch {
            op,
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        })
    }
}

fn rank(op: &'static str, ty: &TensorType, expected: usize) -> Result<(), OpError> {
    if ty.rank() == expected {
        Ok(())
    } else {
        Err(OpError::Rank {
            op,
            expected,
            got: ty.clone(),
        })
    }
}

fn shape_mismatch(op: &'static str, lhs: &TensorType, rhs: &TensorType) -> OpError {
    OpError::ShapeMismatch {
        op,
        lhs: lhs.clone(),
        rhs: rhs.clone(),
    }
}

fn scalar_of(ty: &TensorType) -> TensorType {
    TensorType {
        dtype: ty.dtype,
        shape: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::tensor::DType;

    fn f32(shape: &[u64]) -> TensorType {
        TensorType::f32(shape)
    }

    fn i32(shape: &[u64]) -> TensorType {
        TensorType::i32(shape)
    }

    #[test]
    fn embedding_of_a_v_by_h_table_and_n_ids_is_n_by_h() {
        assert_eq!(
            infer(&TensorOp::Embedding, &[&f32(&[7, 3]), &i32(&[10])], None),
            Ok(f32(&[10, 3]))
        );
    }

    #[test]
    fn embedding_refuses_ids_that_are_not_i32_and_a_table_that_is_not_f32() {
        assert_eq!(
            infer(&TensorOp::Embedding, &[&f32(&[7, 3]), &f32(&[10])], None),
            Err(OpError::OperandDType {
                op: "Embedding",
                index: 1,
                expected: DType::I32,
                got: f32(&[10]),
            })
        );
        assert_eq!(
            infer(&TensorOp::Embedding, &[&i32(&[7, 3]), &i32(&[10])], None),
            Err(OpError::OperandDType {
                op: "Embedding",
                index: 0,
                expected: DType::F32,
                got: i32(&[7, 3]),
            })
        );
    }

    #[test]
    fn embedding_refuses_a_table_not_of_rank_2_and_ids_not_of_rank_1() {
        assert_eq!(
            infer(&TensorOp::Embedding, &[&f32(&[7]), &i32(&[10])], None),
            Err(OpError::Rank {
                op: "Embedding",
                expected: 2,
                got: f32(&[7]),
            })
        );
        assert_eq!(
            infer(&TensorOp::Embedding, &[&f32(&[7, 3]), &i32(&[2, 5])], None),
            Err(OpError::Rank {
                op: "Embedding",
                expected: 1,
                got: i32(&[2, 5]),
            })
        );
    }

    #[test]
    fn embedding_grad_scatters_n_rows_into_a_rows_by_h_table() {
        let op = TensorOp::EmbeddingGrad {
            rows: Dim::Static(7),
        };
        assert_eq!(
            infer(&op, &[&i32(&[10]), &f32(&[10, 3])], None),
            Ok(f32(&[7, 3]))
        );
        assert_eq!(
            infer(&op, &[&i32(&[10]), &f32(&[9, 3])], None),
            Err(OpError::ShapeMismatch {
                op: "EmbeddingGrad",
                lhs: i32(&[10]),
                rhs: f32(&[9, 3]),
            })
        );
        assert_eq!(
            infer(&op, &[&f32(&[10]), &f32(&[10, 3])], None),
            Err(OpError::OperandDType {
                op: "EmbeddingGrad",
                index: 0,
                expected: DType::I32,
                got: f32(&[10]),
            })
        );
    }

    /// Every operand of every other op is `f32`: an `i32` is refused whichever side it is on,
    /// and two `i32`s are refused too.
    #[test]
    fn an_i32_operand_of_any_other_op_is_refused() {
        let ids = i32(&[4, 1]);
        let x = f32(&[4, 1]);
        for (op, operands) in [
            (TensorOp::Add, vec![&ids, &x]),
            (TensorOp::Sub, vec![&x, &ids]),
            (TensorOp::Mul, vec![&ids, &ids]),
            (TensorOp::Mse, vec![&x, &ids]),
            (TensorOp::Relu, vec![&ids]),
            (TensorOp::ReduceSum, vec![&ids]),
            (TensorOp::Transpose, vec![&ids]),
        ] {
            let index = operands.iter().position(|t| t.dtype == DType::I32).unwrap();
            assert_eq!(
                infer(&op, &operands, None),
                Err(OpError::OperandDType {
                    op: op.name(),
                    index,
                    expected: DType::F32,
                    got: ids.clone(),
                }),
                "{op:?}"
            );
        }
        let w = f32(&[1, 4]);
        assert!(matches!(
            infer(&TensorOp::MatMul, &[&ids, &w], None),
            Err(OpError::OperandDType { index: 0, .. })
        ));
    }

    /// An `i32` input is a leaf like any other; an `i32` parameter is refused.
    #[test]
    fn an_i32_input_is_declared_and_an_i32_parameter_is_refused() {
        let input = TensorOp::Input { name: "ids".into() };
        assert_eq!(infer(&input, &[], Some(&i32(&[4]))), Ok(i32(&[4])));
        let read = TensorOp::ParamRead {
            param: crate::ir::tensor::ParamId(0),
        };
        assert_eq!(
            infer(&read, &[], Some(&i32(&[4, 1]))),
            Err(OpError::ParamDType { got: i32(&[4, 1]) })
        );
    }

    #[test]
    fn add_rows_of_an_n_by_h_x_and_an_h_row_is_n_by_h() {
        assert_eq!(
            infer(&TensorOp::AddRows, &[&f32(&[5, 3]), &f32(&[3])], None),
            Ok(f32(&[5, 3]))
        );
    }

    /// A row of another length is a shape mismatch; an `x` not of rank 2 or a `b` not of rank 1
    /// is a rank error.
    #[test]
    fn add_rows_refuses_a_row_of_another_width_and_operands_of_other_ranks() {
        assert_eq!(
            infer(&TensorOp::AddRows, &[&f32(&[5, 3]), &f32(&[4])], None),
            Err(OpError::ShapeMismatch {
                op: "AddRows",
                lhs: f32(&[5, 3]),
                rhs: f32(&[4]),
            })
        );
        assert_eq!(
            infer(&TensorOp::AddRows, &[&f32(&[3]), &f32(&[3])], None),
            Err(OpError::Rank {
                op: "AddRows",
                expected: 2,
                got: f32(&[3]),
            })
        );
        assert_eq!(
            infer(&TensorOp::AddRows, &[&f32(&[5, 3]), &f32(&[1, 3])], None),
            Err(OpError::Rank {
                op: "AddRows",
                expected: 1,
                got: f32(&[1, 3]),
            })
        );
        assert_eq!(
            infer(&TensorOp::AddRows, &[&f32(&[5, 3]), &f32(&[])], None),
            Err(OpError::Rank {
                op: "AddRows",
                expected: 1,
                got: f32(&[]),
            })
        );
    }

    #[test]
    fn sum_rows_of_an_n_by_h_tensor_is_an_h_row() {
        assert_eq!(
            infer(&TensorOp::SumRows, &[&f32(&[5, 3])], None),
            Ok(f32(&[3]))
        );
        assert_eq!(
            infer(&TensorOp::SumRows, &[&f32(&[5])], None),
            Err(OpError::Rank {
                op: "SumRows",
                expected: 2,
                got: f32(&[5]),
            })
        );
    }

    /// `gelu` keeps the shape of any operand; `GeluGrad` needs its two operands of one shape.
    #[test]
    fn gelu_keeps_any_shape_and_gelu_grad_needs_one_shape() {
        for shape in [&[][..], &[7], &[5, 3]] {
            assert_eq!(infer(&TensorOp::Gelu, &[&f32(shape)], None), Ok(f32(shape)));
            assert_eq!(
                infer(&TensorOp::GeluGrad, &[&f32(shape), &f32(shape)], None),
                Ok(f32(shape))
            );
        }
        assert_eq!(
            infer(&TensorOp::GeluGrad, &[&f32(&[5, 3]), &f32(&[3, 5])], None),
            Err(OpError::ShapeMismatch {
                op: "GeluGrad",
                lhs: f32(&[5, 3]),
                rhs: f32(&[3, 5]),
            })
        );
    }

    /// The operands of the four dense ops are `f32`: row ids are refused in every position.
    #[test]
    fn the_dense_ops_refuse_an_i32_operand() {
        let x = f32(&[5, 3]);
        let b = f32(&[3]);
        for (op, operands, index) in [
            (TensorOp::AddRows, vec![i32(&[5, 3]), b.clone()], 0),
            (TensorOp::AddRows, vec![x.clone(), i32(&[3])], 1),
            (TensorOp::SumRows, vec![i32(&[5, 3])], 0),
            (TensorOp::Gelu, vec![i32(&[5, 3])], 0),
            (TensorOp::GeluGrad, vec![x.clone(), i32(&[5, 3])], 1),
        ] {
            let refs: Vec<&TensorType> = operands.iter().collect();
            assert_eq!(
                infer(&op, &refs, None),
                Err(OpError::OperandDType {
                    op: op.name(),
                    index,
                    expected: DType::F32,
                    got: operands[index].clone(),
                }),
                "{op:?}"
            );
        }
    }

    #[test]
    fn layer_norm_of_an_n_by_h_x_and_two_h_rows_is_n_by_h() {
        let (x, row) = (f32(&[4, 5]), f32(&[5]));
        assert_eq!(
            infer(&TensorOp::LayerNorm, &[&x, &row, &row], None),
            Ok(x.clone())
        );
        assert_eq!(
            infer(&TensorOp::LayerNormGradX, &[&x, &row, &x], None),
            Ok(x.clone())
        );
        assert_eq!(
            infer(&TensorOp::LayerNormGradGamma, &[&x, &x], None),
            Ok(row)
        );
    }

    /// A `gamma` or `beta` of another length is a shape mismatch against `x`; an `x` not of rank
    /// 2, or a `gamma` or `beta` not of rank 1, is a rank error.
    #[test]
    fn layer_norm_refuses_rows_of_another_width_and_operands_of_other_ranks() {
        let (x, row, long) = (f32(&[4, 5]), f32(&[5]), f32(&[6]));
        assert_eq!(
            infer(&TensorOp::LayerNorm, &[&x, &long, &row], None),
            Err(OpError::ShapeMismatch {
                op: "LayerNorm",
                lhs: x.clone(),
                rhs: long.clone(),
            })
        );
        assert_eq!(
            infer(&TensorOp::LayerNorm, &[&x, &row, &long], None),
            Err(OpError::ShapeMismatch {
                op: "LayerNorm",
                lhs: x.clone(),
                rhs: long.clone(),
            })
        );
        assert_eq!(
            infer(&TensorOp::LayerNorm, &[&row, &row, &row], None),
            Err(OpError::Rank {
                op: "LayerNorm",
                expected: 2,
                got: row.clone(),
            })
        );
        for (gamma, beta, wrong) in [
            (f32(&[1, 5]), row.clone(), f32(&[1, 5])),
            (row.clone(), f32(&[]), f32(&[])),
        ] {
            assert_eq!(
                infer(&TensorOp::LayerNorm, &[&x, &gamma, &beta], None),
                Err(OpError::Rank {
                    op: "LayerNorm",
                    expected: 1,
                    got: wrong,
                })
            );
        }
        assert_eq!(
            infer(&TensorOp::LayerNormGradX, &[&x, &row, &f32(&[5, 4])], None),
            Err(OpError::ShapeMismatch {
                op: "LayerNormGradX",
                lhs: x.clone(),
                rhs: f32(&[5, 4]),
            })
        );
        assert_eq!(
            infer(&TensorOp::LayerNormGradGamma, &[&x, &f32(&[4, 6])], None),
            Err(OpError::ShapeMismatch {
                op: "LayerNormGradGamma",
                lhs: x,
                rhs: f32(&[4, 6]),
            })
        );
    }

    /// `softmax` keeps the shape of a rank-2 operand and refuses any other rank; `SoftmaxGrad`
    /// needs two operands of one rank-2 shape.
    #[test]
    fn softmax_keeps_a_rank_2_shape_and_refuses_other_ranks() {
        let x = f32(&[4, 6]);
        assert_eq!(infer(&TensorOp::Softmax, &[&x], None), Ok(x.clone()));
        assert_eq!(
            infer(&TensorOp::SoftmaxGrad, &[&x, &x], None),
            Ok(x.clone())
        );
        for shape in [&[][..], &[6], &[2, 4, 6]] {
            assert_eq!(
                infer(&TensorOp::Softmax, &[&f32(shape)], None),
                Err(OpError::Rank {
                    op: "Softmax",
                    expected: 2,
                    got: f32(shape),
                })
            );
        }
        assert_eq!(
            infer(&TensorOp::SoftmaxGrad, &[&x, &f32(&[6, 4])], None),
            Err(OpError::ShapeMismatch {
                op: "SoftmaxGrad",
                lhs: x,
                rhs: f32(&[6, 4]),
            })
        );
    }

    /// The operands of the five norm and softmax ops are `f32`: row ids are refused in every
    /// position.
    #[test]
    fn the_norm_and_softmax_ops_refuse_an_i32_operand() {
        let x = f32(&[4, 5]);
        let row = f32(&[5]);
        for (op, operands, index) in [
            (
                TensorOp::LayerNorm,
                vec![i32(&[4, 5]), row.clone(), row.clone()],
                0,
            ),
            (
                TensorOp::LayerNorm,
                vec![x.clone(), row.clone(), i32(&[5])],
                2,
            ),
            (
                TensorOp::LayerNormGradX,
                vec![x.clone(), i32(&[5]), x.clone()],
                1,
            ),
            (
                TensorOp::LayerNormGradGamma,
                vec![x.clone(), i32(&[4, 5])],
                1,
            ),
            (TensorOp::Softmax, vec![i32(&[4, 5])], 0),
            (TensorOp::SoftmaxGrad, vec![x.clone(), i32(&[4, 5])], 1),
        ] {
            let refs: Vec<&TensorType> = operands.iter().collect();
            assert_eq!(
                infer(&op, &refs, None),
                Err(OpError::OperandDType {
                    op: op.name(),
                    index,
                    expected: DType::F32,
                    got: operands[index].clone(),
                }),
                "{op:?}"
            );
        }
    }

    /// `attention` of `q, k, v: [N, H]` and `key_mask: [B, S]` with `B * S = N` is `[N, H]`, and
    /// each gradient op, with `G` of that shape too, is `[N, H]`.
    #[test]
    fn attention_of_n_by_h_operands_and_a_b_by_s_mask_is_n_by_h() {
        let x = f32(&[6, 4]);
        let mask = f32(&[2, 3]);
        for heads in [1, 2, 4] {
            let op = TensorOp::Attention { heads };
            assert_eq!(infer(&op, &[&x, &x, &x, &mask], None), Ok(x.clone()));
            for grad in [
                TensorOp::AttentionGradQ { heads },
                TensorOp::AttentionGradK { heads },
                TensorOp::AttentionGradV { heads },
            ] {
                assert_eq!(infer(&grad, &[&x, &x, &x, &mask, &x], None), Ok(x.clone()));
            }
        }
    }

    /// `q: [B * L, H]` with `1 <= L <= S` under `k, v: [B * S, H]` and `key_mask: [B, S]` is
    /// `[B * L, H]`, `L` the rows of `q` over the rows of the mask; `AttentionGradQ` with `G` of
    /// `q`'s shape is `q`'s shape and `AttentionGradK` and `AttentionGradV` `k`'s. `q` whose rows
    /// the mask's `B` does not divide, or divides into more than `S`, a `v` of other rows than
    /// `k`, or a `k` of another width than `q`, is a shape mismatch against `q`
    /// (story:attention-fewer-query-rows).
    #[test]
    fn attention_of_fewer_query_rows_than_key_rows() {
        let (kv, mask) = (f32(&[6, 4]), f32(&[2, 3]));
        for l in [1, 2, 3] {
            let q = f32(&[2 * l, 4]);
            let op = TensorOp::Attention { heads: 2 };
            assert_eq!(infer(&op, &[&q, &kv, &kv, &mask], None), Ok(q.clone()));
            for (grad, want) in [
                (TensorOp::AttentionGradQ { heads: 2 }, &q),
                (TensorOp::AttentionGradK { heads: 2 }, &kv),
                (TensorOp::AttentionGradV { heads: 2 }, &kv),
            ] {
                assert_eq!(
                    infer(&grad, &[&q, &kv, &kv, &mask, &q], None),
                    Ok(want.clone())
                );
            }
        }
        let op = TensorOp::Attention { heads: 2 };
        for rows in [3, 5, 8] {
            let q = f32(&[rows, 4]);
            assert_eq!(
                infer(&op, &[&q, &kv, &kv, &mask], None),
                Err(OpError::ShapeMismatch {
                    op: "Attention",
                    lhs: q,
                    rhs: mask.clone(),
                }),
                "{rows} query rows"
            );
        }
        let q = f32(&[2, 4]);
        assert_eq!(
            infer(&op, &[&q, &kv, &f32(&[4, 4]), &mask], None),
            Err(OpError::ShapeMismatch {
                op: "Attention",
                lhs: q.clone(),
                rhs: f32(&[4, 4]),
            })
        );
        assert_eq!(
            infer(&op, &[&q, &f32(&[6, 2]), &f32(&[6, 2]), &mask], None),
            Err(OpError::ShapeMismatch {
                op: "Attention",
                lhs: q.clone(),
                rhs: f32(&[6, 2]),
            })
        );
        assert_eq!(
            infer(
                &TensorOp::AttentionGradK { heads: 2 },
                &[&q, &kv, &kv, &mask, &kv],
                None
            ),
            Err(OpError::ShapeMismatch {
                op: "AttentionGradK",
                lhs: q,
                rhs: kv,
            })
        );
    }

    /// An `i32 [U, 2]` fourth operand is spans (story:attention-event-spans): `q` of the rows of
    /// `k` or of `U` rows gives `q`'s shape, the gradients `q`'s or `k`'s; spans of another
    /// shape or rank, `U = 0`, or `q` of other rows is a shape mismatch against `q`.
    #[test]
    fn attention_over_spans_takes_an_i32_u_by_2_operand() {
        let (kv, spans) = (f32(&[7, 4]), i32(&[3, 2]));
        for q in [f32(&[7, 4]), f32(&[3, 4])] {
            let op = TensorOp::Attention { heads: 2 };
            assert_eq!(infer(&op, &[&q, &kv, &kv, &spans], None), Ok(q.clone()));
            for (grad, want) in [
                (TensorOp::AttentionGradQ { heads: 2 }, &q),
                (TensorOp::AttentionGradK { heads: 2 }, &kv),
                (TensorOp::AttentionGradV { heads: 2 }, &kv),
            ] {
                assert_eq!(
                    infer(&grad, &[&q, &kv, &kv, &spans, &q], None),
                    Ok(want.clone())
                );
            }
        }
        let op = TensorOp::Attention { heads: 2 };
        for (q, spans) in [
            (f32(&[4, 4]), i32(&[3, 2])),
            (f32(&[7, 4]), i32(&[3, 3])),
            (f32(&[7, 4]), i32(&[0, 2])),
            (f32(&[0, 4]), i32(&[0, 2])),
        ] {
            assert_eq!(
                infer(&op, &[&q, &kv, &kv, &spans], None),
                Err(OpError::ShapeMismatch {
                    op: "Attention",
                    lhs: q,
                    rhs: spans,
                })
            );
        }
        assert!(matches!(
            infer(&op, &[&kv, &kv, &kv, &i32(&[6])], None),
            Err(OpError::Rank { .. })
        ));
    }

    /// A key mask whose `B * S` is not `N`, or `k` of another shape, is a shape mismatch;
    /// `heads` that does not divide `H` is [`OpError::HeadsDivide`]; an operand not of rank 2 is
    /// a rank error.
    #[test]
    fn attention_refuses_a_mask_of_other_rows_heads_not_dividing_h_and_other_ranks() {
        let x = f32(&[6, 4]);
        let mask = f32(&[2, 3]);
        let op = TensorOp::Attention { heads: 2 };
        assert_eq!(
            infer(&op, &[&x, &x, &x, &f32(&[2, 4])], None),
            Err(OpError::ShapeMismatch {
                op: "Attention",
                lhs: x.clone(),
                rhs: f32(&[2, 4]),
            })
        );
        assert_eq!(
            infer(&op, &[&x, &f32(&[6, 2]), &x, &mask], None),
            Err(OpError::ShapeMismatch {
                op: "Attention",
                lhs: x.clone(),
                rhs: f32(&[6, 2]),
            })
        );
        assert_eq!(
            infer(
                &TensorOp::Attention { heads: 3 },
                &[&x, &x, &x, &mask],
                None
            ),
            Err(OpError::HeadsDivide {
                op: "Attention",
                heads: 3,
                width: Dim::Static(4),
            })
        );
        assert_eq!(
            infer(&op, &[&x, &x, &x, &f32(&[6])], None),
            Err(OpError::Rank {
                op: "Attention",
                expected: 2,
                got: f32(&[6]),
            })
        );
        assert_eq!(
            infer(
                &TensorOp::AttentionGradV { heads: 2 },
                &[&x, &x, &x, &mask, &f32(&[6, 2])],
                None
            ),
            Err(OpError::ShapeMismatch {
                op: "AttentionGradV",
                lhs: x,
                rhs: f32(&[6, 2]),
            })
        );
    }

    /// `heads` is `attention`'s one named argument, and only its key mask takes no gradient.
    #[test]
    fn attention_takes_heads_by_name_and_its_key_mask_is_constant() {
        let op = TensorOp::Attention { heads: 1 };
        assert_eq!(named_parameters(&op), &[HEADS]);
        assert_eq!(
            with_named(&op, &[NamedValue::Integer(4)]),
            TensorOp::Attention { heads: 4 }
        );
        assert_eq!(constant_operands(&op), &[(3, "key_mask")]);
        assert!(named_parameters(&TensorOp::Softmax).is_empty());
        assert!(constant_operands(&TensorOp::Softmax).is_empty());
        assert_eq!(NamedKind::PositiveInteger.integer(0), None);
        assert_eq!(
            NamedKind::PositiveInteger.integer(2),
            Some(NamedValue::Integer(2))
        );
    }

    /// `dropout` takes `p` by name and an `i32[2]` token the source does not pass; its result and
    /// its gradient op's have the type of the tensor operand, at any rank.
    #[test]
    fn dropout_takes_p_by_name_and_reads_an_i32_token() {
        let op = TensorOp::Dropout { p: 0.0 };
        assert_eq!(named_parameters(&op), &[P]);
        assert_eq!(
            with_named(&op, &[NamedValue::Float(0.25)]),
            TensorOp::Dropout { p: 0.25 }
        );
        assert!(constant_operands(&op).is_empty());
        assert_eq!(token_operand(&op), Some(1));
        assert_eq!(token_operand(&TensorOp::DropoutGrad { p: 0.5 }), Some(0));
        assert_eq!(token_operand(&TensorOp::Embedding), None);
        assert_eq!((arity(&op), source_arity(&op)), (2, 1));
        assert_eq!(source_arity(&TensorOp::MatMul), 2);
        let token = i32(&[2]);
        for x in [f32(&[]), f32(&[5]), f32(&[3, 4]), f32(&[2, 3, 4])] {
            let op = TensorOp::Dropout { p: 0.3 };
            assert_eq!(infer(&op, &[&x, &token], None), Ok(x.clone()));
            let grad = TensorOp::DropoutGrad { p: 0.3 };
            assert_eq!(infer(&grad, &[&token, &x], None), Ok(x.clone()));
        }
    }

    /// `p` outside `[0, 1)` and a token that is not `i32[2]` are refused; `f32` tokens and `i32`
    /// tensors are dtype errors.
    #[test]
    fn dropout_refuses_rates_outside_0_1_and_tokens_other_than_i32_2() {
        let (x, token) = (f32(&[3, 4]), i32(&[2]));
        for p in [1.0f32, -0.1, f32::NAN, 1.5] {
            assert_eq!(
                infer(&TensorOp::Dropout { p }, &[&x, &token], None),
                Err(OpError::DropoutRate {
                    op: "Dropout",
                    bits: p.to_bits()
                })
            );
        }
        assert!(infer(&TensorOp::Dropout { p: -0.0 }, &[&x, &token], None).is_ok());
        assert_eq!(
            infer(&TensorOp::Dropout { p: 0.5 }, &[&x, &i32(&[3])], None),
            Err(OpError::Token {
                op: "Dropout",
                index: 1,
                got: i32(&[3])
            })
        );
        assert_eq!(
            infer(&TensorOp::DropoutGrad { p: 0.5 }, &[&f32(&[2]), &x], None),
            Err(OpError::OperandDType {
                op: "DropoutGrad",
                index: 0,
                expected: DType::I32,
                got: f32(&[2])
            })
        );
        assert_eq!(
            infer(
                &TensorOp::Dropout { p: 0.5 },
                &[&i32(&[3, 4]), &token],
                None
            ),
            Err(OpError::OperandDType {
                op: "Dropout",
                index: 0,
                expected: DType::F32,
                got: i32(&[3, 4])
            })
        );
        assert_eq!(
            OpError::DropoutRate {
                op: "Dropout",
                bits: 1.0f32.to_bits()
            }
            .to_string(),
            "Dropout needs p in [0, 1); got p = 1.0"
        );
    }

    /// A probability is a float literal whose `f32` rounding is in `[0, 1)`; an integer literal is
    /// not one, and a positive integer kind takes no float.
    #[test]
    fn a_probability_is_a_float_literal_in_0_1() {
        let kind = NamedKind::Probability;
        assert_eq!(kind.float(0.0), Some(NamedValue::Float(0.0)));
        assert_eq!(kind.float(0.3), Some(NamedValue::Float(0.3)));
        assert_eq!(kind.float(1.0), None);
        assert_eq!(kind.float(-0.1), None);
        assert_eq!(kind.float(0.999_999_999), None, "rounds to 1.0f32");
        assert_eq!(kind.integer(0), None);
        assert_eq!(NamedKind::PositiveInteger.float(2.0), None);
        assert_eq!(NamedValue::Float(0.3).to_string(), "0.3");
        assert_eq!(NamedValue::Float(0.0).to_string(), "0.0");
    }

    /// A sink that types each emitted node through [`infer`] and records it.
    struct Recorder {
        types: Vec<TensorType>,
        emitted: Vec<(TensorOp, Vec<ValueId>)>,
    }

    impl GradientSink for Recorder {
        type Error = OpError;

        fn emit(&mut self, op: TensorOp, operands: &[ValueId]) -> Result<ValueId, OpError> {
            let types: Vec<&TensorType> =
                operands.iter().map(|o| &self.types[o.0 as usize]).collect();
            let ty = infer(&op, &types, None)?;
            self.types.push(ty);
            self.emitted.push((op, operands.to_vec()));
            Ok(ValueId(self.types.len() as u32 - 1))
        }

        fn type_of(&self, value: ValueId) -> &TensorType {
            &self.types[value.0 as usize]
        }
    }

    /// The gradient of `Dropout` is `DropoutGrad` over the token and `G`; the token takes none,
    /// and `DropoutGrad` has no rule.
    #[test]
    fn dropout_gradient_reads_the_token_and_g() {
        let (x, token, y, g) = (ValueId(0), ValueId(1), ValueId(2), ValueId(3));
        let mut sink = Recorder {
            types: vec![f32(&[3, 4]), i32(&[2]), f32(&[3, 4]), f32(&[3, 4])],
            emitted: Vec::new(),
        };
        let op = TensorOp::Dropout { p: 0.25 };
        let out = gradient(&mut sink, &op, &[x, token], y, g, &[true, true])
            .unwrap()
            .unwrap();
        assert_eq!(out, vec![Some(ValueId(4)), None]);
        assert_eq!(
            sink.emitted,
            vec![(TensorOp::DropoutGrad { p: 0.25 }, vec![token, g])]
        );
        let grad = TensorOp::DropoutGrad { p: 0.25 };
        assert_eq!(
            gradient(&mut sink, &grad, &[token, g], ValueId(4), g, &[false, true]),
            Ok(None)
        );
    }

    /// `stop_gradient` keeps the type of any `f32` operand, refuses an `i32` one, and its rule
    /// gives its operand no contribution and emits nothing, even when the operand is wanted.
    #[test]
    fn stop_gradient_keeps_the_type_and_gives_no_gradient() {
        for shape in [&[][..], &[7], &[5, 3], &[2, 3, 4]] {
            let ty = f32(shape);
            assert_eq!(infer(&TensorOp::StopGradient, &[&ty], None), Ok(ty));
        }
        assert!(matches!(
            infer(&TensorOp::StopGradient, &[&i32(&[4])], None),
            Err(OpError::OperandDType { index: 0, .. })
        ));
        assert_eq!(arity(&TensorOp::StopGradient), 1);
        assert!(constant_operands(&TensorOp::StopGradient).is_empty());
        let (x, y, g) = (ValueId(0), ValueId(1), ValueId(2));
        let mut sink = Recorder {
            types: vec![f32(&[3, 4]), f32(&[3, 4]), f32(&[3, 4])],
            emitted: Vec::new(),
        };
        let out = gradient(&mut sink, &TensorOp::StopGradient, &[x], y, g, &[true]);
        assert_eq!(out, Ok(Some(vec![None])));
        assert!(sink.emitted.is_empty());
    }

    /// `cross_entropy` of `[N, V]` logits, `i32[N]` labels and `f32[N]` weights is rank 0; its
    /// gradient op, with a rank-0 `G`, has the type of the logits.
    #[test]
    fn cross_entropy_of_n_by_v_logits_and_n_labels_and_weights_is_rank_0() {
        let (logits, labels, weights) = (f32(&[6, 5]), i32(&[6]), f32(&[6]));
        assert_eq!(
            infer(&TensorOp::CrossEntropy, &[&logits, &labels, &weights], None),
            Ok(f32(&[]))
        );
        assert_eq!(
            infer(
                &TensorOp::CrossEntropyGrad,
                &[&logits, &labels, &weights, &f32(&[])],
                None
            ),
            Ok(logits.clone())
        );
    }

    /// `f32` labels or `i32` weights or logits are dtype errors at their index; labels or weights
    /// of another length are a shape mismatch against the logits; logits not of rank 2, labels or
    /// weights not of rank 1, or a `G` not of rank 0, are rank errors.
    #[test]
    fn cross_entropy_refuses_other_dtypes_lengths_and_ranks() {
        let (logits, labels, weights) = (f32(&[6, 5]), i32(&[6]), f32(&[6]));
        let op = TensorOp::CrossEntropy;
        for (operands, index, expected) in [
            (vec![&logits, &weights, &weights], 1, DType::I32),
            (vec![&logits, &labels, &labels], 2, DType::F32),
            (vec![&i32(&[6, 5]), &labels, &weights], 0, DType::F32),
        ] {
            assert_eq!(
                infer(&op, &operands, None),
                Err(OpError::OperandDType {
                    op: "CrossEntropy",
                    index,
                    expected,
                    got: operands[index].clone(),
                })
            );
        }
        assert_eq!(
            infer(&op, &[&logits, &i32(&[7]), &weights], None),
            Err(OpError::ShapeMismatch {
                op: "CrossEntropy",
                lhs: logits.clone(),
                rhs: i32(&[7]),
            })
        );
        assert_eq!(
            infer(&op, &[&logits, &labels, &f32(&[5])], None),
            Err(OpError::ShapeMismatch {
                op: "CrossEntropy",
                lhs: logits.clone(),
                rhs: f32(&[5]),
            })
        );
        for (operands, expected, got) in [
            (
                vec![f32(&[6]), labels.clone(), weights.clone()],
                2,
                f32(&[6]),
            ),
            (
                vec![logits.clone(), i32(&[6, 1]), weights.clone()],
                1,
                i32(&[6, 1]),
            ),
            (vec![logits.clone(), labels.clone(), f32(&[])], 1, f32(&[])),
        ] {
            let refs: Vec<&TensorType> = operands.iter().collect();
            assert_eq!(
                infer(&op, &refs, None),
                Err(OpError::Rank {
                    op: "CrossEntropy",
                    expected,
                    got,
                })
            );
        }
        assert_eq!(
            infer(
                &TensorOp::CrossEntropyGrad,
                &[&logits, &labels, &weights, &f32(&[1])],
                None
            ),
            Err(OpError::Rank {
                op: "CrossEntropyGrad",
                expected: 0,
                got: f32(&[1]),
            })
        );
    }

    /// The labels and weights of `cross_entropy` take no gradient; it takes no named argument.
    #[test]
    fn cross_entropy_labels_and_weights_are_constant() {
        assert_eq!(
            constant_operands(&TensorOp::CrossEntropy),
            &[(1, "labels"), (2, "weights")]
        );
        assert!(named_parameters(&TensorOp::CrossEntropy).is_empty());
        assert_eq!(operand_dtype(&TensorOp::CrossEntropyGrad, 1), DType::I32);
        assert_eq!(operand_dtype(&TensorOp::CrossEntropyGrad, 3), DType::F32);
    }

    /// `sigreg(z: f32[N, K], slices = M) -> f32[]`; `z` of another rank is a rank error, an
    /// `i32` `z` an operand dtype error; `slices` is a positive integer and the op is expanded by
    /// the lowering.
    #[test]
    fn sigreg_types_a_rank_2_f32_operand_to_a_scalar() {
        let op = with_named(&TensorOp::Sigreg { slices: 1 }, &[NamedValue::Integer(16)]);
        assert_eq!(op, TensorOp::Sigreg { slices: 16 });
        assert_eq!(named_parameters(&op), &[SLICES]);
        assert_eq!(SLICES.kind.integer(0), None);
        assert!(expanded(&op));
        assert!(!expanded(&TensorOp::SigregColumns));
        assert_eq!(source_arity(&op), 1);
        assert!(constant_operands(&op).is_empty());
        assert_eq!(infer(&op, &[&f32(&[33, 8])], None), Ok(f32(&[])));
        assert_eq!(
            derived_types(&op, &[&f32(&[33, 8])]),
            [f32(&[8, 16]), f32(&[33, 16]), f32(&[16])]
        );
        assert!(derived_types(&op, &[&f32(&[8])]).is_empty());
        assert!(derived_types(&TensorOp::SigregColumns, &[&f32(&[33, 8])]).is_empty());
        assert_eq!(
            infer(&op, &[&f32(&[8])], None),
            Err(OpError::Rank {
                op: "Sigreg",
                expected: 2,
                got: f32(&[8]),
            })
        );
        assert_eq!(
            infer(&op, &[&i32(&[33, 8])], None),
            Err(OpError::OperandDType {
                op: "Sigreg",
                index: 0,
                expected: DType::F32,
                got: i32(&[33, 8]),
            })
        );
    }

    /// `SigregColumns(P: [N, M]) -> [M]`; `SigregColumnsGrad(P, G: [M]) -> [N, M]`, a `G` of
    /// another length a shape mismatch; the gradient of `SigregColumns` is `SigregColumnsGrad`
    /// of its operand and the adjoint.
    #[test]
    fn sigreg_columns_and_its_gradient_type_per_column() {
        let p = f32(&[33, 16]);
        assert_eq!(infer(&TensorOp::SigregColumns, &[&p], None), Ok(f32(&[16])));
        assert_eq!(
            infer(&TensorOp::SigregColumnsGrad, &[&p, &f32(&[16])], None),
            Ok(p.clone())
        );
        assert_eq!(
            infer(&TensorOp::SigregColumnsGrad, &[&p, &f32(&[15])], None),
            Err(OpError::ShapeMismatch {
                op: "SigregColumnsGrad",
                lhs: p.clone(),
                rhs: f32(&[15]),
            })
        );
    }
}
