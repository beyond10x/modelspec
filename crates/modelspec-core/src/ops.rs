//! The operator registry: per op, its arity and its shape and dtype rule (DESIGN.md § 2.2, "one
//! operator registry").
//!
//! The builder, the verifier and the type checker all type an op through [`infer`], so the three
//! cannot disagree. A rule failure is an [`OpError`] carrying the types involved; the checker maps
//! it to a user diagnostic (a [`OpError::MatMulContracting`] is `E-TYPE-012`). The reverse-mode
//! gradient rule of each op is [`gradient`], used by [`crate::autodiff`].

use thiserror::Error;

use crate::ir::tensor::{Dim, TensorOp, TensorType, ValueId};

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
}

/// The number of operands `op` takes.
pub fn arity(op: &TensorOp) -> usize {
    match op {
        TensorOp::Input { .. } | TensorOp::ParamRead { .. } | TensorOp::Scalar(_) => 0,
        TensorOp::Relu
        | TensorOp::Transpose
        | TensorOp::ReduceSum
        | TensorOp::ReduceMean
        | TensorOp::BroadcastScalar { .. } => 1,
        TensorOp::Add
        | TensorOp::Sub
        | TensorOp::Mul
        | TensorOp::MatMul
        | TensorOp::Mse
        | TensorOp::ReluGrad => 2,
    }
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
    match op {
        TensorOp::Input { .. } | TensorOp::ParamRead { .. } => declared
            .cloned()
            .ok_or(OpError::MissingDeclaredType { op: name }),
        TensorOp::Scalar(_) => Ok(TensorType::f32_scalar()),
        TensorOp::Add | TensorOp::Sub | TensorOp::Mul => {
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
        TensorOp::ReluGrad => {
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
        TensorOp::Relu => Ok(operands[0].clone()),
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
    }
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
/// `operands` are the node's operands and `g` the adjoint of its result. Returns, in operand
/// order, the contribution to the adjoint of each operand `i` with `wanted[i]`, and `None` for
/// the others; a contribution has its operand's type. The caller sums the contributions of every
/// use of a value. An operand of rank 0 under a same-shape-or-scalar op receives the sum of its
/// broadcast contribution. Returns `Ok(None)` for the internal gradient ops `ReluGrad` and
/// `BroadcastScalar`, which have no rule.
///
/// The rules, for `Y = op(A, B)` and `G = dL/dY`:
///
/// | op | `dA` | `dB` |
/// | --- | --- | --- |
/// | `Add` | `G` | `G` |
/// | `Sub` | `G` | `-G` |
/// | `Mul` | `G * B` | `G * A` |
/// | `MatMul` | `G @ transpose(B)` | `transpose(A) @ G` |
/// | `Relu` | `where(A > 0, G, 0)` (`ReluGrad`; 0 at `A = 0`) | |
/// | `Transpose` | `transpose(G)` | |
/// | `ReduceSum` | `broadcast_scalar(G, shape(A))` | |
/// | `ReduceMean` | `broadcast_scalar(G * (1 / numel(A)), shape(A))` | |
/// | `Mse` | `(A - B) * (G * (2 / numel(A)))` | `-dA` |
///
/// The constants `1 / numel` and `2 / numel` are `f32` scalars. `Mse`'s `B` is a target in a
/// training program, an input that receives no gradient; a trainable `B` receives `-dA`.
pub fn gradient<S: GradientSink + ?Sized>(
    sink: &mut S,
    op: &TensorOp,
    operands: &[ValueId],
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
        TensorOp::ReluGrad | TensorOp::BroadcastScalar { .. } => return Ok(None),
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
