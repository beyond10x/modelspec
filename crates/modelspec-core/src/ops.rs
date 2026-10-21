//! The operator registry: per op, its arity and its shape and dtype rule (DESIGN.md § 2.2, "one
//! operator registry").
//!
//! The builder, the verifier and the type checker all type an op through [`infer`], so the three
//! cannot disagree. A rule failure is an [`OpError`] carrying the types involved; the checker maps
//! it to a user diagnostic (a [`OpError::MatMulContracting`] is `E-TYPE-012`). Gradient rules are
//! added with reverse-mode autodiff.

use thiserror::Error;

use crate::ir::tensor::{Dim, TensorOp, TensorType};

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
