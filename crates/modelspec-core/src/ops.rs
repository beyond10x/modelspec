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
        | TensorOp::Softmax => 1,
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
        | TensorOp::SoftmaxGrad => 2,
        TensorOp::LayerNorm | TensorOp::LayerNormGradX => 3,
    }
}

/// The element type operand `index` of `op` must have: `i32` for the ids of `Embedding` and
/// `EmbeddingGrad`, `f32` for every other operand (DESIGN.md § 2.2).
pub fn operand_dtype(op: &TensorOp, index: usize) -> DType {
    match (op, index) {
        (TensorOp::Embedding, 1) | (TensorOp::EmbeddingGrad { .. }, 0) => DType::I32,
        _ => DType::F32,
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
    for (index, operand) in operands.iter().enumerate() {
        let expected = operand_dtype(op, index);
        if operand.dtype != expected {
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
        TensorOp::Relu | TensorOp::Gelu => Ok(operands[0].clone()),
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
    }
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
/// `LayerNormGradX`, `LayerNormGradGamma` and `SoftmaxGrad`, which have no rule.
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
///
/// The constants `1 / numel` and `2 / numel` are `f32` scalars. `Mse`'s `B` is a target in a
/// training program, an input that receives no gradient; a trainable `B` receives `-dA`.
/// `EmbeddingGrad` adds the rows of `G` in ascending `n`, so a row read twice accumulates in a
/// fixed order, and a row never read stays exactly zero. `SumRows` starts from row 0 and adds
/// each later row in ascending order. `gelu'` is the derivative of the tanh form of `Gelu`. For
/// `LayerNorm`, `xhat = (A - mean) * rstd` and `rstd = 1 / sqrt(var + 1e-12)` are recomputed from
/// `A` per row; `Softmax`'s rule reads its own result `Y`.
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
        TensorOp::ReluGrad
        | TensorOp::BroadcastScalar { .. }
        | TensorOp::EmbeddingGrad { .. }
        | TensorOp::SumRows
        | TensorOp::GeluGrad
        | TensorOp::LayerNormGradX
        | TensorOp::LayerNormGradGamma
        | TensorOp::SoftmaxGrad => {
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
}
