//! Reverse-mode automatic differentiation over tensor IR (DESIGN.md § 3.3 pass 4, § 4).
//!
//! [`differentiate`] generates one backward graph on the tensor SSA of a forward program, not a
//! runtime tape. It walks the nodes in reverse definition order from a rank-0 loss, applies each
//! op's gradient rule ([`crate::ops::gradient`]) and sums every incoming adjoint of a value
//! exactly once per use. Only nodes the loss depends on and that depend on a parameter read take
//! part, so inputs and targets receive no gradient and a parameter the loss does not reach has
//! none. Every gradient is `f32`, the program's dtype.
//!
//! [`differentiate_inputs`] runs the same walk with the same rules from any rank-0 target and
//! gives the adjoint of named inputs instead, in a separate program; [`output_element_target`]
//! builds such a target for one element of a model's prediction. The training program
//! [`differentiate`] produces does not change.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::ir::tensor::{
    DType, Dim, ParamId, TensorNode, TensorOp, TensorProgram, TensorType, ValueId,
};
use crate::ir::training::{Subgraph, TrainingProgram};
use crate::ir::verify::{TrainingVerifyError, VerifyError, verify, verify_training};
use crate::ops::{self, GradientSink, OpError};

/// Why a program could not be differentiated.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum AutodiffError {
    #[error("program does not verify: {0}")]
    Verify(#[from] VerifyError),
    #[error("loss {loss} is not defined")]
    UndefinedLoss { loss: ValueId },
    #[error("loss {loss} is {ty}, not rank 0")]
    LossNotScalar { loss: ValueId, ty: TensorType },
    #[error("{node}: {op} has no gradient rule")]
    NoGradientRule { node: ValueId, op: &'static str },
    /// A gradient rule built a node the registry refuses: a compiler bug.
    #[error("gradient rule of {node}: {source}")]
    Rule { node: ValueId, source: OpError },
    /// The backward graph needs a value id past `u32::MAX`: the forward program's ids leave too
    /// little room above its largest one.
    #[error("no value id is left for the backward graph: ids run past u32::MAX")]
    ValueIdsExhausted,
    /// The generated program breaks a Training IR invariant: a compiler bug.
    #[error("generated training program does not verify: {0}")]
    Training(#[from] TrainingVerifyError),
}

/// Differentiates `forward` with respect to its parameters, from the rank-0 value `loss`.
///
/// The result keeps every node of `forward` unchanged and tagged [`Subgraph::Forward`] (the loss
/// [`Subgraph::Loss`]), followed by the backward seed `1.0` (tagged `Loss`) and the nodes the
/// gradient rules build: tagged [`Subgraph::Gradient`] when they depend on the seed, `Forward`
/// when they do not (see [`Sink`]). Its outputs are the loss, then the gradient of each
/// loss-reachable parameter in `ParamId` order; the outputs of `forward` are dropped. The result
/// passes [`verify_training`].
pub fn differentiate(
    forward: &TensorProgram,
    loss: ValueId,
) -> Result<TrainingProgram, AutodiffError> {
    verify(forward)?;
    let nodes = &forward.nodes;
    let position: BTreeMap<ValueId, usize> =
        nodes.iter().enumerate().map(|(i, n)| (n.id, i)).collect();
    let &loss_at = position
        .get(&loss)
        .ok_or(AutodiffError::UndefinedLoss { loss })?;
    if nodes[loss_at].ty.rank() != 0 {
        return Err(AutodiffError::LossNotScalar {
            loss,
            ty: nodes[loss_at].ty.clone(),
        });
    }

    let trainable = dependent_nodes(nodes, &position, |node| {
        matches!(node.op, TensorOp::ParamRead { .. })
    });
    let mut sink = Sink::new(forward, loss);
    let mut reads: BTreeMap<ParamId, Vec<ValueId>> = BTreeMap::new();
    for (leaf, g) in reverse_walk(&mut sink, nodes, &position, &trainable, loss_at)? {
        if let TensorOp::ParamRead { param } = nodes[position[&leaf]].op {
            reads.entry(param).or_default().push(g);
        }
    }

    let mut gradients = BTreeMap::new();
    for (param, read_adjoints) in reads {
        let g = sink.sum(&read_adjoints).map_err(rule_error(loss))?;
        gradients.insert(param, g);
    }
    let mut program = sink.program;
    program.outputs = std::iter::once(loss)
        .chain(gradients.values().copied())
        .collect();
    let training = TrainingProgram {
        program,
        subgraphs: sink.subgraphs,
        loss,
        gradients,
    };
    verify_training(&training)?;
    Ok(training)
}

/// The reverse walk from the rank-0 value at position `target_at`: seeds it with `1.0` when it
/// depends on a source (`reaches[target_at]`), then visits the nodes in reverse definition order,
/// sums each node's incoming adjoints once and applies its gradient rule, asking only for the
/// operands that depend on a source (`reaches`). Returns the summed adjoint of each `ParamRead`
/// and `Input` node that received one, in the order the walk reached them.
///
/// Target reachability needs no mask: adjoints flow only from the seed through operands, so a
/// node the target does not depend on never receives one.
fn reverse_walk(
    sink: &mut Sink,
    nodes: &[TensorNode],
    position: &BTreeMap<ValueId, usize>,
    reaches: &[bool],
    target_at: usize,
) -> Result<Vec<(ValueId, ValueId)>, AutodiffError> {
    let target = nodes[target_at].id;
    let mut adjoints: BTreeMap<ValueId, Vec<ValueId>> = BTreeMap::new();
    let mut leaves = Vec::new();
    if reaches[target_at] {
        sink.tag = Subgraph::Loss;
        let seed = sink
            .push(TensorOp::Scalar(1.0), &[])
            .map_err(rule_error(target))?;
        sink.tag = Subgraph::Gradient;
        sink.seed = Some(seed);
        adjoints.insert(target, vec![seed]);
    }

    for node in nodes[..=target_at].iter().rev() {
        let Some(contributions) = adjoints.remove(&node.id) else {
            continue;
        };
        let g = sink.sum(&contributions).map_err(rule_error(node.id))?;
        if matches!(node.op, TensorOp::ParamRead { .. } | TensorOp::Input { .. }) {
            leaves.push((node.id, g));
            continue;
        }
        let wanted: Vec<bool> = node.operands.iter().map(|o| reaches[position[o]]).collect();
        let per_operand = ops::gradient(sink, &node.op, &node.operands, node.id, g, &wanted)
            .map_err(rule_error(node.id))?
            .ok_or(AutodiffError::NoGradientRule {
                node: node.id,
                op: node.op.name(),
            })?;
        for (&operand, contribution) in node.operands.iter().zip(per_operand) {
            if let Some(c) = contribution {
                adjoints.entry(operand).or_default().push(c);
            }
        }
    }
    Ok(leaves)
}

/// Per node position: the node is a source or depends on one through its operands.
fn dependent_nodes(
    nodes: &[TensorNode],
    position: &BTreeMap<ValueId, usize>,
    is_source: impl Fn(&TensorNode) -> bool,
) -> Vec<bool> {
    let mut reaches = vec![false; nodes.len()];
    for (i, node) in nodes.iter().enumerate() {
        reaches[i] = is_source(node) || node.operands.iter().any(|o| reaches[position[o]]);
    }
    reaches
}

/// The name an [`output_element_target`] gives the one-hot mask input it adds.
pub const TARGET_MASK: &str = "target_mask";

/// Why an input adjoint or an output-element target could not be built. A library error with no
/// diagnostic code, like [`AutodiffError`].
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum InputAdjointError {
    #[error("program does not verify: {0}")]
    Verify(#[from] VerifyError),
    #[error("target {target} is not defined")]
    UndefinedTarget { target: ValueId },
    #[error("target {target} is {ty}, not rank 0")]
    TargetNotScalar { target: ValueId, ty: TensorType },
    /// The name is not that of an `Input` node of the program. After dead-code elimination a
    /// declared input the loss does not read has none.
    #[error("`{name}` is not an input of the program")]
    UnknownInput { name: String },
    /// The input is `i32` row ids, which have no gradient (DESIGN.md § 4).
    #[error("`{name}` is an i32 input; row ids have no gradient")]
    IntegerInput { name: String },
    #[error("prediction {prediction} is not defined")]
    UndefinedPrediction { prediction: ValueId },
    #[error("an index of rank {got} for a prediction of rank {expected}")]
    IndexRank { expected: usize, got: usize },
    #[error("index {index:?} is out of bounds for {ty}")]
    IndexOutOfBounds { index: Vec<u64>, ty: TensorType },
    /// The program already has an input named [`TARGET_MASK`].
    #[error("the program already has an input named `{TARGET_MASK}`")]
    MaskNameTaken,
    /// A new node needs a value id past `u32::MAX`.
    #[error("no value id is left for the new nodes: ids run past u32::MAX")]
    ValueIdsExhausted,
    /// The reverse walk failed: an op without a gradient rule, or a rule's node the registry
    /// refuses.
    #[error(transparent)]
    Autodiff(AutodiffError),
    /// The generated program breaks a Tensor IR invariant: a compiler bug.
    #[error("generated program does not verify: {0}")]
    Generated(VerifyError),
}

impl From<AutodiffError> for InputAdjointError {
    fn from(error: AutodiffError) -> Self {
        match error {
            AutodiffError::ValueIdsExhausted => InputAdjointError::ValueIdsExhausted,
            other => InputAdjointError::Autodiff(other),
        }
    }
}

/// The adjoint of named inputs of a rank-0 target, in its own program.
#[derive(Clone, Debug, PartialEq)]
pub struct InputAdjoint {
    /// The forward nodes unchanged, then the backward nodes. Outputs: the target, then each of
    /// [`Self::adjoints`] in order. Inputs and parameter table are the forward program's.
    pub program: TensorProgram,
    /// The rank-0 target.
    pub target: ValueId,
    /// Each input asked for, by name in the order asked, and the value of its adjoint, which has
    /// the input's type.
    pub adjoints: Vec<(String, ValueId)>,
}

/// Differentiates the rank-0 value `target` of `forward` with respect to the inputs `inputs`
/// (DESIGN.md § 3.3 pass 4, § 4).
///
/// The walk and the gradient-rule registry are those of [`differentiate`], with the named
/// `Input` nodes as the sources in place of the parameter reads: only nodes the target depends
/// on and that depend on a named input take part, so no parameter receives a gradient here. An
/// input the target does not read has the adjoint zero, a `Scalar 0.0` broadcast to its shape
/// with `BroadcastScalar` (the `Scalar` itself at rank 0). A name asked for twice is listed twice
/// with one adjoint. The result passes [`verify`]; `forward` and every training program are left
/// as they are. An `i32` input is refused: row ids have no gradient.
pub fn differentiate_inputs(
    forward: &TensorProgram,
    target: ValueId,
    inputs: &[&str],
) -> Result<InputAdjoint, InputAdjointError> {
    verify(forward)?;
    let nodes = &forward.nodes;
    let position: BTreeMap<ValueId, usize> =
        nodes.iter().enumerate().map(|(i, n)| (n.id, i)).collect();
    let &target_at = position
        .get(&target)
        .ok_or(InputAdjointError::UndefinedTarget { target })?;
    if nodes[target_at].ty.rank() != 0 {
        return Err(InputAdjointError::TargetNotScalar {
            target,
            ty: nodes[target_at].ty.clone(),
        });
    }
    let mut asked = Vec::with_capacity(inputs.len());
    for &name in inputs {
        let value = input_named(forward, name).ok_or_else(|| InputAdjointError::UnknownInput {
            name: name.to_owned(),
        })?;
        if nodes[position[&value]].ty.dtype != DType::F32 {
            return Err(InputAdjointError::IntegerInput {
                name: name.to_owned(),
            });
        }
        asked.push((name, value));
    }

    let sources: BTreeSet<ValueId> = asked.iter().map(|&(_, v)| v).collect();
    let reaches = dependent_nodes(nodes, &position, |node| sources.contains(&node.id));
    let mut sink = Sink::new(forward, target);
    let mut resolved: BTreeMap<ValueId, ValueId> =
        reverse_walk(&mut sink, nodes, &position, &reaches, target_at)?
            .into_iter()
            .collect();
    let mut zero = None;
    let mut adjoints = Vec::with_capacity(asked.len());
    for (name, value) in asked {
        let adjoint = match resolved.get(&value) {
            Some(&g) => g,
            None => {
                let z = match zero {
                    Some(z) => z,
                    None => *zero.insert(
                        sink.push(TensorOp::Scalar(0.0), &[])
                            .map_err(rule_error(value))?,
                    ),
                };
                let shape = nodes[position[&value]].ty.shape.clone();
                let g = if shape.is_empty() {
                    z
                } else {
                    sink.push(TensorOp::BroadcastScalar { shape }, &[z])
                        .map_err(rule_error(value))?
                };
                resolved.insert(value, g);
                g
            }
        };
        adjoints.push((name.to_owned(), adjoint));
    }

    let mut program = sink.program;
    program.outputs = std::iter::once(target)
        .chain(adjoints.iter().map(|&(_, v)| v))
        .collect();
    verify(&program).map_err(InputAdjointError::Generated)?;
    Ok(InputAdjoint {
        program,
        target,
        adjoints,
    })
}

/// A rank-0 target for one element of a prediction, in its own program.
#[derive(Clone, Debug, PartialEq)]
pub struct OutputTarget {
    /// The forward nodes unchanged, then the [`TARGET_MASK`] input, `Mul prediction, mask` and
    /// `ReduceSum` of it. Its one output is the target; its inputs are the forward program's,
    /// then the mask.
    pub program: TensorProgram,
    /// The rank-0 target `sum(prediction * mask)`.
    pub target: ValueId,
    /// The [`TARGET_MASK`] input, of the prediction's type. The caller binds it to the one-hot
    /// tensor with `1` at [`Self::offset`], so the target is that element.
    pub mask: ValueId,
    /// The row-major offset of the element.
    pub offset: u64,
}

/// Builds the rank-0 target for element `index` of `prediction`, a value of `forward`, from
/// existing ops: an `Input` named [`TARGET_MASK`] of the prediction's type, `Mul prediction,
/// target_mask` and `ReduceSum`. The result passes [`verify`]; [`differentiate_inputs`] takes it
/// with [`OutputTarget::target`].
pub fn output_element_target(
    forward: &TensorProgram,
    prediction: ValueId,
    index: &[u64],
) -> Result<OutputTarget, InputAdjointError> {
    verify(forward)?;
    let ty = forward
        .node(prediction)
        .ok_or(InputAdjointError::UndefinedPrediction { prediction })?
        .ty
        .clone();
    if input_named(forward, TARGET_MASK).is_some() {
        return Err(InputAdjointError::MaskNameTaken);
    }
    if index.len() != ty.rank() {
        return Err(InputAdjointError::IndexRank {
            expected: ty.rank(),
            got: index.len(),
        });
    }
    let mut offset: u64 = 0;
    for (&i, dim) in index.iter().zip(&ty.shape) {
        let Dim::Static(n) = *dim else {
            unreachable!("a verified program has no symbolic dimension")
        };
        if i >= n {
            return Err(InputAdjointError::IndexOutOfBounds {
                index: index.to_vec(),
                ty: ty.clone(),
            });
        }
        offset = offset * n + i;
    }

    let mut program = forward.clone();
    let mut next = match forward.nodes.iter().map(|n| n.id.0).max() {
        Some(largest) => largest.checked_add(1),
        None => Some(0),
    };
    let mut append = |program: &mut TensorProgram,
                      op: TensorOp,
                      operands: &[ValueId]|
     -> Result<ValueId, InputAdjointError> {
        let id = ValueId(next.ok_or(InputAdjointError::ValueIdsExhausted)?);
        next = id.0.checked_add(1);
        let operand_types: Vec<&TensorType> = operands
            .iter()
            .map(|&o| &program.node(o).expect("an operand defined above").ty)
            .collect();
        let result = ops::infer(&op, &operand_types, Some(&ty))
            .map_err(|source| AutodiffError::Rule { node: id, source })?;
        program.nodes.push(TensorNode {
            id,
            op,
            operands: operands.to_vec(),
            ty: result,
            source_span: None,
        });
        Ok(id)
    };
    let mask = append(
        &mut program,
        TensorOp::Input {
            name: TARGET_MASK.to_owned(),
        },
        &[],
    )?;
    program.inputs.push(mask);
    let product = append(&mut program, TensorOp::Mul, &[prediction, mask])?;
    let target = append(&mut program, TensorOp::ReduceSum, &[product])?;
    program.outputs = vec![target];
    verify(&program).map_err(InputAdjointError::Generated)?;
    Ok(OutputTarget {
        program,
        target,
        mask,
        offset,
    })
}

/// The `Input` node of `program` named `name`.
fn input_named(program: &TensorProgram, name: &str) -> Option<ValueId> {
    program.inputs.iter().copied().find(|&v| {
        matches!(&program.node(v).map(|n| &n.op), Some(TensorOp::Input { name: n }) if n == name)
    })
}

/// The [`AutodiffError`] for a failed emission while differentiating `node`.
fn rule_error(node: ValueId) -> impl Fn(SinkError) -> AutodiffError {
    move |error| match error {
        SinkError::Op(source) => AutodiffError::Rule { node, source },
        SinkError::IdsExhausted => AutodiffError::ValueIdsExhausted,
    }
}

/// Why [`Sink`] could not emit a node.
#[derive(Debug)]
enum SinkError {
    /// The registry refused the node.
    Op(OpError),
    /// No value id is left: the last one handed out was `u32::MAX`.
    IdsExhausted,
}

impl From<OpError> for SinkError {
    fn from(error: OpError) -> Self {
        SinkError::Op(error)
    }
}

/// The program under construction: the forward nodes, then each emitted node, typed through the
/// registry. An emitted node is tagged [`Sink::tag`] when an operand is the seed or tagged
/// `Gradient`, that is when it depends on the seed, and `Forward` otherwise: a rule's constant
/// or a value it computes from forward values alone (`transpose(B)`, `Y - T`) does not depend on
/// the loss. Value ids continue above the forward program's largest; `next` is `None` once
/// `u32::MAX` is handed out.
struct Sink {
    program: TensorProgram,
    subgraphs: BTreeMap<ValueId, Subgraph>,
    types: BTreeMap<ValueId, TensorType>,
    next: Option<u32>,
    seed: Option<ValueId>,
    tag: Subgraph,
}

impl Sink {
    fn new(forward: &TensorProgram, loss: ValueId) -> Self {
        let subgraphs = forward
            .nodes
            .iter()
            .map(|n| {
                let tag = if n.id == loss {
                    Subgraph::Loss
                } else {
                    Subgraph::Forward
                };
                (n.id, tag)
            })
            .collect();
        let types = forward.nodes.iter().map(|n| (n.id, n.ty.clone())).collect();
        let next = match forward.nodes.iter().map(|n| n.id.0).max() {
            Some(largest) => largest.checked_add(1),
            None => Some(0),
        };
        Self {
            program: forward.clone(),
            subgraphs,
            types,
            next,
            seed: None,
            tag: Subgraph::Gradient,
        }
    }

    fn push(&mut self, op: TensorOp, operands: &[ValueId]) -> Result<ValueId, SinkError> {
        let operand_types: Vec<&TensorType> = operands.iter().map(|o| &self.types[o]).collect();
        let ty = ops::infer(&op, &operand_types, None)?;
        let id = ValueId(self.next.ok_or(SinkError::IdsExhausted)?);
        self.next = id.0.checked_add(1);
        let on_seed = operands
            .iter()
            .any(|&o| Some(o) == self.seed || self.subgraphs[&o] == Subgraph::Gradient);
        let tag = if on_seed || self.tag == Subgraph::Loss {
            self.tag
        } else {
            Subgraph::Forward
        };
        self.types.insert(id, ty.clone());
        self.subgraphs.insert(id, tag);
        self.program.nodes.push(TensorNode {
            id,
            op,
            operands: operands.to_vec(),
            ty,
            source_span: None,
        });
        Ok(id)
    }

    /// The sum of `values` in order, each added once; the value itself when there is one.
    fn sum(&mut self, values: &[ValueId]) -> Result<ValueId, SinkError> {
        let (&first, rest) = values
            .split_first()
            .expect("an adjoint has at least one contribution");
        rest.iter()
            .try_fold(first, |acc, &v| self.push(TensorOp::Add, &[acc, v]))
    }
}

impl GradientSink for Sink {
    type Error = SinkError;

    fn emit(&mut self, op: TensorOp, operands: &[ValueId]) -> Result<ValueId, SinkError> {
        self.push(op, operands)
    }

    fn type_of(&self, value: ValueId) -> &TensorType {
        &self.types[&value]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::tensor::{Initializer, TensorProgramBuilder};

    /// `mse(embedding(table, ids), t)` with the table a parameter; returns it and its loss.
    fn lookup() -> (TensorProgram, ValueId) {
        let mut b = TensorProgramBuilder::new();
        let id = b
            .declare_param(
                "table",
                TensorType::f32(&[7, 3]),
                Initializer::XavierUniform,
            )
            .unwrap();
        let table = b.param_read(id).unwrap();
        let ids = b.input("ids", TensorType::i32(&[10])).unwrap();
        let rows = b.embedding(table, ids).unwrap();
        let t = b.input("t", TensorType::f32(&[10, 3])).unwrap();
        let loss = b.mse(rows, t).unwrap();
        b.output(loss).unwrap();
        (b.finish(), loss)
    }

    /// The table's gradient is one `EmbeddingGrad` of the ids and the rows' adjoint.
    #[test]
    fn the_table_gradient_is_an_embedding_grad_of_the_ids() {
        let (forward, loss) = lookup();
        let training = differentiate(&forward, loss).unwrap();
        let g = training.gradients[&ParamId(0)];
        let node = training.program.node(g).unwrap();
        assert_eq!(
            node.op,
            TensorOp::EmbeddingGrad {
                rows: Dim::Static(7)
            }
        );
        assert_eq!(node.operands[0], forward.inputs[0], "the ids");
        assert_eq!(node.ty, TensorType::f32(&[7, 3]));
    }

    /// An `i32` input has no adjoint: asking for one is refused.
    #[test]
    fn the_adjoint_of_an_i32_input_is_refused() {
        let (forward, loss) = lookup();
        assert_eq!(
            differentiate_inputs(&forward, loss, &["t", "ids"]),
            Err(InputAdjointError::IntegerInput { name: "ids".into() })
        );
        assert!(differentiate_inputs(&forward, loss, &["t"]).is_ok());
    }

    /// `mse(gelu(add_rows(matmul(x, w), b)), t)` with `w` and `b` parameters.
    fn dense() -> (TensorProgram, ValueId) {
        let mut b = TensorProgramBuilder::new();
        let w = b
            .declare_param("w", TensorType::f32(&[4, 3]), Initializer::XavierUniform)
            .unwrap();
        let bias = b
            .declare_param("b", TensorType::f32(&[3]), Initializer::Zeros)
            .unwrap();
        let x = b.input("x", TensorType::f32(&[5, 4])).unwrap();
        let w = b.param_read(w).unwrap();
        let bias = b.param_read(bias).unwrap();
        let xw = b.matmul(x, w).unwrap();
        let z = b.add_rows(xw, bias).unwrap();
        let y = b.gelu(z).unwrap();
        let t = b.input("t", TensorType::f32(&[5, 3])).unwrap();
        let loss = b.mse(y, t).unwrap();
        b.output(loss).unwrap();
        (b.finish(), loss)
    }

    /// The bias gradient is one `SumRows` of the adjoint of the `add_rows` result, which is a
    /// `GeluGrad` of the pre-activation; the adjoint of `x W` is that same `GeluGrad`.
    #[test]
    fn the_bias_gradient_is_a_sum_rows_of_a_gelu_grad() {
        let (forward, loss) = dense();
        let training = differentiate(&forward, loss).unwrap();
        let program = &training.program;
        let db = program.node(training.gradients[&ParamId(1)]).unwrap();
        assert_eq!(db.op, TensorOp::SumRows);
        assert_eq!(db.ty, TensorType::f32(&[3]));
        let dz = program.node(db.operands[0]).unwrap();
        assert_eq!(dz.op, TensorOp::GeluGrad);
        let z = forward
            .nodes
            .iter()
            .find(|n| n.op == TensorOp::AddRows)
            .unwrap();
        assert_eq!(dz.operands[0], z.id, "GeluGrad reads the pre-activation");
        let dw = program.node(training.gradients[&ParamId(0)]).unwrap();
        assert_eq!(dw.op, TensorOp::MatMul);
        assert_eq!(dw.operands[1], dz.id, "dW = transpose(x) @ dZ");
    }

    /// `mse(softmax(layer_norm(matmul(x, w), gamma, beta)), t)` with `w`, `gamma` and `beta`
    /// parameters.
    fn norm_softmax() -> (TensorProgram, ValueId) {
        let mut b = TensorProgramBuilder::new();
        let w = b
            .declare_param("w", TensorType::f32(&[3, 5]), Initializer::XavierUniform)
            .unwrap();
        let gamma = b
            .declare_param("gamma", TensorType::f32(&[5]), Initializer::Ones)
            .unwrap();
        let beta = b
            .declare_param("beta", TensorType::f32(&[5]), Initializer::Zeros)
            .unwrap();
        let x = b.input("x", TensorType::f32(&[4, 3])).unwrap();
        let w = b.param_read(w).unwrap();
        let gamma = b.param_read(gamma).unwrap();
        let beta = b.param_read(beta).unwrap();
        let xw = b.matmul(x, w).unwrap();
        let z = b.layer_norm(xw, gamma, beta).unwrap();
        let y = b.softmax(z).unwrap();
        let t = b.input("t", TensorType::f32(&[4, 5])).unwrap();
        let loss = b.mse(y, t).unwrap();
        b.output(loss).unwrap();
        (b.finish(), loss)
    }

    /// The adjoint of the `layer_norm` result is a `SoftmaxGrad` of the `softmax` result; the
    /// gradient of `gamma` is a `LayerNormGradGamma` and of `beta` a `SumRows` of that adjoint,
    /// and the adjoint of `x W` a `LayerNormGradX` reading `x W`, `gamma` and it.
    #[test]
    fn the_norm_gradients_are_their_internal_ops_over_a_softmax_grad() {
        let (forward, loss) = norm_softmax();
        let training = differentiate(&forward, loss).unwrap();
        let program = &training.program;
        let find = |op: TensorOp| forward.nodes.iter().find(|n| n.op == op).unwrap().id;
        let (xw, z, y) = (
            find(TensorOp::MatMul),
            find(TensorOp::LayerNorm),
            find(TensorOp::Softmax),
        );
        let dgamma = program.node(training.gradients[&ParamId(1)]).unwrap();
        assert_eq!(dgamma.op, TensorOp::LayerNormGradGamma);
        assert_eq!(dgamma.ty, TensorType::f32(&[5]));
        let dz = program.node(dgamma.operands[1]).unwrap();
        assert_eq!(dz.op, TensorOp::SoftmaxGrad);
        assert_eq!(dz.operands[0], y, "SoftmaxGrad reads the softmax result");
        assert_eq!(dgamma.operands[0], xw, "LayerNormGradGamma reads x W");
        let dbeta = program.node(training.gradients[&ParamId(2)]).unwrap();
        assert_eq!(dbeta.op, TensorOp::SumRows);
        assert_eq!(dbeta.operands, vec![dz.id]);
        let dw = program.node(training.gradients[&ParamId(0)]).unwrap();
        let dxw = program.node(dw.operands[1]).unwrap();
        assert_eq!(dxw.op, TensorOp::LayerNormGradX);
        let gamma_read =
            forward.nodes[forward.nodes.iter().position(|n| n.id == z).unwrap()].operands[1];
        assert_eq!(dxw.operands, vec![xw, gamma_read, dz.id]);
    }
}
