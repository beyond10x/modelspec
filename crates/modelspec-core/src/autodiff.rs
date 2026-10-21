//! Reverse-mode automatic differentiation over tensor IR (DESIGN.md § 3.3 pass 4, § 4).
//!
//! [`differentiate`] generates one backward graph on the tensor SSA of a forward program, not a
//! runtime tape. It walks the nodes in reverse definition order from a rank-0 loss, applies each
//! op's gradient rule ([`crate::ops::gradient`]) and sums every incoming adjoint of a value
//! exactly once per use. Only nodes the loss depends on and that depend on a parameter read take
//! part, so inputs and targets receive no gradient and a parameter the loss does not reach has
//! none. Every gradient is `f32`, the program's dtype.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::ir::tensor::{ParamId, TensorNode, TensorOp, TensorProgram, TensorType, ValueId};
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

    let trainable = trainable_nodes(nodes, &position);
    let mut sink = Sink::new(forward, loss);
    let mut adjoints: BTreeMap<ValueId, Vec<ValueId>> = BTreeMap::new();
    let mut reads: BTreeMap<ParamId, Vec<ValueId>> = BTreeMap::new();
    if trainable[loss_at] {
        sink.tag = Subgraph::Loss;
        let seed = sink
            .push(TensorOp::Scalar(1.0), &[])
            .map_err(rule_error(loss))?;
        sink.tag = Subgraph::Gradient;
        sink.seed = Some(seed);
        adjoints.insert(loss, vec![seed]);
    }

    for node in nodes[..=loss_at].iter().rev() {
        let Some(contributions) = adjoints.remove(&node.id) else {
            continue;
        };
        let g = sink.sum(&contributions).map_err(rule_error(node.id))?;
        if let TensorOp::ParamRead { param } = node.op {
            reads.entry(param).or_default().push(g);
            continue;
        }
        let wanted: Vec<bool> = node
            .operands
            .iter()
            .map(|o| trainable[position[o]])
            .collect();
        let per_operand = ops::gradient(&mut sink, &node.op, &node.operands, g, &wanted)
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

/// Per node position: the node depends on a parameter read.
///
/// Loss reachability needs no mask: adjoints flow only from the seed through operands, so a node
/// the loss does not depend on never receives one.
fn trainable_nodes(nodes: &[TensorNode], position: &BTreeMap<ValueId, usize>) -> Vec<bool> {
    let mut trainable = vec![false; nodes.len()];
    for (i, node) in nodes.iter().enumerate() {
        trainable[i] = matches!(node.op, TensorOp::ParamRead { .. })
            || node.operands.iter().any(|o| trainable[position[o]]);
    }
    trainable
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
