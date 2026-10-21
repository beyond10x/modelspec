//! Training IR: a forward program with its loss and generated gradients (DESIGN.md § 3.1).
//!
//! A [`TrainingProgram`] is a [`TensorProgram`] plus analysis metadata, not a second container:
//! its SSA graph is the field [`TrainingProgram::program`], whose outputs are the loss followed by
//! one gradient per entry of [`TrainingProgram::gradients`], in `ParamId` order. Any Tensor IR
//! consumer, the CPU interpreter included, runs it through `&training.program`;
//! [`crate::ir::verify::verify_training`] checks it.
//!
//! The subgraph tags are for analysis and are not execution boundaries (DESIGN.md § 3.1). The
//! optimizer subgraph, effect tokens and parameter versions come with SGD.

use std::collections::BTreeMap;
use std::fmt;

use crate::ir::tensor::{ParamId, TensorProgram, ValueId};

/// The Training IR subgraph a node belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Subgraph {
    /// The forward computation the loss is taken over, and any value that does not depend on the
    /// backward seed (a gradient rule's constant, a `transpose` of a forward value).
    Forward,
    /// The loss value and the backward seed `dL/dL = 1`, an operand-free `Scalar(1.0)` from which
    /// every gradient derives. No other node is tagged `Loss`.
    Loss,
    /// The backward computation: every value other than the seed that depends on the seed,
    /// adjoints and parameter gradients.
    Gradient,
}

impl fmt::Display for Subgraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Subgraph::Forward => "forward",
            Subgraph::Loss => "loss",
            Subgraph::Gradient => "gradient",
        })
    }
}

/// A training program: the SSA graph of the forward, loss and backward computations.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingProgram {
    /// The SSA graph; outputs are [`Self::loss`] then each of [`Self::gradients`] in key order. A
    /// parameter the loss does not read has no gradient and so no output.
    pub program: TensorProgram,
    /// The subgraph of every node, keyed by the value it defines.
    pub subgraphs: BTreeMap<ValueId, Subgraph>,
    /// The rank-0 loss value.
    pub loss: ValueId,
    /// The gradient of the loss with respect to each loss-reachable trainable parameter, and only
    /// those: exactly the parameters the loss reads through its operands have an entry. A
    /// declared parameter the loss does not read has no entry, and an optimizer step leaves its
    /// value unchanged.
    pub gradients: BTreeMap<ParamId, ValueId>,
}
