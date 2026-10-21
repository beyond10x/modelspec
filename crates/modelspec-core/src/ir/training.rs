//! Training IR: a forward program with its loss, generated gradients and optimizer step
//! (DESIGN.md § 3.1).
//!
//! A [`TrainingProgram`] is a [`TensorProgram`] plus analysis metadata, not a second container:
//! its SSA graph is the field [`TrainingProgram::program`], whose outputs are the loss followed by
//! one gradient per entry of [`TrainingProgram::gradients`], in `ParamId` order. Any Tensor IR
//! consumer, the CPU interpreter included, runs it through `&training.program`;
//! [`crate::ir::verify::verify_training`] checks it.
//!
//! A [`TrainingStep`] is a training program with the SGD [`Optimizer`] subgraph attached by
//! [`attach_sgd`]: one `p_next = p - lr * grad` output per loss-reachable parameter, after the
//! gradients. Its persistent-state effects are metadata, not `TensorOp`s: every parameter carries a
//! [`StateVersion`], and every `ParamRead` and every `p_next` carries an [`Effect`] with an ordered
//! [`EffectToken`]; an update yields version `n + 1` from `n` (DESIGN.md § 3.1, § 4).
//! [`crate::ir::verify::verify_step`] checks it, the legal-effects rules of DESIGN.md § 3.2
//! included.
//!
//! The subgraph tags are for analysis and are not execution boundaries (DESIGN.md § 3.1).

use std::collections::BTreeMap;
use std::fmt;

use thiserror::Error;

use crate::diag::SourceSpan;
use crate::hir::{HirData, HirTrain};
use crate::ir::tensor::{ParamId, TensorNode, TensorOp, TensorProgram, TensorType, ValueId};
use crate::ir::verify::{TrainingVerifyError, verify_step, verify_training};
use crate::ops::{self, OpError};

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
    /// The optimizer step: the learning rate and each parameter's next value. Only a
    /// [`TrainingStep`] has `Optimizer` nodes, and no node of another subgraph uses one.
    Optimizer,
}

impl fmt::Display for Subgraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Subgraph::Forward => "forward",
            Subgraph::Loss => "loss",
            Subgraph::Gradient => "gradient",
            Subgraph::Optimizer => "optimizer",
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

/// A version of a parameter's persistent state, printed `v<n>`. An update yields `n + 1` from `n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateVersion(pub u64);

impl fmt::Display for StateVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// The position of an effect in a step's effect order, printed `!<n>`: a smaller token happens
/// first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EffectToken(pub u32);

impl fmt::Display for EffectToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "!{}", self.0)
    }
}

/// Whether an effect reads or writes parameter state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Access {
    Read,
    Write,
}

impl fmt::Display for Access {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Access::Read => "read",
            Access::Write => "write",
        })
    }
}

/// One access to a parameter's persistent state, carried by the value that performs it: a
/// `ParamRead` node reads, a `p_next` output writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Effect {
    pub token: EffectToken,
    pub param: ParamId,
    pub access: Access,
    /// The version a read reads, or the version a write produces.
    pub version: StateVersion,
}

/// The SGD optimizer step of a [`TrainingStep`], the one M0 optimizer.
///
/// The versions are those of the step as compiled: every parameter is at version 0, its
/// initializer, when the step first runs. Each run of the step reads version `n` of a trained
/// parameter and writes `n + 1`; a runner tracks `n`, and these versions fix the relation.
#[derive(Clone, Debug, PartialEq)]
pub struct Optimizer {
    /// The SGD learning rate: positive and finite.
    pub lr: f32,
    /// The state version of every declared parameter when the step starts.
    pub versions: BTreeMap<ParamId, StateVersion>,
    /// The next value `p - lr * grad` of each parameter with a gradient, and only those: a
    /// parameter the loss does not read keeps its value and its version.
    pub updates: BTreeMap<ParamId, ValueId>,
    /// The effect of each value that accesses parameter state, keyed by that value: every
    /// `ParamRead` node and every update.
    pub effects: BTreeMap<ValueId, Effect>,
}

impl Optimizer {
    /// The effects in token order.
    pub fn effects_in_order(&self) -> Vec<(ValueId, Effect)> {
        let mut effects: Vec<(ValueId, Effect)> =
            self.effects.iter().map(|(&v, &e)| (v, e)).collect();
        effects.sort_by_key(|(_, e)| e.token);
        effects
    }
}

/// A whole training step: forward, loss, backward and SGD in one program (DESIGN.md § 0
/// invariant 5).
///
/// The outputs of `training.program` are the loss, then each gradient, then each of
/// [`Optimizer::updates`], both in `ParamId` order. [`TrainingProgram::gradients`] and
/// [`TrainingProgram::loss`] keep their meaning.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingStep {
    pub training: TrainingProgram,
    pub optimizer: Optimizer,
}

/// Why [`attach_sgd`] refused.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum AttachError {
    /// The learning rate is not positive and finite; its bits are kept so the error compares.
    #[error("learning rate {} is not positive and finite", f32::from_bits(*lr_bits))]
    LearningRate { lr_bits: u32 },
    /// The optimizer step needs a value id past `u32::MAX`.
    #[error("no value id is left for the optimizer step: ids run past u32::MAX")]
    ValueIdsExhausted,
    /// An optimizer node the registry refuses: a compiler bug.
    #[error("optimizer step for parameter {param}: {source}")]
    Rule { param: ParamId, source: OpError },
    /// The input or the result breaks a Training IR invariant: a compiler bug.
    #[error("training step does not verify: {0}")]
    Training(#[from] TrainingVerifyError),
}

/// Attaches the SGD step `p_next = p - lr * grad` to `training`, one update per parameter with a
/// gradient, in `ParamId` order.
///
/// `training` must pass [`crate::ir::verify::verify_training`]. The result appends, tagged
/// [`Subgraph::Optimizer`], one `Scalar(lr)` and for each parameter `Mul lr, grad` and
/// `Sub p, (lr * grad)`, where `p` is the first `ParamRead` of the parameter in node order; its
/// outputs are those of `training` followed by the updates. Every declared parameter is at
/// [`StateVersion`] 0. Effect tokens count from 0: first a read for each `ParamRead` node in node
/// order, each of version 0, then a write for each update in `ParamId` order, each of version 1.
/// The result passes [`crate::ir::verify::verify_step`].
pub fn attach_sgd(training: TrainingProgram, lr: f32) -> Result<TrainingStep, AttachError> {
    verify_training(&training)?;
    if !(lr.is_finite() && lr > 0.0) {
        return Err(AttachError::LearningRate {
            lr_bits: lr.to_bits(),
        });
    }
    let mut training = training;
    let mut next = match training.program.nodes.iter().map(|n| n.id.0).max() {
        Some(largest) => largest.checked_add(1),
        None => Some(0),
    };
    let mut push = |training: &mut TrainingProgram,
                    param: ParamId,
                    op: TensorOp,
                    operands: Vec<ValueId>|
     -> Result<ValueId, AttachError> {
        let program = &training.program;
        let types: Vec<&TensorType> = operands
            .iter()
            .map(|&o| &program.node(o).expect("a verified operand is defined").ty)
            .collect();
        let ty =
            ops::infer(&op, &types, None).map_err(|source| AttachError::Rule { param, source })?;
        let id = ValueId(next.ok_or(AttachError::ValueIdsExhausted)?);
        next = id.0.checked_add(1);
        training.program.nodes.push(TensorNode {
            id,
            op,
            operands,
            ty,
            source_span: None,
        });
        training.subgraphs.insert(id, Subgraph::Optimizer);
        Ok(id)
    };

    let mut updates = BTreeMap::new();
    let gradients: Vec<(ParamId, ValueId)> =
        training.gradients.iter().map(|(&p, &g)| (p, g)).collect();
    if let Some(&(first, _)) = gradients.first() {
        let rate = push(&mut training, first, TensorOp::Scalar(lr), Vec::new())?;
        for (param, grad) in gradients {
            let read = training
                .program
                .nodes
                .iter()
                .find(|n| n.op == TensorOp::ParamRead { param })
                .expect("a parameter with a gradient is read")
                .id;
            let scaled = push(&mut training, param, TensorOp::Mul, vec![rate, grad])?;
            let update = push(&mut training, param, TensorOp::Sub, vec![read, scaled])?;
            updates.insert(param, update);
        }
    }
    training.program.outputs.extend(updates.values().copied());

    let versions: BTreeMap<ParamId, StateVersion> = training
        .program
        .params
        .keys()
        .map(|&p| (p, StateVersion(0)))
        .collect();
    let mut effects = BTreeMap::new();
    let mut token = 0u32;
    let mut record = |value: ValueId, param: ParamId, access: Access, version: u64| {
        effects.insert(
            value,
            Effect {
                token: EffectToken(token),
                param,
                access,
                version: StateVersion(version),
            },
        );
        token += 1;
    };
    for node in &training.program.nodes {
        if let TensorOp::ParamRead { param } = node.op {
            record(node.id, param, Access::Read, 0);
        }
    }
    for (&param, &update) in &updates {
        record(update, param, Access::Write, 1);
    }

    let step = TrainingStep {
        training,
        optimizer: Optimizer {
            lr,
            versions,
            updates,
            effects,
        },
    };
    verify_step(&step)?;
    Ok(step)
}

/// One `input` or `target` declaration of a `train` block.
#[derive(Clone, Debug, PartialEq)]
pub struct DataDecl {
    pub name: String,
    pub ty: TensorType,
    pub span: SourceSpan,
}

/// The SGD configuration of a `train` block, which the runtime reads (DESIGN.md § 6).
///
/// It records the declarations as written; whether there is exactly one `input` and one `target`
/// is the data loader's to refuse.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainConfig {
    /// The number of steps of one run: positive.
    pub steps: u64,
    /// The SGD learning rate: positive and finite.
    pub lr: f32,
    /// The global seed of every random draw (DESIGN.md § 2.3).
    pub seed: u64,
    /// The `input` declarations, in source order.
    pub inputs: Vec<DataDecl>,
    /// The `target` declarations, in source order.
    pub targets: Vec<DataDecl>,
    /// The `train` block.
    pub span: SourceSpan,
}

impl TrainConfig {
    /// The configuration of a checked `train` block.
    pub fn from_hir(train: &HirTrain) -> Self {
        let decls = |data: &mut dyn Iterator<Item = &HirData>| {
            data.map(|d| DataDecl {
                name: d.name.clone(),
                ty: d.ty.clone(),
                span: d.span.clone(),
            })
            .collect()
        };
        Self {
            steps: train.steps,
            lr: train.lr,
            seed: train.seed,
            inputs: decls(&mut train.inputs()),
            targets: decls(&mut train.targets()),
            span: train.span.clone(),
        }
    }
}
