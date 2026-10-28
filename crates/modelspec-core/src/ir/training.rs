//! Training IR: a forward program with its loss, generated gradients and optimizer step
//! (DESIGN.md § 3.1).
//!
//! A [`TrainingProgram`] is a [`TensorProgram`] plus analysis metadata, not a second container:
//! its SSA graph is the field [`TrainingProgram::program`], whose outputs are the loss followed by
//! one gradient per entry of [`TrainingProgram::gradients`], in `ParamId` order. Any Tensor IR
//! consumer, the CPU interpreter included, runs it through `&training.program`;
//! [`crate::ir::verify::verify_training`] checks it.
//!
//! A [`TrainingStep`] is a training program with an [`Optimizer`] subgraph attached: by
//! [`attach_sgd`], one `p_next = p - lr * grad` output per loss-reachable parameter, after the
//! gradients; or by [`attach_adamw`], the AdamW update of each such parameter, which also reads
//! and writes its first and second moments `m` and `v` ([`Moments`]) and reads three rank-0
//! inputs the training host supplies at every step ([`StepInputs`]); or by [`attach_muon`], the
//! Muon update of each matrix parameter no `embedding` reads ([`muon_params`]), which reads and
//! writes its momentum buffer `b` ([`Buffer`]) and orthogonalizes it by Newton–Schulz, and the
//! AdamW update of every other one. Its persistent-state effects
//! are metadata, not `TensorOp`s: every parameter carries a [`StateVersion`], and every
//! `ParamRead` and every `p_next` carries an [`Effect`] with an ordered [`EffectToken`]; an update
//! yields version `n + 1` from `n` (DESIGN.md § 3.1, § 4). A moment's read and write carry a
//! [`MomentEffect`] at its parameter's versions, in the same token order.
//! [`crate::ir::verify::verify_step`] checks it, the legal-effects rules of DESIGN.md § 3.2
//! included.
//!
//! The subgraph tags are for analysis and are not execution boundaries (DESIGN.md § 3.1).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use thiserror::Error;

use crate::diag::SourceSpan;
use crate::hir::{HirData, HirTrain};
use crate::ir::tensor::{Dim, ParamId, TensorNode, TensorOp, TensorProgram, TensorType, ValueId};
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
    /// The optimizer step: its constants, the host inputs of an AdamW step, each moment's next
    /// value and each parameter's next value. Only a [`TrainingStep`] has `Optimizer` nodes, and
    /// no node of another subgraph uses one.
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

/// The update rule of an [`Optimizer`], the `optimizer` setting of a `train` block (DESIGN.md
/// § 2.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptimizerRule {
    /// `sgd(lr = F)`: `p - lr * grad` at a constant rate.
    Sgd,
    /// `adamw(...)`: AdamW with decoupled weight decay and the linear warmup and decay schedule.
    AdamW(AdamW),
    /// `muon(...)`: Muon, orthogonalized momentum, for each matrix parameter no `embedding` reads
    /// ([`muon_params`]), and AdamW for every other parameter, both on the schedule of `adamw`.
    Muon(Muon),
}

/// The hyperparameters of `adamw(...)` besides its peak learning rate `lr`, as the checker accepts
/// them (DESIGN.md § 2.1, coordinator decision D50).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamW {
    /// The decoupled weight decay: finite and at least 0.
    pub weight_decay: f32,
    /// The first-moment decay rate, in `[0, 1)`.
    pub beta1: f32,
    /// The second-moment decay rate, in `[0, 1)`.
    pub beta2: f32,
    /// The denominator term: positive and finite.
    pub eps: f32,
    /// The steps of the linear warmup from 0, `0` for none; fewer than the run's steps.
    pub warmup_steps: u64,
}

/// The rank-0 values the training host supplies to step `t` of an AdamW run
/// ([`AdamW::scalars`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepScalars {
    /// The learning rate of the step, by the schedule.
    pub lr_t: f32,
    /// The first-moment bias correction `1 / (1 - beta1^t)`.
    pub c1: f32,
    /// The second-moment bias correction `1 / (1 - beta2^t)`.
    pub c2: f32,
}

impl AdamW {
    /// Whether every hyperparameter is in range: `weight_decay` finite and at least 0, each beta
    /// in `[0, 1)`, `eps` positive and finite. Whether `warmup_steps` is below the run's steps is
    /// the checker's and the runner's to say.
    pub fn in_range(&self) -> bool {
        let beta = |b: f32| (0.0..1.0).contains(&b);
        self.weight_decay.is_finite()
            && self.weight_decay >= 0.0
            && beta(self.beta1)
            && beta(self.beta2)
            && self.eps.is_finite()
            && self.eps > 0.0
    }

    /// The constants of the update, in the order [`attach_adamw`] adds their `Scalar` nodes:
    /// `beta1`, `1 - beta1`, `beta2`, `1 - beta2`, `eps`, `weight_decay`. Each `1 - beta` is
    /// computed in `f64` from the `f32` beta and rounded once to `f32`.
    pub fn constants(&self) -> [f32; 6] {
        let complement = |beta: f32| (1.0 - f64::from(beta)) as f32;
        [
            self.beta1,
            complement(self.beta1),
            self.beta2,
            complement(self.beta2),
            self.eps,
            self.weight_decay,
        ]
    }

    /// The host inputs of step `t`, counted from 1, of a run of `steps` steps whose peak rate is
    /// `lr`, each computed in `f64` and rounded once to `f32` (DESIGN.md § 8, coordinator
    /// decision D50, settlement 1): `lr_t = lr * t / warmup_steps` for `t <= warmup_steps`, and
    /// `lr * (steps - t) / (steps - warmup_steps)` after, a linear warmup from 0 and a linear
    /// decay to 0 at the last step (Devlin et al. 2019); `c1 = 1 / (1 - beta1^t)` and
    /// `c2 = 1 / (1 - beta2^t)`. `lr_t` is 0 past the last step and when `warmup_steps` is not
    /// below `steps`, a run the checker and the runner refuse.
    pub fn scalars(&self, lr: f32, t: u64, steps: u64) -> StepScalars {
        let peak = f64::from(lr);
        let warmup = self.warmup_steps;
        let rate = if t <= warmup {
            peak * t as f64 / warmup as f64
        } else {
            match steps.saturating_sub(warmup) {
                0 => 0.0,
                span => peak * steps.saturating_sub(t) as f64 / span as f64,
            }
        };
        let power = i32::try_from(t).unwrap_or(i32::MAX);
        let correction = |beta: f32| (1.0 / (1.0 - f64::from(beta).powi(power))) as f32;
        StepScalars {
            lr_t: rate as f32,
            c1: correction(self.beta1),
            c2: correction(self.beta2),
        }
    }
}

/// The largest `ns_steps` of `muon(...)` the checker accepts.
pub const MUON_MAX_NS_STEPS: u32 = 10;

/// The term added to the Frobenius norm of a Muon update before it divides the update.
pub const MUON_NORM_EPS: f32 = 1e-7;

/// The coefficients `a`, `b` and `c` of one Newton–Schulz step `x = a x + (b A + c A A) x`,
/// `A = x xᵀ` (Jordan et al. 2024).
pub const NEWTON_SCHULZ: [f32; 3] = [3.4445, -4.7750, 2.0315];

/// The hyperparameters of `muon(...)` besides its peak learning rate `lr`, as the checker accepts
/// them (DESIGN.md § 2.1, coordinator decision D67).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Muon {
    /// The momentum of the buffer `b`, in `[0, 1)`.
    pub momentum: f32,
    /// The Newton–Schulz steps of each update, in `[1, MUON_MAX_NS_STEPS]`.
    pub ns_steps: u32,
    /// The peak learning rate of the parameters AdamW updates: positive and finite.
    pub adam_lr: f32,
    /// The AdamW hyperparameters of those parameters. Its `weight_decay` is also the decay of the
    /// Muon parameters and its `warmup_steps` the warmup of both rates.
    pub adamw: AdamW,
}

/// The rank-0 values the training host supplies to step `t` of a Muon run ([`Muon::scalars`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MuonScalars {
    /// The learning rate of the Muon parameters at the step, by the schedule of `lr`.
    pub lr_t: f32,
    /// The host inputs of the AdamW parameters, the rate by the schedule of `adam_lr`.
    pub adamw: StepScalars,
}

impl Muon {
    /// Whether every hyperparameter is in range: `momentum` in `[0, 1)`, `ns_steps` in
    /// `[1, MUON_MAX_NS_STEPS]`, `adam_lr` positive and finite and the AdamW ones in range.
    pub fn in_range(&self) -> bool {
        (0.0..1.0).contains(&self.momentum)
            && (1..=MUON_MAX_NS_STEPS).contains(&self.ns_steps)
            && self.adam_lr.is_finite()
            && self.adam_lr > 0.0
            && self.adamw.in_range()
    }

    /// The constants of a Muon update, in the order [`attach_muon`] adds their `Scalar` nodes:
    /// `momentum`, [`MUON_NORM_EPS`], the three [`NEWTON_SCHULZ`] coefficients and
    /// `weight_decay`. Each parameter's scale ([`Muon::scale`]) is a constant too.
    pub fn constants(&self) -> [f32; 6] {
        let [a, b, c] = NEWTON_SCHULZ;
        [
            self.momentum,
            MUON_NORM_EPS,
            a,
            b,
            c,
            self.adamw.weight_decay,
        ]
    }

    /// The scale `0.2 * sqrt(max(rows, cols))` of the update of a `[rows, cols]` parameter, which
    /// gives it the RMS of an AdamW update (Liu et al. 2025), computed in `f64` and rounded once
    /// to `f32`.
    pub fn scale(rows: u64, cols: u64) -> f32 {
        (0.2 * (rows.max(cols) as f64).sqrt()) as f32
    }

    /// The host inputs of step `t`, counted from 1, of a run of `steps` steps whose Muon peak rate
    /// is `lr`: the schedule of [`AdamW::scalars`] applied to `lr` for the Muon parameters and to
    /// `adam_lr` for the AdamW ones.
    pub fn scalars(&self, lr: f32, t: u64, steps: u64) -> MuonScalars {
        MuonScalars {
            lr_t: self.adamw.scalars(lr, t, steps).lr_t,
            adamw: self.adamw.scalars(self.adam_lr, t, steps),
        }
    }
}

/// The name of the host input of an AdamW step's learning rate, and under Muon of the rate of
/// the parameters AdamW updates.
pub const LR_T_INPUT: &str = "optimizer.lr_t";
/// The name of the host input of a Muon step's learning rate for the parameters Muon updates.
pub const MUON_LR_T_INPUT: &str = "optimizer.muon_lr_t";
/// The name of the host input of an AdamW step's first-moment bias correction.
pub const C1_INPUT: &str = "optimizer.c1";
/// The name of the host input of an AdamW step's second-moment bias correction.
pub const C2_INPUT: &str = "optimizer.c2";

/// A per-parameter optimizer state: the first or the second moment of a parameter under AdamW,
/// printed `m` and `v`, or the momentum buffer of a parameter under Muon, printed `b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Moment {
    First,
    Second,
    /// The momentum buffer `b` of a Muon parameter.
    Momentum,
}

impl Moment {
    /// Both AdamW moments, in the order a step reads and writes them.
    pub const BOTH: [Moment; 2] = [Moment::First, Moment::Second];
}

impl fmt::Display for Moment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Moment::First => "m",
            Moment::Second => "v",
            Moment::Momentum => "b",
        })
    }
}

/// The name of the `Input` a step reads `moment` of the parameter at canonical `path` from:
/// `optimizer.m@<path>`, `optimizer.v@<path>` or `optimizer.b@<path>`.
pub fn moment_input(moment: Moment, path: &str) -> String {
    format!("optimizer.{moment}@{path}")
}

/// The three rank-0 `f32` inputs of an AdamW step, named [`LR_T_INPUT`], [`C1_INPUT`] and
/// [`C2_INPUT`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepInputs {
    pub lr_t: ValueId,
    pub c1: ValueId,
    pub c2: ValueId,
}

/// The moments of one parameter in an AdamW step: the `Input`s the step reads them from, of the
/// parameter's type, and their next values, outputs of the step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Moments {
    pub m: ValueId,
    pub v: ValueId,
    pub m_next: ValueId,
    pub v_next: ValueId,
}

impl Moments {
    /// The `Input` that reads `moment`, [`Moment::First`] or [`Moment::Second`].
    pub fn input(&self, moment: Moment) -> ValueId {
        match moment {
            Moment::First => self.m,
            Moment::Second => self.v,
            Moment::Momentum => unreachable!("AdamW moments hold no momentum buffer"),
        }
    }

    /// The next value of `moment`.
    pub fn next(&self, moment: Moment) -> ValueId {
        match moment {
            Moment::First => self.m_next,
            Moment::Second => self.v_next,
            Moment::Momentum => unreachable!("AdamW moments hold no momentum buffer"),
        }
    }
}

/// The momentum buffer of one parameter in a Muon step: the `Input` the step reads it from, of the
/// parameter's type, its next value, an output of the step, and the Frobenius norm
/// `sqrt(sum(u * u))` its step divides the Nesterov momentum by, which a runtime keeps to stop a
/// step whose norm is not finite (coordinator decision D69).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buffer {
    pub b: ValueId,
    pub b_next: ValueId,
    pub norm: ValueId,
}

/// One per-parameter state of a step ([`Optimizer::state_slots`]): its parameter, which state it
/// is, the `Input` the step reads it from and its next value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateSlot {
    pub param: ParamId,
    pub moment: Moment,
    pub input: ValueId,
    pub next: ValueId,
}

/// One access to a moment of a parameter, carried by the value that performs it: the moment's
/// `Input` reads, its next value writes. A moment is at its parameter's version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MomentEffect {
    pub token: EffectToken,
    pub param: ParamId,
    pub moment: Moment,
    pub access: Access,
    /// The version a read reads, or the version a write produces.
    pub version: StateVersion,
}

/// The optimizer step of a [`TrainingStep`]: SGD, or AdamW with its moments.
///
/// The versions are those of the step as compiled: every parameter is at version 0, its
/// initializer, when the step first runs. Each run of the step reads version `n` of a trained
/// parameter and writes `n + 1`; a runner tracks `n`, and these versions fix the relation.
#[derive(Clone, Debug, PartialEq)]
pub struct Optimizer {
    /// The learning rate of SGD, or the peak rate of AdamW's schedule: positive and finite.
    pub lr: f32,
    /// The rule of the update.
    pub rule: OptimizerRule,
    /// The state version of every declared parameter when the step starts.
    pub versions: BTreeMap<ParamId, StateVersion>,
    /// The next value of each parameter with a gradient, and only those: a parameter the loss
    /// does not read keeps its value and its version.
    pub updates: BTreeMap<ParamId, ValueId>,
    /// The effect of each value that accesses parameter state, keyed by that value: every
    /// `ParamRead` node and every update.
    pub effects: BTreeMap<ValueId, Effect>,
    /// The host inputs of an AdamW step with at least one update, or of a Muon step with at least
    /// one parameter AdamW updates; `None` under SGD.
    pub step_inputs: Option<StepInputs>,
    /// The moments of each parameter AdamW updates, under AdamW or Muon; empty under SGD.
    pub moments: BTreeMap<ParamId, Moments>,
    /// The host input [`MUON_LR_T_INPUT`] of a Muon step with at least one parameter Muon
    /// updates; `None` otherwise.
    pub muon_lr_t: Option<ValueId>,
    /// The momentum buffer of each parameter Muon updates; empty unless the rule is Muon.
    pub buffers: BTreeMap<ParamId, Buffer>,
    /// The effect of each value that accesses a moment or a buffer, keyed by that value: every
    /// state's `Input` and every next state. Its tokens and those of [`Self::effects`] are one
    /// order.
    pub moment_effects: BTreeMap<ValueId, MomentEffect>,
}

impl Optimizer {
    /// Every per-parameter state in `ParamId` order, a parameter's AdamW moments `m` then `v`,
    /// or its Muon buffer `b`: the order of the step's state reads, writes and outputs.
    pub fn state_slots(&self) -> Vec<StateSlot> {
        let mut slots = Vec::new();
        let params: BTreeSet<ParamId> = self
            .moments
            .keys()
            .chain(self.buffers.keys())
            .copied()
            .collect();
        for param in params {
            if let Some(m) = self.moments.get(&param) {
                for moment in Moment::BOTH {
                    slots.push(StateSlot {
                        param,
                        moment,
                        input: m.input(moment),
                        next: m.next(moment),
                    });
                }
            }
            if let Some(b) = self.buffers.get(&param) {
                slots.push(StateSlot {
                    param,
                    moment: Moment::Momentum,
                    input: b.b,
                    next: b.b_next,
                });
            }
        }
        slots
    }

    /// The effects in token order.
    pub fn effects_in_order(&self) -> Vec<(ValueId, Effect)> {
        let mut effects: Vec<(ValueId, Effect)> =
            self.effects.iter().map(|(&v, &e)| (v, e)).collect();
        effects.sort_by_key(|(_, e)| e.token);
        effects
    }

    /// The `Input`s of the step the training host supplies beside the data and the dropout
    /// tokens: the three step inputs, the Muon rate and each state's input ([`Self::state_slots`]),
    /// none under SGD.
    pub fn host_inputs(&self) -> Vec<ValueId> {
        let mut inputs: Vec<ValueId> = self
            .step_inputs
            .iter()
            .flat_map(|s| [s.lr_t, s.c1, s.c2])
            .collect();
        inputs.extend(self.muon_lr_t);
        inputs.extend(self.state_slots().iter().map(|s| s.input));
        inputs
    }
}

/// A whole training step: forward, loss, backward and the optimizer in one program (DESIGN.md
/// § 0 invariant 5).
///
/// The outputs of `training.program` are the loss, then each gradient, then each of
/// [`Optimizer::updates`], then, under AdamW or Muon, the next value of each state of
/// [`Optimizer::state_slots`], all in `ParamId` order. [`TrainingProgram::gradients`] and
/// [`TrainingProgram::loss`] keep their meaning.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingStep {
    pub training: TrainingProgram,
    pub optimizer: Optimizer,
}

/// Why [`attach_sgd`] or [`attach_adamw`] refused.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum AttachError {
    /// The learning rate is not positive and finite; its bits are kept so the error compares.
    #[error("learning rate {} is not positive and finite", f32::from_bits(*lr_bits))]
    LearningRate { lr_bits: u32 },
    /// An AdamW hyperparameter is out of range ([`AdamW::in_range`]).
    #[error("the adamw hyperparameters are out of range")]
    Hyperparameters,
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

/// The dropout sites of `program`: the name of each `Input` a `Dropout` or `DropoutGrad` reads
/// as its RNG token ([`ops::token_operand`]), once each, in input order. The training host
/// supplies each site's token at every step (DESIGN.md § 2.3, coordinator decision D47); a
/// program without dropout has none.
pub fn dropout_sites(program: &TensorProgram) -> Vec<String> {
    let mut tokens = std::collections::BTreeSet::new();
    for node in &program.nodes {
        if let Some(&token) = ops::token_operand(&node.op).and_then(|i| node.operands.get(i)) {
            tokens.insert(token);
        }
    }
    program
        .inputs
        .iter()
        .filter(|value| tokens.contains(value))
        .filter_map(|&value| match &program.node(value)?.op {
            TensorOp::Input { name } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

/// The projection inputs of the `sigreg` sites of `program`: each `Input` that a `MatMul` reads
/// as its right operand where a `SigregColumns` reads that `MatMul`, once each, in input order
/// (DESIGN.md § 2.3, story:jepa-sigreg-cpu). A program without `sigreg` has none.
pub fn sigreg_projection_inputs(program: &TensorProgram) -> Vec<ValueId> {
    let projected: std::collections::BTreeSet<ValueId> = program
        .nodes
        .iter()
        .filter(|n| n.op == TensorOp::SigregColumns)
        .filter_map(|n| n.operands.first().copied())
        .collect();
    let projections: std::collections::BTreeSet<ValueId> = program
        .nodes
        .iter()
        .filter(|n| n.op == TensorOp::MatMul && projected.contains(&n.id))
        .filter_map(|n| n.operands.get(1).copied())
        .collect();
    program
        .inputs
        .iter()
        .copied()
        .filter(|value| projections.contains(value))
        .filter(|&value| {
            matches!(
                program.node(value).map(|n| &n.op),
                Some(TensorOp::Input { .. })
            )
        })
        .collect()
}

/// The `sigreg` sites of `program`: the name of each of its [`sigreg_projection_inputs`], in
/// input order. The training host supplies each site's projection at every step.
pub fn sigreg_sites(program: &TensorProgram) -> Vec<String> {
    sigreg_projection_inputs(program)
        .into_iter()
        .filter_map(|value| match &program.node(value)?.op {
            TensorOp::Input { name } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

/// Appends optimizer nodes to a training program, numbering them after its largest value id.
struct Appender {
    next: Option<u32>,
}

impl Appender {
    fn new(training: &TrainingProgram) -> Self {
        let next = match training.program.nodes.iter().map(|n| n.id.0).max() {
            Some(largest) => largest.checked_add(1),
            None => Some(0),
        };
        Self { next }
    }

    /// Appends `op` over `operands`, tagged [`Subgraph::Optimizer`]; an `Input` takes the type
    /// `declared` and is appended to the program's inputs.
    fn push(
        &mut self,
        training: &mut TrainingProgram,
        param: ParamId,
        op: TensorOp,
        operands: Vec<ValueId>,
        declared: Option<&TensorType>,
    ) -> Result<ValueId, AttachError> {
        let program = &training.program;
        let types: Vec<&TensorType> = operands
            .iter()
            .map(|&o| &program.node(o).expect("a verified operand is defined").ty)
            .collect();
        let ty = ops::infer(&op, &types, declared)
            .map_err(|source| AttachError::Rule { param, source })?;
        let id = ValueId(self.next.ok_or(AttachError::ValueIdsExhausted)?);
        self.next = id.0.checked_add(1);
        let input = matches!(op, TensorOp::Input { .. });
        training.program.nodes.push(TensorNode {
            id,
            op,
            operands,
            ty,
            source_span: None,
        });
        if input {
            training.program.inputs.push(id);
        }
        training.subgraphs.insert(id, Subgraph::Optimizer);
        Ok(id)
    }

    /// Appends the `Input` `name` of type `ty`.
    fn input(
        &mut self,
        training: &mut TrainingProgram,
        param: ParamId,
        name: String,
        ty: &TensorType,
    ) -> Result<ValueId, AttachError> {
        let op = TensorOp::Input { name };
        self.push(training, param, op, Vec::new(), Some(ty))
    }

    /// Appends `op` over `operands`, neither of them an `Input`.
    fn op(
        &mut self,
        training: &mut TrainingProgram,
        param: ParamId,
        op: TensorOp,
        operands: &[ValueId],
    ) -> Result<ValueId, AttachError> {
        self.push(training, param, op, operands.to_vec(), None)
    }
}

/// The first `ParamRead` of `param` in node order.
fn first_read(training: &TrainingProgram, param: ParamId) -> ValueId {
    training
        .program
        .nodes
        .iter()
        .find(|n| n.op == TensorOp::ParamRead { param })
        .expect("a parameter with a gradient is read")
        .id
}

/// The checks every attachment starts with: `training` verifies and `lr` is positive and finite.
fn check_attach(training: &TrainingProgram, lr: f32) -> Result<(), AttachError> {
    verify_training(training)?;
    if !(lr.is_finite() && lr > 0.0) {
        return Err(AttachError::LearningRate {
            lr_bits: lr.to_bits(),
        });
    }
    Ok(())
}

/// Every declared parameter of `training` at [`StateVersion`] 0.
fn initial_versions(training: &TrainingProgram) -> BTreeMap<ParamId, StateVersion> {
    training
        .program
        .params
        .keys()
        .map(|&p| (p, StateVersion(0)))
        .collect()
}

/// [`attach_sgd`], [`attach_adamw`] or [`attach_muon`], as `rule` names.
pub fn attach(
    training: TrainingProgram,
    lr: f32,
    rule: OptimizerRule,
) -> Result<TrainingStep, AttachError> {
    match rule {
        OptimizerRule::Sgd => attach_sgd(training, lr),
        OptimizerRule::AdamW(adamw) => attach_adamw(training, lr, adamw),
        OptimizerRule::Muon(muon) => attach_muon(training, lr, muon),
    }
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
    check_attach(&training, lr)?;
    let mut training = training;
    let mut append = Appender::new(&training);

    let mut updates = BTreeMap::new();
    let gradients: Vec<(ParamId, ValueId)> =
        training.gradients.iter().map(|(&p, &g)| (p, g)).collect();
    if let Some(&(first, _)) = gradients.first() {
        let rate = append.op(&mut training, first, TensorOp::Scalar(lr), &[])?;
        for (param, grad) in gradients {
            let read = first_read(&training, param);
            let scaled = append.op(&mut training, param, TensorOp::Mul, &[rate, grad])?;
            let update = append.op(&mut training, param, TensorOp::Sub, &[read, scaled])?;
            updates.insert(param, update);
        }
    }
    training.program.outputs.extend(updates.values().copied());

    let versions = initial_versions(&training);
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
            rule: OptimizerRule::Sgd,
            versions,
            updates,
            effects,
            step_inputs: None,
            moments: BTreeMap::new(),
            muon_lr_t: None,
            buffers: BTreeMap::new(),
            moment_effects: BTreeMap::new(),
        },
    };
    verify_step(&step)?;
    Ok(step)
}

/// Attaches the AdamW step of `adamw` at peak rate `lr` to `training`: one update per parameter
/// with a gradient, in `ParamId` order, with decoupled weight decay (Loshchilov and Hutter) and
/// the moments `m` and `v` as versioned state (DESIGN.md § 2.1, § 6, coordinator decision D50).
///
/// `training` must pass [`crate::ir::verify::verify_training`], `lr` must be positive and finite
/// and `adamw` in range ([`AdamW::in_range`]). With at least one gradient the result appends,
/// tagged [`Subgraph::Optimizer`], a `Scalar` for each of [`AdamW::constants`] in its order, the
/// rank-0 `f32` inputs `lr_t`, `c1` and `c2` ([`StepInputs`]), and then for each parameter `p`
/// with gradient `g`, in `ParamId` order, the inputs `m` and `v` of its type
/// ([`moment_input`]) and, each one rounded `f32` operation in this order:
///
/// ```text
/// m' = beta1 * m + (1 - beta1) * g
/// v' = beta2 * v + ((1 - beta2) * g) * g
/// p' = p - lr_t * ((m' * c1) / (sqrt(v' * c2) + eps) + weight_decay * p)
/// ```
///
/// where `p` is the first `ParamRead` of the parameter in node order. Its outputs are those of
/// `training`, then the updates, then `m'` and `v'` of each parameter. Every declared parameter
/// is at [`StateVersion`] 0. Effect tokens count from 0: a read for each `ParamRead` node in node
/// order, a read of `m` then of `v` for each parameter, a write for each update, then a write of
/// `m'` then of `v'` for each parameter, each in `ParamId` order; reads are of version 0, writes
/// of version 1. The result passes [`crate::ir::verify::verify_step`].
pub fn attach_adamw(
    training: TrainingProgram,
    lr: f32,
    adamw: AdamW,
) -> Result<TrainingStep, AttachError> {
    check_attach(&training, lr)?;
    if !adamw.in_range() {
        return Err(AttachError::Hyperparameters);
    }
    let mut training = training;
    let mut append = Appender::new(&training);

    let mut updates = BTreeMap::new();
    let mut moments = BTreeMap::new();
    let mut step_inputs = None;
    let gradients: Vec<(ParamId, ValueId)> =
        training.gradients.iter().map(|(&p, &g)| (p, g)).collect();
    if let Some(&(first, _)) = gradients.first() {
        let t = &mut training;
        let (constants, inputs) = adamw_prologue(&mut append, t, first, &adamw)?;
        step_inputs = Some(inputs);
        for (param, g) in gradients {
            let (update, state) = adamw_update(&mut append, t, param, g, constants, inputs)?;
            updates.insert(param, update);
            moments.insert(param, state);
        }
    }
    training.program.outputs.extend(updates.values().copied());
    training
        .program
        .outputs
        .extend(moments.values().flat_map(|m| [m.m_next, m.v_next]));

    let versions = initial_versions(&training);
    let mut effects = BTreeMap::new();
    let mut moment_effects = BTreeMap::new();
    let mut token = 0u32;
    let mut next_token = || {
        let current = EffectToken(token);
        token += 1;
        current
    };
    let effect = |token, param, access, version| Effect {
        token,
        param,
        access,
        version: StateVersion(version),
    };
    let moment_effect = |token, param, moment, access, version| MomentEffect {
        token,
        param,
        moment,
        access,
        version: StateVersion(version),
    };
    for node in &training.program.nodes {
        if let TensorOp::ParamRead { param } = node.op {
            effects.insert(node.id, effect(next_token(), param, Access::Read, 0));
        }
    }
    for (&param, state) in &moments {
        for moment in Moment::BOTH {
            let read = moment_effect(next_token(), param, moment, Access::Read, 0);
            moment_effects.insert(state.input(moment), read);
        }
    }
    for (&param, &update) in &updates {
        effects.insert(update, effect(next_token(), param, Access::Write, 1));
    }
    for (&param, state) in &moments {
        for moment in Moment::BOTH {
            let write = moment_effect(next_token(), param, moment, Access::Write, 1);
            moment_effects.insert(state.next(moment), write);
        }
    }

    let step = TrainingStep {
        training,
        optimizer: Optimizer {
            lr,
            rule: OptimizerRule::AdamW(adamw),
            versions,
            updates,
            effects,
            step_inputs,
            moments,
            muon_lr_t: None,
            buffers: BTreeMap::new(),
            moment_effects,
        },
    };
    verify_step(&step)?;
    Ok(step)
}

/// Appends the `Scalar` of each of [`AdamW::constants`] in its order and the rank-0 `f32` inputs
/// `lr_t`, `c1` and `c2`, attributed to `param`.
fn adamw_prologue(
    append: &mut Appender,
    t: &mut TrainingProgram,
    param: ParamId,
    adamw: &AdamW,
) -> Result<([ValueId; 6], StepInputs), AttachError> {
    let mut constants = [ValueId(0); 6];
    for (slot, value) in constants.iter_mut().zip(adamw.constants()) {
        *slot = append.op(t, param, TensorOp::Scalar(value), &[])?;
    }
    let rank0 = TensorType::f32_scalar();
    let inputs = StepInputs {
        lr_t: append.input(t, param, LR_T_INPUT.into(), &rank0)?,
        c1: append.input(t, param, C1_INPUT.into(), &rank0)?,
        c2: append.input(t, param, C2_INPUT.into(), &rank0)?,
    };
    Ok((constants, inputs))
}

/// Appends the AdamW update of `param` with gradient `g` ([`attach_adamw`]) over the `constants`
/// and `inputs` of [`adamw_prologue`]: its moment inputs, then each operation in order. Returns
/// the update and the moments.
fn adamw_update(
    append: &mut Appender,
    t: &mut TrainingProgram,
    param: ParamId,
    g: ValueId,
    [beta1, rest1, beta2, rest2, eps, decay]: [ValueId; 6],
    inputs: StepInputs,
) -> Result<(ValueId, Moments), AttachError> {
    let decl = &t.program.params[&param];
    let (ty, path) = (decl.ty.clone(), decl.path.clone());
    let p = first_read(t, param);
    let m = append.input(t, param, moment_input(Moment::First, &path), &ty)?;
    let v = append.input(t, param, moment_input(Moment::Second, &path), &ty)?;
    let mut op = |t: &mut TrainingProgram, op: TensorOp, operands: &[ValueId]| {
        append.op(t, param, op, operands)
    };
    let decayed_m = op(t, TensorOp::Mul, &[beta1, m])?;
    let new_m = op(t, TensorOp::Mul, &[rest1, g])?;
    let m_next = op(t, TensorOp::Add, &[decayed_m, new_m])?;
    let decayed_v = op(t, TensorOp::Mul, &[beta2, v])?;
    let scaled_g = op(t, TensorOp::Mul, &[rest2, g])?;
    let new_v = op(t, TensorOp::Mul, &[scaled_g, g])?;
    let v_next = op(t, TensorOp::Add, &[decayed_v, new_v])?;
    let m_hat = op(t, TensorOp::Mul, &[m_next, inputs.c1])?;
    let v_hat = op(t, TensorOp::Mul, &[v_next, inputs.c2])?;
    let root = op(t, TensorOp::Sqrt, &[v_hat])?;
    let denominator = op(t, TensorOp::Add, &[root, eps])?;
    let direction = op(t, TensorOp::Div, &[m_hat, denominator])?;
    let shrink = op(t, TensorOp::Mul, &[decay, p])?;
    let step = op(t, TensorOp::Add, &[direction, shrink])?;
    let scaled = op(t, TensorOp::Mul, &[inputs.lr_t, step])?;
    let update = op(t, TensorOp::Sub, &[p, scaled])?;
    let moments = Moments {
        m,
        v,
        m_next,
        v_next,
    };
    Ok((update, moments))
}

/// The static `[rows, cols]` of a rank-2 type, `None` for any other.
fn matrix_shape(ty: &TensorType) -> Option<(u64, u64)> {
    match ty.shape[..] {
        [Dim::Static(rows), Dim::Static(cols)] => Some((rows, cols)),
        _ => None,
    }
}

/// The parameters `muon(...)` updates with Muon (DESIGN.md § 2.1, coordinator decision D67):
/// each parameter with a gradient whose type is rank 2, of static shape, and that no
/// `embedding` reads as its table, directly or through any chain of operands. A table tied to a
/// decoder through `transpose` is such a parameter's exclusion too: the `embedding` reads it.
/// Every other parameter with a gradient takes AdamW.
pub fn muon_params(training: &TrainingProgram) -> BTreeSet<ParamId> {
    let program = &training.program;
    let nodes: BTreeMap<ValueId, &TensorNode> = program.nodes.iter().map(|n| (n.id, n)).collect();
    let mut tables = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut pending: Vec<ValueId> = program
        .nodes
        .iter()
        .filter(|n| n.op == TensorOp::Embedding)
        .filter_map(|n| n.operands.first().copied())
        .collect();
    while let Some(value) = pending.pop() {
        if !seen.insert(value) {
            continue;
        }
        let Some(node) = nodes.get(&value) else {
            continue;
        };
        if let TensorOp::ParamRead { param } = node.op {
            tables.insert(param);
        }
        pending.extend(node.operands.iter().copied());
    }
    training
        .gradients
        .keys()
        .filter(|p| !tables.contains(p))
        .filter(|p| matrix_shape(&program.params[p].ty).is_some())
        .copied()
        .collect()
}

/// Attaches the Muon step of `muon` at peak rate `lr` to `training` (DESIGN.md § 2.1, § 6, § 8,
/// coordinator decision D67): Muon for each of [`muon_params`] and the AdamW update of
/// [`attach_adamw`] at `adam_lr`, the shared `weight_decay` and the betas and `eps` of
/// [`Muon::adamw`] for every other parameter with a gradient, in `ParamId` order.
///
/// `training` must pass [`crate::ir::verify::verify_training`], `lr` must be positive and finite
/// and `muon` in range ([`Muon::in_range`]). The result appends, tagged [`Subgraph::Optimizer`]:
/// when a parameter takes AdamW, the constants and the inputs `lr_t`, `c1` and `c2` of
/// [`attach_adamw`], `lr_t` the schedule of `adam_lr`; when a parameter takes Muon, a `Scalar`
/// for each of [`Muon::constants`] in its order and the rank-0 `f32` input
/// [`MUON_LR_T_INPUT`]. Then for each parameter, in `ParamId` order, its AdamW update, or for a
/// Muon parameter `p` of shape `[rows, cols]` with gradient `g` the input `b` of its type
/// ([`moment_input`] of [`Moment::Momentum`]), the `Scalar` of its [`Muon::scale`], and, each
/// one rounded `f32` operation or matmul in this order:
///
/// ```text
/// b' = momentum * b + g
/// u  = g + momentum * b'
/// x  = u / (sqrt(sum(u * u)) + 1e-7)            transposed when rows > cols
/// ns_steps times:
///     a = matmul(x, transpose(x))
///     x = 3.4445 * x + matmul(-4.7750 * a + 2.0315 * matmul(a, a), x)
/// x transposed back when rows > cols
/// p' = p - muon_lr_t * (scale * x + weight_decay * p)
/// ```
///
/// Its outputs are those of `training`, then the updates, then the next value of each state of
/// [`Optimizer::state_slots`]. Every declared parameter is at [`StateVersion`] 0. Effect tokens
/// count from 0: a read for each `ParamRead` node in node order, a read of each state in
/// [`Optimizer::state_slots`] order, a write for each update in `ParamId` order, then a write of
/// each state in the same order; reads are of version 0, writes of version 1. The result passes
/// [`crate::ir::verify::verify_step`].
pub fn attach_muon(
    training: TrainingProgram,
    lr: f32,
    muon: Muon,
) -> Result<TrainingStep, AttachError> {
    check_attach(&training, lr)?;
    if !muon.in_range() {
        return Err(AttachError::Hyperparameters);
    }
    let split = muon_params(&training);
    let mut training = training;
    let mut append = Appender::new(&training);

    let gradients: Vec<(ParamId, ValueId)> =
        training.gradients.iter().map(|(&p, &g)| (p, g)).collect();
    let t = &mut training;
    let mut adamw_prelude = None;
    if let Some(&(first, _)) = gradients.iter().find(|(p, _)| !split.contains(p)) {
        adamw_prelude = Some(adamw_prologue(&mut append, t, first, &muon.adamw)?);
    }
    let mut muon_prelude = None;
    if let Some(&(first, _)) = gradients.iter().find(|(p, _)| split.contains(p)) {
        let mut constants = [ValueId(0); 6];
        for (slot, value) in constants.iter_mut().zip(muon.constants()) {
            *slot = append.op(t, first, TensorOp::Scalar(value), &[])?;
        }
        let rate = append.input(t, first, MUON_LR_T_INPUT.into(), &TensorType::f32_scalar())?;
        muon_prelude = Some((constants, rate));
    }

    let mut updates = BTreeMap::new();
    let mut moments = BTreeMap::new();
    let mut buffers = BTreeMap::new();
    for (param, g) in gradients {
        match (muon_prelude, adamw_prelude) {
            (Some((constants, rate)), _) if split.contains(&param) => {
                let (update, buffer) =
                    muon_update(&mut append, t, param, g, &muon, constants, rate)?;
                updates.insert(param, update);
                buffers.insert(param, buffer);
            }
            (_, Some((constants, inputs))) => {
                let (update, state) = adamw_update(&mut append, t, param, g, constants, inputs)?;
                updates.insert(param, update);
                moments.insert(param, state);
            }
            _ => unreachable!("each parameter's prelude is appended above"),
        }
    }

    let mut optimizer = Optimizer {
        lr,
        rule: OptimizerRule::Muon(muon),
        versions: initial_versions(&training),
        updates,
        effects: BTreeMap::new(),
        step_inputs: adamw_prelude.map(|(_, inputs)| inputs),
        moments,
        muon_lr_t: muon_prelude.map(|(_, rate)| rate),
        buffers,
        moment_effects: BTreeMap::new(),
    };
    let slots = optimizer.state_slots();
    training
        .program
        .outputs
        .extend(optimizer.updates.values().copied());
    training
        .program
        .outputs
        .extend(slots.iter().map(|s| s.next));

    let mut token = 0u32;
    let mut next_token = || {
        let current = EffectToken(token);
        token += 1;
        current
    };
    let effect = |token, param, access, version| Effect {
        token,
        param,
        access,
        version: StateVersion(version),
    };
    let moment_effect = |token, slot: &StateSlot, access, version| MomentEffect {
        token,
        param: slot.param,
        moment: slot.moment,
        access,
        version: StateVersion(version),
    };
    for node in &training.program.nodes {
        if let TensorOp::ParamRead { param } = node.op {
            let read = effect(next_token(), param, Access::Read, 0);
            optimizer.effects.insert(node.id, read);
        }
    }
    for slot in &slots {
        let read = moment_effect(next_token(), slot, Access::Read, 0);
        optimizer.moment_effects.insert(slot.input, read);
    }
    for (&param, &update) in &optimizer.updates {
        let write = effect(next_token(), param, Access::Write, 1);
        optimizer.effects.insert(update, write);
    }
    for slot in &slots {
        let write = moment_effect(next_token(), slot, Access::Write, 1);
        optimizer.moment_effects.insert(slot.next, write);
    }

    let step = TrainingStep {
        training,
        optimizer,
    };
    verify_step(&step)?;
    Ok(step)
}

/// Appends the Muon update of `param` with gradient `g` ([`attach_muon`]) over the `constants`
/// of [`Muon::constants`] and the rate input `rate`: its buffer input, its scale, then each
/// operation in order. Returns the update and the buffer, its norm beside it.
fn muon_update(
    append: &mut Appender,
    t: &mut TrainingProgram,
    param: ParamId,
    g: ValueId,
    muon: &Muon,
    [momentum, tiny, a, b, c, decay]: [ValueId; 6],
    rate: ValueId,
) -> Result<(ValueId, Buffer), AttachError> {
    let decl = &t.program.params[&param];
    let (ty, path) = (decl.ty.clone(), decl.path.clone());
    let (rows, cols) = matrix_shape(&ty).expect("a Muon parameter is a static matrix");
    let p = first_read(t, param);
    let buffer = append.input(t, param, moment_input(Moment::Momentum, &path), &ty)?;
    let scale = append.op(t, param, TensorOp::Scalar(Muon::scale(rows, cols)), &[])?;
    let mut op = |t: &mut TrainingProgram, op: TensorOp, operands: &[ValueId]| {
        append.op(t, param, op, operands)
    };
    let decayed = op(t, TensorOp::Mul, &[momentum, buffer])?;
    let b_next = op(t, TensorOp::Add, &[decayed, g])?;
    let ahead = op(t, TensorOp::Mul, &[momentum, b_next])?;
    let u = op(t, TensorOp::Add, &[g, ahead])?;
    let squares = op(t, TensorOp::Mul, &[u, u])?;
    let sum = op(t, TensorOp::ReduceSum, &[squares])?;
    let norm = op(t, TensorOp::Sqrt, &[sum])?;
    let denominator = op(t, TensorOp::Add, &[norm, tiny])?;
    let mut x = op(t, TensorOp::Div, &[u, denominator])?;
    let transposed = rows > cols;
    if transposed {
        x = op(t, TensorOp::Transpose, &[x])?;
    }
    for _ in 0..muon.ns_steps {
        let xt = op(t, TensorOp::Transpose, &[x])?;
        let gram = op(t, TensorOp::MatMul, &[x, xt])?;
        let square = op(t, TensorOp::MatMul, &[gram, gram])?;
        let cubic = op(t, TensorOp::Mul, &[c, square])?;
        let linear = op(t, TensorOp::Mul, &[b, gram])?;
        let poly = op(t, TensorOp::Add, &[linear, cubic])?;
        let applied = op(t, TensorOp::MatMul, &[poly, x])?;
        let kept = op(t, TensorOp::Mul, &[a, x])?;
        x = op(t, TensorOp::Add, &[kept, applied])?;
    }
    if transposed {
        x = op(t, TensorOp::Transpose, &[x])?;
    }
    let scaled_x = op(t, TensorOp::Mul, &[scale, x])?;
    let shrink = op(t, TensorOp::Mul, &[decay, p])?;
    let step = op(t, TensorOp::Add, &[scaled_x, shrink])?;
    let scaled = op(t, TensorOp::Mul, &[rate, step])?;
    let update = op(t, TensorOp::Sub, &[p, scaled])?;
    Ok((
        update,
        Buffer {
            b: buffer,
            b_next,
            norm,
        },
    ))
}

/// One `input` or `target` declaration of a `train` block.
#[derive(Clone, Debug, PartialEq)]
pub struct DataDecl {
    pub name: String,
    pub ty: TensorType,
    pub span: SourceSpan,
}

/// The configuration of a `train` block, which the runtime reads (DESIGN.md § 6). Its optimizer
/// rule is the step's [`Optimizer::rule`].
///
/// It records the declarations as written; whether there is exactly one `input` and one `target`
/// is the data loader's to refuse.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainConfig {
    /// The number of steps of one run: positive.
    pub steps: u64,
    /// The learning rate of `sgd`, or the peak rate of `adamw`: positive and finite.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::tensor::TensorProgramBuilder;

    /// The sites are the token inputs a `Dropout` or `DropoutGrad` reads, once each, in input
    /// order; an `i32` input read as ids is none.
    #[test]
    fn dropout_sites_are_the_token_inputs_in_input_order() {
        let mut b = TensorProgramBuilder::new();
        let x = b.input("x", TensorType::f32(&[3])).unwrap();
        let ids = b.input("ids", TensorType::i32(&[3])).unwrap();
        let second = b.input("M.f.dropout#1", TensorType::i32(&[2])).unwrap();
        let first = b.input("M.f.dropout#0", TensorType::i32(&[2])).unwrap();
        let a = b.dropout(x, first, 0.5).unwrap();
        let c = b.dropout(a, second, 0.5).unwrap();
        let g = b.dropout_grad(second, c, 0.5).unwrap();
        let _ = (ids, g);
        let program = b.finish();
        assert_eq!(dropout_sites(&program), ["M.f.dropout#1", "M.f.dropout#0"]);

        let mut b = TensorProgramBuilder::new();
        b.input("x", TensorType::f32(&[3])).unwrap();
        assert!(dropout_sites(&b.finish()).is_empty());
    }

    const ADAMW: AdamW = AdamW {
        weight_decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        warmup_steps: 2,
    };

    /// `mse(matmul(x, w), y)` differentiated; with `unused`, a parameter nothing reads.
    fn linear(unused: bool) -> TrainingProgram {
        use crate::ir::tensor::Initializer;
        let mut b = TensorProgramBuilder::new();
        let w = b
            .declare_param("L.w", TensorType::f32(&[2, 1]), Initializer::XavierUniform)
            .unwrap();
        if unused {
            b.declare_param("L.u", TensorType::f32(&[2, 1]), Initializer::XavierUniform)
                .unwrap();
        }
        let x = b.input("x", TensorType::f32(&[3, 2])).unwrap();
        let y = b.input("y", TensorType::f32(&[3, 1])).unwrap();
        let w = b.param_read(w).unwrap();
        let h = b.matmul(x, w).unwrap();
        let loss = b.mse(h, y).unwrap();
        crate::autodiff::differentiate(&b.finish(), loss).unwrap()
    }

    /// The schedule warms up linearly from 0 over `warmup_steps` and decays linearly to 0 at the
    /// last step; the bias corrections are `1 / (1 - beta^t)`; each is computed in `f64` and
    /// rounded once.
    #[test]
    fn adamw_scalars_follow_the_settled_schedule() {
        let lr = 0.1f32;
        let rates: Vec<f32> = (1..=5).map(|t| ADAMW.scalars(lr, t, 5).lr_t).collect();
        let peak = f64::from(lr);
        let expected = [peak / 2.0, peak, peak * 2.0 / 3.0, peak / 3.0, 0.0];
        assert_eq!(rates, expected.map(|r| r as f32));
        let no_warmup = AdamW {
            warmup_steps: 0,
            ..ADAMW
        };
        assert_eq!(no_warmup.scalars(lr, 1, 4).lr_t, (peak * 0.75) as f32);
        let first = ADAMW.scalars(lr, 1, 5);
        assert_eq!(first.c1, (1.0 / (1.0 - f64::from(0.9f32))) as f32);
        assert_eq!(first.c2, (1.0 / (1.0 - f64::from(0.999f32))) as f32);
        assert_eq!(ADAMW.scalars(lr, 6, 5).lr_t, 0.0, "past the last step");
        let refused = AdamW {
            warmup_steps: 5,
            ..ADAMW
        };
        assert_eq!(refused.scalars(lr, 6, 5).lr_t, 0.0, "no span to decay over");
    }

    /// The constants are in their order, each `1 - beta` rounded once from `f64`.
    #[test]
    fn adamw_constants_are_the_betas_their_complements_eps_and_the_decay() {
        let [b1, r1, b2, r2, eps, decay] = ADAMW.constants();
        assert_eq!((b1, b2, eps, decay), (0.9, 0.999, 1e-8, 0.01));
        assert_eq!(r1, (1.0 - f64::from(0.9f32)) as f32);
        assert_eq!(r2, (1.0 - f64::from(0.999f32)) as f32);
        assert!(ADAMW.in_range());
        for out in [
            AdamW {
                beta1: 1.0,
                ..ADAMW
            },
            AdamW {
                beta2: -0.1,
                ..ADAMW
            },
            AdamW { eps: 0.0, ..ADAMW },
            AdamW {
                weight_decay: -0.1,
                ..ADAMW
            },
            AdamW {
                weight_decay: f32::INFINITY,
                ..ADAMW
            },
        ] {
            assert!(!out.in_range(), "{out:?}");
        }
    }

    /// The update of each parameter is the AdamW formula in written order, one op each, with
    /// its moments as inputs and outputs and their effects ordered reads before writes.
    #[test]
    fn attach_adamw_appends_the_update_in_written_order() {
        let step = attach_adamw(linear(true), 0.1, ADAMW).unwrap();
        let optimizer = &step.optimizer;
        let program = &step.training.program;
        assert_eq!(optimizer.rule, OptimizerRule::AdamW(ADAMW));
        let w = ParamId(0);
        assert_eq!(optimizer.updates.keys().collect::<Vec<_>>(), vec![&w]);
        assert_eq!(optimizer.moments.keys().collect::<Vec<_>>(), vec![&w]);
        let names: Vec<&str> = program
            .nodes
            .iter()
            .filter(|n| step.training.subgraphs[&n.id] == Subgraph::Optimizer)
            .map(|n| n.op.name())
            .collect();
        let mut expected = vec!["Scalar"; 6];
        expected.extend(["Input"; 5]);
        expected.extend([
            "Mul", "Mul", "Add", "Mul", "Mul", "Mul", "Add", "Mul", "Mul", "Sqrt", "Add", "Div",
            "Mul", "Add", "Mul", "Sub",
        ]);
        assert_eq!(names, expected);
        let moments = optimizer.moments[&w];
        let input = |value: ValueId| match &program.node(value).unwrap().op {
            TensorOp::Input { name } => name.clone(),
            other => panic!("{other:?}"),
        };
        assert_eq!(input(moments.m), "optimizer.m@L.w");
        assert_eq!(input(moments.v), "optimizer.v@L.w");
        let inputs = optimizer.step_inputs.unwrap();
        assert_eq!(
            [input(inputs.lr_t), input(inputs.c1), input(inputs.c2)],
            [LR_T_INPUT, C1_INPUT, C2_INPUT]
        );
        let update = optimizer.updates[&w];
        let mut outputs = vec![step.training.loss, step.training.gradients[&w], update];
        outputs.extend([moments.m_next, moments.v_next]);
        assert_eq!(program.outputs, outputs);
        let tokens = |value: ValueId| optimizer.moment_effects[&value].token.0;
        let read = optimizer.effects_in_order()[0].1;
        assert_eq!((read.token.0, read.access), (0, Access::Read));
        assert_eq!((tokens(moments.m), tokens(moments.v)), (1, 2));
        assert_eq!(optimizer.effects[&update].token.0, 3);
        assert_eq!((tokens(moments.m_next), tokens(moments.v_next)), (4, 5));
        let write = optimizer.moment_effects[&moments.v_next];
        assert_eq!(
            (write.moment, write.access, write.version),
            (Moment::Second, Access::Write, StateVersion(1))
        );
        assert_eq!(optimizer.host_inputs().len(), 5);
    }

    /// Out-of-range hyperparameters and learning rates are refused before anything is appended.
    #[test]
    fn attach_adamw_refuses_out_of_range_values() {
        let beta = AdamW {
            beta1: 1.0,
            ..ADAMW
        };
        assert_eq!(
            attach_adamw(linear(false), 0.1, beta),
            Err(AttachError::Hyperparameters)
        );
        assert!(matches!(
            attach_adamw(linear(false), 0.0, ADAMW),
            Err(AttachError::LearningRate { .. })
        ));
        let sgd = attach(linear(false), 0.1, OptimizerRule::Sgd).unwrap();
        assert_eq!(sgd, attach_sgd(linear(false), 0.1).unwrap());
        assert!(sgd.optimizer.moments.is_empty() && sgd.optimizer.step_inputs.is_none());
    }
}
