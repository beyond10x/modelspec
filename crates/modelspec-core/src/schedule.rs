//! The fixed safe schedule of M1 (DESIGN.md § 5.2, § 5.3): a verified training step to Kernel IR
//! and its execution plans.
//!
//! Ops run in Training IR order, except that the step plan records the `mse` loss sum right
//! after the largest matmul it can share a barrier interval with ([`step_order`]). Each
//! elementwise, broadcast, transpose and SGD op is one Dispatch, one invocation per output
//! element; a matmul is one Dispatch of the workgroups of its [`MatMulTile`]; the `mse` loss is
//! one Dispatch of one workgroup (DESIGN.md § 5.4); every other reduce is four consecutive steps:
//! an `Allocate` of its partials buffer, a stage-1 Dispatch that writes one partial per workgroup
//! into it, a `Barrier`, and a stage-2 Dispatch of one workgroup that reads the partials.
//! Each row-wise op is one Dispatch of workgroups of 64 ([`rowwise_kernel`]): `Embedding`,
//! `AddRows`, `Gelu`, `GeluGrad`, `Dropout` and `DropoutGrad` one invocation per output element,
//! `EmbeddingGrad`, `LayerNorm`, `LayerNormGradX`, `Softmax` and `SoftmaxGrad` one per output row,
//! and the column reductions `SumRows`, `LayerNormGradGamma`, `SigregColumns` and
//! `SigregColumnsGrad` one per output column.
//! `Attention`, `AttentionGradQ`, `AttentionGradK` and `AttentionGradV` are one Dispatch each of
//! workgroups of [`ATTENTION_WORKGROUP`], one per sequence, head and tile of [`attention_tile`]
//! rows ([`attention_workgroups`], story:vulkan-attention-workgroup-per-head), the head count a
//! last immediate operand; `CrossEntropy` is a loss sum of one workgroup, `CrossEntropyGrad` one
//! invocation per row, and `Sqrt` and `Div` elementwise kernels of one invocation per output
//! element that no elementwise chain extends ([`attention_loss_kernel`], coordinator decision
//! D73). The kernels of an attention call share its softmax statistics ([`softmax_stats_sites`],
//! story:vulkan-attention-softmax-stats-reused): an `Allocate` of a `[rows · heads, 3]` buffer,
//! numbered after every other buffer, right before the `Attention` Dispatch, which writes it
//! before its result, as `AttentionGradQ` does, and `AttentionGradK` and `AttentionGradV` read it
//! after their operands.
//! `Input`, `ParamRead` and `Scalar` dispatch nothing: inputs and parameters are bound views, and a scalar
//! is compiled into the kernels that read it.
//!
//! Four exceptions, and the elementwise chains below, remove dispatches without changing a bit
//! of any result (DESIGN.md § 3.1,
//! § 5.4). An `Add`, `Sub` or `Mul` of rank 0 whose operands are all compile-time scalars is
//! folded: the host computes it as one round-to-nearest-even `f32` operation, and every reader
//! takes the result as an immediate. It is not folded when an operand or the result is infinite,
//! NaN or subnormal, when it is the loss, the prediction or an update, or when a reader takes no
//! immediate; a `Relu` or `ReluGrad` of compile-time scalars is never folded and stays a kernel
//! whose operands are immediates. The first `Relu`
//! that reads a `MatMul` output is the matmul kernel's [`Epilogue`]: the kernel writes the
//! activation, and also the pre-activation when another op, the loss or the prediction reads it.
//! A `ReluGrad(x, g)` whose `g` is a `MatMul` output that nothing else reads, and that is neither
//! the loss, the prediction nor an update, is that matmul's `ReluGrad` [`Epilogue`] when `x` is
//! stored before the matmul and no parameter `x` is updated before it: the kernel reads `x` as a
//! third operand and writes `x > 0 ? sum : 0.0`, and `g` has no buffer
//! (story:step-sites-beat-pytorch).
//! An SGD update `p - m` whose `m` is `lr * g`, with one scalar factor and no other reader, is
//! one `SgdUpdate` dispatch over `(p, lr, g)` in the factors' order, and `m` has no buffer.
//!
//! A `Transpose` whose every reader is a `MatMul` is folded into its readers: it dispatches
//! nothing and has no buffer, and each matmul reads the transpose's operand in place, its view
//! bound where the transpose's output was and the operand marked in [`Kernel::transposed`]. It
//! stays a dispatch when another op reads it, when it is the loss or the prediction, or when its
//! operand is a parameter whose in-place update comes before one of its readers. A matmul that
//! reads a folded transpose computes every output element as before (DESIGN.md § 3.1, § 5.2).
//!
//! A chain of elementwise ops (`Add`, `Sub`, `Mul`, `Relu`, `ReluGrad`) not fused above, each op
//! but the last read once and by the next op alone, at that op's type, and neither the loss, the
//! prediction nor a next optimizer state, is one `Elementwise` dispatch at the place of its last op
//! (story:fuse-elementwise-chains-and-gradient-sums). Ops are taken in order: an op extends the
//! chain of its first operand, by position, that qualifies while the chain's buffer operands stay
//! within [`MAX_CHAIN_READS`] and no parameter it reads is updated before the op, and otherwise
//! starts a chain. The kernel's op is the chain's [`ElementwiseChain::name`] and its operands
//! the chain's leaves, so an adjoint's sum `((c0 + c1) + c2) + …` keeps the order the IR adds
//! it in. The emitter rounds each op once, as the unfused kernels do, so the chain's result has
//! the bits of the unfused plan.
//!
//! There is no buffer reuse: every stored value has its own buffer, except that each SGD update
//! writes its parameter's one view in place, each next optimizer state (an AdamW moment's next
//! value) is written in place into the buffer of the `Input` its state is read from
//! ([`host_inputs`]) and is neither folded nor fused into a reader, a folded scalar is an
//! immediate, the `lr * g` of a
//! fused update has no buffer, a pre-activation that nothing but its epilogue `Relu` reads
//! has none either, nor has a matmul output whose `ReluGrad` is its epilogue, and neither has a
//! folded transpose nor a value inside an elementwise chain. A `Barrier` precedes any Dispatch that
//! reads a buffer an earlier Dispatch wrote since the last barrier, or writes one an earlier
//! Dispatch read or wrote since then; it depends on the previous barrier and on every Dispatch
//! since, so a step after it is ordered after everything before it. Each plan ends with its one
//! fence, a `Barrier` with no dependencies, so a runner may record it back to back.
//! A buffer over [`MAX_BUFFER_BYTES`] or a dispatch over
//! [`MAX_INVOCATIONS`] is refused as a target limit. The prediction has its own view: a forward that returns an
//! input or a parameter copies it there with a `Copy` step, and one that returns a literal fills
//! it with a Broadcast dispatch. The workgroup sizes are the constants of [`KernelKind`], except a
//! matmul's, its tile's: [`schedule`] gives every matmul [`MatMulTile::of`], the tile of every
//! Vulkan device, and [`schedule_for`] with the limits of a device profile ([`MatMulLimits`])
//! [`MatMulTile::for_limits`], which takes its invocation limit and workgroup memory (DESIGN.md
//! § 3.1, § 5.2). Nothing here opens a device.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::artifact::{DataRole, DataSpec, ParameterSpec, canonical_parameter_id};
use crate::ir::kernel::{
    BufferView, ChainLink, ElementwiseChain, Epilogue, ExecutionPlan, Kernel, KernelKind,
    MIN_SHARED_MEMORY_BYTES, MatMulLimits, MatMulTile, Operand, PlanOp, PlanStep,
    SOFTMAX_STATS_PER_QUERY,
};
use crate::ir::tensor::{DType, Dim, ParamId, TensorNode, TensorOp, TensorType, ValueId};
use crate::ir::training::{TrainConfig, TrainingStep, sigreg_projection_inputs};
use crate::ops;

/// The most workgroups in one dimension of a dispatch on every Vulkan device.
pub const MAX_WORKGROUPS_PER_DIMENSION: u32 = 65_535;

/// Why [`schedule`] refused a verified step: each is a compiler bug or a size M1 cannot dispatch.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ScheduleError {
    #[error("{value} has unresolved or overflowing shape {ty}")]
    Shape { value: ValueId, ty: TensorType },
    #[error("`{name}` has unresolved or overflowing shape {ty}")]
    DeclarationShape { name: String, ty: TensorType },
    #[error("Input {value} names `{name}`, which the train block does not declare")]
    UndeclaredInput { value: ValueId, name: String },
    #[error("{value} reads {operand}, which is not defined before it")]
    UndefinedOperand { value: ValueId, operand: ValueId },
    #[error("{value} needs {groups} workgroups, more than three dimensions of 65535 hold")]
    TooManyWorkgroups { value: ValueId, groups: u64 },
    #[error("{value} is not a value of the step")]
    MissingValue { value: ValueId },
    #[error("buffer b{buffer} holds {bytes} bytes, more than a storage buffer binding addresses")]
    BufferTooLarge { buffer: u32, bytes: u64 },
    #[error("{value} dispatches {invocations} invocations, more than a 32-bit index counts")]
    TooManyInvocations { value: ValueId, invocations: u64 },
    /// The op has no Vulkan kernel for its operands (`E-TARGET-011`): a row-wise op that reads
    /// an immediate operand, an attention op over more than [`MAX_ATTENTION_KEYS`] keys, or any
    /// other operands [`rowwise_kernel`] and [`attention_loss_kernel`] do not take.
    #[error("{value} is {op}, which has no Vulkan kernel")]
    NoKernel { value: ValueId, op: &'static str },
}

/// The most bytes one storage buffer binding addresses: Vulkan's `maxStorageBufferRange` is a
/// `uint32_t`.
pub const MAX_BUFFER_BYTES: u64 = u32::MAX as u64;

/// The most invocations of one dispatch: each has a distinct 32-bit linear index.
pub const MAX_INVOCATIONS: u64 = 1 << 32;

impl ScheduleError {
    /// Whether the step exceeds a limit of the target rather than breaking an invariant of the
    /// compiler: too many workgroups or invocations, a buffer larger than a binding addresses, or
    /// a static shape whose byte count overflows.
    pub fn is_target_limit(&self) -> bool {
        let fully_static = |ty: &TensorType| ty.shape.iter().all(|d| matches!(d, Dim::Static(_)));
        match self {
            ScheduleError::TooManyWorkgroups { .. }
            | ScheduleError::BufferTooLarge { .. }
            | ScheduleError::TooManyInvocations { .. } => true,
            ScheduleError::Shape { ty, .. } | ScheduleError::DeclarationShape { ty, .. } => {
                fully_static(ty)
            }
            _ => false,
        }
    }
}

/// Where a value lives: a view, or a constant compiled into its readers.
#[derive(Clone, Copy, Debug)]
enum Place {
    View(BufferView),
    Immediate(f32),
}

/// One computing op, or a fused chain of them, with its kernels and views.
struct Work {
    /// The value its plan steps name: the op's, the matmul's of a fused epilogue, the update's
    /// of a fused SGD update.
    value: ValueId,
    /// Every value the work computes, `value` first.
    values: Vec<ValueId>,
    /// One kernel, or stage 1 and stage 2 of a reduce.
    kernels: Vec<Kernel>,
    reads: Vec<BufferView>,
    /// In binding order; one view but for an epilogue that keeps its pre-activation and an
    /// attention kernel that writes softmax statistics before its result.
    writes: Vec<BufferView>,
    /// The partials buffer of a two-stage reduce.
    partials: Option<BufferView>,
    /// The softmax statistics buffer an `Attention` work allocates before its dispatch and
    /// writes first ([`Kernel::softmax_stats`]).
    allocates: Option<BufferView>,
}

/// The fusions and folded constants of a step (the module documentation).
struct Fusions {
    /// The value of each folded op.
    folded: BTreeMap<ValueId, f32>,
    /// Each matmul with a `relu` epilogue: the `relu`, and whether the pre-activation is written.
    relu_of: BTreeMap<ValueId, (ValueId, bool)>,
    /// Each matmul with a `ReluGrad` epilogue: the `ReluGrad`, and the pre-activation it reads.
    relu_grad_of: BTreeMap<ValueId, (ValueId, ValueId)>,
    /// Each fused SGD update: its `lr * g`.
    scaled: BTreeMap<ValueId, ValueId>,
    /// The ops computed inside another op's kernel.
    absorbed: BTreeSet<ValueId>,
    /// Each folded transpose: the operand its matmul readers read in place.
    transposed: BTreeMap<ValueId, ValueId>,
    /// Each fused elementwise chain, by the value of its last op.
    chains: BTreeMap<ValueId, Chain>,
}

/// The most buffers a fused elementwise chain reads: with its result, 4 storage buffers, the
/// Vulkan minimum of `maxPerStageDescriptorStorageBuffers`.
pub const MAX_CHAIN_READS: usize = 3;

/// A chain of elementwise ops one kernel computes (the module documentation).
struct Chain {
    /// Its ops in order, the last the one whose value the dispatch names.
    nodes: Vec<ValueId>,
    /// The first op's operands, then each later op's operands but the result before it.
    leaves: Vec<ValueId>,
    /// The kernel op, [`ElementwiseChain::name`].
    name: &'static str,
}

/// The chain an elementwise op is the last op of, as the pass builds it.
struct OpenChain {
    nodes: Vec<ValueId>,
    leaves: Vec<ValueId>,
    first: &'static str,
    links: Vec<ChainLink>,
}

/// The name of `op` when it is one of the five elementwise ops.
fn elementwise_op(op: &TensorOp) -> Option<&'static str> {
    match op {
        TensorOp::Add | TensorOp::Sub | TensorOp::Mul | TensorOp::Relu | TensorOp::ReluGrad => {
            Some(op.name())
        }
        _ => None,
    }
}

/// Whether `op` reads an immediate operand in its kernel.
fn takes_immediate(op: &TensorOp) -> bool {
    matches!(
        op,
        TensorOp::Add
            | TensorOp::Sub
            | TensorOp::Mul
            | TensorOp::Relu
            | TensorOp::ReluGrad
            | TensorOp::BroadcastScalar { .. }
    )
}

/// `op` over compile-time scalars, as its elementwise kernel computes it: one correctly rounded
/// `f32` instruction, or the compare and select of `relu` and `ReluGrad`.
fn fold(op: &TensorOp, operands: &[f32]) -> Option<f32> {
    let value = match (op, operands) {
        (TensorOp::Add, [a, b]) => a + b,
        (TensorOp::Sub, [a, b]) => a - b,
        (TensorOp::Mul, [a, b]) => a * b,
        _ => return None,
    };
    let exact = |x: f32| x.is_finite() && !x.is_subnormal();
    (operands.iter().all(|&x| exact(x)) && exact(value)).then_some(value)
}

/// The fusions of `nodes`, whose updates are `updates`, the update of each parameter
/// `update_of`, and whose loss, prediction and next optimizer states are `bound`: values a runner
/// reads, which neither fold nor fuse into a reader.
fn fusions(
    nodes: &[TensorNode],
    updates: &BTreeMap<ValueId, usize>,
    update_of: &BTreeMap<ParamId, ValueId>,
    bound: &BTreeSet<ValueId>,
) -> Fusions {
    let mut readers: BTreeMap<ValueId, Vec<&TensorNode>> = BTreeMap::new();
    for node in nodes {
        for operand in &node.operands {
            readers.entry(*operand).or_default().push(node);
        }
    }
    let readers_of = |value: ValueId| readers.get(&value).map_or(&[][..], |r| &r[..]);
    let op_of = |value: ValueId| nodes.iter().find(|n| n.id == value).map(|n| &n.op);

    let mut constants: BTreeMap<ValueId, f32> = BTreeMap::new();
    let mut folded = BTreeMap::new();
    for node in nodes {
        if let TensorOp::Scalar(value) = node.op {
            constants.insert(node.id, value);
            continue;
        }
        let operands: Option<Vec<f32>> = node
            .operands
            .iter()
            .map(|o| constants.get(o).copied())
            .collect();
        let foldable = node.ty.rank() == 0
            && !bound.contains(&node.id)
            && !updates.contains_key(&node.id)
            && readers_of(node.id).iter().all(|r| takes_immediate(&r.op));
        if let Some(value) = operands
            .filter(|_| foldable)
            .and_then(|o| fold(&node.op, &o))
        {
            constants.insert(node.id, value);
            folded.insert(node.id, value);
        }
    }

    let mut relu_of = BTreeMap::new();
    let mut absorbed = BTreeSet::new();
    for node in nodes {
        let [matmul] = node.operands[..] else {
            continue;
        };
        if node.op != TensorOp::Relu
            || op_of(matmul) != Some(&TensorOp::MatMul)
            || relu_of.contains_key(&matmul)
            || updates.contains_key(&matmul)
        {
            continue;
        }
        let pre_activation =
            bound.contains(&matmul) || readers_of(matmul).iter().any(|r| r.id != node.id);
        relu_of.insert(matmul, (node.id, pre_activation));
        absorbed.insert(node.id);
    }

    let mut scaled = BTreeMap::new();
    for node in nodes.iter().filter(|n| updates.contains_key(&n.id)) {
        let [_, product] = node.operands[..] else {
            continue;
        };
        let Some(mul) = nodes.iter().find(|n| n.id == product) else {
            continue;
        };
        let scalars = mul
            .operands
            .iter()
            .filter(|o| constants.contains_key(o))
            .count();
        let only_reader = matches!(readers_of(product), [r] if r.id == node.id);
        if node.op == TensorOp::Sub
            && mul.op == TensorOp::Mul
            && mul.operands.len() == 2
            && scalars == 1
            && only_reader
            && !bound.contains(&product)
            && !folded.contains_key(&product)
        {
            scaled.insert(node.id, product);
            absorbed.insert(product);
        }
    }

    let position: BTreeMap<ValueId, usize> =
        nodes.iter().enumerate().map(|(i, n)| (n.id, i)).collect();
    let mut transposed = BTreeMap::new();
    for node in nodes {
        let [operand] = node.operands[..] else {
            continue;
        };
        let readers = readers_of(node.id);
        if node.op != TensorOp::Transpose
            || bound.contains(&node.id)
            || updates.contains_key(&node.id)
            || readers.is_empty()
            || readers.iter().any(|r| r.op != TensorOp::MatMul)
        {
            continue;
        }
        let updated_at = match op_of(operand) {
            Some(TensorOp::ParamRead { param }) => update_of
                .get(param)
                .and_then(|update| position.get(update))
                .copied(),
            _ => None,
        };
        if updated_at.is_some_and(|at| readers.iter().any(|r| position[&r.id] > at)) {
            continue;
        }
        transposed.insert(node.id, operand);
    }

    let updated_before = |leaf: ValueId, at: usize| match op_of(leaf) {
        Some(TensorOp::ParamRead { param }) => update_of
            .get(param)
            .and_then(|update| position.get(update))
            .is_some_and(|&updated| updated < at),
        _ => false,
    };

    // A `ReluGrad(x, g)` whose adjoint `g` is a matmul output that it alone reads is that
    // matmul's epilogue, when `x` has a buffer before the matmul runs. Of the ops absorbed so
    // far only a fused `lr * g` product has none: a `relu` or `ReluGrad` epilogue stores its
    // value in a view its matmul writes.
    let unstored: BTreeSet<ValueId> = scaled.values().copied().collect();
    let mut relu_grad_of = BTreeMap::new();
    for node in nodes {
        let [x, g] = node.operands[..] else {
            continue;
        };
        let at = position[&node.id];
        let fits = node.op == TensorOp::ReluGrad
            && x != g
            && op_of(g) == Some(&TensorOp::MatMul)
            && matches!(readers_of(g), [r] if r.id == node.id)
            && !relu_of.contains_key(&g)
            && !bound.contains(&g)
            && !updates.contains_key(&g)
            && !updates.contains_key(&node.id)
            && !folded.contains_key(&node.id)
            && !constants.contains_key(&x)
            && !transposed.contains_key(&x)
            && !unstored.contains(&x)
            && position.get(&x).is_some_and(|&p| p < position[&g])
            && !updated_before(x, at);
        if fits {
            relu_grad_of.insert(g, (node.id, x));
            absorbed.insert(node.id);
        }
    }

    // Elementwise chains, in node order: an op extends the chain of its first operand, by
    // position, that it alone reads, once, at its own type, while the chain's buffer leaves stay
    // within MAX_CHAIN_READS and no parameter leaf is updated before the op; else it starts one.
    let fused_before = absorbed.clone();
    let ty_of: BTreeMap<ValueId, &TensorType> = nodes.iter().map(|n| (n.id, &n.ty)).collect();
    let buffers = |values: &[ValueId]| values.iter().filter(|v| !constants.contains_key(v)).count();
    let mut open: BTreeMap<ValueId, OpenChain> = BTreeMap::new();
    for node in nodes {
        let Some(name) = elementwise_op(&node.op) else {
            continue;
        };
        if folded.contains_key(&node.id)
            || fused_before.contains(&node.id)
            || updates.contains_key(&node.id)
        {
            continue;
        }
        let at = position[&node.id];
        let extended = node.operands.iter().enumerate().find_map(|(i, &operand)| {
            let chain = open.get(&operand)?;
            let others: Vec<ValueId> = node
                .operands
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &v)| v)
                .collect();
            let fits = !bound.contains(&operand)
                && matches!(readers_of(operand), [r] if r.id == node.id)
                && ty_of.get(&operand) == Some(&&node.ty)
                && buffers(&chain.leaves) + buffers(&others) <= MAX_CHAIN_READS
                && !chain
                    .leaves
                    .iter()
                    .chain(&others)
                    .any(|&leaf| updated_before(leaf, at));
            fits.then_some((i, operand, others))
        });
        let chain = match extended {
            Some((i, operand, others)) => {
                let mut chain = open.remove(&operand).expect("the chain is open");
                chain.nodes.push(node.id);
                chain.leaves.extend(others);
                chain.links.push(ChainLink { op: name, at: i });
                chain
            }
            None => OpenChain {
                nodes: vec![node.id],
                leaves: node.operands.clone(),
                first: name,
                links: Vec::new(),
            },
        };
        open.insert(node.id, chain);
    }
    let mut chains = BTreeMap::new();
    for (last, chain) in open {
        if chain.links.is_empty() {
            continue;
        }
        absorbed.extend(&chain.nodes[..chain.nodes.len() - 1]);
        let name = ElementwiseChain {
            first: chain.first,
            links: chain.links,
        }
        .name();
        chains.insert(
            last,
            Chain {
                nodes: chain.nodes,
                leaves: chain.leaves,
                name,
            },
        );
    }

    Fusions {
        folded,
        relu_of,
        relu_grad_of,
        scaled,
        absorbed,
        transposed,
        chains,
    }
}

/// Schedules `step`, whose forward prediction is `prediction`, with the data declarations of
/// `config`, for every Vulkan device: [`schedule_for`] without device limits. The result has one view
/// per parameter and passes [`crate::ir::verify::verify_plan`].
pub fn schedule(
    step: &TrainingStep,
    prediction: ValueId,
    config: &TrainConfig,
) -> Result<ExecutionPlan, ScheduleError> {
    schedule_for(step, prediction, config, None)
}

/// [`schedule`] within the `limits` of a device ([`MatMulLimits::of`] its profile): each matmul takes
/// [`MatMulTile::for_limits`] of its shape and transposed operands, and without limits
/// [`MatMulTile::of`]. Limits at the Vulkan minimum of 128 invocations give the plan of
/// [`schedule`]. The result passes [`crate::ir::verify::verify_plan_for`] with the profile of the limits
/// when that profile reports at least 128 invocations, as every conformant device does; under a
/// profile below 128 it does not, since every tile has at least 128 invocations.
pub fn schedule_for(
    step: &TrainingStep,
    prediction: ValueId,
    config: &TrainConfig,
    limits: Option<MatMulLimits>,
) -> Result<ExecutionPlan, ScheduleError> {
    schedule_with(step, prediction, config, limits, Kernels::Vulkan).map(|(plan, _)| plan)
}

/// The views of [`schedule`] alone, for an artifact with no kernels (a Cpu artifact): its
/// buffers, parameters, data, loss view and prediction view, with no kernels and empty plans.
/// An op with no Vulkan kernel ([`ScheduleError::NoKernel`] under [`schedule`]) takes the buffer
/// of its value like any other op, so a program without such an op has the views of
/// [`schedule`] exactly; every other refusal of [`schedule`] stays.
pub fn schedule_views(
    step: &TrainingStep,
    prediction: ValueId,
    config: &TrainConfig,
) -> Result<ExecutionPlan, ScheduleError> {
    let (mut plan, _) = schedule_with(step, prediction, config, None, Kernels::None)?;
    plan.kernels.clear();
    plan.step.clear();
    plan.forward.clear();
    Ok(plan)
}

/// One `Input` of a step that the training host supplies besides the data of the `train` block:
/// the RNG token of a dropout site, an AdamW step's scheduled rate or bias correction, or the
/// optimizer state of a parameter, and the view of the plan it is read from.
#[derive(Clone, Debug, PartialEq)]
pub struct HostInput {
    /// The `Input` node.
    pub value: ValueId,
    /// Its name: the dropout site, the sigreg site, [`crate::ir::training::LR_T_INPUT`],
    /// [`crate::ir::training::C1_INPUT`], [`crate::ir::training::C2_INPUT`] or a
    /// [`crate::ir::training::moment_input`].
    pub name: String,
    pub ty: TensorType,
    /// A buffer of its own. The step plan writes each optimizer state's next value into it, so
    /// the next step reads it there; it writes no other host input.
    pub view: BufferView,
}

/// The host inputs of `step` in node order, each with the view of the plans [`schedule`],
/// [`schedule_for`] and [`schedule_views`] give it: every buffer is numbered alike by the three,
/// a buffer of each host input included. A runner uploads the tokens and the scalars before each
/// step and the initial optimizer states once, and reads the states back from these views.
pub fn host_inputs(
    step: &TrainingStep,
    prediction: ValueId,
    config: &TrainConfig,
) -> Result<Vec<HostInput>, ScheduleError> {
    schedule_with(step, prediction, config, None, Kernels::None).map(|(_, inputs)| inputs)
}

/// Whether a schedule builds the Vulkan kernels of every op, or only the views.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kernels {
    Vulkan,
    None,
}

fn schedule_with(
    step: &TrainingStep,
    prediction: ValueId,
    config: &TrainConfig,
    limits: Option<MatMulLimits>,
    built: Kernels,
) -> Result<(ExecutionPlan, Vec<HostInput>), ScheduleError> {
    let program = &step.training.program;
    let mut buffers = Vec::new();

    let mut data = Vec::new();
    let declared = config
        .inputs
        .iter()
        .map(|d| (d, DataRole::Input))
        .chain(config.targets.iter().map(|d| (d, DataRole::Target)));
    for (decl, role) in declared {
        let view = allocate(&mut buffers, &decl.ty).ok_or(ScheduleError::DeclarationShape {
            name: decl.name.clone(),
            ty: decl.ty.clone(),
        })?;
        data.push(DataSpec {
            name: decl.name.clone(),
            role,
            ty: decl.ty.clone(),
            view,
        });
    }

    let mut parameters = Vec::new();
    for (&param, decl) in &program.params {
        let view = allocate(&mut buffers, &decl.ty).ok_or(ScheduleError::DeclarationShape {
            name: decl.path.clone(),
            ty: decl.ty.clone(),
        })?;
        parameters.push(ParameterSpec {
            param,
            path: decl.path.clone(),
            canonical_parameter_id: canonical_parameter_id(&decl.path),
            ty: decl.ty.clone(),
            views: vec![view],
        });
    }

    let updates: BTreeMap<ValueId, usize> = step
        .optimizer
        .updates
        .iter()
        .map(|(param, &value)| {
            let at = parameters.iter().position(|p| p.param == *param);
            (value, at.expect("an updated parameter is declared"))
        })
        .collect();

    // Each next optimizer state is written in place into the buffer its state is read from, so
    // the next step reads it there; the runner reads it back from that view.
    let in_place: BTreeMap<ValueId, ValueId> = step
        .optimizer
        .state_slots()
        .iter()
        .map(|slot| (slot.next, slot.input))
        .collect();
    let bound: BTreeSet<ValueId> = [step.training.loss, prediction]
        .into_iter()
        .chain(in_place.keys().copied())
        .collect();
    let fused = fusions(&program.nodes, &updates, &step.optimizer.updates, &bound);
    let tokens: BTreeSet<ValueId> = program
        .nodes
        .iter()
        .filter_map(|n| ops::token_operand(&n.op).and_then(|i| n.operands.get(i).copied()))
        .chain(step.optimizer.host_inputs())
        .chain(sigreg_projection_inputs(program))
        .chain(crate::passes::masked_rows::site_inputs(program))
        .collect();
    let mut host_inputs = Vec::new();
    let mut places: BTreeMap<ValueId, Place> = BTreeMap::new();
    let mut works: Vec<Work> = Vec::new();
    let mut prediction_view = None;
    for node in &program.nodes {
        let shape_error = || ScheduleError::Shape {
            value: node.id,
            ty: node.ty.clone(),
        };
        if node.id == prediction
            && matches!(
                node.op,
                TensorOp::Input { .. } | TensorOp::ParamRead { .. } | TensorOp::Scalar(_)
            )
        {
            let own = allocate(&mut buffers, &node.ty).ok_or_else(shape_error)?;
            prediction_view = Some(own);
            works.push(match &node.op {
                TensorOp::Scalar(value) => {
                    let output = elements(&node.ty).ok_or_else(shape_error)?;
                    let kind = KernelKind::Broadcast;
                    let workgroups =
                        spread(node, output.div_ceil(u64::from(kind.workgroup_size())))?;
                    Work {
                        value: node.id,
                        values: vec![node.id],
                        kernels: vec![kernel(
                            kind,
                            node,
                            vec![Operand::Immediate(*value)],
                            node.ty.clone(),
                            None,
                            workgroups,
                        )],
                        reads: Vec::new(),
                        writes: vec![own],
                        partials: None,
                        allocates: None,
                    }
                }
                _ => Work {
                    value: node.id,
                    values: vec![node.id],
                    kernels: Vec::new(),
                    reads: vec![bound_view(&data, &parameters, node)?],
                    writes: vec![own],
                    partials: None,
                    allocates: None,
                },
            });
        }
        match &node.op {
            TensorOp::Input { name } if tokens.contains(&node.id) => {
                // The RNG token of a dropout site, the projection of a sigreg site, the AdamW
                // scalars and moments and the inputs of the masked-row gather, supplied by the host
                // each step, are no data of the train block: each takes its own buffer.
                let view = allocate(&mut buffers, &node.ty).ok_or_else(shape_error)?;
                places.insert(node.id, Place::View(view));
                host_inputs.push(HostInput {
                    value: node.id,
                    name: name.clone(),
                    ty: node.ty.clone(),
                    view,
                });
                continue;
            }
            TensorOp::Input { .. } | TensorOp::ParamRead { .. } => {
                let view = bound_view(&data, &parameters, node)?;
                places.insert(node.id, Place::View(view));
                continue;
            }
            TensorOp::Scalar(value) => {
                places.insert(node.id, Place::Immediate(*value));
                continue;
            }
            _ => {}
        }
        if let Some(&value) = fused.folded.get(&node.id) {
            places.insert(node.id, Place::Immediate(value));
            continue;
        }
        if fused.absorbed.contains(&node.id) {
            continue;
        }
        if let Some(operand) = fused.transposed.get(&node.id) {
            let place = places
                .get(operand)
                .copied()
                .ok_or(ScheduleError::UndefinedOperand {
                    value: node.id,
                    operand: *operand,
                })?;
            places.insert(node.id, place);
            continue;
        }

        let scaled = fused.scaled.get(&node.id).copied();
        let chain = fused.chains.get(&node.id);
        let relu_grad = fused.relu_grad_of.get(&node.id).copied();
        let mut sources = match scaled {
            Some(product) => {
                let mul = program
                    .node(product)
                    .ok_or(ScheduleError::UndefinedOperand {
                        value: node.id,
                        operand: product,
                    })?;
                let mut sources = vec![node.operands[0]];
                sources.extend(&mul.operands);
                sources
            }
            None => match chain {
                Some(chain) => chain.leaves.clone(),
                None => node.operands.clone(),
            },
        };
        sources.extend(relu_grad.map(|(_, pre_activation)| pre_activation));
        let mut operands = Vec::new();
        let mut reads = Vec::new();
        let mut transposed = [false, false];
        if node.op == TensorOp::MatMul {
            for (flag, operand) in transposed.iter_mut().zip(&sources) {
                *flag = fused.transposed.contains_key(operand);
            }
        }
        for &operand in &sources {
            let place = places
                .get(&operand)
                .ok_or(ScheduleError::UndefinedOperand {
                    value: node.id,
                    operand,
                })?;
            match *place {
                Place::View(view) => {
                    let ty = program
                        .node(operand)
                        .ok_or(ScheduleError::UndefinedOperand {
                            value: node.id,
                            operand,
                        })?
                        .ty
                        .clone();
                    operands.push(Operand::Buffer(ty));
                    reads.push(view);
                }
                Place::Immediate(value) => operands.push(Operand::Immediate(value)),
            }
        }

        let update = updates.get(&node.id).copied();
        let relu = fused.relu_of.get(&node.id).copied();
        let mut values = vec![node.id];
        if let Some(chain) = chain {
            values.extend(&chain.nodes[..chain.nodes.len() - 1]);
        }
        let mut writes = Vec::new();
        match (update, relu) {
            (Some(at), _) => {
                writes.push(parameters[at].views[0]);
                places.insert(node.id, Place::View(writes[0]));
                values.extend(scaled);
            }
            (None, Some((activation, pre_activation))) => {
                if pre_activation {
                    let view = allocate(&mut buffers, &node.ty).ok_or_else(shape_error)?;
                    places.insert(node.id, Place::View(view));
                    writes.push(view);
                }
                let view = allocate(&mut buffers, &node.ty).ok_or_else(shape_error)?;
                places.insert(activation, Place::View(view));
                writes.push(view);
                values.push(activation);
            }
            (None, None) => {
                let view = match in_place.get(&node.id).map(|state| places.get(state)) {
                    Some(Some(Place::View(view))) => *view,
                    Some(_) => {
                        return Err(ScheduleError::MissingValue {
                            value: in_place[&node.id],
                        });
                    }
                    None => allocate(&mut buffers, &node.ty).ok_or_else(shape_error)?,
                };
                writes.push(view);
                let computed = match relu_grad {
                    Some((gradient, _)) => {
                        values.push(gradient);
                        gradient
                    }
                    None => node.id,
                };
                places.insert(computed, Place::View(writes[0]));
            }
        }
        let epilogue = match (relu, relu_grad) {
            (Some((_, pre_activation)), _) => Some(Epilogue::Relu {
                writes_pre_activation: pre_activation,
            }),
            (None, Some(_)) => Some(Epilogue::ReluGrad),
            (None, None) => None,
        };

        let output = elements(&node.ty).ok_or_else(shape_error)?;
        let one = |kind: KernelKind, operands: Vec<Operand>| -> Result<Kernel, ScheduleError> {
            let tile = match kind {
                KernelKind::MatMul => {
                    Some(matmul_tile(&operands, transposed, limits).ok_or_else(shape_error)?)
                }
                _ => None,
            };
            let workgroups = match (kind, tile) {
                (KernelKind::LossSum, _) => [1, 1, 1],
                (_, Some((tile, m, n))) => spread(node, tile.workgroups(m, n))?,
                _ => spread(node, output.div_ceil(u64::from(kind.workgroup_size())))?,
            };
            let mut built = kernel(kind, node, operands, node.ty.clone(), None, workgroups);
            if let Some((tile, _, _)) = tile {
                built.workgroup_size = tile.workgroup_size;
                built.epilogue = epilogue;
                built.transposed = transposed;
                built.id = Kernel::module_id(&built.signature());
            }
            Ok(built)
        };
        let (kernels, partials) = match &node.op {
            TensorOp::Add | TensorOp::Sub | TensorOp::Mul | TensorOp::Relu | TensorOp::ReluGrad => {
                let kind = if update.is_some() {
                    KernelKind::SgdUpdate
                } else {
                    KernelKind::Elementwise
                };
                let mut built = one(kind, operands)?;
                if let Some(chain) = chain {
                    built.op = chain.name;
                    built.id = Kernel::module_id(&built.signature());
                }
                (vec![built], None)
            }
            TensorOp::BroadcastScalar { .. } => (vec![one(KernelKind::Broadcast, operands)?], None),
            TensorOp::Transpose => (vec![one(KernelKind::Transpose, operands)?], None),
            TensorOp::MatMul => (vec![one(KernelKind::MatMul, operands)?], None),
            TensorOp::Mse => (vec![one(KernelKind::LossSum, operands)?], None),
            TensorOp::ReduceSum | TensorOp::ReduceMean => {
                let reduced = program
                    .node(node.operands[0])
                    .ok_or(ScheduleError::UndefinedOperand {
                        value: node.id,
                        operand: node.operands[0],
                    })?
                    .ty
                    .clone();
                let count = elements(&reduced).ok_or_else(shape_error)?;
                let groups = count.div_ceil(u64::from(KernelKind::ReduceStage1.workgroup_size()));
                let partial_ty = TensorType::f32(&[groups]);
                let partials = allocate(&mut buffers, &partial_ty).ok_or_else(shape_error)?;
                let stage1 = kernel(
                    KernelKind::ReduceStage1,
                    node,
                    operands,
                    partial_ty.clone(),
                    Some(reduced.clone()),
                    spread(node, groups)?,
                );
                let stage2 = kernel(
                    KernelKind::ReduceStage2,
                    node,
                    vec![Operand::Buffer(partial_ty)],
                    node.ty.clone(),
                    Some(reduced),
                    [1, 1, 1],
                );
                (vec![stage1, stage2], Some(partials))
            }
            TensorOp::Embedding
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
            | TensorOp::Dropout { .. }
            | TensorOp::DropoutGrad { .. }
            | TensorOp::StopGradient
            | TensorOp::SigregColumns
            | TensorOp::SigregColumnsGrad => match built {
                Kernels::None => continue,
                Kernels::Vulkan => {
                    let no_kernel = || ScheduleError::NoKernel {
                        value: node.id,
                        op: node.op.name(),
                    };
                    let (_, invocations) =
                        rowwise_shape(&node.op, &operands, &node.ty).ok_or_else(no_kernel)?;
                    spread(node, invocations.div_ceil(ROWWISE_WORKGROUP))?;
                    let built = rowwise_kernel(&node.op, operands, node.ty.clone())
                        .ok_or_else(no_kernel)?;
                    (vec![built], None)
                }
            },
            TensorOp::Attention { .. }
            | TensorOp::AttentionGradQ { .. }
            | TensorOp::AttentionGradK { .. }
            | TensorOp::AttentionGradV { .. }
            | TensorOp::CrossEntropy
            | TensorOp::CrossEntropyGrad
            | TensorOp::Sqrt
            | TensorOp::Div => match built {
                Kernels::None => continue,
                Kernels::Vulkan => {
                    let no_kernel = || ScheduleError::NoKernel {
                        value: node.id,
                        op: node.op.name(),
                    };
                    let (_, groups, _) = attention_loss_groups(&node.op, &operands, &node.ty)
                        .ok_or_else(no_kernel)?;
                    spread(node, groups)?;
                    let built = attention_loss_kernel(&node.op, operands, node.ty.clone())
                        .ok_or_else(no_kernel)?;
                    (vec![built], None)
                }
            },
            TensorOp::Sigreg { .. } => {
                match built {
                    Kernels::None => continue,
                    // The lowering expands `sigreg` and the verifier refuses it in Tensor IR, so
                    // no plan reaches this: named as the source operator, at the call.
                    Kernels::Vulkan => {
                        return Err(ScheduleError::NoKernel {
                            value: node.id,
                            op: TensorOp::Sigreg { slices: 1 }.name(),
                        });
                    }
                }
            }
            TensorOp::Input { .. } | TensorOp::ParamRead { .. } | TensorOp::Scalar(_) => {
                unreachable!("handled above")
            }
        };
        works.push(Work {
            value: node.id,
            values,
            kernels,
            reads,
            writes,
            partials,
            allocates: None,
        });
    }

    // The softmax statistics of each attention call that shares them: one buffer each, numbered
    // after every other, which the `Attention` work allocates and writes before its result,
    // `AttentionGradQ` writes before its result and the other gradients read after their
    // operands (story:vulkan-attention-softmax-stats-reused).
    if built == Kernels::Vulkan {
        let sites: Vec<[Option<usize>; 4]> = softmax_stats_sites(&program.nodes)
            .into_iter()
            .map(|site| {
                site.map(|value| {
                    value.and_then(|value| works.iter().position(|w| w.value == value))
                })
            })
            .collect();
        for [forward, grad_q, grad_k, grad_v] in sites {
            let (Some(forward), Some(grad_q), Some(grad_k)) = (forward, grad_q, grad_k) else {
                continue;
            };
            let Some(ty) = works[forward].kernels.first().and_then(softmax_stats_type) else {
                continue;
            };
            let Some(view) = allocate(&mut buffers, &ty) else {
                return Err(ScheduleError::Shape {
                    value: works[forward].value,
                    ty,
                });
            };
            for at in [Some(forward), Some(grad_q), Some(grad_k), grad_v]
                .into_iter()
                .flatten()
            {
                let work = &mut works[at];
                let Some(kernel) = work.kernels.first().cloned().and_then(with_softmax_stats)
                else {
                    continue;
                };
                if kernel.writes_softmax_stats() {
                    work.writes.insert(0, view);
                } else {
                    work.reads.push(view);
                }
                work.kernels = vec![kernel];
            }
            works[forward].allocates = Some(view);
        }
    }

    let view_of = |value: ValueId| match places.get(&value) {
        Some(Place::View(view)) => Ok(*view),
        _ => Err(ScheduleError::MissingValue { value }),
    };
    let loss_view = view_of(step.training.loss)?;
    let prediction_view = match prediction_view {
        Some(view) => view,
        None => view_of(prediction)?,
    };

    if let Some((buffer, &bytes)) = buffers
        .iter()
        .enumerate()
        .find(|(_, bytes)| **bytes > MAX_BUFFER_BYTES)
    {
        let buffer = u32::try_from(buffer).expect("allocate numbers buffers in u32");
        return Err(ScheduleError::BufferTooLarge { buffer, bytes });
    }
    for work in &works {
        for kernel in &work.kernels {
            let invocations = kernel
                .workgroups
                .iter()
                .fold(u64::from(kernel.workgroup_size), |acc, &n| {
                    acc.saturating_mul(u64::from(n))
                });
            if invocations > MAX_INVOCATIONS {
                return Err(ScheduleError::TooManyInvocations {
                    value: work.value,
                    invocations,
                });
            }
        }
    }

    let order = step_order(&works);
    let step_plan = record(order.iter().map(|&at| &works[at]));
    let ancestors = ancestors(&program.nodes, prediction);
    let forward_plan = record(
        works
            .iter()
            .filter(|w| w.values.iter().any(|v| ancestors.contains(v))),
    );

    let mut kernels: Vec<Kernel> = Vec::new();
    for kernel in order.iter().flat_map(|&at| &works[at].kernels) {
        if !kernels.iter().any(|k| k.id == kernel.id) {
            kernels.push(kernel.clone());
        }
    }

    let plan = ExecutionPlan {
        buffers,
        parameters,
        data,
        kernels,
        step: step_plan,
        forward: forward_plan,
        loss_view,
        prediction_view,
    };
    Ok((plan, host_inputs))
}

/// The bound view an `Input` or `ParamRead` node reads.
fn bound_view(
    data: &[DataSpec],
    parameters: &[ParameterSpec],
    node: &TensorNode,
) -> Result<BufferView, ScheduleError> {
    match &node.op {
        TensorOp::Input { name } => data
            .iter()
            .find(|d| &d.name == name)
            .map(|d| d.view)
            .ok_or_else(|| ScheduleError::UndeclaredInput {
                value: node.id,
                name: name.clone(),
            }),
        TensorOp::ParamRead { param } => parameters
            .iter()
            .find(|p| p.param == *param)
            .map(|p| p.views[0])
            .ok_or(ScheduleError::MissingValue { value: node.id }),
        _ => Err(ScheduleError::MissingValue { value: node.id }),
    }
}

/// A new buffer holding one value of type `ty`, and its whole view; `None` when the shape is not
/// static or its byte count overflows.
fn allocate(buffers: &mut Vec<u64>, ty: &TensorType) -> Option<BufferView> {
    let count = elements(ty)?;
    let size = count.checked_mul(ty.dtype.size_bytes())?;
    let buffer = u32::try_from(buffers.len()).ok()?;
    buffers.push(size);
    Some(BufferView {
        buffer,
        dtype: ty.dtype,
        element_count: count,
        offset: 0,
        size_bytes: size,
        alignment: DType::size_bytes(ty.dtype),
    })
}

/// The element count of a static shape; 1 for rank 0.
fn elements(ty: &TensorType) -> Option<u64> {
    ty.shape.iter().try_fold(1u64, |acc, dim| match dim {
        Dim::Static(n) => acc.checked_mul(*n),
        Dim::Symbol(_) => None,
    })
}

/// `groups` workgroups over as few dimensions as hold them, each at most
/// [`MAX_WORKGROUPS_PER_DIMENSION`].
fn spread(node: &TensorNode, groups: u64) -> Result<[u32; 3], ScheduleError> {
    spread_groups(groups).ok_or(ScheduleError::TooManyWorkgroups {
        value: node.id,
        groups,
    })
}

/// [`spread`] without a node: `None` past [`MAX_WORKGROUPS_PER_DIMENSION`] in all three.
fn spread_groups(groups: u64) -> Option<[u32; 3]> {
    let max = u64::from(MAX_WORKGROUPS_PER_DIMENSION);
    let x = groups.clamp(1, max);
    let rest = groups.div_ceil(max).max(1);
    let y = rest.min(max);
    let z = rest.div_ceil(max).max(1);
    if z > max {
        return None;
    }
    Some([x as u32, y as u32, z as u32])
}

/// The workgroup size of every row-wise kernel ([`KernelKind::workgroup_size`]).
const ROWWISE_WORKGROUP: u64 = 64;

/// The kernel kind of the row-wise op `op` over `operands` into `result`, and the invocations
/// its dispatch runs: one per output element of a [`KernelKind::PerElement`] kernel, one per
/// output row of a [`KernelKind::PerRow`] kernel and one per output column of a
/// [`KernelKind::PerColumn`] kernel. `None` for another op, an immediate operand, a shape that
/// is not static or a zero dimension, or operands that are not the op's: an `i32` buffer of ids
/// or an `i32[2]` RNG token where the op reads one, `f32` elsewhere.
fn rowwise_shape(
    op: &TensorOp,
    operands: &[Operand],
    result: &TensorType,
) -> Option<(KernelKind, u64)> {
    let mut shapes = Vec::with_capacity(operands.len());
    let mut dtypes = Vec::with_capacity(operands.len());
    for operand in operands {
        let Operand::Buffer(ty) = operand else {
            return None;
        };
        shapes.push(static_dims(ty)?);
        dtypes.push(ty.dtype);
    }
    let out = static_dims(result)?;
    if result.dtype != DType::F32 || out.contains(&0) || shapes.iter().flatten().any(|&d| d == 0) {
        return None;
    }
    let f32s =
        |at: &[usize]| (0..dtypes.len()).all(|i| (dtypes[i] == DType::F32) == at.contains(&i));
    let out_elements = out.iter().product::<u64>();
    let token = [2_u64];
    let views: Vec<&[u64]> = shapes.iter().map(Vec::as_slice).collect();
    let (kind, invocations) = match (op, views.as_slice(), out.as_slice()) {
        (TensorOp::Embedding, [[v, h], [n]], [rn, rh]) if (n, h) == (rn, rh) && *v > 0 => {
            f32s(&[0]).then_some((KernelKind::PerElement, out_elements))?
        }
        (TensorOp::EmbeddingGrad { rows }, [[n], [gn, h]], [v, rh])
            if n == gn && h == rh && *rows == Dim::Static(*v) =>
        {
            f32s(&[1]).then_some((KernelKind::PerRow, *v))?
        }
        (TensorOp::AddRows, [[n, h], [bh]], [rn, rh]) if h == bh && (n, h) == (rn, rh) => {
            f32s(&[0, 1]).then_some((KernelKind::PerElement, out_elements))?
        }
        (TensorOp::SumRows, [[_, h]], [rh]) if h == rh => {
            f32s(&[0]).then_some((KernelKind::PerColumn, *h))?
        }
        (TensorOp::Gelu, [x], _) if *x == out.as_slice() => {
            f32s(&[0]).then_some((KernelKind::PerElement, out_elements))?
        }
        (TensorOp::GeluGrad, [x, g], _) if x == g && *x == out.as_slice() => {
            f32s(&[0, 1]).then_some((KernelKind::PerElement, out_elements))?
        }
        (TensorOp::Dropout { .. }, [x, t], _) if *x == out.as_slice() && *t == token => {
            f32s(&[0]).then_some((KernelKind::PerElement, out_elements))?
        }
        (TensorOp::DropoutGrad { .. }, [t, g], _) if *g == out.as_slice() && *t == token => {
            f32s(&[1]).then_some((KernelKind::PerElement, out_elements))?
        }
        (TensorOp::LayerNorm, [[n, h], [gh], [bh]], [rn, rh])
            if h == gh && h == bh && (n, h) == (rn, rh) =>
        {
            f32s(&[0, 1, 2]).then_some((KernelKind::PerRow, *n))?
        }
        (TensorOp::LayerNormGradX, [[n, h], [gh], g], [rn, rh])
            if h == gh && *g == [*n, *h] && (n, h) == (rn, rh) =>
        {
            f32s(&[0, 1, 2]).then_some((KernelKind::PerRow, *n))?
        }
        (TensorOp::LayerNormGradGamma, [[n, h], g], [rh]) if h == rh && *g == [*n, *h] => {
            f32s(&[0, 1]).then_some((KernelKind::PerColumn, *h))?
        }
        (TensorOp::Softmax, [[n, h]], [rn, rh]) if (n, h) == (rn, rh) => {
            f32s(&[0]).then_some((KernelKind::PerRow, *n))?
        }
        (TensorOp::SoftmaxGrad, [[n, h], g], [rn, rh]) if *g == [*n, *h] && (n, h) == (rn, rh) => {
            f32s(&[0, 1]).then_some((KernelKind::PerRow, *n))?
        }
        (TensorOp::SigregColumns, [[_, m]], [rm]) if m == rm => {
            f32s(&[0]).then_some((KernelKind::PerColumn, *m))?
        }
        (TensorOp::SigregColumnsGrad, [[n, m], [gm]], [rn, rm])
            if m == gm && (n, m) == (rn, rm) =>
        {
            f32s(&[0, 1]).then_some((KernelKind::PerColumn, *m))?
        }
        _ => return None,
    };
    Some((kind, invocations))
}

/// The static dimensions of `ty`.
fn static_dims(ty: &TensorType) -> Option<Vec<u64>> {
    ty.shape
        .iter()
        .map(|dim| match dim {
            Dim::Static(n) => Some(*n),
            Dim::Symbol(_) => None,
        })
        .collect()
}

/// The kernel record of the row-wise op `op` over the buffer operands `operands`, in the op's
/// operand order, into `result`, as [`schedule`] builds it (DESIGN.md § 5.2): its kind
/// ([`KernelKind::PerElement`], [`KernelKind::PerRow`] or [`KernelKind::PerColumn`]),
/// workgroups of 64 over its invocations, and its module id. A `Dropout` or `DropoutGrad`
/// carries its drop probability `p` as a last, immediate operand, so it is in the module id;
/// the run key is not, the pipeline supplies it. `None` where [`rowwise_shape`] is: another
/// op, or operands that are not the op's.
pub fn rowwise_kernel(
    op: &TensorOp,
    mut operands: Vec<Operand>,
    result: TensorType,
) -> Option<Kernel> {
    let (kind, invocations) = rowwise_shape(op, &operands, &result)?;
    let workgroups = spread_groups(invocations.div_ceil(ROWWISE_WORKGROUP))?;
    if let Some(p) = op.dropout_rate() {
        operands.push(Operand::Immediate(p));
    }
    let mut kernel = Kernel {
        id: crate::ir::kernel::KernelModuleId([0; 16]),
        kind,
        op: op.name(),
        dtype: result.dtype,
        operands,
        result,
        reduces: None,
        epilogue: None,
        transposed: [false, false],
        softmax_stats: None,
        workgroup_size: kind.workgroup_size(),
        workgroups,
    };
    kernel.id = Kernel::module_id(&kernel.signature());
    Some(kernel)
}

/// The most keys of one sequence an attention kernel reads: a workgroup stages the weights of
/// its rows over them in workgroup memory ([`attention_tile`]), and a gradient of `k` or `v`
/// computes them for every query of its sequence (coordinator decision D73).
pub const MAX_ATTENTION_KEYS: u64 = 512;

/// The workgroup size of an attention kernel (story:vulkan-attention-workgroup-per-head).
pub const ATTENTION_WORKGROUP: u32 = 128;

/// The most rows of one sequence's head an attention workgroup computes.
pub const MAX_ATTENTION_TILE: u64 = 32;

/// The `f32` values of workgroup memory an attention workgroup of `tile` rows over `keys` keys
/// may stage: two `[tile, keys]` arrays of scores and three statistics per row, or one array
/// and three statistics per key, each with one value per key to spare.
pub fn attention_workgroup_values(tile: u64, keys: u64) -> u64 {
    let rows = 2 * tile * keys + 3 * tile + keys;
    let columns = tile * keys + 4 * keys;
    rows.max(columns)
}

/// The rows of one sequence's head each workgroup of an attention kernel over `keys` keys per
/// sequence computes: the largest power of two of at most [`MAX_ATTENTION_TILE`] and `keys` whose
/// [`attention_workgroup_values`] fit the workgroup memory of every device
/// ([`MIN_SHARED_MEMORY_BYTES`]); 1 past [`MAX_ATTENTION_KEYS`].
pub fn attention_tile(keys: u64) -> u64 {
    let budget = u64::from(MIN_SHARED_MEMORY_BYTES) / 4;
    let mut tile = MAX_ATTENTION_TILE;
    while tile > 1 && (tile > keys || attention_workgroup_values(tile, keys) > budget) {
        tile /= 2;
    }
    tile
}

/// The workgroups of an attention kernel over `sequences` sequences of `keys` keys and `heads`
/// heads: one per sequence, head and tile of [`attention_tile`] rows, the tiles of a head
/// consecutive, the heads of a sequence consecutive.
pub fn attention_workgroups(sequences: u64, keys: u64, heads: u64) -> Option<u64> {
    sequences
        .checked_mul(heads)?
        .checked_mul(keys.div_ceil(attention_tile(keys)))
}

/// The kernel kind of `Attention`, its three gradients, `CrossEntropy`, `CrossEntropyGrad`,
/// `Sqrt` or `Div` over `operands` into `result`, and the invocations its dispatch runs: one per
/// row and head of an attention kernel ([`KernelKind::PerRow`]), one workgroup for the
/// cross-entropy loss ([`KernelKind::LossSum`], 1), one per row of its gradient
/// ([`KernelKind::PerRow`]), one per output element of `Sqrt` and `Div`
/// ([`KernelKind::Elementwise`]). `None` for another op, a shape that is not static or has a
/// zero dimension, or operands that are not the op's:
///
/// - attention: `q`, `k`, `v` (and the gradient `g` of a gradient op) `f32` buffers `[N, H]`, a
///   key mask `f32` `[B, S]` with `B·S = N` and `S` at most [`MAX_ATTENTION_KEYS`], and `heads`
///   dividing `H`;
/// - cross-entropy: logits `f32` `[N, V]`, labels `i32` `[N]`, weights `f32` `[N]`, a rank-0
///   result, and for the gradient a seed that is an immediate or a rank-0 `f32` buffer, into
///   `[N, V]`;
/// - `Sqrt` of one operand and `Div` of two, each an immediate, a rank-0 `f32` buffer or an `f32`
///   buffer of the result's shape.
fn attention_loss_shape(
    op: &TensorOp,
    operands: &[Operand],
    result: &TensorType,
) -> Option<(KernelKind, u64)> {
    let out = static_dims(result)?;
    if result.dtype != DType::F32 || out.contains(&0) {
        return None;
    }
    let buffer = |at: usize, dtype: DType| match operands.get(at)? {
        Operand::Buffer(ty) if ty.dtype == dtype => {
            let dims = static_dims(ty)?;
            (!dims.contains(&0)).then_some(dims)
        }
        _ => None,
    };
    let f32s = |at: usize| buffer(at, DType::F32);
    match op {
        TensorOp::Attention { heads }
        | TensorOp::AttentionGradQ { heads }
        | TensorOp::AttentionGradK { heads }
        | TensorOp::AttentionGradV { heads } => {
            let arity = if matches!(op, TensorOp::Attention { .. }) {
                4
            } else {
                5
            };
            if operands.len() != arity || *heads == 0 {
                return None;
            }
            let rows = f32s(0)?;
            let [n, h] = rows[..] else {
                return None;
            };
            let same = [1, 2].into_iter().chain((arity == 5).then_some(4));
            for at in same {
                if f32s(at)? != rows {
                    return None;
                }
            }
            let [b, s] = f32s(3)?[..] else {
                return None;
            };
            let fits = out == rows
                && h % heads == 0
                && b.checked_mul(s) == Some(n)
                && s <= MAX_ATTENTION_KEYS
                && *heads <= 1 << 24;
            fits.then_some((KernelKind::PerRow, n.checked_mul(*heads)?))
        }
        TensorOp::CrossEntropy | TensorOp::CrossEntropyGrad => {
            let logits = f32s(0)?;
            let [n, v] = logits[..] else {
                return None;
            };
            if buffer(1, DType::I32)? != [n] || f32s(2)? != [n] {
                return None;
            }
            if *op == TensorOp::CrossEntropy {
                return (operands.len() == 3 && out.is_empty()).then_some((KernelKind::LossSum, 1));
            }
            let seed = match operands.get(3)? {
                Operand::Immediate(_) => true,
                Operand::Buffer(ty) => ty.dtype == DType::F32 && ty.rank() == 0,
            };
            (operands.len() == 4 && seed && out == [n, v]).then_some((KernelKind::PerRow, n))
        }
        TensorOp::Sqrt | TensorOp::Div => {
            let arity = if *op == TensorOp::Sqrt { 1 } else { 2 };
            if operands.len() != arity {
                return None;
            }
            for operand in operands {
                if let Operand::Buffer(ty) = operand {
                    let dims = static_dims(ty)?;
                    if ty.dtype != DType::F32 || (!dims.is_empty() && dims != out) {
                        return None;
                    }
                }
            }
            Some((KernelKind::Elementwise, out.iter().product()))
        }
        _ => None,
    }
}

/// The kernel record of `Attention`, `AttentionGradQ`, `AttentionGradK`, `AttentionGradV`,
/// `CrossEntropy`, `CrossEntropyGrad`, `Sqrt` or `Div` over `operands`, in the op's operand
/// order, into `result`, as [`schedule`] builds it (DESIGN.md § 5.2, coordinator decision D73):
/// its kind, workgroups and workgroup size by [`attention_loss_groups`], and its module id. An
/// attention kernel carries its head count as a last, immediate operand ([`Kernel::heads`]), so
/// it is in the module id. `None` for another op or for operands that are not the op's.
pub fn attention_loss_kernel(
    op: &TensorOp,
    mut operands: Vec<Operand>,
    result: TensorType,
) -> Option<Kernel> {
    let (kind, groups, workgroup_size) = attention_loss_groups(op, &operands, &result)?;
    let workgroups = spread_groups(groups)?;
    if let Some(heads) = op.attention_heads() {
        operands.push(Operand::Immediate(heads as f32));
    }
    let mut kernel = Kernel {
        id: crate::ir::kernel::KernelModuleId([0; 16]),
        kind,
        op: op.name(),
        dtype: result.dtype,
        operands,
        result,
        reduces: None,
        epilogue: None,
        transposed: [false, false],
        softmax_stats: None,
        workgroup_size,
        workgroups,
    };
    kernel.id = Kernel::module_id(&kernel.signature());
    Some(kernel)
}

/// The kind of the kernel of [`attention_loss_shape`], its workgroups and their size: for an
/// attention kernel [`attention_workgroups`] of [`ATTENTION_WORKGROUP`], for the cross-entropy
/// loss one of 64, for any other workgroups of 64 over its invocations.
fn attention_loss_groups(
    op: &TensorOp,
    operands: &[Operand],
    result: &TensorType,
) -> Option<(KernelKind, u64, u32)> {
    let (kind, invocations) = attention_loss_shape(op, operands, result)?;
    let size = kind.workgroup_size();
    match (kind, op.attention_heads()) {
        (_, Some(heads)) => {
            let Some(Operand::Buffer(mask)) = operands.get(3) else {
                return None;
            };
            let [b, s] = static_dims(mask)?[..] else {
                return None;
            };
            Some((
                kind,
                attention_workgroups(b, s, heads)?,
                ATTENTION_WORKGROUP,
            ))
        }
        (KernelKind::LossSum, None) => Some((kind, 1, size)),
        _ => Some((kind, invocations.div_ceil(u64::from(size)), size)),
    }
}

/// The type of the softmax statistics an attention kernel of [`attention_loss_kernel`] shares
/// with the other kernels of its call ([`Kernel::softmax_stats`]): `[rows · heads, 3]` `f32`,
/// one maximum, sum and row dot per query head. `None` for another kernel or a shape that is not
/// static.
pub fn softmax_stats_type(kernel: &Kernel) -> Option<TensorType> {
    let heads = kernel.heads()?;
    let Some(Operand::Buffer(q)) = kernel.operands.first() else {
        return None;
    };
    let [Dim::Static(rows), _] = q.shape[..] else {
        return None;
    };
    Some(TensorType::f32(&[
        rows.checked_mul(heads)?,
        SOFTMAX_STATS_PER_QUERY,
    ]))
}

/// `kernel`, an attention kernel of [`attention_loss_kernel`], made to share its softmax
/// statistics ([`Kernel::softmax_stats`] of [`softmax_stats_type`]) with its new module id:
/// an `Attention` kernel writes them, `AttentionGradQ` reads them and writes the row dots, and
/// `AttentionGradK` and `AttentionGradV` read them. `None` for another kernel.
pub fn with_softmax_stats(mut kernel: Kernel) -> Option<Kernel> {
    kernel.softmax_stats = Some(softmax_stats_type(&kernel)?);
    kernel.id = Kernel::module_id(&kernel.signature());
    Some(kernel)
}

/// The attention calls of `nodes` whose kernels share softmax statistics
/// (story:vulkan-attention-softmax-stats-reused): by the value of each call's `Attention`, its
/// `AttentionGradQ`, `AttentionGradK` and, when there is one, `AttentionGradV`. A call shares
/// them when its `q`, `k`, `v`, key mask and head count are those of exactly one `Attention`, one
/// `AttentionGradQ`, one `AttentionGradK` and at most one `AttentionGradV`, the gradients read
/// one `G`, and the forward comes before `AttentionGradQ` and that before every other gradient;
/// the kernels of any other call recompute the statistics where they use them.
fn softmax_stats_sites(nodes: &[TensorNode]) -> Vec<[Option<ValueId>; 4]> {
    type Call = (Vec<ValueId>, u64);
    let mut calls: BTreeMap<Call, [Vec<(usize, &TensorNode)>; 4]> = BTreeMap::new();
    for (at, node) in nodes.iter().enumerate() {
        let role = match node.op {
            TensorOp::Attention { .. } => 0,
            TensorOp::AttentionGradQ { .. } => 1,
            TensorOp::AttentionGradK { .. } => 2,
            TensorOp::AttentionGradV { .. } => 3,
            _ => continue,
        };
        let (Some(heads), Some(operands)) = (node.op.attention_heads(), node.operands.get(..4))
        else {
            continue;
        };
        calls.entry((operands.to_vec(), heads)).or_default()[role].push((at, node));
    }
    calls
        .into_values()
        .filter_map(|[forward, q, k, v]| {
            let ([forward], [q], [k]) = (&forward[..], &q[..], &k[..]) else {
                return None;
            };
            let v = match &v[..] {
                [] => None,
                [v] => Some(*v),
                _ => return None,
            };
            let grads = [Some(*q), Some(*k), v];
            let gradient = |(_, node): &(usize, &TensorNode)| node.operands.get(4).copied();
            let one_g = grads.iter().flatten().all(|g| gradient(g) == gradient(q));
            let ordered = forward.0 < q.0 && k.0 > q.0 && v.is_none_or(|v| v.0 > q.0);
            (one_g && gradient(q).is_some() && ordered).then(|| {
                [
                    Some(forward.1.id),
                    Some(q.1.id),
                    Some(k.1.id),
                    v.map(|(_, node)| node.id),
                ]
            })
        })
        .collect()
}

/// The tile of a matmul over `operands` `[m, k]` and `[k, n]`, reading the operands marked in
/// `transposed`, within the `limits` of a device or, without them, for every Vulkan device;
/// with `m` and `n`.
fn matmul_tile(
    operands: &[Operand],
    transposed: [bool; 2],
    limits: Option<MatMulLimits>,
) -> Option<(MatMulTile, u64, u64)> {
    let (m, k, n) = Kernel::matmul_shape(operands)?;
    let tile = match limits {
        Some(limits) => MatMulTile::for_limits(m, k, n, transposed, limits),
        None => MatMulTile::of(m, k, n),
    };
    Some((tile, m, n))
}

fn kernel(
    kind: KernelKind,
    node: &TensorNode,
    operands: Vec<Operand>,
    result: TensorType,
    reduces: Option<TensorType>,
    workgroups: [u32; 3],
) -> Kernel {
    let mut kernel = Kernel {
        id: crate::ir::kernel::KernelModuleId([0; 16]),
        kind,
        op: node.op.name(),
        dtype: node.ty.dtype,
        operands,
        result,
        reduces,
        epilogue: None,
        transposed: [false, false],
        softmax_stats: None,
        workgroup_size: kind.workgroup_size(),
        workgroups,
    };
    kernel.id = Kernel::module_id(&kernel.signature());
    kernel
}

/// `prediction` and every value it is computed from.
fn ancestors(nodes: &[TensorNode], prediction: ValueId) -> BTreeSet<ValueId> {
    let mut reached = BTreeSet::from([prediction]);
    for node in nodes.iter().rev() {
        if reached.contains(&node.id) {
            reached.extend(node.operands.iter().copied());
        }
    }
    reached
}

/// The order the step plan records `works` in: their order, except that the `mse` loss sum
/// moves to right after the matmul with the most FLOP, the first of equals, among the works
/// after it up to the first that writes a buffer the loss sum reads or reads or writes the one
/// it writes. No op of the step reads the loss, so the one-workgroup sum then shares that
/// matmul's barrier interval instead of holding one of its own (DESIGN.md § 3.1). Without such
/// a matmul the order is unchanged.
fn step_order(works: &[Work]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..works.len()).collect();
    let Some(loss) = works
        .iter()
        .position(|w| matches!(&w.kernels[..], [k] if k.kind == KernelKind::LossSum))
    else {
        return order;
    };
    let buffers = |views: &[BufferView]| views.iter().map(|v| v.buffer).collect::<BTreeSet<u32>>();
    let (reads, writes) = (buffers(&works[loss].reads), buffers(&works[loss].writes));
    let mut beside: Option<(u64, usize)> = None;
    for (at, work) in works.iter().enumerate().skip(loss + 1) {
        let (r, w) = (buffers(&work.reads), buffers(&work.writes));
        let partials = work.partials.map(|p| p.buffer);
        if !w.is_disjoint(&reads)
            || !w.is_disjoint(&writes)
            || !r.is_disjoint(&writes)
            || partials.is_some_and(|p| reads.contains(&p) || writes.contains(&p))
        {
            break;
        }
        if let [kernel] = &work.kernels[..]
            && kernel.kind == KernelKind::MatMul
            && let Some(flop) = kernel.flop()
            && beside.is_none_or(|(most, _)| flop > most)
        {
            beside = Some((flop, at));
        }
    }
    if let Some((_, at)) = beside {
        let moved = order.remove(loss);
        order.insert(at, moved);
    }
    order
}

/// The plan steps of `works` in order, with the barriers the schedule needs.
fn record<'a>(works: impl Iterator<Item = &'a Work>) -> Vec<PlanStep> {
    let mut recorder = Recorder::default();
    for work in works {
        match (&work.kernels[..], work.partials) {
            ([], None) => recorder.copy(work.value, work.reads.clone(), work.writes.clone()),
            ([single], None) => {
                let allocate = work.allocates.map(|stats| {
                    recorder.sync_for(&work.reads, &work.writes);
                    recorder.push(PlanStep {
                        op: PlanOp::Allocate,
                        kernel_module_id: None,
                        workgroups: None,
                        reads: Vec::new(),
                        writes: vec![stats],
                        depends_on: Vec::new(),
                        value: Some(work.value),
                    })
                });
                recorder.dispatch(
                    single,
                    work.value,
                    work.reads.clone(),
                    work.writes.clone(),
                    allocate,
                );
            }
            ([stage1, stage2], Some(partials)) => {
                recorder.sync_for(&work.reads, &[partials]);
                let allocate = recorder.push(PlanStep {
                    op: PlanOp::Allocate,
                    kernel_module_id: None,
                    workgroups: None,
                    reads: Vec::new(),
                    writes: vec![partials],
                    depends_on: Vec::new(),
                    value: Some(work.value),
                });
                recorder.dispatch(
                    stage1,
                    work.value,
                    work.reads.clone(),
                    vec![partials],
                    Some(allocate),
                );
                recorder.dispatch(
                    stage2,
                    work.value,
                    vec![partials],
                    work.writes.clone(),
                    None,
                );
            }
            _ => unreachable!("a work item is a copy, one kernel, or two stages with partials"),
        }
    }
    if !recorder.steps.is_empty() {
        recorder.push(PlanStep {
            op: PlanOp::Barrier,
            kernel_module_id: None,
            workgroups: None,
            reads: Vec::new(),
            writes: Vec::new(),
            depends_on: Vec::new(),
            value: None,
        });
    }
    recorder.steps
}

#[derive(Default)]
struct Recorder {
    steps: Vec<PlanStep>,
    barrier: Option<usize>,
    since: Vec<usize>,
    read: BTreeSet<u32>,
    written: BTreeSet<u32>,
}

impl Recorder {
    fn push(&mut self, step: PlanStep) -> usize {
        self.steps.push(step);
        self.steps.len() - 1
    }

    /// Records a `Barrier` if `reads` or `writes` conflict with an access since the last one.
    fn sync_for(&mut self, reads: &[BufferView], writes: &[BufferView]) {
        let hazard = reads.iter().any(|v| self.written.contains(&v.buffer))
            || writes
                .iter()
                .any(|v| self.written.contains(&v.buffer) || self.read.contains(&v.buffer));
        if !hazard {
            return;
        }
        let depends_on = self
            .barrier
            .into_iter()
            .chain(self.since.drain(..))
            .collect();
        let at = self.push(PlanStep {
            op: PlanOp::Barrier,
            kernel_module_id: None,
            workgroups: None,
            reads: Vec::new(),
            writes: Vec::new(),
            depends_on,
            value: None,
        });
        self.barrier = Some(at);
        self.read.clear();
        self.written.clear();
    }

    fn copy(&mut self, value: ValueId, reads: Vec<BufferView>, writes: Vec<BufferView>) {
        self.sync_for(&reads, &writes);
        let depends_on: Vec<usize> = self.barrier.into_iter().collect();
        self.read.extend(reads.iter().map(|v| v.buffer));
        self.written.extend(writes.iter().map(|v| v.buffer));
        let at = self.push(PlanStep {
            op: PlanOp::Copy,
            kernel_module_id: None,
            workgroups: None,
            reads,
            writes,
            depends_on,
            value: Some(value),
        });
        self.since.push(at);
    }

    fn dispatch(
        &mut self,
        kernel: &Kernel,
        value: ValueId,
        reads: Vec<BufferView>,
        writes: Vec<BufferView>,
        allocate: Option<usize>,
    ) {
        self.sync_for(&reads, &writes);
        let mut depends_on: Vec<usize> = self.barrier.into_iter().chain(allocate).collect();
        depends_on.sort_unstable();
        self.read.extend(reads.iter().map(|v| v.buffer));
        self.written.extend(writes.iter().map(|v| v.buffer));
        let at = self.push(PlanStep {
            op: PlanOp::Dispatch,
            kernel_module_id: Some(kernel.id),
            workgroups: Some(kernel.workgroups),
            reads,
            writes,
            depends_on,
            value: Some(value),
        });
        self.since.push(at);
    }
}
