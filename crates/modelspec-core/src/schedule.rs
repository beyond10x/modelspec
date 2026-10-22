//! The fixed safe schedule of M1 (DESIGN.md § 5.2, § 5.3): a verified training step to Kernel IR
//! and its execution plans.
//!
//! Ops run in Training IR order. Each elementwise, broadcast, transpose, matmul and SGD op is one
//! Dispatch, one invocation per output element; the `mse` loss is one Dispatch of one workgroup
//! (DESIGN.md § 5.4); every other reduce is four consecutive steps: an `Allocate` of its partials
//! buffer, a stage-1 Dispatch that writes one partial per workgroup into it, a `Barrier`, and a
//! stage-2 Dispatch of one workgroup that reads the partials. `Input`, `ParamRead` and `Scalar`
//! dispatch nothing: inputs and parameters are bound views, and a scalar is compiled into the
//! kernels that read it.
//!
//! Every value has its own buffer (no reuse), and each SGD update writes its parameter's one
//! view in place. A `Barrier` precedes any Dispatch that reads a buffer an earlier Dispatch wrote
//! since the last barrier, or writes one an earlier Dispatch read or wrote since then; it depends
//! on the previous barrier and on every Dispatch since, so a step after it is ordered after
//! everything before it. Each plan ends with its one fence, a `Barrier` with no dependencies, so
//! a runner may record it back to back. A buffer over [`MAX_BUFFER_BYTES`] or a dispatch over
//! [`MAX_INVOCATIONS`] is refused as a target limit. The prediction has its own view: a forward that returns an
//! input or a parameter copies it there with a `Copy` step, and one that returns a literal fills
//! it with a Broadcast dispatch. The workgroup sizes are the constants of [`KernelKind`]; nothing
//! here reads a device.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::artifact::{DataRole, DataSpec, ParameterSpec, canonical_parameter_id};
use crate::ir::kernel::{BufferView, ExecutionPlan, Kernel, KernelKind, Operand, PlanOp, PlanStep};
use crate::ir::tensor::{DType, Dim, TensorNode, TensorOp, TensorType, ValueId};
use crate::ir::training::{TrainConfig, TrainingStep};

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

/// One computing op with its kernels and views.
struct Work {
    value: ValueId,
    /// One kernel, or stage 1 and stage 2 of a reduce.
    kernels: Vec<Kernel>,
    reads: Vec<BufferView>,
    writes: BufferView,
    /// The partials buffer of a two-stage reduce.
    partials: Option<BufferView>,
}

/// Schedules `step`, whose forward prediction is `prediction`, with the data declarations of
/// `config`. The result has one view per parameter and passes
/// [`crate::ir::verify::verify_plan`].
pub fn schedule(
    step: &TrainingStep,
    prediction: ValueId,
    config: &TrainConfig,
) -> Result<ExecutionPlan, ScheduleError> {
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
                        kernels: vec![kernel(
                            kind,
                            node,
                            vec![Operand::Immediate(*value)],
                            node.ty.clone(),
                            None,
                            workgroups,
                        )],
                        reads: Vec::new(),
                        writes: own,
                        partials: None,
                    }
                }
                _ => Work {
                    value: node.id,
                    kernels: Vec::new(),
                    reads: vec![bound_view(&data, &parameters, node)?],
                    writes: own,
                    partials: None,
                },
            });
        }
        match &node.op {
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

        let mut operands = Vec::new();
        let mut reads = Vec::new();
        for &operand in &node.operands {
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
        let writes = match update {
            Some(at) => parameters[at].views[0],
            None => allocate(&mut buffers, &node.ty).ok_or_else(shape_error)?,
        };
        places.insert(node.id, Place::View(writes));

        let output = elements(&node.ty).ok_or_else(shape_error)?;
        let one = |kind: KernelKind, operands: Vec<Operand>| -> Result<Kernel, ScheduleError> {
            let workgroups = match kind {
                KernelKind::LossSum => [1, 1, 1],
                _ => spread(node, output.div_ceil(u64::from(kind.workgroup_size())))?,
            };
            Ok(kernel(
                kind,
                node,
                operands,
                node.ty.clone(),
                None,
                workgroups,
            ))
        };
        let (kernels, partials) = match &node.op {
            TensorOp::Add | TensorOp::Sub | TensorOp::Mul | TensorOp::Relu | TensorOp::ReluGrad => {
                let kind = if update.is_some() {
                    KernelKind::SgdUpdate
                } else {
                    KernelKind::Elementwise
                };
                (vec![one(kind, operands)?], None)
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
            TensorOp::Input { .. } | TensorOp::ParamRead { .. } | TensorOp::Scalar(_) => {
                unreachable!("handled above")
            }
        };
        works.push(Work {
            value: node.id,
            kernels,
            reads,
            writes,
            partials,
        });
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

    let step_plan = record(works.iter());
    let ancestors = ancestors(&program.nodes, prediction);
    let forward_plan = record(works.iter().filter(|w| ancestors.contains(&w.value)));

    let mut kernels: Vec<Kernel> = Vec::new();
    for kernel in works.iter().flat_map(|w| &w.kernels) {
        if !kernels.iter().any(|k| k.id == kernel.id) {
            kernels.push(kernel.clone());
        }
    }

    Ok(ExecutionPlan {
        buffers,
        parameters,
        data,
        kernels,
        step: step_plan,
        forward: forward_plan,
        loss_view,
        prediction_view,
    })
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
    let max = u64::from(MAX_WORKGROUPS_PER_DIMENSION);
    let too_many = ScheduleError::TooManyWorkgroups {
        value: node.id,
        groups,
    };
    let x = groups.clamp(1, max);
    let rest = groups.div_ceil(max).max(1);
    let y = rest.min(max);
    let z = rest.div_ceil(max).max(1);
    if z > max {
        return Err(too_many);
    }
    Ok([x as u32, y as u32, z as u32])
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

/// The plan steps of `works` in order, with the barriers the schedule needs.
fn record<'a>(works: impl Iterator<Item = &'a Work>) -> Vec<PlanStep> {
    let mut recorder = Recorder::default();
    for work in works {
        match (&work.kernels[..], work.partials) {
            ([], None) => recorder.copy(work.value, work.reads.clone(), work.writes),
            ([single], None) => {
                recorder.dispatch(single, work.value, work.reads.clone(), work.writes, None);
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
                    partials,
                    Some(allocate),
                );
                recorder.dispatch(stage2, work.value, vec![partials], work.writes, None);
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

    fn copy(&mut self, value: ValueId, reads: Vec<BufferView>, writes: BufferView) {
        self.sync_for(&reads, &[writes]);
        let depends_on: Vec<usize> = self.barrier.into_iter().collect();
        self.read.extend(reads.iter().map(|v| v.buffer));
        self.written.insert(writes.buffer);
        let at = self.push(PlanStep {
            op: PlanOp::Copy,
            kernel_module_id: None,
            workgroups: None,
            reads,
            writes: vec![writes],
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
        writes: BufferView,
        allocate: Option<usize>,
    ) {
        self.sync_for(&reads, &[writes]);
        let mut depends_on: Vec<usize> = self.barrier.into_iter().chain(allocate).collect();
        depends_on.sort_unstable();
        self.read.extend(reads.iter().map(|v| v.buffer));
        self.written.insert(writes.buffer);
        let at = self.push(PlanStep {
            op: PlanOp::Dispatch,
            kernel_module_id: Some(kernel.id),
            workgroups: Some(kernel.workgroups),
            reads,
            writes: vec![writes],
            depends_on,
            value: Some(value),
        });
        self.since.push(at);
    }
}
