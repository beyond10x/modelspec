//! Kernel IR and the execution plan of a training step (DESIGN.md § 3.1, § 5.2, § 7.1).
//!
//! An [`ExecutionPlan`] is what [`crate::schedule::schedule`] makes of a verified training step:
//! the buffer table, the views a runtime binds (one per input and target, one or two per
//! parameter, the loss and the prediction), the [`Kernel`]s, and two lists of [`PlanStep`]s: the
//! step plan, one training step that a runner may record many times, and the forward plan, the
//! prediction alone. [`crate::ir::verify::verify_plan`] checks it; [`crate::ir::print::print_kernel`]
//! prints it.
//!
//! The workgroup size of a kernel is the constant of its [`KernelKind`], never read from a device
//! (DESIGN.md § 5.3, the fixed safe schedule of M1).

use std::fmt;

use crate::artifact::{DataSpec, ParameterSpec};
use crate::ir::tensor::{DType, TensorType, ValueId};

/// The identity of a kernel module, a UUID printed in its hyphenated form. It is derived from the
/// content of the kernel ([`Kernel::module_id`]), so two compiles of one file give equal ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelModuleId(pub [u8; 16]);

impl fmt::Display for KernelModuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, byte) in self.0.iter().enumerate() {
            if matches!(i, 4 | 6 | 8 | 10) {
                f.write_str("-")?;
            }
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// A typed range of one plan buffer (DESIGN.md § 7.1), printed `b3[f32 x 8, bytes 0..32, align 4]`.
///
/// `size_bytes` is `element_count` times the dtype's size and `offset` a multiple of
/// `alignment`; the verifier checks both and that the range lies inside its buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferView {
    /// The plan buffer, an index into [`ExecutionPlan::buffers`].
    pub buffer: u32,
    pub dtype: DType,
    pub element_count: u64,
    pub offset: u64,
    pub size_bytes: u64,
    pub alignment: u64,
}

impl BufferView {
    /// The first byte past the view, or `None` when it overflows.
    pub fn end(&self) -> Option<u64> {
        self.offset.checked_add(self.size_bytes)
    }
}

impl fmt::Display for BufferView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "b{}[{} x {}, bytes {}..{}, align {}]",
            self.buffer,
            self.dtype,
            self.element_count,
            self.offset,
            self.offset.saturating_add(self.size_bytes),
            self.alignment
        )
    }
}

/// The command kinds of the execution DAG (DESIGN.md § 3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlanOp {
    /// Makes a scratch buffer, the view it writes. A run-setup step: a runner executes every
    /// `Allocate` of a plan once, before its first recording, and records only the other steps.
    Allocate,
    Copy,
    Dispatch,
    /// Makes every earlier write visible to, and every earlier read finish before, the steps that
    /// depend on it.
    Barrier,
    /// A CPU-visible completion point. No plan the schedule makes has one: a runner places it.
    Complete,
}

impl fmt::Display for PlanOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            PlanOp::Allocate => "allocate",
            PlanOp::Copy => "copy",
            PlanOp::Dispatch => "dispatch",
            PlanOp::Barrier => "barrier",
            PlanOp::Complete => "complete",
        })
    }
}

/// One command of a plan.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanStep {
    pub op: PlanOp,
    /// The module a Dispatch runs; present exactly on a Dispatch.
    pub kernel_module_id: Option<KernelModuleId>,
    /// The workgroup count per dimension of a Dispatch; present exactly on a Dispatch.
    pub workgroups: Option<[u32; 3]>,
    /// The views the step reads, in the kernel's binding order.
    pub reads: Vec<BufferView>,
    /// The views the step writes; for an `Allocate` the scratch view it makes.
    pub writes: Vec<BufferView>,
    /// Positions of earlier steps of the same plan this one waits on.
    pub depends_on: Vec<usize>,
    /// The Training IR value the step computes or allocates scratch for, if any.
    pub value: Option<ValueId>,
}

/// What a kernel computes. Its workgroup size is a constant of the kind
/// ([`KernelKind::workgroup_size`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KernelKind {
    /// `Add`, `Sub`, `Mul`, `Relu` or `ReluGrad`, one invocation per output element.
    Elementwise,
    /// `BroadcastScalar`, one invocation per output element.
    Broadcast,
    /// The first stage of `ReduceSum` or `ReduceMean`: one partial per workgroup.
    ReduceStage1,
    /// The second stage: one workgroup reads every partial and writes the result.
    ReduceStage2,
    /// The `mse` loss: one workgroup stages its terms in workgroup memory and invocation 0 sums
    /// them in row-major order (DESIGN.md § 5.2, § 5.4).
    LossSum,
    Transpose,
    /// One invocation per output element, with an optional [`Epilogue`] on its rounded sum.
    MatMul,
    /// The SGD update, written in place into the parameter's view: with operands `(p, lr, g)`
    /// (or `(p, g, lr)`) it is `p - (lr * g)`, the product rounded before the subtraction and
    /// neither contracted; with operands `(p, d)` it is `p - d`.
    SgdUpdate,
}

impl KernelKind {
    /// The workgroup size of the kind in the fixed safe schedule: 64 invocations for every kind.
    pub const fn workgroup_size(self) -> u32 {
        64
    }
}

impl fmt::Display for KernelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            KernelKind::Elementwise => "elementwise",
            KernelKind::Broadcast => "broadcast",
            KernelKind::ReduceStage1 => "reduce_stage_1",
            KernelKind::ReduceStage2 => "reduce_stage_2",
            KernelKind::LossSum => "loss_sum",
            KernelKind::Transpose => "transpose",
            KernelKind::MatMul => "matmul",
            KernelKind::SgdUpdate => "sgd_update",
        })
    }
}

/// One operand of a kernel: a bound buffer of a type, or a constant compiled into the kernel.
#[derive(Clone, Debug, PartialEq)]
pub enum Operand {
    Buffer(TensorType),
    Immediate(f32),
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Buffer(ty) => write!(f, "{ty}"),
            Operand::Immediate(value) => write!(f, "{value:?}"),
        }
    }
}

/// An elementwise op a matmul kernel applies to each rounded output element before it stores it
/// (DESIGN.md § 3.1): the same instructions the op's own kernel runs on the value it loads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Epilogue {
    /// `relu`, a compare and a select. With `writes_pre_activation` the kernel also writes the
    /// sum before it, for a reader other than the `relu`; then the sum's binding comes before the
    /// activation's.
    Relu { writes_pre_activation: bool },
}

impl fmt::Display for Epilogue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Epilogue::Relu {
                writes_pre_activation: false,
            } => f.write_str("epilogue Relu"),
            Epilogue::Relu {
                writes_pre_activation: true,
            } => f.write_str("epilogue Relu writes pre-activation"),
        }
    }
}

/// One kernel of Kernel IR. Its bindings are its buffer operands in order, each read, then what
/// it writes: the pre-activation of an [`Epilogue`] that keeps it, then its result.
#[derive(Clone, Debug, PartialEq)]
pub struct Kernel {
    /// [`Kernel::module_id`] of the other fields.
    pub id: KernelModuleId,
    pub kind: KernelKind,
    /// The Tensor IR op the kernel implements, by name.
    pub op: &'static str,
    pub dtype: DType,
    pub operands: Vec<Operand>,
    pub result: TensorType,
    /// The tensor a reduce stage reduces, so that both stages name its element count.
    pub reduces: Option<TensorType>,
    /// What a matmul kernel applies to each output element before storing it; `None` on every
    /// other kind.
    pub epilogue: Option<Epilogue>,
    pub workgroup_size: u32,
    pub workgroups: [u32; 3],
}

impl Kernel {
    /// The bindings the kernel writes, after its buffer operands: 2 for an epilogue that keeps
    /// the pre-activation, else 1. Each has the result's type.
    pub fn written(&self) -> usize {
        match self.epilogue {
            Some(Epilogue::Relu {
                writes_pre_activation: true,
            }) => 2,
            _ => 1,
        }
    }

    /// The text the module id is derived from: every field but the id.
    pub fn signature(&self) -> String {
        let mut text = format!("{} {} {} (", self.kind, self.op, self.dtype);
        for (i, operand) in self.operands.iter().enumerate() {
            if i > 0 {
                text.push_str(", ");
            }
            text.push_str(&operand.to_string());
        }
        text.push_str(&format!(") -> {}", self.result));
        if let Some(reduces) = &self.reduces {
            text.push_str(&format!(" over {reduces}"));
        }
        if let Some(epilogue) = &self.epilogue {
            text.push_str(&format!(" {epilogue}"));
        }
        let [x, y, z] = self.workgroups;
        text.push_str(&format!(
            " workgroup_size {} workgroups [{x}, {y}, {z}]",
            self.workgroup_size
        ));
        text
    }

    /// The module id of a kernel with this [`Kernel::signature`]: the first 16 bytes of its blake3
    /// hash, marked as a version 8 UUID.
    pub fn module_id(signature: &str) -> KernelModuleId {
        let hash = blake3::hash(format!("modelspec-kernel/1\n{signature}").as_bytes());
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&hash.as_bytes()[..16]);
        bytes[6] = (bytes[6] & 0x0f) | 0x80;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        KernelModuleId(bytes)
    }
}

/// Kernel IR and the execution plans of one training step.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionPlan {
    /// The size in bytes of each plan buffer, indexed by [`BufferView::buffer`]; both plans
    /// share the numbering. One buffer per value, none reused.
    pub buffers: Vec<u64>,
    /// Every declared parameter, in `ParamId` order, with its views.
    pub parameters: Vec<ParameterSpec>,
    /// Every `input`, then every `target` declaration of the `train` block, in source order.
    pub data: Vec<DataSpec>,
    /// The distinct kernels, in the order the step plan first dispatches them.
    pub kernels: Vec<Kernel>,
    /// One training step: forward, loss, backward and SGD, with no completion point.
    pub step: Vec<PlanStep>,
    /// The prediction alone: no loss, gradient or update.
    pub forward: Vec<PlanStep>,
    /// The view the step plan writes the loss into.
    pub loss_view: BufferView,
    /// The view both plans write the prediction into.
    pub prediction_view: BufferView,
}

impl ExecutionPlan {
    /// The kernel with module id `id`.
    pub fn kernel(&self, id: KernelModuleId) -> Option<&Kernel> {
        self.kernels.iter().find(|k| k.id == id)
    }
}
