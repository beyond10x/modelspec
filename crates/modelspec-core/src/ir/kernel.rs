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
use crate::ir::tensor::{DType, Dim, TensorType, ValueId};

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
/// ([`KernelKind::workgroup_size`]), except a matmul's, which is its [`MatMulTile`]'s.
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
    /// One invocation per output element in the blocks of its [`MatMulTile`], with an optional
    /// [`Epilogue`] on its rounded sum.
    MatMul,
    /// The SGD update, written in place into the parameter's view: with operands `(p, lr, g)`
    /// (or `(p, g, lr)`) it is `p - (lr * g)`, the product rounded before the subtraction and
    /// neither contracted; with operands `(p, d)` it is `p - d`.
    SgdUpdate,
}

impl KernelKind {
    /// The workgroup size of the kind in the fixed safe schedule: 64 invocations for every kind
    /// but a matmul, which runs [`MatMulTile::workgroup_size`].
    pub const fn workgroup_size(self) -> u32 {
        64
    }
}

/// The workgroup memory every Vulkan device provides, the minimum of
/// `maxComputeSharedMemorySize`: every [`MatMulTile`] stages at most this much.
pub const MIN_SHARED_MEMORY_BYTES: u32 = 16_384;

/// The workgroups a one-row [`MatMulTile`] needs from which its invocations compute two rows
/// each instead: on the RTX 3090 two rows take the `[784, 100] x [100, 64]` matmul (392
/// workgroups of one row) from 5.24 to 4.92 µs and are slower at every MNIST matmul of at most
/// 52 workgroups (DESIGN.md § 5.2).
pub const MATMUL_TWO_ROWS_FROM_WORKGROUPS: u64 = 256;

/// The most values of the contracting index per tile of a two-row [`MatMulTile`], and the
/// tiles it reads ahead: on the RTX 3090 tiles of 16 values read 2 ahead take the
/// `[784, 100] x [100, 64]` matmul from 4.93 to 4.67 µs against 32 values read 6 ahead
/// (DESIGN.md § 5.2).
pub const MATMUL_TWO_ROW_TILE: (u32, u32) = (16, 2);

/// How a matmul kernel `[m, k] x [k, n] -> [m, n]` covers its output (DESIGN.md § 5.2): each
/// workgroup of `workgroup_size` invocations computes a block of
/// [`MatMulTile::rows_per_workgroup`] rows by `cols` columns. Invocation `t` computes column
/// `t % cols` of the rows `t / cols + q * (workgroup_size / cols)` for `q` in `0..rows`, one
/// output element per row, each over ascending `k` from the `k = 0` product. The contracting
/// index is walked in tiles of `depth` values: a tile is read into registers `ahead` tiles
/// before its products, then staged in one of `buffers` workgroup memory buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatMulTile {
    pub workgroup_size: u32,
    /// Output columns per workgroup, a power of two that divides `workgroup_size`.
    pub cols: u32,
    /// Output rows per invocation, each with its own accumulator.
    pub rows: u32,
    /// Values of the contracting index per tile.
    pub depth: u32,
    /// Tiles read into registers before the products of the first one.
    pub ahead: u32,
    /// Workgroup memory buffers, 2 when there is more than one tile, else 1.
    pub buffers: u32,
}

impl MatMulTile {
    /// The tile of a `[m, k] x [k, n]` matmul, a function of its shape alone: 128
    /// invocations, 16 columns (fewer, a power of two, when `n` is smaller), one row per
    /// invocation, or two when one row needs at least [`MATMUL_TWO_ROWS_FROM_WORKGROUPS`]
    /// workgroups, tiles of up to 64 values read 6 tiles ahead for one row and of up to 16
    /// values read 2 tiles ahead for two ([`MATMUL_TWO_ROW_TILE`]), and depth halved until the
    /// staged values fit in [`MIN_SHARED_MEMORY_BYTES`].
    pub fn of(m: u64, k: u64, n: u64) -> Self {
        let workgroup_size = 128;
        let cols = n.clamp(1, 16).next_power_of_two() as u32;
        let one_row = m.div_ceil(u64::from(workgroup_size / cols)) * n.div_ceil(u64::from(cols));
        let (rows, (depth, ahead)) = if one_row >= MATMUL_TWO_ROWS_FROM_WORKGROUPS {
            (2, MATMUL_TWO_ROW_TILE)
        } else {
            (1, (64, 6))
        };
        let mut tile = Self {
            workgroup_size,
            cols,
            rows,
            depth: k.clamp(1, u64::from(depth)) as u32,
            ahead,
            buffers: 1,
        };
        loop {
            tile.buffers = if u64::from(tile.depth) < k { 2 } else { 1 };
            if tile.depth == 1 || tile.shared_bytes() <= MIN_SHARED_MEMORY_BYTES {
                return tile;
            }
            tile.depth /= 2;
        }
    }

    /// Output rows per workgroup.
    pub fn rows_per_workgroup(&self) -> u32 {
        (self.workgroup_size / self.cols) * self.rows
    }

    /// The distance between two columns of the staged, transposed `b` block: `depth` rounded
    /// up to a multiple of 4, plus 4, so the columns an access reads at once fall in distinct
    /// memory banks.
    pub fn b_stride(&self) -> u32 {
        self.depth.next_multiple_of(4) + 4
    }

    /// The `f32` values of one buffer: the `a` block, one row of `depth` values per output
    /// row, then the `b` block transposed, one column per [`MatMulTile::b_stride`].
    pub fn buffer_values(&self) -> u32 {
        self.rows_per_workgroup() * self.depth + self.cols * self.b_stride()
    }

    /// The `f32` values the tile stages, every buffer.
    pub fn shared_values(&self) -> u32 {
        self.buffers * self.buffer_values()
    }

    /// The workgroup memory the tile stages, in bytes.
    pub fn shared_bytes(&self) -> u32 {
        self.shared_values() * 4
    }

    /// The workgroups that cover `[m, n]`: row blocks times column blocks.
    pub fn workgroups(&self, m: u64, n: u64) -> u64 {
        m.div_ceil(u64::from(self.rows_per_workgroup())) * n.div_ceil(u64::from(self.cols))
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

    /// `(m, k, n)` of a kernel whose operands are two static buffers `[m, k]` and `[k, n]`.
    pub fn matmul_shape(operands: &[Operand]) -> Option<(u64, u64, u64)> {
        let dims = |operand: &Operand| match operand {
            Operand::Buffer(ty) => match ty.shape[..] {
                [Dim::Static(rows), Dim::Static(cols)] => Some((rows, cols)),
                _ => None,
            },
            Operand::Immediate(_) => None,
        };
        let [a, b] = operands else {
            return None;
        };
        let ((m, k), (k2, n)) = (dims(a)?, dims(b)?);
        (k == k2).then_some((m, k, n))
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
