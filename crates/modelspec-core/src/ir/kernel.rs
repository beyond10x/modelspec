//! Kernel IR and the execution plan of a training step (DESIGN.md § 3.1, § 5.2, § 7.1).
//!
//! An [`ExecutionPlan`] is what [`crate::schedule::schedule`] makes of a verified training step:
//! the buffer table, the views a runtime binds (one per input and target, one or two per
//! parameter, the loss and the prediction), the [`Kernel`]s, and two lists of [`PlanStep`]s: the
//! step plan, one training step that a runner may record many times, and the forward plan, the
//! prediction alone. [`crate::ir::verify::verify_plan`] checks it; [`crate::ir::print::print_kernel`]
//! prints it.
//!
//! The workgroup size of a kernel is the constant of its [`KernelKind`], except a matmul's, which
//! is its [`MatMulTile`]'s: a function of its shape and of the invocation limit of the device
//! profile its plan is scheduled for, 128 without one (DESIGN.md § 3.1, § 5.2).

use std::collections::BTreeSet;
use std::fmt;

use crate::artifact::{DataSpec, ParameterSpec};
use crate::device::DeviceProfile;
use crate::ir::tensor::{DType, Dim, TensorType, ValueId};
use crate::ir::verify::MAX_WORKGROUP_INVOCATIONS;

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
/// ([`KernelKind::workgroup_size`]), except a matmul's, which is its [`MatMulTile`]'s
/// ([`Kernel::matmul_tile`]).
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
    /// A row-wise op computed one output element per invocation: `Embedding`, `AddRows`,
    /// `Gelu`, `GeluGrad`, `Dropout` and `DropoutGrad` (DESIGN.md § 5.2).
    PerElement,
    /// A row-wise op computed one output row per invocation, its reductions over the row in
    /// ascending column order: `EmbeddingGrad` (each table row, its ids scanned in ascending
    /// order), `LayerNorm`, `LayerNormGradX`, `Softmax` and `SoftmaxGrad`.
    PerRow,
    /// A column reduction over the rows of its operands in ascending row order, one output
    /// column per invocation: `SumRows` and `LayerNormGradGamma`.
    PerColumn,
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
/// `[784, 100] x [100, 64]` matmul from 4.91 to 4.58 µs against 32 values read 6 ahead
/// (DESIGN.md § 5.2).
pub const MATMUL_TWO_ROW_TILE: (u32, u32) = (16, 2);

/// A register-blocked [`MatMulTile`] (DESIGN.md § 5.2): its workgroup size, columns, rows and
/// columns per invocation, its depth before [`MIN_SHARED_MEMORY_BYTES`] lowers it, and the
/// shapes that take it: `n` of at least `cols`, `k` of at least `from_k`, and at least
/// `from_workgroups` workgroups of the block over the output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatMulBlock {
    pub workgroup_size: u32,
    pub cols: u32,
    pub rows: u32,
    pub cols_per_invocation: u32,
    pub depth: u32,
    pub from_k: u64,
    pub from_workgroups: u64,
}

/// The block register-blocked tile: 256 invocations of 8 rows by 8 columns, a block of 128 rows
/// by 128 columns, tiles of 8 values of `k`. Each staged value of `a` serves 128 columns, as in
/// [`MATMUL_WIDE_BLOCK`], and each staged value of `b` 128 rows instead of 64, so a workgroup
/// reads half the operand values per product the wide tile reads. It reads workgroup memory in
/// vectors ([`MatMulTile::vector_reads`]): on the RTX 3090 it takes `[8192, 784] x [784, 4096]`
/// from 2,853.5 µs on the wide tile to 2,600.0 µs and `[784, 8192]ᵀ x [8192, 4096]` from 3,573.6
/// to 2,766.6 µs (DESIGN.md § 3.1, § 5.2). Only a device profile that allows 256 invocations
/// takes it ([`MatMulTile::for_profile`]); `from_k` and `from_workgroups` are the wide tile's.
pub const MATMUL_LARGE_BLOCK: MatMulBlock = MatMulBlock {
    workgroup_size: 256,
    cols: 128,
    rows: 8,
    cols_per_invocation: 8,
    depth: 8,
    from_k: 64,
    from_workgroups: 128,
};

/// The wide register-blocked tile: 128 invocations of 8 rows by 8 columns, a block of 64 rows by
/// 128 columns, tiles of 8 values of `k`. On the RTX 3090 it takes the forward matmul of
/// `mlp_sweep` b8192 h4096, `[8192, 784] x [784, 4096]`, from 16,062 µs (one or two columns per
/// invocation) to 5,487 µs, against 9,042 µs for [`MATMUL_SQUARE_BLOCK`]; at `k = 10`, the
/// `[8192, 10] x [10, 4096]` matmul of that step, the square block is faster (202 against
/// 221 µs), hence `from_k` (DESIGN.md § 5.2).
pub const MATMUL_WIDE_BLOCK: MatMulBlock = MatMulBlock {
    workgroup_size: 128,
    cols: 128,
    rows: 8,
    cols_per_invocation: 8,
    depth: 8,
    from_k: 64,
    from_workgroups: 128,
};

/// The square register-blocked tile: 128 invocations of 8 rows by 4 columns, a block of 64 rows
/// by 64 columns, tiles of 16 values of `k`. Every output of at least 1024 by 1024 that does not
/// take [`MATMUL_WIDE_BLOCK`] takes it (256 workgroups). On the RTX 3090 it is slower than one
/// or two columns per invocation at 52 workgroups (`[784, 8192] x [8192, 256]`, 1,087 against
/// 773 µs) and faster at 128 (`[8192, 784] x [784, 64]`, 123 against 184 µs), hence
/// `from_workgroups` (DESIGN.md § 5.2).
pub const MATMUL_SQUARE_BLOCK: MatMulBlock = MatMulBlock {
    workgroup_size: 128,
    cols: 64,
    rows: 8,
    cols_per_invocation: 4,
    depth: 16,
    from_k: 1,
    from_workgroups: 128,
};

/// How a matmul kernel `[m, k] x [k, n] -> [m, n]` covers its output (DESIGN.md § 5.2): each
/// workgroup of `workgroup_size` invocations computes a block of
/// [`MatMulTile::rows_per_workgroup`] rows by `cols` columns, in `cols / cols_per_invocation`
/// column lanes. Invocation `t` computes the columns `t % lanes + c * lanes` for `c` in
/// `0..cols_per_invocation` of the rows `t / lanes + q * (workgroup_size / lanes)` for `q` in
/// `0..rows`, `rows · cols_per_invocation` output elements, each with its own accumulator over
/// ascending `k` from the `k = 0` product. A tile with [`MatMulTile::vector_reads`] assigns adjacent
/// columns and rows instead, as that method states; each output element still has one
/// invocation and one accumulator. The contracting index is walked in tiles of `depth`
/// values: a tile is read into registers `ahead` tiles before its products, then staged in one
/// of `buffers` workgroup memory buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatMulTile {
    pub workgroup_size: u32,
    /// Output columns per workgroup, a power of two that divides `workgroup_size`.
    pub cols: u32,
    /// Output rows per invocation.
    pub rows: u32,
    /// Output columns per invocation, a power of two that divides `cols`; with 1 the staged `b`
    /// block is transposed, one column per [`MatMulTile::b_stride`], else it is staged as read,
    /// one value of `k` per [`MatMulTile::b_stride`].
    pub cols_per_invocation: u32,
    /// Values of the contracting index per tile.
    pub depth: u32,
    /// Tiles read into registers before the products of the first one.
    pub ahead: u32,
    /// Workgroup memory buffers, 2 when there is more than one tile, else 1.
    pub buffers: u32,
    /// The `f32` values added to the stride of the staged `b` block against bank conflicts.
    pub b_pad: u32,
}

impl MatMulTile {
    /// The tile of a `[m, k] x [k, n]` matmul on every Vulkan device: [`MatMulTile::of_invocations`]
    /// at [`MAX_WORKGROUP_INVOCATIONS`], the Vulkan minimum of `maxComputeWorkGroupInvocations`.
    /// A plan scheduled without a device profile runs it.
    pub fn of(m: u64, k: u64, n: u64) -> Self {
        Self::of_invocations(m, k, n, MAX_WORKGROUP_INVOCATIONS)
    }

    /// The tile of a `[m, k] x [k, n]` matmul whose workgroups may hold `max_invocations`
    /// invocations, a function of the shape and the limit: the first of [`MATMUL_LARGE_BLOCK`],
    /// [`MATMUL_WIDE_BLOCK`] and [`MATMUL_SQUARE_BLOCK`] whose workgroup size is within the limit
    /// and which the shape takes, read 1 tile ahead with `b` unpadded. Otherwise 128 invocations
    /// of one column each, 16 columns (fewer, a power of two, when `n` is smaller), one row per
    /// invocation, or two when one row needs at least [`MATMUL_TWO_ROWS_FROM_WORKGROUPS`]
    /// workgroups, tiles of up to 64 values read 6 tiles ahead for one row and of up to 16 values
    /// read 2 tiles ahead for two ([`MATMUL_TWO_ROW_TILE`]), and the transposed `b` padded by 4.
    /// In every tile the depth is clamped to `k`, then halved until the staged values fit in
    /// [`MIN_SHARED_MEMORY_BYTES`]. A limit below [`MAX_WORKGROUP_INVOCATIONS`] counts as that
    /// minimum, so every limit up to 255 gives the tiles of [`MatMulTile::of`].
    pub fn of_invocations(m: u64, k: u64, n: u64, max_invocations: u32) -> Self {
        let limit = max_invocations.max(MAX_WORKGROUP_INVOCATIONS);
        for block in [MATMUL_LARGE_BLOCK, MATMUL_WIDE_BLOCK, MATMUL_SQUARE_BLOCK] {
            if block.workgroup_size > limit {
                continue;
            }
            let tile = Self {
                workgroup_size: block.workgroup_size,
                cols: block.cols,
                rows: block.rows,
                cols_per_invocation: block.cols_per_invocation,
                depth: block.depth,
                ahead: 1,
                buffers: 1,
                b_pad: 0,
            };
            if n >= u64::from(block.cols)
                && k >= block.from_k
                && tile.workgroups(m, n) >= block.from_workgroups
            {
                return tile.fitted(k);
            }
        }
        let workgroup_size = 128;
        let cols = n.clamp(1, 16).next_power_of_two() as u32;
        let one_row = m.div_ceil(u64::from(workgroup_size / cols)) * n.div_ceil(u64::from(cols));
        let (rows, (depth, ahead)) = if one_row >= MATMUL_TWO_ROWS_FROM_WORKGROUPS {
            (2, MATMUL_TWO_ROW_TILE)
        } else {
            (1, (64, 6))
        };
        Self {
            workgroup_size,
            cols,
            rows,
            cols_per_invocation: 1,
            depth,
            ahead,
            buffers: 1,
            b_pad: 4,
        }
        .fitted(k)
    }

    /// The tile of a `[m, k] x [k, n]` matmul reading the operands marked in `transposed`
    /// ([`Kernel::transposed`]) on a device with `profile`: [`MatMulTile::for_limits`] of
    /// [`MatMulLimits::of`] the profile.
    pub fn for_profile(
        m: u64,
        k: u64,
        n: u64,
        transposed: [bool; 2],
        profile: &DeviceProfile,
    ) -> Self {
        Self::for_limits(m, k, n, transposed, MatMulLimits::of(profile))
    }

    /// The tile of a `[m, k] x [k, n]` matmul reading the operands marked in `transposed`
    /// within `limits`: [`MatMulTile::of_invocations`] at `limits.invocations`, then the next
    /// smaller tile while one of more than [`MAX_WORKGROUP_INVOCATIONS`] invocations stages
    /// more bytes ([`MatMulTile::layout`]) than `limits.shared_bytes`. A tile of the Vulkan
    /// minimum is returned as it is: the emitter refuses one that does not fit.
    pub fn for_limits(m: u64, k: u64, n: u64, transposed: [bool; 2], limits: MatMulLimits) -> Self {
        let mut limit = limits.invocations;
        loop {
            let tile = Self::of_invocations(m, k, n, limit);
            if tile.workgroup_size <= MAX_WORKGROUP_INVOCATIONS
                || tile.layout(transposed).shared_bytes() <= limits.shared_bytes
            {
                return tile;
            }
            limit = tile.workgroup_size - 1;
        }
    }

    /// Where the tile's values lie in workgroup memory for a matmul reading the operands marked
    /// in `transposed`: the `a` block, one line of [`MatMulLayout::a_stride`] per output row,
    /// then the `b` block, `cols` lines of [`MatMulLayout::b_stride`] transposed or `depth`
    /// lines as read, in each of `buffers` buffers.
    ///
    /// An operand read as is keeps the strides of the tile, `depth` and
    /// [`MatMulTile::b_stride`]. A transposed operand is staged in other runs (DESIGN.md
    /// § 5.2), and its stride is padded so that the 32 stores of a subgroup fall into 32
    /// different banks of 4 bytes. A transposed `a` stores 8 rows by 4 values of `k` per
    /// subgroup: with a depth that is a multiple of 4, its stride is the smallest one of at
    /// least `depth` that leaves 4 modulo 8, so the 8 rows start in 8 different groups of 4
    /// banks. A transposed `b` of a register-blocked tile stores `32 / depth` columns by `depth`
    /// values of `k`, `k`-major: with a depth of 2 or more that divides 32, its stride is
    /// `cols + 32 / depth`. A transposed `b` of a one-column tile already has the padded
    /// stride of [`MatMulTile::b_pad`].
    ///
    /// A tile with [`MatMulTile::vector_reads`] takes [`MatMulTile::vector_layout`] instead.
    pub fn layout(&self, transposed: [bool; 2]) -> MatMulLayout {
        let [a_transposed, b_transposed] = transposed;
        let depth = self.depth;
        if self.vector_reads() {
            return self.vector_layout(transposed);
        }
        let a_stride = if a_transposed && depth >= 4 && depth.is_multiple_of(4) {
            depth + (12 - depth % 8) % 8
        } else {
            depth
        };
        let b_stride = if b_transposed
            && self.cols_per_invocation > 1
            && depth > 1
            && 32u32.is_multiple_of(depth)
        {
            self.b_stride() + 32 / depth
        } else {
            self.b_stride()
        };
        let lines = if self.cols_per_invocation == 1 {
            self.cols
        } else {
            self.depth
        };
        let b_offset = self.rows_per_workgroup() * a_stride;
        MatMulLayout {
            a_stride,
            b_stride,
            b_offset,
            buffer_values: b_offset + lines * b_stride,
            buffers: self.buffers,
        }
    }

    /// Whether each invocation reads its values of `a` and `b` of one `k` from workgroup memory
    /// as vectors of 4 `f32` ([`MatMulTile::vector_layout`]): the tiles of more than
    /// [`MAX_WORKGROUP_INVOCATIONS`] invocations, [`MATMUL_LARGE_BLOCK`], whose rows and columns
    /// per invocation are multiples of 4. Invocation `t` then computes 4 adjacent columns from
    /// `4 · (t % lanes)` in each group of `4 · lanes` columns, and 4 adjacent rows from
    /// `4 · (t / lanes)` in each group of `4 · workgroup_size / lanes` rows.
    pub fn vector_reads(&self) -> bool {
        self.workgroup_size > MAX_WORKGROUP_INVOCATIONS
            && self.rows.is_multiple_of(4)
            && self.cols_per_invocation.is_multiple_of(4)
    }

    /// The layout of a tile with [`MatMulTile::vector_reads`]: both blocks `k`-major, so the 4
    /// adjacent rows or columns an invocation reads at one `k` are one aligned vector. The `a`
    /// block holds one line of the block's rows per value of `k`, at a stride of the rows plus
    /// 4 when `a` is read as is and plus 8 when it is transposed, so the 32 stores of a subgroup
    /// (8 values of `k` by 4 rows as is, 4 values of `k` by 8 rows transposed) fall into 32
    /// different banks; the `b` block follows, `k`-major at
    /// [`MatMulTile::b_stride`], padded by `32 / depth` when it is transposed and its depth of 2
    /// to 8 divides 32. Every stride and offset is a multiple of 4 values.
    pub fn vector_layout(&self, transposed: [bool; 2]) -> MatMulLayout {
        let [a_transposed, b_transposed] = transposed;
        let depth = self.depth;
        let a_stride = self.rows_per_workgroup() + if a_transposed { 8 } else { 4 };
        let b_stride = if b_transposed && (2..=8).contains(&depth) && 32u32.is_multiple_of(depth) {
            self.b_stride() + 32 / depth
        } else {
            self.b_stride()
        };
        let b_offset = depth * a_stride;
        MatMulLayout {
            a_stride,
            b_stride,
            b_offset,
            buffer_values: b_offset + depth * b_stride,
            buffers: self.buffers,
        }
    }

    /// The layout a device with `shared_bytes` of workgroup memory stages the tile in:
    /// [`MatMulTile::layout`] when it fits, and otherwise that of a matmul reading neither
    /// operand transposed.
    pub fn layout_within(&self, transposed: [bool; 2], shared_bytes: u32) -> MatMulLayout {
        let padded = self.layout(transposed);
        if padded.shared_bytes() <= shared_bytes {
            padded
        } else {
            self.layout([false, false])
        }
    }

    /// The tile with its depth clamped to `k`, then halved until the staged values fit in
    /// [`MIN_SHARED_MEMORY_BYTES`], and two buffers when there is more than one tile.
    fn fitted(mut self, k: u64) -> Self {
        self.depth = k.clamp(1, u64::from(self.depth)) as u32;
        loop {
            self.buffers = if u64::from(self.depth) < k { 2 } else { 1 };
            if self.depth == 1 || self.shared_bytes() <= MIN_SHARED_MEMORY_BYTES {
                return self;
            }
            self.depth /= 2;
        }
    }

    /// The column lanes of a workgroup: `cols / cols_per_invocation`.
    pub fn lanes(&self) -> u32 {
        self.cols / self.cols_per_invocation
    }

    /// Output rows per workgroup.
    pub fn rows_per_workgroup(&self) -> u32 {
        (self.workgroup_size / self.lanes()) * self.rows
    }

    /// The distance in the staged `b` block between two columns, when it is transposed
    /// (`cols_per_invocation` 1), or between two values of `k`: `depth` or `cols` rounded up
    /// to a multiple of 4, plus [`MatMulTile::b_pad`].
    pub fn b_stride(&self) -> u32 {
        let line = if self.cols_per_invocation == 1 {
            self.depth
        } else {
            self.cols
        };
        line.next_multiple_of(4) + self.b_pad
    }

    /// The `f32` values of one buffer: the `a` block, one row of `depth` values per output
    /// row, then the `b` block, `cols` lines of [`MatMulTile::b_stride`] transposed or `depth`
    /// lines as read.
    pub fn buffer_values(&self) -> u32 {
        let lines = if self.cols_per_invocation == 1 {
            self.cols
        } else {
            self.depth
        };
        self.rows_per_workgroup() * self.depth + lines * self.b_stride()
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

/// The two device limits a matmul tile is chosen within ([`MatMulTile::for_limits`]): the
/// invocations of one workgroup of `LocalSize` x and the bytes of workgroup memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatMulLimits {
    pub invocations: u32,
    pub shared_bytes: u32,
}

impl MatMulLimits {
    /// The limits of the device `profile` describes: the smaller of its
    /// `max_workgroup_invocations` and `max_workgroup_size.x`, the two bounds of a workgroup of
    /// `LocalSize` x, and its `max_shared_memory_bytes`.
    pub fn of(profile: &DeviceProfile) -> Self {
        Self {
            invocations: profile
                .max_workgroup_invocations
                .min(profile.max_workgroup_size.x),
            shared_bytes: profile.max_shared_memory_bytes,
        }
    }

    /// The least invocations of one workgroup every Vulkan device provides, which the smallest
    /// tile needs: the invariant `invocations >= 128` of `spec/domains/artifact.yaml`
    /// `MatMulLimits`.
    pub const MIN_INVOCATIONS: u32 = MAX_WORKGROUP_INVOCATIONS;

    /// The least workgroup memory every Vulkan device provides, which every tile of
    /// [`Self::MIN_INVOCATIONS`] stages within: the invariant `shared_bytes >= 16384` of
    /// `spec/domains/artifact.yaml` `MatMulLimits`.
    pub const MIN_SHARED_BYTES: u32 = MIN_SHARED_MEMORY_BYTES;

    /// The limits, if they hold both invariants of `spec/domains/artifact.yaml` `MatMulLimits`
    /// ([`Self::MIN_INVOCATIONS`], [`Self::MIN_SHARED_BYTES`]); else the first field below its
    /// bound. Every recorded limit passes here: the compile's, from a device profile
    /// ([`crate::compile::compile_artifact_for`]), and the reader's, from a manifest
    /// ([`crate::artifact::read`]).
    pub fn checked(self) -> Result<Self, LimitsBelowMinimum> {
        for (field, bound, value) in [
            ("invocations", Self::MIN_INVOCATIONS, self.invocations),
            ("shared_bytes", Self::MIN_SHARED_BYTES, self.shared_bytes),
        ] {
            if value < bound {
                return Err(LimitsBelowMinimum {
                    field,
                    bound,
                    value,
                });
            }
        }
        Ok(self)
    }
}

/// A [`MatMulLimits`] field below the bound of its invariant ([`MatMulLimits::checked`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LimitsBelowMinimum {
    /// `invocations` or `shared_bytes`.
    pub field: &'static str,
    /// The least value the field may hold.
    pub bound: u32,
    /// The value it holds.
    pub value: u32,
}

impl fmt::Display for LimitsBelowMinimum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "matmul_limits.{} >= {} ({} is {})",
            self.field, self.bound, self.field, self.value
        )
    }
}

impl std::error::Error for LimitsBelowMinimum {}

/// The workgroup memory of a [`MatMulTile`] for a matmul with given `transposed` flags
/// ([`MatMulTile::layout`]), in `f32` values: what the emitter stages and where.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatMulLayout {
    /// The distance between two rows of the staged `a` block.
    pub a_stride: u32,
    /// The distance in the staged `b` block between two columns, when it is transposed
    /// (`cols_per_invocation` 1), or between two values of `k`.
    pub b_stride: u32,
    /// Where the `b` block starts in a buffer, after the `a` block.
    pub b_offset: u32,
    /// The values of one buffer.
    pub buffer_values: u32,
    /// The buffers, the tile's [`MatMulTile::buffers`].
    pub buffers: u32,
}

impl MatMulLayout {
    /// The `f32` values staged, every buffer.
    pub fn shared_values(&self) -> u32 {
        self.buffers * self.buffer_values
    }

    /// The workgroup memory staged, in bytes.
    pub fn shared_bytes(&self) -> u32 {
        self.shared_values() * 4
    }
}

/// Every parameter of the tile, so that a matmul's [`Kernel::signature`] and module id change
/// with the module its tile emits. The destructuring is exhaustive: a new field does not compile
/// until it is printed.
impl fmt::Display for MatMulTile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let MatMulTile {
            workgroup_size,
            cols,
            rows,
            cols_per_invocation,
            depth,
            ahead,
            buffers,
            b_pad,
        } = *self;
        write!(
            f,
            "tile workgroup_size {workgroup_size} cols {cols} rows {rows} \
             cols_per_invocation {cols_per_invocation} depth {depth} ahead {ahead} \
             buffers {buffers} b_pad {b_pad}"
        )
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
            KernelKind::PerElement => "per_element",
            KernelKind::PerRow => "per_row",
            KernelKind::PerColumn => "per_column",
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
    /// `ReluGrad(x, sum)`, the gradient of a `relu` whose adjoint the matmul computes: the
    /// kernel reads the pre-activation `x` as a third buffer operand, of the result's type, and
    /// writes `x > 0 ? sum : 0.0`, a compare and a select; the sum itself is not written.
    ReluGrad,
}

impl Epilogue {
    /// The buffer operands of a matmul kernel with this epilogue: 3 for `ReluGrad`, else 2.
    pub fn matmul_operands(epilogue: Option<Epilogue>) -> usize {
        match epilogue {
            Some(Epilogue::ReluGrad) => 3,
            _ => 2,
        }
    }
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
            Epilogue::ReluGrad => f.write_str("epilogue ReluGrad"),
        }
    }
}

/// The Tensor IR ops an `elementwise` kernel implements, alone or in a chain.
pub const ELEMENTWISE_OPS: [&str; 5] = ["Add", "Sub", "Mul", "Relu", "ReluGrad"];

/// The operand count of the elementwise op `op`: 1 for `Relu`, 2 for `Add`, `Sub`, `Mul` and
/// `ReluGrad`; `None` for an op that is not elementwise.
pub fn elementwise_arity(op: &str) -> Option<usize> {
    match op {
        "Relu" => Some(1),
        "Add" | "Sub" | "Mul" | "ReluGrad" => Some(2),
        _ => None,
    }
}

/// One op of an [`ElementwiseChain`] after its first: the op, and the position among its
/// operands of the result of the op before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChainLink {
    pub op: &'static str,
    pub at: usize,
}

/// A chain of elementwise ops that one `elementwise` kernel computes in order (DESIGN.md § 3.1):
/// each op after the first reads the result of the op before it, which no other op reads, at
/// [`ChainLink::at`]. Its kernel op is its [`ElementwiseChain::name`], `First>Op@at>Op@at...`,
/// for example `Sub>Mul@0`, and its operands are the chain's leaves: the first op's operands in
/// order, then each later op's operands in order without the result before it. Each op rounds
/// its result once, as the unfused kernels do.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ElementwiseChain {
    pub first: &'static str,
    pub links: Vec<ChainLink>,
}

impl ElementwiseChain {
    /// The chain a kernel op names: at least one link, every op elementwise, every position
    /// within its op's operands, and the name in the canonical form [`Self::name`] prints.
    /// `None` for a single op or any other text.
    pub fn parse(name: &str) -> Option<Self> {
        let mut parts = name.split('>');
        let first = static_op(parts.next()?)?;
        let links = parts
            .map(|part| {
                let (op, at) = part.split_once('@')?;
                let op = static_op(op)?;
                let at: usize = at.parse().ok()?;
                (at < elementwise_arity(op)?).then_some(ChainLink { op, at })
            })
            .collect::<Option<Vec<_>>>()?;
        if links.is_empty() {
            return None;
        }
        let chain = ElementwiseChain { first, links };
        (chain.render() == name).then_some(chain)
    }

    fn render(&self) -> String {
        let mut name = self.first.to_owned();
        for link in &self.links {
            name.push_str(&format!(">{}@{}", link.op, link.at));
        }
        name
    }

    /// The kernel op of the chain, interned so that a [`Kernel`] can hold it.
    pub fn name(&self) -> &'static str {
        intern(self.render())
    }

    /// The ops of the chain in order.
    pub fn ops(&self) -> impl Iterator<Item = &'static str> + '_ {
        std::iter::once(self.first).chain(self.links.iter().map(|link| link.op))
    }

    /// The number of the chain's leaves, its kernel's operands.
    pub fn operands(&self) -> usize {
        let arity = |op| elementwise_arity(op).unwrap_or(0);
        arity(self.first)
            + self
                .links
                .iter()
                .map(|link| arity(link.op) - 1)
                .sum::<usize>()
    }
}

/// `op` as the `'static` name of an elementwise op.
fn static_op(op: &str) -> Option<&'static str> {
    ELEMENTWISE_OPS.iter().copied().find(|known| *known == op)
}

/// The one `'static` copy of `name`. A program has few distinct chains, and each is leaked once
/// for the life of the process.
fn intern(name: String) -> &'static str {
    use std::sync::{Mutex, OnceLock};
    static NAMES: OnceLock<Mutex<BTreeSet<&'static str>>> = OnceLock::new();
    let mut names = NAMES
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(known) = names.get(name.as_str()) {
        return known;
    }
    let leaked: &'static str = Box::leak(name.into_boxed_str());
    names.insert(leaked);
    leaked
}

/// One kernel of Kernel IR. Its bindings are its buffer operands in order, each read, then what
/// it writes: the pre-activation of an [`Epilogue`] that keeps it, then its result.
#[derive(Clone, Debug, PartialEq)]
pub struct Kernel {
    /// [`Kernel::module_id`] of the other fields.
    pub id: KernelModuleId,
    pub kind: KernelKind,
    /// The Tensor IR op the kernel implements, by name; for an `elementwise` kernel that fuses a
    /// chain of ops, the [`ElementwiseChain::name`] of the chain.
    pub op: &'static str,
    pub dtype: DType,
    pub operands: Vec<Operand>,
    pub result: TensorType,
    /// The tensor a reduce stage reduces, so that both stages name its element count.
    pub reduces: Option<TensorType>,
    /// What a matmul kernel applies to each output element before storing it; `None` on every
    /// other kind.
    pub epilogue: Option<Epilogue>,
    /// Whether a matmul kernel reads each of its two operands transposed (DESIGN.md § 3.1,
    /// § 5.2). Operand `i` keeps its logical type, `[m, k]` or `[k, n]`, so
    /// [`Kernel::matmul_shape`] and the tile do not change; with `transposed[i]` its binding
    /// holds the row-major transpose of that type, `[k, m]` or `[n, k]`, the view of the tensor
    /// a folded `transpose` read, and the kernel reads it in place. `[false, false]` on every
    /// other kind.
    pub transposed: [bool; 2],
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

    /// `(m, k, n)` of a kernel whose operands are two static buffers `[m, k]` and `[k, n]`, and
    /// for a [`Epilogue::ReluGrad`] a third, the pre-activation, `[m, n]`.
    pub fn matmul_shape(operands: &[Operand]) -> Option<(u64, u64, u64)> {
        let dims = |operand: &Operand| match operand {
            Operand::Buffer(ty) => match ty.shape[..] {
                [Dim::Static(rows), Dim::Static(cols)] => Some((rows, cols)),
                _ => None,
            },
            Operand::Immediate(_) => None,
        };
        let (a, b, x) = match operands {
            [a, b] => (a, b, None),
            [a, b, x] => (a, b, Some(x)),
            _ => return None,
        };
        let ((m, k), (k2, n)) = (dims(a)?, dims(b)?);
        let x_fits = x.is_none_or(|x| dims(x) == Some((m, n)));
        (k == k2 && x_fits).then_some((m, k, n))
    }

    /// The floating-point operations of one dispatch of the kernel under the counting rule of
    /// DESIGN.md § 3.1: `2·m·k·n` for a matmul, its epilogue adding none; per output element,
    /// one for `Add`, `Sub`, `Mul`, `Relu` and `ReluGrad`, one per op for an elementwise chain
    /// ([`ElementwiseChain`]), and two for the scaled SGD update
    /// `p - (lr * g)`; one per element of the buffer operands of a reduction stage or a loss
    /// sum; none for a transpose or a broadcast. A row-wise kernel counts the operations of the
    /// CPU interpreter's definition of its op, each add, subtract, multiply, compare, division,
    /// square root and `exp` one and each fused multiply-add two: none for `Embedding`; one per
    /// output element for `AddRows`, `Dropout` and `DropoutGrad`; one per element of the
    /// gradient for `EmbeddingGrad` and of the operand for `SumRows`; 10 per element for `Gelu`
    /// and 24 for `GeluGrad`; 5 per element for `Softmax` and 4 for `SoftmaxGrad`; for an
    /// `[n, h]` operand, `n·(8h + 3)` for `LayerNorm` and `LayerNormGradGamma` and
    /// `n·(14h + 3)` for `LayerNormGradX` (DESIGN.md § 3.1). `None` for an op without a rule, a
    /// shape with a dimension that is not static, or a count past `u64`.
    pub fn flop(&self) -> Option<u64> {
        let elements = |ty: &TensorType| {
            ty.shape.iter().try_fold(1u64, |n, dim| match dim {
                Dim::Static(d) => n.checked_mul(*d),
                Dim::Symbol(_) => None,
            })
        };
        let per_output = |flop: u64| elements(&self.result)?.checked_mul(flop);
        let operand = |i: usize| match self.operands.get(i)? {
            Operand::Buffer(ty) => Some(ty),
            Operand::Immediate(_) => None,
        };
        // `n·(per_element·h + per_row)` over the rows `[n, h]` of operand 0.
        let rows = |per_element: u64, per_row: u64| match operand(0)?.shape[..] {
            [Dim::Static(n), Dim::Static(h)] => per_element
                .checked_mul(h)?
                .checked_add(per_row)?
                .checked_mul(n),
            _ => None,
        };
        match self.kind {
            KernelKind::MatMul => {
                let (m, k, n) = Kernel::matmul_shape(&self.operands)?;
                2u64.checked_mul(m)?.checked_mul(k)?.checked_mul(n)
            }
            KernelKind::Elementwise => match self.op {
                "Add" | "Sub" | "Mul" | "Relu" | "ReluGrad" => per_output(1),
                chain => per_output(ElementwiseChain::parse(chain)?.ops().count() as u64),
            },
            KernelKind::SgdUpdate => match (self.op, self.operands.len()) {
                ("Sub", 3) => per_output(2),
                ("Sub", 2) => per_output(1),
                _ => None,
            },
            KernelKind::ReduceStage1 | KernelKind::ReduceStage2 | KernelKind::LossSum => self
                .operands
                .iter()
                .try_fold(0u64, |n, operand| match operand {
                    Operand::Buffer(ty) => n.checked_add(elements(ty)?),
                    Operand::Immediate(_) => Some(n),
                }),
            KernelKind::Transpose | KernelKind::Broadcast => Some(0),
            KernelKind::PerElement => match self.op {
                "Embedding" => Some(0),
                "AddRows" | "Dropout" | "DropoutGrad" => per_output(1),
                "Gelu" => per_output(10),
                "GeluGrad" => per_output(24),
                _ => None,
            },
            KernelKind::PerRow => match self.op {
                "EmbeddingGrad" => elements(operand(1)?),
                "LayerNorm" => rows(8, 3),
                "LayerNormGradX" => rows(14, 3),
                "Softmax" => per_output(5),
                "SoftmaxGrad" => per_output(4),
                _ => None,
            },
            KernelKind::PerColumn => match self.op {
                "SumRows" => elements(operand(0)?),
                "LayerNormGradGamma" => rows(8, 3),
                _ => None,
            },
        }
    }

    /// The text the module id is derived from: every field but the id, and a matmul's
    /// [`MatMulTile`], so that a module the tile changes gets a new id. An operand the kernel
    /// reads transposed is followed by ` transposed`; a kernel that reads none has the text it
    /// had before the flag existed.
    pub fn signature(&self) -> String {
        self.signature_with(self.matmul_tile())
    }

    /// The tile a matmul kernel runs: [`MatMulTile::of_invocations`] of its shape at its own
    /// workgroup size, so a kernel of 128 invocations has the tile of [`MatMulTile::of`] and one
    /// the schedule gave a block of 256 has that block. A kernel whose workgroup size is not its
    /// tile's is refused by the emitter. `None` on every other kind and on a shape that is not
    /// two static matrices.
    pub fn matmul_tile(&self) -> Option<MatMulTile> {
        match self.kind {
            KernelKind::MatMul => Kernel::matmul_shape(&self.operands)
                .map(|(m, k, n)| MatMulTile::of_invocations(m, k, n, self.workgroup_size)),
            _ => None,
        }
    }

    fn signature_with(&self, tile: Option<MatMulTile>) -> String {
        let mut text = format!("{} {} {} (", self.kind, self.op, self.dtype);
        for (i, operand) in self.operands.iter().enumerate() {
            if i > 0 {
                text.push_str(", ");
            }
            text.push_str(&operand.to_string());
            if self.transposed.get(i) == Some(&true) {
                text.push_str(" transposed");
            }
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
        if let Some(tile) = tile {
            text.push_str(&format!(" {tile}"));
        }
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

/// The arithmetic and memory traffic of one Dispatch step under the counting rule of DESIGN.md
/// § 3.1, computed from the plan when it is read: no field of an artifact holds it. Printed
/// `flop <n> bytes <n>`, or `uncounted`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DispatchCount {
    /// The floating-point operations of the dispatch, and the bytes of the views it reads plus
    /// the bytes of the views it writes.
    Counted { flop: u64, bytes: u64 },
    /// A dispatch the rule has no count for: its kernel's kind and op have no rule, a shape it
    /// needs is not static, a count overflows, or the plan has no kernel of its module id.
    Uncounted,
}

impl fmt::Display for DispatchCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DispatchCount::Counted { flop, bytes } => write!(f, "flop {flop} bytes {bytes}"),
            DispatchCount::Uncounted => f.write_str("uncounted"),
        }
    }
}

impl ExecutionPlan {
    /// The positions, in plan order, of the step plan's steps a loss evaluation records (the
    /// final loss of a run, `infer`): the last step that writes the loss view, every earlier
    /// Dispatch or Copy that writes the prediction's buffer or a buffer a chosen step reads, and
    /// each Barrier with a chosen step since the previous chosen Barrier and one after it, so any
    /// two chosen steps the plan orders with a barrier stay ordered by one. No `Allocate` is
    /// chosen; empty when no step writes the loss view (DESIGN.md § 3.1).
    pub fn loss_steps(&self) -> Vec<usize> {
        let Some(last) = self.step.iter().rposition(|step| {
            step.writes
                .iter()
                .any(|view| view.buffer == self.loss_view.buffer)
        }) else {
            return Vec::new();
        };
        let mut chosen = vec![false; last + 1];
        chosen[last] = true;
        let mut needed: BTreeSet<u32> = self.step[last].reads.iter().map(|v| v.buffer).collect();
        needed.insert(self.prediction_view.buffer);
        for at in (0..last).rev() {
            let step = &self.step[at];
            if matches!(step.op, PlanOp::Dispatch | PlanOp::Copy)
                && step.writes.iter().any(|v| needed.contains(&v.buffer))
            {
                chosen[at] = true;
                needed.extend(step.reads.iter().map(|v| v.buffer));
            }
        }
        let mut steps = Vec::new();
        let mut since_barrier = false;
        for (at, &is_chosen) in chosen.iter().enumerate() {
            if self.step[at].op == PlanOp::Barrier {
                if since_barrier && chosen[at..].contains(&true) {
                    steps.push(at);
                    since_barrier = false;
                }
            } else if is_chosen {
                steps.push(at);
                since_barrier = true;
            }
        }
        steps
    }

    /// The kernel with module id `id`.
    pub fn kernel(&self, id: KernelModuleId) -> Option<&Kernel> {
        self.kernels.iter().find(|k| k.id == id)
    }

    /// The count of `step` under the counting rule (DESIGN.md § 3.1): its kernel's
    /// [`Kernel::flop`], and the `size_bytes` of its read views plus those of its write views.
    /// `None` for a step that is not a Dispatch.
    pub fn dispatch_count(&self, step: &PlanStep) -> Option<DispatchCount> {
        if step.op != PlanOp::Dispatch {
            return None;
        }
        let counted = || {
            let flop = self.kernel(step.kernel_module_id?)?.flop()?;
            let bytes = step
                .reads
                .iter()
                .chain(&step.writes)
                .try_fold(0u64, |n, view| n.checked_add(view.size_bytes))?;
            Some(DispatchCount::Counted { flop, bytes })
        };
        Some(counted().unwrap_or(DispatchCount::Uncounted))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `[100, 784] x [784, 64]` matmul of the MNIST example, with the tile it runs.
    fn mnist_matmul() -> (Kernel, MatMulTile) {
        let tile = MatMulTile::of(100, 784, 64);
        let kernel = Kernel {
            id: KernelModuleId([0; 16]),
            kind: KernelKind::MatMul,
            op: "MatMul",
            dtype: DType::F32,
            operands: vec![
                Operand::Buffer(TensorType::f32(&[100, 784])),
                Operand::Buffer(TensorType::f32(&[784, 64])),
            ],
            result: TensorType::f32(&[100, 64]),
            reduces: None,
            epilogue: None,
            transposed: [false, false],
            workgroup_size: tile.workgroup_size,
            workgroups: [tile.workgroups(100, 64) as u32, 1, 1],
        };
        (kernel, tile)
    }

    #[test]
    fn a_matmul_signature_is_that_of_the_tile_its_shape_runs() {
        let (kernel, tile) = mnist_matmul();
        assert_eq!(kernel.signature(), kernel.signature_with(Some(tile)));
        assert!(
            kernel.signature().ends_with(&format!(" {tile}")),
            "{}",
            kernel.signature()
        );
    }

    #[test]
    fn two_matmuls_differing_only_in_tile_depth_or_ahead_have_different_signatures() {
        let (kernel, tile) = mnist_matmul();
        let deeper = MatMulTile {
            depth: tile.depth * 2,
            ..tile
        };
        let further = MatMulTile {
            ahead: tile.ahead + 1,
            ..tile
        };
        let base = kernel.signature_with(Some(tile));
        assert_ne!(base, kernel.signature_with(Some(deeper)), "depth: {base}");
        assert_ne!(base, kernel.signature_with(Some(further)), "ahead: {base}");
        assert_ne!(
            Kernel::module_id(&base),
            Kernel::module_id(&kernel.signature_with(Some(deeper)))
        );
        assert_ne!(
            Kernel::module_id(&base),
            Kernel::module_id(&kernel.signature_with(Some(further)))
        );
    }

    /// Exhaustive: a field added to [`MatMulTile`] does not compile here until it has a variant
    /// below, and the variant fails until the field is in the signature.
    #[test]
    fn every_matmul_tile_field_changes_the_signature() {
        let (kernel, tile) = mnist_matmul();
        let MatMulTile {
            workgroup_size,
            cols,
            rows,
            cols_per_invocation,
            depth,
            ahead,
            buffers,
            b_pad,
        } = tile;
        let variants = [
            (
                "workgroup_size",
                MatMulTile {
                    workgroup_size: workgroup_size * 2,
                    ..tile
                },
            ),
            (
                "cols",
                MatMulTile {
                    cols: cols * 2,
                    ..tile
                },
            ),
            (
                "rows",
                MatMulTile {
                    rows: rows + 1,
                    ..tile
                },
            ),
            (
                "depth",
                MatMulTile {
                    depth: depth + 1,
                    ..tile
                },
            ),
            (
                "ahead",
                MatMulTile {
                    ahead: ahead + 1,
                    ..tile
                },
            ),
            (
                "buffers",
                MatMulTile {
                    buffers: buffers + 1,
                    ..tile
                },
            ),
            (
                "cols_per_invocation",
                MatMulTile {
                    cols_per_invocation: cols_per_invocation * 2,
                    ..tile
                },
            ),
            (
                "b_pad",
                MatMulTile {
                    b_pad: b_pad + 1,
                    ..tile
                },
            ),
        ];
        let base = kernel.signature_with(Some(tile));
        for (field, variant) in variants {
            assert_ne!(
                base,
                kernel.signature_with(Some(variant)),
                "{field}: {base}"
            );
        }
    }

    /// A whole-buffer `f32` view of `elements` values.
    fn view(buffer: u32, elements: u64) -> BufferView {
        BufferView {
            buffer,
            dtype: DType::F32,
            element_count: elements,
            offset: 0,
            size_bytes: 4 * elements,
            alignment: 4,
        }
    }

    /// A kernel of `kind` and `op` with its content-derived module id.
    fn kernel(
        kind: KernelKind,
        op: &'static str,
        operands: Vec<Operand>,
        result: TensorType,
        reduces: Option<TensorType>,
    ) -> Kernel {
        let mut kernel = Kernel {
            id: KernelModuleId([0; 16]),
            kind,
            op,
            dtype: DType::F32,
            operands,
            result,
            reduces,
            epilogue: None,
            transposed: [false, false],
            workgroup_size: kind.workgroup_size(),
            workgroups: [1, 1, 1],
        };
        kernel.id = Kernel::module_id(&kernel.signature());
        kernel
    }

    fn dispatch(kernel: &Kernel, reads: Vec<BufferView>, writes: Vec<BufferView>) -> PlanStep {
        PlanStep {
            op: PlanOp::Dispatch,
            kernel_module_id: Some(kernel.id),
            workgroups: Some(kernel.workgroups),
            reads,
            writes,
            depends_on: Vec::new(),
            value: None,
        }
    }

    /// A plan of `kernels` whose step plan is `step`: one buffer per view, no parameter or data.
    fn plan(kernels: Vec<Kernel>, step: Vec<PlanStep>) -> ExecutionPlan {
        let buffers = step
            .iter()
            .flat_map(|s| s.reads.iter().chain(&s.writes))
            .fold(Vec::new(), |mut buffers: Vec<u64>, v| {
                let at = v.buffer as usize;
                if buffers.len() <= at {
                    buffers.resize(at + 1, 0);
                }
                buffers[at] = buffers[at].max(v.size_bytes);
                buffers
            });
        ExecutionPlan {
            buffers,
            parameters: Vec::new(),
            data: Vec::new(),
            kernels,
            step,
            forward: Vec::new(),
            loss_view: view(0, 1),
            prediction_view: view(0, 1),
        }
    }

    fn counted(flop: u64, bytes: u64) -> Option<DispatchCount> {
        Some(DispatchCount::Counted { flop, bytes })
    }

    /// story:profile-achieved-throughput: the `[100, 784] x [784, 64]` matmul of the MNIST
    /// step, with its `relu` epilogue writing the pre-activation and the activation, is
    /// `2 * 100 * 784 * 64` FLOP; the epilogue adds none. Its bytes are x and w1 read and the
    /// two `[100, 64]` views written.
    #[test]
    fn a_matmul_dispatch_counts_2_m_k_n_flop_and_the_bytes_of_its_views() {
        let (mut matmul, _) = mnist_matmul();
        matmul.epilogue = Some(Epilogue::Relu {
            writes_pre_activation: true,
        });
        let step = dispatch(
            &matmul,
            vec![view(0, 100 * 784), view(1, 784 * 64)],
            vec![view(2, 100 * 64), view(3, 100 * 64)],
        );
        let plan = plan(vec![matmul], vec![step.clone()]);
        assert_eq!(
            plan.dispatch_count(&step),
            counted(10_035_200, 313_600 + 200_704 + 25_600 + 25_600)
        );
    }

    /// One FLOP per output element for `sub`; two for the SGD update `p - lr * g`, its product
    /// and its difference, which reads `p` and `g` and writes `p` in place.
    #[test]
    fn an_elementwise_dispatch_counts_its_arithmetic_per_output_element() {
        let ty = TensorType::f32(&[100, 10]);
        let sub = kernel(
            KernelKind::Elementwise,
            "Sub",
            vec![Operand::Buffer(ty.clone()), Operand::Buffer(ty.clone())],
            ty.clone(),
            None,
        );
        let sub_step = dispatch(
            &sub,
            vec![view(0, 1000), view(1, 1000)],
            vec![view(2, 1000)],
        );
        let w1 = TensorType::f32(&[784, 64]);
        let update = kernel(
            KernelKind::SgdUpdate,
            "Sub",
            vec![
                Operand::Buffer(w1.clone()),
                Operand::Immediate(0.1),
                Operand::Buffer(w1.clone()),
            ],
            w1,
            None,
        );
        let update_step = dispatch(
            &update,
            vec![view(3, 50_176), view(4, 50_176)],
            vec![view(3, 50_176)],
        );
        let plan = plan(
            vec![sub, update],
            vec![sub_step.clone(), update_step.clone()],
        );
        assert_eq!(plan.dispatch_count(&sub_step), counted(1000, 12_000));
        assert_eq!(
            plan.dispatch_count(&update_step),
            counted(2 * 50_176, 3 * 200_704)
        );
    }

    /// A `ReduceSum` of 1,000 values is a stage 1 over them into 16 partials and a stage 2 over
    /// the partials; the `mse` loss sum reads two `[100, 10]` tensors. One FLOP per element read.
    #[test]
    fn a_reduction_dispatch_counts_one_flop_per_input_element() {
        let input = TensorType::f32(&[100, 10]);
        let partials = TensorType::f32(&[16]);
        let stage1 = kernel(
            KernelKind::ReduceStage1,
            "ReduceSum",
            vec![Operand::Buffer(input.clone())],
            partials.clone(),
            Some(input.clone()),
        );
        let stage2 = kernel(
            KernelKind::ReduceStage2,
            "ReduceSum",
            vec![Operand::Buffer(partials)],
            TensorType::f32_scalar(),
            Some(input.clone()),
        );
        let loss = kernel(
            KernelKind::LossSum,
            "Mse",
            vec![Operand::Buffer(input.clone()), Operand::Buffer(input)],
            TensorType::f32_scalar(),
            None,
        );
        let steps = vec![
            dispatch(&stage1, vec![view(0, 1000)], vec![view(1, 16)]),
            dispatch(&stage2, vec![view(1, 16)], vec![view(2, 1)]),
            dispatch(&loss, vec![view(0, 1000), view(3, 1000)], vec![view(4, 1)]),
        ];
        let plan = plan(vec![stage1, stage2, loss], steps.clone());
        assert_eq!(plan.dispatch_count(&steps[0]), counted(1000, 4000 + 64));
        assert_eq!(plan.dispatch_count(&steps[1]), counted(16, 64 + 4));
        assert_eq!(plan.dispatch_count(&steps[2]), counted(2000, 8000 + 4));
    }

    /// A transpose and a scalar broadcast compute nothing and move their bytes.
    #[test]
    fn a_transpose_or_broadcast_dispatch_counts_no_flop_and_its_bytes() {
        let transpose = kernel(
            KernelKind::Transpose,
            "Transpose",
            vec![Operand::Buffer(TensorType::f32(&[64, 10]))],
            TensorType::f32(&[10, 64]),
            None,
        );
        let broadcast = kernel(
            KernelKind::Broadcast,
            "BroadcastScalar",
            vec![Operand::Immediate(1.0)],
            TensorType::f32(&[8]),
            None,
        );
        let steps = vec![
            dispatch(&transpose, vec![view(0, 640)], vec![view(1, 640)]),
            dispatch(&broadcast, Vec::new(), vec![view(2, 8)]),
        ];
        let plan = plan(vec![transpose, broadcast], steps.clone());
        assert_eq!(plan.dispatch_count(&steps[0]), counted(0, 5120));
        assert_eq!(plan.dispatch_count(&steps[1]), counted(0, 32));
    }

    /// An elementwise op the rule names no arithmetic for, a matmul whose shape is not static
    /// and a module the plan has no kernel of are `uncounted`, never 0; a barrier has no count.
    #[test]
    fn a_dispatch_with_no_counting_rule_is_uncounted_never_zero() {
        let ty = TensorType::f32(&[8]);
        let exp = kernel(
            KernelKind::Elementwise,
            "Exp",
            vec![Operand::Buffer(ty.clone())],
            ty,
            None,
        );
        let symbolic = TensorType {
            dtype: DType::F32,
            shape: vec![Dim::Symbol(0), Dim::Static(4)],
        };
        let matmul = kernel(
            KernelKind::MatMul,
            "MatMul",
            vec![
                Operand::Buffer(symbolic),
                Operand::Buffer(TensorType::f32(&[4, 1])),
            ],
            TensorType::f32(&[8, 1]),
            None,
        );
        let mut unknown = dispatch(&exp, vec![view(0, 8)], vec![view(1, 8)]);
        unknown.kernel_module_id = Some(KernelModuleId([7; 16]));
        let steps = vec![
            dispatch(&exp, vec![view(0, 8)], vec![view(1, 8)]),
            dispatch(&matmul, vec![view(2, 32), view(3, 4)], vec![view(4, 8)]),
            unknown,
        ];
        let barrier = PlanStep {
            op: PlanOp::Barrier,
            kernel_module_id: None,
            workgroups: None,
            reads: Vec::new(),
            writes: Vec::new(),
            depends_on: Vec::new(),
            value: None,
        };
        let plan = plan(vec![exp, matmul], steps.clone());
        for step in &steps {
            assert_eq!(
                plan.dispatch_count(step),
                Some(DispatchCount::Uncounted),
                "{step:?}"
            );
        }
        assert_eq!(plan.dispatch_count(&barrier), None);
        assert_eq!(DispatchCount::Uncounted.to_string(), "uncounted");
        let printed = crate::ir::print::print_kernel(&plan);
        let dispatches: Vec<&str> = printed
            .lines()
            .filter(|l| l.contains(": dispatch "))
            .collect();
        assert_eq!(dispatches.len(), 3, "{printed}");
        for line in dispatches {
            assert!(line.ends_with(" uncounted"), "{line}");
            assert!(!line.contains(" flop "), "{line}");
        }
    }

    #[test]
    fn a_counted_dispatch_prints_its_flop_and_bytes_on_its_plan_line() {
        let (matmul, _) = mnist_matmul();
        let step = dispatch(
            &matmul,
            vec![view(0, 100 * 784), view(1, 784 * 64)],
            vec![view(2, 100 * 64)],
        );
        assert_eq!(
            DispatchCount::Counted {
                flop: 10_035_200,
                bytes: 539_904
            }
            .to_string(),
            "flop 10035200 bytes 539904"
        );
        let printed = crate::ir::print::print_kernel(&plan(vec![matmul], vec![step]));
        let line = printed
            .lines()
            .find(|l| l.contains(": dispatch "))
            .expect("a dispatch line");
        assert!(
            line.ends_with(" workgroups [52, 1, 1] flop 10035200 bytes 539904"),
            "{line}"
        );
    }

    #[test]
    fn a_kernel_other_than_a_matmul_has_no_tile_in_its_signature() {
        let (mut kernel, _) = mnist_matmul();
        kernel.kind = KernelKind::Transpose;
        assert!(
            !kernel.signature().contains(" tile "),
            "{}",
            kernel.signature()
        );
    }
}
