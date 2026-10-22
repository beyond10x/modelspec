//! What the GPU tests share: the tolerances of DESIGN.md § 9 gate 5, named per operator, the
//! whole-run L2 bound, and the Kernel IR kernels the emitter is tested on.
//!
//! An element `g` from the device passes against the CPU interpreter's `c` when
//! `|g - c| <= GPU_ATOL + GPU_RTOL * r`. The reference `r` is an argument of [`within`], beside
//! the pair each [`Tolerance`] or [`TermTolerance`] carries: an elementwise, broadcast or SGD
//! kernel uses `r = |c|` ([`Reference::AbsCpu`]); a reduction or a matmul element uses the sum of
//! the absolute values of its terms ([`TermReference`]), with these two constants. After a run,
//! every parameter passes [`run_within`]. No other test file repeats their values.
//!
//! Not every test binary that includes this module uses all of it.

use modelspec_core::ir::kernel::{
    Epilogue, Kernel, KernelKind, KernelModuleId, MatMulTile, Operand,
};
use modelspec_core::ir::tensor::{DType, Dim, TensorType};

/// The absolute part of the GPU tolerance.
pub const GPU_ATOL: f32 = 1e-5;
/// The relative part of the GPU tolerance, scaled by the reference `r`.
pub const GPU_RTOL: f32 = 1e-4;

/// Whether `g` passes against `c` with reference `r` and the pair `atol`, `rtol`:
/// `|g - c| <= atol + rtol * r`, computed in `f64` so the check itself does not round.
pub fn within(g: f32, c: f32, r: f32, atol: f32, rtol: f32) -> bool {
    (f64::from(g) - f64::from(c)).abs() <= f64::from(atol) + f64::from(rtol) * f64::from(r)
}

/// The words after each output that no kernel may write.
pub const GUARD_WORDS: usize = 16;

/// The bits of guard word `i`.
pub fn guard(i: usize) -> u32 {
    0x5afe_0000 | i as u32
}

/// `values` followed by the [`GUARD_WORDS`] guard words. The SGD suite guards the gradient as
/// well as the parameter: the guard value is about 3.6e16, so `p - g` at a guard word only
/// changes if `g` there is large, and a gradient of exactly `count` elements read one past its
/// end would leave the parameter's guard word unchanged.
pub fn guarded(values: &[f32]) -> Vec<f32> {
    let mut words = values.to_vec();
    words.extend((0..GUARD_WORDS).map(|i| f32::from_bits(guard(i))));
    words
}

/// The fixed-seed input generator of the device tests (SplitMix64): values in [-4, 4) with
/// exact zeros, `+0.0` and `-0.0` in turn, every seventh element. Each call to
/// [`Draw::values`] moves the zeros one position on, so two operands drawn one after the other
/// never have zeros at the same index: relu-grad sees `x = ±0` with a nonzero gradient.
pub struct Draw {
    state: u64,
    calls: u64,
}

impl Draw {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            calls: 0,
        }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// `count` values; see [`Draw`].
    pub fn values(&mut self, count: u64) -> Vec<f32> {
        let zero_at = (3 + self.calls) % 7;
        self.calls += 1;
        (0..count)
            .map(|i| {
                if i % 7 == zero_at {
                    if i % 14 == zero_at { 0.0 } else { -0.0 }
                } else {
                    ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0
                }
            })
            .collect()
    }
}

/// `values` with every subnormal replaced by a zero of its sign: what a device without
/// `shaderDenormPreserveFloat32` may compute on (DESIGN.md § 9 gate 5).
pub fn flush_subnormals(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|&v| {
            if v.is_subnormal() {
                0.0f32.copysign(v)
            } else {
                v
            }
        })
        .collect()
}

/// The operators of the M1 elementwise, broadcast and SGD kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementwiseOp {
    Add,
    Sub,
    Mul,
    Relu,
    ReluGrad,
    Broadcast,
    Sgd,
}

/// How the reference `r` of [`within`] is taken from the CPU value `c`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reference {
    /// `r = |c|`: elementwise, broadcast and SGD kernels.
    AbsCpu,
}

/// The tolerance of one operator.
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    pub op: ElementwiseOp,
    pub atol: f32,
    pub rtol: f32,
    pub reference: Reference,
}

impl Tolerance {
    /// The reference `r` for the CPU value `c`.
    pub fn r(&self, c: f32) -> f32 {
        match self.reference {
            Reference::AbsCpu => c.abs(),
        }
    }

    /// Whether the device value `g` passes against the CPU value `c`.
    pub fn accepts(&self, g: f32, c: f32) -> bool {
        within(g, c, self.r(c), self.atol, self.rtol)
    }
}

impl std::fmt::Display for Tolerance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = match self.reference {
            Reference::AbsCpu => "|c|",
        };
        write!(
            f,
            "{:?}: |g - c| <= {:e} + {:e} * {r}",
            self.op, self.atol, self.rtol
        )
    }
}

/// The tolerance of `op`. Every M1 elementwise, broadcast and SGD operator has the same pair and
/// `r = |c|` (DESIGN.md § 9 gate 5).
pub fn tolerance(op: ElementwiseOp) -> Tolerance {
    let pair = |reference| Tolerance {
        op,
        atol: GPU_ATOL,
        rtol: GPU_RTOL,
        reference,
    };
    match op {
        ElementwiseOp::Add
        | ElementwiseOp::Sub
        | ElementwiseOp::Mul
        | ElementwiseOp::Relu
        | ElementwiseOp::ReluGrad
        | ElementwiseOp::Broadcast
        | ElementwiseOp::Sgd => pair(Reference::AbsCpu),
    }
}

/// One operand of a catalog case: a tensor of the case's element count, a rank-0 tensor, or an
/// immediate compiled into the kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Form {
    Tensor,
    Rank0,
    Immediate(f32),
}

/// One kernel the emitter builds, in the form the fixed safe schedule gives it.
#[derive(Clone, Debug)]
pub struct Case {
    pub name: String,
    pub op: ElementwiseOp,
    pub operands: Vec<Form>,
}

/// Every elementwise, broadcast and SGD form of M1: add, sub and mul over two tensors, with a
/// rank-0 operand on either side and with an immediate; relu; relu-grad; scalar broadcast of a
/// rank-0 tensor and of an immediate; the SGD update.
pub fn catalog() -> Vec<Case> {
    let mut cases = Vec::new();
    for (op, name) in [
        (ElementwiseOp::Add, "add"),
        (ElementwiseOp::Sub, "sub"),
        (ElementwiseOp::Mul, "mul"),
    ] {
        for (form, operands) in [
            ("tensor_tensor", vec![Form::Tensor, Form::Tensor]),
            ("tensor_rank0", vec![Form::Tensor, Form::Rank0]),
            ("rank0_tensor", vec![Form::Rank0, Form::Tensor]),
            (
                "tensor_immediate",
                vec![Form::Tensor, Form::Immediate(0.375)],
            ),
            (
                "immediate_tensor",
                vec![Form::Immediate(-1.25), Form::Tensor],
            ),
        ] {
            cases.push(Case {
                name: format!("{name}_{form}"),
                op,
                operands,
            });
        }
    }
    cases.push(Case {
        name: "relu".to_owned(),
        op: ElementwiseOp::Relu,
        operands: vec![Form::Tensor],
    });
    cases.push(Case {
        name: "relu_grad".to_owned(),
        op: ElementwiseOp::ReluGrad,
        operands: vec![Form::Tensor, Form::Tensor],
    });
    cases.push(Case {
        name: "broadcast_rank0".to_owned(),
        op: ElementwiseOp::Broadcast,
        operands: vec![Form::Rank0],
    });
    cases.push(Case {
        name: "broadcast_immediate".to_owned(),
        op: ElementwiseOp::Broadcast,
        operands: vec![Form::Immediate(2.5)],
    });
    cases.push(Case {
        name: "sgd".to_owned(),
        op: ElementwiseOp::Sgd,
        operands: vec![Form::Tensor, Form::Tensor],
    });
    cases
}

/// The Tensor IR op name and the kernel kind of `op`.
pub fn op_and_kind(op: ElementwiseOp) -> (&'static str, KernelKind) {
    match op {
        ElementwiseOp::Add => ("Add", KernelKind::Elementwise),
        ElementwiseOp::Sub => ("Sub", KernelKind::Elementwise),
        ElementwiseOp::Mul => ("Mul", KernelKind::Elementwise),
        ElementwiseOp::Relu => ("Relu", KernelKind::Elementwise),
        ElementwiseOp::ReluGrad => ("ReluGrad", KernelKind::Elementwise),
        ElementwiseOp::Broadcast => ("BroadcastScalar", KernelKind::Broadcast),
        ElementwiseOp::Sgd => ("Sub", KernelKind::SgdUpdate),
    }
}

/// The type of an operand of `form` in a case of `count` elements.
pub fn operand_type(form: Form, count: u64) -> Option<TensorType> {
    match form {
        Form::Tensor => Some(TensorType::f32(&[count])),
        Form::Rank0 => Some(TensorType::f32_scalar()),
        Form::Immediate(_) => None,
    }
}

/// The kernel of `case` over `count` elements, as `modelspec_core::schedule` builds one: kind,
/// op, operands, result, the kind's workgroup size and one workgroup per 64 elements.
pub fn kernel(case: &Case, count: u64) -> Kernel {
    let (op, kind) = op_and_kind(case.op);
    let operands = case
        .operands
        .iter()
        .map(|&form| match operand_type(form, count) {
            Some(ty) => Operand::Buffer(ty),
            None => match form {
                Form::Immediate(value) => Operand::Immediate(value),
                _ => unreachable!("only an immediate has no type"),
            },
        })
        .collect();
    let groups = count.div_ceil(u64::from(kind.workgroup_size()));
    let mut kernel = Kernel {
        id: KernelModuleId([0; 16]),
        kind,
        op,
        dtype: DType::F32,
        operands,
        result: TensorType {
            dtype: DType::F32,
            shape: vec![Dim::Static(count)],
        },
        reduces: None,
        epilogue: None,
        workgroup_size: kind.workgroup_size(),
        workgroups: [
            u32::try_from(groups).expect("a test count fits one dimension"),
            1,
            1,
        ],
    };
    kernel.id = Kernel::module_id(&kernel.signature());
    kernel
}

/// The operators of the M1 reduction, loss-sum, transpose and matmul kernels. [`KernelOp::LossSum`]
/// is the `mse` loss sum before its division, the module the bit-equality check runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelOp {
    ReduceSum,
    ReduceMean,
    Mse,
    LossSum,
    Transpose,
    MatMul,
}

/// How the reference `r` of a reduction, loss-sum, transpose or matmul element is taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TermReference {
    /// `r = |c|`: `transpose`, a copy.
    Cpu,
    /// `r` is the sum of the absolute values of the element's terms: the elements of a `sum`,
    /// the squared differences of a loss sum, the `K` products of a matmul element.
    Terms,
    /// That sum divided by the element count: `mean`, and `mse` over its squared differences.
    TermsOverCount,
}

/// The tolerance of one reduction, loss-sum, transpose or matmul operator.
#[derive(Clone, Copy, Debug)]
pub struct TermTolerance {
    pub op: KernelOp,
    pub atol: f32,
    pub rtol: f32,
    pub reference: TermReference,
}

impl TermTolerance {
    /// The reference `r` of an element with CPU value `c`, whose terms have the absolute sum
    /// `abs_terms` and number `count`.
    pub fn r(&self, c: f32, abs_terms: f64, count: u64) -> f32 {
        match self.reference {
            TermReference::Cpu => c.abs(),
            TermReference::Terms => abs_terms as f32,
            TermReference::TermsOverCount => (abs_terms / count as f64) as f32,
        }
    }

    /// Whether the device value `g` passes against the CPU value `c` with reference `r`.
    pub fn accepts(&self, g: f32, c: f32, r: f32) -> bool {
        within(g, c, r, self.atol, self.rtol)
    }
}

impl std::fmt::Display for TermTolerance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = match self.reference {
            TermReference::Cpu => "|c|",
            TermReference::Terms => "sum |term|",
            TermReference::TermsOverCount => "sum |term| / count",
        };
        write!(
            f,
            "{:?}: |g - c| <= {:e} + {:e} * {r}",
            self.op, self.atol, self.rtol
        )
    }
}

/// The tolerance of `op` (DESIGN.md § 9 gate 5): the elementwise pair, with the `r` of its terms.
pub fn term_tolerance(op: KernelOp) -> TermTolerance {
    let reference = match op {
        KernelOp::Transpose => TermReference::Cpu,
        KernelOp::ReduceSum | KernelOp::LossSum | KernelOp::MatMul => TermReference::Terms,
        KernelOp::ReduceMean | KernelOp::Mse => TermReference::TermsOverCount,
    };
    TermTolerance {
        op,
        atol: GPU_ATOL,
        rtol: GPU_RTOL,
        reference,
    }
}

/// The relative bound of the whole-run check: after a run, every parameter passes
/// `||g - c||_2 <= RUN_L2_RTOL * ||c||_2` (DESIGN.md § 9 gate 5).
pub const RUN_L2_RTOL: f32 = 1e-4;

/// Whether the device parameter `g` passes the whole-run check against the CPU parameter `c`,
/// computed in `f64`; a length mismatch fails.
pub fn run_within(g: &[f32], c: &[f32]) -> bool {
    if g.len() != c.len() {
        return false;
    }
    let difference: f64 = g
        .iter()
        .zip(c)
        .map(|(&g, &c)| (f64::from(g) - f64::from(c)).powi(2))
        .sum();
    let norm: f64 = c.iter().map(|&c| f64::from(c).powi(2)).sum();
    difference.sqrt() <= f64::from(RUN_L2_RTOL) * norm.sqrt()
}

fn finish(mut kernel: Kernel) -> Kernel {
    kernel.id = Kernel::module_id(&kernel.signature());
    kernel
}

fn groups_of(count: u64, kind: KernelKind) -> [u32; 3] {
    let groups = count.div_ceil(u64::from(kind.workgroup_size())).max(1);
    [
        u32::try_from(groups).expect("a test count fits one dimension"),
        1,
        1,
    ]
}

/// Stage 1 and stage 2 of `ReduceSum` (`mean` false) or `ReduceMean` over `count` elements, as
/// `modelspec_core::schedule` builds them: one partial per workgroup of 64, then one workgroup.
pub fn reduce_kernels(mean: bool, count: u64) -> (Kernel, Kernel) {
    let op = if mean { "ReduceMean" } else { "ReduceSum" };
    let reduced = TensorType::f32(&[count]);
    let groups = count.div_ceil(u64::from(KernelKind::ReduceStage1.workgroup_size()));
    let partials = TensorType::f32(&[groups]);
    let stage1 = finish(Kernel {
        id: KernelModuleId([0; 16]),
        kind: KernelKind::ReduceStage1,
        op,
        dtype: DType::F32,
        operands: vec![Operand::Buffer(reduced.clone())],
        result: partials.clone(),
        reduces: Some(reduced.clone()),
        epilogue: None,
        workgroup_size: KernelKind::ReduceStage1.workgroup_size(),
        workgroups: groups_of(count, KernelKind::ReduceStage1),
    });
    let stage2 = finish(Kernel {
        id: KernelModuleId([0; 16]),
        kind: KernelKind::ReduceStage2,
        op,
        dtype: DType::F32,
        operands: vec![Operand::Buffer(partials)],
        result: TensorType::f32_scalar(),
        reduces: Some(reduced),
        epilogue: None,
        workgroup_size: KernelKind::ReduceStage2.workgroup_size(),
        workgroups: [1, 1, 1],
    });
    (stage1, stage2)
}

/// The `mse` loss over two tensors of `count` elements: one workgroup of 64 invocations.
pub fn loss_sum_kernel(count: u64) -> Kernel {
    finish(Kernel {
        id: KernelModuleId([0; 16]),
        kind: KernelKind::LossSum,
        op: "Mse",
        dtype: DType::F32,
        operands: vec![
            Operand::Buffer(TensorType::f32(&[count])),
            Operand::Buffer(TensorType::f32(&[count])),
        ],
        result: TensorType::f32_scalar(),
        reduces: None,
        epilogue: None,
        workgroup_size: KernelKind::LossSum.workgroup_size(),
        workgroups: [1, 1, 1],
    })
}

/// `transpose([rows, cols]) -> [cols, rows]`, one invocation per output element.
pub fn transpose_kernel(rows: u64, cols: u64) -> Kernel {
    finish(Kernel {
        id: KernelModuleId([0; 16]),
        kind: KernelKind::Transpose,
        op: "Transpose",
        dtype: DType::F32,
        operands: vec![Operand::Buffer(TensorType::f32(&[rows, cols]))],
        result: TensorType::f32(&[cols, rows]),
        reduces: None,
        epilogue: None,
        workgroup_size: KernelKind::Transpose.workgroup_size(),
        workgroups: groups_of(rows * cols, KernelKind::Transpose),
    })
}

/// `matmul([m, k], [k, n]) -> [m, n]`, the workgroups of its [`MatMulTile`].
pub fn matmul_kernel(m: u64, k: u64, n: u64) -> Kernel {
    matmul_kernel_with(m, k, n, None)
}

/// [`matmul_kernel`] with `epilogue`.
pub fn matmul_kernel_with(m: u64, k: u64, n: u64, epilogue: Option<Epilogue>) -> Kernel {
    let tile = MatMulTile::of(m, k, n);
    finish(Kernel {
        id: KernelModuleId([0; 16]),
        kind: KernelKind::MatMul,
        op: "MatMul",
        dtype: DType::F32,
        operands: vec![
            Operand::Buffer(TensorType::f32(&[m, k])),
            Operand::Buffer(TensorType::f32(&[k, n])),
        ],
        result: TensorType::f32(&[m, n]),
        reduces: None,
        epilogue,
        workgroup_size: tile.workgroup_size,
        workgroups: [
            u32::try_from(tile.workgroups(m, n)).expect("a test matmul fits one dimension"),
            1,
            1,
        ],
    })
}

/// The element counts of the reduction and `mse` cases of `story:m1-vulkan-reduce-matmul`.
pub const REDUCE_COUNTS: [u64; 7] = [1, 63, 64, 65, 1_000, 4_097, 78_400];

/// The `[rows, cols]` of its transpose cases.
pub const TRANSPOSE_SHAPES: [(u64, u64); 4] = [(1, 1), (3, 5), (100, 784), (65, 129)];

/// The `[m, k] x [k, n]` of its matmul cases: the MNIST step shapes, shapes that fill no
/// workgroup, and, for the tile of `story:m1-f-forward-matmul-tiled` (8 rows by 16 columns,
/// tiles of 64 of `k`): exactly one tile, one past a multiple of the tile in each of `m`, `k`
/// and `n`, `k = 1`, and one column, whose tile has 128 rows and a smaller depth.
pub const MATMUL_SHAPES: [(u64, u64, u64); 13] = [
    (100, 784, 64),
    (784, 100, 64),
    (100, 64, 10),
    (64, 100, 10),
    (100, 10, 64),
    (1, 1, 1),
    (17, 33, 65),
    (8, 64, 16),
    (9, 65, 17),
    (130, 129, 33),
    (5, 1, 7),
    (129, 200, 1),
    (3, 128, 3),
];

/// The SPIR-V opcodes and enumerants the module checks read.
pub mod words {
    pub const OP_EXECUTION_MODE: u32 = 16;
    pub const OP_CAPABILITY: u32 = 17;
    pub const OP_DECORATE: u32 = 71;
    pub const OP_F_ADD: u32 = 129;
    pub const OP_F_SUB: u32 = 131;
    pub const OP_F_MUL: u32 = 133;
    pub const DECORATION_NO_CONTRACTION: u32 = 42;
    pub const MODE_DENORM_PRESERVE: u32 = 4459;
    pub const MODE_SIGNED_ZERO_INF_NAN_PRESERVE: u32 = 4461;
    pub const MODE_ROUNDING_MODE_RTE: u32 = 4462;
    /// The three float-control execution modes of DESIGN.md § 5.4.
    pub const FLOAT_CONTROL_MODES: [u32; 3] = [
        MODE_ROUNDING_MODE_RTE,
        MODE_DENORM_PRESERVE,
        MODE_SIGNED_ZERO_INF_NAN_PRESERVE,
    ];

    /// Every instruction of `module` after the header: its opcode and its words.
    pub fn instructions(module: &[u32]) -> Vec<(u32, &[u32])> {
        let mut out = Vec::new();
        let mut at = 5;
        while at < module.len() {
            let length = (module[at] >> 16) as usize;
            assert!(
                length > 0 && at + length <= module.len(),
                "a whole instruction"
            );
            out.push((module[at] & 0xffff, &module[at..at + length]));
            at += length;
        }
        out
    }

    /// Each `OpExecutionMode` of `module`: its mode and its literal operands.
    pub fn execution_modes(module: &[u32]) -> Vec<(u32, Vec<u32>)> {
        instructions(module)
            .into_iter()
            .filter(|(op, _)| *op == OP_EXECUTION_MODE)
            .map(|(_, w)| (w[2], w[3..].to_vec()))
            .collect()
    }

    /// The float-control modes `module` declares, each with its width operand.
    pub fn float_control_modes(module: &[u32]) -> Vec<(u32, Vec<u32>)> {
        execution_modes(module)
            .into_iter()
            .filter(|(mode, _)| FLOAT_CONTROL_MODES.contains(mode))
            .collect()
    }

    /// The `OpFAdd`, `OpFSub` and `OpFMul` of `module` without a `NoContraction` decoration, by
    /// result id, and how many there are in all.
    pub fn uncontracted_arithmetic(module: &[u32]) -> (Vec<u32>, usize) {
        let all = instructions(module);
        let decorated: Vec<u32> = all
            .iter()
            .filter(|(op, w)| *op == OP_DECORATE && w[2] == DECORATION_NO_CONTRACTION)
            .map(|(_, w)| w[1])
            .collect();
        let arithmetic: Vec<u32> = all
            .iter()
            .filter(|(op, _)| [OP_F_ADD, OP_F_SUB, OP_F_MUL].contains(op))
            .map(|(_, w)| w[2])
            .collect();
        let missing = arithmetic
            .iter()
            .copied()
            .filter(|id| !decorated.contains(id))
            .collect();
        (missing, arithmetic.len())
    }
}
