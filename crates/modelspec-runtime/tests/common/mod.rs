//! What the GPU tests share: the elementwise tolerance of DESIGN.md § 9 gate 5, named per
//! operator, and the Kernel IR kernels the emitter is tested on.
//!
//! An element `g` from the device passes against the CPU interpreter's `c` when
//! `|g - c| <= GPU_ATOL + GPU_RTOL * r`. The reference `r` is an argument of [`within`], beside
//! the pair each [`Tolerance`] carries: an elementwise, broadcast or SGD kernel uses `r = |c|`
//! ([`Reference::AbsCpu`]); the reduction and matmul rule of `story:m1-vulkan-reduce-matmul` adds
//! its own `r` beside it, with these two constants. No other test file repeats their values.

use modelspec_core::ir::kernel::{Kernel, KernelKind, KernelModuleId, Operand};
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
