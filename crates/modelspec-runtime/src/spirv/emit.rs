//! The SPIR-V compute module of one M1 kernel (DESIGN.md § 5.2): elementwise, broadcast, SGD,
//! the two stages of a reduction, the `mse` loss sum, transpose and matmul, and the row-wise
//! kernels of the BERT operators (`rowwise`, `per_element`, `per_row` and `per_column`), among
//! them the attention and cross-entropy kernels and the `Sqrt` and `Div` of the
//! AdamW update (`emit/attention.rs`), and the two held-out evaluation kernels `HeldOutRows` and
//! `HeldOutSum` (`emit/heldout.rs`, [`held_out_kernels`]).
//!
//! The module is SPIR-V 1.5, the version Vulkan 1.2 consumes, with capability `Shader`, the
//! `GLSL450` logical memory model and one `GLCompute` entry point `main`, with `LocalSize` the
//! kernel's workgroup size. [`emit_module`] adds the three float-control execution modes of
//! DESIGN.md § 5.4 (`RoundingModeRTE`, `DenormPreserve`, `SignedZeroInfNanPreserve`, width 32) and
//! their capabilities to a loss-sum, matmul or `HeldOutRows` module exactly when the device
//! profile reports all three for f32; no other module declares a float-control mode. A matmul or loss-sum module
//! fuses each product after the first of every sum into the sum, one `GLSL.std.450 Fma` per
//! product-and-sum, the instruction set imported once by such a module, by a row-wise module
//! that fuses a sum or takes an `Exp` or a `Sqrt`, and by no other (DESIGN.md § 12 ADR-009).
//! Every `OpFAdd`, `OpFSub`, `OpFMul` and `Fma` of a reduction, loss-sum or matmul module is
//! decorated `NoContraction`, so no driver fuses a multiply and an add or splits an `Fma`; so
//! are the multiply and the subtraction of an SGD update with three operands, `p - (lr * g)`
//! (DESIGN.md § 3.1), and every `OpFAdd`, `OpFSub`, `OpFMul`, `OpFDiv` and `Fma` of a row-wise
//! module. The modules [`emit_kernels`] builds for an
//! artifact whose `fast_math` is set carry none of these decorations, and nothing else in them
//! changes: the same `Fma` instructions, and a driver free to contract what is left; their
//! cache keys say `fast_math` (DESIGN.md § 12 ADR-008). [`emit`], [`emit_module`] and
//! [`emit_loss_sum_undivided`] build the deterministic module.
//!
//! A matmul with a `relu` epilogue applies the compare and select of the `relu` kernel to its
//! rounded sum and writes that; when it keeps the pre-activation, it writes the sum first. A
//! matmul with a `ReluGrad` epilogue reads the pre-activation `x` of its output element from its
//! third operand and writes `x > 0 ? sum : 0.0`, the compare and select of the `ReluGrad` kernel
//! (story:step-sites-beat-pytorch).
//!
//! Bindings, all in descriptor set 0: each buffer operand of the kernel in order (read, decorated
//! `NonWritable`), then what it writes: the pre-activation of an epilogue that keeps it, then the
//! result. A push constant holds the dispatch's count: the
//! output elements of an elementwise, transpose or matmul kernel, the elements a reduction stage
//! or a loss sum reads, the output rows of a `per_row` kernel (rows times heads for an attention
//! kernel) and the output columns of a
//! `per_column` one. A row-wise module reads an `i32` operand as `u32` words, and a dropout
//! module the run key from two specialization constants. A rank-0 buffer operand of an
//! elementwise kernel is read at element 0 by every invocation; an immediate operand is a
//! constant of the module.
//!
//! The order of every sum is fixed by the shape alone (DESIGN.md § 3.3). Stage 1 of a reduction
//! sums 64 consecutive elements per workgroup with a pairwise tree in workgroup memory and writes
//! one partial per workgroup; stage 2 runs one workgroup whose invocation `i` sums partials `i`,
//! `i + 64`, … in ascending order and folds those 64 sums with the same tree. Neither stage adds a
//! padding term. A matmul stages tiles of its operands in workgroup memory ([`MatMulTile`]), and
//! each output still runs in one invocation, in ascending `k` order from the first term, as the
//! CPU interpreter does; a tile that needs more workgroup memory than the device profile's
//! `max_shared_memory_bytes` is refused with [`EmitError::SharedMemory`]. The loss sum runs one
//! workgroup of 64: its invocations stage the differences `a - b` in workgroup memory, as many
//! at a time as the device profile's `max_shared_memory_bytes` holds, and invocation 0 adds
//! their squares in ascending element order from the first term, each later square fused into
//! the sum, chunk after chunk, as the CPU interpreter does. The plan's module divides the sum by
//! the element count, and also adds the squares of the differences scaled by the reciprocal of
//! the count, the mean, which it writes instead where the sum overflows (DESIGN.md § 5.4);
//! [`emit_loss_sum_undivided`] writes the sum.

use std::collections::{BTreeMap, VecDeque};

use modelspec_core::artifact::{CompiledArtifact, ContentHash};
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{
    ElementwiseChain, Epilogue, Kernel, KernelKind, KernelModuleId, MatMulLayout, MatMulTile,
    Operand,
};
use modelspec_core::ir::tensor::{DType, Dim, TensorType};
use rspirv::binary::Assemble;
use rspirv::dr::{self, Builder};
use rspirv::spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel,
    FunctionControl, GlslStd450Op, LoopControl, MemoryModel, SelectionControl, StorageClass, Word,
};

mod attention;
mod heldout;
mod rowwise;

pub use heldout::{COUNT_LIMB, HELD_OUT_STATE, HeldOutKernels, held_out_kernels, held_out_sums};

/// Why a kernel has no module.
#[derive(Debug, thiserror::Error)]
pub enum EmitError {
    /// The kernel's kind and op are not an M1 kernel the emitter builds.
    #[error("no SPIR-V emitter for {kind} kernel {op}")]
    Unsupported { kind: KernelKind, op: &'static str },
    /// The kernel's operands do not fit its op.
    #[error("{op} kernel has operands the emitter cannot bind: {reason}")]
    Operands { op: &'static str, reason: String },
    /// The kernel stages more workgroup memory than the device provides.
    #[error("{op} kernel stages {needs} bytes of workgroup memory, the device provides {provides}")]
    SharedMemory {
        op: &'static str,
        needs: u32,
        provides: u32,
    },
    /// `rspirv` refused an instruction.
    #[error("the SPIR-V builder refused the module: {0}")]
    Build(String),
}

impl From<dr::Error> for EmitError {
    fn from(error: dr::Error) -> Self {
        Self::Build(error.to_string())
    }
}

/// The capabilities a module with the three float-control modes declares besides `Shader`, in
/// the order its [`KernelCacheKey`] names them.
pub const FLOAT_CONTROL_CAPABILITIES: [&str; 3] = [
    "RoundingModeRTE",
    "DenormPreserve",
    "SignedZeroInfNanPreserve",
];

/// What a compiled module depends on (DESIGN.md § 5.1): the kernel, its shapes and dtype, the
/// compile settings, the device and driver, and the capabilities the module declares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelCacheKey {
    /// blake3 of the kernel's signature and the module variant.
    pub ir_hash: ContentHash,
    /// The shape of each buffer operand in binding order, then each written binding's.
    pub shapes: Vec<Vec<u64>>,
    pub dtype: DType,
    pub fast_math: bool,
    pub deterministic: bool,
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver_version: u32,
    pub capabilities: Vec<String>,
}

/// The SPIR-V module of one kernel.
#[derive(Clone, Debug)]
pub struct KernelModule {
    pub kernel_module_id: KernelModuleId,
    pub entry_point: String,
    pub spirv: Vec<u32>,
    pub cache_key: KernelCacheKey,
}

impl KernelModule {
    /// The module's words as little-endian bytes, as a shader module is created from.
    pub fn bytes(&self) -> Vec<u8> {
        self.spirv
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect()
    }
}

/// Which module of a loss-sum kernel is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    /// The plan's module: for a loss sum, the sum of the squared differences divided by the
    /// element count, or where that sum overflows the sum of the scaled ones.
    Plan,
    /// The loss sum before its division, the value of the bit-for-bit class.
    Undivided,
}

/// The workgroup memory every Vulkan device provides, the minimum of
/// `maxComputeSharedMemorySize`: what a module built without a device profile stages into.
pub use modelspec_core::ir::kernel::MIN_SHARED_MEMORY_BYTES;

/// The SPIR-V words of `kernel`'s compute module, with no float-control mode, for a device with
/// [`MIN_SHARED_MEMORY_BYTES`] of workgroup memory.
pub fn emit(kernel: &Kernel) -> Result<Vec<u32>, EmitError> {
    Ok(build(kernel, Variant::Plan, false, MIN_SHARED_MEMORY_BYTES, false)?.assemble())
}

/// The module of `kernel` for a device with `profile`.
pub fn emit_module(kernel: &Kernel, profile: &DeviceProfile) -> Result<KernelModule, EmitError> {
    module_for(kernel, profile, Variant::Plan, false)
}

/// [`emit_module`] under `fast_math` (DESIGN.md § 12 ADR-008), as [`emit_kernels`] emits a kernel
/// of an artifact: what a runner needs for a kernel that is not in the plan, such as the `p = 0`
/// copy of a `Dropout` the device run evaluates its loss without dropout with.
pub fn emit_module_with(
    kernel: &Kernel,
    profile: &DeviceProfile,
    fast_math: bool,
) -> Result<KernelModule, EmitError> {
    module_for(kernel, profile, Variant::Plan, fast_math)
}

/// The module of the loss-sum `kernel` without its division by the element count: the sum of
/// squared differences the bit-for-bit class of DESIGN.md § 5.4 compares.
pub fn emit_loss_sum_undivided(
    kernel: &Kernel,
    profile: &DeviceProfile,
) -> Result<KernelModule, EmitError> {
    if kernel.kind != KernelKind::LossSum {
        return Err(EmitError::Unsupported {
            kind: kernel.kind,
            op: kernel.op,
        });
    }
    module_for(kernel, profile, Variant::Undivided, false)
}

/// One module per kernel of `artifact`'s execution plan, in plan order; none for a CPU artifact.
pub fn emit_kernels(
    artifact: &CompiledArtifact,
    profile: &DeviceProfile,
) -> Result<Vec<KernelModule>, EmitError> {
    artifact
        .execution
        .kernels
        .iter()
        .map(|kernel| module_for(kernel, profile, Variant::Plan, artifact.fast_math))
        .collect()
}

/// Whether `profile` reports all three f32 float controls of DESIGN.md § 5.4, the rule
/// `negotiate` gives the bit-for-bit class by.
fn reports_float_controls(profile: &DeviceProfile) -> bool {
    [
        profile.shader_rounding_mode_rte_f32,
        profile.shader_denorm_preserve_f32,
        profile.shader_signed_zero_inf_nan_preserve_f32,
    ]
    .iter()
    .all(|value| *value == Some(true))
}

fn module_for(
    kernel: &Kernel,
    profile: &DeviceProfile,
    variant: Variant,
    fast_math: bool,
) -> Result<KernelModule, EmitError> {
    // The row losses of `CrossEntropy` keep the float controls of the loss sum they feed.
    let float_controls = (matches!(kernel.kind, KernelKind::LossSum | KernelKind::MatMul)
        || matches!(
            (kernel.kind, kernel.op),
            (KernelKind::PerRow, "HeldOutRows" | "CrossEntropy")
        ))
        && reports_float_controls(profile);
    let body = body_of(kernel, variant, profile.max_shared_memory_bytes)?;
    let chunk = match body {
        Body::LossSum { chunk, .. } => format!(" chunk {chunk}"),
        // The padded staging is taken only when it fits the profile's workgroup memory, so a
        // module that takes it records its strides: the unpadded layout follows from the
        // signature, and two modules never share a cache key.
        Body::MatMul { tile, layout, .. } if layout != tile.layout([false, false]) => format!(
            " staging a_stride {} b_stride {}",
            layout.a_stride, layout.b_stride
        ),
        _ => String::new(),
    };
    let spirv = build(
        kernel,
        variant,
        float_controls,
        profile.max_shared_memory_bytes,
        fast_math,
    )?
    .assemble();
    let shapes = kernel
        .bindings()
        .into_iter()
        .map(|ty| static_shape(kernel, ty))
        .collect::<Result<Vec<_>, _>>()?;
    let mut capabilities = vec!["Shader".to_owned()];
    if float_controls {
        capabilities.extend(
            FLOAT_CONTROL_CAPABILITIES
                .iter()
                .map(|name| (*name).to_owned()),
        );
    }
    let tag = match variant {
        Variant::Plan => "plan",
        Variant::Undivided => "undivided",
    };
    let ir_hash = ContentHash(
        blake3::hash(
            format!(
                "modelspec-kernel-module/1\n{}\n{tag}{chunk}",
                kernel.signature()
            )
            .as_bytes(),
        )
        .to_hex()
        .to_string(),
    );
    Ok(KernelModule {
        kernel_module_id: kernel.id,
        entry_point: "main".to_owned(),
        spirv,
        cache_key: KernelCacheKey {
            ir_hash,
            shapes,
            dtype: kernel.dtype,
            fast_math,
            deterministic: true,
            vendor_id: profile.vendor_id,
            device_id: profile.device_id,
            driver_version: profile.driver_version,
            capabilities,
        },
    })
}

fn operands_error(kernel: &Kernel, reason: impl Into<String>) -> EmitError {
    EmitError::Operands {
        op: kernel.op,
        reason: reason.into(),
    }
}

fn static_shape(kernel: &Kernel, ty: &TensorType) -> Result<Vec<u64>, EmitError> {
    ty.shape
        .iter()
        .map(|dim| match dim {
            Dim::Static(n) => Ok(*n),
            Dim::Symbol(_) => Err(operands_error(kernel, format!("{ty} is not static"))),
        })
        .collect()
}

fn u32_of(kernel: &Kernel, value: u64) -> Result<u32, EmitError> {
    u32::try_from(value).map_err(|_| operands_error(kernel, format!("{value} does not fit u32")))
}

/// The shapes of `kernel`'s buffer operands; an immediate operand is refused.
fn buffer_shapes(kernel: &Kernel, arity: usize) -> Result<Vec<Vec<u64>>, EmitError> {
    if kernel.operands.len() != arity {
        return Err(operands_error(
            kernel,
            format!("{} operands, the op takes {arity}", kernel.operands.len()),
        ));
    }
    kernel
        .operands
        .iter()
        .map(|operand| match operand {
            Operand::Buffer(ty) => static_shape(kernel, ty),
            Operand::Immediate(_) => Err(operands_error(kernel, "an immediate operand")),
        })
        .collect()
}

/// What one module computes, its constants taken from the kernel's shapes.
enum Body {
    Elementwise(Arith, Vec<Source>),
    /// A fused elementwise chain ([`ElementwiseChain`]): each op and its inputs, in order.
    Chain(Vec<(Arith, Vec<Input>)>),
    Stage1,
    Stage2 {
        divide_by: Option<u32>,
    },
    /// `terms` is the element count of the shape and `chunk` the number of terms staged in
    /// workgroup memory at a time; `divide_by`, the element count the sum is divided by, whose
    /// reciprocal scales the terms of the sum written where it overflows ([`Shell::loss_sum`]),
    /// is `None` for the undivided sum.
    LossSum {
        divide_by: Option<u32>,
        terms: u32,
        chunk: u32,
    },
    Transpose {
        rows: u32,
        cols: u32,
    },
    MatMul {
        m: u32,
        k: u32,
        n: u32,
        epilogue: Option<Epilogue>,
        tile: MatMulTile,
        /// Whether `a` and `b` are bound transposed ([`Kernel::transposed`]).
        transposed: [bool; 2],
        /// Where the tile's values lie in workgroup memory ([`MatMulTile::layout_within`]).
        layout: MatMulLayout,
    },
    /// An attention, cross-entropy or cross-entropy gradient kernel
    /// ([`attention::AttentionLoss`]).
    AttentionLoss(attention::AttentionLoss),
    /// A row-wise kernel ([`rowwise::Rowwise`]).
    Rowwise(rowwise::Rowwise),
    /// A held-out evaluation kernel ([`heldout::HeldOutBody`]).
    HeldOut(heldout::HeldOutBody),
}

/// What `kernel` computes, for a device with `shared_bytes` of workgroup memory.
fn body_of(kernel: &Kernel, variant: Variant, shared_bytes: u32) -> Result<Body, EmitError> {
    let unsupported = || EmitError::Unsupported {
        kind: kernel.kind,
        op: kernel.op,
    };
    let elements = |shape: &[u64]| shape.iter().product::<u64>();
    let body = match (kernel.kind, kernel.op) {
        (KernelKind::Elementwise | KernelKind::Broadcast | KernelKind::SgdUpdate, _) => {
            elementwise(kernel)?
        }
        (
            KernelKind::PerRow,
            "Attention" | "AttentionGradQ" | "AttentionGradK" | "AttentionGradV" | "CrossEntropy"
            | "CrossEntropyGrad",
        )
        | (KernelKind::LossSum, "CrossEntropy")
        | (KernelKind::PerElement, "CrossEntropyGrad") => {
            // Only the `mse` loss sum has an undivided module (DESIGN.md § 5.4).
            if variant == Variant::Undivided {
                return Err(unsupported());
            }
            Body::AttentionLoss(attention::AttentionLoss::of(kernel)?)
        }
        (KernelKind::PerRow, "HeldOutRows") | (KernelKind::LossSum, "HeldOutSum") => {
            if variant == Variant::Undivided {
                return Err(unsupported());
            }
            Body::HeldOut(heldout::HeldOutBody::of(kernel)?)
        }
        (KernelKind::PerElement | KernelKind::PerRow | KernelKind::PerColumn, _) => {
            Body::Rowwise(rowwise::Rowwise::of(kernel)?)
        }
        (KernelKind::ReduceStage1 | KernelKind::ReduceStage2, "ReduceSum" | "ReduceMean") => {
            if kernel.workgroup_size != TREE {
                return Err(operands_error(
                    kernel,
                    format!("a reduction stage runs workgroups of {TREE}"),
                ));
            }
            let shapes = buffer_shapes(kernel, 1)?;
            let reduces = kernel
                .reduces
                .as_ref()
                .ok_or_else(|| operands_error(kernel, "a reduction stage names what it reduces"))?;
            let n = elements(&static_shape(kernel, reduces)?);
            let groups = n.div_ceil(u64::from(TREE));
            let result = elements(&static_shape(kernel, &kernel.result)?);
            if kernel.kind == KernelKind::ReduceStage1 {
                if elements(&shapes[0]) != n || result != groups {
                    return Err(operands_error(
                        kernel,
                        format!("stage 1 reads {n} elements into {groups} partials"),
                    ));
                }
                Body::Stage1
            } else {
                if shapes[0] != [groups] || result != 1 || kernel.result.rank() != 0 {
                    return Err(operands_error(
                        kernel,
                        format!("stage 2 reads f32[{groups}] into a rank-0 result"),
                    ));
                }
                let divide_by = match kernel.op {
                    "ReduceMean" => Some(u32_of(kernel, n)?),
                    _ => None,
                };
                Body::Stage2 { divide_by }
            }
        }
        (KernelKind::LossSum, "Mse") => {
            let shapes = buffer_shapes(kernel, 2)?;
            if shapes[0] != shapes[1] || kernel.result.rank() != 0 {
                return Err(operands_error(
                    kernel,
                    "mse reads two tensors of one shape into a rank-0 result",
                ));
            }
            let n = u32_of(kernel, elements(&shapes[0]))?;
            let divide_by = match variant {
                Variant::Plan => Some(n),
                Variant::Undivided => None,
            };
            if kernel.workgroup_size != TREE {
                return Err(operands_error(
                    kernel,
                    format!("the loss sum runs a workgroup of {TREE}"),
                ));
            }
            let fits = shared_bytes / 4;
            if fits == 0 {
                return Err(operands_error(
                    kernel,
                    format!("{shared_bytes} bytes of workgroup memory hold no f32 term"),
                ));
            }
            Body::LossSum {
                divide_by,
                terms: n,
                chunk: n.clamp(1, fits),
            }
        }
        (KernelKind::Transpose, "Transpose") => {
            let shapes = buffer_shapes(kernel, 1)?;
            let result = static_shape(kernel, &kernel.result)?;
            match (shapes[0].as_slice(), result.as_slice()) {
                ([rows, cols], [r_cols, r_rows]) if rows == r_rows && cols == r_cols => {
                    Body::Transpose {
                        rows: u32_of(kernel, *rows)?,
                        cols: u32_of(kernel, *cols)?,
                    }
                }
                _ => {
                    return Err(operands_error(kernel, "transpose reads [m, n] into [n, m]"));
                }
            }
        }
        (KernelKind::MatMul, "MatMul") => {
            let shapes = buffer_shapes(kernel, Epilogue::matmul_operands(kernel.epilogue))?;
            let result = static_shape(kernel, &kernel.result)?;
            match (
                shapes[0].as_slice(),
                shapes[1].as_slice(),
                result.as_slice(),
            ) {
                ([m, k], [k2, n], [rm, rn])
                    if k == k2
                        && m == rm
                        && n == rn
                        && *k > 0
                        && shapes.get(2).is_none_or(|x| *x == result) =>
                {
                    let tile = MatMulTile::of_invocations(*m, *k, *n, kernel.workgroup_size);
                    let covered = kernel
                        .workgroups
                        .iter()
                        .map(|&w| u64::from(w))
                        .product::<u64>();
                    if kernel.workgroup_size != tile.workgroup_size
                        || covered < tile.workgroups(*m, *n)
                    {
                        return Err(operands_error(
                            kernel,
                            format!(
                                "a [{m}, {k}] x [{k}, {n}] matmul runs {} workgroups of {}",
                                tile.workgroups(*m, *n),
                                tile.workgroup_size
                            ),
                        ));
                    }
                    let layout = tile.layout_within(kernel.transposed, shared_bytes);
                    let needs = layout.shared_bytes();
                    if needs > shared_bytes {
                        return Err(EmitError::SharedMemory {
                            op: kernel.op,
                            needs,
                            provides: shared_bytes,
                        });
                    }
                    u32_of(kernel, m.saturating_mul(*k).max(k.saturating_mul(*n)))?;
                    Body::MatMul {
                        m: u32_of(kernel, *m)?,
                        k: u32_of(kernel, *k)?,
                        n: u32_of(kernel, *n)?,
                        epilogue: kernel.epilogue,
                        tile,
                        transposed: kernel.transposed,
                        layout,
                    }
                }
                _ => {
                    return Err(operands_error(
                        kernel,
                        "matmul reads [m, k] and [k, n], k > 0, and with a ReluGrad epilogue [m, n], into [m, n]",
                    ));
                }
            }
        }
        _ => return Err(unsupported()),
    };
    if variant == Variant::Undivided && kernel.kind != KernelKind::LossSum {
        return Err(unsupported());
    }
    Ok(body)
}

/// The arithmetic of the elementwise, broadcast and SGD kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arith {
    Add,
    Sub,
    Mul,
    Relu,
    ReluGrad,
    /// `a - (b * c)`, the SGD update with its scaled gradient: the product rounded, then the
    /// difference, both decorated `NoContraction`.
    ScaledSub,
    /// `sqrt(a)`, the device's root corrected by one `Fma` residual step ([`Shell::root`]).
    Sqrt,
    /// `a / b`, the device's quotient corrected by one `Fma` residual step
    /// ([`Shell::quotient`]).
    Div,
    Copy,
}

impl Arith {
    fn of(kernel: &Kernel) -> Result<Self, EmitError> {
        let arith = match (kernel.kind, kernel.op) {
            (KernelKind::Elementwise, "Add") => Self::Add,
            (KernelKind::SgdUpdate, "Sub") if kernel.operands.len() == 3 => Self::ScaledSub,
            (KernelKind::Elementwise, "Sub") | (KernelKind::SgdUpdate, "Sub") => Self::Sub,
            (KernelKind::Elementwise, "Mul") => Self::Mul,
            (KernelKind::Elementwise, "Relu") => Self::Relu,
            (KernelKind::Elementwise, "ReluGrad") => Self::ReluGrad,
            (KernelKind::Elementwise, "Sqrt") => Self::Sqrt,
            (KernelKind::Elementwise, "Div") => Self::Div,
            (KernelKind::Broadcast, "BroadcastScalar") => Self::Copy,
            (kind, op) => return Err(EmitError::Unsupported { kind, op }),
        };
        Ok(arith)
    }

    fn arity(self) -> usize {
        match self {
            Self::Relu | Self::Copy | Self::Sqrt => 1,
            Self::Add | Self::Sub | Self::Mul | Self::ReluGrad | Self::Div => 2,
            Self::ScaledSub => 3,
        }
    }
}

/// How one operand of an elementwise kernel is read.
#[derive(Clone, Copy)]
enum Source {
    /// Buffer operand `binding`, at the invocation's index or, for a rank-0 tensor, at 0.
    Buffer { binding: u32, scalar: bool },
    /// A constant of the module.
    Immediate(f32),
}

/// One input of an op of a fused elementwise chain.
#[derive(Clone, Copy)]
enum Input {
    /// A leaf of the chain, an operand of the kernel.
    Source(Source),
    /// The result of the op before it.
    Previous,
}

/// The body of an `elementwise` kernel whose op names the chain `chain`: its operands are the
/// chain's leaves, bound in order.
fn chain_body(kernel: &Kernel, chain: &ElementwiseChain) -> Result<Body, EmitError> {
    if kernel.operands.len() != chain.operands() {
        return Err(operands_error(
            kernel,
            format!(
                "{} operands, the chain takes {}",
                kernel.operands.len(),
                chain.operands()
            ),
        ));
    }
    let mut binding = 0;
    let mut leaves = kernel.operands.iter().map(|operand| match operand {
        Operand::Buffer(ty) => {
            let source = Source::Buffer {
                binding,
                scalar: ty.rank() == 0,
            };
            binding += 1;
            Input::Source(source)
        }
        Operand::Immediate(value) => Input::Source(Source::Immediate(*value)),
    });
    let arith = |op: &str| match op {
        "Add" => Arith::Add,
        "Sub" => Arith::Sub,
        "Mul" => Arith::Mul,
        "Relu" => Arith::Relu,
        _ => Arith::ReluGrad,
    };
    let mut ops = Vec::new();
    let first = arith(chain.first);
    ops.push((first, leaves.by_ref().take(first.arity()).collect()));
    for link in &chain.links {
        let op = arith(link.op);
        let inputs = (0..op.arity())
            .map(|i| match i == link.at {
                true => Some(Input::Previous),
                false => leaves.next(),
            })
            .collect::<Option<Vec<Input>>>()
            .expect("the operand count is the chain's");
        ops.push((op, inputs));
    }
    Ok(Body::Chain(ops))
}

fn elementwise(kernel: &Kernel) -> Result<Body, EmitError> {
    if kernel.kind == KernelKind::Elementwise
        && let Some(chain) = ElementwiseChain::parse(kernel.op)
    {
        return chain_body(kernel, &chain);
    }
    let arith = Arith::of(kernel)?;
    if kernel.operands.len() != arith.arity() {
        return Err(operands_error(
            kernel,
            format!(
                "{} operands, the op takes {}",
                kernel.operands.len(),
                arith.arity()
            ),
        ));
    }
    let mut binding = 0;
    let sources: Vec<Source> = kernel
        .operands
        .iter()
        .map(|operand| match operand {
            Operand::Buffer(ty) => {
                let source = Source::Buffer {
                    binding,
                    scalar: ty.rank() == 0,
                };
                binding += 1;
                source
            }
            Operand::Immediate(value) => Source::Immediate(*value),
        })
        .collect();
    if arith == Arith::Copy && matches!(sources[0], Source::Buffer { scalar: false, .. }) {
        return Err(operands_error(
            kernel,
            "a scalar broadcast reads a rank-0 tensor or an immediate",
        ));
    }
    Ok(Body::Elementwise(arith, sources))
}

/// The workgroup size of both reduction stages and the width of their tree.
const TREE: u32 = 64;

/// The iterations of one loop pass of the loss sum's staging and adding loops
/// ([`Shell::repeat`]), so the driver can issue their loads ahead of the adds.
const UNROLL: u32 = 8;

/// The strides of the pairwise tree over [`TREE`] values.
const TREE_STRIDES: [u32; 6] = [32, 16, 8, 4, 2, 1];

/// `Workgroup` scope.
const SCOPE_WORKGROUP: u32 = 2;

/// `AcquireRelease | WorkgroupMemory`.
const SEMANTICS_WORKGROUP: u32 = 0x8 | 0x100;

fn build(
    kernel: &Kernel,
    variant: Variant,
    float_controls: bool,
    shared_bytes: u32,
    fast_math: bool,
) -> Result<dr::Module, EmitError> {
    let body = body_of(kernel, variant, shared_bytes)?;
    // One flag per read binding: whether it holds `u32` words, an `i32` operand's bits. The
    // softmax statistics a backward attention kernel reads follow its buffer operands.
    let reads: Vec<bool> = kernel
        .operands
        .iter()
        .filter_map(|operand| match operand {
            Operand::Buffer(ty) => Some(ty.dtype == DType::I32),
            Operand::Immediate(_) => None,
        })
        .chain(kernel.reads_softmax_stats().then_some(false))
        .collect();
    let shared = match body {
        Body::Stage1 | Body::Stage2 { .. } => Some((TREE, false)),
        Body::LossSum { chunk, .. } => Some((chunk, false)),
        Body::MatMul { layout, tile, .. } => Some((layout.shared_values(), tile.vector_reads())),
        Body::Rowwise(rowwise) => rowwise
            .shared()
            .map(|values| (values, rowwise.shared_vectors())),
        Body::AttentionLoss(body) => body.shared().map(|values| (values, false)),
        Body::HeldOut(body) => body.shared().map(|values| (values, false)),
        _ => None,
    };
    let matmul_vars = match body {
        Body::MatMul { k, tile, .. } => matmul_accumulators(&tile) + matmul_carried(k, &tile),
        Body::Rowwise(rowwise) => rowwise.f32_vars(),
        Body::AttentionLoss(body) => body.f32_vars(),
        Body::HeldOut(body) => body.f32_vars(),
        // The scaled sum of the plan's loss sum ([`Shell::loss_sum`]).
        Body::LossSum {
            divide_by: Some(_), ..
        } => 1,
        _ => 0,
    };
    let seeded = matches!(body, Body::Rowwise(rowwise) if rowwise.seeded());
    let u32_vars = match body {
        Body::AttentionLoss(body) => body.u32_vars(),
        Body::HeldOut(body) => body.u32_vars(),
        Body::Rowwise(rowwise) => rowwise.u32_vars(),
        _ => 0,
    };
    if kernel.epilogue.is_some() && !matches!(body, Body::MatMul { .. }) {
        return Err(operands_error(
            kernel,
            "only a matmul kernel has an epilogue",
        ));
    }
    // Every float add, subtract, multiply and fused multiply-add of a reduction, loss-sum,
    // transpose, matmul, row-wise, attention or cross-entropy module, the product and difference
    // of an SGD update with three operands, each op of a fused elementwise chain and every one of
    // a `Sqrt` or `Div` module is decorated `NoContraction`; none is in a `fast_math` module
    // (DESIGN.md § 3.1, § 5.4, § 12 ADR-008, ADR-009).
    let exact = !fast_math
        && match body {
            Body::Elementwise(arith, _) => {
                matches!(arith, Arith::ScaledSub | Arith::Sqrt | Arith::Div)
            }
            _ => true,
        };
    let mut m = Shell::new(
        &reads,
        kernel.written(),
        kernel.workgroup_size,
        shared,
        matmul_vars,
        u32_vars,
        float_controls,
        exact,
        seeded,
    )?;
    match body {
        Body::Elementwise(arith, sources) => m.elementwise(arith, &sources)?,
        Body::Chain(ops) => m.chain(&ops)?,
        Body::Stage1 => m.stage1()?,
        Body::Stage2 { divide_by } => m.stage2(divide_by)?,
        Body::LossSum {
            divide_by,
            terms,
            chunk,
        } => m.loss_sum(divide_by, terms, chunk)?,
        Body::Transpose { rows, cols } => m.transpose(rows, cols)?,
        Body::MatMul {
            m: rows,
            k,
            n,
            epilogue,
            tile,
            transposed,
            layout,
        } => m.matmul([rows, k, n], epilogue, tile, transposed, layout)?,
        Body::Rowwise(rowwise) => m.rowwise(rowwise)?,
        Body::AttentionLoss(body) => m.attention_loss(body)?,
        Body::HeldOut(body) => m.held_out(body)?,
    }
    m.finish()
}

/// The tiles of a matmul's contracting index each pass of its tile loop computes. Even, so the
/// workgroup memory buffer of a tile in the loop is a constant of its place in the pass.
const MATMUL_UNROLL: u32 = 4;

/// The passes of the tile loop of a `k`-deep matmul with `tile`. Tile `i` is in the loop when
/// it is not the first, and it and tile `i + ahead`, the one it reads ahead, are full; the
/// loop covers the first [`MATMUL_UNROLL`] times as many of them, from tile 1 up, and the rest
/// are emitted straight. So a module holds at most `MATMUL_UNROLL + ahead + 1` tile bodies
/// besides the first, whatever `k` is.
fn matmul_passes(k: u32, tile: &MatMulTile) -> u32 {
    let full = k / tile.depth;
    full.saturating_sub(tile.ahead.max(1) + 1) / MATMUL_UNROLL
}

/// The accumulators of one invocation: one per output element of its block.
fn matmul_accumulators(tile: &MatMulTile) -> usize {
    (tile.rows * tile.cols_per_invocation) as usize
}

/// The values one invocation reads per tile: its `a` slots, then its `b` slots.
fn matmul_slots(tile: &MatMulTile) -> usize {
    let size = tile.workgroup_size;
    ((tile.rows_per_workgroup() * tile.depth).div_ceil(size)
        + (tile.depth * tile.cols).div_ceil(size)) as usize
}

/// The `f32` function variables that carry the tiles read ahead from one pass of the tile loop
/// to the next: `ahead - 1` tiles of [`matmul_slots`] values, none without a loop.
fn matmul_carried(k: u32, tile: &MatMulTile) -> usize {
    if matmul_passes(k, tile) == 0 {
        return 0;
    }
    (tile.ahead.max(1) as usize - 1) * matmul_slots(tile)
}

/// The values of the block's rows a transposed `a` stages from one value of `k` before the next:
/// 8 values of `f32`, one 32-byte sector of its binding.
const TRANSPOSED_A_RUN: u32 = 8;

/// The order in which a workgroup stages one operand's tile, and where each staged value lies
/// in the operand's binding (DESIGN.md § 5.2). Staging element `e`, `0 ≤ e <` the tile's cells,
/// is element `e / size` of invocation `e % size`'s slots, so consecutive invocations take
/// consecutive elements. Element `e` has an index `along` the contracting index, within the
/// tile, and an index `across` the block's rows (`a`) or columns (`b`); the binding holds its
/// value at `across · across_stride + along · along_stride`, with `across` counted from the
/// block's first row or column and `along` from the tile's first `k`. Runs of `run` consecutive
/// elements take consecutive values of the binding's contiguous index, so they read consecutive
/// addresses.
///
/// An operand read as is is contiguous along `k` (`a`) or across the columns (`b`), and a run
/// spans the tile's whole extent of that index. An operand bound transposed is contiguous along
/// the other index: a transposed `b` along `k`, in runs of the tile's depth, and a transposed
/// `a` across the block's rows, in runs of [`TRANSPOSED_A_RUN`] rows, then `k`, then the next
/// run of rows. Workgroup memory is filled in the order of the operand read as is, a transposed
/// operand's lines at the padded stride of [`MatMulTile::layout`], so that the 32 stores of a
/// subgroup fall into 32 different banks when the device has the memory for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Staging {
    /// The consecutive elements that read consecutive addresses.
    run: u32,
    /// Whether the contiguous index is `along` rather than `across`.
    along_first: bool,
    /// The values of `along` in the tile: its depth.
    depth: u32,
    /// The values of `across` in the tile: the block's rows (`a`) or columns (`b`).
    across: u32,
    /// The binding's distance between two consecutive values of `k`.
    along_stride: u32,
    /// The binding's distance between two consecutive rows (`a`) or columns (`b`).
    across_stride: u32,
}

impl Staging {
    /// The staging of operand `a` (`is_a`) or `b` of the `[m, k] x [k, n]` matmul with `tile`,
    /// bound as is or `transposed`: `a` as `[m, k]` or `[k, m]`, `b` as `[k, n]` or `[n, k]`.
    fn of(tile: &MatMulTile, m: u32, k: u32, n: u32, is_a: bool, transposed: bool) -> Self {
        let depth = tile.depth;
        let rows = tile.rows_per_workgroup();
        match (is_a, transposed) {
            (true, false) => Self {
                run: depth,
                along_first: true,
                depth,
                across: rows,
                along_stride: 1,
                across_stride: k,
            },
            (true, true) => Self {
                run: if rows.is_multiple_of(TRANSPOSED_A_RUN) {
                    TRANSPOSED_A_RUN
                } else {
                    rows
                },
                along_first: false,
                depth,
                across: rows,
                along_stride: m,
                across_stride: 1,
            },
            (false, false) => Self {
                run: tile.cols,
                along_first: false,
                depth,
                across: tile.cols,
                along_stride: n,
                across_stride: 1,
            },
            (false, true) => Self {
                run: depth,
                along_first: true,
                depth,
                across: tile.cols,
                along_stride: 1,
                across_stride: k,
            },
        }
    }

    /// `(along, across)` of staging element `e`.
    #[cfg(test)]
    fn split(&self, e: u32) -> (u32, u32) {
        let (slow, fast) = (e / self.run, e % self.run);
        if self.along_first {
            (fast, slow)
        } else if self.run == self.across {
            (slow, fast)
        } else {
            (slow % self.depth, slow / self.depth * self.run + fast)
        }
    }

    /// The binding element staging element `e` reads in the block whose first row or column
    /// is `origin`, for the tile whose first `k` is `base`.
    #[cfg(test)]
    fn element(&self, e: u32, origin: u32, base: u32) -> u64 {
        let (along, across) = self.split(e);
        u64::from(origin + across) * u64::from(self.across_stride)
            + u64::from(base + along) * u64::from(self.along_stride)
    }
}

/// A module under construction, inside the entry block of `main` once [`Shell::new`] returns.
struct Shell {
    b: Builder,
    bool_ty: Word,
    u32_ty: Word,
    f32_ty: Word,
    element_ptr: Word,
    shared_ptr: Word,
    buffers: Vec<Word>,
    /// The written bindings, after the read ones; the last is the result.
    outs: Vec<Word>,
    shared: Option<Word>,
    /// The `vec4` of `f32` and its workgroup pointer when workgroup memory is an array of them
    /// ([`MatMulTile::vector_reads`]); `None` when it is an array of `f32`.
    shared_vectors: Option<(Word, Word)>,
    /// The invocation's flat index over every workgroup.
    index: Word,
    /// The invocation's index in its workgroup, when the module uses workgroup memory.
    local: Option<Word>,
    /// The push-constant count.
    count: Word,
    /// Two `u32` and one `f32` function variables, declared first in the entry block.
    u32_vars: [Word; 2],
    f32_var: Word,
    /// The `f32` function variables of a matmul invocation: its per-row accumulators, then the
    /// tiles its tile loop carries from one pass to the next ([`matmul_carried`]); of a
    /// row-wise kernel, the accumulators it keeps beside `f32_var`.
    matmul_vars: Vec<Word>,
    /// The `u32` function variables a module keeps beside [`Shell::u32_vars`], declared after
    /// the `f32` ones and only by an attention or cross-entropy module, so the words of every
    /// other module are unchanged.
    extra_u32_vars: Vec<Word>,
    /// The workgroup size, the `LocalSize` x of the entry point.
    group: Word,
    /// Whether [`Shell::f_add`], [`Shell::f_sub`], [`Shell::f_mul`], [`Shell::f_div`] and
    /// [`Shell::f_fma`] decorate their result `NoContraction`.
    exact: bool,
    /// The `GLSL.std.450` instruction set, imported by the first extended instruction of the
    /// module ([`Shell::f_fma`], [`Shell::exp`], [`Shell::sqrt`]), and by no other module.
    glsl: Option<Word>,
    /// The pointer to one `u32` element of a storage buffer: an `i32` operand's ids or RNG
    /// token, read as their bit patterns; declared only by a module that reads one, so the words
    /// of every other module are unchanged.
    u32_element_ptr: Option<Word>,
    /// The run key's low and high 32 bits, the specialization constants of a module that reads
    /// it (`crate::vulkan::pipeline::SEED_SPEC_IDS`).
    seed: Option<[Word; 2]>,
    u32_constants: BTreeMap<u32, Word>,
    f32_constants: BTreeMap<u32, Word>,
}

impl Shell {
    /// `reads` holds one flag per read binding: `true` for a buffer of `u32` (an `i32`
    /// operand's bits), `false` for one of `f32`. `seeded` declares the run key.
    #[allow(clippy::too_many_arguments)]
    fn new(
        reads: &[bool],
        writes: usize,
        workgroup_size: u32,
        shared: Option<(u32, bool)>,
        matmul_vars: usize,
        extra_u32: usize,
        float_controls: bool,
        exact: bool,
        seeded: bool,
    ) -> Result<Self, EmitError> {
        let mut b = Builder::new();
        b.set_version(1, 5);
        b.capability(Capability::Shader);
        if float_controls {
            b.capability(Capability::RoundingModeRTE);
            b.capability(Capability::DenormPreserve);
            b.capability(Capability::SignedZeroInfNanPreserve);
        }
        b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

        let void = b.type_void();
        let fn_void = b.type_function(void, []);
        let bool_ty = b.type_bool();
        let u32_ty = b.type_int(32, 0);
        let f32_ty = b.type_float(32, None);
        let v3u32 = b.type_vector(u32_ty, 3);
        let input_v3u32 = b.type_pointer(None, StorageClass::Input, v3u32);

        let rta = b.type_runtime_array(f32_ty);
        b.decorate(rta, Decoration::ArrayStride, [dr::Operand::LiteralBit32(4)]);
        let buffer_struct = |b: &mut Builder, read: bool| {
            let id = b.id();
            let ty = b.type_struct_id(Some(id), [rta]);
            b.decorate(ty, Decoration::Block, []);
            b.member_decorate(ty, 0, Decoration::Offset, [dr::Operand::LiteralBit32(0)]);
            if read {
                b.member_decorate(ty, 0, Decoration::NonWritable, []);
            }
            ty
        };
        let read_struct = buffer_struct(&mut b, true);
        let write_struct = buffer_struct(&mut b, false);
        let read_ptr = b.type_pointer(None, StorageClass::StorageBuffer, read_struct);
        let write_ptr = b.type_pointer(None, StorageClass::StorageBuffer, write_struct);
        let element_ptr = b.type_pointer(None, StorageClass::StorageBuffer, f32_ty);
        // A read binding of `u32` and its element pointer, declared only by a module that reads
        // one.
        let (u32_element_ptr, u32_read_ptr) = if reads.contains(&true) {
            let element = b.type_pointer(None, StorageClass::StorageBuffer, u32_ty);
            let rta = b.type_runtime_array(u32_ty);
            b.decorate(rta, Decoration::ArrayStride, [dr::Operand::LiteralBit32(4)]);
            let id = b.id();
            let ty = b.type_struct_id(Some(id), [rta]);
            b.decorate(ty, Decoration::Block, []);
            b.member_decorate(ty, 0, Decoration::Offset, [dr::Operand::LiteralBit32(0)]);
            b.member_decorate(ty, 0, Decoration::NonWritable, []);
            let read = b.type_pointer(None, StorageClass::StorageBuffer, ty);
            (Some(element), Some(read))
        } else {
            (None, None)
        };
        let shared_ptr = b.type_pointer(None, StorageClass::Workgroup, f32_ty);
        let u32_fn_ptr = b.type_pointer(None, StorageClass::Function, u32_ty);
        let f32_fn_ptr = b.type_pointer(None, StorageClass::Function, f32_ty);

        let pc_struct = b.type_struct([u32_ty]);
        b.decorate(pc_struct, Decoration::Block, []);
        b.member_decorate(
            pc_struct,
            0,
            Decoration::Offset,
            [dr::Operand::LiteralBit32(0)],
        );
        let pc_ptr = b.type_pointer(None, StorageClass::PushConstant, pc_struct);
        let pc_u32_ptr = b.type_pointer(None, StorageClass::PushConstant, u32_ty);

        let zero = b.constant_bit32(u32_ty, 0);
        let group = b.constant_bit32(u32_ty, workgroup_size);
        let mut u32_constants = BTreeMap::new();
        u32_constants.insert(0, zero);
        u32_constants.insert(workgroup_size, group);

        let builtin = |b: &mut Builder, which: BuiltIn| {
            let var = b.variable(input_v3u32, None, StorageClass::Input, None);
            b.decorate(var, Decoration::BuiltIn, [dr::Operand::BuiltIn(which)]);
            var
        };
        let gid = builtin(&mut b, BuiltIn::GlobalInvocationId);
        let nwg = builtin(&mut b, BuiltIn::NumWorkgroups);
        let lid = shared.map(|_| builtin(&mut b, BuiltIn::LocalInvocationId));
        let bind = |b: &mut Builder, ptr: Word, binding: u32| {
            let var = b.variable(ptr, None, StorageClass::StorageBuffer, None);
            b.decorate(
                var,
                Decoration::DescriptorSet,
                [dr::Operand::LiteralBit32(0)],
            );
            b.decorate(
                var,
                Decoration::Binding,
                [dr::Operand::LiteralBit32(binding)],
            );
            var
        };
        let buffers: Vec<Word> = reads
            .iter()
            .enumerate()
            .map(|(binding, &integer)| {
                let ptr = match (integer, u32_read_ptr) {
                    (true, Some(ptr)) => ptr,
                    _ => read_ptr,
                };
                bind(&mut b, ptr, binding as u32)
            })
            .collect();
        let outs: Vec<Word> = (0..writes as u32)
            .map(|at| bind(&mut b, write_ptr, reads.len() as u32 + at))
            .collect();
        let seed = if seeded {
            let words = crate::vulkan::pipeline::SEED_SPEC_IDS.map(|id| {
                let word = b.spec_constant_bit32(u32_ty, 0);
                b.decorate(word, Decoration::SpecId, [dr::Operand::LiteralBit32(id)]);
                word
            });
            Some(words)
        } else {
            None
        };
        let pc = b.variable(pc_ptr, None, StorageClass::PushConstant, None);
        // Workgroup memory of `length` `f32` values, an array of `vec4` when `vectors`.
        let mut shared_vectors = None;
        let shared_var = if let Some((length, vectors)) = shared {
            let (element, length) = if vectors {
                let vec4 = b.type_vector(f32_ty, 4);
                shared_vectors = Some((vec4, b.type_pointer(None, StorageClass::Workgroup, vec4)));
                (vec4, length / 4)
            } else {
                (f32_ty, length)
            };
            let length = match u32_constants.get(&length) {
                Some(&id) => id,
                None => {
                    let id = b.constant_bit32(u32_ty, length);
                    u32_constants.insert(length, id);
                    id
                }
            };
            let array = b.type_array(element, length);
            let array_ptr = b.type_pointer(None, StorageClass::Workgroup, array);
            Some(b.variable(array_ptr, None, StorageClass::Workgroup, None))
        } else {
            None
        };

        let main = b.id();
        let mut interface = vec![gid, nwg];
        interface.extend(lid);
        interface.extend(&buffers);
        interface.extend(&outs);
        interface.push(pc);
        interface.extend(shared_var);
        b.entry_point(ExecutionModel::GLCompute, main, "main", interface);
        b.execution_mode(main, ExecutionMode::LocalSize, [workgroup_size, 1, 1]);
        if float_controls {
            b.execution_mode(main, ExecutionMode::RoundingModeRTE, [32]);
            b.execution_mode(main, ExecutionMode::DenormPreserve, [32]);
            b.execution_mode(main, ExecutionMode::SignedZeroInfNanPreserve, [32]);
        }

        b.begin_function(void, Some(main), FunctionControl::NONE, fn_void)?;
        b.begin_block(None)?;
        let u32_vars = [
            b.variable(u32_fn_ptr, None, StorageClass::Function, None),
            b.variable(u32_fn_ptr, None, StorageClass::Function, None),
        ];
        let f32_var = b.variable(f32_fn_ptr, None, StorageClass::Function, None);
        let matmul_vars = (0..matmul_vars)
            .map(|_| b.variable(f32_fn_ptr, None, StorageClass::Function, None))
            .collect();
        let extra_u32_vars = (0..extra_u32)
            .map(|_| b.variable(u32_fn_ptr, None, StorageClass::Function, None))
            .collect();
        let ids = b.load(v3u32, None, gid, None, [])?;
        let groups = b.load(v3u32, None, nwg, None, [])?;
        let x = b.composite_extract(u32_ty, None, ids, [0])?;
        let y = b.composite_extract(u32_ty, None, ids, [1])?;
        let z = b.composite_extract(u32_ty, None, ids, [2])?;
        let groups_x = b.composite_extract(u32_ty, None, groups, [0])?;
        let groups_y = b.composite_extract(u32_ty, None, groups, [1])?;
        let row = b.i_mul(u32_ty, None, groups_x, group)?;
        let plane = b.i_mul(u32_ty, None, row, groups_y)?;
        let y_offset = b.i_mul(u32_ty, None, y, row)?;
        let z_offset = b.i_mul(u32_ty, None, z, plane)?;
        let xy = b.i_add(u32_ty, None, x, y_offset)?;
        let index = b.i_add(u32_ty, None, xy, z_offset)?;
        let local = match lid {
            Some(lid) => {
                let ids = b.load(v3u32, None, lid, None, [])?;
                Some(b.composite_extract(u32_ty, None, ids, [0])?)
            }
            None => None,
        };
        let count_ptr = b.access_chain(pc_u32_ptr, None, pc, [zero])?;
        let count = b.load(u32_ty, None, count_ptr, None, [])?;
        Ok(Self {
            b,
            bool_ty,
            u32_ty,
            f32_ty,
            element_ptr,
            shared_ptr,
            buffers,
            outs,
            shared: shared_var,
            shared_vectors,
            index,
            local,
            count,
            u32_vars,
            f32_var,
            matmul_vars,
            extra_u32_vars,
            group,
            exact,
            glsl: None,
            u32_element_ptr,
            seed,
            u32_constants,
            f32_constants: BTreeMap::new(),
        })
    }

    fn finish(mut self) -> Result<dr::Module, EmitError> {
        self.b.ret()?;
        self.b.end_function()?;
        Ok(self.b.module())
    }

    fn uint(&mut self, value: u32) -> Word {
        if let Some(&id) = self.u32_constants.get(&value) {
            return id;
        }
        let id = self.b.constant_bit32(self.u32_ty, value);
        self.u32_constants.insert(value, id);
        id
    }

    fn float(&mut self, value: f32) -> Word {
        if let Some(&id) = self.f32_constants.get(&value.to_bits()) {
            return id;
        }
        let id = self.b.constant_bit32(self.f32_ty, value.to_bits());
        self.f32_constants.insert(value.to_bits(), id);
        id
    }

    fn no_contraction(&mut self, id: Word) -> Word {
        if self.exact {
            self.b.decorate(id, Decoration::NoContraction, []);
        }
        id
    }

    fn f_add(&mut self, a: Word, b: Word) -> Result<Word, EmitError> {
        let id = self.b.f_add(self.f32_ty, None, a, b)?;
        Ok(self.no_contraction(id))
    }

    fn f_sub(&mut self, a: Word, b: Word) -> Result<Word, EmitError> {
        let id = self.b.f_sub(self.f32_ty, None, a, b)?;
        Ok(self.no_contraction(id))
    }

    fn f_mul(&mut self, a: Word, b: Word) -> Result<Word, EmitError> {
        let id = self.b.f_mul(self.f32_ty, None, a, b)?;
        Ok(self.no_contraction(id))
    }

    /// `a * b + c` as one fused multiply-add, `GLSL.std.450 Fma` (DESIGN.md § 12 ADR-009),
    /// decorated `NoContraction` where the arithmetic is: the extended instruction set then
    /// counts the `Fma` as one operation and requires every such `Fma` to be computed the same
    /// way, so the driver neither splits it into a multiply and an add nor treats it as the
    /// expression `a * b + c`.
    fn f_fma(&mut self, a: Word, b: Word, c: Word) -> Result<Word, EmitError> {
        let id = self.glsl_f32(GlslStd450Op::Fma, &[a, b, c])?;
        Ok(self.no_contraction(id))
    }

    /// `a / b`, decorated `NoContraction` where the arithmetic is, so no driver turns it into
    /// a product with a reciprocal.
    fn f_div(&mut self, a: Word, b: Word) -> Result<Word, EmitError> {
        let id = self.b.f_div(self.f32_ty, None, a, b)?;
        Ok(self.no_contraction(id))
    }

    /// `GLSL.std.450 Exp` of `x`.
    fn exp(&mut self, x: Word) -> Result<Word, EmitError> {
        self.glsl_f32(GlslStd450Op::Exp, &[x])
    }

    /// `GLSL.std.450 Sqrt` of `x`.
    fn sqrt(&mut self, x: Word) -> Result<Word, EmitError> {
        self.glsl_f32(GlslStd450Op::Sqrt, &[x])
    }

    /// The `f32` extended instruction `op` of `GLSL.std.450` over `args`, the instruction set
    /// imported by the module's first such instruction.
    fn glsl_f32(&mut self, op: GlslStd450Op, args: &[Word]) -> Result<Word, EmitError> {
        let glsl = match self.glsl {
            Some(glsl) => glsl,
            None => *self.glsl.insert(self.b.ext_inst_import("GLSL.std.450")),
        };
        Ok(self.b.ext_inst(
            self.f32_ty,
            None,
            glsl,
            op as u32,
            args.iter().map(|&arg| dr::Operand::IdRef(arg)),
        )?)
    }

    fn add(&mut self, a: Word, b: Word) -> Result<Word, EmitError> {
        Ok(self.b.i_add(self.u32_ty, None, a, b)?)
    }

    fn mul(&mut self, a: Word, b: Word) -> Result<Word, EmitError> {
        Ok(self.b.i_mul(self.u32_ty, None, a, b)?)
    }

    fn less(&mut self, a: Word, b: Word) -> Result<Word, EmitError> {
        Ok(self.b.u_less_than(self.bool_ty, None, a, b)?)
    }

    fn and(&mut self, a: Word, b: Word) -> Result<Word, EmitError> {
        Ok(self.b.logical_and(self.bool_ty, None, a, b)?)
    }

    fn equal(&mut self, a: Word, b: Word) -> Result<Word, EmitError> {
        Ok(self.b.i_equal(self.bool_ty, None, a, b)?)
    }

    /// The smaller of `a` and `b`.
    fn min(&mut self, a: Word, b: Word) -> Result<Word, EmitError> {
        let below = self.less(a, b)?;
        Ok(self.b.select(self.u32_ty, None, below, a, b)?)
    }

    /// Element `at` of read binding `binding`.
    fn read(&mut self, binding: usize, at: Word) -> Result<Word, EmitError> {
        let zero = self.uint(0);
        let ptr = self
            .b
            .access_chain(self.element_ptr, None, self.buffers[binding], [zero, at])?;
        Ok(self.b.load(self.f32_ty, None, ptr, None, [])?)
    }

    /// Element `at` of read binding `binding`, a buffer of `u32`.
    fn read_u32(&mut self, binding: usize, at: Word) -> Result<Word, EmitError> {
        let element_ptr = self
            .u32_element_ptr
            .ok_or_else(|| EmitError::Build(format!("binding {binding} is not declared as u32")))?;
        let zero = self.uint(0);
        let ptr = self
            .b
            .access_chain(element_ptr, None, self.buffers[binding], [zero, at])?;
        Ok(self.b.load(self.u32_ty, None, ptr, None, [])?)
    }

    /// Element `at` of the result, the last written binding, as this invocation wrote it.
    fn read_result(&mut self, at: Word) -> Result<Word, EmitError> {
        self.read_out(self.outs.len() - 1, at)
    }

    /// Element `at` of written binding `output`, as this invocation or an earlier dispatch
    /// wrote it.
    fn read_out(&mut self, output: usize, at: Word) -> Result<Word, EmitError> {
        let zero = self.uint(0);
        let ptr = self
            .b
            .access_chain(self.element_ptr, None, self.outs[output], [zero, at])?;
        Ok(self.b.load(self.f32_ty, None, ptr, None, [])?)
    }

    /// Writes `value` to element `at` of the result, the last written binding.
    fn write(&mut self, at: Word, value: Word) -> Result<(), EmitError> {
        self.write_to(self.outs.len() - 1, at, value)
    }

    /// Writes `value` to element `at` of written binding `output`.
    fn write_to(&mut self, output: usize, at: Word, value: Word) -> Result<(), EmitError> {
        let zero = self.uint(0);
        let ptr = self
            .b
            .access_chain(self.element_ptr, None, self.outs[output], [zero, at])?;
        self.b.store(ptr, value, None, [])?;
        Ok(())
    }

    /// `relu(x)`: `x` where `x > 0`, else `0.0`, a compare and a select.
    fn relu(&mut self, x: Word) -> Result<Word, EmitError> {
        let f_zero = self.float(0.0);
        let positive = self.b.f_ord_greater_than(self.bool_ty, None, x, f_zero)?;
        Ok(self.b.select(self.f32_ty, None, positive, x, f_zero)?)
    }

    /// `ReluGrad(x, g)`: `g` where `x > 0`, else `0.0`, a compare and a select.
    fn relu_grad(&mut self, x: Word, g: Word) -> Result<Word, EmitError> {
        let f_zero = self.float(0.0);
        let positive = self.b.f_ord_greater_than(self.bool_ty, None, x, f_zero)?;
        Ok(self.b.select(self.f32_ty, None, positive, g, f_zero)?)
    }

    fn shared_at(&mut self, at: Word) -> Result<Word, EmitError> {
        let shared = self.shared.expect("a module with workgroup memory");
        if self.shared_vectors.is_some() {
            let (two, three) = (self.uint(2), self.uint(3));
            let vector = self.b.shift_right_logical(self.u32_ty, None, at, two)?;
            let lane = self.b.bitwise_and(self.u32_ty, None, at, three)?;
            return Ok(self
                .b
                .access_chain(self.shared_ptr, None, shared, [vector, lane])?);
        }
        Ok(self.b.access_chain(self.shared_ptr, None, shared, [at])?)
    }

    /// The 4 `f32` values of vector `at` of workgroup memory declared as `vec4`s, as one load.
    fn load_shared_vector(&mut self, at: Word) -> Result<[Word; 4], EmitError> {
        let shared = self.shared.expect("a module with workgroup memory");
        let (vec4, ptr) = self.shared_vectors.expect("workgroup memory of vectors");
        let ptr = self.b.access_chain(ptr, None, shared, [at])?;
        let vector = self.b.load(vec4, None, ptr, None, [])?;
        let mut values = [0; 4];
        for (lane, value) in values.iter_mut().enumerate() {
            *value = self
                .b
                .composite_extract(self.f32_ty, None, vector, [lane as u32])?;
        }
        Ok(values)
    }

    fn load_shared(&mut self, at: Word) -> Result<Word, EmitError> {
        let ptr = self.shared_at(at)?;
        Ok(self.b.load(self.f32_ty, None, ptr, None, [])?)
    }

    fn store_shared(&mut self, at: Word, value: Word) -> Result<(), EmitError> {
        let ptr = self.shared_at(at)?;
        self.b.store(ptr, value, None, [])?;
        Ok(())
    }

    /// A workgroup barrier on workgroup memory, in uniform control flow.
    fn barrier(&mut self) -> Result<(), EmitError> {
        let scope = self.uint(SCOPE_WORKGROUP);
        let semantics = self.uint(SEMANTICS_WORKGROUP);
        self.b.control_barrier(scope, scope, semantics)?;
        Ok(())
    }

    /// `if cond { then }`, continuing in the merge block.
    fn when(
        &mut self,
        cond: Word,
        then: impl FnOnce(&mut Self) -> Result<(), EmitError>,
    ) -> Result<(), EmitError> {
        let body = self.b.id();
        let merge = self.b.id();
        self.b.selection_merge(merge, SelectionControl::NONE)?;
        self.b.branch_conditional(cond, body, merge, [])?;
        self.b.begin_block(Some(body))?;
        then(self)?;
        self.b.branch(merge)?;
        self.b.begin_block(Some(merge))?;
        Ok(())
    }

    /// `for (i = start; i < end; i += step) { body(i) }` with the counter in function variable
    /// `var`, continuing in the merge block.
    fn counted(
        &mut self,
        var: Word,
        start: Word,
        end: Word,
        step: Word,
        body: impl FnOnce(&mut Self, Word) -> Result<(), EmitError>,
    ) -> Result<(), EmitError> {
        self.counted_with(LoopControl::NONE, var, start, end, step, body)
    }

    /// [`Shell::counted`] with the loop control `control`, a hint to the driver's compiler.
    fn counted_with(
        &mut self,
        control: LoopControl,
        var: Word,
        start: Word,
        end: Word,
        step: Word,
        body: impl FnOnce(&mut Self, Word) -> Result<(), EmitError>,
    ) -> Result<(), EmitError> {
        self.b.store(var, start, None, [])?;
        let header = self.b.id();
        let check = self.b.id();
        let block = self.b.id();
        let next = self.b.id();
        let merge = self.b.id();
        self.b.branch(header)?;
        self.b.begin_block(Some(header))?;
        self.b.loop_merge(merge, next, control, [])?;
        self.b.branch(check)?;
        self.b.begin_block(Some(check))?;
        let i = self.b.load(self.u32_ty, None, var, None, [])?;
        let more = self.less(i, end)?;
        self.b.branch_conditional(more, block, merge, [])?;
        self.b.begin_block(Some(block))?;
        body(self, i)?;
        self.b.branch(next)?;
        self.b.begin_block(Some(next))?;
        let i = self.b.load(self.u32_ty, None, var, None, [])?;
        let stepped = self.add(i, step)?;
        self.b.store(var, stepped, None, [])?;
        self.b.branch(header)?;
        self.b.begin_block(Some(merge))?;
        Ok(())
    }

    /// `acc += value` on the module's `f32` variable.
    fn accumulate(&mut self, value: Word) -> Result<(), EmitError> {
        let acc = self.b.load(self.f32_ty, None, self.f32_var, None, [])?;
        let sum = self.f_add(acc, value)?;
        self.b.store(self.f32_var, sum, None, [])?;
        Ok(())
    }

    /// `acc = fma(d, d, acc)` on the module's `f32` variable: the square of `d` fused into the
    /// sum.
    fn accumulate_square(&mut self, d: Word) -> Result<(), EmitError> {
        let acc = self.b.load(self.f32_ty, None, self.f32_var, None, [])?;
        let sum = self.f_fma(d, d, acc)?;
        self.b.store(self.f32_var, sum, None, [])?;
        Ok(())
    }

    fn elementwise(&mut self, arith: Arith, sources: &[Source]) -> Result<(), EmitError> {
        let (index, count) = (self.index, self.count);
        let inside = self.less(index, count)?;
        self.when(inside, |m| {
            let mut values = Vec::new();
            for source in sources {
                let value = match *source {
                    Source::Buffer { binding, scalar } => {
                        let at = if scalar { m.uint(0) } else { index };
                        m.read(binding as usize, at)?
                    }
                    Source::Immediate(value) => m.float(value),
                };
                values.push(value);
            }
            let f_zero = m.float(0.0);
            let (f32_ty, bool_ty) = (m.f32_ty, m.bool_ty);
            let result = match arith {
                Arith::Add => m.b.f_add(f32_ty, None, values[0], values[1])?,
                Arith::Sub => m.b.f_sub(f32_ty, None, values[0], values[1])?,
                Arith::Mul => m.b.f_mul(f32_ty, None, values[0], values[1])?,
                Arith::Relu => m.relu(values[0])?,
                Arith::ReluGrad => {
                    let positive = m.b.f_ord_greater_than(bool_ty, None, values[0], f_zero)?;
                    m.b.select(f32_ty, None, positive, values[1], f_zero)?
                }
                Arith::ScaledSub => {
                    let product = m.f_mul(values[1], values[2])?;
                    m.f_sub(values[0], product)?
                }
                Arith::Sqrt => m.root(values[0])?,
                Arith::Div => m.quotient(values[0], values[1])?,
                Arith::Copy => values[0],
            };
            m.write(index, result)
        })
    }

    /// A fused elementwise chain: each op computed from its inputs and rounded once, in order,
    /// its add, subtract or multiply decorated `NoContraction` unless the module is `fast_math`,
    /// so no driver contracts two ops into one fused multiply-add; the last op's result is
    /// written. The intermediate results stay in registers.
    fn chain(&mut self, ops: &[(Arith, Vec<Input>)]) -> Result<(), EmitError> {
        let (index, count) = (self.index, self.count);
        let inside = self.less(index, count)?;
        self.when(inside, |m| {
            let mut previous = None;
            for (arith, inputs) in ops {
                let mut values = Vec::new();
                for input in inputs {
                    let value = match *input {
                        Input::Previous => previous.expect("only a later op reads the previous"),
                        Input::Source(Source::Buffer { binding, scalar }) => {
                            let at = if scalar { m.uint(0) } else { index };
                            m.read(binding as usize, at)?
                        }
                        Input::Source(Source::Immediate(value)) => m.float(value),
                    };
                    values.push(value);
                }
                let result = match arith {
                    Arith::Add => m.f_add(values[0], values[1])?,
                    Arith::Sub => m.f_sub(values[0], values[1])?,
                    Arith::Mul => m.f_mul(values[0], values[1])?,
                    Arith::Relu => m.relu(values[0])?,
                    Arith::ReluGrad => {
                        let f_zero = m.float(0.0);
                        let (f32_ty, bool_ty) = (m.f32_ty, m.bool_ty);
                        let positive = m.b.f_ord_greater_than(bool_ty, None, values[0], f_zero)?;
                        m.b.select(f32_ty, None, positive, values[1], f_zero)?
                    }
                    Arith::ScaledSub | Arith::Sqrt | Arith::Div | Arith::Copy => {
                        unreachable!("no op of a chain")
                    }
                };
                previous = Some(result);
            }
            m.write(index, previous.expect("a chain has ops"))
        })
    }

    /// The pairwise tree over the first `valid` values of workgroup memory: at stride `s`,
    /// invocation `i < s` adds value `i + s` to value `i` when `i + s < valid`. Value 0 ends as
    /// the sum; no value at or past `valid` is read.
    fn tree(&mut self, valid: Word) -> Result<(), EmitError> {
        let local = self.local.expect("a module with workgroup memory");
        for stride in TREE_STRIDES {
            let s = self.uint(stride);
            let low = self.less(local, s)?;
            let partner = self.add(local, s)?;
            let filled = self.less(partner, valid)?;
            let folds = self.and(low, filled)?;
            self.when(folds, |m| {
                let a = m.load_shared(local)?;
                let b = m.load_shared(partner)?;
                let sum = m.f_add(a, b)?;
                m.store_shared(local, sum)
            })?;
            self.barrier()?;
        }
        Ok(())
    }

    /// Stage 1: workgroup `w` writes partial `w`, the tree sum of elements `64 w` to
    /// `min(64 w + 64, count) - 1`.
    fn stage1(&mut self) -> Result<(), EmitError> {
        let (index, count) = (self.index, self.count);
        let local = self.local.expect("a module with workgroup memory");
        let inside = self.less(index, count)?;
        self.when(inside, |m| {
            let value = m.read(0, index)?;
            m.store_shared(local, value)
        })?;
        self.barrier()?;
        let base = self.b.i_sub(self.u32_ty, None, index, local)?;
        let started = self.less(base, count)?;
        let left = self.b.i_sub(self.u32_ty, None, count, base)?;
        let zero = self.uint(0);
        let left = self.b.select(self.u32_ty, None, started, left, zero)?;
        let tree = self.uint(TREE);
        let valid = self.min(left, tree)?;
        self.tree(valid)?;
        let first = self.equal(local, zero)?;
        let writes = self.and(first, started)?;
        self.when(writes, |m| {
            let sum = m.load_shared(zero)?;
            let group = m.b.u_div(m.u32_ty, None, base, tree)?;
            m.write(group, sum)
        })
    }

    /// Stage 2, one workgroup over `count` partials: invocation `i` sums partials `i`, `i + 64`,
    /// … in ascending order, then the tree folds the 64 sums; `divide_by` is the element count of
    /// a mean.
    fn stage2(&mut self, divide_by: Option<u32>) -> Result<(), EmitError> {
        let (index, count) = (self.index, self.count);
        let local = self.local.expect("a module with workgroup memory");
        let var = self.u32_vars[0];
        let has = self.less(local, count)?;
        let tree = self.uint(TREE);
        self.when(has, |m| {
            let first = m.read(0, local)?;
            m.b.store(m.f32_var, first, None, [])?;
            let start = m.add(local, tree)?;
            m.counted(var, start, count, tree, |m, j| {
                let value = m.read(0, j)?;
                m.accumulate(value)
            })?;
            let acc = m.b.load(m.f32_ty, None, m.f32_var, None, [])?;
            m.store_shared(local, acc)
        })?;
        self.barrier()?;
        let valid = self.min(count, tree)?;
        self.tree(valid)?;
        let zero = self.uint(0);
        let first = self.equal(index, zero)?;
        let any = self.less(zero, count)?;
        let writes = self.and(first, any)?;
        self.when(writes, |m| {
            let mut sum = m.load_shared(zero)?;
            if let Some(n) = divide_by {
                let n = m.float(n as f32);
                sum = m.b.f_div(m.f32_ty, None, sum, n)?;
            }
            m.write(zero, sum)
        })
    }

    /// `body(i)` for `i` in `0..times`: a loop whose every iteration runs `unroll` of them, then
    /// the rest in straight-line code. The counter is function variable `var`.
    fn repeat(
        &mut self,
        var: Word,
        times: u32,
        unroll: u32,
        mut body: impl FnMut(&mut Self, Word) -> Result<(), EmitError>,
    ) -> Result<(), EmitError> {
        let groups = times / unroll;
        if groups > 0 {
            let (zero, one) = (self.uint(0), self.uint(1));
            let (end, width) = (self.uint(groups), self.uint(unroll));
            self.counted(var, zero, end, one, |m, g| {
                let first = m.mul(g, width)?;
                for k in 0..unroll {
                    let i = match k {
                        0 => first,
                        _ => {
                            let k = m.uint(k);
                            m.add(first, k)?
                        }
                    };
                    body(m, i)?;
                }
                Ok(())
            })?;
        }
        for i in groups * unroll..times {
            let i = self.uint(i);
            body(self, i)?;
        }
        Ok(())
    }

    /// The `mse` loss sum of one workgroup over the `terms` elements of the shape, `chunk` terms
    /// at a time. For the chunk at element `base`, invocation `i` stores the difference `a - b`
    /// of elements `base + i`, `base + i + 64`, … at its offset in workgroup memory; after a
    /// barrier invocation 0 adds the chunk's terms `(a - b) * (a - b)` in ascending order, the
    /// term of element 0 in the first chunk starting the sum as a product and every later one
    /// fused into it, `fma(d, d, acc)` (DESIGN.md § 12 ADR-009); a second barrier frees the
    /// memory for the next chunk. The order of additions is the one-invocation order: the first
    /// term, then each next one. With `divide_by` the element count `n`, in the plan's module, the
    /// same invocation also adds the scaled terms `(d * s) * d`, `fma(d * s, d, acc)` after the
    /// first, with `s = 1/n` rounded to `f32`, into a second accumulator, and writes the sum
    /// divided by `n` where that sum is finite or NaN and the scaled sum where it is infinite: the
    /// scaled sum is the mean, which passes `f32::MAX` where the CPU interpreter's `f64` mean does
    /// up to its rounding, and a finite sum keeps the bits it had (DESIGN.md § 5.4, revision
    /// 0.75). The counts are constants of the shape, so the driver sees every trip count; no term
    /// is added for an empty shape.
    fn loss_sum(
        &mut self,
        divide_by: Option<u32>,
        terms: u32,
        chunk: u32,
    ) -> Result<(), EmitError> {
        if terms == 0 {
            return Ok(());
        }
        let scaled =
            divide_by.map(|n| (self.float((1.0 / f64::from(n)) as f32), self.matmul_vars[0]));
        let chunk_var = self.u32_vars[0];
        let (full, tail) = (terms / chunk, terms % chunk);
        let zero = self.uint(0);
        self.loss_chunk(zero, chunk, scaled, true, full == 1 && tail == 0)?;
        if full > 1 {
            let (one, end, size) = (self.uint(1), self.uint(full), self.uint(chunk));
            self.counted(chunk_var, one, end, one, |m, c| {
                let base = m.mul(c, size)?;
                m.loss_chunk(base, chunk, scaled, false, false)
            })?;
        }
        if tail > 0 {
            let base = self.uint(full * chunk);
            self.loss_chunk(base, tail, scaled, false, true)?;
        }
        let local = self.local.expect("a module with workgroup memory");
        let first_invocation = self.equal(local, zero)?;
        self.when(first_invocation, |m| {
            let mut sum = m.b.load(m.f32_ty, None, m.f32_var, None, [])?;
            if let (Some(n), Some((_, var))) = (divide_by, scaled) {
                let n = m.float(n as f32);
                let divided = m.b.f_div(m.f32_ty, None, sum, n)?;
                let mean = m.b.load(m.f32_ty, None, var, None, [])?;
                let overflowed = m.b.is_inf(m.bool_ty, None, sum)?;
                sum = m.pick(m.f32_ty, overflowed, mean, divided)?;
            }
            m.write(zero, sum)
        })
    }

    /// One chunk of [`Shell::loss_sum`]: the `len` terms from element `base`, and with `scaled`,
    /// the constant `s` and the second accumulator, their scaled terms too. The `opening` chunk
    /// starts each sum with its first term; the `closing` one needs no barrier after it.
    fn loss_chunk(
        &mut self,
        base: Word,
        len: u32,
        scaled: Option<(Word, Word)>,
        opening: bool,
        closing: bool,
    ) -> Result<(), EmitError> {
        let local = self.local.expect("a module with workgroup memory");
        let term_var = self.u32_vars[1];
        let group = self.group;
        let stage = |m: &mut Self, j: Word| -> Result<(), EmitError> {
            let at = m.add(base, j)?;
            let a = m.read(0, at)?;
            let b = m.read(1, at)?;
            let d = m.f_sub(a, b)?;
            m.store_shared(j, d)
        };
        let size = TREE;
        let (rounds, partial) = (len / size, len % size);
        self.repeat(term_var, rounds, UNROLL, |m, r| {
            let offset = m.mul(r, group)?;
            let j = m.add(local, offset)?;
            stage(m, j)
        })?;
        if partial > 0 {
            let partial = self.uint(partial);
            let inside = self.less(local, partial)?;
            let offset = self.uint(rounds * size);
            self.when(inside, |m| {
                let j = m.add(local, offset)?;
                stage(m, j)
            })?;
        }
        self.barrier()?;
        let zero = self.uint(0);
        let first_invocation = self.equal(local, zero)?;
        let start = u32::from(opening);
        self.when(first_invocation, |m| {
            if opening {
                let d = m.load_shared(zero)?;
                let first = m.f_mul(d, d)?;
                m.b.store(m.f32_var, first, None, [])?;
                if let Some((s, var)) = scaled {
                    let e = m.f_mul(d, s)?;
                    let first = m.f_mul(e, d)?;
                    m.b.store(var, first, None, [])?;
                }
            }
            let start = m.uint(start);
            m.repeat(term_var, len - u32::from(opening), UNROLL, |m, t| {
                let j = m.add(t, start)?;
                let d = m.load_shared(j)?;
                m.accumulate_square(d)?;
                if let Some((s, var)) = scaled {
                    let e = m.f_mul(d, s)?;
                    let acc = m.b.load(m.f32_ty, None, var, None, [])?;
                    let sum = m.f_fma(e, d, acc)?;
                    m.b.store(var, sum, None, [])?;
                }
                Ok(())
            })
        })?;
        if !closing {
            self.barrier()?;
        }
        Ok(())
    }

    /// Output element `o` of `[cols, rows]` is input element `(o % rows) * cols + o / rows`.
    fn transpose(&mut self, rows: u32, cols: u32) -> Result<(), EmitError> {
        let (index, count) = (self.index, self.count);
        let inside = self.less(index, count)?;
        self.when(inside, |m| {
            let rows = m.uint(rows);
            let cols = m.uint(cols);
            let i = m.b.u_mod(m.u32_ty, None, index, rows)?;
            let j = m.b.u_div(m.u32_ty, None, index, rows)?;
            let row = m.mul(i, cols)?;
            let at = m.add(row, j)?;
            let value = m.read(0, at)?;
            m.write(index, value)
        })
    }

    /// `value` where `valid`, else `fallback`.
    fn pick(
        &mut self,
        ty: Word,
        valid: Word,
        value: Word,
        fallback: Word,
    ) -> Result<Word, EmitError> {
        Ok(self.b.select(ty, None, valid, value, fallback)?)
    }

    /// The tiled matmul of `[m, k] x [k, n]` (DESIGN.md § 5.2). Workgroup `g` computes the block
    /// of row block `g / ⌈n / cols⌉` and column block `g % ⌈n / cols⌉`. For each tile of `depth`
    /// values of the contracting index, from `k = 0` up, its invocations read the block's
    /// `a[rows, tile]` and `b[tile, cols]` into registers `ahead` tiles early, stage them in
    /// workgroup memory (`b` transposed when an invocation computes one column, as read when it
    /// computes several), wait on a barrier, and each invocation reads its `rows` values of `a`
    /// and `cols_per_invocation` values of `b` of each `k` of the tile into registers and fuses
    /// their products into the accumulator of each output element of its block in ascending `k`,
    /// the first product of the first tile starting the sum. Values outside the matrices are
    /// staged but never enter a written output, and the last tile has its own trip count, so no
    /// padding term is added: every output element is `a[i, 0] * b[0, j]`, then
    /// `fma(a[i, p], b[p, j], acc)` for each next `p` in ascending order, as the CPU computes it
    /// with `mul_add` (DESIGN.md § 12 ADR-009). With a `relu` [`Epilogue`] the rounded
    /// sum is written to the pre-activation binding when the kernel keeps it, and its `relu` to
    /// the result; with a `ReluGrad` epilogue the result is the sum where the third operand, the
    /// pre-activation, is above 0, and 0 elsewhere.
    ///
    /// An operand bound transposed (`transposed`, [`Kernel::transposed`]) is read in place from
    /// its row-major transpose, `[k, m]` for `a` and `[n, k]` for `b`, in the [`Staging`] order
    /// that gives consecutive invocations consecutive addresses, and staged in workgroup memory
    /// in the order of the operand bound as is, each line of the `a` block or of a `k`-major `b`
    /// block at the stride of `layout` ([`MatMulTile::layout_within`]), padded against bank
    /// conflicts when the device has the workgroup memory for it, so the products read the same
    /// values as for the operand bound as is; every product and sum is that of the operand bound
    /// as is, and the instructions for an operand bound as is are unchanged.
    ///
    /// A tile with [`MatMulTile::vector_reads`] (the block of 256 invocations) gives each
    /// invocation runs of 4 adjacent columns and rows, stages both blocks `k`-major
    /// ([`MatMulTile::vector_layout`]) in workgroup memory declared as `vec4`s, and per `k` reads
    /// each run of 4 values of `b` and of `a` with one vector load, then fuses the products of
    /// that `k` in the order of the scalar reads, row by row and column by column: every
    /// product and sum is the one the scalar tile computes for that output element.
    fn matmul(
        &mut self,
        [m, k, n]: [u32; 3],
        epilogue: Option<Epilogue>,
        tile: MatMulTile,
        transposed: [bool; 2],
        layout: MatMulLayout,
    ) -> Result<(), EmitError> {
        let local = self.local.expect("a module with workgroup memory");
        let u32_ty = self.u32_ty;
        let (size, cols, depth) = (tile.workgroup_size, tile.cols, tile.depth);
        let (lanes, per) = (tile.lanes(), tile.cols_per_invocation);
        let [a_transposed, b_transposed] = transposed;
        // `b` staged transposed, one column per stride, or as read, one `k` per stride.
        let transposed = per == 1;
        let block_rows = tile.rows_per_workgroup();
        let thread_rows = size / lanes;
        let a_stride = layout.a_stride;
        let b_offset = layout.b_offset;

        let index = self.index;
        let size_c = self.uint(size);
        let group = self.b.u_div(u32_ty, None, index, size_c)?;
        let col_blocks = self.uint(n.div_ceil(cols));
        let block_row = self.b.u_div(u32_ty, None, group, col_blocks)?;
        let block_col = self.b.u_mod(u32_ty, None, group, col_blocks)?;
        let block_rows_c = self.uint(block_rows);
        let row0 = self.mul(block_row, block_rows_c)?;
        let cols_c = self.uint(cols);
        let col0 = self.mul(block_col, cols_c)?;
        let lanes_c = self.uint(lanes);
        let thread_col = self.b.u_mod(u32_ty, None, local, lanes_c)?;
        let thread_row = self.b.u_div(u32_ty, None, local, lanes_c)?;
        // `k` and `depth` are declared here, before the constants below, though the staging
        // takes them by value (`Staging`): the module of an operand bound as is keeps its order.
        let (m_c, _, n_c) = (self.uint(m), self.uint(k), self.uint(n));
        self.uint(depth);
        let a_stride_c = self.uint(a_stride);
        let b_offset_c = self.uint(b_offset);
        let b_stride_c = self.uint(layout.b_stride);
        // Per column `c` of the invocation's block: its output column `j`, and where its `b`
        // values start in a buffer, at `k` 0 of the tile.
        // A tile with vector reads (`MatMulTile::vector_reads`) gives each invocation runs of 4
        // adjacent columns and rows, both blocks staged `k`-major.
        let vectors = tile.vector_reads();
        let (thread_col4, thread_row4) = if vectors {
            let four = self.uint(4);
            (self.mul(thread_col, four)?, self.mul(thread_row, four)?)
        } else {
            (thread_col, thread_row)
        };
        let mut local_cols = Vec::new();
        let mut b_cols = Vec::new();
        for c in 0..per {
            let col = if vectors {
                let offset = self.uint(c / 4 * lanes * 4 + c % 4);
                self.add(thread_col4, offset)?
            } else {
                let offset = self.uint(c * lanes);
                self.add(thread_col, offset)?
            };
            let at = if transposed {
                self.mul(col, b_stride_c)?
            } else {
                col
            };
            b_cols.push(self.add(b_offset_c, at)?);
            local_cols.push(self.add(col0, col)?);
        }
        let mut local_rows = Vec::new();
        for q in 0..tile.rows {
            if vectors {
                let offset = self.uint(q / 4 * thread_rows * 4 + q % 4);
                local_rows.push(self.add(thread_row4, offset)?);
            } else {
                let offset = self.uint(q * thread_rows);
                local_rows.push(self.add(thread_row, offset)?);
            }
        }
        // Where each row's `a` values start in a buffer, at `k` 0 of the tile: the row of a
        // `k`-major block, else the row times the stride.
        let mut a_rows = Vec::new();
        for &row in &local_rows {
            a_rows.push(if vectors {
                row
            } else {
                self.mul(row, a_stride_c)?
            });
        }
        // With vector reads, the vector of each run of 4 columns and rows at `k` 0 of the tile.
        let mut b_vectors = Vec::new();
        let mut a_vectors = Vec::new();
        if vectors {
            let two = self.uint(2);
            for &at in b_cols.iter().step_by(4) {
                b_vectors.push(self.b.shift_right_logical(u32_ty, None, at, two)?);
            }
            for &at in a_rows.iter().step_by(4) {
                a_vectors.push(self.b.shift_right_logical(u32_ty, None, at, two)?);
            }
        }

        // This invocation's staging slots, the same in every tile: the `a` values it stages,
        // then the `b` values. A slot outside the matrices reads element `0` of the tile's
        // offset instead, a value no written output uses.
        struct Slot {
            /// `a` (binding 0) or `b` (binding 1).
            is_a: bool,
            /// The element read for the tile from `k = 0`, before the tile's offset.
            at: Word,
            /// Where the value goes in a workgroup memory buffer.
            to: Word,
            /// The slot's index along the contracting index, within the tile.
            along: Word,
            /// The binding's distance between two consecutive values of `k`
            /// ([`Staging::along_stride`]).
            along_stride: u32,
            /// Whether the slot lies inside the matrices.
            valid: Word,
            /// On a last step that not every invocation takes, whether this one takes it.
            taken: Option<Word>,
        }
        let mut slots = Vec::new();
        let zero = self.uint(0);
        for (total, is_a) in [(block_rows * depth, true), (depth * cols, false)] {
            let total_c = self.uint(total);
            for step in 0..total.div_ceil(size) {
                let offset = self.uint(step * size);
                let e = self.add(local, offset)?;
                let taken = if (step + 1) * size > total {
                    Some(self.less(e, total_c)?)
                } else {
                    None
                };
                // Element `e` of the operand's tile: its `along` and `across` indices, its row
                // or column, and the element of the binding it reads (`Staging`).
                let staging = Staging::of(
                    &tile,
                    m,
                    k,
                    n,
                    is_a,
                    if is_a { a_transposed } else { b_transposed },
                );
                let run_c = self.uint(staging.run);
                let slow = self.b.u_div(u32_ty, None, e, run_c)?;
                let fast = self.b.u_mod(u32_ty, None, e, run_c)?;
                let (along, across) = if staging.along_first {
                    (fast, slow)
                } else if staging.run == staging.across {
                    (slow, fast)
                } else {
                    let depth_c = self.uint(staging.depth);
                    let along = self.b.u_mod(u32_ty, None, slow, depth_c)?;
                    let runs = self.b.u_div(u32_ty, None, slow, depth_c)?;
                    let runs_at = self.mul(runs, run_c)?;
                    (along, self.add(runs_at, fast)?)
                };
                let (origin, limit) = if is_a { (row0, m_c) } else { (col0, n_c) };
                let line = self.add(origin, across)?;
                let valid = self.less(line, limit)?;
                let at = if staging.along_first {
                    let stride = self.uint(staging.across_stride);
                    let line_at = self.mul(line, stride)?;
                    self.add(line_at, along)?
                } else {
                    let stride = self.uint(staging.along_stride);
                    let along_at = self.mul(along, stride)?;
                    self.add(along_at, line)?
                };
                let to = if is_a && vectors {
                    let k_at = self.mul(along, a_stride_c)?;
                    self.add(k_at, across)?
                } else if is_a {
                    let r_at = self.mul(across, a_stride_c)?;
                    self.add(r_at, along)?
                } else {
                    let (p, c) = (along, across);
                    if transposed {
                        let c_at = self.mul(c, b_stride_c)?;
                        let c_at = self.add(b_offset_c, c_at)?;
                        self.add(c_at, p)?
                    } else {
                        let p_at = self.mul(p, b_stride_c)?;
                        let p_at = self.add(b_offset_c, p_at)?;
                        self.add(p_at, c)?
                    }
                };
                let valid = match taken {
                    Some(taken) => self.and(valid, taken)?,
                    None => valid,
                };
                let at = self.pick(u32_ty, valid, at, zero)?;
                slots.push(Slot {
                    is_a,
                    at,
                    to,
                    along,
                    along_stride: staging.along_stride,
                    valid,
                    taken,
                });
            }
        }

        // Where a tile read starts along the contracting index: a constant `k`, or, in the tile
        // loop, its element offsets into `a` and `b`.
        #[derive(Clone, Copy)]
        enum Start {
            At(u32),
            Looped { a: Word, b: Word },
        }

        // Reads this invocation's share of the tile of `len` values of the contracting index
        // from `start` into registers, one value per slot.
        let fetch = |s: &mut Self, start: Start, len: u32| -> Result<Vec<Word>, EmitError> {
            let mut fetched = Vec::new();
            for slot in &slots {
                let offset = match start {
                    Start::At(base) => s.uint(base * slot.along_stride),
                    Start::Looped { a, b } => {
                        if slot.is_a {
                            a
                        } else {
                            b
                        }
                    }
                };
                let mut at = s.add(slot.at, offset)?;
                if len < depth {
                    let len_c = s.uint(len);
                    let inside = s.less(slot.along, len_c)?;
                    let inside = s.and(slot.valid, inside)?;
                    let zero = s.uint(0);
                    at = s.pick(u32_ty, inside, at, zero)?;
                }
                let value = s.read(if slot.is_a { 0 } else { 1 }, at)?;
                fetched.push(value);
            }
            Ok(fetched)
        };

        // Writes fetched values to the workgroup memory buffer at `buffer`.
        let store = |s: &mut Self, staged: Vec<Word>, buffer: Word| -> Result<(), EmitError> {
            for (value, slot) in staged.into_iter().zip(&slots) {
                let (to, taken) = (slot.to, slot.taken);
                let to = s.add(buffer, to)?;
                match taken {
                    Some(taken) => s.when(taken, |s| s.store_shared(to, value))?,
                    None => s.store_shared(to, value)?,
                }
            }
            Ok(())
        };

        let accumulators = matmul_accumulators(&tile);
        let b_line = layout.b_stride;
        // Fuses the products of the first `len` values of the tile staged in the buffer at
        // `buffer` into the accumulators (`Fma`), the first product starting the sum when
        // `opening`. Per
        // `k`, the invocation reads its `b` value of each of its columns into registers, then
        // per row its `a` value and that row's products, accumulator `q * per + c` holding row
        // `q` and column `c` of its block.
        let compute = |s: &mut Self,
                       buffer: Word,
                       buffer_value: u32,
                       len: u32,
                       opening: bool|
         -> Result<(), EmitError> {
            let vars = s.matmul_vars[..accumulators].to_vec();
            let mut sums: Vec<Option<Word>> = Vec::new();
            for &var in &vars {
                sums.push(if opening {
                    None
                } else {
                    Some(s.b.load(s.f32_ty, None, var, None, [])?)
                });
            }
            let mut b_starts = Vec::new();
            for &b_col in &b_cols {
                b_starts.push(s.add(buffer, b_col)?);
            }
            let mut a_starts = Vec::new();
            for &a_row in &a_rows {
                a_starts.push(s.add(buffer, a_row)?);
            }
            for p in 0..len {
                if vectors {
                    // Per `k`, one vector load per run of 4 columns of `b` and of 4 rows
                    // of `a`, then the products in the order of the scalar reads.
                    let mut bs = Vec::new();
                    for &at in &b_vectors {
                        let offset = s.uint((buffer_value + p * b_line) / 4);
                        let at = s.add(at, offset)?;
                        bs.extend(s.load_shared_vector(at)?);
                    }
                    let mut a_values = Vec::new();
                    for &at in &a_vectors {
                        let offset = s.uint((buffer_value + p * a_stride) / 4);
                        let at = s.add(at, offset)?;
                        a_values.extend(s.load_shared_vector(at)?);
                    }
                    for (row_sums, &a) in sums.chunks_mut(per as usize).zip(&a_values) {
                        for (sum, &b) in row_sums.iter_mut().zip(&bs) {
                            *sum = Some(match *sum {
                                None => s.f_mul(a, b)?,
                                Some(acc) => s.f_fma(a, b, acc)?,
                            });
                        }
                    }
                    continue;
                }
                let p_c = s.uint(if vectors { p * a_stride } else { p });
                let b_step = s.uint(if transposed { p } else { p * b_line });
                let mut bs = Vec::new();
                for &b_start in &b_starts {
                    let b_at = s.add(b_start, b_step)?;
                    bs.push(s.load_shared(b_at)?);
                }
                for (row_sums, &a_start) in sums.chunks_mut(per as usize).zip(&a_starts) {
                    let a_at = s.add(a_start, p_c)?;
                    let a = s.load_shared(a_at)?;
                    for (sum, &b) in row_sums.iter_mut().zip(&bs) {
                        *sum = Some(match *sum {
                            None => s.f_mul(a, b)?,
                            Some(acc) => s.f_fma(a, b, acc)?,
                        });
                    }
                }
            }
            for (&var, sum) in vars.iter().zip(sums) {
                if let Some(sum) = sum {
                    s.b.store(var, sum, None, [])?;
                }
            }
            Ok(())
        };
        let buffer_values = layout.buffer_values;
        let buffers = layout.buffers;
        let buffer_of =
            |s: &mut Self, tile_index: u32| s.uint((tile_index % buffers) * buffer_values);

        // The tiles of the contracting index, from `k = 0` up; the last may be short. Tile `i`
        // is read into registers `ahead` tiles before it is computed, so its reads overlap the
        // products of the tiles before it, and staged in buffer `i % 2`: one barrier per tile
        // both publishes it and frees the buffer of tile `i - 1` for tile `i + 1`.
        let ahead = tile.ahead.max(1) as usize;
        let count = k.div_ceil(depth) as usize;
        let len_of = |i: usize| (k - i as u32 * depth).min(depth);
        // The read of tile `i + ahead`, when there is one, for tile `i` emitted straight.
        let ahead_of = |i: usize| {
            (i + ahead < count).then(|| (Start::At((i + ahead) as u32 * depth), len_of(i + ahead)))
        };
        // Tile `i`: the read of `read`, the products of tile `i`, the staging of tile `i + 1`.
        // `queue` holds the tiles read and not yet staged, the next one first.
        let step = |s: &mut Self,
                    queue: &mut VecDeque<Vec<Word>>,
                    i: usize,
                    read: Option<(Start, u32)>|
         -> Result<(), EmitError> {
            if let Some((start, len)) = read {
                queue.push_back(fetch(s, start, len)?);
            }
            let buffer = buffer_of(s, i as u32);
            let buffer_value = (i as u32 % buffers) * buffer_values;
            compute(s, buffer, buffer_value, len_of(i), i == 0)?;
            if i + 1 < count {
                let next = queue.pop_front().expect("fetched ahead");
                let buffer = buffer_of(s, i as u32 + 1);
                store(s, next, buffer)?;
                s.barrier()?;
            }
            Ok(())
        };

        let mut fetched = VecDeque::new();
        for i in 0..count.min(ahead) {
            fetched.push_back(fetch(self, Start::At(i as u32 * depth), len_of(i))?);
        }
        let buffer = buffer_of(self, 0);
        let first = fetched.pop_front().expect("k is at least 1");
        store(self, first, buffer)?;
        self.barrier()?;
        step(self, &mut fetched, 0, ahead_of(0))?;

        // The tile loop (`matmul_passes`): each pass computes `MATMUL_UNROLL` full tiles and
        // reads the full tiles `ahead` past them. The tiles read and not yet staged cross from
        // one pass to the next in function variables.
        let passes = matmul_passes(k, &tile);
        let mut resume = 1;
        if passes > 0 {
            let carried = self.matmul_vars[accumulators..].to_vec();
            let per = slots.len();
            assert_eq!(carried.len(), (ahead - 1) * per, "matmul_carried");
            let spill = |s: &mut Self, queue: &mut VecDeque<Vec<Word>>| -> Result<(), EmitError> {
                for (q, values) in queue.drain(..).enumerate() {
                    for (at, value) in values.into_iter().enumerate() {
                        s.b.store(carried[q * per + at], value, None, [])?;
                    }
                }
                Ok(())
            };
            let fill = |s: &mut Self, queue: &mut VecDeque<Vec<Word>>| -> Result<(), EmitError> {
                for q in 0..ahead - 1 {
                    let mut values = Vec::new();
                    for at in 0..per {
                        values.push(s.b.load(s.f32_ty, None, carried[q * per + at], None, [])?);
                    }
                    queue.push_back(values);
                }
                Ok(())
            };
            spill(self, &mut fetched)?;
            let (zero, one, end) = (self.uint(0), self.uint(1), self.uint(passes));
            let along = |is_a: bool, transposed: bool| {
                Staging::of(&tile, m, k, n, is_a, transposed).along_stride
            };
            let (a_along, b_along) = (along(true, a_transposed), along(false, b_transposed));
            let pass_a = self.uint(MATMUL_UNROLL * depth * a_along);
            let pass_b = self.uint(MATMUL_UNROLL * depth * b_along);
            let var = self.u32_vars[0];
            self.counted(var, zero, end, one, |s, pass| {
                let mut queue = VecDeque::new();
                fill(s, &mut queue)?;
                let a0 = s.mul(pass, pass_a)?;
                let b0 = s.mul(pass, pass_b)?;
                // Tile `1 + pass * MATMUL_UNROLL + p` is full, not the first, has the parity of
                // `1 + p` and reads the full tile `ahead` past it.
                for p in 0..MATMUL_UNROLL as usize {
                    let i = 1 + p;
                    let base = (i + ahead) as u32 * depth;
                    let a_c = s.uint(base * a_along);
                    let a = s.add(a0, a_c)?;
                    let b_c = s.uint(base * b_along);
                    let b = s.add(b0, b_c)?;
                    step(s, &mut queue, i, Some((Start::Looped { a, b }, depth)))?;
                }
                spill(s, &mut queue)
            })?;
            fill(self, &mut fetched)?;
            resume += passes as usize * MATMUL_UNROLL as usize;
        }
        for i in resume..count {
            step(self, &mut fetched, i, ahead_of(i))?;
        }

        let mut col_inside = Vec::new();
        for &j in &local_cols {
            col_inside.push(self.less(j, n_c)?);
        }
        for (q, &row) in local_rows.iter().enumerate() {
            let row = self.add(row0, row)?;
            let row_inside = self.less(row, m_c)?;
            let row_at = self.mul(row, n_c)?;
            for (c, (&j, &col_inside)) in local_cols.iter().zip(&col_inside).enumerate() {
                let inside = self.and(row_inside, col_inside)?;
                let var = self.matmul_vars[q * per as usize + c];
                self.when(inside, |s| {
                    let at = s.add(row_at, j)?;
                    let sum = s.b.load(s.f32_ty, None, var, None, [])?;
                    match epilogue {
                        None => s.write(at, sum),
                        Some(Epilogue::Relu {
                            writes_pre_activation,
                        }) => {
                            if writes_pre_activation {
                                s.write_to(0, at, sum)?;
                            }
                            let activation = s.relu(sum)?;
                            s.write(at, activation)
                        }
                        Some(Epilogue::ReluGrad) => {
                            let pre_activation = s.read(2, at)?;
                            let gradient = s.relu_grad(pre_activation, sum)?;
                            s.write(at, gradient)
                        }
                    }
                })?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;

    /// `(m, k, n)` of each tile: wide, square, one row and two rows per invocation, the
    /// `mlp_sweep` b8192 h4096 and MNIST matmuls that read an operand transposed among them.
    const SHAPES: [(u32, u32, u32); 8] = [
        (784, 8192, 4096),
        (8192, 784, 4096),
        (8192, 10, 4096),
        (1031, 7, 1029),
        (100, 10, 64),
        (64, 100, 10),
        (784, 100, 64),
        (4096, 8192, 10),
    ];

    /// The staging elements of one operand's tile.
    fn total(tile: &MatMulTile, is_a: bool) -> u32 {
        if is_a {
            tile.rows_per_workgroup() * tile.depth
        } else {
            tile.depth * tile.cols
        }
    }

    /// Acceptance `staged-reads-coalesced`: consecutive staging elements, the elements of
    /// consecutive invocations, read consecutive addresses of the bound view of a transposed
    /// operand, in runs of [`TRANSPOSED_A_RUN`] of the block's rows (`a`, one 32-byte sector) or of
    /// the tile's depth (`b`), and the elements cover the tile once.
    #[test]
    fn a_transposed_operand_is_staged_from_consecutive_addresses_of_its_bound_view() {
        for (m, k, n) in SHAPES {
            let tile = MatMulTile::of(m.into(), k.into(), n.into());
            for is_a in [true, false] {
                let case = format!("[{m}, {k}] x [{k}, {n}] {tile} a {is_a}");
                let staging = Staging::of(&tile, m, k, n, is_a, true);
                let run = if is_a {
                    assert!(
                        tile.rows_per_workgroup().is_multiple_of(TRANSPOSED_A_RUN),
                        "{case}"
                    );
                    TRANSPOSED_A_RUN
                } else {
                    tile.depth
                };
                assert_eq!(staging.run, run, "{case}");
                assert!(staging.run >= 8.min(if is_a { m } else { k }), "{case}");
                let total = total(&tile, is_a);
                for e in 0..total - 1 {
                    if (e + 1) % staging.run == 0 {
                        continue;
                    }
                    for (origin, base) in [(0, 0), (64, tile.depth)] {
                        assert_eq!(
                            staging.element(e + 1, origin, base),
                            staging.element(e, origin, base) + 1,
                            "{case}: element {e} at origin {origin} base {base}"
                        );
                    }
                }
                let cells: BTreeSet<(u32, u32)> = (0..total).map(|e| staging.split(e)).collect();
                assert_eq!(cells.len(), total as usize, "{case}: each cell once");
                let across = if is_a {
                    tile.rows_per_workgroup()
                } else {
                    tile.cols
                };
                assert!(
                    cells
                        .iter()
                        .all(|&(along, a)| along < tile.depth && a < across),
                    "{case}: inside the tile"
                );
            }
        }
    }

    /// An operand read as is keeps the staging order of the plan before the change: `a` along
    /// the contracting index, `b` across the block's columns, each contiguous in its view.
    #[test]
    fn an_operand_read_as_is_keeps_its_staging_order() {
        for (m, k, n) in SHAPES {
            let tile = MatMulTile::of(m.into(), k.into(), n.into());
            let a = Staging::of(&tile, m, k, n, true, false);
            assert_eq!((a.run, a.along_first), (tile.depth, true));
            assert_eq!((a.along_stride, a.across_stride), (1, k));
            let b = Staging::of(&tile, m, k, n, false, false);
            assert_eq!((b.run, b.along_first), (tile.cols, false));
            assert_eq!((b.along_stride, b.across_stride), (n, 1));
        }
    }

    /// The workgroup memory of the RTX 3090 profile, in bytes.
    const RTX3090_SHARED_BYTES: u32 = 49_152;

    /// The workgroup memory address that staging element `e` of operand `a` (`is_a`) or `b`
    /// stores to in `layout`, as [`Shell::matmul`] computes it.
    fn staged_at(tile: &MatMulTile, layout: &MatMulLayout, is_a: bool, at: (u32, u32)) -> u32 {
        let (along, across) = at;
        if is_a && tile.vector_reads() {
            along * layout.a_stride + across
        } else if is_a {
            across * layout.a_stride + along
        } else if tile.cols_per_invocation == 1 {
            layout.b_offset + across * layout.b_stride + along
        } else {
            layout.b_offset + along * layout.b_stride + across
        }
    }

    /// The bank conflicts of staging operand `a` (`is_a`) or `b` of `tile`, bound transposed
    /// when `transposed`,
    /// into `layout`: over every store of 32 consecutive staging elements, the elements of the
    /// 32 invocations of one subgroup, the most distinct addresses one of 32 banks of 4 bytes
    /// takes. 1 is a store without conflicts.
    fn conflicts(
        tile: &MatMulTile,
        layout: &MatMulLayout,
        (m, k, n): (u32, u32, u32),
        is_a: bool,
        transposed: bool,
    ) -> u32 {
        let staging = Staging::of(tile, m, k, n, is_a, transposed);
        let total = total(tile, is_a);
        (0..total)
            .step_by(32)
            .map(|first| {
                let mut banks: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
                for e in first..(first + 32).min(total) {
                    let at = staged_at(tile, layout, is_a, staging.split(e));
                    banks.entry(at % 32).or_default().insert(at);
                }
                banks.values().map(|a| a.len() as u32).max().unwrap_or(0)
            })
            .max()
            .unwrap_or(0)
    }

    /// The flags of a matmul that reads only `a` (`is_a`) or only `b` transposed.
    fn only(is_a: bool) -> [bool; 2] {
        [is_a, !is_a]
    }

    /// Whether the padded layout of the tile removes the bank conflicts of the operand: a
    /// transposed `a` whose depth is a multiple of 4, staged 8 rows by 4 values of `k` per
    /// subgroup, and a transposed `b` of a register-blocked tile whose depth of 2 to 16 divides
    /// 32, staged `k`-major, 32 / depth columns per subgroup. A transposed `b` of a one-column
    /// tile is staged column-major at the stride it had before.
    fn padded(tile: &MatMulTile, is_a: bool) -> bool {
        if is_a {
            tile.depth >= 4 && tile.depth.is_multiple_of(4)
        } else {
            tile.cols_per_invocation > 1 && tile.depth > 1 && 32 % tile.depth == 0
        }
    }

    /// Acceptance `transposed-staging-padded`: the staging stores of a transposed `a` and of a
    /// transposed `b` of a register-blocked tile take one address per bank per subgroup store on
    /// a device with the workgroup memory of the RTX 3090, for the tiles of 128 invocations, the
    /// block of 256 and the `mlp_sweep` and MNIST shapes; unpadded they take 2 to 16.
    #[test]
    fn a_transposed_operand_is_staged_into_workgroup_memory_without_bank_conflicts() {
        let shapes = SHAPES.iter().map(|&s| (s, 128)).chain([
            ((8192, 784, 4096), 256),
            ((784, 8192, 4096), 256),
            ((784, 8192, 1024), 128),
            ((784, 8192, 256), 128),
            ((256, 8192, 784), 128),
        ]);
        let mut checked = 0;
        for ((m, k, n), limit) in shapes {
            let tile = MatMulTile::of_invocations(m.into(), k.into(), n.into(), limit);
            for is_a in [true, false] {
                if !padded(&tile, is_a) {
                    continue;
                }
                let case = format!("[{m}, {k}] x [{k}, {n}] {tile} a {is_a}");
                let layout = tile.layout_within(only(is_a), RTX3090_SHARED_BYTES);
                let before = conflicts(&tile, &tile.layout([false, false]), (m, k, n), is_a, true);
                let after = conflicts(&tile, &layout, (m, k, n), is_a, true);
                assert_eq!(after, 1, "{case}: {before} unpadded, {after} padded");
                checked += 1;
            }
        }
        assert!(checked >= 10, "{checked} operands checked");
    }

    /// The padding never adds conflicts and never moves an operand read as is: over sampled
    /// shapes and both invocation limits, a padded operand stores without conflicts, any other
    /// transposed operand stores as unpadded, and the strides of a matmul reading neither
    /// operand transposed are those of the tile before the change, but for the `k`-major `a`
    /// of a tile with vector reads ([`MatMulTile::vector_layout`]).
    #[test]
    fn the_padding_removes_conflicts_and_leaves_an_operand_read_as_is_where_it_was() {
        let mut state = 0x5eed_u64;
        let mut next = |bound: u64| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            1 + (z ^ (z >> 31)) % bound
        };
        for _ in 0..4000 {
            let (m, k, n) = (next(9000) as u32, next(9000) as u32, next(9000) as u32);
            for limit in [128, 256] {
                let tile = MatMulTile::of_invocations(m.into(), k.into(), n.into(), limit);
                let plain = tile.layout([false, false]);
                if tile.vector_reads() {
                    assert_eq!(plain.a_stride, tile.rows_per_workgroup() + 4, "{tile}");
                } else {
                    assert_eq!(plain.a_stride, tile.depth, "{tile}");
                }
                assert_eq!(plain.b_stride, tile.b_stride(), "{tile}");
                for is_a in [true, false] {
                    let case = format!("[{m}, {k}] x [{k}, {n}] {tile} a {is_a}");
                    let layout = tile.layout_within(only(is_a), RTX3090_SHARED_BYTES);
                    let before = conflicts(&tile, &plain, (m, k, n), is_a, true);
                    let after = conflicts(&tile, &layout, (m, k, n), is_a, true);
                    if padded(&tile, is_a) {
                        assert_eq!(after, 1, "{case}: {before} unpadded");
                    } else {
                        assert_eq!(layout, plain, "{case}");
                    }
                    assert!(after <= before, "{case}: {before} unpadded, {after} padded");
                }
            }
        }
    }

    /// The block of 256 invocations reads vectors ([`MatMulTile::vector_reads`]) and the tiles of
    /// 128 do not; its layout stages both operands `k`-major, every stride, offset and buffer a
    /// multiple of 4 values so that each vector read is aligned, and its staging stores of `a`
    /// and `b`, read as is or transposed, take one address per bank per subgroup store, for the
    /// `mlp_sweep` b8192 h4096 shapes and sampled ones.
    #[test]
    fn the_block_stages_k_major_aligned_for_vector_reads_without_bank_conflicts() {
        let mut shapes = vec![(8192, 784, 4096), (784, 8192, 4096), (8192, 64, 128)];
        let mut state = 0x000b_10c4_u64;
        for _ in 0..200 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let pick = |shift: u32, bound: u64, from: u64| from + (state >> shift) % bound;
            shapes.push((
                pick(10, 9000, 256) as u32,
                pick(30, 9000, 64) as u32,
                pick(45, 9000, 128) as u32,
            ));
        }
        let mut blocks = 0;
        for (m, k, n) in shapes {
            let tile = MatMulTile::of_invocations(m.into(), k.into(), n.into(), 256);
            assert_eq!(tile.vector_reads(), tile.workgroup_size == 256, "{tile}");
            assert!(!MatMulTile::of(m.into(), k.into(), n.into()).vector_reads());
            if !tile.vector_reads() {
                continue;
            }
            blocks += 1;
            for flags in [[false, false], [true, false], [false, true], [true, true]] {
                let layout = tile.layout_within(flags, RTX3090_SHARED_BYTES);
                let case = format!("[{m}, {k}] x [{k}, {n}] {tile} {flags:?} {layout:?}");
                assert_eq!(layout, tile.vector_layout(flags), "{case}");
                for value in [
                    layout.a_stride,
                    layout.b_stride,
                    layout.b_offset,
                    layout.buffer_values,
                ] {
                    assert!(value.is_multiple_of(4), "{case}");
                }
                assert!(layout.a_stride >= tile.rows_per_workgroup(), "{case}");
                assert!(layout.b_stride >= tile.cols, "{case}");
                for (is_a, transposed) in [(true, flags[0]), (false, flags[1])] {
                    let found = conflicts(&tile, &layout, (m, k, n), is_a, transposed);
                    assert_eq!(found, 1, "{case} a {is_a}");
                }
            }
        }
        assert!(blocks >= 50, "{blocks} blocks checked");
    }
}
