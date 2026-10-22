//! The SPIR-V compute module of one M1 kernel (DESIGN.md § 5.2): elementwise, broadcast, SGD,
//! the two stages of a reduction, the `mse` loss sum, transpose and matmul.
//!
//! The module is SPIR-V 1.5, the version Vulkan 1.2 consumes, with capability `Shader`, the
//! `GLSL450` logical memory model and one `GLCompute` entry point `main`, with `LocalSize` the
//! kernel's workgroup size. [`emit_module`] adds the three float-control execution modes of
//! DESIGN.md § 5.4 (`RoundingModeRTE`, `DenormPreserve`, `SignedZeroInfNanPreserve`, width 32) and
//! their capabilities to a loss-sum or matmul module exactly when the device profile reports all
//! three for f32; no other module declares a float-control mode. Every `OpFAdd`, `OpFSub` and
//! `OpFMul` of a reduction, loss-sum or matmul module is decorated `NoContraction`, so no driver
//! fuses a multiply and an add; so are the multiply and the subtraction of an SGD update with
//! three operands, `p - (lr * g)` (DESIGN.md § 3.1).
//!
//! A matmul with a `relu` epilogue applies the compare and select of the `relu` kernel to its
//! rounded sum and writes that; when it keeps the pre-activation, it writes the sum first.
//!
//! Bindings, all in descriptor set 0: each buffer operand of the kernel in order (read, decorated
//! `NonWritable`), then what it writes: the pre-activation of an epilogue that keeps it, then the
//! result. A push constant holds the dispatch's count: the
//! output elements of an elementwise, transpose or matmul kernel, the elements a reduction stage
//! or a loss sum reads. A rank-0 buffer operand of an elementwise kernel is read at element 0 by
//! every invocation; an immediate operand is a constant of the module.
//!
//! The order of every sum is fixed by the shape alone (DESIGN.md § 3.3). Stage 1 of a reduction
//! sums 64 consecutive elements per workgroup with a pairwise tree in workgroup memory and writes
//! one partial per workgroup; stage 2 runs one workgroup whose invocation `i` sums partials `i`,
//! `i + 64`, … in ascending order and folds those 64 sums with the same tree. Neither stage adds a
//! padding term. A matmul stages tiles of its operands in workgroup memory ([`MatMulTile`]), and
//! each output still runs in one invocation, in ascending `k` order from the first term, as the
//! CPU interpreter does; a tile that needs more workgroup memory than the device profile's
//! `max_shared_memory_bytes` is refused with [`EmitError::SharedMemory`]. The loss sum runs one
//! workgroup of 64: its invocations stage the terms `(a - b) * (a - b)` in workgroup memory, as
//! many at a time as the device profile's `max_shared_memory_bytes` holds, and invocation 0 adds
//! them in ascending element order from the first term, chunk after chunk, as the CPU
//! interpreter does.

use std::collections::BTreeMap;

use modelspec_core::artifact::{CompiledArtifact, ContentHash};
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{
    Epilogue, Kernel, KernelKind, KernelModuleId, MatMulTile, Operand,
};
use modelspec_core::ir::tensor::{DType, Dim, TensorType};
use rspirv::binary::Assemble;
use rspirv::dr::{self, Builder};
use rspirv::spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel,
    FunctionControl, LoopControl, MemoryModel, SelectionControl, StorageClass, Word,
};

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
    /// The plan's module.
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
    Ok(build(kernel, Variant::Plan, false, MIN_SHARED_MEMORY_BYTES)?.assemble())
}

/// The module of `kernel` for a device with `profile`.
pub fn emit_module(kernel: &Kernel, profile: &DeviceProfile) -> Result<KernelModule, EmitError> {
    module_for(kernel, profile, Variant::Plan)
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
    module_for(kernel, profile, Variant::Undivided)
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
        .map(|kernel| emit_module(kernel, profile))
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
) -> Result<KernelModule, EmitError> {
    let float_controls = matches!(kernel.kind, KernelKind::LossSum | KernelKind::MatMul)
        && reports_float_controls(profile);
    let body = body_of(kernel, variant, profile.max_shared_memory_bytes)?;
    let chunk = match body {
        Body::LossSum { chunk, .. } => format!(" chunk {chunk}"),
        _ => String::new(),
    };
    let spirv = build(
        kernel,
        variant,
        float_controls,
        profile.max_shared_memory_bytes,
    )?
    .assemble();
    let mut shapes = Vec::new();
    for operand in &kernel.operands {
        if let Operand::Buffer(ty) = operand {
            shapes.push(static_shape(kernel, ty)?);
        }
    }
    let result = static_shape(kernel, &kernel.result)?;
    shapes.extend(std::iter::repeat_n(result, kernel.written()));
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
            fast_math: false,
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
    Stage1,
    Stage2 {
        divide_by: Option<u32>,
    },
    /// `terms` is the element count of the shape and `chunk` the number of terms staged in
    /// workgroup memory at a time.
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
    },
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
            let shapes = buffer_shapes(kernel, 2)?;
            let result = static_shape(kernel, &kernel.result)?;
            match (
                shapes[0].as_slice(),
                shapes[1].as_slice(),
                result.as_slice(),
            ) {
                ([m, k], [k2, n], [rm, rn]) if k == k2 && m == rm && n == rn && *k > 0 => {
                    let tile = MatMulTile::of(*m, *k, *n);
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
                    if tile.shared_bytes() > shared_bytes {
                        return Err(EmitError::SharedMemory {
                            op: kernel.op,
                            needs: tile.shared_bytes(),
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
                    }
                }
                _ => {
                    return Err(operands_error(
                        kernel,
                        "matmul reads [m, k] and [k, n], k > 0, into [m, n]",
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
            (KernelKind::Broadcast, "BroadcastScalar") => Self::Copy,
            (kind, op) => return Err(EmitError::Unsupported { kind, op }),
        };
        Ok(arith)
    }

    fn arity(self) -> usize {
        match self {
            Self::Relu | Self::Copy => 1,
            Self::Add | Self::Sub | Self::Mul | Self::ReluGrad => 2,
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

fn elementwise(kernel: &Kernel) -> Result<Body, EmitError> {
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
) -> Result<dr::Module, EmitError> {
    let body = body_of(kernel, variant, shared_bytes)?;
    let reads = kernel
        .operands
        .iter()
        .filter(|operand| matches!(operand, Operand::Buffer(_)))
        .count();
    let shared = match body {
        Body::Stage1 | Body::Stage2 { .. } => Some(TREE),
        Body::LossSum { chunk, .. } => Some(chunk),
        Body::MatMul { tile, .. } => Some(tile.shared_values()),
        _ => None,
    };
    let accumulators = match body {
        Body::MatMul { tile, .. } => tile.rows as usize,
        _ => 0,
    };
    if kernel.epilogue.is_some() && !matches!(body, Body::MatMul { .. }) {
        return Err(operands_error(
            kernel,
            "only a matmul kernel has an epilogue",
        ));
    }
    let exact = !matches!(body, Body::Elementwise(..));
    let mut m = Shell::new(
        reads,
        kernel.written(),
        kernel.workgroup_size,
        shared,
        accumulators,
        float_controls,
        exact,
    )?;
    match body {
        Body::Elementwise(arith, sources) => m.elementwise(arith, &sources)?,
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
        } => m.matmul(rows, k, n, epilogue, tile)?,
    }
    m.finish()
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
    /// The invocation's flat index over every workgroup.
    index: Word,
    /// The invocation's index in its workgroup, when the module uses workgroup memory.
    local: Option<Word>,
    /// The push-constant count.
    count: Word,
    /// Two `u32` and one `f32` function variables, declared first in the entry block.
    u32_vars: [Word; 2],
    f32_var: Word,
    /// The per-row accumulators of a matmul invocation, `f32` function variables.
    accumulators: Vec<Word>,
    /// The workgroup size, the `LocalSize` x of the entry point.
    group: Word,
    exact: bool,
    u32_constants: BTreeMap<u32, Word>,
    f32_constants: BTreeMap<u32, Word>,
}

impl Shell {
    fn new(
        reads: usize,
        writes: usize,
        workgroup_size: u32,
        shared: Option<u32>,
        accumulators: usize,
        float_controls: bool,
        exact: bool,
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
        let buffers: Vec<Word> = (0..reads as u32)
            .map(|binding| bind(&mut b, read_ptr, binding))
            .collect();
        let outs: Vec<Word> = (0..writes as u32)
            .map(|at| bind(&mut b, write_ptr, reads as u32 + at))
            .collect();
        let pc = b.variable(pc_ptr, None, StorageClass::PushConstant, None);
        let shared_var = if let Some(length) = shared {
            let length = match u32_constants.get(&length) {
                Some(&id) => id,
                None => {
                    let id = b.constant_bit32(u32_ty, length);
                    u32_constants.insert(length, id);
                    id
                }
            };
            let array = b.type_array(f32_ty, length);
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
        let accumulators = (0..accumulators)
            .map(|_| b.variable(f32_fn_ptr, None, StorageClass::Function, None))
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
            index,
            local,
            count,
            u32_vars,
            f32_var,
            accumulators,
            group,
            exact,
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

    fn shared_at(&mut self, at: Word) -> Result<Word, EmitError> {
        let shared = self.shared.expect("a module with workgroup memory");
        Ok(self.b.access_chain(self.shared_ptr, None, shared, [at])?)
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
                    let product = m.b.f_mul(f32_ty, None, values[1], values[2])?;
                    m.b.decorate(product, Decoration::NoContraction, []);
                    let difference = m.b.f_sub(f32_ty, None, values[0], product)?;
                    m.b.decorate(difference, Decoration::NoContraction, []);
                    difference
                }
                Arith::Copy => values[0],
            };
            m.write(index, result)
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
    /// at a time. For the chunk at element `base`, invocation `i` stores the term
    /// `(a - b) * (a - b)` of elements `base + i`, `base + i + 64`, … at its offset in workgroup
    /// memory; after a barrier invocation 0 adds the chunk's terms in ascending order, starting
    /// from the term of element 0 in the first chunk; a second barrier frees the memory for the
    /// next chunk. The order of additions is the one-invocation order: the first term, then each
    /// next one. `divide_by` divides the sum by the element count. The counts are constants of
    /// the shape, so the driver sees every trip count; no term is added for an empty shape.
    fn loss_sum(
        &mut self,
        divide_by: Option<u32>,
        terms: u32,
        chunk: u32,
    ) -> Result<(), EmitError> {
        if terms == 0 {
            return Ok(());
        }
        let chunk_var = self.u32_vars[0];
        let (full, tail) = (terms / chunk, terms % chunk);
        let zero = self.uint(0);
        self.loss_chunk(zero, chunk, true, full == 1 && tail == 0)?;
        if full > 1 {
            let (one, end, size) = (self.uint(1), self.uint(full), self.uint(chunk));
            self.counted(chunk_var, one, end, one, |m, c| {
                let base = m.mul(c, size)?;
                m.loss_chunk(base, chunk, false, false)
            })?;
        }
        if tail > 0 {
            let base = self.uint(full * chunk);
            self.loss_chunk(base, tail, false, true)?;
        }
        let local = self.local.expect("a module with workgroup memory");
        let first_invocation = self.equal(local, zero)?;
        self.when(first_invocation, |m| {
            let mut sum = m.b.load(m.f32_ty, None, m.f32_var, None, [])?;
            if let Some(n) = divide_by {
                let n = m.float(n as f32);
                sum = m.b.f_div(m.f32_ty, None, sum, n)?;
            }
            m.write(zero, sum)
        })
    }

    /// One chunk of [`Shell::loss_sum`]: the `len` terms from element `base`. The `opening` chunk
    /// starts the sum with its first term; the `closing` one needs no barrier after it.
    fn loss_chunk(
        &mut self,
        base: Word,
        len: u32,
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
            let term = m.f_mul(d, d)?;
            m.store_shared(j, term)
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
                let first = m.load_shared(zero)?;
                m.b.store(m.f32_var, first, None, [])?;
            }
            let start = m.uint(start);
            m.repeat(term_var, len - u32::from(opening), UNROLL, |m, t| {
                let j = m.add(t, start)?;
                let term = m.load_shared(j)?;
                m.accumulate(term)
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
    /// workgroup memory (`b` transposed), wait on a barrier, and each invocation adds the
    /// tile's products to each of its `rows` accumulators in ascending `k`, the first product of
    /// the first tile starting the sum. Values outside the matrices are staged but never enter
    /// a written output, and the last tile has its own trip count, so no padding term is added:
    /// every output element is `a[i, 0] * b[0, j]`, then each next product added in ascending
    /// `k`, as the CPU computes it. With an [`Epilogue`] the rounded sum is written to the
    /// pre-activation binding when the kernel keeps it, and its `relu` to the result.
    fn matmul(
        &mut self,
        m: u32,
        k: u32,
        n: u32,
        epilogue: Option<Epilogue>,
        tile: MatMulTile,
    ) -> Result<(), EmitError> {
        let local = self.local.expect("a module with workgroup memory");
        let u32_ty = self.u32_ty;
        let (size, cols, depth) = (tile.workgroup_size, tile.cols, tile.depth);
        let block_rows = tile.rows_per_workgroup();
        let thread_rows = size / cols;
        let a_stride = depth;
        let b_offset = block_rows * a_stride;

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
        let thread_col = self.b.u_mod(u32_ty, None, local, cols_c)?;
        let thread_row = self.b.u_div(u32_ty, None, local, cols_c)?;
        let j = self.add(col0, thread_col)?;
        let (m_c, k_c, n_c) = (self.uint(m), self.uint(k), self.uint(n));
        let depth_c = self.uint(depth);
        let a_stride_c = self.uint(a_stride);
        let b_offset_c = self.uint(b_offset);
        let b_stride_c = self.uint(tile.b_stride());
        let b_col = self.mul(thread_col, b_stride_c)?;
        let b_col = self.add(b_offset_c, b_col)?;
        let mut local_rows = Vec::new();
        for q in 0..tile.rows {
            let offset = self.uint(q * thread_rows);
            local_rows.push(self.add(thread_row, offset)?);
        }
        let mut a_rows = Vec::new();
        for &row in &local_rows {
            a_rows.push(self.mul(row, a_stride_c)?);
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
                let (along, to, at, valid) = if is_a {
                    let r = self.b.u_div(u32_ty, None, e, depth_c)?;
                    let c = self.b.u_mod(u32_ty, None, e, depth_c)?;
                    let row = self.add(row0, r)?;
                    let valid = self.less(row, m_c)?;
                    let row_at = self.mul(row, k_c)?;
                    let at = self.add(row_at, c)?;
                    let r_at = self.mul(r, a_stride_c)?;
                    let to = self.add(r_at, c)?;
                    (c, to, at, valid)
                } else {
                    let p = self.b.u_div(u32_ty, None, e, cols_c)?;
                    let c = self.b.u_mod(u32_ty, None, e, cols_c)?;
                    let col = self.add(col0, c)?;
                    let valid = self.less(col, n_c)?;
                    let p_at = self.mul(p, n_c)?;
                    let at = self.add(p_at, col)?;
                    let c_at = self.mul(c, b_stride_c)?;
                    let c_at = self.add(b_offset_c, c_at)?;
                    let to = self.add(c_at, p)?;
                    (p, to, at, valid)
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
                    valid,
                    taken,
                });
            }
        }

        // Reads this invocation's share of the tile of `len` values of the contracting index
        // from `base` into registers: each value with its slot.
        let fetch = |s: &mut Self, base: u32, len: u32| -> Result<Vec<(Word, &Slot)>, EmitError> {
            let mut fetched = Vec::new();
            for slot in &slots {
                let offset = s.uint(if slot.is_a { base } else { base * n });
                let mut at = s.add(slot.at, offset)?;
                if len < depth {
                    let len_c = s.uint(len);
                    let inside = s.less(slot.along, len_c)?;
                    let inside = s.and(slot.valid, inside)?;
                    let zero = s.uint(0);
                    at = s.pick(u32_ty, inside, at, zero)?;
                }
                let value = s.read(if slot.is_a { 0 } else { 1 }, at)?;
                fetched.push((value, slot));
            }
            Ok(fetched)
        };

        // Writes fetched values to the workgroup memory buffer at `buffer`.
        let store =
            |s: &mut Self, staged: Vec<(Word, &Slot)>, buffer: Word| -> Result<(), EmitError> {
                for (value, slot) in staged {
                    let (to, taken) = (slot.to, slot.taken);
                    let to = s.add(buffer, to)?;
                    match taken {
                        Some(taken) => s.when(taken, |s| s.store_shared(to, value))?,
                        None => s.store_shared(to, value)?,
                    }
                }
                Ok(())
            };

        // Adds the products of the first `len` values of the tile staged in the buffer at
        // `buffer` to the accumulators, the first product starting the sum when `opening`.
        let compute =
            |s: &mut Self, buffer: Word, len: u32, opening: bool| -> Result<(), EmitError> {
                let vars = s.accumulators.clone();
                let mut sums: Vec<Option<Word>> = Vec::new();
                for &var in &vars {
                    sums.push(if opening {
                        None
                    } else {
                        Some(s.b.load(s.f32_ty, None, var, None, [])?)
                    });
                }
                let b_start = s.add(buffer, b_col)?;
                let mut a_starts = Vec::new();
                for &a_row in &a_rows {
                    a_starts.push(s.add(buffer, a_row)?);
                }
                for p in 0..len {
                    let p_c = s.uint(p);
                    let b_at = s.add(b_start, p_c)?;
                    let b = s.load_shared(b_at)?;
                    for (sum, &a_start) in sums.iter_mut().zip(&a_starts) {
                        let a_at = s.add(a_start, p_c)?;
                        let a = s.load_shared(a_at)?;
                        let product = s.f_mul(a, b)?;
                        *sum = Some(match *sum {
                            None => product,
                            Some(acc) => s.f_add(acc, product)?,
                        });
                    }
                }
                for (&var, sum) in vars.iter().zip(sums) {
                    if let Some(sum) = sum {
                        s.b.store(var, sum, None, [])?;
                    }
                }
                Ok(())
            };
        let buffer_values = tile.buffer_values();
        let buffers = tile.buffers;
        let buffer_of =
            |s: &mut Self, tile_index: u32| s.uint((tile_index % buffers) * buffer_values);

        // The tiles of the contracting index, from `k = 0` up; the last may be short. Tile `i`
        // is read into registers `ahead` tiles before it is computed, so its reads overlap the
        // products of the tiles before it, and staged in buffer `i % 2`: one barrier per tile
        // both publishes it and frees the buffer of tile `i - 1` for tile `i + 1`.
        let ahead = tile.ahead.max(1) as usize;
        let tiles: Vec<(u32, u32)> = (0..k.div_ceil(depth))
            .map(|i| (i * depth, (k - i * depth).min(depth)))
            .collect();
        let mut fetched = std::collections::VecDeque::new();
        for &(base, len) in tiles.iter().take(ahead) {
            fetched.push_back(fetch(self, base, len)?);
        }
        let buffer = buffer_of(self, 0);
        let first = fetched.pop_front().expect("k is at least 1");
        store(self, first, buffer)?;
        self.barrier()?;
        for (i, &(_, len)) in tiles.iter().enumerate() {
            if let Some(&(base, next_len)) = tiles.get(i + ahead) {
                fetched.push_back(fetch(self, base, next_len)?);
            }
            let buffer = buffer_of(self, i as u32);
            compute(self, buffer, len, i == 0)?;
            if i + 1 < tiles.len() {
                let next = fetched.pop_front().expect("fetched ahead");
                let buffer = buffer_of(self, i as u32 + 1);
                store(self, next, buffer)?;
                self.barrier()?;
            }
        }

        for (q, &row) in local_rows.iter().enumerate() {
            let row = self.add(row0, row)?;
            let row_inside = self.less(row, m_c)?;
            let col_inside = self.less(j, n_c)?;
            let inside = self.and(row_inside, col_inside)?;
            let var = self.accumulators[q];
            self.when(inside, |s| {
                let row_at = s.mul(row, n_c)?;
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
                }
            })?;
        }
        Ok(())
    }
}
