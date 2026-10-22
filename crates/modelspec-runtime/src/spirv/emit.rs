//! The SPIR-V compute module of one M1 elementwise, broadcast or SGD kernel (DESIGN.md § 5.2).
//!
//! The module is SPIR-V 1.5, the version Vulkan 1.2 consumes, with capability `Shader`, the
//! `GLSL450` logical memory model and one `GLCompute` entry point `main`. Its only execution mode
//! is `LocalSize` with the kernel's workgroup size: no float-control mode is declared.
//!
//! Bindings, all in descriptor set 0: each buffer operand of the kernel in order (read, decorated
//! `NonWritable`), then the result (written). A push constant holds the element count; every
//! invocation whose flat index is not below it writes nothing. A rank-0 buffer operand is read at
//! element 0 by every invocation; an immediate operand is a constant of the module.

use modelspec_core::ir::kernel::{Kernel, KernelKind, Operand};
use rspirv::binary::Assemble;
use rspirv::dr::{self, Builder};
use rspirv::spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel,
    FunctionControl, MemoryModel, SelectionControl, StorageClass, Word,
};

/// Why a kernel has no module.
#[derive(Debug, thiserror::Error)]
pub enum EmitError {
    /// The kernel's kind and op are not an M1 elementwise, broadcast or SGD kernel.
    #[error("no SPIR-V emitter for {kind} kernel {op}")]
    Unsupported { kind: KernelKind, op: &'static str },
    /// The kernel's operands do not fit its op.
    #[error("{op} kernel has operands the emitter cannot bind: {reason}")]
    Operands { op: &'static str, reason: String },
    /// `rspirv` refused an instruction.
    #[error("the SPIR-V builder refused the module: {0}")]
    Build(String),
}

impl From<dr::Error> for EmitError {
    fn from(error: dr::Error) -> Self {
        Self::Build(error.to_string())
    }
}

/// The arithmetic of the supported kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arith {
    Add,
    Sub,
    Mul,
    Relu,
    ReluGrad,
    Copy,
}

impl Arith {
    fn of(kernel: &Kernel) -> Result<Self, EmitError> {
        let arith = match (kernel.kind, kernel.op) {
            (KernelKind::Elementwise, "Add") => Self::Add,
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
        }
    }
}

/// How one operand is read.
#[derive(Clone, Copy)]
enum Source {
    /// Buffer operand `binding`, at the invocation's index or, for a rank-0 tensor, at 0.
    Buffer { binding: u32, scalar: bool },
    /// A constant of the module.
    Immediate(f32),
}

/// The SPIR-V words of `kernel`'s compute module.
pub fn emit(kernel: &Kernel) -> Result<Vec<u32>, EmitError> {
    let arith = Arith::of(kernel)?;
    if kernel.operands.len() != arith.arity() {
        return Err(EmitError::Operands {
            op: kernel.op,
            reason: format!(
                "{} operands, the op takes {}",
                kernel.operands.len(),
                arith.arity()
            ),
        });
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
        return Err(EmitError::Operands {
            op: kernel.op,
            reason: "a scalar broadcast reads a rank-0 tensor or an immediate".to_owned(),
        });
    }
    let result_binding = binding;
    Ok(module(arith, &sources, result_binding, kernel.workgroup_size)?.assemble())
}

fn module(
    arith: Arith,
    sources: &[Source],
    result_binding: u32,
    workgroup_size: u32,
) -> Result<dr::Module, EmitError> {
    let mut b = Builder::new();
    b.set_version(1, 5);
    b.capability(Capability::Shader);
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
    let f_zero = b.constant_bit32(f32_ty, 0.0f32.to_bits());

    let gid = b.variable(input_v3u32, None, StorageClass::Input, None);
    b.decorate(
        gid,
        Decoration::BuiltIn,
        [dr::Operand::BuiltIn(BuiltIn::GlobalInvocationId)],
    );
    let nwg = b.variable(input_v3u32, None, StorageClass::Input, None);
    b.decorate(
        nwg,
        Decoration::BuiltIn,
        [dr::Operand::BuiltIn(BuiltIn::NumWorkgroups)],
    );
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
    let mut buffers = Vec::new();
    for source in sources {
        if let Source::Buffer { binding, .. } = source {
            buffers.push(bind(&mut b, read_ptr, *binding));
        }
    }
    let out = bind(&mut b, write_ptr, result_binding);
    let pc = b.variable(pc_ptr, None, StorageClass::PushConstant, None);

    let main = b.id();
    let mut interface = vec![gid, nwg];
    interface.extend(&buffers);
    interface.extend([out, pc]);
    b.entry_point(ExecutionModel::GLCompute, main, "main", interface);
    b.execution_mode(main, ExecutionMode::LocalSize, [workgroup_size, 1, 1]);

    b.begin_function(void, Some(main), FunctionControl::NONE, fn_void)?;
    b.begin_block(None)?;
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
    let count_ptr = b.access_chain(pc_u32_ptr, None, pc, [zero])?;
    let count = b.load(u32_ty, None, count_ptr, None, [])?;
    let inside = b.u_less_than(bool_ty, None, index, count)?;
    let body = b.id();
    let merge = b.id();
    b.selection_merge(merge, SelectionControl::NONE)?;
    b.branch_conditional(inside, body, merge, [])?;

    b.begin_block(Some(body))?;
    let mut values = Vec::new();
    let mut next_buffer = buffers.iter();
    for source in sources {
        let value = match *source {
            Source::Buffer { scalar, .. } => {
                let var = *next_buffer.next().expect("one variable per buffer operand");
                let at = if scalar { zero } else { index };
                let ptr = b.access_chain(element_ptr, None, var, [zero, at])?;
                b.load(f32_ty, None, ptr, None, [])?
            }
            Source::Immediate(value) => b.constant_bit32(f32_ty, value.to_bits()),
        };
        values.push(value);
    }
    let result = match arith {
        Arith::Add => b.f_add(f32_ty, None, values[0], values[1])?,
        Arith::Sub => b.f_sub(f32_ty, None, values[0], values[1])?,
        Arith::Mul => b.f_mul(f32_ty, None, values[0], values[1])?,
        Arith::Relu => {
            let positive = b.f_ord_greater_than(bool_ty, None, values[0], f_zero)?;
            b.select(f32_ty, None, positive, values[0], f_zero)?
        }
        Arith::ReluGrad => {
            let positive = b.f_ord_greater_than(bool_ty, None, values[0], f_zero)?;
            b.select(f32_ty, None, positive, values[1], f_zero)?
        }
        Arith::Copy => values[0],
    };
    let target = b.access_chain(element_ptr, None, out, [zero, index])?;
    b.store(target, result, None, [])?;
    b.branch(merge)?;

    b.begin_block(Some(merge))?;
    b.ret()?;
    b.end_function()?;
    Ok(b.module())
}
