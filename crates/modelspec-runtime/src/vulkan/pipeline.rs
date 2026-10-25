//! One compute pipeline per SPIR-V module.
//!
//! The pipeline layout matches what `crate::spirv::emit` declares: descriptor set 0 with one
//! storage buffer per binding, and a 4-byte push constant (the element count) for the compute
//! stage.

use ash::vk;
use modelspec_core::ir::kernel::{Kernel, KernelModuleId, Operand};

use super::{Device, ExecError};

/// The bytes of the push constant every pipeline declares: the element count, a `u32`.
pub const PUSH_CONSTANT_BYTES: u32 = 4;

/// A compute pipeline, its layout and its shader module.
pub struct Pipeline<'d> {
    device: &'d ash::Device,
    /// The pipeline executable queries of the device, when it was created with them.
    statistics: Option<&'d ash::khr::pipeline_executable_properties::Device>,
    module: vk::ShaderModule,
    pub(crate) set_layout: vk::DescriptorSetLayout,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) bindings: u32,
    /// Per binding, whether it is a rank-0 broadcast operand (one element read by every
    /// invocation) rather than a tensor of the dispatch's element count.
    pub(crate) broadcast: Vec<bool>,
    pub(crate) local_size: [u32; 3],
}

impl Pipeline<'_> {
    /// The storage buffers the pipeline binds.
    pub fn bindings(&self) -> u32 {
        self.bindings
    }

    /// Per binding, whether it is a rank-0 broadcast operand.
    pub fn broadcast(&self) -> &[bool] {
        &self.broadcast
    }

    /// The workgroup size the module declares.
    pub fn local_size(&self) -> [u32; 3] {
        self.local_size
    }

    /// Every executable of the pipeline and the statistics the driver reports for it, in the
    /// order the driver lists them, through `VK_KHR_pipeline_executable_properties`; `None`
    /// when the device was created without the extension, so the pipeline captured none.
    pub fn statistics(&self) -> Result<Option<Vec<PipelineExecutable>>, ExecError> {
        let Some(queries) = self.statistics else {
            return Ok(None);
        };
        let info = vk::PipelineInfoKHR::default().pipeline(self.pipeline);
        // SAFETY: the pipeline was created on the device of `queries` with
        // `CAPTURE_STATISTICS_KHR`, and `info` outlives the call.
        let executables = unsafe { queries.get_pipeline_executable_properties(&info) }
            .map_err(ExecError::call("vkGetPipelineExecutablePropertiesKHR"))?;
        let mut read = Vec::with_capacity(executables.len());
        for (index, executable) in executables.iter().enumerate() {
            let info = vk::PipelineExecutableInfoKHR::default()
                .pipeline(self.pipeline)
                .executable_index(index as u32);
            // SAFETY: as above; `index` is below the count the driver returned.
            let statistics = unsafe { queries.get_pipeline_executable_statistics(&info) }
                .map_err(ExecError::call("vkGetPipelineExecutableStatisticsKHR"))?;
            read.push(PipelineExecutable {
                name: text(executable.name_as_c_str()),
                description: text(executable.description_as_c_str()),
                subgroup_size: executable.subgroup_size,
                statistics: statistics
                    .iter()
                    .map(|statistic| PipelineStatistic {
                        name: text(statistic.name_as_c_str()),
                        description: text(statistic.description_as_c_str()),
                        value: StatisticValue::of(statistic),
                    })
                    .collect(),
            });
        }
        Ok(Some(read))
    }
}

/// A string the driver returned in a fixed array, up to its NUL; empty without one.
fn text(read: Result<&std::ffi::CStr, std::ffi::FromBytesUntilNulError>) -> String {
    read.map(|text| text.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The value of one statistic, in the format the driver states for it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StatisticValue {
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
}

impl StatisticValue {
    fn of(statistic: &vk::PipelineExecutableStatisticKHR<'_>) -> Self {
        // SAFETY: `format` names the member of the union the driver wrote.
        unsafe {
            match statistic.format {
                vk::PipelineExecutableStatisticFormatKHR::BOOL32 => {
                    Self::Bool(statistic.value.b32 != vk::FALSE)
                }
                vk::PipelineExecutableStatisticFormatKHR::INT64 => Self::Int(statistic.value.i64),
                vk::PipelineExecutableStatisticFormatKHR::FLOAT64 => {
                    Self::Float(statistic.value.f64)
                }
                _ => Self::Uint(statistic.value.u64),
            }
        }
    }
}

impl std::fmt::Display for StatisticValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool(value) => write!(f, "{value}"),
            Self::Int(value) => write!(f, "{value}"),
            Self::Uint(value) => write!(f, "{value}"),
            Self::Float(value) => write!(f, "{value}"),
        }
    }
}

/// One statistic of a pipeline executable, as the driver names, describes and states it.
#[derive(Clone, Debug, PartialEq)]
pub struct PipelineStatistic {
    pub name: String,
    pub description: String,
    pub value: StatisticValue,
}

/// One executable of a pipeline (for a compute pipeline, its compute shader) and its statistics.
#[derive(Clone, Debug, PartialEq)]
pub struct PipelineExecutable {
    pub name: String,
    pub description: String,
    /// The subgroup size the executable was compiled for.
    pub subgroup_size: u32,
    pub statistics: Vec<PipelineStatistic>,
}

/// The executables of the pipeline of one kernel of a plan.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelStatistics {
    pub kernel_module_id: KernelModuleId,
    pub executables: Vec<PipelineExecutable>,
}

impl Drop for Pipeline<'_> {
    fn drop(&mut self) {
        // SAFETY: every object was created on `device`, null handles are ignored, and no
        // submitted work uses the pipeline any more (a `Submitted` borrows it and waits).
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_shader_module(self.module, None);
        }
    }
}

/// The operands of the `LocalSize` execution mode of `words`, if it declares one.
pub fn local_size(words: &[u32]) -> Option<[u32; 3]> {
    let mut at = 5;
    while at < words.len() {
        let length = (words[at] >> 16) as usize;
        let opcode = words[at] & 0xffff;
        if length == 0 || at + length > words.len() {
            return None;
        }
        // OpExecutionMode (16) with mode LocalSize (17): entry point, mode, x, y, z.
        if opcode == 16 && length == 6 && words[at + 2] == 17 {
            return Some([words[at + 3], words[at + 4], words[at + 5]]);
        }
        at += length;
    }
    None
}

/// The literal of every `OpDecorate … decoration <literal>` of `words`, sorted; `None` for a
/// truncated module.
fn decorations(words: &[u32], decoration: u32) -> Option<Vec<u32>> {
    let mut literals = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let length = (words[at] >> 16) as usize;
        let opcode = words[at] & 0xffff;
        if length == 0 || at + length > words.len() {
            return None;
        }
        // OpDecorate (71) with one literal: target, decoration, literal.
        if opcode == 71 && length == 4 && words[at + 2] == decoration {
            literals.push(words[at + 3]);
        }
        at += length;
    }
    literals.sort_unstable();
    Some(literals)
}

/// The operands of every `Binding` decoration of `words`, sorted; `None` for a truncated module.
pub fn declared_bindings(words: &[u32]) -> Option<Vec<u32>> {
    decorations(words, 33)
}

/// The operands of every `DescriptorSet` decoration of `words`, sorted; `None` for a truncated
/// module.
pub fn declared_sets(words: &[u32]) -> Option<Vec<u32>> {
    decorations(words, 34)
}

/// The storage-buffer bindings of a Kernel IR kernel, in binding order: each buffer operand,
/// then each written binding ([`Kernel::written`]). `true` marks a rank-0 operand, which every
/// invocation reads at index 0.
pub fn kernel_broadcast(kernel: &Kernel) -> Vec<bool> {
    kernel
        .operands
        .iter()
        .filter_map(|operand| match operand {
            Operand::Buffer(ty) => Some(ty.rank() == 0),
            Operand::Immediate(_) => None,
        })
        .chain(std::iter::repeat_n(false, kernel.written()))
        .collect()
}

impl Device<'_> {
    /// The compute pipeline of the SPIR-V module `words`, entry point `main`, with `bindings`
    /// storage buffers in descriptor set 0, each a tensor that a dispatch indexes by invocation.
    /// `bindings` must equal what the module's `Binding` decorations declare, 0 to
    /// `bindings - 1`, all in set 0, as `LocalSize` is read from the module too.
    pub fn pipeline(&self, words: &[u32], bindings: u32) -> Result<Pipeline<'_>, ExecError> {
        self.pipeline_with(words, vec![false; bindings as usize])
    }

    /// The compute pipeline of `kernel`, whose SPIR-V module is `words`: its bindings and which
    /// of them are rank-0 come from the kernel ([`kernel_broadcast`]).
    pub fn kernel_pipeline(
        &self,
        kernel: &Kernel,
        words: &[u32],
    ) -> Result<Pipeline<'_>, ExecError> {
        self.pipeline_with(words, kernel_broadcast(kernel))
    }

    fn pipeline_with(
        &self,
        words: &[u32],
        broadcast: Vec<bool>,
    ) -> Result<Pipeline<'_>, ExecError> {
        let bindings = broadcast.len() as u32;
        let local_size = local_size(words).ok_or(ExecError::NoLocalSize)?;
        let declared = declared_bindings(words).unwrap_or_default();
        let sets = declared_sets(words).unwrap_or_default();
        if !declared.iter().copied().eq(0..bindings) || sets != vec![0; declared.len()] {
            return Err(ExecError::ModuleBindings {
                declared,
                sets,
                requested: bindings,
            });
        }
        let mut pipeline = Pipeline {
            device: &self.device,
            statistics: self.pipeline_statistics.as_ref(),
            module: vk::ShaderModule::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            bindings,
            broadcast,
            local_size,
        };
        let info = vk::ShaderModuleCreateInfo::default().code(words);
        // SAFETY: `info` outlives the call; the module is destroyed by `pipeline`'s drop.
        pipeline.module = unsafe { self.device.create_shader_module(&info, None) }
            .map_err(ExecError::call("vkCreateShaderModule"))?;
        let slots: Vec<vk::DescriptorSetLayoutBinding<'_>> = (0..bindings)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&slots);
        // SAFETY: `info` and `slots` outlive the call.
        pipeline.set_layout = unsafe { self.device.create_descriptor_set_layout(&info, None) }
            .map_err(ExecError::call("vkCreateDescriptorSetLayout"))?;
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(PUSH_CONSTANT_BYTES)];
        let set_layouts = [pipeline.set_layout];
        let info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&ranges);
        // SAFETY: `info` and what it points to outlive the call.
        pipeline.layout = unsafe { self.device.create_pipeline_layout(&info, None) }
            .map_err(ExecError::call("vkCreatePipelineLayout"))?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(pipeline.module)
            .name(c"main");
        let flags = if pipeline.statistics.is_some() {
            vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR
        } else {
            vk::PipelineCreateFlags::empty()
        };
        let infos = [vk::ComputePipelineCreateInfo::default()
            .flags(flags)
            .stage(stage)
            .layout(pipeline.layout)];
        // SAFETY: `infos` and what it points to outlive the call.
        let created = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &infos, None)
        }
        .map_err(|(_, result)| ExecError::Vulkan {
            call: "vkCreateComputePipelines",
            result,
        })?;
        pipeline.pipeline = created[0];
        Ok(pipeline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_size_reads_the_execution_mode_and_refuses_a_truncated_module() {
        let header = [0x0723_0203, 0x0001_0500, 0, 9, 0];
        let mut words = header.to_vec();
        // OpCapability Shader, then OpExecutionMode %4 LocalSize 64 1 1.
        words.extend([(2 << 16) | 17, 1, (6 << 16) | 16, 4, 17, 64, 1, 1]);
        assert_eq!(local_size(&words), Some([64, 1, 1]));
        assert_eq!(local_size(&words[..words.len() - 1]), None);
        assert_eq!(local_size(&header), None);
    }

    #[test]
    fn declared_bindings_reads_the_binding_decorations_and_refuses_a_truncated_module() {
        let header = [0x0723_0203, 0x0001_0500, 0, 9, 0];
        let mut words = header.to_vec();
        // OpDecorate %7 Binding 1, OpDecorate %7 DescriptorSet 0, OpDecorate %5 Binding 0.
        words.extend([(4 << 16) | 71, 7, 33, 1, (4 << 16) | 71, 7, 34, 0]);
        words.extend([(4 << 16) | 71, 5, 33, 0]);
        assert_eq!(declared_bindings(&words), Some(vec![0, 1]));
        assert_eq!(declared_sets(&words), Some(vec![0]));
        assert_eq!(declared_bindings(&words[..words.len() - 1]), None);
        assert_eq!(declared_bindings(&header), Some(vec![]));
    }

    #[test]
    fn kernel_broadcast_marks_rank0_operands_and_never_the_result() {
        use modelspec_core::ir::kernel::{KernelKind, KernelModuleId};
        use modelspec_core::ir::tensor::{DType, TensorType};
        let kernel = Kernel {
            id: KernelModuleId([0; 16]),
            kind: KernelKind::Elementwise,
            op: "Mul",
            dtype: DType::F32,
            operands: vec![
                Operand::Buffer(TensorType::f32_scalar()),
                Operand::Immediate(2.0),
                Operand::Buffer(TensorType::f32(&[5])),
            ],
            result: TensorType::f32(&[5]),
            reduces: None,
            epilogue: None,
            transposed: [false, false],
            workgroup_size: 64,
            workgroups: [1, 1, 1],
        };
        assert_eq!(kernel_broadcast(&kernel), vec![true, false, false]);
    }
}
