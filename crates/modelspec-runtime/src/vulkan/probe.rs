//! Probing one physical device into a [`Candidate`]: every value from the Vulkan property and
//! feature chains, as the driver states it (DESIGN.md § 5.1, § 8).

use std::collections::BTreeSet;
use std::ffi::CStr;

use ash::vk;
use modelspec_core::device::{
    CooperativeMatrixShape, DeviceProfile, DeviceProfileId, Extent3D, MemoryHeap,
};
use modelspec_core::diag::Diagnostic;

use super::negotiate::REQUIRED_API_VERSION;
use super::select::{Candidate, E_TARGET_003};

/// `VkShaderStageFlagBits` by value and Vulkan name.
const SHADER_STAGES: &[(u32, &str)] = &[
    (0x1, "VK_SHADER_STAGE_VERTEX_BIT"),
    (0x2, "VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT"),
    (0x4, "VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT"),
    (0x8, "VK_SHADER_STAGE_GEOMETRY_BIT"),
    (0x10, "VK_SHADER_STAGE_FRAGMENT_BIT"),
    (0x20, "VK_SHADER_STAGE_COMPUTE_BIT"),
    (0x40, "VK_SHADER_STAGE_TASK_BIT_EXT"),
    (0x80, "VK_SHADER_STAGE_MESH_BIT_EXT"),
    (0x100, "VK_SHADER_STAGE_RAYGEN_BIT_KHR"),
    (0x200, "VK_SHADER_STAGE_ANY_HIT_BIT_KHR"),
    (0x400, "VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR"),
    (0x800, "VK_SHADER_STAGE_MISS_BIT_KHR"),
    (0x1000, "VK_SHADER_STAGE_INTERSECTION_BIT_KHR"),
    (0x2000, "VK_SHADER_STAGE_CALLABLE_BIT_KHR"),
    (0x4000, "VK_SHADER_STAGE_SUBPASS_SHADING_BIT_HUAWEI"),
    (0x8_0000, "VK_SHADER_STAGE_CLUSTER_CULLING_BIT_HUAWEI"),
];

/// `VkSubgroupFeatureFlagBits` by value and Vulkan name.
const SUBGROUP_OPERATIONS: &[(u32, &str)] = &[
    (0x1, "VK_SUBGROUP_FEATURE_BASIC_BIT"),
    (0x2, "VK_SUBGROUP_FEATURE_VOTE_BIT"),
    (0x4, "VK_SUBGROUP_FEATURE_ARITHMETIC_BIT"),
    (0x8, "VK_SUBGROUP_FEATURE_BALLOT_BIT"),
    (0x10, "VK_SUBGROUP_FEATURE_SHUFFLE_BIT"),
    (0x20, "VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT"),
    (0x40, "VK_SUBGROUP_FEATURE_CLUSTERED_BIT"),
    (0x80, "VK_SUBGROUP_FEATURE_QUAD_BIT"),
    (0x100, "VK_SUBGROUP_FEATURE_PARTITIONED_BIT_NV"),
    (0x200, "VK_SUBGROUP_FEATURE_ROTATE_BIT"),
    (0x400, "VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT"),
];

/// `VkComponentTypeKHR` by value and Vulkan name.
const COMPONENT_TYPES: &[(i32, &str)] = &[
    (0, "VK_COMPONENT_TYPE_FLOAT16_KHR"),
    (1, "VK_COMPONENT_TYPE_FLOAT32_KHR"),
    (2, "VK_COMPONENT_TYPE_FLOAT64_KHR"),
    (3, "VK_COMPONENT_TYPE_SINT8_KHR"),
    (4, "VK_COMPONENT_TYPE_SINT16_KHR"),
    (5, "VK_COMPONENT_TYPE_SINT32_KHR"),
    (6, "VK_COMPONENT_TYPE_SINT64_KHR"),
    (7, "VK_COMPONENT_TYPE_UINT8_KHR"),
    (8, "VK_COMPONENT_TYPE_UINT16_KHR"),
    (9, "VK_COMPONENT_TYPE_UINT32_KHR"),
    (10, "VK_COMPONENT_TYPE_UINT64_KHR"),
    (1_000_141_000, "VK_COMPONENT_TYPE_BFLOAT16_KHR"),
    (1_000_491_000, "VK_COMPONENT_TYPE_SINT8_PACKED_NV"),
    (1_000_491_001, "VK_COMPONENT_TYPE_UINT8_PACKED_NV"),
    (1_000_491_002, "VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT"),
    (1_000_491_003, "VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT"),
];

/// The names of the bits set in `flags`, in bit order; a bit without a name is written as hex.
fn bit_names(flags: u32, names: &[(u32, &str)]) -> Vec<String> {
    (0..32)
        .map(|bit| 1u32 << bit)
        .filter(|bit| flags & bit != 0)
        .map(|bit| match names.iter().find(|(value, _)| *value == bit) {
            Some((_, name)) => (*name).to_owned(),
            None => format!("0x{bit:x}"),
        })
        .collect()
}

fn component_type(value: vk::ComponentTypeKHR) -> String {
    let raw = value.as_raw();
    match COMPONENT_TYPES.iter().find(|(known, _)| *known == raw) {
        Some((_, name)) => (*name).to_owned(),
        None => format!("VkComponentTypeKHR({raw})"),
    }
}

/// A UUID in its hyphenated lowercase form.
fn uuid(bytes: &[u8; vk::UUID_SIZE]) -> String {
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// The highest SPIR-V version a device accepts, packed as in the SPIR-V header, from its API
/// version with the variant bits dropped: SPIR-V 1.0 in Vulkan 1.0, 1.3 in Vulkan 1.1, 1.5 in
/// Vulkan 1.2 and 1.6 from Vulkan 1.3 (the Vulkan specification, "SPIR-V Environment").
fn spirv_version(api_version: u32) -> u32 {
    let version = (
        vk::api_version_major(api_version),
        vk::api_version_minor(api_version),
    );
    if version >= (1, 3) {
        0x0001_0600
    } else if version >= (1, 2) {
        0x0001_0500
    } else if version >= (1, 1) {
        0x0001_0300
    } else {
        0x0001_0000
    }
}

/// The subgroup properties a driver states.
struct Subgroup {
    size: u32,
    stages: u32,
    operations: u32,
}

/// The values a driver states through the Vulkan 1.1 and 1.2 query structures; `None` where the
/// device's API version has no structure that states it.
#[derive(Default)]
struct Stated {
    device_uuid: Option<String>,
    subgroup: Option<Subgroup>,
    f16_storage: Option<bool>,
    f16_compute: Option<bool>,
    rounding_mode_rte_f32: Option<bool>,
    denorm_preserve_f32: Option<bool>,
    signed_zero_inf_nan_preserve_f32: Option<bool>,
}

fn bool_of(value: vk::Bool32) -> bool {
    value == vk::TRUE
}

/// Probes physical device `index` of `instance`.
pub(super) fn candidate(
    entry: &ash::Entry,
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    index: usize,
) -> Result<Candidate, Diagnostic> {
    // SAFETY: `device` was enumerated from `instance`, which is alive for every call below, and
    // each chained structure outlives the call that fills it.
    unsafe {
        let properties = instance.get_physical_device_properties(device);
        let reported = properties.api_version & !(0x7 << 29);
        // A device below 1.2 has no Vulkan 1.2 property and feature chains; its profile records
        // what plain Vulkan 1.0 queries state and `negotiate` refuses it.
        let current = reported >= REQUIRED_API_VERSION;
        let listing_failed = |error: vk::Result| {
            Diagnostic::new(
                E_TARGET_003,
                format!("the extensions of Vulkan physical device {index} could not be listed"),
            )
            .with("error", error.to_string())
        };
        let extensions: BTreeSet<String> = instance
            .enumerate_device_extension_properties(device)
            .map_err(listing_failed)?
            .iter()
            .filter_map(|extension| extension.extension_name_as_c_str().ok())
            .map(|name| name.to_string_lossy().into_owned())
            .collect();
        let has = |name: &CStr| extensions.contains(name.to_str().expect("an ASCII name"));

        let stated = if current {
            let mut vulkan11 = vk::PhysicalDeviceVulkan11Properties::default();
            let mut float_controls = vk::PhysicalDeviceFloatControlsProperties::default();
            let mut properties2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut vulkan11)
                .push_next(&mut float_controls);
            instance.get_physical_device_properties2(device, &mut properties2);
            let mut features11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut features11)
                .push_next(&mut features12);
            instance.get_physical_device_features2(device, &mut features2);
            Stated {
                device_uuid: Some(uuid(&vulkan11.device_uuid)),
                subgroup: Some(Subgroup {
                    size: vulkan11.subgroup_size,
                    stages: vulkan11.subgroup_supported_stages.as_raw(),
                    operations: vulkan11.subgroup_supported_operations.as_raw(),
                }),
                f16_storage: Some(bool_of(features11.storage_buffer16_bit_access)),
                f16_compute: Some(bool_of(features12.shader_float16)),
                rounding_mode_rte_f32: Some(bool_of(
                    float_controls.shader_rounding_mode_rte_float32,
                )),
                denorm_preserve_f32: Some(bool_of(float_controls.shader_denorm_preserve_float32)),
                signed_zero_inf_nan_preserve_f32: Some(bool_of(
                    float_controls.shader_signed_zero_inf_nan_preserve_float32,
                )),
            }
        } else if reported >= vk::API_VERSION_1_1 {
            // Vulkan 1.1 core: the subgroup and ID properties and 16-bit storage. Float16
            // arithmetic and the float controls are Vulkan 1.2 and stay unstated.
            let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
            let mut id = vk::PhysicalDeviceIDProperties::default();
            let mut properties2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut subgroup)
                .push_next(&mut id);
            instance.get_physical_device_properties2(device, &mut properties2);
            let mut storage16 = vk::PhysicalDevice16BitStorageFeatures::default();
            let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut storage16);
            instance.get_physical_device_features2(device, &mut features2);
            Stated {
                device_uuid: Some(uuid(&id.device_uuid)),
                subgroup: Some(Subgroup {
                    size: subgroup.subgroup_size,
                    stages: subgroup.supported_stages.as_raw(),
                    operations: subgroup.supported_operations.as_raw(),
                }),
                f16_storage: Some(bool_of(storage16.storage_buffer16_bit_access)),
                ..Stated::default()
            }
        } else {
            Stated::default()
        };

        let has_budget = reported >= vk::API_VERSION_1_1 && has(ash::ext::memory_budget::NAME);
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let memory = if reported >= vk::API_VERSION_1_1 {
            let mut memory2 = vk::PhysicalDeviceMemoryProperties2::default();
            if has_budget {
                memory2 = memory2.push_next(&mut budget);
            }
            instance.get_physical_device_memory_properties2(device, &mut memory2);
            memory2.memory_properties
        } else {
            instance.get_physical_device_memory_properties(device)
        };
        let memory_heaps = memory
            .memory_heaps_as_slice()
            .iter()
            .enumerate()
            .map(|(heap, properties)| MemoryHeap {
                size_bytes: properties.size,
                device_local: properties.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
                budget_bytes: has_budget.then(|| budget.heap_budget[heap]),
            })
            .collect();

        let families = instance.get_physical_device_queue_family_properties(device);
        let compute_queue_family = families
            .iter()
            .position(|family| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|family| family as u32);
        let timestamp_queries = match compute_queue_family {
            Some(family) => families[family as usize].timestamp_valid_bits > 0,
            None => bool_of(properties.limits.timestamp_compute_and_graphics),
        };

        let khr_cooperative_matrix = has(ash::khr::cooperative_matrix::NAME);
        let cooperative_matrix = khr_cooperative_matrix || has(ash::nv::cooperative_matrix::NAME);
        let cooperative_matrix_shapes = if khr_cooperative_matrix {
            ash::khr::cooperative_matrix::Instance::new(entry, instance)
                .get_physical_device_cooperative_matrix_properties(device)
                .map_err(|error| {
                    Diagnostic::new(
                        E_TARGET_003,
                        format!(
                            "the cooperative-matrix properties of Vulkan physical device {index} \
                             could not be read"
                        ),
                    )
                    .with("error", error.to_string())
                })?
                .iter()
                .map(|shape| CooperativeMatrixShape {
                    m: shape.m_size,
                    n: shape.n_size,
                    k: shape.k_size,
                    a_type: component_type(shape.a_type),
                    b_type: component_type(shape.b_type),
                    c_type: component_type(shape.c_type),
                    result_type: component_type(shape.result_type),
                })
                .collect()
        } else {
            Vec::new()
        };

        let limits = properties.limits;
        let extent = |size: [u32; 3]| Extent3D {
            x: size[0],
            y: size[1],
            z: size[2],
        };
        let profile = DeviceProfile {
            device_profile_id: DeviceProfileId::new(),
            vendor_id: properties.vendor_id,
            device_id: properties.device_id,
            device_uuid: stated.device_uuid,
            driver_version: properties.driver_version,
            subgroup_size: stated.subgroup.as_ref().map(|subgroup| subgroup.size),
            subgroup_stages: stated.subgroup.as_ref().map_or_else(Vec::new, |subgroup| {
                bit_names(subgroup.stages, SHADER_STAGES)
            }),
            subgroup_operations: stated.subgroup.as_ref().map_or_else(Vec::new, |subgroup| {
                bit_names(subgroup.operations, SUBGROUP_OPERATIONS)
            }),
            max_workgroup_invocations: limits.max_compute_work_group_invocations,
            max_workgroup_size: extent(limits.max_compute_work_group_size),
            max_workgroup_count: extent(limits.max_compute_work_group_count),
            max_shared_memory_bytes: limits.max_compute_shared_memory_size,
            f16_storage: stated.f16_storage,
            f16_compute: stated.f16_compute,
            cooperative_matrix,
            timestamp_queries,
            memory_heaps,
            vulkan_api_version: properties.api_version,
            spirv_version: spirv_version(reported),
            max_storage_buffer_range: limits.max_storage_buffer_range,
            min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment,
            cooperative_matrix_shapes,
            shader_rounding_mode_rte_f32: stated.rounding_mode_rte_f32,
            shader_denorm_preserve_f32: stated.denorm_preserve_f32,
            shader_signed_zero_inf_nan_preserve_f32: stated.signed_zero_inf_nan_preserve_f32,
        };
        Ok(Candidate {
            profile,
            compute_queue_family,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_names_flag_bits_in_bit_order_and_unknown_bits_as_hex() {
        assert_eq!(
            bit_names(0x20 | 0x1 | 0x4000_0000, SHADER_STAGES),
            [
                "VK_SHADER_STAGE_VERTEX_BIT",
                "VK_SHADER_STAGE_COMPUTE_BIT",
                "0x40000000"
            ]
        );
    }

    #[test]
    fn probe_writes_a_uuid_hyphenated() {
        let bytes: [u8; 16] = [
            0xa3, 0x24, 0x1a, 0xed, 0x6b, 0x48, 0x2e, 0x51, 0x63, 0x78, 0xb6, 0xd9, 0xd0, 0x25,
            0x52, 0x99,
        ];
        assert_eq!(uuid(&bytes), "a3241aed-6b48-2e51-6378-b6d9d0255299");
    }

    #[test]
    fn probe_spirv_version_follows_the_api_version() {
        assert_eq!(spirv_version(vk::API_VERSION_1_0), 0x0001_0000);
        assert_eq!(spirv_version(vk::API_VERSION_1_1), 0x0001_0300);
        assert_eq!(spirv_version(vk::API_VERSION_1_2), 0x0001_0500);
        assert_eq!(
            spirv_version(vk::make_api_version(1, 1, 1, 0)),
            0x0001_0300,
            "the variant bits do not raise the version"
        );
        assert_eq!(spirv_version(vk::API_VERSION_1_3), 0x0001_0600);
        assert_eq!(
            spirv_version(vk::make_api_version(0, 1, 4, 341)),
            0x0001_0600
        );
    }
}
