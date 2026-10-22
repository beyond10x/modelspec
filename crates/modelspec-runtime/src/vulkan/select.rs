//! Device selection over the probed list: a pure function, so E-TARGET-003 is decided without a
//! Vulkan instance.

use modelspec_core::device::DeviceProfile;
use modelspec_core::diag::{Code, Diagnostic};

/// No Vulkan physical device was found, the device index names none of them, or the chosen
/// device has no compute queue family.
pub const E_TARGET_003: Code = Code::new("E-TARGET-003");

/// One probed physical device: its profile and the index of its first queue family with
/// compute support, if it has one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub profile: DeviceProfile,
    pub compute_queue_family: Option<u32>,
}

/// The indices of the devices to use: `Some(index)` selects that device, which must exist and
/// have a compute queue family; `None` selects every device. An empty list is refused either way.
pub fn select_device(
    candidates: &[Candidate],
    device: Option<usize>,
) -> Result<Vec<usize>, Diagnostic> {
    if candidates.is_empty() {
        return Err(Diagnostic::new(
            E_TARGET_003,
            "no Vulkan physical device was found",
        ));
    }
    let Some(index) = device else {
        return Ok((0..candidates.len()).collect());
    };
    let Some(candidate) = candidates.get(index) else {
        return Err(Diagnostic::new(
            E_TARGET_003,
            format!("there is no Vulkan physical device {index}"),
        )
        .with("device", index.to_string())
        .with("found", candidates.len().to_string()));
    };
    if candidate.compute_queue_family.is_none() {
        return Err(Diagnostic::new(
            E_TARGET_003,
            format!("Vulkan physical device {index} has no compute queue family"),
        )
        .with("device", index.to_string()));
    }
    Ok(vec![index])
}

#[cfg(test)]
mod tests {
    use super::*;
    use modelspec_core::device::{DeviceProfile, DeviceProfileId, Extent3D, MemoryHeap};

    fn profile() -> DeviceProfile {
        let extent = Extent3D { x: 1, y: 1, z: 1 };
        DeviceProfile {
            device_profile_id: DeviceProfileId::new(),
            vendor_id: 1,
            device_id: 2,
            device_uuid: None,
            driver_version: 3,
            subgroup_size: Some(32),
            subgroup_stages: vec!["VK_SHADER_STAGE_COMPUTE_BIT".to_owned()],
            subgroup_operations: vec!["VK_SUBGROUP_FEATURE_BASIC_BIT".to_owned()],
            max_workgroup_invocations: 1024,
            max_workgroup_size: extent.clone(),
            max_workgroup_count: extent,
            max_shared_memory_bytes: 49152,
            f16_storage: Some(false),
            f16_compute: Some(false),
            cooperative_matrix: false,
            timestamp_queries: true,
            memory_heaps: vec![MemoryHeap {
                size_bytes: 1 << 30,
                device_local: true,
                budget_bytes: None,
            }],
            vulkan_api_version: (1 << 22) | (2 << 12),
            spirv_version: 0x0001_0500,
            max_storage_buffer_range: u32::MAX,
            min_storage_buffer_offset_alignment: 16,
            cooperative_matrix_shapes: Vec::new(),
            shader_rounding_mode_rte_f32: Some(true),
            shader_denorm_preserve_f32: Some(true),
            shader_signed_zero_inf_nan_preserve_f32: Some(true),
        }
    }

    fn candidate(compute_queue_family: Option<u32>) -> Candidate {
        Candidate {
            profile: profile(),
            compute_queue_family,
        }
    }

    fn code(result: Result<Vec<usize>, modelspec_core::diag::Diagnostic>) -> String {
        result.unwrap_err().code.as_str().to_owned()
    }

    #[test]
    fn select_device_refuses_an_empty_list() {
        assert_eq!(code(select_device(&[], None)), "E-TARGET-003");
        assert_eq!(code(select_device(&[], Some(0))), "E-TARGET-003");
    }

    #[test]
    fn select_device_refuses_index_1_over_a_one_entry_list() {
        let refusal = select_device(&[candidate(Some(0))], Some(1)).unwrap_err();
        assert_eq!(refusal.code.as_str(), "E-TARGET-003");
        assert!(refusal.to_string().contains("device: 1"), "{refusal}");
    }

    #[test]
    fn select_device_refuses_an_entry_with_no_compute_queue_family() {
        assert_eq!(
            code(select_device(&[candidate(None)], Some(0))),
            "E-TARGET-003"
        );
    }

    #[test]
    fn select_device_picks_the_named_entry_or_lists_every_entry() {
        let list = [candidate(Some(0)), candidate(None), candidate(Some(2))];
        assert_eq!(select_device(&list, Some(2)).unwrap(), vec![2]);
        assert_eq!(select_device(&list, None).unwrap(), vec![0, 1, 2]);
    }
}
