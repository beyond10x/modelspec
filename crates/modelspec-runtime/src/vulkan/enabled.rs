//! What a `VkDeviceCreateInfo` enables, read back from the structure passed to
//! `vkCreateDevice`, so [`super::Device`] reports the set the driver was given rather than the
//! set the caller asked for.

use std::ffi::CStr;

use ash::vk;

/// The `VkPhysicalDeviceFeatures` members set to `VK_TRUE`, by Vulkan member name, in
/// declaration order.
pub(super) fn feature_names(features: &vk::PhysicalDeviceFeatures) -> Vec<String> {
    let f = features;
    [
        (f.robust_buffer_access, "robustBufferAccess"),
        (f.full_draw_index_uint32, "fullDrawIndexUint32"),
        (f.image_cube_array, "imageCubeArray"),
        (f.independent_blend, "independentBlend"),
        (f.geometry_shader, "geometryShader"),
        (f.tessellation_shader, "tessellationShader"),
        (f.sample_rate_shading, "sampleRateShading"),
        (f.dual_src_blend, "dualSrcBlend"),
        (f.logic_op, "logicOp"),
        (f.multi_draw_indirect, "multiDrawIndirect"),
        (f.draw_indirect_first_instance, "drawIndirectFirstInstance"),
        (f.depth_clamp, "depthClamp"),
        (f.depth_bias_clamp, "depthBiasClamp"),
        (f.fill_mode_non_solid, "fillModeNonSolid"),
        (f.depth_bounds, "depthBounds"),
        (f.wide_lines, "wideLines"),
        (f.large_points, "largePoints"),
        (f.alpha_to_one, "alphaToOne"),
        (f.multi_viewport, "multiViewport"),
        (f.sampler_anisotropy, "samplerAnisotropy"),
        (f.texture_compression_etc2, "textureCompressionETC2"),
        (f.texture_compression_astc_ldr, "textureCompressionASTC_LDR"),
        (f.texture_compression_bc, "textureCompressionBC"),
        (f.occlusion_query_precise, "occlusionQueryPrecise"),
        (f.pipeline_statistics_query, "pipelineStatisticsQuery"),
        (
            f.vertex_pipeline_stores_and_atomics,
            "vertexPipelineStoresAndAtomics",
        ),
        (f.fragment_stores_and_atomics, "fragmentStoresAndAtomics"),
        (
            f.shader_tessellation_and_geometry_point_size,
            "shaderTessellationAndGeometryPointSize",
        ),
        (f.shader_image_gather_extended, "shaderImageGatherExtended"),
        (
            f.shader_storage_image_extended_formats,
            "shaderStorageImageExtendedFormats",
        ),
        (
            f.shader_storage_image_multisample,
            "shaderStorageImageMultisample",
        ),
        (
            f.shader_storage_image_read_without_format,
            "shaderStorageImageReadWithoutFormat",
        ),
        (
            f.shader_storage_image_write_without_format,
            "shaderStorageImageWriteWithoutFormat",
        ),
        (
            f.shader_uniform_buffer_array_dynamic_indexing,
            "shaderUniformBufferArrayDynamicIndexing",
        ),
        (
            f.shader_sampled_image_array_dynamic_indexing,
            "shaderSampledImageArrayDynamicIndexing",
        ),
        (
            f.shader_storage_buffer_array_dynamic_indexing,
            "shaderStorageBufferArrayDynamicIndexing",
        ),
        (
            f.shader_storage_image_array_dynamic_indexing,
            "shaderStorageImageArrayDynamicIndexing",
        ),
        (f.shader_clip_distance, "shaderClipDistance"),
        (f.shader_cull_distance, "shaderCullDistance"),
        (f.shader_float64, "shaderFloat64"),
        (f.shader_int64, "shaderInt64"),
        (f.shader_int16, "shaderInt16"),
        (f.shader_resource_residency, "shaderResourceResidency"),
        (f.shader_resource_min_lod, "shaderResourceMinLod"),
        (f.sparse_binding, "sparseBinding"),
        (f.sparse_residency_buffer, "sparseResidencyBuffer"),
        (f.sparse_residency_image2_d, "sparseResidencyImage2D"),
        (f.sparse_residency_image3_d, "sparseResidencyImage3D"),
        (f.sparse_residency2_samples, "sparseResidency2Samples"),
        (f.sparse_residency4_samples, "sparseResidency4Samples"),
        (f.sparse_residency8_samples, "sparseResidency8Samples"),
        (f.sparse_residency16_samples, "sparseResidency16Samples"),
        (f.sparse_residency_aliased, "sparseResidencyAliased"),
        (f.variable_multisample_rate, "variableMultisampleRate"),
        (f.inherited_queries, "inheritedQueries"),
    ]
    .into_iter()
    .filter(|(value, _)| *value == vk::TRUE)
    .map(|(_, name)| name.to_owned())
    .collect()
}

/// The device extensions and optional features `info` enables, read from the structure itself.
///
/// # Safety
///
/// `info` must point only at live memory: its extension name pointers at NUL-terminated strings
/// and its feature pointer, if not null, at a `VkPhysicalDeviceFeatures`.
pub(super) unsafe fn of(info: &vk::DeviceCreateInfo) -> (Vec<String>, Vec<String>) {
    let extensions = if info.enabled_extension_count == 0 {
        Vec::new()
    } else {
        // SAFETY: the caller guarantees the array holds `enabled_extension_count` live names.
        unsafe {
            std::slice::from_raw_parts(
                info.pp_enabled_extension_names,
                info.enabled_extension_count as usize,
            )
            .iter()
            .map(|name| CStr::from_ptr(*name).to_string_lossy().into_owned())
            .collect()
        }
    };
    // SAFETY: the caller guarantees a non-null feature pointer is live.
    let features = match unsafe { info.p_enabled_features.as_ref() } {
        Some(features) => feature_names(features),
        None => Vec::new(),
    };
    (extensions, features)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enabled_reads_the_names_and_features_passed_in_the_create_info() {
        let names = [c"VK_KHR_a".as_ptr(), c"VK_EXT_b".as_ptr()];
        let features = vk::PhysicalDeviceFeatures::default()
            .robust_buffer_access(true)
            .shader_int16(true)
            .inherited_queries(true);
        let info = vk::DeviceCreateInfo::default()
            .enabled_extension_names(&names)
            .enabled_features(&features);
        // SAFETY: `names` and `features` outlive the call.
        let (extensions, enabled) = unsafe { of(&info) };
        assert_eq!(extensions, ["VK_KHR_a", "VK_EXT_b"]);
        assert_eq!(
            enabled,
            ["robustBufferAccess", "shaderInt16", "inheritedQueries"]
        );
    }

    #[test]
    fn enabled_is_empty_for_an_empty_create_info() {
        let features = vk::PhysicalDeviceFeatures::default();
        let info = vk::DeviceCreateInfo::default().enabled_features(&features);
        // SAFETY: `features` outlives the call.
        assert_eq!(unsafe { of(&info) }, (Vec::new(), Vec::new()));
        // SAFETY: a default create info points at nothing.
        assert_eq!(
            unsafe { of(&vk::DeviceCreateInfo::default()) },
            (Vec::new(), Vec::new())
        );
    }

    #[test]
    fn enabled_names_every_feature_member() {
        let all = vk::PhysicalDeviceFeatures {
            robust_buffer_access: vk::TRUE,
            full_draw_index_uint32: vk::TRUE,
            image_cube_array: vk::TRUE,
            independent_blend: vk::TRUE,
            geometry_shader: vk::TRUE,
            tessellation_shader: vk::TRUE,
            sample_rate_shading: vk::TRUE,
            dual_src_blend: vk::TRUE,
            logic_op: vk::TRUE,
            multi_draw_indirect: vk::TRUE,
            draw_indirect_first_instance: vk::TRUE,
            depth_clamp: vk::TRUE,
            depth_bias_clamp: vk::TRUE,
            fill_mode_non_solid: vk::TRUE,
            depth_bounds: vk::TRUE,
            wide_lines: vk::TRUE,
            large_points: vk::TRUE,
            alpha_to_one: vk::TRUE,
            multi_viewport: vk::TRUE,
            sampler_anisotropy: vk::TRUE,
            texture_compression_etc2: vk::TRUE,
            texture_compression_astc_ldr: vk::TRUE,
            texture_compression_bc: vk::TRUE,
            occlusion_query_precise: vk::TRUE,
            pipeline_statistics_query: vk::TRUE,
            vertex_pipeline_stores_and_atomics: vk::TRUE,
            fragment_stores_and_atomics: vk::TRUE,
            shader_tessellation_and_geometry_point_size: vk::TRUE,
            shader_image_gather_extended: vk::TRUE,
            shader_storage_image_extended_formats: vk::TRUE,
            shader_storage_image_multisample: vk::TRUE,
            shader_storage_image_read_without_format: vk::TRUE,
            shader_storage_image_write_without_format: vk::TRUE,
            shader_uniform_buffer_array_dynamic_indexing: vk::TRUE,
            shader_sampled_image_array_dynamic_indexing: vk::TRUE,
            shader_storage_buffer_array_dynamic_indexing: vk::TRUE,
            shader_storage_image_array_dynamic_indexing: vk::TRUE,
            shader_clip_distance: vk::TRUE,
            shader_cull_distance: vk::TRUE,
            shader_float64: vk::TRUE,
            shader_int64: vk::TRUE,
            shader_int16: vk::TRUE,
            shader_resource_residency: vk::TRUE,
            shader_resource_min_lod: vk::TRUE,
            sparse_binding: vk::TRUE,
            sparse_residency_buffer: vk::TRUE,
            sparse_residency_image2_d: vk::TRUE,
            sparse_residency_image3_d: vk::TRUE,
            sparse_residency2_samples: vk::TRUE,
            sparse_residency4_samples: vk::TRUE,
            sparse_residency8_samples: vk::TRUE,
            sparse_residency16_samples: vk::TRUE,
            sparse_residency_aliased: vk::TRUE,
            variable_multisample_rate: vk::TRUE,
            inherited_queries: vk::TRUE,
        };
        let names = feature_names(&all);
        assert_eq!(names.len(), 55);
        let distinct: std::collections::BTreeSet<&String> = names.iter().collect();
        assert_eq!(distinct.len(), 55);
    }
}
