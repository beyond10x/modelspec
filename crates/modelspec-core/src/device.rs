//! `modelspec.device.DeviceProfile` (`spec/domains/device.yaml`): one Vulkan physical device as
//! discovery recorded it (DESIGN.md § 5.1), as plain data with its invariants and its JSON form.
//!
//! Every value is the driver's own; nothing is inferred from a product name (DESIGN.md § 8).
//! No Vulkan type appears here: the runtime's `vulkan` module fills a profile, the compiler only
//! reads one (DESIGN.md § 7).

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// `modelspec.device.DeviceProfileId`: the identity of one probe's record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DeviceProfileId(pub Uuid);

impl DeviceProfileId {
    /// A fresh random identity.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for DeviceProfileId {
    fn default() -> Self {
        Self::new()
    }
}

/// `modelspec.device.Extent3D`: a size along x, y and z, each positive.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Extent3D {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// `modelspec.device.MemoryHeap`: one Vulkan memory heap and, with `VK_EXT_memory_budget`, its
/// budget.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryHeap {
    pub size_bytes: u64,
    pub device_local: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_bytes: Option<u64>,
}

/// `modelspec.device.CooperativeMatrixShape`: one cooperative-matrix property entry, component
/// types by Vulkan enum name.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CooperativeMatrixShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_type: String,
    pub b_type: String,
    pub c_type: String,
    pub result_type: String,
}

/// `modelspec.device.DeviceProfile`. Field order is the order of `spec/domains/device.yaml` and
/// of the JSON form.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceProfile {
    pub device_profile_id: DeviceProfileId,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_uuid: Option<String>,
    pub driver_version: u32,
    pub subgroup_size: Option<u32>,
    pub subgroup_stages: Vec<String>,
    pub subgroup_operations: Vec<String>,
    pub max_workgroup_invocations: u32,
    pub max_workgroup_size: Extent3D,
    pub max_workgroup_count: Extent3D,
    pub max_shared_memory_bytes: u32,
    pub f16_storage: Option<bool>,
    pub f16_compute: Option<bool>,
    pub cooperative_matrix: bool,
    pub timestamp_queries: bool,
    pub memory_heaps: Vec<MemoryHeap>,
    pub vulkan_api_version: u32,
    pub spirv_version: u32,
    pub max_storage_buffer_range: u32,
    pub min_storage_buffer_offset_alignment: u64,
    pub cooperative_matrix_shapes: Vec<CooperativeMatrixShape>,
    pub shader_rounding_mode_rte_f32: Option<bool>,
    pub shader_denorm_preserve_f32: Option<bool>,
    pub shader_signed_zero_inf_nan_preserve_f32: Option<bool>,
}

/// The fields of a profile besides its identity `device_profile_id`, in spec order.
pub const FIELDS: [&str; 24] = [
    "vendor_id",
    "device_id",
    "device_uuid",
    "driver_version",
    "subgroup_size",
    "subgroup_stages",
    "subgroup_operations",
    "max_workgroup_invocations",
    "max_workgroup_size",
    "max_workgroup_count",
    "max_shared_memory_bytes",
    "f16_storage",
    "f16_compute",
    "cooperative_matrix",
    "timestamp_queries",
    "memory_heaps",
    "vulkan_api_version",
    "spirv_version",
    "max_storage_buffer_range",
    "min_storage_buffer_offset_alignment",
    "cooperative_matrix_shapes",
    "shader_rounding_mode_rte_f32",
    "shader_denorm_preserve_f32",
    "shader_signed_zero_inf_nan_preserve_f32",
];

fn positive(name: &str, value: u64) -> Result<(), String> {
    if value > 0 {
        Ok(())
    } else {
        Err(format!("{name} > 0 (got {value})"))
    }
}

fn distinct(name: &str, values: &[String]) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    match values.iter().find(|value| !seen.insert(value.as_str())) {
        None => Ok(()),
        Some(value) => Err(format!("distinct {name} ({value} repeats)")),
    }
}

impl Extent3D {
    fn validate(&self, name: &str) -> Result<(), String> {
        positive(&format!("{name}.x"), self.x.into())?;
        positive(&format!("{name}.y"), self.y.into())?;
        positive(&format!("{name}.z"), self.z.into())
    }
}

impl MemoryHeap {
    fn validate(&self, name: &str) -> Result<(), String> {
        positive(&format!("{name}.size_bytes"), self.size_bytes)?;
        match self.budget_bytes {
            Some(budget) if budget > self.size_bytes => Err(format!(
                "{name}.budget_bytes <= size_bytes (got {budget} > {})",
                self.size_bytes
            )),
            _ => Ok(()),
        }
    }
}

impl DeviceProfile {
    /// Checks every invariant of `spec/domains/device.yaml` on the profile and its parts, and
    /// names the first one it breaks.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(size) = self.subgroup_size {
            positive("subgroup_size", size.into())?;
        }
        positive(
            "max_workgroup_invocations",
            self.max_workgroup_invocations.into(),
        )?;
        positive(
            "max_shared_memory_bytes",
            self.max_shared_memory_bytes.into(),
        )?;
        distinct("subgroup_stages", &self.subgroup_stages)?;
        distinct("subgroup_operations", &self.subgroup_operations)?;
        if !self.cooperative_matrix && !self.cooperative_matrix_shapes.is_empty() {
            return Err(format!(
                "cooperative_matrix_shapes is empty when cooperative_matrix is false (got {} \
                 shapes)",
                self.cooperative_matrix_shapes.len()
            ));
        }
        self.max_workgroup_size.validate("max_workgroup_size")?;
        self.max_workgroup_count.validate("max_workgroup_count")?;
        for (index, heap) in self.memory_heaps.iter().enumerate() {
            heap.validate(&format!("memory_heaps[{index}]"))?;
        }
        Ok(())
    }

    /// Reads the JSON form and checks the invariants.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let profile: Self =
            serde_json::from_str(json).map_err(|e| format!("not a DeviceProfile: {e}"))?;
        profile
            .validate()
            .map_err(|e| format!("DeviceProfile breaks invariant {e}"))?;
        Ok(profile)
    }

    /// The JSON form: pretty-printed with two-space indentation, fields in spec order, ending in
    /// a newline.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a DeviceProfile serializes") + "\n"
    }

    /// Each field of [`FIELDS`] with its value as compact JSON, in spec order, nested fields in
    /// spec order too.
    pub fn fields(&self) -> Vec<(&'static str, String)> {
        fn json(value: &impl Serialize) -> String {
            serde_json::to_string(value).expect("a DeviceProfile field serializes")
        }
        let values = [
            json(&self.vendor_id),
            json(&self.device_id),
            json(&self.device_uuid),
            json(&self.driver_version),
            json(&self.subgroup_size),
            json(&self.subgroup_stages),
            json(&self.subgroup_operations),
            json(&self.max_workgroup_invocations),
            json(&self.max_workgroup_size),
            json(&self.max_workgroup_count),
            json(&self.max_shared_memory_bytes),
            json(&self.f16_storage),
            json(&self.f16_compute),
            json(&self.cooperative_matrix),
            json(&self.timestamp_queries),
            json(&self.memory_heaps),
            json(&self.vulkan_api_version),
            json(&self.spirv_version),
            json(&self.max_storage_buffer_range),
            json(&self.min_storage_buffer_offset_alignment),
            json(&self.cooperative_matrix_shapes),
            json(&self.shader_rounding_mode_rte_f32),
            json(&self.shader_denorm_preserve_f32),
            json(&self.shader_signed_zero_inf_nan_preserve_f32),
        ];
        FIELDS.into_iter().zip(values).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../modelspec-runtime/tests/fixtures/rtx3090.device.json"
    ));

    fn fixture() -> DeviceProfile {
        DeviceProfile::from_json(FIXTURE).expect("the fixture is a valid profile")
    }

    fn refused(profile: DeviceProfile, invariant: &str) {
        let json = profile.to_json();
        let error = profile.validate().expect_err(invariant);
        assert!(error.contains(invariant), "{error}");
        let error = DeviceProfile::from_json(&json).expect_err(invariant);
        assert!(error.contains(invariant), "{error}");
    }

    #[test]
    fn device_fixture_writes_back_byte_for_byte() {
        assert_eq!(fixture().to_json(), FIXTURE);
    }

    #[test]
    fn device_fixture_keeps_every_invariant() {
        fixture().validate().unwrap();
    }

    #[test]
    fn device_subgroup_size_zero_is_refused() {
        let mut profile = fixture();
        profile.subgroup_size = Some(0);
        refused(profile, "subgroup_size > 0");
    }

    #[test]
    fn device_values_a_driver_does_not_state_are_absent_and_read_back() {
        let mut profile = fixture();
        profile.device_uuid = None;
        profile.subgroup_size = None;
        profile.subgroup_stages.clear();
        profile.subgroup_operations.clear();
        profile.f16_storage = None;
        profile.f16_compute = None;
        profile.shader_rounding_mode_rte_f32 = None;
        profile.shader_denorm_preserve_f32 = None;
        profile.shader_signed_zero_inf_nan_preserve_f32 = None;
        let json = profile.to_json();
        assert!(json.contains("\"subgroup_size\": null"), "{json}");
        assert!(
            json.contains("\"shader_denorm_preserve_f32\": null"),
            "{json}"
        );
        assert_eq!(DeviceProfile::from_json(&json).unwrap(), profile);
    }

    #[test]
    fn device_max_workgroup_invocations_zero_is_refused() {
        let mut profile = fixture();
        profile.max_workgroup_invocations = 0;
        refused(profile, "max_workgroup_invocations > 0");
    }

    #[test]
    fn device_max_shared_memory_bytes_zero_is_refused() {
        let mut profile = fixture();
        profile.max_shared_memory_bytes = 0;
        refused(profile, "max_shared_memory_bytes > 0");
    }

    #[test]
    fn device_repeated_subgroup_stage_is_refused() {
        let mut profile = fixture();
        let first = profile.subgroup_stages[0].clone();
        profile.subgroup_stages.push(first);
        refused(profile, "distinct subgroup_stages");
    }

    #[test]
    fn device_repeated_subgroup_operation_is_refused() {
        let mut profile = fixture();
        let first = profile.subgroup_operations[0].clone();
        profile.subgroup_operations.push(first);
        refused(profile, "distinct subgroup_operations");
    }

    #[test]
    fn device_cooperative_matrix_shapes_without_cooperative_matrix_are_refused() {
        let mut profile = fixture();
        profile.cooperative_matrix = false;
        profile.cooperative_matrix_shapes = vec![CooperativeMatrixShape {
            m: 16,
            n: 16,
            k: 16,
            a_type: "VK_COMPONENT_TYPE_FLOAT16_KHR".to_owned(),
            b_type: "VK_COMPONENT_TYPE_FLOAT16_KHR".to_owned(),
            c_type: "VK_COMPONENT_TYPE_FLOAT32_KHR".to_owned(),
            result_type: "VK_COMPONENT_TYPE_FLOAT32_KHR".to_owned(),
        }];
        refused(
            profile,
            "cooperative_matrix_shapes is empty when cooperative_matrix is false",
        );
    }

    #[test]
    fn device_cooperative_matrix_false_with_no_shapes_is_kept() {
        let mut profile = fixture();
        profile.cooperative_matrix = false;
        profile.cooperative_matrix_shapes.clear();
        profile.validate().unwrap();
    }

    #[test]
    fn device_zero_workgroup_extent_is_refused() {
        for axis in 0..3 {
            let mut profile = fixture();
            match axis {
                0 => profile.max_workgroup_size.x = 0,
                1 => profile.max_workgroup_size.y = 0,
                _ => profile.max_workgroup_count.z = 0,
            }
            refused(profile, "> 0");
        }
    }

    #[test]
    fn device_empty_memory_heap_is_refused() {
        let mut profile = fixture();
        profile.memory_heaps[0].size_bytes = 0;
        profile.memory_heaps[0].budget_bytes = None;
        refused(profile, "memory_heaps[0].size_bytes > 0");
    }

    #[test]
    fn device_budget_above_heap_size_is_refused() {
        let mut profile = fixture();
        profile.memory_heaps[0].budget_bytes = Some(profile.memory_heaps[0].size_bytes + 1);
        refused(profile, "memory_heaps[0].budget_bytes <= size_bytes");
    }

    #[test]
    fn device_unknown_json_field_is_refused() {
        let json = FIXTURE.replacen("{", "{\n  \"product_name\": \"RTX 3090\",", 1);
        let error = DeviceProfile::from_json(&json).unwrap_err();
        assert!(error.contains("product_name"), "{error}");
    }

    #[test]
    fn device_text_lines_name_the_24_fields_besides_the_identity() {
        let profile = fixture();
        let names: Vec<&str> = profile.fields().iter().map(|(name, _)| *name).collect();
        assert_eq!(names.len(), 24);
        let json: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
        let mut keys: Vec<&str> = json
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .filter(|key| *key != "device_profile_id")
            .collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        keys.sort_unstable();
        assert_eq!(sorted, keys);
        for (name, value) in profile.fields() {
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&value).unwrap(),
                json[name],
                "{name}"
            );
        }
    }
}
