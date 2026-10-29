//! `modelspec inspect-device` on the stand-in loader of
//! `crates/modelspec-runtime/tests/adversary2_fake_loader.c` (device 0 at Vulkan 1.2.0, 1 at
//! 1.1.0, 2 at 1.0.0), compiled at test time with `cc` into `CARGO_TARGET_TMPDIR`.
#![cfg(feature = "vulkan")]

use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::OnceLock;

use modelspec_core::device::DeviceProfile;

fn stand_in_loader() -> &'static PathBuf {
    static LOADER: OnceLock<PathBuf> = OnceLock::new();
    LOADER.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("inspect_device_stand_in");
        std::fs::create_dir_all(&dir).expect("the scratch directory is created");
        let source = dir.join("stand_in_loader.c");
        let library = dir.join("libstand_in_vulkan.so");
        std::fs::write(
            &source,
            include_str!("../../modelspec-runtime/tests/adversary2_fake_loader.c"),
        )
        .expect("the loader source is written");
        let status = Command::new("cc")
            .args(["-shared", "-fPIC", "-o"])
            .arg(&library)
            .arg(&source)
            .status()
            .expect("cc runs");
        assert!(status.success(), "the stand-in loader compiles");
        library
    })
}

fn inspect(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["inspect-device", "--backend", "vulkan"])
        .args(args)
        .env("MODELSPEC_VULKAN_LIBRARY", stand_in_loader())
        .env_remove("MODELSPEC_VK_VALIDATION")
        .output()
        .expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn each_e_target_004_of_a_listing_names_its_device() {
    let output = inspect(&["--format", "text"]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    let refusals: Vec<&str> = stderr.split("E-TARGET-004: ").skip(1).collect();
    assert_eq!(refusals.len(), 2, "{stderr}");
    assert!(refusals[0].contains("\n  device: 1\n"), "{stderr}");
    assert!(refusals[1].contains("\n  device: 2"), "{stderr}");
}

#[test]
fn a_vulkan_1_0_profile_states_nothing_it_cannot_and_reads_back() {
    let output = inspect(&["--device", "2", "--format", "json"]);
    assert_eq!(output.status.code(), Some(1), "{}", text(&output.stderr));
    let profile = DeviceProfile::from_json(&text(&output.stdout)).expect("the profile reads back");
    assert_eq!(profile.vulkan_api_version, 1 << 22);
    assert_eq!(profile.spirv_version, 0x0001_0000);
    assert_eq!(profile.device_uuid, None);
    assert_eq!(profile.subgroup_size, None);
    assert!(profile.subgroup_stages.is_empty() && profile.subgroup_operations.is_empty());
    assert_eq!(profile.f16_storage, None);
    assert_eq!(profile.f16_compute, None);
    assert_eq!(profile.shader_rounding_mode_rte_f32, None);
    assert_eq!(profile.shader_denorm_preserve_f32, None);
    assert_eq!(profile.shader_signed_zero_inf_nan_preserve_f32, None);
}

#[test]
fn a_vulkan_1_1_profile_carries_its_1_1_values_and_leaves_the_1_2_ones_unstated() {
    let output = inspect(&["--device", "1", "--format", "json"]);
    assert_eq!(output.status.code(), Some(1), "{}", text(&output.stderr));
    let profile = DeviceProfile::from_json(&text(&output.stdout)).expect("the profile reads back");
    assert_eq!(profile.spirv_version, 0x0001_0300);
    assert_eq!(profile.subgroup_size, Some(32));
    assert!(profile.device_uuid.is_some());
    assert!(profile.f16_storage.is_some());
    assert_eq!(profile.f16_compute, None);
    assert_eq!(profile.shader_denorm_preserve_f32, None);
}

/// Under `MODELSPEC_VK_VALIDATION=1` the stand-in, which lists neither the validation layer nor
/// `VK_EXT_debug_utils`, gets an E-TARGET-003 refusal naming the extension, not a process abort
/// (story:vulkan-validation-without-debug-utils).
#[test]
fn validation_on_a_loader_without_debug_utils_is_e_target_003_not_an_abort() {
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["inspect-device", "--backend", "vulkan"])
        .env("MODELSPEC_VULKAN_LIBRARY", stand_in_loader())
        .env("MODELSPEC_VK_VALIDATION", "1")
        .output()
        .expect("modelspec runs");
    let stderr = text(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(1),
        "{:?}\n{stderr}",
        output.status
    );
    assert!(stderr.starts_with("E-TARGET-003: "), "{stderr}");
    assert!(stderr.contains("VK_EXT_debug_utils"), "{stderr}");
    assert!(stderr.contains("VK_LAYER_KHRONOS_validation"), "{stderr}");
}
