//! Adversary pass 2 for `story:m1-inspect-device`: what `inspect-device` reports for a device
//! below Vulkan 1.2, against `spec/domains/device.yaml` and DESIGN.md § 5.1 and § 8 ("prints the
//! device profile ... as the driver reports it").
//!
//! The loader is the stand-in `crates/modelspec-runtime/tests/adversary2_fake_loader.c`, compiled
//! at test time with `cc` into `CARGO_TARGET_TMPDIR`: device 0 reports Vulkan 1.2.0, device 1
//! Vulkan 1.1.0, device 2 Vulkan 1.0.0, and every one of them a subgroup size of 32 through the
//! Vulkan 1.1 structures.
#![cfg(feature = "vulkan")]

use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::OnceLock;

use modelspec_core::device::DeviceProfile;

const FAKE_LOADER: &str = include_str!("../../modelspec-runtime/tests/adversary2_fake_loader.c");

fn fake_loader() -> &'static PathBuf {
    static LOADER: OnceLock<PathBuf> = OnceLock::new();
    LOADER.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("adversary2_inspect_pre_1_2");
        std::fs::create_dir_all(&dir).expect("the scratch directory is created");
        let source = dir.join("fake_loader.c");
        let library = dir.join("libfake_vulkan.so");
        std::fs::write(&source, FAKE_LOADER).expect("the loader source is written");
        let status = Command::new("cc")
            .args(["-shared", "-fPIC", "-o"])
            .arg(&library)
            .arg(&source)
            .status()
            .expect("cc runs");
        assert!(status.success(), "the fake loader compiles");
        library
    })
}

fn inspect(device: &str, format: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args([
            "inspect-device",
            "--backend",
            "vulkan",
            "--device",
            device,
            "--format",
            format,
        ])
        .env("MODELSPEC_VULKAN_LIBRARY", fake_loader())
        .env_remove("MODELSPEC_VK_VALIDATION")
        .output()
        .expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Control: the 1.2 device of the same loader gives a profile that reads back.
#[test]
fn adversary2_control_the_vulkan_1_2_device_profile_reads_back() {
    let output = inspect("0", "json");
    let stdout = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    DeviceProfile::from_json(&stdout).expect("the 1.2 profile keeps its invariants");
}

/// The printed record of a refused Vulkan 1.1 device is still a `DeviceProfile`: it must keep
/// the invariants of `spec/domains/device.yaml` (`subgroup_size > 0`). The driver reports 32
/// through `VkPhysicalDeviceSubgroupProperties`, which is core in Vulkan 1.1.
#[test]
fn adversary2_a_vulkan_1_1_device_profile_keeps_the_spec_invariants() {
    let output = inspect("1", "json");
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.starts_with("E-TARGET-004: "), "{stderr}");
    let profile = DeviceProfile::from_json(&stdout);
    assert!(profile.is_ok(), "{profile:?}\n{stdout}");
}

/// `spirv_version` is the highest SPIR-V version the device accepts (`spec/domains/device.yaml`).
/// A Vulkan 1.1 device accepts SPIR-V 1.0 to 1.3 (Vulkan specification, "SPIR-V Environment");
/// the profile must not claim SPIR-V 1.5 for it.
#[test]
fn adversary2_a_vulkan_1_1_device_does_not_claim_spirv_1_5() {
    let output = inspect("1", "text");
    let stdout = text(&output.stdout);
    let line = stdout
        .lines()
        .find(|line| line.starts_with("spirv_version: "))
        .expect("a spirv_version line");
    let value: u32 = line["spirv_version: ".len()..].parse().expect("a number");
    assert!(
        value <= 0x0001_0300,
        "a Vulkan 1.1 device is reported as accepting SPIR-V {}.{}\n{stdout}",
        value >> 16,
        (value >> 8) & 0xff
    );
}

/// A Vulkan 1.0 device accepts SPIR-V 1.0 only; the profile must not claim SPIR-V 1.5 for it.
#[test]
fn adversary2_a_vulkan_1_0_device_does_not_claim_spirv_1_5() {
    let output = inspect("2", "text");
    let stdout = text(&output.stdout);
    let line = stdout
        .lines()
        .find(|line| line.starts_with("spirv_version: "))
        .expect("a spirv_version line");
    let value: u32 = line["spirv_version: ".len()..].parse().expect("a number");
    assert_eq!(
        value,
        0x0001_0000,
        "a Vulkan 1.0 device is reported as accepting SPIR-V {}.{}\n{stdout}",
        value >> 16,
        (value >> 8) & 0xff
    );
}
