//! `modelspec inspect-device` without a GPU: usage errors exit 2, a missing Vulkan loader exits 1
//! with E-TARGET-002, and a build without the `vulkan` feature refuses the verb with E-TARGET-001.
//! No test here creates a Vulkan instance.

use std::process::{Command, Output};

/// A loader path that does not exist, so no Vulkan library is opened.
#[cfg(feature = "vulkan")]
const NO_LOADER: &str = "/nonexistent/modelspec-test/libvulkan-missing.so.1";

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .env(
            "MODELSPEC_VULKAN_LIBRARY",
            "/nonexistent/modelspec-test/libvulkan-missing.so.1",
        )
        .output()
        .expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn the_cpu_backend_is_a_usage_error() {
    let output = modelspec(&["inspect-device", "--backend", "cpu"]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output.stderr));
    assert!(text(&output.stderr).contains("invalid value 'cpu'"));
}

#[test]
fn the_yaml_format_is_a_usage_error() {
    let output = modelspec(&["inspect-device", "--backend", "vulkan", "--format", "yaml"]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output.stderr));
    assert!(text(&output.stderr).contains("invalid value 'yaml'"));
}

#[test]
fn a_backend_is_required() {
    let output = modelspec(&["inspect-device"]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output.stderr));
    assert!(text(&output.stderr).contains("--backend"));
}

#[cfg(feature = "vulkan")]
#[test]
fn a_missing_loader_exits_1_with_e_target_002() {
    for format in ["text", "json"] {
        let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
            .args(["inspect-device", "--backend", "vulkan", "--format", format])
            .env("MODELSPEC_VULKAN_LIBRARY", NO_LOADER)
            .output()
            .expect("modelspec runs");
        let stderr = text(&output.stderr);
        assert_eq!(output.status.code(), Some(1), "{stderr}");
        assert!(stderr.starts_with("E-TARGET-002: "), "{stderr}");
        assert!(stderr.contains(NO_LOADER), "{stderr}");
        assert!(output.stdout.is_empty(), "{}", text(&output.stdout));
    }
}

#[cfg(feature = "vulkan")]
#[test]
fn a_missing_loader_is_reported_before_the_device_index_is_checked() {
    let output = modelspec(&["inspect-device", "--backend", "vulkan", "--device", "99"]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.starts_with("E-TARGET-002: "), "{stderr}");
}

#[cfg(not(feature = "vulkan"))]
#[test]
fn without_the_vulkan_feature_the_verb_is_refused_with_e_target_001() {
    let output = modelspec(&["inspect-device", "--backend", "vulkan"]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.starts_with("E-TARGET-001: "), "{stderr}");
    assert!(output.stdout.is_empty(), "{}", text(&output.stdout));
}
