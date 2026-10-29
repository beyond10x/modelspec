//! Adversary cases for `story:m1-inspect-device`: a host whose loader lists one Vulkan 1.2 device
//! (index 0) and one Vulkan 1.1 device (index 1).
//!
//! DESIGN.md § 8 (revision 0.10) and the module documentation of `inspect_device.rs` say the verb
//! prints the profile of every physical device, or of the one `--device` names, with a
//! `negotiation: refused (<code>)` line for a device negotiation refuses, and that a `--device`
//! naming no device exits 1 with E-TARGET-003. The loader here is a stand-in written in C and
//! compiled at test time with `cc` into `CARGO_TARGET_TMPDIR`; it answers only the entry points
//! the probe calls and has no driver behind it.
#![cfg(feature = "vulkan")]

use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::OnceLock;

const FAKE_LOADER: &str = r#"
#include <stdint.h>
#include <string.h>
typedef int32_t VkResult;
typedef void (*PFN)(void);
static const uint32_t API[2] = { (1u << 22) | (2u << 12), (1u << 22) | (1u << 12) };
static VkResult create_instance(const void *info, const void *alloc, void **instance) {
    (void)info; (void)alloc; *instance = (void *)0x1000; return 0;
}
static void destroy_instance(void *instance, const void *alloc) { (void)instance; (void)alloc; }
static VkResult enumerate_physical_devices(void *instance, uint32_t *count, void **devices) {
    (void)instance;
    if (devices == 0) { *count = 2; return 0; }
    uint32_t n = *count < 2 ? *count : 2;
    for (uint32_t i = 0; i < n; i++) devices[i] = (void *)(uintptr_t)(0x2000 + i);
    *count = n;
    return n < 2 ? 5 : 0;
}
static int index_of(void *device) { return (int)((uintptr_t)device - 0x2000); }
static void fill(uint32_t *p, void *device) {
    int i = index_of(device);
    p[0] = API[i]; p[1] = 3; p[2] = 1; p[3] = 2 + (uint32_t)i;
}
static void properties(void *device, void *out) { memset(out, 0, 824); fill((uint32_t *)out, device); }
static void properties2(void *device, void *out) { fill((uint32_t *)((char *)out + 16), device); }
static void features2(void *device, void *out) { (void)device; (void)out; }
static void memory2(void *device, void *out) { (void)device; (void)out; }
static VkResult device_extensions(void *device, const char *layer, uint32_t *count, void *out) {
    (void)device; (void)layer; (void)out; *count = 0; return 0;
}
static void queue_families(void *device, uint32_t *count, uint32_t *out) {
    (void)device;
    if (out == 0) { *count = 1; return; }
    if (*count >= 1) {
        memset(out, 0, 24);
        out[0] = 0x2; out[1] = 1; out[2] = 64;
        *count = 1;
    }
}
PFN vkGetInstanceProcAddr(void *instance, const char *name) {
    (void)instance;
    if (strcmp(name, "vkCreateInstance") == 0) return (PFN)create_instance;
    if (strcmp(name, "vkDestroyInstance") == 0) return (PFN)destroy_instance;
    if (strcmp(name, "vkEnumeratePhysicalDevices") == 0) return (PFN)enumerate_physical_devices;
    if (strcmp(name, "vkGetPhysicalDeviceProperties") == 0) return (PFN)properties;
    if (strcmp(name, "vkGetPhysicalDeviceProperties2") == 0) return (PFN)properties2;
    if (strcmp(name, "vkGetPhysicalDeviceFeatures2") == 0) return (PFN)features2;
    if (strcmp(name, "vkGetPhysicalDeviceMemoryProperties2") == 0) return (PFN)memory2;
    if (strcmp(name, "vkEnumerateDeviceExtensionProperties") == 0) return (PFN)device_extensions;
    if (strcmp(name, "vkGetPhysicalDeviceQueueFamilyProperties") == 0) return (PFN)queue_families;
    return 0;
}
"#;

fn fake_loader() -> &'static PathBuf {
    static LOADER: OnceLock<PathBuf> = OnceLock::new();
    LOADER.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("adversary_inspect_old_device");
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

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .env("MODELSPEC_VULKAN_LIBRARY", fake_loader())
        .env_remove("MODELSPEC_VK_VALIDATION")
        .output()
        .expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Device 0 reports Vulkan 1.2.0 and has a compute queue family: `--device 0` names it alone,
/// and nothing in DESIGN.md § 6 or § 8 makes another device's version a reason to refuse it.
#[test]
fn adversary_device_0_is_inspected_although_device_1_is_below_1_2() {
    let output = modelspec(&[
        "inspect-device",
        "--backend",
        "vulkan",
        "--device",
        "0",
        "--format",
        "text",
    ]);
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(stdout.contains("vulkan_api_version: 4202496\n"), "{stdout}");
    assert!(stdout.ends_with("negotiation: ok\n"), "{stdout}");
}

/// `--device` naming no device exits 1 with E-TARGET-003 (DESIGN.md § 8.1, `inspect_device.rs`
/// module documentation, acceptance item 4).
#[test]
fn adversary_device_99_is_e_target_003_whatever_the_other_devices_report() {
    let output = modelspec(&["inspect-device", "--backend", "vulkan", "--device", "99"]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.starts_with("E-TARGET-003: "), "{stderr}");
}

/// Without `--device` every device is printed; a device negotiation refuses is still printed,
/// with `negotiation: refused (E-TARGET-004)`, and its diagnostic goes to stderr
/// (`inspect_device.rs` module documentation, DESIGN.md § 8).
#[test]
fn adversary_listing_prints_the_refused_device_with_its_verdict() {
    let output = modelspec(&["inspect-device", "--backend", "vulkan", "--format", "text"]);
    let stdout = text(&output.stdout);
    let stderr = text(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(stdout.starts_with("device: 0\n"), "{stdout}");
    assert!(stdout.contains("\ndevice: 1\n"), "{stdout}");
    assert!(
        stdout.ends_with("negotiation: refused (E-TARGET-004)\n"),
        "{stdout}"
    );
    assert!(stderr.contains("E-TARGET-004: "), "{stderr}");
}
