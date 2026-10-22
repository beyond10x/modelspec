//! Adversary pass 2 for `story:m1-inspect-device`: `Vulkan::create_device` on a device below
//! Vulkan 1.2.
//!
//! DESIGN.md § 6 "Device": "The runtime requires Vulkan API version 1.2.0 and refuses a device
//! below it (E-TARGET-004)". `Device::api_version` documents "The API version passed to
//! `vkCreateInstance`, which the device was created under"; a logical device on a physical device
//! that reports 1.1 runs at 1.1 whatever the instance asked for (Vulkan specification, "Version
//! Numbers": the device version is the lesser of the two).
//!
//! The loader is the stand-in `adversary2_fake_loader.c`: device 0 reports Vulkan 1.2.0, device
//! 1 Vulkan 1.1.0.
#![cfg(feature = "vulkan")]

use std::path::PathBuf;
use std::process::Command;

use modelspec_runtime::vulkan::Vulkan;
use modelspec_runtime::vulkan::negotiate::negotiate;

fn fake_loader() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("adversary2_create_device_old");
    std::fs::create_dir_all(&dir).expect("the scratch directory is created");
    let source = dir.join("fake_loader.c");
    let library = dir.join("libfake_vulkan.so");
    std::fs::write(&source, include_str!("adversary2_fake_loader.c"))
        .expect("the loader source is written");
    let status = Command::new("cc")
        .args(["-shared", "-fPIC", "-o"])
        .arg(&library)
        .arg(&source)
        .status()
        .expect("cc runs");
    assert!(status.success(), "the fake loader compiles");
    library
}

#[test]
fn adversary2_create_device_refuses_or_reports_a_vulkan_1_1_device_truthfully() {
    let library = fake_loader();
    // SAFETY: this test binary has one test, so no other thread reads the environment.
    unsafe { std::env::set_var("MODELSPEC_VULKAN_LIBRARY", &library) };
    let vulkan = Vulkan::load().expect("the stand-in loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let old = candidates[1].profile.vulkan_api_version;
    assert_eq!(old, (1 << 22) | (1 << 12), "device 1 reports 1.1.0");
    assert!(negotiate(&candidates[1].profile).is_err());

    let negotiated = negotiate(&candidates[0].profile).expect("device 0 passes");
    match vulkan.create_device(1, &negotiated) {
        Err(refusal) => assert_eq!(refusal.code.as_str(), "E-TARGET-004", "{refusal}"),
        Ok(device) => panic!(
            "a logical device was created on the Vulkan 1.1 device 1 and reports api_version {:#x} \
             (1.{}), above the 1.1 the device runs at",
            device.api_version(),
            (device.api_version() >> 12) & 0x3ff
        ),
    }
}
