//! Device run on a real Vulkan device (DESIGN.md § 5.1, § 9 M1 "Runtime finds RTX 3090 through
//! Vulkan and prints verified device features"). Ignored by default: it creates a Vulkan
//! instance and a logical device, which `task ci` never does. Run with
//! `cargo test -p modelspec-runtime --features vulkan --test vulkan_device -- --ignored`.
#![cfg(feature = "vulkan")]

use modelspec_runtime::vulkan::Vulkan;
use modelspec_runtime::vulkan::negotiate::{REQUIRED_API_VERSION, negotiate};
use modelspec_runtime::vulkan::select::select_device;

/// Vulkan packs a version as major << 22 | minor << 12 | patch.
const fn version(major: u32, minor: u32, patch: u32) -> u32 {
    (major << 22) | (minor << 12) | patch
}

#[test]
#[ignore = "needs a Vulkan device"]
fn device_0_is_an_rtx_3090_and_its_logical_device_has_the_negotiated_set() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("physical devices are probed");
    let selected = select_device(&candidates, Some(0)).expect("device 0 is selectable");
    assert_eq!(selected, vec![0]);
    let candidate = &candidates[0];
    let profile = &candidate.profile;
    profile
        .validate()
        .expect("the probed profile keeps its invariants");

    assert_eq!(profile.vendor_id, 0x10de);
    assert_ne!(profile.device_id, 0);
    assert_ne!(profile.driver_version, 0);
    assert!(profile.vulkan_api_version >= version(1, 2, 0));
    assert_eq!(profile.subgroup_size, Some(32));
    assert!(profile.max_workgroup_invocations >= 1024);
    assert!(profile.max_shared_memory_bytes >= 49152);
    assert!(profile.memory_heaps.iter().any(|heap| heap.device_local));
    assert!(candidate.compute_queue_family.is_some());

    let negotiated = negotiate(profile).expect("the device passes negotiation");
    assert_eq!(negotiated.api_version, REQUIRED_API_VERSION);
    assert!(negotiated.features.is_empty());
    assert!(negotiated.extensions.is_empty());

    let device = vulkan
        .create_device(0, &negotiated)
        .expect("a compute-only logical device is created");
    assert_eq!(
        device.enabled_extensions(),
        negotiated.extensions.as_slice()
    );
    assert_eq!(device.enabled_features(), negotiated.features.as_slice());
    assert_eq!(device.api_version(), negotiated.api_version);
    assert_eq!(Some(device.queue_family()), candidate.compute_queue_family);
}
