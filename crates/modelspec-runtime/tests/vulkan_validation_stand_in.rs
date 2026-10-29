//! `Vulkan::load` under `MODELSPEC_VK_VALIDATION=1` on a loader without `VK_EXT_debug_utils`
//! (story:vulkan-validation-without-debug-utils).
//!
//! The loader is the stand-in `adversary2_fake_loader.c`, compiled at test time with `cc` into
//! `CARGO_TARGET_TMPDIR`: it creates an instance whatever layers and extensions are asked for and
//! has no `vkCreateDebugUtilsMessengerEXT`. The load is refused with E-TARGET-003 naming what is
//! missing, where calling the absent entry point aborted the process.
#![cfg(feature = "vulkan")]

use std::path::PathBuf;
use std::process::Command;

use modelspec_runtime::vulkan::{LIBRARY_ENV, VALIDATION_ENV, Vulkan};

fn stand_in_loader() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("vulkan_validation_stand_in");
    std::fs::create_dir_all(&dir).expect("the scratch directory is created");
    let source = dir.join("stand_in_loader.c");
    let library = dir.join("libstand_in_vulkan.so");
    std::fs::write(&source, include_str!("adversary2_fake_loader.c"))
        .expect("the loader source is written");
    let status = Command::new("cc")
        .args(["-shared", "-fPIC", "-o"])
        .arg(&library)
        .arg(&source)
        .status()
        .expect("cc runs");
    assert!(status.success(), "the stand-in loader compiles");
    library
}

#[test]
fn validation_on_a_loader_without_debug_utils_is_refused_with_e_target_003() {
    let library = stand_in_loader();
    // SAFETY: this test binary has one test, so no other thread reads the environment.
    unsafe {
        std::env::set_var(LIBRARY_ENV, &library);
        std::env::set_var(VALIDATION_ENV, "1");
    }
    let refusal = match Vulkan::load() {
        Err(refusal) => refusal,
        Ok(vulkan) => panic!(
            "the stand-in loader was loaded with validation_enabled() = {}",
            vulkan.validation_enabled()
        ),
    };
    let shown = refusal.to_string();
    assert_eq!(refusal.code.as_str(), "E-TARGET-003", "{shown}");
    assert!(shown.contains("VK_EXT_debug_utils"), "{shown}");
    assert!(shown.contains("VK_LAYER_KHRONOS_validation"), "{shown}");
}
