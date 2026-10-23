//! Adversary cases for `modelspec_core::artifact::read` (story:m1-compile-artifact, DESIGN.md
//! § 7.1 "Untrusted serialized artifacts are validated before dispatch", "No ... user-injected
//! SPIR-V in M1").

use std::path::{Path, PathBuf};

use modelspec_core::artifact::{self, Backend, KernelCacheKey, KernelModuleFile};
use modelspec_core::ir::tensor::DType;

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary_artifact_reader")
        .join(name);
    if dir.is_dir() {
        std::fs::remove_dir_all(&dir).expect("the old test dir is removed");
    }
    dir
}

/// A Vulkan artifact whose module files are three bytes of text each, not SPIR-V (no magic
/// number, not a whole number of 32-bit words), written and sealed through the public API.
#[test]
fn a_module_that_is_not_spirv_is_refused() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join("examples/mnist_mlp.mspec");
    let source = std::fs::read_to_string(&path).unwrap();
    let compiled =
        modelspec_core::compile::compile_artifact("mnist_mlp.mspec", &source, Backend::Vulkan)
            .unwrap_or_else(|d| panic!("{d:?}"));
    let modules: Vec<KernelModuleFile> = compiled
        .execution
        .kernels
        .iter()
        .map(|kernel| KernelModuleFile {
            kernel_module_id: kernel.id,
            entry_point: "main".to_owned(),
            cache_key: KernelCacheKey {
                ir_hash: compiled.ir_hash.0.clone(),
                shapes: Vec::new(),
                dtype: DType::F32,
                fast_math: false,
                deterministic: true,
                vendor_id: 0,
                device_id: 0,
                driver_version: 0,
                capabilities: Vec::new(),
            },
            spirv: b"abc".to_vec(),
        })
        .collect();
    let dir = scratch("not-spirv");
    artifact::write(&dir, &compiled, &modules).expect("written");
    match artifact::read(&dir) {
        Ok(read) => panic!(
            "the reader accepted {} kernel modules of 3 bytes each that are not SPIR-V",
            read.modules.len()
        ),
        Err(d) => assert_eq!(d.code.as_str(), "E-ARTIFACT-003", "{d}"),
    }
}
