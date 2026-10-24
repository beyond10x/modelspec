//! `story:fast-math-mode` (DESIGN.md § 3.3, § 7.1, § 12 ADR-008): the `fast_math` of a
//! [`CompiledArtifact`] is what [`artifact::write`] records in the manifest, what
//! [`artifact::read`] gives back, and part of the artifact hash; the manifest's `fast_math` and
//! every kernel cache key's agree, so `write` refuses modules built under the other setting and
//! `read` refuses a resealed manifest whose kernel key disagrees with it (E-ARTIFACT-003).

use std::path::{Path, PathBuf};

use modelspec_core::artifact::{self, Backend, CompiledArtifact, KernelCacheKey, KernelModuleFile};
use modelspec_core::ir::tensor::DType;

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("artifact_fast_math")
        .join(name);
    if dir.is_dir() {
        std::fs::remove_dir_all(&dir).expect("the old test dir is removed");
    }
    dir
}

fn mnist(backend: Backend) -> CompiledArtifact {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source = std::fs::read_to_string(root.join("examples/mnist_mlp.mspec")).unwrap();
    modelspec_core::compile::compile_artifact("mnist_mlp.mspec", &source, backend)
        .unwrap_or_else(|d| panic!("{d:?}"))
}

/// One module per kernel of `compiled`: a SPIR-V header and nothing else, its key naming
/// `fast_math`.
fn modules(compiled: &CompiledArtifact, fast_math: bool) -> Vec<KernelModuleFile> {
    let header: Vec<u8> = [0x0723_0203u32, 0x0001_0500, 0, 1, 0]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    compiled
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
                fast_math,
                deterministic: true,
                vendor_id: 0,
                device_id: 0,
                driver_version: 0,
                capabilities: vec!["Shader".to_owned()],
            },
            spirv: header.clone(),
        })
        .collect()
}

#[test]
fn write_records_fast_math_and_read_gives_it_back_in_a_different_hash() {
    let deterministic = mnist(Backend::Vulkan);
    assert!(
        !deterministic.fast_math,
        "compile_artifact sets fast_math off"
    );
    let mut fast = deterministic.clone();
    fast.fast_math = true;

    let off = scratch("off");
    let off_hash =
        artifact::write(&off, &deterministic, &modules(&deterministic, false)).expect("written");
    let on = scratch("on");
    let on_hash = artifact::write(&on, &fast, &modules(&fast, true)).expect("written");
    assert_ne!(
        off_hash, on_hash,
        "fast_math is recorded in the artifact hash"
    );

    let read_off = artifact::read(&off).expect("the deterministic artifact reads");
    let read_on = artifact::read(&on).expect("the fast_math artifact reads");
    assert!(!read_off.fast_math);
    assert!(!read_off.artifact.fast_math);
    assert!(read_on.fast_math);
    assert!(read_on.artifact.fast_math);
    assert_eq!(
        read_on.artifact, fast,
        "the artifact round-trips with its fast_math"
    );
    assert_eq!(read_off.artifact, deterministic);
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(on.join("artifact.json")).unwrap()).unwrap();
    assert_eq!(manifest["fast_math"], true);
    assert_eq!(
        manifest["deterministic"], true,
        "fast_math leaves deterministic on"
    );
}

#[test]
fn write_refuses_modules_built_under_the_other_fast_math() {
    let deterministic = mnist(Backend::Vulkan);
    let mut fast = deterministic.clone();
    fast.fast_math = true;
    let dir = scratch("mismatch");
    match artifact::write(&dir, &fast, &modules(&fast, false)) {
        Err(artifact::WriteError::Invalid(why)) => assert!(why.contains("fast_math"), "{why}"),
        other => panic!("write accepted modules keyed fast_math false: {other:?}"),
    }
    match artifact::write(&dir, &deterministic, &modules(&deterministic, true)) {
        Err(artifact::WriteError::Invalid(why)) => assert!(why.contains("fast_math"), "{why}"),
        other => panic!("write accepted modules keyed fast_math true: {other:?}"),
    }
}

#[test]
fn read_refuses_a_resealed_manifest_whose_kernel_key_disagrees_on_fast_math() {
    let mut fast = mnist(Backend::Vulkan);
    fast.fast_math = true;
    let dir = scratch("resealed");
    artifact::write(&dir, &fast, &modules(&fast, true)).expect("written");
    let path = dir.join("artifact.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    manifest["kernels"][0]["cache_key"]["fast_math"] = serde_json::Value::Bool(false);
    std::fs::write(&path, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
    artifact::seal(&dir).expect("resealed");
    match artifact::read(&dir) {
        Ok(_) => {
            panic!("the reader accepted a kernel keyed fast_math false in a fast_math artifact")
        }
        Err(d) => {
            assert_eq!(d.code.as_str(), "E-ARTIFACT-003", "{d}");
            assert!(d.to_string().contains("fast_math"), "{d}");
        }
    }
}
