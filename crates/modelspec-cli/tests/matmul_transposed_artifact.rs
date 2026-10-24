//! `story:matmul-reads-transposed-operand`, acceptance `transposed-operand-flag`, on disk:
//! `modelspec compile` writes each matmul's transposed operands into `artifact.json`, and the
//! reader gives back the kernel record the compiler built, flags included, under the module id
//! its content gives. A kernel that reads no operand transposed writes no `transposed` field,
//! so its manifest entry is what it was before the change.
//!
//! No Vulkan instance is created: the loader path does not exist and `--device-profile` names
//! the fixture profile.

use std::path::{Path, PathBuf};
use std::process::Command;

use modelspec_core::artifact::{self, Backend};
use modelspec_core::compile::compile_artifact;
use modelspec_core::ir::kernel::KernelKind;
use serde_json::Value;

const NO_LOADER: &str = "/nonexistent/modelspec-test/libvulkan-missing.so.1";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("matmul_transposed_artifact")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("the old test dir is removed");
    }
    std::fs::create_dir_all(dir.parent().unwrap()).expect("the temp dir is writable");
    dir
}

#[test]
fn the_mnist_artifact_records_and_reads_back_each_transposed_operand() {
    let example = root().join("examples/mnist_mlp.mspec");
    let profile = root().join("crates/modelspec-runtime/tests/fixtures/rtx3090.device.json");
    let out = scratch("mnist");
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .arg("compile")
        .arg(&example)
        .args(["--backend", "vulkan", "--output"])
        .arg(&out)
        .arg("--device-profile")
        .arg(&profile)
        .env("MODELSPEC_VULKAN_LIBRARY", NO_LOADER)
        .output()
        .expect("modelspec runs");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let source = std::fs::read_to_string(&example).expect("the example reads");
    let compiled = compile_artifact(&example.display().to_string(), &source, Backend::Vulkan)
        .expect("the example compiles")
        .execution;
    let loaded = artifact::read(&out).unwrap_or_else(|d| panic!("the reader accepts it: {d}"));
    assert_eq!(loaded.artifact.execution.kernels, compiled.kernels);
    let flagged = compiled
        .kernels
        .iter()
        .filter(|k| k.transposed != [false, false])
        .count();
    assert_eq!(flagged, 3, "three MNIST matmuls read an operand transposed");

    let json = std::fs::read_to_string(out.join("artifact.json")).expect("artifact.json reads");
    let manifest: Value = serde_json::from_str(&json).expect("artifact.json is JSON");
    let entries = manifest["kernels"].as_array().expect("kernels is a list");
    assert_eq!(entries.len(), compiled.kernels.len());
    for (entry, kernel) in entries.iter().zip(&compiled.kernels) {
        assert_eq!(
            entry["kernel_module_id"].as_str(),
            Some(kernel.id.to_string().as_str())
        );
        match kernel.transposed {
            [false, false] => assert!(entry.get("transposed").is_none(), "{entry}"),
            [a, b] => {
                assert_eq!(kernel.kind, KernelKind::MatMul);
                assert_eq!(entry["transposed"], serde_json::json!([a, b]), "{entry}");
            }
        }
    }
}
