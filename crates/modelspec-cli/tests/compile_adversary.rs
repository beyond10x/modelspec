//! The artifact reader of `modelspec_core::artifact` against altered artifacts (DESIGN.md § 7.1,
//! § 9 gate 8): each is refused with the code the reader's order gives, and none panics.
//!
//! The order is the format version (E-ARTIFACT-001), every module file present and no longer
//! than its entry (E-ARTIFACT-004), each module hash and then `artifact_hash` (E-ARTIFACT-002),
//! then the invariants and the plan verifier (E-ARTIFACT-003). An alteration that is not
//! re-sealed stops at 002; one re-sealed with `artifact::seal` passes 002 and stops at the check
//! it breaks. No test here creates a Vulkan instance.

use std::path::{Path, PathBuf};
use std::process::Command;

use modelspec_core::artifact;
use serde_json::Value;

const NO_LOADER: &str = "/nonexistent/modelspec-test/libvulkan-missing.so.1";

/// A fresh Vulkan artifact of the MNIST example, compiled with the fixture profile into `name`.
fn compiled(name: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("compile_adversary")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("the old test dir is removed");
    }
    std::fs::create_dir_all(dir.parent().unwrap()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .arg("compile")
        .arg(root.join("examples/mnist_mlp.mspec"))
        .args(["--backend", "vulkan", "--output"])
        .arg(&dir)
        .arg("--device-profile")
        .arg(root.join("crates/modelspec-runtime/tests/fixtures/rtx3090.device.json"))
        .env("MODELSPEC_VULKAN_LIBRARY", NO_LOADER)
        .output()
        .expect("modelspec runs");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    artifact::read(&dir).unwrap_or_else(|d| panic!("the untouched artifact reads: {d}"));
    dir
}

fn manifest(dir: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("artifact.json")).unwrap()).unwrap()
}

fn write_manifest(dir: &Path, manifest: &Value) {
    std::fs::write(
        dir.join("artifact.json"),
        serde_json::to_string_pretty(manifest).unwrap(),
    )
    .unwrap();
}

/// Applies `alter` to the parsed `artifact.json` and writes it back, re-sealed or not.
fn alter_manifest(dir: &Path, reseal: bool, alter: impl FnOnce(&mut Value)) {
    let mut value = manifest(dir);
    alter(&mut value);
    write_manifest(dir, &value);
    if reseal {
        artifact::seal(dir).unwrap_or_else(|e| panic!("re-seal: {e}"));
    }
}

/// The first kernel module file of the artifact.
fn first_module(dir: &Path) -> PathBuf {
    let id = manifest(dir)["kernels"][0]["kernel_module_id"]
        .as_str()
        .unwrap()
        .to_owned();
    dir.join("kernels").join(format!("{id}.spv"))
}

fn refused_with(dir: &Path, code: &str) {
    match artifact::read(dir) {
        Ok(_) => panic!("the reader accepted an altered artifact; expected {code}"),
        Err(diagnostic) => {
            let rendered = diagnostic.to_string();
            assert_eq!(diagnostic.code.as_str(), code, "{rendered}");
            assert!(rendered.starts_with(&format!("{code}: ")), "{rendered}");
        }
    }
}

/// Makes the first step with a dependency depend on the step after it instead.
fn depend_on_a_later_step(value: &mut Value) {
    let plan = value["plan"].as_array_mut().unwrap();
    let count = plan.len();
    let (position, step) = plan
        .iter_mut()
        .enumerate()
        .find(|(i, s)| i + 1 < count && !s["depends_on"].as_array().unwrap().is_empty())
        .expect("a step with a dependency");
    step["depends_on"][0] = Value::from(position + 1);
}

#[test]
fn one_byte_changed_in_a_module_is_refused_with_002() {
    let dir = compiled("module-byte");
    let path = first_module(&dir);
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&path, bytes).unwrap();
    refused_with(&dir, "E-ARTIFACT-002");
}

#[test]
fn a_dependency_on_a_later_step_not_re_sealed_is_refused_with_002_before_the_plan_verifier() {
    let dir = compiled("later-dependency-unsealed");
    alter_manifest(&dir, false, depend_on_a_later_step);
    refused_with(&dir, "E-ARTIFACT-002");
}

#[test]
fn format_version_2_is_refused_with_001() {
    let dir = compiled("format-version-2");
    alter_manifest(&dir, true, |value| value["format_version"] = Value::from(2));
    refused_with(&dir, "E-ARTIFACT-001");
}

#[test]
fn a_dispatch_naming_a_module_not_in_the_artifact_is_refused_with_003() {
    let dir = compiled("foreign-module");
    alter_manifest(&dir, true, |value| {
        let step = value["plan"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|s| s["op"] == "Dispatch")
            .expect("a Dispatch step");
        step["kernel_module_id"] = Value::from("00000000-0000-8000-8000-000000000000");
    });
    refused_with(&dir, "E-ARTIFACT-003");
}

#[test]
fn a_dependency_on_a_later_step_re_sealed_is_refused_with_003() {
    let dir = compiled("later-dependency-sealed");
    alter_manifest(&dir, true, depend_on_a_later_step);
    refused_with(&dir, "E-ARTIFACT-003");
}

#[test]
fn a_view_past_its_buffer_is_refused_with_003() {
    let dir = compiled("view-past-buffer");
    alter_manifest(&dir, true, |value| {
        let buffer = value["loss_view"]["buffer"].as_u64().unwrap() as usize;
        let size = value["buffers"][buffer].as_u64().unwrap();
        value["loss_view"]["offset"] = Value::from(size);
    });
    refused_with(&dir, "E-ARTIFACT-003");
}

#[test]
fn an_element_count_whose_byte_size_passes_u64_max_is_refused_with_003() {
    let dir = compiled("byte-size-overflow");
    alter_manifest(&dir, true, |value| {
        value["parameters"][0]["views"][0]["element_count"] = Value::from(u64::MAX / 2);
    });
    refused_with(&dir, "E-ARTIFACT-003");
}

#[test]
fn a_deleted_module_file_is_refused_with_004() {
    let dir = compiled("module-deleted");
    std::fs::remove_file(first_module(&dir)).unwrap();
    refused_with(&dir, "E-ARTIFACT-004");
}

#[test]
fn one_byte_appended_to_a_module_is_refused_with_004_before_its_hash_is_read() {
    let dir = compiled("module-appended");
    let path = first_module(&dir);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.push(0);
    std::fs::write(&path, bytes).unwrap();
    refused_with(&dir, "E-ARTIFACT-004");
}
