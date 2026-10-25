//! Adversary cases for story:matmul-block-kernel-throughput, `compile --device-profile` against
//! profiles that are valid `DeviceProfile` records (`spec/domains/device.yaml` asks only
//! `max_workgroup_invocations > 0` and `max_shared_memory_bytes > 0`) but below the Vulkan
//! minimum of 128 invocations and 16,384 bytes that `MatMulLimits` states as invariants.
//!
//! 1. A profile of 64 invocations: the base compiled the plan for every device; the change
//!    schedules for the profile and then reports its own plan as `E-INTERNAL-002` "compiler bug"
//!    ("a fault in modelspec, not in the specification"), a fault the input caused.
//! 2. A profile of 8,192 bytes of workgroup memory: whatever the compile does, it does not write
//!    an artifact whose `matmul_limits` break `shared_bytes >= 16384`, the invariant of
//!    `spec/domains/artifact.yaml` and the `minimum` of the generated JSON Schema.
//!
//! No test here creates a Vulkan instance: every run sets `MODELSPEC_VULKAN_LIBRARY` to a path
//! that does not exist, and every Vulkan compile names `--device-profile`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

const NO_LOADER: &str = "/nonexistent/modelspec-test/libvulkan-missing.so.1";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary_blockperf_compile")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("the old test dir is removed");
    }
    std::fs::create_dir_all(dir.parent().unwrap()).expect("the temp dir is writable");
    dir
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .env("MODELSPEC_VULKAN_LIBRARY", NO_LOADER)
        .output()
        .expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The RTX 3090 fixture with `field` set to `value`, written under the test temp dir.
fn profile_file(name: &str, field: &str, value: u64) -> PathBuf {
    let fixture = root().join("crates/modelspec-runtime/tests/fixtures/rtx3090.device.json");
    let mut json: Value = serde_json::from_str(&std::fs::read_to_string(fixture).unwrap()).unwrap();
    json[field] = Value::from(value);
    let dir = scratch(name);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("profile.json");
    std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    path
}

/// `compile --backend vulkan --device-profile <profile>` of `examples/<example>.mspec` into a
/// fresh directory `name`: the output and the directory.
fn compile(name: &str, example: &str, profile: &Path) -> (Output, PathBuf) {
    let out = scratch(name);
    let source = root().join("examples").join(format!("{example}.mspec"));
    let output = modelspec(&[
        "compile",
        &source.display().to_string(),
        "--backend",
        "vulkan",
        "--output",
        &out.display().to_string(),
        "--device-profile",
        &profile.display().to_string(),
    ]);
    (output, out)
}

/// ADV-BLOCKPERF-2: a profile below 128 invocations is not reported as E-INTERNAL-002 compiler
/// bug.
#[test]
fn a_profile_below_128_invocations_is_not_reported_as_a_compiler_bug() {
    let profile = profile_file("inv64-profile", "max_workgroup_invocations", 64);
    let mut wrong = Vec::new();
    for example in ["linear_regression", "mnist_mlp"] {
        let (output, _) = compile(&format!("inv64-{example}"), example, &profile);
        let stderr = text(&output.stderr);
        if stderr.contains("E-INTERNAL-002") || stderr.contains("compiler bug") {
            wrong.push(format!(
                "{example}: exit {:?}, {}",
                output.status.code(),
                stderr.trim()
            ));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// ADV-BLOCKPERF-3: compile never writes matmul_limits below the invariants of
/// spec/domains/artifact.yaml.
#[test]
fn no_artifact_records_matmul_limits_below_the_vulkan_minimum() {
    let profile = profile_file("sh8k-profile", "max_shared_memory_bytes", 8192);
    let (output, out) = compile("sh8k-linear", "linear_regression", &profile);
    if output.status.code() != Some(0) {
        return;
    }
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("artifact.json")).unwrap()).unwrap();
    let limits = &manifest["matmul_limits"];
    if limits.is_null() {
        return;
    }
    let shared = limits["shared_bytes"].as_u64().expect("shared_bytes");
    let invocations = limits["invocations"].as_u64().expect("invocations");
    assert!(
        shared >= 16_384 && invocations >= 128,
        "exit 0 and the manifest records {limits}: spec/domains/artifact.yaml MatMulLimits \
         invariants invocations >= 128, shared_bytes >= 16384 (stdout {})",
        text(&output.stdout).trim()
    );
}
