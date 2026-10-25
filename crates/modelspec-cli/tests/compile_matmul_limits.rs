//! `story:matmul-block-kernel-throughput`, acceptance `profile-wired`: `modelspec compile
//! --backend vulkan --device-profile <json>` schedules the plan within the profile's workgroup
//! invocations and workgroup memory, so the `mlp_sweep` b8192 h4096 forward matmul takes the
//! block of 256 invocations, and records both limits in the manifest as `matmul_limits`; the
//! compile of a profile of 128 invocations records them too and takes no block. `ir --stage
//! kernel`, which takes no profile, prints no workgroup past 128 invocations.
//!
//! No test here creates a Vulkan instance: every run sets `MODELSPEC_VULKAN_LIBRARY` to a path
//! that does not exist, and every Vulkan compile names `--device-profile`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

/// A loader path that does not exist, so no Vulkan library is opened.
const NO_LOADER: &str = "/nonexistent/modelspec-test/libvulkan-missing.so.1";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh path under the test temp dir that does not exist yet.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("compile_matmul_limits")
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

/// The `mlp_sweep` b8192 h4096 source, written under the test temp dir `name`.
fn sweep_file(name: &str) -> PathBuf {
    let source = "model SweepMlp {
    param w1: Tensor<f32, [784, 4096]> = xavier_uniform();
    param w2: Tensor<f32, [4096, 10]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8192, 784]>)
        -> Tensor<f32, [8192, 10]> {
        return matmul(relu(matmul(x, w1)), w2);
    }
}

train Fit {
    model = SweepMlp;
    input x: Tensor<f32, [8192, 784]>;
    target y: Tensor<f32, [8192, 10]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.001);
    steps = 220;
    seed = 42;
    device = vulkan(0);
}
";
    let dir = scratch(name);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("mlp_sweep_b8192_h4096.mspec");
    std::fs::write(&path, source).unwrap();
    path
}

/// The RTX 3090 fixture with its invocation limit set to `invocations`, written under the test
/// temp dir.
fn profile_file(name: &str, invocations: u64) -> PathBuf {
    let fixture = root().join("crates/modelspec-runtime/tests/fixtures/rtx3090.device.json");
    let mut json: Value = serde_json::from_str(&std::fs::read_to_string(fixture).unwrap()).unwrap();
    json["max_workgroup_invocations"] = Value::from(invocations);
    let dir = scratch(name);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("profile.json");
    std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    path
}

/// The manifest of `compile --backend vulkan --device-profile <profile>` of the sweep source.
fn compiled_for(name: &str, profile: &Path) -> Value {
    let (source, out) = (sweep_file(&format!("{name}-source")), scratch(name));
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
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        text(&output.stderr)
    );
    let json = std::fs::read_to_string(out.join("artifact.json")).expect("artifact.json is read");
    serde_json::from_str(&json).expect("artifact.json is JSON")
}

fn workgroup_sizes(manifest: &Value) -> Vec<u64> {
    manifest["kernels"]
        .as_array()
        .expect("kernels")
        .iter()
        .map(|kernel| {
            kernel["workgroup_size"]
                .as_u64()
                .unwrap_or_else(|| panic!("a workgroup size in {kernel}"))
        })
        .collect()
}

#[test]
fn compile_with_a_profile_schedules_within_its_limits_and_records_them() {
    let rtx3090 = root().join("crates/modelspec-runtime/tests/fixtures/rtx3090.device.json");
    let manifest = compiled_for("rtx3090", &rtx3090);
    assert_eq!(
        manifest["matmul_limits"],
        serde_json::json!({"invocations": 1024, "shared_bytes": 49152})
    );
    let sizes = workgroup_sizes(&manifest);
    assert!(sizes.contains(&256), "the block: {sizes:?}");

    let minimum = compiled_for("minimum", &profile_file("minimum-profile", 128));
    assert_eq!(
        minimum["matmul_limits"],
        serde_json::json!({"invocations": 128, "shared_bytes": 49152})
    );
    let sizes = workgroup_sizes(&minimum);
    assert!(sizes.iter().all(|&size| size <= 128), "no block: {sizes:?}");
}

#[test]
fn ir_stage_kernel_takes_no_profile_and_prints_no_block() {
    let source = sweep_file("ir-source");
    let output = modelspec(&["ir", &source.display().to_string(), "--stage", "kernel"]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        text(&output.stderr)
    );
    let printed = text(&output.stdout);
    let kernels: Vec<&str> = printed
        .lines()
        .filter(|line| line.starts_with("kernel "))
        .collect();
    assert!(
        kernels
            .iter()
            .any(|line| line.contains(" matmul ") && line.contains("workgroup_size 128 ")),
        "{printed}"
    );
    assert!(
        !kernels
            .iter()
            .any(|line| line.contains("workgroup_size 256")),
        "no block without a profile: {printed}"
    );
}
