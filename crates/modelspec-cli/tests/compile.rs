//! `modelspec compile <file> --backend vulkan|cpu --output <dir> [--device-profile <json>]`:
//! the artifact directory on disk (DESIGN.md § 7.1, § 8). Its SPIR-V passes `spirv-val`, two
//! compiles are byte-equal, the Cpu and Vulkan artifacts share their manifest, and the reader of
//! `modelspec_core::artifact` accepts what `compile` writes.
//!
//! No test here creates a Vulkan instance: every run sets `MODELSPEC_VULKAN_LIBRARY` to a path
//! that does not exist, and every Vulkan compile that is meant to succeed names
//! `--device-profile`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

/// A loader path that does not exist, so no Vulkan library is opened.
const NO_LOADER: &str = "/nonexistent/modelspec-test/libvulkan-missing.so.1";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn example() -> String {
    root()
        .join("examples/mnist_mlp.mspec")
        .display()
        .to_string()
}

fn fixture() -> PathBuf {
    root().join("crates/modelspec-runtime/tests/fixtures/rtx3090.device.json")
}

/// A fresh path under the test temp dir that does not exist yet.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("compile")
        .join(name);
    if dir.is_dir() {
        std::fs::remove_dir_all(&dir).expect("the old test dir is removed");
    } else if dir.exists() {
        std::fs::remove_file(&dir).expect("the old test file is removed");
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

/// `modelspec compile` of the MNIST example into `out`; Vulkan with the fixture profile.
fn compile(backend: &str, out: &Path) -> Output {
    let (example, out, profile) = (
        example(),
        out.display().to_string(),
        fixture().display().to_string(),
    );
    let mut args = vec!["compile", &example, "--backend", backend, "--output", &out];
    if backend == "vulkan" {
        args.extend(["--device-profile", profile.as_str()]);
    }
    modelspec(&args)
}

fn compiled(backend: &str, name: &str) -> PathBuf {
    let out = scratch(name);
    let output = compile(backend, &out);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        text(&output.stderr)
    );
    out
}

fn manifest(dir: &Path) -> Value {
    let json = std::fs::read_to_string(dir.join("artifact.json")).expect("artifact.json is read");
    serde_json::from_str(&json).expect("artifact.json is JSON")
}

/// Every file under `dir`, by relative path, with its bytes.
fn tree(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    fn walk(base: &Path, dir: &Path, out: &mut Vec<(PathBuf, Vec<u8>)>) {
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                walk(base, &path, out);
            } else {
                let bytes = std::fs::read(&path).unwrap();
                out.push((path.strip_prefix(base).unwrap().to_path_buf(), bytes));
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out
}

#[test]
fn a_vulkan_compile_writes_one_valid_module_per_kernel_entry_and_the_reader_accepts_it() {
    let dir = compiled("vulkan", "vulkan");
    let manifest = manifest(&dir);
    assert_eq!(manifest["format_version"], 1);
    assert_eq!(manifest["backend"], "Vulkan");
    assert_eq!(manifest["deterministic"], true);
    assert_eq!(manifest["fast_math"], false);
    let kernels = manifest["kernels"].as_array().expect("kernels is a list");
    let entries: BTreeSet<String> = kernels
        .iter()
        .map(|k| format!("{}.spv", k["kernel_module_id"].as_str().unwrap()))
        .collect();
    let files: BTreeSet<String> = std::fs::read_dir(dir.join("kernels"))
        .expect("kernels/ exists")
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(entries.len(), kernels.len(), "kernel ids are distinct");
    assert_eq!(entries, files, "one kernels/<id>.spv per kernel entry");
    let kinds: BTreeSet<&str> = kernels
        .iter()
        .map(|k| k["kind"].as_str().unwrap())
        .collect();
    for kind in [
        "elementwise",
        "loss_sum",
        "transpose",
        "matmul",
        "sgd_update",
    ] {
        assert!(kinds.contains(kind), "no {kind} kernel in {kinds:?}");
    }
    for file in &files {
        let path = dir.join("kernels").join(file);
        let output = Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2"])
            .arg(&path)
            .output()
            .expect("spirv-val runs; install SPIRV-Tools");
        assert!(
            output.status.success(),
            "spirv-val refused {}: {}{}",
            path.display(),
            text(&output.stderr),
            text(&output.stdout)
        );
    }
    let read = modelspec_core::artifact::read(&dir).unwrap_or_else(|d| panic!("{d}"));
    assert_eq!(read.modules.len(), kernels.len());
    assert_eq!(
        read.artifact_hash.0,
        manifest["artifact_hash"].as_str().unwrap()
    );
}

#[test]
fn two_compiles_of_one_file_give_byte_equal_trees() {
    for backend in ["vulkan", "cpu"] {
        let a = compiled(backend, &format!("twice-{backend}-a"));
        let b = compiled(backend, &format!("twice-{backend}-b"));
        let (a, b) = (tree(&a), tree(&b));
        assert!(!a.is_empty());
        let names = |t: &[(PathBuf, Vec<u8>)]| t.iter().map(|(p, _)| p.clone()).collect::<Vec<_>>();
        assert_eq!(names(&a), names(&b), "{backend}");
        assert!(a == b, "{backend}: the trees differ in content");
    }
}

#[test]
fn cpu_and_vulkan_artifacts_share_schema_hash_and_parameters() {
    let vulkan = manifest(&compiled("vulkan", "same-vulkan"));
    let cpu_dir = compiled("cpu", "same-cpu");
    let cpu = manifest(&cpu_dir);
    assert_eq!(cpu["backend"], "Cpu");
    assert_eq!(cpu["schema_hash"], vulkan["schema_hash"]);
    assert_eq!(cpu["parameters"], vulkan["parameters"]);
    assert_eq!(cpu["kernels"], Value::Array(Vec::new()));
    assert!(
        !cpu_dir.join("kernels").exists(),
        "a Cpu artifact has no kernels/"
    );
    for plan in ["plan", "forward_plan"] {
        let steps = cpu[plan].as_array().unwrap();
        assert!(steps.iter().all(|s| s["op"] != "Dispatch"), "{plan}");
    }
    assert!(
        vulkan["plan"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s["op"] == "Dispatch")
    );
    modelspec_core::artifact::read(&cpu_dir).unwrap_or_else(|d| panic!("{d}"));
}

#[test]
fn a_cpu_compile_needs_no_vulkan_loader() {
    let out = scratch("no-loader-cpu");
    let output = compile("cpu", &out);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    assert!(out.join("artifact.json").is_file());
}

#[test]
fn a_vulkan_compile_without_a_profile_and_without_a_loader_exits_1_with_e_target_002() {
    let out = scratch("no-loader-vulkan");
    let example = example();
    let out_arg = out.display().to_string();
    let output = modelspec(&[
        "compile",
        &example,
        "--backend",
        "vulkan",
        "--output",
        &out_arg,
    ]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.starts_with("E-TARGET-002: "), "{stderr}");
    assert!(!out.exists(), "nothing is written");
}

#[test]
fn a_vulkan_compile_with_the_fixture_profile_needs_no_loader() {
    let out = scratch("no-loader-profile");
    let output = compile("vulkan", &out);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
}

/// A copy of the fixture with `field` set to `value`, written under `name`.
fn altered_profile(name: &str, field: &str, value: Value) -> PathBuf {
    let json = std::fs::read_to_string(fixture()).unwrap();
    let mut profile: Value = serde_json::from_str(&json).unwrap();
    profile[field] = value;
    let path = scratch(name);
    std::fs::write(&path, serde_json::to_string_pretty(&profile).unwrap()).unwrap();
    path
}

#[test]
fn a_profile_that_breaks_an_invariant_is_refused_naming_the_file_and_the_invariant() {
    let profile = altered_profile("subgroup-0.device.json", "subgroup_size", Value::from(0));
    let out = scratch("subgroup-0");
    let (example, out_arg, profile_arg) = (
        example(),
        out.display().to_string(),
        profile.display().to_string(),
    );
    let output = modelspec(&[
        "compile",
        &example,
        "--backend",
        "vulkan",
        "--output",
        &out_arg,
        "--device-profile",
        &profile_arg,
    ]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains(&profile_arg), "{stderr}");
    assert!(stderr.contains("subgroup_size > 0"), "{stderr}");
    assert!(!out.exists(), "nothing is written");
}

#[test]
fn a_profile_below_vulkan_1_2_is_refused_with_e_target_004_before_anything_is_written() {
    let profile = altered_profile(
        "vulkan-1-1.device.json",
        "vulkan_api_version",
        Value::from((1u32 << 22) | (1 << 12)),
    );
    let out = scratch("vulkan-1-1");
    let (example, out_arg, profile_arg) = (
        example(),
        out.display().to_string(),
        profile.display().to_string(),
    );
    let output = modelspec(&[
        "compile",
        &example,
        "--backend",
        "vulkan",
        "--output",
        &out_arg,
        "--device-profile",
        &profile_arg,
    ]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.starts_with("E-TARGET-004: "), "{stderr}");
    assert!(!out.exists(), "the output directory is not created");
}

#[test]
fn a_device_profile_with_the_cpu_backend_is_a_usage_error() {
    let out = scratch("cpu-profile");
    let (example, out_arg, profile_arg) = (
        example(),
        out.display().to_string(),
        fixture().display().to_string(),
    );
    let output = modelspec(&[
        "compile",
        &example,
        "--backend",
        "cpu",
        "--output",
        &out_arg,
        "--device-profile",
        &profile_arg,
    ]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output.stderr));
    assert!(!out.exists());
}
