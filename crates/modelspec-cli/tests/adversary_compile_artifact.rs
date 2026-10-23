//! Adversary cases for `modelspec compile` and the artifact reader (story:m1-compile-artifact,
//! DESIGN.md § 7.1, § 8, § 9 gate 8). No test here creates a Vulkan instance: every Vulkan
//! compile names the fixture `--device-profile` and sets `MODELSPEC_VULKAN_LIBRARY` to a path
//! that does not exist.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::mpsc;
use std::time::Duration;

use modelspec_core::artifact;
use serde_json::Value;

const NO_LOADER: &str = "/nonexistent/modelspec-test/libvulkan-missing.so.1";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary_compile_artifact")
        .join(name);
    if dir.is_dir() {
        std::fs::remove_dir_all(&dir).expect("the old test dir is removed");
    }
    std::fs::create_dir_all(dir.parent().unwrap()).unwrap();
    dir
}

fn compile(example: &str, backend: &str, out: &Path) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_modelspec"));
    command
        .arg("compile")
        .arg(root().join("examples").join(example))
        .args(["--backend", backend, "--output"])
        .arg(out)
        .env("MODELSPEC_VULKAN_LIBRARY", NO_LOADER);
    if backend == "vulkan" {
        command
            .arg("--device-profile")
            .arg(root().join("crates/modelspec-runtime/tests/fixtures/rtx3090.device.json"));
    }
    let output = command.output().expect("modelspec runs");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn vulkan_artifact(name: &str) -> PathBuf {
    let dir = scratch(name);
    compile("mnist_mlp.mspec", "vulkan", &dir);
    artifact::read(&dir).unwrap_or_else(|d| panic!("the untouched artifact reads: {d}"));
    dir
}

fn manifest(dir: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("artifact.json")).unwrap()).unwrap()
}

fn first_module(dir: &Path) -> PathBuf {
    let id = manifest(dir)["kernels"][0]["kernel_module_id"]
        .as_str()
        .unwrap()
        .to_owned();
    dir.join("kernels").join(format!("{id}.spv"))
}

fn spv_files(dir: &Path) -> usize {
    match std::fs::read_dir(dir.join("kernels")) {
        Ok(entries) => entries.count(),
        Err(_) => 0,
    }
}

fn code_of(dir: &Path) -> String {
    match artifact::read(dir) {
        Ok(_) => "accepted".to_owned(),
        Err(d) => format!("{}: {d}", d.code.as_str()),
    }
}

/// Acceptance 3 and the `--output` help ("files of an earlier artifact in it are overwritten"):
/// a Cpu compile into a directory that held a Vulkan artifact leaves a Cpu artifact, which has
/// no `kernels/`.
#[test]
fn a_cpu_compile_into_a_reused_vulkan_output_leaves_no_kernels_dir() {
    let dir = vulkan_artifact("reuse-cpu");
    compile("mnist_mlp.mspec", "cpu", &dir);
    assert!(
        !dir.join("kernels").exists(),
        "the Cpu artifact in {} still has kernels/ with {} stale .spv files",
        dir.display(),
        spv_files(&dir)
    );
}

/// Acceptance 1: `artifact.json` has one kernel entry per file in `kernels/`, also when
/// `--output` names the directory of an earlier compile of another file.
#[test]
fn a_vulkan_compile_into_a_reused_output_has_one_entry_per_kernel_file() {
    let dir = vulkan_artifact("reuse-vulkan");
    compile("linear_regression.mspec", "vulkan", &dir);
    let entries = manifest(&dir)["kernels"].as_array().unwrap().len();
    assert_eq!(
        spv_files(&dir),
        entries,
        "kernels/ holds stale modules of the earlier artifact"
    );
}

/// DESIGN.md § 7.1: "every module file present and no longer than its entry, before any module
/// is hashed (E-ARTIFACT-004)". A module that is a symlink to `/dev/zero` reports length 0 to
/// `metadata` and reads without end; it is longer than its entry and must stop at 004, not be
/// read and hashed.
#[test]
fn a_module_symlinked_to_dev_zero_is_refused_with_004() {
    let dir = vulkan_artifact("dev-zero");
    let path = first_module(&dir);
    std::fs::remove_file(&path).unwrap();
    std::os::unix::fs::symlink("/dev/zero", &path).unwrap();
    let found = code_of(&dir);
    assert!(found.starts_with("E-ARTIFACT-004: "), "{found}");
}

/// § 9 gate 8: the reader of an untrusted directory returns. A module that is a FIFO with no
/// writer must be refused (004: not a readable module file), not block `read` forever.
#[test]
fn a_module_that_is_a_fifo_is_refused_without_blocking() {
    let dir = vulkan_artifact("fifo");
    let path = first_module(&dir);
    std::fs::remove_file(&path).unwrap();
    let status = Command::new("mkfifo")
        .arg(&path)
        .status()
        .expect("mkfifo runs");
    assert!(status.success());
    let (send, receive) = mpsc::channel();
    let reader_dir = dir.clone();
    let reader = std::thread::spawn(move || {
        let _ = send.send(code_of(&reader_dir));
    });
    let outcome = receive.recv_timeout(Duration::from_secs(10));
    if outcome.is_err() {
        // Unblock the reader so the thread ends: open the FIFO for writing and close it.
        drop(std::fs::OpenOptions::new().write(true).open(&path));
        let _ = reader.join();
        panic!("artifact::read blocked for 10 s on a FIFO module file");
    }
    let _ = reader.join();
    let found = outcome.unwrap();
    assert!(found.starts_with("E-ARTIFACT-004: "), "{found}");
}

/// `spec/domains/artifact.yaml`, `CompiledArtifact.buffers`: "No buffer exceeds 4294967295
/// bytes, the u32 range of a storage buffer binding". Re-sealed, a buffer of 2^33 bytes must
/// stop at the invariant check (003).
#[test]
fn a_buffer_past_u32_max_re_sealed_is_refused_with_003() {
    let dir = vulkan_artifact("buffer-past-u32");
    let mut value = manifest(&dir);
    let last = value["buffers"].as_array().unwrap().len() - 1;
    value["buffers"][last] = Value::from(1u64 << 33);
    std::fs::write(
        dir.join("artifact.json"),
        serde_json::to_string_pretty(&value).unwrap(),
    )
    .unwrap();
    artifact::seal(&dir).unwrap();
    let found = code_of(&dir);
    assert!(found.starts_with("E-ARTIFACT-003: "), "{found}");
}
