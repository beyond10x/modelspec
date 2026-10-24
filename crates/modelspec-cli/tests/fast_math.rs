//! `story:fast-math-mode` (DESIGN.md § 8, § 12 ADR-008), acceptances `flag-sets-field` and
//! `cli-reference-regenerated`: `compile --fast-math` sets `fast_math` in the manifest and in every
//! kernel cache key, read back through the artifact reader, and the artifact hash differs from the
//! one without the flag; `--fast-math` with `--backend cpu` is a usage error on `compile` and on
//! `train`; the generated CLI reference lists the flag on both commands.
//!
//! No test here creates a Vulkan instance: every run sets `MODELSPEC_VULKAN_LIBRARY` to a path
//! that does not exist, and every Vulkan compile names `--device-profile`.
#![cfg(feature = "vulkan")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use modelspec_core::artifact;

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

fn fixture() -> String {
    root()
        .join("crates/modelspec-runtime/tests/fixtures/rtx3090.device.json")
        .display()
        .to_string()
}

/// A fresh path under the test temp dir that does not exist yet.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("fast_math")
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

/// `compile` of the MNIST example for Vulkan with the fixture profile into a fresh `name`, with
/// `extra` arguments; the directory and the printed artifact hash.
fn compile(name: &str, extra: &[&str]) -> (PathBuf, String) {
    let out = scratch(name);
    let (example, shown, profile) = (example(), out.display().to_string(), fixture());
    let mut args = vec![
        "compile",
        &example,
        "--backend",
        "vulkan",
        "--output",
        &shown,
        "--device-profile",
        &profile,
    ];
    args.extend_from_slice(extra);
    let output = modelspec(&args);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(0), "{args:?}: {stderr}");
    let stdout = text(&output.stdout);
    let hash = stdout
        .split_whitespace()
        .nth(2)
        .unwrap_or_else(|| panic!("an `artifact <id> <hash>` line: {stdout}"))
        .to_owned();
    (out, hash)
}

#[test]
fn compile_fast_math_sets_fast_math_in_the_artifact_and_changes_its_hash() {
    let (off_dir, off_hash) = compile("off", &[]);
    let (on_dir, on_hash) = compile("on", &["--fast-math"]);
    let off = artifact::read(&off_dir).expect("the reader accepts the deterministic artifact");
    let on = artifact::read(&on_dir).expect("the reader accepts the fast_math artifact");
    assert!(
        !off.fast_math,
        "without --fast-math the manifest says false"
    );
    assert!(on.fast_math, "with --fast-math the manifest says true");
    assert!(!off.artifact.fast_math);
    assert!(on.artifact.fast_math);
    assert!(!on.modules.is_empty());
    for module in &off.modules {
        assert!(!module.cache_key.fast_math, "{}", module.kernel_module_id);
    }
    for module in &on.modules {
        assert!(module.cache_key.fast_math, "{}", module.kernel_module_id);
    }
    assert_eq!(
        off.artifact_hash.0, off_hash,
        "the printed hash is the read one"
    );
    assert_eq!(
        on.artifact_hash.0, on_hash,
        "the printed hash is the read one"
    );
    assert_ne!(
        off_hash, on_hash,
        "fast_math is recorded in the artifact hash"
    );
    assert_eq!(
        off.artifact.execution.kernels, on.artifact.execution.kernels,
        "the same kernels and plan"
    );
}

#[test]
fn compile_fast_math_with_the_cpu_backend_is_a_usage_error() {
    let out = scratch("cpu");
    let (example, shown) = (example(), out.display().to_string());
    let output = modelspec(&[
        "compile",
        &example,
        "--backend",
        "cpu",
        "--output",
        &shown,
        "--fast-math",
    ]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("'--fast-math'"), "{stderr}");
    assert!(stderr.contains("'--backend cpu'"), "{stderr}");
    assert!(!out.exists(), "nothing is written");
}

#[test]
fn train_fast_math_with_the_cpu_backend_is_a_usage_error() {
    let example = example();
    let output = modelspec(&["train", &example, "--backend", "cpu", "--fast-math"]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("'--fast-math'"), "{stderr}");
    assert!(stderr.contains("'--backend cpu'"), "{stderr}");
    assert!(output.stdout.is_empty(), "no step runs");
}

#[test]
fn train_fast_math_on_vulkan_reaches_the_loader() {
    let example = example();
    let output = modelspec(&["train", &example, "--backend", "vulkan", "--fast-math"]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("E-TARGET-002"), "{stderr}");
}

/// The section of the generated CLI reference that documents `modelspec <command>`.
fn reference_section(command: &str) -> String {
    let page = std::fs::read_to_string(root().join("website/docs/reference/cli.md"))
        .expect("the CLI reference is read");
    let heading = format!("## `modelspec {command}`");
    let start = page
        .find(&heading)
        .unwrap_or_else(|| panic!("the reference has {heading}"));
    let rest = &page[start + heading.len()..];
    let end = rest.find("\n## ").unwrap_or(rest.len());
    rest[..end].to_owned()
}

#[test]
fn the_cli_reference_lists_fast_math_on_compile_and_train() {
    for command in ["compile", "train"] {
        let section = reference_section(command);
        assert!(
            section.contains("| `--fast-math` |"),
            "the reference of `modelspec {command}` lists --fast-math:\n{section}"
        );
    }
    for command in ["check", "infer", "inspect-device", "ir", "observe"] {
        assert!(
            !reference_section(command).contains("--fast-math"),
            "`modelspec {command}` takes no --fast-math"
        );
    }
}
