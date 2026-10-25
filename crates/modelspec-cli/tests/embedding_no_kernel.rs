//! An `embedding` program and the schedule (DESIGN.md § 5.2): since
//! story:bert-vulkan-kernels-rowwise `embedding` and `EmbeddingGrad` have Vulkan kernels, so
//! `ir --stage kernel` prints them, `compile --backend vulkan` writes an artifact, and
//! `train --backend vulkan` and `infer --backend vulkan` get past the schedule to the missing
//! loader and the missing checkpoint. A Cpu artifact has no kernels, so `compile --backend cpu`
//! writes one, and the artifact reader accepts its `i32` ids view.
//!
//! No test here creates a Vulkan instance: every run sets `MODELSPEC_VULKAN_LIBRARY` to a path
//! that does not exist.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use modelspec_core::artifact::Backend;
use modelspec_core::ir::tensor::DType;

const NO_LOADER: &str = "/nonexistent/modelspec-test/libvulkan-missing.so.1";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/embedding_ids.mspec")
        .display()
        .to_string()
}

fn profile() -> String {
    root()
        .join("crates/modelspec-runtime/tests/fixtures/rtx3090.device.json")
        .display()
        .to_string()
}

/// A fresh path under the test temp dir that does not exist yet.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("embedding_no_kernel")
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

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn every_command_that_needs_a_vulkan_kernel_schedules_embedding() {
    let fixture = fixture();
    let profile = profile();
    let out = scratch("vulkan").join("artifact");
    let missing = scratch("missing").join("no-such-checkpoint");
    for args in [
        vec!["ir", "--stage", "kernel", fixture.as_str()],
        vec![
            "compile",
            fixture.as_str(),
            "--backend",
            "vulkan",
            "--device-profile",
            profile.as_str(),
            "--output",
            out.to_str().unwrap(),
        ],
        vec!["train", fixture.as_str(), "--backend", "vulkan"],
        vec![
            "infer",
            fixture.as_str(),
            "--backend",
            "vulkan",
            "--checkpoint",
            missing.to_str().unwrap(),
        ],
    ] {
        let output = modelspec(&args);
        let text = stderr(&output);
        assert!(!text.contains("E-TARGET-011"), "{args:?}:\n{text}");
        assert!(!text.contains("E-INTERNAL"), "{args:?}:\n{text}");
        match args[0] {
            "ir" => {
                assert_eq!(output.status.code(), Some(0), "{args:?}:\n{text}");
                let printed = String::from_utf8_lossy(&output.stdout);
                for kernel in [" per_element Embedding ", " per_row EmbeddingGrad "] {
                    assert!(
                        printed
                            .lines()
                            .any(|l| l.starts_with("kernel ") && l.contains(kernel)),
                        "no `{kernel}` kernel line:\n{printed}"
                    );
                }
            }
            "compile" => assert_eq!(output.status.code(), Some(0), "{args:?}:\n{text}"),
            "train" => {
                assert_eq!(output.status.code(), Some(1), "{args:?}:\n{text}");
                assert!(text.starts_with("E-TARGET-002: "), "{args:?}:\n{text}");
            }
            _ => {
                assert_eq!(output.status.code(), Some(1), "{args:?}:\n{text}");
                assert!(text.starts_with("E-CKPT-001: "), "{args:?}:\n{text}");
            }
        }
    }
    assert!(out.join("artifact.json").is_file(), "no {}", out.display());
}

#[test]
fn compile_for_the_cpu_writes_an_artifact_the_reader_accepts() {
    let out = scratch("cpu").join("artifact");
    let output = modelspec(&[
        "compile",
        &fixture(),
        "--backend",
        "cpu",
        "--output",
        out.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let read = modelspec_core::artifact::read(&out).unwrap_or_else(|d| panic!("{d}"));
    let execution = &read.artifact.execution;
    assert_eq!(read.artifact.backend, Backend::Cpu);
    assert!(read.modules.is_empty());
    assert!(execution.kernels.is_empty());
    assert!(execution.step.is_empty() && execution.forward.is_empty());
    let ids = execution
        .data
        .iter()
        .find(|data| data.name == "ids")
        .expect("the ids input has a view");
    assert_eq!(ids.view.dtype, DType::I32);
    assert_eq!(ids.view.element_count, 4);
    assert_eq!(execution.prediction_view.element_count, 12);
}
