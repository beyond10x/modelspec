//! Adversary cases for `modelspec` on a file trained with `adamw(...)` (story:bert-adamw-warmup,
//! DESIGN.md § 2.1, § 8, coordinator decision D50).
//!
//! The commands besides `train --backend cpu` that take such a file: `compile`, `ir --stage
//! kernel`, `train --backend vulkan`, the checkpoint flags, and `--steps` at the warm-up's edge.
//! A case named `adv_adamw_f<N>_...` is a finding of the adversary's report and is ignored until
//! its fix lands.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const FIXTURE: &str = "crates/modelspec-cli/tests/fixtures/linear_regression_adamw.mspec";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("UTF-8")
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("adversary_adamw_cli");
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    let path = dir.join(name);
    let _ = std::fs::remove_dir_all(&path);
    let _ = std::fs::remove_file(&path);
    path
}

/// Finding F1. `compile --backend cpu` on an AdamW file reaches the user as `E-INTERNAL-002`, a
/// compiler bug ("No source file is meant to reach it", `compile::codes::COMPILER_BUG`): the
/// scheduler takes the step's host inputs `optimizer.lr_t`, `c1`, `c2` and the moments for train
/// block data it does not declare. A CPU artifact keeps the views and needs no kernel
/// (`compile_artifact_for`), so the file should compile.
#[test]
#[ignore = "finding F1: compile --backend cpu on an adamw file is E-INTERNAL-002"]
fn adv_adamw_f1_compile_backend_cpu_writes_an_artifact() {
    let out = scratch("artifact");
    let output = modelspec(&[
        "compile",
        FIXTURE,
        "--backend",
        "cpu",
        "--output",
        out.to_str().unwrap(),
    ]);
    let stderr = text(&output.stderr);
    assert!(!stderr.contains("E-INTERNAL-002"), "{stderr}");
    assert_eq!(output.status.code(), Some(0), "{stderr}");
    assert!(out.join("artifact.json").exists());
}

/// `ir --stage kernel` and `train --backend vulkan` are E-TARGET-011 at the `train` block, exit
/// 1 as for `embedding`, with nothing on stdout; `ir --stage training` prints the moments.
#[test]
fn adamw_vulkan_paths_are_e_target_011_before_anything_runs() {
    for args in [
        vec!["ir", FIXTURE, "--stage", "kernel"],
        vec!["train", FIXTURE, "--backend", "vulkan"],
    ] {
        let output = modelspec(&args);
        let stderr = text(&output.stderr);
        assert_eq!(output.status.code(), Some(1), "{args:?}: {stderr}");
        assert!(
            output.stdout.is_empty(),
            "{args:?}: {}",
            text(&output.stdout)
        );
        assert_eq!(
            stderr.lines().take(2).collect::<Vec<_>>(),
            [
                "E-TARGET-011: optimizer `adamw` has no Vulkan kernel yet",
                &format!("  {FIXTURE}:10:1"),
            ],
            "{args:?}"
        );
    }
    let output = modelspec(&["ir", FIXTURE, "--stage", "training"]);
    assert_eq!(output.status.code(), Some(0));
    let stdout = text(&output.stdout);
    for line in [
        "output %25 moment m @LinearReg.w",
        "output %29 moment v @LinearReg.w",
    ] {
        assert!(stdout.contains(line), "{stdout}");
    }
}

/// `--checkpoint` with `--checkpoint-every`, and with `--backend vulkan`, are the same usage
/// error, exit 2, one line, nothing written.
#[test]
fn adamw_checkpoint_flags_are_refused_on_every_backend() {
    let path = scratch("every.msckpt");
    for backend in ["cpu", "vulkan"] {
        let output = modelspec(&[
            "train",
            FIXTURE,
            "--backend",
            backend,
            "--checkpoint",
            path.to_str().unwrap(),
            "--checkpoint-every",
            "5",
        ]);
        let stderr = text(&output.stderr);
        assert_eq!(output.status.code(), Some(2), "{backend}: {stderr}");
        assert_eq!(stderr.lines().count(), 1, "{backend}: {stderr}");
        assert!(output.stdout.is_empty());
        assert!(!path.exists());
    }
}

/// `--steps` equal to the warm-up stops the run before its first step, exit 1, one line naming
/// both numbers and no code; a step more runs to the end at a rate that never goes negative.
#[test]
fn adamw_steps_at_the_warmup_edge() {
    let output = modelspec(&["train", FIXTURE, "--backend", "cpu", "--steps", "10"]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(output.stdout.is_empty(), "{}", text(&output.stdout));
    assert_eq!(stderr.lines().count(), 1, "{stderr}");
    assert!(stderr.contains("10 steps"), "{stderr}");
    assert!(!stderr.contains("E-"), "{stderr}");

    let output = modelspec(&["train", FIXTURE, "--backend", "cpu", "--steps", "11"]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    let output = modelspec(&["train", FIXTURE, "--backend", "cpu", "--steps", "400"]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    let stdout = text(&output.stdout);
    assert!(stdout.contains("step 400 loss "), "{stdout}");
}
