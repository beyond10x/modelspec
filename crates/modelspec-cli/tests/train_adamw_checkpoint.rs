//! `modelspec train --checkpoint` and `--resume` on a file trained with AdamW are usage errors
//! (DESIGN.md § 6, § 8; story:bert-adamw-warmup).
//!
//! A checkpoint holds the parameters and no optimizer slots yet, so writing one for an AdamW run
//! would drop its first and second moments, and resuming would restart them at 0: both are
//! refused with exit 2 and a message naming the limit, before anything runs or is written
//! (coordinator decision D50, 2026-10-25, `.engineering/waves/bert-1.md`).
//!
//! The one exception is a run of `--steps 0` from step 0 without `--checkpoint-every`: its moments
//! are still 0, so the checkpoint of its initial parameters drops nothing. It is how the PyTorch
//! opponent of `opponents/bert_tiny_pretrain/` starts from ModelSpec's own initial parameters
//! (story:bert-tiny-vs-pytorch-cuda).

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
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("train_adamw_checkpoint");
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    let path = dir.join(name);
    let _ = std::fs::remove_file(&path);
    path
}

/// `--checkpoint` exits 2 naming the flag and the moments, and writes no file.
#[test]
fn train_adamw_checkpoint_is_a_usage_error() {
    let path = scratch("run.msckpt");
    let output = modelspec(&[
        "train",
        FIXTURE,
        "--backend",
        "cpu",
        "--checkpoint",
        path.to_str().unwrap(),
    ]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(output.stdout.is_empty(), "{}", text(&output.stdout));
    assert!(stderr.contains("'--checkpoint'"), "{stderr}");
    assert!(stderr.contains("adamw"), "{stderr}");
    assert!(stderr.contains("moments"), "{stderr}");
    assert!(!path.exists(), "no checkpoint is written");
}

/// `--resume` exits 2 the same way, before the checkpoint is read.
#[test]
fn train_adamw_resume_is_a_usage_error() {
    let path = scratch("missing.msckpt");
    let output = modelspec(&[
        "train",
        FIXTURE,
        "--backend",
        "cpu",
        "--resume",
        path.to_str().unwrap(),
    ]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("'--resume'"), "{stderr}");
    assert!(stderr.contains("moments"), "{stderr}");
}

/// The manifest of the checkpoint at `path`, read from the 56-byte header's size field.
fn manifest(path: &Path) -> serde_json::Value {
    let bytes = std::fs::read(path).expect("the checkpoint is written");
    assert_eq!(&bytes[0..4], b"MSCK");
    let size = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    serde_json::from_slice(&bytes[56..56 + size]).expect("a JSON manifest")
}

/// `--steps 0 --checkpoint` writes the step-0 checkpoint of the initial parameters, whose moments
/// are 0, with the file's AdamW optimizer; `--steps 1` and `--checkpoint-every` beside
/// `--steps 0` are still refused.
#[test]
fn train_adamw_step_0_checkpoint_is_written() {
    let path = scratch("step0.msckpt");
    let output = modelspec(&[
        "train",
        FIXTURE,
        "--backend",
        "cpu",
        "--steps",
        "0",
        "--checkpoint",
        path.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    let manifest = manifest(&path);
    assert_eq!(manifest["step"], 0, "{manifest}");
    assert_eq!(manifest["optimizer"]["kind"], "AdamW", "{manifest}");
    assert_eq!(
        manifest["optimizer_slots"],
        serde_json::json!([]),
        "{manifest}"
    );

    for extra in [
        &["--steps", "1"][..],
        &["--steps", "0", "--checkpoint-every", "1"][..],
    ] {
        let refused = scratch("refused.msckpt");
        let mut args = vec![
            "train",
            FIXTURE,
            "--backend",
            "cpu",
            "--checkpoint",
            refused.to_str().unwrap(),
        ];
        args.extend_from_slice(extra);
        let output = modelspec(&args);
        let stderr = text(&output.stderr);
        assert_eq!(output.status.code(), Some(2), "{extra:?}: {stderr}");
        assert!(stderr.contains("moments"), "{stderr}");
        assert!(!refused.exists(), "{extra:?} writes no checkpoint");
    }
}
