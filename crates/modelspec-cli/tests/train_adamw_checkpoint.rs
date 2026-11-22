//! AdamW checkpoints resume the saved optimizer on the original absolute step schedule.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const FIXTURE: &str = "crates/modelspec-cli/tests/fixtures/linear_regression_adamw.mspec";

fn train(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["train", FIXTURE, "--backend", "cpu"])
        .args(args)
        .current_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
        .output()
        .expect("modelspec runs")
}

fn success(args: &[&str]) {
    let run = train(args);
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("train_adamw_checkpoint");
    fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn manifest(path: &Path) -> serde_json::Value {
    let bytes = fs::read(path).expect("checkpoint written");
    assert_eq!(&bytes[..4], b"MSCK");
    let length = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    serde_json::from_slice(&bytes[56..56 + length]).unwrap()
}

#[test]
fn interval_checkpoint_restores_both_moments_and_the_parameter() {
    let full = scratch("full.msckpt");
    let intermediate = scratch("full-7.msckpt");
    let resumed = scratch("resumed.msckpt");
    success(&[
        "--steps",
        "18",
        "--checkpoint-every",
        "7",
        "--checkpoint",
        full.to_str().unwrap(),
    ]);
    let saved = manifest(&intermediate);
    assert_eq!(saved["step"], 7);
    let slots = saved["optimizer_slots"].as_array().unwrap();
    assert_eq!(slots.len(), 2);
    for (slot, name) in slots.iter().zip(["m", "v"]) {
        assert_eq!(slot["slot"], name);
        assert_eq!(slot["path"], saved["parameters"][0]["path"]);
        assert_eq!(slot["shape"], saved["parameters"][0]["shape"]);
    }
    success(&[
        "--steps",
        "11",
        "--resume",
        intermediate.to_str().unwrap(),
        "--checkpoint",
        resumed.to_str().unwrap(),
    ]);
    assert_eq!(fs::read(full).unwrap(), fs::read(resumed).unwrap());
}

#[test]
fn missing_adamw_checkpoint_is_reported_by_the_reader() {
    let path = scratch("missing.msckpt");
    let run = train(&["--resume", path.to_str().unwrap()]);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert_eq!(run.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("E-CKPT-001"), "{stderr}");
}

#[test]
fn a_step_zero_checkpoint_without_slots_reproduces_the_run() {
    let initial = scratch("initial.msckpt");
    let resumed = scratch("from-zero.msckpt");
    let full = scratch("from-init.msckpt");
    success(&["--steps", "0", "--checkpoint", initial.to_str().unwrap()]);
    let saved = manifest(&initial);
    assert_eq!(saved["step"], 0);
    assert_eq!(saved["optimizer"]["kind"], "AdamW");
    assert_eq!(saved["optimizer_slots"], serde_json::json!([]));
    success(&[
        "--steps",
        "12",
        "--resume",
        initial.to_str().unwrap(),
        "--checkpoint",
        resumed.to_str().unwrap(),
    ]);
    success(&["--steps", "12", "--checkpoint", full.to_str().unwrap()]);
    assert_eq!(fs::read(full).unwrap(), fs::read(resumed).unwrap());
}
