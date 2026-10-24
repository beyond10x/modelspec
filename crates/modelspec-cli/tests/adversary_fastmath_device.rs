//! Adversary pass on `story:fast-math-mode` (DESIGN.md § 5.4, § 12 ADR-008) on a Vulkan device.
//!
//! - Attack 6, determinism: `train --backend vulkan --fast-math` on the MNIST example prints the
//!   same bytes twice in a row and again after a deterministic run of the same file ran in
//!   between; the deterministic run prints different bytes, so the flag reaches the device.
//! - Attack 9: both runs print `numerics class: tolerance` (an M1 training run is never in the
//!   bit-for-bit set).
//! - ADR-008 "Comparison": the run record (`--record`) of a `fast_math` run names the class
//!   `tolerance`.
//!
//! Ignored by default (they create a Vulkan instance); run with
//! `cargo test --release -p modelspec-cli --test adversary_fastmath_device -- --ignored`.
#![cfg(feature = "vulkan")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn modelspec(args: &[&str]) -> Output {
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(root())
        .output()
        .expect("modelspec runs");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn class_line(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr)
        .lines()
        .find(|l| l.starts_with("numerics class:"))
        .unwrap_or("none")
        .to_owned()
}

const MNIST: &str = "examples/mnist_mlp.mspec";

#[test]
#[ignore = "needs a Vulkan device"]
fn fast_math_runs_repeat_their_bits_across_an_intervening_deterministic_run() {
    let fast = [
        "train",
        MNIST,
        "--backend",
        "vulkan",
        "--fast-math",
        "--stats",
    ];
    let first = modelspec(&fast);
    let second = modelspec(&fast);
    let deterministic = modelspec(&["train", MNIST, "--backend", "vulkan", "--stats"]);
    let third = modelspec(&fast);
    assert!(!first.stdout.is_empty());
    assert_eq!(first.stdout, second.stdout, "two fast_math runs in a row");
    assert_eq!(
        first.stdout, third.stdout,
        "a fast_math run after a deterministic run"
    );
    assert_ne!(
        first.stdout, deterministic.stdout,
        "the fast_math run prints the deterministic run's bits: the flag did not reach the device"
    );
    assert_eq!(class_line(&first), "numerics class: tolerance");
    assert_eq!(class_line(&third), "numerics class: tolerance");
    assert_eq!(class_line(&deterministic), "numerics class: tolerance");
}

/// Red on the unit (finding ADV-FM-2): the `--record` file carries no numerics class and no
/// `fast_math`; only the `--stats` line names the class.
#[test]
#[ignore = "ADV-FM-2: the run record of a fast_math run does not name the class tolerance"]
fn the_run_record_of_a_fast_math_run_names_the_class_tolerance() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("adversary_fastmath_device");
    std::fs::create_dir_all(&dir).unwrap();
    let record = dir.join("fast.jsonl");
    let _ = std::fs::remove_file(&record);
    let shown = record.display().to_string();
    modelspec(&[
        "train",
        MNIST,
        "--backend",
        "vulkan",
        "--fast-math",
        "--steps",
        "2",
        "--record",
        &shown,
    ]);
    let text = std::fs::read_to_string(&record).expect("the record is written");
    assert!(!text.is_empty());
    assert!(
        text.contains("tolerance"),
        "the run record names no numerics class:\n{text}"
    );
}
