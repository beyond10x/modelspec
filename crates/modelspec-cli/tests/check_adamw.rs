//! `modelspec check` on the arguments of `adamw` and on `sgd` with a schedule argument
//! (DESIGN.md § 2.1, § 8.1; story:bert-adamw-warmup).
//!
//! `optimizer = adamw(lr = F, weight_decay = F, beta1 = F, beta2 = F, eps = F, warmup_steps = N)`
//! takes every argument once, named, in any order (coordinator decision D50, 2026-10-25,
//! `.engineering/waves/bert-1.md`). One it omits is `E-TYPE-001` at the `adamw` call; `lr` that
//! is not a positive finite `f32` is `E-TYPE-010`; `weight_decay < 0`, a beta outside `[0, 1)`
//! and `eps <= 0` are `E-TYPE-023`; `warmup_steps >= steps` is `E-TYPE-024`; each at its
//! argument. `sgd(lr = F)` takes no other argument: a schedule argument after `lr` is
//! `E-SYNTAX-002` at that argument, so `sgd` keeps a constant rate. The fixtures are under
//! `tests/fixtures/` of this crate.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const FIXTURES: &str = "crates/modelspec-cli/tests/fixtures";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("UTF-8")
}

fn check(fixture: &str) -> (Output, String) {
    let relative = format!("{FIXTURES}/{fixture}");
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .arg("check")
        .arg(&relative)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs");
    (output, relative)
}

/// The line and column of the first `needle` in the fixture's source, 1-based.
fn position(fixture: &str, needle: &str) -> (usize, usize) {
    let source = std::fs::read_to_string(repo_root().join(FIXTURES).join(fixture))
        .expect("the fixture exists");
    source
        .lines()
        .enumerate()
        .find_map(|(i, line)| line.find(needle).map(|c| (i + 1, c + 1)))
        .unwrap_or_else(|| panic!("{fixture} holds `{needle}`"))
}

/// `check` exits 1 with exactly one diagnostic, `code` at the first `at` of the source, whose
/// first line names `named`; returns stderr.
fn refused(fixture: &str, code: &str, at: &str, named: &str) -> String {
    let (output, relative) = check(fixture);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{fixture}:\n{stderr}");
    assert!(output.stdout.is_empty(), "{fixture}: check wrote to stdout");
    let lines: Vec<&str> = stderr.lines().collect();
    let heads = lines.iter().filter(|l| !l.starts_with("  ")).count();
    assert_eq!(heads, 1, "{fixture}: one diagnostic expected:\n{stderr}");
    assert!(
        lines[0].starts_with(&format!("{code}: ")),
        "{fixture}: expected {code}:\n{stderr}"
    );
    assert!(
        lines[0].contains(named),
        "{fixture}: {:?} does not name {named}",
        lines[0]
    );
    let (line, column) = position(fixture, at);
    assert_eq!(
        lines.get(1).copied(),
        Some(format!("  {relative}:{line}:{column}").as_str()),
        "{fixture}: location\n{stderr}"
    );
    stderr
}

/// Scenario 6. A missing `beta2` is `E-TYPE-001` at the `adamw` call, naming `beta2`.
#[test]
fn check_adamw_missing_beta2_is_e_type_001() {
    refused(
        "adamw_missing_beta2.mspec",
        "E-TYPE-001",
        "adamw(",
        "`beta2`",
    );
}

/// Scenario 6. `beta1 = 1.0` is outside `[0, 1)`: `E-TYPE-023` at `beta1`.
#[test]
fn check_adamw_beta1_one_is_e_type_023() {
    let stderr = refused("adamw_beta1_one.mspec", "E-TYPE-023", "beta1 =", "`beta1`");
    assert!(stderr.contains("[0, 1)"), "{stderr}");
}

/// Scenario 6. `eps = 0.0` is not positive: `E-TYPE-023` at `eps`.
#[test]
fn check_adamw_eps_zero_is_e_type_023() {
    refused("adamw_eps_zero.mspec", "E-TYPE-023", "eps =", "`eps`");
}

/// Scenario 6. `weight_decay = -0.1` is negative: `E-TYPE-023` at `weight_decay`.
#[test]
fn check_adamw_negative_weight_decay_is_e_type_023() {
    let stderr = refused(
        "adamw_weight_decay_negative.mspec",
        "E-TYPE-023",
        "weight_decay =",
        "`weight_decay`",
    );
    assert!(stderr.contains("-0.1"), "{stderr}");
}

/// `warmup_steps` equal to `steps` leaves no step to decay over: `E-TYPE-024` at the argument.
#[test]
fn check_adamw_warmup_as_long_as_the_run_is_e_type_024() {
    refused(
        "adamw_warmup_too_long.mspec",
        "E-TYPE-024",
        "warmup_steps =",
        "`warmup_steps`",
    );
}

/// `lr = 0.0` is the learning-rate code of `sgd`, `E-TYPE-010`, at the `lr` argument.
#[test]
fn check_adamw_zero_learning_rate_is_e_type_010() {
    refused("adamw_lr_zero.mspec", "E-TYPE-010", "lr =", "learning rate");
}

/// An argument `adamw` does not take is `E-TYPE-002` at it.
#[test]
fn check_adamw_unknown_or_repeated_argument_is_e_type_002() {
    refused(
        "adamw_unknown_argument.mspec",
        "E-TYPE-002",
        "momentum =",
        "`momentum`",
    );
}

/// Scenario 6. `sgd(lr = 0.1, warmup_steps = 10)`: `sgd` takes `lr` alone, so the schedule
/// argument is `E-SYNTAX-002` at the comma before it, as the parser places any argument after
/// `lr`, naming the argument.
#[test]
fn check_sgd_with_a_schedule_argument_is_e_syntax_002() {
    refused(
        "sgd_warmup.mspec",
        "E-SYNTAX-002",
        ", warmup_steps",
        "`warmup_steps`",
    );
}

/// The AdamW fixture of `train_adamw` checks clean, with its arguments in another order too.
#[test]
fn check_adamw_checks_clean_in_any_order() {
    for fixture in ["linear_regression_adamw.mspec", "adamw_reordered.mspec"] {
        let (output, _) = check(fixture);
        let stderr = text(&output.stderr);
        assert_eq!(output.status.code(), Some(0), "{fixture}: {stderr}");
        assert_eq!(stderr, "", "{fixture}: a clean check prints nothing");
    }
}
