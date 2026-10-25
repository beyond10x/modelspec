//! `modelspec check` on the arguments of `muon` (DESIGN.md § 2.1, § 8.1; story:muon-optimizer,
//! scenario 4).
//!
//! `optimizer = muon(lr = F, momentum = F, weight_decay = F, ns_steps = N, warmup_steps = N,
//! adam_lr = F, beta1 = F, beta2 = F, eps = F)` takes every argument once, named, in any order,
//! with the parsing rules and codes of `adamw` (coordinator decision D67 (d), 2026-10-25,
//! `.engineering/waves/bert-1.md`). One it omits is `E-TYPE-001` at the `muon` call, one given
//! twice `E-TYPE-002` at the second; `momentum` outside `[0, 1)` and `ns_steps` outside `[1, 10]`
//! are the new `E-TYPE-025`, each at its argument. The fixtures are under `tests/fixtures/` of
//! this crate.

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

/// Scenario 4. `momentum = 1.0` is outside `[0, 1)`: `E-TYPE-025` at `momentum`.
#[test]
fn check_muon_momentum_one_is_e_type_025() {
    let stderr = refused(
        "muon_momentum_one.mspec",
        "E-TYPE-025",
        "momentum =",
        "`momentum`",
    );
    assert!(stderr.contains("[0, 1)"), "{stderr}");
}

/// Scenario 4. `ns_steps = 0` is below `[1, 10]`: `E-TYPE-025` at `ns_steps`.
#[test]
fn check_muon_ns_steps_zero_is_e_type_025() {
    let stderr = refused(
        "muon_ns_steps_zero.mspec",
        "E-TYPE-025",
        "ns_steps =",
        "`ns_steps`",
    );
    assert!(stderr.contains("[1, 10]"), "{stderr}");
}

/// Scenario 4. `ns_steps = 11` is above `[1, 10]`: `E-TYPE-025` at `ns_steps`.
#[test]
fn check_muon_ns_steps_eleven_is_e_type_025() {
    let stderr = refused(
        "muon_ns_steps_eleven.mspec",
        "E-TYPE-025",
        "ns_steps =",
        "`ns_steps`",
    );
    assert!(stderr.contains("11"), "{stderr}");
}

/// Scenario 4. A missing `adam_lr` is `E-TYPE-001` at the `muon` call, naming `adam_lr`.
#[test]
fn check_muon_missing_argument_is_e_type_001() {
    refused(
        "muon_missing_adam_lr.mspec",
        "E-TYPE-001",
        "muon(",
        "`adam_lr`",
    );
}

/// Scenario 4. `momentum` given twice is `E-TYPE-002` at the second.
#[test]
fn check_muon_repeated_argument_is_e_type_002() {
    refused(
        "muon_repeated_momentum.mspec",
        "E-TYPE-002",
        "momentum = 0.5",
        "`momentum`",
    );
}

/// The Muon fixtures of `train_muon` check clean.
#[test]
fn check_muon_checks_clean() {
    for fixture in ["linear_regression_muon.mspec", "mlm_dense.mspec"] {
        let (output, _) = check(fixture);
        let stderr = text(&output.stderr);
        assert_eq!(output.status.code(), Some(0), "{fixture}: {stderr}");
        assert_eq!(stderr, "", "{fixture}: a clean check prints nothing");
    }
}

/// E-SYNTAX-013 names `muon` among the optimizers it accepts.
#[test]
fn check_unknown_optimizer_names_muon() {
    let relative = "tests/language/syntax/E-SYNTAX-013-optimizer.mspec";
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .arg("check")
        .arg(relative)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs");
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    let first = stderr.lines().next().unwrap_or_default();
    assert!(first.starts_with("E-SYNTAX-013: "), "{stderr}");
    assert!(first.contains("`muon`"), "{stderr}");
}
