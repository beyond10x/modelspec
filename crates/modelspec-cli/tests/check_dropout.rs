//! `modelspec check` and `ir` on dropout (DESIGN.md § 2.1, § 2.3, § 4; story:bert-dropout).
//!
//! `dropout(x, p = <float literal>)` takes one operand of any shape and the named compile-time
//! argument `p`, a float literal in `[0, 1)`. A `p` that is not a float literal is `E-TYPE-019`
//! and a float literal outside `[0, 1)` is `E-TYPE-022`, each at the `p` argument of the call
//! (coordinator decision D47, 2026-10-25, `.engineering/waves/bert-1.md`). A file that uses
//! `dropout` checks clean; its Tensor IR names the site on the `Dropout` node and reads the
//! site's RNG token, and `p = 0.0` lowers to no node at all. The fixtures are under
//! `tests/fixtures/` of this crate.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const FIXTURES: &str = "crates/modelspec-cli/tests/fixtures";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Runs `modelspec <command> <fixture> <more>` from the repository root.
fn modelspec(command: &str, fixture: &str, more: &[&str]) -> (Output, String) {
    let relative = format!("{FIXTURES}/{fixture}");
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .arg(command)
        .arg(&relative)
        .args(more)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs");
    (output, relative)
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("UTF-8")
}

/// The line of every fixture's `dropout` call, and the column of its `p` argument.
const P_ARGUMENT: (u32, u32) = (7, 28);

/// `check` exits 1 with exactly one diagnostic, `code` at the `p` argument, whose first line
/// names `named`; returns stderr.
fn refused(fixture: &str, named: &str, code: &str) -> String {
    let (output, relative) = modelspec("check", fixture, &[]);
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
    assert_eq!(
        lines.get(1).copied(),
        Some(format!("  {relative}:{}:{}", P_ARGUMENT.0, P_ARGUMENT.1).as_str()),
        "{fixture}: location\n{stderr}"
    );
    stderr
}

/// Scenario 6. `p = 1.0` would scale by `1 / 0`: `E-TYPE-022` at the argument.
#[test]
fn check_dropout_p_one_is_e_type_022() {
    let stderr = refused("dropout_p_one.mspec", "`p` of `dropout`", "E-TYPE-022");
    assert!(stderr.contains("[0, 1)"), "{stderr}");
}

/// Scenario 6. `p = -0.1` is below 0: `E-TYPE-022` at the argument.
#[test]
fn check_dropout_p_negative_is_e_type_022() {
    let stderr = refused("dropout_p_negative.mspec", "`p` of `dropout`", "E-TYPE-022");
    assert!(stderr.contains("-0.1"), "{stderr}");
}

/// Scenario 6. `p = x` names a value, not a literal: `E-TYPE-019` at the argument.
#[test]
fn check_dropout_p_not_a_literal_is_e_type_019() {
    refused(
        "dropout_p_not_literal.mspec",
        "`p` of `dropout`",
        "E-TYPE-019",
    );
}

/// `p = 0` is an integer literal, not a float literal: `E-TYPE-019` at the argument.
#[test]
fn check_dropout_p_integer_is_e_type_019() {
    refused("dropout_p_integer.mspec", "`p` of `dropout`", "E-TYPE-019");
}

#[test]
fn check_dropout_checks_clean() {
    for fixture in ["dropout_train.mspec", "dropout_p0.mspec"] {
        let (output, _) = modelspec("check", fixture, &[]);
        let stderr = text(&output.stderr);
        assert_eq!(output.status.code(), Some(0), "{fixture}: {stderr}");
        assert!(output.stdout.is_empty(), "{fixture}");
        assert_eq!(stderr, "", "{fixture}: a clean check prints nothing");
    }
}

/// The Tensor IR reads the site's token, an `i32[2]` input named `Reg.forward.dropout#0`, and
/// prints `p` and the site on the `Dropout` node; the training IR holds `DropoutGrad` reading the
/// same token. `p = 0.0` lowers to no `Dropout` node and no token.
#[test]
fn check_dropout_ir_names_the_site_and_its_token() {
    let (output, _) = modelspec("ir", "dropout_train.mspec", &["--stage", "tensor"]);
    let stdout = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    let token = stdout
        .lines()
        .find(|l| l.ends_with("= Input \"Reg.forward.dropout#0\""))
        .unwrap_or_else(|| panic!("no token input:\n{stdout}"));
    assert!(token.contains(": Tensor<i32, [2]> ="), "{token}");
    let id = token.split(':').next().unwrap();
    let node = stdout
        .lines()
        .find(|l| l.contains("= Dropout [p = 0.3, site = Reg.forward.dropout#0] "))
        .unwrap_or_else(|| panic!("no Dropout node:\n{stdout}"));
    assert!(node.ends_with(&format!(", {id}")), "{node} reads {id}");

    let (output, _) = modelspec("ir", "dropout_train.mspec", &["--stage", "training"]);
    let stdout = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    let grad = stdout
        .lines()
        .find(|l| l.contains("= DropoutGrad [p = 0.3, site = Reg.forward.dropout#0] "))
        .unwrap_or_else(|| panic!("no DropoutGrad node:\n{stdout}"));
    assert!(grad.contains(&format!("] {id}, ")), "{grad} reads {id}");

    let (output, _) = modelspec("ir", "dropout_p0.mspec", &["--stage", "training"]);
    let stdout = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    assert!(!stdout.contains("Dropout"), "{stdout}");
    assert!(!stdout.contains("dropout#"), "{stdout}");
}

/// `dropout` has no Vulkan kernel: `compile --backend vulkan` is `E-TARGET-011` at the call,
/// and `compile --backend cpu` writes an artifact, the token taking a buffer of its own.
#[test]
fn compile_dropout_is_e_target_011_for_vulkan_and_compiles_for_cpu() {
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("check_dropout");
    let vulkan = out.join("vulkan");
    let (output, relative) = modelspec(
        "compile",
        "dropout_train.mspec",
        &["--backend", "vulkan", "--output", vulkan.to_str().unwrap()],
    );
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.starts_with("E-TARGET-011: operator `dropout` has no Vulkan kernel yet\n"),
        "{stderr}"
    );
    assert!(stderr.contains(&format!("  {relative}:7:17\n")), "{stderr}");

    let cpu = out.join("cpu");
    let (output, _) = modelspec(
        "compile",
        "dropout_train.mspec",
        &["--backend", "cpu", "--output", cpu.to_str().unwrap()],
    );
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    assert!(cpu.join("artifact.json").is_file());
}
