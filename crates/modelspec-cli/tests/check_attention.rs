//! `modelspec check` and `ir` on multi-head attention with a key mask (DESIGN.md § 2.1, § 2.2,
//! § 4; story:bert-attention).
//!
//! `attention(q, k, v, key_mask, heads = A)` takes `q`, `k`, `v` of shape `[N, H]`, `key_mask` of
//! shape `[B, S]` with `B * S = N`, and the named compile-time argument `heads`, an integer
//! literal that divides `H`. Each refused call exits 1 with one diagnostic at the call, or at
//! the argument it names. A file that uses `attention` correctly checks clean and lowers to a
//! training program holding its three gradient ops. The fixtures are under `tests/fixtures/` of
//! this crate.

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

/// `check` exits 1 with exactly one diagnostic, `code` at `line:column`, whose first line names
/// `named`; returns stderr.
fn refused_at(fixture: &str, named: &str, code: &str, line: u32, column: u32) -> String {
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
        Some(format!("  {relative}:{line}:{column}").as_str()),
        "{fixture}: location\n{stderr}"
    );
    stderr
}

/// The line of every fixture's `attention` call, and the column the call starts at.
const CALL: (u32, u32) = (7, 16);

/// No `heads`: `attention` is called without an argument it needs, `E-TYPE-001` at the call.
#[test]
fn attention_without_heads_is_e_type_001_at_the_call() {
    let stderr = refused_at(
        "attention_no_heads.mspec",
        "`heads`",
        "E-TYPE-001",
        CALL.0,
        CALL.1,
    );
    assert!(stderr.contains("`attention`"), "{stderr}");
}

/// `heads = 3` with `H = 4`: `E-TYPE-020` at the call, naming both numbers.
#[test]
fn attention_with_heads_3_and_h_4_is_e_type_020_at_the_call() {
    let stderr = refused_at(
        "attention_heads_3.mspec",
        "`attention`",
        "E-TYPE-020",
        CALL.0,
        CALL.1,
    );
    assert!(
        stderr.contains("heads = 3") && stderr.contains('4'),
        "{stderr}"
    );
}

/// `heads = x`, a name and not an integer literal: `E-TYPE-019` at the argument.
#[test]
fn attention_with_a_non_literal_heads_is_e_type_019_at_the_argument() {
    refused_at(
        "attention_heads_not_literal.mspec",
        "`heads`",
        "E-TYPE-019",
        CALL.0,
        84,
    );
}

/// A `key_mask` of shape `[2, 4]` for `q` of shape `[6, 4]`: `B * S = 8` is not `N = 6`, the
/// shape code `E-TYPE-003` at the call, both types named.
#[test]
fn attention_with_a_key_mask_whose_b_times_s_is_not_n_is_e_type_003_at_the_call() {
    let stderr = refused_at(
        "attention_mask_shape.mspec",
        "`attention`",
        "E-TYPE-003",
        CALL.0,
        CALL.1,
    );
    assert!(
        stderr.contains("Tensor<f32, [6, 4]>") && stderr.contains("Tensor<f32, [2, 4]>"),
        "{stderr}"
    );
}

/// `causal = 1`, a named argument `attention` does not take: `E-TYPE-002` at the argument.
#[test]
fn attention_with_an_unknown_named_argument_is_e_type_002_at_the_argument() {
    let stderr = refused_at(
        "attention_unknown_named.mspec",
        "`causal`",
        "E-TYPE-002",
        CALL.0,
        95,
    );
    assert!(
        stderr.contains("`heads`"),
        "the message names what it takes:\n{stderr}"
    );
}

/// A `key_mask` that is a parameter: a key mask has no gradient, so the checker refuses it,
/// `E-TYPE-021` at the argument.
#[test]
fn attention_with_a_parameter_as_key_mask_is_e_type_021_at_the_argument() {
    refused_at(
        "attention_param_mask.mspec",
        "`key_mask`",
        "E-TYPE-021",
        CALL.0,
        74,
    );
}

#[test]
fn attention_checks_clean() {
    let (output, _) = modelspec("check", "attention_train.mspec", &[]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(0), "{stderr}");
    assert!(output.stdout.is_empty());
    assert_eq!(stderr, "", "a clean check prints nothing");
}

/// The training program holds `Attention` with its `heads` and each gradient op over the same
/// four operands and the adjoint.
#[test]
fn the_training_ir_holds_attention_and_its_three_gradient_ops() {
    let (output, _) = modelspec("ir", "attention_train.mspec", &["--stage", "training"]);
    let stdout = text(&output.stdout);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{stdout}\n{}",
        text(&output.stderr)
    );
    for op in [
        "= Attention [heads = 2] ",
        "= AttentionGradQ [heads = 2] ",
        "= AttentionGradK [heads = 2] ",
        "= AttentionGradV [heads = 2] ",
    ] {
        assert!(stdout.contains(op), "{op}:\n{stdout}");
    }
}
