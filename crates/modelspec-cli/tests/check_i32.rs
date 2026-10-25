//! `modelspec check` on `i32` tensors (DESIGN.md § 2.2; story:bert-embedding-lookup).
//!
//! An `i32` tensor is declared only by an `input` or `target` of a train block or by a function
//! parameter, and is read only as the ids of `embedding`. Any other use is refused with
//! `E-TYPE-018` at the offending span: arithmetic with it, a `param` of type `i32`, and a
//! function result of type `i32`. The fixtures are under `tests/fixtures/` of this crate.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const FIXTURES: &str = "crates/modelspec-cli/tests/fixtures";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Runs `modelspec check <fixture>` from the repository root.
fn check(fixture: &str) -> (Output, String) {
    let relative = format!("{FIXTURES}/{fixture}");
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["check", &relative])
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs");
    (output, relative)
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr is UTF-8")
}

/// `check` exits 1 with exactly one diagnostic, `E-TYPE-018` at `line:column`, whose first line
/// holds `fragment`.
fn refused_at(fixture: &str, line: u32, column: u32, fragment: &str) {
    let (output, relative) = check(fixture);
    let text = stderr(&output);
    assert_eq!(output.status.code(), Some(1), "{fixture}:\n{text}");
    assert!(output.stdout.is_empty(), "{fixture}: check wrote to stdout");
    let lines: Vec<&str> = text.lines().collect();
    let heads = lines.iter().filter(|l| !l.starts_with("  ")).count();
    assert_eq!(heads, 1, "{fixture}: one diagnostic expected:\n{text}");
    assert!(
        lines[0].starts_with("E-TYPE-018: "),
        "{fixture}: expected E-TYPE-018:\n{text}"
    );
    assert!(
        lines[0].contains(fragment),
        "{fixture}: {:?} lacks {fragment:?}",
        lines[0]
    );
    assert_eq!(
        lines.get(1).copied(),
        Some(format!("  {relative}:{line}:{column}").as_str()),
        "{fixture}: location\n{text}"
    );
}

#[test]
fn an_i32_added_to_an_f32_is_refused_at_the_call() {
    refused_at("i32_add_to_f32.mspec", 5, 16, "`add` operand 1");
}

#[test]
fn an_i32_param_is_refused_at_its_dtype() {
    refused_at("i32_param.mspec", 2, 21, "parameter `w`");
}

#[test]
fn an_i32_function_result_is_refused_at_its_dtype() {
    refused_at("i32_function_result.mspec", 4, 48, "`ids_of`");
}

#[test]
fn ids_read_only_by_embedding_check_clean() {
    let (output, _) = check("embedding_ids.mspec");
    let text = stderr(&output);
    assert_eq!(output.status.code(), Some(0), "{text}");
    assert!(output.stdout.is_empty());
    assert_eq!(text, "", "a clean check prints nothing");
}
