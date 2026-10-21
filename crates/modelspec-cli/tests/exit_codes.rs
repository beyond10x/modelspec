//! The exit-code convention of `modelspec`: 0 success, 1 diagnostics or runtime error, 2 usage.

use std::process::Command;

/// Runs the `modelspec` binary with `args` and returns its exit code.
fn exit_code(args: &[&str]) -> i32 {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .output()
        .expect("modelspec runs")
        .status
        .code()
        .expect("modelspec exits with a code")
}

#[test]
fn help_exits_0() {
    assert_eq!(exit_code(&["--help"]), 0);
}

#[test]
fn no_arguments_exits_2() {
    assert_eq!(exit_code(&[]), 2);
}

#[test]
fn unknown_subcommand_exits_2() {
    assert_eq!(exit_code(&["no-such-subcommand"]), 2);
}

#[test]
fn version_exits_0() {
    assert_eq!(exit_code(&["--version"]), 0);
}

#[test]
fn subcommand_without_its_file_exits_2() {
    assert_eq!(exit_code(&["check"]), 2);
}

#[test]
fn check_on_a_missing_file_exits_1_with_a_diagnostic() {
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["check", "no-such-model.mspec"])
        .output()
        .expect("modelspec runs");
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("no-such-model.mspec"), "{stderr}");
}

#[test]
fn ir_on_a_missing_file_exits_1() {
    assert_eq!(exit_code(&["ir", "model.mspec", "--stage", "tensor"]), 1);
}

#[test]
fn train_on_a_missing_file_exits_1() {
    assert_eq!(exit_code(&["train", "model.mspec", "--backend", "cpu"]), 1);
}
