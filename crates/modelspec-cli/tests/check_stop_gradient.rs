//! `modelspec check`, `ir`, `train` and `compile` on `stop_gradient` (DESIGN.md § 2.2, § 3.3, § 4,
//! § 8.1; story:jepa-stop-gradient).
//!
//! `stop_gradient(x)` takes one operand of any `f32` shape and returns it unchanged; no gradient
//! flows back through it. A parameter that the loss reads only through `stop_gradient` would get
//! no gradient, so `check` refuses it with `E-TYPE-026` at the `stop_gradient` call through
//! which it reaches the loss: directly, through a `let`, or through an argument of a model
//! function that the function passes to `stop_gradient`. A parameter the loss also reads
//! elsewhere, in the same function or in another one, is no reason to refuse. The Tensor IR
//! holds the `StopGradient` node; the training step does not, because a pass after autodiff
//! replaces each use of it by its operand. The fixtures are under `tests/fixtures/` of this
//! crate.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const FIXTURES: &str = "crates/modelspec-cli/tests/fixtures";

/// A path to no Vulkan loader, so `compile` cannot reach a device.
const NO_LOADER: &str = "/nonexistent/modelspec-test/libvulkan-missing.so.1";

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
        .env("MODELSPEC_VULKAN_LIBRARY", NO_LOADER)
        .output()
        .expect("modelspec runs");
    (output, relative)
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("UTF-8")
}

/// `check` exits 1 with exactly one diagnostic, `E-TYPE-026` at `line:column`, naming the
/// parameter `param` and `stop_gradient`; returns stderr.
fn refused(fixture: &str, param: &str, line: u32, column: u32) -> String {
    let (output, relative) = modelspec("check", fixture, &[]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{fixture}:\n{stderr}");
    assert!(output.stdout.is_empty(), "{fixture}: check wrote to stdout");
    let lines: Vec<&str> = stderr.lines().collect();
    let heads = lines.iter().filter(|l| !l.starts_with("  ")).count();
    assert_eq!(heads, 1, "{fixture}: one diagnostic expected:\n{stderr}");
    assert!(
        lines[0].starts_with("E-TYPE-026: "),
        "{fixture}: expected E-TYPE-026:\n{stderr}"
    );
    assert!(
        lines[0].contains(&format!("`{param}`")) && lines[0].contains("`stop_gradient`"),
        "{fixture}: {:?} names neither `{param}` nor `stop_gradient`",
        lines[0]
    );
    assert_eq!(
        lines.get(1).copied(),
        Some(format!("  {relative}:{line}:{column}").as_str()),
        "{fixture}: location\n{stderr}"
    );
    stderr
}

/// `w` is read only under `stop_gradient`: E-TYPE-026 at the call.
#[test]
fn check_stop_gradient_parameter_only_under_it_is_e_type_026() {
    refused("stop_gradient_only.mspec", "w", 6, 34);
}

/// `w` reaches the loss only through a `let` bound to `stop_gradient(...)`: E-TYPE-026 at the
/// call in the `let`.
#[test]
fn check_stop_gradient_through_a_let_is_e_type_026() {
    refused("stop_gradient_let.mspec", "w", 6, 20);
}

/// `w` is read by `encode`, whose result `forward` passes to `stop_gradient`: E-TYPE-026 at the
/// call inside `forward`.
#[test]
fn check_stop_gradient_through_a_function_argument_is_e_type_026() {
    refused("stop_gradient_argument.mspec", "w", 10, 23);
}

/// A parameter the loss also reads outside `stop_gradient`, in the same expression or through
/// another model function, checks clean.
#[test]
fn check_stop_gradient_with_a_live_read_checks_clean() {
    for fixture in [
        "stop_gradient_target.mspec",
        "stop_gradient_live_elsewhere.mspec",
    ] {
        let (output, _) = modelspec("check", fixture, &[]);
        let stderr = text(&output.stderr);
        assert_eq!(output.status.code(), Some(0), "{fixture}: {stderr}");
        assert!(output.stdout.is_empty(), "{fixture}");
        assert_eq!(stderr, "", "{fixture}: a clean check prints nothing");
    }
}

/// `ir --stage tensor` prints the `StopGradient` node over the second `matmul`; `ir --stage
/// training` prints no `StopGradient`, and the forward `Add` reads that `matmul` directly.
#[test]
fn check_stop_gradient_ir_tensor_has_it_and_training_does_not() {
    let (output, _) = modelspec("ir", "stop_gradient_target.mspec", &["--stage", "tensor"]);
    let tensor = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    let stop = tensor
        .lines()
        .find(|l| l.contains("= StopGradient "))
        .unwrap_or_else(|| panic!("no StopGradient node:\n{tensor}"));
    assert!(stop.contains(": Tensor<f32, [6, 4]> ="), "{stop}");
    let operand = stop.rsplit(' ').next().unwrap();
    let matmul = tensor
        .lines()
        .find(|l| l.starts_with(&format!("{operand}: ")))
        .unwrap_or_else(|| panic!("no node {operand}:\n{tensor}"));
    assert!(matmul.contains("= MatMul "), "{matmul}");

    let (output, _) = modelspec("ir", "stop_gradient_target.mspec", &["--stage", "training"]);
    let training = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    assert!(
        !training.contains("StopGradient"),
        "the training step holds StopGradient:\n{training}"
    );
    assert!(
        training
            .lines()
            .any(|l| l.contains("= Add ") && l.ends_with(&format!(", {operand}"))),
        "the Add reads {operand} directly:\n{training}"
    );
}

/// The file trains on the CPU, and compiles for Vulkan against a device profile without a
/// device: the schedule never sees `StopGradient`.
#[test]
fn check_stop_gradient_trains_on_the_cpu_and_compiles_for_vulkan() {
    let (output, _) = modelspec("train", "stop_gradient_target.mspec", &["--backend", "cpu"]);
    let stdout = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    assert!(
        stdout.lines().any(|l| l.starts_with("step 2 loss ")),
        "{stdout}"
    );

    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("check_stop_gradient_artifact");
    if out.exists() {
        std::fs::remove_dir_all(&out).expect("the old artifact is removed");
    }
    let profile = repo_root().join("crates/modelspec-runtime/tests/fixtures/rtx3090.device.json");
    let (output, _) = modelspec(
        "compile",
        "stop_gradient_target.mspec",
        &[
            "--backend",
            "vulkan",
            "--device-profile",
            &profile.display().to_string(),
            "--output",
            &out.display().to_string(),
        ],
    );
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
}
