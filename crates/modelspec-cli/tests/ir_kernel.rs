//! `modelspec ir --stage kernel`: the Kernel IR and both execution plans of a training step
//! (DESIGN.md § 3.1, § 7.1, § 8; story m1-kernel-plan acceptance 6).
//!
//! The golden `tests/golden/linear_regression.kernel.ir` is what `print_kernel` prints for the
//! example's Vulkan artifact; the cases below check it against the plan the compiler builds, so a
//! golden regenerated from a wrong plan does not pass.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use modelspec_core::artifact::Backend;
use modelspec_core::compile;
use modelspec_core::ir::print::print_kernel;

const EXAMPLE: &str = "examples/linear_regression.mspec";
const GOLDEN: &str = include_str!("golden/linear_regression.kernel.ir");

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
    String::from_utf8(bytes.to_vec()).expect("output is UTF-8")
}

#[test]
fn ir_stage_kernel_prints_exactly_the_golden() {
    let output = modelspec(&["ir", EXAMPLE, "--stage", "kernel"]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    assert_eq!(text(&output.stdout), GOLDEN);
    assert_eq!(text(&output.stderr), "");
}

#[test]
fn the_golden_is_the_printed_plan_of_the_vulkan_artifact() {
    let source = std::fs::read_to_string(repo_root().join(EXAMPLE)).expect("the example exists");
    let artifact =
        compile::compile_artifact(EXAMPLE, &source, Backend::Vulkan).expect("the example compiles");
    assert_eq!(print_kernel(&artifact.execution), GOLDEN);
}

#[test]
fn the_golden_shows_both_plans_and_no_completion_point() {
    assert!(GOLDEN.contains("\nplan step\n"), "{GOLDEN}");
    assert!(GOLDEN.contains("\nplan forward\n"), "{GOLDEN}");
    assert!(!GOLDEN.contains(": complete"), "{GOLDEN}");
    let loss_sum = GOLDEN
        .lines()
        .find(|l| l.contains(" loss_sum Mse "))
        .expect("the golden has the mse loss-sum kernel");
    assert!(
        loss_sum.ends_with("workgroup_size 1 workgroups [1, 1, 1]"),
        "{loss_sum}"
    );
}

#[test]
fn ir_stage_kernels_is_a_usage_error() {
    let output = modelspec(&["ir", EXAMPLE, "--stage", "kernels"]);
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(text(&output.stdout), "");
    let stderr = text(&output.stderr);
    assert!(stderr.contains("'kernels'"), "{stderr}");
    assert!(
        stderr.contains("[possible values: tensor, training, kernel]"),
        "{stderr}"
    );
}

/// A `[4, 1073741824]` f32 parameter is 16 GiB, past the `u32` byte range of a storage buffer
/// binding, and its forward needs more invocations than a 32-bit index counts.
#[test]
fn ir_stage_kernel_refuses_a_tensor_past_the_u32_limits_with_its_target_code() {
    let source = std::fs::read_to_string(repo_root().join(EXAMPLE))
        .expect("the example exists")
        .replace("[4, 1]", "[4, 1073741824]")
        .replace("[8, 1]", "[8, 1073741824]");
    assert!(source.contains("Tensor<f32, [4, 1073741824]>"), "{source}");
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("u32_limit.mspec");
    std::fs::write(&path, source).expect("the fixture is written");
    let path = path.to_str().expect("a UTF-8 path");

    assert_eq!(modelspec(&["check", path]).status.code(), Some(0));
    let output = modelspec(&["ir", path, "--stage", "kernel"]);
    assert_eq!(output.status.code(), Some(1), "{}", text(&output.stdout));
    assert_eq!(text(&output.stdout), "");
    let stderr = text(&output.stderr);
    assert!(stderr.contains("E-TARGET-010"), "{stderr}");
}

#[test]
fn ir_stage_kernel_reports_diagnostics_as_check_does() {
    let fixture = "tests/language/check/matmul_mismatch.mspec";
    let check = modelspec(&["check", fixture]);
    let ir = modelspec(&["ir", fixture, "--stage", "kernel"]);
    assert_eq!(check.status.code(), Some(1));
    assert_eq!(ir.status.code(), Some(1));
    assert_eq!(text(&ir.stdout), "");
    assert_eq!(text(&ir.stderr), text(&check.stderr));
}
