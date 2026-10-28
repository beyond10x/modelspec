//! `modelspec check`, `ir` and `compile` on `sigreg` (DESIGN.md § 2.1, § 2.2, § 2.3, § 4;
//! story:jepa-sigreg-cpu).
//!
//! `sigreg(z, slices = M)` takes one operand `z: f32[N, K]` and the named compile-time argument
//! `slices`, an integer literal of at least 1. A `z` of another rank is `E-TYPE-004` and an `i32`
//! `z` is `E-TYPE-018`, each at the call; `slices = 0` is `E-TYPE-019` at the `slices` argument of
//! the call, as every named-argument value is, and a `slices` whose projection or projected rows
//! overflow `u64` is `E-TYPE-016` there. A file that uses `sigreg` checks clean; its Tensor
//! IR reads the site's projection, an internal `f32[K, M]` input named `<Model>.<fn>.sigreg#<k>`,
//! through `MatMul`, then `SigregColumns` and `ReduceMean`. `compile --backend vulkan` writes an
//! artifact with the two `sigreg` kernels (story:jepa-sigreg-vulkan). The fixtures are under
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

/// The line and column of the `sigreg` call of every refused fixture.
const CALL: (u32, u32) = (5, 17);
/// The column of its `slices` argument.
const SLICES_COLUMN: u32 = 27;

/// `check` exits 1 with exactly one diagnostic, `code` at `line:column`, whose first line names
/// `named`; returns stderr.
fn refused(fixture: &str, named: &str, code: &str, (line, column): (u32, u32)) -> String {
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

/// Done when: a `z` of rank 1 exits 1 with `E-TYPE-004` at the call.
#[test]
fn check_sigreg_z_rank_1_is_e_type_004_at_the_call() {
    refused("sigreg_z_rank1.mspec", "sigreg", "E-TYPE-004", CALL);
}

/// Done when: an `i32` `z` exits 1 with `E-TYPE-018` at the call.
#[test]
fn check_sigreg_i32_z_is_e_type_018_at_the_call() {
    refused("sigreg_z_i32.mspec", "sigreg", "E-TYPE-018", CALL);
}

/// Done when: `slices = 0` exits 1 with `E-TYPE-019` on the call's line, at its `slices`
/// argument.
#[test]
fn check_sigreg_slices_zero_is_e_type_019_at_the_argument() {
    let stderr = refused(
        "sigreg_slices_zero.mspec",
        "`slices` of `sigreg`",
        "E-TYPE-019",
        (CALL.0, SLICES_COLUMN),
    );
    assert!(stderr.contains("at least 1"), "{stderr}");
}

/// ADV-JP-1, DESIGN.md § 2.2 (count overflows are hard errors): a `slices` whose derived
/// projection `f32[4, 2^64 - 1]` overflows `u64` exits 1 with `E-TYPE-016` at the `slices`
/// argument, and `ir --stage tensor` refuses it the same way, never as a compiler bug.
#[test]
fn check_sigreg_slices_whose_projection_overflows_is_e_type_016_at_the_argument() {
    let stderr = refused(
        "sigreg_slices_overflow.mspec",
        "`slices` of `sigreg`",
        "E-TYPE-016",
        (CALL.0, SLICES_COLUMN),
    );
    assert!(stderr.contains("[4, 18446744073709551615]"), "{stderr}");
    let (output, _) = modelspec("ir", "sigreg_slices_overflow.mspec", &["--stage", "tensor"]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.starts_with("E-TYPE-016: "), "{stderr}");
}

/// ADV-JP-1, the rest of its class: every value the lowering derives from `slices` is checked,
/// not only the projection. `slices = 2^59` keeps the projection `f32[4, 2^59]` (2^63 bytes)
/// within `u64` but not `P = z @ proj`, `f32[16, 2^59]`: `E-TYPE-016` at the argument.
#[test]
fn check_sigreg_slices_whose_projected_rows_overflow_is_e_type_016_at_the_argument() {
    let stderr = refused(
        "sigreg_slices_overflow_rows.mspec",
        "`slices` of `sigreg`",
        "E-TYPE-016",
        (CALL.0, SLICES_COLUMN),
    );
    assert!(stderr.contains("[16, 576460752303423488]"), "{stderr}");
}

#[test]
fn check_sigreg_checks_clean() {
    let (output, _) = modelspec("check", "sigreg_train.mspec", &[]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(0), "{stderr}");
    assert!(output.stdout.is_empty());
    assert_eq!(stderr, "", "a clean check prints nothing");
}

/// The Tensor IR reads the site's projection, an `f32[8, 16]` input named
/// `Spread.train.sigreg#0` (the call is in the loss), as the right operand of a `MatMul` whose
/// value `SigregColumns` reads, and `ReduceMean` takes the mean of the columns. The training IR
/// holds `SigregColumnsGrad` reading the same `MatMul`.
#[test]
fn check_sigreg_ir_reads_the_projection_of_the_site() {
    let (output, _) = modelspec("ir", "sigreg_train.mspec", &["--stage", "tensor"]);
    let stdout = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    let projection = stdout
        .lines()
        .find(|l| l.ends_with("= Input \"Spread.train.sigreg#0\""))
        .unwrap_or_else(|| panic!("no projection input:\n{stdout}"));
    assert!(
        projection.contains(": Tensor<f32, [8, 16]> ="),
        "{projection}"
    );
    let id = projection.split(':').next().unwrap().trim();
    let matmul = stdout
        .lines()
        .find(|l| l.contains("= MatMul ") && l.ends_with(&format!(", {id}")))
        .unwrap_or_else(|| panic!("no MatMul reading {id}:\n{stdout}"));
    assert!(matmul.contains(": Tensor<f32, [256, 16]> ="), "{matmul}");
    let projected = matmul.split(':').next().unwrap().trim();
    let columns = stdout
        .lines()
        .find(|l| l.contains("= SigregColumns "))
        .unwrap_or_else(|| panic!("no SigregColumns:\n{stdout}"));
    assert!(columns.ends_with(&format!(" {projected}")), "{columns}");
    assert!(columns.contains(": Tensor<f32, [16]> ="), "{columns}");
    let columns_id = columns.split(':').next().unwrap().trim();
    assert!(
        stdout
            .lines()
            .any(|l| l.contains("= ReduceMean ") && l.ends_with(&format!(" {columns_id}"))),
        "no ReduceMean of {columns_id}:\n{stdout}"
    );

    let (output, _) = modelspec("ir", "sigreg_train.mspec", &["--stage", "training"]);
    let stdout = text(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    let grad = stdout
        .lines()
        .find(|l| l.contains("= SigregColumnsGrad "))
        .unwrap_or_else(|| panic!("no SigregColumnsGrad:\n{stdout}"));
    assert!(grad.contains(&format!(" {projected}, ")), "{grad}");
}

/// `compile --backend vulkan` writes an artifact whose kernels hold `SigregColumns` and
/// `SigregColumnsGrad` (story:jepa-sigreg-vulkan, which replaced the `E-TARGET-011` of
/// story:jepa-sigreg-cpu); `compile --backend cpu` writes one too, the projection taking a buffer
/// of its own.
#[test]
fn compile_sigreg_writes_an_artifact_for_vulkan_and_for_cpu() {
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("check_sigreg");
    let vulkan = out.join("vulkan");
    let profile = repo_root().join("crates/modelspec-runtime/tests/fixtures/rtx3090.device.json");
    let (output, _) = modelspec(
        "compile",
        "sigreg_train.mspec",
        &[
            "--backend",
            "vulkan",
            "--device-profile",
            profile.to_str().unwrap(),
            "--output",
            vulkan.to_str().unwrap(),
        ],
    );
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    let artifact =
        std::fs::read_to_string(vulkan.join("artifact.json")).expect("the Vulkan artifact");
    for op in ["\"SigregColumns\"", "\"SigregColumnsGrad\""] {
        assert!(artifact.contains(op), "no kernel {op} in the artifact");
    }

    let cpu = out.join("cpu");
    let (output, _) = modelspec(
        "compile",
        "sigreg_train.mspec",
        &["--backend", "cpu", "--output", cpu.to_str().unwrap()],
    );
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    assert!(cpu.join("artifact.json").is_file());
}
