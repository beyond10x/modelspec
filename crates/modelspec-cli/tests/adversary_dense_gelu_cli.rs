//! Adversary pass 1 on story:bert-dense-gelu, through the `modelspec` binary: the shape rules of
//! `add_rows` at the call, `E-TARGET-011` for every command that needs a Vulkan kernel, the CPU
//! artifact and inference path, what the training fixture exercises, and the diagnostics page.
//!
//! No test here creates a Vulkan instance: every run sets `MODELSPEC_VULKAN_LIBRARY` to a path
//! that does not exist.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const NO_LOADER: &str = "/nonexistent/modelspec-test/libvulkan-missing.so.1";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
        .display()
        .to_string()
}

fn profile() -> String {
    root()
        .join("crates/modelspec-runtime/tests/fixtures/rtx3090.device.json")
        .display()
        .to_string()
}

/// A fresh path under the test temp dir that does not exist yet.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary_dense_gelu_cli")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("the old test dir is removed");
    }
    std::fs::create_dir_all(dir.parent().unwrap()).expect("the temp dir is writable");
    dir
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .env("MODELSPEC_VULKAN_LIBRARY", NO_LOADER)
        .output()
        .expect("modelspec runs")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// `check` exits 1 with one diagnostic `code` at `<fixture>:<line>:<column>`.
fn check_refuses(name: &str, code: &str, at: &str) {
    let output = modelspec(&["check", &fixture(name)]);
    let text = stderr(&output);
    assert_eq!(output.status.code(), Some(1), "{name}:\n{text}");
    let heads: Vec<&str> = text.lines().filter(|l| !l.starts_with("  ")).collect();
    assert_eq!(heads.len(), 1, "{name}: one diagnostic:\n{text}");
    assert!(
        heads[0].starts_with(&format!("{code}: ")),
        "{name}:\n{text}"
    );
    assert!(
        text.contains(&format!("{name}:{at}")),
        "{name}: not at {at}:\n{text}"
    );
}

/// Attack 3: a `b` as long as the rows (`N`), not the columns (`H`), of `x: [3, 4]` is
/// `E-TYPE-003` at the call: the check does not confuse the axes.
#[test]
fn adversary_a_bias_of_length_n_not_h_is_e_type_003_at_the_call() {
    check_refuses(
        "adversary_dense_bias_along_rows.mspec",
        "E-TYPE-003",
        "5:16",
    );
}

/// Attack 3: `N = H` checks clean.
#[test]
fn adversary_a_square_x_with_its_bias_checks_clean() {
    let output = modelspec(&["check", &fixture("adversary_dense_square.mspec")]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    assert_eq!(stderr(&output), "");
}

/// Attack 3: a `b` of rank 2 `[1, H]` is refused (not broadcast) with `E-TYPE-004` at the
/// call, and an `x` of rank 3 `[B, S, H]` likewise.
#[test]
fn adversary_a_rank_2_bias_and_a_rank_3_x_are_e_type_004_at_the_call() {
    check_refuses("adversary_dense_bias_rank2.mspec", "E-TYPE-004", "5:16");
    check_refuses("adversary_dense_rank3_x.mspec", "E-TYPE-004", "5:16");
}

/// The four commands that need a Vulkan kernel, for `fixture`.
fn vulkan_commands(fixture: &str, tag: &str) -> Vec<Vec<String>> {
    let out = scratch(&format!("{tag}-vulkan")).join("artifact");
    let missing = scratch(&format!("{tag}-missing")).join("no-such-checkpoint");
    let s = |v: &str| v.to_owned();
    vec![
        vec![s("ir"), s("--stage"), s("kernel"), s(fixture)],
        vec![
            s("compile"),
            s(fixture),
            s("--backend"),
            s("vulkan"),
            s("--device-profile"),
            profile(),
            s("--output"),
            out.display().to_string(),
        ],
        vec![s("train"), s(fixture), s("--backend"), s("vulkan")],
        vec![
            s("infer"),
            s(fixture),
            s("--backend"),
            s("vulkan"),
            s("--checkpoint"),
            missing.display().to_string(),
        ],
    ]
}

fn refuses_with_no_kernel(name: &str, operator: &str, at: &str) {
    let fixture = fixture(name);
    for args in vulkan_commands(&fixture, operator) {
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        let output = modelspec(&args);
        let text = stderr(&output);
        assert_eq!(output.status.code(), Some(1), "{args:?}:\n{text}");
        assert!(
            text.contains(&format!(
                "E-TARGET-011: operator `{operator}` has no Vulkan kernel yet"
            )),
            "{args:?}:\n{text}"
        );
        assert!(
            text.contains(&format!("{name}:{at}")),
            "{args:?}: not at {at}:\n{text}"
        );
        assert!(!text.contains("E-INTERNAL"), "{args:?}:\n{text}");
    }
}

/// Attack 6: on `dense_gelu_train.mspec` every Vulkan command answers `E-TARGET-011` naming
/// `add_rows`, the first operator without a kernel, at its call (line 6, column 21).
#[test]
fn adversary_every_vulkan_command_refuses_add_rows_at_its_call() {
    refuses_with_no_kernel("dense_gelu_train.mspec", "add_rows", "6:21");
}

/// Attack 6: a program whose only new operator is `gelu` is refused naming `gelu` at its call.
#[test]
fn adversary_every_vulkan_command_refuses_gelu_at_its_call() {
    refuses_with_no_kernel("adversary_dense_gelu_only.mspec", "gelu", "6:16");
}

/// The `step <k> loss <v>` lines of a `train` run.
fn losses(stdout: &str) -> Vec<(u64, f32)> {
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix("step "))
        .map(|rest| {
            let (step, loss) = rest.split_once(" loss ").expect("`step <k> loss <v>`");
            (step.parse().unwrap(), loss.parse().unwrap())
        })
        .collect()
}

/// Attack 6 and 8: `compile --backend cpu` writes `artifact.json`; `train --backend cpu` over
/// 60 steps lowers the loss and writes a checkpoint; `infer --backend cpu` runs it.
#[test]
fn adversary_the_dense_fixture_compiles_trains_down_and_infers_on_the_cpu() {
    let fixture = fixture("dense_gelu_train.mspec");
    let artifact = scratch("cpu-artifact");
    let output = modelspec(&[
        "compile",
        &fixture,
        "--backend",
        "cpu",
        "--output",
        artifact.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    assert!(artifact.join("artifact.json").is_file());

    let checkpoint = scratch("cpu-checkpoint").with_extension("msckpt");
    let output = modelspec(&[
        "train",
        &fixture,
        "--backend",
        "cpu",
        "--steps",
        "60",
        "--checkpoint",
        checkpoint.to_str().unwrap(),
    ]);
    let text = stdout(&output);
    assert_eq!(output.status.code(), Some(0), "{text}\n{}", stderr(&output));
    let losses = losses(&text);
    let (first, last) = (losses.first().unwrap().1, losses.last().unwrap().1);
    assert!(last < first, "loss {first} -> {last}:\n{text}");

    let output = modelspec(&[
        "infer",
        &fixture,
        "--backend",
        "cpu",
        "--checkpoint",
        checkpoint.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    assert!(!stdout(&output).is_empty());
}

/// Attack 8, finding F2: the training fixture feeds `b` as the input, so its training step has
/// no `SumRows`; no `train` run reaches the bias gradient.
#[test]
fn adversary_the_training_fixture_computes_the_bias_gradient() {
    let output = modelspec(&[
        "ir",
        "--stage",
        "training",
        &fixture("dense_gelu_train.mspec"),
    ]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("SumRows"), "{text}");
}

/// Attack 9, finding F1: the diagnostics page names the operators each code covers; `add_rows`
/// now raises `E-TYPE-003` and `E-TYPE-004`, and their rows do not name it.
#[test]
fn adversary_the_diagnostics_page_names_add_rows_for_e_type_003_and_004() {
    let page = std::fs::read_to_string(root().join("website/docs/reference/diagnostics.md"))
        .expect("the page reads");
    for code in ["E-TYPE-003", "E-TYPE-004"] {
        let row = page
            .lines()
            .find(|l| l.contains(&format!("`{code}` |")))
            .unwrap_or_else(|| panic!("no row for {code}"));
        assert!(row.contains("add_rows"), "{code}: {row}");
    }
}
