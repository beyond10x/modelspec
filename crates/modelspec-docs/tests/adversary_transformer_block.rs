//! Adversary pass 1 of story:transformer-block-benchmark: `transformer-block` entries of
//! `benchmarks/runs.yaml` that the unit's own fixtures do not write, driven through the
//! `modelspec-docs` binary the way `task docs-generate` runs it. Each case copies the repository
//! into its own directory under `CARGO_TARGET_TMPDIR`, appends entries to the copy's
//! `runs.yaml` and runs `generate` there.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn copy(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        if matches!(name.to_str(), Some("target" | ".git" | "node_modules")) {
            continue;
        }
        let kind = entry.file_type().unwrap();
        let target = to.join(&name);
        if kind.is_dir() {
            copy(&entry.path(), &target);
        } else if kind.is_file() {
            fs::copy(entry.path(), &target).unwrap();
        }
    }
}

/// A copy of the repository named `case` whose `runs.yaml` ends with `entries`.
fn sandbox(case: &str, entries: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary-transformer-block")
        .join(case);
    let _ = fs::remove_dir_all(&root);
    copy(&repository(), &root);
    let runs = root.join("benchmarks/runs.yaml");
    let mut text = fs::read_to_string(&runs).unwrap();
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(entries);
    fs::write(&runs, text).unwrap();
    root
}

struct Outcome {
    code: Option<i32>,
    stderr: String,
}

fn generate(root: &Path) -> Outcome {
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec-docs"))
        .arg("generate")
        .arg("--root")
        .arg(root)
        .output()
        .unwrap();
    Outcome {
        code: output.status.code(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// The `## Transformer block` section of the copy's generated page.
fn block_section(root: &Path) -> String {
    let page = fs::read_to_string(root.join("website/docs/benchmarks.md")).unwrap();
    let start = page
        .find("\n## Transformer block\n")
        .unwrap_or_else(|| panic!("no Transformer block section:\n{page}"));
    let end = page[start + 1..]
        .find("\n## ")
        .map_or(page.len(), |at| start + 1 + at);
    page[start..end].to_owned()
}

/// A `transformer-block` entry of side `side` with the optional fields given as YAML lines.
fn entry(id: &str, side: &str, fields: &str, median: f64) -> String {
    format!(
        "  - run_id: {id}\n    date: \"2026-10-27\"\n    workload: transformer-block\n    side: {side}\n    mode: Timed\n{fields}    repeat_count: 1\n    repeats_us_per_step: null\n    median_us_per_step: {median}\n    source: verification-report:fixture\n    state: Recorded\n"
    )
}

/// ADV-TB-2. A `transformer-block` run that gives its batch and its `MatmulOnly` count but no
/// `tokens_per_step` is accepted, and the block's section is then the sweep's layout: no
/// tokens-per-second column, although the workload's tokens are 128·B by definition (D81: "the
/// block's table a tokens-per-second column"; the unit's report: "`transformer-block` runs must
/// state tokens = 128·B"). Either the entry is refused at its field or the section shows
/// tokens per second.
#[test]
fn adv_tb_2_a_block_run_without_tokens_per_step_is_refused_or_still_shows_tokens_per_second() {
    let fields = "    batch: 8\n    flop_per_step: 2717908992\n    flop_rule: MatmulOnly\n";
    let root = sandbox(
        "without-tokens",
        &(entry("block-ms-8", "ModelSpecVulkan", fields, 1024.0)
            + &entry("block-pt-8", "PyTorchCudaEager", fields, 4096.0)),
    );
    let outcome = generate(&root);
    if outcome.code == Some(1) {
        assert!(
            outcome.stderr.contains("field `tokens_per_step`"),
            "{}",
            outcome.stderr
        );
        return;
    }
    assert_eq!(outcome.code, Some(0), "{}", outcome.stderr);
    let section = block_section(&root);
    assert!(section.contains("Tokens per second"), "{section}");
    assert!(section.contains("1,000,000"), "{section}");
}

/// ADV-TB-3. The 128·B rule of `tokens_per_step` is checked only when the entry also gives its
/// batch, so `tokens_per_step: 1000`, which is no whole number of sequences of 128, passes on a
/// `transformer-block` run without one.
#[test]
fn adv_tb_3_block_tokens_that_are_no_whole_number_of_sequences_are_refused_without_a_batch() {
    let root = sandbox(
        "tokens-without-batch",
        &entry(
            "block-ms-x",
            "ModelSpecVulkan",
            "    tokens_per_step: 1000\n",
            1000.0,
        ),
    );
    let outcome = generate(&root);
    assert_eq!(outcome.code, Some(1), "accepted:\n{}", outcome.stderr);
    assert!(
        outcome.stderr.contains("field `tokens_per_step`"),
        "{}",
        outcome.stderr
    );
}

/// The section's tokens per second is `tokens_per_step · 10⁶ / µs` and its GFLOP/s
/// `flop_per_step / µs / 1000` at a median that is not a round number, and the block leaves
/// "Not measured".
#[test]
fn block_section_figures_are_recomputed_from_the_entry() {
    let fields = "    batch: 32\n    tokens_per_step: 4096\n    flop_per_step: 10871635968\n    flop_rule: MatmulOnly\n";
    let root = sandbox(
        "figures",
        &entry("block-ms-32", "ModelSpecVulkan", fields, 3000.0),
    );
    let outcome = generate(&root);
    assert_eq!(outcome.code, Some(0), "{}", outcome.stderr);
    let section = block_section(&root);
    let row = section
        .lines()
        .find(|line| line.starts_with("| 32 | 4,096 | ModelSpec Vulkan |"))
        .unwrap_or_else(|| panic!("{section}"));
    let tokens = 4096.0 * 1e6 / 3000.0;
    let gflops = 10_871_635_968.0 / 3000.0 / 1000.0;
    assert!(row.contains("| 1,365,333.333 |"), "{tokens} {row}");
    assert!(row.contains("| 3,623.9 |"), "{gflops} {row}");
    let page = fs::read_to_string(root.join("website/docs/benchmarks.md")).unwrap();
    let not_measured = &page[page.find("## Not measured").unwrap()..];
    assert!(
        !not_measured.contains("Transformer block"),
        "{not_measured}"
    );
}
