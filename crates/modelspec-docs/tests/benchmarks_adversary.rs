//! Adversarial cases for `benchmarks/runs.yaml` (story:benchmark-history), driven through the
//! `modelspec-docs` binary the way `task docs-generate` and `task docs-check` run it: each case
//! copies the repository into its own directory under `CARGO_TARGET_TMPDIR`, changes one thing in
//! the copy and runs `generate` there.

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

/// A fresh copy of the repository named `case`.
fn sandbox(case: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("benchmarks-adversary")
        .join(case);
    let _ = fs::remove_dir_all(&root);
    copy(&repository(), &root);
    root
}

/// The copy's `benchmarks/runs.yaml` with `from` replaced once by `to`; `from` must occur.
fn edit_runs(root: &Path, from: &str, to: &str) {
    let path = root.join("benchmarks/runs.yaml");
    let text = fs::read_to_string(&path).unwrap();
    assert!(text.contains(from), "fixture text not found: {from}");
    fs::write(&path, text.replacen(from, to, 1)).unwrap();
}

struct Outcome {
    code: Option<i32>,
    stderr: String,
}

fn generate(root: &Path, check: bool) -> Outcome {
    let mut command = Command::new(env!("CARGO_BIN_EXE_modelspec-docs"));
    command.arg("generate").arg("--root").arg(root);
    if check {
        command.arg("--check");
    }
    let output = command.output().unwrap();
    Outcome {
        code: output.status.code(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// `generate` refused the file with a message, not a panic, and the message names `prefix`.
fn assert_refused(outcome: &Outcome, prefix: &str) {
    assert!(
        !outcome.stderr.contains("panicked"),
        "generate panicked instead of refusing:\n{}",
        outcome.stderr
    );
    assert_eq!(outcome.code, Some(1), "{}", outcome.stderr);
    assert!(
        outcome.stderr.contains(prefix),
        "expected `{prefix}` in:\n{}",
        outcome.stderr
    );
}

#[test]
fn adversary_zero_repeats_listed_as_empty_is_refused_not_a_panic() {
    let root = sandbox("zero-repeats-empty");
    edit_runs(
        &root,
        "    repeat_count: 3\n    repeats_us_per_step:\n      - 19355.555555555555\n      - 19900\n      - 20700\n",
        "    repeat_count: 0\n    repeats_us_per_step: []\n",
    );
    assert_refused(
        &generate(&root, false),
        "benchmarks/runs.yaml:11: runs[0] (mnist-0.1.1-modelspec-cpu) field `repeat_count`",
    );
}

#[test]
fn adversary_zero_repeat_count_breaks_the_domain_invariant_and_is_refused() {
    let root = sandbox("zero-repeat-count");
    edit_runs(
        &root,
        "    timed_steps: 144000\n    repeat_count: 5\n    repeats_us_per_step: null\n    median_us_per_step: 40.225\n",
        "    timed_steps: 144000\n    repeat_count: 0\n    repeats_us_per_step: null\n    median_us_per_step: 40.225\n",
    );
    assert_refused(
        &generate(&root, false),
        "benchmarks/runs.yaml:181: runs[11] (mnist-0.4.0-modelspec-vulkan) field `repeat_count`",
    );
}

#[test]
fn adversary_flop_per_step_without_flop_rule_is_refused() {
    let root = sandbox("flop-without-rule");
    edit_runs(&root, "    flop_rule: AllSites\n", "");
    assert_refused(
        &generate(&root, false),
        "runs[14] (mnist-2026-10-23-modelspec-vulkan-profiled) field `flop_",
    );
}

#[test]
fn adversary_zero_threads_is_refused() {
    let root = sandbox("zero-threads");
    edit_runs(
        &root,
        "    side: ModelSpecCpu\n    threads: 1\n",
        "    side: ModelSpecCpu\n    threads: 0\n",
    );
    assert_refused(
        &generate(&root, false),
        "benchmarks/runs.yaml:6: runs[0] (mnist-0.1.1-modelspec-cpu) field `threads`",
    );
}

#[test]
fn adversary_a_negative_batch_is_refused_with_its_line_and_field() {
    let root = sandbox("negative-batch");
    edit_runs(
        &root,
        "    batch: 100\n    hidden: 64\n",
        "    batch: -1\n    hidden: 64\n",
    );
    assert_refused(
        &generate(&root, false),
        "benchmarks/runs.yaml:244: runs[15] (sweep-b100-h64-modelspec-vulkan) field `batch`",
    );
}

#[test]
fn adversary_a_whole_float_where_the_schema_takes_an_integer_is_refused_with_its_field() {
    let root = sandbox("float-threads");
    edit_runs(
        &root,
        "    side: ModelSpecCpu\n    threads: 1\n",
        "    side: ModelSpecCpu\n    threads: 1.0\n",
    );
    // The schema's `"type": "integer"` admits 1.0, so the refusal must come from the reader and
    // still name the field.
    assert_refused(
        &generate(&root, false),
        "benchmarks/runs.yaml:6: runs[0] (mnist-0.1.1-modelspec-cpu) field `threads`",
    );
}

#[test]
fn adversary_a_key_written_twice_in_one_entry_is_refused() {
    let root = sandbox("duplicate-key");
    // An entry with no repeats, so the median check cannot catch which value won.
    edit_runs(
        &root,
        "    repeats_us_per_step: null\n    median_us_per_step: 40.225\n",
        "    repeats_us_per_step: null\n    median_us_per_step: 40.225\n    median_us_per_step: 4.0225\n",
    );
    let outcome = generate(&root, false);
    assert!(!outcome.stderr.contains("panicked"), "{}", outcome.stderr);
    assert_eq!(
        outcome.code,
        Some(1),
        "a duplicate `median_us_per_step` was accepted; the page now shows {}",
        fs::read_to_string(root.join("website/docs/benchmarks.md"))
            .unwrap()
            .lines()
            .find(|line| line.starts_with("| `mnist-0.4.0-modelspec-vulkan`"))
            .unwrap_or_default()
    );
    assert!(
        outcome.stderr.contains("median_us_per_step"),
        "{}",
        outcome.stderr
    );
}

#[test]
fn adversary_a_commit_with_a_multibyte_character_does_not_panic() {
    let root = sandbox("multibyte-commit");
    edit_runs(
        &root,
        "    commit: 16d3dd2f576450fcc71b07cfea2d7c2b01d1fb4a\n",
        "    commit: \"16d3dd…\"\n",
    );
    let outcome = generate(&root, false);
    assert!(
        !outcome.stderr.contains("panicked"),
        "a commit the schema accepts made generate panic:\n{}",
        outcome.stderr
    );
}

/// A drift case: its name, what it plants in the copy, and what `--check` must report.
type DriftCase<'a> = (&'a str, &'a dyn Fn(&Path), &'a str);

#[test]
fn adversary_check_fails_on_a_stale_chart_a_stale_page_a_missing_chart_and_a_stray_chart() {
    let images = "website/static/img/benchmarks";
    let cases: [DriftCase; 4] = [
        (
            "stale-chart",
            &|root| {
                let path = root.join(images).join("mnist-784-64-10.svg");
                let text = fs::read_to_string(&path).unwrap();
                fs::write(&path, text.replacen("#1d4ed8", "#1d4ed9", 1)).unwrap();
            },
            "stale or edited mnist-784-64-10.svg",
        ),
        (
            "stale-page",
            &|root| {
                let path = root.join("website/docs/benchmarks.md");
                let text = fs::read_to_string(&path).unwrap();
                assert!(text.contains("27.14"));
                fs::write(&path, text.replacen("27.14", "27.15", 1)).unwrap();
            },
            "stale or edited benchmarks.md",
        ),
        (
            "missing-chart",
            &|root| fs::remove_file(root.join(images).join("mlp-784-h-10-batch-100.svg")).unwrap(),
            "missing mlp-784-h-10-batch-100.svg",
        ),
        (
            "stray-chart",
            &|root| fs::write(root.join(images).join("old.svg"), "<svg/>").unwrap(),
            "not generated old.svg",
        ),
    ];
    for (case, plant, expected) in cases {
        let root = sandbox(case);
        assert_eq!(
            generate(&root, true).code,
            Some(0),
            "{case}: the copy is not current"
        );
        plant(&root);
        let outcome = generate(&root, true);
        assert_eq!(outcome.code, Some(1), "{case}: {}", outcome.stderr);
        assert!(
            outcome.stderr.contains(expected),
            "{case}: {}",
            outcome.stderr
        );
    }
    // A changed figure in runs.yaml makes the committed page stale too.
    let root = sandbox("stale-figure");
    edit_runs(
        &root,
        "    median_us_per_step: 27.14\n",
        "    median_us_per_step: 26.14\n",
    );
    let outcome = generate(&root, true);
    assert_eq!(outcome.code, Some(1), "{}", outcome.stderr);
    assert!(outcome.stderr.contains("has drifted"), "{}", outcome.stderr);
}
