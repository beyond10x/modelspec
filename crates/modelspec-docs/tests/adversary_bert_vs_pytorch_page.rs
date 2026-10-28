//! Adversarial cases for story:bert-tiny-vs-pytorch-cuda, the page section: a `bert-tiny-pretrain`
//! row appended to a copy of `benchmarks/runs.yaml` and the `modelspec-docs` binary run on the
//! copy, as `task docs-generate` runs it. The story's acceptance asks for GFLOP/s by "the same
//! FLOP count for all three" sides and for the crossing step "read at that 100-step resolution".

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh copy of the repository named `case`, without build output.
fn sandbox(case: &str) -> PathBuf {
    fn copy(from: &Path, to: &Path) {
        fs::create_dir_all(to).unwrap();
        for entry in fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name();
            if matches!(
                name.to_str(),
                Some("target" | ".git" | "node_modules" | "build" | ".docusaurus")
            ) {
                continue;
            }
            let kind = entry.file_type().unwrap();
            if kind.is_dir() {
                copy(&entry.path(), &to.join(&name));
            } else if kind.is_file() {
                fs::copy(entry.path(), to.join(&name)).unwrap();
            }
        }
    }
    let root = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary-bert-vs-pytorch-page")
        .join(case);
    let _ = fs::remove_dir_all(&root);
    copy(&repository(), &root);
    root
}

fn append_runs(root: &Path, entries: &str) {
    let path = root.join("benchmarks/runs.yaml");
    let mut text = fs::read_to_string(&path).unwrap();
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(entries);
    fs::write(&path, text).unwrap();
}

/// A ModelSpec Vulkan `bert-tiny-pretrain` row with every field the unit's own fixture
/// (`bert_entry` in `benchmarks.rs`) gives, its count, rule and steps to target as given.
fn bert_row(id: &str, flop: &str, rule: &str, steps_to_target: u64) -> String {
    format!(
        "  - run_id: {id}\n    date: \"2026-10-28\"\n    workload: bert-tiny-pretrain\n    side: ModelSpecVulkan\n    modelspec_version: 0.22.0\n    commit: 0123456789abcdef\n    mode: Timed\n    batch: 32\n    tokens_per_step: 4096\n    flop_per_step: {flop}\n    flop_rule: {rule}\n    target_loss: 6.926537926202018\n    steps_to_target: {steps_to_target}\n    seconds_to_target: 28.5\n    train_seconds: 84.25\n    eval_every: 100\n    final_heldout_loss: 6.8\n    repeat_count: 3\n    repeats_us_per_step: null\n    median_us_per_step: 8192.0\n    source: a fixture of the adversary\n    state: Recorded\n"
    )
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

/// story:bert-tiny-vs-pytorch-cuda, the control of ADV-BP-4 and ADV-BP-5: the same row counted by
/// `MatmulOnly` at 1,170,210,816·32 with a crossing step on the 100-step grid is accepted, so the
/// two cases below are refused for their count and their step and for nothing else.
#[test]
fn adversary_the_control_bert_row_is_accepted() {
    let root = sandbox("control");
    append_runs(
        &root,
        &bert_row("bert-ms-control", "37446746112", "MatmulOnly", 3400),
    );
    let outcome = generate(&root);
    assert_eq!(outcome.code, Some(0), "{}", outcome.stderr);
}

/// story:bert-tiny-vs-pytorch-cuda, finding ADV-BP-4. The story's acceptance gives every side
/// GFLOP/s by "the same FLOP count for all three", and the brief has `benchmarks.rs` check a
/// `bert-tiny-pretrain` count against `1,170,210,816·B`. That check runs only when the row says
/// `MatmulOnly` (`semantic_problems`, `if let (Some(flop), Some("MatmulOnly"))`), so a ModelSpec
/// row counted by `AllSites` with the stepwise `profile step_total` of 39,400,695,809 FLOP (the
/// implementing session's own 10-step profile) is published beside PyTorch's 37,446,746,112, two
/// counts and two GFLOP/s scales in one section. The row must be refused at its count or rule.
#[test]
fn adversary_adv_bp_4_a_bert_row_counted_by_all_sites_is_refused() {
    let root = sandbox("all-sites");
    append_runs(
        &root,
        &bert_row("bert-ms-all-sites", "39400695809", "AllSites", 3400),
    );
    let outcome = generate(&root);
    assert_ne!(
        outcome.code,
        Some(0),
        "a bert-tiny-pretrain row counted by AllSites, 39,400,695,809 FLOP and not \
         1,170,210,816·32 = 37,446,746,112, was accepted"
    );
    assert!(
        outcome.stderr.contains("flop_rule") || outcome.stderr.contains("flop_per_step"),
        "refused, but not at the count or the rule: {}",
        outcome.stderr
    );
}

/// story:bert-tiny-vs-pytorch-cuda, finding ADV-BP-6. The unit's new `BenchmarkRun.target_loss`
/// summary (spec/domains/benchmark.yaml, and every JSON Schema generated from it) calls
/// 6.926537926202018 "the `bert-tiny-pretrain` reference run's final held-out loss". That number
/// is the held-out unigram baseline of `packed.json` (`heldout_unigram_baseline_nats`), the
/// story's target; the reference run's final held-out loss is 6.7674994
/// (website/docs/runs/bert-tiny-gpu-pretraining.md, "ending at a held-out loss of 6.7674994 nats
/// against a unigram baseline of 6.9265379"). A reader who records the final loss as the target
/// times each side to a loss 0.159 nats lower, about step 10,000 instead of about step 1,500.
#[test]
fn adversary_adv_bp_6_the_target_loss_summary_names_the_baseline_not_a_final_loss() {
    let text = fs::read_to_string(repository().join("spec/domains/benchmark.yaml")).unwrap();
    let domain: serde_yaml_ng::Value = serde_yaml_ng::from_str(&text).unwrap();
    let summary = domain["entities"]
        .as_sequence()
        .unwrap()
        .iter()
        .flat_map(|entity| entity["fields"].as_sequence().into_iter().flatten())
        .find(|field| field["name"].as_str() == Some("target_loss"))
        .and_then(|field| field["summary"].as_str())
        .expect("BenchmarkRun.target_loss has a summary")
        .to_owned();
    let reference =
        fs::read_to_string(repository().join("website/docs/runs/bert-tiny-gpu-pretraining.md"))
            .unwrap();
    assert!(
        reference.contains(
            "ending at a held-out loss of 6.7674994 nats against a unigram baseline of 6.9265379"
        ),
        "the reference run's record changed"
    );
    assert!(
        !summary.contains("final held-out loss `6.926537926202018`"),
        "target_loss calls the unigram baseline 6.926537926202018 the reference run's final \
         held-out loss (that is 6.7674994): {summary}"
    );
}

/// story:bert-tiny-vs-pytorch-cuda, finding ADV-BP-5. A row's steps_to_target is "the first
/// evaluated step whose held-out loss is at most target_loss" (spec/domains/benchmark.yaml), and
/// the acceptance reads the crossing step "at that 100-step resolution"; a row that says
/// `eval_every: 100` and `steps_to_target: 3450` states a step that was never evaluated, and the
/// page publishes it. `bert_problems` checks seconds_to_target against train_seconds but not
/// steps_to_target against eval_every.
#[test]
fn adversary_adv_bp_5_a_bert_row_whose_steps_to_target_was_never_evaluated_is_refused() {
    let root = sandbox("off-grid");
    append_runs(
        &root,
        &bert_row("bert-ms-off-grid", "37446746112", "MatmulOnly", 3450),
    );
    let outcome = generate(&root);
    assert_ne!(
        outcome.code,
        Some(0),
        "steps_to_target 3450 with eval_every 100, a step no evaluation ran at, was accepted"
    );
    assert!(
        outcome.stderr.contains("steps_to_target"),
        "refused, but not at steps_to_target: {}",
        outcome.stderr
    );
}
