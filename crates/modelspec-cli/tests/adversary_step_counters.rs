//! Adversarial cases for the run counters of `train --backend vulkan`
//! (`story:vulkan-step-counters`): `barriers per step`, `device allocations` and
//! `peak device bytes`, and their `profile` lines.
#![cfg(feature = "vulkan")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const LINEAR_REGRESSION: &str = "examples/linear_regression.mspec";
const GUIDE: &str = "website/docs/guides/inspect-a-run.md";

/// The `profile` counter lines whose values do not depend on timing.
const COUNTERS: [&str; 8] = [
    "dispatches",
    "submits",
    "host_waits",
    "barriers_per_step",
    "bytes_uploaded",
    "bytes_downloaded",
    "device_allocations",
    "peak_device_bytes",
];

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

fn stderr_of(args: &[&str]) -> String {
    let output = modelspec(args);
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert_eq!(output.status.code(), Some(0), "{args:?}: {stderr}");
    stderr
}

/// The `profile <name> <value>` counter lines of `text`, by name.
fn profile_counters(text: &str) -> BTreeMap<String, String> {
    text.lines()
        .filter_map(|line| line.strip_prefix("profile "))
        .filter_map(|rest| rest.split_once(' '))
        .filter(|(name, _)| COUNTERS.contains(name))
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect()
}

/// The `<name>: <value>` `--stats` lines of `text`, by name.
fn stats_lines(text: &str) -> BTreeMap<String, String> {
    text.lines()
        .filter_map(|line| line.split_once(": "))
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect()
}

/// The guide's section 5 quotes a `--profile` run and says "None of the counters depends on
/// timing". Run the command the guide names and compare every counter that does not depend on
/// timing with the block it quotes.
#[test]
#[ignore = "needs a Vulkan device"]
fn the_guides_counter_block_is_what_its_command_prints() {
    let guide = std::fs::read_to_string(repo_root().join(GUIDE)).expect("the guide reads");
    let section = guide
        .split("## 5. The time of every dispatch")
        .nth(1)
        .expect("the guide has section 5");
    let command = section
        .split("```shell\n")
        .nth(1)
        .and_then(|rest| rest.split("\n```").next())
        .expect("section 5 quotes a command");
    let quoted = section
        .split("```text\n")
        .nth(1)
        .and_then(|rest| rest.split("\n```").next())
        .expect("section 5 quotes the output");
    let args: Vec<&str> = command
        .split_whitespace()
        .skip_while(|word| *word == "modelspec")
        .collect();
    assert!(args.contains(&"--profile"), "{command}");

    let printed = profile_counters(&stderr_of(&args));
    let quoted = profile_counters(quoted);
    assert_eq!(quoted.len(), COUNTERS.len(), "the guide quotes {quoted:?}");
    assert_eq!(
        printed, quoted,
        "`{command}` prints these counters; the guide quotes the right-hand ones"
    );
}

/// `--stats` and `--profile` print the same three counters, in one run and in two.
#[test]
#[ignore = "needs a Vulkan device"]
fn stats_and_profile_print_the_same_three_counters() {
    let pairs = [
        ("barriers per step", "barriers_per_step"),
        ("device allocations", "device_allocations"),
        ("peak device bytes", "peak_device_bytes"),
    ];
    let both = stderr_of(&[
        "train",
        LINEAR_REGRESSION,
        "--backend",
        "vulkan",
        "--stats",
        "--profile",
    ]);
    let stats_only = stderr_of(&["train", LINEAR_REGRESSION, "--backend", "vulkan", "--stats"]);
    let profile_only = stderr_of(&[
        "train",
        LINEAR_REGRESSION,
        "--backend",
        "vulkan",
        "--profile",
    ]);
    let (both_stats, both_profile) = (stats_lines(&both), profile_counters(&both));
    let (stats_only, profile_only) = (stats_lines(&stats_only), profile_counters(&profile_only));
    for (stat, counter) in pairs {
        let value = stats_only.get(stat).expect(stat);
        assert_eq!(both_stats.get(stat), Some(value), "{stat}");
        assert_eq!(both_profile.get(counter), Some(value), "{counter}");
        assert_eq!(profile_only.get(counter), Some(value), "{counter}");
    }
    assert_eq!(stats_only["barriers per step"], "7");
    assert_eq!(stats_only["device allocations"], "11");
}

/// A run that writes checkpoints on the way, and a run resumed from one, record the same step,
/// so the same barriers per step, and make the same buffers, so the same allocations.
#[test]
#[ignore = "needs a Vulkan device"]
fn checkpoints_and_resume_keep_the_per_step_and_allocation_counts() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("adversary-step-counters");
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    let checkpoint = dir.join("lr.msckpt");
    let checkpoint = checkpoint.to_str().expect("a UTF-8 path");
    let plain = stats_lines(&stderr_of(&[
        "train",
        LINEAR_REGRESSION,
        "--backend",
        "vulkan",
        "--stats",
    ]));
    let segmented = stats_lines(&stderr_of(&[
        "train",
        LINEAR_REGRESSION,
        "--backend",
        "vulkan",
        "--stats",
        "--checkpoint",
        checkpoint,
        "--checkpoint-every",
        "100",
    ]));
    for name in [
        "barriers per step",
        "device allocations",
        "peak device bytes",
    ] {
        assert_eq!(segmented.get(name), plain.get(name), "{name}");
    }
    assert_eq!(segmented["host waits"], "5");
    let resumed = stats_lines(&stderr_of(&[
        "train",
        LINEAR_REGRESSION,
        "--backend",
        "vulkan",
        "--stats",
        "--resume",
        checkpoint,
        "--steps",
        "100",
    ]));
    for name in ["barriers per step", "device allocations"] {
        assert_eq!(resumed.get(name), plain.get(name), "{name}");
    }
}
