//! `modelspec train --backend vulkan --profile` prints the pipeline executable statistics of
//! every kernel of the plan (`story:matmul-block-kernel-throughput`, acceptance
//! `pipeline-statistics`; DESIGN.md § 8).
//!
//! Ignored by default because it creates a Vulkan instance, which `task ci` never does; run it
//! with `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-cli --test train_pipeline_statistics -- --ignored`.
//! It needs a device that offers `VK_KHR_pipeline_executable_properties`, as the RTX 3090 does.
#![cfg(feature = "vulkan")]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const MNIST_MLP: &str = "examples/mnist_mlp.mspec";

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
    String::from_utf8_lossy(bytes).into_owned()
}

/// The `--profile` row of the generated CLI reference, from the doc comment of the flag, names
/// the `profile pipeline` lines and the `not reported` line (adversary pass 1, the note on
/// `train.rs`); `task docs-check` keeps the page equal to the doc comment.
#[test]
fn the_profile_flag_documents_the_profile_pipeline_lines() {
    let page = std::fs::read_to_string(repo_root().join("website/docs/reference/cli.md"))
        .expect("the CLI reference reads");
    let row = page
        .lines()
        .find(|l| l.starts_with("| `--profile` |"))
        .expect("the --profile row");
    for words in [
        "profile pipeline <kernel_module_id> executable",
        "statistic",
        "profile pipeline statistics: not reported",
    ] {
        assert!(row.contains(words), "{words}: {row}");
    }
    let help = modelspec(&["train", "--help"]);
    assert!(
        text(&help.stdout).contains("profile pipeline statistics: not reported"),
        "{}",
        text(&help.stdout)
    );
}

#[test]
#[ignore = "needs a Vulkan device"]
fn profile_prints_the_statistics_of_every_kernel_and_leaves_stdout_alone() {
    let ir = modelspec(&["ir", MNIST_MLP, "--stage", "kernel"]);
    assert_eq!(ir.status.code(), Some(0), "{}", text(&ir.stderr));
    let kernels: BTreeMap<String, String> = text(&ir.stdout)
        .lines()
        .filter_map(|l| l.strip_prefix("kernel "))
        .map(|l| {
            let words: Vec<&str> = l.split(' ').collect();
            (words[1].to_owned(), words[2].to_owned())
        })
        .collect();
    assert!(kernels.values().any(|kind| kind == "matmul"), "{kernels:?}");

    let plain = modelspec(&["train", MNIST_MLP, "--backend", "vulkan"]);
    assert_eq!(plain.status.code(), Some(0), "{}", text(&plain.stderr));
    let profiled = modelspec(&["train", MNIST_MLP, "--backend", "vulkan", "--profile"]);
    let stderr = text(&profiled.stderr);
    assert_eq!(profiled.status.code(), Some(0), "{stderr}");
    eprintln!("{stderr}");
    assert_eq!(profiled.stdout, plain.stdout, "stdout is byte-equal");

    // `<id> executable <i> statistic "<name>" <value> description "<description>"`.
    let mut registers: BTreeMap<String, u64> = BTreeMap::new();
    let mut named: BTreeSet<String> = BTreeSet::new();
    for line in stderr
        .lines()
        .filter_map(|l| l.strip_prefix("profile pipeline "))
    {
        let (id, rest) = line.split_once(' ').expect("a kernel id and more");
        assert!(kernels.contains_key(id), "{id} is not a kernel of the plan");
        let Some((_, statistic)) = rest.split_once(" statistic \"") else {
            assert!(rest.starts_with("executable "), "{line}");
            assert!(rest.contains(" subgroup_size "), "{line}");
            continue;
        };
        let (name, after) = statistic.split_once("\" ").expect("a quoted name");
        let (value, description) = after.split_once(' ').expect("a value and a description");
        assert!(description.starts_with("description \""), "{line}");
        named.insert(id.to_owned());
        if name.to_lowercase().contains("register") {
            registers.insert(id.to_owned(), value.parse().expect("an integer count"));
        }
    }
    let every: BTreeSet<String> = kernels.keys().cloned().collect();
    assert_eq!(
        named, every,
        "every kernel of the plan prints its statistics"
    );
    assert_eq!(
        registers.keys().cloned().collect::<BTreeSet<_>>(),
        every,
        "every kernel prints a register count: {registers:?}"
    );
    assert!(registers.values().all(|&r| r > 0), "{registers:?}");
    assert!(!stderr.contains("not reported"), "{stderr}");
}
