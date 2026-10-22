//! Adversarial cases for `modelspec train --profile` (`story:m1-vulkan-profile`).
#![cfg(feature = "vulkan")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const MNIST_MLP: &str = "examples/mnist_mlp.mspec";
const LINEAR_REGRESSION: &str = "examples/linear_regression.mspec";
const NO_LOADER: &str = "/nonexistent/modelspec-test/libvulkan-missing.so.1";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn modelspec(args: &[&str], loader: Option<&str>) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_modelspec"));
    command.args(args).current_dir(repo_root());
    if let Some(loader) = loader {
        command.env("MODELSPEC_VULKAN_LIBRARY", loader);
    }
    command.output().expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn profile_on_the_cpu_is_refused_in_every_flag_order() {
    let orders: [&[&str]; 5] = [
        &["train", "--profile", LINEAR_REGRESSION, "--backend", "cpu"],
        &["train", "--profile", "--backend", "cpu", LINEAR_REGRESSION],
        &["train", "--backend=cpu", LINEAR_REGRESSION, "--profile"],
        &[
            "train",
            LINEAR_REGRESSION,
            "--backend",
            "cpu",
            "--profile",
            "--steps",
            "3",
        ],
        &[
            "train",
            "--steps",
            "3",
            "--profile",
            "--backend",
            "cpu",
            LINEAR_REGRESSION,
        ],
    ];
    for args in orders {
        let output = modelspec(args, Some(NO_LOADER));
        let stderr = text(&output.stderr);
        assert_eq!(output.status.code(), Some(2), "{args:?}: {stderr}");
        assert!(stderr.contains("'--profile'"), "{args:?}: {stderr}");
        assert!(
            stderr.contains("only with '--backend vulkan'"),
            "{args:?}: {stderr}"
        );
        assert!(output.stdout.is_empty(), "{args:?}");
    }
}

/// A `profile dispatch final <position> <id>` line names a Dispatch step of `plan step` as
/// `modelspec ir --stage kernel` prints it: that position is a `dispatch` of the kernel whose
/// module id the line carries.
#[test]
#[ignore = "needs a Vulkan device"]
fn a_final_line_names_a_dispatch_step_of_plan_step() {
    let ir = modelspec(&["ir", MNIST_MLP, "--stage", "kernel"], None);
    assert_eq!(ir.status.code(), Some(0));
    let ir = text(&ir.stdout);
    let kernels: std::collections::BTreeMap<String, String> = ir
        .lines()
        .filter_map(|l| l.strip_prefix("kernel "))
        .map(|l| {
            let mut words = l.split(' ');
            (
                words.next().unwrap().to_owned(),
                words.next().unwrap().to_owned(),
            )
        })
        .collect();
    let step: std::collections::BTreeMap<usize, String> = ir
        .lines()
        .skip_while(|l| *l != "plan step")
        .skip(1)
        .take_while(|l| l.starts_with("  "))
        .filter_map(|l| {
            let (at, step) = l.trim().split_once(": ")?;
            let kernel = step.strip_prefix("dispatch ")?.split(' ').next()?;
            Some((at.parse().unwrap(), kernels[kernel].clone()))
        })
        .collect();
    let output = modelspec(
        &[
            "train",
            MNIST_MLP,
            "--backend",
            "vulkan",
            "--profile",
            "--steps",
            "2",
        ],
        None,
    );
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(0), "{stderr}");
    assert!(!stderr.contains("profile dispatch forward "), "{stderr}");
    let mut finals = 0;
    for line in stderr.lines() {
        let Some(rest) = line.strip_prefix("profile dispatch final ") else {
            continue;
        };
        finals += 1;
        let words: Vec<&str> = rest.split(' ').collect();
        let at: usize = words[0].parse().unwrap();
        assert_eq!(
            step.get(&at),
            Some(&words[1].to_owned()),
            "`{line}` names no dispatch of plan step {step:?}"
        );
    }
    assert!(finals > 0, "no `profile dispatch final` line: {stderr}");
}
