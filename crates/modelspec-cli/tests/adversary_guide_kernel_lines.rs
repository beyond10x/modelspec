//! Adversary case for story m1-fuse-elementwise: the guide `website/docs/guides/inspect-a-run.md`
//! quotes `kernel` lines of `modelspec ir --stage kernel examples/mnist_mlp.mspec` and names the
//! the loss sum by its step position and kernel. The fusions changed both; the guide must say
//! what the command prints.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn kernel_ir() -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["ir", "examples/mnist_mlp.mspec", "--stage", "kernel"])
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf-8")
}

#[test]
fn every_kernel_line_the_guide_quotes_is_printed_by_ir_stage_kernel() {
    let guide = std::fs::read_to_string(repo_root().join("website/docs/guides/inspect-a-run.md"))
        .expect("the guide reads");
    let printed = kernel_ir();
    let quoted: Vec<&str> = guide
        .lines()
        .filter(|l| l.starts_with("kernel k"))
        .collect();
    assert!(!quoted.is_empty(), "the guide quotes kernel lines");
    let missing: Vec<&str> = quoted
        .iter()
        .copied()
        .filter(|q| !printed.lines().any(|p| p == *q))
        .collect();
    assert!(
        missing.is_empty(),
        "quoted in the guide, not printed:\n{}\n\nprinted kernel lines:\n{}",
        missing.join("\n"),
        printed
            .lines()
            .filter(|l| l.starts_with("kernel k"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn the_guide_names_the_loss_sum_at_the_step_position_the_plan_has() {
    let guide = std::fs::read_to_string(repo_root().join("website/docs/guides/inspect-a-run.md"))
        .expect("the guide reads");
    let sentence = guide
        .split("Here step position ")
        .nth(1)
        .expect("the guide names the loss sum's step position");
    let (position, rest) = sentence.split_once(", module `").expect("a module follows");
    let (prefix, rest) = rest.split_once("…`, is `").expect("a kernel follows");
    let (kernel, _) = rest.split_once('`').expect("the kernel is quoted");
    let printed = kernel_ir();
    let loss_kernel = printed
        .lines()
        .find(|l| l.starts_with("kernel k") && l.contains(" loss_sum Mse "))
        .expect("a loss_sum kernel line");
    let mut words = loss_kernel.split_whitespace().skip(1);
    let (id, module) = (words.next().unwrap(), words.next().unwrap());
    assert_eq!(kernel, id, "the loss sum kernel is {loss_kernel}");
    assert!(
        module.starts_with(prefix),
        "the guide's module {prefix}… is not {module}"
    );
    let dispatch = format!("  {position}: dispatch {id} ");
    assert!(
        printed.lines().any(|l| l.starts_with(&dispatch)),
        "plan step has no line starting {dispatch:?}:\n{printed}"
    );
    let profiled = format!("profile dispatch step {position} {module} ");
    assert!(
        guide.lines().any(|l| l.starts_with(&profiled)),
        "the guide's profile sample has no line starting {profiled:?}"
    );
}
