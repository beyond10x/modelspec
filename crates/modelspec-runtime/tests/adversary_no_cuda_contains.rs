//! Adversary case for `story:m1-vulkan-train` acceptance 5, read as written: "the dependency tree
//! of `modelspec-runtime` and `modelspec-cli` with all features names no crate whose name
//! *contains* `cuda`, `cudarc`, `cust` or `nvrtc`".
//!
//! `tests/no_cuda.rs` matches a refused word only as the whole name or as its stem before `-` or
//! `_`, so `burn-cuda`, `cubecl-cuda` and `rustacuda`, all CUDA crates, pass it. This case runs the
//! same `cargo tree` and refuses every name that contains a refused word.
#![cfg(feature = "vulkan")]

use std::path::{Path, PathBuf};
use std::process::Command;

const REFUSED: &[&str] = &["cuda", "cudarc", "cust", "nvrtc"];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tree(package: &str) -> Vec<String> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let output = Command::new(cargo)
        .args([
            "tree",
            "--locked",
            "-p",
            package,
            "--all-features",
            "--target",
            "all",
            "-e",
            "normal,build,dev",
            "--prefix",
            "none",
            "-f",
            "{p}",
        ])
        .current_dir(workspace_root())
        .output()
        .expect("cargo tree runs");
    assert!(
        output.status.success(),
        "cargo tree -p {package}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("cargo tree prints UTF-8")
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_owned)
        .collect()
}

#[test]
fn adversary_no_crate_name_contains_a_cuda_word() {
    for package in ["modelspec-runtime", "modelspec-cli"] {
        let crates = tree(package);
        assert!(
            crates.iter().any(|name| name == "ash"),
            "{package}: {crates:?}"
        );
        let cuda: Vec<_> = crates
            .iter()
            .filter(|name| REFUSED.iter().any(|word| name.contains(word)))
            .collect();
        assert!(cuda.is_empty(), "{package} depends on {cuda:?}");
    }
}
