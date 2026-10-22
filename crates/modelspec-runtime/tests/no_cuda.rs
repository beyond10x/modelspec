//! No CUDA anywhere (AGENTS.md, Rules; `story:m1-vulkan-train`): with every feature on and for
//! every target, the dependency tree of `modelspec-runtime` and `modelspec-cli` holds no crate
//! named `cuda`, `cudarc`, `cust` or `nvrtc`, and the dynamic section of the `modelspec`
//! binary and of this test binary (the runtime with the `vulkan` feature) names no `libcuda`,
//! `libcudart` or `libnvrtc`.
//!
//! The checks run `cargo tree` and `readelf -d` as processes; no crate is added for them. The
//! `modelspec` binary is the one `cargo build --workspace` (`task build`, the first step of
//! `task ci`) leaves beside this test's profile directory.
#![cfg(feature = "vulkan")]

use std::path::{Path, PathBuf};
use std::process::Command;

/// The crate names refused, alone or as the stem of a `-` or `_` suffixed name.
const CUDA_CRATES: &[&str] = &["cuda", "cudarc", "cust", "nvrtc"];

/// The shared libraries refused in a dynamic section.
const CUDA_LIBRARIES: &[&str] = &["libcuda", "libcudart", "libnvrtc"];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn is_cuda_crate(name: &str) -> bool {
    CUDA_CRATES.iter().any(|refused| {
        name == *refused
            || name
                .strip_prefix(refused)
                .is_some_and(|rest| rest.starts_with('-') || rest.starts_with('_'))
    })
}

/// The crate names of the dependency tree of `package`, every feature on, every target.
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

/// The `NEEDED` entries of the dynamic section of `binary`.
fn needed(binary: &Path) -> Vec<String> {
    let output = Command::new("readelf")
        .args(["-d", "-W"])
        .arg(binary)
        .output()
        .expect("readelf runs");
    assert!(
        output.status.success(),
        "readelf -d {}: {}",
        binary.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| line.contains("(NEEDED)"))
        .filter_map(|line| {
            let start = line.find('[')? + 1;
            let end = line[start..].find(']')? + start;
            Some(line[start..end].to_owned())
        })
        .collect()
}

#[test]
fn the_names_refused_are_matched_as_crate_names() {
    for name in [
        "cuda",
        "cudarc",
        "cust",
        "nvrtc",
        "cuda-sys",
        "cust_core",
        "nvrtc-sys",
    ] {
        assert!(is_cuda_crate(name), "{name}");
    }
    for name in ["ash", "custom", "rspirv", "cudaless"] {
        assert!(!is_cuda_crate(name), "{name}");
    }
}

#[test]
fn no_cuda_crate_is_in_the_dependency_tree_with_every_feature() {
    for package in ["modelspec-runtime", "modelspec-cli"] {
        let crates = tree(package);
        assert!(
            crates.iter().any(|name| name == "ash"),
            "{package}: the tree with every feature holds the Vulkan crate ash: {crates:?}"
        );
        let cuda: Vec<_> = crates.iter().filter(|name| is_cuda_crate(name)).collect();
        assert!(cuda.is_empty(), "{package} depends on {cuda:?}");
    }
}

#[test]
fn no_binary_links_a_cuda_library() {
    let this = std::env::current_exe().expect("the test binary has a path");
    let profile = this
        .parent()
        .and_then(Path::parent)
        .expect("the test binary is in <profile>/deps");
    let modelspec = profile.join(format!("modelspec{}", std::env::consts::EXE_SUFFIX));
    assert!(
        modelspec.is_file(),
        "{} is missing: run `cargo build -p modelspec-cli` (task build) first",
        modelspec.display()
    );
    for binary in [this.as_path(), modelspec.as_path()] {
        let libraries = needed(binary);
        assert!(
            libraries.iter().any(|library| library.starts_with("libc.")),
            "{}: the dynamic section names libc: {libraries:?}",
            binary.display()
        );
        let cuda: Vec<_> = libraries
            .iter()
            .filter(|library| {
                CUDA_LIBRARIES
                    .iter()
                    .any(|refused| library.starts_with(&format!("{refused}.")))
            })
            .collect();
        assert!(cuda.is_empty(), "{} links {cuda:?}", binary.display());
    }
}
