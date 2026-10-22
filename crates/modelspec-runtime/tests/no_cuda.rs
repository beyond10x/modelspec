//! No CUDA anywhere (AGENTS.md, Rules; `story:m1-vulkan-train`): with every feature on and for
//! every target, the dependency tree of `modelspec-runtime` and `modelspec-cli` holds no crate
//! whose name contains `cuda`, `cudarc`, `cust` or `nvrtc`, and the dynamic section of the
//! `modelspec` binary and of this test binary (the runtime with the `vulkan` feature) names no
//! `libcuda`, `libcudart` or `libnvrtc`.
//!
//! The checks run `cargo tree`, `cargo build` and `readelf -d` as processes; no crate is added
//! for them. The test builds the `modelspec` binary itself and reads the path cargo reports for
//! it, so it never checks a binary an earlier build left behind. `readelf -d` sees only the
//! libraries linked at build time: a library opened with `dlopen` at run time, as `ash` opens
//! the Vulkan loader, is not in the dynamic section and this test cannot see it.
#![cfg(feature = "vulkan")]

use std::path::{Path, PathBuf};
use std::process::Command;

/// The words refused anywhere in a crate name. No crate of the tree contains one (checked with
/// `cust` too, which also matches names such as `custom`), so a plain substring match is used.
const CUDA_CRATES: &[&str] = &["cuda", "cudarc", "cust", "nvrtc"];

/// The shared libraries refused in a dynamic section.
const CUDA_LIBRARIES: &[&str] = &["libcuda", "libcudart", "libnvrtc"];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn is_cuda_crate(name: &str) -> bool {
    CUDA_CRATES.iter().any(|refused| name.contains(refused))
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

/// Builds the `modelspec` binary with every default feature of `modelspec-cli`, `vulkan` among
/// them, and returns the path of the executable cargo reports for it.
fn build_modelspec() -> PathBuf {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let output = Command::new(cargo)
        .args([
            "build",
            "--locked",
            "-p",
            "modelspec-cli",
            "--bin",
            "modelspec",
            "--message-format",
            "json",
        ])
        .current_dir(workspace_root())
        .output()
        .expect("cargo build runs");
    assert!(
        output.status.success(),
        "cargo build -p modelspec-cli: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("cargo prints UTF-8");
    let executable = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|message| {
            message["reason"] == "compiler-artifact" && message["target"]["name"] == "modelspec"
        })
        .find_map(|message| message["executable"].as_str().map(PathBuf::from))
        .expect("cargo build reports the modelspec executable");
    assert!(executable.is_file(), "{}", executable.display());
    executable
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
fn every_name_containing_a_refused_word_is_refused() {
    for name in [
        "cuda",
        "cudarc",
        "cust",
        "nvrtc",
        "cuda-sys",
        "cust_core",
        "nvrtc-sys",
        "burn-cuda",
        "cubecl-cuda",
        "rustacuda",
    ] {
        assert!(is_cuda_crate(name), "{name}");
    }
    for name in ["ash", "rspirv", "ash-window", "spirv"] {
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
    let modelspec = build_modelspec();
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
