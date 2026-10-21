//! Guards DESIGN.md § 7: `modelspec-core` builds and tests without Vulkan.
//!
//! The dependency graph of `modelspec-core` (normal, build and dev edges) must name no Vulkan
//! crate. The planted case shows the guard is able to fail.

use std::process::Command;

/// Crates whose presence in the dependency graph means a Vulkan dependency.
const VULKAN_CRATES: &[&str] = &["ash", "vulkano", "wgpu", "gpu-allocator", "vk-mem"];

/// Returns the Vulkan crates named in `cargo tree` output, in order of first appearance.
fn vulkan_crates_in(tree: &str) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for line in tree.lines() {
        let mut words = line
            .trim_start_matches(|c: char| !c.is_ascii_alphanumeric())
            .split_whitespace();
        let (Some(name), Some(version)) = (words.next(), words.next()) else {
            continue;
        };
        if !version.starts_with('v') {
            continue;
        }
        if VULKAN_CRATES.contains(&name) && !found.iter().any(|f| f == name) {
            found.push(name.to_owned());
        }
    }
    found
}

/// The guard's `cargo tree` arguments: every edge kind that ends up in a build of
/// `modelspec-core` or its tests, on every target, with every feature enabled.
const TREE_ARGS: &[&str] = &[
    "tree",
    "-p",
    "modelspec-core",
    "-e",
    "normal,build,dev",
    "--target",
    "all",
    "--all-features",
];

/// Runs `cargo tree` with [`TREE_ARGS`] in `dir` and returns its output.
fn cargo_tree(dir: &std::path::Path) -> String {
    let output = Command::new(env!("CARGO"))
        .current_dir(dir)
        .args(TREE_ARGS)
        .output()
        .expect("cargo tree runs");
    assert!(
        output.status.success(),
        "cargo tree failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("cargo tree prints UTF-8")
}

#[test]
fn committed_tree_lists_no_vulkan_crate() {
    let tree = cargo_tree(std::path::Path::new(env!("CARGO_MANIFEST_DIR")));
    assert!(
        tree.starts_with("modelspec-core v"),
        "unexpected cargo tree output:\n{tree}"
    );
    assert_eq!(vulkan_crates_in(&tree), Vec::<String>::new(), "{tree}");
}

#[test]
fn planted_vulkan_crate_is_listed() {
    let tree = "modelspec-core v0.1.0 (/repo/crates/modelspec-core)\n\
                ├── logos v0.16.1\n\
                │   └── ash v0.38.0\n\
                └── thiserror v2.0.21\n";
    assert_eq!(vulkan_crates_in(tree), vec!["ash".to_owned()]);
}

/// Adversary: runs the guard's own `cargo tree` arguments against a throwaway workspace whose
/// `modelspec-core` manifest ends with `core_deps`. The workspace holds a local stand-in crate
/// named `ash` and a crate `gfx-wrapper` that depends on it, so no registry is needed.
fn guard_on_fixture(case: &str, core_deps: &str) -> Vec<String> {
    let root = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("no_vulkan_adversary")
        .join(case);
    let write = |relative: &str, text: &str| {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().expect("fixture path has a parent"))
            .expect("fixture directory is created");
        std::fs::write(&path, text).expect("fixture file is written");
    };
    write(
        "Cargo.toml",
        r#"[workspace]
members = ["modelspec-core"]
resolver = "3"
"#,
    );
    write(
        "ash/Cargo.toml",
        r#"[package]
name = "ash"
version = "0.38.0"
edition = "2024"
"#,
    );
    write("ash/src/lib.rs", "");
    write(
        "gfx-wrapper/Cargo.toml",
        r#"[package]
name = "gfx-wrapper"
version = "0.1.0"
edition = "2024"

[dependencies]
ash = { path = "../ash" }
"#,
    );
    write("gfx-wrapper/src/lib.rs", "");
    write(
        "modelspec-core/Cargo.toml",
        &format!(
            r#"[package]
name = "modelspec-core"
version = "0.1.0"
edition = "2024"

{core_deps}"#
        ),
    );
    write("modelspec-core/src/lib.rs", "");
    let tree = cargo_tree(&root.join("modelspec-core"));
    let found = vulkan_crates_in(&tree);
    eprintln!("{case}:\n{tree}found: {found:?}");
    found
}

#[test]
fn adversary_fixture_direct_vulkan_dependency_is_listed() {
    let deps = r#"[dependencies]
ash = { path = "../ash" }
"#;
    assert_eq!(guard_on_fixture("direct", deps), vec!["ash".to_owned()]);
}

#[test]
fn adversary_renamed_transitive_vulkan_dependency_is_listed() {
    let deps = r#"[dependencies]
gfx = { package = "gfx-wrapper", path = "../gfx-wrapper" }
"#;
    assert_eq!(
        guard_on_fixture("renamed-transitive", deps),
        vec!["ash".to_owned()]
    );
}

#[test]
fn adversary_target_gated_vulkan_dependency_is_listed() {
    let deps = r#"[target.'cfg(target_os = "android")'.dependencies]
ash = { path = "../ash" }
"#;
    assert_eq!(
        guard_on_fixture("target-gated", deps),
        vec!["ash".to_owned()]
    );
}

#[test]
fn adversary_feature_gated_vulkan_dependency_is_listed() {
    let deps = r#"[features]
vulkan = ["dep:ash"]

[dependencies]
ash = { path = "../ash", optional = true }
"#;
    assert_eq!(
        guard_on_fixture("feature-gated", deps),
        vec!["ash".to_owned()]
    );
}

#[test]
fn adversary_every_listed_vulkan_crate_is_detected() {
    let tree = "modelspec-core v0.1.0 (/repo/crates/modelspec-core)\n\
                ├── ash v0.38.0+1.3.281\n\
                ├── vulkano v0.35.1\n\
                ├── wgpu v26.0.1\n\
                ├── gpu-allocator v0.27.0\n\
                └── vk-mem v0.4.0\n";
    assert_eq!(
        vulkan_crates_in(tree),
        vec!["ash", "vulkano", "wgpu", "gpu-allocator", "vk-mem"]
    );
}
