//! The crate list, from `cargo metadata --no-deps`: every workspace package, its directory, the
//! name Rust code uses for it, the workspace packages it depends on, and its `description`.

use std::path::Path;
use std::process::Command;

use serde_json::Value;

use crate::{Result, cell, front_matter};

/// A directory of the repository on GitHub, by path.
const TREE: &str = "https://github.com/beyond10x/modelspec/tree/main";

/// One workspace package as the page lists it.
#[derive(Debug, PartialEq, Eq)]
struct Package {
    name: String,
    directory: String,
    /// The library name (`use <lib>`), when the package has a library.
    lib: Option<String>,
    /// The binaries it builds.
    bins: Vec<String>,
    /// The workspace packages it depends on outside its tests.
    uses: Vec<String>,
    description: String,
}

fn field<'a>(value: &'a Value, key: &str, what: &str) -> Result<&'a str> {
    value[key]
        .as_str()
        .ok_or_else(|| format!("cargo metadata: {what} has no {key}"))
}

/// The packages of a `cargo metadata --format-version 1` document, sorted by name, with their
/// directories relative to `workspace_root`. A package without a description is refused: the page
/// would have nothing to say about it.
fn packages(metadata: &Value) -> Result<Vec<Package>> {
    let root = field(metadata, "workspace_root", "the document")?;
    let listed = metadata["packages"]
        .as_array()
        .ok_or("cargo metadata lists no packages")?;
    let names: Vec<&str> = listed
        .iter()
        .filter_map(|package| package["name"].as_str())
        .collect();
    let mut found = Vec::new();
    for package in listed {
        let name = field(package, "name", "a package")?.to_owned();
        let manifest = field(package, "manifest_path", &name)?;
        let directory = Path::new(manifest)
            .parent()
            .and_then(|dir| dir.strip_prefix(root).ok())
            .ok_or_else(|| format!("{name} lies outside the workspace"))?
            .to_string_lossy()
            .into_owned();
        let description = package["description"]
            .as_str()
            .ok_or_else(|| format!("{name} has no description in its Cargo.toml"))?;
        let mut lib = None;
        let mut bins = Vec::new();
        for target in package["targets"].as_array().into_iter().flatten() {
            let kinds: Vec<&str> = target["kind"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            let target_name = target["name"].as_str().unwrap_or_default().to_owned();
            if kinds.contains(&"lib") {
                lib = Some(target_name);
            } else if kinds.contains(&"bin") {
                bins.push(target_name);
            }
        }
        bins.sort();
        let mut uses: Vec<String> = package["dependencies"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|dependency| dependency["kind"].is_null())
            .filter_map(|dependency| dependency["name"].as_str())
            .filter(|dependency| names.contains(dependency))
            .map(str::to_owned)
            .collect();
        uses.sort();
        found.push(Package {
            name,
            directory,
            lib,
            bins,
            uses,
            description: description.trim().to_owned(),
        });
    }
    if found.is_empty() {
        return Err("cargo metadata lists no packages".into());
    }
    found.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(found)
}

/// The version every package of `metadata` carries, which is the release that ships the binary:
/// the packages inherit `[workspace.package] version`, and a release is tagged with it. Packages
/// of two versions are refused.
fn release(metadata: &Value) -> Result<String> {
    let versions: std::collections::BTreeSet<&str> = metadata["packages"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|package| package["version"].as_str().unwrap_or_default())
        .collect();
    match versions.into_iter().collect::<Vec<_>>().as_slice() {
        [version] if !version.is_empty() => Ok((*version).to_owned()),
        other => Err(format!(
            "cargo metadata: the packages carry versions {other:?}, not one workspace version"
        )),
    }
}

fn table(packages: &[&Package]) -> String {
    let mut out = String::from(
        "| Package | Directory | In Rust | Depends on | What it is |\n|---|---|---|---|---|\n",
    );
    for package in packages {
        let mut forms = Vec::new();
        if let Some(lib) = &package.lib {
            forms.push(format!("`use {lib}`"));
        }
        for bin in &package.bins {
            forms.push(format!("binary `{bin}`"));
        }
        let uses = if package.uses.is_empty() {
            "none".to_owned()
        } else {
            package
                .uses
                .iter()
                .map(|name| format!("`{name}`"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        out.push_str(&format!(
            "| `{}` | [`{dir}`]({TREE}/{dir}) | {} | {uses} | {} |\n",
            package.name,
            forms.join(", "),
            cell(&package.description),
            dir = package.directory,
        ));
    }
    out
}

/// The crate list page for the packages of `metadata`.
fn page_for(metadata: &Value) -> Result<String> {
    let all = packages(metadata)?;
    let version = release(metadata)?;
    let (tooling, product): (Vec<&Package>, Vec<&Package>) = all
        .iter()
        .partition(|package| package.name.ends_with("-docs"));
    let mut out = front_matter(
        "Crates",
        "Crates",
        "Every package of the ModelSpec workspace, generated from cargo metadata.",
        4,
    );
    out.push_str(&format!(
        "Generated by `modelspec-docs` from `cargo metadata --no-deps`: every package of the \
         workspace, with the `description` its `Cargo.toml` gives. There are {} packages. None is \
         on a registry; the [{version} release](https://github.com/beyond10x/modelspec/releases/tag/{version}) \
         ships the `modelspec` binary.\n\n## The compiler and its runtime\n\n`modelspec-core` \
         carries no Vulkan dependency, and `task core-no-vulkan` fails when one appears in its \
         dependency tree (DESIGN.md § 7).\n\n{}\n## Repository tooling\n\nGenerators and checks \
         the repository's own gate runs. Nothing outside the repository depends on them.\n\n{}",
        all.len(),
        table(&product),
        table(&tooling),
    ));
    Ok(out)
}

/// The crate list page for the workspace at `root`, read with `cargo metadata`.
pub fn page(root: &Path) -> Result<String> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(&cargo)
        .args([
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--manifest-path",
        ])
        .arg(root.join("Cargo.toml"))
        .output()
        .map_err(|e| format!("running cargo metadata: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let metadata: Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("parsing cargo metadata: {e}"))?;
    page_for(&metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(description: Option<&str>) -> Value {
        serde_json::json!({
            "workspace_root": "/w",
            "packages": [
                {
                    "name": "modelspec-docs",
                    "version": "0.10.0",
                    "manifest_path": "/w/crates/modelspec-docs/Cargo.toml",
                    "description": "Generates pages.",
                    "targets": [{"kind": ["bin"], "name": "modelspec-docs"}],
                    "dependencies": [
                        {"name": "modelspec-cli", "kind": null},
                        {"name": "clap", "kind": null}
                    ]
                },
                {
                    "name": "modelspec-cli",
                    "version": "0.10.0",
                    "manifest_path": "/w/crates/modelspec-cli/Cargo.toml",
                    "description": description,
                    "targets": [
                        {"kind": ["lib"], "name": "modelspec_cli"},
                        {"kind": ["bin"], "name": "modelspec"},
                        {"kind": ["test"], "name": "check"}
                    ],
                    "dependencies": [{"name": "modelspec-docs", "kind": "dev"}]
                }
            ]
        })
    }

    #[test]
    fn packages_are_sorted_split_and_linked() {
        let page = page_for(&metadata(Some("The command | line."))).unwrap();
        assert!(page.contains("# generated by modelspec-docs, do not edit"));
        assert!(page.contains("There are 2 packages."));
        let product = page.find("## The compiler and its runtime").unwrap();
        let tooling = page.find("## Repository tooling").unwrap();
        let cli = page.find("| `modelspec-cli` |").unwrap();
        let docs = page.find("| `modelspec-docs` |").unwrap();
        assert!(product < cli && cli < tooling && tooling < docs);
        assert!(page.contains(
            "| `modelspec-cli` | [`crates/modelspec-cli`](https://github.com/beyond10x/modelspec/tree/main/crates/modelspec-cli) | `use modelspec_cli`, binary `modelspec` | none | The command \\| line. |"
        ));
        assert!(page.contains("| binary `modelspec-docs` | `modelspec-cli` | Generates pages. |"));
    }

    #[test]
    fn the_release_named_is_the_workspace_version() {
        let page = page_for(&metadata(Some("The command line."))).unwrap();
        assert!(page.contains(
            "the [0.10.0 release](https://github.com/beyond10x/modelspec/releases/tag/0.10.0) ships"
        ));
        assert!(!page.contains("0.1.0"));
        let mut split = metadata(Some("The command line."));
        split["packages"][1]["version"] = "0.9.0".into();
        assert!(page_for(&split).is_err(), "packages of two versions");
    }

    #[test]
    fn the_committed_page_names_the_current_workspace_version() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let committed =
            std::fs::read_to_string(root.join("website/docs/reference/crates.md")).unwrap();
        let version = env!("CARGO_PKG_VERSION");
        assert!(
            committed.contains(&format!("releases/tag/{version})")),
            "website/docs/reference/crates.md does not name {version}; run `task docs-generate`"
        );
    }

    #[test]
    fn a_package_without_a_description_is_refused() {
        assert!(page_for(&metadata(None)).is_err());
    }
}
