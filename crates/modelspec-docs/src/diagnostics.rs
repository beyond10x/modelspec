//! The diagnostic code reference, from the code constants each stage declares.
//!
//! Every stage declares its codes next to the code that emits them (`diag.rs`: "Codes are declared
//! next to the stage that emits them"). The page lists them per stage with the doc comment of the
//! declaration. Two checks keep the page honest: every code the generator lists is a constant of
//! the library, and every `E-...` code a stage's source declares is listed, so a new code cannot
//! be missing from the page.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use modelspec_core::diag::Code;

use crate::{Result, SOURCE, cell, front_matter, io, sorted_entries};

/// One stage and the codes it declares.
struct Stage {
    title: &'static str,
    /// The source file that declares the codes, relative to the repository root.
    file: &'static str,
    /// What the stage checks and when a reader meets its codes.
    about: &'static str,
    codes: Vec<Code>,
}

fn stages() -> Vec<Stage> {
    use modelspec_core::{compile, resolve, syntax, types};
    vec![
        Stage {
            title: "Syntax",
            file: "crates/modelspec-core/src/syntax/mod.rs",
            about: "The parser of the M0 subset. A construct of the target language that M0 does \
                    not have yet has its own code, so the message names what is missing.",
            codes: syntax::codes::ALL.to_vec(),
        },
        Stage {
            title: "Names",
            file: "crates/modelspec-core/src/resolve.rs",
            about: "Name resolution: variables, operators, the model a train block names and the \
                    functions it calls.",
            codes: resolve::codes::ALL.to_vec(),
        },
        Stage {
            title: "Types and shapes",
            file: "crates/modelspec-core/src/types.rs",
            about: "The type and shape checker. Operator rules come from the operator registry \
                    ([Operators](operators.md)).",
            codes: types::codes::ALL.to_vec(),
        },
        Stage {
            title: "Training data",
            file: "crates/modelspec-runtime/src/train/data.rs",
            about: "The data loaders of `modelspec train`, synthetic data, the dense files of \
                    `--data`, `--init` and `--eval` and the packed masked-language-model \
                    directory of `--data`, which run after a file checks and before the first \
                    step, and of `modelspec infer --data --function`, the dense files of a \
                    model function's arguments, read before the function runs and the output \
                    file is written. A `--data` directory without `binding.json` is dense, and \
                    when every declaration names a batch tensor its refusal follows a line \
                    naming the directory and `binding.json`; `--eval` beside a packed `--data` \
                    is a usage error, exit 2, because `--eval` reads dense evaluation tensors.",
            codes: vec![
                modelspec_runtime::train::data::E_DATA_001,
                modelspec_runtime::train::data::E_DATA_002,
                modelspec_runtime::train::data::E_DATA_003,
                modelspec_runtime::train::data::E_DATA_004,
                modelspec_runtime::train::data::E_DATA_005,
                modelspec_runtime::train::data::E_DATA_006,
                modelspec_runtime::train::data::E_DATA_007,
                modelspec_runtime::train::data::E_DATA_008,
                modelspec_runtime::train::data::E_DATA_009,
                modelspec_runtime::train::data::E_DATA_010,
                modelspec_runtime::train::data::E_DATA_011,
            ],
        },
        Stage {
            title: "Training",
            file: "crates/modelspec-runtime/src/train/runner.rs",
            about: "The CPU training loop of `modelspec train`, which checks the loss and every \
                    parameter after each step.",
            codes: vec![modelspec_runtime::train::runner::E_TRAIN_001],
        },
        Stage {
            title: "Checkpoints",
            file: "crates/modelspec-runtime/src/train/checkpoint.rs",
            about: "The `.msckpt` reader of `modelspec train --resume` and `modelspec infer`, \
                    which checks the header, the checksum, every offset and length and the \
                    payload hash before it allocates a tensor, and matches the checkpoint \
                    against the compiled file.",
            codes: vec![
                modelspec_runtime::train::checkpoint::E_CKPT_001,
                modelspec_runtime::train::checkpoint::E_CKPT_002,
                modelspec_runtime::train::checkpoint::E_CKPT_003,
                modelspec_runtime::train::checkpoint::E_CKPT_004,
            ],
        },
        Stage {
            title: "Artifacts",
            file: "crates/modelspec-core/src/artifact.rs",
            about: "The artifact directory reader of `modelspec compile`, which checks the format \
                    version, the size of each plan as the manifest is parsed, that every kernel \
                    module file is present and no longer than its entry, each module hash and \
                    the artifact hash, and then the invariants and the execution plan, in that \
                    order; then, before a device runs it, that the device reports at least the \
                    workgroup limits the plan was scheduled for.",
            codes: vec![
                modelspec_core::artifact::E_ARTIFACT_001,
                modelspec_core::artifact::E_ARTIFACT_002,
                modelspec_core::artifact::E_ARTIFACT_003,
                modelspec_core::artifact::E_ARTIFACT_004,
                modelspec_core::artifact::E_ARTIFACT_005,
            ],
        },
        Stage {
            title: "Target",
            file: "crates/modelspec-cli/src/train.rs",
            about: "Backend selection of `modelspec train` and `modelspec infer`, before the file is \
                    read, Vulkan device discovery and negotiation of `modelspec inspect-device`, \
                    `modelspec train --backend vulkan` and `modelspec infer --backend vulkan`, and \
                    the limits of the Vulkan target the \
                    schedule checks.",
            codes: vec![modelspec_cli::train::E_TARGET_001],
        },
        Stage {
            title: "Target",
            file: "crates/modelspec-runtime/src/vulkan/mod.rs",
            about: "",
            codes: vec![modelspec_runtime::vulkan::E_TARGET_002],
        },
        Stage {
            title: "Target",
            file: "crates/modelspec-runtime/src/vulkan/select.rs",
            about: "",
            codes: vec![modelspec_runtime::vulkan::select::E_TARGET_003],
        },
        Stage {
            title: "Target",
            file: "crates/modelspec-runtime/src/vulkan/negotiate.rs",
            about: "",
            codes: vec![modelspec_runtime::vulkan::negotiate::E_TARGET_004],
        },
        Stage {
            title: "Target",
            file: "crates/modelspec-core/src/compile.rs",
            about: "",
            codes: vec![
                compile::codes::TOO_LARGE_FOR_TARGET,
                compile::codes::NO_KERNEL,
            ],
        },
        Stage {
            title: "Compiler bugs",
            file: "crates/modelspec-core/src/types.rs",
            about: "Internal codes. No source file is meant to reach them; one that does is a bug \
                    to report, not a mistake in the model.",
            codes: vec![types::codes::INTERNAL],
        },
        Stage {
            title: "Compiler bugs",
            file: "crates/modelspec-core/src/compile.rs",
            about: "",
            codes: vec![compile::codes::COMPILER_BUG],
        },
    ]
}

/// The source directories of the crates that print diagnostics, relative to the repository root.
const SOURCE_DIRS: &[&str] = &[
    "crates/modelspec-core/src",
    "crates/modelspec-runtime/src",
    "crates/modelspec-cli/src",
];

/// Every `.rs` file under `dir`, sorted.
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in sorted_entries(dir)? {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// The part of `source` before its unit tests.
fn without_tests(source: &str) -> &str {
    source.split("#[cfg(test)]").next().unwrap_or(source)
}

/// Every code a declaration line of `source` declares, in order: `NAME = "E-..."` inside a
/// `declare!` block, or `Code::new("E-...")`.
fn declared(source: &str) -> Vec<String> {
    let mut found = Vec::new();
    for line in without_tests(source).lines() {
        let trimmed = line.trim();
        let literal = if let Some((_, rest)) = trimmed.split_once("Code::new(\"E-") {
            rest
        } else if let Some((_, rest)) = trimmed.split_once("= \"E-") {
            rest
        } else {
            continue;
        };
        if let Some((code, _)) = literal.split_once('"') {
            found.push(format!("E-{code}"));
        }
    }
    found
}

/// Rustdoc links as code spans: ``[`X`](path)`` and ``[`X`]`` become `` `X` ``.
fn unlink(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("[`") {
        let Some(close) = rest[start..].find("`]") else {
            break;
        };
        out.push_str(&rest[..start]);
        let end = start + close;
        out.push_str(&rest[start + 1..end + 1]);
        rest = &rest[end + 2..];
        if rest.starts_with('(')
            && let Some(paren) = rest.find(')')
        {
            rest = &rest[paren + 1..];
        }
    }
    out.push_str(rest);
    out
}

/// The doc comment of the declaration of `code` in `source`, on one line.
fn doc_of(source: &str, code: &str, file: &str) -> Result<String> {
    let lines: Vec<&str> = without_tests(source).lines().collect();
    let needle = format!("\"{code}\"");
    let at = lines
        .iter()
        .position(|line| line.contains(&needle))
        .ok_or_else(|| format!("{file} declares no {code}"))?;
    let mut doc = Vec::new();
    for line in lines[..at].iter().rev() {
        let trimmed = line.trim();
        if let Some(text) = trimmed.strip_prefix("///") {
            doc.push(text.trim());
        } else if !trimmed.starts_with("#[") {
            break;
        }
    }
    doc.reverse();
    if doc.is_empty() {
        return Err(format!("{file}: {code} has no doc comment"));
    }
    Ok(cell(&unlink(&doc.join(" "))))
}

pub fn page(root: &Path) -> Result<String> {
    let stages = stages();
    let mut out = front_matter(
        "Diagnostic codes",
        "Diagnostics",
        "Every diagnostic code modelspec prints, per compiler stage, from the constants the stages declare.",
        2,
    );
    out.push_str(
        "Every diagnostic modelspec prints starts with a stable code such as `E-TYPE-012`, \
         followed by a message, the source position and the values involved. \
         [A shape error](../examples/shape-error.md) shows one in full, and \
         [Read a diagnostic](../guides/read-a-diagnostic.md) explains each line.\n\n\
         Generated by `modelspec-docs` from the code constants each stage declares; the meaning \
         is the declaration's doc comment. `modelspec check`, `modelspec compile`, `modelspec train`, \
         `modelspec infer` and `modelspec inspect-device` exit `1` when they print one.\n",
    );
    let mut listed = BTreeSet::new();
    for stage in &stages {
        for code in &stage.codes {
            if !listed.insert(code.as_str().to_owned()) {
                return Err(format!("{code} is listed twice"));
            }
        }
    }
    let mut sources = Vec::new();
    for dir in SOURCE_DIRS {
        rust_files(&root.join(dir), &mut sources)?;
    }
    let mut everywhere = BTreeSet::new();
    for path in &sources {
        let source = fs::read_to_string(path).map_err(io(path.display()))?;
        everywhere.extend(declared(&source));
    }
    if listed != everywhere {
        let unlisted: Vec<_> = everywhere.difference(&listed).collect();
        let undeclared: Vec<_> = listed.difference(&everywhere).collect();
        return Err(format!(
            "the diagnostic page lists {} codes, the sources declare {}: not listed {unlisted:?}, \
             not declared {undeclared:?}",
            listed.len(),
            everywhere.len()
        ));
    }
    let mut previous = "";
    for stage in &stages {
        let source = fs::read_to_string(root.join(stage.file)).map_err(io(stage.file))?;
        let mine = declared(&source);
        if let Some(code) = stage
            .codes
            .iter()
            .find(|c| !mine.iter().any(|m| m == c.as_str()))
        {
            return Err(format!("{} does not declare {code}", stage.file));
        }
        let listed: Vec<&str> = stage.codes.iter().map(|c| c.as_str()).collect();
        if stage.title != previous {
            out.push_str(&format!("\n## {}\n\n{}\n", stage.title, stage.about));
            out.push_str("\n| Code | Meaning | Declared in |\n|---|---|---|\n");
            previous = stage.title;
        }
        for code in &listed {
            out.push_str(&format!(
                "| <a id=\"{anchor}\"></a>`{code}` | {} | [`{file}`]({SOURCE}/{file}) |\n",
                doc_of(&source, code, stage.file)?,
                anchor = code.to_lowercase(),
                file = stage.file,
            ));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declarations_are_found_outside_tests_only() {
        let source = "/// A.\nA = \"E-X-001\";\npub const B: Code = Code::new(\"E-X-002\");\n\
                      #[cfg(test)]\nmod tests { const C: Code = Code::new(\"E-X-003\"); }\n";
        assert_eq!(declared(source), ["E-X-001", "E-X-002"]);
    }

    #[test]
    fn the_doc_comment_above_a_declaration_is_its_meaning() {
        let source = "/// First line\n/// second, see [`MAX`](super::MAX) and [`ALL`].\n\
                      #[allow(dead_code)]\nA = \"E-X-001\";\nB = \"E-X-002\";\n";
        assert_eq!(
            doc_of(source, "E-X-001", "f.rs").unwrap(),
            "First line second, see `MAX` and `ALL`."
        );
        assert!(doc_of(source, "E-X-002", "f.rs").is_err());
        assert!(doc_of(source, "E-X-003", "f.rs").is_err());
    }

    #[test]
    fn every_stage_lists_exactly_the_codes_its_source_declares() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let page = page(&root).unwrap();
        for code in [
            "E-SYNTAX-021",
            "E-NAME-008",
            "E-TYPE-017",
            "E-TYPE-018",
            "E-DATA-003",
            "E-DATA-004",
            "E-DATA-010",
            "E-DATA-011",
            "E-TRAIN-001",
            "E-TARGET-001",
            "E-TARGET-002",
            "E-TARGET-003",
            "E-TARGET-004",
            "E-TARGET-010",
            "E-TARGET-011",
            "E-CKPT-001",
            "E-CKPT-002",
            "E-CKPT-003",
            "E-CKPT-004",
            "E-ARTIFACT-001",
            "E-ARTIFACT-002",
            "E-ARTIFACT-003",
            "E-ARTIFACT-004",
            "E-ARTIFACT-005",
        ] {
            let anchor = code.to_lowercase();
            assert!(
                page.contains(&format!("| <a id=\"{anchor}\"></a>`{code}` |")),
                "{code}"
            );
        }
        assert!(
            page.contains("| <a id=\"e-internal-001\"></a>`E-INTERNAL-001` |")
                && page.contains("| <a id=\"e-internal-002\"></a>`E-INTERNAL-002` |")
        );
        assert_eq!(page.matches("## Compiler bugs").count(), 1);
        assert_eq!(page.matches("## Target").count(), 1);
    }
}
