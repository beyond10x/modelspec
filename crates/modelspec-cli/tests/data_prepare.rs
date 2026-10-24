//! `modelspec data prepare <corpus-dir> --out <dir>` (story:bert-corpus-splits): the release's
//! `wiki.train.raw` and `wiki.valid.raw` become the `Train` and `HeldOut` splits of a `Dataset`
//! record that validates against the generated schema (`spec/domains/data.yaml`).
//!
//! The corpus is the fixture under `crates/modelspec-runtime/tests/fixtures/corpus/`, original
//! text in the release's layout, so no test needs the corpus or the network.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

const FIXTURE: &str = "crates/modelspec-runtime/tests/fixtures/corpus";
const SCHEMA: &str = "crates/modelspec-runtime/schema/entities/modelspec.data.Dataset.schema.json";
const RECORD: &str = "dataset.json";
/// The release file of each split role and the file `data prepare` writes it to.
const SPLITS: [(&str, &str, &str); 2] = [
    ("Train", "wiki.train.raw", "train.raw"),
    ("HeldOut", "wiki.valid.raw", "heldout.raw"),
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh, empty directory under this crate's build directory, named for the test.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("data_prepare_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// Runs `modelspec <args>` from the repository root.
fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs")
}

fn prepare(out: &Path) -> Output {
    let out = out.to_str().expect("UTF-8 path");
    modelspec(&["data", "prepare", FIXTURE, "--out", out])
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn assert_success(output: &Output) {
    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        stderr(output)
    );
}

/// The title lines of the articles of a release file: a top-level heading `= <title> =`, one
/// `=` on each side, which a section heading `= = <section> = =` is not.
fn article_titles(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|t| t.starts_with("= ") && t.ends_with(" =") && !t.starts_with("= ="))
        .map(str::to_owned)
        .collect()
}

fn schema() -> jsonschema::Validator {
    let text = std::fs::read_to_string(repo_root().join(SCHEMA)).expect("the schema is generated");
    let schema: Value = serde_json::from_str(&text).expect("the schema is JSON");
    jsonschema::validator_for(&schema).expect("the schema compiles")
}

#[test]
fn splits_one_per_role_cover_every_fixture_document_and_validate() {
    let out = scratch("splits");
    assert_success(&prepare(&out));

    let text = std::fs::read_to_string(out.join(RECORD)).expect("the record is written");
    assert!(text.ends_with('\n'), "the record ends with a newline");
    let record: Value = serde_json::from_str(&text).expect("the record is JSON");
    let errors: Vec<String> = schema()
        .iter_errors(&record)
        .map(|e| e.to_string())
        .collect();
    assert!(errors.is_empty(), "{record}: {errors:?}");
    assert_eq!(record["state"], "Recorded");
    assert_eq!(record["name"], "corpus");

    let splits = record["splits"].as_array().expect("splits is an array");
    let roles: Vec<&str> = splits.iter().map(|s| s["role"].as_str().unwrap()).collect();
    assert_eq!(
        roles,
        ["Train", "HeldOut"],
        "one split per role, Train first"
    );

    // Every fixture document is in exactly one split, and each split file is its release file.
    let mut every_title = Vec::new();
    let mut written_titles = Vec::new();
    for ((role, release, written), split) in SPLITS.iter().zip(splits) {
        let source = std::fs::read(repo_root().join(FIXTURE).join(release)).unwrap();
        let copy = std::fs::read(out.join(written))
            .unwrap_or_else(|e| panic!("{role} is written to {written}: {e}"));
        assert_eq!(copy, source, "{written} is {release} byte for byte");
        let titles = article_titles(std::str::from_utf8(&source).unwrap());
        assert_eq!(split["document_count"], titles.len() as u64, "{role}");
        assert_eq!(split["byte_count"], source.len() as u64, "{role}");
        let hash = split["content_hash"].as_str().unwrap();
        assert!(
            hash.len() == 64 && hash.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
            "{role}: content_hash {hash} is 64 lowercase hex digits"
        );
        every_title.extend(titles.iter().cloned());
        written_titles.push(titles);
    }
    assert_eq!(every_title.len(), 5, "the fixture holds five articles");
    for title in &every_title {
        let holders = written_titles
            .iter()
            .filter(|titles| titles.contains(title))
            .count();
        assert_eq!(holders, 1, "{title} is in exactly one split");
    }
    assert_ne!(
        splits[0]["content_hash"], splits[1]["content_hash"],
        "the two splits differ"
    );
    let id = record["dataset_id"].as_str().unwrap();
    assert_eq!(&id[14..15], "8", "{id} is a version-8 UUID");
}

#[test]
fn reproducible_two_runs_write_byte_identical_files() {
    let first = scratch("reproducible_first");
    let second = scratch("reproducible_second");
    assert_success(&prepare(&first));
    assert_success(&prepare(&second));

    let listing = |dir: &Path| {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        names
    };
    let names = listing(&first);
    assert_eq!(names, ["dataset.json", "heldout.raw", "train.raw"]);
    assert_eq!(listing(&second), names);
    for name in &names {
        assert_eq!(
            std::fs::read(first.join(name)).unwrap(),
            std::fs::read(second.join(name)).unwrap(),
            "{name} is byte-identical across two runs"
        );
    }
}

#[test]
fn refusals_missing_corpus_exits_1_naming_it() {
    let out = scratch("refusals_missing");
    let missing = "crates/modelspec-runtime/tests/fixtures/no-such-corpus";
    let output = modelspec(&["data", "prepare", missing, "--out", out.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1), "stderr: {}", stderr(&output));
    let message = stderr(&output);
    assert!(
        message.starts_with(&format!("modelspec data prepare {missing}")),
        "the message names the missing path: {message}"
    );
    assert!(!message.contains("E-"), "no diagnostic code: {message}");
    assert!(!out.join(RECORD).exists(), "a refused run writes no record");
}

#[test]
fn refusals_missing_split_file_exits_1_naming_it() {
    let corpus = scratch("refusals_split_corpus");
    std::fs::copy(
        repo_root().join(FIXTURE).join("wiki.train.raw"),
        corpus.join("wiki.train.raw"),
    )
    .unwrap();
    let out = scratch("refusals_split_out");
    let output = modelspec(&[
        "data",
        "prepare",
        corpus.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(1), "stderr: {}", stderr(&output));
    let named = corpus.join("wiki.valid.raw").display().to_string();
    assert!(
        stderr(&output).starts_with(&format!("modelspec data prepare {named}: ")),
        "the message names the missing split file: {}",
        stderr(&output)
    );
    let left: Vec<_> = std::fs::read_dir(&out).unwrap().collect();
    assert!(
        left.is_empty(),
        "a refused run leaves the output directory empty"
    );
}

#[test]
fn refusals_no_out_exits_2() {
    let output = modelspec(&["data", "prepare", FIXTURE]);
    assert_eq!(output.status.code(), Some(2), "stderr: {}", stderr(&output));
    assert!(stderr(&output).contains("--out"), "{}", stderr(&output));
}
