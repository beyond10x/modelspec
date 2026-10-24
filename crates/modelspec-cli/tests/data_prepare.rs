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

/// The articles of each fixture file, by title, as the fixture was written. `wiki.train.raw` also
/// holds a table key line ` = Weight ; W = ` with a blank line before it only and an equation line
/// ` = 2 c + 1 = ` with a blank line after it only; neither starts an article.
const ARTICLES: [&[&str]; 2] = [
    &[
        "Lantern Moth",
        "Copper Bridge",
        "Salt Orchard",
        "Pebble Ledger",
    ],
    &["Glass Kettle", "Hollow Hill"],
];

/// The heading-shaped lines of the fixture that are not article headings.
const NOT_ARTICLES: [&str; 2] = [" = Weight ; W = \n", " = 2 c + 1 = \n"];

/// The `dataset_id` of the fixture: the leading 16 bytes of blake3 over the blake3 digest bytes
/// of `wiki.train.raw` then `wiki.valid.raw`, version nibble 8 and RFC variant bits set; computed
/// once outside the code under test.
const FIXTURE_DATASET_ID: &str = "41e72a1c-06bf-89c0-80f7-a29fdc3b1874";

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

    // Each split file is its release file, and its count is the fixture's own article count.
    let mut written = Vec::new();
    for (((role, release, file), split), articles) in SPLITS.iter().zip(splits).zip(ARTICLES) {
        let source = std::fs::read(repo_root().join(FIXTURE).join(release)).unwrap();
        let copy = std::fs::read(out.join(file))
            .unwrap_or_else(|e| panic!("{role} is written to {file}: {e}"));
        assert_eq!(copy, source, "{file} is {release} byte for byte");
        assert_eq!(split["document_count"], articles.len() as u64, "{role}");
        assert_eq!(split["byte_count"], source.len() as u64, "{role}");
        let hash = split["content_hash"].as_str().unwrap();
        assert!(
            hash.len() == 64 && hash.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
            "{role}: content_hash {hash} is 64 lowercase hex digits"
        );
        written.push(String::from_utf8(copy).unwrap());
    }
    for line in NOT_ARTICLES {
        assert!(written[0].contains(line), "the fixture holds {line:?}");
    }
    // Every fixture document is in exactly one split.
    for title in ARTICLES.concat() {
        let heading = format!(" \n = {title} = \n \n");
        let holders: usize = written.iter().map(|s| s.matches(&heading).count()).sum();
        assert_eq!(holders, 1, "{title} is in exactly one split");
    }
    assert_ne!(
        splits[0]["content_hash"], splits[1]["content_hash"],
        "the two splits differ"
    );
}

#[test]
fn reproducible_dataset_id_is_derived_from_the_split_digests() {
    let out = scratch("dataset_id");
    let output = prepare(&out);
    assert_success(&output);
    let record: Value =
        serde_json::from_str(&std::fs::read_to_string(out.join(RECORD)).unwrap()).unwrap();
    let id = record["dataset_id"].as_str().unwrap();
    assert_eq!(id, FIXTURE_DATASET_ID);
    assert_eq!(&id[14..15], "8", "{id} is a version-8 UUID");
    assert!(
        matches!(&id[19..20], "8" | "9" | "a" | "b"),
        "{id}: RFC variant"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.starts_with(&format!("dataset {id} (corpus)\n")),
        "{stdout}"
    );
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

/// Both release files exist, so every check before reading passes; the HeldOut file holds no
/// document, so the run fails while copying. It leaves no `.partial` file, does not leave behind
/// an `--out` directory it created, and leaves an existing one as it was.
#[test]
fn refusals_failed_read_leaves_no_partial_file_and_no_created_out_directory() {
    let case = scratch("refusals_failed_read");
    let corpus = case.join("corpus");
    std::fs::create_dir_all(&corpus).unwrap();
    std::fs::copy(
        repo_root().join(FIXTURE).join("wiki.train.raw"),
        corpus.join("wiki.train.raw"),
    )
    .unwrap();
    std::fs::write(corpus.join("wiki.valid.raw"), " \n \n").unwrap();
    let named = corpus.join("wiki.valid.raw").display().to_string();

    let created = case.join("new").join("nested").join("out");
    let existing = case.join("existing");
    std::fs::create_dir_all(&existing).unwrap();
    for out in [&created, &existing] {
        let output = modelspec(&[
            "data",
            "prepare",
            corpus.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ]);
        assert_eq!(output.status.code(), Some(1), "stderr: {}", stderr(&output));
        assert_eq!(
            stderr(&output),
            format!("modelspec data prepare {named}: holds no document\n")
        );
    }
    assert!(
        !case.join("new").exists(),
        "the directories the run created are gone"
    );
    let left: Vec<_> = std::fs::read_dir(&existing).unwrap().collect();
    assert!(left.is_empty(), "an existing --out keeps no file: {left:?}");
    let mut corpus_files: Vec<String> = std::fs::read_dir(&corpus)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    corpus_files.sort();
    assert_eq!(corpus_files, ["wiki.train.raw", "wiki.valid.raw"]);
}

#[test]
fn refusals_no_out_exits_2() {
    let output = modelspec(&["data", "prepare", FIXTURE]);
    assert_eq!(output.status.code(), Some(2), "stderr: {}", stderr(&output));
    assert!(stderr(&output).contains("--out"), "{}", stderr(&output));
}
