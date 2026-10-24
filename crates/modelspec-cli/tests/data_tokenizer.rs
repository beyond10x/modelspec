//! `modelspec data tokenizer --dataset <dir> --vocab-size <n> --out <file>` (story:bert-tokenizer):
//! a WordPiece tokenizer trained on the `Train` split of a dataset `data prepare` wrote, recorded
//! as a `Tokenizer` that validates against the generated schema (`spec/domains/data.yaml`).
//!
//! The corpus is the fixture under `crates/modelspec-runtime/tests/fixtures/corpus/`, original
//! text in the release's layout, so no test needs the corpus or the network.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

const FIXTURE: &str = "crates/modelspec-runtime/tests/fixtures/corpus";
const SCHEMA: &str =
    "crates/modelspec-runtime/schema/entities/modelspec.data.Tokenizer.schema.json";
/// The special tokens and their ids (coordinator decision D19).
const SPECIALS: [(&str, &str, u64); 5] = [
    ("[PAD]", "pad_token_id", 0),
    ("[UNK]", "unk_token_id", 1),
    ("[CLS]", "cls_token_id", 2),
    ("[SEP]", "sep_token_id", 3),
    ("[MASK]", "mask_token_id", 4),
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh, empty directory under this crate's build directory, named for the test.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("data_tokenizer_{name}"));
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

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn assert_exit(output: &Output, code: i32) {
    assert_eq!(
        output.status.code(),
        Some(code),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        stderr(output)
    );
}

/// The fixture prepared by `data prepare` into `<case>/dataset`.
fn dataset(case: &Path) -> PathBuf {
    let out = case.join("dataset");
    assert_exit(
        &modelspec(&["data", "prepare", FIXTURE, "--out", out.to_str().unwrap()]),
        0,
    );
    out
}

fn tokenizer(dataset: &Path, size: &str, out: &Path) -> Output {
    modelspec(&[
        "data",
        "tokenizer",
        "--dataset",
        dataset.to_str().unwrap(),
        "--vocab-size",
        size,
        "--out",
        out.to_str().unwrap(),
    ])
}

fn schema() -> jsonschema::Validator {
    let text = std::fs::read_to_string(repo_root().join(SCHEMA)).expect("the schema is generated");
    let schema: Value = serde_json::from_str(&text).expect("the schema is JSON");
    jsonschema::validator_for(&schema).expect("the schema compiles")
}

fn record(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).expect("the record is written");
    assert!(text.ends_with('\n'), "the record ends with a newline");
    serde_json::from_str(&text).expect("the record is JSON")
}

/// Scenario 2: `--vocab-size 300` on the fixture dataset writes exactly 300 tokens, no
/// duplicates, every special token at its settled id, a record that validates against the
/// generated schema, and two runs write identical bytes.
#[test]
fn exact_size() {
    let case = scratch("exact_size");
    let dataset = dataset(&case);
    let first = case.join("first.json");
    let second = case.join("second.json");
    let output = tokenizer(&dataset, "300", &first);
    assert_exit(&output, 0);
    assert_exit(&tokenizer(&dataset, "300", &second), 0);

    let record = record(&first);
    let errors: Vec<String> = schema()
        .iter_errors(&record)
        .map(|e| e.to_string())
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(record["kind"], "WordPiece");
    assert_eq!(record["state"], "Recorded");

    let vocabulary = &record["vocabulary"];
    let tokens: Vec<&str> = vocabulary["tokens"]
        .as_array()
        .expect("tokens is an array")
        .iter()
        .map(|t| t.as_str().expect("a token is a string"))
        .collect();
    assert_eq!(tokens.len(), 300, "exactly the requested size");
    let distinct: HashSet<&str> = tokens.iter().copied().collect();
    assert_eq!(distinct.len(), tokens.len(), "no duplicate token");
    for (token, field, id) in SPECIALS {
        assert_eq!(vocabulary[field], id, "{field}");
        assert_eq!(tokens[id as usize], token, "{token} at id {id}");
    }

    let id = record["tokenizer_id"].as_str().unwrap();
    assert_eq!(&id[14..15], "8", "{id} is a version-8 UUID");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout, format!("tokenizer {id} (300 tokens)\n"));

    assert_eq!(
        std::fs::read(&first).unwrap(),
        std::fs::read(&second).unwrap(),
        "two runs write byte-identical records"
    );
}

/// Scenario 5: `--vocab-size 100000` exits 1 naming the size reached; that size trains, one more
/// does not.
#[test]
fn too_small() {
    let case = scratch("too_small");
    let dataset = dataset(&case);
    let out = case.join("tokenizer.json");
    let output = tokenizer(&dataset, "100000", &out);
    assert_exit(&output, 1);
    let message = stderr(&output);
    let split = dataset.join("train.raw").display().to_string();
    let reason = message
        .strip_prefix(&format!("modelspec data tokenizer {split}: "))
        .unwrap_or_else(|| panic!("names the Train split: {message}"));
    assert!(!message.contains("E-"), "no diagnostic code: {message}");
    assert!(reason.contains("100000"), "names the request: {message}");
    assert!(!out.exists(), "a refused run writes no record");
    let reached: Vec<u64> = reason
        .split(|c: char| !c.is_ascii_digit())
        .filter_map(|n| n.parse().ok())
        .filter(|n| *n != 100000)
        .collect();
    let [reached] = reached[..] else {
        panic!("one size reached in {message}")
    };
    assert!(reached > 5 && reached < 100000, "{message}");

    assert_exit(&tokenizer(&dataset, &reached.to_string(), &out), 0);
    let tokens = record(&out)["vocabulary"]["tokens"]
        .as_array()
        .unwrap()
        .len();
    assert_eq!(tokens as u64, reached);
    let more = tokenizer(
        &dataset,
        &(reached + 1).to_string(),
        &case.join("more.json"),
    );
    assert_exit(&more, 1);
    assert!(
        stderr(&more).contains(&format!(" {reached} ")),
        "{}",
        stderr(&more)
    );
}

#[test]
fn refusals_not_a_dataset_exits_1_naming_it() {
    let case = scratch("refusals_not_a_dataset");
    let out = case.join("tokenizer.json");
    let output = tokenizer(&case, "300", &out);
    assert_exit(&output, 1);
    let named = case.join("dataset.json").display().to_string();
    assert!(
        stderr(&output).starts_with(&format!("modelspec data tokenizer {named}: ")),
        "{}",
        stderr(&output)
    );
    assert!(!out.exists());
}

#[test]
fn refusals_usage_exits_2() {
    for args in [
        &[
            "data",
            "tokenizer",
            "--vocab-size",
            "300",
            "--out",
            "x.json",
        ][..],
        &["data", "tokenizer", "--dataset", ".", "--out", "x.json"][..],
        &["data", "tokenizer", "--dataset", ".", "--vocab-size", "300"][..],
        &[
            "data",
            "tokenizer",
            "--dataset",
            ".",
            "--vocab-size",
            "many",
            "--out",
            "x.json",
        ][..],
    ] {
        let output = modelspec(args);
        assert_exit(&output, 2);
    }
}
