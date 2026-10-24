//! Adversary pass 1 on story:bert-tokenizer: the surface of `modelspec data tokenizer` (attacks
//! 5, 7 and 8 of the brief). The corpus is the original-text fixture
//! `crates/modelspec-runtime/tests/fixtures/corpus/`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

const FIXTURE: &str = "crates/modelspec-runtime/tests/fixtures/corpus";
const SCHEMA: &str =
    "crates/modelspec-runtime/schema/entities/modelspec.data.Tokenizer.schema.json";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("adv_tokenizer_cli_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

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

/// The names in `dir`, sorted.
fn listing(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// `--vocab-size 0` and `4` exit 1 with the reason and write nothing; `-1` and `1e3` are usage
/// errors, exit 2.
#[test]
fn adv_cli_vocab_size_below_specials_and_malformed() {
    let case = scratch("below_specials");
    let dataset = dataset(&case);
    let out = case.join("tokenizer.json");
    for size in ["0", "4"] {
        let output = tokenizer(&dataset, size, &out);
        assert_exit(&output, 1);
        assert!(
            stderr(&output).contains("no room for the 5 special tokens"),
            "{}",
            stderr(&output)
        );
    }
    for size in ["-1", "1e3", "18446744073709551616"] {
        assert_exit(&tokenizer(&dataset, size, &out), 2);
    }
    assert_eq!(listing(&case), ["dataset"]);
}

/// `--out` an existing directory exits 1, leaves the directory as it was and no `.partial`.
#[test]
fn adv_cli_out_is_a_directory() {
    let case = scratch("out_dir");
    let dataset = dataset(&case);
    let out = case.join("taken");
    std::fs::create_dir(&out).unwrap();
    let output = tokenizer(&dataset, "300", &out);
    assert_exit(&output, 1);
    assert!(
        stderr(&output).starts_with(&format!("modelspec data tokenizer {}: ", out.display())),
        "{}",
        stderr(&output)
    );
    assert!(out.is_dir());
    assert_eq!(listing(&case), ["dataset", "taken"]);
}

/// `--dataset` naming a file exits 1 and writes nothing.
#[test]
fn adv_cli_dataset_is_a_file() {
    let case = scratch("dataset_file");
    let dataset = dataset(&case);
    let out = case.join("tokenizer.json");
    let output = tokenizer(&dataset.join("dataset.json"), "300", &out);
    assert_exit(&output, 1);
    assert!(!out.exists());
}

/// Attack 5: the `HeldOut` split is never opened. With `heldout.raw` removed the record is the
/// same bytes as with it.
#[test]
fn adv_cli_heldout_missing_trains_the_same_record() {
    let case = scratch("heldout_missing");
    let dataset = dataset(&case);
    let with = case.join("with.json");
    assert_exit(&tokenizer(&dataset, "300", &with), 0);
    std::fs::remove_file(dataset.join("heldout.raw")).unwrap();
    let without = case.join("without.json");
    assert_exit(&tokenizer(&dataset, "300", &without), 0);
    assert_eq!(
        std::fs::read(&with).unwrap(),
        std::fs::read(&without).unwrap()
    );
}

/// Attack 4: a `train.raw` that is not UTF-8 exits 1 naming the split, never a panic, and writes
/// nothing.
#[test]
fn adv_cli_train_split_not_utf8_exits_1() {
    let case = scratch("not_utf8");
    let dataset = dataset(&case);
    std::fs::write(dataset.join("train.raw"), b" = A = \n \n abc \xff def \n").unwrap();
    let out = case.join("tokenizer.json");
    let output = tokenizer(&dataset, "10", &out);
    assert_exit(&output, 1);
    let split = dataset.join("train.raw").display().to_string();
    assert!(
        stderr(&output).starts_with(&format!("modelspec data tokenizer {split}: ")),
        "{}",
        stderr(&output)
    );
    assert!(!out.exists());
}

/// `--out` inside `--dataset` works, and a second run replaces an existing record with the same
/// bytes.
#[test]
fn adv_cli_out_inside_dataset_and_existing() {
    let case = scratch("out_inside");
    let dataset = dataset(&case);
    let out = dataset.join("tokenizer.json");
    std::fs::write(&out, "stale").unwrap();
    assert_exit(&tokenizer(&dataset, "300", &out), 0);
    let first = std::fs::read(&out).unwrap();
    assert_ne!(first, b"stale");
    assert_exit(&tokenizer(&dataset, "300", &out), 0);
    assert_eq!(std::fs::read(&out).unwrap(), first);
    assert!(!dataset.join("tokenizer.json.partial").exists());
}

/// Attack 7: the generated schema holds the settled special ids as constants, so a record with
/// `[PAD]` at another id does not validate; the id is a version-8 UUID with the RFC variant, and
/// two sizes give two ids.
#[test]
fn adv_cli_record_against_the_schema() {
    let case = scratch("record_schema");
    let dataset = dataset(&case);
    let a = case.join("a.json");
    let b = case.join("b.json");
    assert_exit(&tokenizer(&dataset, "300", &a), 0);
    assert_exit(&tokenizer(&dataset, "299", &b), 0);
    let read =
        |p: &Path| -> Value { serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap() };
    let (ra, rb) = (read(&a), read(&b));
    let (ia, ib) = (
        ra["tokenizer_id"].as_str().unwrap(),
        rb["tokenizer_id"].as_str().unwrap(),
    );
    assert_ne!(ia, ib);
    for id in [ia, ib] {
        assert_eq!(&id[14..15], "8", "{id}");
        assert!("89ab".contains(&id[19..20]), "{id}");
    }

    let text = std::fs::read_to_string(repo_root().join(SCHEMA)).unwrap();
    let schema: Value = serde_json::from_str(&text).unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();
    assert!(validator.is_valid(&ra));
    for field in [
        "pad_token_id",
        "unk_token_id",
        "cls_token_id",
        "sep_token_id",
        "mask_token_id",
    ] {
        let mut wrong = ra.clone();
        let id = wrong["vocabulary"][field].as_u64().unwrap();
        wrong["vocabulary"][field] = Value::from(id + 1);
        assert!(!validator.is_valid(&wrong), "{field} off by one validates");
        let mut missing = ra.clone();
        missing["vocabulary"].as_object_mut().unwrap().remove(field);
        assert!(!validator.is_valid(&missing), "{field} missing validates");
    }
}
