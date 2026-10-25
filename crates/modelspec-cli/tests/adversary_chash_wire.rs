//! Adversary pass 1 on story:content-hash-schema-constrained: the newtypes
//! `modelspec.data.SequenceLength` and `modelspec.data.VocabularySize` on the wire. The records
//! `data pack` writes carry them as bare integers, the generated schemas and `mlm::load` refuse
//! the same malformed values, and the schemas' `minimum` 1 and 6 sit exactly where the writer
//! and the reader draw the line.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use modelspec_runtime::dataset::mlm;
use serde_json::{Value, json};

const CORPUS: &str = "crates/modelspec-runtime/tests/fixtures/corpus-tokenizer";
const PACKED_SCHEMA: &str =
    "crates/modelspec-runtime/schema/entities/modelspec.data.PackedDataset.schema.json";
const BINDING_SCHEMA: &str =
    "crates/modelspec-runtime/schema/entities/modelspec.data.TrainingDataBinding.schema.json";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("adversary_chash_{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(root())
        .output()
        .expect("modelspec runs")
}

fn ok(args: &[&str]) {
    let output = modelspec(args);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{args:?}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn s(path: &Path) -> &str {
    path.to_str().unwrap()
}

fn json_of(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn validator(path: &str) -> jsonschema::Validator {
    jsonschema::validator_for(&json_of(&root().join(path))).expect("the schema compiles")
}

/// Prepares the fixture corpus, trains a 300-token tokenizer and packs at `seq_len` into
/// `<case>/packed`.
fn packed(case: &Path, seq_len: &str) -> PathBuf {
    let dataset = case.join("dataset");
    ok(&["data", "prepare", CORPUS, "--out", s(&dataset)]);
    let tokenizer = case.join("tokenizer.json");
    ok(&[
        "data",
        "tokenizer",
        "--dataset",
        s(&dataset),
        "--vocab-size",
        "300",
        "--out",
        s(&tokenizer),
    ]);
    let out = case.join("packed");
    ok(&[
        "data",
        "pack",
        "--dataset",
        s(&dataset),
        "--tokenizer",
        s(&tokenizer),
        "--seq-len",
        seq_len,
        "--out",
        s(&out),
    ]);
    out
}

/// Writes `text` with `from` replaced by `to` as the record `file` of `dir`, and returns what
/// `mlm::load` says of the directory then.
fn load_with(dir: &Path, file: &str, text: &str, from: &str, to: &str) -> Result<(), String> {
    assert!(text.contains(from), "{file} holds {from}");
    fs::write(dir.join(file), text.replacen(from, to, 1)).unwrap();
    let result = mlm::load(dir).map(|_| ()).map_err(|e| e.to_string());
    fs::write(dir.join(file), text).unwrap();
    result
}

/// Attack 3: both records carry `seq_len` (and `PackedDataset.vocabulary_size`) as bare JSON
/// integers, not an object or a string around the newtype; a string in their place is refused
/// by the schema and, for `packed.json`, by `mlm::load`.
#[test]
fn adv_chash_newtypes_are_bare_integers_and_a_string_is_refused() {
    let case = scratch("wire");
    let dir = packed(&case, "16");
    let packed_text = fs::read_to_string(dir.join(mlm::PACKED_FILE)).unwrap();
    let binding_text = fs::read_to_string(dir.join(mlm::BINDING_FILE)).unwrap();
    assert!(packed_text.contains("\"seq_len\": 16,"), "{packed_text}");
    assert!(
        packed_text.contains("\"vocabulary_size\": 300,"),
        "{packed_text}"
    );
    assert!(binding_text.contains("\"seq_len\": 16,"), "{binding_text}");
    assert!(mlm::load(&dir).is_ok(), "the packed directory loads");

    let packed_schema = validator(PACKED_SCHEMA);
    let binding_schema = validator(BINDING_SCHEMA);
    let packed_record: Value = serde_json::from_str(&packed_text).unwrap();
    let binding_record: Value = serde_json::from_str(&binding_text).unwrap();
    assert!(packed_schema.is_valid(&packed_record));
    assert!(binding_schema.is_valid(&binding_record));

    for (field, number, string) in [
        ("seq_len", "\"seq_len\": 16,", "\"seq_len\": \"16\","),
        (
            "vocabulary_size",
            "\"vocabulary_size\": 300,",
            "\"vocabulary_size\": \"300\",",
        ),
    ] {
        let mut broken = packed_record.clone();
        broken[field] = json!(broken[field].to_string());
        assert!(
            !packed_schema.is_valid(&broken),
            "PackedDataset {field} string"
        );
        let refused = load_with(&dir, mlm::PACKED_FILE, &packed_text, number, string);
        assert!(refused.is_err(), "load of PackedDataset {field} string");
    }
    let mut broken = binding_record.clone();
    broken["seq_len"] = json!("16");
    assert!(
        !binding_schema.is_valid(&broken),
        "TrainingDataBinding seq_len string"
    );
    for value in [json!(-1), json!(1.5), json!(null), json!({"value": 16})] {
        let mut broken = packed_record.clone();
        broken["seq_len"] = value.clone();
        assert!(
            !packed_schema.is_valid(&broken),
            "PackedDataset seq_len {value}"
        );
    }
}

/// Attacks 4 and 7: the bounds sit where the writer and `mlm::load` put them. `data pack
/// --seq-len 1` writes records the schemas accept and that load; `seq_len` 0 is refused by both
/// schemas and by `load`; `vocabulary_size` 6 is accepted and 5 refused by the schema, as by
/// `load`. A `minimum` of 0 or 2, or 5 or 7, fails here.
#[test]
fn adv_chash_schema_minimums_sit_at_the_writer_and_reader_bounds() {
    let case = scratch("bounds");
    let dir = packed(&case, "1");
    assert!(mlm::load(&dir).is_ok(), "a seq_len 1 directory loads");
    let packed_text = fs::read_to_string(dir.join(mlm::PACKED_FILE)).unwrap();
    let packed_record = json_of(&dir.join(mlm::PACKED_FILE));
    let binding_record = json_of(&dir.join(mlm::BINDING_FILE));
    assert_eq!(packed_record["seq_len"], 1);
    assert_eq!(binding_record["seq_len"], 1);
    let packed_schema = validator(PACKED_SCHEMA);
    let binding_schema = validator(BINDING_SCHEMA);
    assert!(
        packed_schema.is_valid(&packed_record),
        "PackedDataset seq_len 1"
    );
    assert!(
        binding_schema.is_valid(&binding_record),
        "TrainingDataBinding seq_len 1"
    );

    let mut zero = packed_record.clone();
    zero["seq_len"] = json!(0);
    assert!(!packed_schema.is_valid(&zero), "PackedDataset seq_len 0");
    let mut zero = binding_record.clone();
    zero["seq_len"] = json!(0);
    assert!(
        !binding_schema.is_valid(&zero),
        "TrainingDataBinding seq_len 0"
    );
    let refused = load_with(
        &dir,
        mlm::PACKED_FILE,
        &packed_text,
        "\"seq_len\": 1,",
        "\"seq_len\": 0,",
    );
    assert!(refused.is_err(), "load of seq_len 0");

    let mut six = packed_record.clone();
    six["vocabulary_size"] = json!(6);
    assert!(
        packed_schema.is_valid(&six),
        "PackedDataset vocabulary_size 6"
    );
    let mut five = packed_record.clone();
    five["vocabulary_size"] = json!(5);
    assert!(
        !packed_schema.is_valid(&five),
        "PackedDataset vocabulary_size 5"
    );
    let refused = load_with(
        &dir,
        mlm::PACKED_FILE,
        &packed_text,
        "\"vocabulary_size\": 300,",
        "\"vocabulary_size\": 5,",
    );
    assert!(refused.is_err(), "load of vocabulary_size 5");
}

/// Attack 3: `mlm::load` refuses a `vocabulary_size` above 2^31 (an `i32` id addresses no more)
/// and a `seq_len` written as `16.0`; the generated schema admits both, so "validates against
/// the schema" still says more than the reader accepts at these two edges.
#[test]
#[ignore = "ADV-CHASH-1: the PackedDataset schema accepts vocabulary_size 2^31 + 1 and seq_len 16.0, which mlm::load refuses (pre-existing; no upper bound in spec/domains/data.yaml, and JSON Schema `integer` admits 16.0)"]
fn adv_chash_schema_refuses_what_the_loader_refuses_at_the_upper_edge() {
    let case = scratch("upper");
    let dir = packed(&case, "16");
    let packed_text = fs::read_to_string(dir.join(mlm::PACKED_FILE)).unwrap();
    let packed_record: Value = serde_json::from_str(&packed_text).unwrap();
    let packed_schema = validator(PACKED_SCHEMA);
    let mut accepted = Vec::new();
    for (pointer, value, from, to) in [
        (
            "vocabulary_size",
            json!((1u64 << 31) + 1),
            "\"vocabulary_size\": 300,",
            "\"vocabulary_size\": 2147483649,",
        ),
        (
            "seq_len",
            json!(16.0),
            "\"seq_len\": 16,",
            "\"seq_len\": 16.0,",
        ),
    ] {
        let refused = load_with(&dir, mlm::PACKED_FILE, &packed_text, from, to);
        assert!(refused.is_err(), "load of {pointer} {value}");
        let mut broken = packed_record.clone();
        broken[pointer] = value.clone();
        if packed_schema.is_valid(&broken) {
            accepted.push(format!("{pointer} {value}"));
        }
    }
    assert!(
        accepted.is_empty(),
        "the schema accepts what load refuses: {accepted:?}"
    );
}
