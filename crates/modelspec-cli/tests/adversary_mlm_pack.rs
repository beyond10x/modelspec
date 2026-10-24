//! Adversary pass 1 on story:bert-mlm-batches: `modelspec data pack` at its boundaries, its
//! refusals and its records against the generated schemas (`spec/domains/data.yaml`,
//! `TrainingDataBinding` and `PackedDataset`; `spec/domains/training.yaml`, `TrainingRun`).
//!
//! The corpus is the fixture under `crates/modelspec-runtime/tests/fixtures/corpus-tokenizer/`.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

const FIXTURE: &str = "crates/modelspec-runtime/tests/fixtures/corpus-tokenizer";
const PACKED_SCHEMA: &str =
    "crates/modelspec-runtime/schema/entities/modelspec.data.PackedDataset.schema.json";
const BINDING_SCHEMA: &str =
    "crates/modelspec-runtime/schema/entities/modelspec.data.TrainingDataBinding.schema.json";
const RUN_SCHEMA: &str =
    "crates/modelspec-runtime/schema/entities/modelspec.training.TrainingRun.schema.json";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("adversary_mlm_{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch directory");
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

fn s(path: &Path) -> &str {
    path.to_str().unwrap()
}

fn inputs(case: &Path) -> (PathBuf, PathBuf) {
    let dataset = case.join("dataset");
    assert_exit(
        &modelspec(&["data", "prepare", FIXTURE, "--out", s(&dataset)]),
        0,
    );
    let tokenizer = case.join("tokenizer.json");
    assert_exit(
        &modelspec(&[
            "data",
            "tokenizer",
            "--dataset",
            s(&dataset),
            "--vocab-size",
            "300",
            "--out",
            s(&tokenizer),
        ]),
        0,
    );
    (dataset, tokenizer)
}

fn pack(dataset: &Path, tokenizer: &Path, seq_len: &str, out: &Path) -> Output {
    modelspec(&[
        "data",
        "pack",
        "--dataset",
        s(dataset),
        "--tokenizer",
        s(tokenizer),
        "--seq-len",
        seq_len,
        "--out",
        s(out),
    ])
}

fn listing(dir: &Path) -> BTreeSet<String> {
    fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

fn json(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn validator(path: &str) -> jsonschema::Validator {
    let schema: Value = json(&repo_root().join(path));
    jsonschema::validator_for(&schema).expect("the schema compiles")
}

/// Attack 7: a `--seq-len` that is negative, empty, fractional, past `u64` or not a number is a
/// usage error, exit 2, and nothing is created.
#[test]
fn adv_pack_seq_len_usage_errors_exit_2() {
    let case = scratch("seq_len_usage");
    let out = case.join("out");
    for value in [
        "-1",
        "",
        "1.5",
        "abc",
        "18446744073709551616",
        "0x10",
        " 16",
    ] {
        let output = pack(&case, &case.join("t.json"), value, &out);
        assert_exit(&output, 2);
        assert!(!out.exists(), "--seq-len {value:?} created --out");
    }
}

/// Attack 7: a dataset whose `heldout.raw` is missing exits 1 naming it, and an existing,
/// non-empty `--out` is left exactly as it was: no `.partial`, the earlier file untouched.
#[test]
fn adv_pack_missing_heldout_leaves_out_as_it_was() {
    let case = scratch("missing_heldout");
    let (dataset, tokenizer) = inputs(&case);
    let broken = case.join("broken");
    fs::create_dir_all(&broken).unwrap();
    for file in ["dataset.json", "train.raw"] {
        fs::copy(dataset.join(file), broken.join(file)).unwrap();
    }
    let out = case.join("out");
    fs::create_dir_all(&out).unwrap();
    fs::write(out.join("keep.txt"), "keep").unwrap();
    fs::write(out.join("train.tokens"), "old").unwrap();
    let output = pack(&broken, &tokenizer, "16", &out);
    assert_exit(&output, 1);
    assert!(
        stderr(&output).contains("heldout.raw"),
        "{}",
        stderr(&output)
    );
    assert_eq!(
        listing(&out),
        BTreeSet::from(["keep.txt".to_string(), "train.tokens".to_string()])
    );
    assert_eq!(fs::read(out.join("train.tokens")).unwrap(), b"old");
}

/// Attack 7: a tokenizer record whose id is not the hash of its tokens exits 1 naming the file.
#[test]
fn adv_pack_tokenizer_id_mismatch_exits_1() {
    let case = scratch("tokenizer_id");
    let (dataset, tokenizer) = inputs(&case);
    let mut record = json(&tokenizer);
    let id = record["tokenizer_id"].as_str().unwrap().to_string();
    let last = if id.ends_with('0') { "1" } else { "0" };
    record["tokenizer_id"] = Value::String(format!("{}{last}", &id[..id.len() - 1]));
    let altered = case.join("altered.json");
    fs::write(&altered, serde_json::to_string_pretty(&record).unwrap()).unwrap();
    let out = case.join("out");
    let output = pack(&dataset, &altered, "16", &out);
    assert_exit(&output, 1);
    assert!(
        stderr(&output).starts_with(&format!("modelspec data pack {}: ", altered.display())),
        "{}",
        stderr(&output)
    );
    assert!(!out.exists());
}

/// Attack 7: `--out` inside `--dataset`, and `--out` the dataset itself, pack and leave the
/// dataset packable with the same result.
#[test]
fn adv_pack_out_inside_the_dataset() {
    let case = scratch("out_inside");
    let (dataset, tokenizer) = inputs(&case);
    let record = fs::read(dataset.join("dataset.json")).unwrap();
    let inside = dataset.join("packed");
    assert_exit(&pack(&dataset, &tokenizer, "16", &inside), 0);
    assert_exit(&pack(&dataset, &tokenizer, "16", &dataset), 0);
    assert_eq!(fs::read(dataset.join("dataset.json")).unwrap(), record);
    let again = case.join("again");
    assert_exit(&pack(&dataset, &tokenizer, "16", &again), 0);
    for file in [
        "train.tokens",
        "heldout.tokens",
        "packed.json",
        "binding.json",
    ] {
        assert_eq!(
            fs::read(inside.join(file)).unwrap(),
            fs::read(again.join(file)).unwrap(),
            "{file}"
        );
        assert_eq!(
            fs::read(dataset.join(file)).unwrap(),
            fs::read(again.join(file)).unwrap(),
            "{file}"
        );
    }
}

/// Attack 4: `S = 1` keeps the whole stream; at `S = 16` exactly its last `L mod 16` ids are
/// dropped; `S` equal to the shorter stream packs one sequence of it, and one more is refused.
#[test]
fn adv_pack_tail_is_exactly_the_last_ids() {
    let case = scratch("tail");
    let (dataset, tokenizer) = inputs(&case);
    let one = case.join("one");
    assert_exit(&pack(&dataset, &tokenizer, "1", &one), 0);
    let sixteen = case.join("sixteen");
    assert_exit(&pack(&dataset, &tokenizer, "16", &sixteen), 0);
    let record = json(&one.join("packed.json"));
    assert!(validator(PACKED_SCHEMA).is_valid(&record));
    let mut lengths = Vec::new();
    for (s_, file) in ["train.tokens", "heldout.tokens"].iter().enumerate() {
        let whole = fs::read(one.join(file)).unwrap();
        let cut = fs::read(sixteen.join(file)).unwrap();
        let ids = whole.len() / 4;
        assert_eq!(record["splits"][s_]["sequence_count"], ids as u64, "{file}");
        assert_eq!(cut.len() / 4, ids / 16 * 16, "{file}: the tail is L mod 16");
        assert_eq!(
            cut[..],
            whole[..cut.len()],
            "{file}: the kept ids are the first ones"
        );
        lengths.push(ids);
    }
    let shorter = lengths.iter().min().copied().unwrap();
    let exact = case.join("exact");
    assert_exit(&pack(&dataset, &tokenizer, &shorter.to_string(), &exact), 0);
    let past = case.join("past");
    let output = pack(&dataset, &tokenizer, &(shorter + 1).to_string(), &past);
    assert_exit(&output, 1);
    assert!(!past.exists());
}

/// Attack 8: `TrainingRun.data_binding_id` is optional in the generated schema.
#[test]
fn adv_training_run_data_binding_id_is_optional() {
    let schema = json(&repo_root().join(RUN_SCHEMA));
    assert!(schema["properties"]["data_binding_id"].is_object());
    let required: Vec<&str> = schema["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(!required.contains(&"data_binding_id"), "{required:?}");
}

/// Attack 8: the entity invariants of `spec/domains/data.yaml` (`seq_len > 0` on both records,
/// `vocabulary_size > 5` and `heldout_unigram_baseline_nats > 0` on `PackedDataset`) are in the
/// generated schemas, so a record that breaks one does not validate.
#[test]
#[ignore = "ADV-MLM-3: the entity schemas project none of the entity invariants"]
fn adv_records_breaking_the_spec_invariants_do_not_validate() {
    let case = scratch("invariants");
    let (dataset, tokenizer) = inputs(&case);
    let out = case.join("out");
    assert_exit(&pack(&dataset, &tokenizer, "16", &out), 0);
    let packed = json(&out.join("packed.json"));
    let binding = json(&out.join("binding.json"));
    let packed_validator = validator(PACKED_SCHEMA);
    let binding_validator = validator(BINDING_SCHEMA);
    assert!(packed_validator.is_valid(&packed));
    assert!(binding_validator.is_valid(&binding));
    let mut accepted = Vec::new();
    for (field, value) in [
        ("seq_len", serde_json::json!(0)),
        ("vocabulary_size", serde_json::json!(5)),
        ("heldout_unigram_baseline_nats", serde_json::json!(-1.0)),
    ] {
        let mut broken = packed.clone();
        broken[field] = value.clone();
        if packed_validator.is_valid(&broken) {
            accepted.push(format!("PackedDataset {field} {value}"));
        }
    }
    let mut broken = binding.clone();
    broken["seq_len"] = serde_json::json!(0);
    if binding_validator.is_valid(&broken) {
        accepted.push("TrainingDataBinding seq_len 0".into());
    }
    assert!(
        accepted.is_empty(),
        "validate against the schema: {accepted:?}"
    );
}
