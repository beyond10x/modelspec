//! `modelspec data pack --dataset <dir> --tokenizer <file> --seq-len <S> --out <dir>`
//! (story:bert-mlm-batches): the splits of a prepared dataset encoded, their documents joined
//! with `[SEP]`, cut into sequences of exactly `S` tokens, and recorded with the held-out unigram
//! baseline as a `PackedDataset` and a `TrainingDataBinding` that validate against the generated
//! schemas (`spec/domains/data.yaml`, coordinator decision D26).
//!
//! The corpus is the fixture under `crates/modelspec-runtime/tests/fixtures/corpus-tokenizer/`,
//! original text in the release's layout, so no test needs the corpus or the network.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use modelspec_runtime::dataset::tokenizer::Tokenizer;
use serde_json::Value;

const FIXTURE: &str = "crates/modelspec-runtime/tests/fixtures/corpus-tokenizer";
const PACKED_SCHEMA: &str =
    "crates/modelspec-runtime/schema/entities/modelspec.data.PackedDataset.schema.json";
const BINDING_SCHEMA: &str =
    "crates/modelspec-runtime/schema/entities/modelspec.data.TrainingDataBinding.schema.json";
/// The files `data pack` writes.
const FILES: [&str; 4] = [
    "train.tokens",
    "heldout.tokens",
    "binding.json",
    "packed.json",
];
/// The split files of a prepared dataset and the packed file of each, in record order.
const SPLITS: [(&str, &str, &str); 2] = [
    ("Train", "train.raw", "train.tokens"),
    ("HeldOut", "heldout.raw", "heldout.tokens"),
];
/// `[SEP]`'s id (coordinator decision D19), the separator D26 settles.
const SEP: u32 = 3;
/// Ids below this are the special tokens.
const SPECIALS: u32 = 5;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh, empty directory under this crate's build directory, named for the test.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("data_pack_{name}"));
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

/// The fixture prepared into `<case>/dataset` and a 300-token tokenizer trained on it into
/// `<case>/tokenizer.json`.
fn inputs(case: &Path) -> (PathBuf, PathBuf) {
    let dataset = case.join("dataset");
    assert_exit(
        &modelspec(&[
            "data",
            "prepare",
            FIXTURE,
            "--out",
            dataset.to_str().unwrap(),
        ]),
        0,
    );
    let tokenizer = case.join("tokenizer.json");
    assert_exit(
        &modelspec(&[
            "data",
            "tokenizer",
            "--dataset",
            dataset.to_str().unwrap(),
            "--vocab-size",
            "300",
            "--out",
            tokenizer.to_str().unwrap(),
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
        dataset.to_str().unwrap(),
        "--tokenizer",
        tokenizer.to_str().unwrap(),
        "--seq-len",
        seq_len,
        "--out",
        out.to_str().unwrap(),
    ])
}

fn validator(path: &str) -> jsonschema::Validator {
    let text = std::fs::read_to_string(repo_root().join(path)).expect("the schema is generated");
    let schema: Value = serde_json::from_str(&text).expect("the schema is JSON");
    jsonschema::validator_for(&schema).expect("the schema compiles")
}

fn record(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).expect("the record is written");
    assert!(text.ends_with('\n'), "the record ends with a newline");
    serde_json::from_str(&text).expect("the record is JSON")
}

/// The record `file` lists its top-level fields in the order of the schema's `required` list.
fn assert_schema_order(schema: &str, file: &Path) {
    let text = std::fs::read_to_string(repo_root().join(schema)).unwrap();
    let schema: Value = serde_json::from_str(&text).unwrap();
    let record = std::fs::read_to_string(file).unwrap();
    let at: Vec<usize> = schema["required"]
        .as_array()
        .expect("the schema lists required fields")
        .iter()
        .map(|k| {
            let key = format!("\n  \"{}\":", k.as_str().unwrap());
            record
                .find(&key)
                .unwrap_or_else(|| panic!("{key:?} in {}", file.display()))
        })
        .collect();
    assert!(
        at.is_sorted(),
        "{} in schema order: {record}",
        file.display()
    );
}

fn assert_valid(validator: &jsonschema::Validator, record: &Value) {
    let errors: Vec<String> = validator
        .iter_errors(record)
        .map(|e| e.to_string())
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
}

/// The ids of a packed file: little-endian `i32`, no header.
fn packed_ids(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).expect("the packed file is written");
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::try_from(i32::from_le_bytes(*b)).expect("a token id is not negative"))
        .collect()
}

/// `line` without its terminator.
fn content(line: &str) -> &str {
    let line = line.strip_suffix('\n').unwrap_or(line);
    line.strip_suffix('\r').unwrap_or(line)
}

/// The documents of a split's text, written here from the settlement (`DatasetSplit`'s
/// document_count, coordinator decision D22) and not from the code under test: a document starts
/// at a line ` = <title> = ` with a blank line, a single space, directly before and after it, and
/// non-blank text before the first such line is a document too.
fn documents(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let blank = |i: usize| lines.get(i).is_some_and(|l| content(l) == " ");
    let heading = |i: usize| {
        let shaped = content(lines[i])
            .strip_prefix(" = ")
            .and_then(|r| r.strip_suffix(" = "))
            .is_some_and(|t| !t.starts_with('=') && !t.trim().is_empty());
        shaped && i > 0 && blank(i - 1) && blank(i + 1)
    };
    let mut starts: Vec<usize> = (0..lines.len()).filter(|&i| heading(i)).collect();
    let first = starts.first().copied().unwrap_or(lines.len());
    if lines[..first].iter().any(|l| !l.trim().is_empty()) {
        starts.insert(0, 0);
    }
    starts
        .iter()
        .enumerate()
        .map(|(d, &start)| {
            let end = starts.get(d + 1).copied().unwrap_or(lines.len());
            lines[start..end].concat()
        })
        .collect()
}

/// The encoded token stream of a split: its documents encoded and joined with `[SEP]` between
/// two of them, no `[CLS]` (coordinator decision D26).
fn token_stream(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    let codec = tokenizer.codec();
    let mut stream = Vec::new();
    for (d, document) in documents(text).iter().enumerate() {
        if d > 0 {
            stream.push(SEP);
        }
        stream.extend(codec.encode(document));
    }
    stream
}

fn read_tokenizer(path: &Path) -> Tokenizer {
    let text = std::fs::read_to_string(path).expect("the tokenizer is written");
    serde_json::from_str(&text).expect("the tokenizer record reads back")
}

/// Scenario 2: with `--seq-len 16` on the fixture every sequence has 16 tokens, their
/// concatenation equals the encoded token stream of the split without its dropped tail, the
/// record validates against the generated schema, and two runs write identical bytes.
#[test]
fn sequences() {
    let case = scratch("sequences");
    let (dataset, tokenizer_file) = inputs(&case);
    let first = case.join("first");
    let second = case.join("second");
    let output = pack(&dataset, &tokenizer_file, "16", &first);
    assert_exit(&output, 0);
    assert_exit(&pack(&dataset, &tokenizer_file, "16", &second), 0);

    let tokenizer = read_tokenizer(&tokenizer_file);
    let packed = record(&first.join("packed.json"));
    assert_valid(&validator(PACKED_SCHEMA), &packed);
    let binding = record(&first.join("binding.json"));
    assert_valid(&validator(BINDING_SCHEMA), &binding);
    assert_schema_order(PACKED_SCHEMA, &first.join("packed.json"));
    assert_schema_order(BINDING_SCHEMA, &first.join("binding.json"));
    assert_eq!(packed["seq_len"], 16);
    assert_eq!(packed["vocabulary_size"], 300);
    assert_eq!(packed["state"], "Recorded");
    assert_eq!(packed["binding_id"], binding["binding_id"]);
    let dataset_record = record(&dataset.join("dataset.json"));
    assert_eq!(binding["dataset_id"], dataset_record["dataset_id"]);
    assert_eq!(binding["tokenizer_id"], tokenizer.tokenizer_id.as_str());
    assert_eq!(binding["seq_len"], 16);
    assert_eq!(
        binding["masking_policy"],
        serde_json::json!({
            "selection_rate": 0.15,
            "mask_fraction": 0.8,
            "random_fraction": 0.1,
            "keep_fraction": 0.1
        })
    );
    for id in [&packed["packed_id"], &binding["binding_id"]] {
        let id = id.as_str().unwrap();
        assert_eq!(&id[14..15], "8", "{id} is a version-8 UUID");
    }

    for (s, (role, raw, file)) in SPLITS.iter().enumerate() {
        let split = &packed["splits"][s];
        assert_eq!(split["role"], *role);
        let text = std::fs::read_to_string(dataset.join(raw)).unwrap();
        let stream = token_stream(&tokenizer, &text);
        let documents = dataset_record["splits"][s]["document_count"]
            .as_u64()
            .unwrap();
        let separators = stream.iter().filter(|&&id| id == SEP).count() as u64;
        assert_eq!(
            separators,
            documents - 1,
            "{role}: one [SEP] between two documents"
        );
        let ids = packed_ids(&first.join(file));
        let count = split["sequence_count"].as_u64().unwrap() as usize;
        assert!(count > 0, "{role} holds a sequence");
        assert_eq!(
            ids.len(),
            count * 16,
            "{role}: every sequence has 16 tokens"
        );
        assert_eq!(
            count,
            stream.len() / 16,
            "{role}: only the partial tail is dropped"
        );
        assert_eq!(ids[..], stream[..count * 16], "{role}: the encoded stream");
        assert!(ids.iter().all(|&id| id < 300));
    }

    for file in FILES {
        assert_eq!(
            std::fs::read(first.join(file)).unwrap(),
            std::fs::read(second.join(file)).unwrap(),
            "two runs write identical {file}"
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let id = packed["packed_id"].as_str().unwrap();
    assert!(
        stdout.starts_with(&format!("packed {id} (seq_len 16)\n")),
        "{stdout}"
    );
}

/// Scenario 6: the baseline in the record equals the value computed here from the fixture
/// counts within 1e-9: the cross-entropy in nats of the held-out tokens that are not special
/// under the Train unigram frequencies with add-one smoothing over the 295 tokens that are not
/// special.
#[test]
fn baseline() {
    let case = scratch("baseline");
    let (dataset, tokenizer_file) = inputs(&case);
    let out = case.join("packed");
    let output = pack(&dataset, &tokenizer_file, "16", &out);
    assert_exit(&output, 0);
    let tokenizer = read_tokenizer(&tokenizer_file);
    let packed_stream = |raw: &str| {
        let text = std::fs::read_to_string(dataset.join(raw)).unwrap();
        let mut stream = token_stream(&tokenizer, &text);
        stream.truncate(stream.len() / 16 * 16);
        stream
    };
    let train = packed_stream("train.raw");
    let heldout = packed_stream("heldout.raw");

    let mut counts: HashMap<u32, u64> = HashMap::new();
    let mut total = 0u64;
    for &id in train.iter().filter(|&&id| id >= SPECIALS) {
        *counts.entry(id).or_default() += 1;
        total += 1;
    }
    let types = (300 - SPECIALS) as f64;
    let measured: Vec<u32> = heldout.into_iter().filter(|&id| id >= SPECIALS).collect();
    assert!(!measured.is_empty());
    let nats = -measured
        .iter()
        .map(|id| {
            let c = counts.get(id).copied().unwrap_or(0) as f64;
            ((c + 1.0) / (total as f64 + types)).ln()
        })
        .sum::<f64>()
        / measured.len() as f64;

    let recorded = record(&out.join("packed.json"))["heldout_unigram_baseline_nats"]
        .as_f64()
        .expect("the baseline is a number");
    assert!(
        (recorded - nats).abs() <= 1e-9,
        "recorded {recorded}, computed {nats}"
    );
    assert!(nats > 0.0 && nats < types.ln() + 1.0, "{nats}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains(&format!("heldout unigram baseline {recorded} nats\n")),
        "{stdout}"
    );
}

#[test]
fn refusals_exit_1_naming_the_path_and_write_nothing() {
    let case = scratch("refusals");
    let (dataset, tokenizer_file) = inputs(&case);

    let out = case.join("not-a-dataset");
    let output = pack(&case, &tokenizer_file, "16", &out);
    assert_exit(&output, 1);
    let named = case.join("dataset.json").display().to_string();
    assert!(
        stderr(&output).starts_with(&format!("modelspec data pack {named}: ")),
        "{}",
        stderr(&output)
    );
    assert!(!out.exists(), "a refused run creates no directory");

    let missing = case.join("missing.json");
    let output = pack(&dataset, &missing, "16", &out);
    assert_exit(&output, 1);
    assert!(
        stderr(&output).starts_with(&format!("modelspec data pack {}: ", missing.display())),
        "{}",
        stderr(&output)
    );

    let output = pack(&dataset, &tokenizer_file, "100000", &out);
    assert_exit(&output, 1);
    let message = stderr(&output);
    assert!(message.contains("100000"), "names the length: {message}");
    assert!(!out.exists(), "a refused run writes nothing");

    let altered = case.join("altered");
    std::fs::create_dir_all(&altered).unwrap();
    for file in ["dataset.json", "train.raw", "heldout.raw"] {
        std::fs::copy(dataset.join(file), altered.join(file)).unwrap();
    }
    std::fs::write(altered.join("heldout.raw"), " \n = Other = \n \n text . \n").unwrap();
    let output = pack(&altered, &tokenizer_file, "4", &out);
    assert_exit(&output, 1);
    assert!(
        stderr(&output).contains("heldout.raw"),
        "names the altered split: {}",
        stderr(&output)
    );
    assert!(!out.exists(), "a refused run writes nothing");
}

#[test]
fn refusals_usage_exits_2() {
    for args in [
        &[
            "data",
            "pack",
            "--tokenizer",
            "t.json",
            "--seq-len",
            "16",
            "--out",
            "o",
        ][..],
        &[
            "data",
            "pack",
            "--dataset",
            ".",
            "--seq-len",
            "16",
            "--out",
            "o",
        ][..],
        &[
            "data",
            "pack",
            "--dataset",
            ".",
            "--tokenizer",
            "t.json",
            "--out",
            "o",
        ][..],
        &[
            "data",
            "pack",
            "--dataset",
            ".",
            "--tokenizer",
            "t.json",
            "--seq-len",
            "16",
        ][..],
        &[
            "data",
            "pack",
            "--dataset",
            ".",
            "--tokenizer",
            "t.json",
            "--seq-len",
            "0",
            "--out",
            "o",
        ][..],
    ] {
        assert_exit(&modelspec(args), 2);
    }
}
