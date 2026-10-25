//! Adversary pass 1 on story:content-hash-schema-constrained: every writer of a
//! `modelspec.artifact.ContentHash` (the compiled artifact's manifest and its kernel cache keys,
//! the checkpoint manifest, `dataset.json` and `packed.json`) writes values the generated
//! `ContentHash` schema accepts, and of the 64 digits the specification's `value.count == 64`
//! states, which the schema does not check (beyond10x/ess issue 685).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

const CONTENT_HASH_SCHEMA: &str =
    "crates/modelspec-runtime/schema/types/modelspec.artifact.ContentHash.schema.json";
const NO_LOADER: &str = "/nonexistent/modelspec-test/libvulkan-missing.so.1";
const CORPUS: &str = "crates/modelspec-runtime/tests/fixtures/corpus-tokenizer";

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
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(root())
        .env("MODELSPEC_VULKAN_LIBRARY", NO_LOADER)
        .output()
        .expect("modelspec runs");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{args:?}\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn json(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn s(path: &Path) -> &str {
    path.to_str().unwrap()
}

/// The values of `pointers` in `record` the `ContentHash` schema refuses or that are not 64
/// digits long, each with its pointer.
fn malformed(record: &Value, pointers: &[String]) -> Vec<String> {
    let schema = json(&root().join(CONTENT_HASH_SCHEMA));
    let validator = jsonschema::validator_for(&schema).expect("the schema compiles");
    let mut bad = Vec::new();
    for pointer in pointers {
        let value = record
            .pointer(pointer)
            .unwrap_or_else(|| panic!("the record has {pointer}"));
        let digits = value.as_str().map_or(0, str::len);
        if !validator.is_valid(value) || digits != 64 {
            bad.push(format!("{pointer} = {value}"));
        }
    }
    bad
}

#[test]
fn adv_chash_compiled_artifact_hashes_are_content_hashes() {
    let case = scratch("artifact");
    let profile = root().join("crates/modelspec-runtime/tests/fixtures/rtx3090.device.json");
    for backend in ["cpu", "vulkan"] {
        let out = case.join(backend);
        let mut args = vec![
            "compile",
            "examples/mnist_mlp.mspec",
            "--backend",
            backend,
            "--output",
            s(&out),
        ];
        if backend == "vulkan" {
            args.extend(["--device-profile", s(&profile)]);
        }
        modelspec(&args);
        let manifest = json(&out.join("artifact.json"));
        let mut pointers: Vec<String> = ["artifact_hash", "source_hash", "ir_hash", "schema_hash"]
            .iter()
            .map(|f| format!("/{f}"))
            .collect();
        let kernels = manifest["kernels"].as_array().map_or(0, Vec::len);
        pointers.extend((0..kernels).map(|i| format!("/kernels/{i}/cache_key/ir_hash")));
        if backend == "vulkan" {
            assert!(kernels > 0, "a Vulkan artifact has kernels");
        }
        assert_eq!(
            malformed(&manifest, &pointers),
            Vec::<String>::new(),
            "{backend}"
        );
    }
}

#[test]
fn adv_chash_checkpoint_manifest_hashes_are_content_hashes() {
    let case = scratch("checkpoint");
    let ckpt = case.join("a.msckpt");
    modelspec(&[
        "train",
        "examples/linear_regression.mspec",
        "--backend",
        "cpu",
        "--steps",
        "3",
        "--checkpoint",
        s(&ckpt),
    ]);
    let bytes = fs::read(&ckpt).unwrap();
    let size = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
    let manifest: Value = serde_json::from_slice(&bytes[56..56 + size]).unwrap();
    let pointers = ["/schema_hash".to_owned(), "/payload_hash".to_owned()];
    assert_eq!(malformed(&manifest, &pointers), Vec::<String>::new());
}

#[test]
fn adv_chash_data_records_hashes_are_content_hashes() {
    let case = scratch("data");
    let dataset = case.join("dataset");
    modelspec(&["data", "prepare", CORPUS, "--out", s(&dataset)]);
    let tokenizer = case.join("tokenizer.json");
    modelspec(&[
        "data",
        "tokenizer",
        "--dataset",
        s(&dataset),
        "--vocab-size",
        "300",
        "--out",
        s(&tokenizer),
    ]);
    let packed = case.join("packed");
    modelspec(&[
        "data",
        "pack",
        "--dataset",
        s(&dataset),
        "--tokenizer",
        s(&tokenizer),
        "--seq-len",
        "16",
        "--out",
        s(&packed),
    ]);
    let pointers = [
        "/splits/0/content_hash".to_owned(),
        "/splits/1/content_hash".to_owned(),
    ];
    let record = json(&dataset.join("dataset.json"));
    assert_eq!(malformed(&record, &pointers), Vec::<String>::new());
    let record = json(&packed.join("packed.json"));
    assert_eq!(malformed(&record, &pointers), Vec::<String>::new());
}

/// The schema check of [`malformed`] itself: it refuses what the schema projects (uppercase,
/// non-hex) and what only the specification states (63, 65 and 0 digits), so a writer test above
/// cannot pass on a helper that accepts everything.
#[test]
fn adv_chash_the_check_refuses_each_malformed_form() {
    let hash = "b175c1cce8dbd537b453b815c9e324fb2596856a93834c20b692ae92cd43cc3b";
    let forms = [
        hash.to_uppercase(),
        "not a digest".to_owned(),
        hash[..63].to_owned(),
        format!("{hash}0"),
        String::new(),
    ];
    let record = Value::Array(forms.iter().cloned().map(Value::String).collect());
    let pointers: Vec<String> = (0..forms.len()).map(|i| format!("/{i}")).collect();
    assert_eq!(malformed(&record, &pointers).len(), forms.len());
    let good = Value::Array(vec![Value::String(hash.to_owned())]);
    assert_eq!(malformed(&good, &["/0".to_owned()]), Vec::<String>::new());
}
