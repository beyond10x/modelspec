//! Adversary cases for the generated JSON Schema of the `Dataset` record `modelspec data prepare`
//! writes (story:bert-corpus-splits; `crates/modelspec-runtime/schema/`). The record test of the
//! story validates a correct record; these cases ask whether the schema refuses the records the
//! specification rules out.

use std::path::Path;

use serde_json::{Value, json};

const SCHEMA: &str = "crates/modelspec-runtime/schema/entities/modelspec.data.Dataset.schema.json";

fn validator() -> jsonschema::Validator {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(SCHEMA);
    let schema: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    jsonschema::validator_for(&schema).unwrap()
}

fn split(role: &str, hash: &str) -> Value {
    json!({"role": role, "content_hash": hash, "document_count": 1, "byte_count": 1})
}

fn record(splits: Vec<Value>) -> Value {
    json!({
        "dataset_id": "0ddcfa78-c328-8cc9-ae3b-490c096d55a5",
        "name": "c",
        "splits": splits,
        "state": "Recorded"
    })
}

const HASH: &str = "b175c1cce8dbd537b453b815c9e324fb2596856a93834c20b692ae92cd43cc3b";

#[test]
fn adv_cs_schema_accepts_a_well_formed_record_and_refuses_a_zero_count() {
    let v = validator();
    assert!(v.is_valid(&record(vec![split("Train", HASH), split("HeldOut", HASH)])));
    let mut zero = split("Train", HASH);
    zero["document_count"] = json!(0);
    assert!(!v.is_valid(&record(vec![zero, split("HeldOut", HASH)])));
    let mut extra = record(vec![split("Train", HASH), split("HeldOut", HASH)]);
    extra["path"] = json!("c");
    assert!(!v.is_valid(&extra), "additionalProperties is false");
}

#[test]
fn adv_cs_schema_refuses_a_content_hash_that_is_not_hex() {
    let v = validator();
    assert!(!v.is_valid(&record(vec![
        split("Train", "not a digest"),
        split("HeldOut", HASH)
    ])));
}

#[test]
fn adv_cs_schema_refuses_an_uppercase_content_hash() {
    let v = validator();
    let upper = HASH.to_uppercase();
    assert!(!v.is_valid(&record(vec![
        split("Train", &upper),
        split("HeldOut", HASH)
    ])));
}

#[test]
#[ignore = "ess-limit: text-length-not-projected, https://github.com/beyond10x/ess/issues/685 (ess 0.68.0 projects `value.count == 64` on a String newtype as x-ess-invariants only, no minLength or maxLength)"]
fn adv_cs_schema_refuses_a_content_hash_that_is_not_64_digits() {
    let v = validator();
    for hash in [&HASH[..63], &format!("{HASH}0")[..], ""] {
        assert!(
            !v.is_valid(&record(vec![split("Train", hash), split("HeldOut", HASH)])),
            "{} digits",
            hash.len()
        );
    }
}

#[test]
#[ignore = "ess-limit: distinct-by-key-not-projected, https://github.com/beyond10x/ess/issues/686 (ess 0.68.0 projects no `distinct:` invariant, on an entity or a struct, into the JSON Schema)"]
fn adv_cs_schema_refuses_two_splits_of_one_role() {
    let v = validator();
    assert!(!v.is_valid(&record(vec![split("Train", HASH), split("Train", HASH)])));
}
