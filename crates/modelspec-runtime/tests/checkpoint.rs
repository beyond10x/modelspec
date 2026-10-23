//! The `.msckpt` writer and reader (DESIGN.md § 6.2; `spec/domains/checkpoint.yaml`): the header
//! byte layout, the 64-byte payload alignment, the payload hash, a round trip, a resumed run that
//! equals one that never stopped, and E-CKPT-003 for a checkpoint that does not match.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use modelspec_core::artifact::schema_hash;
use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::tensor::ParamId;
use modelspec_runtime::train::checkpoint::{
    self, ALIGNMENT, Checkpoint, CheckpointError, E_CKPT_003, HEADER_LEN, OptimizerKind,
};
use modelspec_runtime::train::runner;

const LINEAR: &str = include_str!("../../../examples/linear_regression.mspec");
const TWO_LAYER: &str = include_str!("../../modelspec-docs/models/two_layer_relu.mspec");

fn compiled(source: &str, steps: u64) -> CompiledTraining {
    let mut compiled = compile_training("fit.mspec", source).expect("the source compiles");
    compiled.config.steps = steps;
    compiled
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("checkpoint");
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    dir.join(name)
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

fn trained(steps: u64) -> (CompiledTraining, Checkpoint) {
    let c = compiled(LINEAR, steps);
    let run = runner::run(&c).unwrap();
    let ckpt = Checkpoint::of_state(&c, steps, &run.state.values);
    (c, ckpt)
}

/// The header of DESIGN.md § 6.2: magic, version 1, the marker bytes `04 03 02 01`, a zero
/// reserved word, the manifest size and a blake3 checksum over header and manifest with the
/// checksum bytes counted as zero.
#[test]
fn the_header_has_the_documented_byte_layout() {
    let (_, ckpt) = trained(3);
    let bytes = ckpt.encode();
    assert_eq!(HEADER_LEN, 56);
    assert_eq!(&bytes[0..4], b"MSCK");
    assert_eq!(u32_at(&bytes, 4), 1);
    assert_eq!(&bytes[8..12], &[4, 3, 2, 1]);
    assert_eq!(u32_at(&bytes, 12), 0);
    let m = u64_at(&bytes, 16) as usize;
    let mut signed = bytes[..HEADER_LEN + m].to_vec();
    signed[24..56].fill(0);
    assert_eq!(&bytes[24..56], blake3::hash(&signed).as_bytes());
    let manifest: serde_json::Value = serde_json::from_slice(&bytes[56..56 + m]).unwrap();
    assert_eq!(manifest["file_size"], bytes.len() as u64);
    assert_eq!(manifest["alignment"], 64);
    assert_eq!(manifest["step"], 3);
    assert_eq!(manifest["optimizer_slots"], serde_json::json!([]));
    assert_eq!(
        manifest["optimizer"],
        serde_json::json!({"kind": "Sgd", "learning_rate": f64::from(0.1f32)})
    );
    assert_eq!(
        manifest["rng"],
        serde_json::json!({"algorithm": "Philox4x32x10", "seed": 42, "counter": 0})
    );
    let c = compiled(LINEAR, 3);
    assert_eq!(
        manifest["schema_hash"],
        schema_hash(&c.training.training.program.params).0
    );
}

/// Every tensor sits at a multiple of 64 after the manifest, the file ends with the last tensor,
/// and the payload hash is blake3 from the payload start to the end of the file.
#[test]
fn the_payload_is_64_byte_aligned_and_hashed() {
    let source = TWO_LAYER;
    let c = compiled(source, 2);
    let run = runner::run(&c).unwrap();
    let bytes = Checkpoint::of_state(&c, 2, &run.state.values).encode();
    let m = u64_at(&bytes, 16) as usize;
    let manifest: serde_json::Value = serde_json::from_slice(&bytes[56..56 + m]).unwrap();
    let start = (HEADER_LEN + m).next_multiple_of(ALIGNMENT as usize);
    let records = manifest["parameters"].as_array().unwrap();
    assert_eq!(records.len(), 2);
    let paths: Vec<&str> = records
        .iter()
        .map(|r| r["path"].as_str().unwrap())
        .collect();
    assert_eq!(paths, ["TwoLayer.w1", "TwoLayer.w2"]);
    assert_eq!(records[0]["offset"], start as u64);
    let mut end = 0;
    for record in records {
        let offset = record["offset"].as_u64().unwrap();
        let length = record["length"].as_u64().unwrap();
        assert_eq!(offset % 64, 0, "{record}");
        assert!(offset >= end);
        assert!(bytes[end as usize..offset as usize].iter().all(|&b| b == 0) || end == 0);
        end = offset + length;
    }
    assert_eq!(records[0]["length"], 4 * 8 * 4);
    assert_eq!(records[0]["shape"], serde_json::json!([4, 8]));
    assert_eq!(records[0]["dtype"], "F32");
    assert_eq!(end, bytes.len() as u64);
    assert_eq!(
        manifest["payload_hash"],
        blake3::hash(&bytes[start..]).to_hex().to_string()
    );
    let w1 = &run.state.values[&ParamId(0)];
    let offset = records[0]["offset"].as_u64().unwrap() as usize;
    let first = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    assert_eq!(first.to_bits(), w1.data()[0].to_bits());
}

#[test]
fn a_written_checkpoint_reads_back_to_the_same_state() {
    let (c, ckpt) = trained(5);
    let path = scratch("round_trip.msckpt");
    checkpoint::write(&path, &ckpt).unwrap();
    let read = checkpoint::read(&path).unwrap();
    assert_eq!(read, ckpt);
    assert_eq!(read.step, 5);
    assert_eq!(read.optimizer.kind, OptimizerKind::Sgd);
    let params = read.resume_into(&c).unwrap();
    let run = runner::run(&c).unwrap();
    assert_eq!(params, run.state.values);
}

#[test]
fn the_same_state_encodes_to_the_same_bytes() {
    let (_, a) = trained(4);
    let (_, b) = trained(4);
    assert_eq!(a.encode(), b.encode());
    let (_, other) = trained(5);
    assert_ne!(a.encode(), other.encode());
}

/// § 9 gate 7 at the library level: 6 steps, or 2 steps then 4 resumed from their checkpoint,
/// give the same bytes and the same losses from step 2.
#[test]
fn a_resumed_run_equals_one_that_never_stopped() {
    let whole = compiled(LINEAR, 6);
    let uninterrupted = runner::run(&whole).unwrap();
    let a = Checkpoint::of_state(&whole, 6, &uninterrupted.state.values).encode();

    let (_, half) = trained(2);
    let path = scratch("half.msckpt");
    checkpoint::write(&path, &half).unwrap();
    let rest = compiled(LINEAR, 4);
    let loaded = checkpoint::read(&path).unwrap();
    let init = loaded.resume_into(&rest).unwrap();
    let mut seen = Vec::new();
    let resumed = runner::run_from(
        &rest,
        None,
        &init,
        loaded.step,
        |o| seen.push((o.step, o.loss)),
        |_, _| Ok(()),
    )
    .unwrap();
    assert_eq!(resumed.start_step, 2);
    let b = Checkpoint::of_state(&rest, 6, &resumed.state.values).encode();
    assert_eq!(a, b);
    let expected: Vec<(u64, f32)> = (2..)
        .zip(uninterrupted.losses[2..].iter().copied())
        .collect();
    assert_eq!(seen, expected);
}

/// The runner reports the state after every update, numbered by step, to its checkpoint hook.
#[test]
fn the_hook_sees_the_state_after_each_update() {
    let c = compiled(LINEAR, 3);
    let mut seen = Vec::new();
    runner::run_from(
        &c,
        None,
        &BTreeMap::new(),
        10,
        |_| {},
        |k, _| {
            seen.push(k);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(seen, [11, 12, 13]);
}

fn refused(error: CheckpointError) -> String {
    match error {
        CheckpointError::Refused(d) => {
            assert_eq!(d.code, E_CKPT_003, "{d}");
            d.to_string()
        }
        other => panic!("not refused: {other}"),
    }
}

#[test]
fn another_model_is_refused_at_its_first_parameter_path() {
    let c = compiled(TWO_LAYER, 1);
    let run = runner::run(&c).unwrap();
    let ckpt = Checkpoint::of_state(&c, 1, &run.state.values);
    let message = refused(ckpt.resume_into(&compiled(LINEAR, 1)).unwrap_err());
    assert!(message.contains("TwoLayer.w1"), "{message}");
    assert!(message.contains("LinearReg.w"), "{message}");
    let message = refused(ckpt.parameters_for(&compiled(LINEAR, 1)).unwrap_err());
    assert!(message.contains("TwoLayer.w1"), "{message}");
}

#[test]
fn another_shape_is_refused_naming_the_parameter() {
    let (_, ckpt) = trained(1);
    let wider = LINEAR
        .replace("Tensor<f32, [4, 1]>", "Tensor<f32, [5, 1]>")
        .replace("[8, 4]", "[8, 5]");
    let message = refused(ckpt.resume_into(&compiled(&wider, 1)).unwrap_err());
    assert!(message.contains("LinearReg.w"), "{message}");
    assert!(message.contains("shape"), "{message}");
}

#[test]
fn another_learning_rate_is_refused_naming_the_optimizer_on_resume_only() {
    let (_, ckpt) = trained(1);
    let other = compiled(&LINEAR.replace("lr = 0.1", "lr = 0.05"), 1);
    let message = refused(ckpt.resume_into(&other).unwrap_err());
    assert!(message.contains("optimizer"), "{message}");
    assert!(ckpt.parameters_for(&other).is_ok());
}
