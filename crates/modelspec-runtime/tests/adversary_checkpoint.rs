//! Adversary cases for the `.msckpt` reader (DESIGN.md § 6.2; `spec/domains/checkpoint.yaml`).
//!
//! Each case re-assembles a valid checkpoint with an edited manifest, recomputes the layout, the
//! checksum and the payload hash where the case says so, and asserts the refusal the design states
//! and the allocation bound of the story's acceptance 4: no single allocation while the reader
//! runs is larger than the file plus 4,096 bytes.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::{Path, PathBuf};

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::diag::Code;
use modelspec_runtime::train::checkpoint::{
    self, Checkpoint, CheckpointError, E_CKPT_001, E_CKPT_002, E_CKPT_003, E_CKPT_004, HEADER_LEN,
};
use modelspec_runtime::train::runner;
use serde_json::{Value, json};

struct Counting;

thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
    static LARGEST: Cell<usize> = const { Cell::new(0) };
}

fn record(size: usize) {
    let _ = MEASURING.try_with(|on| {
        if on.get() {
            let _ = LARGEST.try_with(|largest| largest.set(largest.get().max(size)));
        }
    });
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const LINEAR: &str = include_str!("../../../examples/linear_regression.mspec");
const TWO_LAYER: &str = include_str!("../../modelspec-docs/models/two_layer_relu.mspec");

fn compiled(source: &str, steps: u64) -> CompiledTraining {
    let mut compiled = compile_training("fit.mspec", source).expect("the source compiles");
    compiled.config.steps = steps;
    compiled
}

fn valid(source: &str) -> Vec<u8> {
    let c = compiled(source, 3);
    let run = runner::run(&c).unwrap();
    Checkpoint::of_state(&c, 3, &run.state.values).encode()
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("adversary_checkpoint");
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    dir.join(format!("{name}.msckpt"))
}

fn manifest_size(bytes: &[u8]) -> usize {
    u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize
}

fn sign(bytes: &mut [u8]) {
    let m = manifest_size(bytes);
    let mut signed = bytes[..HEADER_LEN + m].to_vec();
    signed[24..56].fill(0);
    let hash = blake3::hash(&signed);
    bytes[24..56].copy_from_slice(hash.as_bytes());
}

fn manifest(bytes: &[u8]) -> Value {
    let m = manifest_size(bytes);
    serde_json::from_slice(&bytes[HEADER_LEN..HEADER_LEN + m]).unwrap()
}

/// Header and manifest of `bytes` replaced by `manifest_text` (padded with spaces to `m`),
/// followed by `payload`, re-signed.
fn assemble(bytes: &[u8], manifest_text: &[u8], m: usize, payload: &[u8]) -> Vec<u8> {
    assert!(manifest_text.len() <= m);
    let mut out = bytes[..HEADER_LEN].to_vec();
    out[16..24].copy_from_slice(&(m as u64).to_le_bytes());
    out.extend_from_slice(manifest_text);
    out.resize(HEADER_LEN + m, b' ');
    out.resize((HEADER_LEN + m).next_multiple_of(64), 0);
    out.extend_from_slice(payload);
    sign(&mut out);
    out
}

/// `bytes` re-laid out with `edit` applied to the manifest: the payload is kept, the manifest gets
/// a size whose end is a multiple of 64, every offset and `file_size` follow the new payload start,
/// and the file is re-signed. `edit` sees the new payload start.
fn rebuilt(bytes: &[u8], edit: impl Fn(&mut Value, u64)) -> Vec<u8> {
    let old_m = manifest_size(bytes);
    let old_start = ((HEADER_LEN + old_m) as u64).next_multiple_of(64);
    let payload = bytes[old_start as usize..].to_vec();
    let original = manifest(bytes);
    let lay_out = |start: u64| {
        let mut v = original.clone();
        for r in v["parameters"].as_array_mut().unwrap() {
            let rel = r["offset"].as_u64().unwrap() - old_start;
            r["offset"] = (start + rel).into();
        }
        v["file_size"] = (start + payload.len() as u64).into();
        edit(&mut v, start);
        serde_json::to_vec(&v).unwrap()
    };
    let probe = lay_out(1 << 40).len();
    let m = (HEADER_LEN + probe + 64).next_multiple_of(64) - HEADER_LEN;
    let start = (HEADER_LEN + m) as u64;
    let text = lay_out(start);
    assemble(bytes, &text, m, &payload)
}

fn measured(path: &Path) -> (Result<Checkpoint, CheckpointError>, usize) {
    LARGEST.with(|l| l.set(0));
    MEASURING.with(|m| m.set(true));
    let result = std::panic::catch_unwind(|| checkpoint::read(path));
    MEASURING.with(|m| m.set(false));
    let largest = LARGEST.with(Cell::get);
    (result.expect("the reader does not panic"), largest)
}

/// Reads `bytes` and returns the refusal, after asserting the allocation bound.
fn refusal(name: &str, bytes: &[u8]) -> Result<Checkpoint, CheckpointError> {
    let path = scratch(name);
    std::fs::write(&path, bytes).unwrap();
    let (result, largest) = measured(&path);
    let bound = bytes.len() + 4096;
    assert!(
        largest <= bound,
        "{name}: largest allocation {largest} bytes > file size {} + 4096 (result: {:?})",
        bytes.len(),
        result.as_ref().err().map(ToString::to_string)
    );
    result
}

fn refused_with(name: &str, bytes: &[u8], code: Code) -> String {
    match refusal(name, bytes) {
        Err(CheckpointError::Refused(d)) => {
            assert_eq!(d.code, code, "{name}: {d}");
            d.to_string()
        }
        Err(other) => panic!("{name}: not a refusal: {other}"),
        Ok(_) => panic!("{name}: accepted, expected {code}"),
    }
}

#[test]
fn rebuilt_without_edit_still_reads() {
    let bytes = rebuilt(&valid(LINEAR), |_, _| {});
    assert!(refusal("rebuilt", &bytes).is_ok());
}

// ---- Allocation bound under manifests that are large in elements, not in bytes. ----

#[test]
fn a_shape_of_fifty_thousand_dimensions_stays_within_the_allocation_bound() {
    let bytes = rebuilt(&valid(LINEAR), |v, _| {
        let mut shape = vec![1u64; 50_000];
        shape[0] = 4;
        v["parameters"][0]["shape"] = json!(shape);
    });
    let _ = refusal("rank_50000", &bytes);
}

#[test]
fn twenty_thousand_records_stay_within_the_allocation_bound() {
    let bytes = rebuilt(&valid(LINEAR), |v, _| {
        let one = json!({"path": "", "dtype": "F32", "shape": [], "offset": 0, "length": 0});
        v["parameters"] = Value::Array(vec![one; 20_000]);
    });
    let _ = refusal("records_20000", &bytes);
}

// ---- The manifest against the types of spec/domains/checkpoint.yaml. ----

#[test]
fn a_checkpoint_id_that_is_not_a_uuid_is_001() {
    let bytes = rebuilt(&valid(LINEAR), |v, _| {
        v["checkpoint_id"] = "not-a-uuid".into()
    });
    refused_with("checkpoint_id", &bytes, E_CKPT_001);
}

// ---- Header checks a mutant can drop. ----

#[test]
fn a_bad_magic_version_or_reserved_word_is_001() {
    for (name, at, value) in [
        ("magic", 0usize, *b"MSCX"),
        ("version", 4, 2u32.to_le_bytes()),
        ("reserved", 12, 1u32.to_le_bytes()),
        ("unknown_marker", 8, 0xdead_beefu32.to_le_bytes()),
    ] {
        let mut bytes = valid(LINEAR);
        bytes[at..at + 4].copy_from_slice(&value);
        sign(&mut bytes);
        refused_with(name, &bytes, E_CKPT_001);
    }
}

#[test]
fn a_zero_length_file_and_a_zero_manifest_size_are_002() {
    refused_with("empty", &[], E_CKPT_002);
    let mut bytes = valid(LINEAR);
    bytes[16..24].copy_from_slice(&0u64.to_le_bytes());
    sign(&mut bytes);
    refused_with("manifest_size_0", &bytes, E_CKPT_002);
}

#[test]
fn a_manifest_that_is_not_utf8_or_nests_deeply_is_001() {
    let base = valid(LINEAR);
    let m = manifest_size(&base);
    let start = (HEADER_LEN + m).next_multiple_of(64);
    let payload = base[start..].to_vec();
    let mut text = base[HEADER_LEN..HEADER_LEN + m].to_vec();
    let at = text.iter().position(|&b| b == b'w').unwrap();
    text[at] = 0xff;
    refused_with("not_utf8", &assemble(&base, &text, m, &payload), E_CKPT_001);

    let deep = format!("{{\"x\":{}{}}}", "[".repeat(100_000), "]".repeat(100_000));
    let m = (HEADER_LEN + deep.len()).next_multiple_of(64) - HEADER_LEN;
    let bytes = assemble(&base, deep.as_bytes(), m, &payload);
    refused_with("deep", &bytes, E_CKPT_001);
}

// ---- Layout checks a mutant can drop. ----

#[test]
fn another_alignment_is_002() {
    let bytes = rebuilt(&valid(LINEAR), |v, _| v["alignment"] = 32.into());
    refused_with("alignment", &bytes, E_CKPT_002);
}

#[test]
fn an_offset_inside_the_manifest_is_002() {
    let bytes = rebuilt(&valid(LINEAR), |v, start| {
        v["parameters"][0]["offset"] = (start - 64).into()
    });
    refused_with("inside_manifest", &bytes, E_CKPT_002);
}

#[test]
fn an_offset_plus_length_past_u64_max_is_002() {
    let bytes = rebuilt(&valid(LINEAR), |v, _| {
        let r = &mut v["parameters"][0];
        r["offset"] = (u64::MAX - 63).into();
        r["length"] = 128.into();
        r["shape"] = json!([32]);
    });
    refused_with("overflow", &bytes, E_CKPT_002);
}

#[test]
fn a_shape_whose_byte_count_overflows_is_002() {
    let bytes = rebuilt(&valid(LINEAR), |v, _| {
        v["parameters"][0]["shape"] = json!([u64::MAX, 2]);
    });
    refused_with("shape_overflow", &bytes, E_CKPT_002);
}

#[test]
fn overlapping_tensors_are_002() {
    let bytes = rebuilt(&valid(TWO_LAYER), |v, _| {
        let first = v["parameters"][0]["offset"].clone();
        v["parameters"][1]["offset"] = first;
    });
    refused_with("overlap", &bytes, E_CKPT_002);
}

#[test]
fn a_duplicate_path_is_001() {
    let bytes = rebuilt(&valid(TWO_LAYER), |v, _| {
        let first = v["parameters"][0]["path"].clone();
        v["parameters"][1]["path"] = first;
    });
    refused_with("duplicate", &bytes, E_CKPT_001);
}

#[test]
fn a_zero_dimension_or_no_parameter_is_001() {
    let bytes = rebuilt(&valid(LINEAR), |v, _| {
        v["parameters"][0]["shape"] = json!([0, 1]);
        v["parameters"][0]["length"] = 0.into();
    });
    refused_with("zero_dim", &bytes, E_CKPT_001);
    let bytes = rebuilt(&valid(LINEAR), |v, _| v["parameters"] = json!([]));
    refused_with("no_parameter", &bytes, E_CKPT_001);
}

#[test]
fn an_sgd_checkpoint_with_slots_is_001() {
    let bytes = rebuilt(&valid(LINEAR), |v, _| {
        let mut slot = v["parameters"][0].clone();
        slot["slot"] = "m".into();
        v["optimizer_slots"] = json!([slot]);
    });
    refused_with("sgd_slots", &bytes, E_CKPT_001);
}

#[test]
fn an_sgd_optimizer_with_an_adamw_field_or_a_parameter_with_a_slot_is_001() {
    let bytes = rebuilt(&valid(LINEAR), |v, _| v["optimizer"]["beta1"] = 0.9.into());
    refused_with("sgd_beta1", &bytes, E_CKPT_001);
    let bytes = rebuilt(&valid(LINEAR), |v, _| {
        v["parameters"][0]["slot"] = "m".into()
    });
    refused_with("param_slot", &bytes, E_CKPT_001);
}

#[test]
fn a_payload_hash_over_a_changed_payload_is_004_even_after_a_rebuild() {
    let base = valid(LINEAR);
    let mut bytes = rebuilt(&base, |_, _| {});
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    refused_with("payload_rebuilt", &bytes, E_CKPT_004);
}

// ---- Matching a compiled file. ----

#[test]
fn another_schema_hash_is_003() {
    let c = compiled(LINEAR, 3);
    let run = runner::run(&c).unwrap();
    let mut ckpt = Checkpoint::of_state(&c, 3, &run.state.values);
    ckpt.schema_hash.0 = "0".repeat(64);
    match ckpt.parameters_for(&c) {
        Err(CheckpointError::Refused(d)) => {
            assert_eq!(d.code, E_CKPT_003);
            assert!(d.to_string().contains("schema hash"), "{d}");
        }
        other => panic!("not refused: {other:?}"),
    }
}

#[test]
fn a_checkpoint_with_an_extra_parameter_is_003() {
    let c = compiled(LINEAR, 3);
    let run = runner::run(&c).unwrap();
    let mut ckpt = Checkpoint::of_state(&c, 3, &run.state.values);
    let mut extra = ckpt.parameters[0].clone();
    extra.path = "zzz.extra".into();
    ckpt.parameters.push(extra);
    match ckpt.parameters_for(&c) {
        Err(CheckpointError::Refused(d)) => {
            assert_eq!(d.code, E_CKPT_003);
            assert!(d.to_string().contains("zzz.extra"), "{d}");
        }
        other => panic!("not refused: {other:?}"),
    }
}

// ---- Identity: `checkpoint_id` is the identity of `Checkpoint` (spec/domains/checkpoint.yaml). ----

#[test]
fn two_checkpoints_that_differ_only_in_their_optimizer_have_different_ids() {
    let c = compiled(LINEAR, 0);
    let run = runner::run(&c).unwrap();
    let one = Checkpoint::of_state(&c, 0, &run.state.values);
    let mut two = one.clone();
    two.optimizer.learning_rate = 0.2;
    let (a, b) = (one.encode(), two.encode());
    assert_ne!(a, b, "the two files differ");
    assert_ne!(
        manifest(&a)["checkpoint_id"],
        manifest(&b)["checkpoint_id"],
        "two different checkpoints carry one identity"
    );
}

// ---- Pass 2: the fixes of round 1. ----

use modelspec_runtime::cpu::HostTensor;
use modelspec_runtime::train::checkpoint::Parameter;

/// `valid(LINEAR)`'s manifest with `parameters` (and the rest of the text) replaced by `text`,
/// followed by zero bytes up to `size`, re-signed.
fn raw_file(text: &str, size: usize) -> Vec<u8> {
    let base = valid(LINEAR);
    let m = text.len();
    let start = (HEADER_LEN + m).next_multiple_of(64);
    assert!(size >= start, "{size} < {start}");
    assemble(&base, text.as_bytes(), m, &vec![0u8; size - start])
}

/// The manifest of `valid(LINEAR)` as compact JSON text, with `"parameters":<list>` in place.
fn manifest_text_with_parameters(list: &str) -> String {
    let mut v = manifest(&valid(LINEAR));
    v["parameters"] = json!("__LIST__");
    serde_json::to_string(&v)
        .unwrap()
        .replace("\"__LIST__\"", list)
}

#[test]
fn deep_nesting_inside_the_raw_record_lists_is_001_within_the_bound() {
    let deep = format!("{}{}", "[".repeat(200_000), "]".repeat(200_000));
    let text = manifest_text_with_parameters(&format!("[{deep}]"));
    let size = (HEADER_LEN + text.len()).next_multiple_of(64) + 64;
    refused_with("deep_parameters", &raw_file(&text, size), E_CKPT_001);

    let mut v = manifest(&valid(LINEAR));
    v["optimizer_slots"] = json!("__LIST__");
    let text = serde_json::to_string(&v)
        .unwrap()
        .replace("\"__LIST__\"", &format!("[{deep}]"));
    let size = (HEADER_LEN + text.len()).next_multiple_of(64) + 64;
    refused_with("deep_slots", &raw_file(&text, size), E_CKPT_001);
}

#[test]
fn a_repeated_manifest_key_is_001() {
    let base = valid(LINEAR);
    let mut text = serde_json::to_string(&manifest(&base)).unwrap();
    text.pop();
    text.push_str(",\"parameters\":[]}");
    let size = (HEADER_LEN + text.len()).next_multiple_of(64) + 64;
    refused_with("repeated_key", &raw_file(&text, size), E_CKPT_001);
}

#[test]
fn a_million_empty_elements_are_counted_without_allocating_and_refused_002() {
    let list = format!("[{}{{}}]", "{},".repeat(1_000_000));
    let text = manifest_text_with_parameters(&list);
    let size = (HEADER_LEN + text.len()).next_multiple_of(64) + 64;
    let message = refused_with("million", &raw_file(&text, size), E_CKPT_002);
    assert!(message.contains("records"), "{message}");
}

/// Around every size at which the record bound can change, for any `Record` size from 64 to 128
/// bytes, the largest allocation stays within the file plus 4,096 bytes, whatever the refusal.
#[test]
fn the_record_bound_holds_the_allocation_bound_at_its_edge() {
    let n = 2_000usize;
    let records: Vec<String> = (0..n)
        .map(|i| {
            format!(
                "{{\"path\":\"p{i}\",\"dtype\":\"F32\",\"shape\":[1],\"offset\":0,\"length\":4}}"
            )
        })
        .collect();
    let text = manifest_text_with_parameters(&format!("[{}]", records.join(",")));
    let floor = (HEADER_LEN + text.len()).next_multiple_of(64);
    let mut refused_for_count = false;
    for record in (64..=128).step_by(8) {
        let edge = (record * n).saturating_sub(4096);
        for size in edge.saturating_sub(2)..=edge + 2 {
            if size < floor {
                continue;
            }
            if let Err(CheckpointError::Refused(d)) = refusal("edge", &raw_file(&text, size)) {
                refused_for_count |= d.code == E_CKPT_002 && d.message.contains("records");
            }
        }
    }
    assert!(refused_for_count, "some size refuses the record count");
}

fn synthetic_checkpoint(shapes: &[&[u64]]) -> Checkpoint {
    let c = compiled(LINEAR, 0);
    let run = runner::run(&c).unwrap();
    let mut ckpt = Checkpoint::of_state(&c, 0, &run.state.values);
    ckpt.parameters = shapes
        .iter()
        .enumerate()
        .map(|(i, shape)| {
            let n: u64 = shape.iter().product();
            let data = (0..n)
                .map(|k| (i as f32) + (k as f32) / 8.0 - 3.0)
                .collect();
            Parameter {
                path: format!("M.p{i:02}"),
                value: HostTensor::new(shape.to_vec(), data).unwrap(),
            }
        })
        .collect();
    ckpt
}

const SHAPES: &[&[u64]] = &[
    &[],
    &[1],
    &[3],
    &[16],
    &[17],
    &[2, 3],
    &[4, 4, 4],
    &[1, 1, 1, 1, 1, 1, 1, 5],
    &[31],
    &[],
];

#[test]
fn the_writer_and_the_gap_checks_agree_for_one_and_for_many_tensors() {
    for shapes in [&SHAPES[..1], &SHAPES[2..3], &SHAPES[3..4], SHAPES] {
        let ckpt = synthetic_checkpoint(shapes);
        let bytes = ckpt.encode();
        let name = format!("roundtrip_{}", shapes.len());
        let read = refusal(&name, &bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(read, ckpt, "{name}");
    }
}

#[test]
fn a_non_zero_byte_between_manifest_and_payload_or_between_tensors_is_002() {
    let bytes = synthetic_checkpoint(SHAPES).encode();
    let v = manifest(&bytes);
    let m = manifest_size(&bytes);
    let start = (HEADER_LEN + m).next_multiple_of(64);
    if HEADER_LEN + m < start {
        let mut b = bytes.clone();
        b[HEADER_LEN + m] = 1;
        refused_with("gap_after_manifest", &b, E_CKPT_002);
    }
    let records = v["parameters"].as_array().unwrap();
    let mut gaps = 0;
    for pair in records.windows(2) {
        let end =
            (pair[0]["offset"].as_u64().unwrap() + pair[0]["length"].as_u64().unwrap()) as usize;
        let next = pair[1]["offset"].as_u64().unwrap() as usize;
        if end < next {
            gaps += 1;
            let mut b = bytes.clone();
            b[next - 1] = 0x80;
            refused_with("gap_between", &b, E_CKPT_002);
        }
    }
    assert!(gaps > 0, "the synthetic file has gaps between tensors");
}

#[test]
fn trailing_bytes_counted_in_file_size_are_002_and_no_payload_hash_is_001() {
    let mut bytes = rebuilt(&valid(TWO_LAYER), |v, _| {
        v["file_size"] = (v["file_size"].as_u64().unwrap() + 64).into()
    });
    bytes.extend_from_slice(&[0u8; 64]);
    let message = refused_with("trailing", &bytes, E_CKPT_002);
    assert!(message.contains("does not end"), "{message}");

    let bytes = rebuilt(&valid(LINEAR), |v, _| {
        v.as_object_mut().unwrap().remove("payload_hash");
    });
    refused_with("no_payload_hash", &bytes, E_CKPT_001);
}

#[test]
fn the_id_is_the_same_for_the_same_state_and_differs_with_the_step() {
    let c = compiled(LINEAR, 0);
    let run = runner::run(&c).unwrap();
    let a = Checkpoint::of_state(&c, 0, &run.state.values);
    let mut b = a.clone();
    b.step = 1;
    let id = |c: &Checkpoint| manifest(&c.encode())["checkpoint_id"].clone();
    assert_eq!(id(&a), id(&a.clone()));
    assert_ne!(id(&a), id(&b));
    let text = id(&a);
    let text = text.as_str().unwrap();
    assert_eq!(&text[14..15], "8", "version 8: {text}");
}

// ---- Pass 2: MAX_RANK at its edge, in the reader and in the writer. ----

/// `valid(LINEAR)` with parameter 0 reshaped to `rank` dimensions of the same element count.
fn of_rank(rank: usize) -> Vec<u8> {
    rebuilt(&valid(LINEAR), |v, _| {
        let r = &mut v["parameters"][0];
        let n = r["length"].as_u64().unwrap() / 4;
        let mut shape = vec![1u64; rank - 1];
        shape.push(n);
        r["shape"] = json!(shape);
    })
}

#[test]
fn a_shape_of_eight_dimensions_reads_and_one_of_nine_is_001() {
    assert!(refusal("rank_8", &of_rank(8)).is_ok(), "8 dimensions read");
    let message = refused_with("rank_9", &of_rank(9), E_CKPT_001);
    assert!(message.contains("dimensions"), "{message}");
}

#[test]
fn the_writer_writes_eight_dimensions_and_refuses_nine() {
    let eight: &[u64] = &[1, 1, 1, 1, 1, 1, 1, 2];
    let nine: &[u64] = &[1, 1, 1, 1, 1, 1, 1, 1, 2];
    let path = scratch("write_rank_8");
    checkpoint::write(&path, &synthetic_checkpoint(&[eight])).expect("8 dimensions are written");
    let path = scratch("write_rank_9");
    let _ = std::fs::remove_file(&path);
    match checkpoint::write(&path, &synthetic_checkpoint(&[nine])) {
        Err(CheckpointError::Write { message, .. }) => {
            assert!(message.contains("dimensions"), "{message}")
        }
        other => panic!("9 dimensions: {other:?}"),
    }
    assert!(!path.exists(), "nothing is written for 9 dimensions");
}
