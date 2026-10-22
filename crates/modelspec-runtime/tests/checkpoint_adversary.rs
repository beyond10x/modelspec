//! The `.msckpt` reader against altered files (DESIGN.md § 6.2, § 9 gate 8): each of the eight
//! alterations of the story and three of the layout (a non-zero gap byte, bytes after the last
//! tensor, no payload hash) is refused with its stated code, without a panic, and the reader
//! never makes a single allocation larger than the file plus 4,096 bytes.
//!
//! The counting allocator below is this test binary's own `#[global_allocator]`; it records, per
//! thread, the largest single allocation made while [`measured`] runs the reader. An alteration of
//! the manifest re-signs the file (the checksum over header and manifest), so the reader reaches
//! the check the case names instead of stopping at the checksum.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::{Path, PathBuf};

use modelspec_core::compile::compile_training;
use modelspec_core::diag::Code;
use modelspec_runtime::train::checkpoint::{
    self, Checkpoint, CheckpointError, E_CKPT_001, E_CKPT_002, E_CKPT_004, HEADER_LEN,
};
use modelspec_runtime::train::runner;

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

/// Runs the reader on `path` and returns its result and the largest single allocation it made on
/// this thread.
fn measured(path: &Path) -> (Result<Checkpoint, CheckpointError>, usize) {
    LARGEST.with(|l| l.set(0));
    MEASURING.with(|m| m.set(true));
    let result = std::panic::catch_unwind(|| checkpoint::read(path));
    MEASURING.with(|m| m.set(false));
    let largest = LARGEST.with(Cell::get);
    (result.expect("the reader does not panic"), largest)
}

const LINEAR: &str = include_str!("../../../examples/linear_regression.mspec");

/// A valid checkpoint of 3 steps of linear regression.
fn valid() -> Vec<u8> {
    let mut compiled = compile_training("fit.mspec", LINEAR).unwrap();
    compiled.config.steps = 3;
    let run = runner::run(&compiled).unwrap();
    Checkpoint::of_state(&compiled, 3, &run.state.values).encode()
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

/// `bytes` with the manifest changed by `edit`, padded with spaces to its old size and re-signed.
fn edited(mut bytes: Vec<u8>, edit: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
    let m = manifest_size(&bytes);
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&bytes[HEADER_LEN..HEADER_LEN + m]).unwrap();
    edit(&mut manifest);
    let mut text = serde_json::to_vec(&manifest).unwrap();
    assert!(text.len() <= m, "the edit fits the old manifest");
    text.resize(m, b' ');
    bytes[HEADER_LEN..HEADER_LEN + m].copy_from_slice(&text);
    sign(&mut bytes);
    bytes
}

fn first_record(manifest: &mut serde_json::Value) -> &mut serde_json::Value {
    &mut manifest["parameters"][0]
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("checkpoint_adversary");
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    dir.join(name)
}

/// Writes `bytes`, reads them back and asserts the refusal `code`, that its message contains
/// `naming`, and the allocation bound.
fn refused_with(name: &str, bytes: &[u8], code: Code, naming: &str) {
    let path = scratch(&format!("{name}.msckpt"));
    std::fs::write(&path, bytes).unwrap();
    let (result, largest) = measured(&path);
    match result {
        Err(CheckpointError::Refused(d)) => {
            assert_eq!(d.code, code, "{name}: {d}");
            assert!(d.to_string().contains(naming), "{name}: {d}");
        }
        Err(other) => panic!("{name}: not a refusal: {other}"),
        Ok(_) => panic!("{name}: accepted"),
    }
    let bound = bytes.len() + 4096;
    eprintln!("{name}: refused with {code}, largest allocation {largest} bytes, bound {bound}");
    assert!(
        largest <= bound,
        "{name}: largest allocation {largest} > file size {} + 4096",
        bytes.len()
    );
}

#[test]
fn the_valid_file_reads_within_the_allocation_bound() {
    let bytes = valid();
    let path = scratch("valid.msckpt");
    std::fs::write(&path, &bytes).unwrap();
    let (result, largest) = measured(&path);
    assert!(result.is_ok(), "{:?}", result.err());
    assert!(largest > 0, "the allocator counts on this thread");
    assert!(largest <= bytes.len() + 4096, "{largest}");
}

#[test]
fn a_truncated_file_is_002() {
    let bytes = valid();
    refused_with(
        "truncated",
        &bytes[..bytes.len() - 1],
        E_CKPT_002,
        "is not the file's",
    );
    refused_with(
        "truncated_header",
        &bytes[..HEADER_LEN - 1],
        E_CKPT_002,
        "fewer than the 56-byte header",
    );
}

#[test]
fn a_changed_checksum_byte_is_001() {
    let mut bytes = valid();
    bytes[30] ^= 0x01;
    refused_with("checksum", &bytes, E_CKPT_001, "the checksum differs");
}

#[test]
fn the_big_endian_marker_is_001() {
    let mut bytes = valid();
    bytes[8..12].copy_from_slice(&[1, 2, 3, 4]);
    sign(&mut bytes);
    refused_with("big_endian", &bytes, E_CKPT_001, "big-endian");
}

#[test]
fn an_offset_past_the_file_is_002() {
    let bytes = valid();
    let past = bytes.len() as u64 + 64;
    let bytes = edited(bytes, |m| {
        first_record(m)["offset"] = past.next_multiple_of(64).into()
    });
    refused_with("offset_past", &bytes, E_CKPT_002, "end past the");
}

#[test]
fn an_offset_not_a_multiple_of_64_is_002() {
    let bytes = valid();
    let bytes = edited(bytes, |m| {
        let offset = first_record(m)["offset"].as_u64().unwrap();
        first_record(m)["offset"] = (offset + 4).into();
    });
    refused_with("misaligned", &bytes, E_CKPT_002, "is not a multiple of 64");
}

#[test]
fn a_length_that_is_not_shape_times_dtype_size_is_002() {
    let bytes = valid();
    let bytes = edited(bytes, |m| first_record(m)["length"] = 12.into());
    refused_with("length", &bytes, E_CKPT_002, "is not the shape");
}

#[test]
fn a_manifest_size_of_u64_max_is_002() {
    let mut bytes = valid();
    bytes[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    refused_with("manifest_size", &bytes, E_CKPT_002, "does not fit");
}

#[test]
fn a_changed_payload_byte_is_004() {
    let mut bytes = valid();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x40;
    refused_with("payload", &bytes, E_CKPT_004, "payload hash differs");
}

#[test]
fn a_byte_between_tensors_that_is_not_zero_is_002() {
    let mut bytes = valid();
    let m = manifest_size(&bytes);
    let manifest: serde_json::Value =
        serde_json::from_slice(&bytes[HEADER_LEN..HEADER_LEN + m]).unwrap();
    let tensors: Vec<(usize, usize)> = manifest["parameters"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let at = r["offset"].as_u64().unwrap() as usize;
            (at, at + r["length"].as_u64().unwrap() as usize)
        })
        .collect();
    let gap = (HEADER_LEN + m..bytes.len())
        .find(|&i| tensors.iter().all(|&(from, to)| i < from || i >= to))
        .expect("the valid file has a byte between manifest and tensors or between tensors");
    bytes[gap] = 1;
    refused_with("gap", &bytes, E_CKPT_002, "is not zero");
}

#[test]
fn bytes_after_the_last_tensor_are_002() {
    let mut bytes = valid();
    let size = bytes.len() as u64 + 64;
    bytes.resize(size as usize, 0);
    let bytes = edited(bytes, |m| m["file_size"] = size.into());
    refused_with("trailing", &bytes, E_CKPT_002, "end of its last tensor");
}

#[test]
fn a_manifest_without_a_payload_hash_is_001() {
    let bytes = valid();
    let bytes = edited(bytes, |m| {
        m.as_object_mut().unwrap().remove("payload_hash");
    });
    refused_with("no_payload_hash", &bytes, E_CKPT_001, "payload_hash");
}
