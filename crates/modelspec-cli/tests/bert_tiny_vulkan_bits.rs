//! `examples/bert_tiny.mspec` trained for 3 steps on Vulkan (the file with `steps = 3` and a
//! warm-up of 1 step, its only changed lines) gives the bits recorded from the
//! attention kernels of 0.21.0, which recompute each query's softmax statistics in every key
//! invocation (story:vulkan-attention-softmax-stats-reused, story:vulkan-attention-workgroup-per-head):
//! the loss of every step and every parameter after step 3.
//!
//! The data is a dense directory the test writes: three blocks of `ids`, `positions`, a key mask
//! that masks the tail keys of most sequences, `labels` and `weights` of 0 and 1. The recording
//! holds the bits of each step's loss and a 64-bit FNV-1a hash of the `param` lines, which print
//! every element in the shortest form that reads back to its bits, so one changed bit of one
//! parameter changes it. A device's `exp`, division and square root are its own, so the recording
//! belongs to one device and driver, named beside it; on another the test fails and names both.
//!
//! The case needs a Vulkan device and is ignored as the device tests are; run it with
//! `cargo test --release -p modelspec-cli --features vulkan --test bert_tiny_vulkan_bits -- --ignored`.
#![cfg(feature = "vulkan")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

const EXAMPLE: &str = "examples/bert_tiny.mspec";

/// `vendor_id`, `device_id` and `driver_version` of the device the recording was taken on, as
/// `inspect-device` prints them: the RTX 3090 of the repository's device tests.
const RECORDED_DEVICE: (u64, u64, u64) = (4318, 8708, 2_559_459_584);

/// The bits of the loss of steps 0 to 3 on the recorded device.
const RECORDED_LOSSES: [u32; 4] = [0x4110_6b45, 0x4110_5fa8, 0x4110_562d, 0x410f_da54];

/// FNV-1a 64 of the `param` lines of stdout, each with its newline, on the recorded device.
const RECORDED_PARAMETERS: u64 = 0x7e11_2f36_223d_7cb2;

/// The rows, sequences, keys per sequence and vocabulary of `examples/bert_tiny.mspec`.
const ROWS: usize = 4096;
const SEQUENCES: usize = 32;
const KEYS: usize = 128;
const VOCABULARY: u64 = 8192;
const BLOCKS: usize = 3;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn ok(output: &Output) -> String {
    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout: {}\nstderr: {}",
        text(&output.stdout),
        text(&output.stderr)
    );
    text(&output.stdout)
}

fn path(p: &Path) -> &str {
    p.to_str().expect("a UTF-8 path")
}

/// A fresh, empty directory `name` under this crate's build directory.
fn fresh(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("bert_tiny_vulkan_bits")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("the old fixture is removed");
    }
    std::fs::create_dir_all(&dir).expect("the build directory is writable");
    dir
}

/// A deterministic stream of fixture values: a 64-bit LCG.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn i32_file(dir: &Path, name: &str, values: &[i32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.join(format!("{name}.i32")), bytes).expect("the data file is written");
}

fn f32_file(dir: &Path, name: &str, values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.join(format!("{name}.f32")), bytes).expect("the data file is written");
}

/// Three blocks of the five declarations of the example's `train` block. Sequence `b` of block
/// `k` keeps its first `128 - (7b + 3k) % 48` keys, so most sequences have masked keys and none
/// has all of them masked; about 15 % of the rows carry weight 1.
fn write_data(dir: &Path) {
    let mut s = Stream(0x5eed_0b17);
    let (mut ids, mut positions, mut mask, mut labels, mut weights) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for k in 0..BLOCKS {
        for row in 0..ROWS {
            ids.push(s.below(VOCABULARY) as i32);
            positions.push((row % KEYS) as i32);
            labels.push(s.below(VOCABULARY) as i32);
            weights.push(if s.below(100) < 15 { 1.0f32 } else { 0.0 });
        }
        for b in 0..SEQUENCES {
            let kept = KEYS - (7 * b + 3 * k) % 48;
            mask.extend((0..KEYS).map(|key| if key < kept { 1.0f32 } else { 0.0 }));
        }
    }
    i32_file(dir, "ids", &ids);
    i32_file(dir, "positions", &positions);
    f32_file(dir, "key_mask", &mask);
    i32_file(dir, "labels", &labels);
    f32_file(dir, "weights", &weights);
}

/// The example with `steps = 3` and a warm-up of 1 step, the only lines that differ, written
/// into `dir`: a run's warm-up must be shorter than the run.
fn three_steps(dir: &Path) -> PathBuf {
    let mut source = std::fs::read_to_string(repo_root().join(EXAMPLE)).expect("the example reads");
    for (from, to) in [
        ("warmup_steps = 1000);", "warmup_steps = 1);"),
        ("steps = 10000;", "steps = 3;"),
    ] {
        assert!(source.contains(from), "the example holds {from:?}");
        source = source.replacen(from, to, 1);
    }
    let file = dir.join("bert_tiny_3.mspec");
    std::fs::write(&file, source).expect("the variant is written");
    file
}

fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// `vendor_id`, `device_id` and `driver_version` of device 0 as `inspect-device` prints them.
fn device_zero() -> (u64, u64, u64) {
    let out = ok(&modelspec(&["inspect-device", "--backend", "vulkan"]));
    let field = |name: &str| {
        out.lines()
            .find_map(|line| line.strip_prefix(&format!("{name}: ")))
            .unwrap_or_else(|| panic!("inspect-device prints {name}"))
            .parse::<u64>()
            .unwrap_or_else(|_| panic!("{name} is a number"))
    };
    (
        field("vendor_id"),
        field("device_id"),
        field("driver_version"),
    )
}

/// The bits of each `StepCompleted` loss of the recording, in step order.
fn recorded_losses(file: &Path) -> Vec<(u64, u32)> {
    std::fs::read_to_string(file)
        .expect("the recording")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("a JSON event"))
        .filter(|event| event["body"]["kind"] == "StepCompleted")
        .map(|event| {
            let value = &event["body"]["value"];
            let loss = value["loss"].as_f64().expect("a finite loss") as f32;
            (value["step"].as_u64().expect("a step"), loss.to_bits())
        })
        .collect()
}

/// Acceptance 1 of both stories: 3 steps of `examples/bert_tiny.mspec` on Vulkan give the
/// recorded loss of every step and the recorded parameters after step 3, bit for bit.
#[test]
#[ignore = "needs a Vulkan device"]
fn bert_tiny_three_vulkan_steps_give_the_recorded_bits() {
    let device = device_zero();
    assert_eq!(
        device, RECORDED_DEVICE,
        "the recording is of device {RECORDED_DEVICE:?} (vendor, device, driver); device 0 is \
         {device:?}: record its bits from the 0.21.0 kernels before comparing on it"
    );
    let case = fresh("three_steps");
    let data = case.join("data");
    std::fs::create_dir_all(&data).expect("the data directory");
    write_data(&data);
    let file = three_steps(&case);
    let recording = case.join("run.jsonl");
    let stdout = ok(&modelspec(&[
        "train",
        path(&file),
        "--backend",
        "vulkan",
        "--data",
        path(&data),
        "--record",
        path(&recording),
    ]));
    let losses = recorded_losses(&recording);
    let params: Vec<&str> = stdout
        .lines()
        .filter(|line| line.starts_with("param @"))
        .collect();
    let hash = fnv1a(params.iter().flat_map(|line| line.bytes().chain(*b"\n")));
    let got: Vec<u32> = losses.iter().map(|&(_, bits)| bits).collect();
    eprintln!(
        "losses {:?} bits [{}] parameters {} lines fnv1a {hash:#018x}",
        losses
            .iter()
            .map(|&(k, bits)| (k, f32::from_bits(bits)))
            .collect::<Vec<_>>(),
        got.iter()
            .map(|bits| format!("{bits:#010x}"))
            .collect::<Vec<_>>()
            .join(", "),
        params.len(),
    );
    assert_eq!(
        losses.iter().map(|&(k, _)| k).collect::<Vec<_>>(),
        vec![0, 1, 2, 3],
        "one loss per step"
    );
    assert_eq!(params.len(), 41, "one line per parameter of the example");
    assert_eq!(got, RECORDED_LOSSES, "the loss bits of steps 0 to 3");
    assert_eq!(
        hash, RECORDED_PARAMETERS,
        "the parameters after step 3 differ from the recorded bits"
    );
}
