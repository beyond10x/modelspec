//! `modelspec train --masked-rows <C>` (story:vulkan-mlm-masked-row-gather, DESIGN.md § 3.3): the
//! step computes the MLM head, the vocabulary projection, its bias, the cross-entropy loss and
//! its gradient on the rows of non-zero weight only, gathered at capacity `C`, and gives the bits
//! of the step that computes every row.
//!
//! The CPU cases train the hidden-32 twin of BERT-tiny (`tests/fixtures/bert_cpu.mspec`) for 3
//! steps on a dense directory the test writes and compare the run with and without the gather
//! byte for byte; a batch with more selected rows than the capacity is refused with E-DATA-012.
//! The device cases, ignored as the device tests are, train `examples/bert_tiny.mspec` for 3
//! steps on the data of `bert_tiny_three_vulkan_steps_give_the_recorded_bits` (the bits test of
//! story:vulkan-attention-softmax-stats-reused) and compare with its recorded bits:
//! `cargo test --release -p modelspec-cli --features vulkan --test bert_tiny_masked_rows -- --ignored`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const EXAMPLE: &str = "examples/bert_tiny.mspec";
const TWIN: &str = "crates/modelspec-cli/tests/fixtures/bert_cpu.mspec";
const LINEAR: &str = "examples/linear_regression.mspec";

/// The rows, sequences, keys per sequence and vocabulary of a model's batch.
struct Shape {
    rows: usize,
    sequences: usize,
    keys: usize,
    vocabulary: u64,
}

const BERT_TINY: Shape = Shape {
    rows: 4096,
    sequences: 32,
    keys: 128,
    vocabulary: 8192,
};

const TWIN_SHAPE: Shape = Shape {
    rows: 32,
    sequences: 2,
    keys: 16,
    vocabulary: 300,
};

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

fn exits(output: &Output, code: i32) -> String {
    assert_eq!(
        output.status.code(),
        Some(code),
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
        .join("bert_tiny_masked_rows")
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

/// Three blocks of the five declarations, as the bits test writes them: sequence `b` of block `k`
/// keeps its first `keys - (7b + 3k) % (keys * 3 / 8)` keys, and about 15 % of the rows carry
/// weight 1. Returns the selected rows of each block.
fn write_data(dir: &Path, shape: &Shape) -> Vec<usize> {
    let mut s = Stream(0x5eed_0b17);
    let (mut ids, mut positions, mut mask, mut labels, mut weights) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut selected = Vec::new();
    for k in 0..BLOCKS {
        let mut count = 0;
        for row in 0..shape.rows {
            ids.push(s.below(shape.vocabulary) as i32);
            positions.push((row % shape.keys) as i32);
            labels.push(s.below(shape.vocabulary) as i32);
            let weight = if s.below(100) < 15 { 1.0f32 } else { 0.0 };
            count += usize::from(weight != 0.0);
            weights.push(weight);
        }
        for b in 0..shape.sequences {
            let kept = shape.keys - (7 * b + 3 * k) % (shape.keys * 3 / 8);
            mask.extend((0..shape.keys).map(|key| if key < kept { 1.0f32 } else { 0.0 }));
        }
        selected.push(count);
    }
    i32_file(dir, "ids", &ids);
    i32_file(dir, "positions", &positions);
    f32_file(dir, "key_mask", &mask);
    i32_file(dir, "labels", &labels);
    f32_file(dir, "weights", &weights);
    selected
}

/// The file `model` with `steps = 3` and a warm-up of 1 step, the only lines that differ, written
/// into `dir`.
fn three_steps(dir: &Path, model: &str, warmup: &str, steps: &str) -> PathBuf {
    let mut source = std::fs::read_to_string(repo_root().join(model)).expect("the model reads");
    for (from, to) in [(warmup, "warmup_steps = 1);"), (steps, "steps = 3;")] {
        assert!(source.contains(from), "the model holds {from:?}");
        source = source.replacen(from, to, 1);
    }
    let file = dir.join("three_steps.mspec");
    std::fs::write(&file, source).expect("the variant is written");
    file
}

/// The twin for 3 steps and its data, written under `name`; the selected rows of each block.
fn twin_case(name: &str) -> (PathBuf, PathBuf, Vec<usize>) {
    let case = fresh(name);
    let data = case.join("data");
    std::fs::create_dir_all(&data).expect("the data directory");
    let selected = write_data(&data, &TWIN_SHAPE);
    let file = three_steps(&case, TWIN, "warmup_steps = 80);", "steps = 800;");
    (file, data, selected)
}

fn train(file: &Path, data: &Path, backend: &str, extra: &[&str]) -> Output {
    let mut args = vec![
        "train",
        path(file),
        "--backend",
        backend,
        "--data",
        path(data),
    ];
    args.extend_from_slice(extra);
    modelspec(&args)
}

/// On the CPU, the twin's 3 steps with the gather print what they print without it, byte for
/// byte: every loss and every parameter after the last step.
#[test]
fn the_gathered_cpu_twin_prints_the_ungathered_run() {
    let (file, data, selected) = twin_case("cpu_twin");
    assert!(
        selected.iter().all(|&n| (1..=16).contains(&n)),
        "{selected:?}"
    );
    let full = exits(&train(&file, &data, "cpu", &[]), 0);
    let gathered = exits(&train(&file, &data, "cpu", &["--masked-rows", "16"]), 0);
    assert!(full.lines().any(|line| line.starts_with("param @")));
    assert_eq!(gathered, full);
}

/// A capacity below a batch's selected rows is refused with E-DATA-012 before the step that
/// would read it, naming the step, the count and the capacity; no row is dropped.
#[test]
fn a_batch_past_the_capacity_is_refused_on_the_cpu() {
    let (file, data, selected) = twin_case("cpu_refused");
    let output = train(&file, &data, "cpu", &["--masked-rows", "1"]);
    exits(&output, 1);
    let stderr = text(&output.stderr);
    assert!(stderr.contains("E-DATA-012"), "{stderr}");
    assert!(stderr.contains("step 0"), "{stderr}");
    assert!(
        stderr.contains(&format!("{} rows", selected[0])),
        "{stderr}"
    );
    assert!(!text(&output.stdout).contains("step 0"), "no step ran");
}

/// `--masked-rows` on a model whose loss is no weighted `cross_entropy` is a usage error.
#[test]
fn masked_rows_need_a_weighted_cross_entropy() {
    let output = modelspec(&["train", LINEAR, "--backend", "cpu", "--masked-rows", "4"]);
    exits(&output, 2);
    assert!(
        text(&output.stderr).contains("--masked-rows")
            && text(&output.stderr).contains("cross_entropy"),
        "{}",
        text(&output.stderr)
    );
}

/// `--masked-rows` beside `--eval`, whose dense batches the step's loss would read through the
/// gather of the last trained batch, is a usage error.
#[test]
fn masked_rows_refuse_dense_eval() {
    let (file, data, _) = twin_case("cpu_eval");
    let output = train(
        &file,
        &data,
        "cpu",
        &["--masked-rows", "16", "--eval", path(&data)],
    );
    exits(&output, 2);
    assert!(
        text(&output.stderr).contains("--eval"),
        "{}",
        text(&output.stderr)
    );
}

#[cfg(feature = "vulkan")]
mod device {
    use super::*;
    use serde_json::Value;

    /// `vendor_id`, `device_id` and `driver_version` of the device the recording was taken on:
    /// the RTX 3090 of the repository's device tests.
    const RECORDED_DEVICE: (u64, u64, u64) = (4318, 8708, 2_559_459_584);

    /// The bits of the loss of steps 0 to 3 on the recorded device, as
    /// `bert_tiny_three_vulkan_steps_give_the_recorded_bits` records them.
    const RECORDED_LOSSES: [u32; 4] = [0x4110_6b45, 0x4110_5fa8, 0x4110_562d, 0x410f_da54];

    /// FNV-1a 64 of the `param` lines of stdout, each with its newline, on the recorded device.
    const RECORDED_PARAMETERS: u64 = 0x7e11_2f36_223d_7cb2;

    fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
        bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    fn device_zero() -> (u64, u64, u64) {
        let out = exits(&modelspec(&["inspect-device", "--backend", "vulkan"]), 0);
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

    /// Acceptance 1: 3 Vulkan steps of `examples/bert_tiny.mspec` gathered at 768 rows give the
    /// recorded loss of every step and the recorded parameters after step 3, bit for bit, as the
    /// step that computes every row does.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn bert_tiny_gathered_three_vulkan_steps_give_the_recorded_bits() {
        let device = device_zero();
        assert_eq!(
            device, RECORDED_DEVICE,
            "the recording is of device {RECORDED_DEVICE:?}; device 0 is {device:?}"
        );
        let case = fresh("three_steps_gathered");
        let data = case.join("data");
        std::fs::create_dir_all(&data).expect("the data directory");
        let selected = write_data(&data, &BERT_TINY);
        assert!(selected.iter().all(|&n| n <= 768), "{selected:?}");
        let file = three_steps(&case, EXAMPLE, "warmup_steps = 1000);", "steps = 10000;");
        for (name, extra) in [
            ("full", &[][..]),
            ("gathered", &["--masked-rows", "768"][..]),
        ] {
            let recording = case.join(format!("{name}.jsonl"));
            let mut args = vec!["--record", path(&recording)];
            args.extend_from_slice(extra);
            let stdout = exits(&train(&file, &data, "vulkan", &args), 0);
            let losses = recorded_losses(&recording);
            let params: Vec<&str> = stdout
                .lines()
                .filter(|line| line.starts_with("param @"))
                .collect();
            let hash = fnv1a(params.iter().flat_map(|line| line.bytes().chain(*b"\n")));
            let got: Vec<u32> = losses.iter().map(|&(_, bits)| bits).collect();
            eprintln!(
                "{name}: bits [{}] parameters {} lines fnv1a {hash:#018x}",
                got.iter()
                    .map(|bits| format!("{bits:#010x}"))
                    .collect::<Vec<_>>()
                    .join(", "),
                params.len(),
            );
            assert_eq!(params.len(), 41, "{name}: one line per parameter");
            assert_eq!(
                got, RECORDED_LOSSES,
                "{name}: the loss bits of steps 0 to 3"
            );
            assert_eq!(
                hash, RECORDED_PARAMETERS,
                "{name}: the parameters after step 3"
            );
        }
    }

    /// On Vulkan, the twin's 3 steps with the gather print what they print without it.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn the_gathered_vulkan_twin_prints_the_ungathered_run() {
        let (file, data, _) = twin_case("vulkan_twin");
        let full = exits(&train(&file, &data, "vulkan", &[]), 0);
        let gathered = exits(&train(&file, &data, "vulkan", &["--masked-rows", "16"]), 0);
        assert!(full.lines().any(|line| line.starts_with("param @")));
        assert_eq!(gathered, full);
    }

    /// Acceptance 2 on Vulkan: a batch past the capacity is refused with E-DATA-012.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn a_batch_past_the_capacity_is_refused_on_vulkan() {
        let (file, data, selected) = twin_case("vulkan_refused");
        let output = train(&file, &data, "vulkan", &["--masked-rows", "1"]);
        exits(&output, 1);
        let stderr = text(&output.stderr);
        assert!(stderr.contains("E-DATA-012"), "{stderr}");
        assert!(
            stderr.contains(&format!("{} rows", selected[0])),
            "{stderr}"
        );
    }
}
