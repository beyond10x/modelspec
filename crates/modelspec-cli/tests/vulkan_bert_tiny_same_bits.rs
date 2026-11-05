//! Three training steps of `examples/bert_tiny.mspec` on Vulkan keep the bits they had before
//! the step-time kernels of epic:bert-tiny-step-tenth: story:vulkan-cross-entropy-sum-per-row
//! (the `CrossEntropy` loss as a per-row dispatch and one ordered sum) and
//! story:vulkan-embedding-grad-per-element (`EmbeddingGrad` one invocation per element).
//!
//! The run is the example with `steps = 3` and `warmup_steps = 1` (a run of 3 refuses the
//! example's warm-up of 1,000), over the packed corpus at sequence length 128 that
//! `MODELSPEC_GPU_PRETRAIN_DATA` names, the directory `adversary_gpu_pretrain_cli` reads. The
//! recorded values below are the `f32` bits of the loss of steps 0 to 3 from the
//! `StepCompleted` events of `--record`, and per parameter an FNV-1a digest of the bits of its
//! values on the final `param` lines. A parameter after 3 AdamW steps depends on each step's
//! gradient, the embedding tables' on the `EmbeddingGrad` of each step, so a gradient whose
//! bits change shows as a digest that changes.
//!
//! The values were first read on the RTX 3090 with the binary of 0782eb1d, before either story,
//! and stood through release 0.25.0. They were recorded again on 2026-11-05 (driver
//! 2580660800) after story:vulkan-cross-entropy-row-workgroup, a change of the tolerance class
//! (DESIGN.md § 5.4): it moves the loss of steps 0 and 2 by one and two units in the last place
//! and with it 36 of the 41 parameter digests. On that day and driver the 0.25.0 release binary
//! still gave the former values, and this tree gave the values below with the one-dispatch
//! AdamW update (story:vulkan-adamw-one-dispatch-per-parameter) scheduled and not scheduled,
//! so neither the driver nor the bit-keeping stories of that day moved them.
//!
//! Run it with
//! `MODELSPEC_GPU_PRETRAIN_DATA=<packed-dir> cargo test --release -p modelspec-cli --test vulkan_bert_tiny_same_bits -- --ignored`.
#![cfg(feature = "vulkan")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

const EXAMPLE: &str = "examples/bert_tiny.mspec";
const DATA_ENV: &str = "MODELSPEC_GPU_PRETRAIN_DATA";

/// The `data` line of the recorded run: the packed corpus the values below belong to.
const DATA_LINE: &str = "data dataset 0ddcfa78-c328-8cc9-ae3b-490c096d55a5 tokenizer \
                         dfd98df4-cbd0-8076-b713-99ab202efa85 seq_len 128 masking \
                         0.15/0.8/0.1/0.1 binding 668f3433-dbd7-8871-a5a7-a421c2bd1816";

/// The bits of the `f32` loss of steps 0, 1, 2 and 3.
const LOSS_BITS: [u32; 4] = [0x4110_91cf, 0x4110_a2f1, 0x4110_a2d8, 0x4110_5628];

/// Per parameter path, the FNV-1a digest of its final values' bits ([`digest`]).
const PARAM_DIGESTS: [(&str, u64); 41] = [
    ("BertTiny.emb_ln_beta", 0x3d6a_a123_a88a_ec70),
    ("BertTiny.emb_ln_gamma", 0xff56_b931_d60a_0742),
    ("BertTiny.l0_b1", 0xcbbc_fce5_6d5a_581a),
    ("BertTiny.l0_b2", 0x99cc_8187_12c4_4f7e),
    ("BertTiny.l0_bk", 0x43e9_9565_efd2_d737),
    ("BertTiny.l0_bo", 0x62a3_05c2_688b_e4d7),
    ("BertTiny.l0_bq", 0x2c73_4e1f_f394_4768),
    ("BertTiny.l0_bv", 0x7251_674f_ac2d_2737),
    ("BertTiny.l0_ln1_beta", 0x33ed_b7a8_7ebe_7997),
    ("BertTiny.l0_ln1_gamma", 0x608b_6bee_4040_6394),
    ("BertTiny.l0_ln2_beta", 0xa3d0_e5c1_89d1_e2de),
    ("BertTiny.l0_ln2_gamma", 0x17c8_5ae2_de64_802d),
    ("BertTiny.l0_w1", 0x5335_f4d9_3be1_8e7e),
    ("BertTiny.l0_w2", 0xfca1_e6ca_bcc3_b6a5),
    ("BertTiny.l0_wk", 0x3448_c90d_8a63_6dc7),
    ("BertTiny.l0_wo", 0x2ad9_b471_14f3_5194),
    ("BertTiny.l0_wq", 0x7756_de45_9ff8_ae69),
    ("BertTiny.l0_wv", 0x029e_c6a8_11f7_4c68),
    ("BertTiny.l1_b1", 0x3fb2_537c_b905_6fe2),
    ("BertTiny.l1_b2", 0x0408_c5bc_271f_ec58),
    ("BertTiny.l1_bk", 0x02b5_456f_5377_f020),
    ("BertTiny.l1_bo", 0x5eee_fe29_a4e9_73a9),
    ("BertTiny.l1_bq", 0x782c_31d3_06f7_07fb),
    ("BertTiny.l1_bv", 0x4da0_17e6_62ca_35d9),
    ("BertTiny.l1_ln1_beta", 0xf290_d14f_415f_8ede),
    ("BertTiny.l1_ln1_gamma", 0x87f4_f9e0_4f7e_c5d1),
    ("BertTiny.l1_ln2_beta", 0xd67c_5764_592d_9bbf),
    ("BertTiny.l1_ln2_gamma", 0x9da9_8734_cc18_c188),
    ("BertTiny.l1_w1", 0xe216_2558_61b1_4227),
    ("BertTiny.l1_w2", 0x8abf_fc2a_ed13_6b3f),
    ("BertTiny.l1_wk", 0x5dc1_5d08_b0c1_843f),
    ("BertTiny.l1_wo", 0x0168_a3d3_7367_1b51),
    ("BertTiny.l1_wq", 0xc124_fb60_5908_a648),
    ("BertTiny.l1_wv", 0x37d6_fdc1_5493_ec6f),
    ("BertTiny.mlm_b", 0x8726_8c21_1865_413b),
    ("BertTiny.mlm_bias", 0x57e9_0263_4240_8d83),
    ("BertTiny.mlm_ln_beta", 0x4274_188e_d021_3c54),
    ("BertTiny.mlm_ln_gamma", 0x7aac_d357_9d1a_06d1),
    ("BertTiny.mlm_w", 0x9150_4e98_c871_21b4),
    ("BertTiny.pos_table", 0xef9b_c014_42e6_5410),
    ("BertTiny.tok_table", 0x2aa4_3fa2_88d1_6621),
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn path(p: &Path) -> &str {
    p.to_str().expect("a UTF-8 path")
}

/// FNV-1a over the little-endian bytes of each value's bits, in order.
fn digest(values: &[f32]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// The values of every `param @<path>: <type> = [v, ...]` line, by path.
fn params(out: &str) -> BTreeMap<String, Vec<f32>> {
    out.lines()
        .filter_map(|line| line.strip_prefix("param @"))
        .map(|rest| {
            let (path, rest) = rest.split_once(": ").expect("`@<path>: <type> = [..]`");
            let (_, list) = rest.split_once(" = [").expect("a value list");
            let list = list.strip_suffix(']').expect("a closed list");
            let values = list
                .split(", ")
                .map(|v| v.parse::<f32>().expect("an f32"))
                .collect();
            (path.to_owned(), values)
        })
        .collect()
}

/// The `(step, loss)` of every `StepCompleted` event of a `--record` file.
fn step_losses(events: &str) -> Vec<(u64, f64)> {
    events
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("an event is JSON"))
        .filter(|event| event["body"]["kind"] == "StepCompleted")
        .map(|event| {
            let value = &event["body"]["value"];
            (
                value["step"].as_u64().expect("a step"),
                value["loss"].as_f64().expect("a loss"),
            )
        })
        .collect()
}

#[test]
#[ignore = "needs a Vulkan device and MODELSPEC_GPU_PRETRAIN_DATA"]
fn device_three_bert_tiny_steps_keep_their_bits() {
    let data =
        PathBuf::from(std::env::var(DATA_ENV).unwrap_or_else(|_| panic!("{DATA_ENV} is set")));
    let case = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("vulkan_bert_tiny_same_bits");
    let _ = std::fs::remove_dir_all(&case);
    std::fs::create_dir_all(&case).expect("scratch directory");
    let mut source = std::fs::read_to_string(repo_root().join(EXAMPLE)).expect("the example");
    for (from, to) in [
        ("warmup_steps = 1000)", "warmup_steps = 1)"),
        ("    steps = 10000;", "    steps = 3;"),
    ] {
        assert!(source.contains(from), "the example holds {from:?}");
        source = source.replace(from, to);
    }
    let file = case.join("bert_tiny_3.mspec");
    std::fs::write(&file, source).expect("the variant is written");
    let events = case.join("events.jsonl");
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args([
            "train",
            path(&file),
            "--backend",
            "vulkan",
            "--data",
            path(&data),
            "--record",
            path(&events),
        ])
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        stdout.lines().find(|l| l.starts_with("data ")),
        Some(DATA_LINE),
        "the recorded values belong to this packed corpus"
    );

    let recorded = std::fs::read_to_string(&events).expect("the event file");
    let losses = step_losses(&recorded);
    let bits: Vec<(u64, u32)> = losses
        .iter()
        .map(|&(step, loss)| {
            let narrow = loss as f32;
            assert_eq!(f64::from(narrow), loss, "step {step}: an f32 loss");
            (step, narrow.to_bits())
        })
        .collect();
    let digests: Vec<(String, u64)> = params(&stdout)
        .iter()
        .map(|(path, values)| (path.clone(), digest(values)))
        .collect();
    eprintln!("loss bits {bits:#010x?}\nparameter digests {digests:#018x?}");

    let expected: Vec<(u64, u32)> = (0..).zip(LOSS_BITS).collect();
    assert_eq!(bits, expected, "the loss bits of steps 0 to 3");
    let expected: Vec<(String, u64)> = PARAM_DIGESTS
        .iter()
        .map(|&(path, d)| (path.to_owned(), d))
        .collect();
    assert_eq!(digests, expected, "the final parameters' bits");
}
