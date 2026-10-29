//! Three training steps of `examples/bert_tiny.mspec` on Vulkan keep the bits they had before
//! the step-time kernels of epic:bert-tiny-step-tenth: story:vulkan-cross-entropy-sum-per-row
//! (the `CrossEntropy` loss as a per-row dispatch and one ordered sum) and
//! story:vulkan-embedding-grad-per-element (`EmbeddingGrad` one invocation per element).
//!
//! The run is the example with `steps = 3` and `warmup_steps = 1` (a run of 3 refuses the
//! example's warm-up of 1,000), over the packed corpus at sequence length 128 that
//! `MODELSPEC_GPU_PRETRAIN_DATA` names, the directory `adversary_gpu_pretrain_cli` reads. The
//! recorded values below were read on the RTX 3090 with the binary of 0782eb1d, before either
//! story: the `f32` bits of the loss of steps 0 to 3 from the `StepCompleted` events of
//! `--record`, and per parameter an FNV-1a digest of the bits of its values on the final `param`
//! lines. A parameter after 3 AdamW steps depends on each step's gradient, the embedding tables'
//! on the `EmbeddingGrad` of each step, so a gradient whose bits change shows as a digest that
//! changes.
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
const LOSS_BITS: [u32; 4] = [0x4110_91ce, 0x4110_a2f1, 0x4110_a2d6, 0x4110_5628];

/// Per parameter path, the FNV-1a digest of its final values' bits ([`digest`]).
const PARAM_DIGESTS: [(&str, u64); 41] = [
    ("BertTiny.emb_ln_beta", 0xe9e1_720b_e247_90a2),
    ("BertTiny.emb_ln_gamma", 0xff56_b931_d60a_0742),
    ("BertTiny.l0_b1", 0xd994_99ea_77da_48c5),
    ("BertTiny.l0_b2", 0x77c6_3b18_e317_5517),
    ("BertTiny.l0_bk", 0xe21f_cc6a_2a76_6d9c),
    ("BertTiny.l0_bo", 0xf01c_3bff_afd0_55d0),
    ("BertTiny.l0_bq", 0xa552_8694_9837_0f62),
    ("BertTiny.l0_bv", 0x454d_6dd9_3e0c_2fe5),
    ("BertTiny.l0_ln1_beta", 0x79c5_ec89_26fb_1693),
    ("BertTiny.l0_ln1_gamma", 0xb6ea_550c_3710_3fa1),
    ("BertTiny.l0_ln2_beta", 0x15f1_6d64_33e4_16ae),
    ("BertTiny.l0_ln2_gamma", 0x17c8_5ae2_de64_802d),
    ("BertTiny.l0_w1", 0x5dcc_3df7_06c4_7bd3),
    ("BertTiny.l0_w2", 0xa9fe_fae2_521f_17f4),
    ("BertTiny.l0_wk", 0xca6a_46d9_492b_e421),
    ("BertTiny.l0_wo", 0xd656_f6b1_2a96_8ed2),
    ("BertTiny.l0_wq", 0x4337_e770_4f28_6685),
    ("BertTiny.l0_wv", 0x3848_0984_901a_e574),
    ("BertTiny.l1_b1", 0xcd30_66fc_e9ee_316e),
    ("BertTiny.l1_b2", 0x219a_4144_958f_c51f),
    ("BertTiny.l1_bk", 0x2128_6391_e24c_c645),
    ("BertTiny.l1_bo", 0x416a_8499_2574_f7e2),
    ("BertTiny.l1_bq", 0xb46d_f9f6_f2ad_49b7),
    ("BertTiny.l1_bv", 0x72a9_7129_8a0a_ec48),
    ("BertTiny.l1_ln1_beta", 0x326b_06bb_38e4_0153),
    ("BertTiny.l1_ln1_gamma", 0x87f4_f9e0_4f7e_c5d1),
    ("BertTiny.l1_ln2_beta", 0xa8b4_cb17_279e_d705),
    ("BertTiny.l1_ln2_gamma", 0x9da9_8734_cc18_c188),
    ("BertTiny.l1_w1", 0x1df3_a2e4_eb0c_cea7),
    ("BertTiny.l1_w2", 0x6bdf_856c_44ef_fe40),
    ("BertTiny.l1_wk", 0xbaec_25f3_9bbe_fd7d),
    ("BertTiny.l1_wo", 0x785c_b05c_12b4_2fbb),
    ("BertTiny.l1_wq", 0xb499_4c28_4fa9_7b9e),
    ("BertTiny.l1_wv", 0x3a0e_c1b3_b515_7156),
    ("BertTiny.mlm_b", 0x3aaa_7d74_a8d5_77fa),
    ("BertTiny.mlm_bias", 0x2d78_ea21_56c0_890c),
    ("BertTiny.mlm_ln_beta", 0x715e_4e7d_2e62_8992),
    ("BertTiny.mlm_ln_gamma", 0x7aac_d357_9d1a_06d1),
    ("BertTiny.mlm_w", 0x66ab_c985_229c_61c4),
    ("BertTiny.pos_table", 0x05a6_eafc_44d1_2513),
    ("BertTiny.tok_table", 0x8cf2_c73b_565d_b8cd),
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
