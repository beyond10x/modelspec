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
//! so neither the driver nor the bit-keeping stories of that day moved them. They were recorded
//! once more on the same day and driver with story:vulkan-matmul-split-k, a change of the
//! tolerance class too (DESIGN.md ADR-010): the weight gradients' sums over slices of `k` move
//! the loss of step 1 by one unit in the last place (`0x4110a2f1` before) and with it
//! 38 of the 41 parameter digests.
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
const LOSS_BITS: [u32; 4] = [0x4110_91cf, 0x4110_a2f0, 0x4110_a2d8, 0x4110_5628];

/// Per parameter path, the FNV-1a digest of its final values' bits ([`digest`]).
const PARAM_DIGESTS: [(&str, u64); 41] = [
    ("BertTiny.emb_ln_beta", 0x65dd_e66c_2ee9_a1a4),
    ("BertTiny.emb_ln_gamma", 0xff56_b931_d60a_0742),
    ("BertTiny.l0_b1", 0x0ef0_46aa_f140_fa19),
    ("BertTiny.l0_b2", 0x0eaa_bd0e_4a7b_f8f0),
    ("BertTiny.l0_bk", 0x530b_e11e_0231_1b0b),
    ("BertTiny.l0_bo", 0xc274_ab3a_a704_67ae),
    ("BertTiny.l0_bq", 0x2891_6274_e1d3_def1),
    ("BertTiny.l0_bv", 0x21fa_f811_44a4_4e26),
    ("BertTiny.l0_ln1_beta", 0x32c0_348e_e23e_80e6),
    ("BertTiny.l0_ln1_gamma", 0xb173_ddf0_271f_551f),
    ("BertTiny.l0_ln2_beta", 0x7c5e_7198_aef8_77ca),
    ("BertTiny.l0_ln2_gamma", 0x6aa2_2f19_e6e3_5914),
    ("BertTiny.l0_w1", 0x8310_b188_c9f0_b399),
    ("BertTiny.l0_w2", 0xdb17_3d60_6db1_0c91),
    ("BertTiny.l0_wk", 0xc996_5c77_cf70_7a92),
    ("BertTiny.l0_wo", 0x0570_5395_0306_50a6),
    ("BertTiny.l0_wq", 0x1f01_cac8_f98d_22ba),
    ("BertTiny.l0_wv", 0x7cd9_2c52_5347_7bac),
    ("BertTiny.l1_b1", 0x1d54_5ba7_0614_3c39),
    ("BertTiny.l1_b2", 0x5cb2_8832_27e5_ade3),
    ("BertTiny.l1_bk", 0x3924_05ba_ff38_bc26),
    ("BertTiny.l1_bo", 0x87dc_1a90_ede7_49d9),
    ("BertTiny.l1_bq", 0x3a9e_efc0_1940_496b),
    ("BertTiny.l1_bv", 0x2c5a_2a48_3cd8_8256),
    ("BertTiny.l1_ln1_beta", 0x1a43_03fc_3a4d_f375),
    ("BertTiny.l1_ln1_gamma", 0x87f4_f9e0_4f7e_c5d1),
    ("BertTiny.l1_ln2_beta", 0x0bcb_09cd_f059_e13e),
    ("BertTiny.l1_ln2_gamma", 0x58d8_dcad_2a20_218d),
    ("BertTiny.l1_w1", 0x2d61_036d_58dc_3d7e),
    ("BertTiny.l1_w2", 0xf4c1_9622_0f36_06f0),
    ("BertTiny.l1_wk", 0xe242_4886_d0f3_53fe),
    ("BertTiny.l1_wo", 0x238d_19fb_54ed_c98c),
    ("BertTiny.l1_wq", 0xeb2d_80df_5245_6c0a),
    ("BertTiny.l1_wv", 0x813e_756e_9007_b971),
    ("BertTiny.mlm_b", 0x3e46_f149_e509_9a8f),
    ("BertTiny.mlm_bias", 0x7086_e3e2_83c3_24af),
    ("BertTiny.mlm_ln_beta", 0x97e8_b0ec_adef_a035),
    ("BertTiny.mlm_ln_gamma", 0x7aac_d357_9d1a_06d1),
    ("BertTiny.mlm_w", 0x670a_ef04_95de_1afc),
    ("BertTiny.pos_table", 0x1a21_169b_dfc8_51a2),
    ("BertTiny.tok_table", 0x281c_33d5_bacb_b7ac),
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
