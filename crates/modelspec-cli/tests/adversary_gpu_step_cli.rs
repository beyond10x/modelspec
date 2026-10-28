//! Adversary pass 1 on story:bert-vulkan-kernels-attention-loss-adamw (coordinator decision D73):
//! the CLI side of the stepwise Vulkan run.
//!
//! Two kinds of ignore: a case whose name carries `device` needs a Vulkan device and is ignored
//! as the M1 device tests are ("needs a Vulkan device"); a case whose name carries a finding id
//! (`f2`, `f3`) was red on the unit's commit and is green after fix pass 1 (coordinator decision
//! D74), ignored only if it needs a device.
//! The others run in the default suite and need no device: the compile commands below take
//! `--device-profile` and open no loader.
//!
//! Run the device cases with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-cli --features vulkan --test adversary_gpu_step_cli -- --ignored`.
#![cfg(feature = "vulkan")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

#[allow(dead_code)]
#[path = "../../modelspec-runtime/tests/common/mod.rs"]
mod common;

#[path = "common/markov_corpus.rs"]
mod markov_corpus;

use common::{GPU_RUN_ATOL, GPU_RUN_RTOL, attention_key_biases, within, zero_gradient_bound};

const TWIN: &str = "crates/modelspec-cli/tests/fixtures/bert_cpu.mspec";
const ATTENTION: &str = "crates/modelspec-cli/tests/fixtures/attention_train.mspec";
const DROPOUT: &str = "crates/modelspec-cli/tests/fixtures/dropout_train.mspec";
const PROFILE: &str = "crates/modelspec-runtime/tests/fixtures/rtx3090.device.json";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("adversary_gpu_step_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
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

fn path(p: &Path) -> &str {
    p.to_str().expect("a UTF-8 path")
}

fn assert_exit(output: &Output, code: i32) {
    assert_eq!(
        output.status.code(),
        Some(code),
        "stdout: {}\nstderr: {}",
        text(&output.stdout),
        text(&output.stderr)
    );
}

/// `source` with each `(from, to)` replaced, every `from` present, written to `<dir>/<name>`.
fn variant(dir: &Path, name: &str, source: &str, edits: &[(&str, &str)]) -> String {
    let mut source = source.to_owned();
    for (from, to) in edits {
        assert!(source.contains(from), "the source holds {from:?}");
        source = source.replace(from, to);
    }
    let file = dir.join(name);
    std::fs::write(&file, source).expect("the variant is written");
    path(&file).to_owned()
}

fn fixture(rel: &str) -> String {
    std::fs::read_to_string(repo_root().join(rel)).expect("the fixture reads")
}

/// The synthetic corpus, a 300-token tokenizer and sequences of 16 packed under `case`, as
/// `vulkan_bert_cpu_steps.rs` makes them; returns the packed directory.
fn packed(case: &Path) -> PathBuf {
    let corpus = case.join("corpus");
    markov_corpus::write(&corpus);
    let dataset = case.join("dataset");
    assert_exit(
        &modelspec(&["data", "prepare", path(&corpus), "--out", path(&dataset)]),
        0,
    );
    let tokenizer = case.join("tokenizer.json");
    assert_exit(
        &modelspec(&[
            "data",
            "tokenizer",
            "--dataset",
            path(&dataset),
            "--vocab-size",
            "300",
            "--out",
            path(&tokenizer),
        ]),
        0,
    );
    let packed = case.join("packed");
    assert_exit(
        &modelspec(&[
            "data",
            "pack",
            "--dataset",
            path(&dataset),
            "--tokenizer",
            path(&tokenizer),
            "--seq-len",
            "16",
            "--out",
            path(&packed),
        ]),
        0,
    );
    packed
}

/// The twin with `steps` steps and a warm-up of `steps / 5`, as the unit's 10-step run.
fn twin(case: &Path, steps: u64) -> String {
    variant(
        case,
        &format!("twin{steps}.mspec"),
        &fixture(TWIN),
        &[
            ("steps = 800;", &format!("steps = {steps};")),
            (
                "warmup_steps = 80)",
                &format!("warmup_steps = {})", steps / 5),
            ),
        ],
    )
}

fn step_lines(out: &str) -> Vec<(u64, f32)> {
    out.lines()
        .filter_map(|line| line.strip_prefix("step "))
        .map(|rest| {
            let (step, loss) = rest.split_once(" loss ").expect("`step <k> loss <v>`");
            (step.parse().unwrap(), loss.parse().unwrap())
        })
        .collect()
}

fn eval_lines(out: &str) -> Vec<(u64, f32, f64)> {
    out.lines()
        .filter_map(|line| line.strip_prefix("eval "))
        .map(|rest| {
            let words: Vec<&str> = rest.split_whitespace().collect();
            (
                words[0].parse().unwrap(),
                words[2].parse().unwrap(),
                words[4].parse().unwrap(),
            )
        })
        .collect()
}

fn tensors(out: &str, label: &str) -> BTreeMap<String, Vec<f32>> {
    let prefix = format!("{label} @");
    out.lines()
        .filter_map(|line| line.strip_prefix(&prefix))
        .map(|rest| {
            let (name, rest) = rest.split_once(": ").expect("`@<path>: <type> = [..]`");
            let (_, list) = rest.split_once(" = [").expect("a value list");
            let list = list.strip_suffix(']').expect("a closed list");
            let values = list.split(", ").map(|v| v.parse().unwrap()).collect();
            (name.to_owned(), values)
        })
        .collect()
}

fn bound(c: f32) -> f64 {
    f64::from(GPU_RUN_ATOL) + f64::from(GPU_RUN_RTOL) * f64::from(c).abs()
}

/// `(|g - c| / bound, |g - c|, name, index, c, g)` of one parameter element.
type Worst = (f64, f64, String, usize, f32, f32);

/// The worst final parameter element of `gpu` against `cpu`: `(|g - c| / bound, |g - c|, name,
/// index, c, g)`, and every element outside the run tolerance.
fn worst_parameter(cpu: &str, gpu: &str) -> (Worst, Vec<String>) {
    let (c_params, g_params) = (tensors(cpu, "param"), tensors(gpu, "param"));
    assert_eq!(
        c_params.keys().collect::<Vec<_>>(),
        g_params.keys().collect::<Vec<_>>()
    );
    let mut worst = (0.0, 0.0, String::new(), 0, 0.0, 0.0);
    let mut outside = Vec::new();
    for (name, c) in &c_params {
        for (i, (&g, &c)) in g_params[name].iter().zip(c).enumerate() {
            let diff = (f64::from(g) - f64::from(c)).abs();
            let ratio = diff / bound(c);
            if ratio > worst.0 {
                worst = (ratio, diff, name.clone(), i, c, g);
            }
            if !within(g, c, c.abs(), GPU_RUN_ATOL, GPU_RUN_RTOL) && outside.len() < 16 {
                outside.push(format!("param @{name} [{i}]: gpu {g} cpu {c}"));
            }
        }
    }
    (worst, outside)
}

/// Hypothesis 2: a sequence of 513 or 1,024 keys, and two sequences of 513, are E-TARGET-011 at
/// the `attention` call of `compile --backend vulkan`, exit 1, nothing written to stdout; a
/// sequence of exactly 512 keys compiles. No device: the compile reads a device profile file.
#[test]
fn adv_gpu_step_attention_over_512_keys_is_e_target_011_at_its_call() {
    let case = scratch("keys");
    let source = fixture(ATTENTION);
    for (b, s, refused) in [
        (1u64, 512u64, false),
        (1, 513, true),
        (1, 1024, true),
        (2, 513, true),
    ] {
        let n = b * s;
        let file = variant(
            &case,
            &format!("att_{b}x{s}.mspec"),
            &source,
            &[
                ("[6, 4]", &format!("[{n}, 4]")),
                ("[2, 3]", &format!("[{b}, {s}]")),
            ],
        );
        let out = case.join(format!("art_{b}x{s}"));
        let output = modelspec(&[
            "compile",
            &file,
            "--backend",
            "vulkan",
            "--device-profile",
            PROFILE,
            "--output",
            path(&out),
        ]);
        let stderr = text(&output.stderr);
        if refused {
            assert_exit(&output, 1);
            assert!(
                output.stdout.is_empty(),
                "{b}x{s}: {}",
                text(&output.stdout)
            );
            assert!(
                stderr.starts_with("E-TARGET-011: operator `attention` has no Vulkan kernel yet"),
                "{b}x{s}: {stderr}"
            );
            assert!(
                stderr.contains(&format!("{file}:7:16")),
                "{b}x{s}: {stderr}"
            );
        } else {
            assert_exit(&output, 0);
        }
    }
}

/// Hypothesis 2: `examples/bert_tiny.mspec` (S = 128, V = 8,192, AdamW) schedules for Vulkan:
/// `ir --stage kernel` exits 0 and prints each of the eight new kernel kinds.
#[test]
fn adv_gpu_step_bert_tiny_schedules_the_eight_kernels() {
    let output = modelspec(&["ir", "examples/bert_tiny.mspec", "--stage", "kernel"]);
    assert_exit(&output, 0);
    let stdout = text(&output.stdout);
    for op in [
        " per_row Attention ",
        " per_row AttentionGradQ ",
        " per_row AttentionGradK ",
        " per_row AttentionGradV ",
        " loss_sum CrossEntropy ",
        " per_row CrossEntropyGrad ",
        " elementwise Sqrt ",
        " elementwise Div ",
    ] {
        assert!(
            stdout
                .lines()
                .any(|line| line.starts_with("kernel ") && line.contains(op)),
            "{op}"
        );
    }
}

/// Finding f3 (hypothesis 9): DESIGN.md § 8 said `--profile` refuses a stepwise run (one with
/// dropout, AdamW, `--data` or `--eval-every`), but `train --help`, from which the generated CLI
/// reference page is built, described `--profile` without that limit. Red on the unit; fix pass
/// 1 named the refusal in the help (coordinator decision D74). Since
/// story:bert-tiny-vs-pytorch-cuda a stepwise run is profiled, one query pool per step's command
/// buffer, and the help names the stepwise run's profile instead.
#[test]
fn adv_gpu_step_f3_help_names_the_stepwise_profile() {
    let output = modelspec(&["train", "--help"]);
    assert_exit(&output, 0);
    let help = text(&output.stdout);
    let paragraph: String = help
        .split("--profile")
        .nth(1)
        .expect("a --profile paragraph")
        .split("\n\n")
        .next()
        .unwrap_or_default()
        .to_owned();
    assert!(
        paragraph.contains("stepwise") && !paragraph.contains("refuses"),
        "the --profile help names no stepwise profile:\n{paragraph}"
    );
}

/// Hypothesis 9: on a stepwise run `--profile` prints the profile of every step's command buffer
/// on stderr, exit 0, the unprofiled run's stdout (story:bert-tiny-vs-pytorch-cuda,
/// which replaced the one-line refusal): each `step` site called once per step, each `final` site
/// once, one host wait per step plus one, and `step_total` with no uncounted site; and `--stats`
/// prints its twelve counters, `host waits` one per step plus one.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_gpu_step_device_profile_and_stats_on_a_stepwise_run() {
    let output = modelspec(&["train", DROPOUT, "--backend", "vulkan", "--profile"]);
    assert_exit(&output, 0);
    let plain = modelspec(&["train", DROPOUT, "--backend", "vulkan"]);
    assert_exit(&plain, 0);
    assert_eq!(text(&output.stdout), text(&plain.stdout));
    let stderr = text(&output.stderr);
    let dispatches: Vec<&str> = stderr
        .lines()
        .filter(|line| line.starts_with("profile dispatch "))
        .collect();
    assert!(
        dispatches
            .iter()
            .any(|line| line.starts_with("profile dispatch step ")),
        "{stderr}"
    );
    for line in &dispatches {
        let calls = if line.starts_with("profile dispatch step ") {
            " calls 30 "
        } else {
            " calls 1 "
        };
        assert!(line.contains(calls), "{line}");
    }
    assert!(stderr.contains("profile host_waits 31\n"), "{stderr}");
    assert!(
        stderr.contains("profile steady_us_per_step ")
            && !stderr.contains("profile steady_us_per_step unavailable"),
        "30 steps are past the 20-step warm-up:\n{stderr}"
    );
    let total = stderr.lines().last().expect("a step_total line").to_owned();
    assert!(
        total.starts_with("profile step_total flop ") && total.ends_with(" uncounted_sites 0"),
        "{stderr}"
    );

    let output = modelspec(&["train", DROPOUT, "--backend", "vulkan", "--stats"]);
    assert_exit(&output, 0);
    let stderr = text(&output.stderr);
    for name in [
        "steps",
        "host waits",
        "submits",
        "dispatches",
        "dispatches per step",
        "barriers per step",
        "bytes uploaded",
        "bytes downloaded",
        "device allocations",
        "peak device bytes",
        "numerics class",
        "us per step",
    ] {
        assert!(
            stderr
                .lines()
                .any(|line| line.starts_with(&format!("{name}: "))),
            "{name}: {stderr}"
        );
    }
    assert!(stderr.contains("steps: 30\n"), "{stderr}");
    assert!(stderr.contains("host waits: 31\n"), "{stderr}");
}

/// A `step <k> loss <v> …` or `eval <k> loss <v> …` line split at its loss: the text before it,
/// the value, and the text after it.
fn loss_line(line: &str) -> Option<(&str, f32, &str)> {
    let (head, tail) = line.split_once(" loss ")?;
    if !(head.starts_with("step ") || head.starts_with("eval ")) {
        return None;
    }
    let (value, rest) = tail.split_once(' ').unwrap_or((tail, ""));
    Some((head, value.parse().ok()?, rest))
}

/// The stdout of a Vulkan run against the CPU's (coordinator decision D88): the same lines, each
/// finite loss within the run tolerance `GPU_RUN_ATOL + GPU_RUN_RTOL·|c|` of DESIGN.md § 9, every
/// non-finite one, every other line and the rest of a loss line byte for byte.
fn assert_stdout_as_the_cpu(gpu: &str, cpu: &str, label: &str) {
    let g: Vec<&str> = gpu.lines().collect();
    let c: Vec<&str> = cpu.lines().collect();
    assert_eq!(g.len(), c.len(), "{label}: lines\n{gpu}\n---\n{cpu}");
    let mut losses = 0;
    for (g_line, c_line) in g.iter().zip(&c) {
        match (loss_line(g_line), loss_line(c_line)) {
            (Some((g_head, g_loss, g_rest)), Some((c_head, c_loss, c_rest))) => {
                assert_eq!((g_head, g_rest), (c_head, c_rest), "{label}");
                if !c_loss.is_finite() {
                    assert_eq!(g_line, c_line, "{label}: a non-finite loss");
                    continue;
                }
                assert!(
                    within(g_loss, c_loss, c_loss.abs(), GPU_RUN_ATOL, GPU_RUN_RTOL),
                    "{label}: `{g_line}` against the CPU's `{c_line}`"
                );
                losses += 1;
            }
            _ => assert_eq!(g_line, c_line, "{label}"),
        }
    }
    assert!(losses > 0, "{label}: no loss line in\n{cpu}");
}

/// Hypothesis 5: the SGD dropout fixture on Vulkan prints the CPU's stdout, a resume on Vulkan
/// from a CPU checkpoint at step 10 prints the CPU resume's stdout, and a diverging learning rate
/// gives the CPU's status, stdout and stderr. The unit claims the first in its report and has no
/// test for it. Since story:bert-tiny-vs-pytorch-cuda (coordinator decision D88) every finite
/// loss is compared within the run tolerance of DESIGN.md § 9 and every other stdout line byte
/// for byte: release 0.21.0 sums the CPU's whole-tensor reductions in `f64` (D84), which moved
/// the CPU's step-0 loss by one ULP (`2.426727` against the Vulkan run's `2.4267268` on the RTX
/// 3090).
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_gpu_step_device_dropout_sgd_matches_the_cpu_run_resume_and_divergence() {
    let case = scratch("dropout");
    let cpu = modelspec(&["train", DROPOUT, "--backend", "cpu"]);
    assert_exit(&cpu, 0);
    let gpu = modelspec(&["train", DROPOUT, "--backend", "vulkan"]);
    assert_exit(&gpu, 0);
    assert_stdout_as_the_cpu(
        &text(&gpu.stdout),
        &text(&cpu.stdout),
        "a 30-step dropout run",
    );

    let checkpoint = case.join("cpu10.msckpt");
    assert_exit(
        &modelspec(&[
            "train",
            DROPOUT,
            "--backend",
            "cpu",
            "--steps",
            "10",
            "--checkpoint",
            path(&checkpoint),
        ]),
        0,
    );
    let resumed: Vec<Output> = ["cpu", "vulkan"]
        .iter()
        .map(|backend| {
            modelspec(&[
                "train",
                DROPOUT,
                "--backend",
                backend,
                "--resume",
                path(&checkpoint),
                "--steps",
                "20",
            ])
        })
        .collect();
    assert_exit(&resumed[0], 0);
    assert_exit(&resumed[1], 0);
    assert_stdout_as_the_cpu(
        &text(&resumed[1].stdout),
        &text(&resumed[0].stdout),
        "a resumed dropout run",
    );

    let diverging = variant(
        &case,
        "lr100.mspec",
        &fixture(DROPOUT),
        &[("sgd(lr = 0.05)", "sgd(lr = 100.0)")],
    );
    let cpu = modelspec(&["train", &diverging, "--backend", "cpu"]);
    let gpu = modelspec(&["train", &diverging, "--backend", "vulkan"]);
    assert_exit(&cpu, 1);
    assert_eq!(gpu.status.code(), cpu.status.code());
    assert_stdout_as_the_cpu(
        &text(&gpu.stdout),
        &text(&cpu.stdout),
        "a diverging dropout run",
    );
    assert_eq!(text(&gpu.stderr), text(&cpu.stderr));
}

/// Hypothesis 1: the twin's run tolerance at 10, 20 and 40 steps (warm-up a fifth of the run):
/// every printed step loss and every final parameter element within `1e-4 + 1e-3 |c|` of the
/// CPU run. Prints the drift table: the worst element's ratio to its bound, its parameter, its
/// CPU value and its difference.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_gpu_step_device_twin_drift_over_10_20_and_40_steps() {
    let case = scratch("drift");
    let data = packed(&case);
    let mut outside = Vec::new();
    eprintln!(
        "steps | max |g-c| | bound at that element | ratio | parameter [i] | cpu | gpu | max step-loss diff"
    );
    for steps in [10u64, 20, 40] {
        let file = twin(&case, steps);
        let cpu = modelspec(&["train", &file, "--backend", "cpu", "--data", path(&data)]);
        assert_exit(&cpu, 0);
        let gpu = modelspec(&["train", &file, "--backend", "vulkan", "--data", path(&data)]);
        assert_exit(&gpu, 0);
        let (cpu_out, gpu_out) = (text(&cpu.stdout), text(&gpu.stdout));
        let mut loss_diff = 0.0f64;
        for ((k, c), (gk, g)) in step_lines(&cpu_out).iter().zip(step_lines(&gpu_out)) {
            assert_eq!(*k, gk);
            loss_diff = loss_diff.max((f64::from(g) - f64::from(*c)).abs());
            if !within(g, *c, c.abs(), GPU_RUN_ATOL, GPU_RUN_RTOL) {
                outside.push(format!("{steps} steps, step {k}: gpu {g} cpu {c}"));
            }
        }
        let ((ratio, diff, name, i, c, g), mut out) = worst_parameter(&cpu_out, &gpu_out);
        eprintln!(
            "{steps} | {diff:.3e} | {:.3e} | {ratio:.3} | @{name} [{i}] | {c} | {g} | {loss_diff:.3e}",
            bound(c)
        );
        outside.append(&mut out);
    }
    assert!(outside.is_empty(), "{outside:#?}");
}

/// Finding f2 (hypothesis 1): DESIGN.md § 9 states the run tolerance for "a training run on the
/// device" against the CPU run, every final parameter element within `1e-4 + 1e-3 |c|`, with no
/// bound on its length. The twin at 80, 160 and its own 800 steps (warm-up a tenth, as the
/// fixture's) breaks it. The worst elements are the key biases `l0_bk` and `l1_bk`: softmax is
/// invariant to a shift of a query's scores, so their true gradient is 0 and what each backend
/// computes is rounding noise, which AdamW's `m / (sqrt(v) + eps)` turns into an update of up to
/// `lr_t * noise / eps` per step whose sign is the backend's own. Prints, per run length, the
/// worst element and the worst element of every other parameter. Red on the unit.
///
/// Fix pass 1 (coordinator decision D74) amends the rule rather than widening it: a parameter
/// whose exact gradient is zero by the model's structure, here an attention key bias, which the
/// test finds in the twin's Tensor IR ([`common::attention_key_biases`]), passes when
/// `|g - c| <= GPU_RUN_ATOL + steps · lr · 1e-3` ([`common::zero_gradient_bound`]); every other
/// parameter keeps the element rule, and the worst ratio among them stays at or under 1.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_gpu_step_f2_device_twin_run_tolerance_at_80_160_and_800_steps() {
    let case = scratch("drift_long");
    let data = packed(&case);
    let mut outside = Vec::new();
    eprintln!(
        "steps | max |g-c| | bound | ratio | parameter [i] | cpu | gpu | elements outside | worst ratio outside the key biases | key-bias max |g-c| | key-bias bound | key-bias ratio"
    );
    for steps in [80u64, 160, 800] {
        let file = variant(
            &case,
            &format!("twin{steps}.mspec"),
            &fixture(TWIN),
            &[
                ("steps = 800;", &format!("steps = {steps};")),
                (
                    "warmup_steps = 80)",
                    &format!("warmup_steps = {})", steps / 10),
                ),
            ],
        );
        let cpu = modelspec(&["train", &file, "--backend", "cpu", "--data", path(&data)]);
        assert_exit(&cpu, 0);
        let gpu = modelspec(&["train", &file, "--backend", "vulkan", "--data", path(&data)]);
        assert_exit(&gpu, 0);
        let (cpu_out, gpu_out) = (text(&cpu.stdout), text(&gpu.stdout));
        for ((k, c), (_, g)) in step_lines(&cpu_out).iter().zip(step_lines(&gpu_out)) {
            if !within(g, *c, c.abs(), GPU_RUN_ATOL, GPU_RUN_RTOL) {
                outside.push(format!("{steps} steps, step {k}: gpu {g} cpu {c}"));
            }
        }
        let biases = key_biases(&file);
        assert!(!biases.is_empty(), "the twin has attention key biases");
        let biased = zero_gradient_bound(steps, learning_rate(&file));
        let (c_params, g_params) = (tensors(&cpu_out, "param"), tensors(&gpu_out, "param"));
        let (mut count, mut rest, mut bias_diff) = (0usize, 0.0f64, 0.0f64);
        for (name, c) in &c_params {
            for (i, (&g, &c)) in g_params[name].iter().zip(c).enumerate() {
                let diff = (f64::from(g) - f64::from(c)).abs();
                let ratio = diff / bound(c);
                if ratio > 1.0 {
                    count += 1;
                }
                let passes = if biases.contains(name) {
                    bias_diff = bias_diff.max(diff);
                    diff <= biased
                } else {
                    rest = rest.max(ratio);
                    ratio <= 1.0
                };
                if !passes && outside.len() < 16 {
                    outside.push(format!(
                        "{steps} steps, param @{name} [{i}]: gpu {g} cpu {c}"
                    ));
                }
            }
        }
        let ((ratio, diff, name, i, c, g), _) = worst_parameter(&cpu_out, &gpu_out);
        eprintln!(
            "{steps} | {diff:.3e} | {:.3e} | {ratio:.3} | @{name} [{i}] | {c} | {g} | {count} | {rest:.3} | {bias_diff:.3e} | {biased:.3e} | {:.3}",
            bound(c),
            bias_diff / biased
        );
        assert!(
            rest <= 1.0,
            "{steps} steps: worst ratio outside the key biases {rest}"
        );
    }
    assert!(outside.is_empty(), "{outside:#?}");
}

/// The attention key biases of `file`, from its Tensor IR (`ir --stage tensor`), by the
/// structural rule of [`common::attention_key_biases`].
fn key_biases(file: &str) -> std::collections::BTreeSet<String> {
    let output = modelspec(&["ir", file, "--stage", "tensor"]);
    assert_exit(&output, 0);
    attention_key_biases(&text(&output.stdout))
}

/// The `lr` of the `adamw(...)` optimizer of `file`.
fn learning_rate(file: &str) -> f64 {
    let source = std::fs::read_to_string(file).expect("the file reads");
    let (_, rest) = source
        .split_once("adamw(lr = ")
        .expect("an adamw optimizer");
    let (lr, _) = rest
        .split_once(',')
        .expect("lr is followed by another argument");
    lr.trim().parse().expect("lr is a float literal")
}

/// Finding f2 (hypothesis 1), the acceptance run itself: scenario 4's ten steps of the twin
/// (`warmup_steps = 2`) with the seeds 1 to 8 instead of 42. The unit's 10-step comparison
/// passes with its worst element, a key bias, at 0.54 of the bound. On the unit's commit and the
/// RTX 3090 all eight pass, the worst at 0.85 of the bound (seed 1), each worst element a key
/// bias: the acceptance run holds, with its margin set by rounding noise. Prints the worst
/// element per seed.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_gpu_step_device_twin_10_steps_within_the_run_tolerance_for_eight_seeds() {
    let case = scratch("seeds");
    let data = packed(&case);
    let mut outside = Vec::new();
    eprintln!("seed | max |g-c| | bound | ratio | parameter [i]");
    for seed in 1..=8u64 {
        let file = variant(
            &case,
            &format!("twin10_seed{seed}.mspec"),
            &fixture(TWIN),
            &[
                ("steps = 800;", "steps = 10;"),
                ("warmup_steps = 80)", "warmup_steps = 2)"),
                ("seed = 42;", &format!("seed = {seed};")),
            ],
        );
        let cpu = modelspec(&["train", &file, "--backend", "cpu", "--data", path(&data)]);
        assert_exit(&cpu, 0);
        let gpu = modelspec(&["train", &file, "--backend", "vulkan", "--data", path(&data)]);
        assert_exit(&gpu, 0);
        let ((ratio, diff, name, i, c, _), mut out) =
            worst_parameter(&text(&cpu.stdout), &text(&gpu.stdout));
        eprintln!(
            "{seed} | {diff:.3e} | {:.3e} | {ratio:.3} | @{name} [{i}]",
            bound(c)
        );
        out.truncate(4);
        outside.extend(out.into_iter().map(|line| format!("seed {seed}, {line}")));
    }
    assert!(outside.is_empty(), "{outside:#?}");
}

/// Hypothesis 7: three GPU runs of one seed print identical stdout; `--fast-math` on the
/// stepwise run exits 0, two such runs print identical stdout (DESIGN.md ADR-008: "two runs of it
/// on one device and driver give the same bits"), and its losses and final parameters are within
/// the run tolerance of the CPU run.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_gpu_step_device_three_runs_identical_and_fast_math_within_the_run_tolerance() {
    let case = scratch("determinism");
    let data = packed(&case);
    let file = twin(&case, 10);
    let run = |extra: &[&str]| {
        let mut args = vec![
            "train",
            file.as_str(),
            "--backend",
            "vulkan",
            "--data",
            path(&data),
        ];
        args.extend_from_slice(extra);
        let output = modelspec(&args);
        assert_exit(&output, 0);
        text(&output.stdout)
    };
    let first = run(&[]);
    assert_eq!(run(&[]), first, "second GPU run");
    assert_eq!(run(&[]), first, "third GPU run");
    let fast = run(&["--fast-math"]);
    assert_eq!(run(&["--fast-math"]), fast, "second fast-math run");
    let cpu = modelspec(&["train", &file, "--backend", "cpu", "--data", path(&data)]);
    assert_exit(&cpu, 0);
    let cpu = text(&cpu.stdout);
    let mut outside = Vec::new();
    for ((k, c), (_, g)) in step_lines(&cpu).iter().zip(step_lines(&fast)) {
        if !within(g, *c, c.abs(), GPU_RUN_ATOL, GPU_RUN_RTOL) {
            outside.push(format!("fast-math step {k}: gpu {g} cpu {c}"));
        }
    }
    let ((ratio, diff, name, i, _, _), mut out) = worst_parameter(&cpu, &fast);
    eprintln!(
        "fast-math: worst parameter @{name} [{i}] |g-c| {diff:.3e}, ratio to bound {ratio:.3}"
    );
    eprintln!(
        "fast-math stdout equals the default run's: {}",
        fast == first
    );
    outside.append(&mut out);
    assert!(outside.is_empty(), "{outside:#?}");
}

/// Hypotheses 1 and 6: with `--eval-every 3` over 10 steps both backends evaluate steps 0, 3,
/// 6, 9 and 10, each GPU loss within the run tolerance of the CPU's.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_gpu_step_device_eval_every_3_evaluates_the_cpu_steps() {
    let case = scratch("eval3");
    let data = packed(&case);
    let file = twin(&case, 10);
    let run = |backend: &str| {
        let output = modelspec(&[
            "train",
            &file,
            "--backend",
            backend,
            "--data",
            path(&data),
            "--eval-every",
            "3",
        ]);
        assert_exit(&output, 0);
        eval_lines(&text(&output.stdout))
    };
    let (cpu, gpu) = (run("cpu"), run("vulkan"));
    let steps: Vec<u64> = gpu.iter().map(|(k, _, _)| *k).collect();
    assert_eq!(steps, [0, 3, 6, 9, 10]);
    assert_eq!(cpu.iter().map(|(k, _, _)| *k).collect::<Vec<_>>(), steps);
    for ((k, c, _), (_, g, _)) in cpu.iter().zip(&gpu) {
        assert!(
            within(*g, *c, c.abs(), GPU_RUN_ATOL, GPU_RUN_RTOL),
            "eval {k}: gpu {g} cpu {c}"
        );
    }
}

/// Hypothesis 6: a twin run whose learning rate diverges stops with the CPU's status and its
/// E-TRAIN-001 line on stderr, and the GPU run's record ends with `RunFailed` as the CPU's does.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_gpu_step_device_diverging_twin_fails_as_the_cpu_run() {
    let case = scratch("diverge");
    let data = packed(&case);
    let file = variant(
        &case,
        "twin_diverge.mspec",
        &fixture(TWIN),
        &[
            ("steps = 800;", "steps = 10;"),
            ("warmup_steps = 80)", "warmup_steps = 1)"),
            ("lr = 3e-3", "lr = 1e30"),
        ],
    );
    let mut first_lines = Vec::new();
    let mut last_kinds = Vec::new();
    for backend in ["cpu", "vulkan"] {
        let record = case.join(format!("{backend}.run.jsonl"));
        let output = modelspec(&[
            "train",
            &file,
            "--backend",
            backend,
            "--data",
            path(&data),
            "--record",
            path(&record),
        ]);
        assert_exit(&output, 1);
        first_lines.push(
            text(&output.stderr)
                .lines()
                .next()
                .unwrap_or_default()
                .to_owned(),
        );
        let events: Vec<Value> = std::fs::read_to_string(&record)
            .expect("the record")
            .lines()
            .map(|line| serde_json::from_str(line).expect("a JSON event"))
            .collect();
        last_kinds.push(
            events
                .last()
                .map(|event| {
                    event["body"]["kind"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned()
                })
                .unwrap_or_default(),
        );
    }
    assert_eq!(first_lines[1], first_lines[0]);
    assert_eq!(last_kinds, ["RunFailed", "RunFailed"]);
}

/// Hypothesis 5: dense `--data` (newly accepted on Vulkan) trains on the device: a 2-sample
/// batch with an `order.u32` over 3 steps prints the CPU's step lines and final parameters
/// within the run tolerance.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_gpu_step_device_dense_data_trains_as_the_cpu_run() {
    let case = scratch("dense");
    let source = "\
model Lin {
    param w: Tensor<f32, [2, 2]> = xavier_uniform();

    fn forward(x: Tensor<f32, [2, 2]>)
        -> Tensor<f32, [2, 2]> {
        return matmul(x, w);
    }
}

train Fit {
    model = Lin;
    input x: Tensor<f32, [2, 2]>;
    target y: Tensor<f32, [2, 2]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 3;
    seed = 42;
    device = cpu;
}
";
    let file = case.join("model.mspec");
    std::fs::write(&file, source).unwrap();
    let data = case.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    std::fs::write(
        data.join("x.f32"),
        f32s(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0]),
    )
    .unwrap();
    std::fs::write(
        data.join("y.f32"),
        f32s(&[1.0, 0.0, 1.0, 0.0, 0.0, 3.0, 1.0, 0.0]),
    )
    .unwrap();
    let order: Vec<u8> = [2u32, 3, 0, 1, 3, 2]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    std::fs::write(data.join("order.u32"), order).unwrap();
    let run = |backend: &str| {
        let output = modelspec(&[
            "train",
            path(&file),
            "--backend",
            backend,
            "--data",
            path(&data),
        ]);
        assert_exit(&output, 0);
        text(&output.stdout)
    };
    let (cpu, gpu) = (run("cpu"), run("vulkan"));
    let (c_steps, g_steps) = (step_lines(&cpu), step_lines(&gpu));
    assert_eq!(c_steps.len(), g_steps.len(), "{cpu}\n{gpu}");
    for ((k, c), (_, g)) in c_steps.iter().zip(&g_steps) {
        assert!(
            within(*g, *c, c.abs(), GPU_RUN_ATOL, GPU_RUN_RTOL),
            "step {k}: gpu {g} cpu {c}"
        );
    }
    let (_, outside) = worst_parameter(&cpu, &gpu);
    assert!(outside.is_empty(), "{outside:#?}");
}
