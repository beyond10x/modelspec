//! `modelspec ir --stage training` and `modelspec train` on files trained with `muon(...)`
//! (DESIGN.md § 2.1, § 6, § 8; story:muon-optimizer, scenarios 3, 5 and 6).
//!
//! A parameter takes the Muon update when its rank is 2 and no `embedding` reads it; every other
//! parameter with a gradient takes the AdamW update with `adam_lr` (coordinator decision D67 (d),
//! 2026-10-25, `.engineering/waves/bert-1.md`). `mlm_dense.mspec` is `mlm_bag.mspec` with one
//! hidden dense weight `w1` between the embedding lookup and the tied decoder (D67 (b)); scenario
//! 5 trains it 300 steps on the packed probe corpus of story:bert-heldout-evaluation under `muon`
//! and under `adamw` at the same rates (D67 (c)).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const CORPUS: &str = "crates/modelspec-runtime/tests/fixtures/corpus-tokenizer";
const MLM_BAG: &str = "crates/modelspec-cli/tests/fixtures/mlm_bag.mspec";
const MLM_DENSE: &str = "crates/modelspec-cli/tests/fixtures/mlm_dense.mspec";
const LINEAR: &str = "crates/modelspec-cli/tests/fixtures/linear_regression_muon.mspec";

/// The optimizer line of `mlm_dense.mspec`, and the AdamW run it is compared with: `lr` is the
/// Muon `lr` and `adam_lr`, the weight decay, betas, `eps` and warm-up are the same (D67 (c)).
const MUON_LINE: &str = "    optimizer = muon(lr = 0.01, momentum = 0.95, weight_decay = 0.01, ns_steps = 5, warmup_steps = 30, adam_lr = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8);";
const ADAMW_LINE: &str = "    optimizer = adamw(lr = 0.01, weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, warmup_steps = 30);";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("train_muon_{name}"));
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

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn assert_exit(output: &Output, code: i32) {
    assert_eq!(
        output.status.code(),
        Some(code),
        "stdout: {}\nstderr: {}",
        stdout(output),
        stderr(output)
    );
}

fn path(p: &Path) -> &str {
    p.to_str().expect("a UTF-8 path")
}

/// `source_file` with `edits` applied, written to `<case>/<name>.mspec`.
fn variant(case: &Path, source_file: &str, name: &str, edits: &[(&str, &str)]) -> String {
    let mut source = std::fs::read_to_string(repo_root().join(source_file)).expect("the fixture");
    for (from, to) in edits {
        assert!(source.contains(from), "the fixture holds {from:?}");
        source = source.replace(from, to);
    }
    let file = case.join(format!("{name}.mspec"));
    std::fs::write(&file, source).expect("the variant is written");
    path(&file).to_owned()
}

/// The packed probe corpus of story:bert-heldout-evaluation: the original-text fixture, a
/// 300-token vocabulary and sequences of 16.
fn packed(case: &Path) -> PathBuf {
    let dataset = case.join("dataset");
    assert_exit(
        &modelspec(&["data", "prepare", CORPUS, "--out", path(&dataset)]),
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

/// The `output %<n> update @<path> <kind>` line of `path`, `kind` its update.
fn update_kind(ir: &str, path: &str) -> String {
    let suffix = format!(" update @{path} ");
    ir.lines()
        .filter(|l| l.starts_with("output %"))
        .find_map(|l| l.split_once(&suffix).map(|(_, kind)| kind.to_owned()))
        .unwrap_or_else(|| panic!("no update line of @{path}:\n{ir}"))
}

fn ir_training(file: &str) -> String {
    let output = modelspec(&["ir", file, "--stage", "training"]);
    assert_exit(&output, 0);
    stdout(&output)
}

/// Scenario 3. On `mlm_dense.mspec`, `ir --stage training` lists the token table under AdamW,
/// with its moments `m` and `v`, and the dense weight under Muon, with its momentum buffer `b`
/// read as `optimizer.b@MlmDense.w1`; the step holds the Newton-Schulz matmuls of the weight.
#[test]
fn ir_lists_the_table_under_adamw_and_the_dense_weight_under_muon() {
    let ir = ir_training(MLM_DENSE);
    assert_eq!(update_kind(&ir, "MlmDense.table"), "adamw", "{ir}");
    assert_eq!(update_kind(&ir, "MlmDense.w1"), "muon", "{ir}");
    for (moment, path) in [
        ("m", "MlmDense.table"),
        ("v", "MlmDense.table"),
        ("b", "MlmDense.w1"),
    ] {
        assert!(
            ir.lines()
                .any(|l| l.starts_with("output %") && l.ends_with(&format!(" {moment} @{path}"))),
            "no output of `{moment}` of @{path}:\n{ir}"
        );
        assert!(
            ir.lines().any(|l| l.starts_with("optimizer %")
                && l.contains(&format!("= Input \"optimizer.{moment}@{path}\" {{"))),
            "no input of `{moment}` of @{path}:\n{ir}"
        );
    }
    for absent in ["optimizer.b@MlmDense.table", "optimizer.m@MlmDense.w1"] {
        assert!(!ir.contains(absent), "{absent}:\n{ir}");
    }
    assert!(
        ir.lines().any(|l| l
            == "optimizer muon lr = 0.01 momentum = 0.95 weight_decay = 0.01 ns_steps = 5 \
                warmup_steps = 30 adam_lr = 0.01 beta1 = 0.9 beta2 = 0.999 eps = 1e-8"),
        "{ir}"
    );
    let matmuls = ir
        .lines()
        .filter(|l| l.starts_with("optimizer %") && l.contains("= MatMul %"))
        .count();
    assert_eq!(
        matmuls,
        3 * 5,
        "three matmuls per Newton-Schulz step:\n{ir}"
    );
}

/// Scenario 3. `mlm_bag.mspec` under `muon` lists its table, tied to the decoder through
/// `transpose(table)`, under AdamW, and no parameter under Muon.
#[test]
fn ir_lists_the_tied_table_under_adamw() {
    let case = scratch("tied");
    let file = variant(
        &case,
        MLM_BAG,
        "mlm_bag_muon",
        &[("    optimizer = sgd(lr = 1.0);", MUON_LINE)],
    );
    let ir = ir_training(&file);
    assert_eq!(update_kind(&ir, "MlmBag.table"), "adamw", "{ir}");
    assert!(!ir.contains(" muon\n"), "{ir}");
    assert!(!ir.contains("optimizer.b@"), "{ir}");
}

/// `(step, held-out loss)` of every `eval <k> loss <v> accuracy <a>` line.
fn heldout(out: &str) -> Vec<(u64, f32)> {
    out.lines()
        .filter_map(|l| l.strip_prefix("eval "))
        .map(|rest| {
            let mut words = rest.split(' ');
            let k = words.next().expect("a step").parse().expect("a step");
            assert_eq!(words.next(), Some("loss"), "{rest}");
            let loss = words.next().expect("a loss").parse().expect("a loss");
            (k, loss)
        })
        .collect()
}

/// Scenario 5. `mlm_dense.mspec` trains 300 steps on the packed probe corpus with
/// `--eval-every 50` under `muon` and under `adamw` at the Moonshot-matched rates; both runs
/// evaluate at 0, 50, ..., 300 and stay finite, and the held-out loss at step 300 under Muon is
/// at most the AdamW figure. Both curves are printed for the report (`--nocapture`).
#[test]
fn muon_heldout_loss_at_step_300_is_at_most_adamw() {
    let case = scratch("heldout");
    let packed = packed(&case);
    let adamw = variant(
        &case,
        MLM_DENSE,
        "mlm_dense_adamw",
        &[(MUON_LINE, ADAMW_LINE)],
    );
    let mut curves = Vec::new();
    for file in [MLM_DENSE, adamw.as_str()] {
        let output = modelspec(&[
            "train",
            file,
            "--backend",
            "cpu",
            "--data",
            path(&packed),
            "--eval-every",
            "50",
        ]);
        assert_exit(&output, 0);
        let curve = heldout(&stdout(&output));
        let steps: Vec<u64> = curve.iter().map(|(k, _)| *k).collect();
        assert_eq!(steps, [0, 50, 100, 150, 200, 250, 300], "{file}");
        assert!(curve.iter().all(|(_, loss)| loss.is_finite()), "{curve:?}");
        curves.push(curve);
    }
    let (muon, adamw) = (&curves[0], &curves[1]);
    for ((k, m), (_, a)) in muon.iter().zip(adamw) {
        eprintln!("heldout step {k} muon {m} adamw {a}");
    }
    assert_eq!(muon[0], adamw[0], "one seed, one initial evaluation");
    let (m, a) = (muon[6].1, adamw[6].1);
    assert!(
        m <= a,
        "held-out loss at step 300: muon {m} against adamw {a}"
    );
}

/// Scenario 6. `train --backend vulkan` on a Muon file is E-TARGET-011 at the `train` block
/// before anything runs, as for `adamw`, with nothing on stdout; `ir --stage kernel` too.
#[test]
fn muon_vulkan_paths_are_e_target_011_before_anything_runs() {
    for args in [
        vec!["train", LINEAR, "--backend", "vulkan"],
        vec!["ir", LINEAR, "--stage", "kernel"],
    ] {
        let output = modelspec(&args);
        let err = stderr(&output);
        assert_eq!(output.status.code(), Some(1), "{args:?}: {err}");
        assert!(output.stdout.is_empty(), "{args:?}: {}", stdout(&output));
        assert_eq!(
            err.lines().take(2).collect::<Vec<_>>(),
            [
                "E-TARGET-011: optimizer `muon` has no Vulkan kernel yet",
                &format!("  {LINEAR}:10:1"),
            ],
            "{args:?}"
        );
    }
}

/// Scenario 6. `--checkpoint` and `--resume` on a Muon file exit 2 on one line naming the flag,
/// on either backend, before anything runs or is written.
#[test]
fn muon_checkpoint_and_resume_are_usage_errors() {
    let case = scratch("checkpoint");
    let checkpoint = case.join("run.msckpt");
    for backend in ["cpu", "vulkan"] {
        for flag in ["--checkpoint", "--resume"] {
            let output = modelspec(&[
                "train",
                LINEAR,
                "--backend",
                backend,
                flag,
                path(&checkpoint),
            ]);
            let err = stderr(&output);
            assert_eq!(output.status.code(), Some(2), "{backend} {flag}: {err}");
            assert_eq!(err.lines().count(), 1, "{backend} {flag}: {err}");
            assert!(err.contains(&format!("'{flag}'")), "{err}");
            assert!(err.contains("`muon`"), "{err}");
            assert!(output.stdout.is_empty(), "{}", stdout(&output));
            assert!(!checkpoint.exists(), "no checkpoint is written");
        }
    }
}

/// A Muon run on the CPU lowers the loss of the linear fixture, prints its parameters only and
/// repeats byte for byte under one seed.
#[test]
fn muon_trains_and_repeats() {
    let first = modelspec(&["train", LINEAR, "--backend", "cpu"]);
    assert_exit(&first, 0);
    let out = stdout(&first);
    let losses: Vec<f32> = out
        .lines()
        .filter_map(|l| l.strip_prefix("step "))
        .map(|rest| rest.split_once(" loss ").unwrap().1.parse().unwrap())
        .collect();
    assert!(losses.last() < losses.first(), "{out}");
    let params: Vec<&str> = out.lines().filter(|l| l.starts_with("param @")).collect();
    assert_eq!(params.len(), 1, "{out}");
    let again = modelspec(&["train", LINEAR, "--backend", "cpu"]);
    assert_eq!(stdout(&again), out, "two runs with one seed");
}
