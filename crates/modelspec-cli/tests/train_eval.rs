//! `modelspec train --backend cpu --data <packed-dir> --eval-every <n>`
//! (story:bert-heldout-evaluation, coordinator decision D55): at every multiple of `n` among the
//! run's steps and at the run's last step the forward program without dropout runs over every
//! full batch of the `HeldOut` split, each drawn with the run seed, and the run prints
//! `eval <k> loss <v> accuracy <a>` after the `step <k>` line and streams the two values as the
//! `MetricSample`s `heldout_mlm_loss` and `heldout_mlm_accuracy` of the `StepCompleted` of step
//! `k`.
//!
//! The packed directory is the one `train_data.rs` makes: `modelspec data prepare`, `data
//! tokenizer` and `data pack` on the original-text fixture
//! `crates/modelspec-runtime/tests/fixtures/corpus-tokenizer/`, a 300-token vocabulary and
//! sequences of 16, so `mlm_bag.mspec` declares its 64 rows as `[B·S]` with `B = 4`.
//!
//! The expected values are computed here, independently of `modelspec_runtime::train::eval`:
//! the test draws each `HeldOut` batch with `mlm_batch`, runs the forward program of the
//! compiled file on the CPU interpreter with the parameters the run printed, and reduces the
//! `prediction` logits itself, checking its own per-batch reduction against the program's
//! `cross_entropy` output on the way.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::tensor::{Dim, ParamId};
use modelspec_runtime::cpu::{HostTensor, evaluate_all_with_ids};
use modelspec_runtime::dataset::SplitRole;
use modelspec_runtime::dataset::mlm::{self, MaskingPolicy, TrainingDataBinding, mlm_batch};
use serde_json::Value;

const CORPUS: &str = "crates/modelspec-runtime/tests/fixtures/corpus-tokenizer";
const MLM_BAG: &str = "crates/modelspec-cli/tests/fixtures/mlm_bag.mspec";
const MLM_BAG_DROPOUT: &str = "crates/modelspec-cli/tests/fixtures/mlm_bag_dropout.mspec";
const RUN_EVENT_SCHEMA: &str =
    "crates/modelspec-observatory/schema/entities/modelspec.observatory.RunEvent.schema.json";
/// The fixture's `seed`, batch size and sequence length.
const SEED: u64 = 7;
const B: usize = 4;
const S: usize = 16;
const WAIT: Duration = Duration::from_secs(300);

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh, empty directory under this crate's build directory, named for the test.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("train_eval_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// Runs `modelspec <args>` from the repository root.
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

/// The fixture corpus prepared, a 300-token tokenizer trained on it and the two packed into
/// sequences of 16, all under `case`; returns the packed directory.
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

/// `modelspec train <file> --backend cpu --data <data> <extra...>`.
fn train(file: &str, data: &Path, extra: &[&str]) -> Output {
    let mut args = vec!["train", file, "--backend", "cpu", "--data", path(data)];
    args.extend_from_slice(extra);
    modelspec(&args)
}

/// The fixture's source with `edits` applied, each `(from, to)` replacing every occurrence of a
/// text that must be there, written to `<case>/<name>.mspec`.
fn variant(case: &Path, name: &str, edits: &[(&str, &str)]) -> String {
    let mut source = std::fs::read_to_string(repo_root().join(MLM_BAG)).expect("the fixture");
    for (from, to) in edits {
        assert!(source.contains(from), "the fixture holds {from:?}");
        source = source.replace(from, to);
    }
    let file = case.join(format!("{name}.mspec"));
    std::fs::write(&file, source).expect("the variant is written");
    path(&file).to_owned()
}

/// One `eval <k> loss <v> accuracy <a>` line: `k` and the two values as printed.
#[derive(Clone, Debug, PartialEq)]
struct EvalLine {
    step: u64,
    loss: String,
    accuracy: String,
}

impl EvalLine {
    fn loss(&self) -> f32 {
        self.loss.parse().expect("an f32 loss")
    }

    fn accuracy(&self) -> f32 {
        self.accuracy.parse().expect("an f32 accuracy")
    }
}

/// Every `eval` line of `out`, in order; panics on an `eval` line of another shape.
fn evals(out: &str) -> Vec<EvalLine> {
    out.lines()
        .filter(|line| line.starts_with("eval "))
        .map(|line| {
            let words: Vec<&str> = line.split(' ').collect();
            assert!(
                words.len() == 6 && words[2] == "loss" && words[4] == "accuracy",
                "`eval <k> loss <v> accuracy <a>`: {line:?}"
            );
            EvalLine {
                step: words[1].parse().expect("a step"),
                loss: words[3].to_owned(),
                accuracy: words[5].to_owned(),
            }
        })
        .collect()
}

/// The values of every `<label> @<path>: <type> = [...]` line of `out`, by path.
fn tensors(out: &str, label: &str) -> BTreeMap<String, Vec<f32>> {
    out.lines()
        .filter_map(|line| line.strip_prefix(&format!("{label} @")))
        .map(|rest| {
            let (path, rest) = rest.split_once(": ").expect("`@<path>: <type> = [...]`");
            let (_, list) = rest.split_once(" = [").expect("` = [`");
            let list = list.strip_suffix(']').expect("`]`");
            let values = list
                .split(", ")
                .map(|v| v.parse::<f32>().expect("an f32"))
                .collect();
            (path.to_owned(), values)
        })
        .collect()
}

/// `values` by parameter id of `compiled`'s forward program, each with its declared shape.
fn bind_params(
    compiled: &CompiledTraining,
    values: &BTreeMap<String, Vec<f32>>,
) -> BTreeMap<ParamId, HostTensor<f32>> {
    compiled
        .forward
        .program
        .params
        .iter()
        .map(|(&id, decl)| {
            let shape = decl
                .ty
                .shape
                .iter()
                .map(|d| match *d {
                    Dim::Static(n) => n,
                    Dim::Symbol(_) => panic!("a static shape"),
                })
                .collect();
            let data = values
                .get(&decl.path)
                .unwrap_or_else(|| panic!("printed values for @{}", decl.path))
                .clone();
            (id, HostTensor::new(shape, data).expect("the declared size"))
        })
        .collect()
}

/// The first index of the largest logit of `row`.
fn lowest_argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

/// `max + ln(sum exp(x - max)) - x[label]`, the row's negative log-likelihood, every reduction
/// in ascending order as DESIGN.md § 2.2 states for `cross_entropy`.
fn nll(row: &[f32], label: usize) -> f32 {
    let mut max = row[0];
    for &v in &row[1..] {
        if v > max {
            max = v;
        }
    }
    let mut total = (row[0] - max).exp();
    for &v in &row[1..] {
        total += (v - max).exp();
    }
    max + total.ln() - row[label]
}

/// What the test computes for one evaluation.
#[derive(Debug)]
struct Expected {
    loss: f32,
    accuracy: f32,
    /// The labels of the selected positions of every batch, in batch then row order.
    selected_labels: Vec<i32>,
    /// Every batch's logits, rows of `V`.
    logits: Vec<Vec<f32>>,
}

/// The held-out loss and accuracy of `file` with the printed parameter `values` over the
/// `HeldOut` batches `0..⌊n / B⌋` of `packed_dir`, each drawn with the fixture's seed: the
/// weighted sum of every row's `nll` over every batch, the first product rounded and each later
/// one fused into the sum, over the sum of every weight, and the share of rows of positive
/// weight whose first largest logit is the label. Each batch's own weighted mean is checked
/// against the forward program's `cross_entropy` output on the way.
fn expected(file: &str, packed_dir: &Path, values: &BTreeMap<String, Vec<f32>>) -> Expected {
    let source = std::fs::read_to_string(repo_root().join(file))
        .or_else(|_| std::fs::read_to_string(file))
        .expect("the file");
    let compiled = compile_training(file, &source).expect("the file compiles");
    let params = bind_params(&compiled, values);
    let binding: TrainingDataBinding = serde_json::from_str(
        &std::fs::read_to_string(packed_dir.join("binding.json")).expect("binding.json"),
    )
    .expect("a binding");
    let policy: MaskingPolicy = binding.masking_policy;
    let packed = mlm::load(packed_dir).expect("the packed directory loads");
    let batches = packed.sequence_count(SplitRole::HeldOut) / B;
    assert!(batches >= 1, "the fixture holds a HeldOut batch of {B}");
    let mut numerator: Option<f32> = None;
    let mut denominator: Option<f32> = None;
    let (mut correct, mut selected) = (0u32, 0u32);
    let mut selected_labels = Vec::new();
    let mut all_logits = Vec::new();
    for b in 0..batches {
        let batch = mlm_batch(&packed, SplitRole::HeldOut, &policy, SEED, b as u64, B)
            .expect("a HeldOut batch");
        let rows = (B * S) as u64;
        let floats = BTreeMap::from([(
            "weights".to_owned(),
            HostTensor::new(vec![rows], batch.weights.clone()).unwrap(),
        )]);
        let ids = BTreeMap::from([
            (
                "ids".to_owned(),
                HostTensor::new(vec![rows], batch.ids.clone()).unwrap(),
            ),
            (
                "labels".to_owned(),
                HostTensor::new(vec![rows], batch.labels.clone()).unwrap(),
            ),
        ]);
        let mut evaluation =
            evaluate_all_with_ids(&compiled.forward.program, &floats, &ids, &params)
                .expect("the forward program runs");
        let logits = evaluation
            .values
            .remove(&compiled.forward.prediction)
            .expect("the prediction value");
        assert_eq!(logits.shape(), [rows, 300]);
        let logits = logits.data().to_vec();
        let (mut batch_num, mut batch_den): (Option<f32>, Option<f32>) = (None, None);
        for (n, row) in logits.as_chunks::<300>().0.iter().enumerate() {
            let (w, label) = (batch.weights[n], batch.labels[n]);
            let term = nll(row, label as usize);
            numerator = Some(match numerator {
                None => w * term,
                Some(acc) => w.mul_add(term, acc),
            });
            denominator = Some(denominator.map_or(w, |acc| acc + w));
            batch_num = Some(batch_num.map_or(w * term, |acc| w.mul_add(term, acc)));
            batch_den = Some(batch_den.map_or(w, |acc| acc + w));
            if w > 0.0 {
                selected += 1;
                selected_labels.push(label);
                if lowest_argmax(row) == label as usize {
                    correct += 1;
                }
            }
        }
        let (batch_num, batch_den) = (batch_num.unwrap(), batch_den.unwrap());
        let mean = if batch_den == 0.0 {
            0.0
        } else {
            batch_num / batch_den
        };
        let program_loss = evaluation.outputs[0].data()[0];
        assert!(
            (mean - program_loss).abs() <= 1e-6,
            "batch {b}: the test's weighted mean {mean} is the program's cross_entropy {program_loss}"
        );
        all_logits.push(logits);
    }
    let (numerator, denominator) = (numerator.unwrap(), denominator.unwrap());
    assert!(selected > 0, "the HeldOut batches select a position");
    Expected {
        loss: if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        },
        accuracy: correct as f32 / selected as f32,
        selected_labels,
        logits: all_logits,
    }
}

fn close(printed: f32, computed: f32) -> bool {
    (printed - computed).abs() <= 1e-6
}

/// Scenario 1: `mlm_bag.mspec` (200 steps) with `--eval-every 100` prints `eval` lines for
/// steps 0, 100 and 200, each directly after its `step` line, the last after the final loss
/// line and before the `param` lines; the step-200 values equal those computed from the
/// printed final parameters and the step-0 values those computed from the printed initial
/// ones, within `1e-6`. Every other line is the line of the run without the flag.
#[test]
fn values_at_the_evaluated_steps_equal_the_test_computation() {
    let case = scratch("values");
    let packed = packed(&case);
    let output = train(MLM_BAG, &packed, &["--eval-every", "100"]);
    assert_exit(&output, 0);
    let out = stdout(&output);
    let lines: Vec<&str> = out.lines().collect();
    let evals = evals(&out);
    let steps: Vec<u64> = evals.iter().map(|e| e.step).collect();
    assert_eq!(steps, [0, 100, 200], "{out}");
    for e in &evals {
        let at = lines
            .iter()
            .position(|l| l.starts_with(&format!("eval {} ", e.step)))
            .unwrap();
        assert!(
            lines[at - 1].starts_with(&format!("step {} loss ", e.step)),
            "eval {} follows its step line:\n{out}",
            e.step
        );
    }
    let last_eval = lines.iter().rposition(|l| l.starts_with("eval ")).unwrap();
    assert!(
        lines[last_eval + 1].starts_with("param @MlmBag.table:"),
        "{out}"
    );

    let last = expected(MLM_BAG, &packed, &tensors(&out, "param"));
    assert!(
        close(evals[2].loss(), last.loss) && close(evals[2].accuracy(), last.accuracy),
        "eval 200 printed {:?}, computed loss {} accuracy {}",
        evals[2],
        last.loss,
        last.accuracy
    );
    let first = expected(MLM_BAG, &packed, &tensors(&out, "init"));
    assert!(
        close(evals[0].loss(), first.loss) && close(evals[0].accuracy(), first.accuracy),
        "eval 0 printed {:?}, computed loss {} accuracy {}",
        evals[0],
        first.loss,
        first.accuracy
    );

    let plain = train(MLM_BAG, &packed, &[]);
    assert_exit(&plain, 0);
    let without: String = out
        .lines()
        .filter(|l| !l.starts_with("eval "))
        .map(|l| format!("{l}\n"))
        .collect();
    assert_eq!(without, stdout(&plain), "evaluation changes no other line");

    let again = train(MLM_BAG, &packed, &["--eval-every", "100"]);
    assert_exit(&again, 0);
    assert_eq!(stdout(&again), out, "one seed, one stdout");
}

/// Scenario 1, D55: `--eval-every 30` evaluates at 0, 30, ..., 180 and at the last step 200,
/// and an evaluated step that is no multiple of 50 gets its `step` line too.
#[test]
fn values_at_every_multiple_and_the_last_step() {
    let case = scratch("values_every_30");
    let packed = packed(&case);
    let output = train(MLM_BAG, &packed, &["--eval-every", "30"]);
    assert_exit(&output, 0);
    let out = stdout(&output);
    let steps: Vec<u64> = evals(&out).iter().map(|e| e.step).collect();
    assert_eq!(steps, [0, 30, 60, 90, 120, 150, 180, 200], "{out}");
    let lines: Vec<&str> = out.lines().collect();
    for k in steps {
        let at = lines
            .iter()
            .position(|l| l.starts_with(&format!("eval {k} ")))
            .unwrap();
        assert!(
            lines[at - 1].starts_with(&format!("step {k} loss ")),
            "eval {k} follows its step line:\n{out}"
        );
    }
}

/// Scenario 1, D55: a run resumed at step 100 evaluates at 100 and 200, the steps the unbroken
/// run evaluates from 100 on, and prints their `eval` lines bit for bit.
#[test]
fn values_resume_evaluates_the_steps_of_the_unbroken_run() {
    let case = scratch("values_resume");
    let packed = packed(&case);
    let whole = train(MLM_BAG, &packed, &["--eval-every", "100"]);
    assert_exit(&whole, 0);
    let checkpoint = case.join("half.msckpt");
    let half = train(
        MLM_BAG,
        &packed,
        &["--steps", "100", "--checkpoint", path(&checkpoint)],
    );
    assert_exit(&half, 0);
    let resumed = train(
        MLM_BAG,
        &packed,
        &[
            "--steps",
            "100",
            "--resume",
            path(&checkpoint),
            "--eval-every",
            "100",
        ],
    );
    assert_exit(&resumed, 0);
    let from_100: Vec<EvalLine> = evals(&stdout(&whole))
        .into_iter()
        .filter(|e| e.step >= 100)
        .collect();
    assert_eq!(from_100.len(), 2);
    assert_eq!(evals(&stdout(&resumed)), from_100, "{}", stdout(&resumed));

    let odd = train(
        MLM_BAG,
        &packed,
        &[
            "--steps",
            "100",
            "--resume",
            path(&checkpoint),
            "--eval-every",
            "70",
        ],
    );
    assert_exit(&odd, 0);
    let steps: Vec<u64> = evals(&stdout(&odd)).iter().map(|e| e.step).collect();
    assert_eq!(
        steps,
        [140, 200],
        "the multiples of 70 from 100 on, and the end"
    );
}

/// Scenario 2: `mlm_bag_dropout.mspec` is `mlm_bag.mspec` with one `dropout(..., p = 0.1)` after
/// the embedding. With one seed both print one `init` line and one `eval 0` line bit for bit,
/// because evaluation runs without dropout on equal initial parameters, while their `step 0`
/// lines differ, because the training step applies dropout.
#[test]
fn dropout_off_in_the_evaluation() {
    let plain = std::fs::read_to_string(repo_root().join(MLM_BAG)).unwrap();
    let dropout = std::fs::read_to_string(repo_root().join(MLM_BAG_DROPOUT)).unwrap();
    assert_eq!(
        dropout,
        plain.replace(
            "matmul(embedding(table, ids), transpose(table))",
            "matmul(dropout(embedding(table, ids), p = 0.1), transpose(table))"
        ),
        "the dropout fixture is the plain one with one dropout call after the embedding"
    );
    let case = scratch("dropout_off");
    let packed = packed(&case);
    let a = train(MLM_BAG, &packed, &["--eval-every", "100"]);
    let b = train(MLM_BAG_DROPOUT, &packed, &["--eval-every", "100"]);
    assert_exit(&a, 0);
    assert_exit(&b, 0);
    let (a, b) = (stdout(&a), stdout(&b));
    let line = |out: &str, prefix: &str| -> String {
        out.lines()
            .find(|l| l.starts_with(prefix))
            .unwrap_or_else(|| panic!("a {prefix:?} line: {out}"))
            .to_owned()
    };
    assert_eq!(
        line(&a, "init "),
        line(&b, "init "),
        "equal initial parameters"
    );
    assert_eq!(
        line(&a, "eval 0 "),
        line(&b, "eval 0 "),
        "eval 0 without dropout"
    );
    assert_ne!(
        line(&a, "step 0 "),
        line(&b, "step 0 "),
        "the training step applies dropout"
    );
    let computed = expected(MLM_BAG_DROPOUT, &packed, &tensors(&b, "param"));
    let last = evals(&b).pop().unwrap();
    assert_eq!(last.step, 200);
    assert!(
        close(last.loss(), computed.loss) && close(last.accuracy(), computed.accuracy),
        "eval 200 printed {last:?}, computed without dropout {computed:?}"
    );
}

/// The little-endian bytes of `values`.
fn f32_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}

/// Scenario 3: a model whose logits tie by construction. `mlm_bag.mspec` with its tied decoder
/// replaced by a `decoder: [16, 300]` parameter; `--init` sets every `table` entry to 0.25 and
/// gives `decoder` ones in the two columns `low < high` and zeros elsewhere, so every row's
/// logits are 4 at `low` and `high` and 0 elsewhere. `low` is the most frequent selected
/// held-out label and `high` the first id above it selected another number of times, so the
/// lowest-id rule and the highest-id rule give different accuracies, and the printed one is the
/// lowest-id rule's. A zero decoder ties all 300 ids, whose lowest, id 0, is a special token and
/// never a selected label, so its accuracy is 0.
#[test]
fn accuracy_ties_go_to_the_lowest_id() {
    let case = scratch("accuracy_ties");
    let packed = packed(&case);
    let file = variant(
        &case,
        "decoder",
        &[
            (
                "    param table: Tensor<f32, [300, 16]> = xavier_uniform();\n",
                "    param table: Tensor<f32, [300, 16]> = xavier_uniform();\n    param decoder: Tensor<f32, [16, 300]> = zeros();\n",
            ),
            ("transpose(table)", "decoder"),
        ],
    );
    let table = vec![0.25f32; 300 * 16];
    let zero_decoder = vec![0.0f32; 16 * 300];
    let probe = expected(
        &file,
        &packed,
        &BTreeMap::from([
            ("MlmBag.table".to_owned(), table.clone()),
            ("MlmBag.decoder".to_owned(), zero_decoder.clone()),
        ]),
    );
    let mut counts: BTreeMap<i32, usize> = BTreeMap::new();
    for &label in &probe.selected_labels {
        *counts.entry(label).or_default() += 1;
    }
    let count = |id: i32| counts.get(&id).copied().unwrap_or(0);
    let (&low, _) = counts
        .iter()
        .max_by_key(|&(&id, &c)| (c, std::cmp::Reverse(id)))
        .expect("a selected label");
    let high = (low + 1..300)
        .find(|&id| count(id) != count(low))
        .expect("an id above the most frequent label with another count");
    let selected = probe.selected_labels.len() as f32;
    let lowest_rule = count(low) as f32 / selected;
    let highest_rule = count(high) as f32 / selected;
    assert_ne!(lowest_rule, highest_rule);

    let mut decoder = zero_decoder.clone();
    for k in 0..16 {
        decoder[k * 300 + low as usize] = 1.0;
        decoder[k * 300 + high as usize] = 1.0;
    }
    for (name, decoder, accuracy) in [("pair", decoder, lowest_rule), ("zero", zero_decoder, 0.0)] {
        let init = case.join(format!("init_{name}"));
        std::fs::create_dir_all(&init).unwrap();
        std::fs::write(init.join("table.f32"), f32_bytes(table.iter().copied())).unwrap();
        std::fs::write(init.join("decoder.f32"), f32_bytes(decoder.iter().copied())).unwrap();
        let output = train(
            &file,
            &packed,
            &["--steps", "0", "--init", path(&init), "--eval-every", "1"],
        );
        assert_exit(&output, 0);
        let out = stdout(&output);
        let evals = evals(&out);
        assert_eq!(evals.len(), 1, "{out}");
        assert_eq!(evals[0].step, 0, "{out}");
        let computed = expected(&file, &packed, &tensors(&out, "init"));
        for logits in &computed.logits {
            for row in logits.as_chunks::<300>().0 {
                let max = row.iter().copied().fold(f32::MIN, f32::max);
                let tied = row.iter().filter(|&&v| v == max).count();
                assert!(tied >= 2, "{name}: every row's largest logit ties: {row:?}");
            }
        }
        assert_eq!(
            computed.accuracy, accuracy,
            "{name}: the test's lowest-id rule"
        );
        assert_eq!(
            evals[0].accuracy(),
            accuracy,
            "{name}: printed {:?}, the lowest-id rule gives {accuracy}, the highest-id rule \
             {highest_rule}",
            evals[0]
        );
        assert!(
            close(evals[0].loss(), computed.loss),
            "{name}: {evals:?} {computed:?}"
        );
    }
}

/// A serving `modelspec train` process, killed when dropped.
struct Served {
    child: Child,
    address: String,
    stdout: Option<JoinHandle<Vec<u8>>>,
}

impl Served {
    /// Starts `modelspec <args> --serve 127.0.0.1:0` and waits for the line naming the bound
    /// address.
    fn start(args: &[&str]) -> Served {
        let mut child = Command::new(env!("CARGO_BIN_EXE_modelspec"))
            .args(args)
            .args(["--serve", "127.0.0.1:0"])
            .current_dir(repo_root())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("modelspec starts");
        let mut out = child.stdout.take().expect("stdout is piped");
        let stdout = std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = out.read_to_end(&mut bytes);
            bytes
        });
        let (lines, received) = mpsc::channel();
        let err = child.stderr.take().expect("stderr is piped");
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines() {
                let Ok(line) = line else { break };
                let _ = lines.send(line);
            }
        });
        let line = received.recv_timeout(WAIT).expect("a `serving` line");
        let address = line
            .strip_prefix("serving http://")
            .and_then(|rest| rest.strip_suffix("/events"))
            .unwrap_or_else(|| panic!("`serving http://<addr>/events`: {line}"))
            .to_owned();
        Served {
            child,
            address,
            stdout: Some(stdout),
        }
    }

    /// The data of every frame of `GET /events`, read until the server ends the response.
    fn events(&self) -> Vec<Value> {
        let mut stream = TcpStream::connect(&self.address).expect("a connection");
        stream.set_read_timeout(Some(WAIT)).unwrap();
        write!(
            stream,
            "GET /events HTTP/1.1\r\nHost: {}\r\nAccept: text/event-stream\r\n\r\n",
            self.address
        )
        .unwrap();
        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .expect("the response ends");
        let (_, body) = response.split_once("\r\n\r\n").expect("a response head");
        body.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str(data).expect("data is JSON"))
            .collect()
    }

    /// Kills the process and returns its stdout.
    fn stop(&mut self) -> String {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let bytes = self
            .stdout
            .take()
            .map(|h| h.join().expect("stdout reader"))
            .unwrap_or_default();
        String::from_utf8(bytes).expect("UTF-8")
    }
}

impl Drop for Served {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Scenario 4: with `--serve`, the `StepCompleted` events of steps 0, 100 and 200 carry exactly
/// `heldout_mlm_loss` and `heldout_mlm_accuracy`, equal to the printed values; every other
/// event's `metrics` is empty, and every event validates against the observatory JSON Schema.
#[test]
fn streamed_metric_samples_equal_the_printed_values() {
    let case = scratch("streamed");
    let packed = packed(&case);
    let mut served = Served::start(&[
        "train",
        MLM_BAG,
        "--backend",
        "cpu",
        "--data",
        path(&packed),
        "--eval-every",
        "100",
    ]);
    let events = served.events();
    let out = served.stop();
    let schema = std::fs::read_to_string(repo_root().join(RUN_EVENT_SCHEMA)).unwrap();
    let schema: Value = serde_json::from_str(&schema).unwrap();
    let validator = jsonschema::validator_for(&schema).expect("the schema compiles");
    for event in &events {
        let errors: Vec<String> = validator
            .iter_errors(event)
            .map(|e| e.to_string())
            .collect();
        assert!(errors.is_empty(), "{event}: {errors:?}");
    }
    let printed: BTreeMap<u64, EvalLine> = evals(&out).into_iter().map(|e| (e.step, e)).collect();
    assert_eq!(
        printed.keys().copied().collect::<Vec<_>>(),
        [0, 100, 200],
        "{out}"
    );
    let steps: Vec<&Value> = events
        .iter()
        .filter(|e| e["body"]["kind"] == "StepCompleted")
        .map(|e| &e["body"]["value"])
        .collect();
    assert_eq!(steps.len(), 201);
    for (k, step) in steps.iter().enumerate() {
        let k = k as u64;
        assert_eq!(step["step"], k);
        let metrics = step["metrics"].as_array().expect("a metrics list");
        match printed.get(&k) {
            None => assert!(metrics.is_empty(), "step {k}: {step}"),
            Some(line) => {
                let named: Vec<(&str, String)> = metrics
                    .iter()
                    .map(|m| {
                        let value = m["value"].as_f64().expect("a finite value") as f32;
                        (m["name"].as_str().unwrap(), value.to_string())
                    })
                    .collect();
                assert_eq!(
                    named,
                    [
                        ("heldout_mlm_loss", line.loss.clone()),
                        ("heldout_mlm_accuracy", line.accuracy.clone()),
                    ],
                    "step {k}"
                );
            }
        }
    }
}

/// Scenario 5: `--eval-every 10` without `--data`, `--eval-every 0`, `--eval-every` beside a
/// dense `--data` directory and `--eval-every` with `--backend vulkan` and no `--data` are usage
/// errors, exit 2, on one line naming the flag, with nothing on stdout; `--backend vulkan` takes
/// `--eval-every` beside a packed `--data` (story:bert-vulkan-kernels-attention-loss-adamw).
#[test]
fn usage_errors_name_the_flag() {
    let case = scratch("usage");
    let packed = packed(&case);
    let dataset = case.join("dataset");
    for (args, names) in [
        (
            vec!["train", MLM_BAG, "--backend", "cpu", "--eval-every", "10"],
            vec!["'--eval-every'", "'--data"],
        ),
        (
            vec![
                "train",
                MLM_BAG,
                "--backend",
                "cpu",
                "--data",
                path(&packed),
                "--eval-every",
                "0",
            ],
            vec!["'--eval-every'", "0"],
        ),
        (
            vec![
                "train",
                MLM_BAG,
                "--backend",
                "cpu",
                "--data",
                path(&dataset),
                "--eval-every",
                "10",
            ],
            vec!["'--eval-every'", path(&dataset), "binding.json"],
        ),
        (
            vec![
                "train",
                MLM_BAG,
                "--backend",
                "vulkan",
                "--eval-every",
                "10",
            ],
            vec!["'--eval-every'", "'--data <DIR>'"],
        ),
    ] {
        let output = modelspec(&args);
        assert_exit(&output, 2);
        assert_eq!(stdout(&output), "", "{args:?}");
        let text = stderr(&output);
        assert_eq!(text.lines().count(), 1, "{args:?}: one line\n{text}");
        assert!(text.starts_with("error: "), "{args:?}\n{text}");
        for name in names {
            assert!(text.contains(name), "{args:?}: {name}\n{text}");
        }
    }
}

/// D55: a `HeldOut` split of fewer than `B` sequences is refused before the run starts, exit 1,
/// on one line naming both numbers: the fixture with `B = 16`, which its `Train` split holds and
/// its `HeldOut` split does not.
#[test]
fn heldout_smaller_than_the_batch_is_refused() {
    let case = scratch("heldout_small");
    let packed = packed(&case);
    let loaded = mlm::load(&packed).unwrap();
    let (train_n, held_n) = (
        loaded.sequence_count(SplitRole::Train),
        loaded.sequence_count(SplitRole::HeldOut),
    );
    let b = held_n + 1;
    assert!(
        b <= train_n,
        "the Train split holds a batch of {b}: {train_n}"
    );
    let rows = (b * S).to_string();
    let file = variant(
        &case,
        "large_batch",
        &[
            ("[64]", &format!("[{rows}]")),
            ("[64, 300]", &format!("[{rows}, 300]")),
        ],
    );
    let plain = train(&file, &packed, &["--steps", "1"]);
    assert_exit(&plain, 0);
    let output = train(&file, &packed, &["--steps", "1", "--eval-every", "1"]);
    assert_exit(&output, 1);
    let text = stderr(&output);
    assert_eq!(text.lines().count(), 1, "{text}");
    assert!(text.contains(&format!(" {b} ")), "the batch {b}: {text}");
    assert!(
        text.contains(&format!(" {held_n} ")),
        "the split's {held_n}: {text}"
    );
    assert!(text.contains("HeldOut"), "{text}");
    assert!(
        !stdout(&output).lines().any(|l| l.starts_with("step ")),
        "no step runs: {}",
        stdout(&output)
    );
}
