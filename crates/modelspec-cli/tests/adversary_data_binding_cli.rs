//! Adversary cases of story:bert-train-data-binding against `modelspec train --backend cpu
//! --data <packed-dir>` (coordinator decision D48): the binding checks, the dense/packed
//! decision, the `data` line and `RunStarted.data_binding_id`, the seed reaching the batches,
//! and a resume from a `--checkpoint-every` file.
//!
//! The packed directories come from `modelspec data prepare`, `data tokenizer` and `data pack` on
//! the original-text fixture `crates/modelspec-runtime/tests/fixtures/corpus-tokenizer/`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use modelspec_observatory::{RunEventBody, read_recording};
use serde_json::Value;

const CORPUS: &str = "crates/modelspec-runtime/tests/fixtures/corpus-tokenizer";
const MLM_BAG: &str = "crates/modelspec-cli/tests/fixtures/mlm_bag.mspec";
const LINEAR: &str = "examples/linear_regression.mspec";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("adv_data_binding_{name}"));
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

/// The fixture corpus prepared, a tokenizer of `vocab` tokens trained on it and the two packed
/// into sequences of 16, under `case/<name>`; returns the packed directory.
fn packed_as(case: &Path, name: &str, vocab: &str) -> PathBuf {
    let root = case.join(name);
    let dataset = root.join("dataset");
    assert_exit(
        &modelspec(&["data", "prepare", CORPUS, "--out", path(&dataset)]),
        0,
    );
    let tokenizer = root.join("tokenizer.json");
    assert_exit(
        &modelspec(&[
            "data",
            "tokenizer",
            "--dataset",
            path(&dataset),
            "--vocab-size",
            vocab,
            "--out",
            path(&tokenizer),
        ]),
        0,
    );
    let packed = root.join("packed");
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

fn packed(case: &Path) -> PathBuf {
    packed_as(case, "a", "300")
}

/// A copy of the packed directory `from` at `to`.
fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
    }
}

fn train(file: &str, data: &Path, extra: &[&str]) -> Output {
    let mut args = vec!["train", file, "--backend", "cpu", "--data", path(data)];
    args.extend_from_slice(extra);
    modelspec(&args)
}

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

fn json(file: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(file).unwrap()).unwrap()
}

fn losses(stdout: &str) -> Vec<(u64, String)> {
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix("step "))
        .map(|rest| {
            let (step, loss) = rest.split_once(" loss ").expect("`step <k> loss <v>`");
            (step.parse().unwrap(), loss.to_owned())
        })
        .collect()
}

fn params(out: &str) -> Vec<String> {
    out.lines()
        .filter(|l| l.starts_with("param "))
        .map(str::to_owned)
        .collect()
}

/// Exit 1, nothing on stdout, never a panic, and one stderr line holding each of `names`.
fn assert_refused_naming(output: &Output, names: &[&str]) {
    assert_exit(output, 1);
    let text = stderr(output);
    assert_eq!(stdout(output), "", "nothing runs: {text}");
    assert_eq!(text.trim_end().lines().count(), 1, "one line: {text}");
    for name in names {
        assert!(text.contains(name), "{name}: {text}");
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// H4: a `binding.json` whose id differs in one hex digit, one with `seq_len` 0, a `packed.json`
/// of another binding (another tokenizer, the same sequence length) beside the binding's own
/// `binding.json`, and a `train.tokens` one byte short each exit 1 on one line naming the file.
#[test]
fn adv_db_binding_checks_refuse_naming_the_file() {
    let case = scratch("binding_checks");
    let packed = packed(&case);

    let edited = case.join("edited");
    copy_dir(&packed, &edited);
    let mut binding = json(&edited.join("binding.json"));
    let id = binding["binding_id"].as_str().unwrap().to_owned();
    let last = if id.ends_with('0') { "1" } else { "0" };
    binding["binding_id"] = Value::from(format!("{}{last}", &id[..id.len() - 1]));
    std::fs::write(edited.join("binding.json"), binding.to_string()).unwrap();
    assert_refused_naming(
        &train(MLM_BAG, &edited, &[]),
        &[path(&edited), "binding.json"],
    );

    let zero = case.join("zero");
    copy_dir(&packed, &zero);
    let mut binding = json(&zero.join("binding.json"));
    binding["seq_len"] = Value::from(0);
    std::fs::write(zero.join("binding.json"), binding.to_string()).unwrap();
    assert_refused_naming(&train(MLM_BAG, &zero, &[]), &[path(&zero), "binding.json"]);

    let policy = case.join("policy");
    copy_dir(&packed, &policy);
    let mut binding = json(&policy.join("binding.json"));
    binding["masking_policy"]["selection_rate"] = Value::from(0.5);
    std::fs::write(policy.join("binding.json"), binding.to_string()).unwrap();
    assert_refused_naming(
        &train(MLM_BAG, &policy, &[]),
        &[path(&policy), "binding.json"],
    );
    let text = stderr(&train(MLM_BAG, &policy, &[]));
    assert!(
        !text.contains("packed.json"),
        "refused by the binding's own id: {text}"
    );

    let other = packed_as(&case, "b", "280");
    assert_ne!(
        json(&other.join("binding.json"))["binding_id"],
        json(&packed.join("binding.json"))["binding_id"]
    );
    let foreign = case.join("foreign");
    copy_dir(&other, &foreign);
    std::fs::copy(packed.join("binding.json"), foreign.join("binding.json")).unwrap();
    assert_refused_naming(
        &train(MLM_BAG, &foreign, &[]),
        &[path(&foreign), "packed.json"],
    );

    let short = case.join("short");
    copy_dir(&packed, &short);
    let tokens = std::fs::read(short.join("train.tokens")).unwrap();
    std::fs::write(short.join("train.tokens"), &tokens[..tokens.len() - 1]).unwrap();
    assert_refused_naming(
        &train(MLM_BAG, &short, &[]),
        &[path(&short), "train.tokens"],
    );
}

/// H2 and H4: `B` equal to the `Train` sequence count trains; one more is refused on one line
/// naming the directory, before any step.
#[test]
fn adv_db_batch_of_all_train_sequences_and_one_more() {
    let case = scratch("batch_bound");
    let packed = packed(&case);
    let record = json(&packed.join("packed.json"));
    let train_split = record["splits"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["role"] == "Train")
        .expect("a Train split");
    let n = train_split["sequence_count"].as_u64().unwrap();
    let rows = |b: u64| (b * 16).to_string();
    let all = variant(&case, "all", &[("64", &rows(n))]);
    let output = train(&all, &packed, &["--steps", "1"]);
    assert_exit(&output, 0);
    let over = variant(&case, "over", &[("64", &rows(n + 1))]);
    assert_refused_naming(&train(&over, &packed, &[]), &[path(&packed), "Train"]);
}

/// H3: on a packed directory the dense declarations `x` and `y` are E-DATA-005, and so is a
/// dense directory that also holds a `binding.json` (D48: the file decides); a directory that
/// does not exist, under a model whose declarations are not batch tensors, is the dense reader's
/// refusal naming it, not E-DATA-005.
#[test]
fn adv_db_dense_packed_decision() {
    let case = scratch("decision");
    let packed = packed(&case);
    let output = train(LINEAR, &packed, &[]);
    assert_exit(&output, 1);
    let text = stderr(&output);
    assert!(text.starts_with("E-DATA-005: "), "{text}");
    assert!(text.contains("`x`"), "{text}");
    assert_eq!(stdout(&output), "");

    let stray = case.join("stray");
    std::fs::create_dir_all(&stray).unwrap();
    std::fs::write(stray.join("x.f32"), f32_bytes(&[0.5; 32])).unwrap();
    std::fs::write(stray.join("y.f32"), f32_bytes(&[0.25; 8])).unwrap();
    let dense = train(LINEAR, &stray, &["--steps", "1"]);
    assert_exit(&dense, 0);
    let output = train(MLM_BAG, &stray, &[]);
    assert_exit(&output, 1);
    let text = stderr(&output);
    assert!(
        text.starts_with(&format!(
            "modelspec train {MLM_BAG}: {} holds no binding.json, so it is read as a dense \
             directory\nE-DATA-001: ",
            path(&stray)
        )),
        "D56: the directory and binding.json, then E-DATA-001: {text}"
    );
    assert_eq!(stdout(&output), "");
    std::fs::copy(packed.join("binding.json"), stray.join("binding.json")).unwrap();
    let output = train(LINEAR, &stray, &["--steps", "1"]);
    assert_exit(&output, 1);
    assert!(
        stderr(&output).starts_with("E-DATA-005: "),
        "{}",
        stderr(&output)
    );

    let missing = case.join("no-such-dense-directory");
    let output = train(LINEAR, &missing, &[]);
    assert_exit(&output, 1);
    let text = stderr(&output);
    assert!(!text.contains("E-DATA-005"), "{text}");
    assert!(text.contains(path(&missing)), "{text}");
}

/// H5: a dense run prints no `data` line (its first line is an `init` line) and its `RunStarted`
/// carries no `data_binding_id`; a packed run's recording reads back through
/// `read_recording` with the binding's id as `data_binding_id`.
#[test]
fn adv_db_data_line_and_run_started_only_on_packed_runs() {
    let case = scratch("run_started");
    let packed = packed(&case);
    let dense_dir = case.join("dense");
    std::fs::create_dir_all(&dense_dir).unwrap();
    std::fs::write(dense_dir.join("x.f32"), f32_bytes(&[0.5; 32])).unwrap();
    std::fs::write(dense_dir.join("y.f32"), f32_bytes(&[0.25; 8])).unwrap();
    let record = case.join("dense.jsonl");
    let output = train(
        LINEAR,
        &dense_dir,
        &["--steps", "1", "--record", path(&record)],
    );
    assert_exit(&output, 0);
    assert!(stdout(&output).starts_with("init "), "{}", stdout(&output));
    let text = std::fs::read_to_string(&record).unwrap();
    assert!(!text.contains("data_binding_id"), "{text}");

    let record = case.join("packed.jsonl");
    let output = train(
        MLM_BAG,
        &packed,
        &["--steps", "1", "--record", path(&record)],
    );
    assert_exit(&output, 0);
    let events = read_recording(&record).expect("the recording reads back");
    let RunEventBody::RunStarted(started) = &events[0].body else {
        panic!("the first event is RunStarted: {events:?}");
    };
    let id = json(&packed.join("binding.json"))["binding_id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(
        started.data_binding_id.map(|b| b.0.to_string()),
        Some(id),
        "{started:?}"
    );
}

/// H7: with the parameters fixed by `--init`, a run with another seed prints another step-0
/// loss (the seed reaches the batch), and the `data` line does not depend on the seed.
#[test]
fn adv_db_seed_reaches_the_batches_and_not_the_data_line() {
    let case = scratch("seed");
    let packed = packed(&case);
    let init = case.join("init");
    std::fs::create_dir_all(&init).unwrap();
    let table: Vec<f32> = (0..300 * 16)
        .map(|i| ((i as f32) * 0.37).sin() * 0.1)
        .collect();
    std::fs::write(init.join("table.f32"), f32_bytes(&table)).unwrap();
    let eight = variant(&case, "seed_8", &[("seed = 7;", "seed = 8;")]);
    let a = train(MLM_BAG, &packed, &["--steps", "1", "--init", path(&init)]);
    let b = train(&eight, &packed, &["--steps", "1", "--init", path(&init)]);
    assert_exit(&a, 0);
    assert_exit(&b, 0);
    let (a, b) = (stdout(&a), stdout(&b));
    assert_eq!(a.lines().next(), b.lines().next(), "the data line");
    assert_eq!(
        a.lines().nth(1),
        b.lines().nth(1),
        "the init line: --init fixes the parameters"
    );
    assert_ne!(
        losses(&a)[0],
        losses(&b)[0],
        "seed 7 and seed 8 feed one batch at step 0\n{a}\n{b}"
    );
}

/// H1: a run resumed from the `--checkpoint-every` file of step 150 prints the step-200 loss and
/// the final parameters of the run that never stopped.
#[test]
fn adv_db_resume_from_checkpoint_every_matches_the_unbroken_run() {
    let case = scratch("every");
    let packed = packed(&case);
    let checkpoint = case.join("run.msckpt");
    let whole = train(
        MLM_BAG,
        &packed,
        &[
            "--checkpoint",
            path(&checkpoint),
            "--checkpoint-every",
            "50",
        ],
    );
    assert_exit(&whole, 0);
    let mid = case.join("run-150.msckpt");
    assert!(mid.exists(), "{}", mid.display());
    let resumed = train(MLM_BAG, &packed, &["--steps", "50", "--resume", path(&mid)]);
    assert_exit(&resumed, 0);
    let (whole, resumed) = (stdout(&whole), stdout(&resumed));
    let at = |out: &str, k: u64| losses(out).into_iter().find(|(s, _)| *s == k);
    assert_eq!(at(&whole, 200), at(&resumed, 200), "{resumed}");
    assert_eq!(params(&whole), params(&resumed));
}

/// H2 end to end: a model reading all five batch tensors, `[B·S]` ids beside a `[B, S]` key
/// mask, trains from the CLI.
#[test]
fn adv_db_all_five_tensors_train_from_the_cli() {
    let case = scratch("all_five");
    let packed = packed(&case);
    let file = case.join("mini.mspec");
    std::fs::write(
        &file,
        "model Mini {
    param table: Tensor<f32, [300, 16]> = xavier_uniform();
    param pos: Tensor<f32, [16, 16]> = xavier_uniform();
    param w_q: Tensor<f32, [16, 16]> = xavier_uniform();
    param w_k: Tensor<f32, [16, 16]> = xavier_uniform();
    param w_v: Tensor<f32, [16, 16]> = xavier_uniform();

    fn forward(ids: Tensor<i32, [64]>, positions: Tensor<i32, [64]>, key_mask: Tensor<f32, [4, 16]>) -> Tensor<f32, [64, 300]> {
        return matmul(attention(matmul(add(embedding(table, ids), embedding(pos, positions)), w_q), matmul(add(embedding(table, ids), embedding(pos, positions)), w_k), matmul(add(embedding(table, ids), embedding(pos, positions)), w_v), key_mask, heads = 2), transpose(table));
    }
}

train Fit {
    model = Mini;
    input ids: Tensor<i32, [64]>;
    input positions: Tensor<i32, [64]>;
    input key_mask: Tensor<f32, [4, 16]>;
    target labels: Tensor<i32, [64]>;
    target weights: Tensor<f32, [64]>;
    loss = cross_entropy(model.forward(ids, positions, key_mask), labels, weights);
    optimizer = sgd(lr = 0.5);
    steps = 2;
    seed = 7;
    device = cpu;
}
",
    )
    .unwrap();
    let output = train(path(&file), &packed, &[]);
    assert_exit(&output, 0);
    assert!(
        stdout(&output).starts_with("data dataset "),
        "{}",
        stdout(&output)
    );
}

/// H3: `--eval` beside a packed `--data` (not implemented) is refused before any step, never a
/// panic: a usage error, exit 2, on one line naming the `--data` directory (coordinator decision
/// D56).
#[test]
fn adv_db_eval_beside_packed_data_is_refused_before_any_step() {
    let case = scratch("eval");
    let packed = packed(&case);
    let output = train(MLM_BAG, &packed, &["--eval", path(&packed)]);
    assert_exit(&output, 2);
    assert_eq!(stdout(&output), "", "{}", stderr(&output));
    let text = stderr(&output);
    assert_eq!(text.trim_end().lines().count(), 1, "one line: {text}");
    for name in ["--eval", "--data", path(&packed), "packed MLM directory"] {
        assert!(text.contains(name), "{name}: {text}");
    }
}
