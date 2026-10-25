//! `modelspec train --backend cpu --data <packed-dir>` (story:bert-train-data-binding): each
//! `input` and `target` declaration of the `train` block names one tensor of the
//! masked-language-model batch (`ids`, `positions`, `key_mask`, `labels`, `weights`), step `k`
//! feeds the `Train` batch of step `k` drawn with the run seed, and the run prints a `data` line
//! naming its `TrainingDataBinding` before the first `init` line (coordinator decision D48).
//!
//! The packed directory is made by `modelspec data prepare`, `data tokenizer` and `data pack` from
//! the original-text fixture `crates/modelspec-runtime/tests/fixtures/corpus-tokenizer/`, as
//! `data_pack.rs` makes it: a 300-token vocabulary and sequences of 16, so `mlm_bag.mspec`
//! declares its 64 rows as `[B·S]` with `B = 4`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

const CORPUS: &str = "crates/modelspec-runtime/tests/fixtures/corpus-tokenizer";
const MLM_BAG: &str = "crates/modelspec-cli/tests/fixtures/mlm_bag.mspec";
const RUN_EVENT_SCHEMA: &str =
    "crates/modelspec-observatory/schema/entities/modelspec.observatory.RunEvent.schema.json";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh, empty directory under this crate's build directory, named for the test.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("train_data_{name}"));
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

fn json(file: &Path) -> Value {
    let text = std::fs::read_to_string(file).expect("the record is readable");
    serde_json::from_str(&text).expect("the record is JSON")
}

/// The `data` line D48 settles, from the `binding.json` of `packed`.
fn data_line(packed: &Path) -> String {
    let binding = json(&packed.join("binding.json"));
    let policy = &binding["masking_policy"];
    let rate = |name: &str| policy[name].as_f64().expect("a rate").to_string();
    format!(
        "data dataset {} tokenizer {} seq_len {} masking {}/{}/{}/{} binding {}",
        binding["dataset_id"].as_str().unwrap(),
        binding["tokenizer_id"].as_str().unwrap(),
        binding["seq_len"].as_u64().unwrap(),
        rate("selection_rate"),
        rate("mask_fraction"),
        rate("random_fraction"),
        rate("keep_fraction"),
        binding["binding_id"].as_str().unwrap(),
    )
}

/// The `(step, loss)` of every `step` line.
fn losses(stdout: &str) -> Vec<(u64, f32)> {
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix("step "))
        .map(|rest| {
            let (step, loss) = rest.split_once(" loss ").expect("`step <k> loss <v>`");
            (step.parse().unwrap(), loss.parse().unwrap())
        })
        .collect()
}

/// Scenario 1: the fixture trains on the packed batches: exit 0, the `data` line first, then an
/// `init` line, and the loss of step 200 below the loss of step 0.
#[test]
fn trains_mlm_bag_on_packed_batches() {
    let case = scratch("trains");
    let packed = packed(&case);
    let output = train(MLM_BAG, &packed, &[]);
    assert_exit(&output, 0);
    let out = stdout(&output);
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines[0], data_line(&packed), "{out}");
    assert!(
        lines[0].contains(" masking 0.15/0.8/0.1/0.1 binding "),
        "{out}"
    );
    assert!(
        lines[1].starts_with("init @MlmBag.table: Tensor<f32, [300, 16]> = ["),
        "{out}"
    );
    let losses = losses(&out);
    let steps: Vec<u64> = losses.iter().map(|&(k, _)| k).collect();
    assert_eq!(steps, [0, 50, 100, 150, 200], "{out}");
    let (first, last) = (losses[0].1, losses[4].1);
    assert!(first.is_finite() && last.is_finite(), "{losses:?}");
    assert!(
        last < first,
        "step 200 loss {last} not below step 0 loss {first}"
    );
    assert!(
        out.lines()
            .last()
            .unwrap()
            .starts_with("param @MlmBag.table:"),
        "{out}"
    );
}

/// Scenario 1, the run record: the `RunStarted` event of `--record` carries the binding's id as
/// `data_binding_id` and validates against the generated `RunEvent` schema; a run on synthetic
/// data carries none.
#[test]
fn trains_records_the_data_binding_in_run_started() {
    let case = scratch("trains_record");
    let packed = packed(&case);
    let record = case.join("run.jsonl");
    let output = train(
        MLM_BAG,
        &packed,
        &["--steps", "2", "--record", path(&record)],
    );
    assert_exit(&output, 0);
    let text = std::fs::read_to_string(&record).expect("the record is written");
    let first: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    assert_eq!(first["body"]["kind"], "RunStarted", "{text}");
    let binding = json(&packed.join("binding.json"));
    assert_eq!(
        first["body"]["value"]["data_binding_id"], binding["binding_id"],
        "{text}"
    );
    let schema = std::fs::read_to_string(repo_root().join(RUN_EVENT_SCHEMA)).unwrap();
    let schema: Value = serde_json::from_str(&schema).unwrap();
    let validator = jsonschema::validator_for(&schema).expect("the schema compiles");
    for line in text.lines() {
        let event: Value = serde_json::from_str(line).unwrap();
        let errors: Vec<String> = validator
            .iter_errors(&event)
            .map(|e| e.to_string())
            .collect();
        assert!(errors.is_empty(), "{line}: {errors:?}");
    }

    let synthetic = case.join("synthetic.jsonl");
    let output = modelspec(&[
        "train",
        "examples/linear_regression.mspec",
        "--backend",
        "cpu",
        "--steps",
        "1",
        "--record",
        path(&synthetic),
    ]);
    assert_exit(&output, 0);
    let text = std::fs::read_to_string(&synthetic).unwrap();
    let first: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    assert!(
        first["body"]["value"].get("data_binding_id").is_none(),
        "{text}"
    );
}

/// Scenario 2: two runs with one seed print identical stdout; a run with another seed prints
/// other `step` losses.
#[test]
fn reproducible_with_one_seed_and_not_with_another() {
    let case = scratch("reproducible");
    let packed = packed(&case);
    let first = train(MLM_BAG, &packed, &[]);
    let second = train(MLM_BAG, &packed, &[]);
    assert_exit(&first, 0);
    assert_exit(&second, 0);
    assert_eq!(stdout(&first), stdout(&second));
    let other = variant(&case, "seed_8", &[("seed = 7;", "seed = 8;")]);
    let third = train(&other, &packed, &[]);
    assert_exit(&third, 0);
    let (a, b) = (losses(&stdout(&first)), losses(&stdout(&third)));
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter().zip(&b).all(|(x, y)| x.1 != y.1),
        "seed 7 {a:?}\nseed 8 {b:?}"
    );
}

/// Scenario 2, the seed reaches the batches: with the parameters fixed by `--init`, so that the
/// seed cannot change them, seed 7 and seed 8 print one `init` line and other step-0 losses,
/// because they feed other batches at step 0.
#[test]
fn reproducible_batches_follow_the_seed_with_fixed_init() {
    let case = scratch("reproducible_fixed_init");
    let packed = packed(&case);
    let init = case.join("init");
    std::fs::create_dir_all(&init).unwrap();
    let table: Vec<u8> = (0..300 * 16)
        .flat_map(|i| (((i as f32) * 0.61).cos() * 0.1).to_le_bytes())
        .collect();
    std::fs::write(init.join("table.f32"), table).unwrap();
    let eight = variant(&case, "seed_8", &[("seed = 7;", "seed = 8;")]);
    let seven = train(MLM_BAG, &packed, &["--steps", "1", "--init", path(&init)]);
    let eight = train(&eight, &packed, &["--steps", "1", "--init", path(&init)]);
    assert_exit(&seven, 0);
    assert_exit(&eight, 0);
    let (seven, eight) = (stdout(&seven), stdout(&eight));
    let init_line = |out: &str| {
        out.lines()
            .find(|l| l.starts_with("init "))
            .map(str::to_owned)
    };
    assert!(init_line(&seven).is_some(), "{seven}");
    assert_eq!(
        init_line(&seven),
        init_line(&eight),
        "--init fixes the parameters"
    );
    let (a, b) = (losses(&seven), losses(&eight));
    assert_eq!((a[0].0, b[0].0), (0, 0), "{seven}\n{eight}");
    assert_ne!(
        a[0].1, b[0].1,
        "seed 7 and seed 8 feed one batch at step 0\n{seven}\n{eight}"
    );
}

/// Scenario 2, D48: step `k` feeds the batch of step `k`, so a run resumed at step 100 prints
/// the `step` lines and final parameters of the run that never stopped.
#[test]
fn reproducible_resume_draws_the_batches_of_the_unbroken_run() {
    let case = scratch("reproducible_resume");
    let packed = packed(&case);
    let whole = train(MLM_BAG, &packed, &[]);
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
        &["--steps", "100", "--resume", path(&checkpoint)],
    );
    assert_exit(&resumed, 0);
    let (whole, resumed) = (stdout(&whole), stdout(&resumed));
    let from_100 = |out: &str| -> Vec<(u64, f32)> {
        losses(out).into_iter().filter(|&(k, _)| k >= 100).collect()
    };
    let params = |out: &str| -> Vec<String> {
        out.lines()
            .filter(|l| l.starts_with("param "))
            .map(str::to_owned)
            .collect()
    };
    assert_eq!(from_100(&whole).len(), 3, "{whole}");
    assert_eq!(from_100(&whole), from_100(&resumed), "{resumed}");
    assert_eq!(params(&whole), params(&resumed));
    assert_eq!(
        resumed.lines().next(),
        whole.lines().next(),
        "the same data line"
    );
}

/// Scenario 3: a declaration named `tokens` is E-DATA-005, a `labels` declared `f32` and an `S`
/// of 8 against a sequence length of 16 are E-DATA-006, each naming the declaration, before any
/// file beyond the binding is read and before any line is printed; a missing packed directory
/// and an unreadable packed file exit 1 naming them.
#[test]
fn refusals_name_the_declaration_or_the_directory() {
    let case = scratch("refusals");
    let packed = packed(&case);
    // A copy without its token files: a declaration refusal never reads them.
    let bare = case.join("bare");
    std::fs::create_dir_all(&bare).unwrap();
    std::fs::copy(packed.join("binding.json"), bare.join("binding.json")).unwrap();

    let tokens = variant(&case, "tokens", &[("ids", "tokens")]);
    let labels_f32 = variant(
        &case,
        "labels_f32",
        &[
            (
                "target labels: Tensor<i32, [64]>;",
                "target labels: Tensor<f32, [64]>;",
            ),
            (
                "model.forward(ids), labels, weights",
                "model.forward(ids), ids, weights",
            ),
        ],
    );
    let s_8 = variant(
        &case,
        "s_8",
        &[(
            "    input ids: Tensor<i32, [64]>;\n",
            "    input ids: Tensor<i32, [64]>;\n    input key_mask: Tensor<f32, [4, 8]>;\n",
        )],
    );
    for (file, code, names) in [
        (&tokens, "E-DATA-005", ["`tokens`", "ids"]),
        (&labels_f32, "E-DATA-006", ["`labels`", "i32"]),
        (&s_8, "E-DATA-006", ["`key_mask`", "[4, 16]"]),
    ] {
        for dir in [&packed, &bare] {
            let output = train(file, dir, &[]);
            assert_exit(&output, 1);
            let text = stderr(&output);
            assert!(text.starts_with(&format!("{code}: ")), "{file}:\n{text}");
            for name in names {
                assert!(text.contains(name), "{file}: {name}\n{text}");
            }
            assert_eq!(stdout(&output), "", "{file}: nothing runs");
        }
    }

    let missing = case.join("no-such-packed-directory");
    let output = train(MLM_BAG, &missing, &[]);
    assert_exit(&output, 1);
    let text = stderr(&output);
    assert!(text.contains(path(&missing)), "{text}");
    assert_eq!(stdout(&output), "");

    let output = train(MLM_BAG, &bare, &[]);
    assert_exit(&output, 1);
    let text = stderr(&output);
    assert!(text.contains(path(&bare)), "{text}");
    assert!(text.contains("packed.json"), "{text}");
    assert_eq!(stdout(&output), "");
}

/// The line coordinator decision D56 puts before a dense refusal of a run whose declarations all
/// name batch tensors.
fn read_as_dense(file: &str, dir: &Path) -> String {
    format!(
        "modelspec train {file}: {} holds no binding.json, so it is read as a dense directory",
        path(dir)
    )
}

/// Scenario 3, coordinator decision D56: a directory without `binding.json` is dense (D48), so
/// the fixture on the dataset directory of `data prepare` or on an empty directory is refused,
/// exit 1, with one line naming the directory and `binding.json` and then today's E-DATA-001;
/// any dense refusal of batch-tensor declarations takes that line (here E-DATA-007), and dense
/// files under those names still train without it. `--eval` beside a packed `--data` is a usage
/// error, exit 2, on one line, before anything runs.
#[test]
fn refusals_name_the_missing_binding_and_the_packed_eval() {
    let case = scratch("refusals_d56");
    let packed = packed(&case);
    let dataset = case.join("dataset");
    let empty = case.join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    for dir in [&dataset, &empty] {
        let output = train(MLM_BAG, dir, &[]);
        assert_exit(&output, 1);
        assert_eq!(stdout(&output), "", "nothing runs");
        let text = stderr(&output);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines,
            [
                read_as_dense(MLM_BAG, dir).as_str(),
                "E-DATA-001: dense data needs exactly one input and one target",
                "  crates/modelspec-cli/tests/fixtures/mlm_bag.mspec:9:1",
                "  inputs: 1",
                "  targets: 2",
            ],
            "{text}"
        );
    }

    let renamed = case.join("renamed.mspec");
    let linear = std::fs::read_to_string(repo_root().join("examples/linear_regression.mspec"))
        .unwrap()
        .replace("input x:", "input key_mask:")
        .replace("target y:", "target weights:")
        .replace(
            "mse(model.forward(x), y)",
            "mse(model.forward(key_mask), weights)",
        );
    std::fs::write(&renamed, linear).unwrap();
    let renamed = path(&renamed);
    let output = train(renamed, &empty, &[]);
    assert_exit(&output, 1);
    assert_eq!(stdout(&output), "", "nothing runs");
    let text = stderr(&output);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], read_as_dense(renamed, &empty), "{text}");
    assert!(lines[1].starts_with("E-DATA-007: "), "{text}");
    assert!(text.contains("key_mask.f32"), "{text}");
    let f32_bytes =
        |n: usize, v: f32| -> Vec<u8> { (0..n).flat_map(|_| v.to_le_bytes()).collect() };
    std::fs::write(empty.join("key_mask.f32"), f32_bytes(32, 0.5)).unwrap();
    std::fs::write(empty.join("weights.f32"), f32_bytes(8, 0.25)).unwrap();
    let output = train(renamed, &empty, &["--steps", "1"]);
    assert_exit(&output, 0);
    assert!(stdout(&output).starts_with("init "), "{}", stdout(&output));
    assert!(
        !stderr(&output).contains("binding.json"),
        "{}",
        stderr(&output)
    );

    let output = train(MLM_BAG, &packed, &["--eval", path(&packed)]);
    assert_exit(&output, 2);
    assert_eq!(stdout(&output), "", "nothing runs");
    assert_eq!(
        stderr(&output),
        format!(
            "error: the argument '--eval' reads dense evaluation tensors, and the '--data' \
             directory {} is a packed MLM directory\n",
            path(&packed)
        )
    );
    let output = train(MLM_BAG, &packed, &["--eval", path(&dataset)]);
    assert_exit(&output, 2);
}
