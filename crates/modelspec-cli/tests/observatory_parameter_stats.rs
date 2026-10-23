//! `StepCompleted.parameters` of `modelspec train --backend cpu` (story:observatory-parameter-statistics;
//! `spec/domains/observatory.yaml`, `StepCompleted`, `ParameterStatistics`, `Histogram`), as
//! approval-record:parameter-statistics-settlements settles them: `--statistics-every N`, 100
//! without the flag, samples the run's first step, every multiple of N and its last; a sampled
//! step lists each parameter in path order, its `Value` entry, then its `Gradient` entry unless
//! the step is the last or the loss does not read it; every other step's list is empty.
//!
//! The expected entries are computed here from the runtime's own state, not from the
//! observatory's statistics: the values after `k` updates are `state.values` of `runner::run`
//! with `steps = k`, and the gradient is the one a direct `sgd_step` on that state returns. The
//! norm and the histogram are computed again here, the histogram bin by bin from its definition.
//! Entries are compared as the JSON text a recorded line carries, so every float is compared by
//! its shortest round-trip form, which is one-to-one with its bits.
//!
//! Scenario ids of the story map to the tests: `observatory.stats.values-at-step` is
//! [`observatory_stats_values_at_step`], `observatory.stats.gradient-at-step` is
//! [`observatory_stats_gradient_at_step`] and `observatory.stats.unread-parameter` is
//! [`observatory_stats_unread_parameter`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_core::ir::tensor::ParamId;
use modelspec_runtime::cpu::HostTensor;
use modelspec_runtime::train::data::synthetic;
use modelspec_runtime::train::optimizer::sgd_step;
use modelspec_runtime::train::runner;
use serde_json::Value;

const EXAMPLE: &str = "examples/linear_regression.mspec";
const SCHEMA: &str =
    "crates/modelspec-observatory/schema/entities/modelspec.observatory.RunEvent.schema.json";
const BINS: usize = 32;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh path `name` under the test temp dir, without a file at it.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("observatory_parameter_stats");
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    let path = dir.join(name);
    let _ = std::fs::remove_file(&path);
    path
}

fn path_arg(path: &Path) -> String {
    path.display().to_string()
}

/// Runs `modelspec <args>` from the repository root to its end.
fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .stdin(Stdio::null())
        .output()
        .expect("modelspec runs")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// The file `model` compiled for training, for `steps` steps.
fn compiled(model: &Path, steps: u64) -> CompiledTraining {
    let source = std::fs::read_to_string(repo_root().join(model)).expect("the model is readable");
    let mut compiled = compile_training(&model.display().to_string(), &source)
        .unwrap_or_else(|d| panic!("{model:?} compiles: {d:?}"));
    compiled.config.steps = steps;
    compiled
}

/// Records `model` with `train --backend cpu --record` and `extra`, checking that it exits 0 and
/// prints what the same run without `--record` and `--statistics-every` prints; the lines of the
/// recording.
fn record(model: &Path, name: &str, extra: &[&str]) -> Vec<String> {
    let path = scratch(name);
    let model = path_arg(model);
    let mut args = vec!["train", &model, "--backend", "cpu"];
    args.extend_from_slice(extra);
    let recording = path_arg(&path);
    let mut recorded = args.clone();
    recorded.extend(["--record", recording.as_str()]);
    let run = modelspec(&recorded);
    assert_eq!(run.status.code(), Some(0), "{}", stderr(&run));
    let mut plain = Vec::new();
    let mut flags = args.iter().copied();
    while let Some(arg) = flags.next() {
        if arg == "--statistics-every" {
            flags.next();
        } else {
            plain.push(arg);
        }
    }
    let without = modelspec(&plain);
    assert_eq!(without.status.code(), Some(0), "{}", stderr(&without));
    assert_eq!(
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&without.stdout),
        "stdout is the run's without --record and --statistics-every"
    );
    let content = std::fs::read_to_string(&path).expect("the recording is readable");
    content.lines().map(str::to_owned).collect()
}

fn schema() -> jsonschema::Validator {
    let text = std::fs::read_to_string(repo_root().join(SCHEMA)).expect("the schema is committed");
    let schema: Value = serde_json::from_str(&text).expect("the schema is JSON");
    jsonschema::validator_for(&schema).expect("the schema compiles")
}

/// The `StepCompleted` lines of `lines`, by step, after checking every event against the schema.
fn step_lines(lines: &[String]) -> BTreeMap<u64, &str> {
    let validator = schema();
    let mut steps = BTreeMap::new();
    for line in lines {
        let event: Value = serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {line}"));
        let errors: Vec<String> = validator
            .iter_errors(&event)
            .map(|e| e.to_string())
            .collect();
        assert!(errors.is_empty(), "{line}: {errors:?}");
        if event["body"]["kind"] == "StepCompleted" {
            let step = event["body"]["value"]["step"].as_u64().expect("a step");
            steps.insert(step, line.as_str());
        }
    }
    steps
}

/// A float as JSON writes it: its shortest round-trip form, or `null` when it is not finite.
fn number(value: f64) -> String {
    if value.is_finite() {
        serde_json::to_string(&value).expect("a finite float serializes")
    } else {
        "null".to_owned()
    }
}

/// The L2 norm: the square root of the sum of squares in `f64`, in row-major order.
fn norm(values: &[f32]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |sum, &v| sum + f64::from(v) * f64::from(v))
        .sqrt()
}

/// The JSON of the `Histogram` of `values`, from its definition: 32 bins between the finite min
/// and max, edge `i` at `min + (max - min) * i / 32` and the last at `max`; one bin `[min, max]`
/// when they are equal; NaN, +inf and −inf counted beside the bins.
fn histogram(values: &[f32]) -> String {
    let finite: Vec<f64> = values
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| f64::from(v))
        .collect();
    let nan = values.iter().filter(|v| v.is_nan()).count();
    let positive = values.iter().filter(|&&v| v == f32::INFINITY).count();
    let negative = values.iter().filter(|&&v| v == f32::NEG_INFINITY).count();
    let (edges, counts): (Vec<f64>, Vec<usize>) = if finite.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
        let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if min == max {
            (vec![min, max], vec![finite.len()])
        } else {
            let edges: Vec<f64> = (0..=BINS)
                .map(|i| {
                    if i == BINS {
                        max
                    } else {
                        min + (max - min) * i as f64 / BINS as f64
                    }
                })
                .collect();
            let counts = (0..BINS)
                .map(|i| {
                    finite
                        .iter()
                        .filter(|&&x| {
                            edges[i] <= x && (x < edges[i + 1] || (i == BINS - 1 && x == max))
                        })
                        .count()
                })
                .collect();
            (edges, counts)
        }
    };
    assert_eq!(
        counts.iter().sum::<usize>() + nan + positive + negative,
        values.len(),
        "the reference holds every element"
    );
    let edges: Vec<String> = edges.into_iter().map(number).collect();
    let counts: Vec<String> = counts.iter().map(usize::to_string).collect();
    format!(
        r#"{{"bin_edges":[{}],"counts":[{}],"nan_count":{nan},"positive_infinity_count":{positive},"negative_infinity_count":{negative}}}"#,
        edges.join(","),
        counts.join(",")
    )
}

/// The JSON of one `ParameterStatistics`.
fn entry(path: &str, subject: &str, values: &[f32]) -> String {
    format!(
        r#"{{"path":"{path}","subject":"{subject}","norm":{},"histogram":{}}}"#,
        number(norm(values)),
        histogram(values)
    )
}

/// The `Value` and `Gradient` entries the run of `model` has at step `k`, from the values after
/// `k` updates and, for `k < end`, the gradient a direct `sgd_step` on them returns.
fn expected(model: &Path, k: u64, end: u64) -> Vec<String> {
    let after_k = compiled(model, k);
    let state = runner::run(&after_k).expect("the run to step k").state;
    let gradients: BTreeMap<ParamId, HostTensor<f32>> = if k < end {
        let data = synthetic(&after_k.config, &after_k.forward).expect("synthetic data");
        let mut stepped = state.clone();
        sgd_step(&after_k.training, &data.inputs, &mut stepped)
            .expect("step k runs")
            .gradients
    } else {
        BTreeMap::new()
    };
    let mut params: Vec<(&str, ParamId)> = after_k
        .forward
        .program
        .params
        .iter()
        .map(|(&id, decl)| (decl.path.as_str(), id))
        .collect();
    params.sort_unstable();
    let mut entries = Vec::new();
    for (path, id) in params {
        entries.push(entry(path, "Value", state.values[&id].data()));
        if let Some(gradient) = gradients.get(&id) {
            entries.push(entry(path, "Gradient", gradient.data()));
        }
    }
    entries
}

/// Checks that the line of every step in `start..=end` ends with its `parameters`: `expected`'s
/// at the steps of `sampled`, empty at every other.
fn assert_parameters(lines: &[String], model: &Path, start: u64, end: u64, sampled: &[u64]) {
    let steps = step_lines(lines);
    assert_eq!(
        steps.keys().copied().collect::<Vec<_>>(),
        (start..=end).collect::<Vec<_>>()
    );
    for (&k, line) in &steps {
        let entries = if sampled.contains(&k) {
            expected(model, k, end)
        } else {
            Vec::new()
        };
        let tail = format!(
            r#""parameters":[{}]}}}},"state":"Emitted"}}"#,
            entries.join(",")
        );
        assert!(
            line.ends_with(&tail),
            "step {k}:\n{line}\nends with\n{tail}"
        );
    }
}

/// `observatory.stats.values-at-step`: at every sampled step `k` the `Value` entry of
/// `LinearReg.w` is the norm and histogram of its values after `k` updates, the last step `k =
/// steps` among them.
#[test]
fn observatory_stats_values_at_step() {
    let model = Path::new(EXAMPLE);
    let lines = record(
        model,
        "values.run.jsonl",
        &["--steps", "7", "--statistics-every", "3"],
    );
    assert_parameters(&lines, model, 0, 7, &[0, 3, 6, 7]);
    let last = step_lines(&lines)[&7];
    assert!(
        last.contains(r#""path":"LinearReg.w","subject":"Value""#),
        "{last}"
    );
    assert!(!last.contains(r#""subject":"Gradient""#), "{last}");
}

/// `observatory.stats.gradient-at-step`: at every sampled step `k < steps` the `Gradient` entry is
/// the norm and histogram of the gradient a direct `sgd_step` on the values after `k` updates
/// returns; the step `k = steps` has `Value` entries only. A run of one step still has both.
#[test]
fn observatory_stats_gradient_at_step() {
    let model = Path::new(EXAMPLE);
    let lines = record(
        model,
        "gradient.run.jsonl",
        &["--steps", "5", "--statistics-every", "2"],
    );
    assert_parameters(&lines, model, 0, 5, &[0, 2, 4, 5]);
    let steps = step_lines(&lines);
    for k in [0, 2, 4] {
        assert!(
            steps[&k].contains(r#""path":"LinearReg.w","subject":"Gradient""#),
            "step {k}: {}",
            steps[&k]
        );
    }
    let short = record(model, "one-step.run.jsonl", &["--steps", "1"]);
    assert_parameters(&short, model, 0, 1, &[0, 1]);
}

/// `observatory.stats.unread-parameter`: a parameter the loss does not read has a `Value` entry
/// at every sampled step and never a `Gradient` entry; the one it reads has both.
#[test]
fn observatory_stats_unread_parameter() {
    let source = std::fs::read_to_string(repo_root().join(EXAMPLE)).expect("the example");
    let unread = source.replace(
        "    fn forward",
        "    param u: Tensor<f32, [2, 2]> = xavier_uniform();\n\n    fn forward",
    );
    assert_ne!(unread, source, "the example declares `fn forward`");
    let model = scratch("unread.mspec");
    std::fs::write(&model, unread).expect("the model is written");
    let lines = record(
        &model,
        "unread.run.jsonl",
        &["--steps", "4", "--statistics-every", "2"],
    );
    assert_parameters(&lines, &model, 0, 4, &[0, 2, 4]);
    let steps = step_lines(&lines);
    for k in [0, 2] {
        let line = steps[&k];
        assert!(
            line.contains(r#""path":"LinearReg.u","subject":"Value""#),
            "{line}"
        );
        assert!(
            !line.contains(r#""path":"LinearReg.u","subject":"Gradient""#),
            "{line}"
        );
        assert!(
            line.contains(r#""path":"LinearReg.w","subject":"Gradient""#),
            "{line}"
        );
    }
}

/// A resumed run samples its own first step, the checkpoint's, then the multiples of N and its
/// last; the values at the first are the checkpoint's.
#[test]
fn a_resumed_run_samples_from_its_start_step() {
    let model = Path::new(EXAMPLE);
    let checkpoint = scratch("at-5.msckpt");
    let saved = modelspec(&[
        "train",
        EXAMPLE,
        "--backend",
        "cpu",
        "--steps",
        "5",
        "--checkpoint",
        &path_arg(&checkpoint),
    ]);
    assert_eq!(saved.status.code(), Some(0), "{}", stderr(&saved));
    let lines = record(
        model,
        "resumed.run.jsonl",
        &[
            "--resume",
            &path_arg(&checkpoint),
            "--steps",
            "4",
            "--statistics-every",
            "3",
        ],
    );
    assert_parameters(&lines, model, 5, 9, &[5, 6, 9]);
}

/// `--statistics-every` takes a positive number of steps, and only with `--backend cpu`: a Vulkan
/// run keeps every step's list empty.
#[test]
fn statistics_every_is_a_positive_cpu_only_flag() {
    let zero = modelspec(&[
        "train",
        EXAMPLE,
        "--backend",
        "cpu",
        "--statistics-every",
        "0",
    ]);
    assert_eq!(zero.status.code(), Some(2), "{}", stderr(&zero));
    assert!(
        stderr(&zero).contains("--statistics-every"),
        "{}",
        stderr(&zero)
    );
    let vulkan = modelspec(&[
        "train",
        EXAMPLE,
        "--backend",
        "vulkan",
        "--statistics-every",
        "3",
    ]);
    assert_eq!(vulkan.status.code(), Some(2), "{}", stderr(&vulkan));
    assert_eq!(
        stderr(&vulkan),
        "error: the argument '--statistics-every' cannot be used with '--backend vulkan'; it is \
         available only with '--backend cpu'\n"
    );
    assert!(vulkan.stdout.is_empty());
    let help = modelspec(&["train", "--help"]);
    assert_eq!(help.status.code(), Some(0), "{}", stderr(&help));
    let help = String::from_utf8_lossy(&help.stdout);
    assert!(help.contains("--statistics-every <N>"), "{help}");
    assert!(
        help.contains("a Vulkan run sends every step's parameters list empty"),
        "{help}"
    );
}
