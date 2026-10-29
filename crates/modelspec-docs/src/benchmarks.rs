//! The benchmark history page, `website/docs/benchmarks.md`, and its charts under
//! `website/static/img/benchmarks/`, from `benchmarks/runs.yaml`.

use std::path::Path;

use serde_json::Value;

use crate::{Pages, Result};

/// The benchmark history, relative to the repository root.
pub const RUNS: &str = "benchmarks/runs.yaml";
/// The JSON Schema every run is validated against, relative to the repository root.
pub const SCHEMA: &str =
    "crates/modelspec-observatory/schema/entities/modelspec.benchmark.BenchmarkRun.schema.json";
/// The page's file name under [`crate::DOCS`].
pub const PAGE: &str = "benchmarks.md";
/// Where the charts land, relative to the repository root.
pub const IMAGES: &str = "website/static/img/benchmarks";

/// One validated run: the fields of `modelspec.benchmark.BenchmarkRun` the page reads.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Run {
    pub run_id: String,
    pub date: String,
    pub workload: String,
    pub side: String,
    pub threads: Option<u64>,
    pub modelspec_version: Option<String>,
    pub commit: Option<String>,
    pub opponent_version: Option<String>,
    pub mode: String,
    pub batch: Option<u64>,
    pub hidden: Option<u64>,
    pub tokens_per_step: Option<u64>,
    pub flop_per_step: Option<u64>,
    pub flop_rule: Option<String>,
    pub timed_steps: Option<u64>,
    pub repeat_count: u64,
    pub repeats_us_per_step: Option<Vec<f64>>,
    pub median_us_per_step: f64,
    pub target_loss: Option<f64>,
    pub steps_to_target: Option<u64>,
    pub seconds_to_target: Option<f64>,
    pub train_seconds: Option<f64>,
    pub eval_every: Option<u64>,
    pub final_heldout_loss: Option<f64>,
    pub source: String,
}

/// The first line of each entry of the top-level `runs` list, from 1: a line that starts with
/// `  - `.
fn entry_lines(text: &str) -> Vec<usize> {
    text.lines()
        .enumerate()
        .filter(|(_, line)| line.starts_with("  - "))
        .map(|(n, _)| n + 1)
        .collect()
}

/// The line, from 1, of `field` in the entry whose lines are `start..end`, or `start` when the
/// entry does not write it.
fn field_line(text: &str, start: usize, end: usize, field: &str) -> usize {
    let key = format!("{field}:");
    text.lines()
        .enumerate()
        .skip(start - 1)
        .take(end - start)
        .find(|(n, line)| {
            let body = if *n + 1 == start {
                line.strip_prefix("  - ")
            } else {
                line.strip_prefix("    ")
            };
            body.is_some_and(|body| body.starts_with(&key))
        })
        .map_or(start, |(n, _)| n + 1)
}

/// The `run_id` an entry whose lines are `start..end` writes, read from the text so that a
/// refusal before the entry is parsed can still name it; `?` when it writes none.
fn text_id(text: &str, start: usize, end: usize) -> String {
    let line = field_line(text, start, end, "run_id");
    text.lines()
        .nth(line - 1)
        .and_then(|line| line.split_once("run_id:"))
        .map(|(_, value)| value.trim().trim_matches('"').to_owned())
        .filter(|id| !id.is_empty())
        .unwrap_or_else(|| "?".to_owned())
}

/// The keys an entry whose lines are `start..end` writes more than once, each with the line of
/// its second writing and of its first. A YAML map keeps the last of them, so a figure written
/// twice would silently replace the first; the reader refuses it instead.
fn repeated_keys(text: &str, start: usize, end: usize) -> Vec<(String, usize, usize)> {
    let mut seen = std::collections::BTreeMap::new();
    let mut repeated = Vec::new();
    for (n, line) in text.lines().enumerate().skip(start - 1).take(end - start) {
        let body = if n + 1 == start {
            line.strip_prefix("  - ")
        } else {
            line.strip_prefix("    ")
        };
        let Some((key, _)) = body.and_then(|body| body.split_once(':')) else {
            continue;
        };
        let is_key = !key.is_empty()
            && key
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
        if !is_key {
            continue;
        }
        if let Some(first) = seen.insert(key.to_owned(), n + 1) {
            repeated.push((key.to_owned(), n + 1, first));
        }
    }
    repeated
}

/// The median of `values`: the middle one in ascending order, or the mean of the middle two;
/// none for no values.
fn median(values: &[f64]) -> Option<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    match n {
        0 => None,
        _ if n % 2 == 1 => Some(sorted[n / 2]),
        _ => Some((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0),
    }
}

/// The fields of `modelspec.benchmark.BenchmarkRun` typed `PositiveInteger`.
const POSITIVE_INTEGERS: [&str; 9] = [
    "threads",
    "batch",
    "hidden",
    "tokens_per_step",
    "flop_per_step",
    "timed_steps",
    "repeat_count",
    "steps_to_target",
    "eval_every",
];

/// Whether `value` is a YAML number that is not finite (`.nan`, `.inf`, `-.inf`). JSON has no
/// such number: converted, it would become null and read as "not recorded".
fn non_finite(value: &serde_yaml_ng::Value) -> bool {
    match value {
        serde_yaml_ng::Value::Number(number) => number.as_f64().is_some_and(|f| !f.is_finite()),
        serde_yaml_ng::Value::Sequence(items) => items.iter().any(non_finite),
        serde_yaml_ng::Value::Mapping(map) => map.values().any(non_finite),
        serde_yaml_ng::Value::Tagged(tagged) => non_finite(&tagged.value),
        _ => false,
    }
}

fn is_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 10
        && bytes.iter().enumerate().all(|(i, b)| match i {
            4 | 7 => *b == b'-',
            _ => b.is_ascii_digit(),
        })
}

/// The field of an entry that a schema error is about.
fn schema_field(error: &jsonschema::ValidationError<'_>) -> String {
    match error.kind() {
        jsonschema::error::ValidationErrorKind::Required { property } => {
            property.as_str().unwrap_or_default().to_owned()
        }
        jsonschema::error::ValidationErrorKind::AdditionalProperties { unexpected } => {
            unexpected.first().cloned().unwrap_or_default()
        }
        _ => error
            .instance_path()
            .iter()
            .next()
            .map(|segment| segment.to_string())
            .unwrap_or_default(),
    }
}

/// The MLP sweep's workload, whose `MatmulOnly` count is `3196·B·H`.
const SWEEP_WORKLOAD: &str = "mlp-784-H-10";
/// The transformer-block workload, whose `MatmulOnly` count is [`BLOCK_FLOP_PER_SEQUENCE`]·B.
const BLOCK: &str = "transformer-block";
/// The sequence length of a `transformer-block` step.
const BLOCK_SEQ: u64 = 128;
/// The `MatmulOnly` FLOP of a `transformer-block` step per sequence: `S·(42·H² + 24·H·F + 24·S·H)`
/// with `S = H = 128` and `F = 512`, the closed form of
/// `crates/modelspec-runtime/benches/transformer_block/block.rs`.
const BLOCK_FLOP_PER_SEQUENCE: u64 = 339_738_624;
/// The BERT-tiny pre-training workload (story:bert-tiny-vs-pytorch-cuda), whose `MatmulOnly`
/// count is [`BERT_FLOP_PER_SEQUENCE`]·B and whose runs are timed to a held-out loss.
const BERT: &str = "bert-tiny-pretrain";
/// The sequences of one `bert-tiny-pretrain` step, the batch of `examples/bert_tiny.mspec`.
const BERT_BATCH: u64 = 32;
/// The tokens of one `bert-tiny-pretrain` step, [`BERT_BATCH`] sequences of 128.
const BERT_TOKENS: u64 = BERT_BATCH * 128;
/// The `MatmulOnly` FLOP of a `bert-tiny-pretrain` step per sequence:
/// `S·(54·H² + 24·H·F + 6·H·V + 24·S·H)` with `S = H = 128`, `F = 512` and `V = 8,192`, the
/// closed form of `crates/modelspec-runtime/benches/bert_tiny/flop.rs`.
const BERT_FLOP_PER_SEQUENCE: u64 = 1_170_210_816;

/// What is wrong with the `MatmulOnly` count `flop` of a `workload` run counted as `per_sequence`
/// FLOP per sequence of its batch.
fn per_sequence_problem(run: &Run, flop: u64, workload: &str, per_sequence: u64) -> Option<String> {
    match run.batch {
        Some(batch) => {
            let expected = per_sequence.checked_mul(batch);
            (expected != Some(flop)).then(|| {
                format!(
                    "{flop} is not what flop_rule MatmulOnly counts for `{workload}`, \
                     {per_sequence}·B = {} at batch {batch}",
                    expected.map_or("more than 2^64".to_owned(), |n| n.to_string())
                )
            })
        }
        None => Some(format!(
            "flop_rule MatmulOnly counts {per_sequence}·B for `{workload}`, so the run must give batch"
        )),
    }
}

/// What is wrong with a `MatmulOnly` count of `flop` FLOP per step: the rule is a formula per
/// workload (`modelspec.benchmark.FlopRule`), `3196·B·H` for `mlp-784-H-10`, `339,738,624·B` for
/// `transformer-block` and `1,170,210,816·B` for `bert-tiny-pretrain`, and is defined for no other
/// workload.
fn matmul_only_problems(run: &Run, flop: u64) -> Option<(&'static str, String)> {
    let problem = match run.workload.as_str() {
        SWEEP_WORKLOAD => match (run.batch, run.hidden) {
            (Some(batch), Some(hidden)) => {
                let expected = 3196u64
                    .checked_mul(batch)
                    .and_then(|n| n.checked_mul(hidden));
                (expected != Some(flop)).then(|| {
                    format!(
                        "{flop} is not what flop_rule MatmulOnly counts, 3196·B·H = {} at batch {batch}, hidden {hidden}",
                        expected.map_or("more than 2^64".to_owned(), |n| n.to_string())
                    )
                })
            }
            _ => Some(
                "flop_rule MatmulOnly counts 3196·B·H, so the run must give batch and hidden"
                    .to_owned(),
            ),
        },
        BLOCK => per_sequence_problem(run, flop, BLOCK, BLOCK_FLOP_PER_SEQUENCE),
        BERT => per_sequence_problem(run, flop, BERT, BERT_FLOP_PER_SEQUENCE),
        other => Some(format!(
            "flop_rule MatmulOnly is a formula of `{SWEEP_WORKLOAD}`, `{BLOCK}` and `{BERT}`, not of `{other}`"
        )),
    };
    problem.map(|problem| ("flop_per_step", problem))
}

/// What is wrong with a `bert-tiny-pretrain` run (story:bert-tiny-vs-pytorch-cuda) beyond the
/// schema: it states its target_loss, steps_to_target and seconds_to_target, the figures its
/// section is about, and the 4,096 tokens of a step of 32 sequences of 128; its losses and
/// seconds are positive, and its seconds to the target are within its training seconds; it is
/// counted by `MatmulOnly`, the one count all its sides share (ADV-BP-4), and its steps to target
/// is a step it evaluated, a multiple of its `eval_every` (ADV-BP-5).
fn bert_problems(run: &Run) -> Vec<(&'static str, String)> {
    let mut problems = Vec::new();
    let missing = [
        (
            "target_loss",
            run.target_loss.is_none(),
            "the held-out loss it trains to",
        ),
        (
            "steps_to_target",
            run.steps_to_target.is_none(),
            "the first evaluated step at or below its target",
        ),
        (
            "seconds_to_target",
            run.seconds_to_target.is_none(),
            "the training seconds to that step",
        ),
        (
            "tokens_per_step",
            run.tokens_per_step.is_none(),
            "the 4096 tokens of one step",
        ),
    ];
    for (field, absent, what) in missing {
        if absent {
            problems.push((
                field,
                format!("is missing; a `{BERT}` run must give {field}, {what}"),
            ));
        }
    }
    if let Some(tokens) = run.tokens_per_step.filter(|tokens| *tokens != BERT_TOKENS) {
        problems.push((
            "tokens_per_step",
            format!(
                "{tokens} is not the {BERT_TOKENS} tokens of a `{BERT}` step, {BERT_BATCH} sequences of 128"
            ),
        ));
    }
    if let Some(batch) = run.batch.filter(|batch| *batch != BERT_BATCH) {
        problems.push((
            "batch",
            format!("{batch} is not the {BERT_BATCH} sequences of a `{BERT}` step"),
        ));
    }
    for (field, value) in [
        ("target_loss", run.target_loss),
        ("seconds_to_target", run.seconds_to_target),
        ("train_seconds", run.train_seconds),
        ("final_heldout_loss", run.final_heldout_loss),
    ] {
        if value.is_some_and(|value| !(value.is_finite() && value > 0.0)) {
            problems.push((field, "must be a positive number".to_owned()));
        }
    }
    if let (Some(to_target), Some(total)) = (run.seconds_to_target, run.train_seconds)
        && to_target > total
    {
        problems.push((
            "seconds_to_target",
            format!("{to_target} is more than the run's train_seconds {total}"),
        ));
    }
    for (field, absent, what) in [
        (
            "flop_rule",
            run.flop_rule.is_none(),
            "MatmulOnly, the count all its sides share",
        ),
        (
            "flop_per_step",
            run.flop_per_step.is_none(),
            "the MatmulOnly FLOP of one step",
        ),
        (
            "eval_every",
            run.eval_every.is_none(),
            "the steps between its held-out evaluations",
        ),
    ] {
        if absent {
            problems.push((
                field,
                format!("is missing; a `{BERT}` run must give {field}, {what}"),
            ));
        }
    }
    if let Some(rule) = run
        .flop_rule
        .as_deref()
        .filter(|rule| *rule != "MatmulOnly")
    {
        problems.push((
            "flop_rule",
            format!(
                "`{rule}` is not the `MatmulOnly` rule of a `{BERT}` run, \
                 {BERT_FLOP_PER_SEQUENCE}·B, the count all its sides share"
            ),
        ));
    }
    if let (Some(steps), Some(every)) = (run.steps_to_target, run.eval_every)
        && every > 0
        && !steps.is_multiple_of(every)
    {
        problems.push((
            "steps_to_target",
            format!("{steps} is no step the run evaluated, a multiple of its eval_every {every}"),
        ));
    }
    problems
}

/// What is wrong with the batch and tokens of a `transformer-block` run (coordinator decision
/// D84): the run gives both, since its section is drawn against batch with a tokens-per-second
/// column, and its tokens are `128·B`, a whole number of sequences of 128 whether or not it gives
/// its batch.
fn block_problems(run: &Run) -> Vec<(&'static str, String)> {
    let mut problems = Vec::new();
    if run.batch.is_none() {
        problems.push((
            "batch",
            format!(
                "is missing; a `{BLOCK}` run must give batch, the sequences of {BLOCK_SEQ} tokens of one step"
            ),
        ));
    }
    match (run.tokens_per_step, run.batch) {
        (None, _) => problems.push((
            "tokens_per_step",
            format!(
                "is missing; a `{BLOCK}` run must give tokens_per_step, the {BLOCK_SEQ}·B tokens of one step"
            ),
        )),
        (Some(tokens), Some(batch)) if batch.checked_mul(BLOCK_SEQ) != Some(tokens) => {
            problems.push((
                "tokens_per_step",
                format!(
                    "{tokens} is not the tokens of a `{BLOCK}` step at batch {batch}, {batch}·{BLOCK_SEQ}"
                ),
            ));
        }
        (Some(tokens), None) if !tokens.is_multiple_of(BLOCK_SEQ) => problems.push((
            "tokens_per_step",
            format!(
                "{tokens} is no whole number of `{BLOCK}` sequences of {BLOCK_SEQ} tokens, so it is no {BLOCK_SEQ}·B"
            ),
        )),
        _ => {}
    }
    problems
}

/// What is wrong with a schema-valid run beyond the schema: the invariants of
/// `modelspec.benchmark.BenchmarkRun` (spec/domains/benchmark.yaml), checked here too so that they
/// hold whatever schema is read, its date, its repeats against its repeat count and its median
/// against the median of its repeats; a `MatmulOnly` count against its workload's formula
/// ([`matmul_only_problems`]); the batch and tokens of a `transformer-block` run
/// ([`block_problems`]); the target figures, batch and tokens of a `bert-tiny-pretrain` run
/// ([`bert_problems`]); and its workload against the workloads the page declares
/// ([`WORKLOADS`]).
fn semantic_problems(run: &Run) -> Vec<(&'static str, String)> {
    let mut problems = Vec::new();
    if !is_date(&run.date) {
        problems.push(("date", format!("`{}` is not a `YYYY-MM-DD` date", run.date)));
    }
    if !WORKLOADS.iter().any(|declared| declared.id == run.workload) {
        let declared: Vec<String> = WORKLOADS
            .iter()
            .map(|declared| format!("`{}`", declared.id))
            .collect();
        problems.push((
            "workload",
            format!(
                "`{}` is not a declared workload; the page declares {}",
                run.workload,
                declared.join(", ")
            ),
        ));
    }
    if let (Some(flop), Some("MatmulOnly")) = (run.flop_per_step, run.flop_rule.as_deref()) {
        problems.extend(matmul_only_problems(run, flop));
    }
    if run.workload == BLOCK {
        problems.extend(block_problems(run));
    }
    if run.workload == BERT {
        problems.extend(bert_problems(run));
    }
    let counts = [
        ("threads", run.threads),
        ("batch", run.batch),
        ("hidden", run.hidden),
        ("tokens_per_step", run.tokens_per_step),
        ("flop_per_step", run.flop_per_step),
        ("timed_steps", run.timed_steps),
        ("repeat_count", Some(run.repeat_count)),
        ("steps_to_target", run.steps_to_target),
        ("eval_every", run.eval_every),
    ];
    for (field, value) in counts {
        if value == Some(0) {
            problems.push((field, "must be at least 1".to_owned()));
        }
    }
    match (run.flop_per_step, &run.flop_rule) {
        (Some(_), None) => problems.push((
            "flop_per_step",
            "is given without flop_rule; give both or neither".to_owned(),
        )),
        (None, Some(_)) => problems.push((
            "flop_rule",
            "is given without flop_per_step; give both or neither".to_owned(),
        )),
        _ => {}
    }
    if let Some(repeats) = &run.repeats_us_per_step {
        if repeats.len() as u64 != run.repeat_count {
            problems.push((
                "repeats_us_per_step",
                format!(
                    "{} repeats listed, but repeat_count is {}",
                    repeats.len(),
                    run.repeat_count
                ),
            ));
        } else if repeats.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
            problems.push((
                "repeats_us_per_step",
                "every repeat must be a positive number".to_owned(),
            ));
        } else if let Some(expected) = median(repeats)
            && (run.median_us_per_step - expected).abs() > 1e-9 * expected.abs()
        {
            problems.push((
                "median_us_per_step",
                format!(
                    "{} is not the median of its repeats; the median of its repeats is {expected}",
                    run.median_us_per_step
                ),
            ));
        }
    }
    if !(run.median_us_per_step.is_finite() && run.median_us_per_step > 0.0) {
        problems.push(("median_us_per_step", "must be a positive number".to_owned()));
    }
    problems
}

/// The runs of `text`, the contents of [`RUNS`], each validated against `schema`, the JSON Schema
/// of `modelspec.benchmark.BenchmarkRun`. Every refusal names the file, the line and the field.
pub fn read(text: &str, schema: &Value) -> Result<Vec<Run>> {
    let lines = entry_lines(text);
    let total_lines = text.lines().count() + 1;
    let span = |index: usize| {
        (
            lines[index],
            lines.get(index + 1).copied().unwrap_or(total_lines),
        )
    };
    let mut problems = Vec::new();
    for index in 0..lines.len() {
        let (start, end) = span(index);
        for (key, line, first) in repeated_keys(text, start, end) {
            problems.push(format!(
                "{RUNS}:{line}: runs[{index}] ({}) field `{key}`: written twice in one entry, first at line {first}; a YAML map would keep only the last",
                text_id(text, start, end)
            ));
        }
    }
    if !problems.is_empty() {
        return Err(problems.join("\n"));
    }
    let document: serde_yaml_ng::Value = serde_yaml_ng::from_str(text).map_err(|e| {
        let line = e.location().map_or(1, |location| location.line());
        format!("{RUNS}:{line}: {e}")
    })?;
    let Some(entries) = document
        .get("runs")
        .and_then(serde_yaml_ng::Value::as_sequence)
    else {
        return Err(format!(
            "{RUNS}:1: the file must be a map with a `runs` list"
        ));
    };
    if let Some(key) = document
        .as_mapping()
        .and_then(|map| map.keys().find(|key| key.as_str() != Some("runs")))
    {
        return Err(format!(
            "{RUNS}:1: unexpected top-level key `{}`; the file holds `runs` only",
            serde_yaml_ng::to_string(key).unwrap_or_default().trim_end()
        ));
    }
    if lines.len() != entries.len() {
        return Err(format!(
            "{RUNS}: {} runs, but {} lines start an entry with `  - `; write each run as a `  - ` item of `runs`",
            entries.len(),
            lines.len()
        ));
    }
    let validator = jsonschema::validator_for(schema)
        .map_err(|e| format!("{SCHEMA}: the schema does not compile: {e}"))?;
    let mut runs = Vec::new();
    let mut ids = std::collections::BTreeSet::new();
    for (index, entry) in entries.iter().enumerate() {
        let (start, end) = span(index);
        let id = text_id(text, start, end);
        let mut refuse = |field: &str, message: String| {
            problems.push(format!(
                "{RUNS}:{}: runs[{index}] ({id}) field `{field}`: {message}",
                field_line(text, start, end, field)
            ));
        };
        let infinite: Vec<_> = entry
            .as_mapping()
            .into_iter()
            .flatten()
            .filter(|(_, value)| non_finite(value))
            .filter_map(|(key, _)| key.as_str())
            .collect();
        if !infinite.is_empty() {
            for field in infinite {
                refuse(
                    field,
                    "holds `.nan` or an infinity, which is not a measured µs per step".to_owned(),
                );
            }
            continue;
        }
        let entry: Value = match serde_json::to_value(entry) {
            Ok(entry) => entry,
            Err(e) => {
                refuse("?", e.to_string());
                continue;
            }
        };
        let errors: Vec<_> = validator.iter_errors(&entry).collect();
        if !errors.is_empty() {
            for error in errors {
                refuse(&schema_field(&error), error.to_string());
            }
            continue;
        }
        // The schema's `"type": "integer"` admits a whole float such as `1.0`, which the run's
        // integer fields do not hold; refuse it by field rather than leave it to serde, whose
        // error names no field.
        let fractional: Vec<_> = POSITIVE_INTEGERS
            .into_iter()
            .filter(|field| {
                entry
                    .get(field)
                    .is_some_and(|value| !value.is_null() && value.as_u64().is_none())
            })
            .collect();
        if !fractional.is_empty() {
            for field in fractional {
                refuse(
                    field,
                    format!("`{}` is not a whole number of at least 1", entry[field]),
                );
            }
            continue;
        }
        let run: Run = match serde_json::from_value(entry) {
            Ok(run) => run,
            Err(e) => {
                refuse("?", e.to_string());
                continue;
            }
        };
        if !ids.insert(run.run_id.clone()) {
            refuse(
                "run_id",
                format!("`{}` names an earlier run too", run.run_id),
            );
        }
        for (field, message) in semantic_problems(&run) {
            refuse(field, message);
        }
        runs.push(run);
    }
    if problems.is_empty() {
        Ok(runs)
    } else {
        Err(problems.join("\n"))
    }
}

/// How a side is named on the page, and whether it is ModelSpec's; in the order the page lists
/// sides.
const SIDES: [(&str, &str, bool); 6] = [
    ("ModelSpecVulkan", "ModelSpec Vulkan", true),
    ("ModelSpecCpu", "ModelSpec CPU", true),
    ("PyTorchCudaEager", "PyTorch CUDA eager", false),
    ("PyTorchCudaCompile", "PyTorch CUDA torch.compile", false),
    ("PyTorchCpuEager", "PyTorch CPU eager", false),
    ("PyTorchCpuCompile", "PyTorch CPU torch.compile", false),
];

/// Series colours, ModelSpec's and the opponents', each taken in order.
const MODELSPEC_COLOURS: [&str; 4] = ["#1d4ed8", "#059669", "#7c3aed", "#0891b2"];
const OPPONENT_COLOURS: [&str; 10] = [
    "#dc2626", "#ea580c", "#a16207", "#db2777", "#4b5563", "#92400e", "#be123c", "#ca8a04",
    "#9ca3af", "#f97316",
];

fn side_rank(side: &str) -> usize {
    SIDES
        .iter()
        .position(|(name, _, _)| *name == side)
        .unwrap_or(SIDES.len())
}

fn side_label(side: &str) -> &str {
    SIDES
        .iter()
        .find(|(name, _, _)| *name == side)
        .map_or(side, |(_, label, _)| label)
}

fn is_modelspec(side: &str) -> bool {
    SIDES.iter().any(|(name, _, ours)| *name == side && *ours)
}

fn threads_label(threads: Option<u64>) -> String {
    match threads {
        Some(1) => "1 thread".to_owned(),
        Some(n) => format!("{n} threads"),
        None => String::new(),
    }
}

/// `value` with at most 3 decimals, trailing zeros dropped, and thousands separated by commas.
fn number(value: f64) -> String {
    let text = format!("{value:.3}");
    let text = if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.')
    } else {
        &text
    };
    grouped(text)
}

/// GFLOP/s with 1 decimal and thousands separated by commas; blank for none.
fn gflops_text(value: Option<f64>) -> String {
    value.map_or(String::new(), |value| grouped(&format!("{value:.1}")))
}

/// `text`, a non-negative decimal number, with the thousands of its whole part separated by
/// commas.
fn grouped(text: &str) -> String {
    let (whole, fraction) = text
        .split_once('.')
        .map_or((text, None), |(w, f)| (w, Some(f)));
    let digits: Vec<char> = whole.chars().collect();
    let mut grouped = String::new();
    for (i, digit) in digits.iter().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(*digit);
    }
    match fraction {
        Some(fraction) => format!("{grouped}.{fraction}"),
        None => grouped,
    }
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// One plotted point: its position on the x axis, its µs per step, and a label drawn beside it.
struct Point {
    x: f64,
    us: f64,
    label: Option<String>,
}

struct Series {
    label: String,
    modelspec: bool,
    points: Vec<Point>,
}

/// A chart of µs per step on a log scale against categories on the x axis, at 0, 1, 2, ..., or,
/// with `log_x`, against a quantity on a log scale whose decades are the ticks.
struct Chart {
    title: String,
    x_label: String,
    ticks: Vec<String>,
    log_x: bool,
    series: Vec<Series>,
    /// Whether an opponent's last point is carried to the right edge as a dotted line, the
    /// figure ModelSpec is measured against until the opponent is measured again.
    carry_forward: bool,
}

/// The width a chart is drawn at, in CSS pixels. A phone shows the content column at about
/// 358 px, so a chart is scaled by no less than 358 / 440 there and its text of [`FONT`] px stays
/// above 12 px; a wide screen shows it at this size.
const WIDTH: f64 = 440.0;
const LEFT: f64 = 88.0;
const RIGHT: f64 = 16.0;
/// The top of the plot, below the title's two lines.
const TOP: f64 = 62.0;
const PLOT_HEIGHT: f64 = 300.0;
const LEGEND_ROW: f64 = 24.0;
/// The size of every label and tick; the title is [`TITLE_FONT`].
const FONT: f64 = 15.0;
const TITLE_FONT: f64 = 17.0;
/// About the width of one character at [`FONT`], to lay labels out without a font.
const CHAR_WIDTH: f64 = 0.56 * FONT;

/// The chart's own colours in the light theme: title and legend, ticks, major and minor grid
/// lines, the plot frame, and the inside of an opponent's open square, which is the light page
/// canvas. A chart has no background of its own, so it sits on the page in either theme.
const INK: &str = "#111827";
const MUTED: &str = "#374151";
const GRID: &str = "#d1d5db";
const MINOR: &str = "#e5e7eb";
const FRAME: &str = "#8b95a1";
const HOLLOW: &str = "#f4f2eb";

/// The dark-theme colour of each of the chart's own colours. The dark page canvas is `#0c1924`.
const DARK: [(&str, &str); 6] = [
    (INK, "#f4f5ed"),
    (MUTED, "#c3ccd0"),
    (GRID, "#3a4b57"),
    (MINOR, "#22333f"),
    (FRAME, "#6b7c86"),
    (HOLLOW, "#0c1924"),
];
/// The dark-theme series colours, one for each of [`MODELSPEC_COLOURS`] and of
/// [`OPPONENT_COLOURS`] in their order.
const MODELSPEC_DARK: [&str; 4] = ["#60a5fa", "#34d399", "#a78bfa", "#22d3ee"];
const OPPONENT_DARK: [&str; 10] = [
    "#f87171", "#fb923c", "#facc15", "#f472b6", "#cbd5e1", "#d6a35c", "#fb7185", "#fde047",
    "#e5e7eb", "#fdba74",
];

/// The chart's style: under the dark theme every light colour becomes its dark one. A chart is an
/// `<img>`, and the browser evaluates `prefers-color-scheme` inside it against the
/// `color-scheme` of the page, which the site sets from its theme toggle.
fn style() -> String {
    let rules: String = DARK
        .iter()
        .copied()
        .chain(MODELSPEC_COLOURS.into_iter().zip(MODELSPEC_DARK))
        .chain(OPPONENT_COLOURS.into_iter().zip(OPPONENT_DARK))
        .map(|(light, dark)| {
            format!("[fill=\"{light}\"]{{fill:{dark}}}[stroke=\"{light}\"]{{stroke:{dark}}}")
        })
        .collect();
    format!("<style>@media (prefers-color-scheme: dark){{{rules}}}</style>\n")
}

fn marker(out: &mut String, x: f64, y: f64, colour: &str, modelspec: bool) {
    if modelspec {
        out.push_str(&format!(
            "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"4.5\" fill=\"{colour}\"/>\n"
        ));
    } else {
        out.push_str(&format!(
            "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"8\" height=\"8\" fill=\"{HOLLOW}\" stroke=\"{colour}\" stroke-width=\"2\"/>\n",
            x - 4.0,
            y - 4.0
        ));
    }
}

/// `10` raised to `decade`, written with a superscript exponent: `10⁷`.
fn power_of_ten(decade: i32) -> String {
    const SUPERSCRIPTS: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    let exponent: String = decade
        .to_string()
        .chars()
        .map(|c| match c.to_digit(10) {
            Some(digit) => SUPERSCRIPTS[digit as usize],
            None => '⁻',
        })
        .collect();
    format!("10{exponent}")
}

/// The title on two lines, split after its first `": "`.
fn title_lines(title: &str) -> (String, String) {
    match title.split_once(": ") {
        Some((head, tail)) => (format!("{head}:"), tail.to_owned()),
        None => (title.to_owned(), String::new()),
    }
}

/// The tick labels and the axis label: dates of one year that do not fit their slot are written
/// `MM-DD`, and the year moves into the axis label.
fn ticks_and_axis(chart: &Chart, slot: f64) -> (Vec<String>, String) {
    let too_wide = chart
        .ticks
        .iter()
        .any(|tick| tick.chars().count() as f64 * CHAR_WIDTH > slot - 6.0);
    let years: std::collections::BTreeSet<&str> = chart
        .ticks
        .iter()
        .map(|tick| &tick[..4.min(tick.len())])
        .collect();
    if too_wide && years.len() == 1 && chart.ticks.iter().all(|tick| is_date(tick)) {
        let year = years.into_iter().next().unwrap_or_default();
        (
            chart
                .ticks
                .iter()
                .map(|tick| tick[5..].to_owned())
                .collect(),
            format!("{}, {year}", chart.x_label),
        )
    } else {
        (chart.ticks.clone(), chart.x_label.clone())
    }
}

/// A label beside a point: where the point is, its text and its colour.
struct Label<'a> {
    x: f64,
    y: f64,
    text: &'a str,
    colour: &'a str,
}

/// The labels as `<text>` elements. Labels with the same text whose points lie within a few
/// pixels of each other across (one release at one date, in several series) are drawn once, above
/// the highest point, in the tick colour. Each label takes the first place beside its point, of
/// above right, below right, above left and below left, that stays inside the plot and clear of
/// the labels placed before it.
fn labels(mut found: Vec<Label<'_>>, plot: (f64, f64, f64, f64)) -> String {
    found.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    let mut merged: Vec<Label<'_>> = Vec::new();
    for label in found {
        match merged
            .iter_mut()
            .find(|kept| kept.text == label.text && (kept.x - label.x).abs() < 10.0)
        {
            Some(kept) => {
                if kept.colour != label.colour {
                    kept.colour = MUTED;
                }
                if label.y < kept.y {
                    kept.y = label.y;
                }
            }
            None => merged.push(label),
        }
    }
    let (left, top, right, bottom) = plot;
    let mut placed: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut out = String::new();
    for label in merged {
        let width = label.text.chars().count() as f64 * CHAR_WIDTH;
        let candidates = [
            (label.x + 7.0, label.y - 7.0, false),
            (label.x + 7.0, label.y + 7.0 + FONT, false),
            (label.x - 7.0, label.y - 7.0, true),
            (label.x - 7.0, label.y + 7.0 + FONT, true),
        ];
        let boxed = |(x, baseline, end): (f64, f64, bool)| {
            let x0 = if end { x - width } else { x };
            (x0, baseline - FONT, x0 + width, baseline + 3.0)
        };
        let clear = |b: (f64, f64, f64, f64)| {
            b.0 >= left
                && b.2 <= right
                && b.1 >= top
                && b.3 <= bottom
                && placed
                    .iter()
                    .all(|p| b.2 <= p.0 || p.2 <= b.0 || b.3 <= p.1 || p.3 <= b.1)
        };
        let choice = candidates
            .iter()
            .copied()
            .find(|candidate| clear(boxed(*candidate)))
            .unwrap_or(candidates[0]);
        placed.push(boxed(choice));
        let (x, baseline, end) = choice;
        out.push_str(&format!(
            "<text x=\"{x:.1}\" y=\"{baseline:.1}\"{} fill=\"{}\">{}</text>\n",
            if end { " text-anchor=\"end\"" } else { "" },
            label.colour,
            escape(label.text)
        ));
    }
    out
}

/// The chart as a standalone SVG document.
fn svg(chart: &Chart) -> String {
    let plot_width = WIDTH - LEFT - RIGHT;
    let values: Vec<f64> = chart
        .series
        .iter()
        .flat_map(|series| series.points.iter().map(|point| point.us))
        .collect();
    let low = values.iter().copied().fold(f64::INFINITY, f64::min);
    let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let low = low.log10().floor() as i32;
    let mut high = high.log10().ceil() as i32;
    if high <= low {
        high = low + 1;
    }
    let y_of = |us: f64| TOP + PLOT_HEIGHT * (f64::from(high) - us.log10()) / f64::from(high - low);
    let decades = chart.log_x.then(|| {
        let xs = chart
            .series
            .iter()
            .flat_map(|series| series.points.iter().map(|point| point.x));
        let low = xs.clone().fold(f64::INFINITY, f64::min).log10().floor() as i32;
        let high = xs.fold(f64::NEG_INFINITY, f64::max).log10().ceil() as i32;
        (low, high.max(low + 1))
    });
    let slots = match decades {
        Some((low, high)) => f64::from(high - low + 1),
        None => chart.ticks.len().max(1) as f64,
    };
    let x_of = |x: f64| match decades {
        Some((low, high)) => {
            LEFT + plot_width * (x.log10() - f64::from(low)) / f64::from(high - low)
        }
        None => LEFT + plot_width * (x + 0.5) / slots,
    };
    let axis = TOP + PLOT_HEIGHT;
    let legend = axis + 76.0;
    let height = legend + chart.series.len() as f64 * LEGEND_ROW;
    let (title, subtitle) = title_lines(&chart.title);
    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{WIDTH}\" height=\"{height}\" viewBox=\"0 0 {WIDTH} {height}\" font-family=\"system-ui, -apple-system, Segoe UI, Roboto, sans-serif\" font-size=\"{FONT}\" role=\"img\" aria-label=\"{}\">\n<title>{}</title>\n{}<text x=\"16\" y=\"24\" font-size=\"{TITLE_FONT}\" font-weight=\"600\" fill=\"{INK}\">{}</text>\n<text x=\"16\" y=\"46\" font-size=\"{TITLE_FONT}\" font-weight=\"600\" fill=\"{INK}\">{}</text>\n",
        escape(&chart.title),
        escape(&chart.title),
        style(),
        escape(&title),
        escape(&subtitle),
    );
    for decade in low..=high {
        let base = 10f64.powi(decade);
        let y = y_of(base);
        out.push_str(&format!(
            "<line x1=\"{LEFT}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" stroke=\"{GRID}\"/>\n<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" fill=\"{MUTED}\">{}</text>\n",
            LEFT + plot_width,
            LEFT - 8.0,
            y + 5.0,
            number(base)
        ));
        if decade < high {
            for minor in [2.0, 5.0] {
                let y = y_of(base * minor);
                out.push_str(&format!(
                    "<line x1=\"{LEFT}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" stroke=\"{MINOR}\"/>\n",
                    LEFT + plot_width
                ));
            }
        }
    }
    if let Some((low, high)) = decades {
        for decade in low..=high {
            let base = 10f64.powi(decade);
            let minors: &[f64] = if decade < high {
                &[1.0, 2.0, 5.0]
            } else {
                &[1.0]
            };
            for minor in minors {
                let x = x_of(base * minor);
                out.push_str(&format!(
                    "<line x1=\"{x:.1}\" y1=\"{TOP}\" x2=\"{x:.1}\" y2=\"{:.1}\" stroke=\"{}\"/>\n",
                    TOP + PLOT_HEIGHT,
                    if *minor == 1.0 { GRID } else { MINOR }
                ));
            }
        }
    }
    out.push_str(&format!(
        "<rect x=\"{LEFT}\" y=\"{TOP}\" width=\"{plot_width:.1}\" height=\"{PLOT_HEIGHT}\" fill=\"none\" stroke=\"{FRAME}\"/>\n"
    ));
    let (ticks, x_label): (Vec<(f64, String)>, String) = match decades {
        Some((low, high)) => (
            (low..=high)
                .map(|decade| (x_of(10f64.powi(decade)), power_of_ten(decade)))
                .collect(),
            chart.x_label.clone(),
        ),
        None => {
            let (ticks, x_label) = ticks_and_axis(chart, plot_width / slots);
            (
                ticks
                    .into_iter()
                    .enumerate()
                    .map(|(i, tick)| (x_of(i as f64), tick))
                    .collect(),
                x_label,
            )
        }
    };
    for (x, tick) in &ticks {
        let x = *x;
        out.push_str(&format!(
            "<line x1=\"{x:.1}\" y1=\"{axis:.1}\" x2=\"{x:.1}\" y2=\"{:.1}\" stroke=\"{FRAME}\"/>\n<text x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\" fill=\"{MUTED}\">{}</text>\n",
            axis + 5.0,
            axis + 22.0,
            escape(tick)
        ));
    }
    out.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" fill=\"{INK}\">{}</text>\n<text transform=\"translate(18 {:.1}) rotate(-90)\" text-anchor=\"middle\" fill=\"{INK}\">µs per step, log scale</text>\n",
        LEFT + plot_width / 2.0,
        axis + 46.0,
        escape(&x_label),
        TOP + PLOT_HEIGHT / 2.0
    ));
    let (mut ours, mut theirs) = (0, 0);
    let colours: Vec<&str> = chart
        .series
        .iter()
        .map(|series| {
            if series.modelspec {
                ours += 1;
                MODELSPEC_COLOURS[(ours - 1) % MODELSPEC_COLOURS.len()]
            } else {
                theirs += 1;
                OPPONENT_COLOURS[(theirs - 1) % OPPONENT_COLOURS.len()]
            }
        })
        .collect();
    let mut found = Vec::new();
    for (series, colour) in chart.series.iter().zip(&colours) {
        let mut points: Vec<&Point> = series.points.iter().collect();
        points.sort_by(|a, b| a.x.total_cmp(&b.x));
        if points.len() > 1 {
            let path: Vec<String> = points
                .iter()
                .map(|point| format!("{:.1},{:.1}", x_of(point.x), y_of(point.us)))
                .collect();
            let (width, dash) = if series.modelspec {
                ("2.5", "")
            } else {
                ("1.8", " stroke-dasharray=\"6 4\"")
            };
            out.push_str(&format!(
                "<polyline points=\"{}\" fill=\"none\" stroke=\"{colour}\" stroke-width=\"{width}\"{dash}/>\n",
                path.join(" ")
            ));
        }
        if let (true, false, Some(last)) = (chart.carry_forward, series.modelspec, points.last()) {
            let (x, y) = (x_of(last.x), y_of(last.us));
            if x < LEFT + plot_width - 1.0 {
                out.push_str(&format!(
                    "<line x1=\"{x:.1}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" stroke=\"{colour}\" stroke-width=\"1.6\" stroke-dasharray=\"2 3\"/>\n",
                    LEFT + plot_width
                ));
            }
        }
        for point in &points {
            let (x, y) = (x_of(point.x), y_of(point.us));
            marker(&mut out, x, y, colour, series.modelspec);
            if let Some(label) = &point.label {
                found.push(Label {
                    x,
                    y,
                    text: label,
                    colour,
                });
            }
        }
    }
    out.push_str(&labels(found, (LEFT, TOP, LEFT + plot_width, axis)));
    for (i, (series, colour)) in chart.series.iter().zip(&colours).enumerate() {
        let y = legend + i as f64 * LEGEND_ROW;
        let dash = if series.modelspec {
            ""
        } else {
            " stroke-dasharray=\"6 4\""
        };
        out.push_str(&format!(
            "<line x1=\"16\" y1=\"{y:.1}\" x2=\"44\" y2=\"{y:.1}\" stroke=\"{colour}\" stroke-width=\"2\"{dash}/>\n"
        ));
        marker(&mut out, 30.0, y, colour, series.modelspec);
        out.push_str(&format!(
            "<text x=\"52\" y=\"{:.1}\" fill=\"{INK}\">{}</text>\n",
            y + 5.0,
            escape(&series.label)
        ));
    }
    out.push_str("</svg>\n");
    out
}

/// A workload the page declares: its `workload` in `runs.yaml`, the heading of its section, and
/// what the "Not measured" section says of it while it has no run.
struct Workload {
    id: &'static str,
    title: &'static str,
    not_measured: &'static str,
}

/// The declared workloads. One without a run in `runs.yaml` is listed under "Not measured"; its
/// first run gives it a section of its own, with no change here.
const WORKLOADS: [Workload; 4] = [
    Workload {
        id: "mnist-784-64-10",
        title: "MNIST 784-64-10",
        not_measured: "the MNIST benchmark: the 784 → 64 ReLU → 10 training step at batch 100 against PyTorch.",
    },
    Workload {
        id: "mlp-784-H-10",
        title: "MLP 784-H-10 sweep",
        not_measured: "the MLP size sweep: the 784 → H ReLU → 10 step over batch and hidden width against PyTorch CUDA.",
    },
    Workload {
        id: "transformer-block",
        title: "Transformer block",
        not_measured: "the transformer-block benchmark on synthetic data against PyTorch CUDA: the two BERT-tiny encoder blocks (hidden 128, 2 heads, sequence 128) on random `f32` inputs.",
    },
    Workload {
        id: "bert-tiny-pretrain",
        title: "BERT-tiny pre-training",
        not_measured: "BERT-tiny pre-training timed to the same held-out loss as PyTorch CUDA with the same corpus, batches, initial weights and optimizer.",
    },
];

/// The heading of a workload's section.
fn workload_title(workload: &str) -> String {
    WORKLOADS
        .iter()
        .find(|declared| declared.id == workload)
        .map_or(workload, |declared| declared.title)
        .to_owned()
}

/// The file-name stem of a workload's charts.
fn slug(workload: &str) -> String {
    workload
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

/// What tells the series of an over-date chart apart: side, threads, mode, and the hidden width
/// of a sweep.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SeriesKey {
    rank: usize,
    side: String,
    threads: Option<u64>,
    profiled: bool,
    hidden: Option<u64>,
}

impl SeriesKey {
    fn of(run: &Run, with_hidden: bool) -> Self {
        Self {
            rank: side_rank(&run.side),
            side: run.side.clone(),
            threads: run.threads,
            profiled: run.mode == "Profiled",
            hidden: if with_hidden { run.hidden } else { None },
        }
    }

    fn label(&self) -> String {
        let mut label = side_label(&self.side).to_owned();
        if self.threads.is_some() {
            label.push_str(&format!(", {}", threads_label(self.threads)));
        }
        if self.profiled {
            label.push_str(", profiled");
        }
        if let Some(hidden) = self.hidden {
            label.push_str(&format!(", H {hidden}"));
        }
        label
    }
}

/// µs per step over date: one slot per date, each series nudged a little apart, a series' points on one date spread across its slot
/// in file order, each ModelSpec point labelled with its release.
fn over_date(title: String, runs: &[&Run], with_hidden: bool) -> Chart {
    let dates: std::collections::BTreeSet<&str> =
        runs.iter().map(|run| run.date.as_str()).collect();
    let dates: Vec<&str> = dates.into_iter().collect();
    let mut grouped: std::collections::BTreeMap<SeriesKey, Vec<&Run>> = Default::default();
    for run in runs {
        grouped
            .entry(SeriesKey::of(run, with_hidden))
            .or_default()
            .push(run);
    }
    let count = grouped.len();
    let series = grouped
        .into_iter()
        .enumerate()
        .map(|(index, (key, runs))| {
            let nudge = (index as f64 - (count as f64 - 1.0) / 2.0) * 0.02;
            let mut points = Vec::new();
            for (slot, date) in dates.iter().enumerate() {
                let on_date: Vec<&&Run> = runs.iter().filter(|run| run.date == *date).collect();
                let n = on_date.len();
                for (k, run) in on_date.into_iter().enumerate() {
                    let offset = if n > 1 {
                        -0.3 + 0.6 * k as f64 / (n - 1) as f64
                    } else {
                        0.0
                    };
                    points.push(Point {
                        x: slot as f64 + offset + nudge,
                        us: run.median_us_per_step,
                        label: if is_modelspec(&run.side) {
                            run.modelspec_version.clone()
                        } else {
                            None
                        },
                    });
                }
            }
            Series {
                label: key.label(),
                modelspec: is_modelspec(&key.side),
                points,
            }
        })
        .collect();
    Chart {
        title,
        x_label: "date of the run".to_owned(),
        ticks: dates.iter().map(|date| (*date).to_owned()).collect(),
        series,
        carry_forward: true,
        log_x: false,
    }
}

/// Of each group of `runs` by `key`, the latest: the last of the latest date in file order.
fn latest<K: Ord>(runs: &[&Run], key: impl Fn(&Run) -> K) -> std::collections::BTreeMap<K, Run> {
    let mut out: std::collections::BTreeMap<K, Run> = Default::default();
    for run in runs {
        let k = key(run);
        if out.get(&k).is_none_or(|kept| kept.date <= run.date) {
            out.insert(k, (*run).clone());
        }
    }
    out
}

fn build(run: &Run) -> String {
    match (&run.modelspec_version, &run.commit) {
        (Some(version), Some(commit)) => {
            format!(
                "{version} (`{}`)",
                commit.chars().take(7).collect::<String>()
            )
        }
        (Some(version), None) => version.clone(),
        _ => run.opponent_version.clone().unwrap_or_default(),
    }
}

fn repeats(run: &Run) -> String {
    if run.repeats_us_per_step.is_some() {
        run.repeat_count.to_string()
    } else {
        format!("{}, not recorded", run.repeat_count)
    }
}

/// The section of a workload without batch or hidden parameters: its over-date chart and the
/// latest run per series.
fn fixed_section(workload: &str, runs: &[&Run], images: &mut Pages) -> String {
    let title = workload_title(workload);
    let name = format!("{}.svg", slug(workload));
    images.insert(
        name.clone(),
        svg(&over_date(
            format!("{title}: µs per step over date"),
            runs,
            false,
        )),
    );
    let mut out = format!(
        "## {title}\n\n`{workload}`: the median µs per step of each run over the date it ran, \
         ModelSpec solid with filled circles and the opponents dashed with open squares. A \
         side's runs on one date are spread across that date in the order of the file, and each \
         ModelSpec point carries its release. An opponent's last run is carried to the right \
         edge as a dotted line, the figure ModelSpec is measured against until that opponent \
         is measured again.\n\n![{title}: µs per step over date, log scale](/img/benchmarks/{name})\n\n\
         ### Latest median per side\n\n\
         | Side | Threads | Mode | Date | Build | µs per step | Repeats | Source |\n|---|---|---|---|---|---|---|---|\n"
    );
    for run in latest(runs, |run| SeriesKey::of(run, false)).values() {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | `{}` |\n",
            side_label(&run.side),
            threads_label(run.threads),
            run.mode,
            run.date,
            crate::cell(&build(run)),
            number(run.median_us_per_step),
            repeats(run),
            crate::cell(&run.source)
        ));
    }
    out
}

/// The section of a workload with batch and hidden parameters: the chart against FLOP per step of
/// every batch, then per batch a chart of µs per step against hidden width at the latest run of
/// each point, an over-date chart when the batch has runs on more than one date, and the latest
/// medians with their GFLOP/s.
fn sweep_section(workload: &str, runs: &[&Run], images: &mut Pages) -> String {
    let title = workload_title(workload);
    let mut out = format!(
        "## {title}\n\n`{workload}`: per batch, the median µs per step of the latest run at each \
         hidden width H, ModelSpec solid with filled circles and the opponents dashed with open \
         squares.\n"
    );
    out.push_str(&flop_section(workload, runs, images));
    let batches: std::collections::BTreeSet<Option<u64>> =
        runs.iter().map(|run| run.batch).collect();
    for batch in batches {
        let at_batch: Vec<&Run> = runs
            .iter()
            .copied()
            .filter(|run| run.batch == batch)
            .collect();
        let batch_label = batch.map_or("no batch".to_owned(), |b| format!("batch {b}"));
        let stem = format!(
            "{}-{}",
            slug(workload),
            batch.map_or("no-batch".to_owned(), |b| format!("batch-{b}"))
        );
        let widths: std::collections::BTreeSet<Option<u64>> =
            at_batch.iter().map(|run| run.hidden).collect();
        let widths: Vec<Option<u64>> = widths.into_iter().collect();
        let points = latest(&at_batch, |run| {
            (side_rank(&run.side), run.side.clone(), run.hidden)
        });
        let mut sides: Vec<(usize, String)> =
            points.keys().map(|(r, s, _)| (*r, s.clone())).collect();
        sides.dedup();
        let series = sides
            .iter()
            .map(|(rank, side)| Series {
                label: side_label(side).to_owned(),
                modelspec: is_modelspec(side),
                points: widths
                    .iter()
                    .enumerate()
                    .filter_map(|(i, hidden)| {
                        points
                            .get(&(*rank, side.clone(), *hidden))
                            .map(|run| Point {
                                x: i as f64,
                                us: run.median_us_per_step,
                                label: None,
                            })
                    })
                    .collect(),
            })
            .collect();
        let name = format!("{stem}.svg");
        images.insert(
            name.clone(),
            svg(&Chart {
                title: format!("{title}, {batch_label}: µs per step against hidden width"),
                x_label: "hidden width H".to_owned(),
                ticks: widths
                    .iter()
                    .map(|hidden| hidden.map_or("none".to_owned(), |h| h.to_string()))
                    .collect(),
                series,
                carry_forward: false,
                log_x: false,
            }),
        );
        let dates: std::collections::BTreeSet<&str> =
            at_batch.iter().map(|run| run.date.as_str()).collect();
        let latest_date = dates.iter().next_back().copied().unwrap_or_default();
        out.push_str(&format!(
            "\n### {}\n\n![{title}, {batch_label}: µs per step against hidden width, log scale](/img/benchmarks/{name})\n",
            capitalised(&batch_label)
        ));
        if dates.len() > 1 {
            let name = format!("{stem}-over-date.svg");
            images.insert(
                name.clone(),
                svg(&over_date(
                    format!("{title}, {batch_label}: µs per step over date"),
                    &at_batch,
                    true,
                )),
            );
            out.push_str(&format!(
                "\n![{title}, {batch_label}: µs per step over date, log scale](/img/benchmarks/{name})\n"
            ));
        }
        out.push_str(&format!(
            "\nLatest median µs per step per side, runs of {latest_date} unless the cell says otherwise, \
             and the GFLOP/s of the same run, its FLOP per step / µs per step / 1000, blank where \
             the run states no FLOP per step:\n\n| H | FLOP per step |"
        ));
        for (_, side) in &sides {
            out.push_str(&format!(" {} |", side_label(side)));
        }
        for (_, side) in &sides {
            out.push_str(&format!(" {} GFLOP/s |", side_label(side)));
        }
        out.push_str("\n|---|---|");
        for _ in 0..2 * sides.len() {
            out.push_str("---|");
        }
        out.push('\n');
        for hidden in &widths {
            let flop = sides
                .iter()
                .find_map(|(rank, side)| {
                    points
                        .get(&(*rank, side.clone(), *hidden))
                        .and_then(|run| run.flop_per_step)
                })
                .map_or(String::new(), |flop| number(flop as f64));
            out.push_str(&format!(
                "| {} | {flop} |",
                hidden.map_or("none".to_owned(), |h| h.to_string())
            ));
            for (rank, side) in &sides {
                let value =
                    points
                        .get(&(*rank, side.clone(), *hidden))
                        .map_or(String::new(), |run| {
                            if run.date == latest_date {
                                number(run.median_us_per_step)
                            } else {
                                format!("{} ({})", number(run.median_us_per_step), run.date)
                            }
                        });
                out.push_str(&format!(" {value} |"));
            }
            for (rank, side) in &sides {
                let value = points.get(&(*rank, side.clone(), *hidden)).and_then(gflops);
                out.push_str(&format!(" {} |", gflops_text(value)));
            }
            out.push('\n');
        }
    }
    out
}

/// The GFLOP/s of `run`, its FLOP per step / µs per step / 1000; none when it states no FLOP per
/// step.
fn gflops(run: &Run) -> Option<f64> {
    run.flop_per_step
        .map(|flop| flop as f64 / run.median_us_per_step / 1000.0)
}

/// One point of a sweep's FLOP chart: the latest run of one side at one batch and hidden width
/// among the runs that state `flop_per_step`, at the FLOP per step the point is placed by.
#[derive(Debug, Clone)]
struct FlopPoint {
    side: String,
    batch: Option<u64>,
    hidden: Option<u64>,
    flop_per_step: u64,
    us: f64,
    /// The run, which the tests name.
    #[cfg(test)]
    run_id: String,
}

#[cfg(test)]
impl FlopPoint {
    fn gflops(&self) -> f64 {
        self.flop_per_step as f64 / self.us / 1000.0
    }
}

/// A sweep's FLOP chart: its points in side order, then batch and hidden width; the rules they are
/// placed by; and each batch and hidden width whose sides count by different rules, with those
/// rules.
struct FlopChart {
    points: Vec<FlopPoint>,
    rules: std::collections::BTreeSet<String>,
    mixed: Vec<(Option<u64>, Option<u64>, std::collections::BTreeSet<String>)>,
}

/// The FLOP chart of `runs`, the runs of one workload. Of each side at each batch and hidden width
/// it takes the latest run that states `flop_per_step`. Where the sides at one batch and width
/// count by different rules, every side there is placed at the ModelSpec run's FLOP per step (the
/// first side's when ModelSpec has none), so all sides of a point share one FLOP count.
fn flop_points(runs: &[&Run]) -> FlopChart {
    let stating: Vec<&Run> = runs
        .iter()
        .copied()
        .filter(|run| run.flop_per_step.is_some())
        .collect();
    let chosen = latest(&stating, |run| {
        (
            run.batch,
            run.hidden,
            side_rank(&run.side),
            run.side.clone(),
        )
    });
    let mut at: std::collections::BTreeMap<(Option<u64>, Option<u64>), Vec<&Run>> =
        Default::default();
    for ((batch, hidden, _, _), run) in &chosen {
        at.entry((*batch, *hidden)).or_default().push(run);
    }
    let mut chart = FlopChart {
        points: Vec::new(),
        rules: Default::default(),
        mixed: Vec::new(),
    };
    for ((batch, hidden), sides) in at {
        let rules: std::collections::BTreeSet<String> = sides
            .iter()
            .filter_map(|run| run.flop_rule.clone())
            .collect();
        let reference = sides
            .iter()
            .find(|run| is_modelspec(&run.side))
            .unwrap_or(&sides[0]);
        let shared = rules.len() > 1;
        if shared {
            chart.rules.extend(reference.flop_rule.clone());
            chart.mixed.push((batch, hidden, rules));
        }
        for run in sides.iter().copied() {
            if !shared {
                chart.rules.extend(run.flop_rule.clone());
            }
            let flop = if shared { reference } else { run }.flop_per_step;
            chart.points.push(FlopPoint {
                side: run.side.clone(),
                batch,
                hidden,
                flop_per_step: flop.unwrap_or_default(),
                us: run.median_us_per_step,
                #[cfg(test)]
                run_id: run.run_id.clone(),
            });
        }
    }
    chart.points.sort_by(|a, b| {
        (side_rank(&a.side), &a.side, a.batch, a.hidden).cmp(&(
            side_rank(&b.side),
            &b.side,
            b.batch,
            b.hidden,
        ))
    });
    chart
}

/// What a `flop_rule` counts, as the caption says it (`modelspec.benchmark.FlopRule`).
fn rule_text(rule: &str, workload: &str) -> String {
    match (rule, workload) {
        ("MatmulOnly", BLOCK) => format!(
            "`MatmulOnly`, {}·B, the matrix products of a step of the two encoder layers, \
             forward and backward, at a batch of B sequences of {BLOCK_SEQ}",
            grouped(&BLOCK_FLOP_PER_SEQUENCE.to_string())
        ),
        ("MatmulOnly", BERT) => format!(
            "`MatmulOnly`, {}·B, the matrix products of a pre-training step, the two encoder \
             layers, the MLM transform and the tied decoder, forward and backward, at a batch of \
             B sequences of 128",
            grouped(&BERT_FLOP_PER_SEQUENCE.to_string())
        ),
        ("MatmulOnly", _) => {
            "`MatmulOnly`, 3196·B·H, the matmuls of a 784 → H → 10 step at batch B".to_owned()
        }
        ("AllSites", _) => {
            "`AllSites`, the `profile step_total` count over every dispatch site of the step"
                .to_owned()
        }
        (other, _) => format!("`{other}`"),
    }
}

/// The FLOP chart of a sweep's `runs` and its paragraph, or nothing when no run states
/// `flop_per_step`.
fn flop_section(workload: &str, runs: &[&Run], images: &mut Pages) -> String {
    let chart = flop_points(runs);
    if chart.points.is_empty() {
        return String::new();
    }
    let title = workload_title(workload);
    // One series per side and batch, in the points' order (side, then batch), so each batch's
    // points of a side are joined by a line of their own and the legend names side and batch.
    let mut series: Vec<Series> = Vec::new();
    let mut keys: Vec<(&str, Option<u64>)> = Vec::new();
    for point in &chart.points {
        let key = (point.side.as_str(), point.batch);
        if keys.last() != Some(&key) {
            keys.push(key);
            series.push(Series {
                label: format!(
                    "{}, {}",
                    side_label(&point.side),
                    point
                        .batch
                        .map_or("no batch".to_owned(), |b| format!("batch {b}"))
                ),
                modelspec: is_modelspec(&point.side),
                points: Vec::new(),
            });
        }
        if let Some(last) = series.last_mut() {
            last.points.push(Point {
                x: point.flop_per_step as f64,
                us: point.us,
                label: None,
            });
        }
    }
    let batches: std::collections::BTreeSet<u64> = chart
        .points
        .iter()
        .filter_map(|point| point.batch)
        .collect();
    let batches: Vec<String> = batches.iter().map(u64::to_string).collect();
    let at_batches = match batches.as_slice() {
        [] => String::new(),
        [one] => format!(" at batch {one}"),
        [init @ .., last] => format!(" at batches {} and {last}", init.join(", ")),
    };
    let name = format!("{}-flop.svg", slug(workload));
    images.insert(
        name.clone(),
        svg(&Chart {
            title: format!("{title}: µs per step against FLOP per step"),
            x_label: "FLOP per step, log scale".to_owned(),
            ticks: Vec::new(),
            series,
            carry_forward: false,
            log_x: true,
        }),
    );
    let rules: Vec<String> = chart
        .rules
        .iter()
        .map(|rule| rule_text(rule, workload))
        .collect();
    let mut out = format!(
        "\n### Against FLOP per step\n\n![{title}: µs per step against FLOP per step{at_batches}, both axes log scale](/img/benchmarks/{name})\n\n\
         Every batch and hidden width in one chart: the median µs per step of each side's latest \
         run at each point that states its FLOP per step, against that FLOP per step, both axes \
         log scale. A side's points of one batch are joined by a line of their own, which the \
         legend names by side and batch. The points count FLOP by {}.",
        rules.join(" and by ")
    );
    for (batch, hidden, rules) in &chart.mixed {
        let rules: Vec<String> = rules.iter().map(|rule| format!("`{rule}`")).collect();
        out.push_str(&format!(
            " At {}, H {} the sides count by different rules ({}), so every side there is placed \
             at the FLOP per step of the ModelSpec run (of the first side listed when ModelSpec \
             has none), the same count for all.",
            batch.map_or("no batch".to_owned(), |b| format!("batch {b}")),
            hidden.map_or("none".to_owned(), |h| h.to_string()),
            rules.join(", ")
        ));
    }
    out.push_str(
        " The tables below give each run's GFLOP/s; [Every run](#every-run) names the source \
         of each.\n",
    );
    out
}

/// The tokens per second of `run`, its tokens per step · 10⁶ / µs per step; none when it states
/// no tokens per step.
fn tokens_per_s(run: &Run) -> Option<f64> {
    run.tokens_per_step
        .map(|tokens| tokens as f64 * 1e6 / run.median_us_per_step)
}

/// The section of a sequence workload, one whose runs state `tokens_per_step`: a chart of µs per
/// step against batch at the latest run of each side and batch, and those runs in a table with
/// their tokens per second and GFLOP/s.
fn sequence_section(workload: &str, runs: &[&Run], images: &mut Pages) -> String {
    let title = workload_title(workload);
    let batches: Vec<Option<u64>> = runs
        .iter()
        .map(|run| run.batch)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let points = latest(runs, |run| {
        (side_rank(&run.side), run.side.clone(), run.batch)
    });
    let mut sides: Vec<(usize, String)> = points.keys().map(|(r, s, _)| (*r, s.clone())).collect();
    sides.dedup();
    let batch_label = |batch: &Option<u64>| batch.map_or("none".to_owned(), |b| b.to_string());
    let series = sides
        .iter()
        .map(|(rank, side)| Series {
            label: side_label(side).to_owned(),
            modelspec: is_modelspec(side),
            points: batches
                .iter()
                .enumerate()
                .filter_map(|(i, batch)| {
                    points.get(&(*rank, side.clone(), *batch)).map(|run| Point {
                        x: i as f64,
                        us: run.median_us_per_step,
                        label: None,
                    })
                })
                .collect(),
        })
        .collect();
    let name = format!("{}-batch.svg", slug(workload));
    images.insert(
        name.clone(),
        svg(&Chart {
            title: format!("{title}: µs per step against batch"),
            x_label: "batch, sequences".to_owned(),
            ticks: batches.iter().map(batch_label).collect(),
            series,
            carry_forward: false,
            log_x: false,
        }),
    );
    let rules: std::collections::BTreeSet<&str> = runs
        .iter()
        .filter_map(|run| run.flop_rule.as_deref())
        .collect();
    let rules: Vec<String> = rules.iter().map(|rule| rule_text(rule, workload)).collect();
    let mut out = format!(
        "## {title}\n\n`{workload}`: per batch of sequences, the median µs per step of the latest \
         run of each side, ModelSpec solid with filled circles and the opponents dashed with open \
         squares.\n\n![{title}: µs per step against batch, log scale](/img/benchmarks/{name})\n\n\
         ### Latest median per side and batch\n\n\
         Tokens per second are the run's tokens per step · 10⁶ / µs per step, GFLOP/s its FLOP \
         per step / µs per step / 1000, blank where the run states no tokens or FLOP per step."
    );
    if !rules.is_empty() {
        out.push_str(&format!(" FLOP are counted by {}.", rules.join(" and by ")));
    }
    out.push_str(
        "\n\n| Batch | Tokens per step | Side | Date | Build | µs per step | Tokens per second | GFLOP/s | Repeats |\n\
         |---|---|---|---|---|---|---|---|---|\n",
    );
    for batch in &batches {
        for (rank, side) in &sides {
            let Some(run) = points.get(&(*rank, side.clone(), *batch)) else {
                continue;
            };
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                batch_label(batch),
                run.tokens_per_step
                    .map_or(String::new(), |tokens| grouped(&tokens.to_string())),
                side_label(&run.side),
                run.date,
                crate::cell(&build(run)),
                number(run.median_us_per_step),
                tokens_per_s(run).map_or(String::new(), number),
                gflops_text(gflops(run)),
                repeats(run)
            ));
        }
    }
    out
}

/// Where the report a run's `source` names is read: a verification report under
/// `.engineering/planning/verification-report/`, or `CHANGELOG.md`; none for another source.
fn report_url(source: &str) -> Option<String> {
    let cited = source.split([',', ' ']).next().unwrap_or_default();
    if let Some(name) = cited.strip_prefix("verification-report:") {
        (!name.is_empty()).then(|| {
            format!(
                "{}/.engineering/planning/verification-report/{name}.md",
                crate::SOURCE
            )
        })
    } else if cited == "CHANGELOG.md" {
        Some(format!("{}/CHANGELOG.md", crate::SOURCE))
    } else {
        None
    }
}

/// The section of `bert-tiny-pretrain` (story:bert-tiny-vs-pytorch-cuda), a workload timed to a
/// held-out loss: per side a table of its runs in file order, with µs per step, tokens per
/// second, GFLOP/s, the target, the steps and training seconds to it and the final held-out
/// loss, each figure linked to the report that states it. No chart: the figure that matters is
/// the time to the target, one number per run.
fn target_section(workload: &str, runs: &[&Run]) -> String {
    let title = workload_title(workload);
    let rules: std::collections::BTreeSet<&str> = runs
        .iter()
        .filter_map(|run| run.flop_rule.as_deref())
        .collect();
    let rules: Vec<String> = rules.iter().map(|rule| rule_text(rule, workload)).collect();
    let counts: std::collections::BTreeSet<u64> = runs.iter().map(|run| run.repeat_count).collect();
    let repeats = match counts.iter().collect::<Vec<_>>()[..] {
        [&1] => "1 run".to_owned(),
        [&count] => format!("{count} runs"),
        _ => "its runs".to_owned(),
    };
    // The clocks as each report's "training clock" row states them: 2026-10-28 excludes the
    // held-out evaluations on every side, 2026-10-29 includes them on the ModelSpec side.
    let report = |date: &str| {
        report_url(&format!("verification-report:bert-tiny-vs-pytorch-{date}")).unwrap_or_default()
    };
    let (excluded, included) = (report("2026-10-28"), report("2026-10-29"));
    let mut out = format!(
        "## {title}, time to the held-out baseline\n\n`{workload}`: `examples/bert_tiny.mspec` \
         pre-trained by each side from the same initial weights on the same batches of the same \
         packed corpus, with the same optimizer and schedule, until its held-out loss, evaluated \
         every `Eval every` steps, is at most the target loss. Steps to target is the first \
         evaluated step at or below the target, training seconds to target the training time up \
         to that step without checkpoint writes or events. The held-out evaluations are outside \
         that time on every side in [the report of 2026-10-28]({excluded}) and on the PyTorch \
         sides in [the report of 2026-10-29]({included}); the ModelSpec runs of 2026-10-29 \
         reduce each evaluation on the device inside the evaluated step's command buffer, so \
         their training seconds, and with them their µs per step, include the evaluations. \
         µs per step is a run's training seconds over its steps, the mean of its steps, and the figure is the \
         median of {repeats}, the median of the run's repeats as for every workload; tokens per \
         second is its tokens per step · 10⁶ / µs per step, GFLOP/s its FLOP per step / µs per \
         step / 1000. Each figure links to the report that states it."
    );
    if !rules.is_empty() {
        out.push_str(&format!(" FLOP are counted by {}.", rules.join(" and by ")));
    }
    out.push('\n');
    let mut sides: Vec<(usize, &str)> = runs
        .iter()
        .map(|run| (side_rank(&run.side), run.side.as_str()))
        .collect();
    sides.sort();
    sides.dedup();
    for (_, side) in sides {
        out.push_str(&format!(
            "\n### {}\n\n\
             | Date | Build | µs per step | Tokens per second | GFLOP/s | Target loss | Steps to target | Training seconds to target | Final held-out loss | Eval every |\n\
             |---|---|---|---|---|---|---|---|---|---|\n",
            side_label(side)
        ));
        for run in runs.iter().filter(|run| run.side == side) {
            let url = report_url(&run.source);
            let linked = |figure: Option<String>| match (figure, &url) {
                (None, _) => String::new(),
                (Some(figure), Some(url)) => format!("[{figure}]({url})"),
                (Some(figure), None) => figure,
            };
            let figures = [
                Some(number(run.median_us_per_step)),
                tokens_per_s(run).map(number),
                gflops(run).map(|value| gflops_text(Some(value))),
                run.target_loss.map(|loss| loss.to_string()),
                run.steps_to_target.map(|steps| grouped(&steps.to_string())),
                run.seconds_to_target.map(number),
                run.final_heldout_loss.map(|loss| format!("{loss:.4}")),
            ];
            let figures: Vec<String> = figures.into_iter().map(linked).collect();
            out.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                run.date,
                crate::cell(&build(run)),
                figures.join(" | "),
                run.eval_every
                    .map_or(String::new(), |every| every.to_string())
            ));
        }
    }
    out
}

fn capitalised(text: &str) -> String {
    let mut chars = text.chars();
    chars.next().map_or(String::new(), |first| {
        first.to_uppercase().chain(chars).collect()
    })
}

/// The page and its charts, keyed by file name, from [`RUNS`] under `root`.
pub fn files(root: &Path) -> Result<(String, Pages)> {
    let text = std::fs::read_to_string(root.join(RUNS)).map_err(crate::io(RUNS))?;
    let schema = std::fs::read_to_string(root.join(SCHEMA)).map_err(crate::io(SCHEMA))?;
    let schema: Value = serde_json::from_str(&schema).map_err(|e| format!("{SCHEMA}: {e}"))?;
    let runs = read(&text, &schema)?;
    Ok(render(&runs))
}

/// The page and its charts, keyed by file name, from `runs`, the validated runs of [`RUNS`].
fn render(runs: &[Run]) -> (String, Pages) {
    let mut workloads: Vec<&str> = Vec::new();
    for run in runs {
        if !workloads.contains(&run.workload.as_str()) {
            workloads.push(&run.workload);
        }
    }
    let mut images = Pages::new();
    let mut page = crate::front_matter(
        "Benchmarks",
        "Benchmarks",
        "Every benchmark run of ModelSpec and its opponents, from benchmarks/runs.yaml: µs per step per workload over time.",
        5,
    );
    page.push_str(&format!(
        "Generated by `modelspec-docs` from [`{RUNS}`]({SOURCE}/{RUNS}), which holds every \
         benchmark run measured so far: one entry per run, with the µs per step of each repeat \
         where its source recorded them, their median, and the verification report or CHANGELOG \
         release that states them. Each entry is validated against the JSON Schema of \
         `modelspec.benchmark.BenchmarkRun` ([`spec/domains/benchmark.yaml`]({SOURCE}/spec/domains/benchmark.yaml)), \
         and a median that is not the median of its repeats is refused. Lower is faster; the \
         charts are log scale. GPU clocks were not locked and several runs shared a loaded \
         workstation: the `note` of each entry says how.\n\n",
        SOURCE = crate::SOURCE
    ));
    for workload in &workloads {
        let of_workload: Vec<&Run> = runs
            .iter()
            .filter(|run| run.workload == *workload)
            .collect();
        let section = if *workload == BERT {
            target_section(workload, &of_workload)
        } else if of_workload.iter().any(|run| run.tokens_per_step.is_some()) {
            sequence_section(workload, &of_workload, &mut images)
        } else if of_workload
            .iter()
            .any(|run| run.batch.is_some() || run.hidden.is_some())
        {
            sweep_section(workload, &of_workload, &mut images)
        } else {
            fixed_section(workload, &of_workload, &mut images)
        };
        page.push_str(&section);
        page.push('\n');
    }
    page.push_str(
        "## Every run\n\n| Run | Date | Workload | Side | Threads | Mode | Batch | H | Build | µs per step | Repeats | Source |\n\
         |---|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    for run in runs {
        page.push_str(&format!(
            "| `{}` | {} | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | `{}` |\n",
            run.run_id,
            run.date,
            run.workload,
            side_label(&run.side),
            threads_label(run.threads),
            run.mode,
            run.batch.map_or(String::new(), |b| b.to_string()),
            run.hidden.map_or(String::new(), |h| h.to_string()),
            crate::cell(&build(run)),
            number(run.median_us_per_step),
            repeats(run),
            crate::cell(&run.source)
        ));
    }
    let unmeasured: Vec<&Workload> = WORKLOADS
        .iter()
        .filter(|declared| !workloads.contains(&declared.id))
        .collect();
    if !unmeasured.is_empty() {
        page.push_str(
            "\n## Not measured\n\nWorkloads the page declares that have no run in \
             `benchmarks/runs.yaml` yet, so no figure. Each gets a section of its own above with \
             its first run.\n\n",
        );
        for declared in unmeasured {
            page.push_str(&format!(
                "- **{}** (`{}`): {}\n",
                declared.title, declared.id, declared.not_measured
            ));
        }
    }
    (page, images)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::*;

    fn root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn schema() -> Value {
        serde_json::from_str(&fs::read_to_string(root().join(SCHEMA)).unwrap()).unwrap()
    }

    const TWO_RUNS: &str = "runs:
  - run_id: a
    date: \"2026-10-21\"
    workload: mnist-784-64-10
    side: ModelSpecVulkan
    mode: Timed
    repeat_count: 3
    repeats_us_per_step:
      - 3
      - 1
      - 2
    median_us_per_step: 2
    source: CHANGELOG.md [0.4.0]
    state: Recorded
  - run_id: b
    date: \"2026-10-22\"
    workload: mnist-784-64-10
    side: PyTorchCudaEager
    mode: Timed
    repeat_count: 4
    repeats_us_per_step:
      - 1
      - 4
      - 2
      - 3
    median_us_per_step: 2.5
    source: verification-report:x
    state: Recorded
";

    #[test]
    fn a_well_formed_history_is_read() {
        let runs = read(TWO_RUNS, &schema()).unwrap();
        assert_eq!(runs.len(), 2);
    }

    #[test]
    fn a_missing_required_field_is_refused_with_its_line_and_field() {
        let text = TWO_RUNS.replace("    source: verification-report:x\n", "");
        let error = read(&text, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:15: runs[1] (b) field `source`:"),
            "{error}"
        );
        let text = TWO_RUNS.replacen("    median_us_per_step: 2\n", "", 1);
        let error = read(&text, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:2: runs[0] (a) field `median_us_per_step`:"),
            "{error}"
        );
    }

    #[test]
    fn a_wrong_field_type_or_unknown_field_is_refused_with_its_line_and_field() {
        let text = TWO_RUNS.replace("    repeat_count: 4\n", "    repeat_count: four\n");
        let error = read(&text, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:20: runs[1] (b) field `repeat_count`:"),
            "{error}"
        );
        let text = TWO_RUNS.replace(
            "    mode: Timed\n    repeat_count: 3\n",
            "    mode: Timed\n    speed: 3\n    repeat_count: 3\n",
        );
        let error = read(&text, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:7: runs[0] (a) field `speed`:"),
            "{error}"
        );
    }

    #[test]
    fn a_median_that_is_not_the_median_of_its_repeats_is_refused_with_its_line_and_field() {
        let text = TWO_RUNS.replace("    median_us_per_step: 2\n", "    median_us_per_step: 3\n");
        let error = read(&text, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:12: runs[0] (a) field `median_us_per_step`:"),
            "{error}"
        );
        assert!(error.contains("median of its repeats is 2"), "{error}");
        let text = TWO_RUNS.replace(
            "    median_us_per_step: 2.5\n",
            "    median_us_per_step: 2\n",
        );
        let error = read(&text, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:26: runs[1] (b) field `median_us_per_step`:"),
            "{error}"
        );
    }

    #[test]
    fn a_nan_or_infinite_figure_is_refused_not_read_as_not_recorded() {
        let text = TWO_RUNS.replace("      - 1\n", "      - .nan\n");
        let error = read(&text, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:8: runs[0] (a) field `repeats_us_per_step`:"),
            "{error}"
        );
        let text = TWO_RUNS.replace(
            "    median_us_per_step: 2\n",
            "    median_us_per_step: .inf\n",
        );
        let error = read(&text, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:12: runs[0] (a) field `median_us_per_step`:"),
            "{error}"
        );
    }

    #[test]
    fn repeats_that_do_not_match_their_count_and_a_repeated_id_are_refused() {
        let text = TWO_RUNS.replace("    repeat_count: 3\n", "    repeat_count: 5\n");
        let error = read(&text, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:8: runs[0] (a) field `repeats_us_per_step`:"),
            "{error}"
        );
        let text = TWO_RUNS.replace("  - run_id: b\n", "  - run_id: a\n");
        let error = read(&text, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:15: runs[1] (a) field `run_id`:"),
            "{error}"
        );
        let text = TWO_RUNS.replace("\"2026-10-22\"", "\"22.10.2026\"");
        let error = read(&text, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:16: runs[1] (b) field `date`:"),
            "{error}"
        );
    }

    #[test]
    fn a_yaml_syntax_error_is_refused_with_its_line() {
        let text = TWO_RUNS.replace(
            "    mode: Timed\n    repeat_count: 4",
            "    mode: [Timed\n    repeat_count: 4",
        );
        let error = read(&text, &schema()).err().unwrap();
        assert!(error.starts_with("benchmarks/runs.yaml:"), "{error}");
        assert!(error.contains("line "), "{error}");
    }

    #[test]
    fn every_sweep_entry_repeats_the_raw_record_it_cites() {
        assert_entries_repeat_their_raw_records("mlp-784-H-10", 86);
    }

    /// The nine runs of verification-report:transformer-block-benchmark-2026-10-27 (3 batches ×
    /// ModelSpec Vulkan, PyTorch CUDA eager and `torch.compile`) each repeat their raw record.
    #[test]
    fn every_transformer_block_entry_repeats_the_raw_record_it_cites() {
        assert_entries_repeat_their_raw_records(BLOCK, 9);
    }

    /// Every committed run of `workload`, `count` of them, cites one raw record of its side, batch
    /// and hidden width under `benchmarks/raw/`, no record twice, and states that record's repeats,
    /// median, timed steps, FLOP and tokens per step, and its commit.
    fn assert_entries_repeat_their_raw_records(workload: &str, count: usize) {
        let text = fs::read_to_string(root().join(RUNS)).unwrap();
        let runs = read(&text, &schema()).unwrap();
        let sweep: Vec<&Run> = runs.iter().filter(|run| run.workload == workload).collect();
        assert_eq!(sweep.len(), count, "{workload}");
        let mut used = std::collections::BTreeSet::new();
        for run in sweep {
            let (_, path) = run
                .source
                .split_once("raw records ")
                .unwrap_or_else(|| panic!("{} cites no raw records: {}", run.run_id, run.source));
            assert!(
                path.starts_with("benchmarks/raw/"),
                "{}: {path}",
                run.run_id
            );
            let side = match run.side.as_str() {
                "ModelSpecVulkan" => "modelspec-vulkan",
                "PyTorchCudaEager" => "pytorch-eager",
                "PyTorchCudaCompile" => "pytorch-compile",
                other => panic!("{}: no raw side for {other}", run.run_id),
            };
            let records = fs::read_to_string(root().join(path)).unwrap();
            let matching: Vec<(usize, Value)> = records
                .lines()
                .enumerate()
                .map(|(n, line)| (n, serde_json::from_str::<Value>(line).unwrap()))
                .filter(|(_, record)| {
                    record["side"] == side
                        && record["batch"].as_u64() == run.batch
                        && record["hidden"].as_u64() == run.hidden
                })
                .collect();
            assert_eq!(matching.len(), 1, "{}: records in {path}", run.run_id);
            let (line, record) = &matching[0];
            assert!(
                used.insert((path.to_owned(), *line)),
                "{}: record used twice",
                run.run_id
            );
            let raw: Vec<f64> = record["runs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|repeat| repeat["us_per_step"].as_f64().unwrap())
                .collect();
            assert_eq!(
                run.repeats_us_per_step.as_deref(),
                Some(&raw[..]),
                "{}",
                run.run_id
            );
            assert_eq!(run.repeat_count, raw.len() as u64, "{}", run.run_id);
            assert_eq!(
                Some(run.median_us_per_step),
                record["median_us_per_step"].as_f64(),
                "{}",
                run.run_id
            );
            assert_eq!(run.timed_steps, record["steps"].as_u64(), "{}", run.run_id);
            assert_eq!(
                run.flop_per_step,
                record["flop_per_step"].as_u64(),
                "{}",
                run.run_id
            );
            assert_eq!(
                run.tokens_per_step,
                record["tokens_per_step"].as_u64(),
                "{}",
                run.run_id
            );
            assert_eq!(
                run.commit.as_deref(),
                record["build"]["commit"].as_str(),
                "{}",
                run.run_id
            );
        }
        assert_eq!(used.len(), count, "every raw record is cited once");
    }

    /// The held-out unigram baseline of the packed corpus, `heldout_unigram_baseline_nats` of its
    /// `packed.json`: the target of every `bert-tiny-pretrain` run.
    const BERT_TARGET: f64 = 6.926537926202018;

    /// What one raw record of a `bert-tiny-pretrain` run, one of its three runs, states.
    struct BertRecord {
        us_per_step: f64,
        steps: u64,
        eval_every: u64,
        steps_to_target: u64,
        seconds_to_target: f64,
        train_seconds: f64,
        final_heldout_loss: f64,
        commit: String,
    }

    /// From the `(step, held-out loss, training seconds)` of a run's evaluations: its eval
    /// cadence (every evaluation on it, from step 0), the first evaluated step at or below
    /// [`BERT_TARGET`] and that evaluation's training seconds, and the last held-out loss.
    fn bert_curve(path: &str, evals: &[(u64, f64, f64)]) -> (u64, u64, f64, f64) {
        assert!(evals.len() > 1, "{path}: {} evaluations", evals.len());
        let every = evals[1].0 - evals[0].0;
        for (i, (step, _, _)) in evals.iter().enumerate() {
            assert_eq!(*step, i as u64 * every, "{path}: eval cadence");
        }
        let (step, _, seconds) = evals
            .iter()
            .find(|(_, loss, _)| *loss <= BERT_TARGET)
            .unwrap_or_else(|| panic!("{path}: never at or below {BERT_TARGET}"));
        (every, *step, *seconds, evals[evals.len() - 1].1)
    }

    /// A ModelSpec run's stdout under `train --eval-every --clock`: its `eval <k> loss <l>
    /// accuracy <a> train_seconds <s>` lines and the closing `train_seconds <s> eval_seconds <e>
    /// steps <n>`; µs per step is its training seconds over its steps.
    fn modelspec_bert_record(path: &str, commit: &str) -> BertRecord {
        let text = fs::read_to_string(root().join(path)).unwrap();
        let mut evals = Vec::new();
        let mut total = None;
        for line in text.lines() {
            match line.split(' ').collect::<Vec<_>>()[..] {
                [
                    "eval",
                    step,
                    "loss",
                    loss,
                    "accuracy",
                    _,
                    "train_seconds",
                    seconds,
                ] => evals.push((
                    step.parse().unwrap(),
                    loss.parse().unwrap(),
                    seconds.parse().unwrap(),
                )),
                ["train_seconds", seconds, "eval_seconds", _, "steps", steps] => {
                    total = Some((
                        seconds.parse::<f64>().unwrap(),
                        steps.parse::<u64>().unwrap(),
                    ));
                }
                _ => {}
            }
        }
        let (train_seconds, steps) =
            total.unwrap_or_else(|| panic!("{path}: no closing train_seconds line"));
        let (eval_every, steps_to_target, seconds_to_target, final_heldout_loss) =
            bert_curve(path, &evals);
        assert_eq!(evals[evals.len() - 1].0, steps, "{path}: last evaluation");
        BertRecord {
            us_per_step: train_seconds * 1e6 / steps as f64,
            steps,
            eval_every,
            steps_to_target,
            seconds_to_target,
            train_seconds,
            final_heldout_loss,
            commit: commit.to_owned(),
        }
    }

    /// An opponent run's JSON lines: one `{"step", "heldout_loss", "train_seconds"}` per
    /// evaluation, then its result line, whose target figures must be those of its evaluations
    /// and whose batch, tokens and FLOP must be those of `run`.
    fn pytorch_bert_record(path: &str, run: &Run) -> BertRecord {
        let text = fs::read_to_string(root().join(path)).unwrap();
        let lines: Vec<Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let (result, evals) = lines.split_last().unwrap();
        let evals: Vec<(u64, f64, f64)> = evals
            .iter()
            .map(|eval| {
                (
                    eval["step"].as_u64().unwrap(),
                    eval["heldout_loss"].as_f64().unwrap(),
                    eval["train_seconds"].as_f64().unwrap(),
                )
            })
            .collect();
        let (eval_every, steps_to_target, seconds_to_target, final_heldout_loss) =
            bert_curve(path, &evals);
        assert_eq!(result["status"], "measured", "{path}");
        assert_eq!(result["build"]["dirty"], false, "{path}");
        assert_eq!(result["target_loss"].as_f64(), Some(BERT_TARGET), "{path}");
        assert_eq!(result["eval_every"].as_u64(), Some(eval_every), "{path}");
        assert_eq!(
            result["steps_to_target"].as_u64(),
            Some(steps_to_target),
            "{path}"
        );
        assert_eq!(
            result["seconds_to_target"].as_f64(),
            Some(seconds_to_target),
            "{path}"
        );
        assert_eq!(
            result["final_heldout_loss"].as_f64(),
            Some(final_heldout_loss),
            "{path}"
        );
        assert_eq!(result["batch"].as_u64(), run.batch, "{path}");
        assert_eq!(
            result["tokens_per_step"].as_u64(),
            run.tokens_per_step,
            "{path}"
        );
        assert_eq!(
            result["flop_per_step"].as_u64(),
            run.flop_per_step,
            "{path}"
        );
        BertRecord {
            us_per_step: result["us_per_step"].as_f64().unwrap(),
            steps: result["steps"].as_u64().unwrap(),
            eval_every,
            steps_to_target,
            seconds_to_target,
            train_seconds: result["train_seconds"].as_f64().unwrap(),
            final_heldout_loss,
            commit: result["build"]["commit"].as_str().unwrap().to_owned(),
        }
    }

    /// The three runs of verification-report:bert-tiny-vs-pytorch-2026-10-28: ModelSpec Vulkan,
    /// PyTorch CUDA eager and `torch.compile`, three runs each of the 10,000 steps of
    /// `examples/bert_tiny.mspec`, all counted by one `MatmulOnly` FLOP figure. Each cites the
    /// three raw records of its side in run order and states their µs per step in that order;
    /// the median of their µs per step, steps and training seconds to the target, training
    /// seconds and final held-out loss; and their steps, eval cadence, target and commit.
    #[test]
    fn every_bert_tiny_pretrain_entry_repeats_the_raw_records_it_cites() {
        let text = fs::read_to_string(root().join(RUNS)).unwrap();
        let runs = read(&text, &schema()).unwrap();
        let bert: Vec<&Run> = runs.iter().filter(|run| run.workload == BERT).collect();
        let mut sides: Vec<&str> = bert.iter().map(|run| run.side.as_str()).collect();
        sides.sort_unstable();
        assert_eq!(
            sides,
            ["ModelSpecVulkan", "PyTorchCudaCompile", "PyTorchCudaEager"]
        );
        for run in bert {
            assert_eq!(
                run.flop_per_step,
                Some(BERT_FLOP_PER_SEQUENCE * BERT_BATCH),
                "{}",
                run.run_id
            );
            assert_eq!(run.target_loss, Some(BERT_TARGET), "{}", run.run_id);
            let (_, cited) = run
                .source
                .split_once("raw records ")
                .unwrap_or_else(|| panic!("{} cites no raw records: {}", run.run_id, run.source));
            let mut names = cited.split(", ");
            let (dir, first) = names.next().unwrap().rsplit_once('/').unwrap();
            assert!(dir.starts_with("benchmarks/raw/"), "{}: {dir}", run.run_id);
            let (prefix, extension) = match run.side.as_str() {
                "ModelSpecVulkan" => ("modelspec", "stdout.txt"),
                "PyTorchCudaEager" => ("pytorch-eager", "jsonl"),
                "PyTorchCudaCompile" => ("pytorch-compile", "jsonl"),
                other => panic!("{}: no raw side for {other}", run.run_id),
            };
            let names: Vec<&str> = std::iter::once(first).chain(names).collect();
            let expected: Vec<String> = (1..=3)
                .map(|n| format!("{prefix}-{n}.{extension}"))
                .collect();
            assert_eq!(names, expected, "{}", run.run_id);
            let records: Vec<BertRecord> = names
                .iter()
                .map(|name| {
                    let path = format!("{dir}/{name}");
                    if is_modelspec(&run.side) {
                        let built =
                            fs::read_to_string(root().join(dir).join("commit.txt")).unwrap();
                        let mut built = built.lines();
                        let commit = built.next().unwrap();
                        assert_eq!(
                            built
                                .next()
                                .and_then(|line| line.strip_prefix("modelspec ")),
                            run.modelspec_version.as_deref(),
                            "{}",
                            run.run_id
                        );
                        modelspec_bert_record(&path, commit)
                    } else {
                        pytorch_bert_record(&path, run)
                    }
                })
                .collect();
            let middle = |figure: fn(&BertRecord) -> f64| {
                median(&records.iter().map(figure).collect::<Vec<_>>())
            };
            assert_eq!(
                run.repeats_us_per_step,
                Some(records.iter().map(|record| record.us_per_step).collect()),
                "{}",
                run.run_id
            );
            assert_eq!(run.repeat_count, 3, "{}", run.run_id);
            assert_eq!(
                Some(run.median_us_per_step),
                middle(|record| record.us_per_step),
                "{}",
                run.run_id
            );
            assert_eq!(
                run.seconds_to_target,
                middle(|record| record.seconds_to_target),
                "{}",
                run.run_id
            );
            assert_eq!(
                run.train_seconds,
                middle(|record| record.train_seconds),
                "{}",
                run.run_id
            );
            assert_eq!(
                run.final_heldout_loss,
                middle(|record| record.final_heldout_loss),
                "{}",
                run.run_id
            );
            let mut crossings: Vec<u64> = records
                .iter()
                .map(|record| record.steps_to_target)
                .collect();
            crossings.sort_unstable();
            assert_eq!(run.steps_to_target, Some(crossings[1]), "{}", run.run_id);
            for record in &records {
                assert_eq!(run.timed_steps, Some(record.steps), "{}", run.run_id);
                assert_eq!(run.eval_every, Some(record.eval_every), "{}", run.run_id);
                assert_eq!(
                    run.commit.as_deref(),
                    Some(record.commit.as_str()),
                    "{}",
                    run.run_id
                );
            }
        }
    }

    /// The published `bert-tiny-pretrain` section holds the three sides' medians of
    /// verification-report:bert-tiny-vs-pytorch-2026-10-28, each figure linked to that report.
    #[test]
    fn the_published_page_has_the_bert_tiny_pretrain_figures_of_the_report() {
        let (page, _) = files(&root()).unwrap();
        let start = page
            .find("\n## BERT-tiny pre-training, time to the held-out baseline\n")
            .unwrap_or_else(|| panic!("no bert-tiny-pretrain section"));
        let end = page[start + 1..]
            .find("\n## ")
            .map_or(page.len(), |at| start + 1 + at);
        let section = &page[start..end];
        let url = report_url("verification-report:bert-tiny-vs-pytorch-2026-10-28").unwrap();
        // Date, build, then µs per step, tokens per second, GFLOP/s, target loss, steps to
        // target, training seconds to target and final held-out loss, each linked; eval every.
        let row = |build: &str, figures: [&str; 7]| {
            let figures: Vec<String> = figures
                .iter()
                .map(|figure| format!("[{figure}]({url})"))
                .collect();
            format!("| 2026-10-28 | {build} | {} | 100 |", figures.join(" | "))
        };
        let target = "6.926537926202018";
        let expected = [
            row(
                "0.21.0 (`3b9869f`)",
                [
                    "222,576.036",
                    "18,402.7",
                    "168.2",
                    target,
                    "1,100",
                    "245.969",
                    "6.7675",
                ],
            ),
            row(
                "torch 2.14.1+cu130, CUDA 13.0",
                [
                    "9,897.473",
                    "413,843.025",
                    "3,783.5",
                    target,
                    "1,100",
                    "9.82",
                    "6.7673",
                ],
            ),
            row(
                "torch 2.14.1+cu130, CUDA 13.0",
                [
                    "4,887.491",
                    "838,057.772",
                    "7,661.8",
                    target,
                    "1,100",
                    "4.881",
                    "6.7675",
                ],
            ),
        ];
        let table: Vec<&str> = section
            .lines()
            .filter(|line| line.starts_with("| ") && !line.starts_with("| Date"))
            .collect();
        assert_eq!(table, expected, "{section}");
        for side in [
            "ModelSpec Vulkan",
            "PyTorch CUDA eager",
            "PyTorch CUDA torch.compile",
        ] {
            assert!(section.contains(&format!("\n### {side}\n")), "{side}");
        }
    }

    #[test]
    fn the_history_holds_every_run_the_story_names() {
        let text = fs::read_to_string(root().join(RUNS)).unwrap();
        let runs = read(&text, &schema()).unwrap();
        let mnist = |side: &str, version: Option<&str>, threads: Option<u64>, mode: &str| {
            runs.iter()
                .filter(|run| {
                    run.workload == "mnist-784-64-10"
                        && run.side == side
                        && run.mode == mode
                        && (version.is_none() || run.modelspec_version.as_deref() == version)
                        && (threads.is_none() || run.threads == threads)
                })
                .count()
        };
        assert_eq!(mnist("ModelSpecCpu", Some("0.1.1"), Some(1), "Timed"), 1);
        for version in ["0.4.0", "0.5.0", "0.6.0"] {
            assert_eq!(
                mnist("ModelSpecVulkan", Some(version), None, "Timed"),
                1,
                "{version}"
            );
        }
        assert_eq!(mnist("ModelSpecVulkan", Some("0.6.0"), None, "Profiled"), 1);
        assert!(mnist("PyTorchCpuEager", None, Some(1), "Timed") >= 1);
        assert!(mnist("PyTorchCpuEager", None, Some(8), "Timed") >= 1);
        assert_eq!(mnist("PyTorchCudaEager", None, None, "Timed"), 1);
        assert_eq!(mnist("PyTorchCudaCompile", None, None, "Timed"), 1);
        let sweep: Vec<&Run> = runs
            .iter()
            .filter(|run| {
                run.source
                    .starts_with("verification-report:mlp-size-sweep-2026-10-23, raw records ")
            })
            .collect();
        assert_eq!(sweep.len(), 36);
        for run in &sweep {
            assert_eq!(run.workload, "mlp-784-H-10");
            assert_eq!(run.repeats_us_per_step.as_ref().map(Vec::len), Some(3));
        }
        let mut points: Vec<(u64, u64, &str)> = sweep
            .iter()
            .map(|run| (run.batch.unwrap(), run.hidden.unwrap(), run.side.as_str()))
            .collect();
        points.sort();
        points.dedup();
        assert_eq!(points.len(), 36);
        assert!(!text.contains("/home/"));
    }

    #[test]
    fn the_page_shows_every_release_and_the_cuda_eager_line_and_the_sweep_per_batch() {
        let (page, images) = files(&root()).unwrap();
        let mnist = &images["mnist-784-64-10.svg"];
        for version in ["0.1.1", "0.4.0", "0.5.0", "0.6.0"] {
            assert!(mnist.contains(&format!(">{version}</text>")), "{version}");
        }
        assert!(mnist.contains(">PyTorch CUDA eager</text>"));
        // PyTorch CUDA eager is the first opponent side of SIDES, so it takes the first colour.
        let colour = OPPONENT_COLOURS[0];
        assert!(
            mnist.contains(&format!(
                "stroke=\"{colour}\" stroke-width=\"1.6\" stroke-dasharray=\"2 3\"/>"
            )),
            "the PyTorch CUDA eager line"
        );
        assert!(mnist.contains(">ModelSpec Vulkan</text>"));
        assert!(page.contains("![MNIST 784-64-10"));
        assert!(page.contains("(/img/benchmarks/mnist-784-64-10.svg)"));
        for batch in [100, 1000, 8192] {
            let name = format!("mlp-784-h-10-batch-{batch}.svg");
            let chart = &images[&name];
            assert!(chart.contains(">PyTorch CUDA eager</text>"), "{name}");
            assert!(chart.contains(">ModelSpec Vulkan</text>"), "{name}");
            assert!(chart.contains(">4096</text>"), "{name}");
            assert!(
                page.contains(&format!("(/img/benchmarks/{name})")),
                "{name}"
            );
        }
        assert!(page.contains("| 1 thread"));
    }

    /// The value of every `name="…"` attribute in `svg`.
    fn attribute_values<'a>(svg: &'a str, name: &str) -> Vec<&'a str> {
        let key = format!(" {name}=\"");
        svg.match_indices(&key)
            .map(|(at, _)| {
                let rest = &svg[at + key.len()..];
                &rest[..rest.find('"').unwrap()]
            })
            .collect()
    }

    #[test]
    fn the_charts_draw_no_background_and_follow_the_dark_theme() {
        let (_, images) = files(&root()).unwrap();
        for (name, svg) in &images {
            let lower = svg.to_ascii_lowercase();
            for white in ["#fff\"", "#ffffff", "\"white\"", "rgb(255, 255, 255)"] {
                assert!(!lower.contains(white), "{name} declares {white}");
            }
            let (_, dark) = svg
                .split_once("@media (prefers-color-scheme: dark)")
                .unwrap_or_else(|| panic!("{name} has no dark-theme rules"));
            for colour in attribute_values(svg, "fill")
                .into_iter()
                .chain(attribute_values(svg, "stroke"))
                .filter(|colour| colour.starts_with('#'))
            {
                assert!(
                    dark.contains(&format!("[fill=\"{colour}\"]"))
                        && dark.contains(&format!("[stroke=\"{colour}\"]")),
                    "{name}: {colour} has no dark-theme colour"
                );
            }
        }
    }

    #[test]
    fn chart_text_is_at_least_12_px_in_a_390_px_viewport() {
        // At a 390 px viewport the content column, and so a chart scaled to fit it, is 358 px
        // wide (the site review of 2026-10-25 measured it).
        let (_, images) = files(&root()).unwrap();
        for (name, svg) in &images {
            let width: f64 = attribute_values(svg, "width")[0].parse().unwrap();
            let scale = (358.0 / width).min(1.0);
            let sizes: Vec<f64> = attribute_values(svg, "font-size")
                .into_iter()
                .map(|size| size.parse().unwrap())
                .collect();
            assert!(!sizes.is_empty(), "{name} sets no font size");
            for size in sizes {
                assert!(
                    size * scale >= 12.0,
                    "{name}: font-size {size} renders at {:.1} px",
                    size * scale
                );
            }
        }
    }

    #[test]
    fn the_committed_page_and_charts_equal_the_generated_ones() {
        let (page, images) = files(&root()).unwrap();
        let committed = fs::read_to_string(root().join(crate::DOCS).join(PAGE)).unwrap();
        assert!(
            committed == page,
            "website/docs/benchmarks.md is stale; run `task docs-generate`"
        );
        for (name, svg) in &images {
            let committed = fs::read_to_string(root().join(IMAGES).join(name)).unwrap();
            assert!(
                &committed == svg,
                "{name} is stale; run `task docs-generate`"
            );
        }
        let on_disk = fs::read_dir(root().join(IMAGES)).unwrap().count();
        assert_eq!(on_disk, images.len());
    }

    /// One fixture entry of `runs.yaml`: a run with one repeat that states only its median.
    fn entry(
        id: &str,
        date: &str,
        workload: &str,
        side: &str,
        point: Option<(u64, u64)>,
        flop: Option<(u64, &str)>,
        median: f64,
    ) -> String {
        let mut text = format!(
            "  - run_id: {id}\n    date: \"{date}\"\n    workload: {workload}\n    side: {side}\n    mode: Timed\n"
        );
        if let Some((batch, hidden)) = point {
            text.push_str(&format!("    batch: {batch}\n    hidden: {hidden}\n"));
        }
        if let Some((flop, rule)) = flop {
            text.push_str(&format!(
                "    flop_per_step: {flop}\n    flop_rule: {rule}\n"
            ));
        }
        text.push_str(&format!(
            "    repeat_count: 1\n    repeats_us_per_step: null\n    median_us_per_step: {median}\n    source: verification-report:fixture\n    state: Recorded\n"
        ));
        text
    }

    const SWEEP: &str = "mlp-784-H-10";

    /// A sweep at batch 100 whose FLOP chart points and GFLOP/s are worked out by hand in
    /// [`the_flop_chart_takes_the_latest_run_stating_flop_per_step_per_side_and_point`]: `3196 · 100
    /// · 64 = 20,454,400` and `3196 · 100 · 256 = 81,817,600` FLOP per step under `MatmulOnly`.
    fn flop_fixture() -> String {
        let matmul_64 = Some((20_454_400, "MatmulOnly"));
        let matmul_256 = Some((81_817_600, "MatmulOnly"));
        let mut text = "runs:\n".to_owned();
        for run in [
            entry(
                "mnist",
                "2026-10-21",
                "mnist-784-64-10",
                "ModelSpecVulkan",
                None,
                None,
                40.0,
            ),
            entry(
                "ms-64-old",
                "2026-10-24",
                SWEEP,
                "ModelSpecVulkan",
                Some((100, 64)),
                matmul_64,
                30.0,
            ),
            entry(
                "ms-64",
                "2026-10-25",
                SWEEP,
                "ModelSpecVulkan",
                Some((100, 64)),
                matmul_64,
                25.0,
            ),
            entry(
                "eager-64",
                "2026-10-25",
                SWEEP,
                "PyTorchCudaEager",
                Some((100, 64)),
                matmul_64,
                400.0,
            ),
            entry(
                "ms-256",
                "2026-10-25",
                SWEEP,
                "ModelSpecVulkan",
                Some((100, 256)),
                matmul_256,
                40.0,
            ),
            entry(
                "eager-256-flop",
                "2026-10-24",
                SWEEP,
                "PyTorchCudaEager",
                Some((100, 256)),
                matmul_256,
                450.0,
            ),
            entry(
                "eager-256",
                "2026-10-25",
                SWEEP,
                "PyTorchCudaEager",
                Some((100, 256)),
                None,
                300.0,
            ),
            entry(
                "compile-256",
                "2026-10-25",
                SWEEP,
                "PyTorchCudaCompile",
                Some((100, 256)),
                Some((90_000_000, "AllSites")),
                500.0,
            ),
        ] {
            text.push_str(&run);
        }
        text
    }

    /// A FLOP chart point as the tests compare it: side, batch, hidden width, FLOP per step, µs per
    /// step, run and GFLOP/s.
    type Row<'a> = (&'a str, Option<u64>, Option<u64>, u64, f64, &'a str, String);

    #[test]
    fn the_flop_chart_takes_the_latest_run_stating_flop_per_step_per_side_and_point() {
        let runs = read(&flop_fixture(), &schema()).unwrap();
        let sweep: Vec<&Run> = runs.iter().filter(|run| run.workload == SWEEP).collect();
        let chart = flop_points(&sweep);
        let points: Vec<Row<'_>> = chart
            .points
            .iter()
            .map(|point| {
                (
                    point.side.as_str(),
                    point.batch,
                    point.hidden,
                    point.flop_per_step,
                    point.us,
                    point.run_id.as_str(),
                    gflops_text(Some(point.gflops())),
                )
            })
            .collect();
        // ModelSpec at H 64 is its later run; PyTorch eager at H 256 is the 2026-10-24 run,
        // because its later run states no flop_per_step; torch.compile at H 256 counts by
        // AllSites, which is not the ModelSpec run's rule, so it is placed at the ModelSpec run's
        // 81,817,600 FLOP: 81,817,600 / 500 / 1000 = 163.6352 GFLOP/s.
        assert_eq!(
            points,
            vec![
                (
                    "ModelSpecVulkan",
                    Some(100),
                    Some(64),
                    20_454_400,
                    25.0,
                    "ms-64",
                    "818.2".to_owned()
                ),
                (
                    "ModelSpecVulkan",
                    Some(100),
                    Some(256),
                    81_817_600,
                    40.0,
                    "ms-256",
                    "2,045.4".to_owned()
                ),
                (
                    "PyTorchCudaEager",
                    Some(100),
                    Some(64),
                    20_454_400,
                    400.0,
                    "eager-64",
                    "51.1".to_owned()
                ),
                (
                    "PyTorchCudaEager",
                    Some(100),
                    Some(256),
                    81_817_600,
                    450.0,
                    "eager-256-flop",
                    "181.8".to_owned()
                ),
                (
                    "PyTorchCudaCompile",
                    Some(100),
                    Some(256),
                    81_817_600,
                    500.0,
                    "compile-256",
                    "163.6".to_owned()
                ),
            ]
        );
        // Every point is placed by MatmulOnly; the one point whose sides differ names both.
        assert_eq!(
            chart.rules.iter().map(String::as_str).collect::<Vec<_>>(),
            ["MatmulOnly"]
        );
        let mixed: Vec<(Option<u64>, Option<u64>, Vec<&str>)> = chart
            .mixed
            .iter()
            .map(|(batch, hidden, rules)| {
                (*batch, *hidden, rules.iter().map(String::as_str).collect())
            })
            .collect();
        assert_eq!(
            mixed,
            vec![(Some(100), Some(256), vec!["AllSites", "MatmulOnly"])]
        );
        assert_eq!(gflops_text(None), "");
        assert_eq!(gflops_text(Some(14_834.712_3)), "14,834.7");
        assert_eq!(gflops_text(Some(180.0)), "180.0");
    }

    #[test]
    fn the_sweep_tables_carry_gflops_per_side_blank_where_the_run_states_no_flop() {
        let runs = read(&flop_fixture(), &schema()).unwrap();
        let (page, images) = render(&runs);
        assert!(
            page.contains(
                "| H | FLOP per step | ModelSpec Vulkan | PyTorch CUDA eager | PyTorch CUDA torch.compile | ModelSpec Vulkan GFLOP/s | PyTorch CUDA eager GFLOP/s | PyTorch CUDA torch.compile GFLOP/s |\n|---|---|---|---|---|---|---|---|\n"
            ),
            "{page}"
        );
        // The table shows the latest run of each side: PyTorch eager's H 256 run of 2026-10-25
        // states no flop_per_step, so its GFLOP/s is blank; torch.compile's is its own
        // 90,000,000 / 500 / 1000.
        assert!(
            page.contains("| 64 | 20,454,400 | 25 | 400 |  | 818.2 | 51.1 |  |\n"),
            "{page}"
        );
        assert!(
            page.contains("| 256 | 81,817,600 | 40 | 300 | 500 | 2,045.4 |  | 180.0 |\n"),
            "{page}"
        );
        let chart = &images["mlp-784-h-10-flop.svg"];
        assert!(
            page.contains("(/img/benchmarks/mlp-784-h-10-flop.svg)"),
            "{page}"
        );
        assert!(
            chart.contains(">FLOP per step, log scale</text>"),
            "{chart}"
        );
        for label in [
            "ModelSpec Vulkan",
            "PyTorch CUDA eager",
            "PyTorch CUDA torch.compile",
        ] {
            assert!(
                chart.contains(&format!(">{label}, batch 100</text>")),
                "{label}"
            );
        }
        // Two ModelSpec points and its legend marker; three opponent points and two legend
        // markers.
        assert_eq!(chart.matches("r=\"4.5\"").count(), 3, "{chart}");
        assert_eq!(
            chart.matches("width=\"8\" height=\"8\"").count(),
            5,
            "{chart}"
        );
        assert!(
            page.contains("The points count FLOP by `MatmulOnly`"),
            "{page}"
        );
        assert!(
            page.contains(
                "At batch 100, H 256 the sides count by different rules (`AllSites`, `MatmulOnly`)"
            ),
            "{page}"
        );
        assert!(!images.contains_key("mnist-784-64-10-flop.svg"));
    }

    /// The `## Not measured` section of `page`, to the end of the page, or none.
    fn not_measured(page: &str) -> Option<&str> {
        page.find("\n## Not measured\n").map(|at| &page[at + 1..])
    }

    #[test]
    fn a_declared_workload_without_a_run_is_listed_as_not_measured_and_leaves_with_its_first_run() {
        let runs = read(&flop_fixture(), &schema()).unwrap();
        let (page, _) = render(&runs);
        let section = not_measured(&page).unwrap_or_else(|| panic!("no section: {page}"));
        assert!(
            !section[1..].contains("\n## "),
            "not the last section: {section}"
        );
        assert!(section.contains("**Transformer block** (`transformer-block`)"));
        assert!(
            section
                .contains("the transformer-block benchmark on synthetic data against PyTorch CUDA")
        );
        assert!(section.contains("**BERT-tiny pre-training** (`bert-tiny-pretrain`)"));
        assert!(section.contains(
            "BERT-tiny pre-training timed to the same held-out loss as PyTorch CUDA with the same corpus, batches, initial weights and optimizer"
        ));
        // Public prose names no planning record (coordinator decision D80, ADV-BP-5).
        for declared in &WORKLOADS {
            for kind in ["story:", "epic:", "verification-report:"] {
                assert!(
                    !declared.not_measured.contains(kind),
                    "{}: {kind}",
                    declared.id
                );
            }
        }
        for measured in ["MNIST", "MLP 784-H-10", "mlp-784-H-10", "![", "|"] {
            assert!(!section.contains(measured), "{measured} in {section}");
        }
        assert_eq!(section.matches("\n- ").count(), 2, "{section}");

        let with_block = format!(
            "{}{}",
            flop_fixture(),
            entry(
                "block",
                "2026-10-27",
                "transformer-block",
                "ModelSpecVulkan",
                None,
                None,
                900.0
            )
            .replace(
                "    mode: Timed\n",
                "    mode: Timed\n    batch: 8\n    tokens_per_step: 1024\n"
            )
        );
        let runs = read(&with_block, &schema()).unwrap();
        let (page, images) = render(&runs);
        assert!(page.contains("\n## Transformer block\n"), "{page}");
        assert!(images.contains_key("transformer-block-batch.svg"));
        let section = not_measured(&page).unwrap();
        assert!(!section.contains("Transformer block"), "{section}");
        assert!(section.contains("**BERT-tiny pre-training**"), "{section}");
        assert_eq!(section.matches("\n- ").count(), 1, "{section}");

        let with_both = format!(
            "{with_block}{}",
            bert_entry("bert", "ModelSpecVulkan", 5000.0, (3400, 28.5), 84.25, 6.8)
        );
        let runs = read(&with_both, &schema()).unwrap();
        let (page, _) = render(&runs);
        assert!(
            page.contains("\n## BERT-tiny pre-training, time to the held-out baseline\n"),
            "{page}"
        );
        assert!(not_measured(&page).is_none(), "{page}");
    }

    #[test]
    fn the_published_page_has_the_flop_chart_of_the_sweep_and_every_declared_workload_measured() {
        let text = fs::read_to_string(root().join(RUNS)).unwrap();
        let runs = read(&text, &schema()).unwrap();
        let sweep: Vec<&Run> = runs.iter().filter(|run| run.workload == SWEEP).collect();
        let chart = flop_points(&sweep);
        assert_eq!(chart.points.len(), 36, "3 sides × 3 batches × 4 widths");
        assert_eq!(
            chart.rules.iter().map(String::as_str).collect::<Vec<_>>(),
            ["MatmulOnly"]
        );
        assert!(chart.mixed.is_empty());
        let largest: Vec<(&str, &str, String)> = chart
            .points
            .iter()
            .filter(|point| point.batch == Some(8192) && point.hidden == Some(4096))
            .map(|point| {
                assert_eq!(point.flop_per_step, 3196 * 8192 * 4096);
                (
                    point.side.as_str(),
                    point.run_id.as_str(),
                    gflops_text(Some(point.gflops())),
                )
            })
            .collect();
        assert_eq!(
            largest,
            [
                (
                    "ModelSpecVulkan",
                    "sweep-b8192-h4096-modelspec-vulkan-step-sites",
                    "15,635.1".to_owned()
                ),
                (
                    "PyTorchCudaEager",
                    "sweep-b8192-h4096-pytorch-cuda-eager-step-sites",
                    "16,326.4".to_owned()
                ),
                (
                    "PyTorchCudaCompile",
                    "sweep-b8192-h4096-pytorch-cuda-compile-0.10.0",
                    "15,812.4".to_owned()
                ),
            ]
        );
        let (page, images) = files(&root()).unwrap();
        assert!(images.contains_key("mlp-784-h-10-flop.svg"));
        assert!(!page.contains("the sides count by different rules"));
        // Both transformer workloads have runs since verification-report:bert-tiny-vs-pytorch-
        // 2026-10-28, so no declared workload is left unmeasured and the section is gone.
        assert!(page.contains("\n## Transformer block\n"));
        assert!(page.contains("\n## BERT-tiny pre-training, time to the held-out baseline\n"));
        assert!(not_measured(&page).is_none(), "{page}");
    }

    /// The published transformer-block section holds the nine medians of
    /// verification-report:transformer-block-benchmark-2026-10-27, with the tokens per second and
    /// GFLOP/s of its results table, in one chart series per side.
    #[test]
    fn the_published_page_has_the_transformer_block_figures_of_the_report() {
        let (page, images) = files(&root()).unwrap();
        let start = page
            .find("\n## Transformer block\n")
            .unwrap_or_else(|| panic!("no transformer-block section"));
        let end = page[start + 1..]
            .find("\n## ")
            .map_or(page.len(), |at| start + 1 + at);
        let section = &page[start..end];
        // Batch, tokens per step, side, build, µs per step, tokens per second, GFLOP/s.
        let expected = [
            "| 8 | 1,024 | ModelSpec Vulkan | 2026-10-27 | 0.20.0 (`a123c1a`) | 93,868.4 | 10,908.889 | 29.0 | 3 |",
            "| 8 | 1,024 | PyTorch CUDA eager | 2026-10-27 | torch 2.14.1+cu130, CUDA 13.0 | 3,937.184 | 260,084.353 | 690.3 | 3 |",
            "| 8 | 1,024 | PyTorch CUDA torch.compile | 2026-10-27 | torch 2.14.1+cu130, CUDA 13.0 | 1,971.859 | 519,306.914 | 1,378.3 | 3 |",
            "| 32 | 4,096 | ModelSpec Vulkan | 2026-10-27 | 0.20.0 (`a123c1a`) | 188,352.069 | 21,746.509 | 57.7 | 3 |",
            "| 32 | 4,096 | PyTorch CUDA eager | 2026-10-27 | torch 2.14.1+cu130, CUDA 13.0 | 4,046.869 | 1,012,140.555 | 2,686.4 | 3 |",
            "| 32 | 4,096 | PyTorch CUDA torch.compile | 2026-10-27 | torch 2.14.1+cu130, CUDA 13.0 | 1,926.75 | 2,125,859.797 | 5,642.5 | 3 |",
            "| 128 | 16,384 | ModelSpec Vulkan | 2026-10-27 | 0.20.0 (`a123c1a`) | 641,794.458 | 25,528.422 | 67.8 | 3 |",
            "| 128 | 16,384 | PyTorch CUDA eager | 2026-10-27 | torch 2.14.1+cu130, CUDA 13.0 | 12,091.141 | 1,355,041.638 | 3,596.6 | 3 |",
            "| 128 | 16,384 | PyTorch CUDA torch.compile | 2026-10-27 | torch 2.14.1+cu130, CUDA 13.0 | 5,138.515 | 3,188,469.97 | 8,462.9 | 3 |",
        ];
        let table: Vec<&str> = section
            .lines()
            .filter(|line| line.starts_with("| ") && !line.starts_with("| Batch"))
            .collect();
        assert_eq!(table, expected, "{section}");
        let chart = &images["transformer-block-batch.svg"];
        for label in [
            "ModelSpec Vulkan",
            "PyTorch CUDA eager",
            "PyTorch CUDA torch.compile",
        ] {
            assert!(chart.contains(label), "{label}");
        }
    }

    #[test]
    fn a_run_of_an_undeclared_workload_is_refused_with_its_line_and_workload() {
        let text = format!(
            "runs:\n{}",
            entry(
                "typo",
                "2026-10-27",
                "transformer_block",
                "ModelSpecVulkan",
                None,
                None,
                900.0
            )
        );
        let error = read(&text, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:4: runs[0] (typo) field `workload`:"),
            "{error}"
        );
        assert!(error.contains("`transformer_block`"), "{error}");
        assert!(error.contains("transformer-block"), "{error}");
    }

    /// A `transformer-block` run at `batch` sequences, with its tokens and `MatmulOnly` FLOP per
    /// step where given.
    fn block_entry(
        id: &str,
        side: &str,
        batch: Option<u64>,
        tokens: Option<u64>,
        flop: Option<u64>,
        median: f64,
    ) -> String {
        let mut text = format!(
            "  - run_id: {id}\n    date: \"2026-10-27\"\n    workload: transformer-block\n    side: {side}\n    mode: Timed\n"
        );
        if let Some(batch) = batch {
            text.push_str(&format!("    batch: {batch}\n"));
        }
        if let Some(tokens) = tokens {
            text.push_str(&format!("    tokens_per_step: {tokens}\n"));
        }
        if let Some(flop) = flop {
            text.push_str(&format!(
                "    flop_per_step: {flop}\n    flop_rule: MatmulOnly\n"
            ));
        }
        text.push_str(&format!(
            "    repeat_count: 1\n    repeats_us_per_step: null\n    median_us_per_step: {median}\n    source: verification-report:fixture\n    state: Recorded\n"
        ));
        text
    }

    /// `MatmulOnly` is `339,738,624·B` for `transformer-block`: 2,717,908,992 at batch 8 stands,
    /// another count or a run without a batch is refused, and so is `MatmulOnly` on a workload
    /// it has no formula for.
    #[test]
    fn a_transformer_block_matmul_only_count_is_its_formula_at_the_batch() {
        let at_8 = format!(
            "runs:\n{}",
            block_entry(
                "b8",
                "ModelSpecVulkan",
                Some(8),
                Some(1024),
                Some(2_717_908_992),
                1024.0
            )
        );
        assert_eq!(
            read(&at_8, &schema()).unwrap()[0].tokens_per_step,
            Some(1024)
        );
        let at_128 = format!(
            "runs:\n{}",
            block_entry(
                "b128",
                "PyTorchCudaCompile",
                Some(128),
                Some(16_384),
                Some(43_486_543_872),
                1.0
            )
        );
        assert!(read(&at_128, &schema()).is_ok());

        let wrong = format!(
            "runs:\n{}",
            block_entry(
                "wrong",
                "ModelSpecVulkan",
                Some(8),
                None,
                Some(3196 * 8 * 128),
                1.0
            )
        );
        let error = read(&wrong, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:8: runs[0] (wrong) field `flop_per_step`:"),
            "{error}"
        );
        assert!(
            error.contains("339738624·B = 2717908992 at batch 8"),
            "{error}"
        );

        let no_batch = format!(
            "runs:\n{}",
            block_entry(
                "no-batch",
                "ModelSpecVulkan",
                None,
                None,
                Some(2_717_908_992),
                1.0
            )
        );
        let error = read(&no_batch, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:7: runs[0] (no-batch) field `flop_per_step`:"),
            "{error}"
        );
        assert!(error.contains("must give batch"), "{error}");

        let mnist = format!(
            "runs:\n{}",
            entry(
                "mnist",
                "2026-10-27",
                "mnist-784-64-10",
                "ModelSpecVulkan",
                None,
                Some((1_000, "MatmulOnly")),
                10.0
            )
        );
        let error = read(&mnist, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:7: runs[0] (mnist) field `flop_per_step`:"),
            "{error}"
        );
        assert!(error.contains("not of `mnist-784-64-10`"), "{error}");
    }

    #[test]
    fn transformer_block_tokens_that_are_not_128_per_sequence_are_refused() {
        let text = format!(
            "runs:\n{}",
            block_entry("tokens", "ModelSpecVulkan", Some(8), Some(1000), None, 1.0)
        );
        let error = read(&text, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:8: runs[0] (tokens) field `tokens_per_step`:"),
            "{error}"
        );
        let text = format!(
            "runs:\n{}",
            block_entry("zero", "ModelSpecVulkan", None, Some(0), None, 1.0)
        );
        let error = read(&text, &schema()).err().unwrap();
        assert!(error.contains("field `tokens_per_step`"), "{error}");
    }

    /// ADV-TB-2 and ADV-TB-3 (coordinator decision D84): a `transformer-block` run without
    /// `tokens_per_step` or without `batch` is refused at the entry's first line, naming the
    /// missing field, and tokens that are no whole number of sequences of 128 are refused at
    /// their field without a batch too.
    #[test]
    fn a_transformer_block_run_without_batch_or_tokens_is_refused_at_its_entry() {
        let without_tokens = format!(
            "runs:\n{}",
            block_entry(
                "no-tokens",
                "ModelSpecVulkan",
                Some(8),
                None,
                Some(2_717_908_992),
                1.0
            )
        );
        let error = read(&without_tokens, &schema()).err().unwrap();
        assert_eq!(
            error,
            "benchmarks/runs.yaml:2: runs[0] (no-tokens) field `tokens_per_step`: is missing; a \
             `transformer-block` run must give tokens_per_step, the 128·B tokens of one step"
        );

        let without_batch = format!(
            "runs:\n{}",
            block_entry("no-batch", "ModelSpecVulkan", None, Some(1024), None, 1.0)
        );
        let error = read(&without_batch, &schema()).err().unwrap();
        assert_eq!(
            error,
            "benchmarks/runs.yaml:2: runs[0] (no-batch) field `batch`: is missing; a \
             `transformer-block` run must give batch, the sequences of 128 tokens of one step"
        );

        let neither = format!(
            "runs:\n{}",
            block_entry("neither", "ModelSpecVulkan", None, Some(1000), None, 1.0)
        );
        let error = read(&neither, &schema()).err().unwrap();
        let lines: Vec<&str> = error.lines().collect();
        assert_eq!(lines.len(), 2, "{error}");
        assert!(
            lines[0].starts_with("benchmarks/runs.yaml:2: runs[0] (neither) field `batch`:"),
            "{error}"
        );
        assert!(
            lines[1].starts_with(
                "benchmarks/runs.yaml:7: runs[0] (neither) field `tokens_per_step`: 1000 is no whole number"
            ),
            "{error}"
        );

        let whole = format!(
            "runs:\n{}",
            block_entry("whole", "ModelSpecVulkan", Some(8), Some(1024), None, 1.0)
        );
        assert!(read(&whole, &schema()).is_ok());
    }

    /// A workload whose runs state tokens per step gets a section of its own: a chart against
    /// batch and the latest run of each side and batch with its tokens per second and GFLOP/s.
    #[test]
    fn a_transformer_block_section_shows_tokens_per_second_per_side_and_batch() {
        let text = format!(
            "{}{}{}",
            flop_fixture(),
            block_entry(
                "block-ms-8",
                "ModelSpecVulkan",
                Some(8),
                Some(1024),
                Some(2_717_908_992),
                1024.0
            ),
            block_entry(
                "block-pt-8",
                "PyTorchCudaEager",
                Some(8),
                Some(1024),
                Some(2_717_908_992),
                4096.0
            )
        );
        let runs = read(&text, &schema()).unwrap();
        let (page, images) = render(&runs);
        let start = page
            .find("\n## Transformer block\n")
            .unwrap_or_else(|| panic!("{page}"));
        let end = page[start + 1..]
            .find("\n## ")
            .map_or(page.len(), |at| start + 1 + at);
        let section = &page[start..end];
        assert!(images.contains_key("transformer-block-batch.svg"));
        assert!(!images.contains_key("transformer-block.svg"));
        assert!(
            section.contains("| Batch | Tokens per step | Side | Date | Build | µs per step | Tokens per second | GFLOP/s | Repeats |"),
            "{section}"
        );
        assert!(
            section.contains("| 8 | 1,024 | ModelSpec Vulkan | 2026-10-27 |  | 1,024 | 1,000,000 | 2,654.2 | 1, not recorded |"),
            "{section}"
        );
        assert!(
            section.contains("| 8 | 1,024 | PyTorch CUDA eager | 2026-10-27 |  | 4,096 | 250,000 | 663.6 | 1, not recorded |"),
            "{section}"
        );
        assert!(section.contains("339,738,624·B"), "{section}");
        let not_measured = not_measured(&page).unwrap();
        assert!(
            !not_measured.contains("Transformer block"),
            "{not_measured}"
        );
    }

    /// A `bert-tiny-pretrain` run of `side` (story:bert-tiny-vs-pytorch-cuda) with every figure a
    /// time-to-target run states: 32 sequences of 128, its `MatmulOnly` FLOP
    /// (`1,170,210,816·32`), the target, the steps and training seconds to it, and the final
    /// held-out loss. Its lines, from 2 under `runs:`: 7 batch, 8 tokens_per_step, 9
    /// flop_per_step, 11 target_loss, 12 steps_to_target, 13 seconds_to_target.
    fn bert_entry(
        id: &str,
        side: &str,
        median: f64,
        to_target: (u64, f64),
        train_seconds: f64,
        final_loss: f64,
    ) -> String {
        let (steps, seconds) = to_target;
        let version = if is_modelspec(side) {
            "    modelspec_version: 0.22.0\n    commit: 0123456789abcdef\n".to_owned()
        } else {
            "    opponent_version: torch 2.14.1+cu130\n".to_owned()
        };
        format!(
            "  - run_id: {id}\n    date: \"2026-10-28\"\n    workload: bert-tiny-pretrain\n    side: {side}\n    mode: Timed\n    batch: 32\n    tokens_per_step: 4096\n    flop_per_step: 37446746112\n    flop_rule: MatmulOnly\n    target_loss: 6.926537926202018\n    steps_to_target: {steps}\n    seconds_to_target: {seconds}\n    train_seconds: {train_seconds}\n    eval_every: 100\n    final_heldout_loss: {final_loss}\n{version}    repeat_count: 3\n    repeats_us_per_step: null\n    median_us_per_step: {median}\n    source: verification-report:bert-fixture\n    state: Recorded\n"
        )
    }

    fn bert_run(side: &str) -> String {
        format!(
            "runs:\n{}",
            bert_entry("bert", side, 8192.0, (3400, 28.5), 84.25, 6.8012345)
        )
    }

    /// A `bert-tiny-pretrain` run states its target and the steps and training seconds to it, and
    /// its 4,096 tokens per step, 32 sequences of 128; without one of them, or with other tokens,
    /// another batch or another `MatmulOnly` count, it is refused at the field.
    #[test]
    fn a_bert_tiny_pretrain_run_without_its_target_figures_or_with_other_tokens_is_refused_at_the_field()
     {
        let run = read(&bert_run("ModelSpecVulkan"), &schema()).unwrap();
        assert_eq!(run[0].target_loss, Some(6.926537926202018));
        assert_eq!(run[0].steps_to_target, Some(3400));
        assert_eq!(run[0].seconds_to_target, Some(28.5));
        assert_eq!(run[0].train_seconds, Some(84.25));
        assert_eq!(run[0].eval_every, Some(100));
        assert_eq!(run[0].final_heldout_loss, Some(6.8012345));
        assert!(read(&bert_run("PyTorchCudaCompile"), &schema()).is_ok());

        for (line, field) in [
            ("    target_loss: 6.926537926202018\n", "target_loss"),
            ("    steps_to_target: 3400\n", "steps_to_target"),
            ("    seconds_to_target: 28.5\n", "seconds_to_target"),
            ("    tokens_per_step: 4096\n", "tokens_per_step"),
        ] {
            let text = bert_run("ModelSpecVulkan").replace(line, "");
            let error = read(&text, &schema()).err().unwrap();
            assert!(
                error.starts_with(&format!(
                    "benchmarks/runs.yaml:2: runs[0] (bert) field `{field}`: is missing; a \
                     `bert-tiny-pretrain` run must give {field}"
                )),
                "{error}"
            );
            assert_eq!(error.lines().count(), 1, "{error}");
        }

        let tokens =
            bert_run("ModelSpecVulkan").replace("tokens_per_step: 4096", "tokens_per_step: 1024");
        assert_eq!(
            read(&tokens, &schema()).err().unwrap(),
            "benchmarks/runs.yaml:8: runs[0] (bert) field `tokens_per_step`: 1024 is not the 4096 \
             tokens of a `bert-tiny-pretrain` step, 32 sequences of 128"
        );

        let batch = bert_run("ModelSpecVulkan")
            .replace("batch: 32", "batch: 8")
            .replace("flop_per_step: 37446746112", "flop_per_step: 9361686528");
        let error = read(&batch, &schema()).err().unwrap();
        assert!(
            error.starts_with(
                "benchmarks/runs.yaml:7: runs[0] (bert) field `batch`: 8 is not the 32 sequences"
            ),
            "{error}"
        );

        let flop = bert_run("ModelSpecVulkan")
            .replace("flop_per_step: 37446746112", "flop_per_step: 10871635968");
        let error = read(&flop, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:9: runs[0] (bert) field `flop_per_step`:"),
            "{error}"
        );
        assert!(
            error.contains("1170210816·B = 37446746112 at batch 32"),
            "{error}"
        );

        let seconds =
            bert_run("ModelSpecVulkan").replace("seconds_to_target: 28.5", "seconds_to_target: 0");
        assert_eq!(
            read(&seconds, &schema()).err().unwrap(),
            "benchmarks/runs.yaml:13: runs[0] (bert) field `seconds_to_target`: must be a \
             positive number"
        );
        let after =
            bert_run("ModelSpecVulkan").replace("seconds_to_target: 28.5", "seconds_to_target: 90");
        assert_eq!(
            read(&after, &schema()).err().unwrap(),
            "benchmarks/runs.yaml:13: runs[0] (bert) field `seconds_to_target`: 90 is more than \
             the run's train_seconds 84.25"
        );
        let zero =
            bert_run("ModelSpecVulkan").replace("steps_to_target: 3400", "steps_to_target: 0");
        let error = read(&zero, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:12: runs[0] (bert) field `steps_to_target`:"),
            "{error}"
        );
    }

    /// A `bert-tiny-pretrain` run is counted by `MatmulOnly` at `1,170,210,816·B`, the one count
    /// all three sides share (ADV-BP-4), and its steps to target is a step it evaluated, a
    /// multiple of its `eval_every` (ADV-BP-5); another rule, no rule, no `eval_every` or a step
    /// between two evaluations is refused at the field (story:bert-tiny-vs-pytorch-cuda).
    #[test]
    fn a_bert_tiny_pretrain_run_counted_otherwise_or_off_its_evaluations_is_refused_at_the_field() {
        let all_sites = bert_run("ModelSpecVulkan")
            .replace("flop_per_step: 37446746112", "flop_per_step: 39400695809")
            .replace("flop_rule: MatmulOnly", "flop_rule: AllSites");
        assert_eq!(
            read(&all_sites, &schema()).err().unwrap(),
            "benchmarks/runs.yaml:10: runs[0] (bert) field `flop_rule`: `AllSites` is not the \
             `MatmulOnly` rule of a `bert-tiny-pretrain` run, 1170210816·B, the count all its \
             sides share"
        );
        for (line, field) in [
            ("    flop_rule: MatmulOnly\n", "flop_rule"),
            ("    flop_per_step: 37446746112\n", "flop_per_step"),
            ("    eval_every: 100\n", "eval_every"),
        ] {
            let text = bert_run("ModelSpecVulkan").replace(line, "");
            let error = read(&text, &schema()).err().unwrap();
            assert!(
                error.contains(&format!(
                    "field `{field}`: is missing; a `bert-tiny-pretrain` run must give {field}"
                )),
                "{error}"
            );
        }
        let off_grid =
            bert_run("ModelSpecVulkan").replace("steps_to_target: 3400", "steps_to_target: 3450");
        assert_eq!(
            read(&off_grid, &schema()).err().unwrap(),
            "benchmarks/runs.yaml:12: runs[0] (bert) field `steps_to_target`: 3450 is no step the \
             run evaluated, a multiple of its eval_every 100"
        );
    }

    /// A `bert-tiny-pretrain` run gets the section "BERT-tiny pre-training, time to the held-out
    /// baseline": a table per side with µs per step, tokens per second, GFLOP/s, the target, the
    /// steps and training seconds to it and the final held-out loss, each linked to the run's
    /// report, and no chart.
    #[test]
    fn a_bert_tiny_pretrain_section_shows_time_to_target_per_side_linked_to_the_report() {
        let text = format!(
            "{}{}{}",
            flop_fixture(),
            bert_entry(
                "bert-ms",
                "ModelSpecVulkan",
                8192.0,
                (3400, 28.5),
                84.25,
                6.8012345
            ),
            bert_entry(
                "bert-pt",
                "PyTorchCudaEager",
                16384.0,
                (3400, 57.0),
                170.0,
                6.8011999
            )
        );
        let runs = read(&text, &schema()).unwrap();
        let (page, images) = render(&runs);
        let start = page
            .find("\n## BERT-tiny pre-training, time to the held-out baseline\n")
            .unwrap_or_else(|| panic!("{page}"));
        let end = page[start + 1..]
            .find("\n## ")
            .map_or(page.len(), |at| start + 1 + at);
        let section = &page[start..end];
        assert!(
            !images
                .keys()
                .any(|name| name.starts_with("bert-tiny-pretrain")),
            "{:?}",
            images.keys()
        );
        let report = format!(
            "{}/.engineering/planning/verification-report/bert-fixture.md",
            crate::SOURCE
        );
        let header = "| Date | Build | µs per step | Tokens per second | GFLOP/s | Target loss | Steps to target | Training seconds to target | Final held-out loss | Eval every |";
        assert_eq!(section.matches(header).count(), 2, "{section}");
        let ours = section
            .find("\n### ModelSpec Vulkan\n")
            .unwrap_or_else(|| panic!("{section}"));
        let theirs = section
            .find("\n### PyTorch CUDA eager\n")
            .unwrap_or_else(|| panic!("{section}"));
        assert!(ours < theirs, "{section}");
        let row = |build: &str, figures: [&str; 7]| {
            let linked: Vec<String> = figures
                .iter()
                .map(|figure| format!("[{figure}]({report})"))
                .collect();
            format!("| 2026-10-28 | {build} | {} | 100 |", linked.join(" | "))
        };
        let ms = row(
            "0.22.0 (`0123456`)",
            [
                "8,192",
                "500,000",
                "4,571.1",
                "6.926537926202018",
                "3,400",
                "28.5",
                "6.8012",
            ],
        );
        let pt = row(
            "torch 2.14.1+cu130",
            [
                "16,384",
                "250,000",
                "2,285.6",
                "6.926537926202018",
                "3,400",
                "57",
                "6.8012",
            ],
        );
        assert!(section[ours..theirs].contains(&ms), "{ms}\n{section}");
        assert!(section[theirs..].contains(&pt), "{pt}\n{section}");
        assert!(section.contains("1,170,210,816·B"), "{section}");
        assert!(
            section.contains(
                "µs per step is a run's training seconds over its steps, the mean of its steps, \
                 and the figure is the median of 3 runs"
            ),
            "{section}"
        );
        assert!(section.contains("`bert-tiny-pretrain`"), "{section}");
        for kind in ["story:", "epic:", "verification-report:"] {
            assert!(!section.contains(kind), "{kind} in {section}");
        }
        assert!(
            not_measured(&page).is_none_or(|rest| !rest.contains("**BERT-tiny pre-training**"))
        );
    }

    /// The method text of the `bert-tiny-pretrain` section says which report's clock holds the
    /// held-out evaluations: verification-report:bert-tiny-vs-pytorch-2026-10-28 excludes them on
    /// every side (its "training clock" row), verification-report:bert-tiny-vs-pytorch-2026-10-29
    /// on the PyTorch sides only, its ModelSpec clock "the 101 held-out evaluations included",
    /// each report linked.
    #[test]
    fn the_bert_tiny_pretrain_method_says_which_report_clocks_the_evaluations() {
        let text = format!(
            "{}{}",
            flop_fixture(),
            bert_entry(
                "bert-ms",
                "ModelSpecVulkan",
                8192.0,
                (3400, 28.5),
                84.25,
                6.8012345
            )
        );
        let runs = read(&text, &schema()).unwrap();
        let (page, _) = render(&runs);
        let report = |date: &str| {
            report_url(&format!("verification-report:bert-tiny-vs-pytorch-{date}")).unwrap()
        };
        let expected = format!(
            "training seconds to target the training time up to that step without checkpoint \
             writes or events. The held-out evaluations are outside that time on every side in \
             [the report of 2026-10-28]({}) and on the PyTorch sides in [the report of \
             2026-10-29]({}); the ModelSpec runs of 2026-10-29 reduce each evaluation on the \
             device inside the evaluated step's command buffer, so their training seconds, and \
             with them their µs per step, include the evaluations. µs per step",
            report("2026-10-28"),
            report("2026-10-29")
        );
        assert!(page.contains(&expected), "{expected}\n{page}");
        assert!(
            !page.contains("without the held-out evaluations, checkpoint writes or events"),
            "{page}"
        );
    }

    #[test]
    fn a_matmul_only_count_that_is_not_3196_b_h_is_refused_with_its_line_and_field() {
        // 3196 · 8192 · 4096 = 107,239,964,672; twice that is a multiply-add counted as two.
        let text = format!(
            "runs:\n{}",
            entry(
                "doubled",
                "2026-10-25",
                SWEEP,
                "PyTorchCudaEager",
                Some((8192, 4096)),
                Some((214_479_929_344, "MatmulOnly")),
                6568.5
            )
        );
        let error = read(&text, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:9: runs[0] (doubled) field `flop_per_step`:"),
            "{error}"
        );
        assert!(error.contains("107239964672"), "{error}");
        let text = format!(
            "runs:\n{}",
            entry(
                "no-width",
                "2026-10-25",
                SWEEP,
                "ModelSpecVulkan",
                None,
                Some((1_000, "MatmulOnly")),
                10.0
            )
        );
        let error = read(&text, &schema()).err().unwrap();
        assert!(
            error.starts_with("benchmarks/runs.yaml:7: runs[0] (no-width) field `flop_per_step`:"),
            "{error}"
        );
        // AllSites is not a formula of B and H, so any count stands.
        let text = format!(
            "runs:\n{}",
            entry(
                "all-sites",
                "2026-10-25",
                SWEEP,
                "ModelSpecVulkan",
                Some((100, 64)),
                Some((1_000, "AllSites")),
                10.0
            )
        );
        assert!(read(&text, &schema()).is_ok());
    }

    /// Three sides at batches 100, 1000 and 8192 and widths 64 and 256, each counted by
    /// `MatmulOnly`.
    fn grid_fixture() -> String {
        let mut text = "runs:\n".to_owned();
        for (rank, side) in ["ModelSpecVulkan", "PyTorchCudaEager", "PyTorchCudaCompile"]
            .into_iter()
            .enumerate()
        {
            for batch in [100, 1000, 8192] {
                for hidden in [64, 256] {
                    text.push_str(&entry(
                        &format!("{side}-{batch}-{hidden}"),
                        "2026-10-25",
                        SWEEP,
                        side,
                        Some((batch, hidden)),
                        Some((3196 * batch * hidden, "MatmulOnly")),
                        (rank as f64 + 1.0) * batch as f64 * hidden as f64 / 1000.0,
                    ));
                }
            }
        }
        text
    }

    /// The centre x of every ModelSpec point of `svg`, the legend's marker at x 30 left out.
    fn modelspec_xs(svg: &str) -> Vec<String> {
        attribute_values(svg, "cx")
            .into_iter()
            .filter(|x| *x != "30.0")
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn the_flop_chart_joins_each_side_and_batch_by_its_own_line() {
        let runs = read(&grid_fixture(), &schema()).unwrap();
        let (_, images) = render(&runs);
        let chart = &images["mlp-784-h-10-flop.svg"];
        assert_eq!(
            chart.matches("<polyline ").count(),
            9,
            "3 sides × 3 batches: {chart}"
        );
        for side in [
            "ModelSpec Vulkan",
            "PyTorch CUDA eager",
            "PyTorch CUDA torch.compile",
        ] {
            for batch in [100, 1000, 8192] {
                assert!(
                    chart.contains(&format!(">{side}, batch {batch}</text>")),
                    "{side}, batch {batch}: {chart}"
                );
            }
        }
    }

    #[test]
    fn a_flop_chart_point_lies_at_the_log10_of_its_flop_count() {
        let runs = read(&grid_fixture(), &schema()).unwrap();
        let (page, images) = render(&runs);
        let chart = &images["mlp-784-h-10-flop.svg"];
        // The counts run from 3196 · 100 · 64 = 20,454,400 to 3196 · 8192 · 256 =
        // 6,702,497,792, so the axis spans 10⁷ to 10¹⁰ over the plot's 336 px from x 88:
        // x = 88 + 336 · (log10(FLOP) - 7) / 3, worked out by hand for the six counts.
        let mut xs = modelspec_xs(chart);
        xs.sort_by(|a, b| a.parse::<f64>().unwrap().total_cmp(&b.parse().unwrap()));
        assert_eq!(
            xs,
            ["122.8", "190.2", "234.8", "302.2", "337.1", "404.5"],
            "{chart}"
        );
        for decade in ["10⁷", "10⁸", "10⁹", "10¹⁰"] {
            assert!(chart.contains(&format!(">{decade}</text>")), "{decade}");
        }
        assert!(
            page.contains(
                "![MLP 784-H-10 sweep: µs per step against FLOP per step at batches 100, 1000 and 8192, both axes log scale]"
            ),
            "{page}"
        );
    }
}
