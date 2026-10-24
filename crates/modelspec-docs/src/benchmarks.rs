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
    pub flop_per_step: Option<u64>,
    pub flop_rule: Option<String>,
    pub timed_steps: Option<u64>,
    pub repeat_count: u64,
    pub repeats_us_per_step: Option<Vec<f64>>,
    pub median_us_per_step: f64,
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
const POSITIVE_INTEGERS: [&str; 6] = [
    "threads",
    "batch",
    "hidden",
    "flop_per_step",
    "timed_steps",
    "repeat_count",
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

/// What is wrong with a schema-valid run beyond the schema: the invariants of
/// `modelspec.benchmark.BenchmarkRun` (spec/domains/benchmark.yaml), checked here too so that they
/// hold whatever schema is read, its date, its repeats against its repeat count and its median
/// against the median of its repeats.
fn semantic_problems(run: &Run) -> Vec<(&'static str, String)> {
    let mut problems = Vec::new();
    if !is_date(&run.date) {
        problems.push(("date", format!("`{}` is not a `YYYY-MM-DD` date", run.date)));
    }
    let counts = [
        ("threads", run.threads),
        ("batch", run.batch),
        ("hidden", run.hidden),
        ("flop_per_step", run.flop_per_step),
        ("timed_steps", run.timed_steps),
        ("repeat_count", Some(run.repeat_count)),
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

/// A chart of µs per step on a log scale against categories on the x axis, at 0, 1, 2, ...
struct Chart {
    title: String,
    x_label: String,
    ticks: Vec<String>,
    series: Vec<Series>,
    /// Whether an opponent's last point is carried to the right edge as a dotted line, the
    /// figure ModelSpec is measured against until the opponent is measured again.
    carry_forward: bool,
}

const WIDTH: f64 = 880.0;
const LEFT: f64 = 78.0;
const RIGHT: f64 = 24.0;
const TOP: f64 = 48.0;
const PLOT_HEIGHT: f64 = 340.0;
const LEGEND_ROW: f64 = 20.0;

fn marker(out: &mut String, x: f64, y: f64, colour: &str, modelspec: bool) {
    if modelspec {
        out.push_str(&format!(
            "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"4.5\" fill=\"{colour}\"/>\n"
        ));
    } else {
        out.push_str(&format!(
            "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"8\" height=\"8\" fill=\"#ffffff\" stroke=\"{colour}\" stroke-width=\"2\"/>\n",
            x - 4.0,
            y - 4.0
        ));
    }
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
    let slots = chart.ticks.len().max(1) as f64;
    let x_of = |x: f64| LEFT + plot_width * (x + 0.5) / slots;
    let legend_rows = chart.series.len().div_ceil(2) as f64;
    let height = TOP + PLOT_HEIGHT + 60.0 + legend_rows * LEGEND_ROW + 12.0;
    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{WIDTH}\" height=\"{height}\" viewBox=\"0 0 {WIDTH} {height}\" font-family=\"system-ui, -apple-system, Segoe UI, Roboto, sans-serif\" font-size=\"12\" role=\"img\" aria-label=\"{}\">\n<title>{}</title>\n<rect width=\"{WIDTH}\" height=\"{height}\" fill=\"#ffffff\"/>\n<text x=\"{LEFT}\" y=\"28\" font-size=\"15\" font-weight=\"600\" fill=\"#111827\">{}</text>\n",
        escape(&chart.title),
        escape(&chart.title),
        escape(&chart.title),
    );
    for decade in low..=high {
        let base = 10f64.powi(decade);
        let y = y_of(base);
        out.push_str(&format!(
            "<line x1=\"{LEFT}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" stroke=\"#d1d5db\"/>\n<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" fill=\"#374151\">{}</text>\n",
            LEFT + plot_width,
            LEFT - 8.0,
            y + 4.0,
            number(base)
        ));
        if decade < high {
            for minor in [2.0, 5.0] {
                let y = y_of(base * minor);
                out.push_str(&format!(
                    "<line x1=\"{LEFT}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" stroke=\"#f3f4f6\"/>\n",
                    LEFT + plot_width
                ));
            }
        }
    }
    out.push_str(&format!(
        "<rect x=\"{LEFT}\" y=\"{TOP}\" width=\"{plot_width:.1}\" height=\"{PLOT_HEIGHT}\" fill=\"none\" stroke=\"#9ca3af\"/>\n"
    ));
    let axis = TOP + PLOT_HEIGHT;
    for (i, tick) in chart.ticks.iter().enumerate() {
        let x = x_of(i as f64);
        out.push_str(&format!(
            "<line x1=\"{x:.1}\" y1=\"{axis:.1}\" x2=\"{x:.1}\" y2=\"{:.1}\" stroke=\"#9ca3af\"/>\n<text x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\" fill=\"#374151\">{}</text>\n",
            axis + 5.0,
            axis + 19.0,
            escape(tick)
        ));
    }
    out.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" fill=\"#111827\">{}</text>\n<text transform=\"translate(18 {:.1}) rotate(-90)\" text-anchor=\"middle\" fill=\"#111827\">µs per step, log scale</text>\n",
        LEFT + plot_width / 2.0,
        axis + 40.0,
        escape(&chart.x_label),
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
                out.push_str(&format!(
                    "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"{colour}\" font-size=\"11\">{}</text>\n",
                    x + 7.0,
                    y - 7.0,
                    escape(label)
                ));
            }
        }
    }
    let legend = axis + 60.0;
    for (i, (series, colour)) in chart.series.iter().zip(&colours).enumerate() {
        let x = LEFT + if i % 2 == 0 { 0.0 } else { plot_width / 2.0 };
        let y = legend + (i / 2) as f64 * LEGEND_ROW;
        let dash = if series.modelspec {
            ""
        } else {
            " stroke-dasharray=\"6 4\""
        };
        out.push_str(&format!(
            "<line x1=\"{x:.1}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" stroke=\"{colour}\" stroke-width=\"2\"{dash}/>\n",
            x + 28.0
        ));
        marker(&mut out, x + 14.0, y, colour, series.modelspec);
        out.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#111827\">{}</text>\n",
            x + 36.0,
            y + 4.0,
            escape(&series.label)
        ));
    }
    out.push_str("</svg>\n");
    out
}

/// The heading of a workload's section.
fn workload_title(workload: &str) -> String {
    match workload {
        "mnist-784-64-10" => "MNIST 784-64-10".to_owned(),
        "mlp-784-H-10" => "MLP 784-H-10 sweep".to_owned(),
        other => other.to_owned(),
    }
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

/// The section of a workload with batch and hidden parameters: per batch, a chart of µs per step
/// against hidden width at the latest run of each point, an over-date chart when the batch has
/// runs on more than one date, and the latest medians.
fn sweep_section(workload: &str, runs: &[&Run], images: &mut Pages) -> String {
    let title = workload_title(workload);
    let mut out = format!(
        "## {title}\n\n`{workload}`: per batch, the median µs per step of the latest run at each \
         hidden width H, ModelSpec solid with filled circles and the opponents dashed with open \
         squares.\n"
    );
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
            "\nLatest median µs per step per side, runs of {latest_date} unless the cell says otherwise:\n\n| H | FLOP per step |"
        ));
        for (_, side) in &sides {
            out.push_str(&format!(" {} |", side_label(side)));
        }
        out.push_str("\n|---|---|");
        for _ in &sides {
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
            out.push('\n');
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
    let mut workloads: Vec<&str> = Vec::new();
    for run in &runs {
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
        let section = if of_workload
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
    for run in &runs {
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
    Ok((page, images))
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
        let text = fs::read_to_string(root().join(RUNS)).unwrap();
        let runs = read(&text, &schema()).unwrap();
        let sweep: Vec<&Run> = runs
            .iter()
            .filter(|run| run.workload == "mlp-784-H-10")
            .collect();
        assert_eq!(sweep.len(), 48);
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
        }
        assert_eq!(used.len(), 48, "every raw record is cited once");
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
}
