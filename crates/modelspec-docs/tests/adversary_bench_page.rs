//! Adversarial cases for story:benchmark-results-page: the sweep's chart against FLOP per step,
//! the GFLOP/s columns, the "Not measured" section and pitch slide 15, each checked against
//! `benchmarks/runs.yaml`, the sweep's raw crossover output and the deck itself rather than
//! against the generator's own helpers. Cases that change an input copy the repository into
//! their own directory under `CARGO_TARGET_TMPDIR` and run the `modelspec-docs` binary there, as
//! `task docs-generate` and `task docs-check` do.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_yaml_ng::Value;

const SWEEP: &str = "mlp-784-H-10";
const SVG: &str = "website/static/img/benchmarks/mlp-784-h-10-flop.svg";
const PAGE: &str = "website/docs/benchmarks.md";
const DECK: &str = "docs/pitch/index.html";

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(root: &Path, path: &str) -> String {
    fs::read_to_string(root.join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// One run of `runs.yaml`, as this file reads it, independently of the generator.
#[derive(Debug, Clone)]
struct Run {
    id: String,
    date: String,
    workload: String,
    side: String,
    batch: Option<u64>,
    hidden: Option<u64>,
    flop: Option<u64>,
    rule: Option<String>,
    median: f64,
}

fn runs(root: &Path) -> Vec<Run> {
    let document: Value = serde_yaml_ng::from_str(&read(root, "benchmarks/runs.yaml")).unwrap();
    document["runs"]
        .as_sequence()
        .unwrap()
        .iter()
        .map(|run| Run {
            id: run["run_id"].as_str().unwrap().to_owned(),
            date: run["date"].as_str().unwrap().to_owned(),
            workload: run["workload"].as_str().unwrap().to_owned(),
            side: run["side"].as_str().unwrap().to_owned(),
            batch: run.get("batch").and_then(Value::as_u64),
            hidden: run.get("hidden").and_then(Value::as_u64),
            flop: run.get("flop_per_step").and_then(Value::as_u64),
            rule: run
                .get("flop_rule")
                .and_then(Value::as_str)
                .map(str::to_owned),
            median: run["median_us_per_step"].as_f64().unwrap(),
        })
        .collect()
}

/// Of the runs of one side at one batch and width, the latest: the last of the latest date in
/// file order.
fn latest<'a>(runs: &'a [Run], side: &str, batch: u64, hidden: u64) -> Option<&'a Run> {
    let mut kept: Option<&Run> = None;
    for run in runs {
        if run.workload == SWEEP
            && run.side == side
            && run.batch == Some(batch)
            && run.hidden == Some(hidden)
            && kept.is_none_or(|kept| kept.date <= run.date)
        {
            kept = Some(run);
        }
    }
    kept
}

/// `text` without its thousands separators, as a number.
fn figure(text: &str) -> f64 {
    text.trim()
        .replace(',', "")
        .parse()
        .unwrap_or_else(|e| panic!("`{text}`: {e}"))
}

fn side_of(label: &str) -> &'static str {
    match label {
        "ModelSpec Vulkan" => "ModelSpecVulkan",
        "PyTorch CUDA eager" => "PyTorchCudaEager",
        "PyTorch CUDA torch.compile" => "PyTorchCudaCompile",
        other => panic!("unknown side label `{other}`"),
    }
}

fn cells(row: &str) -> Vec<String> {
    row.trim()
        .trim_matches('|')
        .split('|')
        .map(|cell| cell.trim().to_owned())
        .collect()
}

/// Every GFLOP/s cell of the sweep's latest-median tables equals `flop_per_step / µs / 1000` of
/// the latest run of its side, batch and width, read from `runs.yaml` here, to the printed
/// 1 decimal; a cell is blank exactly when that run states no `flop_per_step`.
#[test]
fn adversary_every_published_gflops_cell_follows_its_run() {
    let root = repository();
    let runs = runs(&root);
    let page = read(&root, PAGE);
    let mut batch = None;
    let mut header: Option<Vec<String>> = None;
    let mut checked = 0;
    let mut wrong = Vec::new();
    for line in page.lines() {
        if let Some(rest) = line.strip_prefix("### Batch ") {
            batch = Some(rest.trim().parse::<u64>().unwrap());
            header = None;
            continue;
        }
        if line.starts_with("## ") {
            batch = None;
            header = None;
            continue;
        }
        let Some(b) = batch else { continue };
        if line.starts_with("| H | FLOP per step |") {
            header = Some(cells(line));
            continue;
        }
        let Some(columns) = &header else { continue };
        if !line.starts_with("| ") || line.starts_with("|---") {
            continue;
        }
        let row = cells(line);
        assert_eq!(row.len(), columns.len(), "{line}");
        let hidden: u64 = row[0].parse().unwrap();
        for (column, cell) in columns.iter().zip(&row) {
            let Some(label) = column.strip_suffix(" GFLOP/s") else {
                continue;
            };
            let side = side_of(label);
            let run = latest(&runs, side, b, hidden)
                .unwrap_or_else(|| panic!("no run of {side} at batch {b}, H {hidden}"));
            checked += 1;
            match run.flop {
                None => {
                    if !cell.is_empty() {
                        wrong.push(format!("{}: `{cell}` but no flop_per_step", run.id));
                    }
                }
                Some(flop) => {
                    let exact = flop as f64 / run.median / 1000.0;
                    if cell.is_empty() || (figure(cell) - exact).abs() > 0.05 + 1e-9 {
                        wrong.push(format!("{}: `{cell}`, exact {exact:.4}", run.id));
                    }
                }
            }
        }
    }
    assert_eq!(checked, 36, "3 batches × 4 widths × 3 sides");
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Every run that counts by `MatmulOnly` states `3196·B·H` (spec/domains/benchmark.yaml,
/// FlopRule), so the sides of one point state one count.
#[test]
fn adversary_every_matmul_only_run_states_3196_b_h() {
    let runs = runs(&repository());
    let mut counted = 0;
    let mut wrong = Vec::new();
    for run in &runs {
        if run.rule.as_deref() != Some("MatmulOnly") {
            continue;
        }
        counted += 1;
        let expected = 3196 * run.batch.unwrap() * run.hidden.unwrap();
        if run.flop != Some(expected) {
            wrong.push(format!("{}: {:?}, rule says {expected}", run.id, run.flop));
        }
    }
    assert!(counted >= 36, "only {counted} MatmulOnly runs");
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// A marker of a plotted point, not of the legend: its centre and whether it is ModelSpec's
/// filled circle.
#[derive(Debug)]
struct Marker {
    x: f64,
    y: f64,
    modelspec: bool,
}

fn attribute(element: &str, name: &str) -> f64 {
    let key = format!(" {name}=\"");
    let at = element
        .find(&key)
        .unwrap_or_else(|| panic!("{name} in {element}"))
        + key.len();
    let end = element[at..].find('"').unwrap() + at;
    element[at..end].parse().unwrap()
}

fn markers(svg: &str) -> Vec<Marker> {
    let mut out = Vec::new();
    for element in svg.split('<').skip(1) {
        if element.starts_with("circle ") && element.contains("r=\"4.5\"") {
            out.push(Marker {
                x: attribute(element, "cx"),
                y: attribute(element, "cy"),
                modelspec: true,
            });
        } else if element.starts_with("rect ") && element.contains("width=\"8\" height=\"8\"") {
            out.push(Marker {
                x: attribute(element, "x") + 4.0,
                y: attribute(element, "y") + 4.0,
                modelspec: false,
            });
        }
    }
    // The legend's markers sit at x 30, left of the plot.
    out.retain(|marker| (marker.x - 30.0).abs() > 0.05);
    out
}

/// The published FLOP chart draws 36 points, 12 ModelSpec circles and 24 opponent squares, at 12
/// FLOP counts with the three sides of each at one x, and joins each side's batch by one line:
/// 9 polylines.
#[test]
fn adversary_the_published_flop_chart_draws_three_sides_at_each_of_twelve_counts() {
    let svg = read(&repository(), SVG);
    let markers = markers(&svg);
    assert_eq!(markers.len(), 36, "{markers:?}");
    assert_eq!(markers.iter().filter(|m| m.modelspec).count(), 12);
    let mut at: BTreeMap<String, usize> = BTreeMap::new();
    for marker in &markers {
        *at.entry(format!("{:.1}", marker.x)).or_default() += 1;
    }
    assert_eq!(at.len(), 12, "{at:?}");
    assert!(at.values().all(|n| *n == 3), "{at:?}");
    assert_eq!(svg.matches("<polyline ").count(), 9);
    assert!(svg.contains(">FLOP per step, log scale</text>"));
}

/// A fresh copy of the repository named `case`, without build output.
fn sandbox(case: &str) -> PathBuf {
    fn copy(from: &Path, to: &Path) {
        fs::create_dir_all(to).unwrap();
        for entry in fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name();
            if matches!(
                name.to_str(),
                Some("target" | ".git" | "node_modules" | "build" | ".docusaurus")
            ) {
                continue;
            }
            let kind = entry.file_type().unwrap();
            if kind.is_dir() {
                copy(&entry.path(), &to.join(&name));
            } else if kind.is_file() {
                fs::copy(entry.path(), to.join(&name)).unwrap();
            }
        }
    }
    let root = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary-bench-page")
        .join(case);
    let _ = fs::remove_dir_all(&root);
    copy(&repository(), &root);
    root
}

/// In the copy's `runs.yaml`, `from` replaced by `to` once inside the entry of `run_id`.
fn edit_entry(root: &Path, run_id: &str, from: &str, to: &str) {
    let path = root.join("benchmarks/runs.yaml");
    let text = fs::read_to_string(&path).unwrap();
    let start = text
        .find(&format!("  - run_id: {run_id}\n"))
        .unwrap_or_else(|| panic!("no entry {run_id}"));
    let end = text[start + 4..]
        .find("\n  - ")
        .map_or(text.len(), |at| start + 4 + at);
    let entry = &text[start..end];
    assert!(entry.contains(from), "`{from}` not in {run_id}");
    let edited = format!(
        "{}{}{}",
        &text[..start],
        entry.replacen(from, to, 1),
        &text[end..]
    );
    fs::write(&path, edited).unwrap();
}

fn append_runs(root: &Path, entries: &str) {
    let path = root.join("benchmarks/runs.yaml");
    let mut text = fs::read_to_string(&path).unwrap();
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(entries);
    fs::write(&path, text).unwrap();
}

fn entry(id: &str, workload: &str, side: &str, extra: &str) -> String {
    format!(
        "  - run_id: {id}\n    date: \"2026-10-27\"\n    workload: {workload}\n    side: {side}\n    mode: Timed\n{extra}    repeat_count: 1\n    repeats_us_per_step: null\n    median_us_per_step: 900.0\n    source: a fixture of the adversary\n    state: Recorded\n"
    )
}

struct Outcome {
    code: Option<i32>,
    stderr: String,
}

fn generate(root: &Path, check: bool) -> Outcome {
    let mut command = Command::new(env!("CARGO_BIN_EXE_modelspec-docs"));
    command.arg("generate").arg("--root").arg(root);
    if check {
        command.arg("--check");
    }
    let output = command.output().unwrap();
    Outcome {
        code: output.status.code(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// ADV-BP-3. One side states twice the `MatmulOnly` count at batch 8192, H 4096 (a multiply-add
/// counted as two). The story asks for one FLOP count for every side of a point, and the doc of
/// `flop_points` says all sides of a point share one; but the sides agree on the rule, so nothing
/// re-places the point and nothing refuses the count: the chart draws that side about 20 px to
/// the right of the other two. Either `generate` refuses the count that does not fit its rule, or
/// the three sides are drawn at one FLOP count.
#[test]
#[ignore = "ADV-BP-3: one rule, two FLOP counts at one point, drawn apart and not refused"]
fn adversary_adv_bp_3_sides_of_one_point_with_one_rule_and_two_counts_share_one_x() {
    let root = sandbox("two-counts");
    edit_entry(
        &root,
        "sweep-b8192-h4096-pytorch-cuda-eager-step-sites",
        "flop_per_step: 107239964672",
        "flop_per_step: 214479929344",
    );
    let outcome = generate(&root, false);
    if outcome.code == Some(1) && outcome.stderr.contains("flop_per_step") {
        return;
    }
    assert_eq!(outcome.code, Some(0), "{}", outcome.stderr);
    let mut markers = markers(&read(&root, SVG));
    markers.sort_by(|a, b| a.y.total_cmp(&b.y));
    let largest: Vec<String> = markers[..3].iter().map(|m| format!("{:.1}", m.x)).collect();
    assert!(
        largest.iter().all(|x| *x == largest[0]),
        "the three sides at batch 8192, H 4096 drawn at x {largest:?}; generate exited {:?}",
        outcome.code
    );
}

/// The planning kinds of the store, each as `<kind>:`.
fn planning_kinds(root: &Path) -> Vec<String> {
    let kinds: Vec<String> = fs::read_dir(root.join(".engineering/planning"))
        .unwrap()
        .map(|entry| entry.unwrap())
        .filter(|entry| entry.file_type().unwrap().is_dir())
        .map(|entry| format!("{}:", entry.file_name().into_string().unwrap()))
        .collect();
    assert!(kinds.iter().any(|kind| kind == "story:"), "{kinds:?}");
    kinds
}

/// ADV-BP-5. The page is public text served with the site, like the deck's speaker notes, which
/// `the_pitch_notes_cite_changelog_releases_and_name_no_planning_record` keeps free of planning
/// ids. The page's prose (every line that is not a table row; the `Source` cells of the tables
/// cite what `runs.yaml` cites) names none either: before this unit it named none, and the
/// "Not measured" section now names `story:transformer-block-benchmark` and
/// `story:bert-tiny-vs-pytorch-cuda`.
#[test]
#[ignore = "ADV-BP-5: the Not measured prose names planning-store story ids"]
fn adversary_adv_bp_5_the_page_prose_names_no_planning_record() {
    let root = repository();
    let page = read(&root, PAGE);
    let mut wrong = Vec::new();
    for (n, line) in page.lines().enumerate() {
        if line.starts_with('|') {
            continue;
        }
        for kind in planning_kinds(&root) {
            if line.contains(&format!("`{kind}")) || line.contains(&format!(" {kind}")) {
                wrong.push(format!("{PAGE}:{}: {kind}", n + 1));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Slide 15 of the deck: the text of each `<div class="v">` value, in order.
fn slide_values(deck: &str, n: u32) -> (String, Vec<String>) {
    let start = deck
        .find(&format!("<section class=\"slide\" data-n=\"{n}\">"))
        .unwrap();
    let end = deck[start..].find("</section>").unwrap() + start;
    let slide = deck[start..end].to_owned();
    let mut values = Vec::new();
    for piece in slide.split("<div class=\"v\"").skip(1) {
        let open = piece.find('>').unwrap() + 1;
        let close = piece.find("</div>").unwrap();
        values.push(piece[open..close].to_owned());
    }
    (slide, values)
}

/// The deck has slides 1 to 18 in order, once each, and its counter says 18; slide 15 is the
/// sweep, after the MNIST slide; its crossover widths are the `pitch_slide: true` lines of the
/// sweep's `crossover.txt`, in the order ModelSpec Vulkan, PyTorch eager, PyTorch compile.
#[test]
fn adversary_slide_15_follows_the_mnist_slide_with_the_crossover_lines() {
    let root = repository();
    let deck = read(&root, DECK);
    let numbers: Vec<u32> = deck
        .match_indices("<section class=\"slide")
        .map(|(at, _)| {
            let rest = &deck[at..];
            let key = rest.find("data-n=\"").unwrap() + "data-n=\"".len();
            let end = rest[key..].find('"').unwrap() + key;
            rest[key..end].parse().unwrap()
        })
        .collect();
    assert_eq!(numbers, (1..=18).collect::<Vec<_>>());
    assert!(deck.contains("content: attr(data-n) \" / 18\""));
    let (mnist, _) = slide_values(&deck, 14);
    assert!(mnist.contains("MNIST"), "slide 14 is not the MNIST slide");
    let (_, values) = slide_values(&deck, 15);
    assert_eq!(values.len(), 6, "{values:?}");
    let crossover = read(
        &root,
        "benchmarks/raw/2026-10-25-sweep-0.10.0/crossover.txt",
    );
    let mut widths = Vec::new();
    for side in ["modelspec-vulkan", "pytorch-eager", "pytorch-compile"] {
        let line = crossover
            .lines()
            .find(|line| {
                line.contains(&format!("\"side\":\"{side}\""))
                    && line.contains("\"pitch_slide\":true")
            })
            .unwrap_or_else(|| panic!("no pitch line for {side}"));
        let line: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(line["batch"], 1000);
        assert_eq!(line["verdict"], "reached");
        widths.push(format!("H {}", line["hidden"]));
    }
    assert_eq!(values[..3], widths[..]);
}

/// ADV-BP-6. The GFLOP/s of the largest point on slide 15 are those of the runs its notes cite,
/// `sweep-b8192-h4096-<side>-0.10.0`, by the page's rule `flop_per_step / µs / 1000`, to the
/// printed 1 decimal. The slide divides by the report's 1-decimal µs instead, so PyTorch eager
/// reads 15,676.1 where its run's median 6,840.956 µs gives 15,676.166.
#[test]
#[ignore = "ADV-BP-6: slide 15 GFLOP/s from rounded µs, off its cited run by more than the printed rounding"]
fn adversary_adv_bp_6_slide_15_gflops_follow_the_cited_runs() {
    let root = repository();
    let runs = runs(&root);
    let (_, values) = slide_values(&read(&root, DECK), 15);
    let mut wrong = Vec::new();
    for (value, side) in values[3..].iter().zip([
        "modelspec-vulkan",
        "pytorch-cuda-eager",
        "pytorch-cuda-compile",
    ]) {
        let id = format!("sweep-b8192-h4096-{side}-0.10.0");
        let run = runs.iter().find(|run| run.id == id).unwrap();
        let exact = run.flop.unwrap() as f64 / run.median / 1000.0;
        if (figure(value) - exact).abs() > 0.05 + 1e-9 {
            wrong.push(format!("{id}: slide {value}, run {exact:.4}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// ADV-BP-7. A run whose `workload` is not one the generator declares (here a misspelt
/// `transformer_block`) is neither refused nor reported: the page gains a section titled with the
/// raw id, and "Not measured" goes on saying the transformer block has no run. The section is
/// read from a declared list; a workload outside it is refused with the entry's line.
#[test]
#[ignore = "ADV-BP-7: an undeclared workload is published under its raw id, not refused"]
fn adversary_adv_bp_7_an_undeclared_workload_is_refused() {
    let root = sandbox("undeclared-workload");
    append_runs(
        &root,
        &entry("block-typo", "transformer_block", "ModelSpecVulkan", ""),
    );
    let outcome = generate(&root, false);
    let page = read(&root, PAGE);
    assert_eq!(
        outcome.code,
        Some(1),
        "generate exited {:?}; the page has `## transformer_block`: {}; Not measured still lists the block: {}",
        outcome.code,
        page.contains("\n## transformer_block\n"),
        page.contains("- **Transformer block** (`transformer-block`)")
    );
    assert!(outcome.stderr.contains("workload"), "{}", outcome.stderr);
}

/// The first transformer-block runs in the shape its story asks for (batch 8 and its 1024 tokens
/// per step, no hidden width, a FLOP count stated for both sides by one rule) leave "Not
/// measured" and give the workload a section and a chart of its own, with no generator change;
/// since fix pass 1 of story:transformer-block-benchmark (coordinator decision D84) a block run
/// states its tokens per step, so the section is the sequence section and its chart is µs per
/// step against batch.
#[test]
fn adversary_a_realistic_first_transformer_run_leaves_not_measured() {
    let root = sandbox("transformer-run");
    let flop = "    batch: 8\n    tokens_per_step: 1024\n    flop_per_step: 1000000000\n    flop_rule: AllSites\n";
    append_runs(
        &root,
        &format!(
            "{}{}",
            entry("block-ms", "transformer-block", "ModelSpecVulkan", flop),
            entry("block-eager", "transformer-block", "PyTorchCudaEager", flop)
        ),
    );
    let outcome = generate(&root, false);
    assert_eq!(outcome.code, Some(0), "{}", outcome.stderr);
    let page = read(&root, PAGE);
    assert!(page.contains("\n## Transformer block\n"), "{page}");
    let not_measured = &page[page
        .find("\n## Not measured\n")
        .expect("BERT is still listed")..];
    assert!(
        !not_measured.contains("Transformer block"),
        "{not_measured}"
    );
    assert!(not_measured.contains("**BERT-tiny pre-training**"));
    assert!(
        root.join("website/static/img/benchmarks/transformer-block-batch.svg")
            .is_file()
    );
}

/// A hand change to the generated FLOP chart, one point moved by a pixel, fails `generate --check`.
#[test]
fn adversary_generate_check_refuses_a_hand_edited_flop_chart() {
    let root = sandbox("hand-edited-svg");
    let path = root.join(SVG);
    let svg = fs::read_to_string(&path).unwrap();
    let at = svg.find("<circle cx=\"").unwrap() + "<circle cx=\"".len();
    let edited = format!("{}1{}", &svg[..at], &svg[at..]);
    fs::write(&path, edited).unwrap();
    let outcome = generate(&root, true);
    assert_eq!(outcome.code, Some(1), "{}", outcome.stderr);
    assert!(
        outcome.stderr.contains("mlp-784-h-10-flop.svg"),
        "{}",
        outcome.stderr
    );
}
