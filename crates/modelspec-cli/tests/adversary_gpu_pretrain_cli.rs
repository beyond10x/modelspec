//! Adversary pass 1 on story:bert-tiny-gpu-pretraining (unit 19, coordinator decisions D75 and
//! D76): the run record page `website/docs/runs/bert-tiny-gpu-pretraining.md` and the recorded
//! run it describes.
//!
//! Three kinds of case:
//!
//! - Cases without `#[ignore]` run in `task ci`. They hold the page against the committed
//!   fixture `tests/fixtures/bert_tiny_gpu_pretrain/stdout-lines.txt`, the `data`, `step` and
//!   `eval` lines of the run's stdout in order (its `init` and `param` lines are left out), and
//!   against arithmetic on those lines. The event file's size and SHA-256 are literals checked
//!   against the file by [`record_events_conform_and_match_stdout`].
//! - "record" cases need the variable `MODELSPEC_GPU_PRETRAIN_RUN`, the directory holding the
//!   run's `full.stdout`, `full.stderr` and `full.events.jsonl`; the cases that read the packed
//!   corpus also need `MODELSPEC_GPU_PRETRAIN_DATA`, the packed directory (`packed.json`,
//!   `binding.json`, `train.tokens`, `heldout.tokens`). Run them with
//!   `cargo test -p modelspec-cli --test adversary_gpu_pretrain_cli -- --ignored record_`.
//! - "device" cases need the variables above and a Vulkan device; run them with
//!   `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-cli --test adversary_gpu_pretrain_cli -- --ignored device_`.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use serde_json::Value;

const PAGE: &str = "website/docs/runs/bert-tiny-gpu-pretraining.md";
const FIXTURE: &str = "crates/modelspec-cli/tests/fixtures/bert_tiny_gpu_pretrain/stdout-lines.txt";
const SCHEMA: &str =
    "crates/modelspec-observatory/schema/entities/modelspec.observatory.RunEvent.schema.json";
const EXAMPLE: &str = "examples/bert_tiny.mspec";
const RUN_ENV: &str = "MODELSPEC_GPU_PRETRAIN_RUN";
const DATA_ENV: &str = "MODELSPEC_GPU_PRETRAIN_DATA";

/// `heldout_unigram_baseline_nats` of the run's `packed.json`.
const BASELINE: f64 = 6.926537926202018;
/// The size of the recorded `full.events.jsonl`, read with `wc -c` and checked by
/// [`record_events_conform_and_match_stdout`].
const EVENTS_BYTES: u64 = 2_609_690;
/// The SHA-256 of the recorded `full.events.jsonl`, read with `sha256sum` and checked by
/// [`record_events_conform_and_match_stdout`].
const EVENTS_SHA256: &str = "2336f1e1f0a7375a6702a48defba50dbc5b7218fbdf7ffc8853bebc4b12e11bd";
/// The selected (weight 1) rows of the 68 full `HeldOut` batches every evaluation tallies, and
/// `pack`'s add-one unigram loss over exactly those rows' labels: the population the eval loss
/// is measured on. Recomputed from the packed corpus by
/// [`record_accuracy_is_whole_hits_and_the_loss_beats_the_baseline_of_the_evaluated_rows`].
const EVALUATED_ROWS: u64 = 41_857;
const EVALUATED_BASELINE: f64 = 6.904154117063165;
/// The run's batch of `examples/bert_tiny.mspec` and its seed.
const BATCH: usize = 32;
const SEED: u64 = 42;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(relative: &str) -> String {
    std::fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|e| panic!("{relative} is readable: {e}"))
}

/// One `eval <step> loss <loss> accuracy <accuracy>` line, its numbers as printed.
#[derive(Clone, Debug, PartialEq)]
struct Eval {
    step: u64,
    loss: String,
    accuracy: String,
}

impl Eval {
    fn loss(&self) -> f64 {
        self.loss.parse().unwrap()
    }
}

/// The `step` lines as (step, loss text) and the `eval` lines of a run's stdout lines.
fn parse_lines(text: &str) -> (Vec<(u64, String)>, Vec<Eval>) {
    let (mut steps, mut evals) = (Vec::new(), Vec::new());
    for line in text.lines() {
        let words: Vec<&str> = line.split(' ').collect();
        match words.as_slice() {
            ["step", step, "loss", loss] => steps.push((step.parse().unwrap(), (*loss).to_owned())),
            ["eval", step, "loss", loss, "accuracy", accuracy] => evals.push(Eval {
                step: step.parse().unwrap(),
                loss: (*loss).to_owned(),
                accuracy: (*accuracy).to_owned(),
            }),
            _ => {}
        }
    }
    (steps, evals)
}

fn fixture() -> (Vec<(u64, String)>, Vec<Eval>) {
    parse_lines(&read(FIXTURE))
}

/// `n` with a comma between groups of three digits, as the page writes counts.
fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (k, c) in digits.chars().enumerate() {
        if k > 0 && (digits.len() - k).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// The rows of the Markdown table whose header line is `header`, each as its trimmed cells.
fn table(page: &str, header: &str) -> Vec<Vec<String>> {
    let mut lines = page.lines().skip_while(|l| *l != header);
    assert_eq!(
        lines.next(),
        Some(header),
        "the page has the table {header}"
    );
    let separator = lines.next().expect("a separator row");
    assert!(separator.starts_with("|---"), "{separator}");
    lines
        .take_while(|l| l.starts_with('|'))
        .map(|l| {
            l.trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_owned())
                .collect()
        })
        .collect()
}

/// The value cell of the two-column row of `rows` whose first cell is `name`.
fn row<'a>(rows: &'a [Vec<String>], name: &str) -> &'a str {
    let found: Vec<&Vec<String>> = rows.iter().filter(|r| r[0] == name).collect();
    assert_eq!(found.len(), 1, "one row {name:?} in {rows:?}");
    &found[0][1]
}

/// The figure the coordinator puts into `status.rs` as the evidence of the shipped row: every
/// number of the page's result table, its eval table and its training-loss table is the one the
/// committed run lines and arithmetic on them give.
#[test]
fn run_record_page_agrees_with_the_committed_run_lines() {
    let page = read(PAGE);
    let (steps, evals) = fixture();

    assert_eq!(evals.len(), 21, "21 eval lines in the fixture");
    assert_eq!(steps.len(), 201, "201 step lines in the fixture");
    let (first, last) = (&evals[0], &evals[evals.len() - 1]);
    assert_eq!(first.step, 0);
    let last_step = steps.last().unwrap().0;
    assert_eq!(last.step, last_step, "the last eval is at the last step");

    // Scenario 1 and 2 on the committed lines.
    assert!(
        last.loss() < BASELINE,
        "the last eval loss {} is below the baseline {BASELINE}",
        last.loss
    );
    assert!(last.loss() < first.loss(), "the eval loss falls");

    // Every eval line, in order and byte for byte, and no other row.
    let rows = table(&page, "| Step | Held-out loss | Held-out accuracy |");
    let expected: Vec<Vec<String>> = evals
        .iter()
        .map(|e| vec![e.step.to_string(), e.loss.clone(), e.accuracy.clone()])
        .collect();
    assert_eq!(rows, expected, "the eval table is the eval lines");

    // The result table.
    let result = table(&page, "| Figure | Value |");
    assert_eq!(
        row(&result, "Held-out unigram baseline (`packed.json`)"),
        format!("{BASELINE} nats")
    );
    assert_eq!(row(&result, "Held-out loss, step 0"), first.loss);
    let at_last = thousands(last.step);
    assert_eq!(
        row(&result, &format!("Held-out loss, step {at_last}")),
        last.loss
    );
    assert_eq!(
        row(&result, &format!("Held-out accuracy, step {at_last}")),
        last.accuracy
    );
    let margin = BASELINE - last.loss();
    let percent = 100.0 * margin / BASELINE;
    assert_eq!(
        row(&result, "Margin below the baseline"),
        format!("{margin:.7} nats, {percent:.2} % of the baseline")
    );
    let below = evals.iter().find(|e| e.loss() < BASELINE).unwrap();
    assert_eq!(
        row(&result, "First evaluation below the baseline"),
        format!("step {} ({})", thousands(below.step), below.loss)
    );
    assert!(
        row(&result, "Steps").starts_with(&format!("0 to {at_last}, `steps = {last_step}`")),
        "{}",
        row(&result, "Steps")
    );
    let events = 1 + (last_step + 1) + 1;
    assert_eq!(
        row(&result, "Recorded event file"),
        format!(
            "`full.events.jsonl`, {} bytes, {} lines",
            thousands(EVENTS_BYTES),
            thousands(events)
        )
    );
    assert_eq!(
        row(&result, "SHA-256 of the event file"),
        format!("`{EVENTS_SHA256}`")
    );

    // The rounded margin of the lede, the description and the opening paragraph.
    let short = format!("{margin:.3} nats ({percent:.2} %)");
    assert_eq!(
        page.matches(&short).count(),
        2,
        "lede and paragraph: {short}"
    );
    let baseline7 = format!("{BASELINE:.7}");
    assert!(page.contains(&format!("against a unigram baseline of {baseline7}.")));
    assert!(page.contains(&format!("held-out unigram baseline of\n{baseline7}")));
    assert!(page.contains(&format!("from {} at step 0", first.loss)));
    assert!(page.contains(&format!("**{}** at step {at_last}", last.loss)));
    assert!(page.contains(&format!("step {at_last} is **{}**", last.accuracy)));

    // The prose on the curve.
    let eval_at = |step: u64| evals.iter().find(|e| e.step == step).unwrap();
    let gap = eval_at(1000).loss() - BASELINE;
    assert!(
        gap > 0.0 && gap < 0.002,
        "within 0.002 nats at step 1,000: {gap}"
    );
    let plateau: Vec<&Eval> = evals
        .iter()
        .filter(|e| e.accuracy == eval_at(1000).accuracy)
        .collect();
    assert_eq!(
        plateau.iter().map(|e| e.step).collect::<Vec<_>>(),
        vec![1000, 1500, 2000, 2500, 3000, 3500]
    );
    assert!(page.contains(&format!(
        "the accuracy is {} at all six evaluations from step 1,000 to step 3,500",
        eval_at(1000).accuracy
    )));

    // Every step line in its cell of the training-loss table.
    let header = "| Steps | +0 | +50 | +100 | +150 | +200 | +250 | +300 | +350 | +400 | +450 |";
    let losses = table(&page, header);
    let mut cells: BTreeMap<u64, String> = BTreeMap::new();
    for r in &losses {
        let base: u64 = r[0].parse().unwrap();
        for (k, cell) in r[1..].iter().enumerate() {
            if !cell.is_empty() {
                cells.insert(base + 50 * k as u64, cell.clone());
            }
        }
    }
    let printed: BTreeMap<u64, String> = steps.iter().cloned().collect();
    assert_eq!(cells, printed, "the training-loss table is the step lines");

    // The wall time from the page's own timestamps.
    let wall = row(&result, "Wall time");
    assert!(
        wall.starts_with("2026-10-26T06:32:19Z to 07:17:24Z, "),
        "{wall}"
    );
    let seconds = (7 * 3600 + 17 * 60 + 24) - (6 * 3600 + 32 * 60 + 19);
    assert_eq!(
        wall,
        format!(
            "2026-10-26T06:32:19Z to 07:17:24Z, {} min {} s ({} s)",
            seconds / 60,
            seconds % 60,
            thousands(seconds)
        )
    );

    // The evaluation-time estimate of "What the record does not hold".
    let trained = (last_step + 1) as f64;
    let (fast, slow) = (0.217 * trained, 0.232 * trained);
    let (low, high) = (seconds as f64 - slow, seconds as f64 - fast);
    let estimate = format!(
        "10,001 steps at that rate are {} to {} s, which leaves an estimated {} to {} s, {} to {} s for each of the {} evaluations",
        thousands(fast.round() as u64),
        thousands(slow.round() as u64),
        low.round(),
        high.round(),
        (low / evals.len() as f64).round(),
        (high / evals.len() as f64).round(),
        evals.len()
    );
    let flat = page.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(flat.contains(&estimate), "{estimate}");
}

/// Finding F1: the curve paragraph states the gain of the last 1,000 steps, `eval 9000` minus
/// `eval 10000`, rounded to four decimals as the page's other figures are rounded.
#[test]
fn run_record_page_rounds_the_gain_of_the_last_1000_steps() {
    let page = read(PAGE);
    let (_, evals) = fixture();
    let eval_at = |step: u64| evals.iter().find(|e| e.step == step).unwrap();
    let gain = eval_at(9000).loss() - eval_at(10000).loss();
    let sentence = format!("the last 1,000 steps gain {gain:.4} nats");
    assert!(
        page.contains(&sentence),
        "{} - {} = {gain}; the page has no {sentence:?}",
        eval_at(9000).loss,
        eval_at(10000).loss
    );
}

/// Finding F2: `packed.json`'s baseline 6.926537926202018 is the add-one unigram loss over every
/// non-special token of all 2,189 `HeldOut` sequences (`dataset/mlm.rs`, `pack`), while every
/// `eval` loss is over the 41,857 selected rows of the 68 full batches (`train/eval.rs`). Over
/// those rows the same unigram loss is 6.904154117063165, so the page's "the loss of predicting
/// every masked token from its frequency in the `Train` split" is the wrong population, the
/// margin on the evaluated rows is 0.1366547 nats, not 0.1590385, and the first evaluation below
/// it is step 2,500, not 1,500. The record names the baseline of the evaluated rows and the
/// margin against it.
#[test]
fn run_record_page_states_the_baseline_over_the_evaluated_rows() {
    let page = read(PAGE);
    let flat = page.split_whitespace().collect::<Vec<_>>().join(" ");
    let (_, evals) = fixture();
    let last = evals.last().unwrap();
    let margin = EVALUATED_BASELINE - last.loss();
    let below = evals
        .iter()
        .find(|e| e.loss() < EVALUATED_BASELINE)
        .unwrap();
    let missing: Vec<String> = [
        format!("{EVALUATED_BASELINE:.7}"),
        thousands(EVALUATED_ROWS),
        format!("{margin:.7}"),
        format!("step {} ({})", thousands(below.step), below.loss),
    ]
    .into_iter()
    .filter(|figure| !flat.contains(figure.as_str()))
    .collect();
    assert!(
        missing.is_empty(),
        "the page does not state, for the evaluated rows: {missing:?}"
    );
}

/// The page's model, corpus and evaluation figures against the example and the packed counts
/// the page itself states.
#[test]
fn run_record_page_agrees_with_the_example_and_its_corpus_table() {
    let page = read(PAGE);
    let example = read(EXAMPLE);
    let hyper = table(&page, "| Setting | Value |");
    let optimizer = row(&hyper, "Optimizer");
    let call = optimizer
        .strip_prefix('`')
        .and_then(|o| o.split('`').next())
        .unwrap();
    assert!(example.contains(&format!("optimizer = {call};")), "{call}");
    for setting in ["Steps", "Seed"] {
        let value = row(&hyper, setting).trim_matches('`');
        assert!(example.contains(&format!("    {value};")), "{value}");
    }
    assert!(
        example.contains("Tensor<f32, [32, 128]>"),
        "batch 32 of 128"
    );
    assert_eq!(row(&hyper, "Batch"), "32 sequences of 128 tokens");

    // 68 full HeldOut batches of 32 out of 2,189 sequences; one batch's logits in bytes.
    let (_, corpus_section) = page.split_once("### The corpus").expect("a corpus section");
    let corpus = table(corpus_section, "| Item | Value |");
    let packed = corpus.iter().find(|r| r[0] == "Packed").unwrap();
    assert!(packed[1].contains("1,049,703 `Train` and 2,189 `HeldOut` sequences"));
    let batches = 2189 / BATCH;
    let flat = page.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(flat.contains(&format!("the {batches} full `HeldOut` batches")));
    let logits = BATCH * 128 * 8192 * 4;
    assert!(flat.contains(&format!("prediction, {} bytes", thousands(logits as u64))));
}

/// The README, AGENTS.md, the docs index, the roadmap, the CHANGELOG and the pitch quote the
/// page's final loss, its rounded margin and its rounded baseline.
#[test]
fn run_record_figures_agree_across_the_repository() {
    let (_, evals) = fixture();
    let last = evals.last().unwrap();
    let margin = BASELINE - last.loss();
    let figures = [
        last.loss.clone(),
        format!("{margin:.3}"),
        format!("{BASELINE:.7}"),
    ];
    for file in [
        "README.md",
        "AGENTS.md",
        "CHANGELOG.md",
        "website/docs/index.md",
        "website/docs/roadmap.md",
    ] {
        let text = read(file).split_whitespace().collect::<Vec<_>>().join(" ");
        for figure in &figures {
            assert!(text.contains(figure.as_str()), "{file} quotes {figure}");
        }
    }
    let pitch = read("docs/pitch/index.html");
    assert!(pitch.contains(&format!(
        "held-out loss {:.3}, {margin:.3} nats below the unigram baseline",
        last.loss()
    )));
    assert!(pitch.contains(&format!("eval {} loss {}", last.step, last.loss)));
}

fn run_dir() -> PathBuf {
    PathBuf::from(std::env::var(RUN_ENV).unwrap_or_else(|_| panic!("{RUN_ENV} is set")))
}

fn data_dir() -> PathBuf {
    PathBuf::from(std::env::var(DATA_ENV).unwrap_or_else(|_| panic!("{DATA_ENV} is set")))
}

fn schema() -> jsonschema::Validator {
    let text = read(SCHEMA);
    let schema: Value = serde_json::from_str(&text).expect("the schema is JSON");
    jsonschema::validator_for(&schema).expect("the schema compiles")
}

/// The value of the metric `name` of a `StepCompleted` value, if it carries one.
fn metric(step: &Value, name: &str) -> Option<f64> {
    step["metrics"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["name"] == name)
        .map(|m| m["value"].as_f64().unwrap())
}

fn f32_text(value: f64) -> String {
    (value as f32).to_string()
}

/// The committed fixture is the `data`, `step` and `eval` lines of the run's stdout, the stdout
/// ends `EXIT=0`, and the stderr is the two lines of `--serve` and `--ui`.
#[test]
#[ignore = "record: needs MODELSPEC_GPU_PRETRAIN_RUN"]
fn record_fixture_is_the_runs_stdout() {
    let dir = run_dir();
    let stdout = std::fs::read_to_string(dir.join("full.stdout")).unwrap();
    let kept: Vec<&str> = stdout
        .lines()
        .filter(|l| l.starts_with("data ") || l.starts_with("step ") || l.starts_with("eval "))
        .collect();
    let fixture = read(FIXTURE);
    assert_eq!(kept, fixture.lines().collect::<Vec<_>>());
    assert_eq!(stdout.lines().last(), Some("EXIT=0"));
    let kinds: BTreeMap<&str, usize> = stdout.lines().fold(BTreeMap::new(), |mut m, l| {
        *m.entry(l.split(' ').next().unwrap()).or_default() += 1;
        m
    });
    assert_eq!(
        kinds,
        BTreeMap::from([
            ("EXIT=0", 1),
            ("data", 1),
            ("eval", 21),
            ("init", 41),
            ("param", 41),
            ("step", 201)
        ])
    );
    let stderr = std::fs::read_to_string(dir.join("full.stderr")).unwrap();
    assert_eq!(
        stderr,
        "serving http://127.0.0.1:7878/events\nui http://127.0.0.1:7878/\n"
    );
}

/// Scenario 3 on the file: every event valid against the RunEvent schema, sequences 0, 1, 2, ...,
/// one `StepCompleted` per step 0 to the last, each printed `step` and `eval` line equal to its
/// event as `f32` shortest round-trip text, no held-out metric on a step without an `eval` line,
/// and the size and SHA-256 the page states.
#[test]
#[ignore = "record: needs MODELSPEC_GPU_PRETRAIN_RUN"]
fn record_events_conform_and_match_stdout() {
    let dir = run_dir();
    let path = dir.join("full.events.jsonl");
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes.len() as u64, EVENTS_BYTES);
    let sha = Command::new("sha256sum").arg(&path).output().unwrap();
    assert!(sha.status.success());
    assert_eq!(
        String::from_utf8(sha.stdout).unwrap().split(' ').next(),
        Some(EVENTS_SHA256)
    );
    let text = String::from_utf8(bytes).unwrap();
    assert!(text.ends_with('\n'));
    let events: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let validator = schema();
    for (k, event) in events.iter().enumerate() {
        let errors: Vec<String> = validator
            .iter_errors(event)
            .map(|e| e.to_string())
            .collect();
        assert!(errors.is_empty(), "event {k}: {errors:?}");
        assert_eq!(event["sequence"], k as u64);
        assert_eq!(event["training_run_id"], events[0]["training_run_id"]);
    }
    assert_eq!(events[0]["body"]["kind"], "RunStarted");
    assert_eq!(events[0]["body"]["value"]["steps"], 10000);
    assert_eq!(events[0]["body"]["value"]["start_step"], 0);
    assert_eq!(events.last().unwrap()["body"]["kind"], "RunFinished");

    let (steps, evals) = fixture();
    let completed: Vec<&Value> = events[1..events.len() - 1]
        .iter()
        .map(|e| {
            assert_eq!(e["body"]["kind"], "StepCompleted");
            &e["body"]["value"]
        })
        .collect();
    assert_eq!(completed.len(), 10001);
    for (k, step) in completed.iter().enumerate() {
        assert_eq!(
            step["step"], k as u64,
            "one StepCompleted per step, in order"
        );
    }
    for (step, loss) in &steps {
        let event = completed[*step as usize];
        assert_eq!(
            &f32_text(event["loss"].as_f64().unwrap()),
            loss,
            "step {step}"
        );
    }
    let evaluated: Vec<u64> = evals.iter().map(|e| e.step).collect();
    for (k, step) in completed.iter().enumerate() {
        let loss = metric(step, "heldout_mlm_loss");
        let accuracy = metric(step, "heldout_mlm_accuracy");
        match evals.iter().find(|e| e.step == k as u64) {
            Some(e) => {
                assert_eq!(loss.map(f32_text), Some(e.loss.clone()), "eval {k}");
                assert_eq!(accuracy.map(f32_text), Some(e.accuracy.clone()), "eval {k}");
            }
            None => assert!(
                loss.is_none() && accuracy.is_none(),
                "step {k} carries a held-out metric without an eval line ({evaluated:?})"
            ),
        }
    }
}

/// Scenario 3, replay: `observe <file> --serve 127.0.0.1:0` streams every line of the file as
/// one frame, in order, ends the response after `RunFinished`, and exits 0 on SIGTERM.
#[test]
#[ignore = "record: needs MODELSPEC_GPU_PRETRAIN_RUN"]
fn record_replay_streams_the_file_and_exits_0() {
    let path = run_dir().join("full.events.jsonl");
    let lines: Vec<String> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    let mut child = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["observe", path.to_str().unwrap(), "--serve", "127.0.0.1:0"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let err = child.stderr.take().unwrap();
    let (sender, received) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut all = String::new();
        for line in BufReader::new(err).lines() {
            let Ok(line) = line else { break };
            all.push_str(&line);
            all.push('\n');
            let _ = sender.send(line);
        }
        all
    });
    let address = loop {
        let line = received
            .recv_timeout(Duration::from_secs(60))
            .expect("a serving line");
        if let Some(rest) = line.strip_prefix("serving http://") {
            break rest.strip_suffix("/events").unwrap().to_owned();
        }
    };
    let mut stream = TcpStream::connect(&address).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(120)))
        .unwrap();
    write!(
        stream,
        "GET /events HTTP/1.1\r\nHost: {address}\r\nAccept: text/event-stream\r\n\r\n"
    )
    .unwrap();
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .expect("the server ends the response");
    let (head, body) = response.split_once("\r\n\r\n").unwrap();
    assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
    let frames: Vec<&str> = body.trim_end_matches('\n').split("\n\n").collect();
    let sent = Command::new("kill")
        .args(["-s", "TERM", &child.id().to_string()])
        .status()
        .unwrap();
    assert!(sent.success());
    let status = child.wait().unwrap();
    let stderr = reader.join().unwrap();
    let mut stdout = String::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut stdout)
        .unwrap();
    assert_eq!(
        status.code(),
        Some(0),
        "observe exits 0 on SIGTERM: {stderr}"
    );
    assert!(stdout.is_empty(), "{stdout}");
    assert_eq!(frames.len(), lines.len(), "one frame per line of the file");
    for (frame, line) in frames.iter().zip(&lines) {
        let event: Value = serde_json::from_str(line).unwrap();
        let expected = format!(
            "event: {}\nid: {}\ndata: {line}",
            event["body"]["kind"].as_str().unwrap(),
            event["sequence"]
        );
        assert_eq!(*frame, expected);
    }
}

/// What the held-out tally counts, recomputed from the packed corpus with the run's seed and
/// batch: the selected rows of the 68 full `HeldOut` batches, the label of each, and the
/// add-one unigram loss of `pack` over exactly those rows.
struct Evaluated {
    selected: u64,
    label_counts: BTreeMap<i32, u64>,
    top_train_label: i32,
    baseline_over_selected: f64,
    baseline_over_heldout: f64,
}

fn evaluated() -> Evaluated {
    use modelspec_runtime::dataset::SplitRole;
    use modelspec_runtime::dataset::mlm::{MaskingPolicy, load, mlm_batch};

    let packed = load(&data_dir()).expect("the packed directory loads");
    let vocabulary = packed.record.vocabulary_size as usize;
    let specials = 5i32;
    let mut train = vec![0u64; vocabulary];
    let mut train_total = 0u64;
    for &t in packed.ids(SplitRole::Train) {
        if t >= specials {
            train[t as usize] += 1;
            train_total += 1;
        }
    }
    let denominator = train_total as f64 + (vocabulary - specials as usize) as f64;
    let nll = |t: i32| -((train[t as usize] as f64 + 1.0) / denominator).ln();
    let (mut all_sum, mut all_n) = (0.0f64, 0u64);
    for &t in packed.ids(SplitRole::HeldOut) {
        if t >= specials {
            all_sum += nll(t);
            all_n += 1;
        }
    }
    let batches = packed.sequence_count(SplitRole::HeldOut) / BATCH;
    let (mut sum, mut selected) = (0.0f64, 0u64);
    let mut label_counts = BTreeMap::new();
    for b in 0..batches {
        let drawn = mlm_batch(
            &packed,
            SplitRole::HeldOut,
            &MaskingPolicy::BERT,
            SEED,
            b as u64,
            BATCH,
        )
        .unwrap();
        for (&label, &weight) in drawn.labels.iter().zip(&drawn.weights) {
            if weight > 0.0 {
                selected += 1;
                sum += nll(label);
                *label_counts.entry(label).or_insert(0) += 1;
            }
        }
    }
    let top_train_label = (0..vocabulary)
        .max_by_key(|&t| (train[t], std::cmp::Reverse(t)))
        .unwrap() as i32;
    Evaluated {
        selected,
        label_counts,
        top_train_label,
        baseline_over_selected: sum / selected as f64,
        baseline_over_heldout: all_sum / all_n as f64,
    }
}

/// Hypothesis 5: every printed accuracy is `k / selected` in `f32` for an integer `k`, the
/// plateau's `k` is the count of one label among the selected rows, and step 0's `k` is small.
/// Hypothesis 6: `packed.json`'s baseline is over every non-special `HeldOut` token, while the
/// eval loss is over the selected rows of the full batches; the last eval loss is below the
/// baseline over those same rows too.
#[test]
#[ignore = "record: needs MODELSPEC_GPU_PRETRAIN_DATA"]
fn record_accuracy_is_whole_hits_and_the_loss_beats_the_baseline_of_the_evaluated_rows() {
    let e = evaluated();
    let (_, evals) = fixture();
    let top = e
        .label_counts
        .iter()
        .max_by_key(|(l, c)| (**c, std::cmp::Reverse(**l)))
        .unwrap();
    eprintln!(
        "selected {} | top held-out label {} x{} | top train label {} x{} | baseline all held-out {} (packed.json {BASELINE}) | baseline selected rows {}",
        e.selected,
        top.0,
        top.1,
        e.top_train_label,
        e.label_counts.get(&e.top_train_label).copied().unwrap_or(0),
        e.baseline_over_heldout,
        e.baseline_over_selected
    );
    let mut hits = Vec::new();
    for eval in &evals {
        let accuracy: f32 = eval.accuracy.parse().unwrap();
        let k = (0..=e.selected)
            .find(|&k| k as f32 / e.selected as f32 == accuracy)
            .unwrap_or_else(|| {
                panic!(
                    "eval {}: {} is no k / {}",
                    eval.step, eval.accuracy, e.selected
                )
            });
        hits.push((eval.step, k));
    }
    eprintln!("hits {hits:?}");
    assert!(
        (e.baseline_over_heldout - BASELINE).abs() < 1e-9,
        "pack's baseline recomputed: {}",
        e.baseline_over_heldout
    );
    let plateau = hits.iter().find(|(s, _)| *s == 1000).unwrap().1;
    assert!(
        e.label_counts.values().any(|&c| c == plateau),
        "the plateau's {plateau} hits are the count of one label"
    );
    assert_eq!(e.selected, EVALUATED_ROWS);
    assert_eq!(e.baseline_over_selected, EVALUATED_BASELINE);
    assert_eq!(top.1, &plateau, "the plateau is the count of the top label");
    assert_eq!(*top.0, e.top_train_label, "which is the top Train label");
    let last: f64 = evals.last().unwrap().loss();
    assert!(
        last < e.baseline_over_selected,
        "the last eval loss {last} is below the baseline over the evaluated rows {}",
        e.baseline_over_selected
    );
}

/// The probe of hypothesis 8 on the device: the example at `steps = 100; warmup_steps = 10` with
/// `--eval-every 50 --record`, twice, under the validation layer with its log forced to a file.
/// Both runs print the same stdout and record the same events but for ids; the step-0 `step`
/// and `eval` lines and every `init` line equal the full run's; the evals are unit 20's
/// 9.029848, 8.713591 and 8.539593; the validation log holds no message.
#[test]
#[ignore = "device: needs a Vulkan device, MODELSPEC_GPU_PRETRAIN_RUN and MODELSPEC_GPU_PRETRAIN_DATA"]
fn device_short_runs_repeat_and_start_where_the_full_run_started() {
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR")).join("adversary_gpu_pretrain");
    std::fs::create_dir_all(&tmp).unwrap();
    let source = read(EXAMPLE)
        .replace("warmup_steps = 1000)", "warmup_steps = 10)")
        .replace("    steps = 10000;", "    steps = 100;");
    assert!(source.contains("    steps = 100;") && source.contains("warmup_steps = 10)"));
    let spec = tmp.join("bert_tiny_100.mspec");
    std::fs::write(&spec, source).unwrap();
    let data = data_dir();
    let run = |name: &str| {
        let record = tmp.join(format!("{name}.events.jsonl"));
        let log = tmp.join(format!("{name}.validation.log"));
        let _ = std::fs::remove_file(&record);
        let _ = std::fs::remove_file(&log);
        let out = Command::new(env!("CARGO_BIN_EXE_modelspec"))
            .args([
                "train",
                spec.to_str().unwrap(),
                "--backend",
                "vulkan",
                "--data",
            ])
            .arg(&data)
            .args(["--eval-every", "50", "--record"])
            .arg(&record)
            .env("MODELSPEC_VK_VALIDATION", "1")
            .env(
                "VK_KHRONOS_VALIDATION_DEBUG_ACTION",
                "VK_DBG_LAYER_ACTION_LOG_MSG",
            )
            .env("VK_KHRONOS_VALIDATION_REPORT_FLAGS", "error,warn,perf,info")
            .env("VK_KHRONOS_VALIDATION_LOG_FILENAME", &log)
            .stdin(Stdio::null())
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(out.status.success(), "{name}: {stderr}");
        let validation = std::fs::read_to_string(&log).unwrap_or_default();
        let events: Vec<Value> = std::fs::read_to_string(&record)
            .unwrap()
            .lines()
            .map(|l| {
                let mut v: Value = serde_json::from_str(l).unwrap();
                v.as_object_mut().unwrap().remove("run_event_id");
                v.as_object_mut().unwrap().remove("training_run_id");
                v
            })
            .collect();
        // The layer announces its log file on stdout; the run's own lines are the rest.
        let stdout: String = String::from_utf8(out.stdout)
            .unwrap()
            .lines()
            .filter(|l| !l.starts_with("Validation Layer Info - Logging validation error to "))
            .map(|l| format!("{l}\n"))
            .collect();
        (stdout, stderr, validation, events)
    };
    let (first, first_err, first_log, first_events) = run("first");
    let (second, _, second_log, second_events) = run("second");
    let differing = first.lines().zip(second.lines()).position(|(a, b)| a != b);
    assert!(
        differing.is_none() && first.lines().count() == second.lines().count(),
        "two runs print the same stdout; first differing line {differing:?}"
    );
    assert!(
        first_events == second_events,
        "two runs record the same events but for ids"
    );
    assert_eq!(first_err, "", "no stderr");
    for log in [&first_log, &second_log] {
        assert!(
            log.contains("CURRENT-VALIDATION-ENABLED"),
            "the layer logs to the file: {log}"
        );
        let reported: Vec<&str> = log
            .lines()
            .filter(|l| l.starts_with("Validation ") && !l.starts_with("Validation Information"))
            .collect();
        assert!(reported.is_empty(), "validation messages: {reported:?}");
    }

    let (_, evals) = parse_lines(&first);
    let losses: Vec<&str> = evals.iter().map(|e| e.loss.as_str()).collect();
    assert_eq!(losses, ["9.029848", "8.713591", "8.539593"]);
    let full = std::fs::read_to_string(run_dir().join("full.stdout")).unwrap();
    let starts = |text: &str| -> Vec<String> {
        text.lines()
            .filter(|l| {
                l.starts_with("data ")
                    || l.starts_with("init ")
                    || l.starts_with("step 0 ")
                    || l.starts_with("eval 0 ")
            })
            .map(str::to_owned)
            .collect()
    };
    let (short, long) = (starts(&first), starts(&full));
    assert_eq!(
        long.len(),
        44,
        "data, 41 init, step 0 and eval 0 of the full run"
    );
    let differing: Vec<String> = short
        .iter()
        .zip(&long)
        .filter(|(a, b)| a != b)
        .map(|(a, _)| a.chars().take(60).collect())
        .collect();
    assert!(
        short.len() == long.len() && differing.is_empty(),
        "the same start as the full run: {} lines against {}, differing {differing:?}",
        short.len(),
        long.len()
    );
}
