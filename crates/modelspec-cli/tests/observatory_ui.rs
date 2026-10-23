//! The observatory UI (story:observatory-ui-live-loss-chart; `spec/domains/observatory.yaml`,
//! `UiDirectory`): `--ui <dir>` on `modelspec train --serve` and `modelspec observe`, the fixture
//! recordings the UI is checked against, and a headless browser reading the charts.
//!
//! Scenario ids of the story map to the tests: `observatory.ui.live-linear-regression` is
//! [`observatory_ui_live_linear_regression`], `observatory.ui.live-two-layer-relu` is
//! [`observatory_ui_live_two_layer_relu`], `observatory.ui.replay-linear-regression` is
//! [`observatory_ui_replay_linear_regression`], `observatory.ui.replay-two-layer-relu` is
//! [`observatory_ui_replay_two_layer_relu`] and `observatory.ui.metric-charts` is
//! [`observatory_ui_metric_charts`]; `observatory.ui.parameter-stats-two-layer-relu`, of
//! story:observatory-parameter-statistics, is [`observatory_ui_parameter_stats_two_layer_relu`].
//! Those six are `#[ignore]`d because they need the built app
//! (`ui/dist`, `task ui`, which needs Node and pnpm) and a Chrome or Chromium (the `CHROME`
//! variable, else the first of `google-chrome-stable`, `chromium`, ... on `PATH`); `task
//! ui-browser`, part of `task ci`, runs them with `--ignored`. Everything else here runs in `task
//! test` without Node, pnpm or a browser.
//!
//! The browser reads each chart from the attributes the app writes on its `svg[data-metric]`:
//! `data-points` (one per `StepCompleted`), `data-last-step`, `data-last-value` (JavaScript's
//! `String` of the value, which reads back as the same binary64, or `null`) and the `d` of its
//! `path[data-line]` (one `M` or `L` command per finite point). The norm charts of a parameter
//! carry the same attributes on an `svg[data-norm=<path>][data-subject=<Value|Gradient>]`, and the
//! histogram of the shown step an `svg[data-histogram=<path>][data-subject=...]` with `data-step`
//! and `data-counts` (the bin counts joined by commas).
//!
//! The fixtures `tests/fixtures/two_layer_relu.run.jsonl` and `tests/fixtures/accuracy.run.jsonl`
//! are rewritten by running [`observatory_ui_two_layer_relu_fixture_current`] and
//! [`observatory_ui_accuracy_fixture_current`] with `MODELSPEC_REGENERATE_FIXTURES=1`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use modelspec_observatory::{
    MetricSample, RunEvent, RunEventBody, RunEventId, RunEventState, RunStarted, StepCompleted,
    TrainingRunId, line, parse_recording,
};
use serde_json::Value;
use uuid::Uuid;

const LINEAR: &str = "examples/linear_regression.mspec";
const TWO_LAYER: &str = "crates/modelspec-docs/models/two_layer_relu.mspec";
const LINEAR_FIXTURE: &str = "crates/modelspec-cli/tests/fixtures/linear_regression.run.jsonl";
const TWO_LAYER_FIXTURE: &str = "crates/modelspec-cli/tests/fixtures/two_layer_relu.run.jsonl";
const ACCURACY_FIXTURE: &str = "crates/modelspec-cli/tests/fixtures/accuracy.run.jsonl";
const BUILT_UI: &str = "ui/dist";
const REGENERATE: &str = "MODELSPEC_REGENERATE_FIXTURES";
const WAIT: Duration = Duration::from_secs(120);

/// The fields of a `RunEvent` that differ per invocation (`RecordedRunFile`).
const PER_INVOCATION: [&str; 2] = ["training_run_id", "run_event_id"];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("output is UTF-8")
}

/// A fresh path `name` under the test temp dir, with nothing at it.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("observatory_ui");
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    let path = dir.join(name);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir_all(&path);
    path
}

/// A fresh directory `name` holding a stand-in for the built app: an `index.html` and one asset.
fn stand_in_ui(name: &str) -> PathBuf {
    let dir = scratch(name);
    std::fs::create_dir_all(dir.join("assets")).expect("the temp dir is writable");
    std::fs::write(
        dir.join("index.html"),
        "<!doctype html><title>stand-in</title>",
    )
    .unwrap();
    std::fs::write(dir.join("assets/app.js"), "void 0;").unwrap();
    dir
}

fn path_arg(path: &Path) -> String {
    path.display().to_string()
}

/// A finished `modelspec` process.
struct Finished {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: String,
}

/// Runs `modelspec <args>` from the repository root to its end, killing it after [`WAIT`].
fn modelspec(args: &[&str]) -> Finished {
    let mut child = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("modelspec starts");
    let mut out = child.stdout.take().expect("stdout is piped");
    let mut err = child.stderr.take().expect("stderr is piped");
    let stdout = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = out.read_to_end(&mut bytes);
        bytes
    });
    let stderr = std::thread::spawn(move || {
        let mut all = String::new();
        let _ = err.read_to_string(&mut all);
        all
    });
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("the status is read") {
            break status;
        }
        if start.elapsed() > WAIT {
            let _ = child.kill();
            let _ = child.wait();
            panic!("modelspec {args:?} still runs after {WAIT:?}");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    Finished {
        status,
        stdout: stdout.join().expect("stdout reader"),
        stderr: stderr.join().expect("stderr reader"),
    }
}

/// A serving `modelspec` process, killed when dropped.
struct Served {
    child: Child,
    address: String,
    lines: Receiver<String>,
    stderr: Option<JoinHandle<String>>,
}

impl Served {
    /// Starts `modelspec <args>` and waits for the `serving http://<addr>/events` line on stderr.
    fn start(args: &[&str]) -> Served {
        let mut child = Command::new(env!("CARGO_BIN_EXE_modelspec"))
            .args(args)
            .current_dir(repo_root())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("modelspec starts");
        let (sender, lines) = mpsc::channel();
        let err = child.stderr.take().expect("stderr is piped");
        let stderr = std::thread::spawn(move || {
            let mut all = String::new();
            for line in BufReader::new(err).lines() {
                let Ok(line) = line else { break };
                all.push_str(&line);
                all.push('\n');
                let _ = sender.send(line);
            }
            all
        });
        let mut served = Served {
            child,
            address: String::new(),
            lines,
            stderr: Some(stderr),
        };
        let line = served.next_line();
        let address = line
            .strip_prefix("serving http://")
            .and_then(|rest| rest.strip_suffix("/events"))
            .unwrap_or_else(|| panic!("`serving http://<addr>/events` first on stderr: {line}"));
        served.address = address.to_owned();
        served
    }

    /// The next line on stderr, waiting up to [`WAIT`].
    fn next_line(&mut self) -> String {
        match self.lines.recv_timeout(WAIT) {
            Ok(line) => line,
            Err(error) => {
                let status = self.child.try_wait();
                let _ = self.child.kill();
                let _ = self.child.wait();
                let stderr = self
                    .stderr
                    .take()
                    .map(|handle| handle.join().expect("stderr reader"))
                    .unwrap_or_default();
                panic!("no line on stderr ({error}, status {status:?}): {stderr}");
            }
        }
    }
}

impl Drop for Served {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The response to `GET <path>` at `address`: its status line and body.
fn get(address: &str, path: &str) -> (String, Vec<u8>) {
    let mut stream = TcpStream::connect(address).expect("the server accepts a connection");
    stream.set_read_timeout(Some(WAIT)).expect("a read timeout");
    write!(stream, "GET {path} HTTP/1.1\r\nHost: {address}\r\n\r\n").expect("the request is sent");
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .expect("the server ends the response");
    let split = response
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .unwrap_or_else(|| panic!("a response head: {response:?}"));
    let head = text(&response[..split]);
    let status = head.lines().next().unwrap_or_default().to_owned();
    (status, response[split + 4..].to_vec())
}

/// The `data:` lines of `GET /events` at `address`, read until the server closes.
fn streamed(address: &str) -> Vec<String> {
    let (status, body) = get(address, "/events");
    assert_eq!(status, "HTTP/1.1 200 OK");
    text(&body)
        .lines()
        .filter_map(|line| line.strip_prefix("data: ").map(str::to_owned))
        .collect()
}

fn json(line: &str) -> Value {
    serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {line}"))
}

/// The lines of a recorded run file, after checking that it ends with a newline.
fn recorded_lines(path: &Path) -> Vec<String> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("the recording {} is readable: {e}", path.display()));
    assert!(content.ends_with('\n'), "the recording ends with a newline");
    content.lines().map(str::to_owned).collect()
}

/// Each event of `lines` without the per-invocation fields.
fn invariant(lines: &[String]) -> Vec<Value> {
    lines
        .iter()
        .map(|line| {
            let mut event = json(line);
            let object = event.as_object_mut().expect("an event is an object");
            for field in PER_INVOCATION {
                assert!(object.remove(field).is_some(), "{field} in {line}");
            }
            event
        })
        .collect()
}

/// The JSON number text after `"<key>":` in `data`, as the server wrote it, so it is read with
/// Rust's correctly rounded `str::parse` and not through a JSON number type.
fn raw_number<'a>(data: &'a str, key: &str) -> &'a str {
    let marker = format!("\"{key}\":");
    let start = data
        .find(&marker)
        .unwrap_or_else(|| panic!("{marker} in {data}"))
        + marker.len();
    let rest = &data[start..];
    let end = rest
        .find([',', '}'])
        .unwrap_or_else(|| panic!("a value after {marker} in {data}"));
    &rest[..end]
}

/// A loss or metric value as a chart compares it: `None` for `null`, else its binary64 bits.
fn value_bits(raw: &str) -> Option<u64> {
    (raw != "null").then(|| {
        raw.parse::<f64>()
            .unwrap_or_else(|e| panic!("{e}: {raw}"))
            .to_bits()
    })
}

/// The `(step, loss)` of every `StepCompleted` among the `data:` lines, losses as raw text.
fn losses(data: &[String]) -> Vec<(u64, String)> {
    data.iter()
        .filter(|line| json(line)["body"]["kind"] == "StepCompleted")
        .map(|line| {
            let step = json(line)["body"]["value"]["step"]
                .as_u64()
                .expect("a step");
            (step, raw_number(line, "loss").to_owned())
        })
        .collect()
}

/// Records `model` with `train --record` and no `--serve` into `name`.
fn record(model: &str, name: &str) -> PathBuf {
    let path = scratch(name);
    let recorded = modelspec(&[
        "train",
        model,
        "--backend",
        "cpu",
        "--record",
        &path_arg(&path),
    ]);
    assert_eq!(recorded.status.code(), Some(0), "{}", recorded.stderr);
    assert!(recorded.stderr.is_empty(), "stderr: {}", recorded.stderr);
    path
}

/// The deterministic run of the `accuracy` fixture: steps 0 to 20, each `StepCompleted` with a
/// loss and one `MetricSample` named `accuracy`, then `RunFinished`. No M0 producer fills
/// `metrics`, so this helper writes the events itself (story, Scope).
fn accuracy_events() -> Vec<RunEvent> {
    const STEPS: u64 = 21;
    let training_run_id = TrainingRunId(Uuid::from_u128(0x6163_6375_7261_6379_0000_0000_0000_0000));
    let mut bodies = vec![RunEventBody::RunStarted(RunStarted {
        steps: STEPS,
        start_step: 0,
    })];
    for step in 0..STEPS {
        let k = step as f64;
        bodies.push(RunEventBody::StepCompleted(StepCompleted {
            step,
            loss: f64::from(2.0f32 / (1.0 + step as f32)),
            metrics: vec![MetricSample {
                name: "accuracy".into(),
                value: 0.25 + 0.03 * k,
            }],
            parameters: Vec::new(),
        }));
    }
    bodies.push(RunEventBody::RunFinished);
    bodies
        .into_iter()
        .zip(0u64..)
        .map(|(body, sequence)| RunEvent {
            run_event_id: RunEventId(Uuid::from_u128(
                0x6576_656e_7400_0000_0000_0000_0000_0000 | u128::from(sequence),
            )),
            training_run_id,
            sequence,
            body,
            state: RunEventState::Emitted,
        })
        .collect()
}

/// `tests/fixtures/two_layer_relu.run.jsonl` equals a fresh recording of
/// `crates/modelspec-docs/models/two_layer_relu.mspec` except for the per-invocation fields.
#[test]
fn observatory_ui_two_layer_relu_fixture_current() {
    let path = record(TWO_LAYER, "two_layer_relu.run.jsonl");
    let fixture = repo_root().join(TWO_LAYER_FIXTURE);
    if std::env::var_os(REGENERATE).is_some() {
        std::fs::copy(&path, &fixture).expect("the fixture is written");
    }
    let committed = recorded_lines(&fixture);
    let content = std::fs::read_to_string(&fixture).unwrap();
    let events = parse_recording(&content).expect("the fixture is a whole recording");
    assert_eq!(
        events.len(),
        203,
        "RunStarted, a StepCompleted for each of steps 0 to 200 and RunFinished"
    );
    assert_eq!(
        invariant(&committed),
        invariant(&recorded_lines(&path)),
        "the fixture is a recording of {TWO_LAYER}; rerun this test with {REGENERATE}=1"
    );
}

/// `tests/fixtures/accuracy.run.jsonl` is, byte for byte, the run [`accuracy_events`] writes, and
/// a recording `modelspec observe` accepts.
#[test]
fn observatory_ui_accuracy_fixture_current() {
    let fresh: String = accuracy_events().iter().map(line).collect();
    let fixture = repo_root().join(ACCURACY_FIXTURE);
    if std::env::var_os(REGENERATE).is_some() {
        std::fs::write(&fixture, &fresh).expect("the fixture is written");
    }
    let committed = std::fs::read_to_string(&fixture)
        .unwrap_or_else(|e| panic!("{ACCURACY_FIXTURE} is committed: {e}"));
    assert_eq!(
        committed, fresh,
        "rerun this test with {REGENERATE}=1 to rewrite {ACCURACY_FIXTURE}"
    );
    assert_eq!(parse_recording(&committed).unwrap(), accuracy_events());
}

/// `observe --ui <dir>` serves the directory at `/` beside the stream and says where on stderr.
#[test]
fn observe_serves_the_ui_beside_the_stream() {
    let ui = stand_in_ui("observe_ui");
    let mut served = Served::start(&["observe", LINEAR_FIXTURE, "--ui", &path_arg(&ui)]);
    assert_eq!(served.next_line(), format!("ui http://{}/", served.address));
    let (status, body) = get(&served.address, "/");
    assert_eq!(status, "HTTP/1.1 200 OK");
    assert_eq!(text(&body), "<!doctype html><title>stand-in</title>");
    assert_eq!(get(&served.address, "/assets/app.js").0, "HTTP/1.1 200 OK");
    assert_eq!(
        streamed(&served.address).len(),
        recorded_lines(&repo_root().join(LINEAR_FIXTURE)).len()
    );
}

/// `train --serve <addr> --ui <dir>` serves the directory at `/` beside the live stream.
#[test]
fn train_serves_the_ui_beside_the_stream() {
    let ui = stand_in_ui("train_ui");
    let mut served = Served::start(&[
        "train",
        LINEAR,
        "--backend",
        "cpu",
        "--serve",
        "127.0.0.1:0",
        "--ui",
        &path_arg(&ui),
    ]);
    assert_eq!(served.next_line(), format!("ui http://{}/", served.address));
    assert_eq!(get(&served.address, "/").0, "HTTP/1.1 200 OK");
    let data = streamed(&served.address);
    assert_eq!(json(&data[data.len() - 1])["body"]["kind"], "RunFinished");
}

/// A `--ui` directory that cannot be served is refused on one line with exit 1, before anything is
/// bound or run.
#[test]
fn a_ui_directory_that_cannot_be_served_is_refused() {
    let empty = scratch("empty_ui");
    std::fs::create_dir_all(&empty).unwrap();
    let missing = scratch("missing_ui");
    for dir in [&empty, &missing] {
        let dir = path_arg(dir);
        for args in [
            vec![
                "train",
                LINEAR,
                "--backend",
                "cpu",
                "--serve",
                "127.0.0.1:0",
                "--ui",
                &dir,
            ],
            vec!["observe", LINEAR_FIXTURE, "--ui", &dir],
        ] {
            let refused = modelspec(&args);
            assert_eq!(
                refused.status.code(),
                Some(1),
                "{args:?}: {}",
                refused.stderr
            );
            assert!(refused.stdout.is_empty(), "{args:?}: no step is run");
            assert_eq!(
                refused.stderr.lines().count(),
                1,
                "{args:?}: {}",
                refused.stderr
            );
            assert!(
                refused.stderr.contains("cannot serve the UI from"),
                "{args:?}: {}",
                refused.stderr
            );
            assert!(
                !refused.stderr.contains("serving"),
                "{args:?}: nothing is served"
            );
        }
    }
}

/// `--ui` serves next to the stream, so `train --ui` without `--serve` is a usage error.
#[test]
fn train_ui_without_serve_is_a_usage_error() {
    let ui = stand_in_ui("usage_ui");
    let refused = modelspec(&["train", LINEAR, "--backend", "cpu", "--ui", &path_arg(&ui)]);
    assert_eq!(refused.status.code(), Some(2), "{}", refused.stderr);
    assert!(refused.stdout.is_empty());
    assert!(refused.stderr.contains("--serve"), "{}", refused.stderr);
}

/// One chart as the browser read it.
#[derive(Debug)]
struct Chart {
    metric: String,
    points: usize,
    last_step: Option<u64>,
    last_value: Option<String>,
    /// The `M` and `L` commands of the chart's line.
    drawn: usize,
}

/// A headless Chrome or Chromium, from `CHROME` or `PATH`.
fn browser() -> headless_chrome::Browser {
    let options = headless_chrome::LaunchOptions::default_builder()
        .headless(true)
        .sandbox(false)
        .idle_browser_timeout(Duration::from_secs(600))
        .build()
        .expect("launch options");
    headless_chrome::Browser::new(options).unwrap_or_else(|e| {
        panic!("a Chrome or Chromium starts (set CHROME to its executable): {e:#}")
    })
}

fn built_ui() -> String {
    let dir = repo_root().join(BUILT_UI);
    assert!(
        dir.join("index.html").is_file(),
        "{BUILT_UI}/index.html exists; run `task ui` first"
    );
    path_arg(&dir)
}

/// What the page said at one reading before the run ended: its run state and the points of its
/// loss chart.
#[derive(Debug, Clone, PartialEq)]
struct Reading {
    state: String,
    loss_points: u64,
}

/// The charts the page drew once the run ended, the state it ended in and every distinct
/// [`Reading`] before that.
struct PageRead {
    state: String,
    charts: Vec<Chart>,
    before: Vec<Reading>,
}

/// Opens `http://<address>/` in a headless browser, reads it every 50 ms until the page says the
/// run ended and reads its charts.
fn charts(address: &str) -> PageRead {
    let browser = browser();
    let tab = browser.new_tab().expect("a tab opens");
    tab.navigate_to(&format!("http://{address}/"))
        .expect("the page is requested");
    tab.wait_until_navigated().expect("the page loads");
    let start = Instant::now();
    let mut before: Vec<Reading> = Vec::new();
    let state = loop {
        let read = tab
            .evaluate(
                "JSON.stringify({
                    state: document.querySelector('main')?.dataset.runState ?? 'none',
                    points: Number(document.querySelector('svg[data-metric=\"loss\"]')?.dataset.points ?? 0),
                })",
                false,
            )
            .expect("the page answers")
            .value
            .and_then(|v| v.as_str().map(str::to_owned))
            .expect("a JSON string");
        let read = json(&read);
        let reading = Reading {
            state: read["state"].as_str().unwrap_or_default().to_owned(),
            loss_points: read["points"].as_u64().unwrap_or_default(),
        };
        if reading.state == "finished" || reading.state == "failed" {
            break reading.state;
        }
        assert!(
            start.elapsed() < WAIT,
            "the page still says {reading:?} after {WAIT:?}"
        );
        if before.last() != Some(&reading) {
            before.push(reading);
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let read = tab
        .evaluate(
            "JSON.stringify([...document.querySelectorAll('svg[data-metric]')].map((svg) => ({
                metric: svg.dataset.metric,
                points: Number(svg.dataset.points),
                lastStep: svg.dataset.lastStep ?? null,
                lastValue: svg.dataset.lastValue ?? null,
                drawn: (svg.querySelector('path[data-line]')?.getAttribute('d') ?? '')
                    .split(/(?=[ML])/).filter((c) => c.trim().length > 0).length,
            })))",
            false,
        )
        .expect("the charts are read")
        .value
        .and_then(|v| v.as_str().map(str::to_owned))
        .expect("a JSON string");
    let charts = json(&read)
        .as_array()
        .expect("an array")
        .iter()
        .map(|chart| Chart {
            metric: chart["metric"].as_str().expect("a metric").to_owned(),
            points: chart["points"].as_u64().expect("points") as usize,
            last_step: chart["lastStep"]
                .as_str()
                .map(|s| s.parse().expect("a step")),
            last_value: chart["lastValue"].as_str().map(str::to_owned),
            drawn: chart["drawn"].as_u64().expect("drawn") as usize,
        })
        .collect();
    PageRead {
        state,
        charts,
        before,
    }
}

/// The loss chart has one point per `StepCompleted` of `data`, a drawn point per finite loss, and
/// its last point is the step and the binary64 of the last `StepCompleted`'s loss.
fn assert_loss_chart(chart: &Chart, data: &[String]) {
    let losses = losses(data);
    assert!(!losses.is_empty(), "the stream has StepCompleted events");
    assert_eq!(chart.metric, "loss");
    assert_eq!(chart.points, losses.len(), "one point per StepCompleted");
    let finite = losses.iter().filter(|(_, raw)| raw != "null").count();
    assert_eq!(chart.drawn, finite, "one drawn point per finite loss");
    let (step, raw) = losses.last().expect("a last StepCompleted");
    assert_eq!(chart.last_step, Some(*step));
    let shown = chart.last_value.as_deref().expect("a last value");
    assert_eq!(
        value_bits(shown),
        value_bits(raw),
        "the chart's last loss {shown} is the stream's {raw}"
    );
}

/// Opens the UI served at `served`, then reads the stream it read and checks the loss chart
/// against it, and that the run's only chart is the loss.
/// Returns what the page said before the run ended and the number of `StepCompleted` events.
fn assert_loss_only(served: &Served) -> (Vec<Reading>, usize) {
    let read = charts(&served.address);
    assert_eq!(read.state, "finished");
    let data = streamed(&served.address);
    let charts = &read.charts;
    assert_eq!(
        charts.iter().map(|c| c.metric.as_str()).collect::<Vec<_>>(),
        ["loss"],
        "{charts:?}"
    );
    assert_loss_chart(&charts[0], &data);
    (read.before, losses(&data).len())
}

/// The steps of a live scenario's run: the models' own 500 and 200 end in under 0.1 s, before the
/// browser has opened the page, and 50000 take 5 and 9 s on the CPU backend (debug build, on the
/// machine this was written on), so the page reads the stream while the run is still emitting it.
const LIVE_STEPS: &str = "50000";

/// A live scenario: the page opened during a run of [`LIVE_STEPS`] steps says `streaming` with
/// some but not all of the loss points at one reading at least, and at the end draws the loss
/// chart the stream gives.
fn live(model: &str) {
    let ui = built_ui();
    let mut served = Served::start(&[
        "train",
        model,
        "--backend",
        "cpu",
        "--steps",
        LIVE_STEPS,
        "--serve",
        "127.0.0.1:0",
        "--ui",
        &ui,
    ]);
    assert_eq!(served.next_line(), format!("ui http://{}/", served.address));
    let (before, points) = assert_loss_only(&served);
    assert!(
        before.iter().any(|r| r.state == "streaming"
            && r.loss_points > 0
            && (r.loss_points as usize) < points),
        "the page drew the run of {points} loss points while it streamed: {before:?}"
    );
}

fn replay(fixture: &str) {
    let ui = built_ui();
    let mut served = Served::start(&["observe", fixture, "--ui", &ui]);
    assert_eq!(served.next_line(), format!("ui http://{}/", served.address));
    assert_loss_only(&served);
}

/// `observatory.ui.live-linear-regression`.
#[test]
#[ignore = "needs ui/dist and a Chrome or Chromium; run by `task ui-browser`"]
fn observatory_ui_live_linear_regression() {
    live(LINEAR);
}

/// `observatory.ui.live-two-layer-relu`.
#[test]
#[ignore = "needs ui/dist and a Chrome or Chromium; run by `task ui-browser`"]
fn observatory_ui_live_two_layer_relu() {
    live(TWO_LAYER);
}

/// `observatory.ui.replay-linear-regression`: `modelspec observe` of the fixture, no training
/// process.
#[test]
#[ignore = "needs ui/dist and a Chrome or Chromium; run by `task ui-browser`"]
fn observatory_ui_replay_linear_regression() {
    replay(LINEAR_FIXTURE);
}

/// `observatory.ui.replay-two-layer-relu`: `modelspec observe` of the fixture, no training
/// process.
#[test]
#[ignore = "needs ui/dist and a Chrome or Chromium; run by `task ui-browser`"]
fn observatory_ui_replay_two_layer_relu() {
    replay(TWO_LAYER_FIXTURE);
}

/// `observatory.ui.metric-charts`: the `accuracy` fixture draws the loss chart and an `accuracy`
/// chart whose last point is the last sample; a recording with no metrics (the linear-regression
/// fixture, checked by [`observatory_ui_replay_linear_regression`] too) draws the loss chart only.
#[test]
#[ignore = "needs ui/dist and a Chrome or Chromium; run by `task ui-browser`"]
fn observatory_ui_metric_charts() {
    let ui = built_ui();
    let mut served = Served::start(&["observe", ACCURACY_FIXTURE, "--ui", &ui]);
    assert_eq!(served.next_line(), format!("ui http://{}/", served.address));
    let PageRead { state, charts, .. } = charts(&served.address);
    assert_eq!(state, "finished");
    assert_eq!(
        charts.iter().map(|c| c.metric.as_str()).collect::<Vec<_>>(),
        ["loss", "accuracy"],
        "{charts:?}"
    );
    assert_loss_chart(&charts[0], &streamed(&served.address));
    let samples: Vec<(u64, f64)> = accuracy_events()
        .iter()
        .filter_map(|event| match &event.body {
            RunEventBody::StepCompleted(step) => Some((step.step, step.metrics[0].value)),
            _ => None,
        })
        .collect();
    let accuracy = &charts[1];
    assert_eq!(accuracy.points, samples.len());
    assert_eq!(accuracy.drawn, samples.len());
    let (step, value) = samples.last().copied().expect("a last sample");
    assert_eq!(accuracy.last_step, Some(step));
    let shown = accuracy.last_value.as_deref().expect("a last value");
    assert_eq!(
        value_bits(shown),
        Some(value.to_bits()),
        "the chart's last accuracy {shown} is the last sample {value}"
    );
    drop(served);

    let mut plain = Served::start(&["observe", LINEAR_FIXTURE, "--ui", &ui]);
    assert_eq!(plain.next_line(), format!("ui http://{}/", plain.address));
    assert_loss_only(&plain);
}

/// One streamed `ParameterStatistics` entry: its step, its norm as the server wrote it and its
/// bin counts.
#[derive(Debug)]
struct Streamed {
    step: u64,
    norm: String,
    counts: Vec<u64>,
}

/// Every entry of `path` and `subject` among the `StepCompleted` events of `data`, in step order.
fn streamed_statistics(data: &[String], path: &str, subject: &str) -> Vec<Streamed> {
    let marker = format!(r#"{{"path":"{path}","subject":"{subject}","norm":"#);
    data.iter()
        .filter(|line| json(line)["body"]["kind"] == "StepCompleted")
        .filter_map(|line| {
            let event = json(line);
            let value = &event["body"]["value"];
            let entry = value["parameters"]
                .as_array()
                .expect("a parameters list")
                .iter()
                .find(|e| e["path"] == path && e["subject"] == subject)?;
            let at = line.find(&marker).expect("the entry's text") + marker.len();
            let rest = &line[at..];
            let norm = rest[..rest.find([',', '}']).expect("a norm")].to_owned();
            let counts = entry["histogram"]["counts"]
                .as_array()
                .expect("counts")
                .iter()
                .map(|c| c.as_u64().expect("a count"))
                .collect();
            Some(Streamed {
                step: value["step"].as_u64().expect("a step"),
                norm,
                counts,
            })
        })
        .collect()
}

/// What the page shows of one parameter for `PATH` (replaced): per subject, the norm chart's
/// points and last point and the histogram's step and counts, `null` where there is none; and the
/// note that stands for a missing gradient histogram.
const READ_PARAMETER: &str = "JSON.stringify({
    subjects: ['Value', 'Gradient'].map((subject) => {
        const norm = document.querySelector('svg[data-norm=\"PATH\"][data-subject=\"' + subject + '\"]');
        const hist = document.querySelector('svg[data-histogram=\"PATH\"][data-subject=\"' + subject + '\"]');
        return {
            subject,
            points: norm === null ? null : Number(norm.dataset.points),
            lastStep: norm?.dataset.lastStep ?? null,
            lastValue: norm?.dataset.lastValue ?? null,
            step: hist?.dataset.step ?? null,
            counts: hist?.dataset.counts ?? null,
        };
    }),
    note: document.querySelector('section[data-parameter=\"PATH\"] .note')?.textContent ?? null,
})";

fn read_parameter(tab: &headless_chrome::Tab, path: &str) -> Value {
    let read = tab
        .evaluate(&READ_PARAMETER.replace("PATH", path), false)
        .expect("the parameter charts are read")
        .value
        .and_then(|v| v.as_str().map(str::to_owned))
        .expect("a JSON string");
    json(&read)
}

/// The counts of a histogram as the page wrote them, `data-counts` split at commas.
fn shown_counts(subject: &Value) -> Option<Vec<u64>> {
    subject["counts"].as_str().map(|counts| {
        counts
            .split(',')
            .filter(|c| !c.is_empty())
            .map(|c| c.parse().expect("a count"))
            .collect()
    })
}

/// `observatory.ui.parameter-stats-two-layer-relu`: `modelspec observe` of the two-layer-relu
/// fixture draws, for each parameter, a value-norm and a gradient-norm chart whose points are the
/// streamed samples and whose last point is the last streamed norm, and the histograms of the
/// latest sampled step; picking another sampled step draws that step's streamed bin counts.
#[test]
#[ignore = "needs ui/dist and a Chrome or Chromium; run by `task ui-browser`"]
fn observatory_ui_parameter_stats_two_layer_relu() {
    let ui = built_ui();
    let mut served = Served::start(&["observe", TWO_LAYER_FIXTURE, "--ui", &ui]);
    assert_eq!(served.next_line(), format!("ui http://{}/", served.address));
    let data = streamed(&served.address);
    let browser = browser();
    let tab = browser.new_tab().expect("a tab opens");
    tab.navigate_to(&format!("http://{}/", served.address))
        .expect("the page is requested");
    tab.wait_until_navigated().expect("the page loads");
    let start = Instant::now();
    loop {
        let state = tab
            .evaluate(
                "document.querySelector('main')?.dataset.runState ?? 'none'",
                false,
            )
            .expect("the page answers")
            .value
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_default();
        if state == "finished" {
            break;
        }
        assert!(
            start.elapsed() < WAIT,
            "the page still says {state} after {WAIT:?}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    for path in ["TwoLayer.w1", "TwoLayer.w2"] {
        let values = streamed_statistics(&data, path, "Value");
        let gradients = streamed_statistics(&data, path, "Gradient");
        assert!(
            values.len() >= 2 && !gradients.is_empty(),
            "{path} has samples"
        );
        let read = read_parameter(&tab, path);
        for (shown, streamed) in read["subjects"]
            .as_array()
            .expect("subjects")
            .iter()
            .zip([&values, &gradients])
        {
            let last = streamed.last().expect("a last sample");
            assert_eq!(
                shown["points"].as_u64(),
                Some(streamed.len() as u64),
                "{shown}"
            );
            assert_eq!(
                shown["lastStep"].as_str(),
                Some(last.step.to_string().as_str())
            );
            let value = shown["lastValue"].as_str().expect("a last norm");
            assert_eq!(
                value_bits(value),
                value_bits(&last.norm),
                "{path} {}: the chart's last norm {value} is the stream's {}",
                shown["subject"],
                last.norm
            );
        }
        let latest = values.last().expect("a latest sample");
        let [value, gradient] = read["subjects"].as_array().expect("subjects").as_slice() else {
            panic!("two subjects: {read}");
        };
        assert_eq!(
            value["step"].as_str(),
            Some(latest.step.to_string().as_str())
        );
        assert_eq!(shown_counts(value).as_ref(), Some(&latest.counts));
        assert!(
            gradients.iter().all(|g| g.step != latest.step),
            "the last step has no gradient"
        );
        assert_eq!(gradient["step"], Value::Null, "{read}");
        assert_eq!(
            read["note"].as_str(),
            Some(format!("no gradient at step {}", latest.step).as_str())
        );
    }

    let picked = streamed_statistics(&data, "TwoLayer.w1", "Gradient")
        .last()
        .expect("a sample with a gradient")
        .step;
    tab.evaluate(
        &format!(
            "(() => {{
                const select = document.querySelector('select[data-step-select]');
                select.value = '{picked}';
                select.dispatchEvent(new Event('change'));
                return select.value;
            }})()"
        ),
        false,
    )
    .expect("a step is picked");
    let start = Instant::now();
    let read = loop {
        let read = read_parameter(&tab, "TwoLayer.w1");
        if read["subjects"][0]["step"].as_str() == Some(picked.to_string().as_str()) {
            break read;
        }
        assert!(start.elapsed() < WAIT, "the page still shows {read}");
        std::thread::sleep(Duration::from_millis(50));
    };
    for (shown, subject) in read["subjects"]
        .as_array()
        .expect("subjects")
        .iter()
        .zip(["Value", "Gradient"])
    {
        let streamed = streamed_statistics(&data, "TwoLayer.w1", subject);
        let at = streamed
            .iter()
            .find(|s| s.step == picked)
            .expect("the picked step was sampled");
        assert_eq!(shown["step"].as_str(), Some(picked.to_string().as_str()));
        assert_eq!(shown_counts(shown).as_ref(), Some(&at.counts), "{subject}");
    }
    assert_eq!(read["note"], Value::Null, "{read}");
}
