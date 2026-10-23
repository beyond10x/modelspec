//! The observatory UI (story:observatory-ui-live-loss-chart; `spec/domains/observatory.yaml`,
//! `UiDirectory`): `--ui <dir>` on `modelspec train --serve` and `modelspec observe`, the fixture
//! recordings the UI is checked against, and a headless browser reading the charts.
//!
//! Scenario ids of the story map to the tests: `observatory.ui.live-linear-regression` is
//! [`observatory_ui_live_linear_regression`], `observatory.ui.live-two-layer-relu` is
//! [`observatory_ui_live_two_layer_relu`], `observatory.ui.replay-linear-regression` is
//! [`observatory_ui_replay_linear_regression`], `observatory.ui.replay-two-layer-relu` is
//! [`observatory_ui_replay_two_layer_relu`] and `observatory.ui.metric-charts` is
//! [`observatory_ui_metric_charts`]. Those five are `#[ignore]`d because they need the built app
//! (`ui/dist`, `task ui`, which needs Node and pnpm) and a Chrome or Chromium (the `CHROME`
//! variable, else the first of `google-chrome-stable`, `chromium`, ... on `PATH`); `task
//! ui-browser`, part of `task ci`, runs them with `--ignored`. Everything else here runs in `task
//! test` without Node, pnpm or a browser.
//!
//! The browser reads each chart from the attributes the app writes on its `svg[data-metric]`:
//! `data-points` (one per `StepCompleted`), `data-last-step`, `data-last-value` (JavaScript's
//! `String` of the value, which reads back as the same binary64, or `null`) and the `d` of its
//! `path[data-line]` (one `M` or `L` command per finite point).
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

/// Opens `http://<address>/` in a headless browser, waits until the page says the run ended and
/// reads its charts.
fn charts(address: &str) -> (String, Vec<Chart>) {
    let browser = browser();
    let tab = browser.new_tab().expect("a tab opens");
    tab.navigate_to(&format!("http://{address}/"))
        .expect("the page is requested");
    tab.wait_until_navigated().expect("the page loads");
    let start = Instant::now();
    let state = loop {
        let state = tab
            .evaluate(
                "document.querySelector('main')?.dataset.runState ?? 'none'",
                false,
            )
            .expect("the page answers")
            .value
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_default();
        if state == "finished" || state == "failed" {
            break state;
        }
        assert!(
            start.elapsed() < WAIT,
            "the page still says {state:?} after {WAIT:?}"
        );
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
    (state, charts)
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
fn assert_loss_only(served: &Served) {
    let (state, charts) = charts(&served.address);
    assert_eq!(state, "finished");
    let data = streamed(&served.address);
    assert_eq!(
        charts.iter().map(|c| c.metric.as_str()).collect::<Vec<_>>(),
        ["loss"],
        "{charts:?}"
    );
    assert_loss_chart(&charts[0], &data);
}

fn live(model: &str) {
    let ui = built_ui();
    let mut served = Served::start(&[
        "train",
        model,
        "--backend",
        "cpu",
        "--serve",
        "127.0.0.1:0",
        "--ui",
        &ui,
    ]);
    assert_eq!(served.next_line(), format!("ui http://{}/", served.address));
    assert_loss_only(&served);
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
    let (state, charts) = charts(&served.address);
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
