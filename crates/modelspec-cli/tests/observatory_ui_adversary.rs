//! Adversarial browser cases for the observatory UI (story:observatory-ui-live-loss-chart).
//!
//! Like the scenarios of `observatory_ui.rs` these need the built app (`ui/dist`, `task ui`) and a
//! Chrome or Chromium (`CHROME`), so they are `#[ignore]`d and run with `--ignored`.
//!
//! The cases here keep a run going while the page reads it, and stop it and serve another run at
//! the same address while the page is open.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use serde_json::Value;

const LINEAR: &str = "examples/linear_regression.mspec";
const LINEAR_FIXTURE: &str = "crates/modelspec-cli/tests/fixtures/linear_regression.run.jsonl";
const BUILT_UI: &str = "ui/dist";
const WAIT: Duration = Duration::from_secs(120);

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn built_ui() -> String {
    let dir = repo_root().join(BUILT_UI);
    assert!(
        dir.join("index.html").is_file(),
        "{BUILT_UI}/index.html exists; run `task ui` first"
    );
    dir.display().to_string()
}

/// A loopback port nothing listens on now, so a second server can bind it after the first.
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("a free port")
        .port()
}

/// A serving `modelspec` process, killed when dropped.
struct Served {
    child: Child,
    lines: Receiver<String>,
}

impl Served {
    /// Starts `modelspec <args>` and waits for its `ui http://<addr>/` line on stderr.
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
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines() {
                let Ok(line) = line else { break };
                let _ = sender.send(line);
            }
        });
        let served = Served { child, lines };
        loop {
            let line = served
                .lines
                .recv_timeout(WAIT)
                .unwrap_or_else(|e| panic!("modelspec {args:?} prints `ui http://...`: {e}"));
            if line.starts_with("ui http://") {
                break;
            }
        }
        served
    }

    fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Served {
    fn drop(&mut self) {
        self.kill();
    }
}

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

/// Opens `http://<address>/` in a new tab of `browser` and returns the tab once the app has
/// mounted (its `main` element exists). Not `wait_until_navigated`: that returns on the
/// `networkAlmostIdle` lifecycle event, which the page's event stream holds off for as long as a
/// fast run emits, so it would return only once the page already says `finished`
/// (`observatory_ui.rs`, `open_page`).
fn open_page(
    browser: &headless_chrome::Browser,
    address: &str,
) -> std::sync::Arc<headless_chrome::Tab> {
    let tab = browser.new_tab().expect("a tab opens");
    tab.navigate_to(&format!("http://{address}/"))
        .expect("the page is requested");
    let start = Instant::now();
    loop {
        let mounted = tab
            .evaluate("document.querySelector('main') !== null", false)
            .ok()
            .and_then(|r| r.value)
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if mounted {
            return tab;
        }
        assert!(
            start.elapsed() < WAIT,
            "the page has not mounted after {WAIT:?}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// What the page shows now.
#[derive(Debug, Clone, PartialEq)]
struct Shown {
    state: String,
    run: String,
    interrupted: bool,
    points: u64,
    last_value: Option<String>,
}

fn shown(tab: &headless_chrome::Tab) -> Shown {
    let read = tab
        .evaluate(
            "JSON.stringify({
                state: document.querySelector('main')?.dataset.runState ?? 'none',
                run: document.querySelector('.run')?.textContent ?? '',
                interrupted: document.querySelector('.interrupted') !== null,
                points: Number(document.querySelector('svg[data-metric=\"loss\"]')?.dataset.points ?? 0),
                lastValue: document.querySelector('svg[data-metric=\"loss\"]')?.dataset.lastValue ?? null,
            })",
            false,
        )
        .expect("the page answers")
        .value
        .and_then(|v| v.as_str().map(str::to_owned))
        .expect("a JSON string");
    let value: Value = serde_json::from_str(&read).expect("JSON");
    Shown {
        state: value["state"].as_str().unwrap_or_default().to_owned(),
        run: value["run"].as_str().unwrap_or_default().to_owned(),
        interrupted: value["interrupted"].as_bool().unwrap_or_default(),
        points: value["points"].as_u64().unwrap_or_default(),
        last_value: value["lastValue"].as_str().map(str::to_owned),
    }
}

/// Polls the page every 50 ms until `done` holds, for at most `limit`; the last reading either
/// way, and every distinct reading seen.
fn watch(
    tab: &headless_chrome::Tab,
    limit: Duration,
    done: impl Fn(&Shown) -> bool,
) -> (Shown, Vec<Shown>) {
    let start = Instant::now();
    let mut seen: Vec<Shown> = Vec::new();
    loop {
        let now = shown(tab);
        if seen.last() != Some(&now) {
            seen.push(now.clone());
        }
        if done(&now) || start.elapsed() > limit {
            return (now, seen);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// A run stopped before it ends (Ctrl-C on `modelspec train --serve`) and a server started again at
/// the same address, the next run of a fixed `--serve 127.0.0.1:<port>`, with the page left open.
/// The page's `EventSource` reconnects to the new server, which serves another run. The page must
/// show the run it is now connected to. It keeps the stopped run instead: `apply` skips every
/// event of another run (`ui/src/lib/series.ts`), and the new server, given the old
/// `Last-Event-ID`, sends at most the tail of its run or an empty response, so the page shows the
/// dead run as `streaming` for as long as it stays open.
#[test]
#[ignore = "needs ui/dist and a Chrome or Chromium; run with --ignored"]
fn observatory_ui_follows_the_run_a_restarted_server_serves() {
    let ui = built_ui();
    let address = format!("127.0.0.1:{}", free_port());
    let mut first = Served::start(&[
        "train",
        LINEAR,
        "--backend",
        "cpu",
        "--steps",
        "2000000",
        "--serve",
        &address,
        "--ui",
        &ui,
    ]);
    let browser = browser();
    let tab = open_page(&browser, &address);
    let (before, _) = watch(&tab, WAIT, |s| s.state == "streaming" && s.points > 0);
    assert_eq!(
        before.state, "streaming",
        "the first run streams: {before:?}"
    );
    first.kill();

    let fixture = std::fs::read_to_string(repo_root().join(LINEAR_FIXTURE)).unwrap();
    let first_line: Value = serde_json::from_str(fixture.lines().next().unwrap()).unwrap();
    let replayed = first_line["training_run_id"].as_str().unwrap().to_owned();
    let steps = fixture.matches("\"kind\":\"StepCompleted\"").count() as u64;
    let _second = Served::start(&["observe", LINEAR_FIXTURE, "--serve", &address, "--ui", &ui]);

    let (after, seen) = watch(&tab, Duration::from_secs(30), |s| {
        s.run.contains(&replayed) && s.points == steps
    });
    assert!(
        after.run.contains(&replayed) && after.points == steps,
        "30 s after the server at {address} was restarted with run {replayed} ({steps} steps), the \
         page still shows {after:?}; it showed {before:?} before the restart; readings since: \
         {seen:?}"
    );
}

/// The page draws a run while it streams: opened during a run of 100000 steps it shows the run
/// `streaming` with some but not all of its points, and at the end one point per step and the last
/// loss the stream sent.
#[test]
#[ignore = "needs ui/dist and a Chrome or Chromium; run with --ignored"]
fn observatory_ui_draws_a_run_while_it_streams() {
    const STEPS: u64 = 100_000;
    let ui = built_ui();
    let address = format!("127.0.0.1:{}", free_port());
    let _run = Served::start(&[
        "train",
        LINEAR,
        "--backend",
        "cpu",
        "--steps",
        &STEPS.to_string(),
        "--serve",
        &address,
        "--ui",
        &ui,
    ]);
    let browser = browser();
    let tab = open_page(&browser, &address);
    let (end, seen) = watch(&tab, WAIT, |s| s.state == "finished" || s.state == "failed");
    assert_eq!(end.state, "finished", "{end:?}");
    assert!(
        seen.iter()
            .any(|s| s.state == "streaming" && s.points > 0 && s.points < STEPS + 1),
        "the page drew the run while it streamed: {seen:?}"
    );
    assert_eq!(
        end.points,
        STEPS + 1,
        "a point for each of steps 0 to {STEPS}"
    );

    // The last loss the stream sent, as the server wrote it.
    let mut stream = std::net::TcpStream::connect(&address).unwrap();
    write!(stream, "GET /events HTTP/1.1\r\nHost: {address}\r\n\r\n").unwrap();
    let mut body = String::new();
    stream.read_to_string(&mut body).unwrap();
    let last = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .rfind(|data| data.contains("\"kind\":\"StepCompleted\""))
        .expect("a StepCompleted");
    let start = last.find("\"loss\":").unwrap() + "\"loss\":".len();
    let raw = &last[start..start + last[start..].find([',', '}']).unwrap()];
    let shown = end.last_value.expect("a last value");
    assert_eq!(
        shown.parse::<f64>().map(f64::to_bits).ok(),
        raw.parse::<f64>().map(f64::to_bits).ok(),
        "the chart's last loss {shown} is the stream's {raw}"
    );
}
