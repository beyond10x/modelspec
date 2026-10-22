//! Adversary cases, pass 2, against `modelspec train --serve` after the fix (story:
//! observatory-live-event-stream): the exit on SIGINT or SIGTERM after the run, and the `null`
//! loss of a diverging run, driven from `spec/domains/observatory.yaml` (`StreamFrame`,
//! `StepCompleted.loss`; wave observatory-1 coordinator, 2026-10-21).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::Value;

const EXAMPLE: &str = "examples/linear_regression.mspec";
const WAIT: Duration = Duration::from_secs(120);

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source_file(name: &str, source: &str) -> String {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("observatory_signal_adversary");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, source).unwrap();
    path.display().to_string()
}

struct Served {
    child: Child,
    address: String,
    stdout: Option<JoinHandle<Vec<u8>>>,
}

impl Served {
    fn start(path: &str) -> Served {
        let mut child = Command::new(env!("CARGO_BIN_EXE_modelspec"))
            .args(["train", path, "--backend", "cpu", "--serve", "127.0.0.1:0"])
            .current_dir(repo_root())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut out = child.stdout.take().unwrap();
        let stdout = std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = out.read_to_end(&mut bytes);
            bytes
        });
        let (lines, received) = mpsc::channel();
        let err = child.stderr.take().unwrap();
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines() {
                let Ok(line) = line else { break };
                let _ = lines.send(line);
            }
        });
        let mut served = Served {
            child,
            address: String::new(),
            stdout: Some(stdout),
        };
        loop {
            let line = received.recv_timeout(WAIT).expect("a `serving` line");
            if let Some(rest) = line.strip_prefix("serving http://") {
                served.address = rest.strip_suffix("/events").unwrap().to_owned();
                return served;
            }
        }
    }

    /// The exit status within `limit`, or `None` when the process still runs.
    fn exit_within(&mut self, limit: Duration) -> Option<ExitStatus> {
        let start = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return Some(status);
            }
            if start.elapsed() > limit {
                return None;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn signal(&self, signals: &[&str]) {
        let pid = self.child.id().to_string();
        for signal in signals {
            let _ = Command::new("kill")
                .args(["-s", signal, &pid])
                .stdin(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
    }

    fn stdout(&mut self) -> Vec<u8> {
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.stdout
            .take()
            .map(|h| h.join().unwrap())
            .unwrap_or_default()
    }
}

impl Drop for Served {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The body of `GET /events`, read until the server closes it.
fn body(address: &str) -> String {
    let mut stream = TcpStream::connect(address).unwrap();
    stream.set_read_timeout(Some(WAIT)).unwrap();
    write!(stream, "GET /events HTTP/1.1\r\nHost: {address}\r\n\r\n").unwrap();
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .expect("the server ends the response");
    let (head, body) = response.split_once("\r\n\r\n").expect("a head");
    assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
    body.to_owned()
}

fn events(body: &str) -> Vec<Value> {
    body.trim_end_matches('\n')
        .split("\n\n")
        .map(|block| {
            let data = block
                .lines()
                .find_map(|l| l.strip_prefix("data: "))
                .unwrap();
            serde_json::from_str(data).expect("data is JSON")
        })
        .collect()
}

fn last_kind(body: &str) -> String {
    events(body).last().unwrap()["body"]["kind"]
        .as_str()
        .unwrap()
        .to_owned()
}

const DIVERGING: &str = "model LinearReg {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>)
        -> Tensor<f32, [8, 1]> {
        return matmul(x, w);
    }
}

train Fit {
    model = LinearReg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 1000.0);
    steps = 50;
    seed = 42;
    device = cpu;
}
";

const UNREAD_X: &str = "model Const {
    param b: Tensor<f32, [8, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>)
        -> Tensor<f32, [8, 1]> {
        return b;
    }
}

train Fit {
    model = Const;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 500;
    seed = 42;
    device = cpu;
}
";

/// `StepCompleted.loss`: "null when the loss is NaN or ±inf ... The CLI's `step <k> loss <v>`
/// line on stdout for that step names which, `NaN`, `inf` or `-inf`." Each `StepCompleted` whose
/// loss is `null` therefore has that line on stdout.
#[test]
fn every_null_loss_has_the_stdout_line_that_names_it() {
    let path = source_file("diverging.mspec", DIVERGING);
    let mut served = Served::start(&path);
    let events = events(&body(&served.address));
    let stdout = String::from_utf8(served.stdout()).unwrap();
    let null_steps: Vec<u64> = events
        .iter()
        .filter(|e| e["body"]["kind"] == "StepCompleted")
        .filter(|e| e["body"]["value"]["loss"].is_null())
        .map(|e| e["body"]["value"]["step"].as_u64().unwrap())
        .collect();
    assert!(!null_steps.is_empty(), "the run does not diverge: {stdout}");
    let unnamed: Vec<u64> = null_steps
        .iter()
        .copied()
        .filter(|k| {
            !["NaN", "inf", "-inf"]
                .iter()
                .any(|v| stdout.lines().any(|l| l == format!("step {k} loss {v}")))
        })
        .collect();
    assert!(
        unnamed.is_empty(),
        "{} of {} null-loss steps have no `step <k> loss NaN|inf|-inf` line on stdout: {unnamed:?}; \
         the step lines are {:?}",
        unnamed.len(),
        null_steps.len(),
        stdout
            .lines()
            .filter(|l| l.starts_with("step "))
            .collect::<Vec<_>>()
    );
}

/// `StreamFrame`: "on SIGINT or SIGTERM after the run ends it stops serving and exits with the
/// run's own status, 0 for a run that finished and 1 for a run that failed". A second signal
/// right behind the first does not change that status.
#[test]
fn two_signals_after_the_run_still_exit_with_the_runs_status() {
    let refused = source_file("unread_x.mspec", UNREAD_X);
    let mut wrong = Vec::new();
    for (path, last, code) in [
        (EXAMPLE, "RunFinished", 0),
        (refused.as_str(), "RunFailed", 1),
    ] {
        for signals in [["TERM", "TERM"], ["INT", "TERM"], ["TERM", "INT"]] {
            let mut served = Served::start(path);
            assert_eq!(last_kind(&body(&served.address)), last, "{path}");
            served.signal(&signals);
            let status = served.exit_within(Duration::from_secs(10));
            if status.and_then(|s| s.code()) != Some(code) {
                wrong.push(format!("{path} on {signals:?}: {status:?}, not {code}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// A client still connected when the signal arrives (one that sent no request head yet, one
/// half-way through its head) does not hold the process: it exits with the run's status at once,
/// not after the server's 10 s head timeout.
#[test]
fn a_connected_client_does_not_hold_the_exit() {
    let mut served = Served::start(EXAMPLE);
    assert_eq!(last_kind(&body(&served.address)), "RunFinished");
    let _silent = TcpStream::connect(&served.address).unwrap();
    let mut half = TcpStream::connect(&served.address).unwrap();
    write!(half, "GET /events HTTP/1.1\r\nHost: x\r\n").unwrap();
    std::thread::sleep(Duration::from_millis(100));
    served.signal(&["TERM"]);
    let status = served.exit_within(Duration::from_secs(3));
    assert_eq!(status.and_then(|s| s.code()), Some(0), "{status:?}");
}
