//! `modelspec train <file> --backend cpu`: compile a `.mspec` file for training and run its SGD
//! steps on the CPU interpreter with M0 synthetic data (DESIGN.md § 6, § 8, § 9 gate 4).
//!
//! The run takes its steps, learning rate and seed from the file's `train` block; the backend
//! comes from `--backend`, not from the block's `device`. Stdout is, one line each:
//!
//! - `init @<path>: <type> = [v, ...]` for every parameter before the first step;
//! - `step <k> loss <v>` at step 0, every 50 steps and the last step, where step `k` is the loss
//!   after `k` updates;
//! - `param @<path>: <type> = [v, ...]` for every parameter after the last step.
//!
//! Parameters are listed by canonical path, values in row-major order, each `f32` in its shortest
//! round-trip decimal form. `--backend vulkan` exits 1 with [`E_TARGET_001`] before the file is
//! read: M0 has no Vulkan backend. A file that does not compile, or data the file's declarations
//! cannot have, prints its diagnostics to stderr and exits 1; a file that cannot be read, a run
//! that fails, or a run that cannot be written to stdout (a closed pipe, a full device) is
//! reported on one line and exits 1.
//!
//! `--data <dir>` trains on dense `f32` files instead of synthetic data, `--init <dir>` starts
//! the parameters that have a file from it, and `--eval <dir>` adds a last line
//! `eval loss=<mean loss> accuracy=<fraction>` (DESIGN.md § 8; `spec/domains/data.yaml`). Every
//! file is read before the first step; one that cannot be used is refused with `E-DATA-007`,
//! `E-DATA-008` or `E-DATA-009` and no `step` line is printed.
//!
//! `--serve <addr>` also streams the run as `text/event-stream` at `http://<addr>/events`
//! (`spec/domains/observatory.yaml`, `ServeAddress` and `StreamFrame`; [`modelspec_observatory`]).
//! Once the file compiles the address is bound and `serving http://<bound address>/events` is
//! printed on stderr; then the stream is `RunStarted`, one `StepCompleted` per loss, for each of the
//! `steps + 1` the run records, and `RunFinished` once stdout is written, or `RunFailed` with the
//! failure stderr reports. Stdout is byte for byte what the run prints without the flag. The
//! process then keeps serving, so a client that connects late reads the whole stream, until
//! SIGINT, SIGTERM or SIGHUP, on which it exits with the run's own status: 0 for a run that
//! finished, 1 for one that failed (wave observatory-1 coordinator, 2026-10-21). The run has ended
//! once it has printed its last stdout line, or for a run that fails its failure on stderr, so a
//! signal sent after that line or after `RunFinished` or `RunFailed` gets the run's status
//! (coordinator, 2026-10-22). A signal before the run ends stops the process as it stops a run
//! without the flag; a file that does not compile is reported and exits 1 without serving.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::mpsc;

use modelspec_core::compile::{self, CompiledTraining};
use modelspec_core::diag::{Code, Diagnostic};
use modelspec_core::ir::tensor::ParamId;
use modelspec_observatory::{
    EVENTS_PATH, RunEventBody, RunFailed, RunStarted, Server, StepCompleted, TrainingRunId,
};
use modelspec_runtime::cpu::HostTensor;
use modelspec_runtime::train::data::{self, DataError, DenseData};
use modelspec_runtime::train::init;
use modelspec_runtime::train::runner::{self, Eval, Run, RunError};

/// The selected backend is not available in this build.
pub const E_TARGET_001: Code = Code::new("E-TARGET-001");

/// A loss line is printed every this many steps, besides the last.
const LOSS_EVERY: u64 = 50;

/// Arguments of `modelspec train`.
#[derive(clap::Args)]
pub struct Args {
    /// The .mspec file to train.
    pub path: PathBuf,
    /// The backend that runs the training step.
    #[arg(long, value_enum)]
    pub backend: Backend,
    /// Stream the run's events as text/event-stream at path /events of ADDR, a loopback address
    /// such as 127.0.0.1:8080 or localhost:8080 (port 0 picks a free one); the process keeps
    /// serving after the run until SIGINT, SIGTERM or SIGHUP, then exits with the run's status.
    #[arg(long, value_name = "ADDR", value_parser = modelspec_observatory::parse_serve_address)]
    pub serve: Option<SocketAddr>,
    /// Train on the dense f32 files of DIR instead of synthetic data: NAME.f32 for the input and
    /// the target declaration NAME, row-major little-endian, and an optional order.u32 of sample
    /// indices.
    #[arg(long, value_name = "DIR")]
    pub data: Option<PathBuf>,
    /// Start each parameter that has a file DIR/P.f32, P its path without the model name, from
    /// that file instead of its initializer.
    #[arg(long, value_name = "DIR")]
    pub init: Option<PathBuf>,
    /// After the last step, run the forward program over every sample of DIR, laid out as for
    /// --data, and print `eval loss=<mean loss> accuracy=<fraction>`.
    #[arg(long, value_name = "DIR")]
    pub eval: Option<PathBuf>,
}

/// The directories of `--data`, `--init` and `--eval`.
struct Files {
    data: Option<PathBuf>,
    init: Option<PathBuf>,
    eval: Option<PathBuf>,
}

/// The files of a run, read before its first step.
struct Loaded {
    data: Option<DenseData>,
    init: BTreeMap<ParamId, HostTensor<f32>>,
    eval: Option<DenseData>,
}

impl Files {
    fn load(&self, compiled: &CompiledTraining) -> Result<Loaded, DataError> {
        let (config, forward) = (&compiled.config, &compiled.forward);
        let data = match &self.data {
            Some(dir) => Some(data::dense(dir, config, forward)?),
            None => None,
        };
        let init = match &self.init {
            Some(dir) => init::read_params(dir, &forward.program)?,
            None => BTreeMap::new(),
        };
        let eval = match &self.eval {
            Some(dir) => Some(data::dense_eval(dir, config, forward)?),
            None => None,
        };
        Ok(Loaded { data, init, eval })
    }
}

/// A backend `modelspec train` can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Backend {
    /// The CPU interpreter.
    Cpu,
    /// Vulkan compute: not available in M0.
    Vulkan,
}

/// Runs `modelspec train` and returns the process exit code.
pub fn run(args: Args, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    if args.backend == Backend::Vulkan {
        let _ = writeln!(
            stderr,
            "{}",
            Diagnostic::new(E_TARGET_001, "the vulkan backend is not available in M0")
                .with("backend", "vulkan")
                .with("available", "cpu")
        );
        return crate::EXIT_FAILURE;
    }
    let file = args.path.display().to_string();
    let source = match std::fs::read_to_string(&args.path) {
        Ok(source) => source,
        Err(error) => {
            let _ = writeln!(stderr, "modelspec train {file}: {error}");
            return crate::EXIT_FAILURE;
        }
    };
    let compiled = match compile::compile_training(&file, &source) {
        Ok(compiled) => compiled,
        Err(diagnostics) => {
            for diagnostic in &diagnostics {
                let _ = writeln!(stderr, "{diagnostic}");
            }
            return crate::EXIT_FAILURE;
        }
    };
    let server = match args.serve {
        None => None,
        Some(address) => match Server::bind(address, TrainingRunId::new()) {
            Ok(server) => {
                let _ = writeln!(
                    stderr,
                    "serving http://{}{EVENTS_PATH}",
                    server.local_addr()
                );
                Some(server)
            }
            Err(error) => {
                let _ = writeln!(
                    stderr,
                    "modelspec train {file}: cannot serve on {address}: {error}"
                );
                return crate::EXIT_FAILURE;
            }
        },
    };
    let _ = stderr.flush();
    let emit = |body: RunEventBody| {
        if let Some(server) = &server {
            server.emit(body);
        }
    };
    emit(RunEventBody::RunStarted(RunStarted {
        steps: compiled.config.steps,
        start_step: 0,
    }));
    let mut handler = None;
    let files = Files {
        data: args.data,
        init: args.init,
        eval: args.eval,
    };
    let outcome = train(
        &file,
        &compiled,
        &files,
        &emit,
        &mut || {
            if server.is_some() && handler.is_none() {
                handler = Some(interrupted());
            }
        },
        stdout,
        stderr,
    );
    let _ = stderr.flush();
    let (code, last) = match outcome {
        Ok(()) => (crate::EXIT_SUCCESS, RunEventBody::RunFinished),
        Err(failed) => (crate::EXIT_FAILURE, RunEventBody::RunFailed(failed)),
    };
    let interrupted = handler.flatten();
    emit(last);
    match (server, interrupted) {
        (Some(_server), Some(interrupted)) => {
            let _ = interrupted.recv();
            code
        }
        (Some(server), None) => {
            server.wait();
            code
        }
        (None, _) => code,
    }
}

/// A receiver that gets a message on SIGINT, SIGTERM or SIGHUP (the three `ctrlc`'s `termination`
/// feature catches) from now on, or `None` when the handler cannot be installed (one is installed
/// already). Installed only as the run ends, before its last stdout line or its failure line on
/// stderr, so a signal earlier in the run stops the process as it stops a run without `--serve`.
fn interrupted() -> Option<mpsc::Receiver<()>> {
    let (send, receive) = mpsc::channel();
    ctrlc::set_handler(move || {
        let _ = send.send(());
    })
    .ok()?;
    Some(receive)
}

/// Trains `compiled`, emits a `StepCompleted` for each loss as it is recorded and writes the run
/// to stdout. A failure is reported on stderr and returned as the `RunFailed` that ends the stream.
/// `end` is called as the run ends: before the last stdout line, and before a failure line.
fn train(
    file: &str,
    compiled: &CompiledTraining,
    files: &Files,
    emit: &dyn Fn(RunEventBody),
    end: &mut dyn FnMut(),
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> Result<(), RunFailed> {
    let observed: Result<(Run, Option<Eval>), RunError> = 'run: {
        let loaded = match files.load(compiled) {
            Ok(loaded) => loaded,
            Err(error) => break 'run Err(error.into()),
        };
        let observe = |step, loss| {
            emit(RunEventBody::StepCompleted(StepCompleted {
                step,
                loss: f64::from(loss),
                metrics: Vec::new(),
                parameters: Vec::new(),
            }));
        };
        let run = match runner::run_with(compiled, loaded.data.as_ref(), &loaded.init, observe) {
            Ok(run) => run,
            Err(error) => break 'run Err(error),
        };
        match &loaded.eval {
            None => Ok((run, None)),
            Some(data) => match runner::evaluate_dense(compiled, data, &run.state.values) {
                Ok(eval) => Ok((run, Some(eval))),
                Err(error) => Err(error),
            },
        }
    };
    let (run, eval) = match observed {
        Ok(done) => done,
        Err(RunError::Data(DataError::Refused(diagnostic))) => {
            end();
            let _ = writeln!(stderr, "{diagnostic}");
            return Err(RunFailed {
                code: Some(diagnostic.code.to_string()),
                message: diagnostic.message.clone(),
            });
        }
        Err(error) => {
            end();
            let _ = writeln!(stderr, "modelspec train {file}: {error}");
            return Err(RunFailed {
                code: None,
                message: error.to_string(),
            });
        }
    };
    let printed = report(compiled, &run, eval);
    let body = printed.trim_end_matches('\n');
    let (head, last) = printed.split_at(body.rfind('\n').map_or(0, |i| i + 1));
    let written = stdout
        .write_all(head.as_bytes())
        .and_then(|()| stdout.flush())
        .and_then(|()| {
            end();
            stdout.write_all(last.as_bytes())
        })
        .and_then(|()| stdout.flush());
    match written {
        Ok(()) => Ok(()),
        Err(error) => {
            end();
            let message = format!("cannot write the run: {error}");
            let _ = writeln!(stderr, "modelspec train {file}: {message}");
            Err(RunFailed {
                code: None,
                message,
            })
        }
    }
}

/// The stdout of a finished run.
fn report(compiled: &CompiledTraining, run: &Run, eval: Option<Eval>) -> String {
    let params: BTreeMap<&str, ParamId> = compiled
        .forward
        .program
        .params
        .iter()
        .map(|(&id, decl)| (decl.path.as_str(), id))
        .collect();
    let mut out = String::new();
    let tensors = |out: &mut String, label: &str, values: &BTreeMap<ParamId, HostTensor<f32>>| {
        for (&path, id) in &params {
            let ty = &compiled.forward.program.params[id].ty;
            let _ = writeln!(out, "{label} @{path}: {ty} = [{}]", list(values[id].data()));
        }
    };
    tensors(&mut out, "init", &run.initial);
    let last = run.losses.len() - 1;
    for (k, loss) in run.losses.iter().enumerate() {
        if (k as u64).is_multiple_of(LOSS_EVERY) || k == last {
            let _ = writeln!(out, "step {k} loss {loss}");
        }
    }
    tensors(&mut out, "param", &run.state.values);
    if let Some(Eval { loss, accuracy }) = eval {
        let _ = writeln!(out, "eval loss={loss} accuracy={accuracy}");
    }
    out
}

fn list(values: &[f32]) -> String {
    values
        .iter()
        .map(f32::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}
