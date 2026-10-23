//! `modelspec train <file> --backend cpu|vulkan`: compile a `.mspec` file for training and run
//! its SGD steps on the CPU interpreter or on one Vulkan device with M0 synthetic data
//! (DESIGN.md § 6, § 8, § 9 gates 4 and 5).
//!
//! The run takes its steps, learning rate and seed from the file's `train` block; the backend
//! comes from `--backend`, not from the block's `device`. Both backends print the same lines,
//! each value the backend's own. Stdout is, one line each:
//!
//! - `init @<path>: <type> = [v, ...]` for every parameter before the first step;
//! - `step <k> loss <v>` at step 0, every 50 steps and the last step, where step `k` is the loss
//!   after `k` updates;
//! - `param @<path>: <type> = [v, ...]` for every parameter after the last step.
//!
//! Parameters are listed by canonical path, values in row-major order, each `f32` in its shortest
//! round-trip decimal form. A file that does not compile, or data the file's declarations
//! cannot have, prints its diagnostics to stderr and exits 1; a file that cannot be read, a run
//! that fails, or a run that cannot be written to stdout (a closed pipe, a full device) is
//! reported on one line and exits 1. A run whose loss or an updated parameter stops being
//! finite stops at that step: stdout has its `init` lines and its `step` lines through the
//! failing step's, no `param` lines, stderr has one [`runner::E_TRAIN_001`] diagnostic, and it
//! exits 1. A Vulkan run checks every loss and the parameters where it reads them back, after
//! the last step and at each checkpoint, so a parameter that goes non-finite while the loss of
//! its step is finite is named at the first later step whose loss or read-back shows it. Its
//! `--stats` and `--profile` lines follow the diagnostic and count the steps the device ran,
//! through the read-back that showed the value.
//!
//! `--backend vulkan` compiles the file's execution plan for Vulkan and trains on device 0, or
//! on the device `--device <index>` names, in the loader's enumeration order
//! ([`modelspec_runtime::vulkan::runner`]): the host waits for the device once after the last
//! step, and once after each step `--checkpoint-every` writes a checkpoint at, whose parameters it
//! reads back then and no other step. `--stats` prints the run's counters on stderr after it (host waits, submits,
//! dispatches, bytes up and down, the numerics class, microseconds per step). `--profile` times
//! every dispatch with GPU timestamp queries, read back after the one host wait, and prints on
//! stderr after it one `profile dispatch <plan> <position> <kernel_module_id> <entry> calls <n>
//! total_us <t> mean_us <m> flop <f> bytes <b> gflop_per_s <g> gb_per_s <h>` line per dispatch
//! site (`uncounted` in place of the last four for a step without a count), then the counters,
//! the compile and pipeline creation times, the steady-state microseconds per step and a
//! `profile step_total` line, the step's counts over that time
//! ([`modelspec_runtime::vulkan::profile`]); stdout is unchanged. The file is
//! compiled before the loader is opened; then a missing loader exits 1 with `E-TARGET-002`, no
//! device or a `--device` that names none with `E-TARGET-003`, and a device below the
//! requirements with `E-TARGET-004`. A build without the `vulkan` feature refuses the backend
//! with [`E_TARGET_001`] before the file is read. `--data`, `--init`, `--eval` and
//! `--statistics-every` are CPU-only, and `--device`, `--stats` and `--profile` Vulkan-only:
//! naming one with the other backend is a usage error (exit 2).
//!
//! `--data <dir>` trains on dense `f32` files instead of synthetic data, `--init <dir>` starts
//! the parameters that have a file from it, and `--eval <dir>` adds a last line
//! `eval loss=<mean loss> accuracy=<fraction>` (DESIGN.md § 8; `spec/domains/data.yaml`). Every
//! file is read before the first step; one that cannot be used is refused with `E-DATA-007`,
//! `E-DATA-008` or `E-DATA-009` and no `step` line is printed.
//!
//! `--steps <n>` runs `n` steps instead of the block's. `--checkpoint <path>` writes the state
//! after the last step as a `.msckpt` file ([`checkpoint`]; DESIGN.md § 6.2), and
//! `--checkpoint-every <n>` also writes `<stem>-<k>.msckpt` beside it at every step `k` after the
//! start that is a multiple of `n` and before the end. `--resume <path>` starts from a checkpoint's
//! parameters and step `s`: the run ends at `s + steps`, its `init` lines are the values at `s`,
//! and its `step` lines start at `s`. A checkpoint that cannot be read or does not match the file
//! and its optimizer is refused with its `E-CKPT` code before anything is served or run; a
//! checkpoint that cannot be written fails the run on one line.
//!
//! `--serve <addr>` also streams the run as `text/event-stream` at `http://<addr>/events`
//! (`spec/domains/observatory.yaml`, `ServeAddress` and `StreamFrame`; [`modelspec_observatory`]).
//! Once the file compiles the address is bound and `serving http://<bound address>/events` is
//! printed on stderr; then the stream is `RunStarted`, one `StepCompleted` per loss, for each of the
//! `steps + 1` the run records, and `RunFinished` once stdout is written, or `RunFailed` with the
//! failure stderr reports. A CPU run's `StepCompleted` carries `parameters` on its first step,
//! every step that is a multiple of `--statistics-every <n>` (100 without the flag) and its last:
//! per parameter in path order the norm and histogram of its values, the ones the step reads, then
//! of its gradient unless the step is the last or the loss does not read it
//! ([`modelspec_observatory::statistics`]); every other step's list is empty. A Vulkan run
//! carries only the value entries of its first step, from the values it uploads, and of its last,
//! from the values it reads back after the last step, and no gradient, which it never reads back,
//! so the statistics add no read-back and no host wait. Stdout is byte for byte what the run
//! prints without the flag. The process then keeps serving, so a client that connects late reads
//! the whole stream, until
//! SIGINT, SIGTERM or SIGHUP, on which it exits with the run's own status: 0 for a run that
//! finished, 1 for one that failed (wave observatory-1 coordinator, 2026-10-21). The run has ended
//! once it has printed its last stdout line, or for a run that fails its failure on stderr, so a
//! signal sent after that line or after `RunFinished` or `RunFailed` gets the run's status
//! (coordinator, 2026-10-22). A signal before the run ends stops the process as it stops a run
//! without the flag; a file that does not compile is reported and exits 1 without serving.
//! `--ui <dir>` beside `--serve` also serves the built observatory UI at `http://<addr>/`
//! (`UiDirectory`; [`modelspec_observatory::Ui`]): the directory is read whole before the address
//! is bound, one that cannot be served is reported on one line and exits 1, and
//! `ui http://<bound address>/` follows the `serving` line on stderr.
//!
//! `--record <path>`, with or without `--serve`, writes the same events to the file `<path>`,
//! one JSON `RunEvent` per line in `sequence` order, each line the `data:` of its frame and
//! written before a client of `--serve` reads it (`spec/domains/observatory.yaml`,
//! `RecordedRunFile`; [`modelspec_observatory::Recorder`]); `modelspec observe <path>` replays it
//! ([`crate::observe`]). The file is created once the file compiles and the `--serve` address
//! is bound, before anything is served or run, so a refused address leaves a file already at
//! `<path>` as it was; one that cannot be created is reported on one line and exits 1. A run
//! whose events cannot all be written reports that on one line on stderr and exits 1, and no
//! event is written to the file after the first failed write; a run that finished then ends its
//! stream with `RunFailed` instead of `RunFinished`.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use modelspec_core::compile::{self, CompiledTraining};
use modelspec_core::diag::{Code, Diagnostic};
use modelspec_core::ir::tensor::ParamId;
use modelspec_observatory::{
    EVENTS_PATH, EventSequence, ParameterStatistics, Recorder, RunEventBody, RunFailed, RunStarted,
    Server, StatisticsSubject, StepCompleted, TrainingRunId, Ui,
};
use modelspec_runtime::cpu::HostTensor;
use modelspec_runtime::train::checkpoint::{self, Checkpoint, CheckpointError};
use modelspec_runtime::train::data::{self, DataError, DenseData};
use modelspec_runtime::train::init;
use modelspec_runtime::train::optimizer::ParamState;
use modelspec_runtime::train::runner::{self, Diverged, Eval, Observed, Run, RunError};

/// The selected backend is not available: `--backend vulkan` of `train` or `infer` in a build
/// without the `vulkan` feature.
pub const E_TARGET_001: Code = Code::new("E-TARGET-001");

/// A loss line is printed every this many steps, besides the last.
const LOSS_EVERY: u64 = 50;

/// Statistics are taken every this many steps without `--statistics-every`, besides the first
/// and the last (approval-record:parameter-statistics-settlements, question 3).
const STATISTICS_EVERY: u64 = 100;

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
    /// Record the run's events to the file PATH, one JSON RunEvent per line in sequence order,
    /// the events --serve streams; `modelspec observe PATH` replays it.
    #[arg(long, value_name = "PATH")]
    pub record: Option<PathBuf>,
    /// With --serve, also serve the built observatory UI in DIR (ui/dist after `task ui`) at
    /// http://ADDR/; the directory is read whole before ADDR is bound.
    #[arg(long, value_name = "DIR", requires = "serve")]
    pub ui: Option<PathBuf>,
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
    /// Run N steps instead of the steps of the file's train block; a resumed run ends at its
    /// checkpoint's step plus N.
    #[arg(long, value_name = "N")]
    pub steps: Option<u64>,
    /// After the last step, write the parameters, step and optimizer to the .msckpt file PATH.
    #[arg(long, value_name = "PATH")]
    pub checkpoint: Option<PathBuf>,
    /// Also write STEM-K.msckpt beside the --checkpoint file at every step K after the start that
    /// is a multiple of N and before the end.
    #[arg(long, value_name = "N", requires = "checkpoint", value_parser = clap::value_parser!(u64).range(1..))]
    pub checkpoint_every: Option<u64>,
    /// Continue from the .msckpt file PATH: its parameters and step instead of the initializers
    /// and step 0, refused unless it matches the compiled file and its optimizer.
    #[arg(long, value_name = "PATH", conflicts_with = "init")]
    pub resume: Option<PathBuf>,
    /// Take the norm and histogram of every parameter and its gradient for the event stream at
    /// the first step, every step that is a multiple of N and the last, instead of every 100;
    /// CPU only: a Vulkan run sends the Value entries of its first and last step and every other
    /// step's list empty.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u64).range(1..))]
    pub statistics_every: Option<u64>,
    /// With --backend vulkan, train on the physical device INDEX, in the order the Vulkan loader
    /// lists them, instead of device 0.
    #[arg(long, value_name = "INDEX")]
    pub device: Option<usize>,
    /// With --backend vulkan, print the run's host waits, submits, dispatches, bytes copied up
    /// and down, numerics class and microseconds per step on stderr after it.
    #[arg(long)]
    pub stats: bool,
    /// With --backend vulkan, time every dispatch with GPU timestamp queries and print one
    /// `profile dispatch` line per dispatch site, with its FLOP, bytes, achieved GFLOP/s and
    /// GB/s, then the run's counters, compile and pipeline creation time, steady-state
    /// microseconds per step and the step's total FLOP, bytes, GFLOP/s and GB/s on stderr after
    /// it.
    #[arg(long)]
    pub profile: bool,
}

/// The flag of `args` the backend cannot take, if one is given.
fn misplaced_flag(args: &Args) -> Option<&'static str> {
    match args.backend {
        Backend::Cpu => [
            ("--device", args.device.is_some()),
            ("--stats", args.stats),
            ("--profile", args.profile),
        ]
        .into_iter()
        .find_map(|(flag, given)| given.then_some(flag)),
        Backend::Vulkan => [
            ("--data", args.data.is_some()),
            ("--init", args.init.is_some()),
            ("--eval", args.eval.is_some()),
            ("--statistics-every", args.statistics_every.is_some()),
        ]
        .into_iter()
        .find_map(|(flag, given)| given.then_some(flag)),
    }
}

/// What runs the steps: the CPU interpreter, or a Vulkan device with the file's execution plan.
enum Engine {
    /// `statistics_every` is the N of `--statistics-every`.
    Cpu { statistics_every: u64 },
    #[cfg(feature = "vulkan")]
    Vulkan {
        artifact: Box<modelspec_core::artifact::CompiledArtifact>,
        device: Option<usize>,
        stats: bool,
        /// With `--profile`, the time the file and its artifact took to compile.
        profile: Option<std::time::Duration>,
    },
}

/// The events of a run: numbered once, then written to the `--record` file and sent to the
/// `--serve` clients, in that order.
struct Events {
    sequence: EventSequence,
    recorder: Option<Recorder>,
    /// The first error writing the `--record` file; no event is written after it.
    record_error: Option<std::io::Error>,
}

/// Why a run produced no stdout: a diagnostic printed as it is, or a failure on one line.
enum Failure {
    Refused(Diagnostic),
    Failed(String),
    /// E-TRAIN-001: the run up to its failing step goes to stdout, then the diagnostic to
    /// stderr, followed by what a device run prints after it: its `--stats` and `--profile`
    /// lines, empty on the CPU.
    Diverged(Box<Diverged>, String),
}

impl From<RunError> for Failure {
    fn from(error: RunError) -> Self {
        match error {
            RunError::Data(DataError::Refused(diagnostic))
            | RunError::Checkpoint(CheckpointError::Refused(diagnostic)) => {
                Failure::Refused(diagnostic)
            }
            RunError::NonFinite(diverged) => Failure::Diverged(diverged, String::new()),
            error => Failure::Failed(error.to_string()),
        }
    }
}

/// The directories of `--data`, `--init` and `--eval`, and the checkpoints of `--resume`,
/// `--checkpoint` and `--checkpoint-every`.
struct Files {
    data: Option<PathBuf>,
    init: Option<PathBuf>,
    eval: Option<PathBuf>,
    checkpoint: Option<PathBuf>,
    checkpoint_every: Option<u64>,
    /// The step and parameters of `--resume`, read and matched before the server binds.
    resume: Option<(u64, BTreeMap<ParamId, HostTensor<f32>>)>,
}

/// The files of a run, read before its first step.
struct Loaded {
    data: Option<DenseData>,
    init: BTreeMap<ParamId, HostTensor<f32>>,
    eval: Option<DenseData>,
}

impl Files {
    /// The step a run starts at.
    fn start(&self) -> u64 {
        self.resume.as_ref().map_or(0, |(step, _)| *step)
    }

    /// The file `<stem>-<k>.msckpt` beside the `--checkpoint` file.
    fn every_path(&self, k: u64) -> Option<PathBuf> {
        let path = self.checkpoint.as_ref()?;
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        Some(path.with_file_name(format!("{stem}-{k}.msckpt")))
    }

    fn load(&self, compiled: &CompiledTraining) -> Result<Loaded, DataError> {
        let forward = &compiled.forward;
        let mut config = compiled.config.clone();
        config.steps = config.steps.saturating_add(self.start());
        let config = &config;
        let data = match &self.data {
            Some(dir) => Some(data::dense(dir, config, forward)?),
            None => None,
        };
        let init = match (&self.init, &self.resume) {
            (_, Some((_, values))) => values.clone(),
            (Some(dir), None) => init::read_params(dir, &forward.program)?,
            (None, None) => BTreeMap::new(),
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
    /// Vulkan compute on one device, in a build with the vulkan feature.
    Vulkan,
}

/// Runs `modelspec train` and returns the process exit code.
pub fn run(args: Args, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    if let Some(flag) = misplaced_flag(&args) {
        let (backend, other) = match args.backend {
            Backend::Cpu => ("cpu", "vulkan"),
            Backend::Vulkan => ("vulkan", "cpu"),
        };
        let _ = writeln!(
            stderr,
            "error: the argument '{flag}' cannot be used with '--backend {backend}'; it is \
             available only with '--backend {other}'"
        );
        return crate::EXIT_USAGE;
    }
    #[cfg(not(feature = "vulkan"))]
    if args.backend == Backend::Vulkan {
        let _ = writeln!(
            stderr,
            "{}",
            Diagnostic::new(
                E_TARGET_001,
                "the vulkan backend is not available in this build"
            )
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
    #[cfg(feature = "vulkan")]
    let compiling = std::time::Instant::now();
    let mut compiled = match compile::compile_training(&file, &source) {
        Ok(compiled) => compiled,
        Err(diagnostics) => {
            for diagnostic in &diagnostics {
                let _ = writeln!(stderr, "{diagnostic}");
            }
            return crate::EXIT_FAILURE;
        }
    };
    if let Some(steps) = args.steps {
        compiled.config.steps = steps;
    }
    let engine = match args.backend {
        Backend::Cpu => Engine::Cpu {
            statistics_every: args.statistics_every.unwrap_or(STATISTICS_EVERY),
        },
        #[cfg(not(feature = "vulkan"))]
        Backend::Vulkan => unreachable!("refused above in a build without the vulkan feature"),
        #[cfg(feature = "vulkan")]
        Backend::Vulkan => {
            use modelspec_core::artifact::Backend as Target;
            match compile::compile_artifact(&file, &source, Target::Vulkan) {
                Ok(artifact) => Engine::Vulkan {
                    artifact: Box::new(artifact),
                    device: args.device,
                    stats: args.stats,
                    profile: args.profile.then(|| compiling.elapsed()),
                },
                Err(diagnostics) => {
                    for diagnostic in &diagnostics {
                        let _ = writeln!(stderr, "{diagnostic}");
                    }
                    return crate::EXIT_FAILURE;
                }
            }
        }
    };
    let resume = match &args.resume {
        None => None,
        Some(path) => {
            match checkpoint::read(path).and_then(|c| Ok((c.step, c.resume_into(&compiled)?))) {
                Ok(resume) => Some(resume),
                Err(error) => {
                    let _ = writeln!(stderr, "{error}");
                    return crate::EXIT_FAILURE;
                }
            }
        }
    };
    let ui = match &args.ui {
        None => None,
        Some(dir) => match Ui::read(dir) {
            Ok(ui) => Some(ui),
            Err(error) => {
                let _ = writeln!(
                    stderr,
                    "modelspec train {file}: cannot serve the UI from {}: {error}",
                    dir.display()
                );
                return crate::EXIT_FAILURE;
            }
        },
    };
    let training_run_id = TrainingRunId::new();
    // The address is bound before the record file is created, so a refused `--serve` leaves a
    // file already at the `--record` path as it was.
    let server = match args.serve {
        None => None,
        Some(address) => match Server::bind_with_ui(address, training_run_id, ui) {
            Ok(server) => Some(server),
            Err(error) => {
                let _ = writeln!(
                    stderr,
                    "modelspec train {file}: cannot serve on {address}: {error}"
                );
                return crate::EXIT_FAILURE;
            }
        },
    };
    let recorder = match &args.record {
        None => None,
        Some(path) => match Recorder::create(path) {
            Ok(recorder) => Some(recorder),
            Err(error) => {
                let _ = writeln!(
                    stderr,
                    "modelspec train {file}: cannot record to {}: {error}",
                    path.display()
                );
                return crate::EXIT_FAILURE;
            }
        },
    };
    if let Some(server) = &server {
        let _ = writeln!(
            stderr,
            "serving http://{}{EVENTS_PATH}",
            server.local_addr()
        );
        if args.ui.is_some() {
            let _ = writeln!(stderr, "ui http://{}/", server.local_addr());
        }
    }
    let _ = stderr.flush();
    let events = RefCell::new(Events {
        sequence: EventSequence::new(training_run_id),
        recorder,
        record_error: None,
    });
    let emit = |body: RunEventBody| {
        let events = &mut *events.borrow_mut();
        let event = events.sequence.event(body);
        if let Some(recorder) = &mut events.recorder
            && let Err(error) = recorder.write(&event)
        {
            // Sticky: the recorder is dropped at the first failed write, so no later event is
            // written and the failure is reported once.
            events.recorder = None;
            events.record_error = Some(error);
        }
        if let Some(server) = &server {
            server.publish(&event);
        }
    };
    let record_failure = |stderr: &mut dyn Write| {
        let error = events.borrow_mut().record_error.take()?;
        let path = args.record.as_deref().unwrap_or_else(|| Path::new(""));
        let message = format!("cannot record to {}: {error}", path.display());
        let _ = writeln!(stderr, "modelspec train {file}: {message}");
        Some(message)
    };
    let files = Files {
        data: args.data,
        init: args.init,
        eval: args.eval,
        checkpoint: args.checkpoint,
        checkpoint_every: args.checkpoint_every,
        resume,
    };
    emit(RunEventBody::RunStarted(RunStarted {
        steps: compiled.config.steps,
        start_step: files.start(),
    }));
    let mut handler = None;
    let outcome = train(
        &file,
        &compiled,
        &engine,
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
    let outcome = match (outcome, record_failure(stderr)) {
        (Ok(()), Some(message)) => Err(RunFailed {
            code: None,
            message,
        }),
        (outcome, _) => outcome,
    };
    let (mut code, last) = match outcome {
        Ok(()) => (crate::EXIT_SUCCESS, RunEventBody::RunFinished),
        Err(failed) => (crate::EXIT_FAILURE, RunEventBody::RunFailed(failed)),
    };
    let interrupted = handler.flatten();
    emit(last);
    if record_failure(stderr).is_some() {
        code = crate::EXIT_FAILURE;
    }
    let _ = stderr.flush();
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

/// Trains `compiled` on `engine`, emits a `StepCompleted` for each loss as it is recorded (on the
/// CPU with the parameter statistics of the steps `statistics_every` samples, on a device with
/// those of its first and last step, [`device_parameters`]) and
/// writes the run to stdout. A failure is reported on stderr and returned as the `RunFailed`
/// that ends the stream. `end` is called as the run ends: before the last stdout line, and
/// before a failure line. A device run records its losses on the device and emits them once the
/// host has read them back.
#[allow(clippy::too_many_arguments)]
fn train(
    file: &str,
    compiled: &CompiledTraining,
    engine: &Engine,
    files: &Files,
    emit: &dyn Fn(RunEventBody),
    end: &mut dyn FnMut(),
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> Result<(), RunFailed> {
    let step_completed = |step, loss: f32, parameters| {
        emit(RunEventBody::StepCompleted(StepCompleted {
            step,
            loss: f64::from(loss),
            metrics: Vec::new(),
            parameters,
        }));
    };
    let params = paths(compiled);
    let observed: Result<(Run, Option<Eval>), Failure> = 'run: {
        #[cfg(feature = "vulkan")]
        if let Engine::Vulkan {
            artifact,
            device,
            stats,
            profile,
        } = engine
        {
            use modelspec_runtime::vulkan::profile::Profile;
            use modelspec_runtime::vulkan::runner::{
                Checkpoints, DeviceRunError, DivergedOnDevice, RunOptions, RunStats,
                VulkanRunError, train_on_device_from,
            };
            // The `--stats` lines, then the `--profile` lines, of what the device ran.
            let counters = |counted: &RunStats, profiled: Option<Profile>| {
                let mut out = String::new();
                if *stats {
                    out.push_str(&counted.to_string());
                }
                if let (Some(mut profiled), Some(compiling)) = (profiled, *profile) {
                    profiled.compile += compiling;
                    out.push_str(&profiled.report(counted));
                }
                out
            };
            let mut write = |k: u64, values: &BTreeMap<ParamId, HostTensor<f32>>| match files
                .every_path(k)
            {
                Some(path) => checkpoint::write(&path, &Checkpoint::of_state(compiled, k, values)),
                None => Ok(()),
            };
            let options = RunOptions {
                start: files.start(),
                init: files
                    .resume
                    .as_ref()
                    .map(|(_, values)| values.clone())
                    .unwrap_or_default(),
                checkpoints: files.checkpoint_every.map(|every| Checkpoints {
                    every,
                    write: &mut write,
                }),
                data: None,
            };
            // The post-loop step; the runner refuses a run whose end is past `u64::MAX`.
            let last = files.start().saturating_add(compiled.config.steps);
            break 'run match train_on_device_from(
                *device,
                compiled,
                artifact,
                profile.is_some(),
                options,
            ) {
                Ok((run, counted, profiled)) => {
                    for (k, &loss) in (run.start_step..).zip(&run.losses) {
                        step_completed(k, loss, device_parameters(&params, &run, last, k));
                    }
                    let _ = write!(stderr, "{}", counters(&counted, profiled));
                    let end = run.start_step + counted.steps;
                    match &files.checkpoint {
                        Some(path) => checkpoint::write(
                            path,
                            &Checkpoint::of_state(compiled, end, &run.state.values),
                        )
                        .map(|()| (run, None))
                        .map_err(|error| RunError::from(error).into()),
                        None => Ok((run, None)),
                    }
                }
                Err(DeviceRunError::Refused(diagnostic)) => Err(Failure::Refused(diagnostic)),
                Err(DeviceRunError::Failed(VulkanRunError::NonFinite(stopped))) => {
                    // A diverged device run streams the losses it read back, through the
                    // failing step's, as the CPU run streams them while it runs, and prints
                    // its counters after the diagnostic.
                    let DivergedOnDevice {
                        diverged,
                        stats: counted,
                        profile: profiled,
                    } = *stopped;
                    for (k, &loss) in (diverged.run.start_step..).zip(&diverged.run.losses) {
                        let parameters = device_parameters(&params, &diverged.run, last, k);
                        step_completed(k, loss, parameters);
                    }
                    Err(Failure::Diverged(
                        Box::new(diverged),
                        counters(&counted, profiled),
                    ))
                }
                Err(DeviceRunError::Failed(VulkanRunError::Run(error))) => Err((*error).into()),
                Err(DeviceRunError::Failed(error)) => Err(Failure::Failed(error.to_string())),
            };
        }
        #[cfg(not(feature = "vulkan"))]
        let &Engine::Cpu { statistics_every } = engine;
        #[cfg(feature = "vulkan")]
        let &Engine::Cpu { statistics_every } = engine else {
            unreachable!("a Vulkan run ends above")
        };
        let loaded = match files.load(compiled) {
            Ok(loaded) => loaded,
            Err(error) => break 'run Err(RunError::from(error).into()),
        };
        let start = files.start();
        let end = start.saturating_add(compiled.config.steps);
        let after = |k: u64, state: &ParamState| match (files.checkpoint_every, files.every_path(k))
        {
            (Some(n), Some(path)) if k.is_multiple_of(n) && k < end => {
                checkpoint::write(&path, &Checkpoint::of_state(compiled, k, &state.values))
            }
            _ => Ok(()),
        };
        let observe = |o: &Observed<'_>| {
            let sampled =
                o.step == start || o.step == end || o.step.is_multiple_of(statistics_every);
            let parameters = if sampled {
                statistics(&params, o.values, o.gradients)
            } else {
                Vec::new()
            };
            step_completed(o.step, o.loss, parameters);
        };
        let run = match runner::run_from(
            compiled,
            loaded.data.as_ref(),
            &loaded.init,
            start,
            observe,
            after,
        ) {
            Ok(run) => run,
            Err(error) => break 'run Err(error.into()),
        };
        let eval = match &loaded.eval {
            None => None,
            Some(data) => match runner::evaluate_dense(compiled, data, &run.state.values) {
                Ok(eval) => Some(eval),
                Err(error) => break 'run Err(error.into()),
            },
        };
        if let Some(path) = &files.checkpoint {
            let last = Checkpoint::of_state(compiled, end, &run.state.values);
            if let Err(error) = checkpoint::write(path, &last) {
                break 'run Err(RunError::from(error).into());
            }
        }
        Ok((run, eval))
    };
    let (run, eval) = match observed {
        Ok(done) => done,
        Err(Failure::Refused(diagnostic)) => {
            end();
            let _ = writeln!(stderr, "{diagnostic}");
            return Err(RunFailed {
                code: Some(diagnostic.code.to_string()),
                message: diagnostic.message.clone(),
            });
        }
        Err(Failure::Diverged(diverged, counters)) => {
            let Diverged { diagnostic, run } = *diverged;
            let _ = stdout
                .write_all(report(compiled, &run, None, false).as_bytes())
                .and_then(|()| stdout.flush());
            end();
            let _ = writeln!(stderr, "{diagnostic}");
            let _ = write!(stderr, "{counters}");
            return Err(RunFailed {
                code: Some(diagnostic.code.to_string()),
                message: diagnostic.message.clone(),
            });
        }
        Err(Failure::Failed(message)) => {
            end();
            let _ = writeln!(stderr, "modelspec train {file}: {message}");
            return Err(RunFailed {
                code: None,
                message,
            });
        }
    };
    let printed = report(compiled, &run, eval, true);
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

/// Every declared parameter of `compiled` by canonical path, so in path order.
fn paths(compiled: &CompiledTraining) -> BTreeMap<&str, ParamId> {
    compiled
        .forward
        .program
        .params
        .iter()
        .map(|(&id, decl)| (decl.path.as_str(), id))
        .collect()
}

/// The `parameters` of a sampled step: for each parameter of `params` in path order, the
/// `Value` entry of `values`, the values the step reads, then the `Gradient` entry of
/// `gradients` when it holds the parameter's.
fn statistics(
    params: &BTreeMap<&str, ParamId>,
    values: &BTreeMap<ParamId, HostTensor<f32>>,
    gradients: Option<&BTreeMap<ParamId, HostTensor<f32>>>,
) -> Vec<ParameterStatistics> {
    let mut entries = Vec::new();
    for (&path, id) in params {
        entries.push(ParameterStatistics::of(
            path,
            StatisticsSubject::Value,
            values[id].data(),
        ));
        if let Some(gradient) = gradients.and_then(|gradients| gradients.get(id)) {
            entries.push(ParameterStatistics::of(
                path,
                StatisticsSubject::Gradient,
                gradient.data(),
            ));
        }
    }
    entries
}

/// The `parameters` of step `k` of a device run `run` whose post-loop step is `end`, from the
/// values the host holds without reading the device again: at the first step the `Value`
/// entries of [`Run::initial`], the values it uploaded and step `k` reads, at `end` those of
/// the final [`ParamState::values`] it read back after the last step, and at any other step
/// none. The device reads back no gradient, so no step has a `Gradient` entry
/// (approval-record:parameter-statistics-settlements, question 3).
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
fn device_parameters(
    params: &BTreeMap<&str, ParamId>,
    run: &Run,
    end: u64,
    k: u64,
) -> Vec<ParameterStatistics> {
    if k == run.start_step {
        statistics(params, &run.initial, None)
    } else if k == end {
        statistics(params, &run.state.values, None)
    } else {
        Vec::new()
    }
}

/// The stdout of a run: of a finished one, or without its `param` lines of one that stopped at
/// its last loss (E-TRAIN-001).
fn report(compiled: &CompiledTraining, run: &Run, eval: Option<Eval>, finished: bool) -> String {
    let params = paths(compiled);
    let mut out = String::new();
    let tensors = |out: &mut String, label: &str, values: &BTreeMap<ParamId, HostTensor<f32>>| {
        for (&path, id) in &params {
            let ty = &compiled.forward.program.params[id].ty;
            let _ = writeln!(out, "{label} @{path}: {ty} = [{}]", list(values[id].data()));
        }
    };
    tensors(&mut out, "init", &run.initial);
    let last = run.start_step + (run.losses.len() as u64 - 1);
    for (k, loss) in (run.start_step..).zip(&run.losses) {
        if k == run.start_step || k.is_multiple_of(LOSS_EVERY) || k == last {
            let _ = writeln!(out, "step {k} loss {loss}");
        }
    }
    if finished {
        tensors(&mut out, "param", &run.state.values);
    }
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

#[cfg(test)]
mod tests;
