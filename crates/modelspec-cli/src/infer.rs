//! `modelspec infer <file> --backend cpu|vulkan --checkpoint <path>`: the forward program of a
//! compiled file with a checkpoint's parameters (DESIGN.md § 6.2, § 8).
//!
//! The checkpoint is read and matched against the compiled file (paths, dtypes, shapes and schema
//! hash; not the optimizer) and refused with its `E-CKPT` code otherwise. The forward program runs
//! on the synthetic batch of the file's `train` block; M1 takes no `--input` (DESIGN.md § 12,
//! "Data interface"). Stdout is two lines:
//!
//! - `prediction: <type> = [v, ...]`, the model output in row-major order;
//! - `loss <v>`, the loss of that batch, which for the checkpoint a run wrote at its last step is
//!   that run's last `step` loss.
//!
//! `--backend cpu` runs it on the CPU interpreter. `--backend vulkan` compiles the file for
//! Vulkan, reads the checkpoint, then opens the loader and runs the steps of the step plan up to
//! the loss once on device 0, or on the device `--device <index>` names, with one host wait
//! ([`modelspec_runtime::vulkan::runner::infer`]); a checkpoint that cannot be used is therefore
//! refused with its `E-CKPT` code before the loader is opened, and then a missing loader exits 1
//! with `E-TARGET-002`, no device or a `--device` that names none with `E-TARGET-003`, and a
//! device below the requirements with `E-TARGET-004`. A build without the `vulkan` feature
//! refuses `--backend vulkan` with [`E_TARGET_001`] before the file is read. `--device` with
//! `--backend cpu` is a usage error (exit 2). A file that does not compile, or data the file's
//! declarations cannot have, prints its diagnostics and exits 1.
//!
//! `--data <dir> --function <name> --out <path>` (story:decider-infer-on-data;
//! `spec/domains/data.yaml`, `InferenceOutputFile`) runs the model function `name` instead,
//! compiled as an entry of its own ([`compile::compile_function`]): it need not be `forward`
//! and need not return a loss. Each argument binds the dense file of `<dir>` with its name,
//! `<name>.f32` or `<name>.i32` by its dtype, as `train --eval` binds a declaration; the function
//! runs with the checkpoint's parameters on every block of `B` samples in block order, and its
//! results, one row per sample, are written to `<path>` once every block has run, row-major
//! little-endian `f32` with no header, through a temporary file in its directory renamed over
//! it, so an existing `<path>` is replaced whole or not at all ([`replace`]). Stdout is one
//! line, `function <name> blocks <blocks> rows <rows> out <path>`. Refused before anything is
//! written, exit 1: a name that is not a function of the model with E-DATA-010 naming its
//! functions; a `<path>` that resolves to a file the run reads (the model file, the checkpoint
//! or an argument's file) with E-DATA-010 naming `--out` and that file, before the checkpoint is
//! read; an argument without its file, or files that cannot be read as samples, with
//! E-DATA-007; a value of an argument's `i32` file outside the `embedding` table or the
//! `cross_entropy` logits of the function that read it with E-DATA-011; a sample count that is
//! not a multiple of `B` with E-DATA-009; a function whose result is not one row per sample
//! with E-DATA-010 naming the model's functions. `--function` or `--out` without `--data`, and
//! `--data` without both, is a usage error (exit 2). A function that calls `sigreg` reads, at
//! every site, the projection of the checkpoint's step on both backends, as the forward program
//! without `--data` does on the CPU. On `--backend vulkan` the function's step runs stepwise
//! with 0 steps from the checkpoint's step ([`modelspec_runtime::vulkan::stepwise`]): one
//! command buffer and host wait per block, each block's result read back before the next, then
//! one for that step.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};

use modelspec_core::compile::{self, CompiledTraining, FunctionError};
#[cfg(not(feature = "vulkan"))]
use modelspec_core::diag::Diagnostic;
use modelspec_core::ir::tensor::{ParamId, TensorOp};
use modelspec_runtime::cpu::{HostTensor, evaluate_all, evaluate_all_with_ids};
use modelspec_runtime::train::checkpoint;
use modelspec_runtime::train::data::{self, DataError, DenseBatch, DenseData};
use modelspec_runtime::train::init::sigreg_projections;

use crate::train::Backend;
#[cfg(not(feature = "vulkan"))]
use crate::train::E_TARGET_001;

/// Arguments of `modelspec infer`.
#[derive(clap::Args)]
pub struct Args {
    /// The .mspec file whose forward program runs.
    pub path: PathBuf,
    /// The backend that runs the forward program.
    #[arg(long, value_enum)]
    pub backend: Backend,
    /// The .msckpt file whose parameters the forward program reads.
    #[arg(long, value_name = "PATH")]
    pub checkpoint: PathBuf,
    /// With --backend vulkan, run on the physical device INDEX, in the order the Vulkan loader
    /// lists them, instead of device 0.
    #[arg(long, value_name = "INDEX")]
    pub device: Option<usize>,
    /// Run the model function --function over every block of B samples of the dense directory
    /// DIR instead of the forward program on the synthetic batch: each argument NAME of the
    /// function reads DIR/NAME.f32 or DIR/NAME.i32 by its dtype, row-major little-endian, a
    /// sample one row along the leading dimension B every argument shares, else the whole
    /// argument (B = 1), and the sample count a multiple of B.
    #[arg(long, value_name = "DIR", requires_all = ["function", "out"])]
    pub data: Option<PathBuf>,
    /// With --data, the model function to run: any function of the model, not only forward,
    /// returning a loss or not.
    #[arg(long, value_name = "NAME", requires = "data")]
    pub function: Option<String>,
    /// With --data, write the function's result for every block, in block order, to FILE: one
    /// row per sample, row-major little-endian f32 with no header, written once every block has
    /// run and not at all on a refusal; an existing FILE is replaced whole or kept. A FILE that
    /// is the model file, the checkpoint or an argument's file of DIR is refused.
    #[arg(long, value_name = "FILE", requires = "data")]
    pub out: Option<PathBuf>,
}

/// Runs `modelspec infer` and returns the process exit code.
pub fn run(args: Args, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    if args.backend == Backend::Cpu && args.device.is_some() {
        let _ = writeln!(
            stderr,
            "error: the argument '--device' cannot be used with '--backend cpu'; it is \
             available only with '--backend vulkan'"
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
            let _ = writeln!(stderr, "modelspec infer {file}: {error}");
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
    if let (Some(dir), Some(name), Some(out)) = (&args.data, &args.function, &args.out) {
        let run = FunctionRun {
            file: &file,
            source: &source,
            compiled: &compiled,
            dir,
            name,
            out,
        };
        return run.run(&args, stdout, stderr);
    }
    #[cfg(feature = "vulkan")]
    let artifact = match args.backend {
        Backend::Cpu => None,
        Backend::Vulkan => {
            use modelspec_core::artifact::Backend as Target;
            match compile::compile_artifact(&file, &source, Target::Vulkan) {
                Ok(artifact) => Some(artifact),
                Err(diagnostics) => {
                    for diagnostic in &diagnostics {
                        let _ = writeln!(stderr, "{diagnostic}");
                    }
                    return crate::EXIT_FAILURE;
                }
            }
        }
    };
    let read = checkpoint::read(&args.checkpoint);
    let (params, step) = match read.and_then(|c| Ok((c.parameters_for(&compiled)?, c.step))) {
        Ok(read) => read,
        Err(error) => {
            let _ = writeln!(stderr, "{error}");
            return crate::EXIT_FAILURE;
        }
    };
    let mut batch = match data::synthetic(&compiled.config, &compiled.forward) {
        Ok(batch) => batch,
        Err(DataError::Refused(diagnostic)) => {
            let _ = writeln!(stderr, "{diagnostic}");
            return crate::EXIT_FAILURE;
        }
        Err(error) => {
            let _ = writeln!(stderr, "modelspec infer {file}: {error}");
            return crate::EXIT_FAILURE;
        }
    };
    // Every `sigreg` site reads its projection of the checkpoint's step, the step whose
    // parameters these are (story:jepa-sigreg-cpu).
    match sigreg_projections(&compiled.forward.program, compiled.config.seed, step) {
        Ok(projections) => batch.inputs.extend(projections),
        Err(error) => {
            let _ = writeln!(stderr, "modelspec infer {file}: {error}");
            return crate::EXIT_FAILURE;
        }
    }
    let forward = &compiled.forward;
    let Some(node) = forward.program.node(forward.prediction) else {
        let _ = writeln!(
            stderr,
            "modelspec infer {file}: the forward program defines no prediction value"
        );
        return crate::EXIT_FAILURE;
    };
    #[cfg(feature = "vulkan")]
    if let Some(artifact) = &artifact {
        use modelspec_runtime::vulkan::runner::{DeviceRunError, infer_on_device};
        return match infer_on_device(args.device, &compiled, artifact, &params) {
            Ok((inference, _)) => print(
                &file,
                &node.ty.to_string(),
                inference.prediction.data(),
                inference.loss,
                stdout,
                stderr,
            ),
            Err(DeviceRunError::Refused(diagnostic)) => {
                let _ = writeln!(stderr, "{diagnostic}");
                crate::EXIT_FAILURE
            }
            Err(DeviceRunError::Failed(error)) => {
                let _ = writeln!(stderr, "modelspec infer {file}: {error}");
                crate::EXIT_FAILURE
            }
        };
    }
    let mut values = match evaluate_all(&forward.program, &batch.inputs, &params) {
        Ok(values) => values,
        Err(error) => {
            let _ = writeln!(stderr, "modelspec infer {file}: {error}");
            return crate::EXIT_FAILURE;
        }
    };
    let Some(prediction) = values.values.remove(&forward.prediction) else {
        let _ = writeln!(
            stderr,
            "modelspec infer {file}: the forward program defines no prediction value"
        );
        return crate::EXIT_FAILURE;
    };
    let loss = values.outputs[0].data()[0];
    print(
        &file,
        &node.ty.to_string(),
        prediction.data(),
        loss,
        stdout,
        stderr,
    )
}

/// Prints the `prediction` and `loss` lines and returns the exit code.
fn print(
    file: &str,
    ty: &str,
    prediction: &[f32],
    loss: f32,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> u8 {
    let list: Vec<String> = prediction.iter().map(f32::to_string).collect();
    let printed = format!("prediction: {ty} = [{}]\nloss {loss}\n", list.join(", "));
    match stdout
        .write_all(printed.as_bytes())
        .and_then(|()| stdout.flush())
    {
        Ok(()) => crate::EXIT_SUCCESS,
        Err(error) => {
            let _ = writeln!(
                stderr,
                "modelspec infer {file}: cannot write the result: {error}"
            );
            crate::EXIT_FAILURE
        }
    }
}

/// Why a run of `infer --data --function` stopped: a diagnostic, or another error, printed
/// after `modelspec infer <file>: `.
enum Stop {
    Refused(Vec<String>),
    Failed(String),
}

impl From<DataError> for Stop {
    fn from(error: DataError) -> Self {
        match error {
            DataError::Refused(diagnostic) => Stop::Refused(vec![diagnostic.to_string()]),
            other => Stop::Failed(other.to_string()),
        }
    }
}

/// The refusal of a function [`compile::compile_function`] did not compile: E-DATA-010 for a
/// name the model does not declare, else the file's diagnostics.
fn function_stop(error: FunctionError) -> Stop {
    match error {
        FunctionError::Unknown { name, functions } => {
            Stop::from(data::unknown_function(&name, &functions))
        }
        FunctionError::Diagnostics(diagnostics) => {
            Stop::Refused(diagnostics.iter().map(ToString::to_string).collect())
        }
    }
}

/// `infer --data <dir> --function <name> --out <out>` of `file`, whose source is `source` and
/// whose `train` block compiled to `compiled`.
struct FunctionRun<'a> {
    file: &'a str,
    source: &'a str,
    compiled: &'a CompiledTraining,
    dir: &'a Path,
    name: &'a str,
    out: &'a Path,
}

impl FunctionRun<'_> {
    /// Runs it and returns the exit code.
    fn run(&self, args: &Args, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
        match self.results(args) {
            Ok((blocks, rows, values)) => self.write(blocks, rows, &values, stdout, stderr),
            Err(Stop::Refused(diagnostics)) => {
                for diagnostic in diagnostics {
                    let _ = writeln!(stderr, "{diagnostic}");
                }
                crate::EXIT_FAILURE
            }
            Err(Stop::Failed(error)) => {
                let _ = writeln!(stderr, "modelspec infer {}: {error}", self.file);
                crate::EXIT_FAILURE
            }
        }
    }

    /// The block count, the row count and the function's result for every block, in block
    /// order: refused in the order of the module documentation, an `--out` that names a file
    /// the run reads once the function compiles and before anything else is read, the
    /// checkpoint read before the data.
    fn results(&self, args: &Args) -> Result<(usize, usize, Vec<f32>), Stop> {
        let mut function =
            compile::compile_function(self.file, self.source, self.name).map_err(function_stop)?;
        function.config.steps = 0;
        let mut reads = vec![
            ("model".to_owned(), args.path.clone()),
            ("checkpoint".to_owned(), args.checkpoint.clone()),
        ];
        reads.extend(data::function_files(self.dir, &function.config));
        data::refuse_out_reading(self.out, &reads)?;
        let functions: Vec<String> = compile::compile_hir(self.file, self.source)
            .map_err(|diagnostics| {
                Stop::Refused(diagnostics.iter().map(ToString::to_string).collect())
            })?
            .model
            .functions
            .iter()
            .map(|f| f.name.clone())
            .collect();
        #[cfg(feature = "vulkan")]
        let artifact = match args.backend {
            Backend::Cpu => None,
            Backend::Vulkan => {
                use modelspec_core::artifact::Backend as Target;
                Some(
                    compile::compile_function_artifact(
                        self.file,
                        self.source,
                        self.name,
                        Target::Vulkan,
                    )
                    .map_err(function_stop)?,
                )
            }
        };
        let (params, step) = checkpoint::read(&args.checkpoint)
            .and_then(|c| Ok((c.parameters_for(self.compiled)?, c.step)))
            .map_err(|error| Stop::Refused(vec![error.to_string()]))?;
        let dense = data::dense_function(
            self.dir,
            self.name,
            &function.config,
            &function.forward,
            &functions,
        )?;
        let forward = &function.forward;
        let node = forward
            .program
            .node(forward.prediction)
            .ok_or_else(|| Stop::Failed("the function's program defines no result value".into()))?;
        let row = data::row_elements(self.name, &node.ty, dense.batch(), &functions)?;
        let blocks = dense.samples() / dense.batch();
        #[cfg(feature = "vulkan")]
        let values = match &artifact {
            Some(artifact) => on_device(args.device, &function, artifact, &params, &dense, step)?,
            None => on_cpu(&function, &dense, &params, step)?,
        };
        #[cfg(not(feature = "vulkan"))]
        let values = on_cpu(&function, &dense, &params, step)?;
        if values.len() != dense.samples() * row {
            return Err(Stop::Failed(format!(
                "the function's results hold {} values, not {} rows of {row}",
                values.len(),
                dense.samples()
            )));
        }
        Ok((blocks, dense.samples(), values))
    }

    /// Writes `values` to the output file ([`replace`]), then the stdout line, and returns the
    /// exit code.
    fn write(
        &self,
        blocks: usize,
        rows: usize,
        values: &[f32],
        stdout: &mut dyn Write,
        stderr: &mut dyn Write,
    ) -> u8 {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        if let Err(error) = replace(self.out, &bytes) {
            let _ = writeln!(
                stderr,
                "modelspec infer {}: cannot write {}: {error}",
                self.file,
                self.out.display()
            );
            return crate::EXIT_FAILURE;
        }
        let line = format!(
            "function {} blocks {blocks} rows {rows} out {}\n",
            self.name,
            self.out.display()
        );
        match stdout
            .write_all(line.as_bytes())
            .and_then(|()| stdout.flush())
        {
            Ok(()) => crate::EXIT_SUCCESS,
            Err(error) => {
                let _ = writeln!(
                    stderr,
                    "modelspec infer {}: cannot write the result: {error}",
                    self.file
                );
                crate::EXIT_FAILURE
            }
        }
    }
}

/// Replaces `out` with `bytes` whole or not at all: the bytes go to a new temporary file
/// `.<name>.<pid>.tmp` in `out`'s directory, which is synced and renamed over `out`. On any
/// error the temporary file is removed and an existing `out` keeps its bytes; a process killed
/// part of the way (`SIGXFSZ` past a file-size limit) leaves `out` as it was and the temporary
/// file beside it.
fn replace(out: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let name = out.file_name().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "the path names no file")
    })?;
    let dir = match out.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    let mut temporary = std::ffi::OsString::from(".");
    temporary.push(name);
    temporary.push(format!(".{}.tmp", std::process::id()));
    let temporary = dir.join(temporary);
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let written = file
        .write_all(bytes)
        .and_then(|()| file.sync_all())
        .and_then(|()| {
            drop(file);
            std::fs::rename(&temporary, out)
        });
    if written.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    written
}

/// The result of `function`, compiled by [`compile::compile_function`], on every block of
/// `dense` on the CPU interpreter with the parameters `params` of the checkpoint of step
/// `step`, in block order. Each block feeds the arguments the function's program reads, and
/// every `sigreg` site of the function the projection of `step`, as `infer` without `--data`
/// reads it (story:jepa-sigreg-cpu).
fn on_cpu(
    function: &CompiledTraining,
    dense: &DenseData,
    params: &BTreeMap<ParamId, HostTensor<f32>>,
    step: u64,
) -> Result<Vec<f32>, Stop> {
    let forward = &function.forward;
    let program = &forward.program;
    let projections = sigreg_projections(program, function.config.seed, step)
        .map_err(|error| Stop::Failed(error.to_string()))?;
    let read: BTreeSet<&str> = program
        .inputs
        .iter()
        .filter_map(|&value| match &program.node(value)?.op {
            TensorOp::Input { name } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    let mut batch = DenseBatch::default();
    let mut values = Vec::with_capacity(dense.samples());
    for b in 0..dense.samples() / dense.batch() {
        dense.fill_eval_batch(b, &mut batch);
        let inputs: BTreeMap<String, HostTensor<f32>> = batch
            .inputs()
            .iter()
            .filter(|(name, _)| read.contains(name.as_str()))
            .map(|(name, value)| (name.clone(), value.clone()))
            .chain(projections.clone())
            .collect();
        let ids: BTreeMap<String, HostTensor<i32>> = batch
            .ids()
            .iter()
            .filter(|(name, _)| read.contains(name.as_str()))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        let mut evaluated = evaluate_all_with_ids(program, &inputs, &ids, params)
            .map_err(|error| Stop::Failed(error.to_string()))?;
        let result = evaluated
            .values
            .remove(&forward.prediction)
            .ok_or_else(|| Stop::Failed(format!("block {b} has no result value")))?;
        values.extend_from_slice(result.data());
    }
    Ok(values)
}

/// [`on_cpu`] on Vulkan device `device` (0 when `None`): the step of `function` runs stepwise
/// with 0 steps from the checkpoint's step `step` and `dense` as its data and its dense eval
/// data, each block's prediction kept; every block reads the `sigreg` projections of the step
/// it is evaluated at, `step`, as [`on_cpu`] does ([`modelspec_runtime::vulkan::stepwise`]).
#[cfg(feature = "vulkan")]
fn on_device(
    device: Option<usize>,
    function: &CompiledTraining,
    artifact: &modelspec_core::artifact::CompiledArtifact,
    params: &BTreeMap<ParamId, HostTensor<f32>>,
    dense: &DenseData,
    step: u64,
) -> Result<Vec<f32>, Stop> {
    use modelspec_runtime::train::runner::Data;
    use modelspec_runtime::vulkan::runner::DeviceRunError;
    use modelspec_runtime::vulkan::stepwise::{StepwiseOptions, train_stepwise_on_device};
    let mut kept = Vec::with_capacity(dense.samples());
    let options = StepwiseOptions {
        start: step,
        init: params.clone(),
        data: Some(Data::Dense(dense)),
        dense_eval: Some(dense),
        dense_predictions: Some(&mut kept),
        ..StepwiseOptions::default()
    };
    match train_stepwise_on_device(device, false, function, artifact, options, &mut |_| Ok(())) {
        Ok(_) => Ok(kept),
        Err(DeviceRunError::Refused(diagnostic)) => {
            Err(Stop::Refused(vec![diagnostic.to_string()]))
        }
        Err(DeviceRunError::Failed(error)) => Err(Stop::Failed(error.to_string())),
    }
}
