//! `modelspec train --backend vulkan` (DESIGN.md § 6, § 8; `story:m1-vulkan-train`).
//!
//! The device test runs `modelspec train examples/mnist_mlp.mspec` with `--backend vulkan` and
//! with `--backend cpu` and checks that both exit 0 and print the same `init`, `step` and
//! `param` lines in the same order, the values aside: the tolerance of the values is the
//! runtime's `tests/vulkan_train.rs`. It is ignored by default because it creates a Vulkan
//! instance, which `task ci` never does; run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-cli --test train_vulkan -- --ignored`.
//!
//! A second device test trains the `sigreg` fixture of story:jepa-sigreg-cpu on both backends
//! and holds every step's recorded loss and every final parameter element to the run tolerance
//! (story:jepa-sigreg-vulkan).
//!
//! The other tests are usage errors, refused before the file is read or a loader opened.
#![cfg(feature = "vulkan")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[allow(dead_code)]
#[path = "../../modelspec-runtime/tests/common/mod.rs"]
mod common;

use common::{GPU_RUN_ATOL, GPU_RUN_RTOL, within};
use serde_json::Value;

const MNIST_MLP: &str = "examples/mnist_mlp.mspec";
/// The `sigreg` fixture of story:jepa-sigreg-cpu: 100 `sgd` steps on `sigreg` of a `[256, 8]`
/// parameter over 16 slices.
const SIGREG_TRAIN: &str = "crates/modelspec-cli/tests/fixtures/sigreg_train.mspec";
const LINEAR_REGRESSION: &str = "examples/linear_regression.mspec";

/// A loader path that does not exist, so a test that reaches the loader fails with E-TARGET-002
/// instead of creating an instance.
const NO_LOADER: &str = "/nonexistent/modelspec-test/libvulkan-missing.so.1";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn modelspec(args: &[&str], loader: Option<&str>) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_modelspec"));
    command.args(args).current_dir(repo_root());
    if let Some(loader) = loader {
        command.env("MODELSPEC_VULKAN_LIBRARY", loader);
    }
    command.output().expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// What a stdout line is, its values aside: `init @<path>: <type>`, `step <k>` or
/// `param @<path>: <type>`.
fn shape_of(line: &str) -> String {
    match line.split_once(" = [") {
        Some((head, _)) => head.to_owned(),
        None => match line.split_once(" loss ") {
            Some((head, _)) => head.to_owned(),
            None => line.to_owned(),
        },
    }
}

#[test]
#[ignore = "needs a Vulkan device"]
fn mnist_mlp_on_vulkan_prints_the_lines_of_the_cpu_run() {
    let gpu = modelspec(
        &["train", MNIST_MLP, "--backend", "vulkan", "--stats"],
        None,
    );
    let stderr = text(&gpu.stderr);
    assert_eq!(gpu.status.code(), Some(0), "{stderr}");
    let cpu = modelspec(&["train", MNIST_MLP, "--backend", "cpu"], None);
    assert_eq!(cpu.status.code(), Some(0), "{}", text(&cpu.stderr));
    let (gpu_out, cpu_out) = (text(&gpu.stdout), text(&cpu.stdout));
    let gpu_lines: Vec<String> = gpu_out.lines().map(shape_of).collect();
    let cpu_lines: Vec<String> = cpu_out.lines().map(shape_of).collect();
    assert_eq!(gpu_lines.len(), cpu_lines.len(), "the number of lines");
    assert_eq!(gpu_lines, cpu_lines, "the lines in order");
    assert_eq!(
        gpu_out.lines().filter(|l| l.starts_with("step ")).count(),
        37,
        "step 0, every 50 steps to 1800"
    );
    let init = |out: &str| -> Vec<String> {
        out.lines()
            .filter(|l| l.starts_with("init "))
            .map(str::to_owned)
            .collect()
    };
    assert_eq!(
        init(&gpu_out),
        init(&cpu_out),
        "the device run starts from the CPU's initial parameters"
    );
    eprintln!("{stderr}");
    assert!(stderr.contains("host waits: 1\n"), "{stderr}");
}

/// `--init` stays CPU-only, a usage error on Vulkan; `--eval` runs on Vulkan since
/// story:decider-dense-named-data, so beside a file that does not exist it is not a usage error
/// but the missing file's refusal, exit 1.
#[test]
fn the_data_flags_are_usage_errors_on_vulkan() {
    let args = [
        "train",
        "no-such-model.mspec",
        "--backend",
        "vulkan",
        "--init",
        "init",
    ];
    let output = modelspec(&args, Some(NO_LOADER));
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("--init"), "{stderr}");
    assert!(output.stdout.is_empty());

    let args = [
        "train",
        "no-such-model.mspec",
        "--backend",
        "vulkan",
        "--eval",
        "eval",
    ];
    let output = modelspec(&args, Some(NO_LOADER));
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(!stderr.contains("--eval"), "{stderr}");
    assert!(stderr.contains("no-such-model.mspec"), "{stderr}");
    assert!(output.stdout.is_empty());
}

#[test]
fn the_device_and_stats_flags_are_usage_errors_on_the_cpu() {
    for flags in [&["--device", "0"][..], &["--stats"]] {
        let mut args = vec!["train", LINEAR_REGRESSION, "--backend", "cpu"];
        args.extend_from_slice(flags);
        let output = modelspec(&args, Some(NO_LOADER));
        let stderr = text(&output.stderr);
        assert_eq!(output.status.code(), Some(2), "{flags:?}: {stderr}");
        assert!(stderr.contains(flags[0]), "{flags:?}: {stderr}");
        assert!(output.stdout.is_empty(), "{flags:?}");
    }
    let output = modelspec(
        &[
            "train",
            LINEAR_REGRESSION,
            "--backend",
            "vulkan",
            "--device",
            "0",
            "--stats",
        ],
        Some(NO_LOADER),
    );
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.starts_with("E-TARGET-002: "), "{stderr}");
}

#[test]
fn profile_is_a_usage_error_on_the_cpu() {
    let output = modelspec(
        &["train", LINEAR_REGRESSION, "--backend", "cpu", "--profile"],
        Some(NO_LOADER),
    );
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("'--profile'"), "{stderr}");
    assert!(
        stderr.contains("only with '--backend vulkan'"),
        "the error names --profile as Vulkan only: {stderr}"
    );
    assert!(output.stdout.is_empty());
}

/// A Dispatch step of the step plan: `(position, kernel module id)`.
type Site = (usize, String);

/// The Dispatch steps of the step plan, as `modelspec ir --stage kernel` prints the plan, and of
/// them those the final loss evaluation records: the steps the loss is computed from
/// (`ExecutionPlan::loss_steps`, story:step-sites-beat-pytorch), no longer every step up to the
/// last that writes the loss.
fn dispatch_sites(file: &str) -> (Vec<Site>, Vec<Site>) {
    let ir = modelspec(&["ir", file, "--stage", "kernel"], None);
    assert_eq!(ir.status.code(), Some(0), "{}", text(&ir.stderr));
    let ir = text(&ir.stdout);
    let kernels: std::collections::BTreeMap<String, String> = ir
        .lines()
        .filter_map(|l| l.strip_prefix("kernel "))
        .map(|l| {
            let mut words = l.split(' ');
            (
                words.next().unwrap().to_owned(),
                words.next().unwrap().to_owned(),
            )
        })
        .collect();
    let mut sites = Vec::new();
    for line in ir
        .lines()
        .skip_while(|l| *l != "plan step")
        .skip(1)
        .take_while(|l| l.starts_with("  "))
    {
        let (at, step) = line.trim().split_once(": ").unwrap();
        let at: usize = at.parse().unwrap();
        if let Some(rest) = step.strip_prefix("dispatch ") {
            let kernel = rest.split(' ').next().unwrap();
            sites.push((at, kernels[kernel].clone()));
        }
    }
    let source = std::fs::read_to_string(repo_root().join(file)).expect("the example reads");
    let artifact = modelspec_core::compile::compile_artifact(
        file,
        &source,
        modelspec_core::artifact::Backend::Vulkan,
    )
    .expect("the example compiles");
    let loss_steps = artifact.execution.loss_steps();
    assert!(!loss_steps.is_empty(), "a step writes the loss");
    let evaluated = sites
        .iter()
        .filter(|(at, _)| loss_steps.contains(at))
        .cloned()
        .collect();
    (sites, evaluated)
}

#[test]
#[ignore = "needs a Vulkan device"]
fn mnist_mlp_profile_prints_every_dispatch_site_and_leaves_stdout_alone() {
    let plain = modelspec(
        &["train", MNIST_MLP, "--backend", "vulkan", "--stats"],
        None,
    );
    assert_eq!(plain.status.code(), Some(0), "{}", text(&plain.stderr));
    let started = std::time::Instant::now();
    let profiled = modelspec(
        &[
            "train",
            MNIST_MLP,
            "--backend",
            "vulkan",
            "--stats",
            "--profile",
        ],
        None,
    );
    let wall_us = started.elapsed().as_secs_f64() * 1e6;
    let stderr = text(&profiled.stderr);
    assert_eq!(profiled.status.code(), Some(0), "{stderr}");
    eprintln!("{stderr}");
    assert_eq!(profiled.stdout, plain.stdout, "stdout is byte-equal");
    assert!(stderr.contains("host waits: 1\n"), "{stderr}");
    assert!(text(&plain.stderr).contains("host waits: 1\n"));

    let (step_sites, final_sites) = dispatch_sites(MNIST_MLP);
    let lines: Vec<Vec<&str>> = stderr
        .lines()
        .filter_map(|l| l.strip_prefix("profile dispatch "))
        .map(|l| l.split(' ').collect())
        .collect();
    let expected: Vec<(String, usize, String, u64)> = step_sites
        .iter()
        .map(|(at, id)| ("step".to_owned(), *at, id.clone(), 1800))
        .chain(
            final_sites
                .iter()
                .map(|(at, id)| ("final".to_owned(), *at, id.clone(), 1)),
        )
        .collect();
    let got: Vec<(String, usize, String, u64)> = lines
        .iter()
        .map(|w| {
            assert_eq!(w.len(), 18, "{w:?}");
            assert_eq!((w[4], w[6], w[8]), ("calls", "total_us", "mean_us"));
            assert_eq!(
                (w[10], w[12], w[14], w[16]),
                ("flop", "bytes", "gflop_per_s", "gb_per_s")
            );
            (
                w[0].to_owned(),
                w[1].parse().unwrap(),
                w[2].to_owned(),
                w[5].parse().unwrap(),
            )
        })
        .collect();
    assert_eq!(got, expected);
    let sum_us: f64 = lines.iter().map(|w| w[7].parse::<f64>().unwrap()).sum();
    assert!(
        sum_us <= wall_us,
        "{sum_us} us of dispatches in {wall_us} us"
    );
    let dispatches = 1800 * step_sites.len() + final_sites.len();
    assert!(
        stderr.contains(&format!("profile dispatches {dispatches}\n")),
        "{stderr}"
    );
    for counter in [
        "profile submits ",
        "profile host_waits 1\n",
        "profile bytes_uploaded 520864\n",
        "profile bytes_downloaded 210468\n",
        "profile pipeline_creation_us ",
        "profile compile_us ",
        "profile steady_us_per_step ",
        "profile step_total flop ",
    ] {
        assert!(stderr.contains(counter), "{counter}: {stderr}");
    }
}

/// The loss of every `StepCompleted` of a recorded run file, by step.
fn recorded_losses(file: &Path) -> BTreeMap<u64, f32> {
    std::fs::read_to_string(file)
        .expect("the recording")
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("a JSON event"))
        .filter(|event| event["body"]["kind"] == "StepCompleted")
        .map(|event| {
            let value = &event["body"]["value"];
            (
                value["step"].as_u64().expect("a step"),
                value["loss"].as_f64().expect("a loss") as f32,
            )
        })
        .collect()
}

/// The values of every `param @<path>: <type> = [v, ...]` line, by path.
fn final_params(out: &str) -> BTreeMap<String, Vec<f32>> {
    out.lines()
        .filter_map(|line| line.strip_prefix("param @"))
        .map(|rest| {
            let (path, rest) = rest.split_once(": ").expect("`@<path>: <type> = [..]`");
            let (_, list) = rest.split_once(" = [").expect("a value list");
            let list = list.strip_suffix(']').expect("a closed list");
            let values = list
                .split(", ")
                .map(|v| v.parse::<f32>().expect("an f32"))
                .collect();
            (path.to_owned(), values)
        })
        .collect()
}

/// story:jepa-sigreg-vulkan: the `sigreg` fixture trained on Vulkan, its projection uploaded
/// before each step, prints the lines of the CPU run in order, and each of its 101 recorded step
/// losses and every element of its parameter after the run is within
/// `GPU_RUN_ATOL + GPU_RUN_RTOL · |c|` of the CPU run's `c`. Prints the largest differences
/// and whether the two runs were bit-equal.
#[test]
#[ignore = "needs a Vulkan device"]
fn sigreg_on_vulkan_matches_the_cpu_run_within_the_run_tolerance() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("train_vulkan_sigreg");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    let run = |backend: &str| {
        let record = dir.join(format!("{backend}.run.jsonl"));
        let output = modelspec(
            &[
                "train",
                SIGREG_TRAIN,
                "--backend",
                backend,
                "--record",
                record.to_str().expect("a UTF-8 path"),
            ],
            None,
        );
        assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
        (text(&output.stdout), recorded_losses(&record))
    };
    let (gpu_out, gpu_losses) = run("vulkan");
    let (cpu_out, cpu_losses) = run("cpu");
    let gpu_lines: Vec<String> = gpu_out.lines().map(shape_of).collect();
    let cpu_lines: Vec<String> = cpu_out.lines().map(shape_of).collect();
    assert_eq!(gpu_lines, cpu_lines, "the lines in order");

    let mut outside = Vec::new();
    assert_eq!(cpu_losses.len(), 101, "step 0 to 100");
    assert_eq!(
        cpu_losses.keys().collect::<Vec<_>>(),
        gpu_losses.keys().collect::<Vec<_>>()
    );
    let (mut loss_diff, mut bits_equal) = (0.0f64, true);
    for (step, &c) in &cpu_losses {
        let g = gpu_losses[step];
        loss_diff = loss_diff.max((f64::from(g) - f64::from(c)).abs());
        bits_equal &= g.to_bits() == c.to_bits();
        if !within(g, c, c.abs(), GPU_RUN_ATOL, GPU_RUN_RTOL) {
            outside.push(format!("step {step}: gpu {g} cpu {c}"));
        }
    }
    let (cpu_params, gpu_params) = (final_params(&cpu_out), final_params(&gpu_out));
    assert_eq!(cpu_params.len(), 1, "the one parameter");
    assert_eq!(
        cpu_params.keys().collect::<Vec<_>>(),
        gpu_params.keys().collect::<Vec<_>>()
    );
    let (mut elements, mut param_diff) = (0usize, 0.0f64);
    for (path, c) in &cpu_params {
        let g = &gpu_params[path];
        assert_eq!(g.len(), c.len(), "{path}");
        for (i, (&g, &c)) in g.iter().zip(c).enumerate() {
            elements += 1;
            param_diff = param_diff.max((f64::from(g) - f64::from(c)).abs());
            bits_equal &= g.to_bits() == c.to_bits();
            if !within(g, c, c.abs(), GPU_RUN_ATOL, GPU_RUN_RTOL) && outside.len() < 8 {
                outside.push(format!("{path}[{i}]: gpu {g} cpu {c}"));
            }
        }
    }
    eprintln!(
        "sigreg run: {} step losses, max |g-c| {loss_diff:.3e}; {elements} parameter elements, \
         max |g-c| {param_diff:.3e}; bit-equal: {}",
        cpu_losses.len(),
        if bits_equal { "yes" } else { "no" }
    );
    assert_eq!(elements, 256 * 8);
    assert!(outside.is_empty(), "{outside:#?}");
}

#[test]
fn a_file_that_does_not_compile_is_reported_before_the_loader_is_opened() {
    let output = modelspec(
        &["train", "no-such-model.mspec", "--backend", "vulkan"],
        Some(NO_LOADER),
    );
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.starts_with("modelspec train no-such-model.mspec: "),
        "{stderr}"
    );
}
