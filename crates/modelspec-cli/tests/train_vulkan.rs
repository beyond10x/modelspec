//! `modelspec train --backend vulkan` (DESIGN.md § 6, § 8; `story:m1-vulkan-train`).
//!
//! The device test runs `modelspec train examples/mnist_mlp.mspec` with `--backend vulkan` and
//! with `--backend cpu` and checks that both exit 0 and print the same `init`, `step` and
//! `param` lines in the same order, the values aside: the tolerance of the values is the
//! runtime's `tests/vulkan_train.rs`. It is ignored by default because it creates a Vulkan
//! instance, which `task ci` never does; run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-cli --test train_vulkan -- --ignored`.
//!
//! The other tests are usage errors, refused before the file is read or a loader opened.
#![cfg(feature = "vulkan")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const MNIST_MLP: &str = "examples/mnist_mlp.mspec";
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

#[test]
fn the_checkpoint_and_data_flags_are_usage_errors_on_vulkan() {
    for flags in [
        &["--checkpoint", "run.msckpt"][..],
        &["--resume", "run.msckpt"],
        &["--data", "data"],
        &["--init", "init"],
        &["--eval", "eval"],
    ] {
        let mut args = vec!["train", "no-such-model.mspec", "--backend", "vulkan"];
        args.extend_from_slice(flags);
        let output = modelspec(&args, Some(NO_LOADER));
        let stderr = text(&output.stderr);
        assert_eq!(output.status.code(), Some(2), "{flags:?}: {stderr}");
        assert!(stderr.contains(flags[0]), "{flags:?}: {stderr}");
        assert!(output.stdout.is_empty(), "{flags:?}");
    }
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

/// The Dispatch steps of the step plan, and of them those up to the last that writes the loss,
/// as `modelspec ir --stage kernel` prints the plan.
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
    let loss = ir
        .lines()
        .find_map(|l| l.strip_prefix("loss "))
        .and_then(|view| view.split('[').next())
        .expect("the plan names the loss view")
        .to_owned();
    let mut sites = Vec::new();
    let mut loss_at = None;
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
        if step
            .split_once(" writes ")
            .is_some_and(|(_, w)| w.starts_with(&format!("{loss}[")))
        {
            loss_at = Some(at);
        }
    }
    let loss_at = loss_at.expect("a step writes the loss");
    let up_to_loss = sites
        .iter()
        .filter(|(at, _)| *at <= loss_at)
        .cloned()
        .collect();
    (sites, up_to_loss)
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
                .map(|(at, id)| ("forward".to_owned(), *at, id.clone(), 1)),
        )
        .collect();
    let got: Vec<(String, usize, String, u64)> = lines
        .iter()
        .map(|w| {
            assert_eq!(w.len(), 10, "{w:?}");
            assert_eq!((w[4], w[6], w[8]), ("calls", "total_us", "mean_us"));
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
    ] {
        assert!(stderr.contains(counter), "{counter}: {stderr}");
    }
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
