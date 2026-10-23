//! `modelspec train --backend vulkan` with `--checkpoint`, `--checkpoint-every` and `--resume`,
//! and `modelspec infer --backend vulkan` (DESIGN.md § 6.2, § 8; `story:m1-vulkan-checkpoint-infer`).
//!
//! The tests that run in `task ci` create no Vulkan instance: every run sets
//! `MODELSPEC_VULKAN_LIBRARY` to a path that does not exist, so a run that reaches the loader
//! exits 1 with E-TARGET-002, and `infer` reads the checkpoint before it opens the loader. A build
//! without the `vulkan` feature refuses `infer --backend vulkan` with E-TARGET-001.
//!
//! The device tests are ignored by default; run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-cli --test checkpoint_vulkan -- --ignored`.
//! Their tolerances are those of `crates/modelspec-runtime/tests/common/mod.rs`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[cfg(feature = "vulkan")]
#[allow(dead_code)]
#[path = "../../modelspec-runtime/tests/common/mod.rs"]
mod common;

const LINEAR_REGRESSION: &str = "examples/linear_regression.mspec";
#[cfg(feature = "vulkan")]
const MNIST_MLP: &str = "examples/mnist_mlp.mspec";

/// A loader path that does not exist, so no Vulkan library is opened.
const NO_LOADER: &str = "/nonexistent/modelspec-test/libvulkan-missing.so.1";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh directory for one test.
#[cfg(feature = "vulkan")]
fn dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("checkpoint_vulkan")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    dir
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

#[cfg(feature = "vulkan")]
fn path(p: &Path) -> &str {
    p.to_str().expect("a UTF-8 path")
}

/// The code a stderr starts with: the text before its first `: `.
#[cfg(feature = "vulkan")]
fn code(stderr: &str) -> &str {
    stderr.split_once(": ").map_or(stderr, |(code, _)| code)
}

#[cfg(not(feature = "vulkan"))]
#[test]
fn infer_on_vulkan_is_refused_without_the_feature() {
    let output = modelspec(
        &[
            "infer",
            LINEAR_REGRESSION,
            "--backend",
            "vulkan",
            "--checkpoint",
            "x.msckpt",
        ],
        Some(NO_LOADER),
    );
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.starts_with("E-TARGET-001: "), "{stderr}");
    assert!(output.stdout.is_empty());
}

/// A checkpoint of a 5-step CPU run of the linear regression example, at `dir/run.msckpt`.
#[cfg(feature = "vulkan")]
fn cpu_checkpoint(dir: &Path) -> PathBuf {
    let checkpoint = dir.join("run.msckpt");
    let output = modelspec(
        &[
            "train",
            LINEAR_REGRESSION,
            "--backend",
            "cpu",
            "--steps",
            "5",
            "--checkpoint",
            path(&checkpoint),
        ],
        Some(NO_LOADER),
    );
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    checkpoint
}

/// Acceptance 5: with a checkpoint that matches the file, the missing loader is the refusal.
#[cfg(feature = "vulkan")]
#[test]
fn infer_on_vulkan_without_a_loader_exits_1_with_e_target_002() {
    let d = dir("no-loader");
    let checkpoint = cpu_checkpoint(&d);
    let output = modelspec(
        &[
            "infer",
            LINEAR_REGRESSION,
            "--backend",
            "vulkan",
            "--checkpoint",
            path(&checkpoint),
        ],
        Some(NO_LOADER),
    );
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.starts_with("E-TARGET-002: "), "{stderr}");
    assert!(output.stdout.is_empty(), "{}", text(&output.stdout));
}

/// Acceptance 5: `infer` reads the checkpoint before it opens the loader, so a malformed one is
/// refused with the code the CPU reader gives, the loader variable still set.
#[cfg(feature = "vulkan")]
#[test]
fn infer_on_vulkan_refuses_a_malformed_checkpoint_before_the_loader() {
    let d = dir("malformed");
    let malformed = d.join("bad.msckpt");
    std::fs::write(&malformed, b"MSCKnot a checkpoint at all").expect("the file is written");
    let infer = |backend: &str| {
        modelspec(
            &[
                "infer",
                LINEAR_REGRESSION,
                "--backend",
                backend,
                "--checkpoint",
                path(&malformed),
            ],
            Some(NO_LOADER),
        )
    };
    let cpu = infer("cpu");
    let cpu_stderr = text(&cpu.stderr);
    assert_eq!(cpu.status.code(), Some(1), "{cpu_stderr}");
    assert!(code(&cpu_stderr).starts_with("E-CKPT-"), "{cpu_stderr}");
    let gpu = infer("vulkan");
    let gpu_stderr = text(&gpu.stderr);
    assert_eq!(gpu.status.code(), Some(1), "{gpu_stderr}");
    assert_eq!(code(&gpu_stderr), code(&cpu_stderr), "{gpu_stderr}");
    assert!(gpu.stdout.is_empty());
}

/// `train --backend vulkan` takes `--checkpoint`, `--checkpoint-every` and `--resume`: it reads
/// the `--resume` checkpoint before the loader and reaches the loader with a good one.
#[cfg(feature = "vulkan")]
#[test]
fn train_on_vulkan_takes_the_checkpoint_flags() {
    let d = dir("train-flags");
    let checkpoint = cpu_checkpoint(&d);
    let out = d.join("out.msckpt");
    let output = modelspec(
        &[
            "train",
            LINEAR_REGRESSION,
            "--backend",
            "vulkan",
            "--checkpoint",
            path(&out),
            "--checkpoint-every",
            "100",
            "--resume",
            path(&checkpoint),
        ],
        Some(NO_LOADER),
    );
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.starts_with("E-TARGET-002: "), "{stderr}");
    assert!(!out.exists(), "no checkpoint without a run");

    let malformed = d.join("bad.msckpt");
    std::fs::write(&malformed, b"not a checkpoint").expect("the file is written");
    let output = modelspec(
        &[
            "train",
            LINEAR_REGRESSION,
            "--backend",
            "vulkan",
            "--resume",
            path(&malformed),
        ],
        Some(NO_LOADER),
    );
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(code(&stderr).starts_with("E-CKPT-"), "{stderr}");
}

/// The value of the `--stats` line `<name>: <value>` in `stderr`.
#[cfg(feature = "vulkan")]
fn stat<'s>(stderr: &'s str, name: &str) -> &'s str {
    stderr
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{name}: ")))
        .unwrap_or_else(|| panic!("no `{name}` line: {stderr}"))
}

/// The `step <k> loss <v>` lines of a train stdout, as `(k, v)`.
#[cfg(feature = "vulkan")]
fn losses(stdout: &str) -> Vec<(u64, f32)> {
    stdout
        .lines()
        .filter_map(|line| {
            let (k, v) = line.strip_prefix("step ")?.split_once(" loss ")?;
            Some((k.parse().ok()?, v.parse().ok()?))
        })
        .collect()
}

#[cfg(feature = "vulkan")]
fn loss_within(g: f32, c: f32) -> bool {
    let tolerance = common::term_tolerance(common::KernelOp::Mse);
    tolerance.accepts(g, c, tolerance.r(c, f64::from(c), 1))
}

#[cfg(feature = "vulkan")]
fn ok(args: &[&str]) -> Output {
    let output = modelspec(args, None);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{args:?}\nstderr: {}",
        text(&output.stderr)
    );
    output
}

/// Acceptance 2 through the CLI: `--checkpoint-every 600` of 1,800 steps waits 3 times and
/// writes 2 files beside the last; without it the run waits once.
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "needs a Vulkan device"]
fn checkpoint_every_600_waits_three_times_and_writes_two_files() {
    let d = dir("every");
    let run = d.join("run.msckpt");
    let output = ok(&[
        "train",
        MNIST_MLP,
        "--backend",
        "vulkan",
        "--stats",
        "--checkpoint",
        path(&run),
        "--checkpoint-every",
        "600",
    ]);
    let stderr = text(&output.stderr);
    eprintln!("{stderr}");
    assert_eq!(stat(&stderr, "host waits"), "3");
    let mut files: Vec<String> = std::fs::read_dir(&d)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    files.sort();
    assert_eq!(files, ["run-1200.msckpt", "run-600.msckpt", "run.msckpt"]);
    let plain = ok(&["train", MNIST_MLP, "--backend", "vulkan", "--stats"]);
    let stderr = text(&plain.stderr);
    assert_eq!(stat(&stderr, "host waits"), "1");
    assert_eq!(
        losses(&text(&plain.stdout)).len(),
        losses(&text(&output.stdout)).len()
    );
}

/// Acceptance 3 through the CLI: `--steps 900`, then `--resume` for 900, against 1,800 steps;
/// the parameters of the two last checkpoints pass the whole-run bound, the printed losses from
/// 900 the `mse` tolerance.
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "needs a Vulkan device"]
fn a_resumed_vulkan_run_passes_against_the_uninterrupted_one() {
    use modelspec_runtime::train::checkpoint;
    let d = dir("resume");
    let (whole, mid, resumed) = (
        d.join("whole.msckpt"),
        d.join("mid.msckpt"),
        d.join("resumed.msckpt"),
    );
    let uninterrupted = ok(&[
        "train",
        MNIST_MLP,
        "--backend",
        "vulkan",
        "--checkpoint",
        path(&whole),
    ]);
    ok(&[
        "train",
        MNIST_MLP,
        "--backend",
        "vulkan",
        "--steps",
        "900",
        "--checkpoint",
        path(&mid),
    ]);
    let second = ok(&[
        "train",
        MNIST_MLP,
        "--backend",
        "vulkan",
        "--steps",
        "900",
        "--resume",
        path(&mid),
        "--checkpoint",
        path(&resumed),
        "--stats",
    ]);
    assert_eq!(stat(&text(&second.stderr), "host waits"), "1");
    let (c, g) = (
        checkpoint::read(&whole).unwrap(),
        checkpoint::read(&resumed).unwrap(),
    );
    assert_eq!((c.step, g.step), (1800, 1800));
    let mut bit_equal = true;
    for (g, c) in g.parameters.iter().zip(&c.parameters) {
        assert_eq!(g.path, c.path);
        assert!(
            common::run_within(g.value.data(), c.value.data()),
            "{}: outside the whole-run bound",
            c.path
        );
        bit_equal &= g.value == c.value;
    }
    let full: Vec<(u64, f32)> = losses(&text(&uninterrupted.stdout))
        .into_iter()
        .filter(|(k, _)| *k >= 900)
        .collect();
    let tail = losses(&text(&second.stdout));
    assert_eq!(
        tail.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        full.iter().map(|(k, _)| *k).collect::<Vec<_>>()
    );
    for ((k, g), (_, c)) in tail.iter().zip(&full) {
        assert!(
            loss_within(*g, *c),
            "step {k}: resumed {g}, uninterrupted {c}"
        );
    }
    eprintln!(
        "parameters bit-equal {bit_equal}, printed losses bit-equal {}",
        tail == full
    );
}

/// Acceptance 4 through the CLI: `infer --backend cpu` and `--backend vulkan` on the last
/// checkpoint of a device run. The one host wait of device inference is the runtime's
/// `tests/vulkan_checkpoint.rs`: `infer` prints no counters.
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "needs a Vulkan device"]
fn cpu_and_vulkan_infer_load_a_device_checkpoint() {
    let d = dir("infer");
    let run = d.join("run.msckpt");
    let trained = ok(&[
        "train",
        MNIST_MLP,
        "--backend",
        "vulkan",
        "--checkpoint",
        path(&run),
    ]);
    let last = losses(&text(&trained.stdout)).last().unwrap().1;
    let infer_loss = |stdout: &str| -> f32 {
        let lines: Vec<&str> = stdout.lines().collect();
        assert_eq!(lines.len(), 2, "{stdout}");
        assert!(lines[0].starts_with("prediction: "), "{stdout}");
        lines[1].strip_prefix("loss ").unwrap().parse().unwrap()
    };
    let cpu = ok(&[
        "infer",
        MNIST_MLP,
        "--backend",
        "cpu",
        "--checkpoint",
        path(&run),
    ]);
    let cpu_loss = infer_loss(&text(&cpu.stdout));
    assert!(loss_within(cpu_loss, last), "cpu {cpu_loss}, run {last}");
    let gpu = ok(&[
        "infer",
        MNIST_MLP,
        "--backend",
        "vulkan",
        "--checkpoint",
        path(&run),
    ]);
    let gpu_loss = infer_loss(&text(&gpu.stdout));
    assert!(
        loss_within(gpu_loss, cpu_loss),
        "vulkan {gpu_loss}, cpu {cpu_loss}"
    );
    eprintln!("run {last}, cpu infer {cpu_loss}, vulkan infer {gpu_loss}");
}
