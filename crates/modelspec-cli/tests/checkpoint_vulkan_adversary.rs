//! Adversarial cases for `train --backend vulkan` with checkpoints and `infer --backend vulkan`
//! (`story:m1-vulkan-checkpoint-infer`, DESIGN.md § 6.2, § 8).
//!
//! The cases without `#[ignore]` create no Vulkan instance: every Vulkan run sets
//! `MODELSPEC_VULKAN_LIBRARY` to a path that does not exist. The device cases are ignored; run
//! them with `cargo test -p modelspec-cli --test checkpoint_vulkan_adversary -- --ignored`.
#![cfg(feature = "vulkan")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const LINEAR_REGRESSION: &str = "examples/linear_regression.mspec";
const MNIST_MLP: &str = "examples/mnist_mlp.mspec";
const NO_LOADER: &str = "/nonexistent/modelspec-adversary/libvulkan-missing.so.1";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("checkpoint_vulkan_adversary")
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

fn p(path: &Path) -> &str {
    path.to_str().expect("a UTF-8 path")
}

fn code(stderr: &str) -> &str {
    stderr.split_once(": ").map_or(stderr, |(code, _)| code)
}

/// A CPU checkpoint of `file` after `steps` steps at `dir/<name>`.
fn cpu_checkpoint(file: &str, dir: &Path, name: &str, steps: &str) -> PathBuf {
    let checkpoint = dir.join(name);
    let output = modelspec(
        &[
            "train",
            file,
            "--backend",
            "cpu",
            "--steps",
            steps,
            "--checkpoint",
            p(&checkpoint),
        ],
        Some(NO_LOADER),
    );
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    checkpoint
}

/// The bad checkpoints a reader must refuse, by name.
fn bad_checkpoints(d: &Path) -> Vec<(&'static str, PathBuf)> {
    let good = cpu_checkpoint(LINEAR_REGRESSION, d, "good.msckpt", "5");
    let bytes = std::fs::read(&good).unwrap();
    let truncated = d.join("truncated.msckpt");
    std::fs::write(&truncated, &bytes[..bytes.len() / 2]).unwrap();
    let mut flipped = bytes.clone();
    let last = flipped.len() - 1;
    flipped[last] ^= 0x40;
    let payload = d.join("payload.msckpt");
    std::fs::write(&payload, &flipped).unwrap();
    let empty = d.join("empty.msckpt");
    std::fs::write(&empty, b"").unwrap();
    let other = cpu_checkpoint(MNIST_MLP, d, "other-file.msckpt", "0");
    vec![
        ("missing", d.join("missing.msckpt")),
        ("empty", empty),
        ("truncated", truncated),
        ("payload byte flipped", payload),
        ("checkpoint of another file", other),
    ]
}

/// Every checkpoint the CPU `infer` refuses is refused by `infer --backend vulkan` with the same
/// code, exit 1, before the loader is opened (DESIGN.md § 8: "reads the checkpoint before it
/// opens the loader, so a checkpoint the CPU refuses is refused with the same E-CKPT code").
#[test]
fn infer_on_vulkan_refuses_every_bad_checkpoint_as_the_cpu_does() {
    let d = dir("infer-bad");
    for (what, bad) in bad_checkpoints(&d) {
        let infer = |backend: &str| {
            modelspec(
                &[
                    "infer",
                    LINEAR_REGRESSION,
                    "--backend",
                    backend,
                    "--checkpoint",
                    p(&bad),
                ],
                Some(NO_LOADER),
            )
        };
        let cpu = infer("cpu");
        let gpu = infer("vulkan");
        let (cs, gs) = (text(&cpu.stderr), text(&gpu.stderr));
        assert_eq!(cpu.status.code(), Some(1), "{what}: {cs}");
        assert!(code(&cs).starts_with("E-CKPT-"), "{what}: {cs}");
        assert_eq!(gpu.status.code(), Some(1), "{what}: {gs}");
        assert_eq!(code(&gs), code(&cs), "{what}: vulkan {gs} cpu {cs}");
        assert!(gpu.stdout.is_empty(), "{what}");
    }
}

/// The same for `train --backend vulkan --resume`, which reads its checkpoint before the loader.
#[test]
fn train_on_vulkan_refuses_every_bad_resume_as_the_cpu_does() {
    let d = dir("resume-bad");
    for (what, bad) in bad_checkpoints(&d) {
        let train = |backend: &str| {
            modelspec(
                &[
                    "train",
                    LINEAR_REGRESSION,
                    "--backend",
                    backend,
                    "--steps",
                    "3",
                    "--resume",
                    p(&bad),
                ],
                Some(NO_LOADER),
            )
        };
        let cpu = train("cpu");
        let gpu = train("vulkan");
        let (cs, gs) = (text(&cpu.stderr), text(&gpu.stderr));
        assert_eq!(cpu.status.code(), Some(1), "{what}: {cs}");
        assert!(code(&cs).starts_with("E-CKPT-"), "{what}: {cs}");
        assert_eq!(gpu.status.code(), Some(1), "{what}: {gs}");
        assert_eq!(code(&gs), code(&cs), "{what}: vulkan {gs} cpu {cs}");
        assert!(gpu.stdout.is_empty(), "{what}");
    }
}

/// `--checkpoint-every 0` and `--checkpoint-every` without `--checkpoint` are usage errors on the
/// Vulkan backend as on the CPU, before the loader.
#[test]
fn checkpoint_every_usage_errors_on_vulkan() {
    let d = dir("usage");
    let out = d.join("out.msckpt");
    for args in [
        vec!["--checkpoint", p(&out), "--checkpoint-every", "0"],
        vec!["--checkpoint-every", "5"],
    ] {
        let mut all = vec!["train", LINEAR_REGRESSION, "--backend", "vulkan"];
        all.extend(args.iter().copied());
        let output = modelspec(&all, Some(NO_LOADER));
        assert_eq!(
            output.status.code(),
            Some(2),
            "{args:?}: {}",
            text(&output.stderr)
        );
    }
    let output = modelspec(
        &[
            "infer",
            LINEAR_REGRESSION,
            "--backend",
            "cpu",
            "--device",
            "0",
            "--checkpoint",
            p(&out),
        ],
        Some(NO_LOADER),
    );
    assert_eq!(output.status.code(), Some(2), "{}", text(&output.stderr));
}

/// A checkpoint write that fails part-way leaves the file at `--checkpoint` as it was. The run
/// resumes from and checkpoints to the same file, the natural way to continue a run in place;
/// the file size limit makes the write fail after half the checkpoint's bytes (SIGXFSZ ignored,
/// so `write` returns EFBIG as on a full disk). The shared writer is `checkpoint::write`, which
/// both backends call (`crates/modelspec-cli/src/train.rs`).
#[test]
fn a_failed_checkpoint_write_keeps_the_checkpoint_it_replaces() {
    let d = dir("partial");
    let run = cpu_checkpoint(LINEAR_REGRESSION, &d, "run.msckpt", "5");
    let before = std::fs::read(&run).unwrap();
    let limit = format!("--fsize={}", before.len() / 2);
    let output = Command::new("prlimit")
        .args([
            limit.as_str(),
            "env",
            "--ignore-signal=XFSZ",
            env!("CARGO_BIN_EXE_modelspec"),
            "train",
            LINEAR_REGRESSION,
            "--backend",
            "cpu",
            "--steps",
            "5",
            "--resume",
            p(&run),
            "--checkpoint",
            p(&run),
        ])
        .current_dir(repo_root())
        .output()
        .expect("prlimit runs");
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "the write fails: {stderr}");
    let after = std::fs::read(&run).unwrap();
    assert_eq!(
        after.len(),
        before.len(),
        "the failed write left a {}-byte file where a {}-byte checkpoint was: {stderr}",
        after.len(),
        before.len()
    );
    assert_eq!(after, before);
}

// ---------------------------------------------------------------------------------------------
// Device cases.
// ---------------------------------------------------------------------------------------------

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

fn files(d: &Path, prefix: &str) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(d)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with(prefix))
        .collect();
    names.sort();
    names
}

fn rel_l2(g: &[f32], c: &[f32]) -> f64 {
    let diff: f64 = g
        .iter()
        .zip(c)
        .map(|(g, c)| (f64::from(*g) - f64::from(*c)).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm: f64 = c.iter().map(|c| f64::from(*c).powi(2)).sum::<f64>().sqrt();
    diff / norm
}

/// For each `(--steps, --checkpoint-every, resume step)` the Vulkan run writes the same set of
/// files as the CPU run, each with the same step, schema hash, optimizer, rng and parameter
/// paths, and parameters within the whole-run bound of the CPU's; a Vulkan run resumes from a
/// CPU checkpoint.
#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_checkpoint_files_match_the_cpu_at_every_boundary() {
    use modelspec_runtime::train::checkpoint;
    let cases: [(&str, &str, Option<&str>); 6] = [
        ("10", "3", None),
        ("10", "10", None),
        ("10", "11", None),
        ("10", "1", None),
        ("10", "3", Some("4")),
        ("600", "256", None),
    ];
    for (steps, every, resume) in cases {
        let d = dir(&format!("boundary-{steps}-{every}-{resume:?}"));
        let start = resume.map(|s| cpu_checkpoint(LINEAR_REGRESSION, &d, "start.msckpt", s));
        for backend in ["cpu", "vulkan"] {
            let out = d.join(format!("{backend}.msckpt"));
            let mut args = vec![
                "train",
                LINEAR_REGRESSION,
                "--backend",
                backend,
                "--steps",
                steps,
                "--checkpoint",
                p(&out),
                "--checkpoint-every",
                every,
            ];
            if let Some(start) = &start {
                args.extend(["--resume", p(start)]);
            }
            ok(&args);
        }
        let cpu_files = files(&d, "cpu");
        let gpu_files = files(&d, "vulkan");
        let strip = |v: &[String], prefix: &str| -> Vec<String> {
            v.iter()
                .map(|n| n.trim_start_matches(prefix).to_owned())
                .collect()
        };
        assert_eq!(
            strip(&gpu_files, "vulkan"),
            strip(&cpu_files, "cpu"),
            "steps {steps} every {every} resume {resume:?}"
        );
        for (c, g) in cpu_files.iter().zip(&gpu_files) {
            let c = checkpoint::read(&d.join(c)).unwrap();
            let g = checkpoint::read(&d.join(g)).unwrap();
            assert_eq!(g.step, c.step);
            assert_eq!(g.schema_hash, c.schema_hash);
            assert_eq!(g.optimizer, c.optimizer);
            assert_eq!(g.rng, c.rng);
            assert_eq!(g.parameters.len(), c.parameters.len());
            for (gp, cp) in g.parameters.iter().zip(&c.parameters) {
                assert_eq!(gp.path, cp.path);
                let r = rel_l2(gp.value.data(), cp.value.data());
                assert!(r <= 1e-4, "step {}: {} rel l2 {r:e}", c.step, cp.path);
            }
        }
    }
}

/// `infer --backend vulkan` prints the prediction of `--backend cpu` within the whole-run bound,
/// not only its loss (DESIGN.md § 8: "prints the same two lines, each value within the
/// tolerance of § 9 gate 5 of `--backend cpu`'s").
#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_infer_prediction_matches_the_cpu() {
    let d = dir("infer-prediction");
    let run = d.join("run.msckpt");
    ok(&[
        "train",
        MNIST_MLP,
        "--backend",
        "vulkan",
        "--steps",
        "300",
        "--checkpoint",
        p(&run),
    ]);
    let prediction = |backend: &str| -> (String, Vec<f32>) {
        let out = ok(&[
            "infer",
            MNIST_MLP,
            "--backend",
            backend,
            "--checkpoint",
            p(&run),
        ]);
        let stdout = text(&out.stdout);
        let line = stdout.lines().next().unwrap().to_owned();
        let (head, list) = line.split_once(" = [").unwrap();
        let values = list
            .trim_end_matches(']')
            .split(", ")
            .map(|v| v.parse().unwrap())
            .collect();
        (head.to_owned(), values)
    };
    let (ch, c) = prediction("cpu");
    let (gh, g) = prediction("vulkan");
    assert_eq!(gh, ch);
    assert_eq!(g.len(), c.len());
    let r = rel_l2(&g, &c);
    eprintln!("prediction rel l2 {r:e}, bit-equal {}", g == c);
    assert!(r <= 1e-4, "prediction rel l2 {r:e}");
}

/// `--profile` with `--checkpoint-every` at the chunk boundary prints the same dispatch sites and
/// call counts as without checkpoints.
#[test]
#[ignore = "needs a Vulkan device"]
fn profile_with_checkpoints_at_the_chunk_boundary() {
    let d = dir("profile");
    let out = d.join("run.msckpt");
    let sites = |stderr: &str| -> Vec<String> {
        stderr
            .lines()
            .filter(|l| l.starts_with("profile dispatch "))
            .map(|l| {
                let (head, _) = l.split_once(" calls ").unwrap();
                let calls = l
                    .split(" calls ")
                    .nth(1)
                    .unwrap()
                    .split(' ')
                    .next()
                    .unwrap();
                format!("{head} calls {calls}")
            })
            .collect()
    };
    let with = ok(&[
        "train",
        MNIST_MLP,
        "--backend",
        "vulkan",
        "--steps",
        "600",
        "--profile",
        "--checkpoint",
        p(&out),
        "--checkpoint-every",
        "256",
    ]);
    let without = ok(&[
        "train",
        MNIST_MLP,
        "--backend",
        "vulkan",
        "--steps",
        "600",
        "--profile",
    ]);
    let (w, wo) = (sites(&text(&with.stderr)), sites(&text(&without.stderr)));
    assert!(!wo.is_empty(), "{}", text(&without.stderr));
    assert_eq!(w, wo);
    assert_eq!(
        files(&d, "run"),
        ["run-256.msckpt", "run-512.msckpt", "run.msckpt"]
    );
}
