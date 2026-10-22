//! Adversary cases for `train --checkpoint`, `--checkpoint-every`, `--resume`, `--steps` and
//! `infer --backend cpu` (DESIGN.md § 6.2, § 8, § 9 gate 7): the edges of `--checkpoint-every`,
//! `--resume` with `--steps 0`, the step lines after a resume at a step that is not a multiple of
//! 50, chained resumes, and the bytes of two runs.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const EXAMPLE: &str = "examples/linear_regression.mspec";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary_checkpoint_cpu")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    dir
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("output is UTF-8")
}

fn ok(args: &[&str]) -> String {
    let output = modelspec(args);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{args:?}\nstderr: {}",
        text(&output.stderr)
    );
    text(&output.stdout)
}

fn path(p: &Path) -> &str {
    p.to_str().expect("a UTF-8 path")
}

fn train(extra: &[&str]) -> String {
    let mut args = vec!["train", EXAMPLE, "--backend", "cpu"];
    args.extend_from_slice(extra);
    ok(&args)
}

fn step_lines(stdout: &str) -> Vec<(u64, String)> {
    stdout
        .lines()
        .filter_map(|line| {
            let k = line
                .strip_prefix("step ")?
                .split_once(' ')?
                .0
                .parse::<u64>()
                .ok()?;
            Some((k, line.to_owned()))
        })
        .collect()
}

fn names(d: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(d)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

#[test]
fn resume_with_steps_zero_writes_the_checkpoint_it_read() {
    let d = dir("steps_zero");
    let (h, b) = (d.join("h.msckpt"), d.join("b.msckpt"));
    let first = train(&["--steps", "250", "--checkpoint", path(&h)]);
    let rest = train(&[
        "--resume",
        path(&h),
        "--steps",
        "0",
        "--checkpoint",
        path(&b),
    ]);
    assert_eq!(std::fs::read(&h).unwrap(), std::fs::read(&b).unwrap());
    let last = step_lines(&first).pop().unwrap();
    assert_eq!(last.0, 250);
    assert_eq!(step_lines(&rest), vec![last], "{rest}");
}

#[test]
fn checkpoint_every_at_or_past_the_end_writes_no_extra_file() {
    for every in ["500", "501", "100000"] {
        let d = dir(&format!("every_{every}"));
        train(&[
            "--steps",
            "500",
            "--checkpoint",
            path(&d.join("a.msckpt")),
            "--checkpoint-every",
            every,
        ]);
        assert_eq!(names(&d), ["a.msckpt"], "--checkpoint-every {every}");
    }
    let d = dir("every_1");
    train(&[
        "--steps",
        "3",
        "--checkpoint",
        path(&d.join("a.msckpt")),
        "--checkpoint-every",
        "1",
    ]);
    assert_eq!(names(&d), ["a-1.msckpt", "a-2.msckpt", "a.msckpt"]);
}

#[test]
fn a_resumed_run_writes_the_same_periodic_files_after_its_start() {
    let whole_dir = dir("periodic_whole");
    let rest_dir = dir("periodic_rest");
    let half = dir("periodic_half").join("h.msckpt");
    train(&[
        "--steps",
        "500",
        "--checkpoint",
        path(&whole_dir.join("a.msckpt")),
        "--checkpoint-every",
        "100",
    ]);
    train(&["--steps", "250", "--checkpoint", path(&half)]);
    train(&[
        "--resume",
        path(&half),
        "--steps",
        "250",
        "--checkpoint",
        path(&rest_dir.join("a.msckpt")),
        "--checkpoint-every",
        "100",
    ]);
    assert_eq!(
        names(&rest_dir),
        ["a-300.msckpt", "a-400.msckpt", "a.msckpt"]
    );
    for name in names(&rest_dir) {
        assert_eq!(
            std::fs::read(whole_dir.join(&name)).unwrap(),
            std::fs::read(rest_dir.join(&name)).unwrap(),
            "{name}"
        );
    }
}

#[test]
fn a_resume_at_step_120_prints_its_start_and_then_the_uninterrupted_lines() {
    let d = dir("resume_120");
    let (h, a, b) = (d.join("h.msckpt"), d.join("a.msckpt"), d.join("b.msckpt"));
    let whole = train(&["--steps", "500", "--checkpoint", path(&a)]);
    train(&["--steps", "120", "--checkpoint", path(&h)]);
    let rest = train(&[
        "--resume",
        path(&h),
        "--steps",
        "380",
        "--checkpoint",
        path(&b),
    ]);
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
    let rest = step_lines(&rest);
    assert_eq!(rest[0].0, 120, "the first line is the start step");
    let after: Vec<_> = step_lines(&whole)
        .into_iter()
        .filter(|(k, _)| *k > 120)
        .collect();
    assert_eq!(rest[1..], after[..]);
}

#[test]
fn three_chained_segments_equal_one_run() {
    let d = dir("chained");
    let a = d.join("a.msckpt");
    train(&["--steps", "500", "--checkpoint", path(&a)]);
    let (s1, s2, s3) = (
        d.join("s1.msckpt"),
        d.join("s2.msckpt"),
        d.join("s3.msckpt"),
    );
    train(&["--steps", "1", "--checkpoint", path(&s1)]);
    train(&[
        "--resume",
        path(&s1),
        "--steps",
        "298",
        "--checkpoint",
        path(&s2),
    ]);
    train(&[
        "--resume",
        path(&s2),
        "--steps",
        "201",
        "--checkpoint",
        path(&s3),
    ]);
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&s3).unwrap());
}

#[test]
fn two_runs_write_the_same_bytes() {
    let d = dir("twice");
    let (x, y) = (d.join("x.msckpt"), d.join("y.msckpt"));
    let one = train(&["--steps", "37", "--checkpoint", path(&x)]);
    let two = train(&["--steps", "37", "--checkpoint", path(&y)]);
    assert_eq!(one, two);
    assert_eq!(std::fs::read(&x).unwrap(), std::fs::read(&y).unwrap());
}

#[test]
fn infer_on_a_periodic_file_prints_that_steps_loss() {
    let d = dir("infer_periodic");
    let run = train(&[
        "--steps",
        "500",
        "--checkpoint",
        path(&d.join("a.msckpt")),
        "--checkpoint-every",
        "100",
    ]);
    let at_100 = step_lines(&run)
        .into_iter()
        .find(|(k, _)| *k == 100)
        .unwrap()
        .1;
    let infer = ok(&[
        "infer",
        EXAMPLE,
        "--backend",
        "cpu",
        "--checkpoint",
        path(&d.join("a-100.msckpt")),
    ]);
    assert_eq!(
        infer.lines().nth(1).unwrap(),
        at_100.replacen("step 100 loss", "loss", 1)
    );
}

#[test]
fn resume_with_init_is_a_usage_error() {
    let output = modelspec(&[
        "train",
        EXAMPLE,
        "--backend",
        "cpu",
        "--resume",
        "x.msckpt",
        "--init",
        "some-dir",
    ]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output.stderr));
}
