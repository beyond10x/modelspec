//! `modelspec train --checkpoint`, `--checkpoint-every`, `--resume`, `--steps` and
//! `modelspec infer --backend cpu` on `examples/linear_regression.mspec` (DESIGN.md § 6.2, § 8,
//! § 9 gate 7).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const EXAMPLE: &str = "examples/linear_regression.mspec";
const TWO_LAYER: &str = "crates/modelspec-docs/models/two_layer_relu.mspec";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh directory for one test.
fn dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("checkpoint_cpu")
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

/// The `step` lines of a train stdout from `from` on.
fn steps_from(stdout: &str, from: u64) -> Vec<String> {
    stdout
        .lines()
        .filter(|line| {
            line.strip_prefix("step ")
                .and_then(|rest| rest.split_once(' '))
                .and_then(|(k, _)| k.parse::<u64>().ok())
                .is_some_and(|k| k >= from)
        })
        .map(str::to_owned)
        .collect()
}

#[test]
fn resume_equals_uninterrupted() {
    let d = dir("resume");
    let (a, h, b) = (d.join("a.msckpt"), d.join("h.msckpt"), d.join("b.msckpt"));
    let whole = ok(&[
        "train",
        EXAMPLE,
        "--backend",
        "cpu",
        "--steps",
        "500",
        "--checkpoint",
        path(&a),
    ]);
    let first = ok(&[
        "train",
        EXAMPLE,
        "--backend",
        "cpu",
        "--steps",
        "250",
        "--checkpoint",
        path(&h),
    ]);
    let rest = ok(&[
        "train",
        EXAMPLE,
        "--backend",
        "cpu",
        "--resume",
        path(&h),
        "--steps",
        "250",
        "--checkpoint",
        path(&b),
    ]);
    let a_bytes = std::fs::read(&a).unwrap();
    let b_bytes = std::fs::read(&b).unwrap();
    assert!(a_bytes == b_bytes, "a.msckpt and b.msckpt differ");
    assert_ne!(std::fs::read(&h).unwrap(), a_bytes);
    let expected = steps_from(&whole, 250);
    assert_eq!(expected.len(), 6, "{whole}");
    assert!(expected[0].starts_with("step 250 loss "));
    assert!(expected[5].starts_with("step 500 loss "));
    assert_eq!(steps_from(&rest, 0), expected, "{rest}");
    assert!(first.contains("step 250 loss "), "{first}");
}

#[test]
fn infer_reproduces_the_final_loss_and_checkpoint_every_writes_four_files() {
    let d = dir("infer");
    let a = d.join("a.msckpt");
    let run = ok(&[
        "train",
        EXAMPLE,
        "--backend",
        "cpu",
        "--steps",
        "500",
        "--checkpoint",
        path(&a),
        "--checkpoint-every",
        "100",
    ]);
    let last = run
        .lines()
        .find_map(|line| line.strip_prefix("step 500 loss "))
        .expect("a step 500 line")
        .to_owned();
    let infer = ok(&[
        "infer",
        EXAMPLE,
        "--backend",
        "cpu",
        "--checkpoint",
        path(&a),
    ]);
    let lines: Vec<&str> = infer.lines().collect();
    assert_eq!(lines.len(), 2, "{infer}");
    assert!(
        lines[0].starts_with("prediction: Tensor<f32, [8, 1]> = ["),
        "{infer}"
    );
    assert_eq!(lines[1], format!("loss {last}"));

    let mut names: Vec<String> = std::fs::read_dir(&d)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "a-100.msckpt",
            "a-200.msckpt",
            "a-300.msckpt",
            "a-400.msckpt",
            "a.msckpt"
        ]
    );
    let at_200 = ok(&[
        "train",
        EXAMPLE,
        "--backend",
        "cpu",
        "--steps",
        "200",
        "--checkpoint",
        path(&d.join("x.msckpt")),
    ]);
    assert!(at_200.contains("step 200 loss "));
    assert_eq!(
        std::fs::read(d.join("x.msckpt")).unwrap(),
        std::fs::read(d.join("a-200.msckpt")).unwrap()
    );
}

fn refused(args: &[&str], code: &str) -> String {
    let output = modelspec(args);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{args:?}\nstderr: {stderr}");
    assert!(stderr.starts_with(&format!("{code}: ")), "{stderr}");
    assert!(!text(&output.stdout).contains("step "), "no step ran");
    stderr
}

#[test]
fn resume_into_another_model_names_the_first_differing_parameter_path() {
    let d = dir("other_model");
    let c = d.join("two.msckpt");
    ok(&[
        "train",
        TWO_LAYER,
        "--backend",
        "cpu",
        "--steps",
        "3",
        "--checkpoint",
        path(&c),
    ]);
    let stderr = refused(
        &[
            "train",
            EXAMPLE,
            "--backend",
            "cpu",
            "--resume",
            path(&c),
            "--steps",
            "1",
        ],
        "E-CKPT-003",
    );
    assert!(stderr.contains("TwoLayer.w1"), "{stderr}");
    assert!(stderr.contains("LinearReg.w"), "{stderr}");
    let stderr = refused(
        &[
            "infer",
            EXAMPLE,
            "--backend",
            "cpu",
            "--checkpoint",
            path(&c),
        ],
        "E-CKPT-003",
    );
    assert!(stderr.contains("TwoLayer.w1"), "{stderr}");
}

#[test]
fn resume_with_another_learning_rate_names_the_optimizer() {
    let d = dir("other_lr");
    let c = d.join("lin.msckpt");
    ok(&[
        "train",
        EXAMPLE,
        "--backend",
        "cpu",
        "--steps",
        "3",
        "--checkpoint",
        path(&c),
    ]);
    let source = std::fs::read_to_string(repo_root().join(EXAMPLE))
        .unwrap()
        .replace("lr = 0.1", "lr = 0.05");
    let other = d.join("other.mspec");
    std::fs::write(&other, source).unwrap();
    let stderr = refused(
        &[
            "train",
            path(&other),
            "--backend",
            "cpu",
            "--resume",
            path(&c),
            "--steps",
            "1",
        ],
        "E-CKPT-003",
    );
    assert!(stderr.contains("optimizer"), "{stderr}");
}

#[test]
fn a_file_that_is_not_a_checkpoint_is_refused() {
    let d = dir("not_a_checkpoint");
    let c = d.join("junk.msckpt");
    std::fs::write(
        &c,
        b"not a checkpoint at all, just some bytes that are long enough",
    )
    .unwrap();
    refused(
        &["train", EXAMPLE, "--backend", "cpu", "--resume", path(&c)],
        "E-CKPT-001",
    );
    refused(
        &[
            "infer",
            EXAMPLE,
            "--backend",
            "cpu",
            "--checkpoint",
            path(&d.join("missing.msckpt")),
        ],
        "E-CKPT-001",
    );
}

#[test]
fn infer_on_vulkan_and_checkpoint_every_without_checkpoint_are_refused() {
    let output = modelspec(&[
        "infer",
        EXAMPLE,
        "--backend",
        "vulkan",
        "--checkpoint",
        "x.msckpt",
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(text(&output.stderr).starts_with("E-TARGET-001: "));
    let output = modelspec(&[
        "train",
        EXAMPLE,
        "--backend",
        "cpu",
        "--checkpoint-every",
        "10",
    ]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output.stderr));
    let output = modelspec(&[
        "train",
        EXAMPLE,
        "--backend",
        "cpu",
        "--checkpoint-every",
        "0",
        "--checkpoint",
        "x.msckpt",
    ]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output.stderr));
}
