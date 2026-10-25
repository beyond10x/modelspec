//! Adversary pass 1 of story:bert-dropout: `modelspec check`, `ir --stage tensor` and
//! `train --backend cpu` with `--checkpoint` and `--resume` on `dropout` (DESIGN.md § 2.3, § 8,
//! coordinator decision D47).
//!
//! Each case writes its source under this crate's `CARGO_TARGET_TMPDIR`. A case that failed on
//! the unit as written carries its finding id in its name and passes after the fix.

use std::path::PathBuf;
use std::process::Command;

fn dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("adversary_dropout_cli");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Writes `text` as `<name>.mspec` and runs `modelspec <command> <file> <more>`; returns the exit
/// code, stdout and stderr.
fn run(name: &str, text: &str, command: &str, more: &[&str]) -> (Option<i32>, String, String) {
    let path = dir().join(format!("{name}.mspec"));
    std::fs::write(&path, text).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .arg(command)
        .arg(&path)
        .args(more)
        .output()
        .expect("modelspec runs");
    let shown = path.display().to_string();
    let text = |b: &[u8]| String::from_utf8_lossy(b).replace(&shown, "F");
    (
        output.status.code(),
        text(&output.stdout),
        text(&output.stderr),
    )
}

/// A model whose `forward` holds `body` as its return expression over `x: [8, 4]` and `w`.
fn model(functions: &str, body: &str, steps: u64) -> String {
    model_with_loss(functions, body, "mse(model.forward(x), y)", steps)
}

/// [`model`] with the train block's loss `loss`.
fn model_with_loss(functions: &str, body: &str, loss: &str, steps: u64) -> String {
    format!(
        "model Reg {{
    param w: Tensor<f32, [4, 3]> = xavier_uniform();
    {functions}
    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {{
        return {body};
    }}
}}

train Fit {{
    model = Reg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 3]>;
    loss = {loss};
    optimizer = sgd(lr = 0.05);
    steps = {steps};
    seed = 5;
    device = cpu;
}}
"
    )
}

const BLOCK: &str = "fn block(h: Tensor<f32, [8, 3]>) -> Tensor<f32, [8, 3]> {
        return dropout(h, p = 0.5);
    }
";

/// The `site = ...` of every `Dropout` line of `ir --stage tensor`.
fn dropout_sites(ir: &str) -> Vec<String> {
    ir.lines()
        .filter(|l| l.contains("= Dropout "))
        .map(|l| {
            let site = l.split_once("site = ").expect("a site").1;
            site.split_once(']').expect("`]`").0.to_owned()
        })
        .collect()
}

/// ADV-DROP-1. `ir --stage tensor` of the loss `mse(block(block(forward(x))), y)` prints two
/// `Dropout` nodes on two sites, one per call, so the two applications draw two masks: the
/// first inlining is `Reg.block.dropout#0`, the second `Reg.block@1.dropout#0` (coordinator
/// decision D54, 2026-10-25). (A model function cannot call another, E-NAME-007, so the loss is
/// where a function is inlined twice.)
#[test]
fn adv_drop_1_ir_prints_one_site_per_inlined_call() {
    let text = model_with_loss(
        BLOCK,
        "matmul(x, w)",
        "mse(model.block(model.block(model.forward(x))), y)",
        2,
    );
    let (code, ir, stderr) = run("inlined_twice", &text, "ir", &["--stage", "tensor"]);
    assert_eq!(code, Some(0), "{stderr}");
    let sites = dropout_sites(&ir);
    assert_eq!(sites.len(), 2, "{ir}");
    assert_ne!(sites[0], sites[1], "both calls on one site:\n{ir}");
    assert_eq!(
        sites,
        ["Reg.block.dropout#0", "Reg.block@1.dropout#0"],
        "{ir}"
    );
}

/// The first line of stderr of a refused `check`, or `ok` for exit 0.
fn verdict(name: &str, body: &str) -> String {
    let text = model("", body, 2);
    let (code, _, stderr) = run(name, &text, "check", &[]);
    match code {
        Some(0) => "ok".to_owned(),
        _ => stderr
            .lines()
            .next()
            .unwrap_or("")
            .split(':')
            .next()
            .unwrap_or("")
            .to_owned(),
    }
}

/// Calls the unit's `check_dropout.rs` does not try, each with the code it must give: a
/// positional `p` and an operand after `p` are argument errors, an unknown named argument is
/// E-TYPE-002, a literal that rounds to `0.0` in `f32` is the identity, one that rounds to
/// infinity is outside `[0, 1)`.
#[test]
fn adv_drop_checker_codes_of_calls_the_unit_does_not_try() {
    let w = "matmul(D, w)";
    let cases = [
        ("positional", "dropout(x, 0.3)", "E-TYPE-001"),
        ("named_first", "dropout(p = 0.3, x)", "E-TYPE-002"),
        ("unknown_named", "dropout(x, p = 0.3, q = 1)", "E-TYPE-002"),
        ("tiny", "dropout(x, p = 1e-400)", "ok"),
        ("negative_zero", "dropout(x, p = -0.0)", "ok"),
        ("huge", "dropout(x, p = 1e39)", "E-TYPE-022"),
        ("largest_below_1", "dropout(x, p = 0.99999994)", "ok"),
    ];
    let mut got = Vec::new();
    for (name, call, _) in &cases {
        got.push(verdict(name, &w.replace('D', call)));
    }
    let want: Vec<&str> = cases.iter().map(|c| c.2).collect();
    assert_eq!(got, want);
}

/// The `param ...` lines and the last `step <k> loss <v>` line of a `train` stdout.
fn final_state(stdout: &str) -> (Vec<&str>, &str) {
    let params = stdout
        .lines()
        .filter(|l| l.starts_with("param @"))
        .collect();
    let last = stdout
        .lines()
        .rfind(|l| l.starts_with("step "))
        .expect("a loss line");
    (params, last)
}

/// A run of 12 steps equals a run of 5 steps checkpointed and resumed for 7 more: the resumed
/// steps draw the masks of their global steps 5 to 11, so the final parameters and the final loss
/// line agree bit for bit.
#[test]
fn adv_drop_resume_draws_the_masks_of_the_global_steps() {
    let body = "matmul(dropout(x, p = 0.4), w)";
    let text = model("", body, 12);
    let (code, whole, stderr) = run("resume_whole", &text, "train", &["--backend", "cpu"]);
    assert_eq!(code, Some(0), "{stderr}");

    let checkpoint = dir().join("resume_first.msckpt");
    let _ = std::fs::remove_file(&checkpoint);
    let checkpoint = checkpoint.display().to_string();
    let (code, _, stderr) = run(
        "resume_first",
        &text,
        "train",
        &[
            "--backend",
            "cpu",
            "--steps",
            "5",
            "--checkpoint",
            &checkpoint,
        ],
    );
    assert_eq!(code, Some(0), "{stderr}");
    let (code, resumed, stderr) = run(
        "resume_first",
        &text,
        "train",
        &["--backend", "cpu", "--steps", "7", "--resume", &checkpoint],
    );
    assert_eq!(code, Some(0), "{stderr}");

    let (whole_params, whole_last) = final_state(&whole);
    let (resumed_params, resumed_last) = final_state(&resumed);
    assert!(whole_last.starts_with("step 12 "), "{whole}");
    assert_eq!(whole_last, resumed_last, "{whole}\n{resumed}");
    assert_eq!(whole_params, resumed_params, "{whole}\n{resumed}");
}
