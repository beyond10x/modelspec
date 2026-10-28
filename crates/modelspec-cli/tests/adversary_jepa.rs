//! Adversary cases for `stop_gradient` and `sigreg` through the `modelspec` binary
//! (story:jepa-stop-gradient, story:jepa-sigreg-cpu; DESIGN.md § 2.2, § 2.3, § 8.1).
//!
//! The checker's codes and spans, programs the checker accepts that a later stage refuses,
//! a JEPA-shaped training file (`mse(predict(encode(x)), stop_gradient(encode(y))) +
//! λ sigreg(encode(x))`) run twice, resumed, and given one more `sigreg` site, and a non-finite
//! `sigreg` loss. Every source is written here, into this test binary's
//! `CARGO_TARGET_TMPDIR/adversary_jepa/`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("adversary_jepa");
    std::fs::create_dir_all(&dir).expect("the scratch directory");
    dir
}

/// Writes `source` as `<name>.mspec` in the scratch directory and returns its path.
fn write(name: &str, source: &str) -> String {
    let path = scratch().join(format!("{name}.mspec"));
    std::fs::write(&path, source).expect("the source is written");
    path.display().to_string()
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .env(
            "MODELSPEC_VULKAN_LIBRARY",
            "/nonexistent/adversary/libvulkan.so.1",
        )
        .output()
        .expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("UTF-8")
}

/// The 1-based line and column of the `nth` (from 0) occurrence of `needle` in `source`.
fn position(source: &str, needle: &str, nth: usize) -> (u32, u32) {
    let offset = source
        .match_indices(needle)
        .nth(nth)
        .unwrap_or_else(|| panic!("no occurrence {nth} of `{needle}`"))
        .0;
    let before = &source[..offset];
    let line = before.matches('\n').count() + 1;
    let column = offset - before.rfind('\n').map_or(0, |i| i + 1) + 1;
    (line as u32, column as u32)
}

/// `check` exits 1 with exactly one diagnostic, `code` at `at`.
fn refused(name: &str, source: &str, code: &str, at: (u32, u32)) {
    let path = write(name, source);
    let output = modelspec(&["check", &path]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{name}:\n{stderr}");
    let lines: Vec<&str> = stderr.lines().collect();
    let heads = lines.iter().filter(|l| !l.starts_with("  ")).count();
    assert_eq!(heads, 1, "{name}: one diagnostic expected:\n{stderr}");
    assert!(
        lines[0].starts_with(&format!("{code}: ")),
        "{name}: expected {code}:\n{stderr}"
    );
    assert_eq!(
        lines.get(1).copied(),
        Some(format!("  {path}:{}:{}", at.0, at.1).as_str()),
        "{name}: location\n{stderr}"
    );
}

/// A file whose forward computes `sigreg(x, <args>)` of a `[16, 4]` input, the call at line 5,
/// column 17, as in the story's own fixtures.
fn sigreg_file(args: &str) -> String {
    format!(
        "model Spread {{
    param w: Tensor<f32, [1, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [16, 4]>) -> Tensor<f32, []> {{
        let s = sigreg(x, {args});
        return add(s, sum(w));
    }}
}}

train Fit {{
    model = Spread;
    input x: Tensor<f32, [16, 4]>;
    target y: Tensor<f32, []>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}}
"
    )
}

// ---------------------------------------------------------------------------------------------
// The checker
// ---------------------------------------------------------------------------------------------

/// Priority 4: `slices` is an integer literal of at least 1 and no more is checked, so
/// `slices = 18446744073709551615` makes the projection `f32[4, 2^64 - 1]`, whose element count
/// does not fit `u64` (DESIGN.md § 2.2: "integer overflows in element/byte counts ... are hard
/// errors"). `check` must refuse it with a type diagnostic; it accepts it.
#[test]
fn adversary_check_sigreg_slices_whose_projection_overflows_is_refused() {
    let source = sigreg_file("slices = 18446744073709551615");
    let path = write("slices_overflow", &source);
    let output = modelspec(&["check", &path]);
    let stderr = text(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(1),
        "check accepts a projection of 4 x (2^64 - 1) elements:\n{stderr}"
    );
    assert!(stderr.starts_with("E-TYPE-"), "{stderr}");
}

/// Priority 4: whatever `check` says, a checked file never reaches the compiler as a compiler bug:
/// `ir --stage tensor` on the overflowing `slices` exits with a user diagnostic, not
/// `compiler bug`.
#[test]
fn adversary_ir_sigreg_slices_whose_projection_overflows_is_no_compiler_bug() {
    let source = sigreg_file("slices = 18446744073709551615");
    let path = write("slices_overflow_ir", &source);
    let output = modelspec(&["ir", &path, "--stage", "tensor"]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(
        !stderr.contains("compiler bug"),
        "a checked file is a compiler bug at lowering:\n{stderr}"
    );
}

/// Priority 4: `slices = 2.5` is not an integer literal: E-TYPE-019 at the argument.
#[test]
fn adversary_check_sigreg_float_slices_is_e_type_019_at_the_argument() {
    refused(
        "slices_float",
        &sigreg_file("slices = 2.5"),
        "E-TYPE-019",
        (5, 27),
    );
}

/// Priority 4: `sigreg(x)` without `slices`: E-TYPE-001 at the call.
#[test]
fn adversary_check_sigreg_without_slices_is_e_type_001_at_the_call() {
    let source = sigreg_file("").replace("sigreg(x, )", "sigreg(x)");
    refused("slices_missing", &source, "E-TYPE-001", (5, 17));
}

/// A file whose only read of `w` is `<expr>` inside `forward`, beside a live read of `v`.
fn stopped_file(expr: &str, loss: &str) -> String {
    format!(
        "model Target {{
    param w: Tensor<f32, [4, 4]> = xavier_uniform();
    param v: Tensor<f32, [4, 4]> = xavier_uniform();

    fn forward(x: Tensor<f32, [6, 4]>) -> Tensor<f32, [6, 4]> {{
        return add(matmul(x, v), {expr});
    }}
}}

train Fit {{
    model = Target;
    input x: Tensor<f32, [6, 4]>;
    target y: Tensor<f32, [6, 4]>;
    loss = {loss};
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}}
"
    )
}

/// Priority 4: `stop_gradient(stop_gradient(matmul(x, w)))`: E-TYPE-026 once, at the outer call
/// (DESIGN.md § 8.1).
#[test]
fn adversary_check_nested_stop_gradient_is_e_type_026_at_the_outer_call() {
    let source = stopped_file(
        "stop_gradient(stop_gradient(matmul(x, w)))",
        "mse(model.forward(x), y)",
    );
    let at = position(&source, "stop_gradient(stop_gradient", 0);
    refused("nested", &source, "E-TYPE-026", at);
}

/// Priority 2 and 4: the whole loss under `stop_gradient`: E-TYPE-026 once for each of the two
/// parameters, both at the loss's `stop_gradient` call.
#[test]
fn adversary_check_the_whole_loss_under_stop_gradient_is_e_type_026() {
    let source = stopped_file("matmul(x, w)", "stop_gradient(mse(model.forward(x), y))");
    let path = write("whole_loss", &source);
    let output = modelspec(&["check", &path]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    let at = position(&source, "stop_gradient(mse", 0);
    let heads: Vec<&str> = stderr.lines().filter(|l| !l.starts_with("  ")).collect();
    assert_eq!(heads.len(), 2, "one E-TYPE-026 per parameter:\n{stderr}");
    assert!(
        heads.iter().all(|h| h.starts_with("E-TYPE-026: ")),
        "{stderr}"
    );
    let location = format!("  {path}:{}:{}", at.0, at.1);
    assert_eq!(
        stderr.lines().filter(|l| *l == location).count(),
        2,
        "both at the loss's call:\n{stderr}"
    );
}

/// Priority 2: `stop_gradient` of an `i32` value is E-TYPE-018 at the call.
#[test]
fn adversary_check_stop_gradient_of_i32_ids_is_e_type_018() {
    let source = "model Lookup {
    param table: Tensor<f32, [6, 3]> = xavier_uniform();

    fn forward(ids: Tensor<i32, [4]>) -> Tensor<f32, [4, 3]> {
        return embedding(table, stop_gradient(ids));
    }
}

train Fit {
    model = Lookup;
    input ids: Tensor<i32, [4]>;
    target y: Tensor<f32, [4, 3]>;
    loss = mse(model.forward(ids), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}
";
    let at = position(source, "stop_gradient(ids)", 0);
    refused("i32_ids", source, "E-TYPE-018", at);
}

// ---------------------------------------------------------------------------------------------
// A JEPA-shaped training file
// ---------------------------------------------------------------------------------------------

/// The decider's shape: an encoder `enc`, a predictor `pred`, the predictor's output fitted to the
/// stopped encoding of the target (`mse(..., stop_gradient(...))`, the stop on `mse`'s second
/// operand), plus `0.1 sigreg` of the encoding, one site in a model function and one in the loss.
/// `extra`, when not empty, is added to the loss.
fn jepa(extra: &str) -> String {
    let base = "add(add(mse(model.forward(x), stop_gradient(model.encode(y))), mul(model.spread(x), 0.1)), mul(sigreg(model.encode(x), slices = 3), 0.1))";
    let loss = if extra.is_empty() {
        base.to_owned()
    } else {
        format!("add({base}, {extra})")
    };
    format!(
        "model Jepa {{
    param enc: Tensor<f32, [8, 8]> = xavier_uniform();
    param pred: Tensor<f32, [8, 8]> = xavier_uniform();

    fn encode(x: Tensor<f32, [32, 8]>) -> Tensor<f32, [32, 8]> {{
        return matmul(x, enc);
    }}

    fn spread(x: Tensor<f32, [32, 8]>) -> Tensor<f32, []> {{
        return sigreg(matmul(x, enc), slices = 4);
    }}

    fn forward(x: Tensor<f32, [32, 8]>) -> Tensor<f32, [32, 8]> {{
        return matmul(matmul(x, enc), pred);
    }}
}}

train Fit {{
    model = Jepa;
    input x: Tensor<f32, [32, 8]>;
    target y: Tensor<f32, [32, 8]>;
    loss = {loss};
    optimizer = sgd(lr = 0.05);
    steps = 6;
    seed = 13;
    device = cpu;
}}
"
    )
}

/// The stdout of a successful `train <source> --backend cpu <more>`.
fn train(name: &str, source: &str, more: &[&str]) -> String {
    let path = write(name, source);
    let mut args = vec!["train", path.as_str(), "--backend", "cpu"];
    args.extend_from_slice(more);
    let output = modelspec(&args);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{name}: {}",
        text(&output.stderr)
    );
    text(&output.stdout)
}

/// The `step <k> loss <v>` lines and the final `param @...` lines of a run.
fn trajectory(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .filter(|l| l.starts_with("step ") || l.starts_with("param @"))
        .map(str::to_owned)
        .collect()
}

/// Priority 3: two runs of the JEPA file from one seed print the same bytes, and `enc` and
/// `pred` both train.
#[test]
fn adversary_jepa_file_trains_reproducibly() {
    let source = jepa("");
    let first = train("jepa_a", &source, &[]);
    let second = train("jepa_b", &source, &[]);
    assert_eq!(first, second, "two runs from one seed");
    let lines = trajectory(&first);
    assert!(
        lines.iter().any(|l| l.starts_with("step 6 loss ")),
        "{first}"
    );
    for name in ["enc", "pred"] {
        let init = first
            .lines()
            .find(|l| l.starts_with(&format!("init @Jepa.{name}: ")))
            .unwrap_or_else(|| panic!("no init line for {name}:\n{first}"));
        let last = first
            .lines()
            .find(|l| l.starts_with(&format!("param @Jepa.{name}: ")))
            .unwrap_or_else(|| panic!("no param line for {name}:\n{first}"));
        assert_ne!(
            init.split_once(" = ").unwrap().1,
            last.split_once(" = ").unwrap().1,
            "{name} trains"
        );
    }
}

/// Priority 3: the projections of a step depend on the step alone, so 3 steps, a checkpoint and
/// 3 resumed steps end where 6 straight steps end, loss line and parameters bit for bit.
#[test]
fn adversary_jepa_resumed_run_equals_the_straight_run() {
    let source = jepa("");
    let straight = train("jepa_straight", &source, &[]);
    let checkpoint = scratch().join("jepa_half.msckpt");
    let checkpoint = checkpoint.to_str().unwrap();
    train(
        "jepa_half",
        &source,
        &["--steps", "3", "--checkpoint", checkpoint],
    );
    let resumed = train(
        "jepa_resumed",
        &source,
        &["--resume", checkpoint, "--steps", "3"],
    );
    let tail = |s: &str| -> Vec<String> {
        let lines = trajectory(s);
        let last = lines.iter().rfind(|l| l.starts_with("step ")).cloned();
        let params = lines.into_iter().filter(|l| l.starts_with("param @"));
        last.into_iter().chain(params).collect()
    };
    assert_eq!(
        tail(&resumed),
        tail(&straight),
        "{resumed}\n---\n{straight}"
    );
}

/// Priority 3: two files that differ only in one more loss `sigreg` site, after the others and
/// weighted by `0.0`, inside a term both add (one more `spread` inlining in each): the new site
/// changes no other site's projection, so every loss and parameter keeps its bits.
#[test]
fn adversary_jepa_one_more_site_changes_no_other_projection() {
    let base = train("jepa_base", &jepa("mul(model.spread(x), 0.0)"), &[]);
    let more = train(
        "jepa_more",
        &jepa("mul(add(model.spread(x), sigreg(model.encode(x), slices = 5)), 0.0)"),
        &[],
    );
    assert_eq!(trajectory(&more), trajectory(&base), "{more}\n---\n{base}");
}

/// Priority 1: a `sigreg` operand that overflows `f32` gives a non-finite statistic, reported as
/// E-TRAIN-001 with exit 1, not a finite loss.
#[test]
fn adversary_sigreg_of_an_overflowing_operand_is_e_train_001() {
    let source = jepa("").replace(
        "mul(sigreg(model.encode(x), slices = 3), 0.1)",
        "mul(sigreg(mul(model.encode(x), 1e38), slices = 3), 0.1)",
    );
    let path = write("jepa_overflow", &source);
    let output = modelspec(&["train", &path, "--backend", "cpu"]);
    let stderr = text(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(1),
        "{}\n{stderr}",
        text(&output.stdout)
    );
    assert!(stderr.contains("E-TRAIN-001"), "{stderr}");
}
