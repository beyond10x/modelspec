//! Adversary pass 1 of story:bert-mlm-cross-entropy: `modelspec check` and `ir --stage training`
//! on `cross_entropy` calls the unit's own fixtures do not cover (DESIGN.md § 2.2, § 4).
//!
//! Each case writes its source under this crate's `CARGO_TARGET_TMPDIR`. A case that fails on the
//! unit as written is `#[ignore]`d with its finding id in its name.

use std::path::PathBuf;
use std::process::Command;

const LABELS: &str = "Tensor<i32, [6]>";
const WEIGHTS: &str = "Tensor<f32, [6]>";

/// A tied-decoder model with a rank-1 parameter `p`, the model functions `functions`, and a train
/// block declaring `labels` and `weights` of the types given, with the loss `loss` on line 15.
fn source(functions: &str, labels: &str, weights: &str, loss: &str) -> String {
    format!(
        "model Mlm {{
    param table: Tensor<f32, [5, 4]> = xavier_uniform();
    param p: Tensor<f32, [6]> = ones();
    {functions}
    fn forward(ids: Tensor<i32, [6]>) -> Tensor<f32, [6, 5]> {{
        return matmul(embedding(table, ids), transpose(table));
    }}
}}

train Fit {{
    model = Mlm;
    input ids: Tensor<i32, [6]>;
    target labels: {labels};
    target weights: {weights};
    loss = {loss};
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 7;
    device = cpu;
}}
"
    )
}

/// Writes `text` as `<name>.mspec` and runs `modelspec <args> <file>`; returns the exit code,
/// stdout, stderr with the path replaced by `F`, and the text.
fn run(name: &str, text: &str, args: &[&str]) -> (Option<i32>, String, String) {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("adversary_cross_entropy_cli");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.mspec"));
    std::fs::write(&path, text).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .arg(&path)
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

/// The heads of the diagnostics in `stderr`, each with its location line.
fn heads(stderr: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = stderr.lines().collect();
    lines
        .iter()
        .enumerate()
        .filter(|(_, l)| !l.starts_with("  "))
        .map(|(i, l)| {
            (
                l.to_string(),
                lines
                    .get(i + 1)
                    .map(|s| s.trim().to_owned())
                    .unwrap_or_default(),
            )
        })
        .collect()
}

fn check(
    name: &str,
    functions: &str,
    labels: &str,
    weights: &str,
    loss: &str,
) -> (Option<i32>, String) {
    let (code, stdout, stderr) = run(name, &source(functions, labels, weights, loss), &["check"]);
    assert!(stdout.is_empty(), "{name}: {stdout}");
    (code, stderr)
}

/// H5: a named argument to `cross_entropy`, after or before an operand, is E-TYPE-002 at the
/// name, first.
#[test]
fn adv_named_argument_is_e_type_002_first() {
    for (name, loss, column) in [
        (
            "named_after",
            "cross_entropy(model.forward(ids), labels, weights, heads = 1)",
            63,
        ),
        (
            "named_before",
            "cross_entropy(model.forward(ids), labels, heads = 1, weights)",
            54,
        ),
    ] {
        let (code, stderr) = check(name, "", LABELS, WEIGHTS, loss);
        assert_eq!(code, Some(1), "{stderr}");
        let heads = heads(&stderr);
        assert!(heads[0].0.starts_with("E-TYPE-002: "), "{stderr}");
        assert_eq!(heads[0].1, format!("F:15:{column}"), "{stderr}");
    }
}

/// H5: labels of rank 2 with logits of rank 2 are E-TYPE-004 at the call, naming `labels`.
#[test]
fn adv_labels_of_rank_2_are_e_type_004_naming_labels() {
    let loss = "cross_entropy(model.forward(ids), labels, weights)";
    let (code, stderr) = check("labels_rank2", "", "Tensor<i32, [6, 1]>", WEIGHTS, loss);
    assert_eq!(code, Some(1));
    let heads = heads(&stderr);
    assert_eq!(heads.len(), 1, "{stderr}");
    assert!(heads[0].0.starts_with("E-TYPE-004: "), "{stderr}");
    assert!(
        heads[0].0.contains("`labels` has type Tensor<i32, [6, 1]>"),
        "{stderr}"
    );
    assert_eq!(heads[0].1, "F:15:12");
}

/// H5: `i32` weights are E-TYPE-018 for operand 3; a rank-0 weight is E-TYPE-004 naming
/// `weights`; two arguments are E-TYPE-001.
#[test]
fn adv_i32_weights_scalar_weights_and_two_arguments() {
    let loss = "cross_entropy(model.forward(ids), labels, weights)";
    let (code, stderr) = check("weights_i32", "", LABELS, "Tensor<i32, [6]>", loss);
    assert_eq!(code, Some(1));
    assert!(
        stderr.starts_with("E-TYPE-018: `cross_entropy` operand 3 must be f32"),
        "{stderr}"
    );
    let (_, stderr) = check("weights_rank0", "", LABELS, "Tensor<f32, []>", loss);
    assert!(stderr.starts_with("E-TYPE-004: "), "{stderr}");
    assert!(
        stderr.contains("`weights` has type Tensor<f32, []>"),
        "{stderr}"
    );
    let (_, stderr) = check(
        "two_arguments",
        "",
        LABELS,
        WEIGHTS,
        "cross_entropy(model.forward(ids), labels)",
    );
    assert!(
        stderr.starts_with("E-TYPE-001: `cross_entropy` takes 3 argument(s); got 2"),
        "{stderr}"
    );
}

/// H5: labels also read as an `f32` operand elsewhere (the third use) stay E-TYPE-018, at that
/// operator.
#[test]
fn adv_labels_read_as_f32_elsewhere_are_e_type_018_at_that_operator() {
    let loss = "add(cross_entropy(model.forward(ids), labels, weights), sum(labels))";
    let (code, stderr) = check("labels_third_use", "", LABELS, WEIGHTS, loss);
    assert_eq!(code, Some(1));
    let heads = heads(&stderr);
    assert_eq!(heads.len(), 1, "{stderr}");
    assert!(
        heads[0]
            .0
            .starts_with("E-TYPE-018: `sum` operand 1 must be f32"),
        "{stderr}"
    );
    assert_eq!(heads[0].1, "F:15:68");
}

/// H5: weights that read a parameter through a `let` inside a model function are E-TYPE-021 at
/// the `cross_entropy` argument inside that function, naming `weights`.
#[test]
fn adv_weights_reading_a_parameter_through_a_let_inside_a_function_are_e_type_021() {
    let head = "fn head(z: Tensor<f32, [6, 5]>, l: Tensor<i32, [6]>, w: Tensor<f32, [6]>) -> Tensor<f32, []> { let s = mul(w, p); return cross_entropy(z, l, s); }";
    let loss = "model.head(model.forward(ids), labels, weights)";
    let (code, stderr) = check("weights_let_param", head, LABELS, WEIGHTS, loss);
    assert_eq!(code, Some(1));
    let heads = heads(&stderr);
    assert_eq!(heads.len(), 1, "{stderr}");
    assert!(
        heads[0]
            .0
            .starts_with("E-TYPE-021: `weights` of `cross_entropy`"),
        "{stderr}"
    );
    let column = 4 + head.find("l, s)").unwrap() + 3 + 1;
    assert_eq!(heads[0].1, format!("F:4:{column}"), "{stderr}");
}

/// H5: labels through a `let` and a model function argument check and lower; weights through a
/// parameter-free `relu` take no gradient (no `ReluGrad` in the training program).
#[test]
fn adv_labels_through_a_function_and_weights_through_relu_lower() {
    let head = "fn head(z: Tensor<f32, [6, 5]>, l: Tensor<i32, [6]>, w: Tensor<f32, [6]>) -> Tensor<f32, []> { let m = l; return cross_entropy(z, m, w); }";
    let text = source(
        head,
        LABELS,
        WEIGHTS,
        "model.head(model.forward(ids), labels, weights)",
    );
    let (code, stdout, stderr) = run("labels_through_fn", &text, &["ir", "--stage", "training"]);
    assert_eq!(code, Some(0), "{stderr}");
    assert!(stdout.contains("= CrossEntropyGrad %"), "{stdout}");
    assert!(stdout.contains(" = Add %"), "{stdout}");
    let text = source(
        "",
        LABELS,
        WEIGHTS,
        "cross_entropy(model.forward(ids), labels, relu(weights))",
    );
    let (code, stdout, stderr) = run("weights_relu", &text, &["ir", "--stage", "training"]);
    assert_eq!(code, Some(0), "{stderr}");
    assert!(!stdout.contains("ReluGrad"), "{stdout}");
    assert!(stdout.contains("= CrossEntropyGrad %"), "{stdout}");
}

/// N1: logits `[5, 6]` (the transposed decoder output) against labels and weights of length 6:
/// the two per-row operands agree and the logits have 5 rows, so the diagnostic names `logits`,
/// not `labels`.
#[test]
fn adv_n1_wrong_row_count_of_logits_is_not_blamed_on_labels() {
    let loss = "cross_entropy(transpose(model.forward(ids)), labels, weights)";
    let (code, stderr) = check("logits_transposed", "", LABELS, WEIGHTS, loss);
    assert_eq!(code, Some(1));
    assert!(stderr.starts_with("E-TYPE-003: "), "{stderr}");
    assert!(!stderr.contains("`labels` has type"), "{stderr}");
}
