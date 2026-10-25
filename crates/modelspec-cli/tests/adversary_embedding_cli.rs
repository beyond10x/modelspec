//! Adversary cases for story:bert-embedding-lookup through the `modelspec` binary: every other
//! use of `i32` is refused with a code, the shapes `embedding` refuses, `E-DATA-004` and its
//! reach, and what a user sees when a command needs the step's schedule.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// The scratch directory of this test binary, under the build's own target directory.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary_embedding_cli")
        .join(name);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// Writes `source` as `<name>.mspec` in this test's scratch directory and returns its path.
fn write(name: &str, source: &str) -> PathBuf {
    let path = scratch(name).join(format!("{name}.mspec"));
    std::fs::write(&path, source).expect("fixture written");
    path
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .output()
        .expect("modelspec runs")
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr is UTF-8")
}

/// The diagnostic heads of `text`: the lines that are not indented.
fn heads(text: &str) -> Vec<&str> {
    text.lines().filter(|l| !l.starts_with("  ")).collect()
}

/// `modelspec check` on `source` exits 1 with exactly one diagnostic, of `code`, at
/// `line:column` when given.
fn refused(name: &str, source: &str, code: &str, at: Option<(u32, u32)>) {
    let path = write(name, source);
    let output = modelspec(&["check", path.to_str().unwrap()]);
    let text = stderr(&output);
    assert_eq!(output.status.code(), Some(1), "{name}:\n{text}");
    let heads = heads(&text);
    assert_eq!(heads.len(), 1, "{name}: one diagnostic expected:\n{text}");
    assert!(
        heads[0].starts_with(&format!("{code}: ")),
        "{name}: expected {code}:\n{text}"
    );
    assert!(!text.contains("E-INTERNAL"), "{name}:\n{text}");
    if let Some((line, column)) = at {
        let location = text.lines().nth(1).unwrap_or_default();
        assert!(
            location.ends_with(&format!(".mspec:{line}:{column}")),
            "{name}: location {location:?}\n{text}"
        );
    }
}

/// A model over `ids: i32[4, 4]` and `x: f32[4, 4]` whose `forward` returns `body` (line 4,
/// column 16), and a train block that feeds both.
fn operand_model(body: &str) -> String {
    format!(
        "model M {{
    param w: Tensor<f32, [4, 4]> = xavier_uniform();
    fn forward(ids: Tensor<i32, [4, 4]>, x: Tensor<f32, [4, 4]>) -> Tensor<f32, [4, 4]> {{
        return {body};
    }}
}}

train T {{
    model = M;
    input ids: Tensor<i32, [4, 4]>;
    input x: Tensor<f32, [4, 4]>;
    target y: Tensor<f32, [4, 4]>;
    loss = mse(model.forward(ids, x), y);
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 0;
    device = cpu;
}}
"
    )
}

/// Attack 1: an `i32` in each operand position of every operator in `OPERATORS` other than the
/// ids of `embedding` is E-TYPE-018 at the call, and only that.
#[test]
fn adversary_embedding_i32_in_every_other_operand_position_is_e_type_018() {
    for (name, body) in [
        ("matmul_lhs", "matmul(ids, x)"),
        ("matmul_rhs", "matmul(x, ids)"),
        ("add_lhs", "add(ids, x)"),
        ("add_rhs", "add(x, ids)"),
        ("add_scalar", "add(ids, 1.0)"),
        ("sub_lhs", "sub(ids, x)"),
        ("sub_rhs", "sub(x, ids)"),
        ("mul_both", "mul(ids, ids)"),
        ("relu", "relu(ids)"),
        ("transpose", "transpose(ids)"),
        ("sum", "add(x, sum(ids))"),
        ("mean", "add(x, mean(ids))"),
        ("mse_lhs", "add(x, mse(ids, x))"),
        ("mse_rhs", "add(x, mse(x, ids))"),
        ("embedding_table", "embedding(ids, ids)"),
        ("nested", "relu(add(x, matmul(w, ids)))"),
    ] {
        let column = 16 + body.find(name_of(name)).unwrap() as u32;
        refused(
            &format!("op_{name}"),
            &operand_model(body),
            "E-TYPE-018",
            Some((4, column)),
        );
    }
}

/// The operator a case of the matrix above calls with the `i32`.
fn name_of(case: &str) -> &'static str {
    match case {
        "matmul_lhs" | "matmul_rhs" => "matmul",
        "nested" => "matmul",
        "add_lhs" | "add_rhs" | "add_scalar" => "add",
        "sub_lhs" | "sub_rhs" => "sub",
        "mul_both" => "mul",
        "relu" => "relu",
        "transpose" => "transpose",
        "sum" => "sum",
        "mean" => "mean",
        "mse_lhs" | "mse_rhs" => "mse",
        "embedding_table" => "embedding",
        other => panic!("{other}"),
    }
}

/// Attack 1: a float literal as the ids is E-TYPE-018 (an `f32` where `i32` is wanted), an
/// integer literal is E-TYPE-005, and a missing operand is E-TYPE-001.
#[test]
fn adversary_embedding_ids_that_are_not_an_i32_tensor_are_refused() {
    for (name, body, code) in [
        ("float_ids", "embedding(w, 1.0)", "E-TYPE-018"),
        ("f32_ids", "embedding(w, x)", "E-TYPE-018"),
        ("int_ids", "embedding(w, 3)", "E-TYPE-005"),
        ("one_operand", "embedding(w)", "E-TYPE-001"),
    ] {
        refused(name, &operand_model(body), code, None);
    }
}

/// A model over `ids: <ids>` and a table `<table>`, `forward` returning `embedding(table, ids)`
/// declared `-> <result>`.
fn lookup_model(table: &str, ids: &str, result: &str) -> String {
    format!(
        "model M {{
    param table: Tensor<f32, {table}> = xavier_uniform();
    fn forward(ids: Tensor<i32, {ids}>) -> Tensor<f32, {result}> {{
        return embedding(table, ids);
    }}
}}

train T {{
    model = M;
    input ids: Tensor<i32, {ids}>;
    target y: Tensor<f32, {result}>;
    loss = mse(model.forward(ids), y);
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 0;
    device = cpu;
}}
"
    )
}

/// A model whose table is a function argument of type `<table>`, so the parameter rank rule
/// does not see it first.
fn table_argument_model(table: &str) -> String {
    format!(
        "model M {{
    param w: Tensor<f32, [4, 1]> = xavier_uniform();
    fn forward(table: Tensor<f32, {table}>, ids: Tensor<i32, [4]>) -> Tensor<f32, [4, 3]> {{
        return embedding(table, ids);
    }}
}}

train T {{
    model = M;
    input table: Tensor<f32, {table}>;
    input ids: Tensor<i32, [4]>;
    target y: Tensor<f32, [4, 3]>;
    loss = mse(model.forward(table, ids), y);
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 0;
    device = cpu;
}}
"
    )
}

/// Attack 2: ids of rank 0 or 2 and a table of rank 1 or 3 are E-TYPE-004 at the call; `N = 0`
/// is E-TYPE-015 at the declaration, never a panic.
#[test]
fn adversary_embedding_misshapen_lookups_are_refused_with_codes() {
    refused(
        "ids_rank_0",
        &lookup_model("[6, 3]", "[]", "[1, 3]"),
        "E-TYPE-004",
        Some((4, 16)),
    );
    refused(
        "ids_rank_2",
        &lookup_model("[6, 3]", "[2, 4]", "[2, 4, 3]"),
        "E-TYPE-004",
        Some((4, 16)),
    );
    refused(
        "table_rank_1",
        &table_argument_model("[6]"),
        "E-TYPE-004",
        Some((4, 16)),
    );
    refused(
        "table_rank_3",
        &table_argument_model("[6, 3, 2]"),
        "E-TYPE-004",
        Some((4, 16)),
    );
    let path = write("ids_zero", &lookup_model("[6, 3]", "[0]", "[0, 3]"));
    let output = modelspec(&["check", path.to_str().unwrap()]);
    let text = stderr(&output);
    assert_eq!(output.status.code(), Some(1), "{text}");
    assert!(
        heads(&text).iter().all(|h| h.starts_with("E-TYPE-015: ")),
        "{text}"
    );
}

/// Attack 1: an `i32` returned through a `let` from a function declared `f32` is E-TYPE-007;
/// an `i32` train input passed to an `f32` function argument, and an `f32` one to an `i32`
/// argument, are E-TYPE-008; an `i32` target read by `mse` is E-TYPE-018 at the loss.
#[test]
fn adversary_embedding_i32_through_let_arguments_and_targets_is_refused() {
    refused(
        "let_return",
        "model M {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();
    fn forward(ids: Tensor<i32, [4]>) -> Tensor<f32, [4]> {
        let k = ids;
        return k;
    }
}

train T {
    model = M;
    input ids: Tensor<i32, [4]>;
    target y: Tensor<f32, [4]>;
    loss = mse(model.forward(ids), y);
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 0;
    device = cpu;
}
",
        "E-TYPE-007",
        Some((5, 16)),
    );
    let mut source = lookup_model("[6, 3]", "[4]", "[4, 3]");
    source = source.replace("input ids: Tensor<i32, [4]>", "input ids: Tensor<f32, [4]>");
    refused("f32_into_i32_argument", &source, "E-TYPE-008", None);
    refused(
        "i32_into_f32_argument",
        "model M {
    param w: Tensor<f32, [4, 1]> = xavier_uniform();
    fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 1]> {
        return matmul(x, w);
    }
}

train T {
    model = M;
    input x: Tensor<i32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 0;
    device = cpu;
}
",
        "E-TYPE-008",
        None,
    );
    let mut source = lookup_model("[6, 3]", "[4]", "[4, 3]");
    source = source.replace(
        "target y: Tensor<f32, [4, 3]>",
        "target y: Tensor<i32, [4, 3]>",
    );
    refused("i32_target_in_mse", &source, "E-TYPE-018", Some((12, 12)));
}

/// The embedding fixture of the unit, from the repository root.
fn fixture() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/embedding_ids.mspec")
        .display()
        .to_string()
}

/// Attack 6: an `i32` input that no operator reads still checks, and `train` refuses it with
/// E-DATA-004 (not E-DATA-003, which follows it); an `i32` target the loss does not read is
/// E-DATA-004 at the target; dense data refuses before it opens a file.
#[test]
fn adversary_embedding_e_data_004_reaches_unread_ids_targets_and_dense_data() {
    let unread = write(
        "unread_ids",
        "model M {
    param w: Tensor<f32, [4, 3]> = xavier_uniform();
    fn forward(ids: Tensor<i32, [4]>) -> Tensor<f32, [4, 3]> {
        return relu(w);
    }
}

train T {
    model = M;
    input ids: Tensor<i32, [4]>;
    target y: Tensor<f32, [4, 3]>;
    loss = mse(model.forward(ids), y);
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 0;
    device = cpu;
}
",
    );
    let path = unread.to_str().unwrap();
    let output = modelspec(&["check", path]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let output = modelspec(&["train", path, "--backend", "cpu"]);
    let text = stderr(&output);
    assert_eq!(output.status.code(), Some(1), "{text}");
    assert!(output.stdout.is_empty());
    assert!(
        text.starts_with("E-DATA-004: M0 synthetic data cannot supply the i32 input `ids`\n"),
        "{text}"
    );
    assert!(text.contains(".mspec:10:5\n"), "{text}");

    let target = write(
        "unread_target",
        "model M {
    param w: Tensor<f32, [4, 3]> = xavier_uniform();
    fn forward(x: Tensor<f32, [2, 4]>) -> Tensor<f32, [2, 3]> {
        return matmul(x, w);
    }
}

train T {
    model = M;
    input x: Tensor<f32, [2, 4]>;
    target labels: Tensor<i32, [2, 3]>;
    loss = sum(model.forward(x));
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 0;
    device = cpu;
}
",
    );
    let output = modelspec(&["train", target.to_str().unwrap(), "--backend", "cpu"]);
    let text = stderr(&output);
    assert_eq!(output.status.code(), Some(1), "{text}");
    assert!(
        text.starts_with("E-DATA-004: M0 synthetic data cannot supply the i32 target `labels`\n"),
        "{text}"
    );

    let missing = scratch("dense").join("no-such-directory");
    let output = modelspec(&[
        "train",
        &fixture(),
        "--backend",
        "cpu",
        "--data",
        missing.to_str().unwrap(),
    ]);
    let text = stderr(&output);
    assert_eq!(output.status.code(), Some(1), "{text}");
    assert!(
        text.starts_with("E-DATA-004: dense data cannot supply the i32 input `ids`\n"),
        "{text}"
    );
}

/// Attack 8, finding F1: `compile --backend cpu` on the unit's own clean fixture. A CPU artifact
/// has no kernels and empty plans (`compile_artifact`'s doc), yet the command runs the Vulkan
/// schedule first and stops with E-INTERNAL-002, "a fault in modelspec".
#[test]
fn adversary_embedding_compile_for_the_cpu_writes_an_artifact() {
    let out = scratch("compile_cpu").join("artifact");
    let _ = std::fs::remove_dir_all(&out);
    let output = modelspec(&[
        "compile",
        &fixture(),
        "--backend",
        "cpu",
        "--output",
        out.to_str().unwrap(),
    ]);
    let text = stderr(&output);
    assert_eq!(output.status.code(), Some(0), "{text}");
    assert!(out.join("artifact.json").is_file(), "{text}");
}

/// Attack 8: whatever `compile --backend cpu` answers, a refused compile leaves no artifact
/// directory behind.
#[test]
fn adversary_embedding_a_refused_compile_writes_nothing() {
    let out = scratch("compile_refused").join("artifact");
    let _ = std::fs::remove_dir_all(&out);
    let output = modelspec(&[
        "compile",
        &fixture(),
        "--backend",
        "cpu",
        "--output",
        out.to_str().unwrap(),
    ]);
    if output.status.code() != Some(0) {
        assert!(!out.exists(), "a refused compile left {}", out.display());
    }
}

/// Attack 8, finding F2: a command that needs the schedule of an embedding program answers with
/// a user-facing code, not E-INTERNAL-002, whose documentation says no source file is meant to
/// reach it. Since story:bert-vulkan-kernels-rowwise `embedding` has kernels: `ir --stage kernel`
/// prints the schedule, `train --backend vulkan` stops at the missing loader (E-TARGET-002) and
/// `infer --backend vulkan` at the missing checkpoint (E-CKPT-001), each run given a Vulkan
/// library path that does not exist.
#[test]
fn adversary_embedding_no_kernel_is_not_a_compiler_bug() {
    let fixture = fixture();
    let missing = scratch("no_kernel").join("no-such-checkpoint");
    for (args, refusal) in [
        (vec!["ir", "--stage", "kernel", fixture.as_str()], None),
        (
            vec!["train", fixture.as_str(), "--backend", "vulkan"],
            Some("E-TARGET-002: "),
        ),
        (
            vec![
                "infer",
                fixture.as_str(),
                "--backend",
                "vulkan",
                "--checkpoint",
                missing.to_str().unwrap(),
            ],
            Some("E-CKPT-001: "),
        ),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
            .args(&args)
            .env(
                "MODELSPEC_VULKAN_LIBRARY",
                "/nonexistent/modelspec-test/libvulkan-missing.so.1",
            )
            .output()
            .expect("modelspec runs");
        let text = stderr(&output);
        match refusal {
            None => assert_eq!(output.status.code(), Some(0), "{args:?}:\n{text}"),
            Some(code) => {
                assert_eq!(output.status.code(), Some(1), "{args:?}:\n{text}");
                assert!(text.starts_with(code), "{args:?}:\n{text}");
            }
        }
        assert!(
            !text.contains("E-INTERNAL") && !text.contains("compiler bug"),
            "{args:?}:\n{text}"
        );
    }
}
