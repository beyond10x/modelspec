//! Adversary pass 1 on story:bert-attention, through the `modelspec` binary: the named argument
//! `heads` and the operand rules of `attention` at the call, `E-TYPE-021` for a key mask that
//! reads a parameter, the training IR of the three gradient ops, `E-TARGET-011` for every command
//! that needs a Vulkan kernel, and the key mask check on the one CLI path that runs `attention` on
//! the CPU interpreter today.
//!
//! The sources are written under this test's `CARGO_TARGET_TMPDIR`. No test here creates a Vulkan
//! instance: every run sets `MODELSPEC_VULKAN_LIBRARY` to a path that does not exist.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const NO_LOADER: &str = "/nonexistent/modelspec-test/libvulkan-missing.so.1";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh directory under the test temp dir for `name`.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary_attention_cli")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("the old test dir is removed");
    }
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    dir
}

/// `source` written as `<scratch>/<name>.mspec`; returns its path.
fn write(name: &str, source: &str) -> String {
    let path = scratch(name).join(format!("{name}.mspec"));
    std::fs::write(&path, source).expect("the source is written");
    path.display().to_string()
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .env("MODELSPEC_VULKAN_LIBRARY", NO_LOADER)
        .current_dir(root())
        .output()
        .expect("modelspec runs")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// The parts of the attack 6 model that vary.
struct Model<'a> {
    width: u64,
    mask: &'a str,
    lets: &'a str,
    result: &'a str,
    loss: &'a str,
}

impl Default for Model<'_> {
    fn default() -> Self {
        Self {
            width: 4,
            mask: "Tensor<f32, [2, 3]>",
            lets: "",
            result: "attention(x, x, x, m, heads = 2)",
            loss: "mse(model.forward(x, m), y)",
        }
    }
}

/// The line of `return` in [`Model::source`] and the column its expression starts at.
const RESULT: (usize, usize) = (7, 16);

impl Model<'_> {
    fn source(&self) -> String {
        let Model {
            width,
            mask,
            lets,
            result,
            loss,
        } = self;
        format!(
            "model M {{
    param w: Tensor<f32, [{width}, {width}]> = xavier_uniform();
    param p: Tensor<f32, [2, 3]> = ones();
    param u: Tensor<f32, [{width}, 8]> = xavier_uniform();
    fn forward(x: Tensor<f32, [6, {width}]>, m: {mask}) -> Tensor<f32, [6, {width}]> {{
        {lets}
        return {result};
    }}
}}

train T {{
    model = M;
    input x: Tensor<f32, [6, {width}]>;
    input m: {mask};
    target y: Tensor<f32, [6, {width}]>;
    loss = {loss};
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 0;
    device = cpu;
}}
"
        )
    }

    /// The location of `needle` in the result expression, as `line:column`.
    fn at(&self, needle: &str) -> String {
        let offset = self
            .result
            .find(needle)
            .unwrap_or_else(|| panic!("{needle} not in {}", self.result));
        format!("{}:{}", RESULT.0, RESULT.1 + offset)
    }
}

/// `check` of `source` as `name`: the exit code, stderr with the path replaced by `<file>`, and
/// the code of every diagnostic with its location.
fn check(name: &str, source: &str) -> (Option<i32>, String, Vec<(String, String)>) {
    let path = write(name, source);
    let output = modelspec(&["check", &path]);
    let text = stderr(&output).replace(&path, "<file>");
    let lines: Vec<&str> = text.lines().collect();
    let mut found = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if line.starts_with("  ") {
            continue;
        }
        let code = line.split(':').next().unwrap_or("").to_owned();
        let at = lines
            .get(i + 1)
            .and_then(|l| l.strip_prefix("  <file>:"))
            .unwrap_or("")
            .to_owned();
        found.push((code, at));
    }
    (output.status.code(), text, found)
}

/// `check` exits 1 with exactly one diagnostic, `code` at `at`; returns stderr.
fn refused(name: &str, model: &Model<'_>, code: &str, at: &str) -> String {
    let (status, text, found) = check(name, &model.source());
    assert_eq!(status, Some(1), "{name}:\n{text}");
    assert_eq!(
        found,
        [(code.to_owned(), at.to_owned())],
        "{name}: one {code} at {at}:\n{text}"
    );
    text
}

// ---------------------------------------------------------------------------------------------
// Attack 6: the checker and the named argument.
// ---------------------------------------------------------------------------------------------

/// Attack 6. `heads = 1` and `heads = H` check clean.
#[test]
fn adversary_attention_a6_heads_1_and_heads_h_check_clean() {
    for (name, result) in [
        ("heads1", "attention(x, x, x, m, heads = 1)"),
        ("heads4", "attention(x, x, x, m, heads = 4)"),
    ] {
        let model = Model {
            result,
            ..Model::default()
        };
        let (status, text, _) = check(name, &model.source());
        assert_eq!(status, Some(0), "{name}:\n{text}");
        assert_eq!(text, "", "{name}");
    }
}

/// Attack 6. `heads = -1` and `heads = 1_000` are not integer literals of the grammar: one
/// syntax diagnostic each, exit 1. `heads = 2^64 - 1` with `H = 4` is `E-TYPE-020` at the call;
/// `heads = 2^64` is out of range for an integer literal, a syntax diagnostic.
#[test]
fn adversary_attention_a6_heads_negative_underscored_and_huge() {
    for (name, result) in [
        ("negative", "attention(x, x, x, m, heads = -1)"),
        ("underscored", "attention(x, x, x, m, heads = 1_000)"),
        (
            "overflow",
            "attention(x, x, x, m, heads = 18446744073709551616)",
        ),
    ] {
        let model = Model {
            result,
            ..Model::default()
        };
        let (status, text, found) = check(name, &model.source());
        assert_eq!(status, Some(1), "{name}:\n{text}");
        assert_eq!(found.len(), 1, "{name}: one diagnostic:\n{text}");
        assert!(found[0].0.starts_with("E-SYNTAX-"), "{name}:\n{text}");
    }
    let model = Model {
        result: "attention(x, x, x, m, heads = 18446744073709551615)",
        ..Model::default()
    };
    refused("u64max", &model, "E-TYPE-020", &model.at("attention"));
}

/// Attack 6. `heads = 2` with `H = 3` is `E-TYPE-020` at the call, naming `3` and `heads = 2`.
#[test]
fn adversary_attention_a6_heads_2_with_h_3_is_e_type_020_naming_both() {
    let model = Model {
        width: 3,
        ..Model::default()
    };
    let text = refused("h3", &model, "E-TYPE-020", &model.at("attention"));
    assert!(text.contains("width 3"), "{text}");
    assert!(text.contains("heads = 2"), "{text}");
}

/// Attack 6. `heads = heads` names a value, not a literal: `E-TYPE-019` at the argument.
#[test]
fn adversary_attention_a6_heads_equals_heads_is_e_type_019_at_the_argument() {
    let model = Model {
        result: "attention(x, x, x, m, heads = heads)",
        ..Model::default()
    };
    let text = refused("heads_heads", &model, "E-TYPE-019", &model.at("heads"));
    assert!(text.contains("`heads` of `attention`"), "{text}");
}

/// Attack 6. `heads` given positionally, `attention(x, x, x, m, 2)`: the named argument is
/// missing, `E-TYPE-001` at the call; the integer literal in an operand position is also
/// reported, and nothing is a compiler bug.
#[test]
fn adversary_attention_a6_positional_heads_is_e_type_001_at_the_call() {
    let model = Model {
        result: "attention(x, x, x, m, 2)",
        ..Model::default()
    };
    let (status, text, found) = check("positional", &model.source());
    assert_eq!(status, Some(1), "{text}");
    assert!(
        found.contains(&("E-TYPE-001".to_owned(), model.at("attention"))),
        "{text}"
    );
    assert!(!text.contains("E-INTERNAL"), "{text}");
    assert!(text.contains("`heads`"), "{text}");
}

/// Attack 6. `heads = 2, heads = 2` is one `E-TYPE-002` at the second.
#[test]
fn adversary_attention_a6_heads_twice_is_e_type_002_at_the_second() {
    let model = Model {
        result: "attention(x, x, x, m, heads = 2, heads = 2)",
        ..Model::default()
    };
    let second = RESULT.1 + model.result.rfind("heads").unwrap();
    let text = refused(
        "twice",
        &model,
        "E-TYPE-002",
        &format!("{}:{second}", RESULT.0),
    );
    assert!(text.contains("given twice"), "{text}");
}

/// Attack 6. DESIGN.md § 2.1: "An operator call takes its operands as positional arguments, and
/// after them the operator's named compile-time arguments". `heads = 2` before the operands is
/// therefore not a call of that form and is refused.
#[test]
fn adversary_attention_a6_named_argument_before_the_operands_is_refused() {
    let model = Model {
        result: "attention(heads = 2, x, x, x, m)",
        ..Model::default()
    };
    let (status, text, _) = check("named_first", &model.source());
    assert_eq!(
        status,
        Some(1),
        "a named argument before the operands checks:\n{text}"
    );
}

/// Attack 6. A named argument on `relu` is `E-TYPE-002` at the argument. Its value is still
/// typed as an operand, so the integer literal `2` is also `E-TYPE-005` at the literal, as the
/// positional-argument check did at the base commit.
#[test]
fn adversary_attention_a6_named_argument_on_relu_is_e_type_002() {
    let model = Model {
        result: "relu(x, heads = 2)",
        ..Model::default()
    };
    let (status, text, found) = check("relu_named", &model.source());
    assert_eq!(status, Some(1), "{text}");
    assert_eq!(
        found,
        [
            ("E-TYPE-002".to_owned(), model.at("heads")),
            ("E-TYPE-005".to_owned(), model.at("2)")),
        ],
        "{text}"
    );
}

/// Attack 6. A named argument on a model function call is `E-TYPE-002`, and nothing is a
/// compiler bug.
#[test]
fn adversary_attention_a6_named_argument_on_a_model_function_is_e_type_002() {
    let model = Model {
        loss: "mse(model.forward(x, m, heads = 2), y)",
        ..Model::default()
    };
    let (status, text, found) = check("model_named", &model.source());
    assert_eq!(status, Some(1), "{text}");
    assert!(found.iter().any(|(code, _)| code == "E-TYPE-002"), "{text}");
    assert!(!text.contains("E-INTERNAL"), "{text}");
}

/// Attack 6. A key mask of `i32` is `E-TYPE-018` at the call, naming operand 4.
#[test]
fn adversary_attention_a6_i32_key_mask_is_e_type_018() {
    let model = Model {
        mask: "Tensor<i32, [2, 3]>",
        ..Model::default()
    };
    let text = refused("mask_i32", &model, "E-TYPE-018", &model.at("attention"));
    assert!(text.contains("operand 4"), "{text}");
}

/// Attack 6. A key mask of rank 1 is `E-TYPE-004`; one of `[3, 3]` (`B * S = 9`, not `N = 6`)
/// and a `v` of width 8 are `E-TYPE-003`; all at the call, naming the offending type.
#[test]
fn adversary_attention_a6_key_mask_and_v_shapes() {
    let rank1 = Model {
        mask: "Tensor<f32, [6]>",
        ..Model::default()
    };
    let text = refused("mask_rank1", &rank1, "E-TYPE-004", &rank1.at("attention"));
    assert!(text.contains("Tensor<f32, [6]>"), "{text}");
    let nine = Model {
        mask: "Tensor<f32, [3, 3]>",
        ..Model::default()
    };
    let text = refused("mask_3x3", &nine, "E-TYPE-003", &nine.at("attention"));
    assert!(text.contains("Tensor<f32, [3, 3]>"), "{text}");
    assert!(text.contains("key_mask"), "{text}");
    let wide_v = Model {
        result: "attention(x, x, matmul(x, u), m, heads = 2)",
        ..Model::default()
    };
    let text = refused("v_width8", &wide_v, "E-TYPE-003", &wide_v.at("attention"));
    assert!(text.contains("Tensor<f32, [6, 8]>"), "{text}");
}

/// Attack 6. A key mask that is a parameter through two `let`s, or `relu` of a parameter, is
/// `E-TYPE-021` at the argument; `relu` of the input checks clean.
#[test]
fn adversary_attention_a6_key_mask_reading_a_parameter() {
    let lets = Model {
        lets: "let a = p; let b = a;",
        result: "attention(x, x, x, b, heads = 2)",
        ..Model::default()
    };
    refused("two_lets", &lets, "E-TYPE-021", &lets.at("b,"));
    let relu_p = Model {
        result: "attention(x, x, x, relu(p), heads = 2)",
        ..Model::default()
    };
    refused("relu_p", &relu_p, "E-TYPE-021", &relu_p.at("relu"));
    let relu_m = Model {
        result: "attention(x, x, x, relu(m), heads = 2)",
        ..Model::default()
    };
    let (status, text, _) = check("relu_m", &relu_m.source());
    assert_eq!(status, Some(0), "{text}");
}

/// A model whose `mask` function reads `p`, and whose `forward` takes its key mask as an
/// argument; the train block's loss is `loss`.
fn mask_function_model(mask_body: &str, loss: &str) -> String {
    format!(
        "model M {{
    param w: Tensor<f32, [4, 4]> = xavier_uniform();
    param p: Tensor<f32, [2, 3]> = ones();
    fn mask(m: Tensor<f32, [2, 3]>) -> Tensor<f32, [2, 3]> {{
        return {mask_body};
    }}
    fn forward(x: Tensor<f32, [6, 4]>, m: Tensor<f32, [2, 3]>) -> Tensor<f32, [6, 4]> {{
        return attention(matmul(x, w), x, x, m, heads = 2);
    }}
}}

train T {{
    model = M;
    input x: Tensor<f32, [6, 4]>;
    input m: Tensor<f32, [2, 3]>;
    target y: Tensor<f32, [6, 4]>;
    loss = {loss};
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 0;
    device = cpu;
}}
"
    )
}

/// Attack 6. DESIGN.md § 2.2: a key mask that reads a `param` "through a model function call"
/// is `E-TYPE-021`. Here the train block passes `model.mask(m)`, which reads `p`, as the key mask
/// argument of `forward`: inside `forward` the mask is an argument, which `reads_param` takes to
/// read no parameter. `check` must refuse it with `E-TYPE-021`, and `ir --stage training` must not
/// answer with a compiler bug.
#[test]
fn adversary_attention_a6_key_mask_reading_a_parameter_through_a_function_argument() {
    let source = mask_function_model("mul(m, p)", "mse(model.forward(x, model.mask(m)), y)");
    let (status, text, found) = check("mask_arg", &source);
    let path = write("mask_arg_ir", &source);
    let ir = modelspec(&["ir", &path, "--stage", "training"]);
    let ir_text = stderr(&ir).replace(&path, "<file>");
    assert!(
        !ir_text.contains("E-INTERNAL"),
        "ir --stage training answers with a compiler bug:\n{ir_text}"
    );
    assert_eq!(status, Some(1), "check accepts it:\n{text}");
    assert!(found.iter().any(|(code, _)| code == "E-TYPE-021"), "{text}");
}

/// Attack 6. A key mask computed in the train block by a model function that reads no parameter,
/// `model.ident(m)` returning `relu(m)`, is refused by `E-TYPE-021` although it reads no `param`:
/// the rule takes every model function call to read the model's parameters.
#[test]
fn adversary_attention_a6_key_mask_from_a_parameter_free_model_function_checks() {
    let source = mask_function_model(
        "relu(m)",
        "mse(attention(model.forward(x, m), x, x, model.mask(m), heads = 2), y)",
    );
    let (status, text, _) = check("mask_free", &source);
    assert_eq!(
        status,
        Some(0),
        "a parameter-free key mask is refused:\n{text}"
    );
}

// ---------------------------------------------------------------------------------------------
// Attack 7: the training IR.
// ---------------------------------------------------------------------------------------------

/// Each printed node line, `[tag ]%n: <type> = <Op> [<attrs>] <operands>[ {effect}]`, by its id:
/// the op with its attributes and the operand list.
fn nodes(ir: &str) -> BTreeMap<String, (String, Vec<String>)> {
    ir.lines()
        .filter_map(|line| {
            let line = match line.find('%') {
                Some(at) if !line.starts_with("output") => &line[at..],
                _ => return None,
            };
            let line = line.split(" {").next().unwrap_or(line);
            let (id, rest) = line.split_once(": ")?;
            let (_, rhs) = rest.split_once(" = ")?;
            let (op, operands) = match rhs.split_once("] ") {
                Some((op, operands)) if op.contains(" [") => (format!("{op}]"), operands),
                _ => {
                    let (op, operands) = rhs.split_once(' ').unwrap_or((rhs, ""));
                    (op.to_owned(), operands)
                }
            };
            let operands = operands
                .split(", ")
                .filter(|o| !o.is_empty())
                .map(str::to_owned)
                .collect();
            Some((id.to_owned(), (op, operands)))
        })
        .collect()
}

/// Attack 7. `ir --stage training` of the unit's fixture prints `Attention [heads = 2]` over
/// four operands and each gradient op with the same `[heads = 2]` over those four operands and one
/// adjoint, the same adjoint for all three, which is not the attention output itself.
#[test]
fn adversary_attention_a7_training_ir_gradient_operands() {
    let fixture = root().join("crates/modelspec-cli/tests/fixtures/attention_train.mspec");
    let output = modelspec(&["ir", &fixture.display().to_string(), "--stage", "training"]);
    let text = stdout(&output);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let nodes = nodes(&text);
    let find = |op: &str| -> Vec<(String, Vec<String>)> {
        nodes
            .iter()
            .filter(|(_, (o, _))| o == op)
            .map(|(id, (_, args))| (id.clone(), args.clone()))
            .collect()
    };
    let forward = find("Attention [heads = 2]");
    assert_eq!(forward.len(), 1, "{text}");
    let (y, operands) = &forward[0];
    assert_eq!(operands.len(), 4, "{text}");
    let mut adjoints = Vec::new();
    for op in ["AttentionGradQ", "AttentionGradK", "AttentionGradV"] {
        let grads = find(&format!("{op} [heads = 2]"));
        assert_eq!(grads.len(), 1, "{op}:\n{text}");
        let args = &grads[0].1;
        assert_eq!(args.len(), 5, "{op}:\n{text}");
        assert_eq!(&args[..4], &operands[..], "{op}:\n{text}");
        assert_ne!(&args[4], y, "{op}: the adjoint is not the output");
        adjoints.push(args[4].clone());
    }
    assert!(adjoints.iter().all(|a| a == &adjoints[0]), "{adjoints:?}");
}

// ---------------------------------------------------------------------------------------------
// Attack 8: the Vulkan arm.
// ---------------------------------------------------------------------------------------------

/// Attack 8. On the unit's fixture, `ir --stage kernel` and `compile --backend vulkan
/// --device-profile` schedule `attention` and its three gradients as `per_row` kernels
/// (story:bert-vulkan-kernels-attention-loss-adamw, coordinator decision D73): each exits 0 and
/// reports no compiler bug, and the kernel stage prints the four records.
#[test]
fn adversary_attention_a8_every_vulkan_compile_schedules_attention_kernels() {
    let fixture = root()
        .join("crates/modelspec-cli/tests/fixtures/attention_train.mspec")
        .display()
        .to_string();
    let profile = root()
        .join("crates/modelspec-runtime/tests/fixtures/rtx3090.device.json")
        .display()
        .to_string();
    let out = scratch("vulkan").join("artifact").display().to_string();
    let commands: [Vec<&str>; 2] = [
        vec!["ir", "--stage", "kernel", &fixture],
        vec![
            "compile",
            &fixture,
            "--backend",
            "vulkan",
            "--device-profile",
            &profile,
            "--output",
            &out,
        ],
    ];
    for args in &commands {
        let output = modelspec(args);
        let text = stderr(&output);
        assert_eq!(output.status.code(), Some(0), "{args:?}:\n{text}");
        assert!(!text.contains("E-INTERNAL"), "{args:?}:\n{text}");
    }
    let printed = stdout(&modelspec(&commands[0]));
    for op in [
        "Attention",
        "AttentionGradQ",
        "AttentionGradK",
        "AttentionGradV",
    ] {
        assert!(
            printed.lines().any(
                |line| line.starts_with("kernel ") && line.contains(&format!(" per_row {op} "))
            ),
            "{op}:\n{printed}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Attack 9: the key mask check through the CLI.
// ---------------------------------------------------------------------------------------------

/// A file with one `input`, `m: [1, 1]`, which is the key mask and, through `matmul(m, w)`, also
/// `q`, `k` and `v`: one sequence of one key. `train` accepts its data declarations (one input,
/// one target), so it runs `attention` on the CPU interpreter.
const ONE_INPUT: &str = "model One {
    param w: Tensor<f32, [1, 4]> = xavier_uniform();
    fn forward(m: Tensor<f32, [1, 1]>) -> Tensor<f32, [1, 4]> {
        return attention(matmul(m, w), matmul(m, w), matmul(m, w), m, heads = 2);
    }
}

train Fit {
    model = One;
    input m: Tensor<f32, [1, 1]>;
    target y: Tensor<f32, [1, 4]>;
    loss = mse(model.forward(m), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 1;
    device = cpu;
}
";

fn f32_file(path: &Path, values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).expect("the data file is written");
}

/// Attack 9. `train --backend cpu` runs `attention` today: on synthetic data (a standard normal
/// key mask) it exits 1 with the key mask error naming sequence 0 and key 0; on a dense key mask
/// of `0.0` it exits 1 naming sequence 0 as fully masked; on a key mask of `1.0` it trains and
/// exits 0.
#[test]
fn adversary_attention_a9_train_on_the_cpu_runs_the_key_mask_check() {
    let path = write("one_input", ONE_INPUT);
    let synthetic = modelspec(&["train", &path, "--backend", "cpu"]);
    let text = stderr(&synthetic);
    assert_eq!(synthetic.status.code(), Some(1), "synthetic:\n{text}");
    assert!(
        text.contains("holds a value other than 0 or 1 at sequence 0, key 0"),
        "synthetic:\n{text}"
    );
    for (mask, status, message) in [
        (0.0f32, Some(1), Some("masks every key of sequence 0")),
        (1.0f32, Some(0), None),
    ] {
        let dir = scratch(&format!("dense_{mask}"));
        f32_file(&dir.join("m.f32"), &[mask]);
        f32_file(&dir.join("y.f32"), &[0.5, -0.25, 1.0, 0.0]);
        let output = modelspec(&[
            "train",
            &path,
            "--backend",
            "cpu",
            "--data",
            &dir.display().to_string(),
        ]);
        let text = stderr(&output);
        assert_eq!(output.status.code(), status, "mask {mask}:\n{text}");
        if let Some(message) = message {
            assert!(text.contains(message), "mask {mask}:\n{text}");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Attack 10: the diagnostics page.
// ---------------------------------------------------------------------------------------------

/// Attack 10. The diagnostics page has a row for each new code, naming what the CLI refuses with
/// it: `heads` for `E-TYPE-019` and `E-TYPE-020`, `key_mask` for `E-TYPE-021`.
#[test]
fn adversary_attention_a10_diagnostics_page_rows() {
    let page = std::fs::read_to_string(root().join("website/docs/reference/diagnostics.md"))
        .expect("the page reads");
    let row = |code: &str| {
        page.lines()
            .find(|l| l.contains(&format!("`{code}` |")))
            .unwrap_or_else(|| panic!("no row for {code}"))
            .to_owned()
    };
    assert!(row("E-TYPE-019").contains("heads"));
    assert!(row("E-TYPE-020").contains("heads"));
    assert!(row("E-TYPE-021").contains("key_mask"));
    assert!(row("E-TYPE-001").contains("heads"));
}
