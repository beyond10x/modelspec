//! Adversary pass 1 on story:bert-layer-norm-softmax, through the `modelspec` binary: the shape,
//! rank and dtype rules of `layer_norm` and `softmax` at the call, `E-TYPE-017` with a norm in the
//! model, the diagnostics page, `E-TARGET-011` for every command that needs a Vulkan kernel, and
//! the operand order the IR printer shows for the new ops and their gradients.
//!
//! No test here creates a Vulkan instance: every run sets `MODELSPEC_VULKAN_LIBRARY` to a path
//! that does not exist.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const NO_LOADER: &str = "/nonexistent/modelspec-test/libvulkan-missing.so.1";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
        .display()
        .to_string()
}

fn profile() -> String {
    root()
        .join("crates/modelspec-runtime/tests/fixtures/rtx3090.device.json")
        .display()
        .to_string()
}

/// A fresh path under the test temp dir that does not exist yet.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary_norm_cli")
        .join(name);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("the old test dir is removed");
    }
    std::fs::create_dir_all(dir.parent().unwrap()).expect("the temp dir is writable");
    dir
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .env("MODELSPEC_VULKAN_LIBRARY", NO_LOADER)
        .output()
        .expect("modelspec runs")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// `check` exits 1 with one diagnostic `code` at `<fixture>:<at>`; returns stderr.
fn check_refuses(name: &str, code: &str, at: &str) -> String {
    let output = modelspec(&["check", &fixture(name)]);
    let text = stderr(&output);
    assert_eq!(output.status.code(), Some(1), "{name}:\n{text}");
    let heads: Vec<&str> = text.lines().filter(|l| !l.starts_with("  ")).collect();
    assert_eq!(heads.len(), 1, "{name}: one diagnostic:\n{text}");
    assert!(
        heads[0].starts_with(&format!("{code}: ")),
        "{name}: expected {code}:\n{text}"
    );
    assert!(
        text.contains(&format!("{name}:{at}")),
        "{name}: not at {at}:\n{text}"
    );
    text
}

// ---------------------------------------------------------------------------------------------
// Attack 5: the checker.
// ---------------------------------------------------------------------------------------------

/// Attack 5. A `beta` of shape `[H + 1]` is `E-TYPE-003` at the call, like a long `gamma`.
#[test]
fn adversary_norm_a_beta_of_h_plus_1_is_e_type_003_at_the_call() {
    let text = check_refuses("adversary_norm_beta_too_long.mspec", "E-TYPE-003", "7:16");
    assert!(text.contains("`layer_norm`"), "{text}");
}

/// Attack 5, finding N4. A long `gamma` and a long `beta` give the same diagnostic: same code,
/// same location, same `lhs` and `rhs`, same note; nothing says which operand is wrong. The
/// `add_rows` rank message names its operand (coordinator decision D33); `layer_norm`'s shape
/// and rank messages do not.
#[test]
fn adversary_norm_the_diagnostic_names_the_operand_that_is_wrong() {
    let strip = |name: &str| {
        let output = modelspec(&["check", &fixture(name)]);
        stderr(&output).replace(&fixture(name), "<file>")
    };
    let gamma = strip("layer_norm_gamma_too_long.mspec");
    let beta = strip("adversary_norm_beta_too_long.mspec");
    assert_ne!(
        gamma, beta,
        "a long gamma and a long beta read the same:\n{gamma}"
    );
    let rank = check_refuses("adversary_norm_gamma_rank2.mspec", "E-TYPE-004", "7:16");
    assert!(rank.contains("`gamma` has type"), "{rank}");
}

/// Attack 5. A `gamma` of rank 2 (`[1, H]`, not broadcast), an `x` of rank 1 and an `x` of rank
/// 3 are `E-TYPE-004` at the call.
#[test]
fn adversary_norm_operands_of_the_wrong_rank_are_e_type_004_at_the_call() {
    for (name, at, got) in [
        (
            "adversary_norm_gamma_rank2.mspec",
            "7:16",
            "Tensor<f32, [1, 3]>",
        ),
        ("adversary_norm_x_rank1.mspec", "8:17", "Tensor<f32, [3]>"),
        (
            "adversary_norm_x_rank3.mspec",
            "6:16",
            "Tensor<f32, [2, 3, 4]>",
        ),
    ] {
        let text = check_refuses(name, "E-TYPE-004", at);
        assert!(text.contains("`layer_norm`"), "{name}:\n{text}");
        assert!(text.contains(got), "{name}:\n{text}");
    }
}

/// Attack 5. `softmax` of a rank-3 and of a rank-0 tensor is `E-TYPE-004` at the call.
#[test]
fn adversary_norm_softmax_of_rank_3_and_rank_0_is_e_type_004_at_the_call() {
    for (name, at, got) in [
        (
            "adversary_norm_softmax_rank3.mspec",
            "5:16",
            "Tensor<f32, [2, 3, 4]>",
        ),
        (
            "adversary_norm_softmax_rank0.mspec",
            "5:17",
            "Tensor<f32, []>",
        ),
    ] {
        let text = check_refuses(name, "E-TYPE-004", at);
        assert!(text.contains("`softmax`"), "{name}:\n{text}");
        assert!(text.contains(got), "{name}:\n{text}");
    }
}

/// Attack 5. An `i32` `gamma` is `E-TYPE-018` at the call, naming operand 2.
#[test]
fn adversary_norm_an_i32_gamma_is_e_type_018_at_the_call() {
    let text = check_refuses("adversary_norm_i32_gamma.mspec", "E-TYPE-018", "6:16");
    assert!(
        text.contains("`layer_norm` operand 2 must be f32"),
        "{text}"
    );
}

/// Attack 4. A loss that calls a `model.forward` holding `layer_norm` twice is `E-TYPE-017`.
#[test]
fn adversary_norm_two_forward_calls_are_e_type_017() {
    let output = modelspec(&["check", &fixture("adversary_norm_two_forward_calls.mspec")]);
    let text = stderr(&output);
    assert_eq!(output.status.code(), Some(1), "{text}");
    assert!(text.starts_with("E-TYPE-017: "), "{text}");
}

/// Attack 5. The diagnostics page names `layer_norm` for `E-TYPE-003` and `layer_norm` and
/// `softmax` for `E-TYPE-004`.
#[test]
fn adversary_norm_the_diagnostics_page_names_both_operators() {
    let page = std::fs::read_to_string(root().join("website/docs/reference/diagnostics.md"))
        .expect("the page reads");
    let row = |code: &str| {
        page.lines()
            .find(|l| l.contains(&format!("`{code}` |")))
            .unwrap_or_else(|| panic!("no row for {code}"))
            .to_owned()
    };
    assert!(row("E-TYPE-003").contains("`layer_norm(x, gamma, beta)`"));
    let rank = row("E-TYPE-004");
    assert!(rank.contains("`layer_norm(x, gamma, beta)`"), "{rank}");
    assert!(rank.contains("`softmax`"), "{rank}");
}

// ---------------------------------------------------------------------------------------------
// Attack 6: the Vulkan arm.
// ---------------------------------------------------------------------------------------------

/// The four commands that need a Vulkan kernel, for `fixture`.
fn vulkan_commands(fixture: &str, tag: &str) -> Vec<Vec<String>> {
    let out = scratch(&format!("{tag}-vulkan")).join("artifact");
    let missing = scratch(&format!("{tag}-missing")).join("no-such-checkpoint");
    let s = |v: &str| v.to_owned();
    vec![
        vec![s("ir"), s("--stage"), s("kernel"), s(fixture)],
        vec![
            s("compile"),
            s(fixture),
            s("--backend"),
            s("vulkan"),
            s("--device-profile"),
            profile(),
            s("--output"),
            out.display().to_string(),
        ],
        vec![s("train"), s(fixture), s("--backend"), s("vulkan")],
        vec![
            s("infer"),
            s(fixture),
            s("--backend"),
            s("vulkan"),
            s("--checkpoint"),
            missing.display().to_string(),
        ],
    ]
}

fn refuses_with_no_kernel(name: &str, operator: &str, at: &str) {
    let fixture = fixture(name);
    for args in vulkan_commands(&fixture, operator) {
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        let output = modelspec(&args);
        let text = stderr(&output);
        assert_eq!(output.status.code(), Some(1), "{args:?}:\n{text}");
        assert!(
            text.contains(&format!(
                "E-TARGET-011: operator `{operator}` has no Vulkan kernel yet"
            )),
            "{args:?}:\n{text}"
        );
        assert!(
            text.contains(&format!("{name}:{at}")),
            "{args:?}: not at {at}:\n{text}"
        );
        assert!(!text.contains("E-INTERNAL"), "{args:?}:\n{text}");
        assert!(stdout(&output).is_empty(), "{args:?}: printed a plan");
    }
}

/// Attack 6. On `norm_softmax_train.mspec` every Vulkan command answers `E-TARGET-011` naming
/// `layer_norm`, the first operator without a kernel, at its call (line 7, column 24).
#[test]
fn adversary_norm_every_vulkan_command_refuses_layer_norm_at_its_call() {
    refuses_with_no_kernel("norm_softmax_train.mspec", "layer_norm", "7:24");
}

/// Attack 6. A program whose only new operator is `softmax` is refused naming `softmax`.
#[test]
fn adversary_norm_every_vulkan_command_refuses_softmax_at_its_call() {
    refuses_with_no_kernel("adversary_norm_softmax_only.mspec", "softmax", "5:16");
}

// ---------------------------------------------------------------------------------------------
// Attack 7: printing.
// ---------------------------------------------------------------------------------------------

/// Each printed node line, `[tag ]%n: <type> = <Op> <operands>[ {effect}]`, by its id: the op
/// and the operand list as printed, without the effect.
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
            let (op, operands) = rhs.split_once(' ').unwrap_or((rhs, ""));
            let operands = operands
                .split(", ")
                .filter(|o| !o.is_empty())
                .map(str::to_owned)
                .collect();
            Some((id.to_owned(), (op.to_owned(), operands)))
        })
        .collect()
}

fn id_of(nodes: &BTreeMap<String, (String, Vec<String>)>, op: &str, operands: &[&str]) -> String {
    nodes
        .iter()
        .find(|(_, (o, args))| {
            o == op
                && (operands.is_empty()
                    || args.iter().map(String::as_str).eq(operands.iter().copied()))
        })
        .map(|(id, _)| id.clone())
        .unwrap_or_else(|| panic!("no {op} {operands:?} in {nodes:?}"))
}

/// Attack 7. `ir --stage training` prints `LayerNorm x, gamma, beta` (operand order of the
/// source call), `Softmax` of it, `SoftmaxGrad` of the `Softmax` result and the loss's adjoint,
/// `LayerNormGradX x, gamma, dZ`, `LayerNormGradGamma x, dZ` and `SumRows dZ`, `dZ` the
/// `SoftmaxGrad`; `--stage tensor` prints the same forward lines.
#[test]
fn adversary_norm_the_printed_ir_shows_the_operands_in_order() {
    let fixture = fixture("norm_softmax_train.mspec");
    let output = modelspec(&["ir", "--stage", "training", &fixture]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let training = stdout(&output);
    let n = nodes(&training);
    let x = id_of(&n, "Input", &["\"x\""]);
    let w = id_of(&n, "ParamRead", &["@Norm.w"]);
    let xw = id_of(&n, "MatMul", &[&x, &w]);
    let gamma = id_of(&n, "ParamRead", &["@Norm.gamma"]);
    let beta = id_of(&n, "ParamRead", &["@Norm.beta"]);
    let z = id_of(&n, "LayerNorm", &[&xw, &gamma, &beta]);
    let y = id_of(&n, "Softmax", &[&z]);
    let dz = n
        .iter()
        .find(|(_, (op, args))| op == "SoftmaxGrad" && args.first() == Some(&y))
        .map(|(id, _)| id.clone())
        .unwrap_or_else(|| panic!("no SoftmaxGrad of {y}:\n{training}"));
    id_of(&n, "LayerNormGradX", &[&xw, &gamma, &dz]);
    id_of(&n, "LayerNormGradGamma", &[&xw, &dz]);
    id_of(&n, "SumRows", &[&dz]);

    let output = modelspec(&["ir", "--stage", "tensor", &fixture]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let tensor = stdout(&output);
    let t = nodes(&tensor);
    let x = id_of(&t, "Input", &["\"x\""]);
    let w = id_of(&t, "ParamRead", &["@Norm.w"]);
    let xw = id_of(&t, "MatMul", &[&x, &w]);
    let gamma = id_of(&t, "ParamRead", &["@Norm.gamma"]);
    let beta = id_of(&t, "ParamRead", &["@Norm.beta"]);
    let z = id_of(&t, "LayerNorm", &[&xw, &gamma, &beta]);
    id_of(&t, "Softmax", &[&z]);
}
