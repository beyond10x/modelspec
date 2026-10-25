//! `modelspec ir --stage kernel`: the Kernel IR and both execution plans of a training step
//! (DESIGN.md § 3.1, § 7.1, § 8; story m1-kernel-plan acceptance 6).
//!
//! The golden `tests/golden/linear_regression.kernel.ir` is what `print_kernel` prints for the
//! example's Vulkan artifact; the cases below check it against the plan the compiler builds, so a
//! golden regenerated from a wrong plan does not pass. `tests/golden/mnist_mlp.kernel.ir` is the
//! MNIST example's, with the fusions of DESIGN.md § 3.1 (story m1-fuse-elementwise), the `mse` loss
//! sum beside the `w1` gradient and the relu gradient a matmul epilogue (story:step-sites-beat-pytorch).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use modelspec_core::artifact::Backend;
use modelspec_core::compile;
use modelspec_core::ir::print::print_kernel;

const EXAMPLE: &str = "examples/linear_regression.mspec";
const GOLDEN: &str = include_str!("golden/linear_regression.kernel.ir");

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
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

#[test]
fn ir_stage_kernel_prints_exactly_the_golden() {
    let output = modelspec(&["ir", EXAMPLE, "--stage", "kernel"]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    assert_eq!(text(&output.stdout), GOLDEN);
    assert_eq!(text(&output.stderr), "");
}

#[test]
fn the_golden_is_the_printed_plan_of_the_vulkan_artifact() {
    let source = std::fs::read_to_string(repo_root().join(EXAMPLE)).expect("the example exists");
    let artifact =
        compile::compile_artifact(EXAMPLE, &source, Backend::Vulkan).expect("the example compiles");
    assert_eq!(print_kernel(&artifact.execution), GOLDEN);
}

/// story:fuse-elementwise-chains-and-gradient-sums: the `Sub`→`Mul` chain of the linear
/// regression's loss gradient is one dispatch, and its `Sub` has no buffer.
#[test]
fn the_golden_runs_the_loss_gradient_chain_as_one_dispatch() {
    let chain = GOLDEN
        .lines()
        .find(|l| l.starts_with("kernel ") && l.contains(" elementwise "))
        .expect("an elementwise kernel");
    assert!(
        chain.contains(
            " elementwise Sub>Mul@0 f32 (Tensor<f32, [8, 1]>, Tensor<f32, [8, 1]>, 0.25) \
             -> Tensor<f32, [8, 1]> "
        ),
        "{chain}"
    );
    let step = plan_lines(GOLDEN, "step");
    let dispatches = step.iter().filter(|l| l.contains(": dispatch ")).count();
    let barriers = step.iter().filter(|l| l.contains(": barrier")).count();
    assert_eq!((dispatches, barriers), (5, 4), "{GOLDEN}");
    let buffers = GOLDEN.lines().filter(|l| l.starts_with("buffer ")).count();
    assert_eq!(buffers, 7, "no buffer for the chain's `Sub`: {GOLDEN}");
}

#[test]
fn the_golden_shows_both_plans_and_no_completion_point() {
    assert!(GOLDEN.contains("\nplan step\n"), "{GOLDEN}");
    assert!(GOLDEN.contains("\nplan forward\n"), "{GOLDEN}");
    assert!(!GOLDEN.contains(": complete"), "{GOLDEN}");
    let loss_sum = GOLDEN
        .lines()
        .find(|l| l.contains(" loss_sum Mse "))
        .expect("the golden has the mse loss-sum kernel");
    assert!(
        loss_sum.ends_with("workgroup_size 64 workgroups [1, 1, 1]"),
        "{loss_sum}"
    );
}

const MNIST: &str = "examples/mnist_mlp.mspec";
const MNIST_GOLDEN: &str = include_str!("golden/mnist_mlp.kernel.ir");

/// The lines of `plan` in `text`, from its header to the next plan or the end.
fn plan_lines<'a>(text: &'a str, plan: &str) -> Vec<&'a str> {
    text.lines()
        .skip_while(|l| *l != format!("plan {plan}"))
        .skip(1)
        .take_while(|l| !l.starts_with("plan "))
        .collect()
}

/// The `kernel k<n>` line of `text`.
fn kernel_line(text: &str, n: usize) -> &str {
    text.lines()
        .find(|l| l.starts_with(&format!("kernel k{n} ")))
        .unwrap_or_else(|| panic!("no kernel k{n}"))
}

/// Story m1-fuse-elementwise acceptance 1, read off what `modelspec ir` prints.
#[test]
fn ir_stage_kernel_of_mnist_shows_the_fused_step_plan() {
    let output = modelspec(&["ir", MNIST, "--stage", "kernel"]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    let printed = text(&output.stdout);
    assert_eq!(printed, MNIST_GOLDEN);

    let step = plan_lines(&printed, "step");
    let dispatches = step.iter().filter(|l| l.contains(": dispatch ")).count();
    let barriers = step.iter().filter(|l| l.contains(": barrier")).count();
    // The `(y - t) * c` of the loss gradient is one fused elementwise chain since
    // story:fuse-elementwise-chains-and-gradient-sums: 11 dispatches and 8 barriers before it.
    // The relu gradient is the epilogue of the matmul that produces its adjoint since
    // story:step-sites-beat-pytorch: 10 dispatches and 7 barriers before that.
    assert_eq!((dispatches, barriers), (9, 6), "{printed}");
    let relu_grad = kernel_line(&printed, 3);
    assert!(
        relu_grad.contains(" matmul MatMul ") && relu_grad.contains(" epilogue ReluGrad "),
        "{relu_grad}"
    );
    assert!(!printed.contains(" elementwise ReluGrad "), "{printed}");
    let chains: Vec<&str> = printed
        .lines()
        .filter(|l| l.starts_with("kernel ") && l.contains(" elementwise "))
        .filter(|l| {
            l.split(" elementwise ")
                .nth(1)
                .and_then(|rest| rest.split(' ').next())
                .is_some_and(|op| op.contains('>'))
        })
        .collect();
    let [chain] = chains[..] else {
        panic!("one fused elementwise chain: {chains:#?}\n{printed}");
    };
    assert!(
        chain.contains(
            " elementwise Sub>Mul@0 f32 (Tensor<f32, [100, 10]>, Tensor<f32, [100, 10]>, 0.002) \
             -> Tensor<f32, [100, 10]> "
        ),
        "{chain}"
    );
    assert!(!printed.contains(" elementwise Sub f32 "), "{printed}");
    assert!(!printed.contains(" elementwise Mul f32 "), "{printed}");
    // Each transpose of the step is read only by matmuls, which read it in place (DESIGN.md § 3.1).
    assert!(!printed.contains(" transpose Transpose "), "{printed}");
    assert_eq!(printed.matches("> transposed").count(), 3, "{printed}");

    for line in printed.lines().filter(|l| l.starts_with("kernel ")) {
        let operands = line
            .split_once(" (")
            .and_then(|(_, rest)| rest.split_once(") -> "))
            .map(|(operands, _)| operands)
            .expect("a kernel line lists its operands");
        assert!(
            operands.contains("Tensor<"),
            "a kernel with compile-time scalars alone: {line}"
        );
    }

    let k0 = kernel_line(&printed, 0);
    assert!(
        k0.contains(" matmul MatMul ") && k0.contains(" epilogue Relu writes pre-activation "),
        "{k0}"
    );
    assert!(!printed.contains(" elementwise Relu "), "{printed}");
    let first = step
        .iter()
        .find(|l| l.contains(": dispatch "))
        .expect("a dispatch");
    assert!(first.contains(": dispatch k0 "), "{first}");
    let writes = first
        .split_once(" writes ")
        .map(|(_, w)| w)
        .expect("k0 writes");
    assert_eq!(
        writes.matches("[f32 x 6400,").count(),
        2,
        "k0 writes the pre-activation and the activation: {first}"
    );

    let updates: Vec<&str> = printed
        .lines()
        .filter(|l| l.contains(" sgd_update "))
        .collect();
    assert_eq!(
        updates.len(),
        2,
        "one sgd_update per parameter: {updates:?}"
    );
    for update in &updates {
        assert!(
            update.contains(" sgd_update Sub f32 (Tensor<") && update.contains(", 0.1, Tensor<"),
            "the update reads p, lr and the gradient: {update}"
        );
    }
    assert!(
        !printed.contains(" elementwise Mul f32 (0.1, "),
        "no separate lr * g kernel: {printed}"
    );
}

/// The `flop <n> bytes <n>` of a dispatch line, or `None` when it has none.
fn counts(line: &str) -> Option<(u64, u64)> {
    let (_, rest) = line.split_once(" flop ")?;
    let mut words = rest.split(' ');
    let flop = words.next()?.parse().ok()?;
    (words.next()? == "bytes").then_some(())?;
    Some((flop, words.next()?.parse().ok()?))
}

/// story:profile-achieved-throughput: `ir --stage kernel` shows the FLOP and byte count of every
/// dispatch of both plans. The five matmuls of the MNIST step are `2·m·k·n` each:
/// 10,035,200 for `x · w1` and for the `w1` gradient, 128,000 for each of the other three.
#[test]
fn ir_stage_kernel_of_mnist_shows_the_counts_of_every_dispatch() {
    let output = modelspec(&["ir", MNIST, "--stage", "kernel"]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    let printed = text(&output.stdout);
    let kinds: std::collections::BTreeMap<String, String> = printed
        .lines()
        .filter_map(|l| l.strip_prefix("kernel "))
        .map(|l| {
            let words: Vec<&str> = l.split(' ').collect();
            (words[0].to_owned(), words[2].to_owned())
        })
        .collect();
    for plan in ["step", "forward"] {
        for line in plan_lines(&printed, plan) {
            if line.contains(": dispatch ") {
                assert!(counts(line).is_some(), "{plan}: {line}");
            } else {
                assert!(!line.contains(" flop "), "{plan}: {line}");
            }
        }
    }
    let matmuls: Vec<u64> = plan_lines(&printed, "step")
        .into_iter()
        .filter_map(|line| {
            let kernel = line.split_once(": dispatch ")?.1.split(' ').next()?;
            (kinds[kernel] == "matmul").then(|| counts(line).expect("counted").0)
        })
        .collect();
    assert_eq!(
        matmuls,
        [10_035_200, 128_000, 128_000, 128_000, 10_035_200],
        "{printed}"
    );
    assert_eq!(matmuls.iter().sum::<u64>(), 20_454_400);
    let first = plan_lines(&printed, "step")[0];
    assert_eq!(counts(first), Some((10_035_200, 565_504)), "{first}");
}

#[test]
fn the_mnist_golden_is_the_printed_plan_of_the_vulkan_artifact() {
    let source = std::fs::read_to_string(repo_root().join(MNIST)).expect("the example exists");
    let artifact =
        compile::compile_artifact(MNIST, &source, Backend::Vulkan).expect("the example compiles");
    assert_eq!(print_kernel(&artifact.execution), MNIST_GOLDEN);
}

#[test]
fn ir_stage_kernels_is_a_usage_error() {
    let output = modelspec(&["ir", EXAMPLE, "--stage", "kernels"]);
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(text(&output.stdout), "");
    let stderr = text(&output.stderr);
    assert!(stderr.contains("'kernels'"), "{stderr}");
    assert!(
        stderr.contains("[possible values: tensor, training, kernel]"),
        "{stderr}"
    );
}

/// A `[4, 1073741824]` f32 parameter is 16 GiB, past the `u32` byte range of a storage buffer
/// binding, and its forward needs more invocations than a 32-bit index counts.
#[test]
fn ir_stage_kernel_refuses_a_tensor_past_the_u32_limits_with_its_target_code() {
    let source = std::fs::read_to_string(repo_root().join(EXAMPLE))
        .expect("the example exists")
        .replace("[4, 1]", "[4, 1073741824]")
        .replace("[8, 1]", "[8, 1073741824]");
    assert!(source.contains("Tensor<f32, [4, 1073741824]>"), "{source}");
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("u32_limit.mspec");
    std::fs::write(&path, source).expect("the fixture is written");
    let path = path.to_str().expect("a UTF-8 path");

    assert_eq!(modelspec(&["check", path]).status.code(), Some(0));
    let output = modelspec(&["ir", path, "--stage", "kernel"]);
    assert_eq!(output.status.code(), Some(1), "{}", text(&output.stdout));
    assert_eq!(text(&output.stdout), "");
    let stderr = text(&output.stderr);
    assert!(stderr.contains("E-TARGET-010"), "{stderr}");
}

#[test]
fn ir_stage_kernel_reports_diagnostics_as_check_does() {
    let fixture = "tests/language/check/matmul_mismatch.mspec";
    let check = modelspec(&["check", fixture]);
    let ir = modelspec(&["ir", fixture, "--stage", "kernel"]);
    assert_eq!(check.status.code(), Some(1));
    assert_eq!(ir.status.code(), Some(1));
    assert_eq!(text(&ir.stdout), "");
    assert_eq!(text(&ir.stderr), text(&check.stderr));
}
