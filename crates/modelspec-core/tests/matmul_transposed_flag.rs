//! `story:matmul-reads-transposed-operand`, acceptance `transposed-operand-flag`: the matmul
//! kernel record states per operand whether it is read transposed, the printed kernel line shows
//! it, the signature and module id change with it, and the tile is still `MatMulTile::of` of
//! the logical `(m, k, n)`.

use std::path::{Path, PathBuf};

use modelspec_core::artifact::Backend;
use modelspec_core::compile::compile_artifact;
use modelspec_core::ir::kernel::{ExecutionPlan, Kernel, KernelKind, MatMulTile};
use modelspec_core::ir::print::print_kernel;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn mnist_plan() -> ExecutionPlan {
    let source = std::fs::read_to_string(repo_root().join("examples/mnist_mlp.mspec"))
        .expect("the example reads");
    compile_artifact("examples/mnist_mlp.mspec", &source, Backend::Vulkan)
        .expect("the example compiles")
        .execution
}

/// The MNIST matmul kernel `[m, k] x [k, n]`.
fn matmul(plan: &ExecutionPlan, m: u64, k: u64, n: u64) -> &Kernel {
    plan.kernels
        .iter()
        .find(|kernel| {
            kernel.kind == KernelKind::MatMul
                && Kernel::matmul_shape(&kernel.operands) == Some((m, k, n))
        })
        .unwrap_or_else(|| panic!("a [{m}, {k}] x [{k}, {n}] matmul"))
}

#[test]
fn each_mnist_matmul_states_which_operand_it_reads_transposed() {
    let plan = mnist_plan();
    for ((m, k, n), transposed) in [
        ((100, 784, 64), [false, false]),
        ((100, 64, 10), [false, false]),
        ((100, 10, 64), [false, true]),
        ((64, 100, 10), [true, false]),
        ((784, 100, 64), [true, false]),
    ] {
        let kernel = matmul(&plan, m, k, n);
        assert_eq!(kernel.transposed, transposed, "{}", kernel.signature());
    }
    for kernel in plan.kernels.iter().filter(|k| k.kind != KernelKind::MatMul) {
        assert_eq!(kernel.transposed, [false, false], "{}", kernel.signature());
    }
}

#[test]
fn the_flag_is_printed_on_its_operand_and_changes_the_signature_and_module_id() {
    let plan = mnist_plan();
    let printed = print_kernel(&plan);
    let read_a = matmul(&plan, 784, 100, 64);
    let read_b = matmul(&plan, 100, 10, 64);
    for (kernel, operand) in [
        (
            read_a,
            "(Tensor<f32, [784, 100]> transposed, Tensor<f32, [100, 64]>)",
        ),
        (
            read_b,
            "(Tensor<f32, [100, 10]>, Tensor<f32, [10, 64]> transposed)",
        ),
    ] {
        let signature = kernel.signature();
        assert!(signature.contains(operand), "{signature}");
        let line = format!("{} {signature}", kernel.id);
        assert!(
            printed.lines().any(|l| l.ends_with(&line)),
            "printed: {line}\n{printed}"
        );
        assert_eq!(kernel.id, Kernel::module_id(&signature));

        let mut plain = kernel.clone();
        plain.transposed = [false, false];
        let plain_signature = plain.signature();
        assert!(!plain_signature.contains("transposed"), "{plain_signature}");
        assert_ne!(plain_signature, signature);
        assert_ne!(Kernel::module_id(&plain_signature), kernel.id);

        let (m, k, n) = Kernel::matmul_shape(&kernel.operands).expect("a static matmul");
        let tile = MatMulTile::of(m, k, n);
        assert!(signature.ends_with(&format!(" {tile}")), "{signature}");
        assert!(
            plain_signature.ends_with(&format!(" {tile}")),
            "{plain_signature}"
        );
    }
    let mut both = read_a.clone();
    both.transposed = [true, true];
    assert!(
        both.signature()
            .contains("(Tensor<f32, [784, 100]> transposed, Tensor<f32, [100, 64]> transposed)"),
        "{}",
        both.signature()
    );
    assert_ne!(Kernel::module_id(&both.signature()), read_a.id);
}

#[test]
fn a_matmul_reading_no_operand_transposed_keeps_its_signature() {
    let plan = mnist_plan();
    let forward = matmul(&plan, 100, 784, 64);
    assert!(!forward.signature().contains("transposed"));
    assert_eq!(
        forward.id.to_string(),
        "6bd32d9c-1c0c-8fd4-96b2-46d2c722adc9",
        "the module id the forward matmul had before the change"
    );
}
