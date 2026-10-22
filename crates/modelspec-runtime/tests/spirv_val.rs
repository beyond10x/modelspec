//! DESIGN.md § 9 gate 6: every elementwise, broadcast and SGD module the emitter builds passes
//! `spirv-val --target-env vulkan1.2`, in `task ci`. No test here creates a Vulkan instance; each
//! needs `spirv-val` on the path and fails, not skips, without it.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::path::PathBuf;

use modelspec_core::artifact::Backend;
use modelspec_core::compile::compile_artifact;
use modelspec_core::ir::kernel::{Kernel, KernelKind, Operand};
use modelspec_core::ir::tensor::TensorType;
use modelspec_runtime::spirv::emit::{EmitError, emit};
use modelspec_runtime::spirv::validate::{TARGET_ENV, ValidateError, validate, validate_with};

/// The SPIR-V opcode of `OpReturn`.
const OP_RETURN: u32 = 253;

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The elementwise, broadcast and SGD kernels of the Vulkan artifact of `path`.
fn kernels_of(path: &str) -> Vec<Kernel> {
    let source = std::fs::read_to_string(repository().join(path)).expect("the example reads");
    let artifact = compile_artifact(path, &source, Backend::Vulkan).expect("the example compiles");
    artifact
        .execution
        .kernels
        .into_iter()
        .filter(|kernel| {
            matches!(
                kernel.kind,
                KernelKind::Elementwise | KernelKind::Broadcast | KernelKind::SgdUpdate
            )
        })
        .collect()
}

fn assert_valid(kernel: &Kernel) {
    let words = emit(kernel).unwrap_or_else(|error| panic!("{}: {error}", kernel.signature()));
    if let Err(error) = validate(&words) {
        panic!("{}: {error}", kernel.signature());
    }
}

#[test]
fn spirv_val_accepts_every_elementwise_broadcast_and_sgd_module_of_the_examples() {
    let mut kinds = Vec::new();
    let mut ops = Vec::new();
    let mut total = 0;
    for path in [
        "examples/linear_regression.mspec",
        "crates/modelspec-docs/models/two_layer_relu.mspec",
    ] {
        let kernels = kernels_of(path);
        assert!(!kernels.is_empty(), "{path} has elementwise kernels");
        for kernel in &kernels {
            assert_valid(kernel);
            kinds.push(kernel.kind);
            ops.push(kernel.op);
        }
        total += kernels.len();
    }
    for kind in [KernelKind::Elementwise, KernelKind::SgdUpdate] {
        assert!(kinds.contains(&kind), "the examples have a {kind} kernel");
    }
    for op in ["Sub", "Mul", "Relu", "ReluGrad"] {
        assert!(ops.contains(&op), "the examples have a {op} kernel");
    }
    eprintln!("spirv_val: {total} example modules pass spirv-val --target-env {TARGET_ENV}");
}

#[test]
fn spirv_val_accepts_every_form_of_the_catalog_at_every_test_size() {
    let mut total = 0;
    for count in [1, 63, 64, 65, 4_097, 78_400] {
        for case in common::catalog() {
            assert_valid(&common::kernel(&case, count));
            total += 1;
        }
    }
    eprintln!("spirv_val: {total} catalog modules pass spirv-val --target-env {TARGET_ENV}");
}

/// `words` with its first `OpReturn` removed.
fn without_return(words: &[u32]) -> Vec<u32> {
    let mut out = words[..5].to_vec();
    let mut at = 5;
    let mut removed = false;
    while at < words.len() {
        let length = (words[at] >> 16) as usize;
        let opcode = words[at] & 0xffff;
        assert!(length > 0, "an instruction has a length");
        if opcode == OP_RETURN && !removed {
            removed = true;
        } else {
            out.extend_from_slice(&words[at..at + length]);
        }
        at += length;
    }
    assert!(removed, "the module has an OpReturn");
    out
}

#[test]
fn spirv_val_refuses_a_module_with_one_instruction_removed() {
    let kernel = common::kernel(&common::catalog()[0], 65);
    let words = emit(&kernel).expect("the kernel emits");
    validate(&words).expect("the whole module is valid");
    match validate(&without_return(&words)) {
        Err(ValidateError::Invalid { output }) => assert!(!output.is_empty()),
        other => panic!("a module without its OpReturn passed or did not run: {other:?}"),
    }
}

#[test]
fn spirv_val_absent_is_an_error_not_a_pass() {
    let words = emit(&common::kernel(&common::catalog()[0], 1)).expect("the kernel emits");
    match validate_with("modelspec-no-such-spirv-val".as_ref(), &words) {
        Err(ValidateError::Unavailable { program, .. }) => {
            assert_eq!(program, "modelspec-no-such-spirv-val");
        }
        other => panic!("a missing spirv-val was not an error: {other:?}"),
    }
}

#[test]
fn emit_refuses_a_kernel_outside_elementwise_broadcast_and_sgd() {
    let mut kernel = common::kernel(&common::catalog()[0], 8);
    kernel.kind = KernelKind::MatMul;
    kernel.op = "MatMul";
    kernel.operands = vec![
        Operand::Buffer(TensorType::f32(&[2, 4])),
        Operand::Buffer(TensorType::f32(&[4, 4])),
    ];
    assert!(matches!(emit(&kernel), Err(EmitError::Unsupported { .. })));
}

#[test]
fn emit_declares_the_workgroup_size_of_the_kind_and_no_float_control_mode() {
    let kernel = common::kernel(&common::catalog()[0], 65);
    let words = emit(&kernel).expect("the kernel emits");
    let mut at = 5;
    let mut local_size = None;
    let mut modes = Vec::new();
    while at < words.len() {
        let length = (words[at] >> 16) as usize;
        // OpExecutionMode: entry point, mode, operands.
        if words[at] & 0xffff == 16 {
            modes.push(words[at + 2]);
            // ExecutionMode LocalSize is 17.
            if words[at + 2] == 17 {
                local_size = Some([words[at + 3], words[at + 4], words[at + 5]]);
            }
        }
        at += length;
    }
    assert_eq!(local_size, Some([64, 1, 1]));
    assert_eq!(modes, [17], "LocalSize is the only execution mode");
    assert_eq!(
        words[1], 0x0001_0500,
        "SPIR-V 1.5, the version of Vulkan 1.2"
    );
}
