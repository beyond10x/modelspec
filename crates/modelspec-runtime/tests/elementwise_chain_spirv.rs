//! story:fuse-elementwise-chains-and-gradient-sums, no Vulkan instance: the module of a fused
//! elementwise chain (DESIGN.md § 3.1) passes `spirv-val --target-env vulkan1.2`, decorates
//! every `OpFAdd`, `OpFSub` and `OpFMul` it computes `NoContraction` in the deterministic mode,
//! so no driver contracts `a * b + c` into a fused multiply-add the unfused kernels never
//! computed, and drops every decoration under `fast_math`; a kernel op that names no canonical
//! chain does not emit. Needs `spirv-val` on the path.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::path::PathBuf;

use common::words::{
    DECORATION_NO_CONTRACTION, OP_DECORATE, instructions, uncontracted_arithmetic,
};
use modelspec_core::artifact::Backend;
use modelspec_core::compile::compile_artifact;
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{Kernel, KernelKind, KernelModuleId, Operand};
use modelspec_core::ir::tensor::{DType, Dim, TensorType};
use modelspec_runtime::spirv::emit::{emit, emit_kernels};
use modelspec_runtime::spirv::validate::validate;

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

fn tensor(count: u64) -> Operand {
    Operand::Buffer(TensorType::f32(&[count]))
}

fn kernel(op: &'static str, operands: Vec<Operand>, count: u64) -> Kernel {
    let kind = KernelKind::Elementwise;
    let groups = count.div_ceil(u64::from(kind.workgroup_size()));
    let mut kernel = Kernel {
        id: KernelModuleId([0; 16]),
        kind,
        op,
        dtype: DType::F32,
        operands,
        result: TensorType {
            dtype: DType::F32,
            shape: vec![Dim::Static(count)],
        },
        reduces: None,
        epilogue: None,
        transposed: [false, false],
        softmax_stats: None,
        workgroup_size: kind.workgroup_size(),
        workgroups: [u32::try_from(groups).expect("one dimension"), 1, 1],
    };
    kernel.id = Kernel::module_id(&kernel.signature());
    kernel
}

/// Each chain, its operands over `count` elements, and how many `OpFAdd`, `OpFSub` and `OpFMul`
/// its module computes.
fn chains(count: u64) -> Vec<(&'static str, Vec<Operand>, usize)> {
    let t = || tensor(count);
    let s = || Operand::Buffer(TensorType::f32_scalar());
    vec![
        ("Sub>Mul@0", vec![t(), t(), Operand::Immediate(0.002)], 2),
        ("Add>Add@0", vec![t(), t(), t()], 2),
        ("Add>ReluGrad@1", vec![t(), t(), t()], 1),
        ("Mul>Add@0", vec![t(), t(), t()], 2),
        ("Mul>Sub@1", vec![t(), t(), t()], 2),
        ("Relu>Mul@1", vec![t(), s()], 1),
        ("Relu>Add@1", vec![t(), t()], 1),
        (
            "Sub>Mul@0>Add@1",
            vec![t(), t(), Operand::Immediate(0.375), t()],
            3,
        ),
        ("Mul>Add@0>Relu@0", vec![t(), t(), t()], 2),
        ("Add>Add@0", vec![Operand::Immediate(-1.25), t(), s()], 2),
    ]
}

#[test]
fn every_chain_module_passes_spirv_val_with_every_float_op_uncontracted() {
    let mut modules = 0;
    for count in [1, 64, 65, 4_097] {
        for (op, operands, arithmetic) in chains(count) {
            let k = kernel(op, operands, count);
            let words = emit(&k).unwrap_or_else(|e| panic!("{}: {e}", k.signature()));
            if let Err(e) = validate(&words) {
                panic!("{}: {e}", k.signature());
            }
            let (missing, all) = uncontracted_arithmetic(&words);
            assert_eq!(all, arithmetic, "{}: float arithmetic", k.signature());
            assert!(
                missing.is_empty(),
                "{}: no NoContraction on {missing:?}",
                k.signature()
            );
            modules += 1;
        }
    }
    eprintln!("elementwise_chain_spirv: {modules} chain modules pass spirv-val");
}

#[test]
fn a_chain_kernel_has_its_own_module_id_and_single_ops_keep_theirs() {
    let chain = kernel(
        "Sub>Mul@0",
        vec![tensor(8), tensor(8), Operand::Immediate(0.25)],
        8,
    );
    let sub = kernel("Sub", vec![tensor(8), tensor(8)], 8);
    assert_ne!(chain.id, sub.id);
    assert!(
        chain.signature().starts_with("elementwise Sub>Mul@0 f32 "),
        "{}",
        chain.signature()
    );
    let words = emit(&sub).expect("a single op emits");
    let (missing, all) = uncontracted_arithmetic(&words);
    assert_eq!(
        (missing.len(), all),
        (1, 1),
        "a single elementwise op stays undecorated, its words unchanged"
    );
}

#[test]
fn an_op_that_names_no_canonical_chain_does_not_emit() {
    let t = || tensor(8);
    for (op, operands) in [
        ("Sub>Mul", vec![t(), t(), t()]),
        ("Sub>Mul@2", vec![t(), t(), t()]),
        ("Sub>Mul@0>", vec![t(), t(), t()]),
        ("Mul@0>Sub@0", vec![t(), t(), t()]),
        ("Sub>Relu@1", vec![t(), t()]),
        ("Sub>Foo@0", vec![t(), t(), t()]),
        ("Sub>Mul@00", vec![t(), t(), t()]),
        ("Sub>Mul@0", vec![t(), t()]),
        ("Sub>Mul@0", vec![t(), t(), t(), t()]),
        ("Sub>MatMul@0", vec![t(), t(), t()]),
    ] {
        let k = kernel(op, operands, 8);
        assert!(emit(&k).is_err(), "{} emitted", k.signature());
    }
}

fn no_contraction_count(module: &[u32]) -> usize {
    instructions(module)
        .into_iter()
        .filter(|(op, w)| *op == OP_DECORATE && w[2] == DECORATION_NO_CONTRACTION)
        .count()
}

#[test]
fn the_fast_math_chain_module_drops_every_no_contraction_and_nothing_else() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source =
        std::fs::read_to_string(root.join("examples/mnist_mlp.mspec")).expect("the example reads");
    let deterministic =
        compile_artifact("mnist_mlp.mspec", &source, Backend::Vulkan).expect("it compiles");
    let mut fast = deterministic.clone();
    fast.fast_math = true;
    let profile = DeviceProfile::from_json(FIXTURE).expect("the fixture is a profile");
    let off = emit_kernels(&deterministic, &profile).expect("emits");
    let on = emit_kernels(&fast, &profile).expect("emits");
    let mut chains = 0;
    for ((kernel, off), on) in deterministic.execution.kernels.iter().zip(&off).zip(&on) {
        if !kernel.op.contains('>') {
            continue;
        }
        chains += 1;
        let (_, all) = uncontracted_arithmetic(&off.spirv);
        assert_eq!(
            no_contraction_count(&off.spirv),
            all,
            "{}",
            kernel.signature()
        );
        assert_eq!(no_contraction_count(&on.spirv), 0, "{}", kernel.signature());
        assert_eq!(
            uncontracted_arithmetic(&on.spirv).1,
            all,
            "{}",
            kernel.signature()
        );
        assert_eq!(
            on.spirv.len() + 3 * all,
            off.spirv.len(),
            "{}",
            kernel.signature()
        );
    }
    assert_eq!(chains, 1, "the MNIST step has one chain kernel");
}
