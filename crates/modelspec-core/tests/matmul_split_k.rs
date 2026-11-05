//! `story:vulkan-matmul-split-k` (DESIGN.md § 5.2, § 5.4): a matmul with a long `k` and a small
//! output is two dispatches, the partial products of the slices of `k` and their sum.
//!
//! - [`matmul_split_k_slices`] cuts a `k` of at least [`MATMUL_SPLIT_K_FROM`] and below
//!   [`MATMUL_SPLIT_K_BELOW`] that is a multiple of [`MATMUL_SPLIT_K_SLICE`] into at most
//!   [`MATMUL_SPLIT_K_MAX_SLICES`] slices.
//! - The step plan of `examples/bert_tiny.mspec` has 13 split matmul dispatches, each followed by
//!   one `SumRows` dispatch that reads its partial products and writes the matmul's value.
//! - A step without an operator of the tolerance class keeps every matmul whole, and so does a
//!   matmul with an epilogue.
//! - The plan scheduled within a device's limits splits the same matmuls into the same slices.

use std::path::{Path, PathBuf};

use modelspec_core::artifact::Backend;
use modelspec_core::compile::{CompiledTraining, compile_artifact, compile_training};
use modelspec_core::ir::kernel::{
    ExecutionPlan, Kernel, KernelKind, MATMUL_SPLIT_K_BELOW, MATMUL_SPLIT_K_FROM,
    MATMUL_SPLIT_K_MAX_SLICES, MATMUL_SPLIT_K_SLICE, MatMulLimits, Operand, PlanOp, PlanStep,
};
use modelspec_core::ir::tensor::{Dim, TensorType};
use modelspec_core::ir::verify::verify_plan;
use modelspec_core::schedule::{matmul_split_k_slices, schedule, schedule_for};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn bert_tiny() -> ExecutionPlan {
    let source = std::fs::read_to_string(repo_root().join("examples/bert_tiny.mspec"))
        .expect("the example reads");
    match compile_artifact("bert_tiny.mspec", &source, Backend::Vulkan) {
        Ok(artifact) => artifact.execution,
        Err(diagnostics) => panic!("compile_artifact refused the example: {diagnostics:#?}"),
    }
}

/// `examples/linear_regression.mspec` at `[8, 4096] x [4096, 1]` with `forward` returning
/// `body`.
fn long_k(body: &str) -> CompiledTraining {
    let source = std::fs::read_to_string(repo_root().join("examples/linear_regression.mspec"))
        .expect("the example reads")
        .replace("[4, 1]", "[4096, 1]")
        .replace("[8, 4]", "[8, 4096]")
        .replace("matmul(x, w)", body);
    assert!(source.contains(body), "the example has the replaced call");
    compile_training("long_k.mspec", &source).expect("it compiles")
}

fn plan_of(compiled: &CompiledTraining) -> ExecutionPlan {
    schedule(
        &compiled.training,
        compiled.forward.prediction,
        &compiled.config,
    )
    .expect("the step schedules")
}

fn kernel_of<'a>(plan: &'a ExecutionPlan, step: &PlanStep) -> Option<&'a Kernel> {
    if step.op != PlanOp::Dispatch {
        return None;
    }
    plan.kernel(step.kernel_module_id?)
}

/// The positions in `plan.step` of the dispatches of a matmul kernel with more than one slice,
/// each with its slices.
fn split(plan: &ExecutionPlan) -> Vec<(usize, u64)> {
    plan.step
        .iter()
        .enumerate()
        .filter_map(|(at, step)| {
            let slices = kernel_of(plan, step)?.matmul_slices()?;
            (slices > 1).then_some((at, slices))
        })
        .collect()
}

fn matmuls(plan: &ExecutionPlan) -> Vec<&Kernel> {
    plan.kernels
        .iter()
        .filter(|kernel| kernel.kind == KernelKind::MatMul)
        .collect()
}

#[test]
fn a_long_k_is_cut_into_slices_of_128_and_at_most_64_of_them() {
    assert_eq!(
        (
            MATMUL_SPLIT_K_FROM,
            MATMUL_SPLIT_K_BELOW,
            MATMUL_SPLIT_K_SLICE,
            MATMUL_SPLIT_K_MAX_SLICES
        ),
        (4096, 16384, 128, 64)
    );
    for (k, slices) in [
        (4096, Some(32)),
        (4224, Some(33)),
        (8192, Some(64)),
        (8448, Some(33)),
        (12288, Some(48)),
        (16128, Some(63)),
        // From the bound of the long-`k` tiles on.
        (16384, None),
        (16512, None),
        (65536, None),
        // Below the bound.
        (128, None),
        (2048, None),
        (3968, None),
        (4095, None),
        // No multiple of a slice.
        (4097, None),
        (4160, None),
        (8191, None),
        // 65 and 127 slices of 128 are odd and above the bound.
        (65 * 128, None),
        (127 * 128, None),
    ] {
        assert_eq!(matmul_split_k_slices(k), slices, "k {k}");
        if let Some(slices) = slices {
            assert!(k.is_multiple_of(slices), "k {k}");
            assert!((k / slices).is_multiple_of(MATMUL_SPLIT_K_SLICE), "k {k}");
        }
    }
}

#[test]
fn bert_tiny_has_13_split_matmuls_each_summed_by_one_dispatch() {
    let plan = bert_tiny();
    verify_plan(&plan).unwrap_or_else(|error| panic!("the plan verifies: {error}"));
    let split = split(&plan);
    assert_eq!(split.len(), 13, "{split:?}");
    let mut shapes = std::collections::BTreeMap::<String, usize>::new();
    for &(at, slices) in &split {
        let step = &plan.step[at];
        let kernel = kernel_of(&plan, step).expect("a dispatch of a kernel");
        let (m, k, n) = Kernel::matmul_shape(&kernel.operands).expect("a matmul");
        assert_eq!(
            Some(slices),
            matmul_split_k_slices(k),
            "{}",
            kernel.signature()
        );
        assert_eq!(
            kernel.result,
            TensorType::f32(&[slices * m, n]),
            "{}",
            kernel.signature()
        );
        assert_eq!(kernel.epilogue, None);
        assert_eq!(kernel.operands.len(), 2);
        assert_eq!(kernel.id, Kernel::module_id(&kernel.signature()));
        // The tile is that of the matmul of one slice, laid over every slice.
        let tile = kernel.matmul_tile().expect("a tile");
        assert_eq!(
            u64::from(kernel.workgroups[0]),
            slices * tile.workgroups(m, n),
            "{}",
            kernel.signature()
        );
        assert_eq!(kernel.workgroup_size, tile.workgroup_size);
        assert_eq!(step.reads.len(), 2);
        assert_eq!(step.writes.len(), 1, "the partial products");
        // One later dispatch reads the partial products, and it is the sum of the slices.
        let readers: Vec<usize> = plan
            .step
            .iter()
            .enumerate()
            .filter(|(_, later)| {
                later.op == PlanOp::Dispatch && later.reads.contains(&step.writes[0])
            })
            .map(|(reader, _)| reader)
            .collect();
        assert_eq!(readers.len(), 1, "step {at}: {readers:?}");
        let sum = &plan.step[readers[0]];
        assert!(readers[0] > at);
        assert_eq!(sum.value, step.value, "both stages compute one value");
        assert_eq!(sum.reads, step.writes);
        assert_eq!(sum.writes.len(), 1);
        assert_ne!(sum.writes, step.writes);
        let stage2 = kernel_of(&plan, sum).expect("a dispatch of a kernel");
        assert_eq!(stage2.op, "SumRows");
        assert_eq!(
            stage2.operands,
            vec![Operand::Buffer(TensorType::f32(&[slices, m * n]))]
        );
        assert_eq!(stage2.result, TensorType::f32(&[m * n]));
        *shapes
            .entry(format!("[{m}, {k}] x [{k}, {n}] in {slices}"))
            .or_default() += 1;
    }
    let shapes: Vec<(&str, usize)> = shapes.iter().map(|(shape, n)| (&shape[..], *n)).collect();
    assert_eq!(
        shapes,
        vec![
            // The weight gradients of the eight attention projections and of the head's dense
            // layer, then those of each block's two feed-forward layers.
            ("[128, 4096] x [4096, 128] in 32", 9),
            ("[128, 4096] x [4096, 512] in 32", 2),
            ("[512, 4096] x [4096, 128] in 32", 2),
        ]
    );
    // Every other matmul of the plan writes its own `[m, n]`.
    for kernel in matmuls(&plan) {
        let (m, _, n) = Kernel::matmul_shape(&kernel.operands).expect("a matmul");
        let rows = match kernel.result.shape[..] {
            [Dim::Static(rows), Dim::Static(cols)] if cols == n => rows,
            _ => panic!("{}", kernel.signature()),
        };
        assert_eq!(
            kernel.matmul_slices(),
            Some(rows / m),
            "{}",
            kernel.signature()
        );
    }
}

#[test]
fn the_limits_of_a_device_change_no_split() {
    let source = std::fs::read_to_string(repo_root().join("examples/bert_tiny.mspec"))
        .expect("the example reads");
    let compiled = compile_training("bert_tiny.mspec", &source).expect("it compiles");
    // Each split site: its value, its shape and its slices.
    let sites = |plan: &ExecutionPlan| -> Vec<String> {
        split(plan)
            .into_iter()
            .map(|(at, slices)| {
                let step = &plan.step[at];
                let kernel = kernel_of(plan, step).expect("a dispatch of a kernel");
                format!(
                    "{:?} {:?} {:?} in {slices}",
                    step.value,
                    Kernel::matmul_shape(&kernel.operands),
                    kernel.transposed
                )
            })
            .collect()
    };
    let every_device = sites(&plan_of(&compiled));
    assert_eq!(every_device.len(), 13);
    // The RTX 3090's limits, then workgroup memory for no block of 256 invocations.
    for (invocations, shared_bytes) in [(1024, 49152), (256, 16384), (128, 16384)] {
        let plan = schedule_for(
            &compiled.training,
            compiled.forward.prediction,
            &compiled.config,
            Some(MatMulLimits {
                invocations,
                shared_bytes,
            }),
        )
        .expect("the step schedules within the limits");
        assert_eq!(sites(&plan), every_device, "{invocations} {shared_bytes}");
    }
}

#[test]
fn a_step_without_a_tolerance_class_operator_keeps_its_long_k_matmul_whole() {
    let plan = plan_of(&long_k("matmul(x, w)"));
    verify_plan(&plan).unwrap_or_else(|error| panic!("the plan verifies: {error}"));
    assert_eq!(split(&plan), vec![]);
    let forward: Vec<&Kernel> = matmuls(&plan)
        .into_iter()
        .filter(|kernel| Kernel::matmul_shape(&kernel.operands) == Some((8, 4096, 1)))
        .collect();
    assert_eq!(forward.len(), 1);
    assert_eq!(forward[0].result, TensorType::f32(&[8, 1]));
    assert_eq!(forward[0].matmul_slices(), Some(1));
}

#[test]
fn the_same_matmul_is_split_in_a_step_with_a_gelu() {
    let plan = plan_of(&long_k("matmul(gelu(x), w)"));
    verify_plan(&plan).unwrap_or_else(|error| panic!("the plan verifies: {error}"));
    let split = split(&plan);
    assert_eq!(split.len(), 1, "{split:?}");
    let (at, slices) = split[0];
    assert_eq!(slices, 32);
    let kernel = kernel_of(&plan, &plan.step[at]).expect("a dispatch of a kernel");
    assert_eq!(
        Kernel::matmul_shape(&kernel.operands),
        Some((8, 4096, 1)),
        "{}",
        kernel.signature()
    );
    assert_eq!(kernel.result, TensorType::f32(&[256, 1]));
}

#[test]
fn a_matmul_with_an_epilogue_is_not_split() {
    let plan = plan_of(&long_k("relu(matmul(gelu(x), w))"));
    verify_plan(&plan).unwrap_or_else(|error| panic!("the plan verifies: {error}"));
    assert_eq!(split(&plan), vec![]);
    let forward: Vec<&Kernel> = matmuls(&plan)
        .into_iter()
        .filter(|kernel| Kernel::matmul_shape(&kernel.operands) == Some((8, 4096, 1)))
        .collect();
    assert_eq!(forward.len(), 1);
    assert!(forward[0].epilogue.is_some(), "{}", forward[0].signature());
    assert_eq!(forward[0].result, TensorType::f32(&[8, 1]));
}
