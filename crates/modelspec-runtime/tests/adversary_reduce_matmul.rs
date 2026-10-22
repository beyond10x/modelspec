//! Adversary pass 1 on `story:m1-vulkan-reduce-matmul`: device cases the unit's own suite does not
//! hold. The reduction, `mse` and transpose kernels are the ones `compile_artifact` schedules (not
//! hand-built copies), the partials buffer is the view of the plan's stage-1 dispatch, and the
//! inputs are chosen so that one missing, duplicated or misplaced term is far outside the
//! operator's gate-5 tolerance: sparse markers of one magnitude at the last element of every
//! stage-1 block, single one-hot terms at block and stage-2 boundaries, iota transposes, and
//! sparse integer matmuls. Counts reach past 64 * 64 partials and onto a two-dimensional stage-1
//! grid (`[65535, 2, 1]`).
//!
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test adversary_reduce_matmul -- --ignored`
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;

use common::{GUARD_WORDS, KernelOp, guard, guarded};
use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::compile_artifact;
use modelspec_core::ir::kernel::{BufferView, Kernel, KernelKind, PlanOp};
use modelspec_core::ir::tensor::{TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::interpreter::evaluate;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::{KernelModule, emit_loss_sum_undivided, emit_module};
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, Vulkan};

const UNWRITTEN: u32 = 0x7fc0_dead;

/// A training source whose loss is `loss` over a `[n, 1]` prediction.
fn source(n: u64, loss: &str) -> String {
    format!(
        "model R {{\n    param w: Tensor<f32, [{n}, 1]> = xavier_uniform();\n\n    \
         fn forward(x: Tensor<f32, [{n}, 1]>)\n        -> Tensor<f32, [{n}, 1]> {{\n        \
         return mul(x, w);\n    }}\n}}\n\ntrain Fit {{\n    model = R;\n    \
         input x: Tensor<f32, [{n}, 1]>;\n    target y: Tensor<f32, [{n}, 1]>;\n    \
         loss = {loss};\n    optimizer = sgd(lr = 0.1);\n    steps = 1;\n    seed = 42;\n    \
         device = vulkan(0);\n}}\n"
    )
}

fn artifact(n: u64, loss: &str) -> CompiledArtifact {
    compile_artifact("adv.mspec", &source(n, loss), Backend::Vulkan)
        .unwrap_or_else(|d| panic!("n={n} {loss}: {d:?}"))
}

/// The stage-1 and stage-2 kernels of the loss reduction over `[n, 1]`, and the partials view the
/// plan's stage-1 dispatch writes.
fn planned_reduction(n: u64, mean: bool) -> (Kernel, Kernel, BufferView) {
    let op = if mean { "mean" } else { "sum" };
    let artifact = artifact(n, &format!("{op}(mul(model.forward(x), y))"));
    let plan = &artifact.execution;
    let reduced = TensorType::f32(&[n, 1]);
    let stage1 = plan
        .kernels
        .iter()
        .find(|k| k.kind == KernelKind::ReduceStage1 && k.reduces.as_ref() == Some(&reduced))
        .unwrap_or_else(|| panic!("n={n}: a stage 1 over [{n}, 1]"))
        .clone();
    let stage2 = plan
        .kernels
        .iter()
        .find(|k| k.kind == KernelKind::ReduceStage2 && k.reduces.as_ref() == Some(&reduced))
        .expect("a stage 2")
        .clone();
    let step = plan
        .step
        .iter()
        .find(|s| s.op == PlanOp::Dispatch && s.kernel_module_id == Some(stage1.id))
        .expect("the stage-1 dispatch");
    let view = step.writes[0];
    (stage1, stage2, view)
}

fn planned_loss_sum(n: u64) -> Kernel {
    let artifact = artifact(n, "mse(model.forward(x), y)");
    artifact
        .execution
        .kernels
        .iter()
        .find(|k| k.kind == KernelKind::LossSum)
        .expect("a loss sum")
        .clone()
}

/// The CPU interpreter's result of `op` on `inputs`.
fn cpu(op: KernelOp, inputs: &[(Vec<u64>, Vec<f32>)]) -> Vec<f32> {
    let mut builder = TensorProgramBuilder::new();
    let mut tensors = BTreeMap::new();
    let mut values = Vec::new();
    for (i, (shape, data)) in inputs.iter().enumerate() {
        let name = format!("x{i}");
        tensors.insert(
            name.clone(),
            HostTensor::new(shape.clone(), data.clone()).expect("the input has its shape"),
        );
        values.push(builder.input(name, TensorType::f32(shape)).expect("new"));
    }
    let result = match op {
        KernelOp::ReduceSum => builder.reduce_sum(values[0]),
        KernelOp::ReduceMean => builder.reduce_mean(values[0]),
        KernelOp::Mse => builder.mse(values[0], values[1]),
        KernelOp::LossSum => {
            let d = builder.sub(values[0], values[1]).expect("same shapes");
            let square = builder.mul(d, d).expect("same shapes");
            builder.reduce_sum(square)
        }
        KernelOp::Transpose => builder.transpose(values[0]),
        KernelOp::MatMul => builder.matmul(values[0], values[1]),
    }
    .expect("well typed");
    builder.output(result).expect("a value");
    let program = builder.finish();
    let outputs = evaluate(&program, &tensors, &BTreeMap::new()).expect("the CPU evaluates");
    outputs[0].data().to_vec()
}

/// Per output element, the absolute sum of its terms and their number (DESIGN.md § 9 gate 5).
fn terms(op: KernelOp, inputs: &[(Vec<u64>, Vec<f32>)]) -> Vec<(f64, u64)> {
    match op {
        KernelOp::ReduceSum | KernelOp::ReduceMean => {
            let x = &inputs[0].1;
            vec![(x.iter().map(|&v| f64::from(v).abs()).sum(), x.len() as u64)]
        }
        KernelOp::Mse | KernelOp::LossSum => {
            let (a, b) = (&inputs[0].1, &inputs[1].1);
            let s = a
                .iter()
                .zip(b)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum();
            vec![(s, a.len() as u64)]
        }
        KernelOp::Transpose => inputs[0].1.iter().map(|_| (0.0, 1)).collect(),
        KernelOp::MatMul => {
            let (m, k) = (inputs[0].0[0] as usize, inputs[0].0[1] as usize);
            let n = inputs[1].0[1] as usize;
            let (a, b) = (&inputs[0].1, &inputs[1].1);
            let mut out = Vec::with_capacity(m * n);
            for i in 0..m {
                for j in 0..n {
                    let s = (0..k)
                        .map(|p| (f64::from(a[i * k + p]) * f64::from(b[p * n + j])).abs())
                        .sum();
                    out.push((s, k as u64));
                }
            }
            out
        }
    }
}

struct Case {
    name: String,
    op: KernelOp,
    kernels: Vec<Kernel>,
    partials: Option<BufferView>,
    inputs: Vec<(Vec<u64>, Vec<f32>)>,
}

/// The marker magnitude: a power of two, so every sum below is exact in any order.
const M: f32 = 1024.0;

/// Markers at element 0, at the last element, and at the last element of each stage-1 block `p`
/// that is one of the first 64 partials or that ends a stage-2 round (`p % 64 == 63`); every
/// partial carries one when there are at most 4096 of them.
fn markers(n: u64) -> Vec<f32> {
    let groups = n.div_ceil(64);
    (0..n)
        .map(|i| {
            let p = i / 64;
            let block_last = i % 64 == 63;
            let marked =
                i == 0 || i == n - 1 || (block_last && (groups <= 4096 || p < 64 || p % 64 == 63));
            if marked { M } else { 0.0 }
        })
        .collect()
}

fn one_hot(n: u64, at: u64) -> Vec<f32> {
    (0..n).map(|i| if i == at { M } else { 0.0 }).collect()
}

/// Positions of the one-hot cases at count `n`: the first and last element, the last of block 0,
/// the first of block 1, the first element of partial 64 (stage-2 round 2) and of partial 4096,
/// the middle, and the first element of the last block.
fn hot_positions(n: u64) -> Vec<u64> {
    let mut at = vec![
        0,
        n - 1,
        63,
        64,
        64 * 64,
        64 * 64 * 64,
        n / 2,
        (n - 1) / 64 * 64,
    ];
    at.retain(|&p| p < n);
    at.sort_unstable();
    at.dedup();
    at
}

const REDUCE_COUNTS: [u64; 17] = [
    1, 2, 63, 64, 65, 127, 128, 129, 4_095, 4_096, 4_097, 4_160, 4_161, 8_193, 262_144, 262_145,
    4_194_305,
];

fn reduction_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for n in REDUCE_COUNTS {
        for (op, mean) in [(KernelOp::ReduceSum, false), (KernelOp::ReduceMean, true)] {
            let (stage1, stage2, view) = planned_reduction(n, mean);
            let mut push = |what: String, data: Vec<f32>| {
                cases.push(Case {
                    name: format!("{op:?} n={n} {what}"),
                    op,
                    kernels: vec![stage1.clone(), stage2.clone()],
                    partials: Some(view),
                    inputs: vec![(vec![n, 1], data)],
                });
            };
            push("markers".into(), markers(n));
            for at in hot_positions(n) {
                push(format!("one-hot at {at}"), one_hot(n, at));
            }
        }
    }
    cases
}

const LOSS_COUNTS: [u64; 9] = [1, 2, 63, 64, 65, 1_000, 4_097, 78_400, 262_145];

fn loss_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for n in LOSS_COUNTS {
        let kernel = planned_loss_sum(n);
        let a_markers: Vec<f32> = markers(n).iter().map(|&v| v / 32.0).collect();
        let b_thirds: Vec<f32> = (0..n).map(|i| if i % 3 == 1 { 8.0 } else { 0.0 }).collect();
        let zeros = vec![0.0f32; n as usize];
        let mut data = vec![("markers - thirds".to_owned(), a_markers, b_thirds)];
        for at in [0, n - 1] {
            let a: Vec<f32> = one_hot(n, at).iter().map(|&v| v / 32.0).collect();
            data.push((format!("one-hot at {at}"), a, zeros.clone()));
        }
        for (what, a, b) in data {
            for op in [KernelOp::Mse, KernelOp::LossSum] {
                cases.push(Case {
                    name: format!("{op:?} n={n} {what}"),
                    op,
                    kernels: vec![kernel.clone()],
                    partials: None,
                    inputs: vec![(vec![n, 1], a.clone()), (vec![n, 1], b.clone())],
                });
            }
        }
    }
    cases
}

const TRANSPOSE_SHAPES: [(u64, u64); 18] = [
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 63),
    (1, 64),
    (1, 65),
    (1, 4_097),
    (63, 1),
    (64, 1),
    (65, 1),
    (4_097, 1),
    (2, 4_097),
    (4_097, 2),
    (64, 64),
    (63, 65),
    (65, 63),
    (7, 13),
    (784, 100),
];

fn transpose_cases() -> Vec<Case> {
    TRANSPOSE_SHAPES
        .iter()
        .map(|&(rows, cols)| Case {
            name: format!("Transpose [{rows}, {cols}] iota"),
            op: KernelOp::Transpose,
            kernels: vec![common::transpose_kernel(rows, cols)],
            partials: None,
            inputs: vec![(
                vec![rows, cols],
                (0..rows * cols).map(|i| (i + 1) as f32).collect(),
            )],
        })
        .collect()
}

const MATMUL_SHAPES: [(u64, u64, u64); 14] = [
    (1, 1, 65),
    (65, 1, 1),
    (1, 65, 1),
    (63, 7, 65),
    (65, 7, 63),
    (2, 3, 4_097),
    (4_097, 3, 2),
    (64, 64, 64),
    (65, 65, 65),
    (1, 4_097, 1),
    (3, 1, 1),
    (100, 784, 64),
    (784, 100, 64),
    (64, 100, 10),
];

/// Dense small integers: every product and sum exact, rows of `a` and columns of `b` distinct.
fn dense(m: u64, k: u64, n: u64) -> (Vec<f32>, Vec<f32>) {
    let a = (0..m * k)
        .map(|e| ((e / k * 31 + e % k * 17 + 3) % 9) as f32 - 4.0)
        .collect();
    let b = (0..k * n)
        .map(|e| ((e / n * 13 + e % n * 7 + 5) % 11) as f32 - 5.0)
        .collect();
    (a, b)
}

/// Few nonzero products per output element: `a[i, p]` at `p` in `{0, k - 1, i % k}`, `b[p, j]` at
/// `p` in `{0, k - 1, j % k}`, so a missing or extra `k` is far outside the tolerance.
fn sparse(m: u64, k: u64, n: u64) -> (Vec<f32>, Vec<f32>) {
    let a = (0..m * k)
        .map(|e| {
            let (i, p) = (e / k, e % k);
            if p == 0 || p == k - 1 || p == i % k {
                (1 + i % 5) as f32
            } else {
                0.0
            }
        })
        .collect();
    let b = (0..k * n)
        .map(|e| {
            let (p, j) = (e / n, e % n);
            if p == 0 || p == k - 1 || p == j % k {
                (1 + j % 3) as f32
            } else {
                0.0
            }
        })
        .collect();
    (a, b)
}

fn matmul_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for (m, k, n) in MATMUL_SHAPES {
        for (what, (a, b)) in [("dense", dense(m, k, n)), ("sparse", sparse(m, k, n))] {
            cases.push(Case {
                name: format!("MatMul [{m}, {k}] x [{k}, {n}] {what}"),
                op: KernelOp::MatMul,
                kernels: vec![common::matmul_kernel(m, k, n)],
                partials: None,
                inputs: vec![(vec![m, k], a), (vec![k, n], b)],
            });
        }
    }
    cases
}

fn modules(case: &Case, profile: &modelspec_core::device::DeviceProfile) -> Vec<KernelModule> {
    case.kernels
        .iter()
        .map(|kernel| match case.op {
            KernelOp::LossSum => emit_loss_sum_undivided(kernel, profile),
            _ => emit_module(kernel, profile),
        })
        .map(|m| m.expect("the kernel emits"))
        .collect()
}

fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

/// One run of `case`: guarded inputs and output, the dispatch (a reduction through
/// `Recording::reduce` with the plan's partials view), one host wait. The output buffer whole.
fn run(device: &Device<'_>, case: &Case, modules: &[KernelModule], outputs: usize) -> Vec<f32> {
    let uploads: Vec<Vec<f32>> = case
        .inputs
        .iter()
        .map(|(_, data)| guarded(data))
        .chain([guarded(&vec![f32::from_bits(UNWRITTEN); outputs])])
        .collect();
    let buffers: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|v| {
            device
                .device_local_buffer(bytes_of(v.len()))
                .expect("buffer")
        })
        .collect();
    let mut staging: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|v| {
            let mut b = device.staging_buffer(bytes_of(v.len())).expect("staging");
            b.write_f32(v).expect("written");
            b
        })
        .collect();
    let pipelines: Vec<_> = case
        .kernels
        .iter()
        .zip(modules)
        .map(|(k, m)| device.kernel_pipeline(k, &m.spirv).expect("pipeline"))
        .collect();
    let partials = case
        .partials
        .map(|view| device.scratch_buffer(&view).expect("partials"));
    let output = buffers.len() - 1;
    let mut recording = device.record().expect("recording");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("upload");
    }
    recording.barrier(Barrier::TransferToCompute);
    match &partials {
        Some(partials) => recording
            .reduce(
                (&pipelines[0], &case.kernels[0]),
                (&pipelines[1], &case.kernels[1]),
                &buffers[0],
                partials,
                &buffers[output],
            )
            .expect("the reduction is recorded"),
        None => {
            let bound: Vec<&Buffer<'_>> = buffers.iter().collect();
            recording
                .dispatch_kernel(&pipelines[0], &case.kernels[0], &bound)
                .expect("the dispatch is recorded");
        }
    }
    recording.barrier(Barrier::ComputeToTransfer);
    recording
        .copy(&buffers[output], &staging[output])
        .expect("download");
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("ends");
    device
        .submit(recorded)
        .expect("submitted")
        .wait()
        .expect("finishes");
    staging[output].read_f32().expect("reads")
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary_planned_kernels_hold_every_term_at_boundaries_twice() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("probed");
    let profile = candidates[0].profile.clone();
    let negotiated = negotiate(&profile).expect("negotiates");
    eprintln!("class: {}", negotiated.numerics);
    let device = vulkan.create_device(0, &negotiated).expect("opens");
    let setup = vulkan.take_validation_messages();
    assert!(setup.is_empty(), "{setup:#?}");
    let flushes = profile.shader_denorm_preserve_f32 != Some(true);

    let mut cases = reduction_cases();
    cases.extend(loss_cases());
    cases.extend(transpose_cases());
    cases.extend(matmul_cases());
    let mut failures = Vec::new();
    let mut elements = 0u64;
    let mut messages = Vec::new();
    for case in &cases {
        let modules = modules(case, &profile);
        let inputs: Vec<(Vec<u64>, Vec<f32>)> = case
            .inputs
            .iter()
            .map(|(s, d)| {
                let d = if flushes {
                    common::flush_subnormals(d)
                } else {
                    d.clone()
                };
                (s.clone(), d)
            })
            .collect();
        let expected = cpu(case.op, &inputs);
        let terms = terms(case.op, &inputs);
        let n = expected.len();
        let first = run(&device, case, &modules, n);
        let second = run(&device, case, &modules, n);
        let tol = common::term_tolerance(case.op);
        let mut wrong = 0;
        for i in 0..n {
            let (abs_terms, count) = terms[i];
            let r = tol.r(expected[i], abs_terms, count);
            if !tol.accepts(first[i], expected[i], r) {
                wrong += 1;
                if wrong <= 3 {
                    failures.push(format!(
                        "{} [{i}]: gpu {:?} cpu {:?} r {r:e} outside {tol}",
                        case.name, first[i], expected[i]
                    ));
                }
            }
        }
        if wrong > 3 {
            failures.push(format!("{}: {wrong} elements outside in all", case.name));
        }
        for output in [&first, &second] {
            let changed = (0..GUARD_WORDS)
                .filter(|&i| output[n + i].to_bits() != guard(i))
                .count();
            if changed > 0 {
                failures.push(format!("{}: {changed} guard words changed", case.name));
            }
        }
        if first
            .iter()
            .map(|v| v.to_bits())
            .ne(second.iter().map(|v| v.to_bits()))
        {
            failures.push(format!("{}: the two runs differ", case.name));
        }
        elements += n as u64;
        for message in vulkan.take_validation_messages() {
            messages.push(format!("{}: {message}", case.name));
        }
    }
    eprintln!(
        "adversary_reduce_matmul: {} cases run twice, {elements} elements, {} failures, {} \
         validation messages (validation {})",
        cases.len(),
        failures.len(),
        messages.len(),
        if vulkan.validation_enabled() {
            "on"
        } else {
            "off"
        }
    );
    assert!(failures.is_empty(), "{failures:#?}");
    assert!(messages.is_empty(), "{messages:#?}");
}

/// Stage 1 writes exactly its partials and nothing past them: the partials buffer carries guard
/// words, at counts whose plan has one stage-1 workgroup per partial and at a count whose plan
/// spreads stage 1 over `[65535, 2, 1]`, where 65533 workgroups have no block. Partial `p` is the
/// sum of block `p` (`markers` puts one `M` at the end of each block, two in block 0 and the last
/// block when it is short), and the reduced value still matches the CPU.
#[test]
#[ignore = "needs a Vulkan device"]
fn adversary_stage1_writes_no_partial_past_the_last_block() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("probed");
    let profile = candidates[0].profile.clone();
    let negotiated = negotiate(&profile).expect("negotiates");
    let device = vulkan.create_device(0, &negotiated).expect("opens");
    assert!(vulkan.take_validation_messages().is_empty());
    let mut failures = Vec::new();
    for n in [65u64, 4_097, 4_194_240, 4_194_305] {
        let (stage1, stage2, view) = planned_reduction(n, false);
        let groups = n.div_ceil(64) as usize;
        assert_eq!(view.element_count as usize, groups);
        let s1 = emit_module(&stage1, &profile).expect("emits");
        let s2 = emit_module(&stage2, &profile).expect("emits");
        let data: Vec<f32> = (0..n)
            .map(|i| if i % 64 == 63 || i == n - 1 { M } else { 0.0 })
            .collect();
        let uploads = [
            guarded(&data),
            guarded(&vec![f32::from_bits(UNWRITTEN); groups]),
            guarded(&[f32::from_bits(UNWRITTEN)]),
        ];
        let buffers: Vec<Buffer<'_>> = uploads
            .iter()
            .map(|v| {
                device
                    .device_local_buffer(bytes_of(v.len()))
                    .expect("buffer")
            })
            .collect();
        let mut staging: Vec<Buffer<'_>> = uploads
            .iter()
            .map(|v| {
                let mut b = device.staging_buffer(bytes_of(v.len())).expect("staging");
                b.write_f32(v).expect("written");
                b
            })
            .collect();
        let p1 = device
            .kernel_pipeline(&stage1, &s1.spirv)
            .expect("pipeline");
        let p2 = device
            .kernel_pipeline(&stage2, &s2.spirv)
            .expect("pipeline");
        let mut recording = device.record().expect("recording");
        for (from, to) in staging.iter().zip(&buffers) {
            recording.copy(from, to).expect("upload");
        }
        recording.barrier(Barrier::TransferToCompute);
        recording
            .reduce(
                (&p1, &stage1),
                (&p2, &stage2),
                &buffers[0],
                &buffers[1],
                &buffers[2],
            )
            .expect("recorded");
        recording.barrier(Barrier::ComputeToTransfer);
        recording.copy(&buffers[1], &staging[1]).expect("download");
        recording.copy(&buffers[2], &staging[2]).expect("download");
        recording.barrier(Barrier::TransferToHost);
        let recorded = recording.finish().expect("ends");
        device
            .submit(recorded)
            .expect("submitted")
            .wait()
            .expect("finishes");
        let partials = staging[1].read_f32().expect("reads");
        let result = staging[2].read_f32().expect("reads");
        let changed = (0..GUARD_WORDS)
            .filter(|&i| partials[groups + i].to_bits() != guard(i))
            .count();
        if changed > 0 {
            failures.push(format!(
                "n={n} grid {:?}: {changed} guard words after the {groups} partials changed",
                stage1.workgroups
            ));
        }
        let wrong_partials = (0..groups)
            .filter(|&p| {
                let block = &data[p * 64..((p + 1) * 64).min(n as usize)];
                partials[p].to_bits() != block.iter().sum::<f32>().to_bits()
            })
            .count();
        if wrong_partials > 0 {
            failures.push(format!("n={n}: {wrong_partials} partials wrong"));
        }
        let expected = cpu(KernelOp::ReduceSum, &[(vec![n, 1], data.clone())]);
        let tol = common::term_tolerance(KernelOp::ReduceSum);
        let r = tol.r(
            expected[0],
            terms(KernelOp::ReduceSum, &[(vec![n, 1], data)])[0].0,
            n,
        );
        if !tol.accepts(result[0], expected[0], r) {
            failures.push(format!(
                "n={n}: gpu {} cpu {} outside {tol}",
                result[0], expected[0]
            ));
        }
        for message in vulkan.take_validation_messages() {
            failures.push(format!("n={n}: {message}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
