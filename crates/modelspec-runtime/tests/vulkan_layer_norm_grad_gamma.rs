//! story:vulkan-layer-norm-grad-gamma-row-moments: `LayerNormGradGamma` is two dispatches in the
//! step plan, a per-row kernel that writes the mean and `rstd` of every row into an `f32[n, 2]`
//! buffer, a barrier, then the column kernel that reads them instead of computing them again in
//! every workgroup. Each column still adds its terms in ascending row order from `-0.0`, so on
//! the device every element of the gradient is the CPU interpreter's bits.
//!
//! The plan test runs by default. The device test is ignored; run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_layer_norm_grad_gamma -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;

use common::{Draw, GUARD_WORDS, guard, guarded};
use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{compile_artifact_with, compile_training_with};
use modelspec_core::ir::kernel::{BufferView, Kernel, KernelKind, Operand, PlanOp, PlanStep};
use modelspec_core::ir::tensor::{Dim, TensorOp, TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::interpreter::evaluate_seeded;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

const BERT_TINY: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../examples/bert_tiny.mspec"
));

/// The masked-row gather of BERT-tiny's packed training data: 832 rows reach the MLM head.
const MASKED_ROWS: u64 = 832;

/// The `[n, h]` shapes of the layer-norm programs: rows one past a multiple of every tile and
/// columns one past a multiple of every workgroup's, one element, and a row count one past a
/// workgroup with a width under a vector.
const SHAPES: [(u64, u64); 3] = [(4097, 129), (1, 1), (65, 3)];

/// The bits each output element holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_dead;

/// A program whose training step differentiates one `layer_norm` over `[n, h]`.
fn layer_norm_program(n: u64, h: u64) -> String {
    format!(
        "model Norm {{
    param gamma: Tensor<f32, [{h}]> = ones();
    param beta: Tensor<f32, [{h}]> = zeros();

    fn forward(x: Tensor<f32, [{n}, {h}]>) -> Tensor<f32, [{n}, {h}]> {{
        return layer_norm(x, gamma, beta);
    }}
}}

train Fit {{
    model = Norm;
    input x: Tensor<f32, [{n}, {h}]>;
    target y: Tensor<f32, [{n}, {h}]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 7;
    device = cpu;
}}
"
    )
}

/// A compiled program, its `LayerNormGradGamma` nodes and its Vulkan artifact.
struct Compiled {
    label: String,
    sites: usize,
    artifact: CompiledArtifact,
}

fn compile(label: &str, source: &str, masked_rows: Option<u64>) -> Compiled {
    let training = compile_training_with(label, source, masked_rows)
        .unwrap_or_else(|d| panic!("{label} compiles: {d:#?}"));
    let sites = training
        .training
        .training
        .program
        .nodes
        .iter()
        .filter(|node| node.op == TensorOp::LayerNormGradGamma)
        .count();
    let artifact = compile_artifact_with(label, source, Backend::Vulkan, None, masked_rows)
        .unwrap_or_else(|d| panic!("{label} compiles for Vulkan: {d:#?}"));
    Compiled {
        label: label.to_owned(),
        sites,
        artifact,
    }
}

fn programs() -> Vec<Compiled> {
    let mut programs = vec![compile("bert_tiny.mspec", BERT_TINY, Some(MASKED_ROWS))];
    for (n, h) in SHAPES {
        let label = format!("norm_{n}x{h}.mspec");
        programs.push(compile(&label, &layer_norm_program(n, h), None));
    }
    programs
}

/// The rows and columns of a buffer operand `[n, h]`.
fn matrix(operand: Option<&Operand>) -> Option<(u64, u64)> {
    match operand? {
        Operand::Buffer(ty) => match ty.shape[..] {
            [Dim::Static(n), Dim::Static(h)] => Some((n, h)),
            _ => None,
        },
        Operand::Immediate(_) => None,
    }
}

/// One `LayerNormGradGamma` site of a plan: its moments kernel and its column kernel.
struct Site {
    moments: Kernel,
    column: Kernel,
}

/// The sites of `compiled`'s step plan, and what in it is not the story: each column dispatch of
/// `LayerNormGradGamma` comes right after a barrier, which comes right after the moments dispatch
/// of the same value, a `per_row` kernel over `x: [n, h]` into `f32[n, 2]` that an `Allocate`
/// right before it provides; the moments dispatch reads `x` and writes that buffer only, the
/// column dispatch reads the buffer, then `x`, then `g`, and there is one such site per
/// `LayerNormGradGamma` of the training step.
fn sites(compiled: &Compiled) -> (Vec<Site>, Vec<String>) {
    let plan = &compiled.artifact.execution;
    let label = &compiled.label;
    let kernel = |step: &PlanStep| step.kernel_module_id.and_then(|id| plan.kernel(id));
    let is = |step: &PlanStep, kind: KernelKind| {
        step.op == PlanOp::Dispatch
            && kernel(step).is_some_and(|k| k.op == "LayerNormGradGamma" && k.kind == kind)
    };
    let mut bad = Vec::new();
    let mut found = Vec::new();
    let columns: Vec<usize> = (0..plan.step.len())
        .filter(|&i| is(&plan.step[i], KernelKind::PerColumn))
        .collect();
    let moments = (0..plan.step.len())
        .filter(|&i| is(&plan.step[i], KernelKind::PerRow))
        .count();
    if columns.len() != compiled.sites {
        bad.push(format!(
            "{label}: {} column dispatches for {} LayerNormGradGamma sites",
            columns.len(),
            compiled.sites
        ));
    }
    if moments != columns.len() {
        bad.push(format!(
            "{label}: {moments} moments dispatches for {} column dispatches",
            columns.len()
        ));
    }
    for at in columns {
        let column_step = &plan.step[at];
        let column = kernel(column_step).expect("a dispatched kernel").clone();
        let shape = matrix(column.operands.get(1)).or_else(|| matrix(column.operands.first()));
        let here = format!("{label} step {at} {shape:?}");
        if at < 3 {
            bad.push(format!(
                "{here}: no moments dispatch before the column dispatch"
            ));
            continue;
        }
        let (allocate, moments_step, barrier) =
            (&plan.step[at - 3], &plan.step[at - 2], &plan.step[at - 1]);
        if barrier.op != PlanOp::Barrier {
            bad.push(format!(
                "{here}: step {} is {:?}, not a barrier",
                at - 1,
                barrier.op
            ));
        }
        if !is(moments_step, KernelKind::PerRow) || moments_step.value != column_step.value {
            bad.push(format!(
                "{here}: step {} is not the moments dispatch of the same value",
                at - 2
            ));
            continue;
        }
        let moments = kernel(moments_step).expect("a dispatched kernel").clone();
        let Some((n, h)) = matrix(moments.operands.first()) else {
            bad.push(format!(
                "{here}: the moments kernel reads no [n, h] operand"
            ));
            continue;
        };
        if moments.operands.len() != 1 || moments.result != TensorType::f32(&[n, 2]) {
            bad.push(format!(
                "{here}: the moments kernel is {}, not [{n}, {h}] -> f32[{n}, 2]",
                moments.signature()
            ));
        }
        let rows_ty = TensorType::f32(&[n, h]);
        let wanted = vec![
            Operand::Buffer(TensorType::f32(&[n, 2])),
            Operand::Buffer(rows_ty.clone()),
            Operand::Buffer(rows_ty),
        ];
        if column.operands != wanted || column.result != TensorType::f32(&[h]) {
            bad.push(format!(
                "{here}: the column kernel is {}, not (moments, x, g) -> f32[{h}]",
                column.signature()
            ));
        }
        let views = |step: &PlanStep| -> (Vec<BufferView>, Vec<BufferView>) {
            (step.reads.clone(), step.writes.clone())
        };
        let (m_reads, m_writes) = views(moments_step);
        let (c_reads, _) = views(column_step);
        if allocate.op != PlanOp::Allocate || allocate.writes != m_writes {
            bad.push(format!(
                "{here}: no Allocate of the moments buffer right before"
            ));
        }
        if m_reads.len() != 1 || m_writes.len() != 1 || m_writes[0].element_count != 2 * n {
            bad.push(format!("{here}: the moments dispatch is not x -> [n, 2]"));
        } else if c_reads.len() != 3 || c_reads[0] != m_writes[0] || c_reads[1] != m_reads[0] {
            bad.push(format!(
                "{here}: the column dispatch does not read the moments, then x, then g"
            ));
        }
        found.push(Site { moments, column });
    }
    (found, bad)
}

/// The step plans of BERT-tiny (with its 832 masked rows) and of the `[4097, 129]`, `[1, 1]`
/// and `[65, 3]` layer norms run every `LayerNormGradGamma` as a moments dispatch, a barrier and
/// the column dispatch that reads the moments.
#[test]
fn layer_norm_grad_gamma_is_a_moments_dispatch_a_barrier_and_a_column_dispatch_at_every_site() {
    let mut bad = Vec::new();
    for compiled in programs() {
        assert!(
            compiled.sites > 0,
            "{} has a LayerNormGradGamma",
            compiled.label
        );
        bad.extend(sites(&compiled).1);
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

/// The CPU interpreter's `LayerNormGradGamma` of `x` and `g`, both `[n, h]`.
fn cpu(n: u64, h: u64, x: &[f32], g: &[f32]) -> Vec<f32> {
    let ty = TensorType::f32(&[n, h]);
    let mut builder = TensorProgramBuilder::new();
    let mut inputs = BTreeMap::new();
    let mut values = Vec::new();
    for (name, data) in [("x", x), ("g", g)] {
        inputs.insert(
            name.to_owned(),
            HostTensor::new(vec![n, h], data.to_vec()).expect("a host tensor"),
        );
        values.push(
            builder
                .input(name.to_owned(), ty.clone())
                .expect("an input"),
        );
    }
    let out = builder
        .layer_norm_grad_gamma(values[0], values[1])
        .expect("the op is well typed");
    builder.output(out).expect("the result is a value");
    let program = builder.finish();
    evaluate_seeded(&program, &inputs, &BTreeMap::new(), &BTreeMap::new(), 0)
        .expect("the CPU evaluates")[0]
        .data()
        .to_vec()
}

fn bytes_of(words: u64) -> u64 {
    words * 4
}

/// Runs `site` over `x` and `g` once on `device` as the step plan does, in one command buffer
/// with one host wait: the moments kernel over `x` into a buffer of `[n, 2]`, a barrier, then the
/// column kernel over the moments, `x` and `g`. Returns the gradient with its guard words.
fn gpu(device: &Device<'_>, site: &Site, n: u64, h: u64, x: &[f32], g: &[f32]) -> Vec<f32> {
    let pipeline = |kernel: &Kernel| {
        let words = emit(kernel).expect("the kernel emits");
        device
            .kernel_pipeline(kernel, &words)
            .expect("the pipeline is created")
    };
    let (moments_pipeline, column_pipeline) = (pipeline(&site.moments), pipeline(&site.column));
    let uploads = [
        x.to_vec(),
        g.to_vec(),
        guarded(&vec![f32::from_bits(UNWRITTEN); h as usize]),
    ];
    let buffers: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|v| {
            device
                .device_local_buffer(bytes_of(v.len() as u64))
                .expect("a buffer")
        })
        .collect();
    let mut staging: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|v| {
            let mut buffer = device
                .staging_buffer(bytes_of(v.len() as u64))
                .expect("a staging buffer");
            buffer.write_f32(v).expect("the staging buffer is written");
            buffer
        })
        .collect();
    let moments = device
        .device_local_buffer(bytes_of(2 * n))
        .expect("a moments buffer");
    let mut recording = device.record().expect("a recording starts");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("the upload is recorded");
    }
    recording.barrier(Barrier::TransferToCompute);
    recording
        .dispatch_kernel(&moments_pipeline, &site.moments, &[&buffers[0], &moments])
        .expect("the moments dispatch is recorded");
    recording.barrier(Barrier::ComputeToCompute);
    recording
        .dispatch_kernel(
            &column_pipeline,
            &site.column,
            &[&moments, &buffers[0], &buffers[1], &buffers[2]],
        )
        .expect("the column dispatch is recorded");
    recording.barrier(Barrier::ComputeToTransfer);
    recording
        .copy(&buffers[2], &staging[2])
        .expect("the download is recorded");
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    device
        .submit(recorded)
        .expect("submitted")
        .wait()
        .expect("the device finishes");
    staging[2].read_f32().expect("the output reads")
}

/// On the device, the moments and column kernels of every `LayerNormGradGamma` site of
/// BERT-tiny (`[4096, 128]` and the masked `[832, 128]`) and of the `[4097, 129]`, `[1, 1]` and
/// `[65, 3]` layer norms give the CPU's bits for every element of the gradient, in two runs, with
/// the guard words after it unchanged, under the validation layer when it is on.
#[test]
#[ignore = "needs a Vulkan device"]
fn layer_norm_grad_gamma_from_row_moments_is_the_cpu_bits_on_the_device() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        std::env::var(VALIDATION_ENV).as_deref() == Ok("1"),
        "the validation layer is on exactly when {VALIDATION_ENV}=1"
    );
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let mut draw = Draw::new(0x6c6e_6767_6d6f_6d73);
    let mut bad = Vec::new();
    let mut shapes = Vec::new();
    for compiled in programs() {
        let (found, structure) = sites(&compiled);
        bad.extend(structure);
        let mut seen = Vec::new();
        for site in found {
            if seen.contains(&site.column.id) {
                continue;
            }
            seen.push(site.column.id);
            let (n, h) = matrix(site.moments.operands.first()).expect("an [n, h] operand");
            let label = format!("{} [{n}, {h}]", compiled.label);
            let (x, g) = (draw.values(n * h), draw.values(n * h));
            let want = cpu(n, h, &x, &g);
            let first = gpu(&device, &site, n, h, &x, &g);
            let second = gpu(&device, &site, n, h, &x, &g);
            let differ: Vec<usize> = (0..h as usize)
                .filter(|&j| first[j].to_bits() != want[j].to_bits())
                .collect();
            for &j in differ.iter().take(4) {
                bad.push(format!(
                    "{label} column {j}: gpu {:?} cpu {:?}",
                    first[j], want[j]
                ));
            }
            if !differ.is_empty() {
                bad.push(format!(
                    "{label}: {} of {h} columns differ from the CPU",
                    differ.len()
                ));
            }
            for i in 0..GUARD_WORDS {
                if first[h as usize + i].to_bits() != guard(i) {
                    bad.push(format!("{label}: guard word {i} changed"));
                }
            }
            if first
                .iter()
                .map(|v| v.to_bits())
                .ne(second.iter().map(|v| v.to_bits()))
            {
                bad.push(format!("{label}: two runs differ"));
            }
            eprintln!("{label}: {} columns differ from the CPU", differ.len());
            shapes.push((n, h));
        }
    }
    for wanted in [
        (4096, 128),
        (MASKED_ROWS, 128),
        (4097, 129),
        (1, 1),
        (65, 3),
    ] {
        if !shapes.contains(&wanted) {
            bad.push(format!("no site of shape {wanted:?} ran"));
        }
    }
    let messages = vulkan.take_validation_messages();
    assert!(messages.is_empty(), "{messages:#?}");
    assert!(bad.is_empty(), "{bad:#?}");
}
