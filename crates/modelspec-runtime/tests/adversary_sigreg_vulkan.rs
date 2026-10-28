//! Adversary of story:jepa-sigreg-vulkan. Attacks the two `sigreg` kernels and the stepwise
//! projection upload outside the story's own grid:
//!
//! - kernel shapes with `N` or `M` at 1023, 1025 and 4096 (device);
//! - columns of constants, zeros, `1e6`, `1e-6`, a NaN, an infinity and values whose quadrant
//!   count leaves `i32` (device), the CPU and the GPU agreeing on what happens;
//! - training runs on Vulkan against the CPU: `K` of 1, 7, 9 and 64, `N`/`M` at 1023, 1025 and
//!   4096, three `sigreg` sites in one program, `sigreg` with dropout and AdamW, a resumed run
//!   against the run that never stopped, and a NaN parameter stopping both backends alike
//!   (device);
//! - the FLOP rules against the instructions each module executes, from its loop bounds, for
//!   shapes other than the story's (no device);
//! - `infer --backend vulkan`'s refusal along the CLI's own path (no device).
//!
//! Device cases are ignored; run them with
//! `cargo test -p modelspec-runtime --features vulkan --test adversary_sigreg_vulkan -- --ignored --test-threads=1`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use common::words::{GLSL_FMA, OP_EXT_INST, OP_F_ADD, OP_F_MUL, OP_F_SUB, instructions};
use common::{
    Draw, GPU_ATOL, GPU_RTOL, GPU_RUN_ATOL, GPU_RUN_RTOL, GUARD_WORDS, guard, guarded, within,
};
use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{CompiledTraining, compile_artifact, compile_training};
use modelspec_core::ir::kernel::{Kernel, Operand};
use modelspec_core::ir::tensor::{Dim, ParamId, TensorOp, TensorProgramBuilder, TensorType};
use modelspec_core::schedule::rowwise_kernel;
use modelspec_runtime::cpu::interpreter::evaluate_seeded;
use modelspec_runtime::cpu::tensor::{HostTensor, sin_cos};
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::spirv::validate::validate;
use modelspec_runtime::train::checkpoint::{self, Checkpoint};
use modelspec_runtime::train::data::synthetic;
use modelspec_runtime::train::init::sigreg_projections;
use modelspec_runtime::train::runner::{self, Run, RunError};
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::runner::{
    DeviceRunError, RunOptions, VulkanRunError, infer_on_device, train_on_device_from,
};
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

const UNWRITTEN: u32 = 0x7fc0_dead;
const SEED: u64 = 5;

// ---------------------------------------------------------------------------------------------
// Kernel harness (the device test's, restated: its helpers are private to its file).
// ---------------------------------------------------------------------------------------------

fn knots() -> [(f64, f64, f64); 8] {
    std::array::from_fn(|k| {
        let t = 0.625 * (k + 1) as f64;
        let phi = (-0.5 * t * t).exp();
        let share = if k == 7 { 1.0 } else { 2.0 };
        (t, phi, share * 0.625 * phi)
    })
}

fn column_moments(p: &[f32], n: usize, m: usize, column: usize) -> [(f64, f64, f64, f64); 8] {
    let knots = knots();
    std::array::from_fn(|k| {
        let t = knots[k].0 as f32;
        let (mut cos, mut sin, mut abs_cos, mut abs_sin) = (0.0, 0.0, 0.0, 0.0);
        for row in 0..n {
            let (s, c) = sin_cos(t * p[row * m + column]);
            sin += f64::from(s);
            cos += f64::from(c);
            abs_sin += f64::from(s).abs();
            abs_cos += f64::from(c).abs();
        }
        (cos / n as f64, sin / n as f64, abs_cos, abs_sin)
    })
}

fn columns_reference(p: &[f32], n: usize, m: usize) -> Vec<f32> {
    let knots = knots();
    (0..m)
        .map(|column| {
            let moments = column_moments(p, n, m, column);
            let mut r = 0.0;
            for (&(_, phi, weight), &(cos, sin, abs_cos, abs_sin)) in knots.iter().zip(&moments) {
                let d = cos - phi;
                r += n as f64 * weight * (sin * sin + d * d);
                r += 2.0 * weight * (sin.abs() * abs_sin + d.abs() * abs_cos);
            }
            r as f32
        })
        .collect()
}

fn grad_reference(p: &[f32], g: &[f32], n: usize, m: usize) -> Vec<f32> {
    let knots = knots();
    let moments: Vec<_> = (0..m)
        .map(|column| column_moments(p, n, m, column))
        .collect();
    p.iter()
        .enumerate()
        .map(|(i, &x)| {
            let column = i % m;
            let mut r = 0.0;
            for (&(t, phi, weight), &(cos, sin, _, _)) in knots.iter().zip(&moments[column]) {
                let (s, c) = sin_cos(t as f32 * x);
                let d = cos - phi;
                r += weight * t * (sin.abs() * f64::from(c).abs() + d.abs() * f64::from(s).abs());
            }
            (2.0 * f64::from(g[column]).abs() * r) as f32
        })
        .collect()
}

struct Case {
    op: TensorOp,
    operands: Vec<(String, TensorType, Vec<f32>)>,
    result: TensorType,
}

fn cpu(case: &Case) -> Vec<f32> {
    let mut builder = TensorProgramBuilder::new();
    let mut inputs = BTreeMap::new();
    let mut values = Vec::new();
    for (name, ty, data) in &case.operands {
        let shape = ty
            .shape
            .iter()
            .map(|dim| match dim {
                Dim::Static(n) => *n,
                Dim::Symbol(_) => unreachable!(),
            })
            .collect();
        inputs.insert(name.clone(), HostTensor::new(shape, data.clone()).unwrap());
        values.push(
            builder
                .input(name.clone(), ty.clone())
                .expect("a new input"),
        );
    }
    let result = match case.op {
        TensorOp::SigregColumns => builder.sigreg_columns(values[0]),
        TensorOp::SigregColumnsGrad => builder.sigreg_columns_grad(values[0], values[1]),
        _ => unreachable!(),
    }
    .expect("well typed");
    builder.output(result).expect("a value");
    let program = builder.finish();
    evaluate_seeded(&program, &inputs, &BTreeMap::new(), &BTreeMap::new(), SEED)
        .expect("the CPU evaluates")[0]
        .data()
        .to_vec()
}

fn kernel_of(case: &Case) -> Kernel {
    let operands = case
        .operands
        .iter()
        .map(|(_, ty, _)| Operand::Buffer(ty.clone()))
        .collect();
    rowwise_kernel(&case.op, operands, case.result.clone()).expect("a per-column kernel")
}

fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

fn gpu(device: &Device<'_>, case: &Case, elements: usize) -> Vec<f32> {
    let kernel = kernel_of(case);
    let words = emit(&kernel).expect("the kernel emits");
    let pipeline = device
        .kernel_pipeline(&kernel, &words)
        .expect("the pipeline is created");
    let mut uploads: Vec<Vec<f32>> = case.operands.iter().map(|(_, _, d)| d.clone()).collect();
    uploads.push(guarded(&vec![f32::from_bits(UNWRITTEN); elements]));
    let buffers: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|d| {
            device
                .device_local_buffer(bytes_of(d.len()))
                .expect("a buffer")
        })
        .collect();
    let mut staging: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|d| {
            let mut b = device.staging_buffer(bytes_of(d.len())).expect("staging");
            b.write_f32(d).expect("written");
            b
        })
        .collect();
    let output = buffers.len() - 1;
    let bound: Vec<&Buffer<'_>> = buffers.iter().collect();
    let mut recording = device.record().expect("a recording starts");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("upload");
    }
    recording.barrier(Barrier::TransferToCompute);
    recording
        .dispatch_kernel(&pipeline, &kernel, &bound)
        .expect("dispatch");
    recording.barrier(Barrier::ComputeToTransfer);
    recording
        .copy(&buffers[output], &staging[output])
        .expect("download");
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    device
        .submit(recorded)
        .expect("submitted")
        .wait()
        .expect("the device finishes");
    staging[output].read_f32().expect("the output reads")
}

fn open() -> Vulkan {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        std::env::var(VALIDATION_ENV).as_deref() == Ok("1")
    );
    vulkan
}

/// Compares `got` (with guard words) with `expected` under `r`; a NaN on the CPU must be a NaN
/// on the GPU, and the reverse. Returns the failures, at most `limit` per call.
fn compare(label: &str, got: &[f32], expected: &[f32], r: &[f32], limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    let count = expected.len();
    for (i, ((&g, &c), &r)) in got.iter().zip(expected).zip(r).enumerate() {
        let ok = if c.is_nan() || g.is_nan() {
            c.is_nan() && g.is_nan()
        } else if c.is_infinite() || g.is_infinite() {
            g == c
        } else {
            within(g, c, r, GPU_ATOL, GPU_RTOL)
        };
        if !ok && out.len() < limit {
            out.push(format!("{label} [{i}]: gpu {g:?} cpu {c:?} r {r:?}"));
        }
    }
    for i in 0..GUARD_WORDS {
        if got[count + i].to_bits() != guard(i) {
            out.push(format!("{label}: guard word {i} changed"));
        }
    }
    out
}

/// Runs both kernels on `p: [n, m]` with upstream gradient `g` and returns every failure.
fn both_kernels(
    device: &Device<'_>,
    label: &str,
    p: &[f32],
    g: &[f32],
    n: u64,
    m: u64,
) -> Vec<String> {
    let p_type = TensorType::f32(&[n, m]);
    let g_type = TensorType::f32(&[m]);
    let forward = Case {
        op: TensorOp::SigregColumns,
        operands: vec![("p".into(), p_type.clone(), p.to_vec())],
        result: g_type.clone(),
    };
    let backward = Case {
        op: TensorOp::SigregColumnsGrad,
        operands: vec![
            ("p".into(), p_type.clone(), p.to_vec()),
            ("g".into(), g_type, g.to_vec()),
        ],
        result: p_type,
    };
    let (nu, mu) = (n as usize, m as usize);
    let mut out = Vec::new();
    let expected = cpu(&forward);
    let got = gpu(device, &forward, mu);
    out.extend(compare(
        &format!("{label} SigregColumns [{n}, {m}]"),
        &got,
        &expected,
        &columns_reference(p, nu, mu),
        4,
    ));
    let expected = cpu(&backward);
    let got = gpu(device, &backward, nu * mu);
    out.extend(compare(
        &format!("{label} SigregColumnsGrad [{n}, {m}]"),
        &got,
        &expected,
        &grad_reference(p, g, nu, mu),
        4,
    ));
    out
}

// ---------------------------------------------------------------------------------------------
// 1. Kernel shapes the story's grid misses.
// ---------------------------------------------------------------------------------------------

/// `N` or `M` at 1023, 1025 and 4096; values drawn in `[-2, 2)`.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_kernels_match_the_cpu_at_n_and_m_of_1023_1025_and_4096() {
    let vulkan = open();
    let candidates = vulkan.candidates().expect("probed");
    let negotiated = negotiate(&candidates[0].profile).expect("negotiates");
    let device = vulkan.create_device(0, &negotiated).expect("opens");
    let mut draw = Draw::new(0xad5e_1a77);
    let shapes: [(u64, u64); 10] = [
        (1023, 1),
        (1025, 7),
        (4096, 3),
        (1, 1023),
        (7, 1025),
        (3, 4096),
        (1023, 1025),
        (1025, 1023),
        (4096, 65),
        (65, 4096),
    ];
    let mut failures = Vec::new();
    let mut messages = Vec::new();
    for (n, m) in shapes {
        let p: Vec<f32> = draw.values(n * m).iter().map(|v| v * 0.5).collect();
        let g = draw.values(m);
        failures.extend(both_kernels(&device, "shape", &p, &g, n, m));
        messages.extend(vulkan.take_validation_messages());
    }
    assert!(failures.is_empty(), "{failures:#?}");
    assert!(messages.is_empty(), "{messages:#?}");
}

// ---------------------------------------------------------------------------------------------
// 2. Values.
// ---------------------------------------------------------------------------------------------

/// The columns of the value cases, by name, each a function of the row.
fn value_columns(n: u64, draw: &mut Draw) -> Vec<(&'static str, Vec<f32>)> {
    let drawn = |draw: &mut Draw, scale: f32| -> Vec<f32> {
        draw.values(n).iter().map(|v| v * scale).collect()
    };
    let mut nan = drawn(draw, 0.5);
    nan[(n as usize) / 2] = f32::NAN;
    let mut pos_inf = drawn(draw, 0.5);
    pos_inf[0] = f32::INFINITY;
    let mut neg_inf = drawn(draw, 0.5);
    let last = nan.len() - 1;
    neg_inf[last] = f32::NEG_INFINITY;
    vec![
        ("zeros", vec![0.0; n as usize]),
        ("negative zeros", vec![-0.0; n as usize]),
        ("constant 0.7", vec![0.7; n as usize]),
        ("constant 1e6", vec![1e6; n as usize]),
        ("constant 1e-6", vec![1e-6; n as usize]),
        ("drawn x 1e6", drawn(draw, 1e6)),
        ("drawn x 1e-6", drawn(draw, 1e-6)),
        ("one NaN", nan),
        ("one +inf", pos_inf),
        ("one -inf", neg_inf),
        ("constant 1e10", vec![1e10; n as usize]),
        (
            "alternating 3e9",
            (0..n)
                .map(|i| if i % 2 == 0 { 3e9 } else { -3e9 })
                .collect(),
        ),
        ("constant f32::MAX", vec![f32::MAX; n as usize]),
        ("drawn control", drawn(draw, 0.5)),
    ]
}

/// Each named column alone, as a `[n, 1]` operand, at `n` of 1, 64 and 65.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_kernels_agree_with_the_cpu_on_constant_extreme_and_non_finite_columns() {
    let vulkan = open();
    let candidates = vulkan.candidates().expect("probed");
    let negotiated = negotiate(&candidates[0].profile).expect("negotiates");
    let device = vulkan.create_device(0, &negotiated).expect("opens");
    let mut draw = Draw::new(0x7a1e_5000);
    let mut failures = Vec::new();
    let mut messages = Vec::new();
    for n in [1u64, 64, 65] {
        for (name, column) in value_columns(n, &mut draw) {
            for g in [1.0f32, -3.5] {
                failures.extend(both_kernels(&device, name, &column, &[g], n, 1));
            }
            messages.extend(vulkan.take_validation_messages());
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
    assert!(messages.is_empty(), "{messages:#?}");
}

/// All the columns side by side in one `[65, M]` operand: a NaN or an infinity in one column
/// leaves its neighbours as the CPU computes them.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_a_non_finite_column_does_not_reach_its_neighbours() {
    let vulkan = open();
    let candidates = vulkan.candidates().expect("probed");
    let negotiated = negotiate(&candidates[0].profile).expect("negotiates");
    let device = vulkan.create_device(0, &negotiated).expect("opens");
    let mut draw = Draw::new(0x0e16_4b07);
    let n = 65u64;
    let columns = value_columns(n, &mut draw);
    let m = columns.len() as u64;
    let mut p = vec![0.0f32; (n * m) as usize];
    for (j, (_, column)) in columns.iter().enumerate() {
        for (i, &v) in column.iter().enumerate() {
            p[i * m as usize + j] = v;
        }
    }
    let g = draw.values(m);
    let failures = both_kernels(&device, "side by side", &p, &g, n, m);
    let messages = vulkan.take_validation_messages();
    assert!(failures.is_empty(), "{failures:#?}");
    assert!(messages.is_empty(), "{messages:#?}");
}

// ---------------------------------------------------------------------------------------------
// 3. Training on Vulkan against the CPU.
// ---------------------------------------------------------------------------------------------

/// `sigreg` of a `[n, k]` parameter over `m` slices, trained by `sgd` for `steps` steps.
fn single_site(n: u64, k: u64, m: u64, steps: u64) -> String {
    format!(
        "model Spread {{
    param z: Tensor<f32, [{n}, {k}]> = xavier_uniform();

    fn forward(x: Tensor<f32, [{n}, {k}]>) -> Tensor<f32, [{n}, {k}]> {{
        return add(z, mul(x, 0.0));
    }}
}}

train Fit {{
    model = Spread;
    input x: Tensor<f32, [{n}, {k}]>;
    target y: Tensor<f32, [{n}, {k}]>;
    loss = add(sigreg(model.forward(x), slices = {m}), mul(mse(x, y), 0.0));
    optimizer = sgd(lr = 0.5);
    steps = {steps};
    seed = {SEED};
    device = cpu;
}}
"
    )
}

/// Three `sigreg` sites in one loss, `model.forward` read once (E-TYPE-017): two of the same
/// projection type `[8, 16]` over the prediction and the input, one of `[8, 5]` over the target.
const THREE_SITES: &str = "model Spread {
    param z: Tensor<f32, [65, 8]> = xavier_uniform();

    fn forward(x: Tensor<f32, [65, 8]>) -> Tensor<f32, [65, 8]> {
        return add(z, mul(x, 0.0));
    }
}

train Fit {
    model = Spread;
    input x: Tensor<f32, [65, 8]>;
    target y: Tensor<f32, [65, 8]>;
    loss = add(add(sigreg(model.forward(x), slices = 16), sigreg(mul(x, 2.0), slices = 16)), add(sigreg(y, slices = 5), mul(mse(x, y), 0.0)));
    optimizer = sgd(lr = 0.25);
    steps = 12;
    seed = 9;
    device = cpu;
}
";

/// `sigreg` of a dropout output plus half the target, trained by AdamW.
const DROPOUT_ADAMW: &str = "model Drop {
    param w: Tensor<f32, [16, 8]> = xavier_uniform();

    fn forward(x: Tensor<f32, [64, 16]>) -> Tensor<f32, [64, 8]> {
        return dropout(matmul(x, w), p = 0.1);
    }
}

train Fit {
    model = Drop;
    input x: Tensor<f32, [64, 16]>;
    target y: Tensor<f32, [64, 8]>;
    loss = sigreg(add(model.forward(x), mul(y, 0.5)), slices = 12);
    optimizer = adamw(lr = 0.01, weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, warmup_steps = 2);
    steps = 20;
    seed = 7;
    device = cpu;
}
";

/// The same model trained by `sgd`, which a checkpoint can resume.
const DROPOUT_SGD: &str = "model Drop {
    param w: Tensor<f32, [16, 8]> = xavier_uniform();

    fn forward(x: Tensor<f32, [64, 16]>) -> Tensor<f32, [64, 8]> {
        return dropout(matmul(x, w), p = 0.1);
    }
}

train Fit {
    model = Drop;
    input x: Tensor<f32, [64, 16]>;
    target y: Tensor<f32, [64, 8]>;
    loss = sigreg(add(model.forward(x), mul(y, 0.5)), slices = 12);
    optimizer = sgd(lr = 0.05);
    steps = 20;
    seed = 7;
    device = cpu;
}
";

fn compiled(name: &str, source: &str) -> (CompiledTraining, CompiledArtifact) {
    let compiled = compile_training(name, source)
        .unwrap_or_else(|errors| panic!("{name} compiles: {errors:#?}"));
    let artifact = compile_artifact(name, source, Backend::Vulkan)
        .unwrap_or_else(|errors| panic!("{name} compiles for Vulkan: {errors:#?}"));
    (compiled, artifact)
}

fn on_vulkan(
    compiled: &CompiledTraining,
    artifact: &CompiledArtifact,
    start: u64,
    init: BTreeMap<ParamId, HostTensor<f32>>,
) -> Result<Run, DeviceRunError> {
    train_on_device_from(
        None,
        compiled,
        artifact,
        false,
        RunOptions {
            start,
            init,
            ..RunOptions::default()
        },
    )
    .map(|(run, _, _)| run)
}

/// Every loss and every final parameter element of `gpu` within `atol + rtol · |c|` of `cpu`.
fn run_differences(label: &str, gpu: &Run, cpu: &Run, atol: f32, rtol: f32) -> Vec<String> {
    let mut out = Vec::new();
    if gpu.losses.len() != cpu.losses.len() {
        out.push(format!(
            "{label}: {} losses on the GPU, {} on the CPU",
            gpu.losses.len(),
            cpu.losses.len()
        ));
    }
    let (mut max_loss, mut max_param, mut bits) = (0.0f64, 0.0f64, true);
    for (i, (&g, &c)) in gpu.losses.iter().zip(&cpu.losses).enumerate() {
        max_loss = max_loss.max((f64::from(g) - f64::from(c)).abs());
        bits &= g.to_bits() == c.to_bits();
        if !within(g, c, c.abs(), atol, rtol) && out.len() < 6 {
            out.push(format!(
                "{label}: loss at step {}: gpu {g:?} cpu {c:?}",
                cpu.start_step + i as u64
            ));
        }
    }
    for (param, c) in &cpu.state.values {
        let Some(g) = gpu.state.values.get(param) else {
            out.push(format!("{label}: no parameter {param:?} on the GPU"));
            continue;
        };
        for (i, (&g, &c)) in g.data().iter().zip(c.data()).enumerate() {
            max_param = max_param.max((f64::from(g) - f64::from(c)).abs());
            bits &= g.to_bits() == c.to_bits();
            if !within(g, c, c.abs(), atol, rtol) && out.len() < 12 {
                out.push(format!("{label}: {param:?}[{i}]: gpu {g:?} cpu {c:?}"));
            }
        }
    }
    // A run whose parameters never left their start compares nothing about the update.
    if cpu.start_step == 0 && cpu.state.values == cpu.initial {
        out.push(format!(
            "{label}: the CPU run left every parameter at its start"
        ));
    }
    eprintln!(
        "{label}: max loss |g-c| {max_loss:.3e}, max parameter |g-c| {max_param:.3e}, bit-equal {bits}"
    );
    out
}

/// `K` of 1, 7, 9 and 64, and `N` or `M` at 1023, 1025 and 4096, each trained 6 steps on both
/// backends, held to the run tolerance.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_training_with_other_k_and_large_n_or_m_matches_the_cpu_run() {
    let shapes: [(u64, u64, u64); 9] = [
        (65, 1, 33),
        (65, 7, 33),
        (65, 9, 33),
        (65, 64, 33),
        (1023, 8, 16),
        (1025, 9, 1023),
        (4096, 8, 4),
        (64, 8, 4096),
        (65, 7, 1025),
    ];
    let mut failures = Vec::new();
    for (n, k, m) in shapes {
        let label = format!("N {n} K {k} M {m}");
        let source = single_site(n, k, m, 6);
        let (compiled, artifact) = compiled("spread.mspec", &source);
        let cpu = runner::run(&compiled).unwrap_or_else(|e| panic!("{label}: CPU run: {e}"));
        match on_vulkan(&compiled, &artifact, 0, BTreeMap::new()) {
            Ok(gpu) => failures.extend(run_differences(
                &label,
                &gpu,
                &cpu,
                GPU_RUN_ATOL,
                GPU_RUN_RTOL,
            )),
            Err(error) => failures.push(format!("{label}: the device run failed: {error}")),
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Three sites, two of one projection type: each site's projection of each step reaches its
/// own buffer, so every loss and the parameter stay within the run tolerance of the CPU.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_three_sigreg_sites_each_read_their_own_projection_of_each_step() {
    let (compiled, artifact) = compiled("three.mspec", THREE_SITES);
    let sites = modelspec_core::ir::training::sigreg_sites(&compiled.training.training.program);
    assert_eq!(sites.len(), 3, "{sites:?}");
    let cpu = runner::run(&compiled).expect("the CPU run finishes");
    let gpu = on_vulkan(&compiled, &artifact, 0, BTreeMap::new()).expect("the device run");
    let failures = run_differences("three sites", &gpu, &cpu, GPU_RUN_ATOL, GPU_RUN_RTOL);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// `sigreg` with dropout and AdamW on Vulkan against the CPU run.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_sigreg_with_dropout_and_adamw_matches_the_cpu_run() {
    let (compiled, artifact) = compiled("drop.mspec", DROPOUT_ADAMW);
    let cpu = runner::run(&compiled).expect("the CPU run finishes");
    let gpu = on_vulkan(&compiled, &artifact, 0, BTreeMap::new()).expect("the device run");
    let failures = run_differences("dropout adamw", &gpu, &cpu, GPU_RUN_ATOL, GPU_RUN_RTOL);
    assert!(failures.is_empty(), "{failures:#?}");
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary_sigreg_vulkan")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// 10 steps, a checkpoint, 10 steps resumed from it on the device, against the 20 steps that
/// never stopped on the device and on the CPU: the projection (and dropout token) of step `k`
/// depends on `k`, not on how many steps this process ran.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_a_resumed_sigreg_run_reproduces_the_run_that_never_stopped() {
    let (whole, artifact) = compiled("drop.mspec", DROPOUT_SGD);
    let mut half = whole.clone();
    half.config.steps = 10;
    let dir = scratch("resume");
    let mid = dir.join("mid.msckpt");

    let uninterrupted = on_vulkan(&whole, &artifact, 0, BTreeMap::new()).expect("whole run");
    let first = on_vulkan(&half, &artifact, 0, BTreeMap::new()).expect("first half");
    checkpoint::write(&mid, &Checkpoint::of_state(&half, 10, &first.state.values))
        .expect("the checkpoint is written");
    let read = checkpoint::read(&mid).expect("the checkpoint reads");
    let init = read.resume_into(&half).expect("the checkpoint matches");
    let resumed = on_vulkan(&half, &artifact, read.step, init).expect("the resumed run");
    assert_eq!(resumed.start_step, 10);

    let tail = Run {
        start_step: 10,
        losses: uninterrupted.losses[10..].to_vec(),
        ..uninterrupted.clone()
    };
    let mut failures = run_differences(
        "resumed vs uninterrupted (device)",
        &resumed,
        &tail,
        GPU_ATOL,
        GPU_RTOL,
    );
    let cpu = runner::run(&whole).expect("the CPU run");
    failures.extend(run_differences(
        "uninterrupted device vs CPU",
        &uninterrupted,
        &cpu,
        GPU_RUN_ATOL,
        GPU_RUN_RTOL,
    ));
    let cpu_tail = Run {
        start_step: 10,
        losses: cpu.losses[10..].to_vec(),
        ..cpu.clone()
    };
    failures.extend(run_differences(
        "resumed device vs CPU",
        &resumed,
        &cpu_tail,
        GPU_RUN_ATOL,
        GPU_RUN_RTOL,
    ));
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A NaN in the parameter `sigreg` reads stops both backends with E-TRAIN-001 at the same step
/// with the same diagnostic.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_a_nan_parameter_stops_both_backends_alike() {
    let source = single_site(65, 8, 16, 4);
    let (compiled, artifact) = compiled("spread.mspec", &source);
    let clean = runner::run(&compiled).expect("the clean CPU run");
    let mut init = clean.initial.clone();
    let (&param, value) = init.iter_mut().next().expect("one parameter");
    let mut data = value.data().to_vec();
    data[17] = f32::NAN;
    *value = HostTensor::new(value.shape().to_vec(), data).unwrap();

    let cpu = match runner::run_from(&compiled, None, &init, 0, |_| {}, |_, _| Ok(())) {
        Err(RunError::NonFinite(diverged)) => diverged,
        other => panic!("the CPU run stops with E-TRAIN-001, not {other:?}"),
    };
    let gpu = match on_vulkan(&compiled, &artifact, 0, init.clone()) {
        Err(DeviceRunError::Failed(VulkanRunError::NonFinite(diverged))) => diverged,
        Err(other) => panic!("the device run stops with E-TRAIN-001, not {other}"),
        Ok(run) => panic!(
            "the device run with a NaN in {param:?} finished: losses {:?}",
            run.losses
        ),
    };
    assert_eq!(
        gpu.diverged.diagnostic.to_string(),
        cpu.diagnostic.to_string()
    );
    assert_eq!(gpu.diverged.run.losses.len(), cpu.run.losses.len());
}

// ---------------------------------------------------------------------------------------------
// 4. FLOP rules against the instructions each module executes.
// ---------------------------------------------------------------------------------------------

const OP_CONSTANT: u32 = 43;
const OP_STORE: u32 = 62;
const OP_LOOP_MERGE: u32 = 246;
const OP_LABEL: u32 = 248;
const OP_BRANCH: u32 = 249;
const OP_BRANCH_CONDITIONAL: u32 = 250;
const OP_U_LESS_THAN: u32 = 176;
const OP_F_DIV: u32 = 136;
const OP_F_ORD_NOT_EQUAL: u32 = 182;

fn weight(op: u32, w: &[u32]) -> u64 {
    match op {
        OP_F_ADD | OP_F_SUB | OP_F_MUL | OP_F_DIV | OP_F_ORD_NOT_EQUAL => 1,
        OP_EXT_INST if w[4] == GLSL_FMA => 2,
        _ => 0,
    }
}

/// One block of a module: its label and its instructions, each an opcode with its words.
type Block<'m> = (u32, Vec<(u32, &'m [u32])>);

/// The weighted float instructions one invocation executes: each block once, the body of each
/// counted loop once per iteration, its bounds read from the constants the loop compares and
/// starts from. Fails on a loop nested in a loop or a bound that is not a constant.
fn executed_flop(module: &[u32]) -> u64 {
    let all = instructions(module);
    let constants: HashMap<u32, u32> = all
        .iter()
        .filter(|(op, w)| *op == OP_CONSTANT && w.len() == 4)
        .map(|(_, w)| (w[2], w[3]))
        .collect();
    // Blocks in emission order: label, instructions.
    let mut blocks: Vec<Block<'_>> = Vec::new();
    for &(op, w) in &all {
        if op == OP_LABEL {
            blocks.push((w[1], Vec::new()));
        } else if let Some(block) = blocks.last_mut() {
            block.1.push((op, w));
        }
    }
    let index: HashMap<u32, usize> = blocks.iter().enumerate().map(|(i, b)| (b.0, i)).collect();
    let mut times = vec![1u64; blocks.len()];
    for (h, (_, body)) in blocks.iter().enumerate() {
        let Some(&(_, merge)) = body.iter().find(|(op, _)| *op == OP_LOOP_MERGE) else {
            continue;
        };
        let continue_target = merge[2];
        // The start: the last store before the branch into the header, in the block before it.
        let (_, before) = &blocks[h - 1];
        let start = before
            .iter()
            .rev()
            .find(|(op, _)| *op == OP_STORE)
            .map(|(_, w)| constants[&w[2]])
            .expect("a constant start");
        // The check block: the header's branch target.
        let check = body
            .iter()
            .find(|(op, _)| *op == OP_BRANCH)
            .map(|(_, w)| index[&w[1]])
            .expect("a branch to the check");
        let less = blocks[check]
            .1
            .iter()
            .find(|(op, _)| *op == OP_U_LESS_THAN)
            .expect("a compare");
        let end = constants[&less.1[4]];
        let (_, branch) = blocks[check]
            .1
            .iter()
            .find(|(op, _)| *op == OP_BRANCH_CONDITIONAL)
            .expect("a conditional");
        let first = index[&branch[2]];
        let last = index[&continue_target];
        let iterations = u64::from(end.saturating_sub(start));
        for t in times.iter_mut().take(last).skip(first) {
            assert_eq!(*t, 1, "a nested loop");
            *t = iterations;
        }
    }
    blocks
        .iter()
        .zip(&times)
        .map(|((_, body), &t)| t * body.iter().map(|&(op, w)| weight(op, w)).sum::<u64>())
        .sum()
}

fn f32s(shape: &[u64]) -> Operand {
    Operand::Buffer(TensorType::f32(shape))
}

/// For shapes the story does not test, each kernel's FLOP is `M` times what one invocation
/// executes, and each module passes `spirv-val`.
#[test]
fn adv_flop_rules_equal_the_executed_instructions_at_other_shapes() {
    let mut wrong = Vec::new();
    for (n, m) in [
        (1u64, 1u64),
        (2, 3),
        (7, 1),
        (1023, 1025),
        (1025, 1023),
        (4096, 4096),
        (100_000, 2),
    ] {
        let columns = rowwise_kernel(
            &TensorOp::SigregColumns,
            vec![f32s(&[n, m])],
            TensorType::f32(&[m]),
        )
        .expect("a kernel");
        let grad = rowwise_kernel(
            &TensorOp::SigregColumnsGrad,
            vec![f32s(&[n, m]), f32s(&[m])],
            TensorType::f32(&[n, m]),
        )
        .expect("a kernel");
        for kernel in [&columns, &grad] {
            let words = emit(kernel).expect("emits");
            if let Err(error) = validate(&words) {
                wrong.push(format!("{} [{n}, {m}]: spirv-val: {error}", kernel.op));
            }
            let executed = m * executed_flop(&words);
            if kernel.flop() != Some(executed) {
                wrong.push(format!(
                    "{} [{n}, {m}]: rule {:?}, executed {executed}",
                    kernel.op,
                    kernel.flop()
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

// ---------------------------------------------------------------------------------------------
// 5. Refusals.
// ---------------------------------------------------------------------------------------------

/// `infer --backend vulkan` along the CLI's own path (`modelspec-cli/src/infer.rs`): compile for
/// Vulkan, a CPU checkpoint read back, the synthetic batch and the projections of the
/// checkpoint's step, then the device call: E-TARGET-011 at the first `sigreg` call, for the
/// three-site program and for the one whose `sigreg` reads a dropout output.
#[test]
fn adv_device_inference_with_sigreg_is_e_target_011_along_the_cli_path() {
    for (name, source, op) in [
        ("three.mspec", THREE_SITES, "sigreg("),
        ("drop.mspec", DROPOUT_SGD, "dropout("),
    ] {
        let (compiled, artifact) = compiled(name, source);
        let mut short = compiled.clone();
        short.config.steps = 2;
        let run = runner::run(&short).expect("the CPU run");
        let dir = scratch(&format!("infer-{name}"));
        let path = dir.join("ckpt.msckpt");
        checkpoint::write(&path, &Checkpoint::of_state(&short, 2, &run.state.values))
            .expect("written");
        let read = checkpoint::read(&path).expect("reads");
        let params = read.parameters_for(&compiled).expect("matches");
        let mut batch = synthetic(&compiled.config, &compiled.forward).expect("synthetic data");
        batch.inputs.extend(
            sigreg_projections(&compiled.forward.program, compiled.config.seed, read.step)
                .expect("projections"),
        );
        let refused = match infer_on_device(None, &compiled, &artifact, &params) {
            Err(DeviceRunError::Refused(diagnostic)) => diagnostic,
            Err(other) => panic!("{name}: refused before the run, not {other}"),
            Ok(_) => panic!("{name}: inference with sigreg is refused"),
        };
        assert_eq!(
            refused.code.to_string(),
            "E-TARGET-011",
            "{name}: {refused}"
        );
        let (line, column) = source
            .lines()
            .enumerate()
            .find_map(|(i, text)| text.find(op).map(|at| (i + 1, at + 1)))
            .expect("the call");
        let span = refused.span.as_ref().expect("a span");
        assert_eq!(
            (span.line as usize, span.column as usize),
            (line, column),
            "{name}: {refused}"
        );
    }
}
