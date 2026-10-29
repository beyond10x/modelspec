//! Adversary pass on story:vulkan-column-sums-coalesced: `SumRows` and `LayerNormGradGamma`
//! split columns over workgroups of 64 invocations ([`column_sum_columns`]), stage tiles of rows
//! in workgroup memory, add every column in ascending row order and keep the CPU interpreter's
//! bits, the `LayerNormGradGamma` sums starting at `-0.0`.
//!
//! The host tests run by default. The device tests are ignored; run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test adversary_column_sums -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;
use std::time::Instant;

use common::words::instructions;
use common::{Draw, GPU_ATOL, GPU_RTOL, GUARD_WORDS, flush_subnormals, guard, guarded, within};
use modelspec_core::ir::kernel::{Kernel, Operand};
use modelspec_core::ir::tensor::{TensorOp, TensorProgramBuilder, TensorType, ValueId};
use modelspec_core::schedule::{column_sum_columns, layer_norm_grad_gamma_kernels, rowwise_kernel};
use modelspec_runtime::cpu::interpreter::evaluate_seeded;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::pipeline::Pipeline;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

/// The bits each output element holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_dead;

/// `OpTypeFloat`, `OpTypeVector`, `OpTypeArray`, `OpTypePointer`, `OpConstant`, `OpVariable`
/// and the `Workgroup` storage class.
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_POINTER: u32 = 32;
const OP_CONSTANT: u32 = 43;
const OP_VARIABLE: u32 = 59;
const STORAGE_WORKGROUP: u32 = 4;

/// The workgroup memory every Vulkan device has (`maxComputeSharedMemorySize` minimum).
const MIN_SHARED_BYTES: u64 = 16 * 1024;

/// Row counts on both sides of every tile: 1, 63/64/65 (the 64-row `LayerNormGradGamma` tile),
/// 257 (one past the 256-row `SumRows` tile and its 1 + 256 first full tile), 4097.
const ROWS: [u64; 7] = [1, 2, 63, 64, 65, 257, 4097];
/// Widths around vec4 tails, the 16-column moment blocks, the workgroup columns and 4097.
const WIDTHS: [u64; 13] = [1, 2, 3, 5, 15, 16, 17, 31, 63, 65, 129, 1025, 4097];

struct Case {
    op: TensorOp,
    n: u64,
    h: u64,
    operands: Vec<(&'static str, Vec<f32>)>,
}

impl Case {
    fn label(&self) -> String {
        format!("{} [{}, {}]", self.op.name(), self.n, self.h)
    }
}

fn arity(op: &TensorOp) -> usize {
    match op {
        TensorOp::SumRows => 1,
        TensorOp::LayerNormGradGamma => 2,
        other => unreachable!("{} is no column sum", other.name()),
    }
}

fn names(op: &TensorOp) -> [&'static str; 2] {
    match op {
        TensorOp::SumRows => ["g", "unused"],
        _ => ["x", "g"],
    }
}

fn case(op: TensorOp, n: u64, h: u64, values: Vec<Vec<f32>>) -> Case {
    let names = names(&op);
    assert_eq!(values.len(), arity(&op));
    Case {
        op,
        n,
        h,
        operands: names.into_iter().zip(values).collect(),
    }
}

/// The kernels of `op` over `[n, h]`, as the schedule builds them: one for `SumRows`, the row
/// moments and the column sum for `LayerNormGradGamma`
/// (story:vulkan-layer-norm-grad-gamma-row-moments).
fn kernels_of(op: &TensorOp, n: u64, h: u64) -> Vec<Kernel> {
    let operands = (0..arity(op))
        .map(|_| Operand::Buffer(TensorType::f32(&[n, h])))
        .collect();
    let kernels = if *op == TensorOp::LayerNormGradGamma {
        layer_norm_grad_gamma_kernels(operands, TensorType::f32(&[h])).map(Vec::from)
    } else {
        rowwise_kernel(op, operands, TensorType::f32(&[h])).map(|kernel| vec![kernel])
    };
    kernels.unwrap_or_else(|| panic!("{} [{n}, {h}] has its kernels", op.name()))
}

/// The kernel of `op` over `[n, h]` that adds the columns: the last one.
fn kernel_of(op: &TensorOp, n: u64, h: u64) -> Kernel {
    kernels_of(op, n, h)
        .pop()
        .expect("a kernel writes the column sums")
}

fn cpu(case: &Case) -> Vec<f32> {
    let ty = TensorType::f32(&[case.n, case.h]);
    let mut builder = TensorProgramBuilder::new();
    let mut inputs = BTreeMap::new();
    let mut values: Vec<ValueId> = Vec::new();
    for (name, data) in &case.operands {
        inputs.insert(
            (*name).to_owned(),
            HostTensor::new(vec![case.n, case.h], data.clone()).expect("a host tensor"),
        );
        values.push(
            builder
                .input((*name).to_owned(), ty.clone())
                .expect("an input"),
        );
    }
    let out = match case.op {
        TensorOp::SumRows => builder.sum_rows(values[0]),
        TensorOp::LayerNormGradGamma => builder.layer_norm_grad_gamma(values[0], values[1]),
        ref other => unreachable!("{} is no column sum", other.name()),
    }
    .expect("the case is well typed");
    builder.output(out).expect("the result is a value");
    let program = builder.finish();
    evaluate_seeded(&program, &inputs, &BTreeMap::new(), &BTreeMap::new(), 0)
        .expect("the CPU evaluates")[0]
        .data()
        .to_vec()
}

fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

/// The kernels of a case and their pipelines.
struct Run<'d> {
    kernels: Vec<Kernel>,
    pipelines: Vec<Pipeline<'d>>,
}

/// The pipelines of the kernels of `op` over `[n, h]`, and the seconds the slowest took to
/// create.
fn pipeline<'d>(device: &'d Device<'_>, op: &TensorOp, n: u64, h: u64) -> (Run<'d>, f64) {
    let kernels = kernels_of(op, n, h);
    let mut slowest: f64 = 0.0;
    let pipelines = kernels
        .iter()
        .map(|kernel| {
            let words = emit(kernel).expect("the kernel emits");
            let started = Instant::now();
            let pipeline = device
                .kernel_pipeline(kernel, &words)
                .expect("the pipeline is created");
            slowest = slowest.max(started.elapsed().as_secs_f64());
            pipeline
        })
        .collect();
    (Run { kernels, pipelines }, slowest)
}

/// One run of `run` over `case` in one command buffer; the output and its guard words. The row
/// moments of `LayerNormGradGamma` read `x` into a buffer of `[n, 2]`, and after a barrier the
/// column sum reads it, then `x` and `g`, as the step plan runs them.
fn dispatch(device: &Device<'_>, run: &Run<'_>, case: &Case) -> Vec<f32> {
    let mut uploads: Vec<Vec<f32>> = case.operands.iter().map(|(_, v)| v.clone()).collect();
    uploads.push(guarded(&vec![f32::from_bits(UNWRITTEN); case.h as usize]));
    let buffers: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|v| {
            device
                .device_local_buffer(bytes_of(v.len()))
                .expect("a buffer")
        })
        .collect();
    let mut staging: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|v| {
            let mut buffer = device
                .staging_buffer(bytes_of(v.len()))
                .expect("a staging buffer");
            buffer.write_f32(v).expect("the staging buffer is written");
            buffer
        })
        .collect();
    let output = buffers.len() - 1;
    let moments = device
        .device_local_buffer(bytes_of(2 * case.n as usize))
        .expect("a moments buffer");
    let mut recording = device.record().expect("a recording starts");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("the upload is recorded");
    }
    recording.barrier(Barrier::TransferToCompute);
    match (&run.kernels[..], &run.pipelines[..]) {
        ([kernel], [pipeline]) => {
            let bound: Vec<&Buffer<'_>> = buffers.iter().collect();
            recording
                .dispatch_kernel(pipeline, kernel, &bound)
                .expect("the dispatch is recorded");
        }
        ([first, second], [first_pipeline, second_pipeline]) => {
            recording
                .dispatch_kernel(first_pipeline, first, &[&buffers[0], &moments])
                .expect("the moments dispatch is recorded");
            recording.barrier(Barrier::ComputeToCompute);
            let bound: Vec<&Buffer<'_>> = [&moments].into_iter().chain(&buffers).collect();
            recording
                .dispatch_kernel(second_pipeline, second, &bound)
                .expect("the column dispatch is recorded");
        }
        _ => unreachable!("one kernel or two"),
    }
    recording.barrier(Barrier::ComputeToTransfer);
    recording
        .copy(&buffers[output], &staging[output])
        .expect("the download is recorded");
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    device
        .submit(recorded)
        .expect("submitted")
        .wait()
        .expect("the device finishes");
    staging[output].read_f32().expect("the output reads")
}

fn with_device(body: impl FnOnce(&Vulkan, &Device<'_>)) {
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
    body(&vulkan, &device);
    let messages = vulkan.take_validation_messages();
    assert!(messages.is_empty(), "{messages:#?}");
}

/// How an element of the device result is compared with the CPU's.
#[derive(Clone, Copy, PartialEq)]
enum Compare {
    /// Bit for bit.
    Bits,
    /// Bit for bit, except that any NaN agrees with any NaN (a NaN's payload is the
    /// device's own, as before this unit).
    BitsNanClass,
    /// DESIGN.md § 9 gate 5 at `r = |c|`, NaN agreeing with NaN.
    Gate5,
}

fn agrees(g: f32, c: f32, how: Compare) -> bool {
    match how {
        Compare::Bits => g.to_bits() == c.to_bits(),
        Compare::BitsNanClass => (g.is_nan() && c.is_nan()) || g.to_bits() == c.to_bits(),
        Compare::Gate5 => {
            (g.is_nan() && c.is_nan()) || g == c || within(g, c, c.abs(), GPU_ATOL, GPU_RTOL)
        }
    }
}

/// What differs between `got` (with its guard words) and `want`: at most four columns named.
fn differences(label: &str, got: &[f32], want: &[f32], how: Compare) -> Vec<String> {
    let mut bad = Vec::new();
    let mut count = 0;
    for (i, (&g, &c)) in got.iter().zip(want).enumerate() {
        if !agrees(g, c, how) {
            count += 1;
            if count <= 4 {
                bad.push(format!(
                    "{label} column {i}: gpu {g:?} ({:#010x}) cpu {c:?} ({:#010x})",
                    g.to_bits(),
                    c.to_bits()
                ));
            }
        }
    }
    if count > 0 {
        bad.push(format!("{label}: {count} of {} columns differ", want.len()));
    }
    for i in 0..GUARD_WORDS {
        if got[want.len() + i].to_bits() != guard(i) {
            bad.push(format!("{label}: guard word {i} changed"));
        }
    }
    bad
}

/// Runs `case` `runs` times on one pipeline: what differs from `want`, and whether the runs
/// differ from each other.
fn check(device: &Device<'_>, case: &Case, want: &[f32], how: Compare, runs: usize) -> Vec<String> {
    let label = case.label();
    let (run, seconds) = pipeline(device, &case.op, case.n, case.h);
    if seconds > 1.0 {
        eprintln!("{label}: pipeline creation {seconds:.2} s");
    }
    let first = dispatch(device, &run, case);
    let mut bad = differences(&label, &first, want, how);
    for k in 1..runs {
        let again = dispatch(device, &run, case);
        if again
            .iter()
            .map(|v| v.to_bits())
            .ne(first.iter().map(|v| v.to_bits()))
        {
            bad.push(format!("{label}: run {k} differs from run 0"));
        }
    }
    bad
}

/// The `f32` values a module's workgroup memory holds, from its `OpVariable`s of the
/// `Workgroup` storage class.
fn workgroup_bytes(words: &[u32]) -> u64 {
    let all = instructions(words);
    let mut sizes: BTreeMap<u32, u64> = BTreeMap::new();
    let mut constants: BTreeMap<u32, u64> = BTreeMap::new();
    let mut pointees: BTreeMap<u32, u32> = BTreeMap::new();
    let mut total = 0;
    for (op, w) in &all {
        match *op {
            OP_TYPE_FLOAT => {
                sizes.insert(w[1], u64::from(w[2] / 8));
            }
            OP_TYPE_VECTOR => {
                let size = sizes.get(&w[2]).copied().unwrap_or(0) * u64::from(w[3]);
                sizes.insert(w[1], size);
            }
            OP_CONSTANT => {
                constants.insert(w[2], u64::from(w[3]));
            }
            OP_TYPE_ARRAY => {
                let size = sizes.get(&w[2]).copied().unwrap_or(0) * constants[&w[3]];
                sizes.insert(w[1], size);
            }
            OP_TYPE_POINTER if w[2] == STORAGE_WORKGROUP => {
                pointees.insert(w[1], w[3]);
            }
            OP_VARIABLE if w[3] == STORAGE_WORKGROUP => {
                total += sizes[&pointees[&w[1]]];
            }
            _ => {}
        }
    }
    total
}

/// Every attack shape of both column sums has a kernel of workgroups of 64 covering
/// `ceil(h / C)` workgroups of [`column_sum_columns`] columns, that emits, whose workgroup memory
/// is within the 16 KiB every device has.
#[test]
fn adv_cs_every_attack_shape_emits_within_the_minimum_workgroup_memory() {
    let mut bad = Vec::new();
    for op in [TensorOp::SumRows, TensorOp::LayerNormGradGamma] {
        for n in ROWS {
            for h in WIDTHS.into_iter().chain([8192, 16384, 65536]) {
                let label = format!("{} [{n}, {h}]", op.name());
                let kernel = kernel_of(&op, n, h);
                let columns = column_sum_columns(&op, h).expect("a column sum");
                let groups: u64 = kernel.workgroups.iter().map(|&w| u64::from(w)).product();
                if kernel.workgroup_size != 64 || groups != h.div_ceil(columns) {
                    bad.push(format!(
                        "{label}: {groups} workgroups of {}, want {} of 64",
                        kernel.workgroup_size,
                        h.div_ceil(columns)
                    ));
                }
                match emit(&kernel) {
                    Ok(words) => {
                        let bytes = workgroup_bytes(&words);
                        if bytes == 0 || bytes > MIN_SHARED_BYTES {
                            bad.push(format!("{label}: {bytes} bytes of workgroup memory"));
                        }
                    }
                    Err(e) => bad.push(format!("{label}: does not emit: {e}")),
                }
                let mut kernels = kernels_of(&op, n, h);
                kernels.pop();
                for moments in kernels {
                    match emit(&moments) {
                        Ok(words) => {
                            let bytes = workgroup_bytes(&words);
                            if bytes > MIN_SHARED_BYTES {
                                bad.push(format!(
                                    "{label} moments: {bytes} bytes of workgroup memory"
                                ));
                            }
                        }
                        Err(e) => bad.push(format!("{label} moments: does not emit: {e}")),
                    }
                }
            }
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

/// A column sum with zero rows or zero columns has no row-wise kernel, and a kernel built by
/// hand over such operands does not emit.
#[test]
fn adv_cs_zero_rows_or_columns_are_refused_by_schedule_and_emitter() {
    let mut bad = Vec::new();
    for op in [TensorOp::SumRows, TensorOp::LayerNormGradGamma] {
        for (n, h) in [(0, 5), (5, 0), (0, 0)] {
            let operands: Vec<Operand> = (0..arity(&op))
                .map(|_| Operand::Buffer(TensorType::f32(&[n, h])))
                .collect();
            let scheduled = match op {
                TensorOp::LayerNormGradGamma => {
                    layer_norm_grad_gamma_kernels(operands.clone(), TensorType::f32(&[h])).is_some()
                }
                _ => rowwise_kernel(&op, operands.clone(), TensorType::f32(&[h])).is_some(),
            };
            if scheduled {
                bad.push(format!(
                    "{} [{n}, {h}]: the schedule gives a kernel",
                    op.name()
                ));
            }
            let mut kernels = kernels_of(&op, 1, 5);
            let mut column = kernels.pop().expect("a column kernel");
            let moments = TensorType::f32(&[n, 2]);
            column.operands = match op {
                TensorOp::LayerNormGradGamma => [Operand::Buffer(moments.clone())]
                    .into_iter()
                    .chain(operands.clone())
                    .collect(),
                _ => operands.clone(),
            };
            column.result = TensorType::f32(&[h]);
            if emit(&column).is_ok() {
                bad.push(format!("{} [{n}, {h}]: the emitter emits", op.name()));
            }
            for mut kernel in kernels {
                kernel.operands = vec![operands[0].clone()];
                kernel.result = moments.clone();
                if emit(&kernel).is_ok() {
                    bad.push(format!(
                        "{} [{n}, {h}] moments: the emitter emits",
                        op.name()
                    ));
                }
            }
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

/// Drawn values over every attack shape: both column sums are the CPU's bits, the guard words
/// stay, and three runs of one pipeline give the same bits (a missing barrier shows as runs
/// that differ or wrong bits on some shapes).
#[test]
#[ignore = "device: column sums over the attack grid against the CPU bits"]
fn adv_cs_attack_grid_is_the_cpu_bits_in_three_runs() {
    with_device(|_, device| {
        let mut draw = Draw::new(0x6164_765f_6373_3031);
        let mut bad = Vec::new();
        for n in ROWS {
            for h in WIDTHS {
                if n * h > 4097 * 1025 && !(n == 4097 && h == 4097) {
                    continue;
                }
                for op in [TensorOp::SumRows, TensorOp::LayerNormGradGamma] {
                    let values = (0..arity(&op)).map(|_| draw.values(n * h)).collect();
                    let case = case(op, n, h, values);
                    let want = cpu(&case);
                    bad.extend(check(device, &case, &want, Compare::Bits, 3));
                }
            }
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// Signed zeros. `SumRows` columns of all `-0.0`, all `+0.0` and alternating signs are the
/// CPU's bits (`-0.0`, `+0.0`, by the first terms). `LayerNormGradGamma` over rows of `x` all
/// `-0.0` (mean `-0.0`, deviations `+0.0`, so `xhat = +0.0`) and over rows of ones (`xhat` an
/// exact `+0.0`), with `g` of one sign per column: every term is a zero, and the CPU's sum is
/// the sign of its first term, which the `-0.0` start must keep; a start at `+0.0`, of either
/// the column sum or the row mean, flips it.
#[test]
#[ignore = "device: signed-zero column sums against the CPU bits"]
fn adv_cs_signed_zero_columns_are_the_cpu_bits() {
    with_device(|_, device| {
        let mut bad = Vec::new();
        for n in [1, 2, 64, 65, 257, 4097] {
            for h in [1, 3, 5, 17, 129] {
                let zeros: Vec<f32> = (0..n * h)
                    .map(|i| {
                        let (r, c) = (i / h, i % h);
                        match c % 3 {
                            0 => -0.0,
                            1 => 0.0,
                            _ if r % 2 == 0 => -0.0,
                            _ => 0.0,
                        }
                    })
                    .collect();
                let sum = case(TensorOp::SumRows, n, h, vec![zeros.clone()]);
                let want = cpu(&sum);
                bad.extend(check(device, &sum, &want, Compare::Bits, 3));
                let g: Vec<f32> = (0..n * h)
                    .map(|i| {
                        let c = i % h;
                        let magnitude = 0.25 + (i % 11) as f32;
                        if c % 2 == 0 { -magnitude } else { magnitude }
                    })
                    .collect();
                for x in [
                    vec![-0.0_f32; (n * h) as usize],
                    vec![1.0; (n * h) as usize],
                ] {
                    let gamma = case(TensorOp::LayerNormGradGamma, n, h, vec![x, g.clone()]);
                    let want = cpu(&gamma);
                    bad.extend(check(device, &gamma, &want, Compare::Bits, 3));
                }
                let gamma = case(
                    TensorOp::LayerNormGradGamma,
                    n,
                    h,
                    vec![vec![1.0; (n * h) as usize], zeros],
                );
                let want = cpu(&gamma);
                bad.extend(check(device, &gamma, &want, Compare::Bits, 3));
            }
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// Infinities and NaNs in `g` and `x` at rows on tile edges (0, 63, 64, 255, 256, 257, the
/// last), in some columns: the CPU's bits, any NaN agreeing with any NaN.
#[test]
#[ignore = "device: non-finite column sums against the CPU"]
fn adv_cs_non_finite_terms_are_the_cpu_class() {
    with_device(|_, device| {
        let mut draw = Draw::new(0x6164_765f_6373_3032);
        let mut bad = Vec::new();
        for (n, h) in [(1, 5), (65, 17), (257, 33), (4097, 129)] {
            let edge = |r: u64| [0, 63, 64, 255, 256, 257, n - 1].contains(&r);
            let poison = |values: &mut [f32], salt: u64| {
                for (i, v) in values.iter_mut().enumerate() {
                    let (r, c) = (i as u64 / h, i as u64 % h);
                    if edge(r) && (c + salt).is_multiple_of(4) {
                        *v = match (r + c) % 3 {
                            0 => f32::INFINITY,
                            1 => f32::NEG_INFINITY,
                            _ => f32::NAN,
                        };
                    }
                }
            };
            let mut g = draw.values(n * h);
            poison(&mut g, 0);
            let sum = case(TensorOp::SumRows, n, h, vec![g.clone()]);
            let want = cpu(&sum);
            bad.extend(check(device, &sum, &want, Compare::BitsNanClass, 2));
            let mut x = draw.values(n * h);
            poison(&mut x, 1);
            for operands in [
                vec![x.clone(), draw.values(n * h)],
                vec![draw.values(n * h), g],
            ] {
                let gamma = case(TensorOp::LayerNormGradGamma, n, h, operands);
                let want = cpu(&gamma);
                bad.extend(check(device, &gamma, &want, Compare::BitsNanClass, 2));
            }
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// Large-magnitude cancellation, where any other order of addition changes the bits: `SumRows`
/// of columns mixing `±2^24..2^30` and small values, `LayerNormGradGamma` over rows offset by
/// `1e4` with a spread of `1e-2` and gradients of `±1e3`.
#[test]
#[ignore = "device: cancelling column sums against the CPU bits"]
fn adv_cs_cancelling_magnitudes_are_the_cpu_bits() {
    with_device(|_, device| {
        let mut draw = Draw::new(0x6164_765f_6373_3033);
        let mut bad = Vec::new();
        for (n, h) in [(2, 3), (65, 5), (257, 63), (4097, 129), (1000, 1025)] {
            let base = draw.values(n * h);
            let big: Vec<f32> = base
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    if i % 3 == 0 {
                        v * f32::from_bits(0x4b80_0000 + ((i as u32 % 7) << 23))
                    } else {
                        v
                    }
                })
                .collect();
            let sum = case(TensorOp::SumRows, n, h, vec![big]);
            let want = cpu(&sum);
            bad.extend(check(device, &sum, &want, Compare::Bits, 3));
            let x: Vec<f32> = draw
                .values(n * h)
                .iter()
                .map(|v| 1e4 + v * 2.5e-3)
                .collect();
            let g: Vec<f32> = draw.values(n * h).iter().map(|v| v * 250.0).collect();
            let gamma = case(TensorOp::LayerNormGradGamma, n, h, vec![x, g]);
            let want = cpu(&gamma);
            bad.extend(check(device, &gamma, &want, Compare::Bits, 3));
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// Subnormal operands: against DESIGN.md § 9 gate 5's reference, the CPU on the operands as
/// they are on a device that preserves subnormals, else with every subnormal flushed; `SumRows`
/// within gate 5 there (a class of § 5.4), `LayerNormGradGamma` too.
#[test]
#[ignore = "device: subnormal column sums against the flushed reference"]
fn adv_cs_subnormal_terms_are_within_the_flushed_reference() {
    with_device(|vulkan, device| {
        let candidates = vulkan.candidates().expect("the devices are probed");
        let preserves = candidates[0].profile.shader_denorm_preserve_f32 == Some(true);
        let mut draw = Draw::new(0x6164_765f_6373_3034);
        let mut bad = Vec::new();
        for (n, h) in [(1, 5), (65, 17), (4097, 129)] {
            let sub: Vec<f32> = draw
                .values(n * h)
                .iter()
                .enumerate()
                .map(|(i, &v)| if i % 2 == 0 { v * 1e-39 } else { v })
                .collect();
            let reference = |values: &[f32]| {
                if preserves {
                    values.to_vec()
                } else {
                    flush_subnormals(values)
                }
            };
            let sum = case(TensorOp::SumRows, n, h, vec![sub.clone()]);
            let want = cpu(&case(TensorOp::SumRows, n, h, vec![reference(&sub)]));
            bad.extend(check(device, &sum, &want, Compare::Gate5, 2));
            let x = draw.values(n * h);
            let gamma = case(
                TensorOp::LayerNormGradGamma,
                n,
                h,
                vec![x.clone(), sub.clone()],
            );
            let want = cpu(&case(
                TensorOp::LayerNormGradGamma,
                n,
                h,
                vec![x, reference(&sub)],
            ));
            bad.extend(check(device, &gamma, &want, Compare::Gate5, 2));
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// Wide rows: `SumRows` over 65536 columns and `LayerNormGradGamma` over 2048 are the CPU's
/// bits.
#[test]
#[ignore = "device: wide-row column sums against the CPU bits"]
fn adv_cs_wide_rows_are_the_cpu_bits() {
    with_device(|_, device| {
        let mut draw = Draw::new(0x6164_765f_6373_3035);
        let mut bad = Vec::new();
        for (op, n, h) in [
            (TensorOp::SumRows, 65, 65536),
            (TensorOp::LayerNormGradGamma, 65, 2048),
        ] {
            let values = (0..arity(&op)).map(|_| draw.values(n * h)).collect();
            let case = case(op, n, h, values);
            let want = cpu(&case);
            bad.extend(check(device, &case, &want, Compare::Bits, 2));
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// The `LayerNormGradGamma` kernels over `[rows, width]`, as the schedule builds them: the row
/// moments, then the column sum.
fn layer_norm_grad_gamma(rows: u64, width: u64) -> Vec<Kernel> {
    kernels_of(&TensorOp::LayerNormGradGamma, rows, width)
}

/// Finding CS-1: the `LayerNormGradGamma` module of story:vulkan-column-sums-coalesced unrolled
/// the moment loop, two steps of 16 loads, stores and adds for every 16 columns of the row, so
/// its size grew with the row width. Since story:vulkan-layer-norm-grad-gamma-row-moments the
/// moments are a kernel of their own that loops over the columns, and the column sum unrolls
/// nothing per column. Over 4096 columns each module is at most twice its size over 128.
#[test]
fn adv_cs_1_layer_norm_grad_gamma_module_does_not_grow_with_the_row_width() {
    let mut words: Vec<Vec<usize>> = Vec::new();
    for width in [128, 1024, 4096] {
        let modules: Vec<usize> = layer_norm_grad_gamma(4097, width)
            .iter()
            .map(|kernel| emit(kernel).expect("the kernel emits").len())
            .collect();
        eprintln!("LayerNormGradGamma [4097, {width}]: {modules:?} words");
        words.push(modules);
    }
    for (k, (&wide, &narrow)) in words[2].iter().zip(&words[0]).enumerate() {
        assert!(
            wide <= 2 * narrow,
            "LayerNormGradGamma kernel {k} over [4097, 4096] is {wide} words, over [4097, 128] \
             {narrow}"
        );
    }
}

/// Finding CS-1 on the device: `train --backend vulkan` creates every pipeline of the step
/// before the first step, and the `LayerNormGradGamma` pipeline of a `[4097, 4096]` column sum
/// (a `layer_norm` of width 4096 over 4097 rows) took the driver minutes to compile. Each of its
/// pipelines is created within 10 s here.
#[test]
#[ignore = "device: LayerNormGradGamma pipelines of up to 4096 columns against a 10 s bound"]
fn adv_cs_1_layer_norm_grad_gamma_pipeline_of_4096_columns_is_created_within_10_s() {
    with_device(|_, device| {
        let mut bad = Vec::new();
        for width in [128, 1024, 4096] {
            for (k, kernel) in layer_norm_grad_gamma(4097, width).iter().enumerate() {
                let words = emit(kernel).expect("the kernel emits");
                let started = Instant::now();
                let created = device.kernel_pipeline(kernel, &words);
                let seconds = started.elapsed().as_secs_f64();
                eprintln!(
                    "LayerNormGradGamma [4097, {width}] kernel {k}: {} words, pipeline \
                     creation {seconds:.2} s",
                    words.len()
                );
                if created.is_err() {
                    bad.push(format!(
                        "[4097, {width}] kernel {k}: the pipeline is not created"
                    ));
                } else if seconds > 10.0 {
                    bad.push(format!(
                        "[4097, {width}] kernel {k}: pipeline creation took {seconds:.1} s"
                    ));
                }
            }
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}
