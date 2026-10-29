//! story:vulkan-column-sums-coalesced: the column sums of the BERT step, `SumRows` and
//! `LayerNormGradGamma`, add each column's rows from row 0 in ascending order, as the CPU
//! interpreter does, and spread the columns over workgroups of at most
//! [`MOST_COLUMNS`] columns that stage tiles of rows in workgroup memory, so a `[4097, 129]`
//! column sum runs on at least 17 workgroups instead of 3. On the device both give the CPU's
//! bits for `[4097, 129]`: 4097 rows, one past a multiple of every tile, and 129 columns, one
//! past a multiple of every workgroup's columns.
//!
//! The device test is ignored by default. Run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_column_sums -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;

use common::words::instructions;
use common::{Draw, GUARD_WORDS, guard, guarded};
use modelspec_core::ir::kernel::{Kernel, Operand};
use modelspec_core::ir::tensor::{TensorOp, TensorProgramBuilder, TensorType, ValueId};
use modelspec_core::schedule::{layer_norm_grad_gamma_kernels, rowwise_kernel};
use modelspec_runtime::cpu::interpreter::evaluate_seeded;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

/// Rows and columns of the cases.
const N: u64 = 4097;
const H: u64 = 129;

/// The most columns one workgroup of a column sum adds.
const MOST_COLUMNS: u64 = 8;

/// `OpVariable` and its `Workgroup` storage class.
const OP_VARIABLE: u32 = 59;
const STORAGE_WORKGROUP: u32 = 4;

/// The bits each output element holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_dead;

/// One column sum: the op and its operands' names and values, each `[N, H]`.
struct Case {
    op: TensorOp,
    operands: Vec<(&'static str, Vec<f32>)>,
}

fn rows() -> TensorType {
    TensorType::f32(&[N, H])
}

fn result() -> TensorType {
    TensorType::f32(&[H])
}

fn cases() -> Vec<Case> {
    let mut draw = Draw::new(0x636f_6c75_6d6e_7321);
    let count = N * H;
    vec![
        Case {
            op: TensorOp::SumRows,
            operands: vec![("g", draw.values(count))],
        },
        Case {
            op: TensorOp::LayerNormGradGamma,
            operands: vec![("x", draw.values(count)), ("g", draw.values(count))],
        },
    ]
}

/// The kernels of `op` over `[N, H]` operands, as the schedule builds them: one for `SumRows`,
/// the row moments and the column sum for `LayerNormGradGamma`
/// (story:vulkan-layer-norm-grad-gamma-row-moments).
fn kernels_of(op: &TensorOp, operands: usize) -> Vec<Kernel> {
    let operands = (0..operands).map(|_| Operand::Buffer(rows())).collect();
    if *op == TensorOp::LayerNormGradGamma {
        let kernels = layer_norm_grad_gamma_kernels(operands, result());
        return Vec::from(kernels.expect("LayerNormGradGamma has its two kernels"));
    }
    vec![
        rowwise_kernel(op, operands, result())
            .unwrap_or_else(|| panic!("{} has a row-wise kernel", op.name())),
    ]
}

/// The kernel of `op` that adds the columns: the last one.
fn kernel_of(op: &TensorOp, operands: usize) -> Kernel {
    kernels_of(op, operands)
        .pop()
        .expect("a kernel writes the column sums")
}

/// What the kernel of `op` does not do of the story: workgroups of 64 invocations, at least
/// `ceil(H / MOST_COLUMNS)` of them, a module that stages rows in workgroup memory.
fn structure(op: &TensorOp, operands: usize) -> Vec<String> {
    let kernel = kernel_of(op, operands);
    let name = op.name();
    let mut bad = Vec::new();
    if kernel.workgroup_size != 64 {
        bad.push(format!("{name}: workgroup size {}", kernel.workgroup_size));
    }
    let workgroups: u64 = kernel.workgroups.iter().map(|&w| u64::from(w)).product();
    let least = H.div_ceil(MOST_COLUMNS);
    if workgroups < least {
        bad.push(format!(
            "{name} [{N}, {H}]: {workgroups} workgroups, fewer than {least} of at most \
             {MOST_COLUMNS} columns"
        ));
    }
    let words = emit(&kernel).unwrap_or_else(|e| panic!("{name} emits: {e}"));
    let staged = instructions(&words)
        .iter()
        .any(|(op, w)| *op == OP_VARIABLE && w[3] == STORAGE_WORKGROUP);
    if !staged {
        bad.push(format!("{name}: no workgroup memory"));
    }
    bad
}

/// The `[4097, 129]` `SumRows` and `LayerNormGradGamma` spread their 129 columns over workgroups
/// of 64 invocations and at most 8 columns, each staging rows in workgroup memory.
#[test]
fn column_sums_spread_over_workgroups_of_at_most_eight_columns_that_stage_rows() {
    let mut bad = structure(&TensorOp::SumRows, 1);
    bad.extend(structure(&TensorOp::LayerNormGradGamma, 2));
    assert!(bad.is_empty(), "{bad:#?}");
}

/// The CPU interpreter's result of `case`: a program of its one op over its operands.
fn cpu(case: &Case) -> Vec<f32> {
    let mut builder = TensorProgramBuilder::new();
    let mut inputs = BTreeMap::new();
    let mut values: Vec<ValueId> = Vec::new();
    for (name, data) in &case.operands {
        inputs.insert(
            (*name).to_owned(),
            HostTensor::new(vec![N, H], data.clone()).expect("a host tensor"),
        );
        values.push(
            builder
                .input((*name).to_owned(), rows())
                .expect("a new input"),
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

/// Runs the kernels of `case` once on `device` in one command buffer with one host wait, as the
/// step plan runs them, and returns its output with the guard words after it. The row moments
/// of `LayerNormGradGamma` read `x` into a buffer of `[N, 2]`, and after a barrier the column
/// sum reads it, then `x` and `g`.
fn gpu(device: &Device<'_>, case: &Case) -> Vec<f32> {
    let kernels = kernels_of(&case.op, case.operands.len());
    let pipelines: Vec<_> = kernels
        .iter()
        .map(|kernel| {
            let words = emit(kernel).expect("the kernel emits");
            device
                .kernel_pipeline(kernel, &words)
                .expect("the pipeline is created")
        })
        .collect();
    let mut uploads: Vec<Vec<f32>> = case.operands.iter().map(|(_, v)| v.clone()).collect();
    uploads.push(guarded(&vec![f32::from_bits(UNWRITTEN); H as usize]));
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
        .device_local_buffer(bytes_of(2 * N as usize))
        .expect("a moments buffer");
    let mut recording = device.record().expect("a recording starts");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("the upload is recorded");
    }
    recording.barrier(Barrier::TransferToCompute);
    match (&kernels[..], &pipelines[..]) {
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

/// On the device, the `[4097, 129]` `SumRows` and `LayerNormGradGamma` are the CPU's bits in two
/// runs, the guard words after their output unchanged, and their kernels are the spread ones of
/// [`column_sums_spread_over_workgroups_of_at_most_eight_columns_that_stage_rows`].
#[test]
#[ignore = "needs a Vulkan device"]
fn column_sums_of_4097_by_129_are_the_cpu_bits_on_the_device() {
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
    let mut bad = Vec::new();
    for case in cases() {
        let name = case.op.name();
        bad.extend(structure(&case.op, case.operands.len()));
        let want = cpu(&case);
        let first = gpu(&device, &case);
        let second = gpu(&device, &case);
        let mut differ = 0;
        for (i, (&g, &c)) in first.iter().zip(&want).enumerate() {
            if g.to_bits() != c.to_bits() {
                differ += 1;
                if differ <= 4 {
                    bad.push(format!("{name} column {i}: gpu {g:?} cpu {c:?}"));
                }
            }
        }
        if differ > 0 {
            bad.push(format!(
                "{name}: {differ} of {H} columns differ from the CPU"
            ));
        }
        for i in 0..GUARD_WORDS {
            if first[want.len() + i].to_bits() != guard(i) {
                bad.push(format!("{name}: guard word {i} changed"));
            }
        }
        if first
            .iter()
            .map(|v| v.to_bits())
            .ne(second.iter().map(|v| v.to_bits()))
        {
            bad.push(format!("{name}: two runs differ"));
        }
        eprintln!("{name} [{N}, {H}]: {differ} columns differ from the CPU");
    }
    let messages = vulkan.take_validation_messages();
    assert!(messages.is_empty(), "{messages:#?}");
    assert!(bad.is_empty(), "{bad:#?}");
}
