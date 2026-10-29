//! story:vulkan-layer-norm-coalesced-rows: the `per_row` kernels of `LayerNorm`,
//! `LayerNormGradX` and the row moments of `LayerNormGradGamma` stage the rows of their workgroup
//! in workgroup memory, so a row is read as consecutive values by consecutive invocations, and
//! one invocation still adds each row in ascending order from its first term, as the CPU
//! interpreter does. On the device every element is the CPU's bits on the decider shapes
//! (`[65536, 256]`, `[4096, 256]`), BERT-tiny's (`[4096, 128]`, `[832, 128]`) and rows and widths
//! that do not fill a workgroup or a tile, in two runs, with the guard words unchanged; on the
//! decider's `[65536, 256]` a `LayerNorm` dispatch takes at most 536 µs and a `LayerNormGradX`
//! dispatch at most 1,167 µs, half of their mean site time in the 50ec6c2b profile of the
//! decider step (19.3 ms over 18 sites, 21.0 ms over 9).
//!
//! The host test runs by default. The device tests are ignored; run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_layer_norm_rows -- --ignored`,
//! the timing test with the GPU to itself and the display awake (AGENTS.md § Rules).
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;

use common::words::instructions;
use common::{Draw, GUARD_WORDS, guard, guarded};
use modelspec_core::ir::kernel::{Kernel, MIN_SHARED_MEMORY_BYTES, Operand};
use modelspec_core::ir::tensor::{TensorOp, TensorProgramBuilder, TensorType};
use modelspec_core::schedule::{layer_norm_grad_gamma_kernels, layer_norm_rows, rowwise_kernel};
use modelspec_runtime::cpu::interpreter::evaluate_seeded;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

/// `[n, h]`: the decider's token and context rows, BERT-tiny's rows and masked rows, then rows
/// one past a workgroup with a width one past a power of two, a width whose rows fill workgroup
/// memory in 8 rows, a width under a vector, one element, a width of no power of two, a width
/// whose `LayerNormGradX` rows do not fit workgroup memory whole, and one whose `LayerNorm` rows
/// do not either (`modelspec_core::schedule::layer_norm_rows`): those keep one invocation per
/// row reading the binding.
const SHAPES: [(u64, u64); 11] = [
    (65536, 256),
    (4096, 256),
    (4096, 128),
    (832, 128),
    (4097, 129),
    (130, 512),
    (65, 3),
    (1, 1),
    (63, 1000),
    (5, 3000),
    (3, 4097),
];

/// The decider shape the timing test runs.
const DECIDER: (u64, u64) = (65536, 256);

/// The most microseconds one dispatch of `LayerNorm` and of `LayerNormGradX` over [`DECIDER`]
/// may take: half the mean site time of the 50ec6c2b decider profile, 19.3 ms over 18 sites and
/// 21.0 ms over 9.
const LAYER_NORM_US: f64 = 19_300.0 / 18.0 / 2.0;
const LAYER_NORM_GRAD_X_US: f64 = 21_000.0 / 9.0 / 2.0;

/// The dispatches of the one timed command buffer of a kernel, and the first of them the timing
/// leaves out: a device that idled while the host drew and uploaded the operands runs them at
/// lower clocks. The rest are three runs of 100.
const DISPATCHES: u32 = 500;
const WARM: usize = 200;

/// The bits each output element holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_dead;

/// `OpVariable`, the type instructions that size workgroup memory and the `Workgroup` storage
/// class.
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_POINTER: u32 = 32;
const OP_CONSTANT: u32 = 43;
const OP_VARIABLE: u32 = 59;
const STORAGE_WORKGROUP: u32 = 4;

/// The three `per_row` kernels of the story.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Which {
    LayerNorm,
    LayerNormGradX,
    Moments,
}

const ALL: [Which; 3] = [Which::LayerNorm, Which::LayerNormGradX, Which::Moments];

impl Which {
    fn name(self) -> &'static str {
        match self {
            Which::LayerNorm => "LayerNorm",
            Which::LayerNormGradX => "LayerNormGradX",
            Which::Moments => "LayerNormGradGamma moments",
        }
    }

    /// The op whose rows the kernel adds.
    fn op(self) -> TensorOp {
        match self {
            Which::LayerNorm => TensorOp::LayerNorm,
            Which::LayerNormGradX => TensorOp::LayerNormGradX,
            Which::Moments => TensorOp::LayerNormGradGamma,
        }
    }

    /// The `[n, h]` bindings the kernel stages: `x`, and `g` for `LayerNormGradX`.
    fn staged(self) -> u64 {
        if self == Which::LayerNormGradX { 2 } else { 1 }
    }

    /// The `[n, h]` operands' lengths: `x, gamma, beta`; `x, gamma, g`; `x`.
    fn operand_lengths(self, n: u64, h: u64) -> Vec<u64> {
        match self {
            Which::LayerNorm | Which::LayerNormGradX => {
                vec![n * h, h, if self == Which::LayerNorm { h } else { n * h }]
            }
            Which::Moments => vec![n * h],
        }
    }

    fn result_len(self, n: u64, h: u64) -> u64 {
        match self {
            Which::Moments => 2 * n,
            _ => n * h,
        }
    }

    /// The kernel as the schedule builds it.
    fn kernel(self, n: u64, h: u64) -> Kernel {
        let rows = || Operand::Buffer(TensorType::f32(&[n, h]));
        let row = || Operand::Buffer(TensorType::f32(&[h]));
        let kernel = match self {
            Which::LayerNorm => rowwise_kernel(
                &TensorOp::LayerNorm,
                vec![rows(), row(), row()],
                TensorType::f32(&[n, h]),
            ),
            Which::LayerNormGradX => rowwise_kernel(
                &TensorOp::LayerNormGradX,
                vec![rows(), row(), rows()],
                TensorType::f32(&[n, h]),
            ),
            Which::Moments => {
                layer_norm_grad_gamma_kernels(vec![rows(), rows()], TensorType::f32(&[h]))
                    .map(|[moments, _]| moments)
            }
        };
        kernel.unwrap_or_else(|| panic!("{} [{n}, {h}] has a kernel", self.name()))
    }

    /// The CPU interpreter's result over `inputs`.
    fn cpu(self, n: u64, h: u64, inputs: &[Vec<f32>]) -> Vec<f32> {
        if self == Which::Moments {
            return inputs[0]
                .chunks(h as usize)
                .flat_map(|row| {
                    let (mean, rstd) = cpu_moments(row);
                    [mean, rstd]
                })
                .collect();
        }
        let mut builder = TensorProgramBuilder::new();
        let mut host = BTreeMap::new();
        let mut values = Vec::new();
        for (i, data) in inputs.iter().enumerate() {
            let row = i == 1 || (i == 2 && self == Which::LayerNorm);
            let shape = if row { vec![h] } else { vec![n, h] };
            let name = format!("v{i}");
            values.push(
                builder
                    .input(name.clone(), TensorType::f32(&shape))
                    .expect("an input"),
            );
            host.insert(
                name,
                HostTensor::new(shape, data.clone()).expect("a host tensor"),
            );
        }
        let out = match self {
            Which::LayerNorm => builder.layer_norm(values[0], values[1], values[2]),
            Which::LayerNormGradX => builder.layer_norm_grad_x(values[0], values[1], values[2]),
            Which::Moments => unreachable!("written out above"),
        }
        .expect("the op is well typed");
        builder.output(out).expect("the result is a value");
        let program = builder.finish();
        evaluate_seeded(&program, &host, &BTreeMap::new(), &BTreeMap::new(), 0)
            .expect("the CPU evaluates")[0]
            .data()
            .to_vec()
    }
}

/// The CPU interpreter's `row_moments` (`cpu/interpreter.rs`, DESIGN.md § 4), written out: the
/// row summed from its first element over the width, the squared deviations as a sum of
/// products over the width, and `1 / sqrt(variance + 1e-12)`, each operation rounded once.
fn cpu_moments(row: &[f32]) -> (f32, f32) {
    let width = row.len() as f32;
    let mean = row[1..].iter().fold(row[0], |acc, &x| acc + x) / width;
    let first = row[0] - mean;
    let squares = row[1..]
        .iter()
        .fold(first * first, |acc, &x| (x - mean).mul_add(x - mean, acc));
    let variance = squares / width;
    let rstd = 1.0f32 / (variance + 1e-12f32).sqrt();
    (mean, rstd)
}

/// The bytes of a module's workgroup memory, from its `OpVariable`s of the `Workgroup` storage
/// class.
fn workgroup_bytes(words: &[u32]) -> u64 {
    let mut sizes: BTreeMap<u32, u64> = BTreeMap::new();
    let mut constants: BTreeMap<u32, u64> = BTreeMap::new();
    let mut pointees: BTreeMap<u32, u32> = BTreeMap::new();
    let mut total = 0;
    for (op, w) in instructions(words) {
        match op {
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

/// Every shape's three kernels run workgroups of 64. Where [`layer_norm_rows`] rows of the width
/// fit, a workgroup stages exactly those rows of each binding it stages in workgroup memory,
/// within the 16 KiB every device has, and its workgroups cover them; where none fits, the kernel
/// keeps no workgroup memory and one invocation per row.
#[test]
fn layer_norm_row_kernels_stage_their_rows_in_workgroup_memory_within_16_kib() {
    let mut bad = Vec::new();
    let mut staged_shapes = 0;
    for (n, h) in SHAPES {
        for which in ALL {
            let kernel = which.kernel(n, h);
            let label = format!("{} [{n}, {h}]", which.name());
            if kernel.workgroup_size != 64 {
                bad.push(format!("{label}: workgroup size {}", kernel.workgroup_size));
            }
            let rows = layer_norm_rows(&which.op(), h).expect("a layer-norm op");
            let words = emit(&kernel).unwrap_or_else(|e| panic!("{label} emits: {e}"));
            let bytes = workgroup_bytes(&words);
            let invocations: u64 = kernel
                .workgroups
                .iter()
                .map(|&w| u64::from(w))
                .product::<u64>()
                * u64::from(kernel.workgroup_size);
            let (want_bytes, want_invocations) = if rows == 0 {
                (0, n)
            } else {
                staged_shapes += 1;
                (rows * h * which.staged() * 4, n.div_ceil(rows) * 64)
            };
            if bytes != want_bytes {
                bad.push(format!(
                    "{label}: {bytes} bytes of workgroup memory, not {want_bytes}"
                ));
            }
            if bytes > u64::from(MIN_SHARED_MEMORY_BYTES) {
                bad.push(format!(
                    "{label}: {bytes} bytes of workgroup memory, over 16 KiB"
                ));
            }
            if invocations < want_invocations {
                bad.push(format!(
                    "{label}: {invocations} invocations, fewer than {want_invocations}"
                ));
            }
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
    assert!(staged_shapes >= 3 * 9, "{staged_shapes} kernels stage rows");
}

fn bytes_of(words: u64) -> u64 {
    words * 4
}

fn open() -> Vulkan {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        std::env::var(VALIDATION_ENV).as_deref() == Ok("1"),
        "the validation layer is on exactly when {VALIDATION_ENV}=1"
    );
    vulkan
}

fn with_device(body: impl FnOnce(&Device<'_>)) {
    let vulkan = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    body(&device);
    let messages = vulkan.take_validation_messages();
    assert!(messages.is_empty(), "{messages:#?}");
}

/// The operands of `which` over `[n, h]`, drawn.
fn inputs(which: Which, n: u64, h: u64, draw: &mut Draw) -> Vec<Vec<f32>> {
    which
        .operand_lengths(n, h)
        .into_iter()
        .map(|len| draw.values(len))
        .collect()
}

/// Uploads `inputs` and a guarded output of `result` elements, then records `dispatches`
/// dispatches of `kernel`, each between two timestamps and followed by a barrier, and reads the
/// output back, in one command buffer with one host wait. Returns the output with its guard
/// words and the microseconds of each dispatch.
fn run(
    device: &Device<'_>,
    kernel: &Kernel,
    inputs: &[Vec<f32>],
    result: u64,
    dispatches: u32,
) -> (Vec<f32>, Vec<f64>) {
    let words = emit(kernel).expect("the kernel emits");
    let pipeline = device
        .kernel_pipeline(kernel, &words)
        .expect("the pipeline is created");
    let mut uploads = inputs.to_vec();
    uploads.push(guarded(&vec![f32::from_bits(UNWRITTEN); result as usize]));
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
    let pool = device
        .timestamp_pool(2 * dispatches)
        .expect("a timestamp pool");
    let output = buffers.len() - 1;
    let mut recording = device.record().expect("a recording starts");
    recording.reset_timestamps(&pool);
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("the upload is recorded");
    }
    recording.barrier(Barrier::TransferToCompute);
    let bound: Vec<&Buffer<'_>> = buffers.iter().collect();
    for i in 0..dispatches {
        recording
            .timestamp(&pool, 2 * i, false)
            .expect("the start is stamped");
        recording
            .dispatch_kernel(&pipeline, kernel, &bound)
            .expect("the dispatch is recorded");
        recording
            .timestamp(&pool, 2 * i + 1, true)
            .expect("the end is stamped");
        recording.barrier(Barrier::ComputeToCompute);
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
    let stamps = pool.read().expect("the timestamps read");
    let clock = device.clock().expect("the queue has a timestamp counter");
    let mask = match clock.valid_bits {
        0 | 64.. => u64::MAX,
        bits => (1u64 << bits) - 1,
    };
    let times = stamps
        .chunks(2)
        .map(|pair| {
            let ticks = (pair[1] & mask).wrapping_sub(pair[0] & mask) & mask;
            ticks as f64 * f64::from(clock.period_ns) / 1e3
        })
        .collect();
    (staging[output].read_f32().expect("the output reads"), times)
}

/// On the device, each of the three kernels gives the CPU's bits for every element of every
/// shape, in two runs, with the guard words after the output unchanged, under the validation
/// layer when it is on.
#[test]
#[ignore = "needs a Vulkan device"]
fn layer_norm_row_kernels_are_the_cpu_bits_on_the_device() {
    with_device(|device| {
        let mut draw = Draw::new(0x6c6e_726f_7773_0001);
        let mut bad = Vec::new();
        for (n, h) in SHAPES {
            for which in ALL {
                let label = format!("{} [{n}, {h}]", which.name());
                let inputs = inputs(which, n, h, &mut draw);
                let want = which.cpu(n, h, &inputs);
                let kernel = which.kernel(n, h);
                let len = which.result_len(n, h);
                let (first, _) = run(device, &kernel, &inputs, len, 1);
                let (second, _) = run(device, &kernel, &inputs, len, 1);
                let differ: Vec<usize> = (0..len as usize)
                    .filter(|&i| first[i].to_bits() != want[i].to_bits())
                    .collect();
                for &i in differ.iter().take(4) {
                    bad.push(format!(
                        "{label} element {i}: gpu {:?} cpu {:?}",
                        first[i], want[i]
                    ));
                }
                if !differ.is_empty() {
                    bad.push(format!(
                        "{label}: {} of {len} elements differ from the CPU",
                        differ.len()
                    ));
                }
                for i in 0..GUARD_WORDS {
                    if first[len as usize + i].to_bits() != guard(i) {
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
                eprintln!("{label}: {} elements differ from the CPU", differ.len());
            }
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// The median of `values`.
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// On the device, the median of three runs of the median time of 100 dispatches, after 200 that
/// bring the clocks up in the same command buffer, over the decider's `[65536, 256]` is at most
/// [`LAYER_NORM_US`] for `LayerNorm` and at most
/// [`LAYER_NORM_GRAD_X_US`] for `LayerNormGradX`; the moments kernel's is printed.
#[test]
#[ignore = "needs a Vulkan device, alone, at P0 or P2"]
fn layer_norm_row_kernels_of_the_decider_shape_take_half_their_profiled_time() {
    with_device(|device| {
        let (n, h) = DECIDER;
        let mut draw = Draw::new(0x6c6e_726f_7773_0002);
        let mut bad = Vec::new();
        for which in ALL {
            let inputs = inputs(which, n, h, &mut draw);
            let kernel = which.kernel(n, h);
            let len = which.result_len(n, h);
            let times = run(device, &kernel, &inputs, len, DISPATCHES).1;
            let runs: Vec<f64> = times[WARM..]
                .chunks((DISPATCHES as usize - WARM) / 3)
                .map(|run| median(run.to_vec()))
                .collect();
            let us = median(runs.clone());
            let moved: u64 = which.operand_lengths(n, h).iter().sum::<u64>() + len;
            let gbps = bytes_of(moved) as f64 / (us * 1e3);
            eprintln!(
                "{} [{n}, {h}]: {us:.1} µs per dispatch (runs {runs:.1?}), {gbps:.0} GB/s of its operands and result",
                which.name()
            );
            let bound = match which {
                Which::LayerNorm => Some(LAYER_NORM_US),
                Which::LayerNormGradX => Some(LAYER_NORM_GRAD_X_US),
                Which::Moments => None,
            };
            if let Some(bound) = bound.filter(|&bound| us > bound) {
                bad.push(format!(
                    "{} [{n}, {h}]: {us:.1} µs per dispatch, more than {bound:.1}",
                    which.name()
                ));
            }
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}
