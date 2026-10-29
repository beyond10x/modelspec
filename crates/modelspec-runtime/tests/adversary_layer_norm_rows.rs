//! Adversary pass on story:vulkan-layer-norm-coalesced-rows: the `per_row` kernels of
//! `LayerNorm`, `LayerNormGradX` and the row moments of `LayerNormGradGamma` stage
//! `schedule::layer_norm_rows` whole rows per workgroup of 64 in 16 KiB of workgroup memory, or
//! keep one invocation per row where fewer than `schedule::LAYER_NORM_MIN_ROWS` rows fit.
//!
//! Host tests run by default. Device tests are ignored; run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test adversary_layer_norm_rows -- --ignored`.
//! The `timing_` tests need the GPU to themselves.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use common::{Draw, GUARD_WORDS, guard, guarded};
use modelspec_core::ir::kernel::{Kernel, MIN_SHARED_MEMORY_BYTES, Operand};
use modelspec_core::ir::tensor::{TensorOp, TensorProgramBuilder, TensorType};
use modelspec_core::schedule::{
    LAYER_NORM_MIN_ROWS, LAYER_NORM_MOMENTS_MAX_ROWS, layer_norm_grad_gamma_kernels,
    layer_norm_per_row, layer_norm_row_stride, layer_norm_rows, rowwise_kernel,
};
use modelspec_runtime::cpu::interpreter::evaluate_seeded;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

const UNWRITTEN: u32 = 0x7fc0_dead;

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

    fn op(self) -> TensorOp {
        match self {
            Which::LayerNorm => TensorOp::LayerNorm,
            Which::LayerNormGradX => TensorOp::LayerNormGradX,
            Which::Moments => TensorOp::LayerNormGradGamma,
        }
    }

    fn staged(self) -> u64 {
        if self == Which::LayerNormGradX { 2 } else { 1 }
    }

    fn rows(self, h: u64) -> u64 {
        layer_norm_rows(&self.op(), h).expect("a layer-norm op")
    }

    fn operand_lengths(self, n: u64, h: u64) -> Vec<u64> {
        match self {
            Which::LayerNorm => vec![n * h, h, h],
            Which::LayerNormGradX => vec![n * h, h, n * h],
            Which::Moments => vec![n * h],
        }
    }

    fn result_len(self, n: u64, h: u64) -> u64 {
        match self {
            Which::Moments => 2 * n,
            _ => n * h,
        }
    }

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

    /// The CPU interpreter's result; the moments as `row_moments` computes them.
    fn cpu(self, n: u64, h: u64, inputs: &[Vec<f32>]) -> Vec<f32> {
        if self == Which::Moments {
            return inputs[0]
                .chunks(h as usize)
                .flat_map(|row| {
                    let (mean, rstd) = row_moments(row);
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
            Which::Moments => unreachable!(),
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

/// `cpu/interpreter.rs` `row_moments` for `f32`, written out.
fn row_moments(row: &[f32]) -> (f32, f32) {
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

/// Uploads `inputs` (each followed by guard words) and a guarded output, records `dispatches`
/// dispatches, reads back the output and every input. Returns output, inputs as read back,
/// and the microseconds of each dispatch.
fn run(
    device: &Device<'_>,
    kernel: &Kernel,
    inputs: &[Vec<f32>],
    result: u64,
    dispatches: u32,
) -> (Vec<f32>, Vec<Vec<f32>>, Vec<f64>) {
    let words = emit(kernel).expect("the kernel emits");
    let pipeline = device
        .kernel_pipeline(kernel, &words)
        .expect("the pipeline is created");
    let mut uploads: Vec<Vec<f32>> = inputs.to_vec();
    uploads.push(guarded(&vec![f32::from_bits(UNWRITTEN); result as usize]));
    let buffers: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|v| {
            device
                .device_local_buffer(4 * v.len() as u64)
                .expect("a buffer")
        })
        .collect();
    let mut staging: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|v| {
            let mut buffer = device
                .staging_buffer(4 * v.len() as u64)
                .expect("a staging buffer");
            buffer.write_f32(v).expect("the staging buffer is written");
            buffer
        })
        .collect();
    let pool = device
        .timestamp_pool(2 * dispatches)
        .expect("a timestamp pool");
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
    for (from, to) in buffers.iter().zip(&staging) {
        recording.copy(from, to).expect("the download is recorded");
    }
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
    let mut read: Vec<Vec<f32>> = staging
        .iter_mut()
        .map(|s| s.read_f32().expect("reads"))
        .collect();
    let out = read.pop().expect("the output");
    (out, read, times)
}

/// Same bits, or both NaN (the payload is not claimed).
fn same(g: f32, c: f32) -> bool {
    g.to_bits() == c.to_bits() || (g.is_nan() && c.is_nan())
}

/// Runs `which` over `[n, h]` with `inputs` and returns the mismatches against the CPU.
fn check(device: &Device<'_>, which: Which, n: u64, h: u64, inputs: &[Vec<f32>]) -> Vec<String> {
    let label = format!(
        "{} [{n}, {h}] (rows per group {})",
        which.name(),
        which.rows(h)
    );
    let want = which.cpu(n, h, inputs);
    let kernel = which.kernel(n, h);
    let len = which.result_len(n, h) as usize;
    let (got, read, _) = run(device, &kernel, inputs, len as u64, 1);
    let mut bad = Vec::new();
    let differ: Vec<usize> = (0..len).filter(|&i| !same(got[i], want[i])).collect();
    for &i in differ.iter().take(3) {
        bad.push(format!(
            "{label} element {i} (row {}): gpu {:?} ({:#010x}) cpu {:?} ({:#010x})",
            if which == Which::Moments {
                i / 2
            } else {
                i / h as usize
            },
            got[i],
            got[i].to_bits(),
            want[i],
            want[i].to_bits()
        ));
    }
    if !differ.is_empty() {
        bad.push(format!(
            "{label}: {} of {len} elements differ",
            differ.len()
        ));
    }
    for i in 0..GUARD_WORDS {
        if got[len + i].to_bits() != guard(i) {
            bad.push(format!("{label}: guard word {i} changed"));
        }
    }
    for (k, (before, after)) in inputs.iter().zip(&read).enumerate() {
        if before
            .iter()
            .map(|v| v.to_bits())
            .ne(after.iter().map(|v| v.to_bits()))
        {
            bad.push(format!("{label}: input {k} changed"));
        }
    }
    bad
}

fn drawn(which: Which, n: u64, h: u64, draw: &mut Draw) -> Vec<Vec<f32>> {
    which
        .operand_lengths(n, h)
        .into_iter()
        .map(|len| draw.values(len))
        .collect()
}

/// The widths at either side of every place the rows per workgroup change, and odd widths.
fn edge_widths() -> Vec<u64> {
    let mut widths: BTreeSet<u64> = [
        1, 2, 3, 5, 7, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257, 511, 512, 513,
        767, 768, 769, 1000, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097,
    ]
    .into_iter()
    .collect();
    for which in ALL {
        for rows in [2u64, 3, 5, 7, 9, 15, 16, 17, 33, 63, 64] {
            let limit = 4096 / which.staged() / rows;
            widths.extend([limit - 1, limit, limit + 1]);
        }
    }
    widths.into_iter().collect()
}

// ---------------------------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------------------------

/// `layer_norm_rows` is the most rows of the staged bindings that fit 16 KiB, each row padded
/// to the odd stride `layer_norm_row_stride`, at most 64 and for the moments at most
/// `LAYER_NORM_MOMENTS_MAX_ROWS` (16), for every width to 8192; 0 exactly
/// where fewer than `LAYER_NORM_MIN_ROWS` (3) fit, where a workgroup with fewer rows is slower
/// than the per-row kernel (ADV-LNROWS-1). The widest staged widths: 1365 for `LayerNorm` and
/// the moments, 681 for `LayerNormGradX`.
#[test]
fn adv_lnrows_rows_per_workgroup_fill_16_kib_at_every_width() {
    let bytes = u64::from(MIN_SHARED_MEMORY_BYTES);
    let mut bad = Vec::new();
    for which in ALL {
        for h in 1..=8192u64 {
            let rows = which.rows(h);
            let stride = layer_norm_row_stride(h);
            if stride.is_multiple_of(2) || stride < h || stride > h + 1 {
                bad.push(format!("width {h}: stride {stride}"));
            }
            let each = 4 * stride * which.staged();
            let most = if which == Which::Moments {
                LAYER_NORM_MOMENTS_MAX_ROWS
            } else {
                64
            };
            let ok = if rows == 0 {
                LAYER_NORM_MIN_ROWS * each > bytes
            } else {
                (LAYER_NORM_MIN_ROWS..=most).contains(&rows)
                    && rows * each <= bytes
                    && (rows == most || (rows + 1) * each > bytes)
            };
            if !ok {
                bad.push(format!("{} width {h}: {rows} rows", which.name()));
            }
        }
    }
    let limits = [
        (Which::LayerNorm, 1365, 3),
        (Which::LayerNorm, 1366, 0),
        (Which::LayerNorm, 2047, 0),
        (Which::LayerNorm, 4096, 0),
        (Which::LayerNorm, 1024, 3),
        (Which::LayerNorm, 1023, 4),
        (Which::Moments, 1365, 3),
        (Which::Moments, 1366, 0),
        (Which::Moments, 2047, 0),
        (Which::Moments, 4096, 0),
        (Which::LayerNormGradX, 681, 3),
        (Which::LayerNormGradX, 682, 0),
        (Which::LayerNormGradX, 1023, 0),
        (Which::LayerNormGradX, 2048, 0),
        (Which::LayerNorm, 256, 15),
        (Which::LayerNormGradX, 256, 7),
        (Which::Moments, 128, 16),
        (Which::Moments, 255, 16),
        (Which::Moments, 256, 15),
        (Which::Moments, 1, 16),
        (Which::LayerNorm, 63, 64),
        (Which::LayerNorm, 64, 63),
        (Which::LayerNorm, 65, 63),
        (Which::LayerNormGradX, 31, 64),
        (Which::LayerNormGradX, 32, 62),
        (Which::LayerNormGradX, 33, 62),
    ];
    for (which, h, want) in limits {
        if which.rows(h) != want {
            bad.push(format!(
                "{} width {h}: {} rows, not {want}",
                which.name(),
                which.rows(h)
            ));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

/// The moments stage at most `LAYER_NORM_MOMENTS_MAX_ROWS` (16) rows per workgroup, exactly 16
/// at every width where 16 fit, and their plan dispatches one workgroup per 16 rows there, while
/// `LayerNorm` at the same widths keeps every row that fits; inside
/// `layer_norm_per_row::with` no kernel stages rows and each runs one invocation per row, and
/// the switch is off again after it, also after a panic.
#[test]
fn adv_lnrows_moments_stage_at_most_16_rows_and_the_per_row_switch_stages_none() {
    let mut bad = Vec::new();
    let n = 16384u64;
    for h in 1..=8192u64 {
        let rows = Which::Moments.rows(h);
        let fit = 4096 / layer_norm_row_stride(h);
        if rows > LAYER_NORM_MOMENTS_MAX_ROWS {
            bad.push(format!("moments width {h}: {rows} rows"));
        }
        if fit >= 16 {
            if rows != 16 {
                bad.push(format!("moments width {h}: {rows} rows, not 16"));
            }
            let kernel = Which::Moments.kernel(n, h);
            let groups: u64 = kernel.workgroups.iter().map(|&w| u64::from(w)).product();
            if groups != n.div_ceil(16) {
                bad.push(format!("moments [{n}, {h}]: {groups} workgroups"));
            }
            if Which::LayerNorm.rows(h) != fit.min(64) {
                bad.push(format!(
                    "LayerNorm width {h}: {} rows",
                    Which::LayerNorm.rows(h)
                ));
            }
        }
    }
    for h in [1u64, 64, 127, 128, 256, 681] {
        for which in ALL {
            let (rows, groups) = layer_norm_per_row::with(|| {
                let kernel = which.kernel(n, h);
                let groups: u64 = kernel.workgroups.iter().map(|&w| u64::from(w)).product();
                (which.rows(h), groups)
            });
            if rows != 0 || groups != n.div_ceil(64) {
                bad.push(format!(
                    "{} [{n}, {h}] per row: {rows} rows, {groups} workgroups",
                    which.name()
                ));
            }
            if which.rows(h) == 0 {
                bad.push(format!("{} width {h}: the switch stayed on", which.name()));
            }
        }
    }
    let panicked = std::panic::catch_unwind(|| layer_norm_per_row::with(|| panic!("inside")));
    if panicked.is_ok() || layer_norm_per_row::forced() {
        bad.push("the switch stayed on after a panic".to_string());
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

/// The module of each kernel does not grow with the width (CS-1): over every edge width that
/// stages rows, the largest module is within 64 words of the smallest.
#[test]
fn adv_lnrows_module_size_is_flat_over_every_staged_width() {
    let mut bad = Vec::new();
    for which in ALL {
        let sizes: Vec<(u64, usize)> = edge_widths()
            .into_iter()
            .filter(|&h| which.rows(h) > 0)
            .map(|h| (h, emit(&which.kernel(65, h)).expect("emits").len()))
            .collect();
        let min = sizes.iter().map(|s| s.1).min().unwrap();
        let max = sizes.iter().map(|s| s.1).max().unwrap();
        eprintln!("{}: {min}..{max} words", which.name());
        if max > min + 64 {
            bad.push(format!("{}: {min}..{max} words: {sizes:?}", which.name()));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

// ---------------------------------------------------------------------------------------------
// Device: bits
// ---------------------------------------------------------------------------------------------

/// Every row count from 1 to 70 at widths 1, 2, 3, 63, 64, 65, 256, 257, 511, 512, the staging
/// limits 681 (`LayerNormGradX`) and 1365 (`LayerNorm`, the moments) with the column under and
/// over each, and 1024, 2048 and 4096: the CPU's bits, guard words and inputs unchanged.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_lnrows_row_counts_1_to_70_at_edge_widths_are_the_cpu_bits() {
    with_device(|device| {
        let mut draw = Draw::new(0xad75_0001);
        let mut bad = Vec::new();
        let mut cases = 0;
        for h in [
            1u64, 2, 3, 63, 64, 65, 256, 257, 511, 512, 680, 681, 682, 1024, 1364, 1365, 1366,
            2048, 4096,
        ] {
            for n in 1..=70u64 {
                for which in ALL {
                    let inputs = drawn(which, n, h, &mut draw);
                    bad.extend(check(device, which, n, h, &inputs));
                    cases += 1;
                }
            }
        }
        eprintln!("{cases} kernels, {} problems", bad.len());
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// At every edge width, row counts one under and one over each multiple of the rows per
/// workgroup and a single row: the last workgroup partial in every way.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_lnrows_partial_last_workgroups_at_every_edge_width_are_the_cpu_bits() {
    with_device(|device| {
        let mut draw = Draw::new(0xad75_0002);
        let mut bad = Vec::new();
        let mut cases = 0;
        for h in edge_widths() {
            for which in ALL {
                let r = which.rows(h).max(1);
                let counts: BTreeSet<u64> = [1, r - 1, r, r + 1, 2 * r - 1, 2 * r + 1, 3 * r + 2]
                    .into_iter()
                    .filter(|&n| n > 0 && n * h <= 1 << 21)
                    .collect();
                for n in counts {
                    let inputs = drawn(which, n, h, &mut draw);
                    bad.extend(check(device, which, n, h, &inputs));
                    cases += 1;
                }
            }
        }
        eprintln!("{cases} kernels, {} problems", bad.len());
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// Rows of special values: all `+0`, all `-0`, constant rows (the `rstd` of a zero variance),
/// a NaN, the infinities, sums and squares that overflow, tiny normal values, `gamma` and
/// `beta` holding `-0`, and a `g` holding the infinities. The CPU's bits, or NaN where it is
/// NaN.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_lnrows_special_value_rows_are_the_cpu_bits() {
    let (inf, nan, max) = (f32::INFINITY, f32::NAN, f32::MAX);
    let special = |k: usize, h: usize, draw: &mut Draw| -> Vec<f32> {
        let base = draw.values(h as u64);
        (0..h)
            .map(|j| match k {
                0 => 0.0,
                1 => -0.0,
                2 => 3.0,
                3 => 0.1,
                4 => {
                    if j + 1 == h {
                        nan
                    } else {
                        base[j]
                    }
                }
                5 => {
                    if j == 0 {
                        inf
                    } else {
                        base[j]
                    }
                }
                6 => [inf, -inf][j % 2],
                7 => max,
                8 => [max, -max][j % 2],
                9 => base[j] * 1e-15,
                10 => base[j] * 1e19,
                11 => {
                    if j == h / 2 {
                        0.0
                    } else {
                        -0.0
                    }
                }
                12 => -1e-3,
                _ => base[j],
            })
            .collect()
    };
    with_device(|device| {
        let mut draw = Draw::new(0xad75_0003);
        let mut bad = Vec::new();
        for h in [1u64, 2, 3, 64, 65, 257, 2048, 2049, 4096, 4097] {
            for which in ALL {
                let r = which.rows(h).max(1);
                let n = (r + 1).max(16);
                let x: Vec<f32> = (0..n as usize)
                    .flat_map(|k| special(k % 16, h as usize, &mut draw))
                    .collect();
                let mut gamma = draw.values(h);
                gamma[0] = -0.0;
                let mut beta = draw.values(h);
                beta[(h as usize) - 1] = -0.0;
                let mut g = draw.values(n * h);
                g[h as usize] = inf;
                g[(n * h) as usize - 1] = -inf;
                let inputs = match which {
                    Which::LayerNorm => vec![x, gamma, beta],
                    Which::LayerNormGradX => vec![x, gamma, g],
                    Which::Moments => vec![x],
                };
                bad.extend(check(device, which, n, h, &inputs));
                if which == Which::LayerNorm {
                    // beta all -0 and gamma all +1 over all -0 rows: -0 + -0 stays -0.
                    let x = vec![-0.0; (n * h) as usize];
                    let inputs = vec![x, vec![1.0; h as usize], vec![-0.0; h as usize]];
                    bad.extend(check(device, which, n, h, &inputs));
                }
            }
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// Row counts whose workgroups need the second dispatch dimension (more than 65535 groups):
/// `[4194369, 1]` and `[4194369, 3]`, 64 rows per workgroup, 65537 groups, and for the moments
/// 16 rows (`LAYER_NORM_MOMENTS_MAX_ROWS`), 262149 groups.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_lnrows_dispatches_over_two_dimensions_are_the_cpu_bits() {
    with_device(|device| {
        let mut draw = Draw::new(0xad75_0004);
        let mut bad = Vec::new();
        let n = 65_535 * 64 + 129;
        for h in [1u64, 3] {
            for which in ALL {
                let most = if which == Which::Moments {
                    LAYER_NORM_MOMENTS_MAX_ROWS
                } else {
                    64
                };
                assert_eq!(which.rows(h), most);
                let kernel = which.kernel(n, h);
                assert!(kernel.workgroups[1] > 1, "{:?}", kernel.workgroups);
                eprintln!(
                    "{} [{n}, {h}]: workgroups {:?}",
                    which.name(),
                    kernel.workgroups
                );
                let inputs = drawn(which, n, h, &mut draw);
                bad.extend(check(device, which, n, h, &inputs));
            }
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

// ---------------------------------------------------------------------------------------------
// Device: timing (GPU to itself)
// ---------------------------------------------------------------------------------------------

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// The least time, in microseconds, of the command buffer [`dispatch_us`] times: the RTX 3090
/// drops to P3 or P5 while the host draws and uploads operands, and a buffer of 150 dispatches
/// of 20 to 150 µs can run below P0 throughout (2026-10-29: the same
/// `LayerNormGradX [16384, 256]` took 156.7 µs in one run and 653.3 in the next, while the
/// per-row kernel timed after it did not move).
const TIMED_BUFFER_US: f64 = 400_000.0;

/// Median microseconds of one dispatch of `which` over `[n, h]`: a first buffer of 20 dispatches
/// estimates it, then one buffer of at least 150 dispatches and at least [`TIMED_BUFFER_US`],
/// whose first half brings the clocks up and whose second half is timed.
fn dispatch_us(device: &Device<'_>, which: Which, n: u64, h: u64, draw: &mut Draw) -> f64 {
    let inputs = drawn(which, n, h, draw);
    let kernel = which.kernel(n, h);
    let len = which.result_len(n, h);
    let estimate = median(run(device, &kernel, &inputs, len, 20).2[5..].to_vec()).max(1.0);
    let count = ((TIMED_BUFFER_US / estimate).ceil() as u32).clamp(150, 6000);
    let times = run(device, &kernel, &inputs, len, count).2;
    median(times[times.len() / 2..].to_vec())
}

/// The widest width at which `which` stages rows: past it, fewer than `LAYER_NORM_MIN_ROWS`
/// rows fit and the per-row kernel runs.
fn staging_limit(which: Which) -> u64 {
    (1..=8192u64)
        .rev()
        .find(|&h| which.rows(h) > 0)
        .expect("some width stages rows")
}

/// The staged kernel at the staging limit, one column under it (an even width, whose rows are
/// padded to an odd stride) and at half of it, against the per-row kernel one column past the limit, over `n`
/// rows: the problems where its time per column is over 1.25 times the per-row one.
fn cliff(device: &Device<'_>, n: u64, draw: &mut Draw) -> Vec<String> {
    let mut bad = Vec::new();
    let mut pairs = Vec::new();
    for which in ALL {
        let limit = staging_limit(which);
        for staged in [limit, limit - 1, limit / 2] {
            pairs.push((which, staged, limit + 1));
        }
    }
    for (which, staged, fallback) in pairs {
        assert!(which.rows(staged) > 0 && which.rows(fallback) == 0);
        let a = dispatch_us(device, which, n, staged, draw);
        let b = dispatch_us(device, which, n, fallback, draw);
        let (pa, pb) = (a / staged as f64, b / fallback as f64);
        eprintln!(
            "{} [{n}, {staged}] staged {} rows/group: {a:.1} µs ({:.4} µs/col); \
             [{n}, {fallback}] per-row: {b:.1} µs ({:.4} µs/col)",
            which.name(),
            which.rows(staged),
            pa,
            pb
        );
        if pa > 1.25 * pb {
            bad.push(format!(
                "{} [{n}, {staged}] staged: {a:.1} µs, {:.2}x the per-column time of the \
                 per-row kernel at [{n}, {fallback}] ({b:.1} µs)",
                which.name(),
                pa / pb
            ));
        }
    }
    bad
}

/// [`cliff`] over 16384 rows, enough workgroups to fill the device at every width.
#[test]
#[ignore = "device timing: the GPU to itself"]
fn adv_lnrows_timing_staged_kernel_of_16384_rows_at_the_fit_limit_is_not_slower_than_the_fallback()
{
    with_device(|device| {
        let mut draw = Draw::new(0xad75_0006);
        let bad = cliff(device, 16384, &mut draw);
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// At the fit limit of 16 KiB a workgroup would stage one row and 63 of its 64 invocations
/// would idle while the owner adds it (ADV-LNROWS-1), so staging stops where fewer than
/// `LAYER_NORM_MIN_ROWS` rows fit; one column past that limit the previous kernel runs one
/// invocation per row. The staged kernel at the limit takes at most 1.25 times the time per element
/// of the previous kernel one column past it, and so one column under the limit and at half
/// of it.
#[test]
#[ignore = "device timing: the GPU to itself"]
fn adv_lnrows_timing_staged_kernel_at_the_fit_limit_is_not_slower_than_the_fallback() {
    with_device(|device| {
        let mut draw = Draw::new(0xad75_0005);
        let n = 4096;
        let bad = cliff(device, n, &mut draw);
        for h in [128u64, 256, 512, 768, 1024] {
            for which in ALL {
                let us = dispatch_us(device, which, n, h, &mut draw);
                eprintln!(
                    "{} [{n}, {h}] ({} rows/group): {us:.1} µs, {:.4} µs/col",
                    which.name(),
                    which.rows(h),
                    us / h as f64
                );
            }
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// Over 16384 rows at every width from 120 to 130, where the moments would stage 31 to 33 rows
/// without `LAYER_NORM_MOMENTS_MAX_ROWS` and took up to 1.33 times the per-row kernel, each
/// staged kernel takes at most 1.25 times the per-row kernel of the same width, built inside
/// `layer_norm_per_row::with`.
#[test]
#[ignore = "device timing: the GPU to itself"]
fn adv_lnrows_timing_staged_kernels_of_widths_120_to_130_are_not_slower_than_per_row() {
    with_device(|device| {
        let mut draw = Draw::new(0xad75_0007);
        let n = 16384;
        let mut bad = Vec::new();
        for h in 120..=130u64 {
            for which in ALL {
                let a = dispatch_us(device, which, n, h, &mut draw);
                let b = layer_norm_per_row::with(|| dispatch_us(device, which, n, h, &mut draw));
                eprintln!(
                    "{} [{n}, {h}] staged {} rows/group: {a:.1} µs; per-row: {b:.1} µs; {:.2}x",
                    which.name(),
                    which.rows(h),
                    a / b
                );
                if a > 1.25 * b {
                    bad.push(format!(
                        "{} [{n}, {h}] staged: {a:.1} µs, {:.2}x the per-row kernel ({b:.1} µs)",
                        which.name(),
                        a / b
                    ));
                }
            }
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// Every edge width's pipeline is created within 2 s (CS-1 on the device).
#[test]
#[ignore = "device timing: the GPU to itself"]
fn adv_lnrows_timing_pipelines_of_every_edge_width_are_created_within_2_s() {
    with_device(|device| {
        let mut bad = Vec::new();
        let mut worst = 0.0f64;
        for h in edge_widths() {
            for which in ALL {
                let kernel = which.kernel(4097, h);
                let words = emit(&kernel).expect("emits");
                let started = Instant::now();
                let created = device.kernel_pipeline(&kernel, &words);
                let seconds = started.elapsed().as_secs_f64();
                worst = worst.max(seconds);
                if created.is_err() || seconds > 2.0 {
                    bad.push(format!(
                        "{} [4097, {h}]: {seconds:.2} s, {}",
                        which.name(),
                        created.is_ok()
                    ));
                }
            }
        }
        eprintln!("slowest pipeline {worst:.3} s");
        assert!(bad.is_empty(), "{bad:#?}");
    });
}
