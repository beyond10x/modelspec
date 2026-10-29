//! Adversary pass on story:vulkan-layer-norm-grad-gamma-row-moments: `LayerNormGradGamma` on
//! Vulkan is a `per_row` moments kernel into an allocated `f32[n, 2]`, a barrier, and a
//! `per_column` column sum that reads the moments; the result is the CPU interpreter's bits,
//! neither module grows with the row width, and the moments buffers are numbered after every
//! other buffer and before the `EmbeddingGrad` sorted ids.
//!
//! The host tests run by default. The device tests are ignored; run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test adversary_lngg_moments -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::{Draw, GUARD_WORDS, guard, guarded};
use modelspec_core::artifact::Backend;
use modelspec_core::compile::{compile_artifact_with, compile_training_with};
use modelspec_core::ir::kernel::{Kernel, KernelKind, Operand, PlanOp};
use modelspec_core::ir::tensor::{TensorOp, TensorProgramBuilder, TensorType};
use modelspec_core::schedule::layer_norm_grad_gamma_kernels;
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

/// The masked-row gather of BERT-tiny's packed training data.
const MASKED_ROWS: u64 = 832;

/// The bits each output element holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_dead;

/// The two kernels of `LayerNormGradGamma` over `[n, h]`, as the schedule builds them.
fn kernels(n: u64, h: u64) -> [Kernel; 2] {
    let operands = vec![
        Operand::Buffer(TensorType::f32(&[n, h])),
        Operand::Buffer(TensorType::f32(&[n, h])),
    ];
    layer_norm_grad_gamma_kernels(operands, TensorType::f32(&[h]))
        .unwrap_or_else(|| panic!("LayerNormGradGamma [{n}, {h}] has its two kernels"))
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

fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

fn with_device(body: impl FnOnce(&Device<'_>)) {
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
    body(&device);
    let messages = vulkan.take_validation_messages();
    assert!(messages.is_empty(), "{messages:#?}");
}

/// One run of both kernels over `x` and `g` (`[n, h]`) as the step plan runs them, in one
/// command buffer: the moments into a guarded `[n, 2]` buffer, a barrier, the column sum into
/// a guarded `[h]`. Returns the moments and the gradient, each with its guard words.
fn run(device: &Device<'_>, n: u64, h: u64, x: &[f32], g: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let [moments_kernel, column_kernel] = kernels(n, h);
    let pipeline = |kernel: &Kernel| {
        let words = emit(kernel).expect("the kernel emits");
        device
            .kernel_pipeline(kernel, &words)
            .expect("the pipeline is created")
    };
    let (moments_pipeline, column_pipeline) = (pipeline(&moments_kernel), pipeline(&column_kernel));
    let uploads = [
        x.to_vec(),
        g.to_vec(),
        guarded(&vec![f32::from_bits(UNWRITTEN); 2 * n as usize]),
        guarded(&vec![f32::from_bits(UNWRITTEN); h as usize]),
    ];
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
    let mut recording = device.record().expect("a recording starts");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("the upload is recorded");
    }
    recording.barrier(Barrier::TransferToCompute);
    recording
        .dispatch_kernel(
            &moments_pipeline,
            &moments_kernel,
            &[&buffers[0], &buffers[2]],
        )
        .expect("the moments dispatch is recorded");
    recording.barrier(Barrier::ComputeToCompute);
    recording
        .dispatch_kernel(
            &column_pipeline,
            &column_kernel,
            &[&buffers[2], &buffers[0], &buffers[1], &buffers[3]],
        )
        .expect("the column dispatch is recorded");
    recording.barrier(Barrier::ComputeToTransfer);
    for k in [2, 3] {
        recording
            .copy(&buffers[k], &staging[k])
            .expect("the download is recorded");
    }
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    device
        .submit(recorded)
        .expect("submitted")
        .wait()
        .expect("the device finishes");
    let moments = staging[2].read_f32().expect("the moments read");
    let gradient = staging[3].read_f32().expect("the gradient reads");
    (moments, gradient)
}

/// The guard words after the first `count` values of `got` that changed.
fn guards_changed(label: &str, got: &[f32], count: usize) -> Vec<String> {
    (0..GUARD_WORDS)
        .filter(|&i| got[count + i].to_bits() != guard(i))
        .map(|i| format!("{label}: guard word {i} changed"))
        .collect()
}

/// The gradient columns whose bits differ from the CPU's: at most four named, then the count.
fn columns_differ(label: &str, got: &[f32], want: &[f32]) -> Vec<String> {
    let differ: Vec<usize> = (0..want.len())
        .filter(|&j| got[j].to_bits() != want[j].to_bits())
        .collect();
    let mut bad: Vec<String> = differ
        .iter()
        .take(4)
        .map(|&j| {
            format!(
                "{label} column {j}: gpu {:?} ({:#010x}) cpu {:?} ({:#010x})",
                got[j],
                got[j].to_bits(),
                want[j],
                want[j].to_bits()
            )
        })
        .collect();
    if !differ.is_empty() {
        bad.push(format!(
            "{label}: {} of {} columns differ from the CPU",
            differ.len(),
            want.len()
        ));
    }
    bad
}

/// SplitMix64 values with a full 24-bit significand, a random sign and a binary exponent in
/// `[-20, 6]`: rows whose mean and variance are not exact, with variances from about `1e-13`
/// (below `eps = 1e-12`) to about `1e4`, and no subnormal operand, square or result.
struct Wide {
    state: u64,
}

impl Wide {
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn values(&mut self, count: u64) -> Vec<f32> {
        (0..count)
            .map(|_| {
                let r = self.next();
                let significand = (r & 0x7f_ffff) as u32;
                let exponent = (127 - 20 + (r >> 23) % 27) as u32;
                let sign = ((r >> 40) & 1) as u32;
                f32::from_bits(sign << 31 | exponent << 23 | significand)
            })
            .collect()
    }
}

/// Rows past `65535 * 64`: the moments kernel spreads its 65,537 workgroups over two
/// dimensions, and four million rows of three full-significand values with variances from
/// below `eps` up reach the corrected quotient and reciprocal square root on every kind of
/// operand. Every row's mean and `rstd` in the moments buffer is the CPU's `row_moments` bits
/// (the column kernel reads them as the CPU's), the guard words after the `[n, 2]` and after
/// the gradient stay, and every column of the gradient is the CPU interpreter's bits.
#[test]
#[ignore = "device: four million rows of moments against the CPU bits"]
fn adv_lngg_moments_of_four_million_full_significand_rows_are_the_cpu_bits() {
    let (n, h) = (65_535 * 64 + 65, 3u64);
    let mut wide = Wide {
        state: 0x6164_765f_6c6e_6767,
    };
    let (x, g) = (wide.values(n * h), wide.values(n * h));
    let want = cpu(n, h, &x, &g);
    with_device(|device| {
        let (moments, gradient) = run(device, n, h, &x, &g);
        let label = format!("LayerNormGradGamma [{n}, {h}]");
        let mut bad = Vec::new();
        let mut differ = 0usize;
        let mut means = 0usize;
        for (r, row) in x.chunks_exact(h as usize).enumerate() {
            let (mean, rstd) = cpu_moments(row);
            let (gm, gr) = (moments[2 * r], moments[2 * r + 1]);
            if gm.to_bits() != mean.to_bits() || gr.to_bits() != rstd.to_bits() {
                differ += 1;
                if gm.to_bits() != mean.to_bits() {
                    means += 1;
                }
                if differ <= 4 {
                    bad.push(format!(
                        "{label} row {r} {row:?}: gpu (mean {gm:e}, rstd {gr:e} {:#010x}) cpu \
                         (mean {mean:e}, rstd {rstd:e} {:#010x})",
                        gr.to_bits(),
                        rstd.to_bits()
                    ));
                }
            }
        }
        if differ > 0 {
            bad.push(format!(
                "{label}: {differ} of {n} rows' moments differ from the CPU's ({means} means)"
            ));
        }
        bad.extend(guards_changed(
            &format!("{label} moments"),
            &moments,
            2 * n as usize,
        ));
        bad.extend(columns_differ(&label, &gradient, &want));
        bad.extend(guards_changed(&label, &gradient, h as usize));
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// Rows of 8192 columns and more: the column kernel now tiles `LayerNormGradGamma` as
/// `SumRows` (64 columns per workgroup, 16 rows per tile, 16 values per invocation, four
/// registers each), a tile no `LayerNormGradGamma` had before this unit and no width of the
/// unit's or the column-sum adversary's grids reaches (their widest is 4097, 32 columns). Rows
/// on both sides of the 16-row tile: the CPU's bits, guard words kept, three runs equal.
#[test]
#[ignore = "device: 64-column LayerNormGradGamma tiles against the CPU bits"]
fn adv_lngg_sixty_four_column_tiles_are_the_cpu_bits_in_three_runs() {
    let shapes: [(u64, u64); 7] = [
        (1, 8192),
        (15, 8192),
        (16, 8192),
        (17, 8192),
        (33, 8193),
        (17, 12_289),
        (3, 1 << 16),
    ];
    let mut draw = Draw::new(0x6164_765f_6c6e_3634);
    let cases: Vec<(u64, u64, Vec<f32>, Vec<f32>)> = shapes
        .iter()
        .map(|&(n, h)| (n, h, draw.values(n * h), draw.values(n * h)))
        .collect();
    with_device(|device| {
        let mut bad = Vec::new();
        for (n, h, x, g) in &cases {
            let (n, h) = (*n, *h);
            let label = format!("LayerNormGradGamma [{n}, {h}]");
            let want = cpu(n, h, x, g);
            let (moments, first) = run(device, n, h, x, g);
            bad.extend(guards_changed(
                &format!("{label} moments"),
                &moments,
                2 * n as usize,
            ));
            bad.extend(columns_differ(&label, &first, &want));
            bad.extend(guards_changed(&label, &first, h as usize));
            for k in 1..3 {
                let (_, again) = run(device, n, h, x, g);
                if again
                    .iter()
                    .map(|v| v.to_bits())
                    .ne(first.iter().map(|v| v.to_bits()))
                {
                    bad.push(format!("{label}: run {k} differs from run 0"));
                }
            }
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// "Module size no longer grows with the row width": over up to 4097 rows (fewer where the rows
/// would pass 2^31 elements) of widths from 1 to 2^20 both kernels emit, no moments module is
/// larger than at width 128, and no column module is more than twice the smallest.
#[test]
fn adv_lngg_module_sizes_do_not_depend_on_the_width_from_1_to_2_pow_20() {
    let widths: [u64; 12] = [1, 2, 3, 5, 127, 128, 129, 4096, 4097, 8192, 65_536, 1 << 20];
    let mut sizes = Vec::new();
    for h in widths {
        let n = 4097.min((1u64 << 31) / h);
        let [moments, column] = kernels(n, h);
        let m = emit(&moments).expect("the moments kernel emits").len();
        let c = emit(&column).expect("the column kernel emits").len();
        eprintln!("[{n}, {h}]: moments {m} words, column {c} words");
        sizes.push((h, m, c));
    }
    let at_128 = sizes.iter().find(|s| s.0 == 128).map_or(0, |s| s.1);
    let moments: Vec<_> = sizes.iter().filter(|s| s.1 > at_128).collect();
    let least = sizes.iter().map(|&(_, _, c)| c).min().expect("a width");
    let grown: Vec<_> = sizes.iter().filter(|&&(_, _, c)| c > 2 * least).collect();
    assert!(
        moments.is_empty() && grown.is_empty(),
        "moments modules over the {at_128} words at width 128: {moments:?}; column modules over twice the smallest ({least}): \
         {grown:?}"
    );
}

/// The step plan of BERT-tiny: each `LayerNormGradGamma` has its own moments buffer of
/// `8 n` bytes, numbered after every buffer that is neither moments nor sorted ids and before
/// every `EmbeddingGrad` sorted-ids buffer; only its `Allocate` and its moments dispatch write
/// it, only its column dispatch reads it, and the forward plan never touches it.
#[test]
fn adv_lngg_bert_moments_buffers_are_numbered_between_and_never_shared() {
    let training = compile_training_with("bert_tiny.mspec", BERT_TINY, Some(MASKED_ROWS))
        .unwrap_or_else(|d| panic!("bert_tiny compiles: {d:#?}"));
    let sites = training
        .training
        .training
        .program
        .nodes
        .iter()
        .filter(|node| node.op == TensorOp::LayerNormGradGamma)
        .count();
    let artifact = compile_artifact_with(
        "bert_tiny.mspec",
        BERT_TINY,
        Backend::Vulkan,
        None,
        Some(MASKED_ROWS),
    )
    .unwrap_or_else(|d| panic!("bert_tiny compiles for Vulkan: {d:#?}"));
    let plan = &artifact.execution;
    let kernel_of = |at: usize| {
        plan.step[at]
            .kernel_module_id
            .and_then(|id| plan.kernel(id))
            .filter(|_| plan.step[at].op == PlanOp::Dispatch)
    };
    let mut moments: BTreeMap<u32, (usize, u64)> = BTreeMap::new();
    let mut sorted: BTreeSet<u32> = BTreeSet::new();
    for at in 0..plan.step.len() {
        let Some(kernel) = kernel_of(at) else {
            continue;
        };
        if kernel.kind != KernelKind::PerRow {
            continue;
        }
        let written: Vec<u32> = plan.step[at].writes.iter().map(|v| v.buffer).collect();
        match kernel.op {
            "LayerNormGradGamma" => {
                let rows = kernel.result.shape.first().map_or(0, |d| match d {
                    modelspec_core::ir::tensor::Dim::Static(n) => *n,
                    _ => 0,
                });
                for b in written {
                    assert!(
                        moments.insert(b, (at, rows)).is_none(),
                        "buffer {b} holds the moments of two sites"
                    );
                }
            }
            "EmbeddingGrad" => sorted.extend(written),
            _ => {}
        }
    }
    let mut bad = Vec::new();
    if moments.len() != sites {
        bad.push(format!(
            "{} moments buffers for {sites} sites",
            moments.len()
        ));
    }
    let others: Vec<u32> = (0..plan.buffers.len() as u32)
        .filter(|b| !moments.contains_key(b) && !sorted.contains(b))
        .collect();
    let (first_m, last_m) = (moments.keys().min(), moments.keys().max());
    if let (Some(&last_other), Some(&first_m)) = (others.iter().max(), first_m)
        && last_other > first_m
    {
        bad.push(format!(
            "buffer {last_other} is numbered after moments buffer {first_m}"
        ));
    }
    if let (Some(&last_m), Some(&first_s)) = (last_m, sorted.iter().min())
        && last_m > first_s
    {
        bad.push(format!(
            "moments buffer {last_m} is numbered after sorted ids {first_s}"
        ));
    }
    for (&b, &(at, rows)) in &moments {
        if plan.buffers[b as usize] != 8 * rows {
            bad.push(format!(
                "moments buffer {b} is {} bytes for {rows} rows",
                plan.buffers[b as usize]
            ));
        }
        for (s, step) in plan.step.iter().enumerate() {
            let writes = step.writes.iter().any(|v| v.buffer == b);
            let reads = step.reads.iter().any(|v| v.buffer == b);
            let allowed_write = s == at || (step.op == PlanOp::Allocate && s + 1 == at);
            if writes && !allowed_write {
                bad.push(format!(
                    "step {s} ({:?}) writes moments buffer {b}",
                    step.op
                ));
            }
            if reads {
                let column = kernel_of(s).is_some_and(|k| {
                    k.op == "LayerNormGradGamma" && k.kind == KernelKind::PerColumn
                }) && step.value == plan.step[at].value;
                if !column {
                    bad.push(format!("step {s} ({:?}) reads moments buffer {b}", step.op));
                }
            }
        }
        if plan
            .forward
            .iter()
            .any(|step| step.reads.iter().chain(&step.writes).any(|v| v.buffer == b))
        {
            bad.push(format!("the forward plan touches moments buffer {b}"));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

/// Row 1,792,871 of the four-million-row case above, `x = [0.00241724, -0.5908017,
/// 0.0007464285]`, alone, with `g = 1`: the gradient of a one-row `[1, 3]` layer norm is that
/// row's `xhat`, so it is the CPU's bits only if the device's `rstd` is. The CPU rounds
/// `sqrt(variance + eps)` and then its reciprocal; the moments kernel's corrected reciprocal
/// square root (`Shell::reciprocal_sqrt`) gives `0x40652eca` where the CPU gives `0x40652ecb`.
#[test]
#[ignore = "device: one row whose rstd the device rounds off the CPU"]
fn adv_lngg_one_row_whose_rstd_is_off_by_one_ulp_gives_a_gradient_off_the_cpu_bits() {
    let (rows, h, at) = (65_535 * 64 + 65u64, 3u64, 1_792_871usize);
    let mut wide = Wide {
        state: 0x6164_765f_6c6e_6767,
    };
    let x = wide.values(rows * h)[3 * at..3 * at + 3].to_vec();
    let bits: Vec<String> = x.iter().map(|v| format!("{:#010x}", v.to_bits())).collect();
    eprintln!("x = {x:?} = {bits:?}");
    let g = vec![1.0f32; 3];
    let want = cpu(1, h, &x, &g);
    with_device(|device| {
        let (moments, gradient) = run(device, 1, h, &x, &g);
        let (mean, rstd) = cpu_moments(&x);
        let label = format!("LayerNormGradGamma [1, 3] of x {bits:?}");
        let mut bad = columns_differ(&label, &gradient, &want);
        if moments[1].to_bits() != rstd.to_bits() || moments[0].to_bits() != mean.to_bits() {
            bad.push(format!(
                "{label}: moments gpu ({:#010x}, {:#010x}) cpu ({:#010x}, {:#010x})",
                moments[0].to_bits(),
                moments[1].to_bits(),
                mean.to_bits(),
                rstd.to_bits()
            ));
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}
