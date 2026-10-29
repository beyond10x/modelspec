//! `story:m1-loss-sum-staged` acceptance 1: the `mse` loss sum runs one workgroup of 64 that
//! stages its terms in workgroup memory, chunk after chunk, and invocation 0 adds them in
//! row-major order from the first. On a Vulkan device, at 1, 63, 64, 65, 1,000, 12,289 and
//! 100,000 terms, with the workgroup memory the device reports and with copies of its profile
//! that report 256 and 260 bytes (chunks of 64 and 65 terms), each module
//!
//! - gives the CPU's bits for the sum before its division when `negotiate` gives `bit-for-bit`,
//!   and is within the loss-sum and `mse` tolerances of `tests/common/mod.rs` on every device;
//! - gives the same bits on a second run and writes no guard word;
//! - on terms `2^24, 1, 1, …` gives exactly `2^24`, the row-major sum: under round-to-nearest-even
//!   and round-toward-zero alike each `2^24 + 1` rounds back to `2^24`, and an order that adds two
//!   of the ones together before the `2^24` ends above it;
//! - on terms `1, 1, …, 2^24`, with one `0` before the `2^24` when that makes the ones even,
//!   gives exactly `2^24` plus the ones, which is representable, while an order that adds the
//!   `2^24` before two of the ones loses them. Every partial sum of both inputs is an integer
//!   and either exact or a rounding both modes resolve the same way, so these two checks are exact
//!   on every device, inside the bit-for-bit class or not.
//!
//! `story:vulkan-loss-sum-overflow-parity`: at 2, 8, 65 and 1,000 terms whose `f32` sum passes
//! `f32::MAX`, the undivided module writes `inf` and the plan's module writes the mean the CPU
//! interpreter writes, within the `mse` tolerance where that mean is finite and `inf` where it is
//! not (DESIGN.md § 5.4, revision 0.75).
//!
//! The device test is ignored by default: it creates a Vulkan instance, which `task ci` never
//! does. Run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_loss_sum -- --ignored`;
//! with `MODELSPEC_VK_VALIDATION=1` every message of the Khronos validation layer fails it.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use common::{Draw, GUARD_WORDS, KernelOp, guard, guarded};
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::Kernel;
use modelspec_runtime::spirv::emit::{KernelModule, emit_loss_sum_undivided, emit_module};
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::{NumericsClass, negotiate};
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

/// The term counts of the acceptance, and one past a full chunk of the RTX 3090's 48 KiB.
const COUNTS: [u64; 7] = [1, 63, 64, 65, 1_000, 12_289, 100_000];

/// The bits the output holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_dead;

/// `(a - b) * (a - b)` of each pair, then their sum from the first term in row-major order, as
/// the CPU interpreter computes the `mse` sum.
fn cpu_sum(a: &[f32], b: &[f32]) -> f32 {
    let mut terms = a.iter().zip(b).map(|(&x, &y)| {
        let d = x - y;
        d * d
    });
    let first = terms.next().expect("one term at least");
    terms.fold(first, |acc, term| acc + term)
}

/// The absolute sum of the terms, in `f64`: the `r` of DESIGN.md § 9 gate 5.
fn abs_terms(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum()
}

/// Runs the loss-sum `module` of `kernel` once on `a` and `b`; the output and its guard words.
fn run(
    device: &Device<'_>,
    kernel: &Kernel,
    module: &KernelModule,
    a: &[f32],
    b: &[f32],
) -> Vec<f32> {
    let uploads = [
        guarded(a),
        guarded(b),
        guarded(&[f32::from_bits(UNWRITTEN)]),
    ];
    let bytes = |values: &Vec<f32>| (values.len() * 4) as u64;
    let buffers: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|values| {
            device
                .device_local_buffer(bytes(values))
                .expect("a device buffer")
        })
        .collect();
    let mut staging: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|values| {
            let mut buffer = device
                .staging_buffer(bytes(values))
                .expect("a staging buffer");
            buffer
                .write_f32(values)
                .expect("the staging buffer is written");
            buffer
        })
        .collect();
    let pipeline = device
        .kernel_pipeline(kernel, &module.spirv)
        .expect("the pipeline is created");
    let mut recording = device.record().expect("a recording starts");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("the upload is recorded");
    }
    recording.barrier(Barrier::TransferToCompute);
    let bound: Vec<&Buffer<'_>> = buffers.iter().collect();
    recording
        .dispatch_kernel(&pipeline, kernel, &bound)
        .expect("the dispatch is recorded");
    recording.barrier(Barrier::ComputeToTransfer);
    recording
        .copy(&buffers[2], &staging[2])
        .expect("the download is recorded");
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    device
        .submit(recorded)
        .expect("the commands are submitted")
        .wait()
        .expect("the device finishes");
    staging[2].read_f32().expect("the output reads")
}

/// The profiles the modules are built for: the device's, and copies that report 256 and 260
/// bytes of workgroup memory.
fn profiles(device: &DeviceProfile) -> Vec<DeviceProfile> {
    let mut out = vec![device.clone()];
    for bytes in [256, 260] {
        let mut copy = device.clone();
        copy.max_shared_memory_bytes = bytes;
        out.push(copy);
    }
    out
}

#[test]
#[ignore = "needs a Vulkan device"]
fn the_staged_loss_sum_keeps_the_row_major_order_at_every_count_and_chunk() {
    let started = std::time::Instant::now();
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        std::env::var(VALIDATION_ENV).as_deref() == Ok("1"),
        "the validation layer is on exactly when {VALIDATION_ENV}=1"
    );
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = candidates[0].profile.clone();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let class = negotiated.numerics;
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let setup = vulkan.take_validation_messages();
    assert!(setup.is_empty(), "validation messages at setup: {setup:#?}");
    let flushes = profile.shader_denorm_preserve_f32 != Some(true);
    let sum_tol = common::term_tolerance(KernelOp::LossSum);
    let mse_tol = common::term_tolerance(KernelOp::Mse);

    let mut draw = Draw::new(0x6c6f_7373_5f73_756d);
    let mut wrong = Vec::new();
    let mut runs = 0;
    let mut bit_checked = 0;
    let mut bit_equal = 0;
    for count in COUNTS {
        let random = (draw.values(count), draw.values(count));
        let mut ordered = (vec![1.0f32; count as usize], vec![0.0f32; count as usize]);
        ordered.0[0] = 4096.0;
        let n = count as usize;
        let mut last = (vec![1.0f32; n], vec![0.0f32; n]);
        last.0[n - 1] = 4096.0;
        if n >= 2 && (n - 1) % 2 == 1 {
            last.0[n - 2] = 0.0;
        }
        let ones = last.0[..n - 1].iter().filter(|&&x| x == 1.0).count();
        for (inputs, name) in [
            (&random, "random"),
            (&ordered, "2^24, 1, 1, …"),
            (&last, "1, 1, …, 2^24"),
        ] {
            let (a, b) = inputs;
            let (ra, rb) = if flushes {
                (common::flush_subnormals(a), common::flush_subnormals(b))
            } else {
                (a.clone(), b.clone())
            };
            let expected = cpu_sum(&ra, &rb);
            if name == "2^24, 1, 1, …" {
                assert_eq!(expected, 16_777_216.0, "the CPU sum of {name}");
            }
            if name == "1, 1, …, 2^24" {
                assert_eq!(
                    expected,
                    16_777_216.0 + ones as f32,
                    "the CPU sum of {name}"
                );
            }
            let r = sum_tol.r(expected, abs_terms(&ra, &rb), count);
            let mean = expected / count as f32;
            let r_mean = mse_tol.r(mean, abs_terms(&ra, &rb), count);
            let kernel = common::loss_sum_kernel(count);
            for target in profiles(&profile) {
                let what = format!(
                    "{count} terms {name}, {} bytes",
                    target.max_shared_memory_bytes
                );
                let undivided = emit_loss_sum_undivided(&kernel, &target).expect("it emits");
                let plan = emit_module(&kernel, &target).expect("it emits");
                let first = run(&device, &kernel, &undivided, a, b);
                let second = run(&device, &kernel, &undivided, a, b);
                let divided = run(&device, &kernel, &plan, a, b);
                runs += 3;
                for output in [&first, &second, &divided] {
                    for i in 0..GUARD_WORDS {
                        if output[1 + i].to_bits() != guard(i) {
                            wrong.push(format!("{what}: guard word {i} changed"));
                        }
                    }
                }
                if first[0].to_bits() != second[0].to_bits() {
                    wrong.push(format!("{what}: {:?} then {:?}", first[0], second[0]));
                }
                if !sum_tol.accepts(first[0], expected, r) {
                    wrong.push(format!(
                        "{what}: sum gpu {:?} cpu {expected:?} outside {sum_tol}",
                        first[0]
                    ));
                }
                if !mse_tol.accepts(divided[0], mean, r_mean) {
                    wrong.push(format!(
                        "{what}: mse gpu {:?} cpu {mean:?} outside {mse_tol}",
                        divided[0]
                    ));
                }
                if first[0].to_bits() == expected.to_bits() {
                    bit_equal += 1;
                }
                if name != "random" && first[0] != expected {
                    wrong.push(format!(
                        "{what}: gpu {:?} cpu {expected:?}, so a term was added out of row-major order",
                        first[0]
                    ));
                }
                if class == NumericsClass::BitForBit {
                    bit_checked += 1;
                    if first[0].to_bits() != expected.to_bits() {
                        wrong.push(format!(
                            "{what}: gpu {:?} cpu {expected:?} not bit for bit",
                            first[0]
                        ));
                    }
                }
                for message in vulkan.take_validation_messages() {
                    wrong.push(format!("{what}: {message}"));
                }
            }
        }
    }
    eprintln!(
        "vulkan_loss_sum: class: {class}, shared memory {} bytes, {runs} runs over {} counts and 3 \
         chunk sizes, {bit_checked} sums checked bit for bit, {bit_equal} of {} equal to the CPU \
         bit for bit (reported, not checked, outside the class), {} findings, validation {}, \
         {:.2} s",
        profile.max_shared_memory_bytes,
        COUNTS.len(),
        runs / 3,
        wrong.len(),
        if vulkan.validation_enabled() {
            "on"
        } else {
            "off"
        },
        started.elapsed().as_secs_f64()
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// The term counts of the overflow case: two terms, eight (the 8-row fits of
/// `train_vulkan_non_finite`), one past a chunk of 64, and 1,000.
const OVERFLOW_COUNTS: [u64; 4] = [2, 8, 65, 1_000];

#[test]
#[ignore = "needs a Vulkan device"]
fn a_loss_sum_past_f32_max_writes_the_mean_the_cpu_writes() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = candidates[0].profile.clone();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let setup = vulkan.take_validation_messages();
    assert!(setup.is_empty(), "validation messages at setup: {setup:#?}");
    let mse_tol = common::term_tolerance(KernelOp::Mse);
    let mut wrong = Vec::new();
    let mut runs = 0;
    // Each square of 1.35e19 is past half of `f32::MAX`, so two overflow an `f32` sum while their
    // mean does not; each square of 2e19 is past `f32::MAX`, and so is the mean.
    for (d, finite) in [(1.35e19f32, true), (2e19f32, false)] {
        for count in OVERFLOW_COUNTS {
            let n = count as usize;
            let (a, b) = (vec![d; n], vec![0.0f32; n]);
            let squares = abs_terms(&a, &b);
            let mean = (squares / count as f64) as f32;
            assert_eq!(
                mean.is_finite(),
                finite,
                "the CPU mean of {count} terms {d:e}"
            );
            let r_mean = mse_tol.r(mean, squares, count);
            let kernel = common::loss_sum_kernel(count);
            for target in profiles(&profile) {
                let what = format!(
                    "{count} terms {d:e}, {} bytes",
                    target.max_shared_memory_bytes
                );
                let undivided = emit_loss_sum_undivided(&kernel, &target).expect("it emits");
                let plan = emit_module(&kernel, &target).expect("it emits");
                let sum = run(&device, &kernel, &undivided, &a, &b);
                let divided = run(&device, &kernel, &plan, &a, &b);
                runs += 2;
                for output in [&sum, &divided] {
                    for i in 0..GUARD_WORDS {
                        if output[1 + i].to_bits() != guard(i) {
                            wrong.push(format!("{what}: guard word {i} changed"));
                        }
                    }
                }
                if sum[0] != f32::INFINITY {
                    wrong.push(format!("{what}: sum gpu {:?}, not inf", sum[0]));
                }
                let passes = if finite {
                    divided[0].is_finite() && mse_tol.accepts(divided[0], mean, r_mean)
                } else {
                    divided[0].to_bits() == mean.to_bits()
                };
                if !passes {
                    wrong.push(format!(
                        "{what}: mse gpu {:?} cpu {mean:?} outside {mse_tol}",
                        divided[0]
                    ));
                }
                for message in vulkan.take_validation_messages() {
                    wrong.push(format!("{what}: {message}"));
                }
            }
        }
    }
    eprintln!(
        "vulkan_loss_sum overflow: {runs} runs over {} counts, 2 magnitudes and 3 chunk sizes, \
         {} findings",
        OVERFLOW_COUNTS.len(),
        wrong.len()
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}
