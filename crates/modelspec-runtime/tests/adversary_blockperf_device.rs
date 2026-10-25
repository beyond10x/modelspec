//! Adversary cases for story:matmul-block-kernel-throughput on a Vulkan device: the block of 256
//! invocations with vector reads of workgroup memory (`MatMulTile::vector_reads`).
//!
//! 1. Every boundary of DESIGN.md § 9 gate 5 the block can take: `m` or `n` in {127, 128, 129,
//!    255, 256, 257, 511, 512, 513}, the other side long enough for 128 blocks and 1, 2 or 3
//!    past a multiple of 4 (so the last run of 4 rows or columns is partial), `k` from 64 (the
//!    least the block takes) through 513 with every residue mod 4 and mod 8, each operand read as
//!    is and transposed: the block writes the CPU oracle's bits (ADR-009, `mul_add` from the
//!    `k = 0` product) and the bits of the tile of 128 invocations of the same shape.
//! 2. Signed zeros, subnormals (the RTX 3090 reports `shaderDenormPreserveFloat32` false),
//!    values near the overflow bound and products that cancel: the block and the tile of 128
//!    invocations write the same bits element for element; how each compares with the CPU
//!    oracle, on the inputs as given and with subnormals flushed, is printed.
//! 3. The block under `fast_math`: within the tolerance of § 9 gate 5 against the CPU oracle,
//!    its module differs from the deterministic one, its cache key says `fast_math`.
//!
//! Every case fails on a validation message when run with `MODELSPEC_VK_VALIDATION=1`.
//! Ignored by default: they create a Vulkan instance. Run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test adversary_blockperf_device -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use common::{Draw, KernelOp, flush_subnormals, term_tolerance};
use modelspec_core::artifact::Backend;
use modelspec_core::compile::compile_artifact;
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{Kernel, KernelKind, KernelModuleId, MatMulTile, Operand};
use modelspec_core::ir::tensor::{DType, TensorType};
use modelspec_runtime::spirv::emit::{emit_kernels, emit_module};
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

/// The bits each output element holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_7256;

const FLAGS: [[bool; 2]; 4] = [[false, false], [true, false], [false, true], [true, true]];

fn matmul(m: u64, k: u64, n: u64, limit: u32, transposed: [bool; 2]) -> Kernel {
    let tile = MatMulTile::of_invocations(m, k, n, limit);
    let mut kernel = Kernel {
        id: KernelModuleId([0; 16]),
        kind: KernelKind::MatMul,
        op: "MatMul",
        dtype: DType::F32,
        operands: vec![
            Operand::Buffer(TensorType::f32(&[m, k])),
            Operand::Buffer(TensorType::f32(&[k, n])),
        ],
        result: TensorType::f32(&[m, n]),
        reduces: None,
        epilogue: None,
        transposed,
        workgroup_size: tile.workgroup_size,
        workgroups: [
            u32::try_from(tile.workgroups(m, n)).expect("one dimension"),
            1,
            1,
        ],
    };
    kernel.id = Kernel::module_id(&kernel.signature());
    kernel
}

/// Each element one sum from the `k = 0` product in ascending `k`, each later product fused
/// (DESIGN.md § 12 ADR-009); and the sum of the absolute values of its terms, the `r` of § 9
/// gate 5.
fn oracle(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f64>) {
    let mut out = vec![0.0f32; m * n];
    let mut terms = vec![0.0f64; m * n];
    for i in 0..m {
        let row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let mut sum = row[0] * b[j];
            let mut abs = f64::from((row[0] * b[j]).abs());
            for p in 1..k {
                let bv = b[p * n + j];
                sum = row[p].mul_add(bv, sum);
                abs += f64::from((row[p] * bv).abs());
            }
            out[i * n + j] = sum;
            terms[i * n + j] = abs;
        }
    }
    (out, terms)
}

fn transpose(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; values.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = values[r * cols + c];
        }
    }
    out
}

fn run(device: &Device<'_>, kernel: &Kernel, spirv: &[u32], inputs: [&[f32]; 2]) -> Vec<f32> {
    let (m, _, n) = Kernel::matmul_shape(&kernel.operands).expect("a matmul");
    let written = vec![f32::from_bits(UNWRITTEN); (m * n) as usize];
    let uploads: Vec<&[f32]> = vec![inputs[0], inputs[1], &written];
    let bytes = |values: &[f32]| (values.len() * 4) as u64;
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
            let mut buffer = device.staging_buffer(bytes(values)).expect("staging");
            buffer.write_f32(values).expect("written");
            buffer
        })
        .collect();
    let pipeline = device
        .kernel_pipeline(kernel, spirv)
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
        .expect("submitted")
        .wait()
        .expect("the device finishes");
    staging[2].read_f32().expect("the output reads")
}

fn open() -> (Vulkan, DeviceProfile) {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        std::env::var(VALIDATION_ENV).as_deref() == Ok("1")
    );
    let profile = vulkan.candidates().expect("the devices are probed")[0]
        .profile
        .clone();
    assert!(profile.max_workgroup_invocations >= 256 && profile.max_workgroup_size.x >= 256);
    (vulkan, profile)
}

/// The first element where `got` and `want` differ in bits, and how many do.
fn differ(got: &[f32], want: &[f32], n: usize) -> Option<String> {
    let differ: Vec<usize> = (0..got.len())
        .filter(|&i| got[i].to_bits() != want[i].to_bits())
        .collect();
    differ.first().map(|&i| {
        format!(
            "{} of {} differ, first [{}, {}] {:?} ({:#010x}) against {:?} ({:#010x})",
            differ.len(),
            got.len(),
            i / n,
            i % n,
            got[i],
            got[i].to_bits(),
            want[i],
            want[i].to_bits()
        )
    })
}

/// Runs `[m, k] x [k, n]` for every flag combination on the block (limit 1024) and on the tile
/// of 128 invocations; returns what differed from `want` (bits) or between the two.
#[allow(clippy::too_many_arguments)]
fn block_and_tile(
    vulkan: &Vulkan,
    device: &Device<'_>,
    profile: &DeviceProfile,
    (m, k, n): (u64, u64, u64),
    a: &[f32],
    b: &[f32],
    want: Option<&[f32]>,
    wrong: &mut Vec<String>,
) -> u64 {
    let (mu, ku, nu) = (m as usize, k as usize, n as usize);
    let (a_t, b_t) = (transpose(a, mu, ku), transpose(b, ku, nu));
    let mut compared = 0;
    for flags in FLAGS {
        let a_bound: &[f32] = if flags[0] { &a_t } else { a };
        let b_bound: &[f32] = if flags[1] { &b_t } else { b };
        let mut outputs = Vec::new();
        for limit in [1024, 128] {
            let kernel = matmul(m, k, n, limit, flags);
            let tile = MatMulTile::of_invocations(m, k, n, limit);
            let case = format!("[{m}, {k}] x [{k}, {n}] {flags:?} {tile}");
            let module = match emit_module(&kernel, profile) {
                Ok(module) => module,
                Err(e) => {
                    wrong.push(format!("{case}: does not emit: {e}"));
                    continue;
                }
            };
            let got = run(device, &kernel, &module.spirv, [a_bound, b_bound]);
            compared += got.len() as u64;
            if let Some(want) = want
                && let Some(why) = differ(&got, want, nu)
            {
                wrong.push(format!("{case} against the CPU oracle: {why}"));
            }
            for message in vulkan.take_validation_messages() {
                wrong.push(format!("{case}: {message}"));
            }
            outputs.push((case, got));
        }
        if let [(block, got), (_, tile)] = &outputs[..]
            && let Some(why) = differ(got, tile, nu)
        {
            wrong.push(format!(
                "{block} against the tile of 128 invocations: {why}"
            ));
        }
    }
    compared
}

/// The boundary shapes: `b` on one side, the other side the least past a multiple of 128 that
/// gives 128 blocks of 128 by 128, plus 1, 2 or 3; `k` rotating through 64 to 513.
fn boundary_shapes() -> Vec<(u64, u64, u64)> {
    let ks = [
        64, 65, 66, 67, 127, 128, 129, 255, 256, 257, 511, 512, 513, 72, 100, 199, 300,
    ];
    let mut shapes = Vec::new();
    let mut at = 0;
    let mut next_k = || {
        let k = ks[at % ks.len()];
        at += 1;
        k
    };
    for (i, b) in [127u64, 128, 129, 255, 256, 257, 511, 512, 513]
        .into_iter()
        .enumerate()
    {
        let long = 128 * 128u64.div_ceil(b.div_ceil(128)) + 1 + i as u64 % 3;
        shapes.push((b, next_k(), long));
        if b >= 128 {
            shapes.push((long + 1, next_k(), b));
        }
    }
    shapes
}

#[test]
#[ignore = "needs a Vulkan device"]
fn the_block_at_every_boundary_gives_the_oracle_bits_and_the_bits_of_the_tile_of_128() {
    let (vulkan, profile) = open();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let shapes = boundary_shapes();
    let mut wrong = Vec::new();
    let mut compared = 0;
    for &(m, k, n) in &shapes {
        let tile = MatMulTile::of_invocations(m, k, n, 1024);
        assert!(tile.vector_reads(), "[{m}, {k}] x [{k}, {n}] takes {tile}");
        let mut draw = Draw::new(0xb10c_9e4f ^ (m << 40) ^ (k << 20) ^ n);
        let a = draw.values(m * k);
        let b = draw.values(k * n);
        let (want, _) = oracle(&a, &b, m as usize, k as usize, n as usize);
        compared += block_and_tile(
            &vulkan,
            &device,
            &profile,
            (m, k, n),
            &a,
            &b,
            Some(&want),
            &mut wrong,
        );
    }
    eprintln!(
        "validation {}; {} block shapes x 4 flags x 2 tiles, {compared} elements: {shapes:?}",
        vulkan.validation_enabled(),
        shapes.len()
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Special values by position: signed zeros, subnormals of both signs, the least normal,
/// values whose products near the overflow bound, and normal draws.
fn special(count: u64, seed: u64) -> Vec<f32> {
    let mut draw = Draw::new(seed);
    let normal = draw.values(count);
    (0..count as usize)
        .map(|i| match (i * 7 + seed as usize) % 16 {
            0 => 0.0,
            1 => -0.0,
            2 => f32::from_bits(1),
            3 => -f32::from_bits(0x0004_0000),
            4 => f32::MIN_POSITIVE,
            5 => 3.0e18,
            6 => -3.0e18,
            7 => 1.0e-20,
            _ => normal[i],
        })
        .collect()
}

#[test]
#[ignore = "needs a Vulkan device"]
fn signed_zeros_subnormals_and_cancellation_give_the_block_the_bits_of_the_tile_of_128() {
    let (vulkan, profile) = open();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let (m, k, n) = (1153u64, 129u64, 1921u64);
    let (mu, ku, nu) = (m as usize, k as usize, n as usize);
    let a = special(m * k, 3);
    let mut b = special(k * n, 11);
    // Rows 0..8 of `b`: each odd row the negation of the row before, so the products of a row
    // of `a` with equal neighbours cancel to their rounding error.
    for p in (1..8).step_by(2) {
        for j in 0..nu {
            b[p * nu + j] = -b[(p - 1) * nu + j];
        }
    }
    // Tiny inputs, so that sums stay subnormal or zero: `a` of signed zeros, subnormals and
    // 1e-20 (every row a multiple of 5 all -0.0), `b` of signed zeros, 1e-20, 1.0 and -1.0.
    let tiny_a: Vec<f32> = (0..mu * ku)
        .map(|i| {
            if (i / ku).is_multiple_of(5) {
                return -0.0;
            }
            match i % 6 {
                0 => 0.0,
                1 => -0.0,
                2 => f32::from_bits(7),
                3 => -f32::from_bits(0x0002_0000),
                4 => 1.0e-20,
                _ => -1.0e-20,
            }
        })
        .collect();
    let tiny_b: Vec<f32> = (0..ku * nu)
        .map(|i| match i % 5 {
            0 => 0.0,
            1 => 1.0e-20,
            2 => 1.0,
            3 => -1.0,
            _ => f32::from_bits(3),
        })
        .collect();
    let mut wrong = Vec::new();
    let mut compared = 0;
    for (name, a, b) in [("special", &a, &b), ("tiny", &tiny_a, &tiny_b)] {
        compared += block_and_tile(
            &vulkan,
            &device,
            &profile,
            (m, k, n),
            a,
            b,
            None,
            &mut wrong,
        );
        // How the block compares with the oracle on the inputs as given and flushed, printed
        // only: the device declares no DenormPreserve, so neither is the contract here.
        let kernel = matmul(m, k, n, 1024, [false, false]);
        let module = emit_module(&kernel, &profile).expect("the block emits");
        let got = run(&device, &kernel, &module.spirv, [a, b]);
        let (exact, _) = oracle(a, b, mu, ku, nu);
        let (flushed, _) = oracle(&flush_subnormals(a), &flush_subnormals(b), mu, ku, nu);
        let subnormal_out = got.iter().filter(|v| v.is_subnormal()).count();
        let negative_zero = got.iter().filter(|v| v.to_bits() == 0x8000_0000).count();
        eprintln!(
            "{name}: validation {}; block against the oracle: {}; against the oracle on flushed \
             inputs: {}; subnormal outputs {subnormal_out}, -0.0 outputs {negative_zero}; \
             denorm preserve reported {:?}",
            vulkan.validation_enabled(),
            differ(&got, &exact, nu).unwrap_or_else(|| "same bits".to_owned()),
            differ(&got, &flushed, nu).unwrap_or_else(|| "same bits".to_owned()),
            profile.shader_denorm_preserve_f32
        );
        for message in vulkan.take_validation_messages() {
            wrong.push(message);
        }
    }
    eprintln!("{compared} elements block against tile");
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn the_block_under_fast_math_is_within_the_gate_5_tolerance() {
    let (vulkan, profile) = open();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/linear_regression.mspec"
    ))
    .expect("the example reads");
    let mut artifact =
        compile_artifact("linear_regression.mspec", &source, Backend::Vulkan).expect("compiles");
    let (m, k, n) = (1153u64, 129u64, 1921u64);
    let (mu, ku, nu) = (m as usize, k as usize, n as usize);
    let mut draw = Draw::new(0xfa57_3a70);
    let a = draw.values(m * k);
    let b = draw.values(k * n);
    let (want, terms) = oracle(&a, &b, mu, ku, nu);
    let (a_t, b_t) = (transpose(&a, mu, ku), transpose(&b, ku, nu));
    let tolerance = term_tolerance(KernelOp::MatMul);
    let mut wrong = Vec::new();
    let mut identical = 0;
    for flags in FLAGS {
        let kernel = matmul(m, k, n, 1024, flags);
        assert!(MatMulTile::of_invocations(m, k, n, 1024).vector_reads());
        artifact.execution.kernels = vec![kernel.clone()];
        artifact.fast_math = false;
        let exact = emit_kernels(&artifact, &profile).expect("emits").remove(0);
        artifact.fast_math = true;
        let fast = emit_kernels(&artifact, &profile).expect("emits").remove(0);
        let case = format!("[{m}, {k}] x [{k}, {n}] {flags:?}");
        if fast.spirv == exact.spirv {
            wrong.push(format!(
                "{case}: the fast_math module is the deterministic one"
            ));
        }
        if !fast.cache_key.fast_math || exact.cache_key.fast_math {
            wrong.push(format!(
                "{case}: cache keys {:?} {:?}",
                exact.cache_key, fast.cache_key
            ));
        }
        let a_bound: &[f32] = if flags[0] { &a_t } else { &a };
        let b_bound: &[f32] = if flags[1] { &b_t } else { &b };
        let got_exact = run(&device, &kernel, &exact.spirv, [a_bound, b_bound]);
        let got_fast = run(&device, &kernel, &fast.spirv, [a_bound, b_bound]);
        if let Some(why) = differ(&got_exact, &want, nu) {
            wrong.push(format!("{case} deterministic against the oracle: {why}"));
        }
        let outside: Vec<usize> = (0..got_fast.len())
            .filter(|&i| {
                let r = tolerance.r(want[i], terms[i], k);
                !tolerance.accepts(got_fast[i], want[i], r)
            })
            .collect();
        if let Some(&i) = outside.first() {
            wrong.push(format!(
                "{case} fast_math: {} outside {tolerance}, first [{}, {}] {} against {}",
                outside.len(),
                i / nu,
                i % nu,
                got_fast[i],
                want[i]
            ));
        }
        if differ(&got_fast, &got_exact, nu).is_none() {
            identical += 1;
        }
        for message in vulkan.take_validation_messages() {
            wrong.push(format!("{case}: {message}"));
        }
    }
    eprintln!(
        "validation {}; fast_math bit-identical to deterministic in {identical} of 4 flag sets; \
         module ids are shared (the id is the kernel's, the cache key carries fast_math)",
        vulkan.validation_enabled()
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}
