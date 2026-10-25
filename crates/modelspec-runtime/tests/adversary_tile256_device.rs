//! Adversary cases for story:matmul-tile-from-device-invocation-limit on a Vulkan device.
//!
//! 1. The 128 by 128 block of 256 invocations at the edges: `m` and `n` not multiples of 128,
//!    `k` not a multiple of 8 (65, 783, 1023) and a long `k`, each operand read as is or
//!    transposed, bound in buffers of exactly its size, writes the bits of the CPU oracle (one
//!    sum per element from the `k = 0` product in ascending `k`) in every element, with no
//!    validation message.
//! 2. Every padded layout of a transposed operand: the square tile at depths 2, 4, 8, 12 and 16,
//!    the wide tile and the block at depth 8, one- and two-row tiles at depths that are
//!    multiples of 4 (`a` padded) and not, each emitted for the device's own workgroup memory
//!    (padded) and for 16,384 bytes (the padding dropped where it does not fit), against the CPU
//!    oracle bit for bit; the block, whose least staging is 16,640 bytes, for 16,640 bytes
//!    instead, and refused for 16,384.
//!
//! Ignored by default: they create a Vulkan instance. Run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test adversary_tile256_device -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use common::Draw;
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{Kernel, KernelKind, KernelModuleId, MatMulTile, Operand};
use modelspec_core::ir::tensor::{DType, TensorType};
use modelspec_runtime::spirv::emit::emit_module;
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

/// Each element one sum from the `k = 0` product in ascending `k`, each later product fused into
/// it (DESIGN.md § 12 ADR-009).
fn oracle(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let mut sum = row[0] * b[j];
            for p in 1..k {
                sum = row[p].mul_add(b[p * n + j], sum);
            }
            out[i * n + j] = sum;
        }
    }
    out
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
    (vulkan, profile)
}

/// Runs each shape at `limit` for every flag combination and every profile of `profiles`,
/// against the CPU oracle; returns what differed and the elements compared.
fn compare(
    vulkan: &Vulkan,
    device: &Device<'_>,
    profiles: &[DeviceProfile],
    shapes: &[(u64, u64, u64)],
    limit: u32,
    want_size: Option<u32>,
) -> (Vec<String>, u64) {
    let mut wrong = Vec::new();
    let mut compared = 0u64;
    for &(m, k, n) in shapes {
        let tile = MatMulTile::of_invocations(m, k, n, limit);
        if let Some(size) = want_size {
            assert_eq!(tile.workgroup_size, size, "[{m}, {k}] x [{k}, {n}] {tile}");
        }
        let (mu, ku, nu) = (m as usize, k as usize, n as usize);
        let mut draw = Draw::new(0x7432_3536 ^ (m << 40) ^ (k << 20) ^ n);
        let a = draw.values(m * k);
        let b = draw.values(k * n);
        let want = oracle(&a, &b, mu, ku, nu);
        let (a_t, b_t) = (transpose(&a, mu, ku), transpose(&b, ku, nu));
        for flags in FLAGS {
            let kernel = matmul(m, k, n, limit, flags);
            for profile in profiles {
                let layout = tile.layout_within(flags, profile.max_shared_memory_bytes);
                let case = format!(
                    "[{m}, {k}] x [{k}, {n}] {tile} {flags:?} shared {} strides a {} b {}",
                    profile.max_shared_memory_bytes, layout.a_stride, layout.b_stride
                );
                let module = match emit_module(&kernel, profile) {
                    Ok(module) => module,
                    Err(e) => {
                        wrong.push(format!("{case}: does not emit: {e}"));
                        continue;
                    }
                };
                let a_bound: &[f32] = if flags[0] { &a_t } else { &a };
                let b_bound: &[f32] = if flags[1] { &b_t } else { &b };
                let got = run(device, &kernel, &module.spirv, [a_bound, b_bound]);
                compared += got.len() as u64;
                let differ: Vec<usize> = (0..got.len())
                    .filter(|&i| got[i].to_bits() != want[i].to_bits())
                    .collect();
                if let Some(&i) = differ.first() {
                    wrong.push(format!(
                        "{case}: {} of {} differ, first [{}, {}] {:?} against {:?}",
                        differ.len(),
                        got.len(),
                        i / nu,
                        i % nu,
                        got[i],
                        want[i]
                    ));
                }
                for message in vulkan.take_validation_messages() {
                    wrong.push(format!("{case}: {message}"));
                }
            }
        }
    }
    (wrong, compared)
}

#[test]
#[ignore = "needs a Vulkan device"]
fn the_256_invocation_block_gives_the_cpu_oracle_bits_at_partial_blocks_and_partial_tiles() {
    let (vulkan, profile) = open();
    assert!(profile.max_workgroup_invocations >= 256 && profile.max_workgroup_size.x >= 256);
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let shapes = [
        (1153, 65, 1921),
        (8191, 783, 255),
        (2049, 1023, 1000),
        (129, 64, 8191),
        (255, 65, 8191),
        (16_385, 72, 129),
        (1031, 1023, 2049),
        (129, 4097, 8065),
    ];
    let (wrong, compared) = compare(&vulkan, &device, &[profile], &shapes, 1024, Some(256));
    eprintln!(
        "validation {}; {} block shapes x 4 flags, {compared} elements against the CPU oracle",
        vulkan.validation_enabled(),
        shapes.len()
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn every_padded_layout_and_its_dropped_padding_give_the_cpu_oracle_bits() {
    let (vulkan, profile) = open();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let mut small = profile.clone();
    small.max_shared_memory_bytes = 16_384;
    let profiles = [profile, small];
    // Square tile at depths 2, 4, 8, 12, 16 (k clamps the depth); wide at depth 8; one row at
    // depths 24, 40, 56, 64 and 9; two rows at depth 16.
    let shapes = [
        (1025, 2, 1031),
        (1025, 4, 1031),
        (1025, 8, 1031),
        (1025, 12, 1031),
        (1025, 33, 1031),
        (1031, 65, 1029),
        (127, 24, 15),
        (127, 40, 15),
        (65, 56, 129),
        (65, 129, 129),
        (127, 36, 1),
        (2049, 33, 13),
        (4097, 24, 7),
    ];
    let (mut wrong, mut compared) = compare(&vulkan, &device, &profiles, &shapes, 128, None);
    // The block stages at least 16,640 bytes (`a` lines of 132 values, k-major for vector
    // reads), so it is emitted for the device's memory and for exactly 16,640 bytes, where
    // only the layout of neither operand transposed fits and the transposed reads drop their
    // padding; for 16,384 bytes it is refused, never emitted short.
    let (m, k, n) = (1153, 65, 1921);
    let [profile, small] = profiles;
    let mut least = profile.clone();
    least.max_shared_memory_bytes = 16_640;
    let (block_wrong, block_compared) = compare(
        &vulkan,
        &device,
        &[profile, least],
        &[(m, k, n)],
        1024,
        Some(256),
    );
    wrong.extend(block_wrong);
    compared += block_compared;
    for flags in FLAGS {
        match emit_module(&matmul(m, k, n, 1024, flags), &small) {
            Err(e) if e.to_string().contains("16640") && e.to_string().contains("16384") => {}
            Err(e) => wrong.push(format!(
                "the block for 16,384 bytes {flags:?}: refused as {e}"
            )),
            Ok(_) => wrong.push(format!("the block for 16,384 bytes {flags:?}: emitted")),
        }
    }
    eprintln!(
        "validation {}; {} shapes, {compared} elements against the CPU oracle",
        vulkan.validation_enabled(),
        shapes.len() + 1
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}
