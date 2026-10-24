//! `story:matmul-reads-transposed-operand`, acceptance `same-bits`, per tile: a matmul kernel
//! that reads an operand transposed, bound to the row-major transpose of that operand, writes
//! the bits the kernel that reads it as is writes, bound to the operand itself, on the same
//! device. Each shape takes one of the four tiles (the wide and the square register-blocked
//! tiles, one and two rows per invocation), with whole and partial blocks, `k = 1`, a `k`
//! shorter than one tile and one whose tile loop runs several passes; each with `a`, `b` and
//! both read transposed, with and without the `relu` epilogue that keeps its pre-activation.
//! The guard words after every written view stay unchanged.
//!
//! The kernel that reads its operands as is emits what it emitted before the change, so equal
//! bits here mean the bits of the plan before the change.
//!
//! Ignored by default: it creates a Vulkan instance, which `task ci` never does. Run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test vulkan_matmul_transposed -- --ignored`.
#![cfg(feature = "vulkan")]

use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{
    Epilogue, Kernel, KernelKind, KernelModuleId, MatMulTile, Operand,
};
use modelspec_core::ir::tensor::{DType, TensorType};
use modelspec_runtime::spirv::emit::emit_module;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

const GUARD_WORDS: usize = 16;

/// The bits each output element holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_dead;

const KEEP: Option<Epilogue> = Some(Epilogue::Relu {
    writes_pre_activation: true,
});

/// `(m, k, n)` and the tile family it is meant to take.
const SHAPES: [((u64, u64, u64), Family); 14] = [
    ((1024, 64, 1024), Family::Wide),
    ((1025, 100, 1031), Family::Wide),
    ((1088, 300, 1040), Family::Wide),
    ((1031, 7, 1029), Family::Square),
    ((1024, 1, 1024), Family::Square),
    ((1100, 33, 1500), Family::Square),
    ((100, 784, 64), Family::OneRow),
    ((64, 100, 10), Family::OneRow),
    ((100, 10, 64), Family::OneRow),
    ((37, 1000, 5), Family::OneRow),
    ((5, 3, 1), Family::OneRow),
    ((784, 100, 64), Family::TwoRow),
    ((2051, 300, 10), Family::TwoRow),
    ((2049, 17, 13), Family::TwoRow),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Wide,
    Square,
    OneRow,
    TwoRow,
}

fn family(tile: &MatMulTile) -> Family {
    match (tile.cols_per_invocation, tile.cols, tile.rows) {
        (8, 128, 8) => Family::Wide,
        (4, 64, 8) => Family::Square,
        (1, _, 1) => Family::OneRow,
        (1, _, 2) => Family::TwoRow,
        _ => panic!("an unknown tile {tile}"),
    }
}

/// The matmul kernel of `[m, k] x [k, n]` reading its operands transposed as `transposed`
/// says, with the workgroups of its tile.
fn matmul(m: u64, k: u64, n: u64, epilogue: Option<Epilogue>, transposed: [bool; 2]) -> Kernel {
    let tile = MatMulTile::of(m, k, n);
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
        epilogue,
        transposed,
        workgroup_size: tile.workgroup_size,
        workgroups: [
            u32::try_from(tile.workgroups(m, n)).expect("a test matmul fits one dimension"),
            1,
            1,
        ],
    };
    kernel.id = Kernel::module_id(&kernel.signature());
    kernel
}

/// SplitMix64 values in [-4, 4), every seventh element a zero of alternating sign.
fn values(seed: u64, count: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|i| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            match i % 14 {
                3 => 0.0,
                10 => -0.0,
                _ => ((z >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0,
            }
        })
        .collect()
}

/// The row-major `[cols, rows]` transpose of the row-major `[rows, cols]` `values`.
fn transpose(values: &[f32], rows: u64, cols: u64) -> Vec<f32> {
    let (rows, cols) = (rows as usize, cols as usize);
    let mut out = vec![0.0; values.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = values[r * cols + c];
        }
    }
    out
}

fn guard(i: usize) -> u32 {
    0x5afe_0000 | i as u32
}

fn guarded(values: &[f32]) -> Vec<f32> {
    let mut words = values.to_vec();
    words.extend((0..GUARD_WORDS).map(|i| f32::from_bits(guard(i))));
    words
}

fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

/// Runs `kernel` once over `inputs`, each written view guarded and filled with [`UNWRITTEN`];
/// returns every written view whole, its guard words included, in binding order.
fn run(
    device: &Device<'_>,
    kernel: &Kernel,
    spirv: &[u32],
    inputs: [&[f32]; 2],
    count: usize,
) -> Vec<Vec<f32>> {
    let uploads: Vec<Vec<f32>> = inputs
        .iter()
        .map(|values| guarded(values))
        .chain((0..kernel.written()).map(|_| guarded(&vec![f32::from_bits(UNWRITTEN); count])))
        .collect();
    let buffers: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|values| {
            device
                .device_local_buffer(bytes_of(values.len()))
                .expect("a device buffer")
        })
        .collect();
    let mut staging: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|values| {
            let mut buffer = device
                .staging_buffer(bytes_of(values.len()))
                .expect("a staging buffer");
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
    for at in 2..buffers.len() {
        recording
            .copy(&buffers[at], &staging[at])
            .expect("the download is recorded");
    }
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    device
        .submit(recorded)
        .expect("submitted")
        .wait()
        .expect("the device finishes");
    staging[2..]
        .iter_mut()
        .map(|buffer| buffer.read_f32().expect("the output reads"))
        .collect()
}

fn open() -> (Vulkan, DeviceProfile) {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        std::env::var(VALIDATION_ENV).as_deref() == Ok("1")
    );
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = candidates[0].profile.clone();
    (vulkan, profile)
}

#[test]
#[ignore = "needs a Vulkan device"]
fn a_matmul_reading_a_transposed_operand_writes_the_bits_of_the_kernel_reading_it_as_is() {
    let (vulkan, profile) = open();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let mut wrong = Vec::new();
    let mut compared = 0u64;
    for ((m, k, n), expected) in SHAPES {
        let tile = MatMulTile::of(m, k, n);
        assert_eq!(family(&tile), expected, "[{m}, {k}] x [{k}, {n}]: {tile}");
        let a = values(0x7472_616e ^ (m << 40) ^ (k << 20) ^ n, m * k);
        let b = values(0x7370_6f73 ^ (n << 40) ^ (k << 20) ^ m, k * n);
        let (a_t, b_t) = (transpose(&a, m, k), transpose(&b, k, n));
        let count = (m * n) as usize;
        for epilogue in [None, KEEP] {
            let plain = matmul(m, k, n, epilogue, [false, false]);
            let module = emit_module(&plain, &profile).expect("the kernel emits");
            let want = run(&device, &plain, &module.spirv, [&a, &b], count);
            for transposed in [[true, false], [false, true], [true, true]] {
                let kernel = matmul(m, k, n, epilogue, transposed);
                assert_ne!(kernel.id, plain.id, "{}", kernel.signature());
                let case =
                    format!("[{m}, {k}] x [{k}, {n}] {epilogue:?} transposed {transposed:?}");
                let module = emit_module(&kernel, &profile).expect("the kernel emits");
                let a_bound: &[f32] = if transposed[0] { &a_t } else { &a };
                let b_bound: &[f32] = if transposed[1] { &b_t } else { &b };
                let got = run(&device, &kernel, &module.spirv, [a_bound, b_bound], count);
                assert_eq!(got.len(), want.len(), "{case}: one view per write");
                for (at, (got, want)) in got.iter().zip(&want).enumerate() {
                    compared += count as u64;
                    let differ: Vec<usize> = (0..count)
                        .filter(|&i| got[i].to_bits() != want[i].to_bits())
                        .collect();
                    if let Some(&i) = differ.first() {
                        wrong.push(format!(
                            "{case} view {at}: {} of {count} elements differ, first [{i}] {:?} against {:?}",
                            differ.len(),
                            got[i],
                            want[i]
                        ));
                    }
                    let unwritten = (0..count)
                        .filter(|&i| got[i].to_bits() == UNWRITTEN)
                        .count();
                    if unwritten > 0 {
                        wrong.push(format!("{case} view {at}: {unwritten} elements unwritten"));
                    }
                    let changed = (0..GUARD_WORDS)
                        .filter(|&i| got[count + i].to_bits() != guard(i))
                        .count();
                    if changed > 0 {
                        wrong.push(format!("{case} view {at}: {changed} guard words changed"));
                    }
                }
                for message in vulkan.take_validation_messages() {
                    wrong.push(format!("{case}: {message}"));
                }
            }
        }
    }
    eprintln!(
        "transposed operands: {} shapes, {compared} elements compared bit for bit",
        SHAPES.len()
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}
