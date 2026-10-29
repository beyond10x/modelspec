//! `story:vulkan-long-k-weight-grad-matmul`, acceptance `same-bits`: a matmul whose `k` of at
//! least 16,384 takes a long-`k` tile ([`MATMUL_LONG_K_BLOCK`], [`MATMUL_LONG_K_NARROW_BLOCK`])
//! writes, for every output element, the bits of the CPU's AVX2 or portable kernel, the sum of
//! its products in ascending `k` from the `k = 0` product, each later one fused with `mul_add`.
//!
//! The shapes are the three weight-gradient matmuls of the decider model, `a` read transposed,
//! then shapes with partial blocks of rows and columns, a last tile shorter than the depth, and
//! `a`, `b`, both or neither read transposed, one with the `relu` epilogue that keeps its
//! pre-activation. Every element is compared, every element is written and the guard words
//! after every written view stay unchanged.
//!
//! Ignored by default: it creates a Vulkan instance, which `task ci` never does. Run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test vulkan_matmul_long_k -- --ignored`.
#![cfg(feature = "vulkan")]

use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{
    Epilogue, Kernel, KernelKind, KernelModuleId, MATMUL_LONG_K_BLOCK, MATMUL_LONG_K_NARROW_BLOCK,
    MatMulBlock, MatMulTile, Operand,
};
use modelspec_core::ir::tensor::{DType, TensorType};
use modelspec_runtime::cpu::matmul::{self as cpu, Lhs};
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

/// `(m, k, n)`, the operands read transposed, the epilogue and the block the shape takes.
type Case = ((u64, u64, u64), [bool; 2], Option<Epilogue>, MatMulBlock);

const CASES: [Case; 7] = [
    (
        (256, 65536, 256),
        [true, false],
        None,
        MATMUL_LONG_K_NARROW_BLOCK,
    ),
    ((256, 65536, 1024), [true, false], None, MATMUL_LONG_K_BLOCK),
    ((1024, 65536, 256), [true, false], None, MATMUL_LONG_K_BLOCK),
    ((250, 16389, 1000), [true, false], None, MATMUL_LONG_K_BLOCK),
    (
        (255, 16400, 250),
        [false, false],
        KEEP,
        MATMUL_LONG_K_NARROW_BLOCK,
    ),
    (
        (300, 16384, 520),
        [false, true],
        None,
        MATMUL_LONG_K_NARROW_BLOCK,
    ),
    (
        (1030, 16385, 40),
        [true, true],
        None,
        MATMUL_LONG_K_NARROW_BLOCK,
    ),
];

/// Whether `tile` is `block` fitted to its shape.
fn is_block(tile: &MatMulTile, block: &MatMulBlock) -> bool {
    (
        tile.workgroup_size,
        tile.cols,
        tile.rows,
        tile.cols_per_invocation,
        tile.depth,
        tile.ahead,
        tile.b_pad,
    ) == (
        block.workgroup_size,
        block.cols,
        block.rows,
        block.cols_per_invocation,
        block.depth,
        block.ahead,
        0,
    )
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
        softmax_stats: None,
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

/// SplitMix64 values in [-1, 1), every seventh element a zero of alternating sign.
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
                _ => ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0,
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
fn a_long_k_matmul_writes_the_bits_of_the_cpu_fma_chain() {
    let (vulkan, profile) = open();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let cpu_kernel = cpu::Kernel::detected();
    let mut wrong = Vec::new();
    let mut compared = 0u64;
    for ((m, k, n), transposed, epilogue, block) in CASES {
        let case = format!("[{m}, {k}] x [{k}, {n}] transposed {transposed:?} {epilogue:?}");
        let tile = MatMulTile::of(m, k, n);
        if !is_block(&tile, &block) {
            wrong.push(format!("{case}: takes {tile}, not {block:?}"));
            continue;
        }
        let a = values(0x6c6f_6e67 ^ (m << 40) ^ (k << 20) ^ n, m * k);
        let b = values(0x6b6b_6b6b ^ (n << 40) ^ (k << 20) ^ m, k * n);
        let sum = cpu::matmul(
            cpu_kernel,
            &a,
            Lhs::RowMajor,
            &b,
            m as usize,
            k as usize,
            n as usize,
        )
        .expect("the CPU kernel runs");
        let want: Vec<Vec<f32>> = match epilogue {
            None => vec![sum],
            Some(_) => {
                let relu = sum.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
                vec![sum, relu]
            }
        };
        let a_bound = if transposed[0] {
            transpose(&a, m, k)
        } else {
            a
        };
        let b_bound = if transposed[1] {
            transpose(&b, k, n)
        } else {
            b
        };
        let kernel = matmul(m, k, n, epilogue, transposed);
        let module = emit_module(&kernel, &profile).expect("the kernel emits");
        let count = (m * n) as usize;
        let got = run(&device, &kernel, &module.spirv, [&a_bound, &b_bound], count);
        assert_eq!(got.len(), want.len(), "{case}: one view per write");
        for (at, (got, want)) in got.iter().zip(&want).enumerate() {
            compared += count as u64;
            let differ: Vec<usize> = (0..count)
                .filter(|&i| got[i].to_bits() != want[i].to_bits())
                .collect();
            if let Some(&i) = differ.first() {
                wrong.push(format!(
                    "{case} {tile} view {at}: {} of {count} elements differ, first [{i}] {:?} against the CPU's {:?}",
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
    eprintln!(
        "long k: {} cases, {compared} elements compared bit for bit with the CPU {cpu_kernel:?} kernel",
        CASES.len()
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}
