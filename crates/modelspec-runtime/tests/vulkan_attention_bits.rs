//! story:vulkan-attention-workgroup-per-head on a Vulkan device: the four attention kernels
//! compute the bits they computed before their workgroups were laid out per sequence, head and
//! tile of rows. Each kernel runs alone, recomputing its softmax statistics, and the four run
//! as one call sharing them (`Kernel::softmax_stats`), over shapes that fill no tile, a head of
//! 4 columns, the 128 keys and 64 columns of `bert_tiny`, the 256 and 32 keys of four heads of
//! 64 columns of the decider's event and context encoders, and 512 keys, under a key mask
//! that masks some keys of every sequence, one that keeps every key, and one that masks all of
//! one sequence. The FNV-1a hash of every result's bits is the one recorded from the kernels of
//! story:vulkan-attention-softmax-stats-reused.
//!
//! Ignored by default: it creates a Vulkan instance, which `task ci` never does. Run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_attention_bits -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use common::Draw;
use modelspec_core::ir::kernel::{Kernel, Operand};
use modelspec_core::ir::tensor::{TensorOp, TensorType};
use modelspec_core::schedule::{attention_loss_kernel, with_softmax_stats};
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, Vulkan};

/// Which keys a case's mask keeps.
#[derive(Clone, Copy, Debug)]
enum Mask {
    /// Key 0 of every sequence and about two in three of the others.
    Some,
    /// Every key, as a packed MLM batch has.
    All,
    /// Every key of sequence 0 but none of sequence 1.
    NoneOfOne,
}

/// The shapes `[b, s, h, heads]` and masks of the cases.
const CASES: [(u64, u64, u64, u64, Mask); 9] = [
    (2, 16, 32, 2, Mask::Some),
    (2, 16, 32, 2, Mask::NoneOfOne),
    (3, 65, 8, 2, Mask::Some),
    (2, 5, 12, 3, Mask::Some),
    (1, 512, 16, 2, Mask::Some),
    (2, 128, 128, 2, Mask::All),
    (2, 128, 128, 2, Mask::Some),
    (2, 256, 256, 4, Mask::Some),
    (4, 32, 256, 4, Mask::Some),
];

/// The hashes of the results of each case, in the order of [`CASES`]: `Attention`,
/// `AttentionGradQ`, `AttentionGradK` and `AttentionGradV` alone, then the shared statistics and
/// the four results of the call that shares them.
const RECORDED: [[u64; 9]; 9] = [
    [
        0xdc3620e2eecd8df8,
        0x4670d6007ae00ec9,
        0xb251715af1c76d0c,
        0x665fa91db0c52e85,
        0xfa4b77930f571d3b,
        0xdc3620e2eecd8df8,
        0x4670d6007ae00ec9,
        0xb251715af1c76d0c,
        0x665fa91db0c52e85,
    ],
    [
        0xd65384ad758a0f26,
        0x90b9c4ce03cf23ad,
        0xd1af89e4b5f6a617,
        0xc730ff30aa4b6088,
        0xc1fcc7ffb39657de,
        0xd65384ad758a0f26,
        0x90b9c4ce03cf23ad,
        0xd1af89e4b5f6a617,
        0xc730ff30aa4b6088,
    ],
    [
        0x06afd7cb222c827c,
        0x13110bd9bce4d338,
        0xdf70b3147d9b44e3,
        0x0e50d8527e00025f,
        0x7535b5acf34bb203,
        0x06afd7cb222c827c,
        0x13110bd9bce4d338,
        0xdf70b3147d9b44e3,
        0x0e50d8527e00025f,
    ],
    [
        0xa5fc3dd6316c5661,
        0x08a7fdc0400cc125,
        0x65bdf48a1cfcb523,
        0x65b534f2fac0599e,
        0xa2852b90f1d8594a,
        0xa5fc3dd6316c5661,
        0x08a7fdc0400cc125,
        0x65bdf48a1cfcb523,
        0x65b534f2fac0599e,
    ],
    [
        0x77f579462aa5d6b4,
        0x42b7d7f2b7c68150,
        0x5a028544a427afe9,
        0x350a7138671a9d5e,
        0x4908ca8022b28442,
        0x77f579462aa5d6b4,
        0x42b7d7f2b7c68150,
        0x5a028544a427afe9,
        0x350a7138671a9d5e,
    ],
    [
        0xbe90f0ba346e6655,
        0x4fedb084d25cffd5,
        0x56fc09eb472f030e,
        0x73a927477dcf29c2,
        0xd1de0ef737545041,
        0xbe90f0ba346e6655,
        0x4fedb084d25cffd5,
        0x56fc09eb472f030e,
        0x73a927477dcf29c2,
    ],
    [
        0x42b91fd07df21ab0,
        0xafc5d20114218169,
        0x04c6627c5d7cecc3,
        0xdfa586b42223be68,
        0x72cb3ca29f17505a,
        0x42b91fd07df21ab0,
        0xafc5d20114218169,
        0x04c6627c5d7cecc3,
        0xdfa586b42223be68,
    ],
    [
        0xcac7d8af479f58a1,
        0x463805851be48502,
        0x60afb4565e45602d,
        0x88b41724b2cb9360,
        0xbc1d1147c5408c2f,
        0xcac7d8af479f58a1,
        0x463805851be48502,
        0x60afb4565e45602d,
        0x88b41724b2cb9360,
    ],
    [
        0x60a611d9fd34228b,
        0x7b8042ebcb36076a,
        0x3ecff08e443668c5,
        0xd78730cebb828740,
        0x3ac867809ed1b1b7,
        0x60a611d9fd34228b,
        0x7b8042ebcb36076a,
        0x3ecff08e443668c5,
        0xd78730cebb828740,
    ],
];

fn key_mask(b: u64, s: u64, mask: Mask) -> Vec<f32> {
    (0..b * s)
        .map(|i| {
            let (seq, j) = (i / s, i % s);
            let keep = match mask {
                Mask::Some => j == 0 || (j * 7 + seq * 5) % 3 != 0,
                Mask::All => true,
                Mask::NoneOfOne => seq == 0,
            };
            if keep { 1.0 } else { 0.0 }
        })
        .collect()
}

/// FNV-1a over the bits of `values`, low byte first.
fn fnv(values: &[f32]) -> u64 {
    values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
}

/// The kernel of `op` over `arity` buffers of the case's types, as the schedule builds it.
fn kernel_of(op: &TensorOp, rows: &TensorType, mask: &TensorType) -> Kernel {
    let arity = if matches!(op, TensorOp::Attention { .. }) {
        4
    } else {
        5
    };
    let operands = (0..arity)
        .map(|at| Operand::Buffer(if at == 3 { mask.clone() } else { rows.clone() }))
        .collect();
    attention_loss_kernel(op, operands, rows.clone()).expect("an attention kernel")
}

/// Runs `kernels` in order in one recording, each bound to the inputs it reads (`q`, `k`, `v`,
/// the mask and `g` of `inputs`), the statistics buffer where it shares them, and its own
/// result; returns the statistics and every result.
fn run(
    device: &Device<'_>,
    kernels: &[Kernel],
    inputs: &[Vec<f32>],
    stats: usize,
) -> Vec<Vec<f32>> {
    let words: Vec<Vec<u32>> = kernels.iter().map(|k| emit(k).expect("emits")).collect();
    let pipelines: Vec<_> = kernels
        .iter()
        .zip(&words)
        .map(|(k, w)| device.kernel_pipeline(k, w).expect("a pipeline"))
        .collect();
    let rows = inputs[0].len();
    let mut lengths: Vec<usize> = inputs.iter().map(Vec::len).collect();
    lengths.push(stats.max(1));
    lengths.extend(kernels.iter().map(|_| rows));
    let buffers: Vec<Buffer<'_>> = lengths
        .iter()
        .map(|&n| device.device_local_buffer(4 * n as u64).expect("a buffer"))
        .collect();
    let mut staging: Vec<Buffer<'_>> = lengths
        .iter()
        .map(|&n| {
            device
                .staging_buffer(4 * n as u64)
                .expect("a staging buffer")
        })
        .collect();
    let unwritten = |n: usize| vec![f32::from_bits(0x7fc0_dead); n];
    for (at, buffer) in staging.iter_mut().enumerate() {
        let values = match inputs.get(at) {
            Some(values) => values.clone(),
            None => unwritten(lengths[at]),
        };
        buffer.write_f32(&values).expect("written");
    }
    let mut recording = device.record().expect("a recording");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("an upload");
    }
    recording.barrier(Barrier::TransferToCompute);
    let stats_at = inputs.len();
    for (at, (kernel, pipeline)) in kernels.iter().zip(&pipelines).enumerate() {
        let reads = kernel
            .operands
            .iter()
            .filter(|o| matches!(o, Operand::Buffer(_)))
            .count();
        let mut bound: Vec<&Buffer<'_>> = buffers[..reads].iter().collect();
        if kernel.softmax_stats.is_some() {
            bound.push(&buffers[stats_at]);
        }
        bound.push(&buffers[stats_at + 1 + at]);
        recording
            .dispatch_kernel(pipeline, kernel, &bound)
            .expect("a dispatch");
        recording.barrier(Barrier::ComputeToCompute);
    }
    recording.barrier(Barrier::ComputeToTransfer);
    for at in stats_at..buffers.len() {
        recording
            .copy(&buffers[at], &staging[at])
            .expect("a download");
    }
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    device
        .submit(recorded)
        .expect("submitted")
        .wait()
        .expect("the device finishes");
    staging[stats_at..]
        .iter_mut()
        .map(|b| b.read_f32().expect("read"))
        .collect()
}

#[test]
#[ignore = "needs a Vulkan device"]
fn the_attention_kernels_compute_the_bits_recorded_before_their_tiling() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let mut seen = Vec::new();
    for (b, s, h, heads, mask) in CASES {
        let mut draw = Draw::new(0x6174_7465_6e74_6f6e ^ s ^ (h << 16));
        let n = b * s;
        let rows = TensorType::f32(&[n, h]);
        let mask_ty = TensorType::f32(&[b, s]);
        let scaled = |draw: &mut Draw, by: f32| -> Vec<f32> {
            draw.values(n * h).into_iter().map(|x| x * by).collect()
        };
        let q = scaled(&mut draw, 0.25);
        let k = scaled(&mut draw, 0.25);
        let v = scaled(&mut draw, 1.0);
        let g = scaled(&mut draw, 1.0);
        let inputs = vec![q, k, v, key_mask(b, s, mask), g];
        let ops = [
            TensorOp::Attention { heads },
            TensorOp::AttentionGradQ { heads },
            TensorOp::AttentionGradK { heads },
            TensorOp::AttentionGradV { heads },
        ];
        let mut hashes = Vec::new();
        for op in &ops {
            let kernel = kernel_of(op, &rows, &mask_ty);
            let out = run(&device, &[kernel], &inputs, 0);
            hashes.push(fnv(&out[1]));
        }
        let shared: Vec<Kernel> = ops
            .iter()
            .map(|op| with_softmax_stats(kernel_of(op, &rows, &mask_ty)).expect("shares"))
            .collect();
        let out = run(&device, &shared, &inputs, (n * heads * 3) as usize);
        hashes.extend(out.iter().map(|values| fnv(values)));
        let messages = vulkan.take_validation_messages();
        assert!(
            messages.is_empty(),
            "[{b}, {s}] x {h} heads {heads} {mask:?}: {messages:?}"
        );
        seen.push(hashes);
    }
    let table: Vec<String> = seen
        .iter()
        .map(|hashes| {
            let words: Vec<String> = hashes.iter().map(|h| format!("0x{h:016x}")).collect();
            format!("    [{}],", words.join(", "))
        })
        .collect();
    let expected: Vec<Vec<u64>> = RECORDED.iter().map(|r| r.to_vec()).collect();
    assert_eq!(seen, expected, "seen:\n{}", table.join("\n"));
}
