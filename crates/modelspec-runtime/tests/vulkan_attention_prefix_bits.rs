//! story:vulkan-attention-tiled-decider-shapes on a Vulkan device: the four attention kernels
//! compute the bits they computed before they took a key mask that keeps a prefix of each
//! sequence's keys, as a padded batch has, for keys `0..L` alone. The cases are the decider's
//! event encoder (`[65536, 256]` rows of four heads of 64 columns over 256 sequences of 256
//! keys) and context encoder (`[4096, 256]` over 128 sequences of 32 keys), smaller ones of
//! those widths, BERT-tiny's 128 keys of two heads, 65 keys of a head of 4 columns that fill no
//! tile, and 512 keys; each sequence keeps its own prefix length, among them none, one, a tile's
//! worth and all. Two masks that are not prefixes keep the path every other mask takes: the last
//! keys of each sequence, and every key but one in the middle. Each kernel runs alone,
//! recomputing its softmax statistics, and the four run as one call sharing them. The FNV-1a
//! hash of every result's bits is the one recorded from the kernels of 0.23.0 (perf/decider-step
//! 2fe753ab), and the results of the cases of at most 2,048 rows whose every sequence keeps a
//! key (the CPU refuses another) are within `GPU_ATOL + GPU_RTOL * |c|` of the CPU interpreter's
//! `c`.
//!
//! Ignored by default: it creates a Vulkan instance, which `task ci` never does. Run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_attention_prefix_bits -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;

use common::{Draw, GPU_ATOL, GPU_RTOL, within};
use modelspec_core::ir::kernel::{Kernel, Operand};
use modelspec_core::ir::tensor::{TensorOp, TensorProgramBuilder, TensorType};
use modelspec_core::schedule::{attention_loss_kernel, with_softmax_stats};
use modelspec_runtime::cpu::interpreter::evaluate_seeded;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, Vulkan};

/// Which keys of each sequence a case's mask keeps.
#[derive(Clone, Copy, Debug)]
enum Mask {
    /// Keys `0..L` of sequence `seq`, `L` the `seq`-th of the lengths, taken in turn.
    Prefix(&'static [u64]),
    /// The last `L` keys of sequence `seq`, `L` as for `Prefix`.
    Suffix(&'static [u64]),
    /// Every key but key `s / 2`.
    AllButMiddle,
}

/// Prefix lengths of 256 keys: none, one, the tile edges about 4 and 32, the decider's mean of
/// about 88, half, one short of all, all.
const LENGTHS_256: &[u64] = &[0, 1, 3, 4, 5, 20, 31, 32, 33, 88, 128, 255, 256];
const LENGTHS_32: &[u64] = &[0, 1, 4, 5, 17, 31, 32];
const LENGTHS_128: &[u64] = &[128, 64, 7, 1, 127];
const LENGTHS_32_KEPT: &[u64] = &[1, 4, 5, 17, 31, 32];
const LENGTHS_65: &[u64] = &[65, 33, 2];
const LENGTHS_512: &[u64] = &[300, 512];

/// The shapes `[b, s, h, heads]` and masks of the cases.
const CASES: [(u64, u64, u64, u64, Mask); 11] = [
    (256, 256, 256, 4, Mask::Prefix(LENGTHS_256)),
    (128, 32, 256, 4, Mask::Prefix(LENGTHS_32)),
    (13, 256, 256, 4, Mask::Prefix(LENGTHS_256)),
    (7, 32, 256, 4, Mask::Prefix(LENGTHS_32)),
    (6, 32, 256, 4, Mask::Prefix(LENGTHS_32_KEPT)),
    (5, 128, 128, 2, Mask::Prefix(LENGTHS_128)),
    (3, 65, 8, 2, Mask::Prefix(LENGTHS_65)),
    (2, 512, 16, 2, Mask::Prefix(LENGTHS_512)),
    (6, 256, 256, 4, Mask::Suffix(LENGTHS_256)),
    (2, 256, 256, 4, Mask::AllButMiddle),
    (3, 32, 256, 4, Mask::AllButMiddle),
];

/// The hashes of the results of each case, in the order of [`CASES`]: `Attention`,
/// `AttentionGradQ`, `AttentionGradK` and `AttentionGradV` alone, then the shared statistics and
/// the four results of the call that shares them, recorded from the kernels of 0.23.0.
const RECORDED: [[u64; 9]; 11] = [
    [
        0xb7756a46d1626b49,
        0x6c825267083ba47c,
        0x0cedc472834a74b9,
        0xdad6456cbe4659b5,
        0x336a15c25c39e733,
        0xb7756a46d1626b49,
        0x6c825267083ba47c,
        0x0cedc472834a74b9,
        0xdad6456cbe4659b5,
    ],
    [
        0x463640340d44c1c5,
        0xfa43f6a8db39480a,
        0x23658460f18d6c38,
        0xc03c3e182f7d93c3,
        0xbe50a064de570132,
        0x463640340d44c1c5,
        0xfa43f6a8db39480a,
        0x23658460f18d6c38,
        0xc03c3e182f7d93c3,
    ],
    [
        0xb236dc391a21e11f,
        0x5f247054fabe43cc,
        0xb866249f468d9f41,
        0xec33c6b1a4d800e2,
        0xc7eaea4cdb9846fc,
        0xb236dc391a21e11f,
        0x5f247054fabe43cc,
        0xb866249f468d9f41,
        0xec33c6b1a4d800e2,
    ],
    [
        0x92b12b90494b295d,
        0xf86e86a6b7271e57,
        0x6e58d055ad6dd794,
        0x14980c200b8d6ff5,
        0x5693918c9831fc31,
        0x92b12b90494b295d,
        0xf86e86a6b7271e57,
        0x6e58d055ad6dd794,
        0x14980c200b8d6ff5,
    ],
    [
        0xedf5d4427e606530,
        0x698a5a08d0532bab,
        0x5b9d272b2367ae1c,
        0xe8138a727d58ddf8,
        0x8609258f01eed1bd,
        0xedf5d4427e606530,
        0x698a5a08d0532bab,
        0x5b9d272b2367ae1c,
        0xe8138a727d58ddf8,
    ],
    [
        0xdfb1e7d23aee533c,
        0x08733d2622a2fc68,
        0xf001ac3980204442,
        0x24cbb9f3b22dbac6,
        0x6f3f9d058ab153c8,
        0xdfb1e7d23aee533c,
        0x08733d2622a2fc68,
        0xf001ac3980204442,
        0x24cbb9f3b22dbac6,
    ],
    [
        0x329751a0b6596d60,
        0xedbafb1bf902481c,
        0x0e5319d99fd3733a,
        0xcb2a18c3d00513b2,
        0x3fafd0c7610b4c52,
        0x329751a0b6596d60,
        0xedbafb1bf902481c,
        0x0e5319d99fd3733a,
        0xcb2a18c3d00513b2,
    ],
    [
        0xf3e578a5aff5bf01,
        0x365f6f6e0cb56677,
        0x380909c2a2767134,
        0xc7d956eb294e67b8,
        0x7f19a2a2a7317e68,
        0xf3e578a5aff5bf01,
        0x365f6f6e0cb56677,
        0x380909c2a2767134,
        0xc7d956eb294e67b8,
    ],
    [
        0x2b269e41f2ce00b3,
        0xa000082da824517a,
        0x9932f43ee178eae3,
        0xd25086e3f911853a,
        0xb79ae80aa266dbf7,
        0x2b269e41f2ce00b3,
        0xa000082da824517a,
        0x9932f43ee178eae3,
        0xd25086e3f911853a,
    ],
    [
        0x688073ceb154d8b5,
        0xeaa8f942494bb56f,
        0xdeb257393df59d9a,
        0xb05a1ff03c3e98ac,
        0x77d1ec832e1db10f,
        0x688073ceb154d8b5,
        0xeaa8f942494bb56f,
        0xdeb257393df59d9a,
        0xb05a1ff03c3e98ac,
    ],
    [
        0x9ef83e2d7d534501,
        0xdfe31978d843fb36,
        0xb042c0092af853d8,
        0x221531982ea5a89a,
        0x7773bfe0edba5af0,
        0x9ef83e2d7d534501,
        0xdfe31978d843fb36,
        0xb042c0092af853d8,
        0x221531982ea5a89a,
    ],
];

fn key_mask(b: u64, s: u64, mask: Mask) -> Vec<f32> {
    (0..b * s)
        .map(|i| {
            let (seq, j) = (i / s, i % s);
            let keep = match mask {
                Mask::Prefix(lengths) => j < lengths[seq as usize % lengths.len()],
                Mask::Suffix(lengths) => j >= s - lengths[seq as usize % lengths.len()],
                Mask::AllButMiddle => j != s / 2,
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

/// The CPU interpreter's result of `op` over the first `arity` of `inputs` (`q`, `k`, `v`, the
/// mask and `g`).
fn cpu(op: &TensorOp, inputs: &[Vec<f32>], shapes: &[Vec<u64>]) -> Vec<f32> {
    let arity = if matches!(op, TensorOp::Attention { .. }) {
        4
    } else {
        5
    };
    let mut builder = TensorProgramBuilder::new();
    let mut values = Vec::new();
    let mut host = BTreeMap::new();
    for at in 0..arity {
        let name = format!("x{at}");
        let ty = TensorType::f32(&shapes[at]);
        host.insert(
            name.clone(),
            HostTensor::new(shapes[at].clone(), inputs[at].clone()).expect("a tensor"),
        );
        values.push(builder.input(name, ty).expect("a new input"));
    }
    let result = builder.push(op.clone(), &values).expect("well typed");
    builder.output(result).expect("the result is a value");
    let program = builder.finish();
    evaluate_seeded(&program, &host, &BTreeMap::new(), &BTreeMap::new(), 0)
        .expect("the CPU evaluates")[0]
        .data()
        .to_vec()
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
fn the_attention_kernels_compute_the_recorded_bits_under_prefix_masks() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let mut seen = Vec::new();
    let mut outside = Vec::new();
    for (b, s, h, heads, mask) in CASES {
        let mut draw = Draw::new(0x7072_6566_6978_6573 ^ s ^ (h << 16) ^ (b << 32));
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
        let keeps_a_key = inputs[3]
            .chunks(s as usize)
            .all(|keys| keys.iter().any(|&m| m != 0.0));
        let shapes = vec![vec![n, h], vec![n, h], vec![n, h], vec![b, s], vec![n, h]];
        let ops = [
            TensorOp::Attention { heads },
            TensorOp::AttentionGradQ { heads },
            TensorOp::AttentionGradK { heads },
            TensorOp::AttentionGradV { heads },
        ];
        let label = format!("[{b}, {s}] x {h} heads {heads} {mask:?}");
        let mut hashes = Vec::new();
        for op in &ops {
            let kernel = kernel_of(op, &rows, &mask_ty);
            let out = run(&device, &[kernel], &inputs, 0);
            hashes.push(fnv(&out[1]));
            if n <= 2048 && keeps_a_key {
                let expected = cpu(op, &inputs, &shapes);
                let wrong = out[1]
                    .iter()
                    .zip(&expected)
                    .enumerate()
                    .filter(|(_, (g, c))| !within(**g, **c, c.abs(), GPU_ATOL, GPU_RTOL))
                    .take(2)
                    .map(|(i, (g, c))| format!("{} {label} [{i}]: gpu {g:?} cpu {c:?}", op.name()));
                outside.extend(wrong);
            }
        }
        let shared: Vec<Kernel> = ops
            .iter()
            .map(|op| with_softmax_stats(kernel_of(op, &rows, &mask_ty)).expect("shares"))
            .collect();
        let out = run(&device, &shared, &inputs, (n * heads * 3) as usize);
        hashes.extend(out.iter().map(|values| fnv(values)));
        let messages = vulkan.take_validation_messages();
        assert!(messages.is_empty(), "{label}: {messages:?}");
        seen.push(hashes);
    }
    assert!(outside.is_empty(), "{outside:#?}");
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
