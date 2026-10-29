//! story:attention-fewer-query-rows on a Vulkan device: the four attention kernels over `q` of
//! `B * L` rows, `k` and `v` of `B * S` and a `[B, S]` key mask, `1 <= L <= S`. Query `i` of
//! sequence `b` is row `b * L + i`. The cases are the decider's event encoder read at one query
//! per sequence (256 sequences of 256 keys, `[256, 256]` queries of four heads of 64 columns)
//! under prefix masks, the last keys and every key but the middle one, its context encoder (128
//! sequences of 32 keys) at one query, 5 of 32 queries, 33 of 65 queries of a head of 4 columns
//! that fill no tile, one query of 512 keys, and BERT-tiny's 128 queries of 128 keys of two
//! heads (`L = S`).
//!
//! For each case the device's `Attention` and `AttentionGradQ` of the `L` queries of each
//! sequence are, bit for bit, rows `b * S .. b * S + L` of its result over all `S` queries,
//! the first `L` of which are those `L`, and its `AttentionGradK` and `AttentionGradV` are the
//! result over all `S` queries with `G` zero but in those rows, equal as values (the zeros of
//! `G` may turn a `-0.0` sum to `+0.0`, so the sign of a zero is not compared). Each kernel runs
//! alone, recomputing its softmax statistics, and the four run as one call sharing them, with
//! the bits of the alone results. Where every sequence keeps a key (the CPU refuses another)
//! and the case is small enough for the CPU, the results are within `GPU_ATOL + GPU_RTOL * |c|`
//! of the CPU interpreter's `c`. No case raises a validation message.
//!
//! Ignored by default: it creates a Vulkan instance, which `task ci` never does. Run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_attention_query_rows -- --ignored`.
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

const LENGTHS_256: &[u64] = &[0, 1, 3, 4, 5, 20, 31, 32, 33, 88, 128, 255, 256];
const LENGTHS_256_KEPT: &[u64] = &[1, 3, 4, 5, 20, 31, 32, 33, 88, 128, 255, 256];
const LENGTHS_32: &[u64] = &[0, 1, 4, 5, 17, 31, 32];
const LENGTHS_32_KEPT: &[u64] = &[1, 4, 5, 17, 31, 32];
const LENGTHS_128: &[u64] = &[128, 64, 7, 1, 127];
const LENGTHS_65: &[u64] = &[65, 33, 2];
const LENGTHS_512: &[u64] = &[300, 512];

/// The shapes `[b, s, h, heads, l]` and masks of the cases.
const CASES: [(u64, u64, u64, u64, u64, Mask); 11] = [
    (256, 256, 256, 4, 1, Mask::Prefix(LENGTHS_256)),
    (24, 256, 256, 4, 1, Mask::Prefix(LENGTHS_256_KEPT)),
    (6, 256, 256, 4, 1, Mask::Suffix(LENGTHS_256)),
    (2, 256, 256, 4, 1, Mask::AllButMiddle),
    (128, 32, 256, 4, 1, Mask::Prefix(LENGTHS_32)),
    (12, 32, 256, 4, 1, Mask::Prefix(LENGTHS_32_KEPT)),
    (7, 32, 256, 4, 5, Mask::Prefix(LENGTHS_32_KEPT)),
    (3, 32, 256, 4, 1, Mask::AllButMiddle),
    (3, 65, 8, 2, 33, Mask::Prefix(LENGTHS_65)),
    (2, 512, 16, 2, 1, Mask::Prefix(LENGTHS_512)),
    (5, 128, 128, 2, 128, Mask::Prefix(LENGTHS_128)),
];

/// The most multiply-adds `B * L * S * H` of a case the CPU computes.
const CPU_WORK: u64 = 1 << 25;

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

/// The arity of `op`: 4 for `Attention`, 5 for a gradient.
fn arity(op: &TensorOp) -> usize {
    if matches!(op, TensorOp::Attention { .. }) {
        4
    } else {
        5
    }
}

/// Whether `op` writes one row per query (`Attention`, `AttentionGradQ`) rather than per key.
fn per_query(op: &TensorOp) -> bool {
    matches!(
        op,
        TensorOp::Attention { .. } | TensorOp::AttentionGradQ { .. }
    )
}

/// The kernel of `op` over `q` and `g` of type `queries`, `k` and `v` of `keys` and the mask, as
/// the schedule builds it.
fn kernel_of(op: &TensorOp, queries: &TensorType, keys: &TensorType, mask: &TensorType) -> Kernel {
    let types = [queries, keys, keys, mask, queries];
    let operands = types[..arity(op)]
        .iter()
        .map(|&t| Operand::Buffer(t.clone()))
        .collect();
    let result = if per_query(op) { queries } else { keys };
    attention_loss_kernel(op, operands, result.clone()).expect("an attention kernel")
}

/// The CPU interpreter's result of `op` over the first `arity` of `inputs` (`q`, `k`, `v`, the
/// mask and `g`) of `shapes`.
fn cpu(op: &TensorOp, inputs: &[Vec<f32>], shapes: &[Vec<u64>]) -> Vec<f32> {
    let mut builder = TensorProgramBuilder::new();
    let mut values = Vec::new();
    let mut host = BTreeMap::new();
    for at in 0..arity(op) {
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

/// Runs `kernels` of `ops` in order in one recording, each bound to the inputs it reads (`q`,
/// `k`, `v`, the mask and `g` of `inputs`), the statistics buffer of `stats` values where it
/// shares them, and its own result; returns the statistics and every result.
fn run(
    device: &Device<'_>,
    ops: &[TensorOp],
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
    let mut lengths: Vec<usize> = inputs.iter().map(Vec::len).collect();
    lengths.push(stats.max(1));
    lengths.extend(ops.iter().map(|op| {
        if per_query(op) {
            inputs[0].len()
        } else {
            inputs[1].len()
        }
    }));
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

/// Rows `b * s .. b * s + l` of every sequence `b` of `x`, rows of `h` values.
fn first_rows(x: &[f32], s: u64, l: u64, h: u64) -> Vec<f32> {
    let (s, l, h) = (s as usize, l as usize, h as usize);
    x.chunks(s * h)
        .flat_map(|seq| seq[..l * h].to_vec())
        .collect()
}

/// `x` with every row past the first `l` of each sequence of `s` rows `0.0`.
fn only_first_rows(x: &[f32], s: u64, l: u64, h: u64) -> Vec<f32> {
    let (s, l, h) = (s as usize, l as usize, h as usize);
    x.chunks(h)
        .enumerate()
        .flat_map(|(row, values)| {
            if row % s < l {
                values.to_vec()
            } else {
                vec![0.0; h]
            }
        })
        .collect()
}

/// The first two indices at which `a` and `b` differ in bits, or with `zeros_alike` in value
/// (`+0.0` and `-0.0` alike), labelled.
fn differences(a: &[f32], b: &[f32], zeros_alike: bool, label: &str) -> Vec<String> {
    let mut out: Vec<String> = a
        .iter()
        .zip(b)
        .enumerate()
        .filter(|(_, (x, y))| {
            if zeros_alike {
                x != y && !(x.is_nan() && y.is_nan() && x.to_bits() == y.to_bits())
            } else {
                x.to_bits() != y.to_bits()
            }
        })
        .take(2)
        .map(|(i, (x, y))| format!("{label} [{i}]: {x:?} against {y:?}"))
        .collect();
    if a.len() != b.len() {
        out.push(format!("{label}: {} values against {}", a.len(), b.len()));
    }
    out
}

#[test]
#[ignore = "needs a Vulkan device"]
fn fewer_query_rows_compute_the_bits_of_those_rows_of_every_query() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let mut wrong = Vec::new();
    let mut compared_to_cpu = 0;
    for (b, s, h, heads, l, mask) in CASES {
        let mut draw = Draw::new(0x7172_6f77_7300_0000 ^ s ^ (h << 16) ^ (b << 32) ^ (l << 48));
        let (n, m) = (b * l, b * s);
        let queries = TensorType::f32(&[n, h]);
        let every_query = TensorType::f32(&[m, h]);
        let keys = TensorType::f32(&[m, h]);
        let mask_ty = TensorType::f32(&[b, s]);
        let scaled = |draw: &mut Draw, by: f32| -> Vec<f32> {
            draw.values(m * h).into_iter().map(|x| x * by).collect()
        };
        let q_all = scaled(&mut draw, 0.25);
        let k = scaled(&mut draw, 0.25);
        let v = scaled(&mut draw, 1.0);
        let g_all = scaled(&mut draw, 1.0);
        let key_mask = key_mask(b, s, mask);
        let inputs = vec![
            first_rows(&q_all, s, l, h),
            k.clone(),
            v.clone(),
            key_mask.clone(),
            first_rows(&g_all, s, l, h),
        ];
        let every = vec![
            q_all.clone(),
            k.clone(),
            v.clone(),
            key_mask.clone(),
            only_first_rows(&g_all, s, l, h),
        ];
        let keeps_a_key = key_mask
            .chunks(s as usize)
            .all(|keys| keys.iter().any(|&m| m != 0.0));
        let shapes = vec![vec![n, h], vec![m, h], vec![m, h], vec![b, s], vec![n, h]];
        let ops = [
            TensorOp::Attention { heads },
            TensorOp::AttentionGradQ { heads },
            TensorOp::AttentionGradK { heads },
            TensorOp::AttentionGradV { heads },
        ];
        let label = format!("[{b}, {s}] x {h} heads {heads} queries {l} {mask:?}");
        let mut alone = Vec::new();
        for op in &ops {
            let at = format!("{} {label}", op.name());
            let kernel = kernel_of(op, &queries, &keys, &mask_ty);
            let out = run(&device, std::slice::from_ref(op), &[kernel], &inputs, 0).remove(1);
            let full = kernel_of(op, &every_query, &keys, &mask_ty);
            let all = run(&device, std::slice::from_ref(op), &[full], &every, 0).remove(1);
            if per_query(op) {
                let rows = first_rows(&all, s, l, h);
                wrong.extend(differences(&out, &rows, false, &at));
            } else {
                wrong.extend(differences(&out, &all, true, &at));
            }
            if keeps_a_key && b * l * s * h <= CPU_WORK {
                let expected = cpu(op, &inputs, &shapes);
                wrong.extend(
                    out.iter()
                        .zip(&expected)
                        .enumerate()
                        .filter(|(_, (g, c))| !within(**g, **c, c.abs(), GPU_ATOL, GPU_RTOL))
                        .take(2)
                        .map(|(i, (g, c))| format!("{at} [{i}]: gpu {g:?} cpu {c:?}")),
                );
                compared_to_cpu += 1;
            }
            assert!(
                out.iter().any(|&x| x != 0.0) || !keeps_a_key,
                "{at}: all zero"
            );
            alone.push(out);
        }
        let shared: Vec<Kernel> = ops
            .iter()
            .map(|op| with_softmax_stats(kernel_of(op, &queries, &keys, &mask_ty)).expect("shares"))
            .collect();
        let out = run(&device, &ops, &shared, &inputs, (n * heads * 3) as usize);
        for ((op, shared), alone) in ops.iter().zip(&out[1..]).zip(&alone) {
            let at = format!("{} shared {label}", op.name());
            wrong.extend(differences(shared, alone, false, &at));
        }
        let messages = vulkan.take_validation_messages();
        assert!(messages.is_empty(), "{label}: {messages:?}");
    }
    assert!(compared_to_cpu >= 24, "{compared_to_cpu} CPU comparisons");
    assert!(wrong.is_empty(), "{wrong:#?}");
}
