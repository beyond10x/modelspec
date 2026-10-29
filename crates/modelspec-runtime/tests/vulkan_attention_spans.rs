//! story:attention-event-spans on a Vulkan device: the four attention kernels over an `i32
//! [U, 2]` operand of `(start, length)` spans in place of the `f32 [B, S]` key mask. Span `u` is
//! rows `start .. start + length` of `k` and `v`; with `q` of the rows of `k` query row `i`
//! attends to the keys of its span and a row in no span is `+0`, with `q` of `U` rows query `u`
//! attends to span `u`.
//!
//! The cases are the decider's event encoder on its compact layout (256 spans of 1 to 256 tokens
//! back to back in 32,768 rows, a few rows between some spans and many after the last, four heads
//! of 64 columns) read at every row and at one query per span, the same spans over fewer rows,
//! one span of 512 keys, and heads of 4 columns that fill no tile. For each case the device's
//! span result is compared with its own result on the padded layout under the prefix mask that
//! keeps the same tokens (sequence `u` of `S` rows, the first `length` kept): `Attention` and
//! `AttentionGradQ` in bits on every span row, `AttentionGradK` and `AttentionGradV` as values
//! under a `G` that is zero in the masked padded rows (a zero term may turn a `-0.0` sum into
//! `+0.0`), every row outside the spans `+0` in bits though `q`, `k`, `v` and `G` hold values
//! there and the result buffer starts as NaN. The four kernels sharing softmax statistics give the
//! bits of each alone, and where the CPU's work is small the results are within `GPU_ATOL +
//! GPU_RTOL * |c|` of the CPU interpreter's `c`. No case raises a validation message. Over
//! spans whose scores are `-0.0` then `+0.0`, the shared statistics hold the CPU's softmax
//! maximum in bits, the first key's `-0.0`.
//!
//! Ignored by default: it creates a Vulkan instance, which `task ci` never does. Run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_attention_spans -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;

use common::{Draw, GPU_ATOL, GPU_RTOL, within};
use modelspec_core::ir::kernel::{Kernel, Operand};
use modelspec_core::ir::tensor::{TensorOp, TensorProgramBuilder, TensorType};
use modelspec_core::schedule::{attention_loss_kernel, with_softmax_stats};
use modelspec_runtime::cpu::interpreter::evaluate_with_ids;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, Vulkan};

/// Span lengths of the decider-like cases, taken in turn.
const DECIDER: &[u64] = &[1, 3, 4, 256, 88, 2, 128, 33, 5, 20, 1, 255, 31, 32, 4, 1];
const LONG: &[u64] = &[300, 512, 1];
const NARROW: &[u64] = &[65, 33, 2, 1];

/// `U`, `S`, `H`, `heads`, `rows`, the span lengths and whether every row queries.
type Case = (u64, u64, u64, u64, u64, &'static [u64], bool);

/// The cases `[U, S, H, heads, rows, every row]`: `U` spans of the lengths, at most `S` each,
/// in `rows` rows; queries at every row, or one per span.
const CASES: [Case; 8] = [
    (256, 256, 256, 4, 32768, DECIDER, true),
    (256, 256, 256, 4, 32768, DECIDER, false),
    (16, 256, 256, 4, 1024, DECIDER, true),
    (16, 256, 256, 4, 1024, DECIDER, false),
    (128, 32, 256, 4, 4096, &[1, 4, 5, 17, 31, 32], true),
    (3, 512, 16, 2, 1000, LONG, true),
    (4, 65, 8, 2, 120, NARROW, true),
    (4, 65, 8, 2, 120, NARROW, false),
];

/// The most multiply-adds `sum(length^2) * H` of a case the CPU computes.
const CPU_WORK: u64 = 1 << 27;

/// The spans of a case: lengths in turn from `lengths`, back to back but for one row before the
/// first span and before every seventh, the rest of the `rows` after the last.
fn spans_of(u: u64, lengths: &[u64], rows: u64) -> Vec<(u64, u64)> {
    let mut at = 1;
    let spans: Vec<(u64, u64)> = (0..u)
        .map(|i| {
            if i % 7 == 6 {
                at += 1;
            }
            let length = lengths[i as usize % lengths.len()];
            let span = (at, length);
            at += length;
            span
        })
        .collect();
    assert!(at <= rows, "{at} rows of spans in {rows}");
    spans
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

/// The kernel of `op` over `q` and `g` of type `queries`, `k` and `v` of `keys` and the fourth
/// operand, as the schedule builds it.
fn kernel_of(
    op: &TensorOp,
    queries: &TensorType,
    keys: &TensorType,
    fourth: &TensorType,
) -> Kernel {
    let types = [queries, keys, keys, fourth, queries];
    let operands = types[..arity(op)]
        .iter()
        .map(|&t| Operand::Buffer(t.clone()))
        .collect();
    let result = if per_query(op) { queries } else { keys };
    attention_loss_kernel(op, operands, result.clone()).expect("an attention kernel")
}

/// The CPU interpreter's result of `op` over `q`, `k`, `v`, the spans and `g` of `inputs`.
fn cpu(op: &TensorOp, inputs: &[Vec<f32>], shapes: &[Vec<u64>], spans: &[i32]) -> Vec<f32> {
    let mut builder = TensorProgramBuilder::new();
    let mut values = Vec::new();
    let mut host = BTreeMap::new();
    let mut ids = BTreeMap::new();
    for at in 0..arity(op) {
        let name = format!("x{at}");
        if at == 3 {
            ids.insert(
                name.clone(),
                HostTensor::new(shapes[at].clone(), spans.to_vec()).expect("a tensor"),
            );
            values.push(
                builder
                    .input(name, TensorType::i32(&shapes[at]))
                    .expect("a new input"),
            );
            continue;
        }
        host.insert(
            name.clone(),
            HostTensor::new(shapes[at].clone(), inputs[at].clone()).expect("a tensor"),
        );
        values.push(
            builder
                .input(name, TensorType::f32(&shapes[at]))
                .expect("a new input"),
        );
    }
    let result = builder.push(op.clone(), &values).expect("well typed");
    builder.output(result).expect("the result is a value");
    let program = builder.finish();
    evaluate_with_ids(&program, &host, &ids, &BTreeMap::new()).expect("the CPU evaluates")[0]
        .data()
        .to_vec()
}

/// Runs `kernels` of `ops` in order in one recording, each bound to the inputs it reads (`q`,
/// `k`, `v`, the fourth operand and `g` of `inputs`), the statistics buffer of `stats` values
/// where it shares them, and its own result, every written buffer NaN before; returns the
/// statistics and every result.
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
fn spans_compute_the_bits_of_the_padded_prefix_mask() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let mut wrong = Vec::new();
    let mut compared_to_cpu = 0;
    for (u, s, h, heads, rows, lengths, every_row) in CASES {
        let label = format!(
            "[{u} spans of at most {s} in {rows}] x {h} heads {heads} every row {every_row}"
        );
        let spans = spans_of(u, lengths, rows);
        let hs = h as usize;
        let mut draw = Draw::new(0x7370_616e_7300_0000 ^ s ^ (h << 16) ^ (u << 32) ^ rows);
        let scaled = |draw: &mut Draw, n: u64, by: f32| -> Vec<f32> {
            draw.values(n * h).into_iter().map(|x| x * by).collect()
        };
        // The padded layout: sequence `u` of `s` rows, its span's tokens first.
        let (pq, pk, pv, pg) = (
            scaled(&mut draw, u * s, 0.25),
            scaled(&mut draw, u * s, 0.25),
            scaled(&mut draw, u * s, 1.0),
            scaled(&mut draw, u * s, 1.0),
        );
        let mask: Vec<f32> = spans
            .iter()
            .flat_map(|&(_, length)| (0..s).map(move |j| if j < length { 1.0 } else { 0.0 }))
            .collect();
        let padded_row = |span: usize, j: u64| span * s as usize + j as usize;
        // The span layout: the same tokens at their span rows, other values elsewhere.
        let place = |padded: &[f32], fill: Vec<f32>| -> Vec<f32> {
            let mut out = fill;
            for (i, &(start, length)) in spans.iter().enumerate() {
                for j in 0..length {
                    let (to, from) = ((start + j) as usize, padded_row(i, j));
                    out[to * hs..(to + 1) * hs]
                        .copy_from_slice(&padded[from * hs..(from + 1) * hs]);
                }
            }
            out
        };
        let (cq, ck, cv, cg) = (
            place(&pq, scaled(&mut draw, rows, 0.25)),
            place(&pk, scaled(&mut draw, rows, 0.25)),
            place(&pv, scaled(&mut draw, rows, 1.0)),
            place(&pg, scaled(&mut draw, rows, 1.0)),
        );
        let in_span: Vec<bool> = (0..rows)
            .map(|r| {
                spans
                    .iter()
                    .any(|&(start, length)| (start..start + length).contains(&r))
            })
            .collect();
        let span_words: Vec<f32> = spans
            .iter()
            .flat_map(|&(start, length)| [start as u32, length as u32])
            .map(f32::from_bits)
            .collect();
        let span_ids: Vec<i32> = spans
            .iter()
            .flat_map(|&(start, length)| [start as i32, length as i32])
            .collect();
        let pg0: Vec<f32> = pg
            .chunks(hs)
            .enumerate()
            .flat_map(|(r, row)| {
                if mask[r] != 0.0 {
                    row.to_vec()
                } else {
                    vec![0.0; hs]
                }
            })
            .collect();
        // One query per span: the span's first row, against the padded layout's first rows.
        let first = |x: &[f32], per: u64| -> Vec<f32> {
            x.chunks(hs * per as usize)
                .flat_map(|seq| seq[..hs].to_vec())
                .collect()
        };
        let span_first = |x: &[f32]| -> Vec<f32> {
            spans
                .iter()
                .flat_map(|&(start, _)| x[start as usize * hs..(start as usize + 1) * hs].to_vec())
                .collect()
        };
        let q_rows = if every_row { rows } else { u };
        let (span_q, span_g) = if every_row {
            (cq.clone(), cg.clone())
        } else {
            (span_first(&cq), span_first(&cg))
        };
        let (pad_q_rows, pad_q, pad_g) = if every_row {
            (u * s, pq.clone(), pg.clone())
        } else {
            (u, first(&pq, s), first(&pg, s))
        };
        let pad_g0 = if every_row {
            pg0.clone()
        } else {
            pad_g.clone()
        };
        let inputs = vec![span_q, ck.clone(), cv.clone(), span_words, span_g];
        let padded = |g: &Vec<f32>| {
            vec![
                pad_q.clone(),
                pk.clone(),
                pv.clone(),
                mask.clone(),
                g.clone(),
            ]
        };
        let queries = TensorType::f32(&[q_rows, h]);
        let keys = TensorType::f32(&[rows, h]);
        let span_ty = TensorType::i32(&[u, 2]);
        let pad_queries = TensorType::f32(&[pad_q_rows, h]);
        let pad_keys = TensorType::f32(&[u * s, h]);
        let mask_ty = TensorType::f32(&[u, s]);
        let ops = [
            TensorOp::Attention { heads },
            TensorOp::AttentionGradQ { heads },
            TensorOp::AttentionGradK { heads },
            TensorOp::AttentionGradV { heads },
        ];
        let mut alone = Vec::new();
        for op in &ops {
            let at = format!("{} {label}", op.name());
            let kernel = kernel_of(op, &queries, &keys, &span_ty);
            let out = run(&device, std::slice::from_ref(op), &[kernel], &inputs, 0).remove(1);
            let pad_kernel = kernel_of(op, &pad_queries, &pad_keys, &mask_ty);
            let g = if per_query(op) { &pad_g } else { &pad_g0 };
            let want = run(
                &device,
                std::slice::from_ref(op),
                &[pad_kernel],
                &padded(g),
                0,
            )
            .remove(1);
            let mut got_rows = Vec::new();
            let mut want_rows = Vec::new();
            let mut outside = Vec::new();
            if per_query(op) && !every_row {
                got_rows = out.clone();
                want_rows = want.clone();
            } else {
                for (i, &(start, length)) in spans.iter().enumerate() {
                    for j in 0..length {
                        let (row, from) = ((start + j) as usize, padded_row(i, j));
                        got_rows.extend_from_slice(&out[row * hs..(row + 1) * hs]);
                        want_rows.extend_from_slice(&want[from * hs..(from + 1) * hs]);
                    }
                }
                for (r, _) in in_span.iter().enumerate().filter(|(_, inside)| !**inside) {
                    outside.extend_from_slice(&out[r * hs..(r + 1) * hs]);
                }
            }
            wrong.extend(differences(&got_rows, &want_rows, !per_query(op), &at));
            wrong.extend(differences(
                &outside,
                &vec![0.0; outside.len()],
                false,
                &format!("{at} outside"),
            ));
            let work: u64 = spans.iter().map(|&(_, l)| l * l * h).sum();
            if work <= CPU_WORK {
                let shapes = vec![
                    vec![q_rows, h],
                    vec![rows, h],
                    vec![rows, h],
                    vec![u, 2],
                    vec![q_rows, h],
                ];
                let expected = cpu(op, &inputs, &shapes, &span_ids);
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
            assert!(out.iter().any(|&x| x != 0.0), "{at}: all zero");
            alone.push(out);
        }
        let shared: Vec<Kernel> = ops
            .iter()
            .map(|op| with_softmax_stats(kernel_of(op, &queries, &keys, &span_ty)).expect("shares"))
            .collect();
        let out = run(
            &device,
            &ops,
            &shared,
            &inputs,
            (q_rows * heads * 3) as usize,
        );
        for ((op, shared), alone) in ops.iter().zip(&out[1..]).zip(&alone) {
            let at = format!("{} shared {label}", op.name());
            wrong.extend(differences(shared, alone, false, &at));
        }
        let messages = vulkan.take_validation_messages();
        assert!(messages.is_empty(), "{label}: {messages:?}");
    }
    assert!(compared_to_cpu >= 16, "{compared_to_cpu} CPU comparisons");
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// The softmax maximum the CPU interpreter takes over the scores of a span's keys, in
/// ascending key order: the first, replaced by each later greater one
/// (`cpu::interpreter`, `Attend::softmax`), so scores `-0.0` then `+0.0` keep `-0.0`.
fn cpu_maximum(scores: impl Iterator<Item = f32>) -> f32 {
    scores
        .reduce(|m, x| if x > m { x } else { m })
        .expect("a span has a key")
}

/// The F1 case of `adversary_attention_tiled.rs`
/// (`softmax_maximum_of_zero_scores_is_the_first_keys_negative_zero`) over spans: every query
/// `-0.0`, the keys alternately `0.5` and `-0.5`, so a span's scores alternate `-0.0` and
/// `+0.0` from its first key. The statistics buffer of the four kernels sharing softmax
/// statistics holds the CPU interpreter's maximum in bits for every query of every span, at
/// every row and at one query per span: `-0.0` for the spans from an even row, `+0.0` for the
/// span from an odd row.
#[test]
#[ignore = "needs a Vulkan device"]
fn spans_softmax_maximum_of_zero_scores_is_the_first_keys_negative_zero() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let (rows, h, heads) = (16u64, 1u64, 1u64);
    let spans: [(u64, u64); 3] = [(0, 5), (6, 3), (9, 7)];
    let k: Vec<f32> = (0..rows)
        .map(|j| if j % 2 == 0 { 0.5 } else { -0.5 })
        .collect();
    let v: Vec<f32> = (0..rows).map(|j| j as f32).collect();
    let span_words: Vec<f32> = spans
        .iter()
        .flat_map(|&(start, length)| [start as u32, length as u32])
        .map(f32::from_bits)
        .collect();
    let u = spans.len() as u64;
    let ops = [
        TensorOp::Attention { heads },
        TensorOp::AttentionGradQ { heads },
        TensorOp::AttentionGradK { heads },
        TensorOp::AttentionGradV { heads },
    ];
    let mut wrong = Vec::new();
    for every_row in [true, false] {
        let q_rows = if every_row { rows } else { u };
        let q = vec![-0.0f32; q_rows as usize];
        let g = vec![1.0f32; q_rows as usize];
        let inputs = vec![q.clone(), k.clone(), v.clone(), span_words.clone(), g];
        let queries = TensorType::f32(&[q_rows, h]);
        let keys = TensorType::f32(&[rows, h]);
        let span_ty = TensorType::i32(&[u, 2]);
        let shared: Vec<Kernel> = ops
            .iter()
            .map(|op| with_softmax_stats(kernel_of(op, &queries, &keys, &span_ty)).expect("shares"))
            .collect();
        let out = run(
            &device,
            &ops,
            &shared,
            &inputs,
            (q_rows * heads * 3) as usize,
        );
        for (at, &(start, length)) in spans.iter().enumerate() {
            let query_rows: Vec<u64> = if every_row {
                (start..start + length).collect()
            } else {
                vec![at as u64]
            };
            for i in query_rows {
                let want = cpu_maximum(
                    (start..start + length).map(|j| q[i as usize] * k[j as usize] * 1.0),
                );
                let got = out[0][(i * heads * 3) as usize];
                if got.to_bits() != want.to_bits() {
                    wrong.push(format!(
                        "every row {every_row} span {at} query {i}: {:08x} against the CPU's {:08x}",
                        got.to_bits(),
                        want.to_bits()
                    ));
                }
            }
        }
        let messages = vulkan.take_validation_messages();
        assert!(messages.is_empty(), "every row {every_row}: {messages:?}");
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
