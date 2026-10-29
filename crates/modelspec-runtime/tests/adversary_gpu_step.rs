//! Adversary pass 1 on story:bert-vulkan-kernels-attention-loss-adamw (coordinator decision D73):
//! the attention, cross-entropy, `Sqrt` and `Div` kernels on edges the unit's device test does
//! not draw, each against the CPU interpreter.
//!
//! Every case needs a Vulkan device. A case whose name carries `device` is ignored as the M1
//! device tests are ("needs a Vulkan device"), and so is a case whose name carries a finding id
//! (`f1`, ...), which was red on the unit's commit and is green after fix pass 1 (coordinator
//! decision D74).
//!
//! Run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test adversary_gpu_step -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;

use common::{Draw, GPU_ATOL, GPU_RTOL, GUARD_WORDS, guard, guarded, within};
use modelspec_core::ir::kernel::{Kernel, Operand};
use modelspec_core::ir::tensor::{Dim, TensorOp, TensorProgramBuilder, TensorType, ValueId};
use modelspec_core::schedule::{
    attention_loss_kernel, cross_entropy_grad_kernels, cross_entropy_kernels,
};
use modelspec_runtime::cpu::interpreter::evaluate_seeded;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

const UNWRITTEN: u32 = 0x7fc0_dead;

#[derive(Clone)]
enum Data {
    F32(Vec<f32>),
    I32(Vec<i32>),
    Immediate(f32),
}

struct Case {
    label: String,
    op: TensorOp,
    operands: Vec<(String, TensorType, Data)>,
    result: TensorType,
}

fn shape_of(ty: &TensorType) -> Vec<u64> {
    ty.shape
        .iter()
        .map(|dim| match dim {
            Dim::Static(n) => *n,
            Dim::Symbol(_) => unreachable!("a test type is static"),
        })
        .collect()
}

fn cpu(case: &Case) -> Vec<f32> {
    let mut builder = TensorProgramBuilder::new();
    let mut inputs = BTreeMap::new();
    let mut ids = BTreeMap::new();
    let mut values: Vec<ValueId> = Vec::new();
    for (name, ty, data) in &case.operands {
        let shape = shape_of(ty);
        let value = match data {
            Data::F32(v) => {
                inputs.insert(name.clone(), HostTensor::new(shape, v.clone()).unwrap());
                builder
                    .input(name.clone(), ty.clone())
                    .expect("a new input")
            }
            Data::I32(v) => {
                ids.insert(name.clone(), HostTensor::new(shape, v.clone()).unwrap());
                builder
                    .input(name.clone(), ty.clone())
                    .expect("a new input")
            }
            Data::Immediate(x) => builder.scalar(*x),
        };
        values.push(value);
    }
    let result = builder
        .push(case.op.clone(), &values)
        .expect("the case is well typed");
    builder.output(result).expect("the result is a value");
    let program = builder.finish();
    evaluate_seeded(&program, &inputs, &ids, &BTreeMap::new(), 0).expect("the CPU evaluates")[0]
        .data()
        .to_vec()
}

/// The kernels of `case`, as the schedule builds them: one, or for `CrossEntropy` the row
/// losses and their sum (story:vulkan-cross-entropy-sum-per-row).
fn kernels_of(case: &Case) -> Vec<Kernel> {
    let operands: Vec<Operand> = case
        .operands
        .iter()
        .map(|(_, ty, data)| match data {
            Data::Immediate(x) => Operand::Immediate(*x),
            _ => Operand::Buffer(ty.clone()),
        })
        .collect();
    let missing = format!("{} {} has a kernel", case.op.name(), case.label);
    if case.op == TensorOp::CrossEntropy {
        return Vec::from(cross_entropy_kernels(operands, case.result.clone()).expect(&missing));
    }
    if case.op == TensorOp::CrossEntropyGrad {
        let kernels = cross_entropy_grad_kernels(operands, case.result.clone());
        return Vec::from(kernels.expect(&missing));
    }
    vec![attention_loss_kernel(&case.op, operands, case.result.clone()).expect(&missing)]
}

fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

/// The kernels of `case` run once on `device`, its output followed by the guard words. Two
/// kernels run as the step plan runs a two-stage work: stage 1 reads the first operands and
/// writes an intermediate buffer, and after a barrier stage 2 reads it, then the last operands its
/// kernel names.
fn gpu(device: &Device<'_>, case: &Case) -> Vec<f32> {
    let kernels = kernels_of(case);
    let pipelines: Vec<_> = kernels
        .iter()
        .map(|kernel| {
            let words = emit(kernel).expect("the kernel emits");
            device
                .kernel_pipeline(kernel, &words)
                .expect("the pipeline is created")
        })
        .collect();
    let result: u64 = shape_of(&case.result).iter().product();
    let mut uploads: Vec<Data> = case
        .operands
        .iter()
        .filter(|(_, _, d)| !matches!(d, Data::Immediate(_)))
        .map(|(_, _, d)| d.clone())
        .collect();
    uploads.push(Data::F32(guarded(&vec![
        f32::from_bits(UNWRITTEN);
        result as usize
    ])));
    let lengths: Vec<usize> = uploads
        .iter()
        .map(|data| match data {
            Data::F32(v) => v.len(),
            Data::I32(v) => v.len(),
            Data::Immediate(_) => unreachable!("filtered above"),
        })
        .collect();
    let buffers: Vec<Buffer<'_>> = lengths
        .iter()
        .map(|&n| device.device_local_buffer(bytes_of(n)).expect("a buffer"))
        .collect();
    let mut staging: Vec<Buffer<'_>> = uploads
        .iter()
        .zip(&lengths)
        .map(|(data, &n)| {
            let mut buffer = device
                .staging_buffer(bytes_of(n))
                .expect("a staging buffer");
            match data {
                Data::F32(v) => buffer.write_f32(v),
                Data::I32(v) => buffer.write_i32(v),
                Data::Immediate(_) => unreachable!("filtered above"),
            }
            .expect("the staging buffer is written");
            buffer
        })
        .collect();
    let output = buffers.len() - 1;
    let intermediate = match &kernels[..] {
        [_] => None,
        [first, _] => {
            let n: u64 = shape_of(&first.result).iter().product();
            Some(
                device
                    .device_local_buffer(bytes_of(n as usize))
                    .expect("a buffer"),
            )
        }
        _ => panic!("one kernel or two stages"),
    };
    let mut recording = device.record().expect("a recording starts");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("the upload is recorded");
    }
    recording.barrier(Barrier::TransferToCompute);
    match (&kernels[..], &intermediate) {
        ([kernel], None) => {
            let bound: Vec<&Buffer<'_>> = buffers.iter().collect();
            recording
                .dispatch_kernel(&pipelines[0], kernel, &bound)
                .expect("the dispatch is recorded");
        }
        ([first, second], Some(between)) => {
            let reads = first
                .operands
                .iter()
                .filter(|o| matches!(o, Operand::Buffer(_)))
                .count();
            let bound: Vec<&Buffer<'_>> = buffers[..reads].iter().chain([between]).collect();
            recording
                .dispatch_kernel(&pipelines[0], first, &bound)
                .expect("stage 1 is recorded");
            recording.barrier(Barrier::ComputeToCompute);
            let rest = second
                .operands
                .iter()
                .filter(|o| matches!(o, Operand::Buffer(_)))
                .count()
                - 1;
            let bound: Vec<&Buffer<'_>> = [between]
                .into_iter()
                .chain(&buffers[output - rest..])
                .collect();
            recording
                .dispatch_kernel(&pipelines[1], second, &bound)
                .expect("stage 2 is recorded");
        }
        _ => unreachable!("an intermediate buffer exactly for two stages"),
    }
    recording.barrier(Barrier::ComputeToTransfer);
    recording
        .copy(&buffers[output], &staging[output])
        .expect("the download is recorded");
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    device
        .submit(recorded)
        .expect("submitted")
        .wait()
        .expect("the device finishes");
    staging[output].read_f32().expect("the output reads")
}

fn open() -> Vulkan {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        std::env::var(VALIDATION_ENV).as_deref() == Ok("1"),
        "the validation layer is on exactly when {VALIDATION_ENV}=1"
    );
    vulkan
}

/// Runs every case on device 0 and returns, per case, the CPU and the device output (guard
/// words checked and stripped), failing on any validation message.
fn run_all(cases: &[Case]) -> Vec<(Vec<f32>, Vec<f32>)> {
    let vulkan = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let mut results = Vec::new();
    let mut messages = Vec::new();
    for case in cases {
        let expected = cpu(case);
        let got = gpu(&device, case);
        let count = expected.len();
        for i in 0..GUARD_WORDS {
            assert_eq!(
                got[count + i].to_bits(),
                guard(i),
                "{} {}: guard word {i}",
                case.op.name(),
                case.label
            );
        }
        for message in vulkan.take_validation_messages() {
            messages.push(format!("{} {}: {message}", case.op.name(), case.label));
        }
        results.push((expected, got[..count].to_vec()));
    }
    assert!(messages.is_empty(), "{messages:#?}");
    results
}

/// The elements of each case outside the kernel tolerance, at most four per case, and the
/// largest `|g - c|` per op.
fn outside_tolerance(cases: &[Case], results: &[(Vec<f32>, Vec<f32>)]) -> Vec<String> {
    let mut outside = Vec::new();
    let mut worst: BTreeMap<&str, f64> = BTreeMap::new();
    for (case, (expected, got)) in cases.iter().zip(results) {
        let mut listed = 0;
        for (i, (&g, &c)) in got.iter().zip(expected).enumerate() {
            let entry = worst.entry(case.op.name()).or_default();
            *entry = entry.max((f64::from(g) - f64::from(c)).abs());
            if !within(g, c, c.abs(), GPU_ATOL, GPU_RTOL) && listed < 4 {
                outside.push(format!(
                    "{} {} [{i}]: gpu {g:?} cpu {c:?}",
                    case.op.name(),
                    case.label
                ));
                listed += 1;
            }
        }
    }
    for (op, max) in worst {
        eprintln!("{op}: max |g-c| {max:.3e}");
    }
    outside
}

/// The four attention cases over `q`, `k`, `v`, `g` and the key mask `mask` of `b` sequences of
/// `s` rows of width `h` in `heads` heads.
fn attention_cases(
    label: &str,
    (b, s, h, heads): (u64, u64, u64, u64),
    [q, k, v, g]: [Vec<f32>; 4],
    mask: Vec<f32>,
) -> Vec<Case> {
    let n = b * s;
    let rows = TensorType::f32(&[n, h]);
    let forward = vec![
        ("q".to_owned(), rows.clone(), Data::F32(q)),
        ("k".to_owned(), rows.clone(), Data::F32(k)),
        ("v".to_owned(), rows.clone(), Data::F32(v)),
        (
            "key_mask".to_owned(),
            TensorType::f32(&[b, s]),
            Data::F32(mask),
        ),
    ];
    let mut backward = forward.clone();
    backward.push(("g".to_owned(), rows.clone(), Data::F32(g)));
    [
        (TensorOp::Attention { heads }, forward),
        (TensorOp::AttentionGradQ { heads }, backward.clone()),
        (TensorOp::AttentionGradK { heads }, backward.clone()),
        (TensorOp::AttentionGradV { heads }, backward),
    ]
    .into_iter()
    .map(|(op, operands)| Case {
        label: format!("{label} [{b}, {s}] x {h} heads {heads}"),
        op,
        operands,
        result: rows.clone(),
    })
    .collect()
}

/// Random `q`, `k` scaled by 0.25 and `v`, `g` for `n` rows of width `h`.
fn draws(draw: &mut Draw, n: u64, h: u64) -> [Vec<f32>; 4] {
    let mut scaled =
        |by: f32| -> Vec<f32> { draw.values(n * h).into_iter().map(|x| x * by).collect() };
    [scaled(0.25), scaled(0.25), scaled(1.0), scaled(1.0)]
}

/// A `[b, s]` key mask from `keep(sequence, key)`.
fn mask(b: u64, s: u64, keep: impl Fn(u64, u64) -> bool) -> Vec<f32> {
    (0..b * s)
        .map(|i| if keep(i / s, i % s) { 1.0 } else { 0.0 })
        .collect()
}

/// Hypothesis 3: attention and its three gradients with a mask that keeps one key per sequence
/// (first, last, middle), a mask of all ones, sequences of different kept lengths in one batch,
/// one key per sequence (`S = 1`), heads of width 1, and 1, 4 and 8 heads, each within the
/// kernel tolerance of the CPU interpreter.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_gpu_step_device_attention_mask_and_head_edges() {
    let mut draw = Draw::new(0x6164_7631_6d61_736b);
    let mut cases = Vec::new();
    let (b, s, h) = (3u64, 16u64, 32u64);
    let only = |at: fn(u64, u64) -> u64| move |seq: u64, key: u64| key == at(seq, 16);
    let masks: Vec<(&str, Vec<f32>)> = vec![
        ("one key, the first", mask(b, s, only(|_, _| 0))),
        ("one key, the last", mask(b, s, only(|_, s| s - 1))),
        ("one key, the middle", mask(b, s, only(|seq, _| 5 + seq))),
        ("all ones", mask(b, s, |_, _| true)),
        (
            "prefixes of 1, 8, 16",
            mask(b, s, |seq, key| key < [1, 8, 16][seq as usize]),
        ),
        (
            "suffixes",
            mask(b, s, |seq, key| key >= [15, 8, 0][seq as usize]),
        ),
    ];
    for (label, m) in masks {
        for heads in [1, 2, 4, 8] {
            cases.extend(attention_cases(
                label,
                (b, s, h, heads),
                draws(&mut draw, b * s, h),
                m.clone(),
            ));
        }
    }
    cases.extend(attention_cases(
        "S = 1",
        (4, 1, 8, 2),
        draws(&mut draw, 4, 8),
        vec![1.0; 4],
    ));
    cases.extend(attention_cases(
        "heads of width 1",
        (2, 16, 4, 4),
        draws(&mut draw, 32, 4),
        mask(2, 16, |seq, key| (key + seq) % 3 != 1),
    ));
    let results = run_all(&cases);
    let outside = outside_tolerance(&cases, &results);
    assert!(outside.is_empty(), "{outside:#?}");
}

/// Hypothesis 1 and mutant (b): with `q = 0` every score is 0, `exp(0) = 1` and each
/// probability is `1 / kept`, exact for 16 and 8 kept keys, so the `Attention` output and the
/// `AttentionGradV` output are the probability times each value (gradient) row summed in the
/// CPU's ascending key (query) order, the first product rounded and each later one fused: the
/// device's bits equal the CPU's. A kernel that sums its keys or queries in another order
/// differs in the last bits.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_gpu_step_device_uniform_attention_is_bit_equal() {
    let mut draw = Draw::new(0x7569_6e69_666f_726d);
    let (b, s, h, heads) = (2u64, 16u64, 32u64, 2u64);
    let n = b * s;
    let [_, k, v, g] = draws(&mut draw, n, h);
    let q = vec![0.0; (n * h) as usize];
    let mut cases = Vec::new();
    for (label, m) in [
        ("all ones", mask(b, s, |_, _| true)),
        ("every other key", mask(b, s, |_, key| key % 2 == 0)),
    ] {
        let four = attention_cases(
            label,
            (b, s, h, heads),
            [q.clone(), k.clone(), v.clone(), g.clone()],
            m,
        );
        for case in four {
            if matches!(
                case.op,
                TensorOp::Attention { .. } | TensorOp::AttentionGradV { .. }
            ) {
                cases.push(case);
            }
        }
    }
    let results = run_all(&cases);
    let mut differ = Vec::new();
    for (case, (expected, got)) in cases.iter().zip(&results) {
        let bad = got
            .iter()
            .zip(expected)
            .filter(|(g, c)| g.to_bits() != c.to_bits())
            .count();
        eprintln!(
            "{} {}: {bad} of {} elements differ in bits",
            case.op.name(),
            case.label,
            got.len()
        );
        if bad > 0 {
            differ.push(format!("{} {}: {bad} elements", case.op.name(), case.label));
        }
    }
    assert!(differ.is_empty(), "{differ:#?}");
}

/// Hypothesis 3: cross-entropy and its gradient with logits that tie exactly in every row, the
/// labels at class 0 and class V - 1, a single weight that is not 0, fractional weights, and the
/// 8,192 classes of `examples/bert_tiny.mspec`, each within the kernel tolerance.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_gpu_step_device_cross_entropy_ties_labels_and_weights() {
    let mut draw = Draw::new(0x6164_7631_6365_6e74);
    let mut cases = Vec::new();
    let mut add =
        |label: &str, n: u64, v: u64, logits: Vec<f32>, labels: Vec<i32>, weights: Vec<f32>| {
            let operands = vec![
                (
                    "logits".to_owned(),
                    TensorType::f32(&[n, v]),
                    Data::F32(logits),
                ),
                (
                    "labels".to_owned(),
                    TensorType::i32(&[n]),
                    Data::I32(labels),
                ),
                (
                    "weights".to_owned(),
                    TensorType::f32(&[n]),
                    Data::F32(weights),
                ),
            ];
            let mut grad = operands.clone();
            grad.push((
                "seed".to_owned(),
                TensorType::f32_scalar(),
                Data::Immediate(1.0),
            ));
            cases.push(Case {
                label: label.to_owned(),
                op: TensorOp::CrossEntropy,
                operands,
                result: TensorType::f32_scalar(),
            });
            cases.push(Case {
                label: label.to_owned(),
                op: TensorOp::CrossEntropyGrad,
                operands: grad,
                result: TensorType::f32(&[n, v]),
            });
        };
    let (n, v) = (8u64, 64u64);
    add(
        "ties",
        n,
        v,
        vec![0.75; (n * v) as usize],
        (0..n as i32)
            .map(|i| if i % 2 == 0 { 0 } else { v as i32 - 1 })
            .collect(),
        vec![1.0; n as usize],
    );
    add(
        "labels 0 and V-1, one weight",
        n,
        v,
        draw.values(n * v),
        (0..n as i32)
            .map(|i| if i % 2 == 0 { 0 } else { v as i32 - 1 })
            .collect(),
        (0..n).map(|i| if i == 5 { 1.0 } else { 0.0 }).collect(),
    );
    add(
        "fractional weights",
        n,
        v,
        draw.values(n * v),
        (0..n as i32).map(|i| (i * 7) % v as i32).collect(),
        (0..n).map(|i| 0.125 * (i + 1) as f32).collect(),
    );
    let (n, v) = (64u64, 8192u64);
    add(
        "the bert_tiny vocabulary",
        n,
        v,
        draw.values(n * v),
        (0..n as i32).map(|i| (i * 131) % v as i32).collect(),
        (0..n).map(|i| if i % 7 == 0 { 0.0 } else { 1.0 }).collect(),
    );
    let results = run_all(&cases);
    let outside = outside_tolerance(&cases, &results);
    assert!(outside.is_empty(), "{outside:#?}");
}

/// The distance in units in the last place between two finite `f32` of one sign.
fn ulps(a: f32, b: f32) -> u32 {
    a.to_bits().abs_diff(b.to_bits())
}

/// Hypothesis 1: `Sqrt` and `Div` over operands from `1e-30` to `1e30` with results that are
/// normal and finite: every element within 1 ulp of the CPU's correctly rounded result, as the
/// emitter's documentation states; the count of elements that are not bit-equal is printed.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_gpu_step_device_sqrt_and_div_within_one_ulp_over_a_wide_range() {
    let count = 4096u64;
    let mut draw = Draw::new(0x7371_7274_6469_7631);
    let wide = |draw: &mut Draw| -> Vec<f32> {
        draw.values(count)
            .into_iter()
            .map(|x| {
                let e = (x * 7.5).round();
                let m = 1.0 + x.abs() / 4.0;
                m * 10f32.powf(e)
            })
            .collect()
    };
    let ty = TensorType::f32(&[count]);
    let roots = wide(&mut draw);
    let a = wide(&mut draw);
    let b = wide(&mut draw);
    let cases = vec![
        Case {
            label: "wide".into(),
            op: TensorOp::Sqrt,
            operands: vec![("x".into(), ty.clone(), Data::F32(roots))],
            result: ty.clone(),
        },
        Case {
            label: "wide".into(),
            op: TensorOp::Div,
            operands: vec![
                ("a".into(), ty.clone(), Data::F32(a)),
                ("b".into(), ty.clone(), Data::F32(b)),
            ],
            result: ty.clone(),
        },
    ];
    let results = run_all(&cases);
    let mut far = Vec::new();
    for (case, (expected, got)) in cases.iter().zip(&results) {
        let mut not_equal = 0;
        for (i, (&g, &c)) in got.iter().zip(expected).enumerate() {
            if !c.is_normal() {
                continue;
            }
            if g.to_bits() != c.to_bits() {
                not_equal += 1;
            }
            if g.is_sign_negative() != c.is_sign_negative() || ulps(g, c) > 1 {
                far.push(format!("{} [{i}]: gpu {g:?} cpu {c:?}", case.op.name()));
            }
        }
        eprintln!(
            "{}: {not_equal} of {count} elements not bit-equal",
            case.op.name()
        );
    }
    assert!(far.is_empty(), "{:#?}", &far[..far.len().min(8)]);
}

/// Finding f1: `Div` of `1e30 / 1e-10`, whose correctly rounded quotient overflows to `+inf`, is
/// NaN on the device: the device's quotient `q = +inf` makes the residual `fma(-q, b, a) = -inf`,
/// which is not 0 or NaN, so the corrected `q + residual / b = inf - inf` is taken. Red on the
/// unit; the CPU gives `+inf`, and so does an uncorrected division. Fix pass 1 keeps a quotient
/// that is not finite as it is (coordinator decision D74).
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_gpu_step_f1_div_overflow_is_inf_not_nan() {
    let ty = TensorType::f32(&[4]);
    let cases = vec![Case {
        label: "overflow".into(),
        op: TensorOp::Div,
        operands: vec![
            (
                "a".into(),
                ty.clone(),
                Data::F32(vec![1e30, -1e30, 3e38, 1.0]),
            ),
            (
                "b".into(),
                ty.clone(),
                Data::F32(vec![1e-10, 1e-10, 0.5, 1e-39]),
            ),
        ],
        result: ty,
    }];
    let results = run_all(&cases);
    let (expected, got) = &results[0];
    eprintln!("cpu {expected:?}\ngpu {got:?}");
    for (i, (&g, &c)) in got.iter().zip(expected).enumerate().take(3) {
        assert_eq!(g.to_bits(), c.to_bits(), "[{i}]: gpu {g:?} cpu {c:?}");
    }
}
