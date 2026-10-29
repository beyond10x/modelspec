//! story:bert-vulkan-kernels-attention-loss-adamw on a Vulkan device, DESIGN.md § 9 gate 5:
//! `Attention`, `AttentionGradQ`, `AttentionGradK`, `AttentionGradV`, `CrossEntropy`,
//! `CrossEntropyGrad`, `Sqrt` and `Div` run once per shape and every element of their results is
//! within `GPU_ATOL + GPU_RTOL * |c|` of the CPU interpreter's `c` (`tests/common/mod.rs`), with
//! the guard words after the output unchanged: attention with `heads = 2` and a key mask that
//! masks some keys of every sequence, cross-entropy over up to 512 classes with some and with
//! all weights zero. Three AdamW steps of a loss whose gradient is the same at every step put the
//! parameters and both moments within the same rule of the CPU run's (coordinator decision D73,
//! 2026-10-25).
//!
//! Ignored by default: they create a Vulkan instance, which `task ci` never does. Run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_bert_attention_loss_adamw -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;

use common::{Draw, GPU_ATOL, GPU_RTOL, GUARD_WORDS, guard, guarded, within};
use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::compile_training;
use modelspec_core::ir::kernel::{Kernel, Operand};
use modelspec_core::ir::tensor::{Dim, TensorOp, TensorProgramBuilder, TensorType, ValueId};
use modelspec_core::schedule::{
    attention_loss_kernel, cross_entropy_grad_kernels, cross_entropy_kernels, schedule,
};
use modelspec_runtime::cpu::interpreter::evaluate_seeded;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::train::runner;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::stepwise::{StepwiseOptions, train_stepwise};
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

/// The bits each output element holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_dead;

/// An operand of a case: a buffer's host value, or an immediate compiled into the kernel.
#[derive(Clone)]
enum Data {
    F32(Vec<f32>),
    I32(Vec<i32>),
    Immediate(f32),
}

/// One kernel run: the op, its operands with their names and types, their values, and the
/// result type.
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

/// A `[b, s]` key mask of 0 and 1 in which key 0 of every sequence is kept and, of the others,
/// about one in three is masked, at other places in each sequence.
fn key_mask(b: u64, s: u64) -> Vec<f32> {
    (0..b * s)
        .map(|i| {
            let (seq, j) = (i / s, i % s);
            if j == 0 || (j * 7 + seq * 5) % 3 != 0 {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// The four attention cases of `b` sequences of `s` rows of width `h` in `heads` heads.
fn attention_cases(b: u64, s: u64, h: u64, heads: u64, draw: &mut Draw) -> Vec<Case> {
    let n = b * s;
    let rows = TensorType::f32(&[n, h]);
    let mask = TensorType::f32(&[b, s]);
    let scaled = |draw: &mut Draw, by: f32| -> Vec<f32> {
        draw.values(n * h).into_iter().map(|x| x * by).collect()
    };
    let q = ("q".to_owned(), rows.clone(), Data::F32(scaled(draw, 0.25)));
    let k = ("k".to_owned(), rows.clone(), Data::F32(scaled(draw, 0.25)));
    let v = ("v".to_owned(), rows.clone(), Data::F32(scaled(draw, 1.0)));
    let m = ("key_mask".to_owned(), mask, Data::F32(key_mask(b, s)));
    let g = ("g".to_owned(), rows.clone(), Data::F32(scaled(draw, 1.0)));
    let forward = vec![q, k, v, m];
    let mut backward = forward.clone();
    backward.push(g);
    let label = format!("[{b}, {s}] x {h} heads {heads}");
    [
        (TensorOp::Attention { heads }, forward),
        (TensorOp::AttentionGradQ { heads }, backward.clone()),
        (TensorOp::AttentionGradK { heads }, backward.clone()),
        (TensorOp::AttentionGradV { heads }, backward),
    ]
    .into_iter()
    .map(|(op, operands)| Case {
        label: label.clone(),
        op,
        operands,
        result: rows.clone(),
    })
    .collect()
}

/// The cross-entropy cases of `n` rows of `v` classes: the loss, its gradient with the backward
/// seed 1 as an immediate and with a seed of 0.37 in a rank-0 buffer. Every third weight is 0,
/// or every weight with `all_zero`.
fn cross_entropy_cases(n: u64, v: u64, all_zero: bool, draw: &mut Draw) -> Vec<Case> {
    let logits = (
        "logits".to_owned(),
        TensorType::f32(&[n, v]),
        Data::F32(draw.values(n * v)),
    );
    let labels = (
        "labels".to_owned(),
        TensorType::i32(&[n]),
        Data::I32((0..n).map(|i| ((i * 37 + 11) % v) as i32).collect()),
    );
    let weights = (
        "weights".to_owned(),
        TensorType::f32(&[n]),
        Data::F32(
            (0..n)
                .map(|i| {
                    if all_zero || i % 3 == 1 {
                        0.0
                    } else {
                        0.5 + (i % 5) as f32 * 0.25
                    }
                })
                .collect(),
        ),
    );
    let operands = vec![logits, labels, weights];
    let mut immediate = operands.clone();
    immediate.push((
        "seed".to_owned(),
        TensorType::f32_scalar(),
        Data::Immediate(1.0),
    ));
    let mut buffered = operands.clone();
    buffered.push((
        "g".to_owned(),
        TensorType::f32_scalar(),
        Data::F32(vec![0.37]),
    ));
    let label = format!("[{n}, {v}]{}", if all_zero { " zero weights" } else { "" });
    vec![
        Case {
            label: label.clone(),
            op: TensorOp::CrossEntropy,
            operands,
            result: TensorType::f32_scalar(),
        },
        Case {
            label: format!("{label} seed 1"),
            op: TensorOp::CrossEntropyGrad,
            operands: immediate,
            result: TensorType::f32(&[n, v]),
        },
        Case {
            label: format!("{label} seed 0.37"),
            op: TensorOp::CrossEntropyGrad,
            operands: buffered,
            result: TensorType::f32(&[n, v]),
        },
    ]
}

/// The `Sqrt` and `Div` cases of `shape`: the root of non-negative values, quotients of two
/// tensors and of a tensor by a rank-0 buffer, divisors away from 0.
fn optimizer_cases(shape: &[u64], draw: &mut Draw) -> Vec<Case> {
    let ty = TensorType::f32(shape);
    let count: u64 = shape.iter().product();
    let positive: Vec<f32> = draw.values(count).iter().map(|x| x * x * 1e-3).collect();
    let divisor: Vec<f32> = draw.values(count).iter().map(|x| x.abs() + 0.5).collect();
    let label = format!("{shape:?}");
    vec![
        Case {
            label: label.clone(),
            op: TensorOp::Sqrt,
            operands: vec![("x".to_owned(), ty.clone(), Data::F32(positive))],
            result: ty.clone(),
        },
        Case {
            label: label.clone(),
            op: TensorOp::Div,
            operands: vec![
                ("a".to_owned(), ty.clone(), Data::F32(draw.values(count))),
                ("b".to_owned(), ty.clone(), Data::F32(divisor)),
            ],
            result: ty.clone(),
        },
        Case {
            label: format!("{label} by a scalar"),
            op: TensorOp::Div,
            operands: vec![
                ("a".to_owned(), ty.clone(), Data::F32(draw.values(count))),
                (
                    "b".to_owned(),
                    TensorType::f32_scalar(),
                    Data::F32(vec![1.7]),
                ),
            ],
            result: ty,
        },
    ]
}

/// The CPU interpreter's result of `case`: a program of its one op over its operands as inputs,
/// an immediate as a `Scalar`.
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

/// Runs the kernels of `case` once on `device`: uploads its buffer operands and the guarded
/// output, dispatches, reads back, in one command buffer with one host wait. Two kernels run as
/// the step plan runs a two-stage work: stage 1 reads the first operands and writes an
/// intermediate buffer, and after a barrier stage 2 reads it, then the last operands its kernel
/// names. Returns the output with its guard words.
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

/// What one op showed over every shape.
#[derive(Default)]
struct Observed {
    kind: String,
    workgroup_size: u32,
    elements: usize,
    max_abs: f64,
    max_rel: f64,
    bits_equal: bool,
    outside: Vec<String>,
}

/// The attention shapes `[b, s, h, heads]`: the twin's, edges that do not fill a workgroup, and
/// 512 keys, the longest softmax of a device test.
const ATTENTION_SHAPES: [(u64, u64, u64, u64); 3] =
    [(2, 16, 32, 2), (3, 65, 8, 2), (1, 512, 16, 2)];

/// The cross-entropy shapes `[n, v]` and whether every weight is 0: the twin's, 512 classes,
/// and a batch of zero weights.
const CROSS_ENTROPY_SHAPES: [(u64, u64, bool); 4] = [
    (32, 300, false),
    (65, 512, false),
    (3, 5, false),
    (7, 512, true),
];

/// The optimizer shapes: unaligned vectors and the twin's token table.
const OPTIMIZER_SHAPES: [&[u64]; 3] = [&[1], &[65], &[300, 32]];

#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_attention_loss_and_optimizer_kernels_match_the_cpu_interpreter_within_tolerance() {
    let vulkan = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let mut draw = Draw::new(0x6174_7465_6e74_696f);
    let mut cases = Vec::new();
    for (b, s, h, heads) in ATTENTION_SHAPES {
        cases.extend(attention_cases(b, s, h, heads, &mut draw));
    }
    for (n, v, all_zero) in CROSS_ENTROPY_SHAPES {
        cases.extend(cross_entropy_cases(n, v, all_zero, &mut draw));
    }
    for shape in OPTIMIZER_SHAPES {
        cases.extend(optimizer_cases(shape, &mut draw));
    }
    let mut observed: Vec<(String, Observed)> = Vec::new();
    let mut messages = Vec::new();
    for case in &cases {
        let name = case.op.name().to_owned();
        let kernels = kernels_of(case);
        let kernel = kernels.last().expect("a kernel");
        let expected = cpu(case);
        let got = gpu(&device, case);
        let at = match observed.iter().position(|(op, _)| *op == name) {
            Some(at) => at,
            None => {
                observed.push((
                    name.clone(),
                    Observed {
                        kind: kernels
                            .iter()
                            .map(|k| k.kind.to_string())
                            .collect::<Vec<_>>()
                            .join(" then "),
                        workgroup_size: kernel.workgroup_size,
                        bits_equal: true,
                        ..Observed::default()
                    },
                ));
                observed.len() - 1
            }
        };
        let seen = &mut observed[at].1;
        let count = expected.len();
        seen.elements += count;
        for (i, (&g, &c)) in got.iter().zip(&expected).enumerate() {
            let error = (f64::from(g) - f64::from(c)).abs();
            if g.to_bits() != c.to_bits() {
                seen.bits_equal = false;
            }
            seen.max_abs = seen.max_abs.max(error);
            if c != 0.0 {
                seen.max_rel = seen.max_rel.max(error / f64::from(c).abs());
            }
            if !within(g, c, c.abs(), GPU_ATOL, GPU_RTOL) && seen.outside.len() < 4 {
                seen.outside
                    .push(format!("{name} {} [{i}]: gpu {g:?} cpu {c:?}", case.label));
            }
        }
        for i in 0..GUARD_WORDS {
            if got[count + i].to_bits() != guard(i) {
                seen.outside
                    .push(format!("{name} {}: guard word {i} changed", case.label));
            }
        }
        for message in vulkan.take_validation_messages() {
            messages.push(format!("{name} {}: {message}", case.label));
        }
    }
    eprintln!("op | kind | workgroup size | elements | max |g-c| | max rel | bits equal");
    for (op, seen) in &observed {
        eprintln!(
            "{op} | {} | {} | {} | {:.3e} | {:.3e} | {}",
            seen.kind,
            seen.workgroup_size,
            seen.elements,
            seen.max_abs,
            seen.max_rel,
            if seen.bits_equal { "yes" } else { "no" }
        );
    }
    let outside: Vec<&String> = observed.iter().flat_map(|(_, s)| &s.outside).collect();
    assert!(outside.is_empty(), "{outside:#?}");
    assert!(messages.is_empty(), "{messages:#?}");
    assert_eq!(observed.len(), 8);
}

/// A linear model whose loss `sum(prediction * y)` has the same gradient at every step:
/// `x^T y` for `w` and the column sums of `y` for `b`, on the one synthetic batch.
const IDENTICAL_GRADIENTS: &str = "model Lin {
    param w: Tensor<f32, [65, 3]> = xavier_uniform();
    param b: Tensor<f32, [3]> = zeros();

    fn forward(x: Tensor<f32, [8, 65]>) -> Tensor<f32, [8, 3]> {
        return add_rows(matmul(x, w), b);
    }
}

train Fit {
    model = Lin;
    input x: Tensor<f32, [8, 65]>;
    target y: Tensor<f32, [8, 3]>;
    loss = sum(mul(model.forward(x), y));
    optimizer = adamw(lr = 0.01, weight_decay = 0.1, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, warmup_steps = 1);
    steps = 3;
    seed = 11;
    device = cpu;
}
";

/// Scenario 3, AdamW: three steps on identical gradients, the update dispatched as the
/// `Optimizer` subgraph's kernels with `m` and `v` in device buffers kept across the steps and
/// the scheduled rate and bias corrections uploaded before each step: every parameter element,
/// `m` and `v` is within `GPU_ATOL + GPU_RTOL * |c|` of the CPU run's, and every loss too.
#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_adamw_steps_on_identical_gradients_match_the_cpu_parameters_and_moments() {
    let compiled = compile_training("lin.mspec", IDENTICAL_GRADIENTS).expect("the source compiles");
    let plan = schedule(
        &compiled.training,
        compiled.forward.prediction,
        &compiled.config,
    )
    .expect("the AdamW step schedules for Vulkan");
    let artifact = CompiledArtifact::new(
        Backend::Vulkan,
        IDENTICAL_GRADIENTS,
        &compiled.training,
        plan,
    );
    let cpu = runner::run(&compiled).expect("the CPU run finishes");

    let vulkan = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = candidates[0].profile.clone();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let (gpu, stats) = train_stepwise(
        &device,
        &profile,
        &negotiated,
        &compiled,
        &artifact,
        StepwiseOptions::default(),
        &mut |_| Ok(()),
    )
    .expect("the device run finishes");
    eprintln!("vulkan adamw: {stats}");

    let mut outside = Vec::new();
    let mut compared = 0;
    let mut check = |what: String, g: &[f32], c: &[f32]| {
        assert_eq!(g.len(), c.len(), "{what}");
        let mut max_abs = 0.0f64;
        for (i, (&g, &c)) in g.iter().zip(c).enumerate() {
            max_abs = max_abs.max((f64::from(g) - f64::from(c)).abs());
            if !within(g, c, c.abs(), GPU_ATOL, GPU_RTOL) && outside.len() < 8 {
                outside.push(format!("{what} [{i}]: gpu {g:?} cpu {c:?}"));
            }
            compared += 1;
        }
        eprintln!("{what}: {} elements, max |g-c| {max_abs:.3e}", g.len());
    };
    check("losses".into(), &gpu.losses, &cpu.losses);
    for (param, c) in &cpu.state.values {
        let g = &gpu.state.values[param];
        check(format!("parameter {param:?}"), g.data(), c.data());
    }
    assert_eq!(cpu.state.moments.len(), 4, "m and v of w and b");
    for (key, c) in &cpu.state.moments {
        let g = gpu
            .state
            .moments
            .get(key)
            .unwrap_or_else(|| panic!("the device run reads back {key:?}"));
        check(format!("moment {key:?}"), g.data(), c.data());
    }
    let messages = vulkan.take_validation_messages();
    assert!(outside.is_empty(), "{outside:#?}");
    assert!(messages.is_empty(), "{messages:#?}");
    assert_eq!(compared, 4 + 3 * (65 * 3 + 3));
    assert_eq!(stats.steps, 3);
}
