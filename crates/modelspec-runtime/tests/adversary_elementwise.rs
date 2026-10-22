//! Adversary cases for story:m1-vulkan-elementwise.
//!
//! Mutants are applied to SPIR-V words inside these tests (one opcode swapped); no
//! implementation file is touched. Device cases are ignored by default, like the suite's:
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test adversary_elementwise -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;

use common::{Case, ElementwiseOp, Form, Reference, Tolerance, tolerance};
use modelspec_core::ir::tensor::{TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::interpreter::evaluate;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, ExecError, Vulkan};

const OP_U_LESS_THAN: u32 = 176;
const OP_U_LESS_THAN_EQUAL: u32 = 178;
const OP_F_ORD_GREATER_THAN: u32 = 186;
const OP_F_ORD_GREATER_THAN_EQUAL: u32 = 190;

const GUARD_WORDS: usize = 16;

fn guard(i: usize) -> u32 {
    0x5afe_0000 | i as u32
}

/// `words` with every instruction of opcode `from` turned into `to` (same operand layout).
fn mutate(words: &[u32], from: u32, to: u32) -> Vec<u32> {
    let mut out = words.to_vec();
    let mut at = 5;
    let mut hits = 0;
    while at < out.len() {
        let length = (out[at] >> 16) as usize;
        if out[at] & 0xffff == from {
            out[at] = (out[at] & 0xffff_0000) | to;
            hits += 1;
        }
        at += length.max(1);
    }
    assert!(hits > 0, "the module has opcode {from}");
    out
}

/// The suite's input draw: SplitMix64, values in [-4, 4), every seventh exactly zero.
struct Draw(u64);

impl Draw {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn values(&mut self, count: u64) -> Vec<f32> {
        (0..count)
            .map(|i| {
                if i % 7 == 3 {
                    0.0
                } else {
                    ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0
                }
            })
            .collect()
    }
}

fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

fn open() -> Vulkan {
    Vulkan::load().expect("the Vulkan loader opens")
}

fn device(vulkan: &Vulkan) -> Device<'_> {
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens")
}

/// The probed profile of device 0.
fn device_profile(vulkan: &Vulkan) -> modelspec_core::device::DeviceProfile {
    vulkan.candidates().expect("the devices are probed")[0]
        .profile
        .clone()
}

/// Uploads `uploads`, dispatches `words` with buffers `bound` (indices into `uploads`, the last
/// one written), downloads that last buffer whole: one command buffer, one wait.
fn run(
    device: &Device<'_>,
    words: &[u32],
    uploads: &[Vec<f32>],
    bound: &[usize],
    count: u32,
    workgroups: [u32; 3],
) -> Vec<f32> {
    let buffers: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|values| device.device_local_buffer(bytes_of(values.len())).unwrap())
        .collect();
    let mut staging: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|values| {
            let mut buffer = device.staging_buffer(bytes_of(values.len())).unwrap();
            buffer.write_f32(values).unwrap();
            buffer
        })
        .collect();
    let pipeline = device.pipeline(words, bound.len() as u32).unwrap();
    let refs: Vec<&Buffer<'_>> = bound.iter().map(|&i| &buffers[i]).collect();
    let out = *bound.last().unwrap();
    let mut recording = device.record().unwrap();
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).unwrap();
    }
    recording.barrier(Barrier::TransferToCompute);
    recording
        .dispatch(&pipeline, &refs, count, workgroups)
        .unwrap();
    recording.barrier(Barrier::ComputeToTransfer);
    recording.copy(&buffers[out], &staging[out]).unwrap();
    recording.barrier(Barrier::TransferToHost);
    device
        .submit(recording.finish().unwrap())
        .unwrap()
        .wait()
        .unwrap();
    staging[out].read_f32().unwrap()
}

fn case(op: ElementwiseOp, operands: Vec<Form>) -> Case {
    Case {
        name: format!("{op:?}"),
        op,
        operands,
    }
}

/// The CPU interpreter on tensors of `inputs.len()` operands of `n` elements each.
fn cpu(op: ElementwiseOp, inputs: &[Vec<f32>]) -> Vec<f32> {
    let n = inputs[0].len() as u64;
    let mut builder = TensorProgramBuilder::new();
    let mut ids = Vec::new();
    let mut tensors = BTreeMap::new();
    for (i, values) in inputs.iter().enumerate() {
        let name = format!("x{i}");
        ids.push(builder.input(name.clone(), TensorType::f32(&[n])).unwrap());
        tensors.insert(name, HostTensor::new(vec![n], values.clone()).unwrap());
    }
    let result = match op {
        ElementwiseOp::Add => builder.add(ids[0], ids[1]),
        ElementwiseOp::Sub | ElementwiseOp::Sgd => builder.sub(ids[0], ids[1]),
        ElementwiseOp::Mul => builder.mul(ids[0], ids[1]),
        ElementwiseOp::Relu => builder.relu(ids[0]),
        ElementwiseOp::ReluGrad => builder.relu_grad(ids[0], ids[1]),
        ElementwiseOp::Broadcast => unreachable!("not a tensor-tensor op"),
    }
    .unwrap();
    builder.output(result).unwrap();
    let program = builder.finish();
    evaluate(&program, &tensors, &BTreeMap::new()).unwrap()[0]
        .data()
        .to_vec()
}

fn outside(op: ElementwiseOp, got: &[f32], expected: &[f32]) -> Vec<String> {
    let tol = tolerance(op);
    expected
        .iter()
        .enumerate()
        .filter(|&(i, &c)| !tol.accepts(got[i], c))
        .take(4)
        .map(|(i, &c)| format!("[{i}] gpu {:?} cpu {c:?}", got[i]))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Tolerance lookup (no device).

#[test]
fn adversary_tolerance_accepts_applies_the_pair_the_operator_carries() {
    // A per-operator tolerance whose pair is zero must refuse any difference. `accepts` calls
    // `within`, which reads GPU_ATOL and GPU_RTOL directly, so the pair the lookup returns is
    // never applied.
    let exact = Tolerance {
        op: ElementwiseOp::Add,
        atol: 0.0,
        rtol: 0.0,
        reference: Reference::AbsCpu,
    };
    let g = 1.0f32 + f32::EPSILON;
    assert!(
        !exact.accepts(g, 1.0),
        "{exact} accepted gpu {g:?} against cpu 1.0: the operator's pair is ignored"
    );
}

// ---------------------------------------------------------------------------------------------
// Device: typed errors.

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary_pipeline_refuses_fewer_bindings_than_the_module_declares() {
    let vulkan = open();
    let device = device(&vulkan);
    let _ = vulkan.take_validation_messages();
    let kernel = common::kernel(
        &case(ElementwiseOp::Add, vec![Form::Tensor, Form::Tensor]),
        65,
    );
    let words = emit(&kernel).unwrap();
    // The module binds 0, 1 and 2; the layout is asked for 2.
    let made = device.pipeline(&words, 2);
    let messages = vulkan.take_validation_messages();
    assert!(
        made.is_err(),
        "a pipeline with 2 layout bindings for a 3-binding module was created; validation \
         messages: {messages:#?}"
    );
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary_dispatch_refuses_a_count_past_the_bound_buffers() {
    let vulkan = open();
    let device = device(&vulkan);
    let kernel = common::kernel(
        &case(ElementwiseOp::Add, vec![Form::Tensor, Form::Tensor]),
        64,
    );
    let pipeline = device.pipeline(&emit(&kernel).unwrap(), 3).unwrap();
    let local = || device.device_local_buffer(bytes_of(64)).unwrap();
    let (a, b, out) = (local(), local(), local());
    let mut recording = device.record().unwrap();
    // 128 elements over buffers of 64: every invocation from 64 on reads and writes past the end.
    // Never submitted.
    let recorded = recording.dispatch(&pipeline, &[&a, &b, &out], 128, [2, 1, 1]);
    assert!(
        recorded.is_err(),
        "a dispatch of 128 elements over 64-element buffers was recorded: {recorded:?}"
    );
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary_zero_byte_buffer_is_a_typed_error() {
    let vulkan = open();
    let device = device(&vulkan);
    assert!(matches!(
        device.device_local_buffer(0),
        Err(ExecError::ZeroSize)
    ));
    assert!(matches!(device.staging_buffer(0), Err(ExecError::ZeroSize)));
}

// ---------------------------------------------------------------------------------------------
// Device: the SGD guard words of the suite cannot see a bounds-check mutant.

/// The SGD update over `count`, in place, laid out as `vulkan_elementwise.rs` lays it out: the
/// parameter carries 16 guard words, the gradient is exactly `count` long. With
/// `guard_gradient`, the gradient also carries the guard words, bound whole. Returns how many
/// parameter guard words changed.
fn sgd_guards_changed(
    device: &Device<'_>,
    words: &[u32],
    count: u64,
    guard_gradient: bool,
) -> usize {
    let n = count as usize;
    let mut draw = Draw(0x5eed_0000 ^ count);
    let mut p = draw.values(count);
    let mut g = draw.values(count);
    p.extend((0..GUARD_WORDS).map(|i| f32::from_bits(guard(i))));
    if guard_gradient {
        g.extend((0..GUARD_WORDS).map(|i| f32::from_bits(guard(i))));
    }
    let groups = count.div_ceil(64) as u32;
    let got = run(
        device,
        words,
        &[p, g],
        &[0, 1, 0],
        count as u32,
        [groups, 1, 1],
    );
    (0..GUARD_WORDS)
        .filter(|&i| got[n + i].to_bits() != guard(i))
        .count()
}

fn sgd_words(count: u64) -> Vec<u32> {
    emit(&common::kernel(
        &case(ElementwiseOp::Sgd, vec![Form::Tensor, Form::Tensor]),
        count,
    ))
    .unwrap()
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary_suite_sgd_guard_words_catch_the_le_bounds_mutant() {
    let vulkan = open();
    let device = device(&vulkan);
    let mut missed = Vec::new();
    for count in [1u64, 63, 65, 4_097] {
        let mutant = mutate(&sgd_words(count), OP_U_LESS_THAN, OP_U_LESS_THAN_EQUAL);
        if sgd_guards_changed(&device, &mutant, count, true) == 0 {
            missed.push(count);
        }
    }
    assert!(
        missed.is_empty(),
        "with the bounds check as `index <= count`, the suite's SGD layout changed no guard word \
         at counts {missed:?}"
    );
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary_sgd_with_a_guarded_gradient_catches_the_le_bounds_mutant() {
    let vulkan = open();
    let device = device(&vulkan);
    // Not 64: a count that fills its workgroups has no invocation past the end to catch.
    for count in [1u64, 63, 65, 4_097] {
        let words = sgd_words(count);
        assert_eq!(
            sgd_guards_changed(&device, &words, count, true),
            0,
            "n={count}: the emitted SGD module changed a guard word"
        );
        let mutant = mutate(&words, OP_U_LESS_THAN, OP_U_LESS_THAN_EQUAL);
        assert!(
            sgd_guards_changed(&device, &mutant, count, true) > 0,
            "n={count}: the `<=` mutant changed no guard word even with a guarded gradient"
        );
    }
    assert!(vulkan.take_validation_messages().is_empty());
}

// ---------------------------------------------------------------------------------------------
// Device: relu-grad at exactly 0.

fn relu_grad_words(count: u64) -> Vec<u32> {
    emit(&common::kernel(
        &case(ElementwiseOp::ReluGrad, vec![Form::Tensor, Form::Tensor]),
        count,
    ))
    .unwrap()
}

fn relu_grad_outside(device: &Device<'_>, words: &[u32], x: Vec<f32>, g: Vec<f32>) -> Vec<String> {
    let n = x.len();
    let expected = cpu(ElementwiseOp::ReluGrad, &[x.clone(), g.clone()]);
    let out = vec![f32::from_bits(0x7fc0_dead); n];
    let groups = (n as u64).div_ceil(64) as u32;
    let got = run(
        device,
        words,
        &[x, g, out],
        &[0, 1, 2],
        n as u32,
        [groups, 1, 1],
    );
    outside(ElementwiseOp::ReluGrad, &got, &expected)
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary_suite_relu_grad_inputs_catch_the_ge_mutant() {
    let vulkan = open();
    let device = device(&vulkan);
    let count = 4_097;
    // The suite's generator, drawing both operands in the order the suite does.
    let mut draw = common::Draw::new(0x6d6f_6465_6c73_7065);
    let (x, g) = (draw.values(count), draw.values(count));
    let mutant = mutate(
        &relu_grad_words(count),
        OP_F_ORD_GREATER_THAN,
        OP_F_ORD_GREATER_THAN_EQUAL,
    );
    let found = relu_grad_outside(&device, &mutant, x, g);
    assert!(
        !found.is_empty(),
        "relu-grad as `x >= 0` passed every element of the suite's inputs: at every x == 0 the \
         suite's gradient is also 0"
    );
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary_relu_grad_at_zero_passes_no_gradient() {
    let vulkan = open();
    let device = device(&vulkan);
    let count = 65;
    let x: Vec<f32> = (0..count)
        .map(|i| match i % 3 {
            0 => 0.0,
            1 => -0.0,
            _ => 1.0 + i as f32,
        })
        .collect();
    let g: Vec<f32> = (0..count).map(|i| 1.5 + i as f32).collect();
    let words = relu_grad_words(count as u64);
    let real = relu_grad_outside(&device, &words, x.clone(), g.clone());
    assert!(real.is_empty(), "the emitted relu-grad: {real:#?}");
    let mutant = mutate(&words, OP_F_ORD_GREATER_THAN, OP_F_ORD_GREATER_THAN_EQUAL);
    assert!(
        !relu_grad_outside(&device, &mutant, x, g).is_empty(),
        "these inputs do not catch relu-grad as `x >= 0`"
    );
    assert!(vulkan.take_validation_messages().is_empty());
}

// ---------------------------------------------------------------------------------------------
// Device: non-finite and subnormal inputs against the CPU interpreter.

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary_nonfinite_inputs_match_the_cpu_interpreter() {
    let vulkan = open();
    let device = device(&vulkan);
    let specials = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE / 4.0,
        -f32::MIN_POSITIVE / 4.0,
        f32::MAX,
        f32::MIN,
        3.0,
    ];
    let mut a = Vec::new();
    let mut b = Vec::new();
    for &x in &specials {
        for &y in &specials {
            a.push(x);
            b.push(y);
        }
    }
    let n = a.len();
    let groups = (n as u64).div_ceil(64) as u32;
    let unwritten = vec![f32::from_bits(0x7fc0_dead); n];
    let mut wrong = Vec::new();
    for op in [
        ElementwiseOp::Add,
        ElementwiseOp::Sub,
        ElementwiseOp::Mul,
        ElementwiseOp::Relu,
        ElementwiseOp::ReluGrad,
        ElementwiseOp::Sgd,
    ] {
        let unary = op == ElementwiseOp::Relu;
        let forms = if unary {
            vec![Form::Tensor]
        } else {
            vec![Form::Tensor, Form::Tensor]
        };
        let words = emit(&common::kernel(&case(op, forms), n as u64)).unwrap();
        let inputs: Vec<Vec<f32>> = if unary {
            vec![a.clone()]
        } else {
            vec![a.clone(), b.clone()]
        };
        // DESIGN.md § 9 gate 5: on a device without denorm preservation the reference is the
        // CPU interpreter on the inputs with subnormals flushed to signed zero.
        let preserves = device_profile(&vulkan).shader_denorm_preserve_f32 == Some(true);
        let reference: Vec<Vec<f32>> = if preserves {
            inputs.clone()
        } else {
            inputs.iter().map(|v| common::flush_subnormals(v)).collect()
        };
        let expected = cpu(op, &reference);
        let got = if op == ElementwiseOp::Sgd {
            run(
                &device,
                &words,
                &inputs,
                &[0, 1, 0],
                n as u32,
                [groups, 1, 1],
            )
        } else {
            let mut uploads = inputs.clone();
            uploads.push(unwritten.clone());
            let bound: Vec<usize> = (0..uploads.len()).collect();
            run(&device, &words, &uploads, &bound, n as u32, [groups, 1, 1])
        };
        let tol = tolerance(op);
        for i in 0..n {
            let (g, c) = (got[i], expected[i]);
            let same = (g.is_nan() && c.is_nan())
                || (c.is_infinite() && g == c)
                || (c.is_finite() && g.is_finite() && tol.accepts(g, c));
            if !same {
                wrong.push(format!(
                    "{op:?}({:?}, {:?}): gpu {g:?} cpu {c:?}",
                    a[i],
                    if unary { f32::NAN } else { b[i] }
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{} mismatches: {wrong:#?}", wrong.len());
    assert!(vulkan.take_validation_messages().is_empty());
}

// ---------------------------------------------------------------------------------------------
// Device: the y dimension of the flat index, which no suite count reaches.

#[test]
#[ignore = "needs a Vulkan device"]
fn adversary_two_dimensional_workgroups_cover_every_element_once() {
    let vulkan = open();
    let device = device(&vulkan);
    let count = 300u64;
    let mut draw = Draw(0x2d2d);
    let (a, b) = (draw.values(count), draw.values(count));
    let words = emit(&common::kernel(
        &case(ElementwiseOp::Add, vec![Form::Tensor, Form::Tensor]),
        count,
    ))
    .unwrap();
    let expected = cpu(ElementwiseOp::Add, &[a.clone(), b.clone()]);
    let mut out = vec![f32::from_bits(0x7fc0_dead); count as usize];
    out.extend((0..GUARD_WORDS).map(|i| f32::from_bits(guard(i))));
    let got = run(
        &device,
        &words,
        &[a, b, out],
        &[0, 1, 2],
        count as u32,
        [3, 2, 1],
    );
    let bad = outside(ElementwiseOp::Add, &got, &expected);
    assert!(bad.is_empty(), "{bad:#?}");
    let n = count as usize;
    assert!((0..GUARD_WORDS).all(|i| got[n + i].to_bits() == guard(i)));
    assert!(vulkan.take_validation_messages().is_empty());
}
