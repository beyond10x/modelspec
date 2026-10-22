//! DESIGN.md § 9 gates 5 and 6 for the M1 elementwise, broadcast and SGD kernels: each module
//! the emitter builds runs on a Vulkan device and matches the CPU interpreter within the
//! operator's tolerance (`tests/common/mod.rs`), on sizes that do not fill a workgroup.
//!
//! The device tests are ignored by default: they create a Vulkan instance, which `task ci` never
//! does. Run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_elementwise -- --ignored`;
//! with `MODELSPEC_VK_VALIDATION=1` every message of the Khronos validation layer fails the case
//! that caused it.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use common::{
    Case, Draw, ElementwiseOp, Form, GPU_ATOL, GPU_RTOL, GUARD_WORDS, guard, guarded, tolerance,
};
use modelspec_core::ir::kernel::Kernel;
use modelspec_core::ir::tensor::{Dim, TensorProgramBuilder, ValueId};
use modelspec_runtime::cpu::interpreter::evaluate;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, ExecError, VALIDATION_ENV, Vulkan};

/// The element counts of every case: one, one short of a workgroup, one workgroup, one past it,
/// one past 64 workgroups, and the MNIST hidden activations of a 100-row batch (784 x 100).
const COUNTS: [u64; 6] = [1, 63, 64, 65, 4_097, 78_400];

/// The bits each output element holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_dead;

/// The CPU interpreter's result for `case` over `count` elements, with `data[i]` the value of
/// buffer operand `i`.
fn cpu(case: &Case, count: u64, data: &[Vec<f32>]) -> Vec<f32> {
    let mut builder = TensorProgramBuilder::new();
    let mut inputs = BTreeMap::new();
    let mut operands: Vec<ValueId> = Vec::new();
    let mut buffer = 0;
    for (i, &form) in case.operands.iter().enumerate() {
        let value = match common::operand_type(form, count) {
            Some(ty) => {
                let name = format!("x{i}");
                let shape: Vec<u64> = ty
                    .shape
                    .iter()
                    .map(|dim| match dim {
                        Dim::Static(n) => *n,
                        Dim::Symbol(_) => unreachable!("a test type is static"),
                    })
                    .collect();
                inputs.insert(
                    name.clone(),
                    HostTensor::new(shape, data[buffer].clone()).expect("the input has its shape"),
                );
                buffer += 1;
                builder.input(name, ty).expect("the input is new")
            }
            None => match form {
                Form::Immediate(value) => builder.scalar(value),
                _ => unreachable!("only an immediate has no type"),
            },
        };
        operands.push(value);
    }
    let result = match case.op {
        ElementwiseOp::Add => builder.add(operands[0], operands[1]),
        ElementwiseOp::Sub | ElementwiseOp::Sgd => builder.sub(operands[0], operands[1]),
        ElementwiseOp::Mul => builder.mul(operands[0], operands[1]),
        ElementwiseOp::Relu => builder.relu(operands[0]),
        ElementwiseOp::ReluGrad => builder.relu_grad(operands[0], operands[1]),
        ElementwiseOp::Broadcast => builder.broadcast_scalar(operands[0], vec![Dim::Static(count)]),
    }
    .expect("the case is well typed");
    builder.output(result).expect("the result is a value");
    let program = builder.finish();
    let outputs = evaluate(&program, &inputs, &BTreeMap::new()).expect("the CPU evaluates");
    outputs[0].data().to_vec()
}

/// What one case left on the device.
struct Outcome {
    outside: Vec<String>,
    guards_changed: usize,
    elements: u64,
}

/// `words` as the bytes of a buffer.
fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

/// Runs `kernel` of `case` once: uploads its inputs and the guarded output, dispatches, reads
/// back, in one command buffer with one host wait.
fn run_case(
    device: &Device<'_>,
    case: &Case,
    kernel: &Kernel,
    count: u64,
    draw: &mut Draw,
) -> Outcome {
    let n = count as usize;
    let data: Vec<Vec<f32>> = case
        .operands
        .iter()
        .filter_map(|&form| common::operand_type(form, count))
        .map(|ty| draw.values(if ty.rank() == 0 { 1 } else { count }))
        .collect();
    let expected = cpu(case, count, &data);
    let got = dispatch_case(device, case, kernel, count, &data);
    let tol = tolerance(case.op);
    let outside = (0..n)
        .filter(|&i| !tol.accepts(got[i], expected[i]))
        .take(4)
        .map(|i| {
            format!(
                "{} n={count} [{i}]: gpu {:?} cpu {:?} outside {tol}",
                case.name, got[i], expected[i]
            )
        })
        .collect();
    let guards_changed = (0..GUARD_WORDS)
        .filter(|&i| got[n + i].to_bits() != guard(i))
        .count();
    Outcome {
        outside,
        guards_changed,
        elements: count,
    }
}

/// Runs `kernel` of `case` once on the buffer operands `data`: uploads them and the guarded
/// output, dispatches, reads back, in one command buffer with one host wait. Returns the output
/// buffer whole, its guard words included.
fn dispatch_case(
    device: &Device<'_>,
    case: &Case,
    kernel: &Kernel,
    count: u64,
    data: &[Vec<f32>],
) -> Vec<f32> {
    let words = emit(kernel).expect("the kernel emits");
    let n = count as usize;
    let in_place = case.op == ElementwiseOp::Sgd;

    // One device buffer per buffer operand, then the output; an in-place SGD update writes its
    // parameter, operand 0, which carries the guard words. Its gradient carries them too, so a
    // read one past the end subtracts a guard word and changes the parameter's.
    let mut uploads: Vec<Vec<f32>> = data.to_vec();
    if in_place {
        uploads[0] = guarded(&uploads[0]);
        uploads[1] = guarded(&uploads[1]);
    } else {
        uploads.push(guarded(&vec![f32::from_bits(UNWRITTEN); n]));
    }
    let output = if in_place { 0 } else { uploads.len() - 1 };

    let buffers: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|values| {
            device
                .device_local_buffer(bytes_of(values.len()))
                .expect("a device buffer is allocated")
        })
        .collect();
    let mut staging: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|values| {
            let mut buffer = device
                .staging_buffer(bytes_of(values.len()))
                .expect("a staging buffer is allocated");
            buffer
                .write_f32(values)
                .expect("the staging buffer is written");
            buffer
        })
        .collect();
    let pipeline = device
        .kernel_pipeline(kernel, &words)
        .expect("the pipeline is created");

    let mut bound: Vec<&Buffer<'_>> = buffers[..data.len()].iter().collect();
    bound.push(&buffers[output]);

    let mut recording = device.record().expect("a recording starts");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("the upload is recorded");
    }
    recording.barrier(Barrier::TransferToCompute);
    recording
        .dispatch(&pipeline, &bound, count as u32, kernel.workgroups)
        .expect("the dispatch is recorded");
    recording.barrier(Barrier::ComputeToTransfer);
    recording
        .copy(&buffers[output], &staging[output])
        .expect("the download is recorded");
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    let before = device.host_waits();
    let submitted = device.submit(recorded).expect("the commands are submitted");
    submitted.wait().expect("the device finishes");
    assert_eq!(
        device.host_waits() - before,
        1,
        "{}: one host wait",
        case.name
    );

    staging[output].read_f32().expect("the output reads")
}

fn validation_requested() -> bool {
    std::env::var(VALIDATION_ENV).as_deref() == Ok("1")
}

/// Device 0, after negotiation, with the validation layer when [`VALIDATION_ENV`] asks for it.
fn open() -> Vulkan {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        validation_requested(),
        "the validation layer is on exactly when {VALIDATION_ENV}=1"
    );
    vulkan
}

#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_elementwise_kernels_match_the_cpu_interpreter_with_guard_words_unchanged() {
    let started = std::time::Instant::now();
    let vulkan = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let setup = vulkan.take_validation_messages();
    assert!(setup.is_empty(), "validation messages at setup: {setup:#?}");

    let mut draw = Draw::new(0x6d6f_6465_6c73_7065);
    let mut cases = 0;
    let mut elements = 0;
    let mut outside = Vec::new();
    let mut guards_changed = 0;
    let mut messages = Vec::new();
    let waits_before = device.host_waits();
    for count in COUNTS {
        for case in common::catalog() {
            let kernel = common::kernel(&case, count);
            let outcome = run_case(&device, &case, &kernel, count, &mut draw);
            cases += 1;
            elements += outcome.elements;
            outside.extend(outcome.outside);
            if outcome.guards_changed > 0 {
                outside.push(format!(
                    "{} n={count}: {} guard words changed",
                    case.name, outcome.guards_changed
                ));
            }
            guards_changed += outcome.guards_changed;
            for message in vulkan.take_validation_messages() {
                messages.push(format!("{} n={count}: {message}", case.name));
            }
        }
    }
    let waits = device.host_waits() - waits_before;
    eprintln!(
        "vulkan_elementwise: {cases} cases, {elements} elements, {} outside tolerance, \
         {guards_changed} guard words changed, validation {} with {} messages, {waits} host waits, \
         {:.2} s",
        outside.len(),
        if vulkan.validation_enabled() {
            "on"
        } else {
            "off"
        },
        messages.len(),
        started.elapsed().as_secs_f64()
    );
    assert!(outside.is_empty(), "{outside:#?}");
    assert!(messages.is_empty(), "{messages:#?}");
    assert_eq!(waits, cases, "one host wait per case");
}

/// Whether the device value `g` matches the reference `c` of `op`: both NaN, the same infinity,
/// a zero of the same sign as a zero reference, or finite and within the operator's tolerance.
fn matches_reference(op: ElementwiseOp, g: f32, c: f32) -> bool {
    if c.is_nan() || g.is_nan() {
        return c.is_nan() && g.is_nan();
    }
    if c.is_infinite() || g.is_infinite() {
        return g == c;
    }
    if c == 0.0 {
        return g == 0.0 && g.is_sign_negative() == c.is_sign_negative();
    }
    tolerance(op).accepts(g, c)
}

#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_subnormal_inputs_are_flushed_to_signed_zero() {
    let vulkan = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = &candidates[0].profile;
    let negotiated = negotiate(profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let flushes = profile.shader_denorm_preserve_f32 != Some(true);
    let sub = f32::MIN_POSITIVE / 4.0;
    // Each pair is one element; under the flush rule every one of them has a different result.
    let cases: [(ElementwiseOp, &[(f32, f32)]); 4] = [
        (
            ElementwiseOp::Mul,
            &[
                (sub, f32::INFINITY),
                (-sub, f32::INFINITY),
                (f32::NEG_INFINITY, sub),
            ],
        ),
        (
            ElementwiseOp::Add,
            &[(-sub, -0.0), (sub, 0.0), (-0.0, -sub)],
        ),
        (
            ElementwiseOp::ReluGrad,
            &[(sub, 1.5), (sub, -2.0), (sub, f32::MAX)],
        ),
        (ElementwiseOp::Sgd, &[(-0.0, sub), (0.0, -sub), (-sub, 0.0)]),
    ];
    let mut wrong = Vec::new();
    let mut differ = 0;
    let mut elements = 0;
    for (op, pairs) in cases {
        let count = pairs.len() as u64;
        let case = Case {
            name: format!("{op:?}"),
            op,
            operands: vec![Form::Tensor, Form::Tensor],
        };
        let data = vec![
            pairs.iter().map(|p| p.0).collect::<Vec<f32>>(),
            pairs.iter().map(|p| p.1).collect::<Vec<f32>>(),
        ];
        let flushed: Vec<Vec<f32>> = data.iter().map(|v| common::flush_subnormals(v)).collect();
        let preserved = cpu(&case, count, &data);
        let reference = if flushes {
            cpu(&case, count, &flushed)
        } else {
            preserved.clone()
        };
        let got = dispatch_case(&device, &case, &common::kernel(&case, count), count, &data);
        for (i, &(a, b)) in pairs.iter().enumerate() {
            elements += 1;
            if !matches_reference(op, preserved[i], reference[i]) {
                differ += 1;
            }
            if !matches_reference(op, got[i], reference[i]) {
                wrong.push(format!(
                    "{op:?}({a:?}, {b:?}): gpu {:?} reference {:?}",
                    got[i], reference[i]
                ));
            }
        }
    }
    let messages = vulkan.take_validation_messages();
    eprintln!(
        "vulkan_subnormal: shaderDenormPreserveFloat32 {:?}, {elements} elements against the {} \
         reference, {differ} differ from the unflushed one, {} wrong, {} validation messages",
        profile.shader_denorm_preserve_f32,
        if flushes { "flushed" } else { "unflushed" },
        wrong.len(),
        messages.len()
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
    assert!(messages.is_empty(), "{messages:#?}");
    if flushes {
        assert_eq!(differ, elements, "every case tells the flush rule apart");
    }
}

#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_dispatch_binds_one_element_only_where_the_kernel_has_a_rank0_operand() {
    let vulkan = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let case = Case {
        name: "add_tensor_rank0".to_owned(),
        op: ElementwiseOp::Add,
        operands: vec![Form::Tensor, Form::Rank0],
    };
    let kernel = common::kernel(&case, 65);
    let words = emit(&kernel).expect("the kernel emits");
    let (a, scalar, out) = (
        device.device_local_buffer(bytes_of(65)).unwrap(),
        device.device_local_buffer(bytes_of(1)).unwrap(),
        device.device_local_buffer(bytes_of(65)).unwrap(),
    );
    let ranked = device.kernel_pipeline(&kernel, &words).unwrap();
    assert_eq!(ranked.broadcast(), [false, true, false]);
    let untyped = device.pipeline(&words, 3).unwrap();
    let mut recording = device.record().unwrap();
    let accepted = recording.dispatch(&ranked, &[&a, &scalar, &out], 65, kernel.workgroups);
    let refused = recording.dispatch(&untyped, &[&a, &scalar, &out], 65, kernel.workgroups);
    let short = recording.dispatch(&ranked, &[&a, &scalar, &scalar], 65, kernel.workgroups);
    drop(recording);
    eprintln!(
        "vulkan_rank0_binding: kernel pipeline {accepted:?}, all-tensor pipeline {refused:?}, \
         one-element output {short:?}"
    );
    assert!(accepted.is_ok());
    assert!(matches!(
        refused,
        Err(ExecError::CountPastBuffer {
            binding: 1,
            needed: 65,
            elements: 1,
            ..
        })
    ));
    assert!(matches!(
        short,
        Err(ExecError::CountPastBuffer { binding: 2, .. })
    ));
}

/// Story m1-fuse-elementwise: the MNIST step scales its loss gradient by `1.0 * 0.002`, two
/// compile-time scalars. Before the fusion a rank-0 `elementwise Mul (1.0, 0.002)` kernel computed
/// it on the device; the schedule now folds it into its reader's immediate. The folded immediate
/// must be the bits that kernel leaves on this device.
#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_the_folded_scalar_is_the_product_the_device_computed() {
    use modelspec_core::artifact::Backend;
    use modelspec_core::compile::compile_artifact;
    use modelspec_core::ir::kernel::Operand;
    use modelspec_core::ir::tensor::TensorType;

    let vulkan = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let case = Case {
        name: "mul_immediate_immediate".to_owned(),
        op: ElementwiseOp::Mul,
        operands: vec![Form::Immediate(1.0), Form::Immediate(0.002)],
    };
    let mut kernel = common::kernel(&case, 1);
    kernel.result = TensorType::f32_scalar();
    kernel.id = Kernel::module_id(&kernel.signature());
    assert_eq!(
        kernel.id.to_string(),
        "5bb7932a-569c-85fe-8946-5a688a1299e5",
        "the module k5 of the unfused MNIST plan: {}",
        kernel.signature()
    );
    let got = dispatch_case(&device, &case, &kernel, 1, &[]);
    let messages = vulkan.take_validation_messages();

    let file = "examples/mnist_mlp.mspec";
    let source = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(file),
    )
    .expect("the example reads");
    let artifact = compile_artifact(file, &source, Backend::Vulkan).expect("the example compiles");
    let folded: Vec<u32> = artifact
        .execution
        .kernels
        .iter()
        .filter(|k| k.op == "Mul")
        .flat_map(|k| k.operands.iter())
        .filter_map(|o| match o {
            Operand::Immediate(v) => Some(v.to_bits()),
            Operand::Buffer(_) => None,
        })
        .collect();
    eprintln!(
        "vulkan_folded_scalar: device 1.0 * 0.002 = {:#010x}, MNIST Mul immediates {folded:08x?}, \
         {} validation messages",
        got[0].to_bits(),
        messages.len()
    );
    assert!(messages.is_empty(), "{messages:#?}");
    assert_eq!(got[0].to_bits(), 0x3B03_126F, "the device's product");
    assert_eq!(
        folded,
        vec![got[0].to_bits()],
        "the folded immediate is the device's product"
    );
    assert_eq!(
        (0..GUARD_WORDS)
            .filter(|&i| got[1 + i].to_bits() != guard(i))
            .count(),
        0,
        "the guard words are unchanged"
    );
}

/// `a + b`, then `relu`, then the SGD update of `p` by the result, in place.
fn chain_cpu(a: &[f32], b: &[f32], p: &[f32]) -> Vec<f32> {
    let n = a.len() as u64;
    let mut builder = TensorProgramBuilder::new();
    let ty = || modelspec_core::ir::tensor::TensorType::f32(&[n]);
    let (xa, xb, xp) = (
        builder.input("a", ty()).unwrap(),
        builder.input("b", ty()).unwrap(),
        builder.input("p", ty()).unwrap(),
    );
    let sum = builder.add(xa, xb).unwrap();
    let relu = builder.relu(sum).unwrap();
    let update = builder.sub(xp, relu).unwrap();
    builder.output(sum).unwrap();
    builder.output(relu).unwrap();
    builder.output(update).unwrap();
    let program = builder.finish();
    let tensor = |values: &[f32]| HostTensor::new(vec![n], values.to_vec()).unwrap();
    let inputs = BTreeMap::from([
        ("a".to_owned(), tensor(a)),
        ("b".to_owned(), tensor(b)),
        ("p".to_owned(), tensor(p)),
    ]);
    let outputs = evaluate(&program, &inputs, &BTreeMap::new()).unwrap();
    outputs[2].data().to_vec()
}

/// Records add, relu and SGD over `count` elements. With `wait_each`, each dispatch is its own
/// submission and wait; otherwise all three share one command buffer and one wait. Returns the
/// updated parameter and the host waits taken.
fn chain(
    device: &Device<'_>,
    count: u64,
    wait_each: bool,
    draw: &mut Draw,
) -> (Vec<f32>, Vec<f32>, u64) {
    let n = count as usize;
    let (a, b, p) = (draw.values(count), draw.values(count), draw.values(count));
    let expected = chain_cpu(&a, &b, &p);
    let case = |op, operands| Case {
        name: format!("{op:?}"),
        op,
        operands,
    };
    let add = common::kernel(
        &case(ElementwiseOp::Add, vec![Form::Tensor, Form::Tensor]),
        count,
    );
    let relu = common::kernel(&case(ElementwiseOp::Relu, vec![Form::Tensor]), count);
    let sgd = common::kernel(
        &case(ElementwiseOp::Sgd, vec![Form::Tensor, Form::Tensor]),
        count,
    );
    let pipelines: Vec<_> = [&add, &relu, &sgd]
        .iter()
        .map(|kernel| {
            device
                .kernel_pipeline(kernel, &emit(kernel).unwrap())
                .unwrap()
        })
        .collect();
    let bytes = bytes_of(n);
    let local = || device.device_local_buffer(bytes).unwrap();
    let (da, db, dsum, drelu, dp) = (local(), local(), local(), local(), local());
    let mut staged: Vec<Buffer<'_>> = Vec::new();
    for values in [&a, &b, &p] {
        let mut buffer = device.staging_buffer(bytes).unwrap();
        buffer.write_f32(values).unwrap();
        staged.push(buffer);
    }
    let before = device.host_waits();
    let steps: [(usize, Vec<&Buffer<'_>>); 3] = [
        (0, vec![&da, &db, &dsum]),
        (1, vec![&dsum, &drelu]),
        (2, vec![&dp, &drelu, &dp]),
    ];
    let mut recording = device.record().unwrap();
    for (from, to) in staged.iter().zip([&da, &db, &dp]) {
        recording.copy(from, to).unwrap();
    }
    recording.barrier(Barrier::TransferToCompute);
    for (at, (pipeline, buffers)) in steps.iter().enumerate() {
        let kernel = [&add, &relu, &sgd][*pipeline];
        if at > 0 {
            recording.barrier(Barrier::ComputeToCompute);
        }
        recording
            .dispatch(
                &pipelines[*pipeline],
                buffers,
                count as u32,
                kernel.workgroups,
            )
            .unwrap();
        if wait_each && at < 2 {
            let finished = std::mem::replace(&mut recording, device.record().unwrap());
            device
                .submit(finished.finish().unwrap())
                .unwrap()
                .wait()
                .unwrap();
        }
    }
    recording.barrier(Barrier::ComputeToTransfer);
    recording.copy(&dp, &staged[2]).unwrap();
    recording.barrier(Barrier::TransferToHost);
    device
        .submit(recording.finish().unwrap())
        .unwrap()
        .wait()
        .unwrap();
    let waits = device.host_waits() - before;
    (staged[2].read_f32().unwrap(), expected, waits)
}

#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_chained_add_relu_sgd_in_one_command_buffer_waits_once() {
    let vulkan = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let mut draw = Draw::new(0x6368_6169_6e00_0001);
    let mut waits = Vec::new();
    let mut messages = Vec::new();
    for count in COUNTS {
        for wait_each in [false, true] {
            let (got, expected, taken) = chain(&device, count, wait_each, &mut draw);
            let tol = tolerance(ElementwiseOp::Sgd);
            for (i, (&g, &c)) in got.iter().zip(&expected).enumerate() {
                assert!(
                    tol.accepts(g, c),
                    "n={count} [{i}]: gpu {g:?} cpu {c:?} outside {tol}"
                );
            }
            waits.push((count, wait_each, taken));
            messages.extend(vulkan.take_validation_messages());
        }
    }
    let one: Vec<u64> = waits.iter().filter(|w| !w.1).map(|w| w.2).collect();
    let each: Vec<u64> = waits.iter().filter(|w| w.1).map(|w| w.2).collect();
    eprintln!(
        "vulkan_chain: host waits per case {one:?} in one command buffer, {each:?} waiting per \
         dispatch; validation {} with {} messages",
        if vulkan.validation_enabled() {
            "on"
        } else {
            "off"
        },
        messages.len()
    );
    assert!(messages.is_empty(), "{messages:#?}");
    assert!(one.iter().all(|&w| w == 1), "{one:?}");
    assert!(each.iter().all(|&w| w == 3), "{each:?}");
}

/// The test files of the GPU tests: every file of `tests/` named `vulkan_*` or `spirv_*`.
fn gpu_test_files() -> Vec<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("tests/ lists")
        .map(|entry| entry.expect("an entry reads").path())
        .filter(|path| {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            (name.starts_with("vulkan_") || name.starts_with("spirv_")) && name.ends_with(".rs")
        })
        .collect();
    files.sort();
    files
}

/// The ways a Rust source spells `value`.
fn spellings(value: f32) -> Vec<String> {
    let exponent = format!("{value:e}");
    let (mantissa, power) = exponent.split_once('e').expect("an exponent form");
    vec![
        exponent.clone(),
        exponent.to_uppercase(),
        format!("{mantissa}.0e{power}"),
        format!(
            "{mantissa}e{}{:02}",
            &power[..1],
            power[1..].parse::<u32>().expect("a decimal exponent")
        ),
        format!("{value}"),
    ]
}

/// The value DESIGN.md § 9 gate 5 gives `name`, as written after `name = `.
fn design_value(design: &str, name: &str) -> f32 {
    let at = design
        .find(&format!("`{name} = "))
        .unwrap_or_else(|| panic!("DESIGN.md states `{name} = …`"));
    let rest = &design[at + name.len() + 4..];
    let end = rest.find('`').expect("the value closes its code span");
    rest[..end].parse().expect("the value is a number")
}

#[test]
fn tolerance_is_documented_in_design_gate_5_and_defined_once_in_the_tests() {
    let design =
        std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../DESIGN.md"))
            .expect("DESIGN.md reads");
    let gate = design
        .lines()
        .find(|line| line.starts_with("5. Vulkan vs CPU"))
        .expect("DESIGN.md § 9 has gate 5");
    assert_eq!(design_value(gate, "GPU_ATOL"), GPU_ATOL);
    assert_eq!(design_value(gate, "GPU_RTOL"), GPU_RTOL);
    assert!(gate.contains("r = |c|"), "gate 5 states r = |c|: {gate}");
    for kind in ["elementwise", "broadcast", "SGD"] {
        assert!(gate.contains(kind), "gate 5 names {kind}: {gate}");
    }
    assert!(
        gate.contains("story:m1-vulkan-reduce-matmul"),
        "gate 5 leaves reductions and matmul to story:m1-vulkan-reduce-matmul: {gate}"
    );

    let files = gpu_test_files();
    assert!(files.len() >= 2, "{files:?}");
    for file in files {
        let text = std::fs::read_to_string(&file).expect("the test file reads");
        for value in [GPU_ATOL, GPU_RTOL] {
            for spelled in spellings(value) {
                assert!(
                    !text.contains(&spelled),
                    "{} repeats {spelled}; name it through tests/common/mod.rs",
                    file.display()
                );
            }
        }
    }
    for op in [
        ElementwiseOp::Add,
        ElementwiseOp::Sub,
        ElementwiseOp::Mul,
        ElementwiseOp::Relu,
        ElementwiseOp::ReluGrad,
        ElementwiseOp::Broadcast,
        ElementwiseOp::Sgd,
    ] {
        let tol = tolerance(op);
        assert_eq!((tol.op, tol.atol, tol.rtol), (op, GPU_ATOL, GPU_RTOL));
        assert!(tol.accepts(1.0, 1.0));
        assert_eq!(tol.r(-3.0), 3.0);
    }
}
