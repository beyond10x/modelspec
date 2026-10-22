//! DESIGN.md § 9 gates 5 and 6 for the M1 reduction, loss-sum, transpose and matmul kernels
//! (`story:m1-vulkan-reduce-matmul`): each module runs on a Vulkan device and matches the CPU
//! interpreter within the operator's tolerance (`tests/common/mod.rs`) at the MNIST shapes and at
//! shapes that do not fill a workgroup, gives the same bits on a second run, and, when `negotiate`
//! puts the device in the bit-for-bit class, gives the CPU's bits for every matmul output and
//! every `mse` loss sum before its division.
//!
//! The device test is ignored by default: it creates a Vulkan instance, which `task ci` never
//! does. Run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_reduce_matmul -- --ignored`;
//! with `MODELSPEC_VK_VALIDATION=1` every message of the Khronos validation layer fails it.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use common::{Draw, GUARD_WORDS, KernelOp, RUN_L2_RTOL, TermTolerance, guard, guarded};
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{BufferView, Kernel, KernelKind};
use modelspec_core::ir::tensor::{DType, TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::interpreter::evaluate;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::{KernelModule, emit_loss_sum_undivided, emit_module};
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::{NumericsClass, negotiate};
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

/// The bits each output element holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_dead;

/// One case: the kernels it dispatches (one, or the two stages of a reduction), the operator
/// whose tolerance it uses, and its input tensors with their shapes.
struct Case {
    name: String,
    op: KernelOp,
    kernels: Vec<Kernel>,
    inputs: Vec<(Vec<u64>, Vec<f32>)>,
}

/// The CPU interpreter's result of `op` on `inputs`.
fn cpu(op: KernelOp, inputs: &[(Vec<u64>, Vec<f32>)]) -> Vec<f32> {
    let mut builder = TensorProgramBuilder::new();
    let mut tensors = BTreeMap::new();
    let mut values = Vec::new();
    for (i, (shape, data)) in inputs.iter().enumerate() {
        let name = format!("x{i}");
        let ty = TensorType::f32(shape);
        tensors.insert(
            name.clone(),
            HostTensor::new(shape.clone(), data.clone()).expect("the input has its shape"),
        );
        values.push(builder.input(name, ty).expect("the input is new"));
    }
    let result = match op {
        KernelOp::ReduceSum => builder.reduce_sum(values[0]),
        KernelOp::ReduceMean => builder.reduce_mean(values[0]),
        KernelOp::Mse => builder.mse(values[0], values[1]),
        KernelOp::LossSum => {
            let d = builder.sub(values[0], values[1]).expect("same shapes");
            let square = builder.mul(d, d).expect("same shapes");
            builder.reduce_sum(square)
        }
        KernelOp::Transpose => builder.transpose(values[0]),
        KernelOp::MatMul => builder.matmul(values[0], values[1]),
    }
    .expect("the case is well typed");
    builder.output(result).expect("the result is a value");
    let program = builder.finish();
    let outputs = evaluate(&program, &tensors, &BTreeMap::new()).expect("the CPU evaluates");
    outputs[0].data().to_vec()
}

/// Per output element, the absolute sum of its terms and their number (DESIGN.md § 9 gate 5).
fn terms(op: KernelOp, inputs: &[(Vec<u64>, Vec<f32>)]) -> Vec<(f64, u64)> {
    let data = |i: usize| -> &[f32] { &inputs[i].1 };
    match op {
        KernelOp::ReduceSum | KernelOp::ReduceMean => {
            let x = data(0);
            vec![(x.iter().map(|&v| f64::from(v).abs()).sum(), x.len() as u64)]
        }
        KernelOp::Mse | KernelOp::LossSum => {
            let (a, b) = (data(0), data(1));
            let squares = a.iter().zip(b).map(|(&a, &b)| {
                let d = f64::from(a) - f64::from(b);
                d * d
            });
            vec![(squares.sum(), a.len() as u64)]
        }
        KernelOp::Transpose => data(0).iter().map(|_| (0.0, 1)).collect(),
        KernelOp::MatMul => {
            let (m, k) = (inputs[0].0[0] as usize, inputs[0].0[1] as usize);
            let n = inputs[1].0[1] as usize;
            let (a, b) = (data(0), data(1));
            let mut out = Vec::with_capacity(m * n);
            for i in 0..m {
                for j in 0..n {
                    let sum: f64 = (0..k)
                        .map(|p| (f64::from(a[i * k + p]) * f64::from(b[p * n + j])).abs())
                        .sum();
                    out.push((sum, k as u64));
                }
            }
            out
        }
    }
}

/// Every case of acceptance 4, its inputs drawn from `draw`.
fn cases(draw: &mut Draw) -> Vec<Case> {
    let mut cases = Vec::new();
    for count in common::REDUCE_COUNTS {
        for (op, mean) in [(KernelOp::ReduceSum, false), (KernelOp::ReduceMean, true)] {
            let (stage1, stage2) = common::reduce_kernels(mean, count);
            cases.push(Case {
                name: format!("{op:?} n={count}"),
                op,
                kernels: vec![stage1, stage2],
                inputs: vec![(vec![count], draw.values(count))],
            });
        }
        let (a, b) = (draw.values(count), draw.values(count));
        for op in [KernelOp::Mse, KernelOp::LossSum] {
            cases.push(Case {
                name: format!("{op:?} n={count}"),
                op,
                kernels: vec![common::loss_sum_kernel(count)],
                inputs: vec![(vec![count], a.clone()), (vec![count], b.clone())],
            });
        }
    }
    for (rows, cols) in common::TRANSPOSE_SHAPES {
        cases.push(Case {
            name: format!("Transpose [{rows}, {cols}]"),
            op: KernelOp::Transpose,
            kernels: vec![common::transpose_kernel(rows, cols)],
            inputs: vec![(vec![rows, cols], draw.values(rows * cols))],
        });
    }
    for (m, k, n) in common::MATMUL_SHAPES {
        // One draw per row of `a`, so the exact zeros of `Draw` move one column on per row and
        // no column of `a` is all zeros when `k` is a multiple of 7 (784): an all-zero column
        // would hide a kernel that drops that `k`.
        let a: Vec<f32> = (0..m).flat_map(|_| draw.values(k)).collect();
        cases.push(Case {
            name: format!("MatMul [{m}, {k}] x [{k}, {n}]"),
            op: KernelOp::MatMul,
            kernels: vec![common::matmul_kernel(m, k, n)],
            inputs: vec![(vec![m, k], a), (vec![k, n], draw.values(k * n))],
        });
    }
    cases
}

/// The modules of `case` for `profile`: the plan's modules, or for [`KernelOp::LossSum`] the loss
/// sum built without its division.
fn modules(case: &Case, profile: &DeviceProfile) -> Vec<KernelModule> {
    case.kernels
        .iter()
        .map(|kernel| match case.op {
            KernelOp::LossSum => emit_loss_sum_undivided(kernel, profile),
            _ => emit_module(kernel, profile),
        })
        .map(|module| module.expect("the kernel emits"))
        .collect()
}

/// `words` as the bytes of a buffer.
fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

/// The partials view an `Allocate` step makes for a reduction's stage 1.
fn partials_view(stage1: &Kernel) -> BufferView {
    let count = stage1
        .result
        .shape
        .iter()
        .map(|dim| match dim {
            modelspec_core::ir::tensor::Dim::Static(n) => *n,
            modelspec_core::ir::tensor::Dim::Symbol(_) => unreachable!("a test type is static"),
        })
        .product::<u64>();
    BufferView {
        buffer: 0,
        dtype: DType::F32,
        element_count: count,
        offset: 0,
        size_bytes: count * 4,
        alignment: 4,
    }
}

/// Runs `case` once on the device: uploads its guarded inputs and the guarded output, dispatches
/// (a reduction as two dispatches with a barrier between them), reads back, in one command buffer
/// with one host wait. Returns the output buffer whole, its guard words included.
fn run(device: &Device<'_>, case: &Case, modules: &[KernelModule], outputs: usize) -> Vec<f32> {
    let uploads: Vec<Vec<f32>> = case
        .inputs
        .iter()
        .map(|(_, data)| guarded(data))
        .chain([guarded(&vec![f32::from_bits(UNWRITTEN); outputs])])
        .collect();
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
    let pipelines: Vec<_> = case
        .kernels
        .iter()
        .zip(modules)
        .map(|(kernel, module)| {
            device
                .kernel_pipeline(kernel, &module.spirv)
                .expect("the pipeline is created")
        })
        .collect();
    let partials = match case.kernels[0].kind {
        KernelKind::ReduceStage1 => Some(
            device
                .scratch_buffer(&partials_view(&case.kernels[0]))
                .expect("the partials buffer is allocated"),
        ),
        _ => None,
    };
    let output = buffers.len() - 1;

    let mut recording = device.record().expect("a recording starts");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("the upload is recorded");
    }
    recording.barrier(Barrier::TransferToCompute);
    match &partials {
        Some(partials) => recording
            .reduce(
                (&pipelines[0], &case.kernels[0]),
                (&pipelines[1], &case.kernels[1]),
                &buffers[0],
                partials,
                &buffers[output],
            )
            .expect("the reduction is recorded"),
        None => {
            let bound: Vec<&Buffer<'_>> = buffers.iter().collect();
            recording
                .dispatch_kernel(&pipelines[0], &case.kernels[0], &bound)
                .expect("the dispatch is recorded");
        }
    }
    recording.barrier(Barrier::ComputeToTransfer);
    recording
        .copy(&buffers[output], &staging[output])
        .expect("the download is recorded");
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    let before = device.host_waits();
    device
        .submit(recorded)
        .expect("the commands are submitted")
        .wait()
        .expect("the device finishes");
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

/// Whether `op`'s modules are in the M1 bit-for-bit class (DESIGN.md § 5.4).
fn bit_class(op: KernelOp) -> bool {
    matches!(op, KernelOp::MatMul | KernelOp::LossSum)
}

#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_reduce_transpose_matmul_match_the_cpu_twice_with_guard_words_unchanged() {
    let started = std::time::Instant::now();
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        validation_requested(),
        "the validation layer is on exactly when {VALIDATION_ENV}=1"
    );
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = candidates[0].profile.clone();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let class = negotiated.numerics;
    eprintln!("class: {class}");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let setup = vulkan.take_validation_messages();
    assert!(setup.is_empty(), "validation messages at setup: {setup:#?}");
    let flushes = profile.shader_denorm_preserve_f32 != Some(true);

    let mut draw = Draw::new(0x7265_6475_6365_6d6d);
    let cases = cases(&mut draw);
    let mut elements = 0u64;
    let mut outside = Vec::new();
    let mut outside_count = 0;
    let mut guards_changed = 0;
    let mut not_repeated = Vec::new();
    let mut bit_checked = 0u64;
    let mut bit_wrong = Vec::new();
    let mut mode_wrong = Vec::new();
    let mut messages = Vec::new();
    let mut runs = 0u64;
    let waits_before = device.host_waits();
    for case in &cases {
        let modules = modules(case, &profile);
        for (kernel, module) in case.kernels.iter().zip(&modules) {
            let declared = !common::words::float_control_modes(&module.spirv).is_empty();
            let wanted = bit_class(case.op) && class == NumericsClass::BitForBit;
            if declared != wanted {
                mode_wrong.push(format!(
                    "{} {}: float-control modes declared {declared}, class {class}",
                    case.name, kernel.kind
                ));
            }
        }
        let reference_inputs: Vec<(Vec<u64>, Vec<f32>)> = case
            .inputs
            .iter()
            .map(|(shape, data)| {
                let data = if flushes {
                    common::flush_subnormals(data)
                } else {
                    data.clone()
                };
                (shape.clone(), data)
            })
            .collect();
        let expected = cpu(case.op, &reference_inputs);
        let terms = terms(case.op, &reference_inputs);
        let n = expected.len();
        let first = run(&device, case, &modules, n);
        let second = run(&device, case, &modules, n);
        runs += 2;
        let tol: TermTolerance = common::term_tolerance(case.op);
        let mut wrong_here = 0;
        for i in 0..n {
            let (abs_terms, count) = terms[i];
            let r = tol.r(expected[i], abs_terms, count);
            if !tol.accepts(first[i], expected[i], r) {
                wrong_here += 1;
                if wrong_here <= 4 {
                    outside.push(format!(
                        "{} [{i}]: gpu {:?} cpu {:?} r {r:e} outside {tol}",
                        case.name, first[i], expected[i]
                    ));
                }
            }
        }
        outside_count += wrong_here;
        elements += n as u64;
        for output in [&first, &second] {
            let changed = (0..GUARD_WORDS)
                .filter(|&i| output[n + i].to_bits() != guard(i))
                .count();
            if changed > 0 {
                outside.push(format!("{}: {changed} guard words changed", case.name));
            }
            guards_changed += changed;
        }
        if first
            .iter()
            .map(|v| v.to_bits())
            .ne(second.iter().map(|v| v.to_bits()))
        {
            not_repeated.push(case.name.clone());
        }
        if class == NumericsClass::BitForBit && bit_class(case.op) {
            for i in 0..n {
                bit_checked += 1;
                if first[i].to_bits() != expected[i].to_bits() {
                    bit_wrong.push(format!(
                        "{} [{i}]: gpu {:?} cpu {:?}",
                        case.name, first[i], expected[i]
                    ));
                }
            }
        }
        for message in vulkan.take_validation_messages() {
            messages.push(format!("{}: {message}", case.name));
        }
    }
    let waits = device.host_waits() - waits_before;
    eprintln!(
        "vulkan_reduce_matmul: class: {class}, {} cases run twice, {elements} elements compared, \
         {outside_count} outside tolerance, {guards_changed} guard words changed, {} of {} cases \
         differ between the two runs, {bit_checked} elements checked bit for bit with {} \
         differing, {} float-control mode mismatches, validation {} with {} messages, {waits} host \
         waits, {:.2} s",
        cases.len(),
        not_repeated.len(),
        cases.len(),
        bit_wrong.len(),
        mode_wrong.len(),
        if vulkan.validation_enabled() {
            "on"
        } else {
            "off"
        },
        messages.len(),
        started.elapsed().as_secs_f64()
    );
    assert!(outside.is_empty(), "{outside:#?}");
    assert!(
        not_repeated.is_empty(),
        "not bit-equal twice: {not_repeated:#?}"
    );
    assert!(mode_wrong.is_empty(), "{mode_wrong:#?}");
    assert!(
        bit_wrong.is_empty(),
        "{:#?}",
        &bit_wrong[..bit_wrong.len().min(8)]
    );
    assert!(messages.is_empty(), "{messages:#?}");
    assert_eq!(waits, runs, "one host wait per run");
}

// Acceptance 1: DESIGN.md and the spec state the rule; run in `task ci`.

fn repository_file(path: &str) -> String {
    std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(path),
    )
    .unwrap_or_else(|error| panic!("{path} reads: {error}"))
}

/// The text of DESIGN.md section `heading` up to the next heading of the same or a higher level.
fn section<'a>(design: &'a str, heading: &str) -> &'a str {
    let at = design
        .find(&format!("\n{heading}"))
        .unwrap_or_else(|| panic!("DESIGN.md has {heading}"));
    let rest = &design[at + 1..];
    let level = heading.chars().take_while(|&c| c == '#').count();
    let end = rest[1..]
        .match_indices('\n')
        .map(|(i, _)| i + 2)
        .find(|&i| {
            let hashes = rest[i..].chars().take_while(|&c| c == '#').count();
            hashes > 0 && hashes <= level && rest[i + hashes..].starts_with(' ')
        })
        .unwrap_or(rest.len());
    &rest[..end]
}

#[test]
fn design_and_spec_state_the_reduction_rule_the_orders_and_the_bit_class() {
    let design = repository_file("DESIGN.md");
    let passes = section(&design, "### 3.3 ");
    assert!(
        !passes.contains("reassociate floating-point reductions in deterministic mode"),
        "§ 3.3 no longer forbids the two-stage reduction"
    );
    for phrase in [
        "an order fixed by the shape alone",
        "compile-time constant",
        "one admissible order",
    ] {
        assert!(passes.contains(phrase), "§ 3.3 states {phrase:?}");
    }
    let kernels = section(&design, "### 5.2 ");
    for phrase in [
        "64 consecutive elements",
        "pairwise tree",
        "`i + 64`",
        "no atomics",
        "`(a - b) * (a - b)`",
        "starting from the `k = 0` product",
        "no split-K",
    ] {
        assert!(kernels.contains(phrase), "§ 5.2 states {phrase:?}");
    }
    let classes = section(&design, "### 5.4 ");
    for phrase in [
        "In M1 the bit-for-bit class holds the `mse` loss sum before its division and every matmul output",
        "`RoundingModeRTE`, `DenormPreserve` and `SignedZeroInfNanPreserve`",
        "Two-stage reductions, the `mse` value after its division and every other M1 kernel",
    ] {
        assert!(classes.contains(phrase), "§ 5.4 states {phrase:?}");
    }
    let gate = design
        .lines()
        .find(|line| line.starts_with("5. Vulkan vs CPU"))
        .expect("DESIGN.md § 9 has gate 5");
    assert!(
        gate.contains("the sum of the absolute values of its terms"),
        "gate 5 states the r of a reduction and a matmul element: {gate}"
    );
    let bound = gate
        .split("`||g - c||_2 <= ")
        .nth(1)
        .and_then(|rest| rest.split(" * ||c||_2`").next())
        .expect("gate 5 states the whole-run bound");
    assert_eq!(bound.parse::<f32>().expect("a number"), RUN_L2_RTOL);
    assert!(gate.contains("§ 5.4 item 3"), "{gate}");

    let spec = repository_file("spec/domains/artifact.yaml");
    let entity = &spec[spec
        .find("- name: modelspec.artifact.CompiledArtifact")
        .expect("the spec has CompiledArtifact")..];
    let field = &entity[entity
        .find("- name: deterministic")
        .expect("CompiledArtifact has deterministic")..];
    let summary = field
        .lines()
        .find(|line| line.trim_start().starts_with("summary:"))
        .expect("deterministic has a summary");
    assert!(
        !summary.contains("no floating-point reduction is reassociated"),
        "{summary}"
    );
    for phrase in [
        "an order fixed by the shape alone",
        "compile-time constants",
        "one admissible order",
        "artifact hash",
    ] {
        assert!(
            summary.contains(phrase),
            "the spec summary states {phrase:?}: {summary}"
        );
    }
}

#[test]
fn the_whole_run_check_bounds_the_l2_error_relative_to_the_cpu_norm() {
    let c = [3.0f32, -4.0];
    assert!(common::run_within(&c, &c));
    let at_bound = [3.0 + 5.0 * RUN_L2_RTOL * 0.99, -4.0];
    assert!(common::run_within(&at_bound, &c));
    let past = [3.0 + 5.0 * RUN_L2_RTOL * 1.01, -4.0];
    assert!(!common::run_within(&past, &c));
    assert!(!common::run_within(&c[..1], &c));
    for op in [
        KernelOp::ReduceSum,
        KernelOp::ReduceMean,
        KernelOp::Mse,
        KernelOp::LossSum,
        KernelOp::Transpose,
        KernelOp::MatMul,
    ] {
        let tol = common::term_tolerance(op);
        assert_eq!((tol.atol, tol.rtol), (common::GPU_ATOL, common::GPU_RTOL));
        let r = tol.r(-2.0, 10.0, 4);
        let wanted = match op {
            KernelOp::Transpose => 2.0,
            KernelOp::ReduceMean | KernelOp::Mse => 2.5,
            _ => 10.0,
        };
        assert_eq!(r, wanted, "{op:?}");
    }
}
