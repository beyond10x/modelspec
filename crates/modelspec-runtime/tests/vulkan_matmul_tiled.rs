//! `story:m1-f-forward-matmul-tiled` on a Vulkan device: the tiled matmul module, with no
//! epilogue, with the `relu` epilogue, and with the `relu` epilogue that keeps its
//! pre-activation, at every shape of `common::MATMUL_SHAPES` (the MNIST shapes, shapes that do
//! not fill a tile, and shapes one past a tile in each of `m`, `k` and `n`, and `k = 1`). Each
//! written view matches the CPU interpreter: bit for bit when `negotiate` puts the device in
//! the bit-for-bit class, otherwise within the matmul tolerance of `tests/common/mod.rs` (the
//! `relu` moves no value further from the CPU's than its operand is). The guard words after each
//! view stay unchanged and a second run gives the same bits.
//!
//! Ignored by default: it creates a Vulkan instance, which `task ci` never does. Run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_matmul_tiled -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;

use common::{Draw, GUARD_WORDS, KernelOp, guard, guarded};
use modelspec_core::ir::kernel::{Epilogue, Kernel};
use modelspec_core::ir::tensor::{TensorProgramBuilder, TensorType};
use modelspec_runtime::cpu::interpreter::evaluate;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::emit_module;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::{NumericsClass, negotiate};
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

/// The bits each output element holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_dead;

/// The CPU interpreter's `matmul(a, b)` and `relu(matmul(a, b))`.
fn cpu(a: (&[u64], &[f32]), b: (&[u64], &[f32])) -> (Vec<f32>, Vec<f32>) {
    let mut builder = TensorProgramBuilder::new();
    let mut tensors = BTreeMap::new();
    let mut values = Vec::new();
    for (name, (shape, data)) in [("a", a), ("b", b)] {
        tensors.insert(
            name.to_owned(),
            HostTensor::new(shape.to_vec(), data.to_vec()).expect("the input has its shape"),
        );
        let ty = TensorType::f32(shape);
        values.push(builder.input(name, ty).expect("the input is new"));
    }
    let product = builder.matmul(values[0], values[1]).expect("well typed");
    let activation = builder.relu(product).expect("well typed");
    builder.output(product).expect("a value");
    builder.output(activation).expect("a value");
    let outputs = evaluate(&builder.finish(), &tensors, &BTreeMap::new()).expect("it evaluates");
    (outputs[0].data().to_vec(), outputs[1].data().to_vec())
}

/// Per output element, the absolute sum of its `k` products.
fn terms(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f64> {
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            out.push(
                (0..k)
                    .map(|p| (f64::from(a[i * k + p]) * f64::from(b[p * n + j])).abs())
                    .sum(),
            );
        }
    }
    out
}

fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

/// Runs `kernel` once on the device over `a` and `b`, each written view guarded and filled with
/// [`UNWRITTEN`]; returns every written view whole, its guard words included, in binding order.
fn run(device: &Device<'_>, kernel: &Kernel, spirv: &[u32], a: &[f32], b: &[f32]) -> Vec<Vec<f32>> {
    let outputs = kernel.written();
    let n = (kernel.result.shape.iter())
        .map(|d| match d {
            modelspec_core::ir::tensor::Dim::Static(n) => *n as usize,
            modelspec_core::ir::tensor::Dim::Symbol(_) => unreachable!("static"),
        })
        .product::<usize>();
    let uploads: Vec<Vec<f32>> = [guarded(a), guarded(b)]
        .into_iter()
        .chain((0..outputs).map(|_| guarded(&vec![f32::from_bits(UNWRITTEN); n])))
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

#[test]
#[ignore = "needs a Vulkan device"]
fn the_tiled_matmul_with_and_without_its_relu_epilogue_matches_the_cpu() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        std::env::var(VALIDATION_ENV).as_deref() == Ok("1")
    );
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = candidates[0].profile.clone();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let class = negotiated.numerics;
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let flushes = profile.shader_denorm_preserve_f32 != Some(true);
    let tol = common::term_tolerance(KernelOp::MatMul);

    let mut draw = Draw::new(0x7469_6c65_646d_6d31);
    let mut wrong = Vec::new();
    let mut cases = 0;
    let mut elements = 0u64;
    let mut bit_checked = 0u64;
    let mut bit_equal = 0u64;
    for (m, k, n) in common::MATMUL_SHAPES {
        let a: Vec<f32> = (0..m).flat_map(|_| draw.values(k)).collect();
        let b = draw.values(k * n);
        let (ra, rb) = if flushes {
            (common::flush_subnormals(&a), common::flush_subnormals(&b))
        } else {
            (a.clone(), b.clone())
        };
        let (sum, activation) = cpu((&[m, k], &ra), (&[k, n], &rb));
        let abs_terms = terms(m as usize, k as usize, n as usize, &ra, &rb);
        for epilogue in [
            None,
            Some(Epilogue::Relu {
                writes_pre_activation: false,
            }),
            Some(Epilogue::Relu {
                writes_pre_activation: true,
            }),
        ] {
            let kernel = common::matmul_kernel_with(m, k, n, epilogue);
            let name = format!("[{m}, {k}] x [{k}, {n}] {epilogue:?}");
            let module = emit_module(&kernel, &profile).expect("the kernel emits");
            let first = run(&device, &kernel, &module.spirv, &a, &b);
            let second = run(&device, &kernel, &module.spirv, &a, &b);
            cases += 1;
            let expected: Vec<&Vec<f32>> = match epilogue {
                None => vec![&sum],
                Some(Epilogue::Relu {
                    writes_pre_activation: false,
                }) => vec![&activation],
                Some(Epilogue::Relu {
                    writes_pre_activation: true,
                }) => vec![&sum, &activation],
            };
            assert_eq!(first.len(), expected.len(), "{name}: one view per write");
            for (view, (got, want)) in first.iter().zip(&expected).enumerate() {
                let count = want.len();
                elements += count as u64;
                let mut here = 0;
                for i in 0..count {
                    if got[i].to_bits() == want[i].to_bits() {
                        bit_equal += 1;
                    }
                    let ok = if class == NumericsClass::BitForBit {
                        bit_checked += 1;
                        got[i].to_bits() == want[i].to_bits()
                    } else {
                        tol.accepts(got[i], want[i], abs_terms[i] as f32)
                    };
                    if !ok {
                        here += 1;
                        if here <= 4 {
                            wrong.push(format!(
                                "{name} view {view} [{i}]: gpu {:?} cpu {:?}",
                                got[i], want[i]
                            ));
                        }
                    }
                }
                let changed = (0..GUARD_WORDS)
                    .filter(|&i| got[count + i].to_bits() != guard(i))
                    .count();
                if changed > 0 {
                    wrong.push(format!("{name} view {view}: {changed} guard words changed"));
                }
            }
            let bits = |runs: &Vec<Vec<f32>>| -> Vec<u32> {
                runs.iter().flatten().map(|v| v.to_bits()).collect()
            };
            if bits(&first) != bits(&second) {
                wrong.push(format!("{name}: the second run differs"));
            }
        }
        for message in vulkan.take_validation_messages() {
            wrong.push(format!("[{m}, {k}] x [{k}, {n}]: {message}"));
        }
    }
    eprintln!(
        "vulkan_matmul_tiled: class {class}, {cases} cases run twice, {elements} elements, \
         {bit_checked} checked bit for bit, {bit_equal} bit-equal to the CPU, {} failures",
        wrong.len()
    );
    assert!(wrong.is_empty(), "{:#?}", &wrong[..wrong.len().min(12)]);
}
