//! `story:deterministic-fma`, acceptance `oracle-matches`, on a Vulkan device (DESIGN.md § 12
//! ADR-009, § 9 gate 5, bit-for-bit class of § 5.4): every matmul site and the `mse` loss sum
//! of the step plans of the MNIST example and of the `mlp_sweep` sources at b1000 h1024, b8192
//! h1024 and b8192 h4096 give the CPU oracle's bits, the kernels fusing each later product
//! into its sum with `Fma` and the oracle with `mul_add`.
//!
//! Each plan is taken as `compile_artifact` schedules it (tiles of 128 invocations) and as
//! scheduled for the profile of the device the test runs on (the RTX 3090 takes the 128 by 128
//! block of 256 invocations at three sites). A site's operands are drawn from a generator seeded
//! by the plan and the site, values of normal magnitude with signed zeros, so no operand,
//! product or partial sum is subnormal and the comparison is exact in any numerics class; an
//! operand the kernel reads transposed is bound as the row-major transpose of the drawn one.
//! The CPU side is `cpu::matmul` with the kernel this CPU runs, checked at sampled elements
//! against the written-out reference (`a[i, 0] * b[0, j]`, then `mul_add` per later `k`), and
//! the interpreter's `mse` against the same reference over the terms. A `relu` epilogue's
//! result is the compare and select of the CPU sum. The guard words after each view stay
//! unchanged.
//!
//! Ignored by default: it creates a Vulkan instance, which `task ci` never does. Run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test vulkan_fma_oracle -- --ignored`.
#![cfg(feature = "vulkan")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{compile_artifact, compile_training};
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{
    Epilogue, ExecutionPlan, Kernel, KernelKind, MatMulLimits, PlanOp,
};
use modelspec_core::ir::tensor::{TensorProgramBuilder, TensorType};
use modelspec_core::schedule::schedule_for;
use modelspec_runtime::cpu::matmul::{self, Lhs};
use modelspec_runtime::cpu::{HostTensor, evaluate};
use modelspec_runtime::spirv::emit::{emit_loss_sum_undivided, emit_module};
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

const GUARD_WORDS: usize = 16;

/// The bits each output element holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_dead;

/// Elements per site checked against the written-out reference.
const SAMPLES: u64 = 97;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The source `benches/mlp_sweep.rs` compiles at batch `batch` and hidden width `hidden`.
fn sweep(batch: u64, hidden: u64) -> String {
    format!(
        "model SweepMlp {{
    param w1: Tensor<f32, [784, {hidden}]> = xavier_uniform();
    param w2: Tensor<f32, [{hidden}, 10]> = xavier_uniform();

    fn forward(x: Tensor<f32, [{batch}, 784]>)
        -> Tensor<f32, [{batch}, 10]> {{
        return matmul(relu(matmul(x, w1)), w2);
    }}
}}

train Fit {{
    model = SweepMlp;
    input x: Tensor<f32, [{batch}, 784]>;
    target y: Tensor<f32, [{batch}, 10]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.001);
    steps = 220;
    seed = 42;
    device = vulkan(0);
}}
"
    )
}

fn sources() -> Vec<(String, String)> {
    let mnist = std::fs::read_to_string(repo_root().join("examples/mnist_mlp.mspec"))
        .expect("the example reads");
    let mut sources = vec![("mnist_mlp".to_owned(), mnist)];
    for (batch, hidden) in [(1000, 1024), (8192, 1024), (8192, 4096)] {
        sources.push((format!("sweep_b{batch}_h{hidden}"), sweep(batch, hidden)));
    }
    sources
}

/// The plan of `source` scheduled for `profile`.
fn scheduled(name: &str, source: &str, profile: &DeviceProfile) -> ExecutionPlan {
    let compiled = compile_training(name, source).unwrap_or_else(|d| panic!("{name}: {d:?}"));
    let plan = schedule_for(
        &compiled.training,
        compiled.forward.prediction,
        &compiled.config,
        Some(MatMulLimits::of(profile)),
    )
    .expect("the profile's schedule");
    CompiledArtifact::new(Backend::Vulkan, source, &compiled.training, plan).execution
}

/// The kernel of each dispatch of `plan`'s step plan of `kind`, in plan order.
fn sites(plan: &ExecutionPlan, kind: KernelKind) -> Vec<&Kernel> {
    plan.step
        .iter()
        .filter(|s| s.op == PlanOp::Dispatch)
        .map(|s| {
            plan.kernel(s.kernel_module_id.expect("a dispatch names its kernel"))
                .expect("the plan has the kernel")
        })
        .filter(|k| k.kind == kind)
        .collect()
}

/// SplitMix64 values in [-4, 4), every seventh element a zero of alternating sign.
fn values(seed: u64, count: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|i| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            match i % 14 {
                3 => 0.0,
                10 => -0.0,
                _ => ((z >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0,
            }
        })
        .collect()
}

/// A seed of the text `name`, the number `site` and the operand.
fn seed_of(name: &str, site: usize, operand: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.bytes().chain((site as u64).to_le_bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash ^ operand
}

/// The row-major `[cols, rows]` transpose of the row-major `[rows, cols]` `values`.
fn transpose(values: &[f32], rows: u64, cols: u64) -> Vec<f32> {
    let (rows, cols) = (rows as usize, cols as usize);
    let mut out = vec![0.0; values.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = values[r * cols + c];
        }
    }
    out
}

fn guard(i: usize) -> u32 {
    0x5afe_0000 | i as u32
}

fn guarded(values: &[f32]) -> Vec<f32> {
    let mut words = values.to_vec();
    words.extend((0..GUARD_WORDS).map(|i| f32::from_bits(guard(i))));
    words
}

fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

/// Runs `kernel` once over `inputs`, each written view guarded and filled with [`UNWRITTEN`];
/// returns every written view whole, its guard words included, in binding order.
fn run(
    device: &Device<'_>,
    kernel: &Kernel,
    spirv: &[u32],
    inputs: [&[f32]; 2],
    count: usize,
) -> Vec<Vec<f32>> {
    let uploads: Vec<Vec<f32>> = inputs
        .iter()
        .map(|values| guarded(values))
        .chain((0..kernel.written()).map(|_| guarded(&vec![f32::from_bits(UNWRITTEN); count])))
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

/// Element `(i, j)` of the written-out reference.
fn reference(a: &[f32], b: &[f32], (k, n): (usize, usize), i: usize, j: usize) -> f32 {
    let mut acc = a[i * k] * b[j];
    for p in 1..k {
        acc = a[i * k + p].mul_add(b[p * n + j], acc);
    }
    acc
}

/// The `relu` epilogue: `x` where `x > 0`, else `+0`.
fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

/// The views `kernel` writes from the sum `sum`, in binding order.
fn expected_views(kernel: &Kernel, sum: &[f32]) -> Vec<Vec<f32>> {
    match kernel.epilogue {
        None => vec![sum.to_vec()],
        Some(Epilogue::Relu {
            writes_pre_activation,
        }) => {
            let activation: Vec<f32> = sum.iter().map(|&x| relu(x)).collect();
            if writes_pre_activation {
                vec![sum.to_vec(), activation]
            } else {
                vec![activation]
            }
        }
    }
}

/// The interpreter's `mse` of `a` and `b` over `shape`.
fn cpu_mse(shape: &[u64], a: &[f32], b: &[f32]) -> f32 {
    let mut builder = TensorProgramBuilder::new();
    let x = builder.input("a", TensorType::f32(shape)).unwrap();
    let y = builder.input("b", TensorType::f32(shape)).unwrap();
    let z = builder.mse(x, y).unwrap();
    builder.output(z).unwrap();
    let program = builder.finish();
    let inputs: BTreeMap<String, HostTensor<f32>> = [("a", a), ("b", b)]
        .into_iter()
        .map(|(name, data)| {
            (
                name.to_owned(),
                HostTensor::new(shape.to_vec(), data.to_vec()).expect("a tensor"),
            )
        })
        .collect();
    evaluate(&program, &inputs, &BTreeMap::new()).expect("evaluates")[0].data()[0]
}

#[test]
#[ignore = "needs a Vulkan device"]
fn every_matmul_and_loss_sum_of_the_step_plans_gives_the_cpu_oracle_bits() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        std::env::var(VALIDATION_ENV).as_deref() == Ok("1")
    );
    let profile = vulkan.candidates().expect("the devices are probed")[0]
        .profile
        .clone();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let kernel_cpu = matmul::Kernel::detected();
    // The device against the CPU, and the CPU against the written-out reference.
    let (mut wrong, mut oracle_wrong) = (Vec::new(), Vec::new());
    let (mut elements, mut matmul_sites, mut blocks, mut losses) = (0u64, 0, 0, 0);
    for (name, source) in sources() {
        let plain = compile_artifact(&name, &source, Backend::Vulkan)
            .unwrap_or_else(|d| panic!("{name} compiles: {d:?}"))
            .execution;
        let fitted = scheduled(&name, &source, &profile);
        let plans = [("no profile", &plain), ("device profile", &fitted)];
        let counts: Vec<usize> = plans
            .iter()
            .map(|(_, p)| sites(p, KernelKind::MatMul).len())
            .collect();
        assert_eq!(
            counts[0], counts[1],
            "{name}: the schedules dispatch the same matmuls"
        );
        for site in 0..counts[0] {
            let first = sites(&plain, KernelKind::MatMul)[site];
            let (m, k, n) = Kernel::matmul_shape(&first.operands).expect("a static matmul");
            let a = values(seed_of(&name, site, 1), m * k);
            let b = values(seed_of(&name, site, 2), k * n);
            let sum = matmul::matmul(
                kernel_cpu,
                &a,
                Lhs::RowMajor,
                &b,
                m as usize,
                k as usize,
                n as usize,
            )
            .expect("the CPU kernel runs");
            let step = (m * n / SAMPLES).max(1);
            for at in (0..m * n).step_by(step as usize) {
                let (i, j) = ((at / n) as usize, (at % n) as usize);
                let want = reference(&a, &b, (k as usize, n as usize), i, j);
                if sum[at as usize].to_bits() != want.to_bits() {
                    oracle_wrong.push(format!(
                        "{name} matmul {site} [{m}, {k}] x [{k}, {n}]: CPU {kernel_cpu:?} \
                         element ({i}, {j}) {:e}, reference {want:e}",
                        sum[at as usize]
                    ));
                }
            }
            for (schedule, plan) in plans {
                let kernel = sites(plan, KernelKind::MatMul)[site];
                assert_eq!(
                    Kernel::matmul_shape(&kernel.operands),
                    Some((m, k, n)),
                    "{name} {schedule} site {site}"
                );
                blocks += usize::from(kernel.workgroup_size == 256);
                let [ta, tb] = kernel.transposed;
                let a_bound = if ta { transpose(&a, m, k) } else { a.clone() };
                let b_bound = if tb { transpose(&b, k, n) } else { b.clone() };
                let module = emit_module(kernel, &profile).expect("the kernel emits");
                let count = (m * n) as usize;
                let views = run(&device, kernel, &module.spirv, [&a_bound, &b_bound], count);
                let case = format!(
                    "{name} {schedule} matmul {site} [{m}, {k}] x [{k}, {n}] {:?} {:?} wg {}",
                    kernel.epilogue, kernel.transposed, kernel.workgroup_size
                );
                for (at, (view, want)) in views.iter().zip(expected_views(kernel, &sum)).enumerate()
                {
                    let changed = (0..GUARD_WORDS)
                        .filter(|&i| view[count + i].to_bits() != guard(i))
                        .count();
                    if changed > 0 {
                        wrong.push(format!("{case} view {at}: {changed} guard words changed"));
                    }
                    let differ: Vec<usize> = (0..count)
                        .filter(|&e| view[e].to_bits() != want[e].to_bits())
                        .collect();
                    if let Some(&e) = differ.first() {
                        wrong.push(format!(
                            "{case} view {at}: {} of {count} elements differ from the CPU, \
                             first ({}, {}) device {:e} CPU {:e}",
                            differ.len(),
                            e / n as usize,
                            e % n as usize,
                            view[e],
                            want[e]
                        ));
                    }
                    elements += count as u64;
                }
                for message in vulkan.take_validation_messages() {
                    wrong.push(format!("{case}: {message}"));
                }
                matmul_sites += 1;
            }
        }
        for (schedule, plan) in plans {
            for (site, kernel) in sites(plan, KernelKind::LossSum).into_iter().enumerate() {
                let shape = match &kernel.operands[0] {
                    modelspec_core::ir::kernel::Operand::Buffer(ty) => ty
                        .shape
                        .iter()
                        .map(|d| match d {
                            modelspec_core::ir::tensor::Dim::Static(n) => *n,
                            other => panic!("{name}: a static loss shape, got {other:?}"),
                        })
                        .collect::<Vec<u64>>(),
                    other => panic!("{name}: a buffer operand, got {other:?}"),
                };
                let count = shape.iter().product::<u64>();
                let a = values(seed_of(&name, 1000 + site, 1), count);
                let b = values(seed_of(&name, 1000 + site, 2), count);
                let mut want = 0.0f32;
                for (i, (&x, &y)) in a.iter().zip(&b).enumerate() {
                    let d = x - y;
                    want = if i == 0 { d * d } else { d.mul_add(d, want) };
                }
                let interpreted = cpu_mse(&shape, &a, &b);
                let divided = want / count as f32;
                if interpreted.to_bits() != divided.to_bits() {
                    oracle_wrong.push(format!(
                        "{name} mse {shape:?}: interpreter {interpreted:e}, reference {divided:e}"
                    ));
                }
                let module = emit_loss_sum_undivided(kernel, &profile).expect("the sum emits");
                let views = run(&device, kernel, &module.spirv, [&a, &b], 1);
                let got = views[0][0];
                if got.to_bits() != want.to_bits() {
                    wrong.push(format!(
                        "{name} {schedule} mse sum over {shape:?}: device {got:e}, CPU {want:e}"
                    ));
                }
                if (0..GUARD_WORDS).any(|i| views[0][1 + i].to_bits() != guard(i)) {
                    wrong.push(format!("{name} {schedule} mse: a guard word changed"));
                }
                for message in vulkan.take_validation_messages() {
                    wrong.push(format!("{name} {schedule} mse: {message}"));
                }
                losses += 1;
            }
        }
    }
    eprintln!(
        "vulkan_fma_oracle: {matmul_sites} matmul sites ({blocks} of 256 invocations, \
         {elements} elements) and {losses} loss sums compared bit for bit with the CPU \
         ({kernel_cpu:?})"
    );
    assert!(matmul_sites == 40 && losses == 8, "{matmul_sites} {losses}");
    assert!(
        wrong.is_empty() && oracle_wrong.is_empty(),
        "{} device views or sums differ from the CPU: {:#?}\n{} CPU values differ from the \
         written-out reference: {:#?}",
        wrong.len(),
        &wrong[..wrong.len().min(16)],
        oracle_wrong.len(),
        &oracle_wrong[..oracle_wrong.len().min(8)]
    );
}
