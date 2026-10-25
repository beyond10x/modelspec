//! Adversary pass 1 of story:fuse-elementwise-chains-and-gradient-sums on a Vulkan device, with
//! the chain kernels a compiled artifact holds (not hand-built ones):
//!
//! - the adjoint sum of four contributions, the plan's `Add>Add@0` then `Add>ReluGrad@1`, over
//!   the contributions `1e8, 1, -1e8, 1`, writes `1` on every element: the IR's left-to-right
//!   sum `((c0 + c1) + c2) + c3`. Right to left it is `0`, and `(c0 + c2) + …` is `2`, so any
//!   other association changes the bits;
//! - every chain kernel of a model whose chains end in a contractible `mul` then `add`
//!   (`Sub>Mul@0>Add@0`, `Mul>Add@1`, `Sub>Mul@0`) writes in its deterministic module the host's
//!   `f32` arithmetic, each op rounded once, bit for bit, and its `fast_math` module stays within
//!   the § 9 gate 5 tolerance of the deterministic one (`GPU_ATOL + GPU_RTOL * |c|`); the count
//!   of elements the two differ on is printed, the contraction `fast_math` permits.
//!
//! Ignored by default. Run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test adversary_fuse1_chain_device -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use common::{Draw, GPU_ATOL, GPU_RTOL, within};
use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::compile_artifact;
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{Kernel, KernelKind, Operand};
use modelspec_runtime::spirv::emit::emit_kernels;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

/// `h` read by four matmuls: the adjoint of `h` sums four contributions.
const ADJOINT_OF_FOUR: &str = "model Adv {
    param w1: Tensor<f32, [12, 9]> = xavier_uniform();
    param w2: Tensor<f32, [9, 5]> = xavier_uniform();
    param w3: Tensor<f32, [9, 5]> = xavier_uniform();
    param w4: Tensor<f32, [9, 5]> = xavier_uniform();
    param w5: Tensor<f32, [9, 5]> = xavier_uniform();

    fn forward(x: Tensor<f32, [10, 12]>)
        -> Tensor<f32, [10, 5]> {
        let h = relu(matmul(x, w1));
        return add(add(add(matmul(h, w2), matmul(h, w3)), matmul(h, w4)), matmul(h, w5));
    }
}

train Fit {
    model = Adv;
    input x: Tensor<f32, [10, 12]>;
    target y: Tensor<f32, [10, 5]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.05);
    steps = 30;
    seed = 7;
    device = cpu;
}
";

/// Chains over 64 x 257 elements whose ops end in a product then a sum.
const CONTRACTIBLE: &str = "model Adv {
    param w1: Tensor<f32, [300, 257]> = xavier_uniform();
    param s: Tensor<f32, [64, 257]> = ones();

    fn forward(x: Tensor<f32, [64, 300]>)
        -> Tensor<f32, [64, 257]> {
        return add(mul(sub(matmul(x, w1), s), 0.3), s);
    }
}

train Fit {
    model = Adv;
    input x: Tensor<f32, [64, 300]>;
    target y: Tensor<f32, [64, 257]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.05);
    steps = 30;
    seed = 7;
    device = cpu;
}
";

fn artifact(source: &str) -> CompiledArtifact {
    compile_artifact("adv1.mspec", source, Backend::Vulkan).expect("the source compiles")
}

fn elements(kernel: &Kernel) -> u64 {
    kernel
        .result
        .shape
        .iter()
        .map(|d| match d {
            modelspec_core::ir::tensor::Dim::Static(n) => *n,
            other => panic!("{other:?} is not static"),
        })
        .product()
}

fn chain_kernels(artifact: &CompiledArtifact) -> Vec<(usize, &Kernel)> {
    artifact
        .execution
        .kernels
        .iter()
        .enumerate()
        .filter(|(_, k)| k.kind == KernelKind::Elementwise && k.op.contains('>'))
        .collect()
}

/// Uploads `uploads` into one device buffer each, runs `steps` (a kernel, its module words,
/// the buffers it binds in order, reads then the write) with a compute barrier between two
/// steps, and reads back buffer `out`, in one submission and one host wait.
fn run(
    device: &Device<'_>,
    uploads: &[Vec<f32>],
    steps: &[(&Kernel, &[u32], Vec<usize>)],
    out: usize,
) -> Vec<f32> {
    let bytes = |n: usize| (n * 4) as u64;
    let buffers: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|v| {
            device
                .device_local_buffer(bytes(v.len()))
                .expect("a buffer")
        })
        .collect();
    let mut staging: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|v| {
            let mut b = device.staging_buffer(bytes(v.len())).expect("staging");
            b.write_f32(v).expect("written");
            b
        })
        .collect();
    let pipelines: Vec<_> = steps
        .iter()
        .map(|(k, words, _)| device.kernel_pipeline(k, words).expect("a pipeline"))
        .collect();
    let mut recording = device.record().expect("a recording");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("upload");
    }
    recording.barrier(Barrier::TransferToCompute);
    for (at, ((k, _, bound), pipeline)) in steps.iter().zip(&pipelines).enumerate() {
        if at > 0 {
            recording.barrier(Barrier::ComputeToCompute);
        }
        let bound: Vec<&Buffer<'_>> = bound.iter().map(|&b| &buffers[b]).collect();
        recording
            .dispatch_kernel(pipeline, k, &bound)
            .expect("the dispatch is recorded");
    }
    recording.barrier(Barrier::ComputeToTransfer);
    recording
        .copy(&buffers[out], &staging[out])
        .expect("download");
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    device
        .submit(recorded)
        .expect("submitted")
        .wait()
        .expect("the device finishes");
    staging[out].read_f32().expect("the output reads")
}

/// The host's value of chain `op` over `leaves` (one value per leaf at this element), each op
/// rounded once to `f32`.
fn host(op: &str, leaves: &[f32]) -> f32 {
    let mut leaves = leaves.iter().copied();
    let mut previous = None;
    for (i, link) in op.split('>').enumerate() {
        let (name, at) = match link.split_once('@') {
            Some((name, at)) => (name, Some(at.parse::<usize>().expect("a position"))),
            None => {
                assert_eq!(i, 0);
                (link, None)
            }
        };
        let arity = if name == "Relu" { 1 } else { 2 };
        let args: Vec<f32> = (0..arity)
            .map(|j| {
                if Some(j) == at {
                    previous.expect("a previous result")
                } else {
                    leaves.next().expect("a leaf")
                }
            })
            .collect();
        previous = Some(match name {
            "Add" => args[0] + args[1],
            "Sub" => args[0] - args[1],
            "Mul" => args[0] * args[1],
            "Relu" => {
                if args[0] > 0.0 {
                    args[0]
                } else {
                    0.0
                }
            }
            "ReluGrad" => {
                if args[0] > 0.0 {
                    args[1]
                } else {
                    0.0
                }
            }
            other => panic!("no op {other}"),
        });
    }
    previous.expect("a chain has ops")
}

fn open_device(vulkan: &Vulkan) -> (Device<'_>, DeviceProfile) {
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = candidates[0].profile.clone();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    (device, profile)
}

fn validation_requested() -> bool {
    std::env::var(VALIDATION_ENV).as_deref() == Ok("1")
}

#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_the_plans_adjoint_sum_of_four_adds_left_to_right() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(vulkan.validation_enabled(), validation_requested());
    let (device, profile) = open_device(&vulkan);
    let artifact = artifact(ADJOINT_OF_FOUR);
    let modules = emit_kernels(&artifact, &profile).expect("the kernels emit");
    let chains = chain_kernels(&artifact);
    let find = |op: &str| {
        chains
            .iter()
            .find(|(_, k)| k.op == op && elements(k) == 90)
            .copied()
            .unwrap_or_else(|| panic!("a {op} kernel over 10 x 9 in {chains:?}"))
    };
    let (sum_at, sum) = find("Add>Add@0");
    let (grad_at, grad) = find("Add>ReluGrad@1");
    let n = 90;
    let fill = |v: f32| vec![v; n];
    // Buffers: c0, c1, c2, the sum s, c3, the pre-activation z, the result.
    let uploads = vec![
        fill(1e8),
        fill(1.0),
        fill(-1e8),
        fill(f32::NAN),
        fill(1.0),
        fill(0.5),
        fill(f32::NAN),
    ];
    let steps = [
        (sum, &modules[sum_at].spirv[..], vec![0, 1, 2, 3]),
        (grad, &modules[grad_at].spirv[..], vec![3, 4, 5, 6]),
    ];
    let got = run(&device, &uploads, &steps, 6);
    let left_to_right = ((1e8f32 + 1.0) + -1e8) + 1.0;
    let right_to_left = 1e8f32 + (1.0 + (-1e8 + 1.0));
    assert_eq!(
        (left_to_right, right_to_left),
        (1.0, 0.0),
        "the case tells the orders apart"
    );
    let wrong: Vec<(usize, f32)> = got[..n]
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, v)| v.to_bits() != left_to_right.to_bits())
        .collect();
    let messages = vulkan.take_validation_messages();
    eprintln!(
        "adversary_fuse1_chain_device adjoint: {n} elements, {} not {left_to_right}, validation {} with {} messages",
        wrong.len(),
        if vulkan.validation_enabled() {
            "on"
        } else {
            "off"
        },
        messages.len()
    );
    assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
    assert!(messages.is_empty(), "{messages:#?}");
}

#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_artifact_chains_write_host_bits_and_fast_math_stays_within_tolerance() {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(vulkan.validation_enabled(), validation_requested());
    let (device, profile) = open_device(&vulkan);
    let deterministic = artifact(CONTRACTIBLE);
    let mut fast = deterministic.clone();
    fast.fast_math = true;
    let off = emit_kernels(&deterministic, &profile).expect("the kernels emit");
    let on = emit_kernels(&fast, &profile).expect("the fast_math kernels emit");
    let chains = chain_kernels(&deterministic);
    let names: Vec<&str> = chains.iter().map(|(_, k)| k.op).collect();
    for expected in ["Sub>Mul@0>Add@0", "Mul>Add@1", "Sub>Mul@0"] {
        assert!(names.contains(&expected), "{expected} in {names:?}");
    }
    let mut draw = Draw::new(0x6164_7631_6675_7365);
    let mut report = Vec::new();
    let mut failures = Vec::new();
    for (at, kernel) in chains {
        let n = elements(kernel) as usize;
        let mut uploads: Vec<Vec<f32>> = Vec::new();
        let mut leaves: Vec<Option<usize>> = Vec::new();
        for operand in &kernel.operands {
            match operand {
                Operand::Buffer(ty) => {
                    let count = if ty.rank() == 0 { 1 } else { n as u64 };
                    leaves.push(Some(uploads.len()));
                    uploads.push(draw.values(count));
                }
                Operand::Immediate(_) => leaves.push(None),
            }
        }
        let out = uploads.len();
        uploads.push(vec![f32::NAN; n]);
        let bound: Vec<usize> = (0..=out).collect();
        let det = run(
            &device,
            &uploads,
            &[(kernel, &off[at].spirv[..], bound.clone())],
            out,
        );
        let fm = run(
            &device,
            &uploads,
            &[(kernel, &on[at].spirv[..], bound)],
            out,
        );
        let (mut host_differ, mut fast_differ, mut outside) = (0, 0, 0);
        for i in 0..n {
            let values: Vec<f32> = kernel
                .operands
                .iter()
                .zip(&leaves)
                .map(|(operand, leaf)| match (operand, leaf) {
                    (Operand::Immediate(v), _) => *v,
                    (Operand::Buffer(ty), Some(b)) => {
                        uploads[*b][if ty.rank() == 0 { 0 } else { i }]
                    }
                    _ => unreachable!(),
                })
                .collect();
            let h = host(kernel.op, &values);
            if h.to_bits() != det[i].to_bits() {
                host_differ += 1;
            }
            if fm[i].to_bits() != det[i].to_bits() {
                fast_differ += 1;
            }
            if !within(fm[i], det[i], det[i].abs(), GPU_ATOL, GPU_RTOL) {
                outside += 1;
            }
        }
        report.push(format!(
            "{} n={n}: deterministic vs host differ {host_differ}, fast_math vs deterministic differ {fast_differ}, outside tolerance {outside}",
            kernel.op
        ));
        if host_differ > 0 || outside > 0 {
            failures.push(report.last().cloned().expect("a line"));
        }
    }
    let messages = vulkan.take_validation_messages();
    eprintln!(
        "adversary_fuse1_chain_device chains, validation {} with {} messages:\n  {}",
        if vulkan.validation_enabled() {
            "on"
        } else {
            "off"
        },
        messages.len(),
        report.join("\n  ")
    );
    assert!(failures.is_empty(), "{failures:#?}");
    assert!(messages.is_empty(), "{messages:#?}");
}
