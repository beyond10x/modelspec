//! story:fuse-elementwise-chains-and-gradient-sums on a Vulkan device: a fused elementwise chain
//! kernel writes the bits the unfused plan writes, one kernel per op with every intermediate
//! stored, on every element (DESIGN.md § 3.1, § 5.4).
//!
//! Each chain runs at every count of [`COUNTS`] on two sets of operands: every combination of
//! [`SPECIAL`] values (zeros of both signs, infinities, NaN, subnormals, the largest and the
//! smallest normal) on the first elements, then the fixed-seed [`Draw`] values. The fused output
//! equals the unfused output bit for bit everywhere; on the drawn elements both equal the `f32`
//! arithmetic of the host, ops applied in the order of the chain, each rounded once. The guard
//! words after the output stay unchanged.
//!
//! Ignored by default: it creates a Vulkan instance, which `task ci` never does. Run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test vulkan_elementwise_chain_same_bits -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use common::{Draw, GUARD_WORDS, guard, guarded};
use modelspec_core::ir::kernel::{Kernel, KernelKind, KernelModuleId, Operand};
use modelspec_core::ir::tensor::{DType, Dim, TensorType};
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

/// One element, one short of a workgroup, one workgroup, one past it, one past 64 workgroups,
/// and the MNIST hidden activations of a 100-row batch.
const COUNTS: [u64; 6] = [1, 63, 64, 65, 4_097, 78_400];

/// The bits each output element holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_dead;

/// The values whose every combination the first elements of each operand take.
const SPECIAL: [f32; 11] = [
    0.0,
    -0.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    f32::from_bits(0x0001_16c2),
    f32::from_bits(0x8001_16c2),
    f32::MAX,
    f32::MIN_POSITIVE,
    1.0,
    -3.0,
];

/// A leaf operand of a chain.
#[derive(Clone, Copy, Debug)]
enum Leaf {
    Tensor,
    Rank0,
    Immediate(f32),
}

/// The chains the device checks, by kernel op, with their leaves in binding order.
fn chains() -> Vec<(&'static str, Vec<Leaf>)> {
    use Leaf::{Immediate, Rank0, Tensor};
    vec![
        ("Sub>Mul@0", vec![Tensor, Tensor, Immediate(0.002)]),
        ("Add>Add@0", vec![Tensor, Tensor, Tensor]),
        ("Add>ReluGrad@1", vec![Tensor, Tensor, Tensor]),
        ("Mul>Add@0", vec![Tensor, Tensor, Tensor]),
        ("Mul>Sub@1", vec![Tensor, Tensor, Tensor]),
        ("Relu>Mul@1", vec![Tensor, Rank0]),
        (
            "Sub>Mul@0>Add@1",
            vec![Tensor, Tensor, Immediate(0.375), Tensor],
        ),
        ("Mul>Add@0>Relu@0", vec![Tensor, Tensor, Tensor]),
    ]
}

fn arity(op: &str) -> usize {
    if op == "Relu" { 1 } else { 2 }
}

/// The ops of chain `name`, each with the position of the previous result among its operands
/// (`None` for the first).
fn links(name: &'static str) -> Vec<(&'static str, Option<usize>)> {
    name.split('>')
        .enumerate()
        .map(|(i, link)| match link.split_once('@') {
            Some((op, at)) => (op, Some(at.parse().expect("a position"))),
            None => {
                assert_eq!(i, 0, "{name}");
                (link, None)
            }
        })
        .collect()
}

/// Where one operand of an unfused op comes from.
#[derive(Clone, Copy, Debug)]
enum Arg {
    Leaf(usize),
    Previous,
}

/// The operands of each op of chain `name`, in order.
fn op_operands(name: &'static str) -> Vec<(&'static str, Vec<Arg>)> {
    let mut next = 0;
    links(name)
        .into_iter()
        .map(|(op, at)| {
            let operands = (0..arity(op))
                .map(|i| {
                    if Some(i) == at {
                        Arg::Previous
                    } else {
                        next += 1;
                        Arg::Leaf(next - 1)
                    }
                })
                .collect();
            (op, operands)
        })
        .collect()
}

fn ty(leaf: Leaf, count: u64) -> Option<TensorType> {
    match leaf {
        Leaf::Tensor => Some(TensorType::f32(&[count])),
        Leaf::Rank0 => Some(TensorType::f32_scalar()),
        Leaf::Immediate(_) => None,
    }
}

fn operand(leaf: Leaf, count: u64) -> Operand {
    match leaf {
        Leaf::Immediate(value) => Operand::Immediate(value),
        _ => Operand::Buffer(ty(leaf, count).expect("a buffer leaf")),
    }
}

/// The elementwise kernel `op` over `operands` and `count` elements, as the schedule builds it.
fn kernel(op: &'static str, operands: Vec<Operand>, count: u64) -> Kernel {
    let kind = KernelKind::Elementwise;
    let groups = count.div_ceil(u64::from(kind.workgroup_size()));
    let mut kernel = Kernel {
        id: KernelModuleId([0; 16]),
        kind,
        op,
        dtype: DType::F32,
        operands,
        result: TensorType {
            dtype: DType::F32,
            shape: vec![Dim::Static(count)],
        },
        reduces: None,
        epilogue: None,
        transposed: [false, false],
        softmax_stats: None,
        workgroup_size: kind.workgroup_size(),
        workgroups: [u32::try_from(groups).expect("one dimension"), 1, 1],
    };
    kernel.id = Kernel::module_id(&kernel.signature());
    kernel
}

/// The host's `f32` result of one op.
fn apply(op: &str, a: f32, b: f32) -> f32 {
    match op {
        "Add" => a + b,
        "Sub" => a - b,
        "Mul" => a * b,
        "Relu" => {
            if a > 0.0 {
                a
            } else {
                0.0
            }
        }
        "ReluGrad" => {
            if a > 0.0 {
                b
            } else {
                0.0
            }
        }
        other => panic!("no op {other}"),
    }
}

/// The host's result of chain `name` at element `i`.
fn host(name: &'static str, leaves: &[Leaf], values: &[Vec<f32>], i: usize) -> f32 {
    let leaf_value = |l: usize| -> f32 {
        match leaves[l] {
            Leaf::Immediate(v) => v,
            Leaf::Rank0 => values[l][0],
            Leaf::Tensor => values[l][i],
        }
    };
    let mut previous = f32::NAN;
    for (op, operands) in op_operands(name) {
        let args: Vec<f32> = operands
            .iter()
            .map(|from| match from {
                Arg::Leaf(l) => leaf_value(*l),
                Arg::Previous => previous,
            })
            .collect();
        previous = apply(op, args[0], *args.get(1).unwrap_or(&args[0]));
    }
    previous
}

fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

/// Uploads `uploads` into one device buffer each, runs `steps` (a kernel, the buffers it reads
/// in binding order, the buffer it writes), a compute barrier between two steps, and reads
/// back buffer `out`, in one command buffer with one host wait.
fn run(
    device: &Device<'_>,
    uploads: &[Vec<f32>],
    steps: &[(&Kernel, Vec<usize>, usize)],
    out: usize,
    count: u64,
) -> Vec<f32> {
    let buffers: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|v| {
            device
                .device_local_buffer(bytes_of(v.len()))
                .expect("a device buffer")
        })
        .collect();
    let mut staging: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|v| {
            let mut b = device
                .staging_buffer(bytes_of(v.len()))
                .expect("a staging buffer");
            b.write_f32(v).expect("the staging buffer is written");
            b
        })
        .collect();
    let pipelines: Vec<_> = steps
        .iter()
        .map(|(k, _, _)| {
            let words = emit(k).unwrap_or_else(|e| panic!("{}: {e}", k.signature()));
            device
                .kernel_pipeline(k, &words)
                .expect("the pipeline is created")
        })
        .collect();
    let mut recording = device.record().expect("a recording starts");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("the upload is recorded");
    }
    recording.barrier(Barrier::TransferToCompute);
    for (at, ((k, reads, write), pipeline)) in steps.iter().zip(&pipelines).enumerate() {
        if at > 0 {
            recording.barrier(Barrier::ComputeToCompute);
        }
        let mut bound: Vec<&Buffer<'_>> = reads.iter().map(|&r| &buffers[r]).collect();
        bound.push(&buffers[*write]);
        recording
            .dispatch(pipeline, &bound, count as u32, k.workgroups)
            .expect("the dispatch is recorded");
    }
    recording.barrier(Barrier::ComputeToTransfer);
    recording
        .copy(&buffers[out], &staging[out])
        .expect("the download is recorded");
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    device
        .submit(recorded)
        .expect("submitted")
        .wait()
        .expect("the device finishes");
    staging[out].read_f32().expect("the output reads")
}

/// The leaf values of one case: every [`SPECIAL`] combination on the first elements of the
/// tensor leaves, then drawn values. Returns them and the number of special elements.
fn values(leaves: &[Leaf], count: u64, draw: &mut Draw) -> (Vec<Vec<f32>>, usize) {
    let tensors = leaves.iter().filter(|l| matches!(l, Leaf::Tensor)).count() as u32;
    let combinations = SPECIAL.len().pow(tensors);
    let special = (count as usize).min(combinations);
    let mut t = 0u32;
    let values = leaves
        .iter()
        .map(|leaf| match leaf {
            Leaf::Tensor => {
                let mut v = draw.values(count);
                let stride = SPECIAL.len().pow(t);
                t += 1;
                for (i, x) in v.iter_mut().enumerate().take(special) {
                    *x = SPECIAL[(i / stride) % SPECIAL.len()];
                }
                v
            }
            Leaf::Rank0 => draw.values(1),
            Leaf::Immediate(_) => Vec::new(),
        })
        .collect();
    (values, special)
}

fn validation_requested() -> bool {
    std::env::var(VALIDATION_ENV).as_deref() == Ok("1")
}

#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_a_fused_elementwise_chain_writes_the_bits_of_the_unfused_kernels() {
    let started = std::time::Instant::now();
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(vulkan.validation_enabled(), validation_requested());
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let setup = vulkan.take_validation_messages();
    assert!(setup.is_empty(), "validation messages at setup: {setup:#?}");

    let mut draw = Draw::new(0x6368_6169_6e73_0001);
    let (mut cases, mut elements, mut specials) = (0, 0u64, 0usize);
    let mut differ = Vec::new();
    let mut messages = Vec::new();
    for count in COUNTS {
        for (name, leaves) in chains() {
            let n = count as usize;
            let (data, special) = values(&leaves, count, &mut draw);
            let operands: Vec<Operand> = leaves.iter().map(|&l| operand(l, count)).collect();
            let fused = kernel(name, operands, count);
            assert_eq!(
                fused.operands.len(),
                op_operands(name)
                    .iter()
                    .flat_map(|(_, o)| o)
                    .filter(|f| matches!(f, Arg::Leaf(_)))
                    .count(),
                "{name}"
            );

            // Buffers: each buffer leaf, then one per unfused op result, then the fused output.
            let buffer_leaves: Vec<usize> = (0..leaves.len())
                .filter(|&l| !matches!(leaves[l], Leaf::Immediate(_)))
                .collect();
            let mut uploads: Vec<Vec<f32>> =
                buffer_leaves.iter().map(|&l| data[l].clone()).collect();
            let ops = op_operands(name);
            let first_result = uploads.len();
            for _ in 0..=ops.len() {
                uploads.push(guarded(&vec![f32::from_bits(UNWRITTEN); n]));
            }
            let fused_out = uploads.len() - 1;
            let unfused_out = fused_out - 1;

            let mut singles = Vec::new();
            for (k, (op, from)) in ops.iter().enumerate() {
                let mut operands = Vec::new();
                let mut reads = Vec::new();
                for f in from {
                    match *f {
                        Arg::Previous => {
                            operands.push(Operand::Buffer(TensorType::f32(&[count])));
                            reads.push(first_result + k - 1);
                        }
                        Arg::Leaf(l) => {
                            operands.push(operand(leaves[l], count));
                            if let Some(b) = buffer_leaves.iter().position(|&x| x == l) {
                                reads.push(b);
                            }
                        }
                    }
                }
                singles.push((kernel(op, operands, count), reads, first_result + k));
            }
            let unfused_steps: Vec<(&Kernel, Vec<usize>, usize)> =
                singles.iter().map(|(k, r, w)| (k, r.clone(), *w)).collect();
            let unfused = run(&device, &uploads, &unfused_steps, unfused_out, count);
            let fused_reads: Vec<usize> = (0..buffer_leaves.len()).collect();
            let got = run(
                &device,
                &uploads,
                &[(&fused, fused_reads, fused_out)],
                fused_out,
                count,
            );
            for i in 0..n {
                let (f, u) = (got[i], unfused[i]);
                if f.to_bits() != u.to_bits() {
                    differ.push(format!(
                        "{name} n={count} [{i}]: fused {f:?} ({:#010x}) unfused {u:?} ({:#010x})",
                        f.to_bits(),
                        u.to_bits()
                    ));
                } else if i >= special {
                    let h = host(name, &leaves, &data, i);
                    if h.to_bits() != f.to_bits() {
                        differ.push(format!("{name} n={count} [{i}]: device {f:?} host {h:?}"));
                    }
                }
            }
            for (i, word) in got[n..n + GUARD_WORDS].iter().enumerate() {
                if word.to_bits() != guard(i) {
                    differ.push(format!("{name} n={count}: guard word {i} changed"));
                }
            }
            for message in vulkan.take_validation_messages() {
                messages.push(format!("{name} n={count}: {message}"));
            }
            cases += 1;
            elements += count;
            specials += special;
        }
    }
    eprintln!(
        "vulkan_elementwise_chain_same_bits: {cases} cases, {elements} elements ({specials} \
         special), {} differ, validation {} with {} messages, {:.2} s",
        differ.len(),
        if vulkan.validation_enabled() {
            "on"
        } else {
            "off"
        },
        messages.len(),
        started.elapsed().as_secs_f64()
    );
    assert!(differ.is_empty(), "{:#?}", &differ[..differ.len().min(20)]);
    assert!(messages.is_empty(), "{messages:#?}");
}
