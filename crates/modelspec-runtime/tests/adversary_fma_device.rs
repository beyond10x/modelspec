//! Adversary cases for story:deterministic-fma on a Vulkan device (DESIGN.md § 12 ADR-009).
//!
//! Every matmul tile, each selected explicitly by its shape: one row per invocation, two rows,
//! the square and the wide register-blocked tiles, and the 128 by 128 block of 256 invocations
//! under the device's profile; each operand read as is or transposed.
//!
//! 1. At `k` = 1, 2, 3, 5, 33, 127 and 1000 (not multiples of the tile depth, and below and
//!    above the register-blocked tiles' bound of 64), drawn operands give the bits of the CPU
//!    kernel in every element, and the CPU kernel those of the written-out reference
//!    (`a[i, 0] * b[0, j]`, then `mul_add` per later `k`) at sampled elements; a `relu`
//!    epilogue that keeps the pre-activation at `k` = 127.
//! 2. A product that overflows when rounded alone and is cancelled by the running sum,
//!    `-f32::MAX + 2^64 * 2^64`, is `2^104` in every element: the device fuses it, as the CPU
//!    oracle's `mul_add` does, instead of computing `+inf`.
//! 3. A sum whose every product is `-1 * +0 = -0` stays `-0`: the `k = 0` product starts the
//!    sum, not `fma(a0, b0, +0)`.
//!
//! Ignored by default: they create a Vulkan instance. Run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test adversary_fma_device -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use common::Draw;
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{
    Epilogue, Kernel, KernelKind, KernelModuleId, MatMulTile, Operand,
};
use modelspec_core::ir::tensor::{DType, TensorType};
use modelspec_runtime::cpu::matmul::{self as cpu, Lhs};
use modelspec_runtime::spirv::emit::emit_module;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

const UNWRITTEN: u32 = 0x7fc0_f3a1;

const FLAGS: [[bool; 2]; 4] = [[false, false], [true, false], [false, true], [true, true]];

const KS: [u64; 7] = [1, 2, 3, 5, 33, 127, 1000];

/// `(name, m, n, invocation limit)`: shapes that select each tile at `k >= 64`; below 64 the
/// wide and block shapes take the square tile.
const TILES: [(&str, u64, u64, u32); 5] = [
    ("one row", 17, 65, 128),
    ("two rows", 2049, 17, 128),
    ("square", 1024, 512, 128),
    ("wide", 2048, 1024, 128),
    ("block", 2048, 2048, 256),
];

const SAMPLES: usize = 97;

fn matmul_kernel(
    m: u64,
    k: u64,
    n: u64,
    limit: u32,
    transposed: [bool; 2],
    epilogue: Option<Epilogue>,
) -> Kernel {
    let tile = MatMulTile::of_invocations(m, k, n, limit);
    let mut kernel = Kernel {
        id: KernelModuleId([0; 16]),
        kind: KernelKind::MatMul,
        op: "MatMul",
        dtype: DType::F32,
        operands: vec![
            Operand::Buffer(TensorType::f32(&[m, k])),
            Operand::Buffer(TensorType::f32(&[k, n])),
        ],
        result: TensorType::f32(&[m, n]),
        reduces: None,
        epilogue,
        transposed,
        workgroup_size: tile.workgroup_size,
        workgroups: [
            u32::try_from(tile.workgroups(m, n)).expect("one dimension"),
            1,
            1,
        ],
    };
    kernel.id = Kernel::module_id(&kernel.signature());
    kernel
}

fn transpose(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; values.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = values[r * cols + c];
        }
    }
    out
}

fn reference(a: &[f32], b: &[f32], k: usize, n: usize, i: usize, j: usize) -> f32 {
    let mut acc = a[i * k] * b[j];
    for p in 1..k {
        acc = a[i * k + p].mul_add(b[p * n + j], acc);
    }
    acc
}

fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

/// Runs `kernel` once; returns every written view whole, guard words included.
fn run(
    device: &Device<'_>,
    kernel: &Kernel,
    spirv: &[u32],
    inputs: [&[f32]; 2],
    count: usize,
) -> Vec<Vec<f32>> {
    let uploads: Vec<Vec<f32>> = inputs
        .iter()
        .map(|values| common::guarded(values))
        .chain(
            (0..kernel.written()).map(|_| common::guarded(&vec![f32::from_bits(UNWRITTEN); count])),
        )
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

struct Gpu {
    vulkan: Vulkan,
    profile: DeviceProfile,
}

fn gpu() -> Gpu {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        std::env::var(VALIDATION_ENV).as_deref() == Ok("1")
    );
    let profile = vulkan.candidates().expect("the devices are probed")[0]
        .profile
        .clone();
    Gpu { vulkan, profile }
}

/// Runs `kernel` on `a`, `b` (row-major, bound transposed as the kernel reads them) and returns
/// the differences of each written view from `want` (one per view), and from the guards.
#[allow(clippy::too_many_arguments)]
fn compare(
    gpu: &Gpu,
    device: &Device<'_>,
    kernel: &Kernel,
    a: &[f32],
    b: &[f32],
    (m, k, n): (u64, u64, u64),
    want: &[Vec<f32>],
    case: &str,
    wrong: &mut Vec<String>,
) {
    let [ta, tb] = kernel.transposed;
    let a_bound = if ta {
        transpose(a, m as usize, k as usize)
    } else {
        a.to_vec()
    };
    let b_bound = if tb {
        transpose(b, k as usize, n as usize)
    } else {
        b.to_vec()
    };
    let module = match emit_module(kernel, &gpu.profile) {
        Ok(module) => module,
        Err(error) => {
            wrong.push(format!("{case}: does not emit: {error:?}"));
            return;
        }
    };
    let count = (m * n) as usize;
    let views = run(device, kernel, &module.spirv, [&a_bound, &b_bound], count);
    assert_eq!(views.len(), want.len(), "{case}");
    for (at, (view, want)) in views.iter().zip(want).enumerate() {
        if (0..common::GUARD_WORDS).any(|i| view[count + i].to_bits() != common::guard(i)) {
            wrong.push(format!("{case} view {at}: a guard word changed"));
        }
        let differ: Vec<usize> = (0..count)
            .filter(|&e| view[e].to_bits() != want[e].to_bits())
            .collect();
        if let Some(&e) = differ.first() {
            wrong.push(format!(
                "{case} view {at}: {} of {count} elements differ, first ({}, {}) device {:e} \
                 ({:#010x}) want {:e} ({:#010x})",
                differ.len(),
                e / n as usize,
                e % n as usize,
                view[e],
                view[e].to_bits(),
                want[e],
                want[e].to_bits()
            ));
        }
    }
    for message in gpu.vulkan.take_validation_messages() {
        wrong.push(format!("{case}: {message}"));
    }
}

fn tile_name(kernel: &Kernel) -> String {
    let t = kernel.matmul_tile().expect("a tile");
    format!(
        "wg {} rows {} cols/inv {} depth {}",
        t.workgroup_size, t.rows, t.cols_per_invocation, t.depth
    )
}

#[test]
#[ignore = "needs a Vulkan device"]
fn every_tile_at_every_k_boundary_gives_the_fused_reference_bits() {
    let gpu = gpu();
    let negotiated = negotiate(&gpu.profile).expect("device 0 negotiates");
    let device = gpu
        .vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let cpu_kernel = cpu::Kernel::detected();
    let (mut wrong, mut runs, mut tiles) = (Vec::new(), 0, std::collections::BTreeSet::new());
    for (case_no, &(name, m, n, limit)) in TILES.iter().enumerate() {
        for &k in &KS {
            let mut draw = Draw::new(0xf3a1_0000 + (case_no as u64) * 64 + k);
            let a = draw.values(m * k);
            let b = draw.values(k * n);
            let sum = cpu::matmul(
                cpu_kernel,
                &a,
                Lhs::RowMajor,
                &b,
                m as usize,
                k as usize,
                n as usize,
            )
            .expect("the CPU kernel runs");
            let step = ((m * n) as usize / SAMPLES).max(1);
            for at in (0..(m * n) as usize).step_by(step) {
                let (i, j) = (at / n as usize, at % n as usize);
                let want = reference(&a, &b, k as usize, n as usize, i, j);
                if sum[at].to_bits() != want.to_bits() {
                    wrong.push(format!(
                        "{name} k {k}: CPU {cpu_kernel:?} ({i}, {j}) {:e} reference {want:e}",
                        sum[at]
                    ));
                }
            }
            for transposed in FLAGS {
                let kernel = matmul_kernel(m, k, n, limit, transposed, None);
                tiles.insert(tile_name(&kernel));
                let case = format!(
                    "{name} [{m}, {k}] x [{k}, {n}] {transposed:?} {}",
                    tile_name(&kernel)
                );
                compare(
                    &gpu,
                    &device,
                    &kernel,
                    &a,
                    &b,
                    (m, k, n),
                    std::slice::from_ref(&sum),
                    &case,
                    &mut wrong,
                );
                runs += 1;
            }
            if k == 127 {
                let kernel = matmul_kernel(
                    m,
                    k,
                    n,
                    limit,
                    [false, false],
                    Some(Epilogue::Relu {
                        writes_pre_activation: true,
                    }),
                );
                let activation: Vec<f32> = sum.iter().map(|&x| relu(x)).collect();
                let case = format!("{name} [{m}, {k}] x [{k}, {n}] relu+pre");
                compare(
                    &gpu,
                    &device,
                    &kernel,
                    &a,
                    &b,
                    (m, k, n),
                    &[sum.clone(), activation],
                    &case,
                    &mut wrong,
                );
                runs += 1;
            }
        }
    }
    eprintln!("adversary_fma_device: {runs} matmul runs over tiles {tiles:#?}");
    assert!(tiles.len() >= 5, "{tiles:?}");
    assert!(
        wrong.is_empty(),
        "{} cases: {:#?}",
        wrong.len(),
        &wrong[..wrong.len().min(20)]
    );
}

/// The `[m, k]`, `[k, n]` operands whose every element is `first * 1 + big * big + 0 * 0 …`.
fn cancelling(m: u64, k: u64, n: u64, first: f32, big: f32) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..m)
        .flat_map(|_| {
            (0..k).map(move |p| match p {
                0 => first,
                1 => big,
                _ => 0.0,
            })
        })
        .collect();
    let b: Vec<f32> = (0..k)
        .flat_map(|p| {
            (0..n).map(move |_| match p {
                0 => 1.0,
                1 => big,
                _ => 0.0,
            })
        })
        .collect();
    (a, b)
}

#[test]
#[ignore = "needs a Vulkan device"]
fn an_overflowing_product_cancelled_by_the_sum_is_finite_on_every_tile() {
    let gpu = gpu();
    let negotiated = negotiate(&gpu.profile).expect("device 0 negotiates");
    let device = gpu
        .vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let want = 2f32.powi(104);
    let big = 2f32.powi(64);
    assert!((big * big).is_infinite());
    assert_eq!(big.mul_add(big, -f32::MAX), want);
    let mut wrong = Vec::new();
    let mut runs = 0;
    for &(name, m, n, limit) in &TILES {
        for k in [2u64, 3, 33, 127] {
            let (a, b) = cancelling(m, k, n, -f32::MAX, big);
            let expected = vec![want; (m * n) as usize];
            let cpu_sum = cpu::matmul(
                cpu::Kernel::detected(),
                &a,
                Lhs::RowMajor,
                &b,
                m as usize,
                k as usize,
                n as usize,
            )
            .expect("runs");
            if cpu_sum.iter().any(|v| v.to_bits() != want.to_bits()) {
                wrong.push(format!("{name} k {k}: the CPU kernel is not 2^104"));
            }
            for transposed in FLAGS {
                let kernel = matmul_kernel(m, k, n, limit, transposed, None);
                let case = format!(
                    "cancelling {name} [{m}, {k}] x [{k}, {n}] {transposed:?} {}",
                    tile_name(&kernel)
                );
                compare(
                    &gpu,
                    &device,
                    &kernel,
                    &a,
                    &b,
                    (m, k, n),
                    std::slice::from_ref(&expected),
                    &case,
                    &mut wrong,
                );
                runs += 1;
            }
        }
    }
    eprintln!("adversary_fma_device: {runs} cancelling runs");
    assert!(
        wrong.is_empty(),
        "{} cases: {:#?}",
        wrong.len(),
        &wrong[..wrong.len().min(20)]
    );
}

#[test]
#[ignore = "needs a Vulkan device"]
fn a_sum_of_negative_zero_products_stays_negative_zero_on_every_tile() {
    let gpu = gpu();
    let negotiated = negotiate(&gpu.profile).expect("device 0 negotiates");
    let device = gpu
        .vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let mut wrong = Vec::new();
    let mut runs = 0;
    for &(name, m, n, limit) in &TILES {
        for k in [1u64, 2, 3, 33, 127] {
            let a = vec![-1.0f32; (m * k) as usize];
            let b = vec![0.0f32; (k * n) as usize];
            let expected = vec![-0.0f32; (m * n) as usize];
            for transposed in [[false, false], [true, true]] {
                let kernel = matmul_kernel(m, k, n, limit, transposed, None);
                let case = format!(
                    "negative zero {name} [{m}, {k}] x [{k}, {n}] {transposed:?} {}",
                    tile_name(&kernel)
                );
                compare(
                    &gpu,
                    &device,
                    &kernel,
                    &a,
                    &b,
                    (m, k, n),
                    std::slice::from_ref(&expected),
                    &case,
                    &mut wrong,
                );
                runs += 1;
            }
        }
    }
    eprintln!("adversary_fma_device: {runs} negative-zero runs");
    assert!(
        wrong.is_empty(),
        "{} cases: {:#?}",
        wrong.len(),
        &wrong[..wrong.len().min(20)]
    );
}
