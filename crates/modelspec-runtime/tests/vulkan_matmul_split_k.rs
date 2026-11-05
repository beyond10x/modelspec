//! `story:vulkan-matmul-split-k` (DESIGN.md § 5.2, § 5.4): the two kernels of a split-K matmul.
//!
//! The first, a `MatMul` kernel over `[m, k]` and `[k, n]` whose result is `[P · m, n]`, writes
//! at rows `s · m ..` the product over the `k / P` values of `k` from `s · k / P`: for every
//! element the bits of the CPU kernel over that slice of both operands, the sum of its products
//! in ascending `k` from the slice's first product, each later one fused with `mul_add`. The
//! second, `SumRows` over `[P, m · n]`, adds the slices in ascending order from slice 0.
//!
//! - Without a device: every such module passes `spirv-val --target-env vulkan1.2` for the RTX
//!   3090 profile and for a profile of 128 invocations, and the emitter refuses a split result
//!   with an epilogue, with too few workgroups, or whose rows or `k` the slices do not divide.
//! - On the device (`--ignored`): the bits of both stages for the three weight-gradient shapes
//!   of `examples/bert_tiny.mspec` and for shapes with partial blocks of rows and columns, `a`,
//!   `b`, both or neither read transposed; every element written, every guard word unchanged,
//!   and the sum within the rounding bound of the CPU's unsplit matmul.
//!
//! The device test creates a Vulkan instance, which `task ci` never does. Run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test vulkan_matmul_split_k -- --ignored`.
#![cfg(feature = "vulkan")]

use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{
    Epilogue, Kernel, KernelKind, KernelModuleId, MatMulLimits, MatMulTile, Operand,
};
use modelspec_core::ir::tensor::{DType, TensorOp, TensorType};
use modelspec_core::schedule::{matmul_split_k_slices, rowwise_kernel};
use modelspec_runtime::cpu::matmul::{self as cpu, Lhs};
use modelspec_runtime::spirv::emit::emit_module;
use modelspec_runtime::spirv::validate::validate;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

const GUARD_WORDS: usize = 16;

/// The bits each output element holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_dead;

const RTX3090: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

/// `(m, k, n)`, the slices of `k` and the operands read transposed.
type Case = ((u64, u64, u64), u64, [bool; 2]);

/// The first three are the weight gradients of `examples/bert_tiny.mspec`; the slices of the
/// last one are not those of the schedule's rule.
const CASES: [Case; 9] = [
    ((128, 4096, 128), 32, [true, false]),
    ((512, 4096, 128), 32, [true, false]),
    ((128, 4096, 512), 32, [true, false]),
    ((8, 4096, 1), 32, [false, false]),
    ((1, 4096, 300), 32, [true, false]),
    ((37, 8192, 50), 64, [false, true]),
    ((130, 12288, 70), 48, [true, true]),
    ((5, 4224, 3), 33, [false, false]),
    ((9, 600, 7), 3, [true, false]),
];

fn rtx3090() -> DeviceProfile {
    DeviceProfile::from_json(RTX3090).expect("the fixture is a valid profile")
}

/// The first kernel of the split-K matmul of `[m, k] x [k, n]` in `slices`, as
/// `schedule::schedule_for` builds it within `limits`.
fn stage1(
    (m, k, n): (u64, u64, u64),
    slices: u64,
    transposed: [bool; 2],
    limits: MatMulLimits,
) -> Kernel {
    let tile = MatMulTile::for_limits(m * slices, k / slices, n, transposed, limits);
    let mut kernel = Kernel {
        id: KernelModuleId([0; 16]),
        kind: KernelKind::MatMul,
        op: "MatMul",
        dtype: DType::F32,
        operands: vec![
            Operand::Buffer(TensorType::f32(&[m, k])),
            Operand::Buffer(TensorType::f32(&[k, n])),
        ],
        result: TensorType::f32(&[slices * m, n]),
        reduces: None,
        epilogue: None,
        transposed,
        softmax_stats: None,
        workgroup_size: tile.workgroup_size,
        workgroups: [
            u32::try_from(slices * tile.workgroups(m, n)).expect("a test matmul fits a dimension"),
            1,
            1,
        ],
    };
    kernel.id = Kernel::module_id(&kernel.signature());
    kernel
}

/// The second kernel: `SumRows` over the `slices` partial products of `count` elements.
fn stage2(slices: u64, count: u64) -> Kernel {
    rowwise_kernel(
        &TensorOp::SumRows,
        vec![Operand::Buffer(TensorType::f32(&[slices, count]))],
        TensorType::f32(&[count]),
    )
    .expect("the schedule has a SumRows kernel")
}

fn signed(mut kernel: Kernel) -> Kernel {
    kernel.id = Kernel::module_id(&kernel.signature());
    kernel
}

#[test]
fn the_rule_gives_the_slices_of_every_case_but_the_last() {
    for ((_, k, _), slices, _) in &CASES[..CASES.len() - 1] {
        assert_eq!(matmul_split_k_slices(*k), Some(*slices), "k {k}");
    }
    assert_eq!(matmul_split_k_slices(600), None);
}

#[test]
fn every_split_module_passes_spirv_val() {
    let wide = rtx3090();
    let mut narrow = rtx3090();
    narrow.max_workgroup_invocations = 128;
    for profile in [&wide, &narrow] {
        let limits = MatMulLimits::of(profile);
        for (shape, slices, transposed) in CASES {
            let kernel = stage1(shape, slices, transposed, limits);
            assert_eq!(
                kernel.matmul_slices(),
                Some(slices),
                "{}",
                kernel.signature()
            );
            let module = emit_module(&kernel, profile)
                .unwrap_or_else(|error| panic!("{}: {error}", kernel.signature()));
            if let Err(error) = validate(&module.spirv) {
                panic!("{}: {error}", kernel.signature());
            }
            let sum = stage2(slices, shape.0 * shape.2);
            let module = emit_module(&sum, profile)
                .unwrap_or_else(|error| panic!("{}: {error}", sum.signature()));
            if let Err(error) = validate(&module.spirv) {
                panic!("{}: {error}", sum.signature());
            }
        }
    }
}

#[test]
fn the_emitter_refuses_a_split_result_it_cannot_cover() {
    let profile = rtx3090();
    let limits = MatMulLimits::of(&profile);
    let good = stage1((128, 4096, 128), 32, [true, false], limits);
    emit_module(&good, &profile).expect("the kernel of the schedule emits");

    // An epilogue on a split result.
    let mut kernel = good.clone();
    kernel.epilogue = Some(Epilogue::Relu {
        writes_pre_activation: false,
    });
    assert_eq!(kernel.matmul_slices(), None);
    assert!(emit_module(&signed(kernel), &profile).is_err());

    // One workgroup too few for the last slice.
    let mut kernel = good.clone();
    kernel.workgroups[0] -= 1;
    assert!(emit_module(&signed(kernel), &profile).is_err());

    // Rows that are no multiple of `m`.
    let mut kernel = good.clone();
    kernel.result = TensorType::f32(&[32 * 128 + 1, 128]);
    assert_eq!(kernel.matmul_slices(), None);
    assert!(emit_module(&signed(kernel), &profile).is_err());

    // A `k` the slices do not divide.
    let mut kernel = good.clone();
    kernel.operands = vec![
        Operand::Buffer(TensorType::f32(&[128, 4097])),
        Operand::Buffer(TensorType::f32(&[4097, 128])),
    ];
    assert_eq!(kernel.matmul_slices(), None);
    assert!(emit_module(&signed(kernel), &profile).is_err());

    // A third operand, the pre-activation of `ReluGrad`, on a split result.
    let mut kernel = good.clone();
    kernel.operands.push(Operand::Buffer(kernel.result.clone()));
    kernel.epilogue = Some(Epilogue::ReluGrad);
    assert_eq!(kernel.matmul_slices(), None);
    assert!(emit_module(&signed(kernel), &profile).is_err());
}

/// SplitMix64 values in [-1, 1), every seventh element a zero of alternating sign.
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
                _ => ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0,
            }
        })
        .collect()
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

/// Runs `kernel` once over `inputs`, its written view of `count` elements guarded and filled
/// with [`UNWRITTEN`]; returns that view whole, its guard words included.
fn run(
    device: &Device<'_>,
    kernel: &Kernel,
    spirv: &[u32],
    inputs: &[&[f32]],
    count: usize,
) -> Vec<f32> {
    assert_eq!(kernel.written(), 1);
    let uploads: Vec<Vec<f32>> = inputs
        .iter()
        .map(|values| guarded(values))
        .chain([guarded(&vec![f32::from_bits(UNWRITTEN); count])])
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
    let out = inputs.len();
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

/// What is wrong with `got`, a written view of `want.len()` elements and its guard words.
fn compare(label: &str, got: &[f32], want: &[f32], wrong: &mut Vec<String>) {
    let count = want.len();
    let differ: Vec<usize> = (0..count)
        .filter(|&i| got[i].to_bits() != want[i].to_bits())
        .collect();
    if let Some(&i) = differ.first() {
        wrong.push(format!(
            "{label}: {} of {count} elements differ, first [{i}] {:?} against {:?}",
            differ.len(),
            got[i],
            want[i]
        ));
    }
    let unwritten = (0..count)
        .filter(|&i| got[i].to_bits() == UNWRITTEN)
        .count();
    if unwritten > 0 {
        wrong.push(format!("{label}: {unwritten} elements unwritten"));
    }
    let changed = (0..GUARD_WORDS)
        .filter(|&i| got[count + i].to_bits() != guard(i))
        .count();
    if changed > 0 {
        wrong.push(format!("{label}: {changed} guard words changed"));
    }
}

fn open() -> (Vulkan, DeviceProfile) {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        std::env::var(VALIDATION_ENV).as_deref() == Ok("1")
    );
    let candidates = vulkan.candidates().expect("the devices are probed");
    let profile = candidates[0].profile.clone();
    (vulkan, profile)
}

#[test]
#[ignore = "needs a Vulkan device"]
fn both_stages_write_the_bits_of_the_slices_and_of_their_ascending_sum() {
    let (vulkan, profile) = open();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let limits = MatMulLimits::of(&profile);
    let cpu_kernel = cpu::Kernel::detected();
    let mut wrong = Vec::new();
    let mut compared = 0u64;
    for ((m, k, n), slices, transposed) in CASES {
        let case = format!("[{m}, {k}] x [{k}, {n}] in {slices} transposed {transposed:?}");
        let a = values(0x7370_6c69 ^ (m << 40) ^ (k << 20) ^ n, m * k);
        let b = values(0x6b6b_6b6b ^ (n << 40) ^ (k << 20) ^ m, k * n);
        let (mu, ku, nu) = (m as usize, k as usize, n as usize);
        let slice = ku / slices as usize;
        let count = mu * nu;
        // The partial product of each slice: the CPU kernel over its columns of `a` and its
        // rows of `b`.
        let mut partials = Vec::with_capacity(slices as usize * count);
        for s in 0..slices as usize {
            let a_slice: Vec<f32> = (0..mu)
                .flat_map(|row| a[row * ku + s * slice..row * ku + (s + 1) * slice].to_vec())
                .collect();
            let b_slice = &b[s * slice * nu..(s + 1) * slice * nu];
            partials.extend(
                cpu::matmul(cpu_kernel, &a_slice, Lhs::RowMajor, b_slice, mu, slice, nu)
                    .expect("the CPU kernel runs"),
            );
        }
        // Their sum from slice 0 in ascending order.
        let mut sum = partials[..count].to_vec();
        for s in 1..slices as usize {
            for (at, total) in sum.iter_mut().enumerate() {
                *total += partials[s * count + at];
            }
        }
        let whole = cpu::matmul(cpu_kernel, &a, Lhs::RowMajor, &b, mu, ku, nu)
            .expect("the CPU kernel runs");
        let a_bound = if transposed[0] {
            transpose(&a, m, k)
        } else {
            a.clone()
        };
        let b_bound = if transposed[1] {
            transpose(&b, k, n)
        } else {
            b.clone()
        };

        let first = stage1((m, k, n), slices, transposed, limits);
        let module = emit_module(&first, &profile).expect("the first stage emits");
        let got = run(
            &device,
            &first,
            &module.spirv,
            &[&a_bound, &b_bound],
            partials.len(),
        );
        compare(
            &format!("{case} stage 1 {}", first.matmul_tile().expect("a tile")),
            &got,
            &partials,
            &mut wrong,
        );
        compared += partials.len() as u64;

        let second = stage2(slices, m * n);
        let module = emit_module(&second, &profile).expect("the second stage emits");
        let got = run(
            &device,
            &second,
            &module.spirv,
            &[&got[..partials.len()]],
            count,
        );
        compare(&format!("{case} stage 2"), &got, &sum, &mut wrong);
        compared += count as u64;

        // Against the unsplit sum: both round at most `k` times, each by half an ulp of a
        // partial sum no larger than the sum of the products' magnitudes.
        let far: Vec<usize> = (0..count)
            .filter(|&at| {
                let (row, col) = (at / nu, at % nu);
                let magnitude: f32 = (0..ku)
                    .map(|i| (a[row * ku + i] * b[i * nu + col]).abs())
                    .sum();
                (got[at] - whole[at]).abs() > k as f32 * f32::EPSILON * magnitude
            })
            .collect();
        if let Some(&at) = far.first() {
            wrong.push(format!(
                "{case}: {} of {count} sums beyond the bound, first [{at}] {:?} against the unsplit {:?}",
                far.len(),
                got[at],
                whole[at]
            ));
        }
        for message in vulkan.take_validation_messages() {
            wrong.push(format!("{case}: {message}"));
        }
    }
    eprintln!(
        "split k: {} cases, {compared} elements compared bit for bit with the CPU {cpu_kernel:?} kernel",
        CASES.len()
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}
