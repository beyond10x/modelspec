//! Adversary cases for story:matmul-reads-transposed-operand on a Vulkan device.
//!
//! 1. Partial tiles on the transposed operand: a matmul kernel reading `a`, `b` or both
//!    transposed, bound to exactly-sized buffers holding the row-major transposes, writes the
//!    bits of the CPU oracle (one sum per element from the `k = 0` product in ascending `k`) at
//!    `m`, `k`, `n` of 1, 7, 63, 65, 127, 129 and at partial blocks of every tile family, with
//!    no element left unwritten and no validation message.
//! 2. The fold rule on sources the implementor's cases do not train: a transpose read by an
//!    `add`, a prediction that is a transpose read only by a matmul of the loss, a transpose
//!    read by two matmuls one with the `relu` epilogue, a chain of transposes, and a matmul of
//!    two transposed inputs at the one-row, wide and square tiles. Each trains on the device
//!    within the whole-run bound and the `mse` tolerance of the CPU run of the same file.
//!
//! Ignored by default: they create a Vulkan instance. Run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test adversary_transposed_device -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use common::{Draw, KernelOp, run_within, term_tolerance};
use modelspec_core::artifact::Backend;
use modelspec_core::compile::{compile_artifact, compile_training};
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{Kernel, KernelKind, KernelModuleId, MatMulTile, Operand};
use modelspec_core::ir::tensor::{DType, TensorType};
use modelspec_runtime::spirv::emit::emit_module;
use modelspec_runtime::train::runner;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::runner::train;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

/// The bits each output element holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_beef;

/// A source to train, by name, and the tile family `(rows, cols_per_invocation)` of its one
/// kernel that flags both operands, when it has one.
type FoldCase = (&'static str, String, Option<(u32, u32)>);

/// `(m, k, n)`: the edges the brief names and a partial block of every tile family.
const SHAPES: [(u64, u64, u64); 15] = [
    (1, 1, 1),
    (1, 7, 129),
    (7, 1, 63),
    (63, 65, 127),
    (129, 63, 65),
    (127, 129, 1),
    (65, 127, 129),
    (2049, 129, 7),
    (2049, 65, 13),
    (4097, 7, 63),
    (1031, 63, 1029),
    (1031, 1, 1029),
    (1025, 65, 1031),
    (1025, 129, 1031),
    (1088, 300, 1040),
];

fn matmul(m: u64, k: u64, n: u64, transposed: [bool; 2]) -> Kernel {
    let tile = MatMulTile::of(m, k, n);
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
        epilogue: None,
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

/// The CPU oracle of DESIGN.md § 5.2: each element one sum from the `k = 0` product, ascending.
fn oracle(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = a[i * k] * b[j];
            for p in 1..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    out
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

/// Runs `kernel` once over `inputs` in buffers of exactly their size and returns the output,
/// filled with [`UNWRITTEN`] before the dispatch.
fn run(device: &Device<'_>, kernel: &Kernel, spirv: &[u32], inputs: [&[f32]; 2]) -> Vec<f32> {
    let written = vec![f32::from_bits(UNWRITTEN); output_len(kernel)];
    let uploads: Vec<&[f32]> = vec![inputs[0], inputs[1], &written];
    let bytes = |values: &[f32]| (values.len() * 4) as u64;
    let buffers: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|values| {
            device
                .device_local_buffer(bytes(values))
                .expect("a device buffer")
        })
        .collect();
    let mut staging: Vec<Buffer<'_>> = uploads
        .iter()
        .map(|values| {
            let mut buffer = device.staging_buffer(bytes(values)).expect("staging");
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
    recording
        .copy(&buffers[2], &staging[2])
        .expect("the download is recorded");
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    device
        .submit(recorded)
        .expect("submitted")
        .wait()
        .expect("the device finishes");
    staging[2].read_f32().expect("the output reads")
}

fn output_len(kernel: &Kernel) -> usize {
    let (m, _, n) = Kernel::matmul_shape(&kernel.operands).expect("a matmul");
    (m * n) as usize
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
fn a_transposed_operand_in_an_exactly_sized_buffer_gives_the_cpu_oracle_bits_at_partial_tiles() {
    let (vulkan, profile) = open();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let mut wrong = Vec::new();
    let mut compared = 0u64;
    for (m, k, n) in SHAPES {
        let tile = MatMulTile::of(m, k, n);
        let (mu, ku, nu) = (m as usize, k as usize, n as usize);
        let mut draw = Draw::new(0x6164_7631 ^ (m << 32) ^ (k << 16) ^ n);
        let a = draw.values(m * k);
        let b = draw.values(k * n);
        let want = oracle(&a, &b, mu, ku, nu);
        let (a_t, b_t) = (transpose(&a, mu, ku), transpose(&b, ku, nu));
        for transposed in [[false, false], [true, false], [false, true], [true, true]] {
            let case = format!("[{m}, {k}] x [{k}, {n}] {tile} transposed {transposed:?}");
            let kernel = matmul(m, k, n, transposed);
            let module = emit_module(&kernel, &profile).expect("the kernel emits");
            let a_bound: &[f32] = if transposed[0] { &a_t } else { &a };
            let b_bound: &[f32] = if transposed[1] { &b_t } else { &b };
            let got = run(&device, &kernel, &module.spirv, [a_bound, b_bound]);
            compared += got.len() as u64;
            let differ: Vec<usize> = (0..got.len())
                .filter(|&i| got[i].to_bits() != want[i].to_bits())
                .collect();
            if let Some(&i) = differ.first() {
                wrong.push(format!(
                    "{case}: {} of {} differ, first [{}, {}] {:?} against {:?}",
                    differ.len(),
                    got.len(),
                    i / nu,
                    i % nu,
                    got[i],
                    want[i]
                ));
            }
            for message in vulkan.take_validation_messages() {
                wrong.push(format!("{case}: {message}"));
            }
        }
    }
    eprintln!(
        "validation {}; {} shapes x 4, {compared} elements against the CPU oracle",
        vulkan.validation_enabled(),
        SHAPES.len()
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}

const ADD_READER: &str = "model AddReader {
    param w: Tensor<f32, [10, 3]> = xavier_uniform();

    fn forward(p: Tensor<f32, [10, 12]>) -> Tensor<f32, [12, 3]> {
        let t = transpose(p);
        return add(matmul(t, w), matmul(add(t, t), w));
    }
}

train Fit {
    model = AddReader;
    input p: Tensor<f32, [10, 12]>;
    target y: Tensor<f32, [12, 3]>;
    loss = mse(model.forward(p), y);
    optimizer = sgd(lr = 0.05);
    steps = 40;
    seed = 11;
    device = vulkan(0);
}
";

/// The prediction is a transpose read in the step only by the loss's matmul (synthetic data takes
/// one input and a target of the output type, so the loss multiplies by `x · xᵀ`).
const PREDICTION_TRANSPOSE: &str = "model PredT {
    param w: Tensor<f32, [12, 3]> = xavier_uniform();

    fn forward(x: Tensor<f32, [10, 12]>) -> Tensor<f32, [3, 10]> {
        return transpose(matmul(x, w));
    }
}

train Fit {
    model = PredT;
    input x: Tensor<f32, [10, 12]>;
    target y: Tensor<f32, [3, 10]>;
    loss = mse(matmul(model.forward(x), matmul(x, transpose(x))), y);
    optimizer = sgd(lr = 0.00001);
    steps = 40;
    seed = 12;
    device = vulkan(0);
}
";

const TWO_READERS: &str = "model TwoReaders {
    param v: Tensor<f32, [4, 12]> = xavier_uniform();

    fn forward(a: Tensor<f32, [10, 12]>) -> Tensor<f32, [10, 4]> {
        let t = transpose(v);
        return add(relu(matmul(a, t)), matmul(mul(a, a), t));
    }
}

train Fit {
    model = TwoReaders;
    input a: Tensor<f32, [10, 12]>;
    target y: Tensor<f32, [10, 4]>;
    loss = mse(model.forward(a), y);
    optimizer = sgd(lr = 0.02);
    steps = 40;
    seed = 13;
    device = vulkan(0);
}
";

const CHAIN: &str = "model Chain {
    param w: Tensor<f32, [10, 3]> = xavier_uniform();

    fn forward(a: Tensor<f32, [10, 12]>) -> Tensor<f32, [12, 3]> {
        return matmul(transpose(transpose(transpose(a))), w);
    }
}

train Fit {
    model = Chain;
    input a: Tensor<f32, [10, 12]>;
    target y: Tensor<f32, [12, 3]>;
    loss = mse(model.forward(a), y);
    optimizer = sgd(lr = 0.05);
    steps = 40;
    seed = 14;
    device = vulkan(0);
}
";

/// `[m, k] x [k, m]` with both operands folded transposes of computed values of the one input
/// `x: [k, m]`: `transpose(relu(x))` and `transpose(relu(transpose(x)))`.
fn both(m: u64, k: u64, out: u64, steps: u64) -> String {
    format!(
        "model Both {{
    param w: Tensor<f32, [{m}, {out}]> = xavier_uniform();

    fn forward(x: Tensor<f32, [{k}, {m}]>) -> Tensor<f32, [{m}, {out}]> {{
        return matmul(matmul(transpose(relu(x)), transpose(relu(transpose(x)))), w);
    }}
}}

train Fit {{
    model = Both;
    input x: Tensor<f32, [{k}, {m}]>;
    target y: Tensor<f32, [{m}, {out}]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.00001);
    steps = {steps};
    seed = 15;
    device = vulkan(0);
}}
"
    )
}

#[test]
#[ignore = "needs a Vulkan device"]
fn each_fold_case_trains_on_the_device_within_tolerance_of_the_cpu() {
    let (vulkan, profile) = open();
    let negotiated = negotiate(&profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    // The tile family `(rows, cols_per_invocation)` of the kernel that flags both operands.
    let cases: Vec<FoldCase> = vec![
        ("add_reader", ADD_READER.to_owned(), None),
        (
            "prediction_transpose",
            PREDICTION_TRANSPOSE.to_owned(),
            None,
        ),
        ("two_readers", TWO_READERS.to_owned(), None),
        ("chain", CHAIN.to_owned(), None),
        ("both one-row", both(10, 12, 5, 40), Some((1, 1))),
        ("both wide", both(1024, 96, 3, 3), Some((8, 8))),
        ("both square", both(1031, 7, 3, 3), Some((8, 4))),
    ];
    let mut wrong = Vec::new();
    for (name, source, both_tile) in cases {
        let compiled = compile_training(name, &source).expect("compiles");
        let artifact = compile_artifact(name, &source, Backend::Vulkan).expect("compiles");
        let flagged = artifact
            .execution
            .kernels
            .iter()
            .filter(|k| k.transposed.contains(&true))
            .count();
        if let Some(family) = both_tile {
            let tiles: Vec<(u32, u32)> = artifact
                .execution
                .kernels
                .iter()
                .filter(|k| k.transposed == [true, true])
                .map(|k| {
                    let (m, k, n) = Kernel::matmul_shape(&k.operands).expect("a matmul");
                    let tile = MatMulTile::of(m, k, n);
                    (tile.rows, tile.cols_per_invocation)
                })
                .collect();
            assert_eq!(tiles, [family], "{name}: one kernel flags both operands");
        }
        let cpu = runner::run(&compiled).expect("the CPU run finishes");
        let (gpu, _) = train(&device, &profile, &negotiated, &compiled, &artifact)
            .expect("the device run finishes");
        for message in vulkan.take_validation_messages() {
            wrong.push(format!("{name}: {message}"));
        }
        assert_eq!(gpu.initial, cpu.initial, "{name}: the same start");
        for (param, c) in &cpu.state.values {
            let g = &gpu.state.values[param];
            if !run_within(g.data(), c.data()) {
                wrong.push(format!("{name}: {param:?} outside the whole-run bound"));
            }
        }
        let tolerance = term_tolerance(KernelOp::Mse);
        let outside: Vec<(usize, f32, f32)> = gpu
            .losses
            .iter()
            .zip(&cpu.losses)
            .enumerate()
            .filter(|(_, (g, c))| !tolerance.accepts(**g, **c, tolerance.r(**c, f64::from(**c), 1)))
            .map(|(i, (g, c))| (i, *g, *c))
            .collect();
        if !outside.is_empty() {
            wrong.push(format!(
                "{name}: {} losses outside, first {:?}",
                outside.len(),
                outside[0]
            ));
        }
        eprintln!(
            "{name}: {flagged} flagged kernels, {} losses, last cpu {} gpu {}",
            cpu.losses.len(),
            cpu.losses.last().unwrap(),
            gpu.losses.last().unwrap()
        );
    }
    eprintln!("validation {}", vulkan.validation_enabled());
    assert!(wrong.is_empty(), "{wrong:#?}");
}
