//! story:bert-vulkan-kernels-rowwise on a Vulkan device, DESIGN.md § 9 gate 5: each of the
//! thirteen row-wise kernels runs once per shape and every element of its result is within
//! `GPU_ATOL + GPU_RTOL * |c|` of the CPU interpreter's `c` (`tests/common/mod.rs`), on shapes
//! whose edges do not fill a workgroup and reductions of at most 512 terms, with the guard words
//! after the output unchanged. The dropout masks are the CPU's bit for bit for three seeds and
//! two steps, the embedding gradient of repeated ids gives the same bits in two runs, and the
//! scheduled sort and gradient of `EmbeddingGrad` over repeated, absent and boundary ids of a
//! word table of BERT-tiny's shape give the CPU's bits.
//!
//! Ignored by default: they create a Vulkan instance, which `task ci` never does. Run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_bert_rowwise -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::{Draw, GPU_ATOL, GPU_RTOL, GUARD_WORDS, guard, guarded, run_within, within};
use modelspec_core::artifact::Backend;
use modelspec_core::compile::{compile_artifact, compile_artifact_for, compile_training};
use modelspec_core::ir::kernel::{Kernel, Operand};
use modelspec_core::ir::tensor::{Dim, TensorOp, TensorProgramBuilder, TensorType, ValueId};
use modelspec_core::schedule::{
    embedding_grad_kernels, layer_norm_grad_gamma_kernels, rowwise_kernel,
};
use modelspec_runtime::cpu::interpreter::evaluate_seeded;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::train::init::{dropout_keeps, dropout_token, token_id};
use modelspec_runtime::train::runner;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::runner::{device_profile, train_on_device};
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

/// The bits each output element holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_dead;

/// The dropout site whose token the cases read.
const SITE: &str = "Block.forward.dropout#0";

/// The drop probability of the dropout cases.
const P: f32 = 0.25;

/// `[n, h, v]`: rows, row width and table rows. One element; edges one short of, at and one
/// past a workgroup of 64; rows of 512 terms; and 512 rows, the longest column reduction.
const SHAPES: [(u64, u64, u64); 6] = [
    (1, 1, 1),
    (3, 5, 7),
    (63, 65, 64),
    (65, 63, 3),
    (130, 512, 129),
    (512, 33, 17),
];

/// The ops whose device result must be the CPU's bits: a copy, one rounded operation per
/// element, or rounded adds in the CPU's order.
const BIT_EQUAL: [&str; 6] = [
    "Embedding",
    "EmbeddingGrad",
    "AddRows",
    "SumRows",
    "Dropout",
    "DropoutGrad",
];

/// A buffer operand's host value.
#[derive(Clone)]
enum Data {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

/// One kernel run: the op, its operands with their names and types, their values, the result
/// type and the run key of a dropout.
struct Case {
    op: TensorOp,
    operands: Vec<(String, TensorType, Data)>,
    result: TensorType,
    seed: u64,
}

fn shape_of(ty: &TensorType) -> Vec<u64> {
    ty.shape
        .iter()
        .map(|dim| match dim {
            Dim::Static(n) => *n,
            Dim::Symbol(_) => unreachable!("a test type is static"),
        })
        .collect()
}

/// `count` ids in `0..rows`, hashed (SplitMix64) from `salt` and the index: with more ids than
/// rows, ids repeat.
fn ids(count: u64, rows: u64, salt: u64) -> Vec<i32> {
    (0..count)
        .map(|i| {
            let mut z = salt
                .wrapping_add(i.wrapping_mul(0x9e37_79b9_7f4a_7c15))
                .wrapping_add(0x9e37_79b9_7f4a_7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((z ^ (z >> 31)) % rows) as i32
        })
        .collect()
}

fn token_data(step: u64) -> Data {
    Data::I32(dropout_token(SITE, step).data().to_vec())
}

/// The thirteen cases of one shape, in the order of the story.
fn cases(n: u64, h: u64, v: u64, draw: &mut Draw) -> Vec<Case> {
    let rows = TensorType::f32(&[n, h]);
    let row = TensorType::f32(&[h]);
    let table = TensorType::f32(&[v, h]);
    let id_type = TensorType::i32(&[n]);
    let token = TensorType::i32(&[2]);
    let mut f = |name: &str, ty: &TensorType| {
        let count = shape_of(ty).iter().product();
        (name.to_owned(), ty.clone(), Data::F32(draw.values(count)))
    };
    let id_operand = (
        "ids".to_owned(),
        id_type.clone(),
        Data::I32(ids(n, v, n * 1_000 + h)),
    );
    let token_operand = (SITE.to_owned(), token.clone(), token_data(1));
    let case =
        |op: TensorOp, operands: Vec<(String, TensorType, Data)>, result: &TensorType| Case {
            op,
            operands,
            result: result.clone(),
            seed: 7,
        };
    vec![
        case(
            TensorOp::Embedding,
            vec![f("table", &table), id_operand.clone()],
            &rows,
        ),
        case(
            TensorOp::EmbeddingGrad {
                rows: Dim::Static(v),
            },
            vec![id_operand, f("g", &rows)],
            &table,
        ),
        case(TensorOp::AddRows, vec![f("x", &rows), f("b", &row)], &rows),
        case(TensorOp::SumRows, vec![f("g", &rows)], &row),
        case(TensorOp::Gelu, vec![f("x", &rows)], &rows),
        case(
            TensorOp::GeluGrad,
            vec![f("x", &rows), f("g", &rows)],
            &rows,
        ),
        case(
            TensorOp::Dropout { p: P },
            vec![f("x", &rows), token_operand.clone()],
            &rows,
        ),
        case(
            TensorOp::DropoutGrad { p: P },
            vec![token_operand, f("g", &rows)],
            &rows,
        ),
        case(
            TensorOp::LayerNorm,
            vec![f("x", &rows), f("gamma", &row), f("beta", &row)],
            &rows,
        ),
        case(
            TensorOp::LayerNormGradX,
            vec![f("x", &rows), f("gamma", &row), f("g", &rows)],
            &rows,
        ),
        case(
            TensorOp::LayerNormGradGamma,
            vec![f("x", &rows), f("g", &rows)],
            &row,
        ),
        case(TensorOp::Softmax, vec![f("x", &rows)], &rows),
        case(
            TensorOp::SoftmaxGrad,
            vec![f("y", &rows), f("g", &rows)],
            &rows,
        ),
    ]
}

/// The CPU interpreter's result of `case`: a program of its one op over its operands as inputs.
fn cpu(case: &Case) -> Vec<f32> {
    let mut builder = TensorProgramBuilder::new();
    let mut inputs = BTreeMap::new();
    let mut ids = BTreeMap::new();
    let mut values: Vec<ValueId> = Vec::new();
    for (name, ty, data) in &case.operands {
        let shape = shape_of(ty);
        match data {
            Data::F32(values) => {
                inputs.insert(
                    name.clone(),
                    HostTensor::new(shape, values.clone()).unwrap(),
                );
            }
            Data::I32(values) => {
                ids.insert(
                    name.clone(),
                    HostTensor::new(shape, values.clone()).unwrap(),
                );
            }
        }
        values.push(
            builder
                .input(name.clone(), ty.clone())
                .expect("a new input"),
        );
    }
    let v = &values;
    let result = match &case.op {
        TensorOp::Embedding => builder.embedding(v[0], v[1]),
        TensorOp::EmbeddingGrad { rows } => builder.embedding_grad(v[0], v[1], *rows),
        TensorOp::AddRows => builder.add_rows(v[0], v[1]),
        TensorOp::SumRows => builder.sum_rows(v[0]),
        TensorOp::Gelu => builder.gelu(v[0]),
        TensorOp::GeluGrad => builder.gelu_grad(v[0], v[1]),
        TensorOp::Dropout { p } => builder.dropout(v[0], v[1], *p),
        TensorOp::DropoutGrad { p } => builder.dropout_grad(v[0], v[1], *p),
        TensorOp::LayerNorm => builder.layer_norm(v[0], v[1], v[2]),
        TensorOp::LayerNormGradX => builder.layer_norm_grad_x(v[0], v[1], v[2]),
        TensorOp::LayerNormGradGamma => builder.layer_norm_grad_gamma(v[0], v[1]),
        TensorOp::Softmax => builder.softmax(v[0]),
        TensorOp::SoftmaxGrad => builder.softmax_grad(v[0], v[1]),
        other => unreachable!("{} is no row-wise case", other.name()),
    }
    .expect("the case is well typed");
    builder.output(result).expect("the result is a value");
    let program = builder.finish();
    evaluate_seeded(&program, &inputs, &ids, &BTreeMap::new(), case.seed)
        .expect("the CPU evaluates")[0]
        .data()
        .to_vec()
}

/// The kernels of `case`, as the schedule builds them: one for every op but `EmbeddingGrad`,
/// which is a sort and a gradient, and `LayerNormGradGamma`, which is the row moments and the
/// column sum.
fn kernels_of(case: &Case) -> Vec<Kernel> {
    let operands = case
        .operands
        .iter()
        .map(|(_, ty, _)| Operand::Buffer(ty.clone()))
        .collect();
    if matches!(case.op, TensorOp::EmbeddingGrad { .. }) {
        let kernels = embedding_grad_kernels(operands, case.result.clone());
        return Vec::from(kernels.expect("EmbeddingGrad has its two kernels"));
    }
    if case.op == TensorOp::LayerNormGradGamma {
        let kernels = layer_norm_grad_gamma_kernels(operands, case.result.clone());
        return Vec::from(kernels.expect("LayerNormGradGamma has its two kernels"));
    }
    vec![
        rowwise_kernel(&case.op, operands, case.result.clone())
            .unwrap_or_else(|| panic!("{} has a row-wise kernel", case.op.name())),
    ]
}

fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

/// Runs the kernels of `case` once on `device` ([`run`]).
fn gpu(device: &Device<'_>, case: &Case) -> Vec<f32> {
    run(device, case, &kernels_of(case))
}

/// Runs `kernels` over the operands of `case` once on `device`: uploads its operands and the
/// guarded output, dispatches, reads back, in one command buffer with one host wait. Returns the
/// output with its guard words. Two kernels run as the step plan runs a two-stage work: stage 1
/// reads the first operands and writes an intermediate buffer, and after a barrier stage 2 reads
/// it, then the last operands its kernel names.
fn run(device: &Device<'_>, case: &Case, kernels: &[Kernel]) -> Vec<f32> {
    let pipelines: Vec<_> = kernels
        .iter()
        .map(|kernel| {
            let words = emit(kernel).expect("the kernel emits");
            if matches!(
                case.op,
                TensorOp::Dropout { .. } | TensorOp::DropoutGrad { .. }
            ) {
                device.kernel_pipeline_seeded(kernel, &words, case.seed)
            } else {
                device.kernel_pipeline(kernel, &words)
            }
            .expect("the pipeline is created")
        })
        .collect();
    let result: u64 = shape_of(&case.result).iter().product();
    let mut uploads: Vec<Data> = case.operands.iter().map(|(_, _, d)| d.clone()).collect();
    uploads.push(Data::F32(guarded(&vec![
        f32::from_bits(UNWRITTEN);
        result as usize
    ])));
    let lengths: Vec<usize> = uploads
        .iter()
        .map(|data| match data {
            Data::F32(v) => v.len(),
            Data::I32(v) => v.len(),
        })
        .collect();
    let buffers: Vec<Buffer<'_>> = lengths
        .iter()
        .map(|&n| device.device_local_buffer(bytes_of(n)).expect("a buffer"))
        .collect();
    let mut staging: Vec<Buffer<'_>> = uploads
        .iter()
        .zip(&lengths)
        .map(|(data, &n)| {
            let mut buffer = device
                .staging_buffer(bytes_of(n))
                .expect("a staging buffer");
            match data {
                Data::F32(v) => buffer.write_f32(v),
                Data::I32(v) => buffer.write_i32(v),
            }
            .expect("the staging buffer is written");
            buffer
        })
        .collect();
    let output = buffers.len() - 1;
    let buffer_operands = |kernel: &Kernel| {
        kernel
            .operands
            .iter()
            .filter(|o| matches!(o, Operand::Buffer(_)))
            .count()
    };
    let between = match kernels {
        [_] => None,
        [first, _] => {
            let n: u64 = shape_of(&first.result).iter().product();
            Some(
                device
                    .device_local_buffer(bytes_of(n as usize))
                    .expect("a buffer"),
            )
        }
        _ => panic!("one kernel or two stages"),
    };
    let mut recording = device.record().expect("a recording starts");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("the upload is recorded");
    }
    recording.barrier(Barrier::TransferToCompute);
    match (kernels, &between) {
        ([kernel], None) => {
            let bound: Vec<&Buffer<'_>> = buffers.iter().collect();
            recording
                .dispatch_kernel(&pipelines[0], kernel, &bound)
                .expect("the dispatch is recorded");
        }
        ([first, second], Some(between)) => {
            let reads = buffer_operands(first);
            let bound: Vec<&Buffer<'_>> = buffers[..reads].iter().chain([between]).collect();
            recording
                .dispatch_kernel(&pipelines[0], first, &bound)
                .expect("stage 1 is recorded");
            recording.barrier(Barrier::ComputeToCompute);
            let rest = buffer_operands(second) - 1;
            let bound: Vec<&Buffer<'_>> = [between]
                .into_iter()
                .chain(&buffers[output - rest..])
                .collect();
            recording
                .dispatch_kernel(&pipelines[1], second, &bound)
                .expect("stage 2 is recorded");
        }
        _ => unreachable!("an intermediate buffer exactly for two stages"),
    }
    recording.barrier(Barrier::ComputeToTransfer);
    recording
        .copy(&buffers[output], &staging[output])
        .expect("the download is recorded");
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    device
        .submit(recorded)
        .expect("submitted")
        .wait()
        .expect("the device finishes");
    staging[output].read_f32().expect("the output reads")
}

fn open() -> Vulkan {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    assert_eq!(
        vulkan.validation_enabled(),
        std::env::var(VALIDATION_ENV).as_deref() == Ok("1"),
        "the validation layer is on exactly when {VALIDATION_ENV}=1"
    );
    vulkan
}

/// What one op showed over every shape.
#[derive(Default)]
struct Observed {
    kind: String,
    workgroup_size: u32,
    elements: usize,
    max_abs: f64,
    max_rel: f64,
    bits_equal: bool,
    outside: Vec<String>,
}

#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_rowwise_kernels_match_the_cpu_interpreter_within_tolerance_with_guard_words_unchanged() {
    let vulkan = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let mut draw = Draw::new(0x726f_7777_6973_6521);
    let mut observed: Vec<(String, Observed)> = Vec::new();
    let mut messages = Vec::new();
    for (n, h, v) in SHAPES {
        for case in cases(n, h, v, &mut draw) {
            let name = case.op.name().to_owned();
            let kernel = kernels_of(&case).pop().expect("a kernel writes the result");
            let expected = cpu(&case);
            let got = gpu(&device, &case);
            let at = match observed.iter().position(|(op, _)| *op == name) {
                Some(at) => at,
                None => {
                    observed.push((
                        name.clone(),
                        Observed {
                            kind: kernel.kind.to_string(),
                            workgroup_size: kernel.workgroup_size,
                            bits_equal: true,
                            ..Observed::default()
                        },
                    ));
                    observed.len() - 1
                }
            };
            let seen = &mut observed[at].1;
            let count = expected.len();
            seen.elements += count;
            for (i, (&g, &c)) in got.iter().zip(&expected).enumerate() {
                let error = (f64::from(g) - f64::from(c)).abs();
                if g.to_bits() != c.to_bits() {
                    seen.bits_equal = false;
                }
                seen.max_abs = seen.max_abs.max(error);
                if c != 0.0 {
                    seen.max_rel = seen.max_rel.max(error / f64::from(c).abs());
                }
                if !within(g, c, c.abs(), GPU_ATOL, GPU_RTOL) && seen.outside.len() < 4 {
                    seen.outside.push(format!(
                        "{name} [{n}, {h}] v {v} [{i}]: gpu {g:?} cpu {c:?}"
                    ));
                }
            }
            for i in 0..GUARD_WORDS {
                if got[count + i].to_bits() != guard(i) {
                    seen.outside
                        .push(format!("{name} [{n}, {h}] v {v}: guard word {i} changed"));
                }
            }
            for message in vulkan.take_validation_messages() {
                messages.push(format!("{name} [{n}, {h}]: {message}"));
            }
        }
    }
    eprintln!("op | kind | workgroup size | elements | max |g-c| | max rel | bits equal");
    for (op, seen) in &observed {
        eprintln!(
            "{op} | {} | {} | {} | {:.3e} | {:.3e} | {}",
            seen.kind,
            seen.workgroup_size,
            seen.elements,
            seen.max_abs,
            seen.max_rel,
            if seen.bits_equal { "yes" } else { "no" }
        );
    }
    let outside: Vec<&String> = observed.iter().flat_map(|(_, s)| &s.outside).collect();
    assert!(outside.is_empty(), "{outside:#?}");
    assert!(messages.is_empty(), "{messages:#?}");
    assert_eq!(observed.len(), 13);
    for (op, seen) in &observed {
        if BIT_EQUAL.contains(&op.as_str()) {
            assert!(seen.bits_equal, "{op} is the CPU's bits");
        }
    }
}

/// The dropout masks of three seeds and two steps: `Dropout` of ones gives `1 / (1 - p)` exactly
/// where the CPU's Philox draw keeps the element and `+0` elsewhere, and `Dropout` and
/// `DropoutGrad` of drawn values are the CPU's bits.
#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_dropout_masks_are_the_cpu_masks_bit_for_bit_for_three_seeds_and_two_steps() {
    let vulkan = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let (n, h) = (65_u64, 513_u64);
    let rows = TensorType::f32(&[n, h]);
    let token = TensorType::i32(&[2]);
    let count = (n * h) as usize;
    let scale = 1.0f32 / (1.0f32 - P);
    let mut draw = Draw::new(0x6472_6f70_6f75_7421);
    let mut masks = 0;
    let mut kept = 0;
    for seed in [7_u64, 0x1234_5678_9abc_def0, u64::MAX] {
        for step in [0_u64, 1] {
            let id = token_id([
                dropout_token(SITE, step).data()[0],
                dropout_token(SITE, step).data()[1],
            ]);
            let ones = Case {
                op: TensorOp::Dropout { p: P },
                operands: vec![
                    ("x".to_owned(), rows.clone(), Data::F32(vec![1.0; count])),
                    (SITE.to_owned(), token.clone(), token_data(step)),
                ],
                result: rows.clone(),
                seed,
            };
            let got = gpu(&device, &ones);
            let cpu_mask = cpu(&ones);
            for i in 0..count {
                let keeps = dropout_keeps(seed, id, i as u32, P);
                let want = if keeps { scale } else { 0.0 };
                assert_eq!(
                    got[i].to_bits(),
                    want.to_bits(),
                    "seed {seed:#x} step {step} [{i}]"
                );
                assert_eq!(cpu_mask[i].to_bits(), want.to_bits());
                kept += usize::from(keeps);
            }
            masks += 1;
            let values = draw.values(n * h);
            for op in [TensorOp::Dropout { p: P }, TensorOp::DropoutGrad { p: P }] {
                let x = ("x".to_owned(), rows.clone(), Data::F32(values.clone()));
                let t = (SITE.to_owned(), token.clone(), token_data(step));
                let operands = match op {
                    TensorOp::Dropout { .. } => vec![x, t],
                    _ => vec![t, x],
                };
                let case = Case {
                    op,
                    operands,
                    result: rows.clone(),
                    seed,
                };
                let got = gpu(&device, &case);
                let want = cpu(&case);
                let differ = (0..count)
                    .filter(|&i| got[i].to_bits() != want[i].to_bits())
                    .count();
                assert_eq!(differ, 0, "{} seed {seed:#x} step {step}", case.op.name());
            }
        }
    }
    let messages = vulkan.take_validation_messages();
    eprintln!(
        "vulkan dropout: {masks} masks of {count} elements bit-equal, {kept} kept of {} at p {P}",
        masks * count
    );
    assert!(messages.is_empty(), "{messages:#?}");
    assert_eq!(masks, 6);
}

/// `EmbeddingGrad` of 512 ids into 3 rows, about 170 contributions per row: two runs give the
/// same bits, and those are the CPU's.
#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_embedding_grad_of_repeated_ids_gives_the_same_bits_in_two_runs() {
    let vulkan = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let (n, h, v) = (512_u64, 65_u64, 3_u64);
    let mut draw = Draw::new(0x656d_6265_6464_2121);
    let case = Case {
        op: TensorOp::EmbeddingGrad {
            rows: Dim::Static(v),
        },
        operands: vec![
            (
                "ids".to_owned(),
                TensorType::i32(&[n]),
                Data::I32(ids(n, v, 99)),
            ),
            (
                "g".to_owned(),
                TensorType::f32(&[n, h]),
                Data::F32(draw.values(n * h)),
            ),
        ],
        result: TensorType::f32(&[v, h]),
        seed: 0,
    };
    let first = gpu(&device, &case);
    let second = gpu(&device, &case);
    let want = cpu(&case);
    let count = (v * h) as usize;
    let bits = |values: &[f32]| {
        values[..count]
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&first), bits(&second), "two runs, the same bits");
    assert_eq!(bits(&first), bits(&want), "the CPU's bits");
    let messages = vulkan.take_validation_messages();
    eprintln!("vulkan embedding_grad: {n} ids into {v} rows of {h}, two runs bit-equal");
    assert!(messages.is_empty(), "{messages:#?}");
}

/// A word table of BERT-tiny's shape, 8,192 rows of 128, and a batch of 4,096 ids.
const WORDS: &str = "model Words {
    param table: Tensor<f32, [8192, 128]> = xavier_uniform();

    fn forward(ids: Tensor<i32, [4096]>) -> Tensor<f32, [4096, 128]> {
        return embedding(table, ids);
    }
}

train Fit {
    model = Words;
    input ids: Tensor<i32, [4096]>;
    target y: Tensor<f32, [4096, 128]>;
    loss = mse(model.forward(ids), y);
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 7;
    device = cpu;
}
";

/// `n` ids into a `v`-row table with repeated, absent and boundary ids: every 16th position
/// holds id 0, every 16th from 5 the last id `v - 1` and every 16th from 9 id `v / 2`; position
/// 1 holds `v / 2 + 1` and position `n - 1` holds `v - 2`, each once; the rest cycle through
/// ids 1 to 97 in a descending order, so equal ids sit at positions far apart and ids fall as
/// positions rise. Every other row is absent.
fn boundary_ids(n: u64, v: u64) -> Vec<i32> {
    (0..n)
        .map(|i| {
            let id = match (i, i % 16) {
                (1, _) => v / 2 + 1,
                (i, _) if i == n - 1 => v - 2,
                (_, 0) => 0,
                (_, 5) => v - 1,
                (_, 9) => v / 2,
                _ => 1 + ((n - i) * 37) % 97,
            };
            id as i32
        })
        .collect()
}

/// story:vulkan-embedding-grad-sorted-ids: the step plan computes `EmbeddingGrad` of BERT-tiny's
/// word table as two kernels, a sort of the batch's ids by id and position (`per_row`, into
/// `i32[4096, 2]`) and the gradient over the sorted ids (`per_element`), and their result over
/// repeated, absent and boundary ids (0 and 8,191) is the CPU interpreter's bit for bit, every
/// absent row `+0.0`, with the guard words after it unchanged.
#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_embedding_grad_sorts_the_ids_and_gives_the_cpu_bits_on_repeated_absent_and_boundary_ids()
{
    let (n, h, v) = (4096_u64, 128_u64, 8192_u64);
    let artifact = compile_artifact("words.mspec", WORDS, Backend::Vulkan)
        .unwrap_or_else(|errors| panic!("the word table compiles for Vulkan: {errors:#?}"));
    let kernels: Vec<Kernel> = artifact
        .execution
        .kernels
        .iter()
        .filter(|k| k.op == "EmbeddingGrad")
        .cloned()
        .collect();
    let vulkan = open();
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    let ids = boundary_ids(n, v);
    let mut draw = Draw::new(0x736f_7274_6564_2121);
    let case = Case {
        op: TensorOp::EmbeddingGrad {
            rows: Dim::Static(v),
        },
        operands: vec![
            (
                "ids".to_owned(),
                TensorType::i32(&[n]),
                Data::I32(ids.clone()),
            ),
            (
                "g".to_owned(),
                TensorType::f32(&[n, h]),
                Data::F32(draw.values(n * h)),
            ),
        ],
        result: TensorType::f32(&[v, h]),
        seed: 0,
    };
    let got = run(&device, &case, &kernels);
    let want = cpu(&case);
    let count = (v * h) as usize;
    let differ: Vec<usize> = (0..count)
        .filter(|&i| got[i].to_bits() != want[i].to_bits())
        .collect();
    let present: BTreeSet<i32> = ids.iter().copied().collect();
    let absent = (0..v as i32).filter(|id| !present.contains(id)).count();
    let absent_zero = (0..v as usize)
        .filter(|&r| !present.contains(&(r as i32)))
        .all(|r| {
            got[r * h as usize..(r + 1) * h as usize]
                .iter()
                .all(|x| x.to_bits() == 0)
        });
    let guards = (0..GUARD_WORDS).all(|i| got[count + i].to_bits() == guard(i));
    let messages = vulkan.take_validation_messages();
    let kinds: Vec<String> = kernels.iter().map(|k| k.kind.to_string()).collect();
    eprintln!(
        "vulkan embedding_grad: {n} ids, {} distinct, {absent} absent rows of {v}, kernels {kinds:?}, \
         {} elements differ",
        present.len(),
        differ.len()
    );
    assert!(
        differ.is_empty(),
        "{} elements differ, first at {:?}",
        differ.len(),
        differ
            .iter()
            .take(4)
            .map(|&i| (i / h as usize, i % h as usize, got[i], want[i]))
            .collect::<Vec<_>>()
    );
    assert!(absent_zero, "every absent row is +0.0");
    assert!(guards, "the guard words are unchanged");
    assert!(messages.is_empty(), "{messages:#?}");
    assert!(present.contains(&0) && present.contains(&(v as i32 - 1)));
    assert_eq!(
        kinds,
        ["per_row", "per_element"],
        "the sort, then the gradient over the sorted ids"
    );
    assert_eq!(kernels[0].result, TensorType::i32(&[n, 2]));
    assert_eq!(
        kernels[1].operands[0],
        Operand::Buffer(TensorType::i32(&[n, 2]))
    );
}

/// A model of matmul, `add_rows` and `gelu` trains on the device: `dense_gelu_train.mspec`,
/// whose every parameter ends within the whole-run bound of the CPU run's.
const DENSE_GELU: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../modelspec-cli/tests/fixtures/dense_gelu_train.mspec"
));

#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_trains_a_matmul_add_rows_gelu_model_to_the_cpu_run() {
    let profile = device_profile(None).expect("device 0");
    let training = compile_training("dense_gelu_train.mspec", DENSE_GELU).expect("compiles");
    let artifact = compile_artifact_for(
        "dense_gelu_train.mspec",
        DENSE_GELU,
        Backend::Vulkan,
        Some(&profile),
    )
    .unwrap_or_else(|d| panic!("{d:?}"));
    let (device_run, _) = train_on_device(None, &training, &artifact).expect("runs");
    let cpu_run = runner::run(&training).expect("the CPU runs");
    assert_eq!(device_run.losses.len(), cpu_run.losses.len());
    for (param, cpu_value) in &cpu_run.state.values {
        let device_value = &device_run.state.values[param];
        assert!(
            run_within(device_value.data(), cpu_value.data()),
            "{param:?}: {:?} vs {:?}",
            device_value.data(),
            cpu_value.data()
        );
    }
    let last = |losses: &[f32]| *losses.last().expect("a loss");
    eprintln!(
        "vulkan dense_gelu_train: {} losses, final {} on the device, {} on the CPU",
        device_run.losses.len(),
        last(&device_run.losses),
        last(&cpu_run.losses)
    );
    assert!(within(
        last(&device_run.losses),
        last(&cpu_run.losses),
        last(&cpu_run.losses).abs(),
        GPU_ATOL,
        GPU_RTOL
    ));
}
