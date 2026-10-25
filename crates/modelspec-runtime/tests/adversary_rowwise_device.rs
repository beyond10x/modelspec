//! Adversary pass 1 on story:bert-vulkan-kernels-rowwise, on a Vulkan device: the thirteen
//! row-wise kernels on shapes, values and drop probabilities the unit's device suite does not
//! run, against the CPU interpreter, with DESIGN.md § 9 gate 5's `within` and the bit claims of
//! DESIGN.md revision 0.46 (copies, sums of single rounded adds and dropout masks bit for bit).
//!
//! Ignored by default: they create a Vulkan instance. Run them with
//! `cargo test -p modelspec-runtime --features vulkan --test adversary_rowwise_device -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;
use std::time::Instant;

use common::{GPU_ATOL, GPU_RTOL, GUARD_WORDS, guard, guarded, within};
use modelspec_core::ir::kernel::{Kernel, Operand};
use modelspec_core::ir::tensor::{Dim, TensorOp, TensorProgramBuilder, TensorType, ValueId};
use modelspec_core::schedule::rowwise_kernel;
use modelspec_runtime::cpu::interpreter::evaluate_seeded;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::emit;
use modelspec_runtime::train::init::{dropout_keeps, dropout_token, token_id};
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, Vulkan};

const UNWRITTEN: u32 = 0x7fc0_dead;

const SITES: [&str; 3] = [
    "Block.forward.dropout#0",
    "Block.forward.dropout#1",
    "Block.forward@1.dropout#0",
];

const BIT_EQUAL: [&str; 6] = [
    "Embedding",
    "EmbeddingGrad",
    "AddRows",
    "SumRows",
    "Dropout",
    "DropoutGrad",
];

#[derive(Clone)]
enum Data {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

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

/// SplitMix64 values.
struct Gen(u64);

impl Gen {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn uniform(&mut self, count: u64, lo: f32, hi: f32) -> Vec<f32> {
        (0..count).map(|_| lo + (hi - lo) * self.unit()).collect()
    }

    fn ids(&mut self, count: u64, rows: u64) -> Vec<i32> {
        (0..count).map(|_| (self.next() % rows) as i32).collect()
    }
}

fn f(name: &str, ty: &TensorType, values: Vec<f32>) -> (String, TensorType, Data) {
    (name.to_owned(), ty.clone(), Data::F32(values))
}

fn token(site: &str, step: u64) -> (String, TensorType, Data) {
    (
        site.to_owned(),
        TensorType::i32(&[2]),
        Data::I32(dropout_token(site, step).data().to_vec()),
    )
}

/// The thirteen cases of one `[n, h]` shape and a `v`-row table, operands uniform in
/// `[lo, hi)`.
fn cases(n: u64, h: u64, v: u64, rng: &mut Gen, lo: f32, hi: f32) -> Vec<Case> {
    let rows = TensorType::f32(&[n, h]);
    let row = TensorType::f32(&[h]);
    let table = TensorType::f32(&[v, h]);
    let ids = (
        "ids".to_owned(),
        TensorType::i32(&[n]),
        Data::I32(rng.ids(n, v)),
    );
    let mut u = |count: u64| rng.uniform(count, lo, hi);
    let (nh, vh) = (n * h, v * h);
    let case =
        |op: TensorOp, operands: Vec<(String, TensorType, Data)>, result: &TensorType| Case {
            op,
            operands,
            result: result.clone(),
            seed: 0x5eed,
        };
    vec![
        case(
            TensorOp::Embedding,
            vec![f("table", &table, u(vh)), ids.clone()],
            &rows,
        ),
        case(
            TensorOp::EmbeddingGrad {
                rows: Dim::Static(v),
            },
            vec![ids, f("g", &rows, u(nh))],
            &table,
        ),
        case(
            TensorOp::AddRows,
            vec![f("x", &rows, u(nh)), f("b", &row, u(h))],
            &rows,
        ),
        case(TensorOp::SumRows, vec![f("g", &rows, u(nh))], &row),
        case(TensorOp::Gelu, vec![f("x", &rows, u(nh))], &rows),
        case(
            TensorOp::GeluGrad,
            vec![f("x", &rows, u(nh)), f("g", &rows, u(nh))],
            &rows,
        ),
        case(
            TensorOp::Dropout { p: 0.1 },
            vec![f("x", &rows, u(nh)), token(SITES[0], 3)],
            &rows,
        ),
        case(
            TensorOp::DropoutGrad { p: 0.1 },
            vec![token(SITES[0], 3), f("g", &rows, u(nh))],
            &rows,
        ),
        case(
            TensorOp::LayerNorm,
            vec![
                f("x", &rows, u(nh)),
                f("gamma", &row, u(h)),
                f("beta", &row, u(h)),
            ],
            &rows,
        ),
        case(
            TensorOp::LayerNormGradX,
            vec![
                f("x", &rows, u(nh)),
                f("gamma", &row, u(h)),
                f("g", &rows, u(nh)),
            ],
            &rows,
        ),
        case(
            TensorOp::LayerNormGradGamma,
            vec![f("x", &rows, u(nh)), f("g", &rows, u(nh))],
            &row,
        ),
        case(TensorOp::Softmax, vec![f("x", &rows, u(nh))], &rows),
        case(
            TensorOp::SoftmaxGrad,
            vec![f("y", &rows, u(nh)), f("g", &rows, u(nh))],
            &rows,
        ),
    ]
}

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
        values.push(builder.input(name.clone(), ty.clone()).expect("an input"));
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

fn kernel_of(case: &Case) -> Kernel {
    let operands = case
        .operands
        .iter()
        .map(|(_, ty, _)| Operand::Buffer(ty.clone()))
        .collect();
    rowwise_kernel(&case.op, operands, case.result.clone())
        .unwrap_or_else(|| panic!("{} has a row-wise kernel", case.op.name()))
}

fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
}

/// One dispatch of `case`'s kernel; the output and its guard words.
fn gpu(device: &Device<'_>, case: &Case) -> Vec<f32> {
    let kernel = kernel_of(case);
    let words = emit(&kernel).expect("the kernel emits");
    let pipeline = if matches!(
        case.op,
        TensorOp::Dropout { .. } | TensorOp::DropoutGrad { .. }
    ) {
        device.kernel_pipeline_seeded(&kernel, &words, case.seed)
    } else {
        device.kernel_pipeline(&kernel, &words)
    }
    .expect("the pipeline is created");
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
            let mut buffer = device.staging_buffer(bytes_of(n)).expect("staging");
            match data {
                Data::F32(v) => buffer.write_f32(v),
                Data::I32(v) => buffer.write_i32(v),
            }
            .expect("written");
            buffer
        })
        .collect();
    let output = buffers.len() - 1;
    let bound: Vec<&Buffer<'_>> = buffers.iter().collect();
    let mut recording = device.record().expect("a recording starts");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("upload");
    }
    recording.barrier(Barrier::TransferToCompute);
    recording
        .dispatch_kernel(&pipeline, &kernel, &bound)
        .expect("dispatch");
    recording.barrier(Barrier::ComputeToTransfer);
    recording
        .copy(&buffers[output], &staging[output])
        .expect("download");
    recording.barrier(Barrier::TransferToHost);
    let recorded = recording.finish().expect("the recording ends");
    device
        .submit(recorded)
        .expect("submitted")
        .wait()
        .expect("the device finishes");
    staging[output].read_f32().expect("the output reads")
}

fn with_device(body: impl FnOnce(&Vulkan, &Device<'_>)) {
    let vulkan = Vulkan::load().expect("the Vulkan loader opens");
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    let device = vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens");
    body(&vulkan, &device);
    let messages = vulkan.take_validation_messages();
    assert!(messages.is_empty(), "{messages:#?}");
}

/// `g` agrees with `c`: both NaN, equal (infinities included), or within gate 5 at `r = |c|`.
fn agrees(g: f32, c: f32) -> bool {
    (g.is_nan() && c.is_nan()) || g == c || within(g, c, c.abs(), GPU_ATOL, GPU_RTOL)
}

/// The elements of `got` that disagree with `want` (bitwise when `bits`), at most `limit`
/// described, the guard words checked, and the worst absolute error.
fn compare(label: &str, got: &[f32], want: &[f32], bits: bool, limit: usize) -> (Vec<String>, f64) {
    let mut bad = Vec::new();
    let mut worst = 0.0_f64;
    let mut count = 0usize;
    for (i, (&g, &c)) in got.iter().zip(want).enumerate() {
        let err = (f64::from(g) - f64::from(c)).abs();
        if err.is_finite() {
            worst = worst.max(err);
        }
        let ok = if bits {
            g.to_bits() == c.to_bits()
        } else {
            agrees(g, c)
        };
        if !ok {
            count += 1;
            if bad.len() < limit {
                bad.push(format!(
                    "{label} [{i}]: gpu {g:?} ({:#010x}) cpu {c:?} ({:#010x})",
                    g.to_bits(),
                    c.to_bits()
                ));
            }
        }
    }
    if count > limit {
        bad.push(format!("{label}: {count} elements disagree in all"));
    }
    for i in 0..GUARD_WORDS {
        if got[want.len() + i].to_bits() != guard(i) {
            bad.push(format!("{label}: guard word {i} changed"));
        }
    }
    (bad, worst)
}

/// Hypothesis 1: shapes the unit's six did not cover. N of 1, 64, 65 and 4096 rows, widths of
/// 1, 2, 64, 65, 768 and 1024, tables of 1 and 50,000 rows, each op against the CPU, bit-equal
/// where the unit claims bits and within gate 5 at `r = |c|` elsewhere. `LayerNormGradX` of
/// rows of two columns is left to `adv_rowwise_2_*`, which it fails (ADV-ROWWISE-2).
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_rowwise_shapes_beyond_the_unit_grid_match_the_cpu() {
    const SHAPES: [(u64, u64, u64); 9] = [
        (1, 1, 1),
        (64, 1, 1),
        (65, 2, 1),
        (1, 64, 50_000),
        (65, 65, 50_000),
        (4096, 2, 3),
        (4096, 65, 7),
        (8, 768, 30),
        (3, 1024, 4),
    ];
    with_device(|_, device| {
        let mut rng = Gen(0xadd5_0001);
        let mut bad = Vec::new();
        let mut modules = 0;
        for (n, h, v) in SHAPES {
            for case in cases(n, h, v, &mut rng, -4.0, 4.0) {
                let name = case.op.name();
                if name == "LayerNormGradX" && h <= 2 {
                    continue;
                }
                let want = cpu(&case);
                let got = gpu(device, &case);
                let bits = BIT_EQUAL.contains(&name);
                let (found, worst) =
                    compare(&format!("{name} [{n}, {h}] v {v}"), &got, &want, bits, 3);
                if !found.is_empty() || worst > 0.0 {
                    eprintln!("{name} [{n}, {h}] v {v}: worst |g - c| {worst:.3e}");
                }
                bad.extend(found);
                modules += 1;
            }
        }
        eprintln!("{modules} kernels run against the CPU");
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

fn rows_case(op: TensorOp, x: Vec<f32>, n: u64, h: u64) -> Case {
    let rows = TensorType::f32(&[n, h]);
    Case {
        op,
        operands: vec![f("x", &rows, x)],
        result: rows,
        seed: 1,
    }
}

/// Hypothesis 2: edge values. Softmax rows holding `-inf`, `+inf`, NaN and the largest finite
/// `f32`, a row of one column, an all `-inf` row; `gelu` and its gradient at ±10, ±100, ±1e4,
/// ±1e20, the infinities, NaN, a subnormal and `-0.0`; `layer_norm` of a constant row whose mean
/// divides exactly, of one column and of two adjacent floats (a constant row whose mean does not
/// divide exactly is `adv_rowwise_1_*`). Device and CPU agree: both NaN, equal, or within gate 5.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_rowwise_edge_values_agree_with_the_cpu() {
    with_device(|_, device| {
        let mut bad = Vec::new();
        let (inf, nan, max) = (f32::INFINITY, f32::NAN, f32::MAX);
        let softmax_rows: Vec<[f32; 3]> = vec![
            [-inf, 0.0, 1.0],
            [inf, 0.0, 1.0],
            [nan, 0.0, 1.0],
            [0.0, nan, 1.0],
            [max, 0.0, -max],
            [-inf, -inf, -inf],
            [1e-45, -1e-45, 0.0],
            [88.0, 89.0, 90.0],
            [-100.0, 0.0, -100.0],
        ];
        let x: Vec<f32> = softmax_rows.iter().flatten().copied().collect();
        let n = softmax_rows.len() as u64;
        let case = rows_case(TensorOp::Softmax, x, n, 3);
        let (found, _) = compare(
            "Softmax specials",
            &gpu(device, &case),
            &cpu(&case),
            false,
            20,
        );
        bad.extend(found);
        let one = rows_case(TensorOp::Softmax, vec![-inf, 3.0, nan, max, 0.0], 5, 1);
        let (found, _) = compare("Softmax width 1", &gpu(device, &one), &cpu(&one), false, 20);
        bad.extend(found);

        let points: Vec<f32> = vec![
            10.0, -10.0, 100.0, -100.0, 1e4, -1e4, 1e20, -1e20, inf, -inf, nan, 1e-40, -0.0, 0.0,
            -3.0, 3.0, 5.5e19, -5.5e19, 40.0, -40.0, 45.0, -45.0, 87.0, -87.0,
        ];
        let len = points.len() as u64;
        let ty = TensorType::f32(&[len]);
        let gelu = Case {
            op: TensorOp::Gelu,
            operands: vec![f("x", &ty, points.clone())],
            result: ty.clone(),
            seed: 1,
        };
        let (found, _) = compare("Gelu specials", &gpu(device, &gelu), &cpu(&gelu), false, 30);
        bad.extend(found);
        let grad = Case {
            op: TensorOp::GeluGrad,
            operands: vec![
                f("x", &ty, points.clone()),
                f("g", &ty, vec![1.0; len as usize]),
            ],
            result: ty.clone(),
            seed: 1,
        };
        let (found, _) = compare(
            "GeluGrad specials",
            &gpu(device, &grad),
            &cpu(&grad),
            false,
            30,
        );
        bad.extend(found);

        let next = f32::from_bits(1.5f32.to_bits() + 1);
        let norm_rows: Vec<(u64, Vec<f32>)> = vec![
            (4, vec![0.1, 0.1, 0.1, 0.1]),
            (1, vec![7.0]),
            (2, vec![1.5, next]),
            (4, vec![1e6, 1e6, 1e6, 1e6 + 0.0625]),
        ];
        for (h, x) in norm_rows {
            let rows = TensorType::f32(&[1, h]);
            let row = TensorType::f32(&[h]);
            let ln = Case {
                op: TensorOp::LayerNorm,
                operands: vec![
                    f("x", &rows, x.clone()),
                    f("gamma", &row, vec![1.0; h as usize]),
                    f("beta", &row, vec![0.0; h as usize]),
                ],
                result: rows.clone(),
                seed: 1,
            };
            let (found, _) = compare(
                &format!("LayerNorm {x:?}"),
                &gpu(device, &ln),
                &cpu(&ln),
                false,
                8,
            );
            bad.extend(found);
            let gx = Case {
                op: TensorOp::LayerNormGradX,
                operands: vec![
                    f("x", &rows, x.clone()),
                    f("gamma", &row, vec![1.0; h as usize]),
                    f("g", &rows, (0..h).map(|j| j as f32 - 0.5).collect()),
                ],
                result: rows.clone(),
                seed: 1,
            };
            let (found, _) = compare(
                &format!("LayerNormGradX {x:?}"),
                &gpu(device, &gx),
                &cpu(&gx),
                false,
                8,
            );
            bad.extend(found);
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// Finding ADV-ROWWISE-3: `AddRows`, `SumRows`, `Embedding`, `EmbeddingGrad`, `Dropout` and
/// `DropoutGrad` over operands holding subnormals give the CPU interpreter's bits, as DESIGN.md
/// revision 0.46 and § 5.2 state for them without a condition. On a device without
/// `shaderDenormPreserveFloat32` (the RTX 3090) the arithmetic ones flush.
#[test]
#[ignore = "needs a Vulkan device; red: ADV-ROWWISE-3"]
fn adv_rowwise_3_bit_claims_hold_for_subnormal_operands() {
    with_device(|_, device| {
        let (n, h, v) = (5_u64, 7_u64, 3_u64);
        let rows = TensorType::f32(&[n, h]);
        let row = TensorType::f32(&[h]);
        let sub = |k: u64| -> Vec<f32> {
            (0..k)
                .map(|i| {
                    let bits = 0x0000_1000 + (i as u32) * 0x0001_2345;
                    let x = f32::from_bits(bits & 0x007f_ffff);
                    if i % 2 == 0 { x } else { -x }
                })
                .collect()
        };
        let ids = (
            "ids".to_owned(),
            TensorType::i32(&[n]),
            Data::I32(vec![0, 2, 1, 2, 0]),
        );
        let table = TensorType::f32(&[v, h]);
        let cases = vec![
            Case {
                op: TensorOp::AddRows,
                operands: vec![f("x", &rows, sub(n * h)), f("b", &row, sub(h))],
                result: rows.clone(),
                seed: 1,
            },
            Case {
                op: TensorOp::SumRows,
                operands: vec![f("g", &rows, sub(n * h))],
                result: row.clone(),
                seed: 1,
            },
            Case {
                op: TensorOp::Embedding,
                operands: vec![f("table", &table, sub(v * h)), ids.clone()],
                result: rows.clone(),
                seed: 1,
            },
            Case {
                op: TensorOp::EmbeddingGrad {
                    rows: Dim::Static(v),
                },
                operands: vec![ids, f("g", &rows, sub(n * h))],
                result: table.clone(),
                seed: 1,
            },
            Case {
                op: TensorOp::Dropout { p: 0.5 },
                operands: vec![f("x", &rows, sub(n * h)), token(SITES[1], 0)],
                result: rows.clone(),
                seed: 1,
            },
            Case {
                op: TensorOp::DropoutGrad { p: 0.5 },
                operands: vec![token(SITES[1], 0), f("g", &rows, sub(n * h))],
                result: rows.clone(),
                seed: 1,
            },
        ];
        let mut bad = Vec::new();
        for case in cases {
            let name = case.op.name();
            let (found, _) = compare(name, &gpu(device, &case), &cpu(&case), true, 4);
            bad.extend(found);
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// Hypothesis 2, dropout at the ends of `[0, 1)`: `p = 0`, `0.999`, the largest `f32` below 1
/// and the smallest positive normal. `Dropout` of ones and of drawn values and `DropoutGrad`
/// are the CPU's bits, and `p = 0` keeps every element unscaled.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_rowwise_dropout_at_extreme_drop_probabilities_is_the_cpus_bits() {
    with_device(|_, device| {
        let (n, h) = (129_u64, 257_u64);
        let rows = TensorType::f32(&[n, h]);
        let mut rng = Gen(0xadd5_0002);
        let values = rng.uniform(n * h, -4.0, 4.0);
        let below_one = f32::from_bits(1.0f32.to_bits() - 1);
        let mut bad = Vec::new();
        for p in [0.0_f32, 0.999, below_one, f32::MIN_POSITIVE] {
            let mut kept = 0usize;
            for (label, x) in [
                ("ones", vec![1.0; (n * h) as usize]),
                ("values", values.clone()),
            ] {
                for grad in [false, true] {
                    let t = token(SITES[2], 11);
                    let operands = if grad {
                        vec![t, f("g", &rows, x.clone())]
                    } else {
                        vec![f("x", &rows, x.clone()), t]
                    };
                    let case = Case {
                        op: if grad {
                            TensorOp::DropoutGrad { p }
                        } else {
                            TensorOp::Dropout { p }
                        },
                        operands,
                        result: rows.clone(),
                        seed: 0xdead_beef_0bad_f00d,
                    };
                    let got = gpu(device, &case);
                    let want = cpu(&case);
                    if label == "ones" && !grad {
                        kept = want[..(n * h) as usize]
                            .iter()
                            .filter(|&&y| y != 0.0)
                            .count();
                    }
                    let (found, _) = compare(
                        &format!("{} p {p:e} {label}", case.op.name()),
                        &got,
                        &want,
                        true,
                        3,
                    );
                    bad.extend(found);
                    if p == 0.0 {
                        for (i, (&g, &x)) in got.iter().zip(&x).enumerate() {
                            if g.to_bits() != x.to_bits() {
                                bad.push(format!("p 0 [{i}]: {g:?} is not {x:?}"));
                                break;
                            }
                        }
                    }
                }
            }
            eprintln!("dropout p {p:e}: {kept} kept of {}", n * h);
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// Hypothesis 3: ten tokens over three sites, a mask of 1,000,003 elements (not a multiple of
/// 64 or of 4) and one of 4,194,311 elements, past 65,535 workgroups of 64 in one dimension:
/// the device mask is `dropout_keeps` of `train/init.rs` at every element, the last included.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_rowwise_dropout_masks_of_a_million_elements_for_ten_tokens_are_the_cpus() {
    with_device(|_, device| {
        let p = 0.3_f32;
        let scale = 1.0f32 / (1.0f32 - p);
        let seed = 0x0123_4567_89ab_cdef_u64;
        let mut bad = Vec::new();
        let mut checked = 0usize;
        for k in 0..11_u64 {
            let count: u64 = if k == 10 { 4_194_311 } else { 1_000_003 };
            let site = SITES[(k % 3) as usize];
            let step = k * 7 + 1;
            let ty = TensorType::f32(&[count]);
            let case = Case {
                op: TensorOp::Dropout { p },
                operands: vec![f("x", &ty, vec![1.0; count as usize]), token(site, step)],
                result: ty,
                seed,
            };
            let t = dropout_token(site, step);
            let id = token_id([t.data()[0], t.data()[1]]);
            let got = gpu(device, &case);
            let mut differ = 0usize;
            assert_eq!(
                got.len(),
                count as usize + GUARD_WORDS,
                "{site} step {step}: element count"
            );
            for (i, &g) in got.iter().enumerate().take(count as usize) {
                let want = if dropout_keeps(seed, id, i as u32, p) {
                    scale
                } else {
                    0.0
                };
                if g.to_bits() != want.to_bits() {
                    if differ < 3 {
                        bad.push(format!("{site} step {step} [{i}]: {g:?} not {want:?}"));
                    }
                    differ += 1;
                }
            }
            for i in 0..GUARD_WORDS {
                if got[count as usize + i].to_bits() != guard(i) {
                    bad.push(format!("{site} step {step}: guard word {i} changed"));
                }
            }
            checked += count as usize;
        }
        eprintln!("dropout: {checked} mask elements checked over 11 tokens");
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// Hypothesis 4: `EmbeddingGrad` of 8,192 ids, every one row 5 of a 32-row table: twenty runs
/// give identical bits, those of the CPU (ascending `n`), and every other row is `+0`.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_rowwise_embedding_grad_of_one_repeated_row_is_identical_over_twenty_runs() {
    with_device(|_, device| {
        let (n, h, v) = (8192_u64, 65_u64, 32_u64);
        let mut rng = Gen(0xadd5_0004);
        let case = Case {
            op: TensorOp::EmbeddingGrad {
                rows: Dim::Static(v),
            },
            operands: vec![
                (
                    "ids".to_owned(),
                    TensorType::i32(&[n]),
                    Data::I32(vec![5; n as usize]),
                ),
                f(
                    "g",
                    &TensorType::f32(&[n, h]),
                    rng.uniform(n * h, -4.0, 4.0),
                ),
            ],
            result: TensorType::f32(&[v, h]),
            seed: 0,
        };
        let want = cpu(&case);
        let count = (v * h) as usize;
        let mut bad = Vec::new();
        for run in 0..20 {
            let got = gpu(device, &case);
            let (found, _) = compare(&format!("run {run}"), &got, &want, true, 3);
            bad.extend(found);
        }
        for (i, &x) in want[..count].iter().enumerate() {
            let row = i as u64 / h;
            if row != 5 && x.to_bits() != 0 {
                bad.push(format!("cpu row {row} [{i}] is {x:?}"));
            }
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// Hypothesis 4: `LayerNormGradGamma` over row counts on both sides of the 64-row blocks it
/// stages: within gate 5 of the CPU at `r = |c|` (the unit's rule) and two runs bit-equal.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_rowwise_layer_norm_grad_gamma_at_block_edges_matches_the_cpu() {
    with_device(|_, device| {
        let h = 65_u64;
        let mut rng = Gen(0xadd5_0005);
        let mut bad = Vec::new();
        for n in [1_u64, 63, 64, 65, 127, 128, 129, 191, 500] {
            let rows = TensorType::f32(&[n, h]);
            let case = Case {
                op: TensorOp::LayerNormGradGamma,
                operands: vec![
                    f("x", &rows, rng.uniform(n * h, -4.0, 4.0)),
                    f("g", &rows, rng.uniform(n * h, -4.0, 4.0)),
                ],
                result: TensorType::f32(&[h]),
                seed: 0,
            };
            let want = cpu(&case);
            let first = gpu(device, &case);
            let second = gpu(device, &case);
            let (found, worst) = compare(&format!("LNGG n {n}"), &first, &want, false, 3);
            bad.extend(found);
            if first
                .iter()
                .map(|x| x.to_bits())
                .ne(second.iter().map(|x| x.to_bits()))
            {
                bad.push(format!("LNGG n {n}: two runs differ"));
            }
            eprintln!("LNGG n {n}: worst |g - c| {worst:.3e}");
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// Hypothesis 2: `Embedding` reads an id outside the table (negative or at or past its rows)
/// as `0.0` for every element of its row, as DESIGN.md § 5.2 states, and in-table ids as the
/// table row.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_rowwise_embedding_of_an_id_outside_the_table_writes_zero() {
    with_device(|_, device| {
        let (v, h) = (4_u64, 3_u64);
        let ids = vec![-1, 4, 11, i32::MIN, i32::MAX, 0, 3, 2];
        let n = ids.len() as u64;
        let table: Vec<f32> = (0..v * h).map(|i| i as f32 + 0.5).collect();
        let case = Case {
            op: TensorOp::Embedding,
            operands: vec![
                f("table", &TensorType::f32(&[v, h]), table.clone()),
                (
                    "ids".to_owned(),
                    TensorType::i32(&[n]),
                    Data::I32(ids.clone()),
                ),
            ],
            result: TensorType::f32(&[n, h]),
            seed: 0,
        };
        let got = gpu(device, &case);
        for (k, &id) in ids.iter().enumerate() {
            for j in 0..h as usize {
                let want = if (0..v as i32).contains(&id) {
                    table[id as usize * h as usize + j]
                } else {
                    0.0
                };
                assert_eq!(
                    got[k * h as usize + j].to_bits(),
                    want.to_bits(),
                    "id {id} [{j}]"
                );
            }
        }
    });
}

/// Hypothesis 1: `SumRows` over 100,000 rows, one invocation per column summing them all:
/// it finishes and gives the CPU's bits (ascending rows from row 0), at widths 1 and 3.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_rowwise_sum_rows_over_a_hundred_thousand_rows_is_the_cpus_bits() {
    with_device(|_, device| {
        let mut rng = Gen(0xadd5_0006);
        let mut bad = Vec::new();
        for h in [1_u64, 3] {
            let n = 100_000_u64;
            let case = Case {
                op: TensorOp::SumRows,
                operands: vec![f(
                    "g",
                    &TensorType::f32(&[n, h]),
                    rng.uniform(n * h, -4.0, 4.0),
                )],
                result: TensorType::f32(&[h]),
                seed: 0,
            };
            let want = cpu(&case);
            let started = Instant::now();
            let got = gpu(device, &case);
            eprintln!("SumRows [{n}, {h}]: {:?} with uploads", started.elapsed());
            let (found, _) = compare(&format!("SumRows h {h}"), &got, &want, true, 3);
            bad.extend(found);
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// The sum of a column starts from its row 0, not from `+0.0`: a column of `-0.0` sums to
/// `-0.0` on the CPU and must on the device. Kills the mutant that starts `SumRows` from `+0.0`,
/// which the unit's suite does not.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_rowwise_sum_rows_of_negative_zero_columns_is_negative_zero() {
    with_device(|_, device| {
        let mut bad = Vec::new();
        for n in [1_u64, 2, 65] {
            let h = 3_u64;
            let mut values = Vec::new();
            for row in 0..n {
                values.extend([-0.0_f32, 0.0, if row % 2 == 0 { -0.0 } else { 0.0 }]);
            }
            let case = Case {
                op: TensorOp::SumRows,
                operands: vec![f("g", &TensorType::f32(&[n, h]), values)],
                result: TensorType::f32(&[h]),
                seed: 0,
            };
            let want = cpu(&case);
            assert_eq!(
                want[0].to_bits(),
                (-0.0_f32).to_bits(),
                "the CPU sums -0.0 columns to -0.0"
            );
            let got = gpu(device, &case);
            let (found, _) = compare(&format!("SumRows -0.0 n {n}"), &got, &want, true, 3);
            bad.extend(found);
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// `n` rows of `h` values centred on `offset` with spread `scale`.
fn narrow_rows(rng: &mut Gen, n: u64, h: u64, offset: f32, scale: f32) -> Vec<f32> {
    rng.uniform(n * h, -1.0, 1.0)
        .into_iter()
        .map(|u| offset + scale * u)
        .collect()
}

/// `layer_norm` rows of small spread, about the scale of BERT's initial embeddings and below,
/// with a gradient drawn independently of the row: the three kernels against the CPU within
/// gate 5 at `r = |c|`, the rule the unit's own device test applies. The absolute error grows
/// with `rstd` (4.6e-5 at spread 0.01), and so does `|c|`, so the relative part covers it.
#[test]
#[ignore = "needs a Vulkan device"]
fn adv_rowwise_layer_norm_grads_of_narrow_rows_are_within_tolerance() {
    with_device(|_, device| {
        let mut rng = Gen(0xadd5_0007);
        let mut bad = Vec::new();
        for (h, scale) in [(64_u64, 0.035_f32), (64, 0.01), (512, 0.035), (768, 0.035)] {
            let n = 64_u64;
            let rows = TensorType::f32(&[n, h]);
            let row = TensorType::f32(&[h]);
            let x = narrow_rows(&mut rng, n, h, 0.0, scale);
            let g = rng.uniform(n * h, -1.0, 1.0);
            let gamma = vec![1.0; h as usize];
            let cases = [
                Case {
                    op: TensorOp::LayerNorm,
                    operands: vec![
                        f("x", &rows, x.clone()),
                        f("gamma", &row, gamma.clone()),
                        f("beta", &row, vec![0.0; h as usize]),
                    ],
                    result: rows.clone(),
                    seed: 0,
                },
                Case {
                    op: TensorOp::LayerNormGradX,
                    operands: vec![
                        f("x", &rows, x.clone()),
                        f("gamma", &row, gamma.clone()),
                        f("g", &rows, g.clone()),
                    ],
                    result: rows.clone(),
                    seed: 0,
                },
                Case {
                    op: TensorOp::LayerNormGradGamma,
                    operands: vec![f("x", &rows, x.clone()), f("g", &rows, g.clone())],
                    result: row.clone(),
                    seed: 0,
                },
            ];
            for case in cases {
                let name = case.op.name();
                let want = cpu(&case);
                let got = gpu(device, &case);
                let (found, worst) = compare(
                    &format!("{name} h {h} spread {scale}"),
                    &got,
                    &want,
                    false,
                    3,
                );
                eprintln!(
                    "{name} h {h} spread {scale}: worst |g - c| {worst:.3e}, {} outside",
                    found.len()
                );
                bad.extend(found);
            }
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// The three `layer_norm` cases of one `[1, h]` row `x` with `gamma` ones, `beta` zeros and
/// the gradient `g[j] = j - h / 2`.
fn norm_cases(x: &[f32]) -> Vec<Case> {
    let h = x.len() as u64;
    let rows = TensorType::f32(&[1, h]);
    let row = TensorType::f32(&[h]);
    let g: Vec<f32> = (0..h).map(|j| j as f32 - (h / 2) as f32).collect();
    vec![
        Case {
            op: TensorOp::LayerNorm,
            operands: vec![
                f("x", &rows, x.to_vec()),
                f("gamma", &row, vec![1.0; h as usize]),
                f("beta", &row, vec![0.0; h as usize]),
            ],
            result: rows.clone(),
            seed: 0,
        },
        Case {
            op: TensorOp::LayerNormGradX,
            operands: vec![
                f("x", &rows, x.to_vec()),
                f("gamma", &row, vec![1.0; h as usize]),
                f("g", &rows, g.clone()),
            ],
            result: rows.clone(),
            seed: 0,
        },
        Case {
            op: TensorOp::LayerNormGradGamma,
            operands: vec![f("x", &rows, x.to_vec()), f("g", &rows, g)],
            result: row,
            seed: 0,
        },
    ]
}

/// Finding ADV-ROWWISE-1: `layer_norm` of a constant row. The CPU's mean of `h` copies of `c`
/// rounds back to `c` for each row here, so every deviation is exactly 0 and the normalized row
/// is `beta`, its gradient `rstd * (g - mean(g))`. The device divides the sum by `h` with
/// `OpFDiv`, which Vulkan does not round correctly (2.5 ulp); a mean one ulp off `c` makes each
/// deviation one ulp of `c`, and with the variance far below `eps = 1e-12`, `rstd` near `1e6`
/// turns it into `x̂ = ±ulp(c) · rstd`, of order 1e-2 to 1 where the CPU has 0.
#[test]
#[ignore = "needs a Vulkan device; red: ADV-ROWWISE-1"]
fn adv_rowwise_1_layer_norm_of_a_constant_row_is_within_tolerance() {
    with_device(|_, device| {
        let rows: Vec<(f32, usize)> = vec![
            (0.1, 3),
            (0.3, 7),
            (100.0, 3),
            (1.0 / 3.0, 5),
            (0.7, 11),
            (12.34, 13),
            (1e-3, 3),
            (0.2, 768),
        ];
        let mut bad = Vec::new();
        let mut exact = 0;
        for (c, h) in rows {
            let x = vec![c; h];
            for case in norm_cases(&x) {
                let name = case.op.name();
                let want = cpu(&case);
                if name == "LayerNorm" && want[..h].iter().all(|&y| y == 0.0) {
                    exact += 1;
                }
                let got = gpu(device, &case);
                let (found, worst) = compare(&format!("{name} [{c}; {h}]"), &got, &want, false, 2);
                eprintln!("{name} [{c}; {h}]: worst |g - c| {worst:.3e}");
                bad.extend(found);
            }
        }
        eprintln!("{exact} of 8 constant rows normalize to exactly beta on the CPU");
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// Finding ADV-ROWWISE-2: `LayerNormGradX` where the CPU's element cancels to (nearly) 0 while
/// its terms are of order `rstd`: (a) rows of two columns with operands drawn as the unit's
/// tests draw them, where `x̂ = ±1` and the CPU's gradient is exactly 0 (the gradient of a
/// two-column `layer_norm` vanishes but for `eps`); (b) the gradient equal to the row's own
/// `layer_norm` output (gamma 1, beta 0), the gradient of any loss of `|y|` alone such as
/// `mse(layer_norm(x), 0)`, which `layer_norm`'s scale invariance sends to 0. The device's
/// `rstd`, a few ulp off the CPU's, leaves about `rstd` times a few ulp, past `GPU_ATOL` once
/// `rstd` is in the tens.
#[test]
#[ignore = "needs a Vulkan device; red: ADV-ROWWISE-2"]
fn adv_rowwise_2_layer_norm_grad_x_of_a_cancelling_gradient_is_within_tolerance() {
    with_device(|_, device| {
        let mut rng = Gen(0xadd5_0008);
        let mut bad = Vec::new();
        let (n, h) = (4096_u64, 2_u64);
        let rows = TensorType::f32(&[n, h]);
        let row = TensorType::f32(&[h]);
        let two = Case {
            op: TensorOp::LayerNormGradX,
            operands: vec![
                f("x", &rows, rng.uniform(n * h, -4.0, 4.0)),
                f("gamma", &row, rng.uniform(h, -4.0, 4.0)),
                f("g", &rows, rng.uniform(n * h, -4.0, 4.0)),
            ],
            result: rows,
            seed: 0,
        };
        let want = cpu(&two);
        let zeros = want.iter().filter(|&&c| c == 0.0).count();
        let (found, worst) = compare("LNGX [4096, 2]", &gpu(device, &two), &want, false, 3);
        eprintln!(
            "LNGX [4096, 2]: {zeros} of {} CPU elements 0, worst {worst:.3e}",
            n * h
        );
        bad.extend(found);
        for (h, spread) in [(64_u64, 0.035_f32), (64, 0.01), (768, 0.035)] {
            let n = 16_u64;
            let rows = TensorType::f32(&[n, h]);
            let row = TensorType::f32(&[h]);
            let x = narrow_rows(&mut rng, n, h, 0.0, spread);
            let forward = Case {
                op: TensorOp::LayerNorm,
                operands: vec![
                    f("x", &rows, x.clone()),
                    f("gamma", &row, vec![1.0; h as usize]),
                    f("beta", &row, vec![0.0; h as usize]),
                ],
                result: rows.clone(),
                seed: 0,
            };
            let y = cpu(&forward);
            let case = Case {
                op: TensorOp::LayerNormGradX,
                operands: vec![
                    f("x", &rows, x),
                    f("gamma", &row, vec![1.0; h as usize]),
                    f("g", &rows, y),
                ],
                result: rows,
                seed: 0,
            };
            let want = cpu(&case);
            let largest = want.iter().fold(0.0_f32, |m, &c| m.max(c.abs()));
            let label = format!("LNGX g = y, h {h} spread {spread}");
            let (found, worst) = compare(&label, &gpu(device, &case), &want, false, 3);
            eprintln!("{label}: largest |c| {largest:.3e}, worst |g - c| {worst:.3e}");
            bad.extend(found);
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}

/// Finding ADV-ROWWISE-3: with every operand normal, so that DESIGN.md § 9 gate 5's flushed
/// reference (the CPU on operands with subnormals flushed, for a device without
/// `shaderDenormPreserveFloat32`) is the CPU itself, `AddRows`, `SumRows` and `EmbeddingGrad`
/// whose adds land on a subnormal give the CPU's bits, as DESIGN.md revision 0.46 claims.
#[test]
#[ignore = "needs a Vulkan device; red: ADV-ROWWISE-3"]
fn adv_rowwise_3_bit_claims_hold_against_the_flushed_reference_for_subnormal_results() {
    with_device(|_, device| {
        let (n, h) = (2_u64, 4_u64);
        let rows = TensorType::f32(&[n, h]);
        let row = TensorType::f32(&[h]);
        let a = f32::MIN_POSITIVE * 1.5;
        let x = vec![a, -a, a, -a, -a * 0.75, a * 0.75, -a * 0.875, a * 0.6875];
        assert!(x.iter().all(|v| v.is_normal()));
        let b = vec![-a * 0.75, a * 0.75, -a * 0.875, a * 0.6875];
        let cases = vec![
            Case {
                op: TensorOp::AddRows,
                operands: vec![f("x", &rows, x.clone()), f("b", &row, b)],
                result: rows.clone(),
                seed: 0,
            },
            Case {
                op: TensorOp::SumRows,
                operands: vec![f("g", &rows, x.clone())],
                result: row.clone(),
                seed: 0,
            },
            Case {
                op: TensorOp::EmbeddingGrad {
                    rows: Dim::Static(1),
                },
                operands: vec![
                    (
                        "ids".to_owned(),
                        TensorType::i32(&[n]),
                        Data::I32(vec![0, 0]),
                    ),
                    f("g", &rows, x.clone()),
                ],
                result: TensorType::f32(&[1, h]),
                seed: 0,
            },
        ];
        let mut bad = Vec::new();
        for case in cases {
            let name = case.op.name();
            let want = cpu(&case);
            let subnormal = want.iter().filter(|c| c.is_subnormal()).count();
            eprintln!("{name}: {subnormal} subnormal CPU results");
            let (found, _) = compare(name, &gpu(device, &case), &want, true, 4);
            bad.extend(found);
        }
        assert!(bad.is_empty(), "{bad:#?}");
    });
}
