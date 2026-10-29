//! Adversary cases for story:attention-event-spans.
//!
//! - Host data path: an `i32` spans input read from a dense `--data` directory is held to its
//!   rule before the first step, as every other `i32` input that bounds a read is
//!   (`E-DATA-011`). The CPU interpreter refuses a malformed span only at the step that reads it,
//!   and the Vulkan run reads the words as they are, clamping them, so without a host check a
//!   span the CPU refuses trains silently on the device.
//! - Synthetic data: an `i32` spans input is no key mask; the run without `--data` refuses by a
//!   named data code instead of filling the spans with `f32` ones.
//! - Device (ignored, `--features vulkan`): a span longer than `MAX_SPAN_KEYS = 512`, which the
//!   CPU interpreter accepts, gives the CPU's result on the device, or nothing does; spans at row
//!   0, back to back and ending at the last row give the CPU's result and `+0` outside them.
//!
//! Run the device cases with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test adversary_attention_spans -- --include-ignored`.

use std::fs;
use std::path::PathBuf;

use modelspec_core::compile::{CompiledTraining, compile_training};
use modelspec_runtime::train::data::{DataError, dense, synthetic};

const SPANS_MODEL: &str = "\
model Spans {
    param w: Tensor<f32, [4, 4]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>, spans: Tensor<i32, [2, 2]>) -> Tensor<f32, [8, 4]> {
        let h = matmul(x, w);
        return attention(h, h, h, spans, heads = 2);
    }
}

train Fit {
    model = Spans;
    input x: Tensor<f32, [8, 4]>;
    input spans: Tensor<i32, [2, 2]>;
    target y: Tensor<f32, [8, 4]>;
    loss = mse(model.forward(x, spans), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 3;
    device = cpu;
}
";

fn compiled() -> CompiledTraining {
    compile_training("spans.mspec", SPANS_MODEL).expect("the spans model compiles")
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("adversary_spans_{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// A dense directory of two samples: sample 0 with the valid spans `(0, 3), (4, 4)`, sample 1
/// with `bad`.
fn write_dense(name: &str, bad: [i32; 4]) -> PathBuf {
    let dir = scratch(name);
    let f32s = |n: usize| -> Vec<u8> {
        (0..n)
            .flat_map(|i| ((i % 7) as f32 * 0.125 - 0.3).to_le_bytes())
            .collect()
    };
    fs::write(dir.join("x.f32"), f32s(2 * 32)).expect("x");
    fs::write(dir.join("y.f32"), f32s(2 * 32)).expect("y");
    let spans: Vec<u8> = [0, 3, 4, 4]
        .into_iter()
        .chain(bad)
        .flat_map(i32::to_le_bytes)
        .collect();
    fs::write(dir.join("spans.i32"), spans).expect("spans");
    dir
}

/// Every span the CPU interpreter refuses (`SpanNegative`, `SpanEmpty`, `SpanOutOfRange`,
/// `SpanOverlap`, and spans out of order) is refused by the dense data reader before the first
/// step, as an `i32` id outside its `embedding` table is (E-DATA-011, "the refusal comes before
/// the first step on both backends"). On the Vulkan backend nothing else refuses it.
#[test]
fn dense_data_refuses_every_span_the_cpu_refuses_before_the_first_step() {
    let compiled = compiled();
    let cases: [(&str, [i32; 4]); 7] = [
        ("negative_start", [-1, 2, 4, 4]),
        ("negative_length", [0, -1, 4, 4]),
        ("empty", [0, 0, 4, 4]),
        ("past_the_rows", [0, 3, 4, 5]),
        ("overlapping", [0, 3, 2, 4]),
        ("out_of_order", [4, 4, 0, 3]),
        ("huge", [0, i32::MAX, 0, i32::MAX]),
    ];
    let mut accepted = Vec::new();
    for (name, bad) in cases {
        let dir = write_dense(name, bad);
        match dense(&dir, &compiled.config, &compiled.forward) {
            Err(DataError::Refused(_)) => {}
            Err(other) => accepted.push(format!("{name}: {bad:?} refused unnamed: {other}")),
            Ok(_) => accepted.push(format!("{name}: {bad:?} accepted")),
        }
    }
    assert!(
        accepted.is_empty(),
        "spans the CPU interpreter refuses, read by the dense reader without a refusal: {accepted:#?}"
    );
}

/// The valid sample alone is accepted, so the case above fails on the bad spans only.
#[test]
fn dense_data_accepts_valid_spans() {
    let compiled = compiled();
    let dir = write_dense("valid", [0, 1, 2, 6]);
    assert!(dense(&dir, &compiled.config, &compiled.forward).is_ok());
}

/// Without `--data` the spans input is no key mask: `synthetic` refuses the block by a named data
/// code (two inputs besides the key masks), not with an interpreter error from an `f32` tensor of
/// ones standing in for the `i32` spans.
#[test]
fn synthetic_data_does_not_fill_an_i32_spans_input_as_a_key_mask() {
    let compiled = compiled();
    match synthetic(&compiled.config, &compiled.forward) {
        Ok(data) => assert!(
            !data.inputs.contains_key("spans"),
            "synthetic data holds an f32 tensor for the i32 spans"
        ),
        Err(DataError::Refused(_)) => {}
        Err(other) => panic!("synthetic data over spans fails unnamed: {other:?}"),
    }
}

#[cfg(feature = "vulkan")]
#[allow(dead_code)]
mod common;

#[cfg(feature = "vulkan")]
mod device {
    use std::collections::BTreeMap;

    use super::common::{Draw, GPU_ATOL, GPU_RTOL, within};
    use modelspec_core::ir::kernel::{Kernel, Operand};
    use modelspec_core::ir::tensor::{TensorOp, TensorProgramBuilder, TensorType};
    use modelspec_core::schedule::attention_loss_kernel;
    use modelspec_runtime::cpu::interpreter::evaluate_with_ids;
    use modelspec_runtime::cpu::tensor::HostTensor;
    use modelspec_runtime::spirv::emit::emit;
    use modelspec_runtime::vulkan::memory::Buffer;
    use modelspec_runtime::vulkan::negotiate::negotiate;
    use modelspec_runtime::vulkan::sync::Barrier;
    use modelspec_runtime::vulkan::{Device, Vulkan};

    const OPS: [TensorOp; 4] = [
        TensorOp::Attention { heads: 2 },
        TensorOp::AttentionGradQ { heads: 2 },
        TensorOp::AttentionGradK { heads: 2 },
        TensorOp::AttentionGradV { heads: 2 },
    ];
    const H: u64 = 8;

    fn arity(op: &TensorOp) -> usize {
        if matches!(op, TensorOp::Attention { .. }) {
            4
        } else {
            5
        }
    }

    fn per_query(op: &TensorOp) -> bool {
        matches!(
            op,
            TensorOp::Attention { .. } | TensorOp::AttentionGradQ { .. }
        )
    }

    fn shapes(n: u64, m: u64, u: u64) -> Vec<Vec<u64>> {
        vec![vec![n, H], vec![m, H], vec![m, H], vec![u, 2], vec![n, H]]
    }

    fn cpu(op: &TensorOp, inputs: &[Vec<f32>], shapes: &[Vec<u64>], spans: &[i32]) -> Vec<f32> {
        let mut builder = TensorProgramBuilder::new();
        let mut values = Vec::new();
        let mut host = BTreeMap::new();
        let mut ids = BTreeMap::new();
        for at in 0..arity(op) {
            let name = format!("x{at}");
            if at == 3 {
                ids.insert(
                    name.clone(),
                    HostTensor::new(shapes[at].clone(), spans.to_vec()).expect("a tensor"),
                );
                values.push(
                    builder
                        .input(name, TensorType::i32(&shapes[at]))
                        .expect("an input"),
                );
                continue;
            }
            host.insert(
                name.clone(),
                HostTensor::new(shapes[at].clone(), inputs[at].clone()).expect("a tensor"),
            );
            values.push(
                builder
                    .input(name, TensorType::f32(&shapes[at]))
                    .expect("an input"),
            );
        }
        let result = builder.push(op.clone(), &values).expect("well typed");
        builder.output(result).expect("the result is a value");
        let program = builder.finish();
        evaluate_with_ids(&program, &host, &ids, &BTreeMap::new()).expect("the CPU evaluates")[0]
            .data()
            .to_vec()
    }

    /// One kernel of `op` on the device over `inputs` (the spans as `u32` words in slot 3), its
    /// result buffer NaN before.
    fn gpu(
        device: &Device<'_>,
        op: &TensorOp,
        inputs: &[Vec<f32>],
        shapes: &[Vec<u64>],
    ) -> Vec<f32> {
        let types: Vec<TensorType> = (0..arity(op))
            .map(|at| {
                if at == 3 {
                    TensorType::i32(&shapes[at])
                } else {
                    TensorType::f32(&shapes[at])
                }
            })
            .collect();
        let operands = types.iter().cloned().map(Operand::Buffer).collect();
        let result = if per_query(op) { &types[0] } else { &types[1] };
        let kernel: Kernel =
            attention_loss_kernel(op, operands, result.clone()).expect("an attention kernel");
        let words = emit(&kernel).expect("emits");
        let pipeline = device.kernel_pipeline(&kernel, &words).expect("a pipeline");
        let out_len = if per_query(op) {
            inputs[0].len()
        } else {
            inputs[1].len()
        };
        let mut lengths: Vec<usize> = inputs[..arity(op)].iter().map(Vec::len).collect();
        lengths.push(out_len);
        let buffers: Vec<Buffer<'_>> = lengths
            .iter()
            .map(|&n| device.device_local_buffer(4 * n as u64).expect("a buffer"))
            .collect();
        let mut staging: Vec<Buffer<'_>> = lengths
            .iter()
            .map(|&n| {
                device
                    .staging_buffer(4 * n as u64)
                    .expect("a staging buffer")
            })
            .collect();
        for (at, buffer) in staging.iter_mut().enumerate() {
            let values = match inputs.get(at).filter(|_| at < arity(op)) {
                Some(values) => values.clone(),
                None => vec![f32::from_bits(0x7fc0_dead); lengths[at]],
            };
            buffer.write_f32(&values).expect("written");
        }
        let mut recording = device.record().expect("a recording");
        for (from, to) in staging.iter().zip(&buffers) {
            recording.copy(from, to).expect("an upload");
        }
        recording.barrier(Barrier::TransferToCompute);
        let bound: Vec<&Buffer<'_>> = buffers.iter().collect();
        recording
            .dispatch_kernel(&pipeline, &kernel, &bound)
            .expect("a dispatch");
        recording.barrier(Barrier::ComputeToTransfer);
        let last = buffers.len() - 1;
        recording
            .copy(&buffers[last], &staging[last])
            .expect("a download");
        recording.barrier(Barrier::TransferToHost);
        let recorded = recording.finish().expect("the recording ends");
        device
            .submit(recorded)
            .expect("submitted")
            .wait()
            .expect("the device finishes");
        staging[last].read_f32().expect("read")
    }

    /// For each op over `spans` in `m` key rows, with queries at every row (`every_row`) or one
    /// per span: the device's result against the CPU interpreter's, `+0` in bits where the CPU
    /// has `+0` outside every span, within `GPU_ATOL + GPU_RTOL * |c|` elsewhere.
    fn compare(
        device: &Device<'_>,
        vulkan: &Vulkan,
        label: &str,
        m: u64,
        spans: &[(u64, u64)],
        every_row: bool,
    ) -> Vec<String> {
        let u = spans.len() as u64;
        let n = if every_row { m } else { u };
        let shapes = shapes(n, m, u);
        let mut draw = Draw::new(0x6164_7673_7061_6e73 ^ m ^ (u << 20));
        let mut values = |rows: u64, by: f32| -> Vec<f32> {
            draw.values(rows * H).into_iter().map(|x| x * by).collect()
        };
        let words: Vec<f32> = spans
            .iter()
            .flat_map(|&(s, l)| [s as u32, l as u32])
            .map(f32::from_bits)
            .collect();
        let ids: Vec<i32> = spans
            .iter()
            .flat_map(|&(s, l)| [s as i32, l as i32])
            .collect();
        let inputs = vec![
            values(n, 0.25),
            values(m, 0.25),
            values(m, 1.0),
            words,
            values(n, 1.0),
        ];
        let mut wrong = Vec::new();
        for op in &OPS {
            let want = cpu(op, &inputs, &shapes, &ids);
            let got = gpu(device, op, &inputs, &shapes);
            let bad: Vec<String> = got
                .iter()
                .zip(&want)
                .enumerate()
                .filter(|(_, (g, c))| {
                    if c.to_bits() == 0 {
                        g.to_bits() != 0 && !within(**g, **c, 0.0, GPU_ATOL, GPU_RTOL)
                    } else {
                        g.is_nan() || !within(**g, **c, c.abs(), GPU_ATOL, GPU_RTOL)
                    }
                })
                .map(|(i, (g, c))| format!("row {} [{i}]: gpu {g:?} cpu {c:?}", i as u64 / H))
                .collect();
            if !bad.is_empty() {
                wrong.push(format!(
                    "{} {label}: {} of {} values differ, first {:?}, last {:?}",
                    op.name(),
                    bad.len(),
                    got.len(),
                    bad.first(),
                    bad.last()
                ));
            }
            let outside_negative_zero = got
                .iter()
                .zip(&want)
                .filter(|(g, c)| c.to_bits() == 0 && g.to_bits() == 0x8000_0000)
                .count();
            if outside_negative_zero > 0 {
                wrong.push(format!(
                    "{} {label}: {outside_negative_zero} -0 where the CPU has +0",
                    op.name()
                ));
            }
        }
        let messages = vulkan.take_validation_messages();
        assert!(messages.is_empty(), "{label}: {messages:?}");
        wrong
    }

    fn open() -> Vulkan {
        Vulkan::load().expect("the Vulkan loader opens")
    }

    /// A model of `rows` rows whose one `attention` reads `u` spans, trained on dense data.
    fn long_model(rows: u64, u: u64) -> String {
        format!(
            "\
model Long {{
    param w: Tensor<f32, [8, 8]> = xavier_uniform();

    fn forward(x: Tensor<f32, [{rows}, 8]>, spans: Tensor<i32, [{u}, 2]>) -> Tensor<f32, [{rows}, 8]> {{
        let h = matmul(x, w);
        return attention(h, h, h, spans, heads = 2);
    }}
}}

train Fit {{
    model = Long;
    input x: Tensor<f32, [{rows}, 8]>;
    input spans: Tensor<i32, [{u}, 2]>;
    target y: Tensor<f32, [{rows}, 8]>;
    loss = mse(model.forward(x, spans), y);
    optimizer = sgd(lr = 0.1);
    steps = 2;
    seed = 3;
    device = cpu;
}}
"
        )
    }

    /// A dense directory of one sample of `rows` rows over `spans`.
    fn long_dense(name: &str, rows: u64, spans: &[(u64, u64)]) -> std::path::PathBuf {
        let dir = super::scratch(name);
        let f32s: Vec<u8> = (0..rows * 8)
            .flat_map(|i| ((i % 11) as f32 * 0.0625 - 0.3).to_le_bytes())
            .collect();
        std::fs::write(dir.join("x.f32"), &f32s).expect("x");
        std::fs::write(dir.join("y.f32"), &f32s).expect("y");
        let words: Vec<u8> = spans
            .iter()
            .flat_map(|&(s, l)| [s as i32, l as i32])
            .flat_map(i32::to_le_bytes)
            .collect();
        std::fs::write(dir.join("spans.i32"), words).expect("spans");
        dir
    }

    /// A span of more than `MAX_SPAN_KEYS = 512` keys, which the CPU interpreter accepts and
    /// computes whole. The device kernels read at most 512 keys of a span, so the device never
    /// returns a result for it that differs from the CPU's: the Vulkan run refuses it with
    /// `E-TARGET-011` before the first step, as training data (`train --data`) and as the
    /// dense eval data of `--eval` and `infer --data --function`. This case first asked the
    /// device for the CPU's result over the long span; the coordinator decided the Vulkan backend
    /// refuses such a span instead, so it now asserts "refused with E-TARGET-011, no step run".
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn a_span_past_512_keys_is_refused_by_the_device_before_the_first_step() {
        use modelspec_core::artifact::Backend;
        use modelspec_core::compile::{codes::NO_KERNEL, compile_artifact_for, compile_training};
        use modelspec_runtime::train::data::{DataError, dense, dense_eval};
        use modelspec_runtime::train::runner::{Data, RunError, run_with};
        use modelspec_runtime::vulkan::runner::VulkanRunError;
        use modelspec_runtime::vulkan::stepwise::{StepwiseOptions, train_stepwise};

        let vulkan = open();
        let candidates = vulkan.candidates().expect("the devices are probed");
        let profile = candidates[0].profile.clone();
        let negotiated = negotiate(&profile).expect("device 0 negotiates");
        let device = vulkan
            .create_device(0, &negotiated)
            .expect("device 0 opens");
        let mut wrong = Vec::new();
        type Spans = &'static [(u64, u64)];
        let cases: [(&str, u64, Spans, Spans); 2] = [
            ("one span of 600", 600, &[(0, 600)], &[(0, 512)]),
            (
                "spans (0,3) (4,513)",
                520,
                &[(0, 3), (4, 513)],
                &[(0, 3), (4, 512)],
            ),
        ];
        for (label, rows, long, short) in cases {
            let source = long_model(rows, long.len() as u64);
            let compiled = compile_training("long.mspec", &source).expect("the model compiles");
            let artifact =
                compile_artifact_for("long.mspec", &source, Backend::Vulkan, Some(&profile))
                    .unwrap_or_else(|d| panic!("{label}: the model compiles for Vulkan: {d:?}"));
            let tag = label.replace([' ', '(', ')', ','], "_");
            let long_dir = long_dense(&format!("long_{tag}"), rows, long);
            let short_dir = long_dense(&format!("short_{tag}"), rows, short);
            let long_data = dense(&long_dir, &compiled.config, &compiled.forward)
                .unwrap_or_else(|e| panic!("{label}: the CPU's dense reader accepts it: {e}"));
            let long_eval = dense_eval(&long_dir, &compiled.config, &compiled.forward)
                .unwrap_or_else(|e| panic!("{label}: the CPU's eval reader accepts it: {e}"));
            let short_data = dense(&short_dir, &compiled.config, &compiled.forward)
                .unwrap_or_else(|e| panic!("{label}: the short spans are read: {e}"));
            if let Err(error) = run_with(&compiled, Some(&long_data), &BTreeMap::new(), |_| {}) {
                wrong.push(format!(
                    "{label}: the CPU run refuses the long span: {error}"
                ));
            }
            let runs: [(&str, StepwiseOptions<'_, '_>); 2] = [
                (
                    "train --data",
                    StepwiseOptions {
                        data: Some(Data::Dense(&long_data)),
                        ..StepwiseOptions::default()
                    },
                ),
                (
                    "--eval",
                    StepwiseOptions {
                        data: Some(Data::Dense(&short_data)),
                        dense_eval: Some(&long_eval),
                        ..StepwiseOptions::default()
                    },
                ),
            ];
            for (how, options) in runs {
                let mut steps = 0;
                let result = train_stepwise(
                    &device,
                    &profile,
                    &negotiated,
                    &compiled,
                    &artifact,
                    options,
                    &mut |_| {
                        steps += 1;
                        Ok(())
                    },
                );
                match result {
                    Err(VulkanRunError::Run(error)) => match *error {
                        RunError::Data(DataError::Refused(d)) if d.code == NO_KERNEL => {}
                        other => wrong.push(format!(
                            "{label} {how}: refused, not with {}: {other}",
                            NO_KERNEL.as_str()
                        )),
                    },
                    Err(other) => wrong.push(format!(
                        "{label} {how}: failed, not refused with {}: {other}",
                        NO_KERNEL.as_str()
                    )),
                    Ok(_) => wrong.push(format!(
                        "{label} {how}: the device ran a span past 512 keys to a result"
                    )),
                }
                if steps > 0 {
                    wrong.push(format!(
                        "{label} {how}: {steps} steps ran before the refusal"
                    ));
                }
            }
        }
        let messages = vulkan.take_validation_messages();
        assert!(messages.is_empty(), "{messages:?}");
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Spans from row 0, back to back, and one ending at the last row of `k`: the CPU's result,
    /// `+0` (never `-0`) in the rows between spans.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn spans_at_the_buffer_edges_give_the_cpu_result_on_the_device() {
        let vulkan = open();
        let candidates = vulkan.candidates().expect("the devices are probed");
        let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
        let device = vulkan
            .create_device(0, &negotiated)
            .expect("device 0 opens");
        let mut wrong = Vec::new();
        for every_row in [true, false] {
            wrong.extend(compare(
                &device,
                &vulkan,
                "edges",
                8,
                &[(0, 3), (3, 1), (6, 2)],
                every_row,
            ));
            wrong.extend(compare(
                &device,
                &vulkan,
                "one span of 512 at the end",
                530,
                &[(18, 512)],
                every_row,
            ));
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }
}
