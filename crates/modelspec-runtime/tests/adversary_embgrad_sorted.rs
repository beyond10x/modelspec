//! Adversary pass on story:vulkan-embedding-grad-sorted-ids: the sort of `EmbeddingGrad` by id
//! and position into `i32[n, 2]` and the gradient over the sorted ids, on inputs the unit's
//! device test does not run (one id, a one-row table, `n` off a workgroup multiple, every id
//! equal, ids outside the table, gradient rows of `-0.0`, batches far past BERT-tiny's), with the
//! sorted rows read back and checked against the stable sort by `(id as u32, position)`, guard
//! words after both written buffers, and the step plan of BERT-tiny with the masked-row gather.
//!
//! The device tests are ignored by default: they create a Vulkan instance. Run them with
//! `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test adversary_embgrad_sorted -- --ignored`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use common::{Draw, GUARD_WORDS, guard, guarded};
use modelspec_core::artifact::{self, Backend, KernelCacheKey, KernelModuleFile};
use modelspec_core::compile::{compile_artifact, compile_artifact_with};
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{DispatchCount, Kernel, KernelKind, Operand, PlanOp};
use modelspec_core::ir::tensor::{DType, Dim, TensorProgramBuilder, TensorType};
use modelspec_core::schedule::embedding_grad_kernels;
use modelspec_runtime::cpu::interpreter::evaluate_seeded;
use modelspec_runtime::cpu::tensor::HostTensor;
use modelspec_runtime::spirv::emit::{emit, emit_kernels};
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

/// The bits every written word starts as, a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_dead;

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

const BERT_TINY: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../examples/bert_tiny.mspec"
));

/// What one run of the two kernels left: the sorted rows and the table gradient, each followed
/// by its guard words, as bits.
struct Ran {
    sorted: Vec<u32>,
    grad: Vec<u32>,
}

fn bytes_of(words: usize) -> u64 {
    (words * 4) as u64
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

fn device_of(vulkan: &Vulkan) -> Device<'_> {
    let candidates = vulkan.candidates().expect("the devices are probed");
    let negotiated = negotiate(&candidates[0].profile).expect("device 0 negotiates");
    vulkan
        .create_device(0, &negotiated)
        .expect("device 0 opens")
}

/// The two kernels the schedule builds for `ids: i32[n]`, `g: f32[n, h]` into `f32[v, h]`.
fn kernels(n: u64, h: u64, v: u64) -> [Kernel; 2] {
    embedding_grad_kernels(
        vec![
            Operand::Buffer(TensorType::i32(&[n])),
            Operand::Buffer(TensorType::f32(&[n, h])),
        ],
        TensorType::f32(&[v, h]),
    )
    .expect("EmbeddingGrad has its two kernels")
}

/// Runs the sort, a compute-to-compute barrier and the gradient once, as the step plan records
/// a two-stage work, with the sorted buffer and the gradient both guarded and both read back.
fn run(device: &Device<'_>, ids: &[i32], g: &[f32], h: u64, v: u64) -> Ran {
    let n = ids.len() as u64;
    let [sort, grad] = kernels(n, h, v);
    let pipeline = |kernel: &Kernel| {
        let words = emit(kernel).expect("the kernel emits");
        device
            .kernel_pipeline(kernel, &words)
            .expect("the pipeline is created")
    };
    let (sort_pipeline, grad_pipeline) = (pipeline(&sort), pipeline(&grad));
    let sorted_init = guarded(&vec![f32::from_bits(UNWRITTEN); 2 * ids.len()]);
    let grad_init = guarded(&vec![f32::from_bits(UNWRITTEN); (v * h) as usize]);
    let lengths = [ids.len(), g.len(), sorted_init.len(), grad_init.len()];
    let buffers: Vec<Buffer<'_>> = lengths
        .iter()
        .map(|&n| device.device_local_buffer(bytes_of(n)).expect("a buffer"))
        .collect();
    let mut staging: Vec<Buffer<'_>> = lengths
        .iter()
        .map(|&n| {
            device
                .staging_buffer(bytes_of(n))
                .expect("a staging buffer")
        })
        .collect();
    staging[0].write_i32(ids).expect("ids staged");
    staging[1].write_f32(g).expect("g staged");
    staging[2].write_f32(&sorted_init).expect("sorted staged");
    staging[3].write_f32(&grad_init).expect("grad staged");
    let mut recording = device.record().expect("a recording starts");
    for (from, to) in staging.iter().zip(&buffers) {
        recording.copy(from, to).expect("the upload is recorded");
    }
    recording.barrier(Barrier::TransferToCompute);
    recording
        .dispatch_kernel(&sort_pipeline, &sort, &[&buffers[0], &buffers[2]])
        .expect("the sort is recorded");
    recording.barrier(Barrier::ComputeToCompute);
    recording
        .dispatch_kernel(
            &grad_pipeline,
            &grad,
            &[&buffers[2], &buffers[1], &buffers[3]],
        )
        .expect("the gradient is recorded");
    recording.barrier(Barrier::ComputeToTransfer);
    for at in [2, 3] {
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
    let bits = |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    Ran {
        sorted: bits(staging[2].read_f32().expect("the sorted ids read")),
        grad: bits(staging[3].read_f32().expect("the gradient reads")),
    }
}

/// The CPU interpreter's `EmbeddingGrad` of `ids` and `g` into `v` rows, or its refusal.
fn cpu(ids: &[i32], g: &[f32], h: u64, v: u64) -> Result<Vec<u32>, String> {
    let n = ids.len() as u64;
    let mut builder = TensorProgramBuilder::new();
    let id_value = builder
        .input("ids", TensorType::i32(&[n]))
        .expect("an input");
    let g_value = builder
        .input("g", TensorType::f32(&[n, h]))
        .expect("an input");
    let result = builder
        .embedding_grad(id_value, g_value, Dim::Static(v))
        .expect("well typed");
    builder.output(result).expect("an output");
    let program = builder.finish();
    let inputs = BTreeMap::from([(
        "g".to_owned(),
        HostTensor::new(vec![n, h], g.to_vec()).expect("a tensor"),
    )]);
    let id_inputs = BTreeMap::from([(
        "ids".to_owned(),
        HostTensor::new(vec![n], ids.to_vec()).expect("a tensor"),
    )]);
    evaluate_seeded(&program, &inputs, &id_inputs, &BTreeMap::new(), 0)
        .map(|out| out[0].data().iter().map(|x| x.to_bits()).collect())
        .map_err(|error| format!("{error:?}"))
}

/// The stable sort of the positions by id compared as `u32`, as `(id, position)` rows.
fn stable_sort(ids: &[i32]) -> Vec<u32> {
    let mut positions: Vec<usize> = (0..ids.len()).collect();
    positions.sort_by_key(|&k| (ids[k] as u32, k));
    positions
        .into_iter()
        .flat_map(|k| [ids[k] as u32, k as u32])
        .collect()
}

/// The table gradient by the CPU's definition over the ids inside the table only: each element
/// from `+0.0`, plus the column of every position whose id is its row, in ascending position.
fn reference(ids: &[i32], g: &[f32], h: u64, v: u64) -> Vec<u32> {
    let h = h as usize;
    let mut out = vec![0.0f32; v as usize * h];
    for (k, &id) in ids.iter().enumerate() {
        if id < 0 || id as u64 >= v {
            continue;
        }
        let row = id as usize * h;
        for j in 0..h {
            out[row + j] += g[k * h + j];
        }
    }
    out.into_iter().map(f32::to_bits).collect()
}

/// `count` ids in `0..rows`, SplitMix64 of `salt` and the index.
fn hashed(count: u64, rows: u64, salt: u64) -> Vec<i32> {
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

/// The failures of one run against the stable sort, the guard words and `want`.
fn check(label: &str, ran: &Ran, ids: &[i32], want: &[u32]) -> Vec<String> {
    let mut failures = Vec::new();
    let sorted_len = 2 * ids.len();
    let expected = stable_sort(ids);
    if ran.sorted[..sorted_len] != expected[..] {
        let first = (0..sorted_len)
            .find(|&i| ran.sorted[i] != expected[i])
            .expect("a difference");
        failures.push(format!(
            "{label}: sorted row {} is ({:#x}, {:#x}), the stable sort has ({:#x}, {:#x})",
            first / 2,
            ran.sorted[first / 2 * 2],
            ran.sorted[first / 2 * 2 + 1],
            expected[first / 2 * 2],
            expected[first / 2 * 2 + 1]
        ));
    }
    for (name, words, len) in [
        ("sorted", &ran.sorted, sorted_len),
        ("gradient", &ran.grad, want.len()),
    ] {
        if (0..GUARD_WORDS).any(|i| words[len + i] != guard(i)) {
            failures.push(format!(
                "{label}: a guard word after the {name} buffer changed"
            ));
        }
    }
    let differ: Vec<usize> = (0..want.len())
        .filter(|&i| ran.grad[i] != want[i])
        .collect();
    if !differ.is_empty() {
        let at = differ[0];
        failures.push(format!(
            "{label}: {} gradient elements differ, first {at}: {:#010x} vs {:#010x}",
            differ.len(),
            ran.grad[at],
            want[at]
        ));
    }
    failures
}

/// Boundaries the unit's device test does not run: one id, a one-row table, `n` one below, at
/// and one above a workgroup multiple, every id equal (a one-row table, the last row, and the
/// masked-row gather of zero selected rows, all padding at row 0), descending ids, and the
/// masked-row shape of ascending distinct rows then padding. Each sort is the stable sort by
/// id and position, each gradient the CPU interpreter's bits, and both guard words intact.
#[test]
#[ignore = "device: opens a Vulkan instance"]
fn adv_eg_edge_shapes_sort_stably_and_give_the_cpu_bits() {
    let vulkan = open();
    let device = device_of(&vulkan);
    let mut draw = Draw::new(0x6164_765f_6567_0001);
    let masked: Vec<i32> = (0..100)
        .map(|i| i * 37 + 3)
        .chain(std::iter::repeat_n(0, 540))
        .collect();
    let cases: Vec<(&str, Vec<i32>, u64, u64)> = vec![
        ("n=1 v=1", vec![0], 1, 1),
        ("n=1 last row", vec![4], 3, 5),
        ("n=2 reversed", vec![1, 0], 1, 2),
        ("n=63", hashed(63, 2, 1), 5, 2),
        ("n=64 one row", vec![0; 64], 1, 1),
        ("n=65", hashed(65, 3, 2), 7, 3),
        ("n=127 descending", (0..127).rev().collect(), 2, 200),
        ("n=129 one row", vec![0; 129], 3, 1),
        ("n=4097 all last", vec![8; 4097], 1, 9),
        ("n=4097 hashed", hashed(4097, 8192, 3), 2, 8192),
        ("masked zero rows", vec![0; 640], 4, 4096),
        ("masked 100 rows", masked, 4, 4096),
    ];
    let mut failures = Vec::new();
    for (label, ids, h, v) in &cases {
        let g = draw.values(ids.len() as u64 * h);
        let want = cpu(ids, &g, *h, *v).expect("the CPU evaluates");
        assert_eq!(want, reference(ids, &g, *h, *v), "{label}: the reference");
        let ran = run(&device, ids, &g, *h, *v);
        failures.extend(check(label, &ran, ids, &want));
    }
    let messages = vulkan.take_validation_messages();
    eprintln!(
        "adv embgrad edge shapes: {} cases, {} failures",
        cases.len(),
        failures.len()
    );
    assert!(failures.is_empty(), "{failures:#?}");
    assert!(messages.is_empty(), "{messages:#?}");
}

/// Gradient rows of `-0.0`, as a weight-`+0` padding row of the masked-row gather can carry:
/// the CPU adds them to `+0.0` and writes `+0.0`, so the device must too, for every id equal
/// (zero selected rows) and for hashed ids.
#[test]
#[ignore = "device: opens a Vulkan instance"]
fn adv_eg_negative_zero_rows_sum_to_the_cpus_plus_zero() {
    let vulkan = open();
    let device = device_of(&vulkan);
    let (h, v) = (4_u64, 4096_u64);
    let mut failures = Vec::new();
    for (label, ids) in [
        ("all padding", vec![0; 640]),
        ("hashed", hashed(640, 64, 9)),
    ] {
        let g = vec![-0.0f32; ids.len() * h as usize];
        let want = cpu(&ids, &g, h, v).expect("the CPU evaluates");
        assert!(
            want.iter().all(|&b| b == 0),
            "{label}: the CPU writes +0.0 everywhere"
        );
        let ran = run(&device, &ids, &g, h, v);
        failures.extend(check(label, &ran, &ids, &want));
    }
    let messages = vulkan.take_validation_messages();
    assert!(failures.is_empty(), "{failures:#?}");
    assert!(messages.is_empty(), "{messages:#?}");
}

/// Ids the CPU refuses, negative and at or past the table: the device sorts them as `u32` after
/// every id inside the table, writes nothing past either buffer, and leaves every row of the
/// table the sum over the ids inside it.
#[test]
#[ignore = "device: opens a Vulkan instance"]
fn adv_eg_out_of_range_ids_write_nothing_outside_and_leave_in_range_rows() {
    let vulkan = open();
    let device = device_of(&vulkan);
    let (h, v) = (3_u64, 17_u64);
    let mut ids = hashed(200, v, 5);
    for (at, id) in [
        (0, -1),
        (7, i32::MIN),
        (8, v as i32),
        (63, v as i32 + 1),
        (64, i32::MAX),
        (199, -2),
    ] {
        ids[at] = id;
    }
    let mut draw = Draw::new(0x6164_765f_6567_0003);
    let g = draw.values(ids.len() as u64 * h);
    assert!(
        cpu(&ids, &g, h, v).is_err(),
        "the CPU refuses an id outside the table"
    );
    let want = reference(&ids, &g, h, v);
    let ran = run(&device, &ids, &g, h, v);
    let failures = check("out of range", &ran, &ids, &want);
    let messages = vulkan.take_validation_messages();
    assert!(failures.is_empty(), "{failures:#?}");
    assert!(messages.is_empty(), "{messages:#?}");
}

/// Batches 16 and 64 times BERT-tiny's 4,096 ids: the sort compares every id with every id, so
/// these run 2^32 and 2^36 compares; the result is still the CPU's bits and the stable sort.
#[test]
#[ignore = "device: opens a Vulkan instance"]
fn adv_eg_batches_far_past_bert_tiny_still_sort_and_give_the_cpu_bits() {
    let vulkan = open();
    let device = device_of(&vulkan);
    let (h, v) = (2_u64, 8192_u64);
    let mut draw = Draw::new(0x6164_765f_6567_0004);
    let mut failures = Vec::new();
    for n in [65_536_u64, 262_144] {
        let ids = hashed(n, v, n);
        let g = draw.values(n * h);
        let want = cpu(&ids, &g, h, v).expect("the CPU evaluates");
        let started = Instant::now();
        let ran = run(&device, &ids, &g, h, v);
        eprintln!(
            "adv embgrad n={n}: {:.1} ms with uploads",
            started.elapsed().as_secs_f64() * 1e3
        );
        failures.extend(check(&format!("n={n}"), &ran, &ids, &want));
    }
    let messages = vulkan.take_validation_messages();
    assert!(failures.is_empty(), "{failures:#?}");
    assert!(messages.is_empty(), "{messages:#?}");
}

/// The step plan of BERT-tiny with the masked-row gather at 640 rows: three `EmbeddingGrad`
/// sites (word table, position table, masked-row scatter), each an `Allocate` of its own
/// `i32[n, 2]` buffer right before its sort, the sort reading the `n` ids and writing that
/// buffer only, a barrier, then the gradient reading the buffer, then `g`. No other step reads
/// or writes a sorted buffer, every sorted buffer is numbered after every other buffer, the
/// sort counts 0 FLOP and the bytes of its two views, and the two sorts over 4,096 ids are one
/// module.
#[test]
fn adv_eg_bert_plan_allocates_sorts_and_barriers_before_each_gradient_at_three_sites() {
    let artifact = compile_artifact_with(
        "bert_tiny.mspec",
        BERT_TINY,
        Backend::Vulkan,
        None,
        Some(640),
    )
    .unwrap_or_else(|d| panic!("BERT-tiny compiles: {d:#?}"));
    let plan = &artifact.execution;
    let kernel = |step: &modelspec_core::ir::kernel::PlanStep| {
        step.kernel_module_id.and_then(|id| plan.kernel(id))
    };
    let is = |step: &modelspec_core::ir::kernel::PlanStep, kind: KernelKind| {
        step.op == PlanOp::Dispatch
            && kernel(step).is_some_and(|k| k.op == "EmbeddingGrad" && k.kind == kind)
    };
    let grads: Vec<usize> = (0..plan.step.len())
        .filter(|&i| is(&plan.step[i], KernelKind::PerElement))
        .collect();
    assert_eq!(grads.len(), 3, "three EmbeddingGrad sites");
    let mut sorted = BTreeSet::new();
    let mut ns = Vec::new();
    for &at in &grads {
        let grad = &plan.step[at];
        let sort_at = (0..at)
            .rev()
            .find(|&i| is(&plan.step[i], KernelKind::PerRow) && plan.step[i].value == grad.value)
            .expect("a sort before the gradient");
        let sort = &plan.step[sort_at];
        let allocate = &plan.step[sort_at - 1];
        assert_eq!(allocate.op, PlanOp::Allocate, "the allocate right before");
        assert_eq!(allocate.writes, sort.writes);
        assert_eq!(sort.reads.len(), 1);
        assert_eq!(sort.writes.len(), 1);
        let (ids, out) = (sort.reads[0], sort.writes[0]);
        assert_eq!(ids.dtype, DType::I32);
        assert_eq!(out.dtype, DType::I32);
        assert_eq!(out.element_count, 2 * ids.element_count);
        assert!(
            plan.step[sort_at + 1..at]
                .iter()
                .any(|s| s.op == PlanOp::Barrier),
            "a barrier between the sort and the gradient"
        );
        assert_eq!(grad.reads.len(), 2);
        assert_eq!(
            grad.reads[0], out,
            "the gradient reads the sorted ids first"
        );
        assert_eq!(grad.reads[1].element_count % ids.element_count, 0);
        assert_eq!(
            plan.dispatch_count(sort),
            Some(DispatchCount::Counted {
                flop: 0,
                bytes: ids.size_bytes + out.size_bytes
            })
        );
        for (i, step) in plan.step.iter().enumerate() {
            if [sort_at - 1, sort_at, at].contains(&i) {
                continue;
            }
            assert!(
                !step
                    .reads
                    .iter()
                    .chain(&step.writes)
                    .any(|view| view.buffer == out.buffer),
                "step {i} touches sorted buffer {}",
                out.buffer
            );
        }
        sorted.insert(out.buffer);
        ns.push(ids.element_count);
    }
    assert_eq!(sorted.len(), 3, "one sorted buffer per site");
    let others = plan
        .step
        .iter()
        .chain(&plan.forward)
        .flat_map(|s| s.reads.iter().chain(&s.writes))
        .map(|view| view.buffer)
        .filter(|b| !sorted.contains(b))
        .max()
        .expect("buffers");
    assert!(
        sorted.iter().all(|&b| b > others),
        "sorted buffers {sorted:?} after every other, the largest other {others}"
    );
    assert!(
        !plan
            .forward
            .iter()
            .any(|s| is(s, KernelKind::PerRow) || is(s, KernelKind::PerElement)),
        "no EmbeddingGrad in the forward plan"
    );
    let ids: BTreeSet<_> = plan.kernels.iter().map(|k| k.id).collect();
    assert_eq!(
        ids.len(),
        plan.kernels.len(),
        "each module once in the plan"
    );
    let modules = |kind: KernelKind| {
        plan.kernels
            .iter()
            .filter(|k| k.op == "EmbeddingGrad" && k.kind == kind)
            .count()
    };
    assert_eq!(
        (modules(KernelKind::PerRow), modules(KernelKind::PerElement)),
        (2, 3),
        "the word and position sorts over 4,096 ids share one module"
    );
    let dispatches = plan
        .step
        .iter()
        .filter(|s| s.op == PlanOp::Dispatch)
        .count();
    eprintln!("adv embgrad bert plan: ids {ns:?}, {dispatches} dispatches per step");
}

/// A word table compiled for Vulkan with its sort, written as an artifact and read back, is the
/// same artifact: the sort's `i32` kernel record survives the manifest.
#[test]
fn adv_eg_artifact_with_the_sort_reads_back_unchanged() {
    const WORDS: &str = "model Words {
    param table: Tensor<f32, [37, 5]> = xavier_uniform();

    fn forward(ids: Tensor<i32, [65]>) -> Tensor<f32, [65, 5]> {
        return embedding(table, ids);
    }
}

train Fit {
    model = Words;
    input ids: Tensor<i32, [65]>;
    target y: Tensor<f32, [65, 5]>;
    loss = mse(model.forward(ids), y);
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 7;
    device = cpu;
}
";
    let compiled = compile_artifact("words.mspec", WORDS, Backend::Vulkan)
        .unwrap_or_else(|d| panic!("compiles: {d:#?}"));
    assert!(
        compiled
            .execution
            .kernels
            .iter()
            .any(|k| k.op == "EmbeddingGrad" && k.kind == KernelKind::PerRow),
        "the sort is a kernel of the plan"
    );
    let profile = DeviceProfile::from_json(FIXTURE).expect("the fixture is a profile");
    let modules: Vec<KernelModuleFile> = emit_kernels(&compiled, &profile)
        .expect("emits")
        .into_iter()
        .map(|m| {
            let spirv = m.bytes();
            let key = m.cache_key;
            KernelModuleFile {
                kernel_module_id: m.kernel_module_id,
                entry_point: m.entry_point,
                cache_key: KernelCacheKey {
                    ir_hash: key.ir_hash.0,
                    shapes: key.shapes,
                    dtype: key.dtype,
                    fast_math: key.fast_math,
                    deterministic: key.deterministic,
                    vendor_id: key.vendor_id,
                    device_id: key.device_id,
                    driver_version: key.driver_version,
                    capabilities: key.capabilities,
                },
                spirv,
            }
        })
        .collect();
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("adv-embgrad-artifact");
    artifact::write(&dir, &compiled, &modules).expect("written");
    let read = artifact::read(&dir).unwrap_or_else(|d| panic!("reads back: {d:?}"));
    assert_eq!(read.artifact, compiled);
}
