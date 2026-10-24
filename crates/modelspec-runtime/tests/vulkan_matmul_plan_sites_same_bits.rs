//! `story:matmul-reads-transposed-operand`, acceptance `same-bits`: every matmul site of the
//! step plans of the MNIST example and of the `mlp_sweep` sources at b1000 h1024, b8192 h1024
//! and b8192 h4096 writes, on the same device, the bits the same site wrote in the plan before
//! the change, over the same logical operands.
//!
//! A site is the `i`-th matmul dispatch of a step plan. Its operands are drawn from a generator
//! seeded by the plan and the site; an operand the kernel reads transposed is bound as the
//! row-major transpose of the drawn operand, which is what the plan binds there (the view of the
//! tensor the folded transpose read). `tests/fixtures/matmul_plan_sites_same_bits.txt` holds one
//! FNV-1a digest per written view, recorded at the base commit (where no matmul reads an operand
//! transposed) with `MODELSPEC_SAME_BITS_RECORD=1`, under a header naming the device and driver
//! it ran on; a run on another device or driver fails rather than comparing. Re-recorded once at
//! the tree of story:deterministic-fma, whose fused multiply-add changed every view (DESIGN.md
//! § 12 ADR-009).
//!
//! Ignored by default: it creates a Vulkan instance, which `task ci` never does. Run it with
//! `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test vulkan_matmul_plan_sites_same_bits -- --ignored`.
#![cfg(feature = "vulkan")]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use modelspec_core::artifact::Backend;
use modelspec_core::compile::compile_artifact;
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{Epilogue, ExecutionPlan, Kernel, KernelKind, PlanOp};
use modelspec_runtime::spirv::emit::emit_module;
use modelspec_runtime::vulkan::memory::Buffer;
use modelspec_runtime::vulkan::negotiate::negotiate;
use modelspec_runtime::vulkan::sync::Barrier;
use modelspec_runtime::vulkan::{Device, VALIDATION_ENV, Vulkan};

/// Set to `1` to write the fixture from this tree instead of comparing against it.
const RECORD_ENV: &str = "MODELSPEC_SAME_BITS_RECORD";

const GUARD_WORDS: usize = 16;

/// The bits each output element holds before the dispatch: a NaN no kernel computes.
const UNWRITTEN: u32 = 0x7fc0_dead;

/// Which operands `kernel` reads transposed.
fn reads_transposed(kernel: &Kernel) -> [bool; 2] {
    kernel.transposed
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/matmul_plan_sites_same_bits.txt")
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

/// The step plans the story names, by name.
fn plans() -> Vec<(String, ExecutionPlan)> {
    let mnist = std::fs::read_to_string(repo_root().join("examples/mnist_mlp.mspec"))
        .expect("the example reads");
    let mut sources = vec![("mnist_mlp".to_owned(), mnist)];
    for (batch, hidden) in [(1000, 1024), (8192, 1024), (8192, 4096)] {
        sources.push((format!("sweep_b{batch}_h{hidden}"), sweep(batch, hidden)));
    }
    sources
        .into_iter()
        .map(|(name, source)| {
            let plan = compile_artifact(&name, &source, Backend::Vulkan)
                .unwrap_or_else(|d| panic!("{name} compiles: {d:?}"))
                .execution;
            (name, plan)
        })
        .collect()
}

/// The matmul kernel of each matmul dispatch of `plan`'s step plan, in plan order.
fn matmul_sites(plan: &ExecutionPlan) -> Vec<&Kernel> {
    plan.step
        .iter()
        .filter(|s| s.op == PlanOp::Dispatch)
        .map(|s| {
            plan.kernel(s.kernel_module_id.expect("a dispatch names its kernel"))
                .expect("the plan has the kernel")
        })
        .filter(|k| k.kind == KernelKind::MatMul)
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

/// A seed of the text `name` and the number `site`.
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

/// FNV-1a over the little-endian bytes of each value's bits.
fn digest(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
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

fn device_line(profile: &DeviceProfile) -> String {
    format!(
        "device vulkan {:04x}:{:04x} driver_version {}",
        profile.vendor_id, profile.device_id, profile.driver_version
    )
}

fn epilogue_name(epilogue: Option<Epilogue>) -> &'static str {
    match epilogue {
        None => "none",
        Some(Epilogue::Relu {
            writes_pre_activation: true,
        }) => "relu-keep",
        Some(Epilogue::Relu {
            writes_pre_activation: false,
        }) => "relu",
    }
}

#[test]
#[ignore = "needs a Vulkan device"]
fn every_matmul_site_of_the_step_plans_writes_the_bits_of_the_plan_before_the_change() {
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
    let mut lines = String::new();
    writeln!(lines, "{}", device_line(&profile)).expect("a string");
    let mut wrong = Vec::new();
    let mut flagged = 0;
    for (name, plan) in plans() {
        for (site, kernel) in matmul_sites(&plan).into_iter().enumerate() {
            let (m, k, n) = Kernel::matmul_shape(&kernel.operands).expect("a static matmul");
            let transposed = reads_transposed(kernel);
            flagged += transposed.iter().filter(|t| **t).count();
            let a = values(seed_of(&name, site, 1), m * k);
            let b = values(seed_of(&name, site, 2), k * n);
            let a_bound = if transposed[0] {
                transpose(&a, m, k)
            } else {
                a
            };
            let b_bound = if transposed[1] {
                transpose(&b, k, n)
            } else {
                b
            };
            let module = emit_module(kernel, &profile).expect("the kernel emits");
            let count = (m * n) as usize;
            let views = run(&device, kernel, &module.spirv, [&a_bound, &b_bound], count);
            let case = format!(
                "{name} matmul {site} {m} {k} {n} {}",
                epilogue_name(kernel.epilogue)
            );
            for (at, view) in views.iter().enumerate() {
                let changed = (0..GUARD_WORDS)
                    .filter(|&i| view[count + i].to_bits() != guard(i))
                    .count();
                if changed > 0 {
                    wrong.push(format!("{case} view {at}: {changed} guard words changed"));
                }
                writeln!(lines, "{case} view {at} {:016x}", digest(&view[..count]))
                    .expect("a string");
            }
            for message in vulkan.take_validation_messages() {
                wrong.push(format!("{case}: {message}"));
            }
        }
    }
    if std::env::var(RECORD_ENV).as_deref() == Ok("1") {
        std::fs::write(fixture_path(), &lines).expect("the fixture is written");
        eprintln!(
            "recorded {} ({flagged} transposed operands)",
            fixture_path().display()
        );
    } else {
        let recorded = std::fs::read_to_string(fixture_path()).expect("the fixture reads");
        let mut recorded_lines = recorded.lines();
        let header = recorded_lines.next().unwrap_or_default();
        assert_eq!(
            header,
            device_line(&profile),
            "the fixture was recorded on another device or driver; record it again at the \
             base commit with {RECORD_ENV}=1"
        );
        let now: Vec<&str> = lines.lines().skip(1).collect();
        let before: Vec<&str> = recorded_lines.collect();
        assert_eq!(now.len(), before.len(), "one digest per written view");
        for (now, before) in now.iter().zip(&before) {
            if now != before {
                wrong.push(format!("now {now}, before {before}"));
            }
        }
        eprintln!(
            "same bits: {} views of 20 matmul sites compared against {}, {flagged} operands read transposed",
            now.len(),
            fixture_path().display()
        );
        assert!(flagged > 0, "the plans read operands transposed");
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
