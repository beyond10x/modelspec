//! Adversary cases for story:deterministic-fma without a device (DESIGN.md § 12 ADR-008,
//! ADR-009).
//!
//! 1. `fast-math-unchanged-in-meaning`, word for word: every `fast_math` module of the step
//!    plans of the MNIST and linear-regression examples and of the `mlp_sweep` sources at b1000
//!    h1024, b8192 h1024 and b8192 h4096, scheduled without a profile and for the RTX 3090
//!    profile, is its deterministic module with exactly its `OpDecorate … NoContraction`
//!    instructions removed: the same header, the same instructions in the same order, the same
//!    `Fma` instructions. The unit's own check compares counts and lengths only.
//! 2. The `mse` loss sum at 1 to 9 terms (no unrolled pass) holds exactly `count - 1` `Fma`,
//!    each squaring one staged difference (`Fma(d, d, acc)`, both factors the same id), and one
//!    `OpFMul` of one difference by itself; every `OpFSub` is decorated `NoContraction`.

#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use common::words::{
    DECORATION_NO_CONTRACTION, GLSL_FMA, OP_DECORATE, OP_EXT_INST, OP_F_MUL, OP_F_SUB, instructions,
};
use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{compile_artifact, compile_training};
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::MatMulLimits;
use modelspec_core::schedule::schedule_for;
use modelspec_runtime::spirv::emit::{emit_kernels, emit_loss_sum_undivided, emit_module};

const RTX3090: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

/// The words of the SPIR-V header before the first instruction.
const HEADER_WORDS: usize = 5;

fn rtx3090() -> DeviceProfile {
    DeviceProfile::from_json(RTX3090).expect("the fixture is a valid profile")
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

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

fn sources() -> Vec<(String, String)> {
    let mut sources = Vec::new();
    for example in ["mnist_mlp", "linear_regression"] {
        let path = repo_root().join(format!("examples/{example}.mspec"));
        sources.push((
            example.to_owned(),
            std::fs::read_to_string(path).expect("the example reads"),
        ));
    }
    for (batch, hidden) in [(1000, 1024), (8192, 1024), (8192, 4096)] {
        sources.push((format!("sweep_b{batch}_h{hidden}"), sweep(batch, hidden)));
    }
    sources
}

fn scheduled(name: &str, source: &str, profile: &DeviceProfile) -> CompiledArtifact {
    let compiled = compile_training(name, source).unwrap_or_else(|d| panic!("{name}: {d:?}"));
    let plan = schedule_for(
        &compiled.training,
        compiled.forward.prediction,
        &compiled.config,
        Some(MatMulLimits::of(profile)),
    )
    .expect("the profile's schedule");
    CompiledArtifact::new(Backend::Vulkan, source, &compiled.training, plan)
}

/// `module` without its `OpDecorate %x NoContraction` instructions, header included.
fn without_no_contraction(module: &[u32]) -> Vec<u32> {
    let mut out = module[..HEADER_WORDS].to_vec();
    for (op, words) in instructions(module) {
        if op == OP_DECORATE && words.len() == 3 && words[2] == DECORATION_NO_CONTRACTION {
            continue;
        }
        out.extend_from_slice(words);
    }
    out
}

#[test]
fn every_fast_math_module_is_its_deterministic_module_less_no_contraction_word_for_word() {
    let profile = rtx3090();
    let mut wrong = Vec::new();
    let (mut compared, mut with_fma) = (0, 0);
    for (name, source) in sources() {
        let plain = compile_artifact(&name, &source, Backend::Vulkan).expect("compiles");
        let fitted = scheduled(&name, &source, &profile);
        for (schedule, artifact) in [("no profile", plain), ("rtx3090", fitted)] {
            let mut fast = artifact.clone();
            fast.fast_math = true;
            let exact = emit_kernels(&artifact, &profile).expect("emits");
            let loose = emit_kernels(&fast, &profile).expect("emits");
            assert_eq!(exact.len(), loose.len());
            for (kernel, (e, f)) in artifact
                .execution
                .kernels
                .iter()
                .zip(exact.iter().zip(&loose))
            {
                let stripped = without_no_contraction(&e.spirv);
                if stripped != f.spirv {
                    let first = stripped
                        .iter()
                        .zip(&f.spirv)
                        .position(|(x, y)| x != y)
                        .unwrap_or(stripped.len().min(f.spirv.len()));
                    wrong.push(format!(
                        "{name} {schedule} {}: fast_math differs from the stripped deterministic \
                         module at word {first} ({} against {} words)",
                        kernel.signature(),
                        f.spirv.len(),
                        stripped.len()
                    ));
                }
                if instructions(&e.spirv)
                    .iter()
                    .any(|(op, w)| *op == OP_EXT_INST && w[4] == GLSL_FMA)
                {
                    with_fma += 1;
                }
                compared += 1;
            }
        }
    }
    eprintln!("adversary_fma_spirv: {compared} module pairs, {with_fma} with Fma");
    assert!(with_fma >= 50, "{with_fma} modules with Fma");
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn the_loss_sum_squares_each_staged_difference_once_and_fuses_every_later_square() {
    let profile = rtx3090();
    let mut wrong = Vec::new();
    for count in 1..=9u32 {
        let kernel = common::loss_sum_kernel(u64::from(count));
        let plan = emit_module(&kernel, &profile).expect("emits");
        let undivided = emit_loss_sum_undivided(&kernel, &profile).expect("emits");
        for (variant, module) in [("plan", &plan.spirv), ("undivided", &undivided.spirv)] {
            let all = instructions(module);
            let decorated: BTreeSet<u32> = all
                .iter()
                .filter(|(op, w)| *op == OP_DECORATE && w[2] == DECORATION_NO_CONTRACTION)
                .map(|(_, w)| w[1])
                .collect();
            let fmas: Vec<&[u32]> = all
                .iter()
                .filter(|(op, w)| *op == OP_EXT_INST && w[4] == GLSL_FMA)
                .map(|(_, w)| *w)
                .collect();
            let muls: Vec<&[u32]> = all
                .iter()
                .filter(|(op, _)| *op == OP_F_MUL)
                .map(|(_, w)| *w)
                .collect();
            let subs: Vec<&[u32]> = all
                .iter()
                .filter(|(op, _)| *op == OP_F_SUB)
                .map(|(_, w)| *w)
                .collect();
            let what = format!("mse over {count} terms {variant}");
            if fmas.len() != count as usize - 1 {
                wrong.push(format!("{what}: {} Fma, wanted {}", fmas.len(), count - 1));
            }
            for w in &fmas {
                if w[5] != w[6] {
                    wrong.push(format!(
                        "{what}: Fma %{} multiplies %{} by %{}",
                        w[2], w[5], w[6]
                    ));
                }
            }
            match muls.as_slice() {
                [w] if w[3] == w[4] => {}
                other => wrong.push(format!("{what}: {} OpFMul, not one square", other.len())),
            }
            if subs.iter().any(|w| !decorated.contains(&w[2])) {
                wrong.push(format!("{what}: an OpFSub without NoContraction"));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
