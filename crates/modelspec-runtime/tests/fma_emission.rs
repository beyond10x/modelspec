//! `story:deterministic-fma`, acceptance `fma-emitted` and `fast-math-unchanged-in-meaning`,
//! without a device (DESIGN.md § 12 ADR-009): every product-and-sum of a matmul or loss-sum
//! module is one `OpExtInst GLSL.std.450 Fma`.
//!
//! For every module of the step plans of the MNIST and linear-regression examples and of the
//! `mlp_sweep` sources at b1000 h1024, b8192 h1024 and b8192 h4096, scheduled without a profile
//! (tiles of 128 invocations) and for the RTX 3090 profile (`tests/fixtures/rtx3090.device.json`,
//! where three matmuls take the 128 by 128 block of 256 invocations), deterministic and
//! `fast_math`, and for the matmuls of `common::MATMUL_SHAPES` with each epilogue and each
//! combination of operands read transposed:
//!
//! - a matmul module holds no `OpFAdd`; one `OpFMul` per accumulator, the `k = 0` product that
//!   starts its sum; and one `Fma` per accumulator and later `k` it emits, whose two factors are
//!   workgroup-memory loads and whose addend is the accumulator: the `k = 0` product, an earlier
//!   `Fma` or a load of the accumulator's function variable;
//! - a loss-sum module holds no `OpFAdd`, one `OpFMul` (the first term), and an `Fma` for every
//!   later term, each squaring one difference `a - b` that the module stages;
//! - no `OpFAdd` anywhere reads an `OpFMul` result, the factors of every `Fma` are not themselves
//!   products, and `GLSL.std.450` is imported exactly once by a module that uses it and by no
//!   other module;
//! - in a deterministic module every remaining `OpFAdd`, `OpFSub` and `OpFMul` is decorated
//!   `NoContraction`; the `fast_math` module is the deterministic one less those decorations,
//!   with the same `Fma` instructions;
//! - every module passes `spirv-val --target-env vulkan1.2`.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use common::words::{
    DECORATION_NO_CONTRACTION, OP_DECORATE, OP_F_ADD, OP_F_MUL, OP_F_SUB, instructions,
};
use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{compile_artifact, compile_training};
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{Epilogue, Kernel, KernelKind, MatMulLimits, MatMulTile};
use modelspec_core::schedule::schedule_for;
use modelspec_runtime::spirv::emit::{emit_kernels, emit_loss_sum_undivided, emit_module};
use modelspec_runtime::spirv::validate::validate;

const OP_EXT_INST_IMPORT: u32 = 11;
const OP_EXT_INST: u32 = 12;
const OP_LOAD: u32 = 61;
const OP_ACCESS_CHAIN: u32 = 65;
const OP_VARIABLE: u32 = 59;
/// `GLSL.std.450` instruction `Fma`.
const GLSL_FMA: u32 = 50;
const STORAGE_FUNCTION: u32 = 7;
const STORAGE_WORKGROUP: u32 = 4;

const RTX3090: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

fn rtx3090() -> DeviceProfile {
    DeviceProfile::from_json(RTX3090).expect("the fixture is a valid profile")
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
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

/// The Vulkan artifact of `source` with the plan scheduled for `profile`.
fn artifact_for(name: &str, source: &str, profile: &DeviceProfile) -> CompiledArtifact {
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

/// What the arithmetic of one module is.
#[derive(Debug, Default)]
struct Census {
    f_add: usize,
    f_sub: usize,
    f_mul: usize,
    fma: usize,
    /// `OpExtInstImport "GLSL.std.450"` instructions.
    imports: usize,
    /// `OpFAdd`, `OpFSub` and `OpFMul` results not decorated `NoContraction`.
    uncontracted: usize,
    /// `Fma` results not decorated `NoContraction`.
    bare_fma: usize,
    /// `NoContraction` decorations.
    decorations: usize,
    /// What breaks the fused form, one line each.
    wrong: Vec<String>,
}

/// The census of `module`, whose accumulators are function variables and whose factors are
/// staged in workgroup memory.
fn census(module: &[u32]) -> Census {
    let all = instructions(module);
    let mut c = Census::default();
    let mut decorated = BTreeSet::new();
    let mut glsl = BTreeSet::new();
    // Result id to its opcode, and for an `Fma` its operands.
    let mut defined: BTreeMap<u32, u32> = BTreeMap::new();
    let mut fmas: Vec<(u32, [u32; 3])> = Vec::new();
    let mut variables: BTreeMap<u32, u32> = BTreeMap::new();
    let mut chains: BTreeMap<u32, u32> = BTreeMap::new();
    let mut loads: BTreeMap<u32, u32> = BTreeMap::new();
    let mut f_adds: Vec<[u32; 2]> = Vec::new();
    for (op, w) in &all {
        match *op {
            OP_DECORATE if w[2] == DECORATION_NO_CONTRACTION => {
                decorated.insert(w[1]);
                c.decorations += 1;
            }
            OP_EXT_INST_IMPORT => {
                let name: Vec<u8> = w[2..]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .take_while(|b| *b != 0)
                    .collect();
                if name == b"GLSL.std.450" {
                    glsl.insert(w[1]);
                    c.imports += 1;
                }
            }
            OP_VARIABLE => {
                variables.insert(w[2], w[3]);
            }
            OP_ACCESS_CHAIN => {
                chains.insert(w[2], w[3]);
            }
            OP_LOAD => {
                loads.insert(w[2], w[3]);
                defined.insert(w[2], *op);
            }
            OP_EXT_INST if glsl.contains(&w[3]) && w[4] == GLSL_FMA => {
                c.fma += 1;
                fmas.push((w[2], [w[5], w[6], w[7]]));
                defined.insert(w[2], OP_EXT_INST);
            }
            OP_F_ADD | OP_F_SUB | OP_F_MUL => {
                match *op {
                    OP_F_ADD => {
                        c.f_add += 1;
                        f_adds.push([w[3], w[4]]);
                    }
                    OP_F_SUB => c.f_sub += 1,
                    _ => c.f_mul += 1,
                }
                defined.insert(w[2], *op);
            }
            _ => {
                if w.len() > 2 {
                    defined.entry(w[2]).or_insert(*op);
                }
            }
        }
    }
    for (op, w) in &all {
        if [OP_F_ADD, OP_F_SUB, OP_F_MUL].contains(op) && !decorated.contains(&w[2]) {
            c.uncontracted += 1;
        }
    }
    c.bare_fma = fmas
        .iter()
        .filter(|(id, _)| !decorated.contains(id))
        .count();
    // Where a load reads: a function variable or workgroup memory.
    let storage_of = |load: u32| -> Option<u32> {
        let pointer = *loads.get(&load)?;
        let base = chains.get(&pointer).copied().unwrap_or(pointer);
        variables.get(&base).copied()
    };
    for operands in &f_adds {
        for operand in operands {
            if defined.get(operand) == Some(&OP_F_MUL) {
                c.wrong.push(format!("OpFAdd reads the OpFMul %{operand}"));
            }
        }
    }
    for (id, [x, y, acc]) in &fmas {
        for factor in [x, y] {
            if storage_of(*factor) != Some(STORAGE_WORKGROUP) {
                c.wrong.push(format!(
                    "Fma %{id}: factor %{factor} is not a value staged in workgroup memory"
                ));
            }
        }
        let addend = match defined.get(acc) {
            Some(&OP_F_MUL) | Some(&OP_EXT_INST) => true,
            Some(&OP_LOAD) => storage_of(*acc) == Some(STORAGE_FUNCTION),
            _ => false,
        };
        if !addend {
            c.wrong.push(format!(
                "Fma %{id}: addend %{acc} is not the product, an Fma or the accumulator"
            ));
        }
    }
    c
}

/// The `k` values a `k`-deep matmul with `tile` emits products for: every tile but those of
/// its tile loop past the loop's one body of 4.
fn emitted_k(k: u64, tile: &MatMulTile) -> u64 {
    let depth = u64::from(tile.depth);
    let full = k / depth;
    let passes = full.saturating_sub(u64::from(tile.ahead.max(1)) + 1) / 4;
    let looped = passes * 4;
    k - looped.saturating_sub(4) * depth
}

/// Checks one deterministic `module` of `kernel` and its `fast_math` twin, when there is one.
fn check(what: &str, kernel: &Kernel, module: &[u32], fast: Option<&[u32]>) -> Vec<String> {
    let mut wrong = Vec::new();
    if let Err(error) = validate(module) {
        wrong.push(format!("{what}: {error}"));
    }
    let c = census(module);
    wrong.extend(c.wrong.iter().map(|w| format!("{what}: {w}")));
    let decorates = match kernel.kind {
        KernelKind::Elementwise | KernelKind::Broadcast => false,
        KernelKind::SgdUpdate => kernel.operands.len() == 3,
        _ => true,
    };
    if decorates && c.uncontracted + c.bare_fma > 0 {
        wrong.push(format!(
            "{what}: {} arithmetic results and {} Fma without NoContraction",
            c.uncontracted, c.bare_fma
        ));
    }
    let uses = c.fma > 0;
    if c.imports != usize::from(uses) {
        wrong.push(format!(
            "{what}: {} GLSL.std.450 imports for {} Fma",
            c.imports, c.fma
        ));
    }
    match kernel.kind {
        KernelKind::MatMul => {
            let (_, k, _) = Kernel::matmul_shape(&kernel.operands).expect("a static matmul");
            let tile = kernel.matmul_tile().expect("a tile");
            let accumulators = (tile.rows * tile.cols_per_invocation) as usize;
            let fused = accumulators * (emitted_k(k, &tile) - 1) as usize;
            if c.f_add != 0 || c.f_mul != accumulators || c.fma != fused {
                wrong.push(format!(
                    "{what}: {} OpFAdd, {} OpFMul, {} Fma; wanted 0, {accumulators} (the k = 0 \
                     products), {fused}",
                    c.f_add, c.f_mul, c.fma
                ));
            }
        }
        KernelKind::LossSum => {
            if c.f_add != 0 || c.f_mul != 1 || c.f_sub == 0 {
                wrong.push(format!(
                    "{what}: {} OpFAdd, {} OpFMul, {} OpFSub, {} Fma; wanted no OpFAdd and one \
                     OpFMul, the first term",
                    c.f_add, c.f_mul, c.f_sub, c.fma
                ));
            }
        }
        KernelKind::ReduceStage1 | KernelKind::ReduceStage2 => {
            if c.f_mul != 0 || c.fma != 0 {
                wrong.push(format!("{what}: a reduction multiplies"));
            }
        }
        _ => {
            if c.fma != 0 {
                wrong.push(format!(
                    "{what}: {} Fma outside a matmul or loss sum",
                    c.fma
                ));
            }
        }
    }
    if let Some(fast) = fast {
        let f = census(fast);
        if f.decorations != 0 || f.fma != c.fma || f.imports != c.imports {
            wrong.push(format!(
                "{what}: fast_math keeps {} NoContraction, {} Fma against {}",
                f.decorations, f.fma, c.fma
            ));
        }
        if fast.len() + 3 * c.decorations != module.len() {
            wrong.push(format!(
                "{what}: the fast_math module is {} words, the deterministic one {} with {} \
                 NoContraction",
                fast.len(),
                module.len(),
                c.decorations
            ));
        }
    }
    wrong
}

/// Acceptance `fma-emitted` and `fast-math-unchanged-in-meaning` over the step plans.
#[test]
fn every_product_and_sum_of_the_step_plans_is_one_fma() {
    let profile = rtx3090();
    let mut wrong = Vec::new();
    let (mut matmuls, mut blocks, mut losses, mut fused) = (0, 0, 0, 0);
    for (name, source) in sources() {
        let plain = compile_artifact(&name, &source, Backend::Vulkan).expect("compiles");
        let scheduled = artifact_for(&name, &source, &profile);
        for (schedule, artifact) in [("no profile", plain), ("rtx3090", scheduled)] {
            let mut fast = artifact.clone();
            fast.fast_math = true;
            let modules = emit_kernels(&artifact, &profile).expect("every kernel emits");
            let fast_modules = emit_kernels(&fast, &profile).expect("every kernel emits");
            for ((kernel, module), fast) in artifact
                .execution
                .kernels
                .iter()
                .zip(&modules)
                .zip(&fast_modules)
            {
                let what = format!("{name} {schedule}: {}", kernel.signature());
                wrong.extend(check(&what, kernel, &module.spirv, Some(&fast.spirv)));
                fused += census(&module.spirv).fma;
                match kernel.kind {
                    KernelKind::MatMul => {
                        matmuls += 1;
                        blocks += usize::from(kernel.workgroup_size == 256);
                    }
                    KernelKind::LossSum => {
                        losses += 1;
                        let undivided =
                            emit_loss_sum_undivided(kernel, &profile).expect("the sum emits");
                        wrong.extend(check(
                            &format!("{what} undivided"),
                            kernel,
                            &undivided.spirv,
                            None,
                        ));
                    }
                    _ => {}
                }
            }
        }
    }
    eprintln!(
        "fma_emission: {matmuls} matmul and {losses} loss-sum modules of the step plans, \
         {blocks} of 256 invocations, {fused} Fma"
    );
    assert!(
        matmuls >= 40 && blocks == 3 && losses == 10,
        "{matmuls} {blocks} {losses}"
    );
    assert!(wrong.is_empty(), "{:#?}", &wrong[..wrong.len().min(16)]);
}

/// Acceptance `fma-emitted` per tile: every shape of `common::MATMUL_SHAPES` (one and two rows
/// per invocation, the square and the wide register-blocked tiles, short and looped `k`), each
/// epilogue, each operand read as is or transposed, without and with the RTX 3090 profile.
#[test]
fn every_matmul_tile_and_staging_path_fuses_each_product_into_its_sum() {
    let epilogues = [
        None,
        Some(Epilogue::Relu {
            writes_pre_activation: false,
        }),
        Some(Epilogue::Relu {
            writes_pre_activation: true,
        }),
    ];
    let profile = rtx3090();
    let mut wrong = Vec::new();
    let mut tiles = BTreeSet::new();
    let mut checked = 0;
    let shapes = common::MATMUL_SHAPES.iter().copied().chain([
        (8192, 784, 4096),
        (8192, 10, 4096),
        (784, 8192, 4096),
        (4096, 8192, 10),
    ]);
    for (m, k, n) in shapes {
        for epilogue in epilogues {
            for transposed in [[false, false], [true, false], [false, true], [true, true]] {
                let mut kernel = common::matmul_kernel_with(m, k, n, epilogue);
                kernel.transposed = transposed;
                kernel.id = Kernel::module_id(&kernel.signature());
                let tile = kernel.matmul_tile().expect("a tile");
                tiles.insert((tile.rows, tile.cols_per_invocation, tile.workgroup_size));
                let what = format!("[{m}, {k}] x [{k}, {n}] {epilogue:?} {transposed:?}");
                let module = emit_module(&kernel, &profile).expect("emits");
                wrong.extend(check(&what, &kernel, &module.spirv, None));
                checked += 1;
            }
        }
    }
    eprintln!("fma_emission: {checked} matmul modules, tiles {tiles:?}");
    assert!(tiles.len() >= 4, "{tiles:?}");
    assert!(wrong.is_empty(), "{:#?}", &wrong[..wrong.len().min(16)]);
}

/// Acceptance `fma-emitted` for the `mse` loss sum: at term counts that do not fill the
/// workgroup, fill it, take several chunks and leave a partial one, with the workgroup memory
/// of the RTX 3090 and with copies that hold 64 and 65 terms, the plan's module and the
/// undivided one.
#[test]
fn the_loss_sum_fuses_each_square_into_its_sum() {
    let mut wrong = Vec::new();
    let mut checked = 0;
    for count in [1, 2, 63, 64, 65, 1_000, 12_289, 100_000] {
        let kernel = common::loss_sum_kernel(count);
        for bytes in [None, Some(256), Some(260)] {
            let mut profile = rtx3090();
            if let Some(bytes) = bytes {
                profile.max_shared_memory_bytes = bytes;
            }
            let what = format!("mse over {count} terms, {bytes:?} bytes");
            let plan = emit_module(&kernel, &profile).expect("emits");
            let undivided = emit_loss_sum_undivided(&kernel, &profile).expect("emits");
            for (variant, module) in [("plan", &plan.spirv), ("undivided", &undivided.spirv)] {
                let mut found = check(&format!("{what} {variant}"), &kernel, module, None);
                let c = census(module);
                if count > 1 && c.fma == 0 {
                    found.push(format!("{what} {variant}: no Fma for {count} terms"));
                }
                wrong.extend(found);
                checked += 1;
            }
        }
    }
    eprintln!("fma_emission: {checked} loss-sum modules");
    assert!(wrong.is_empty(), "{:#?}", &wrong[..wrong.len().min(16)]);
}
