//! story:jepa-sigreg-vulkan, DESIGN.md § 5.2 and § 9 gate 6, no Vulkan instance: `SigregColumns`
//! and `SigregColumnsGrad` are per-column kernels, one dispatch each with a kernel record of
//! workgroup size 64, counted by the FLOP rules of DESIGN.md § 5.2; each module passes
//! `spirv-val --target-env vulkan1.2` on the shapes of a compiled program and on every shape of
//! the device test, computes `sin` and `cos` by the CPU's written-out routine (`RoundEven` for the
//! quadrant, never `GLSL.std.450 Sin` or `Cos`), and decorates every float add, subtract,
//! multiply and `Fma` `NoContraction` in the deterministic mode and none under `fast_math`.
//! Inference on the device of a program with `sigreg` is refused before Vulkan is loaded.
//! Needs `spirv-val` on the path and fails, not skips, without it.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::words::{
    DECORATION_NO_CONTRACTION, GLSL_FMA, OP_DECORATE, OP_EXT_INST, OP_F_ADD, OP_F_MUL, OP_F_SUB,
    instructions,
};
use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::{compile_artifact, compile_training};
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{DispatchCount, Kernel, KernelKind, Operand, PlanOp};
use modelspec_core::ir::print::print_kernel;
use modelspec_core::ir::tensor::{TensorOp, TensorType};
use modelspec_core::schedule::rowwise_kernel;
use modelspec_runtime::spirv::emit::{emit, emit_kernels};
use modelspec_runtime::spirv::validate::validate;
use modelspec_runtime::vulkan::runner::{DeviceRunError, infer_on_device};

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

/// `GLSL.std.450` instruction numbers: `RoundEven`, `Sin`, `Cos`.
const GLSL_ROUND_EVEN: u32 = 2;
const GLSL_SIN: u32 = 13;
const GLSL_COS: u32 = 14;

/// The two ops by their Tensor IR names.
const SIGREG: [&str; 2] = ["SigregColumns", "SigregColumnsGrad"];

/// A program whose loss is `sigreg` of a `[65, 8]` parameter over 63 slices: `P` is `[65, 63]`,
/// and neither dimension fills a workgroup.
const SPREAD: &str = "model Spread {
    param z: Tensor<f32, [65, 8]> = xavier_uniform();

    fn forward(x: Tensor<f32, [65, 8]>) -> Tensor<f32, [65, 8]> {
        return add(z, mul(x, 0.0));
    }
}

train Fit {
    model = Spread;
    input x: Tensor<f32, [65, 8]>;
    target y: Tensor<f32, [65, 8]>;
    loss = add(sigreg(model.forward(x), slices = 63), mul(mse(x, y), 0.0));
    optimizer = sgd(lr = 0.5);
    steps = 2;
    seed = 5;
    device = cpu;
}
";

fn spread() -> CompiledArtifact {
    compile_artifact("spread.mspec", SPREAD, Backend::Vulkan)
        .unwrap_or_else(|errors| panic!("the program compiles for Vulkan: {errors:#?}"))
}

/// The FLOP rule of `SigregColumns` over `[n, m]` (DESIGN.md § 5.2).
fn columns_flop(n: u64, m: u64) -> u64 {
    m * 8 * (27 * n + 16)
}

/// The FLOP rule of `SigregColumnsGrad` over `[n, m]` (DESIGN.md § 5.2).
fn grad_flop(n: u64, m: u64) -> u64 {
    m * (456 * n + 97)
}

fn f32s(shape: &[u64]) -> Operand {
    Operand::Buffer(TensorType::f32(shape))
}

/// The two kernels of `P: [n, m]`, as the schedule builds them.
fn kernels_of_shape(n: u64, m: u64) -> [Kernel; 2] {
    let columns = rowwise_kernel(
        &TensorOp::SigregColumns,
        vec![f32s(&[n, m])],
        TensorType::f32(&[m]),
    )
    .unwrap_or_else(|| panic!("SigregColumns [{n}, {m}] has a per-column kernel"));
    let grad = rowwise_kernel(
        &TensorOp::SigregColumnsGrad,
        vec![f32s(&[n, m]), f32s(&[m])],
        TensorType::f32(&[n, m]),
    )
    .unwrap_or_else(|| panic!("SigregColumnsGrad [{n}, {m}] has a per-column kernel"));
    [columns, grad]
}

fn assert_valid(kernel: &Kernel) {
    let words = emit(kernel).unwrap_or_else(|error| panic!("{}: {error}", kernel.signature()));
    if let Err(error) = validate(&words) {
        panic!("{}: {error}", kernel.signature());
    }
}

/// Every module of the program's Vulkan artifact passes `spirv-val` as `emit_kernels` builds it
/// for the RTX 3090 profile; each of the two ops is one per-column kernel of workgroup size 64
/// whose workgroups cover the 63 columns, dispatched with a counted FLOP of its rule, printed by
/// `ir --stage kernel` with its kind and never `uncounted`.
#[test]
fn both_sigreg_kernels_of_a_compiled_program_are_counted_per_column_dispatches() {
    let artifact = spread();
    let plan = &artifact.execution;
    let profile = DeviceProfile::from_json(FIXTURE).expect("the fixture is a profile");
    let modules = emit_kernels(&artifact, &profile).expect("every kernel emits");
    for (kernel, module) in plan.kernels.iter().zip(&modules) {
        if let Err(error) = validate(&module.spirv) {
            panic!("{}: {error}", kernel.signature());
        }
    }
    let printed = print_kernel(plan);
    let (n, m) = (65, 63);
    for (op, flop) in SIGREG
        .into_iter()
        .zip([columns_flop(n, m), grad_flop(n, m)])
    {
        let kernels: Vec<&Kernel> = plan.kernels.iter().filter(|k| k.op == op).collect();
        assert_eq!(kernels.len(), 1, "one kernel of {op}:\n{printed}");
        let kernel = kernels[0];
        assert_eq!(kernel.kind, KernelKind::PerColumn, "{}", kernel.signature());
        assert_eq!(kernel.workgroup_size, 64, "{}", kernel.signature());
        assert_eq!(kernel.workgroups, [1, 1, 1], "{}", kernel.signature());
        assert_eq!(kernel.id, Kernel::module_id(&kernel.signature()));
        assert_eq!(kernel.flop(), Some(flop), "{}", kernel.signature());
        let dispatches: Vec<_> = plan
            .step
            .iter()
            .filter(|s| s.op == PlanOp::Dispatch && s.kernel_module_id == Some(kernel.id))
            .collect();
        assert_eq!(dispatches.len(), 1, "{op} is dispatched once per step");
        assert!(
            matches!(
                plan.dispatch_count(dispatches[0]),
                Some(DispatchCount::Counted { flop: f, bytes }) if f == flop && bytes > 0
            ),
            "{op}: {:?}",
            plan.dispatch_count(dispatches[0])
        );
        let id = kernel.id.to_string();
        let line = printed
            .lines()
            .find(|l| l.starts_with("kernel ") && l.contains(&id))
            .unwrap_or_else(|| panic!("no kernel line of {op}:\n{printed}"));
        assert!(line.contains(&format!(" {id} per_column {op} ")), "{line}");
        assert!(line.contains("workgroup_size 64"), "{line}");
    }
    for line in printed.lines().filter(|l| l.contains(": dispatch ")) {
        assert!(!line.ends_with(" uncounted"), "{line}");
    }
}

/// The shapes of the device test: `N` and `M` each one element, one short of, at and one past a
/// workgroup of 64, and the longest of the story.
const ROWS: [u64; 5] = [1, 63, 64, 65, 256];
const COLUMNS: [u64; 5] = [1, 63, 64, 65, 1024];

/// Both modules pass `spirv-val` on every shape of the device test, each with workgroups that
/// cover its columns, and the FLOP of each is its rule.
#[test]
fn both_sigreg_kernels_pass_spirv_val_on_every_shape_of_the_device_test() {
    let mut modules = 0;
    for n in ROWS {
        for m in COLUMNS {
            let [columns, grad] = kernels_of_shape(n, m);
            assert_eq!(columns.flop(), Some(columns_flop(n, m)));
            assert_eq!(grad.flop(), Some(grad_flop(n, m)));
            for kernel in [&columns, &grad] {
                assert_eq!(kernel.kind, KernelKind::PerColumn);
                let covered: u64 = kernel.workgroups.iter().map(|&w| u64::from(w)).product();
                assert_eq!(covered, m.div_ceil(64), "{}", kernel.signature());
                assert_valid(kernel);
                modules += 1;
            }
        }
    }
    assert_eq!(modules, 2 * ROWS.len() * COLUMNS.len());
}

/// `OpFDiv` and `OpFOrdNotEqual`, the division and compare of a corrected division.
const OP_F_DIV: u32 = 136;
const OP_F_ORD_NOT_EQUAL: u32 = 182;

/// The float operations a module holds, each instruction once, by the counting rule of
/// DESIGN.md § 3.1: each add, subtract, multiply, division and compare one, each `Fma` two;
/// negations, `RoundEven`, conversions and selects none.
fn weighted_float_instructions(words: &[u32]) -> u64 {
    instructions(words)
        .into_iter()
        .map(|(op, w)| match op {
            OP_F_ADD | OP_F_SUB | OP_F_MUL | OP_F_DIV | OP_F_ORD_NOT_EQUAL => 1,
            OP_EXT_INST if w[4] == GLSL_FMA => 2,
            _ => 0,
        })
        .sum()
}

/// D-S3-1: the FLOP rules are the count of the emitted instructions. Each module holds its
/// first row's operations straight, then one loop body per row: `SigregColumns` row 0 and one
/// body of its loop over rows `1..N`, so per column the rule at `N = 2`; `SigregColumnsGrad`
/// that and the body of its loop over every row once, so the rule at `N = 1` plus the means'
/// loop body, the rule of `SigregColumns` from one row to the next (216).
#[test]
fn the_flop_rules_count_the_emitted_float_instructions() {
    for (n, m) in [(1, 1), (65, 63), (256, 1024)] {
        let [columns, grad] = kernels_of_shape(n, m);
        let words = emit(&columns).expect("emits");
        assert_eq!(weighted_float_instructions(&words), columns_flop(2, 1));
        let words = emit(&grad).expect("emits");
        let means_row = columns_flop(2, 1) - columns_flop(1, 1);
        assert_eq!(means_row, 216);
        assert_eq!(
            weighted_float_instructions(&words),
            grad_flop(1, 1) + means_row
        );
    }
}

/// The float adds, subtracts, multiplies and `Fma`s of a module, and how many of them are
/// decorated `NoContraction`.
fn float_arithmetic(words: &[u32]) -> (usize, usize) {
    let all = instructions(words);
    let decorated: BTreeSet<u32> = all
        .iter()
        .filter(|(op, w)| *op == OP_DECORATE && w[2] == DECORATION_NO_CONTRACTION)
        .map(|(_, w)| w[1])
        .collect();
    let mut total = 0;
    let mut exact = 0;
    for (op, w) in &all {
        let result = match *op {
            OP_F_ADD | OP_F_SUB | OP_F_MUL => w[2],
            OP_EXT_INST if w[4] == GLSL_FMA => w[2],
            _ => continue,
        };
        total += 1;
        if decorated.contains(&result) {
            exact += 1;
        }
    }
    (total, exact)
}

/// The `GLSL.std.450` instruction numbers a module uses.
fn glsl_instructions(words: &[u32]) -> BTreeSet<u32> {
    instructions(words)
        .into_iter()
        .filter(|(op, _)| *op == OP_EXT_INST)
        .map(|(_, w)| w[4])
        .collect()
}

/// ADR-009 for the two kernels: in the deterministic mode every float add, subtract, multiply
/// and `Fma` is decorated `NoContraction`, the `fast_math` modules carry none, and both compute
/// `sin` and `cos` by the written-out routine: `RoundEven` and `Fma`, never `Sin` or `Cos`.
#[test]
fn every_sigreg_float_op_is_uncontracted_and_sin_cos_is_the_written_out_routine() {
    let artifact = spread();
    let mut fast = artifact.clone();
    fast.fast_math = true;
    let profile = DeviceProfile::from_json(FIXTURE).expect("the fixture is a profile");
    let off = emit_kernels(&artifact, &profile).expect("emits");
    let on = emit_kernels(&fast, &profile).expect("emits");
    let mut seen = BTreeSet::new();
    for ((kernel, off), on) in artifact.execution.kernels.iter().zip(&off).zip(&on) {
        if !SIGREG.contains(&kernel.op) {
            continue;
        }
        seen.insert(kernel.op);
        let (total, exact) = float_arithmetic(&off.spirv);
        assert!(total > 0, "{}", kernel.signature());
        assert_eq!(exact, total, "{}", kernel.signature());
        let (fast_total, fast_exact) = float_arithmetic(&on.spirv);
        assert_eq!(fast_total, total, "{}", kernel.signature());
        assert_eq!(fast_exact, 0, "{}", kernel.signature());
        let glsl = glsl_instructions(&off.spirv);
        assert!(glsl.contains(&GLSL_ROUND_EVEN), "{}", kernel.signature());
        assert!(glsl.contains(&GLSL_FMA), "{}", kernel.signature());
        assert!(!glsl.contains(&GLSL_SIN), "{}", kernel.signature());
        assert!(!glsl.contains(&GLSL_COS), "{}", kernel.signature());
    }
    assert_eq!(seen, SIGREG.into_iter().collect());
}

/// A training run uploads each site's projection before each step, but inference runs the step
/// plan with no step to draw it for: inference on the device of a program with `sigreg` is
/// `E-TARGET-011` at its `sigreg` call, refused before Vulkan is loaded, so this needs no device.
#[test]
fn device_inference_with_sigreg_is_refused_before_vulkan_is_loaded() {
    let training = compile_training("spread.mspec", SPREAD).expect("the program compiles");
    let artifact = spread();
    let refused = match infer_on_device(None, &training, &artifact, &BTreeMap::new()) {
        Err(DeviceRunError::Refused(diagnostic)) => diagnostic,
        Err(other) => panic!("refused before the run, not {other}"),
        Ok(_) => panic!("inference with sigreg is refused"),
    };
    assert_eq!(refused.code.to_string(), "E-TARGET-011", "{refused}");
    assert!(
        refused
            .to_string()
            .starts_with("E-TARGET-011: operator `sigreg` has no Vulkan inference yet"),
        "{refused}"
    );
    let (line, column) = SPREAD
        .lines()
        .enumerate()
        .find_map(|(i, text)| text.find("sigreg(").map(|at| (i + 1, at + 1)))
        .expect("the call");
    let span = refused.span.as_ref().expect("a span");
    assert_eq!(
        (span.line as usize, span.column as usize),
        (line, column),
        "{refused}"
    );
}
