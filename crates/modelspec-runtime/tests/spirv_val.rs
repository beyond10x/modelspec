//! DESIGN.md § 9 gate 6: every module the emitter builds passes
//! `spirv-val --target-env vulkan1.2`, in `task ci`: the elementwise, broadcast and SGD modules,
//! the reduction, loss-sum, transpose and matmul modules, and every module `emit_kernels` builds
//! for a whole artifact against `tests/fixtures/rtx3090.device.json`. No test here creates a
//! Vulkan instance; each needs `spirv-val` on the path and fails, not skips, without it.
#![cfg(feature = "vulkan")]

#[allow(dead_code)]
mod common;

use std::collections::BTreeSet;
use std::path::PathBuf;

use modelspec_core::artifact::{Backend, CompiledArtifact};
use modelspec_core::compile::compile_artifact;
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::{
    Epilogue, Kernel, KernelKind, MIN_SHARED_MEMORY_BYTES, MatMulTile, Operand,
};
use modelspec_core::ir::tensor::{DType, TensorType};
use modelspec_runtime::spirv::emit::{
    EmitError, FLOAT_CONTROL_CAPABILITIES, KernelCacheKey, emit, emit_kernels,
    emit_loss_sum_undivided, emit_module,
};
use modelspec_runtime::spirv::validate::{TARGET_ENV, ValidateError, validate, validate_with};

/// The SPIR-V opcode of `OpReturn`.
const OP_RETURN: u32 = 253;

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The elementwise, broadcast and SGD kernels of the Vulkan artifact of `path`, and its matmul
/// kernels with an epilogue.
fn kernels_of(path: &str) -> Vec<Kernel> {
    let source = std::fs::read_to_string(repository().join(path)).expect("the example reads");
    let artifact = compile_artifact(path, &source, Backend::Vulkan).expect("the example compiles");
    artifact
        .execution
        .kernels
        .into_iter()
        .filter(|kernel| {
            matches!(
                kernel.kind,
                KernelKind::Elementwise | KernelKind::Broadcast | KernelKind::SgdUpdate
            ) || kernel.epilogue.is_some()
        })
        .collect()
}

fn assert_valid(kernel: &Kernel) {
    let words = emit(kernel).unwrap_or_else(|error| panic!("{}: {error}", kernel.signature()));
    if let Err(error) = validate(&words) {
        panic!("{}: {error}", kernel.signature());
    }
}

#[test]
fn spirv_val_accepts_every_elementwise_broadcast_and_sgd_module_of_the_examples() {
    let mut kinds = Vec::new();
    let mut ops = Vec::new();
    let mut total = 0;
    for path in [
        "examples/linear_regression.mspec",
        "crates/modelspec-docs/models/two_layer_relu.mspec",
        "examples/mnist_mlp.mspec",
    ] {
        let kernels = kernels_of(path);
        assert!(!kernels.is_empty(), "{path} has elementwise kernels");
        for kernel in &kernels {
            assert_valid(kernel);
            kinds.push(kernel.kind);
            match kernel.epilogue {
                Some(_) => ops.push("Relu"),
                // A fused elementwise chain (story:fuse-elementwise-chains-and-gradient-sums)
                // counts as each of its ops.
                None => ops.extend(
                    kernel
                        .op
                        .split('>')
                        .map(|link| link.split('@').next().unwrap_or(link)),
                ),
            }
            if kernel.kind == KernelKind::SgdUpdate && kernel.operands.len() == 3 {
                let words = emit(kernel).expect("the update emits");
                let (missing, all) = common::words::uncontracted_arithmetic(&words);
                assert_eq!(all, 2, "{path}: the scaled update multiplies and subtracts");
                assert!(
                    missing.is_empty(),
                    "{path}: no NoContraction on {missing:?}"
                );
                ops.push("ScaledSub");
            }
        }
        total += kernels.len();
    }
    for kind in [
        KernelKind::Elementwise,
        KernelKind::SgdUpdate,
        KernelKind::MatMul,
    ] {
        assert!(kinds.contains(&kind), "the examples have a {kind} kernel");
    }
    for op in ["Sub", "Mul", "Relu", "ReluGrad", "ScaledSub"] {
        assert!(ops.contains(&op), "the examples have a {op} kernel");
    }
    eprintln!("spirv_val: {total} example modules pass spirv-val --target-env {TARGET_ENV}");
}

#[test]
fn spirv_val_accepts_every_form_of_the_catalog_at_every_test_size() {
    let mut total = 0;
    for count in [1, 63, 64, 65, 4_097, 78_400] {
        for case in common::catalog() {
            assert_valid(&common::kernel(&case, count));
            total += 1;
        }
    }
    eprintln!("spirv_val: {total} catalog modules pass spirv-val --target-env {TARGET_ENV}");
}

/// `words` with its first `OpReturn` removed.
fn without_return(words: &[u32]) -> Vec<u32> {
    let mut out = words[..5].to_vec();
    let mut at = 5;
    let mut removed = false;
    while at < words.len() {
        let length = (words[at] >> 16) as usize;
        let opcode = words[at] & 0xffff;
        assert!(length > 0, "an instruction has a length");
        if opcode == OP_RETURN && !removed {
            removed = true;
        } else {
            out.extend_from_slice(&words[at..at + length]);
        }
        at += length;
    }
    assert!(removed, "the module has an OpReturn");
    out
}

#[test]
fn spirv_val_refuses_a_module_with_one_instruction_removed() {
    let kernel = common::kernel(&common::catalog()[0], 65);
    let words = emit(&kernel).expect("the kernel emits");
    validate(&words).expect("the whole module is valid");
    match validate(&without_return(&words)) {
        Err(ValidateError::Invalid { output }) => assert!(!output.is_empty()),
        other => panic!("a module without its OpReturn passed or did not run: {other:?}"),
    }
}

#[test]
fn spirv_val_absent_is_an_error_not_a_pass() {
    let words = emit(&common::kernel(&common::catalog()[0], 1)).expect("the kernel emits");
    match validate_with("modelspec-no-such-spirv-val".as_ref(), &words) {
        Err(ValidateError::Unavailable { program, .. }) => {
            assert_eq!(program, "modelspec-no-such-spirv-val");
        }
        other => panic!("a missing spirv-val was not an error: {other:?}"),
    }
}

#[test]
fn emit_refuses_an_op_its_kind_does_not_have_and_a_matmul_of_the_wrong_result_shape() {
    let mut kernel = common::kernel(&common::catalog()[0], 8);
    kernel.op = "MatMul";
    assert!(matches!(emit(&kernel), Err(EmitError::Unsupported { .. })));
    kernel.kind = KernelKind::MatMul;
    kernel.operands = vec![
        Operand::Buffer(TensorType::f32(&[2, 4])),
        Operand::Buffer(TensorType::f32(&[4, 4])),
    ];
    assert!(
        matches!(emit(&kernel), Err(EmitError::Operands { .. })),
        "a [2, 4] x [4, 4] matmul into an f32[8] has no module"
    );
}

// story:m1-vulkan-reduce-matmul acceptance 2 and 3.

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/rtx3090.device.json"
));

fn fixture() -> DeviceProfile {
    DeviceProfile::from_json(FIXTURE).expect("the fixture is a valid profile")
}

/// The fixture with the three float-control values set to `values`.
fn fixture_with(values: [bool; 3]) -> DeviceProfile {
    let mut profile = fixture();
    profile.shader_rounding_mode_rte_f32 = Some(values[0]);
    profile.shader_denorm_preserve_f32 = Some(values[1]);
    profile.shader_signed_zero_inf_nan_preserve_f32 = Some(values[2]);
    profile
}

/// The linear-regression example with the loss `mean(mul(model.forward(x), y))`, the source of
/// `story:m1-kernel-plan` acceptance 3.
fn mean_loss_source() -> String {
    let source = std::fs::read_to_string(repository().join("examples/linear_regression.mspec"))
        .expect("the example reads");
    let replaced = source.replace(
        "loss = mse(model.forward(x), y);",
        "loss = mean(mul(model.forward(x), y));",
    );
    assert_ne!(replaced, source, "the loss was replaced");
    replaced
}

/// A program whose forward multiplies the transpose of its input by a parameter, elementwise: its
/// transpose is read by an elementwise kernel, so it stays a `Transpose` dispatch, where every
/// transpose of the examples is read only by matmuls, which read it in place (DESIGN.md § 3.1).
fn elementwise_transpose_source() -> String {
    "model T {
    param w: Tensor<f32, [24, 24]> = xavier_uniform();

    fn forward(x: Tensor<f32, [24, 24]>)
        -> Tensor<f32, [24, 24]> {
        return mul(transpose(x), w);
    }
}

train Fit {
    model = T;
    input x: Tensor<f32, [24, 24]>;
    target y: Tensor<f32, [24, 24]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 1;
    seed = 42;
    device = vulkan(0);
}
"
    .to_owned()
}

/// The Vulkan artifacts of acceptance 2, by name.
fn acceptance_artifacts() -> Vec<(String, CompiledArtifact)> {
    let mut artifacts = Vec::new();
    for path in [
        "examples/linear_regression.mspec",
        "crates/modelspec-docs/models/two_layer_relu.mspec",
        "examples/mnist_mlp.mspec",
    ] {
        let source = std::fs::read_to_string(repository().join(path)).expect("the example reads");
        let artifact =
            compile_artifact(path, &source, Backend::Vulkan).expect("the example compiles");
        artifacts.push((path.to_owned(), artifact));
    }
    let artifact = compile_artifact("mean_loss.mspec", &mean_loss_source(), Backend::Vulkan)
        .expect("the mean source compiles");
    artifacts.push(("mean_loss.mspec".to_owned(), artifact));
    let artifact = compile_artifact(
        "elementwise_transpose.mspec",
        &elementwise_transpose_source(),
        Backend::Vulkan,
    )
    .expect("the transpose source compiles");
    artifacts.push(("elementwise_transpose.mspec".to_owned(), artifact));
    artifacts
}

fn has_float_controls(kind: KernelKind) -> bool {
    matches!(kind, KernelKind::LossSum | KernelKind::MatMul)
}

#[test]
fn emit_kernels_gives_one_valid_module_per_kernel_of_every_acceptance_artifact() {
    let profile = fixture();
    let mut kinds = BTreeSet::new();
    let mut total = 0;
    let mut checked_arithmetic = 0;
    let mut transposed = 0;
    for (name, artifact) in acceptance_artifacts() {
        let modules = emit_kernels(&artifact, &profile).unwrap_or_else(|e| panic!("{name}: {e}"));
        let kernels = &artifact.execution.kernels;
        assert_eq!(
            modules.len(),
            kernels.len(),
            "{name}: one module per kernel"
        );
        for (module, kernel) in modules.iter().zip(kernels) {
            let what = format!("{name}: {}", kernel.signature());
            assert_eq!(module.kernel_module_id, kernel.id, "{what}");
            assert_eq!(module.entry_point, "main", "{what}");
            assert_eq!(module.bytes().len(), module.spirv.len() * 4, "{what}");
            if let Err(error) = validate(&module.spirv) {
                panic!("{what}: {error}");
            }
            if has_float_controls(kernel.kind) {
                let (missing, all) = common::words::uncontracted_arithmetic(&module.spirv);
                assert!(all > 0, "{what}: the module computes");
                assert!(
                    missing.is_empty(),
                    "{what}: no NoContraction on {missing:?}"
                );
                checked_arithmetic += all;
            }
            kinds.insert(kernel.kind);
            transposed += usize::from(kernel.transposed.contains(&true));
        }
        total += modules.len();
    }
    for kind in [
        KernelKind::Elementwise,
        KernelKind::ReduceStage1,
        KernelKind::ReduceStage2,
        KernelKind::LossSum,
        KernelKind::Transpose,
        KernelKind::MatMul,
        KernelKind::SgdUpdate,
    ] {
        assert!(kinds.contains(&kind), "the artifacts have a {kind} kernel");
    }
    assert!(transposed > 0, "a matmul reads an operand transposed");
    eprintln!(
        "spirv_val: emit_kernels gave {total} modules over 5 artifacts, kinds {kinds:?}, {transposed} matmuls reading an \
         operand transposed, all pass \
         spirv-val --target-env {TARGET_ENV}; {checked_arithmetic} OpFAdd/OpFSub/OpFMul of \
         loss-sum and matmul modules carry NoContraction"
    );
}

#[test]
fn emit_kernels_of_a_cpu_artifact_is_empty() {
    let path = "examples/linear_regression.mspec";
    let source = std::fs::read_to_string(repository().join(path)).expect("the example reads");
    let artifact = compile_artifact(path, &source, Backend::Cpu).expect("the example compiles");
    assert!(emit_kernels(&artifact, &fixture()).unwrap().is_empty());
}

/// Every reduction, loss-sum, transpose and matmul kernel of the device test, at its shapes.
fn device_test_kernels() -> Vec<Kernel> {
    let mut kernels = Vec::new();
    for count in common::REDUCE_COUNTS {
        for mean in [false, true] {
            let (stage1, stage2) = common::reduce_kernels(mean, count);
            kernels.push(stage1);
            kernels.push(stage2);
        }
        kernels.push(common::loss_sum_kernel(count));
    }
    for (rows, cols) in common::TRANSPOSE_SHAPES {
        kernels.push(common::transpose_kernel(rows, cols));
    }
    for (m, k, n) in common::MATMUL_SHAPES {
        kernels.push(common::matmul_kernel(m, k, n));
    }
    kernels
}

#[test]
fn spirv_val_accepts_every_reduction_transpose_and_matmul_module_of_the_device_test() {
    let mut total = 0;
    for profile in [fixture(), fixture_with([true; 3])] {
        for kernel in device_test_kernels() {
            let module = emit_module(&kernel, &profile)
                .unwrap_or_else(|error| panic!("{}: {error}", kernel.signature()));
            if let Err(error) = validate(&module.spirv) {
                panic!("{}: {error}", kernel.signature());
            }
            total += 1;
            if kernel.kind == KernelKind::LossSum {
                let undivided = emit_loss_sum_undivided(&kernel, &profile).expect("it emits");
                validate(&undivided.spirv).expect("the undivided loss sum is valid");
                assert_ne!(undivided.spirv, module.spirv, "the division is gone");
                assert_ne!(undivided.cache_key, module.cache_key);
                let (missing, _) = common::words::uncontracted_arithmetic(&undivided.spirv);
                assert!(missing.is_empty(), "{missing:?}");
                total += 1;
            }
        }
    }
    eprintln!("spirv_val: {total} device-test modules pass spirv-val --target-env {TARGET_ENV}");
}

/// The capabilities among the three float-control names that `key` names.
fn float_control_capabilities(key: &KernelCacheKey) -> Vec<&str> {
    key.capabilities
        .iter()
        .map(String::as_str)
        .filter(|name| FLOAT_CONTROL_CAPABILITIES.contains(name))
        .collect()
}

#[test]
fn float_control_modes_are_declared_exactly_on_loss_sum_and_matmul_when_all_three_are_reported() {
    let mut kernels = device_test_kernels();
    for (_, artifact) in acceptance_artifacts() {
        kernels.extend(artifact.execution.kernels);
    }
    let all_three = fixture_with([true; 3]);
    let with = |kernel: &Kernel, profile: &DeviceProfile| {
        let module = emit_module(kernel, profile).expect("the kernel emits");
        validate(&module.spirv).expect("the module is valid");
        module
    };
    let mut declaring = 0;
    for kernel in &kernels {
        let what = kernel.signature();
        let module = with(kernel, &all_three);
        let modes = common::words::float_control_modes(&module.spirv);
        let names = float_control_capabilities(&module.cache_key);
        assert!(module.cache_key.capabilities.contains(&"Shader".to_owned()));
        if has_float_controls(kernel.kind) {
            let mut declared: Vec<u32> = modes.iter().map(|(mode, _)| *mode).collect();
            declared.sort_unstable();
            let mut wanted = common::words::FLOAT_CONTROL_MODES.to_vec();
            wanted.sort_unstable();
            assert_eq!(declared, wanted, "{what}: the three modes");
            assert!(modes.iter().all(|(_, width)| width == &[32]), "{what}");
            assert_eq!(
                names, FLOAT_CONTROL_CAPABILITIES,
                "{what}: the key names them"
            );
            declaring += 1;
        } else {
            assert!(modes.is_empty(), "{what}: {modes:?}");
            assert!(names.is_empty(), "{what}: {names:?}");
        }
        for off in 0..3 {
            let mut values = [true; 3];
            values[off] = false;
            let module = with(kernel, &fixture_with(values));
            let modes = common::words::float_control_modes(&module.spirv);
            assert!(modes.is_empty(), "{what} with value {off} false: {modes:?}");
            let names = float_control_capabilities(&module.cache_key);
            assert!(names.is_empty(), "{what} with value {off} false: {names:?}");
        }
        let module = with(kernel, &fixture());
        assert!(common::words::float_control_modes(&module.spirv).is_empty());
    }
    assert!(declaring > 0);
    eprintln!(
        "spirv_val: {} modules, {declaring} declare RoundingModeRTE, DenormPreserve and \
         SignedZeroInfNanPreserve for width 32 on the all-true fixture copy, none on a copy with \
         one false",
        kernels.len()
    );
}

#[test]
fn the_cache_key_names_the_kernel_and_the_device() {
    let profile = fixture();
    let kernel = common::matmul_kernel(17, 33, 65);
    let key = emit_module(&kernel, &profile).unwrap().cache_key;
    assert_eq!(key.shapes, vec![vec![17, 33], vec![33, 65], vec![17, 65]]);
    assert_eq!(key.dtype, DType::F32);
    assert!(key.deterministic);
    assert!(!key.fast_math);
    assert_eq!(
        (key.vendor_id, key.device_id, key.driver_version),
        (profile.vendor_id, profile.device_id, profile.driver_version)
    );
    assert_eq!(key.ir_hash.0.len(), 64);
    let other = emit_module(&common::matmul_kernel(17, 33, 64), &profile)
        .unwrap()
        .cache_key;
    assert_ne!(key.ir_hash, other.ir_hash);
    let mut capabilities = key.capabilities.clone();
    capabilities.sort();
    capabilities.dedup();
    assert_eq!(capabilities.len(), key.capabilities.len(), "distinct");
}

#[test]
fn emit_declares_the_workgroup_size_of_the_kind_and_no_float_control_mode() {
    let kernel = common::kernel(&common::catalog()[0], 65);
    let words = emit(&kernel).expect("the kernel emits");
    let mut at = 5;
    let mut local_size = None;
    let mut modes = Vec::new();
    while at < words.len() {
        let length = (words[at] >> 16) as usize;
        // OpExecutionMode: entry point, mode, operands.
        if words[at] & 0xffff == 16 {
            modes.push(words[at + 2]);
            // ExecutionMode LocalSize is 17.
            if words[at + 2] == 17 {
                local_size = Some([words[at + 3], words[at + 4], words[at + 5]]);
            }
        }
        at += length;
    }
    assert_eq!(local_size, Some([64, 1, 1]));
    assert_eq!(modes, [17], "LocalSize is the only execution mode");
    assert_eq!(
        words[1], 0x0001_0500,
        "SPIR-V 1.5, the version of Vulkan 1.2"
    );
}

// story:m1-loss-sum-staged acceptance 2: the loss sum runs one workgroup of 64 that stages its
// terms in workgroup memory, sized by the shape and the device's `max_shared_memory_bytes`.

/// The `OpVariable`, `OpTypePointer`, `OpTypeArray`, `OpConstant` and `OpControlBarrier` opcodes,
/// and the `Workgroup` storage class.
const OP_VARIABLE: u32 = 59;
const OP_TYPE_POINTER: u32 = 32;
const OP_TYPE_ARRAY: u32 = 28;
const OP_CONSTANT: u32 = 43;
const OP_CONTROL_BARRIER: u32 = 224;
const STORAGE_WORKGROUP: u32 = 4;

/// The length of every `Workgroup` array variable of `module`, in declaration order.
fn workgroup_array_lengths(module: &[u32]) -> Vec<u32> {
    let all = common::words::instructions(module);
    let find = |op: u32, id: u32| {
        all.iter()
            .find(|(o, w)| *o == op && w[1] == id)
            .map(|(_, w)| *w)
            .unwrap_or_else(|| panic!("opcode {op} defines %{id}"))
    };
    all.iter()
        .filter(|(op, w)| *op == OP_VARIABLE && w[3] == STORAGE_WORKGROUP)
        .map(|(_, w)| {
            let pointer = find(OP_TYPE_POINTER, w[1]);
            assert_eq!(pointer[2], STORAGE_WORKGROUP);
            let array = find(OP_TYPE_ARRAY, pointer[3]);
            all.iter()
                .find(|(o, c)| *o == OP_CONSTANT && c[2] == array[3])
                .map(|(_, c)| c[3])
                .expect("the array length is a constant")
        })
        .collect()
}

fn local_size(module: &[u32]) -> Vec<u32> {
    common::words::execution_modes(module)
        .into_iter()
        .find(|(mode, _)| *mode == 17)
        .map(|(_, size)| size)
        .expect("the module declares LocalSize")
}

/// The `OpLoad`, `OpStore`, `OpAccessChain` and `OpInBoundsAccessChain` opcodes.
const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
const OP_ACCESS_CHAIN: u32 = 65;
const OP_IN_BOUNDS_ACCESS_CHAIN: u32 = 66;

/// Walks the module in layout order, in which the emitter places each staging loop before the
/// sum that reads it and each chunk after the one before, and returns the stores and loads of
/// workgroup memory as `'S'` and `'L'`, with `'B'` for each `OpControlBarrier`, plus every
/// switch between a store and a load, either way round, that has no barrier between them: a
/// read of a term before every invocation has staged it, or a store over a term before
/// invocation 0 has added it.
fn workgroup_accesses(module: &[u32]) -> (String, Vec<usize>) {
    let all = common::words::instructions(module);
    let mut shared: Vec<u32> = all
        .iter()
        .filter(|(op, w)| *op == OP_VARIABLE && w[3] == STORAGE_WORKGROUP)
        .map(|(_, w)| w[2])
        .collect();
    let mut trace = String::new();
    for (op, w) in &all {
        match *op {
            OP_ACCESS_CHAIN | OP_IN_BOUNDS_ACCESS_CHAIN if shared.contains(&w[3]) => {
                shared.push(w[2]);
            }
            OP_STORE if shared.contains(&w[1]) => trace.push('S'),
            OP_LOAD if shared.contains(&w[3]) => trace.push('L'),
            OP_CONTROL_BARRIER => trace.push('B'),
            _ => {}
        }
    }
    let mut unguarded = Vec::new();
    let mut last: Option<char> = None;
    for (at, access) in trace.chars().enumerate() {
        match (last, access) {
            (_, 'B') => last = None,
            (Some(before), now) if before != now => {
                unguarded.push(at);
                last = Some(now);
            }
            (_, now) => last = Some(now),
        }
    }
    (trace, unguarded)
}

#[test]
fn the_loss_sum_stages_its_terms_in_workgroup_memory_sized_by_the_shape_and_the_profile() {
    let mut total = 0;
    let mut hazards = Vec::new();
    for bytes in [49_152u32, 260, 256] {
        let mut profile = fixture_with([true; 3]);
        profile.max_shared_memory_bytes = bytes;
        for count in [1u64, 63, 64, 65, 1_000, 100_000] {
            let kernel = common::loss_sum_kernel(count);
            assert_eq!(kernel.workgroup_size, 64);
            let chunk = count.min(u64::from(bytes / 4)) as u32;
            for module in [
                emit_module(&kernel, &profile).expect("the loss sum emits"),
                emit_loss_sum_undivided(&kernel, &profile).expect("the undivided sum emits"),
            ] {
                let what = format!("{count} terms, {bytes} bytes");
                validate(&module.spirv).unwrap_or_else(|error| panic!("{what}: {error}"));
                assert_eq!(local_size(&module.spirv), [64, 1, 1], "{what}");
                assert_eq!(
                    workgroup_array_lengths(&module.spirv),
                    [chunk],
                    "{what}: one f32 array of min(count, bytes / 4)"
                );
                let (trace, unguarded) = workgroup_accesses(&module.spirv);
                let chunks = count.div_ceil(u64::from(chunk)) as usize;
                assert!(
                    trace.contains('S') && trace.contains('L'),
                    "{what}: stores and loads of workgroup memory, got {trace}"
                );
                if !unguarded.is_empty() {
                    hazards.push(format!(
                        "{what} ({chunks} chunks): unguarded at {unguarded:?} of {trace}"
                    ));
                }
                if (chunks > 1) != trace.contains("LBS") {
                    hazards.push(format!(
                        "{what} ({chunks} chunks): a later chunk stages after a barrier after a \
                         sum exactly when there is more than one chunk, in {trace}"
                    ));
                }
                let (missing, all) = common::words::uncontracted_arithmetic(&module.spirv);
                assert!(all > 0 && missing.is_empty(), "{what}: {missing:?}");
                total += 1;
            }
        }
    }
    assert!(
        hazards.is_empty(),
        "a barrier between staging and summing, and between a chunk's sum and the next chunk's \
         staging: {hazards:#?}"
    );
    let kernel = common::loss_sum_kernel(1_000);
    let mut small = fixture();
    small.max_shared_memory_bytes = 256;
    assert_ne!(
        emit_module(&kernel, &fixture()).unwrap().cache_key.ir_hash,
        emit_module(&kernel, &small).unwrap().cache_key.ir_hash,
        "the chunk is in the key"
    );
    let words = emit(&common::loss_sum_kernel(100_000)).expect("emits without a profile");
    validate(&words).expect("valid");
    assert_eq!(
        workgroup_array_lengths(&words),
        [4_096],
        "without a profile, the 16384 bytes every Vulkan device has"
    );
    small.max_shared_memory_bytes = 3;
    assert!(matches!(
        emit_module(&kernel, &small),
        Err(EmitError::Operands { .. })
    ));
    eprintln!(
        "spirv_val: {total} staged loss-sum modules pass spirv-val --target-env {TARGET_ENV}"
    );
}

// story:m1-f-forward-matmul-tiled: the matmul stages tiles of its operands in workgroup memory.

/// `trace` with each run of one access collapsed to one letter.
fn collapsed(trace: &str) -> String {
    let mut out = String::new();
    for c in trace.chars() {
        if !out.ends_with(c) || c == 'B' {
            out.push(c);
        }
    }
    out
}

/// The three epilogues a matmul kernel has.
const EPILOGUES: [Option<Epilogue>; 3] = [
    None,
    Some(Epilogue::Relu {
        writes_pre_activation: false,
    }),
    Some(Epilogue::Relu {
        writes_pre_activation: true,
    }),
];

#[test]
fn a_matmul_module_runs_its_tile_with_one_barrier_per_tile_of_k() {
    let mut total = 0;
    for (m, k, n) in common::MATMUL_SHAPES {
        let tile = MatMulTile::of(m, k, n);
        let what = format!("[{m}, {k}] x [{k}, {n}], {tile:?}");
        assert!(tile.shared_bytes() <= MIN_SHARED_MEMORY_BYTES, "{what}");
        assert!(tile.workgroup_size.is_multiple_of(tile.cols), "{what}");
        assert!(tile.cols.is_power_of_two() && u64::from(tile.cols) <= n.next_power_of_two());
        assert!(u64::from(tile.depth) <= k, "{what}");
        let tiles = k.div_ceil(u64::from(tile.depth)) as usize;
        assert_eq!(tile.buffers, if tiles > 1 { 2 } else { 1 }, "{what}");
        assert!(
            tile.workgroups(m, n) * u64::from(tile.rows_per_workgroup()) * u64::from(tile.cols)
                >= m * n,
            "{what}: the workgroups cover the output"
        );
        for epilogue in EPILOGUES {
            let kernel = common::matmul_kernel_with(m, k, n, epilogue);
            for profile in [fixture(), fixture_with([true; 3])] {
                let module = emit_module(&kernel, &profile).expect("the matmul emits");
                validate(&module.spirv).unwrap_or_else(|error| panic!("{what}: {error}"));
                assert_eq!(
                    local_size(&module.spirv),
                    [tile.workgroup_size, 1, 1],
                    "{what}"
                );
                assert_eq!(
                    workgroup_array_lengths(&module.spirv),
                    [tile.shared_values()],
                    "{what}: one f32 array of every buffer"
                );
                let (looped, loops) = (tile_loop(k, &tile), loop_merges(&module.spirv));
                assert_eq!(
                    loops,
                    usize::from(looped > 0),
                    "{what}: one tile loop or none"
                );
                let emitted = tiles - looped.saturating_sub(TILE_LOOP_BODY);
                let (trace, _) = workgroup_accesses(&module.spirv);
                let wanted = format!("SB{}L", "LSB".repeat(emitted - 1));
                assert_eq!(
                    collapsed(&trace),
                    wanted,
                    "{what}: stage the first tile, a barrier, then per tile its products and \
                     the next tile's staging into the other buffer, a barrier, with the \
                     {looped} tiles of the loop emitted as one body of {TILE_LOOP_BODY}"
                );
                let emitted_k =
                    k - (looped.saturating_sub(TILE_LOOP_BODY) as u64) * u64::from(tile.depth);
                let (missing, all) = common::words::uncontracted_arithmetic(&module.spirv);
                let (missing_fma, fma) = common::words::uncontracted_fma(&module.spirv);
                let rows = u64::from(tile.rows);
                assert!(
                    all as u64 == rows
                        && fma as u64 == rows * (emitted_k - 1)
                        && missing.is_empty()
                        && missing_fma.is_empty(),
                    "{what}: {all} arithmetic and {fma} Fma, one chain per row, no NoContraction \
                     on {missing:?} {missing_fma:?}"
                );
                total += 1;
            }
        }
    }
    eprintln!("spirv_val: {total} tiled matmul modules pass spirv-val --target-env {TARGET_ENV}");
}

/// The tiles of the matmul's tile loop body.
const TILE_LOOP_BODY: usize = 4;

/// The tiles a `k`-deep matmul with `tile` runs in its loop: from tile 1 up, the tiles that
/// are full and read a full tile `ahead` past them, rounded down to whole loop bodies.
fn tile_loop(k: u64, tile: &MatMulTile) -> usize {
    let full = (k / u64::from(tile.depth)) as usize;
    let candidates = full.saturating_sub(tile.ahead as usize + 1);
    candidates - candidates % TILE_LOOP_BODY
}

/// The `OpLoopMerge` instructions of `module`.
fn loop_merges(module: &[u32]) -> usize {
    common::words::instructions(module)
        .iter()
        .filter(|(op, _)| *op == OP_LOOP_MERGE)
        .count()
}

const OP_LOOP_MERGE: u32 = 246;

// review-result:m1-f-forward-adversary-1, blockers 1 and 2: the module of a matmul is bounded
// whatever its `k`. review-result:m1-f-forward-adversary-2: the slack over the 16-tile bound is
// the `k`-dependent constants of `common::matmul_k_constants`, not a fixed 4; `[208, k] x [k, 2]`
// holds 5 of them apart.

#[test]
fn a_matmul_module_is_no_larger_at_any_k_than_at_a_k_of_sixteen_tiles() {
    for (m, n) in [(100, 64), (4, 1), (9, 17), (784, 64), (5, 3), (208, 2)] {
        let depth = u64::from(MatMulTile::of(m, 1 << 20, n).depth);
        let words = |k: u64| -> (usize, u32) {
            let module = emit(&common::matmul_kernel(m, k, n)).expect("the matmul emits");
            (module.len(), module[3])
        };
        // Every k up to 16 tiles: its straight tiles after the loop and the length of its
        // last tile take every value a larger k gives them.
        let mut small = (0, 0);
        for at in 1..=16 * depth {
            let (len, bound) = words(at);
            small = (small.0.max(len), small.1.max(bound));
        }
        let limit = u64::from(u32::MAX) / n.max(m);
        for base in [10_000, 100_000, 600_000, 1_000_000, 4_000_000] {
            for at in [base, base + 1, base + depth - 1, base + 3 * depth + 5] {
                if at > limit {
                    continue;
                }
                let (len, bound) = words(at);
                let what = format!("[{m}, {at}] x [{at}, {n}]: {len} words, id bound {bound}");
                let spare = common::matmul_k_constants(at, &MatMulTile::of(m, at, n));
                assert!(
                    len <= small.0 + common::CONSTANT_WORDS * spare
                        && bound <= small.1 + spare as u32,
                    "{what}, above {small:?}, the largest at k up to {}, by more than its {spare} \
                     k-dependent constants",
                    16 * depth
                );
                let module = emit(&common::matmul_kernel(m, at, n)).expect("emits");
                validate(&module).unwrap_or_else(|error| panic!("{what}: {error}"));
            }
        }
    }
}

#[test]
fn a_matmul_tile_beyond_the_profiles_workgroup_memory_is_refused_before_any_pipeline() {
    for (m, k, n) in common::MATMUL_SHAPES {
        let tile = MatMulTile::of(m, k, n);
        let kernel = common::matmul_kernel(m, k, n);
        let mut profile = fixture();
        profile.max_shared_memory_bytes = tile.shared_bytes();
        emit_module(&kernel, &profile).expect("a tile that fits emits");
        profile.max_shared_memory_bytes = tile.shared_bytes() - 4;
        match emit_module(&kernel, &profile) {
            Err(EmitError::SharedMemory {
                needs, provides, ..
            }) => {
                assert_eq!(
                    (needs, provides),
                    (tile.shared_bytes(), tile.shared_bytes() - 4)
                );
            }
            other => panic!("[{m}, {k}] x [{k}, {n}]: {:?}", other.map(|m| m.cache_key)),
        }
    }
    let (name, artifact) = acceptance_artifacts()
        .into_iter()
        .find(|(name, _)| name.contains("mnist"))
        .expect("the MNIST artifact");
    let mut small = fixture();
    small.max_shared_memory_bytes = 1_024;
    assert!(
        matches!(
            emit_kernels(&artifact, &small),
            Err(EmitError::SharedMemory { .. })
        ),
        "{name}: emit_kernels, which runs before any pipeline is created, refuses the artifact"
    );
}

#[test]
fn a_matmul_kernel_whose_workgroups_are_not_its_tiles_is_refused() {
    let mut kernel = common::matmul_kernel(17, 33, 65);
    kernel.workgroup_size = 64;
    assert!(matches!(emit(&kernel), Err(EmitError::Operands { .. })));
    let mut kernel = common::matmul_kernel(17, 33, 65);
    kernel.workgroups[0] -= 1;
    assert!(matches!(emit(&kernel), Err(EmitError::Operands { .. })));
}
