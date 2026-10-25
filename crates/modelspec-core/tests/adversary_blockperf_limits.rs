//! Adversary cases for story:matmul-block-kernel-throughput, the scheduling contract of
//! `matmul_limits` (DESIGN.md § 5.2, § 7.1; `spec/domains/artifact.yaml`).
//!
//! 1. The artifact hash covers `matmul_limits`: a manifest whose recorded limits are changed and
//!    not resealed is refused with E-ARTIFACT-002.
//! 2. The reader enforces the invariants the specification gives `MatMulLimits`
//!    (`invocations >= 128`, `shared_bytes >= 16384`, also the `minimum` of the generated JSON
//!    Schema): a resealed manifest recording fewer bytes or invocations is refused with
//!    E-ARTIFACT-003, the code of "the artifact breaks an invariant of
//!    `spec/domains/artifact.yaml`".
//! 3. The block is never taken below `k = 64`, so the `k` of 1 to 5 the vector reads might trip
//!    on never reach them: the vector runs are along rows and columns, not along `k`.

use std::path::{Path, PathBuf};

use modelspec_core::artifact::{
    self, Backend, CompiledArtifact, E_ARTIFACT_002, E_ARTIFACT_003, KernelCacheKey,
    KernelModuleFile,
};
use modelspec_core::compile::compile_artifact_for;
use modelspec_core::device::DeviceProfile;
use modelspec_core::ir::kernel::MatMulTile;
use modelspec_core::ir::tensor::DType;

const RTX3090: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../modelspec-runtime/tests/fixtures/rtx3090.device.json"
));

fn rtx3090() -> DeviceProfile {
    DeviceProfile::from_json(RTX3090).expect("the fixture is a valid profile")
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary_blockperf_limits")
        .join(name);
    if dir.is_dir() {
        std::fs::remove_dir_all(&dir).expect("the old test dir is removed");
    }
    dir
}

const SWEEP: &str = "model SweepMlp {
    param w1: Tensor<f32, [784, 4096]> = xavier_uniform();
    param w2: Tensor<f32, [4096, 10]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8192, 784]>)
        -> Tensor<f32, [8192, 10]> {
        return matmul(relu(matmul(x, w1)), w2);
    }
}

train Fit {
    model = SweepMlp;
    input x: Tensor<f32, [8192, 784]>;
    target y: Tensor<f32, [8192, 10]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.001);
    steps = 220;
    seed = 42;
    device = vulkan(0);
}
";

fn scheduled() -> CompiledArtifact {
    compile_artifact_for("mlp_sweep.mspec", SWEEP, Backend::Vulkan, Some(&rtx3090()))
        .unwrap_or_else(|d| panic!("{d:?}"))
}

/// One module per kernel: a SPIR-V header and nothing else.
fn modules(compiled: &CompiledArtifact) -> Vec<KernelModuleFile> {
    let header: Vec<u8> = [0x0723_0203u32, 0x0001_0500, 0, 1, 0]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    compiled
        .execution
        .kernels
        .iter()
        .map(|kernel| KernelModuleFile {
            kernel_module_id: kernel.id,
            entry_point: "main".to_owned(),
            cache_key: KernelCacheKey {
                ir_hash: compiled.ir_hash.0.clone(),
                shapes: Vec::new(),
                dtype: DType::F32,
                fast_math: false,
                deterministic: true,
                vendor_id: 0,
                device_id: 0,
                driver_version: 0,
                capabilities: vec!["Shader".to_owned()],
            },
            spirv: header.clone(),
        })
        .collect()
}

/// Writes the scheduled artifact to `name`, replaces its manifest's `matmul_limits` with
/// `limits`, and reseals it when `reseal`.
fn edited(name: &str, limits: serde_json::Value, reseal: bool) -> PathBuf {
    let compiled = scheduled();
    let dir = scratch(name);
    artifact::write(&dir, &compiled, &modules(&compiled)).expect("written");
    let path = dir.join(artifact::MANIFEST_FILE);
    let mut json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    json.as_object_mut()
        .unwrap()
        .insert("matmul_limits".to_owned(), limits);
    std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap() + "\n").unwrap();
    if reseal {
        artifact::seal(&dir).expect("resealed");
    }
    dir
}

#[test]
fn a_changed_matmul_limits_without_reseal_is_refused_with_the_hash_code() {
    for (name, limits) in [
        (
            "shared-raised",
            serde_json::json!({"invocations": 1024, "shared_bytes": 49153}),
        ),
        (
            "invocations-lowered",
            serde_json::json!({"invocations": 256, "shared_bytes": 49152}),
        ),
    ] {
        let dir = edited(name, limits, false);
        let refused = artifact::read(&dir).expect_err("the hash covers matmul_limits");
        assert_eq!(refused.code, E_ARTIFACT_002, "{name}: {refused}");
    }
}

/// ADV-BLOCKPERF-1: the reader enforces the MatMulLimits invariants of
/// spec/domains/artifact.yaml.
#[test]
fn a_manifest_breaking_the_matmul_limits_invariants_is_refused() {
    let mut accepted = Vec::new();
    for (name, limits) in [
        (
            "shared-below-16k",
            serde_json::json!({"invocations": 1024, "shared_bytes": 8192}),
        ),
        (
            "shared-zero",
            serde_json::json!({"invocations": 1024, "shared_bytes": 0}),
        ),
    ] {
        let dir = edited(name, limits.clone(), true);
        match artifact::read(&dir) {
            Ok(read) => accepted.push(format!(
                "{name}: accepted, matmul_limits {:?}",
                read.artifact.matmul_limits
            )),
            Err(refused) if refused.code == E_ARTIFACT_003 => {}
            Err(refused) => accepted.push(format!("{name}: refused with {refused}")),
        }
    }
    assert!(
        accepted.is_empty(),
        "spec/domains/artifact.yaml MatMulLimits: shared_bytes >= 16384: {accepted:#?}"
    );
}

/// The `mlp_sweep` source at batch `b` and hidden width `h`.
fn sweep_at(b: u64, h: u64) -> String {
    SWEEP
        .replace("8192", &b.to_string())
        .replace("4096", &h.to_string())
}

/// DESIGN.md § 3.1 "Tiles under the RTX 3090 profile", read as the specification: in the
/// `mlp_sweep` plans at b8192 h4096, b8192 h1024 and b1000 h1024 the block takes the forward
/// matmul `[8192, 784] x [784, H]` at batch 8192 and the weight gradient
/// `[784, 8192]ᵀ x [8192, 4096]`, and no other matmul.
#[test]
fn the_block_takes_exactly_the_matmuls_design_names_in_the_sweep_plans() {
    let mut found = Vec::new();
    for (b, h) in [(8192, 4096), (8192, 1024), (1000, 1024)] {
        let compiled = compile_artifact_for(
            "mlp_sweep.mspec",
            &sweep_at(b, h),
            Backend::Vulkan,
            Some(&rtx3090()),
        )
        .unwrap_or_else(|d| panic!("{d:?}"));
        for kernel in &compiled.execution.kernels {
            if kernel.workgroup_size == 256 {
                let (m, k, n) = modelspec_core::ir::kernel::Kernel::matmul_shape(&kernel.operands)
                    .expect("a block is a matmul");
                found.push((b, h, m, k, n, kernel.transposed));
            }
        }
    }
    assert_eq!(
        found,
        vec![
            (8192, 4096, 8192, 784, 4096, [false, false]),
            (8192, 4096, 784, 8192, 4096, [true, false]),
            (8192, 1024, 8192, 784, 1024, [false, false]),
        ],
        "DESIGN.md § 3.1"
    );
}

#[test]
fn the_block_and_its_vector_reads_are_never_taken_below_k_64() {
    let mut taken = Vec::new();
    for k in 1..64u64 {
        for (m, n) in [(8192, 4096), (16_385, 129), (127, 16_385), (65_536, 65_536)] {
            for limit in [256, 1024] {
                let tile = MatMulTile::of_invocations(m, k, n, limit);
                if tile.vector_reads() || tile.workgroup_size > 128 {
                    taken.push(format!("[{m}, {k}] x [{k}, {n}] at {limit}: {tile}"));
                }
            }
        }
    }
    assert!(taken.is_empty(), "{taken:#?}");
    let tile = MatMulTile::of_invocations(8192, 64, 4096, 1024);
    assert!(tile.vector_reads(), "k = 64 takes the block: {tile}");
}
