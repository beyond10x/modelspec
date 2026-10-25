//! `story:matmul-block-kernel-throughput`, acceptance `profile-wired` (DESIGN.md § 5.2, § 7.1):
//! an artifact compiled for a device profile ([`compile_artifact_for`]) is scheduled within the
//! profile's workgroup invocations and workgroup memory, records them as `matmul_limits`, and
//! the reader and the plan verifier refuse a device that reports less, naming both values; an
//! artifact compiled without a profile is the one [`compile_artifact`] gives, its manifest
//! without `matmul_limits`.

use std::path::{Path, PathBuf};

use modelspec_core::artifact::{
    self, Backend, CompiledArtifact, E_ARTIFACT_003, E_ARTIFACT_005, KernelCacheKey,
    KernelModuleFile,
};
use modelspec_core::compile::{compile_artifact, compile_artifact_for};
use modelspec_core::device::DeviceProfile;
use modelspec_core::diag::Diagnostic;
use modelspec_core::ir::kernel::MatMulLimits;
use modelspec_core::ir::tensor::DType;
use modelspec_core::ir::verify::{
    PlanVerifyError, verify_device_limits, verify_plan, verify_plan_within,
};

const RTX3090: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../modelspec-runtime/tests/fixtures/rtx3090.device.json"
));

fn rtx3090() -> DeviceProfile {
    DeviceProfile::from_json(RTX3090).expect("the fixture is a valid profile")
}

/// The RTX 3090 profile with `invocations` and `shared_bytes` in place of its limits.
fn limited(invocations: u32, shared_bytes: u32) -> DeviceProfile {
    let mut profile = rtx3090();
    profile.max_workgroup_invocations = invocations;
    profile.max_shared_memory_bytes = shared_bytes;
    profile
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("artifact_matmul_limits")
        .join(name);
    if dir.is_dir() {
        std::fs::remove_dir_all(&dir).expect("the old test dir is removed");
    }
    dir
}

/// The source `benches/mlp_sweep.rs` compiles at batch 8192 and hidden width 4096.
fn sweep() -> String {
    "model SweepMlp {
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
"
    .to_owned()
}

fn compiled(backend: Backend, profile: Option<&DeviceProfile>) -> CompiledArtifact {
    compile_artifact_for("mlp_sweep.mspec", &sweep(), backend, profile)
        .unwrap_or_else(|d| panic!("{d:?}"))
}

/// One module per kernel of `compiled`: a SPIR-V header and nothing else.
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

fn detail<'d>(diagnostic: &'d Diagnostic, label: &str) -> Option<&'d str> {
    diagnostic
        .details
        .iter()
        .find(|d| d.label.as_deref() == Some(label))
        .map(|d| d.text.as_str())
}

/// Compiled for the RTX 3090 profile, the artifact records the profile's limits and its plan
/// takes the block of 256 invocations; without a profile it is `compile_artifact`'s artifact,
/// no kernel past 128 invocations and no limits.
#[test]
fn an_artifact_compiled_for_a_profile_records_its_limits_and_takes_the_block() {
    let profile = rtx3090();
    let limits = MatMulLimits::of(&profile);
    assert_eq!(
        (limits.invocations, limits.shared_bytes),
        (1024, 49_152),
        "the RTX 3090 fixture"
    );
    let scheduled = compiled(Backend::Vulkan, Some(&profile));
    assert_eq!(scheduled.matmul_limits, Some(limits));
    assert!(
        scheduled
            .execution
            .kernels
            .iter()
            .any(|kernel| kernel.workgroup_size == 256),
        "the forward matmul takes the block"
    );
    verify_plan_within(&scheduled.execution, scheduled.matmul_limits).expect("within its limits");
    assert!(matches!(
        verify_plan(&scheduled.execution),
        Err(PlanVerifyError::WorkgroupTooLarge { size: 256, .. })
    ));

    let every = compiled(Backend::Vulkan, None);
    let shipped = compile_artifact("mlp_sweep.mspec", &sweep(), Backend::Vulkan).unwrap();
    assert_eq!(
        every, shipped,
        "no profile: the artifact of compile_artifact"
    );
    assert_eq!(every.matmul_limits, None);
    assert!(
        every
            .execution
            .kernels
            .iter()
            .all(|kernel| kernel.workgroup_size <= 128)
    );
}

/// The verifier accepts the profile the plan was scheduled for and any that reports more, and
/// refuses one that reports fewer invocations or less workgroup memory with
/// `DeviceBelowLimits`, whose message names the recorded and the reported value of both.
#[test]
fn the_verifier_refuses_a_device_below_the_recorded_limits_naming_both_values() {
    let scheduled = compiled(Backend::Vulkan, Some(&rtx3090()));
    let recorded = scheduled.matmul_limits;
    verify_device_limits(recorded, &rtx3090()).expect("the profile it was scheduled for");
    verify_device_limits(recorded, &limited(2048, 65_536)).expect("a larger device");
    verify_device_limits(None, &limited(128, 16_384)).expect("no limits: every device");
    for (invocations, shared) in [(512, 49_152), (1024, 49_148), (128, 16_384)] {
        let error = verify_device_limits(recorded, &limited(invocations, shared))
            .expect_err("a device below the limits");
        let PlanVerifyError::DeviceBelowLimits {
            recorded: was,
            reported,
        } = error
        else {
            panic!("{error:?}");
        };
        assert_eq!(Some(was), recorded);
        assert_eq!(
            (reported.invocations, reported.shared_bytes),
            (invocations, shared)
        );
        let message = error.to_string();
        for value in [
            "1024",
            "49152",
            &invocations.to_string(),
            &shared.to_string(),
        ] {
            assert!(message.contains(value), "{message} names {value}");
        }
    }
}

/// Written and read back, the limits are in the manifest, in the artifact hash, and in what
/// `read` gives back; `read_for_device` refuses a device below them with E-ARTIFACT-005 naming
/// both values of both limits and accepts the profile. An artifact without limits writes no
/// `matmul_limits` key.
#[test]
fn the_reader_records_the_limits_and_refuses_a_device_below_them() {
    let profile = rtx3090();
    let scheduled = compiled(Backend::Vulkan, Some(&profile));
    let dir = scratch("scheduled");
    let hash = artifact::write(&dir, &scheduled, &modules(&scheduled)).expect("written");
    let manifest = std::fs::read_to_string(dir.join(artifact::MANIFEST_FILE)).unwrap();
    let json: serde_json::Value = serde_json::from_str(&manifest).unwrap();
    assert_eq!(
        json["matmul_limits"],
        serde_json::json!({"invocations": 1024, "shared_bytes": 49152}),
        "{manifest}"
    );
    let read = artifact::read(&dir).expect("a valid artifact");
    assert_eq!(read.artifact_hash, hash);
    assert_eq!(read.artifact.matmul_limits, scheduled.matmul_limits);
    artifact::read_for_device(&dir, &profile).expect("the profile it was scheduled for");

    let refused = artifact::read_for_device(&dir, &limited(512, 32_768))
        .expect_err("a device below the limits");
    assert_eq!(refused.code, E_ARTIFACT_005, "{refused}");
    for (label, value) in [
        ("recorded_invocations", "1024"),
        ("reported_invocations", "512"),
        ("recorded_shared_bytes", "49152"),
        ("reported_shared_bytes", "32768"),
    ] {
        assert_eq!(detail(&refused, label), Some(value), "{refused}");
    }
    assert!(artifact::check_device(&scheduled, &limited(1024, 49_151)).is_err());

    let every = compiled(Backend::Vulkan, None);
    let plain = scratch("every");
    let plain_hash = artifact::write(&plain, &every, &modules(&every)).expect("written");
    let manifest = std::fs::read_to_string(plain.join(artifact::MANIFEST_FILE)).unwrap();
    assert!(!manifest.contains("matmul_limits"), "{manifest}");
    assert_ne!(plain_hash, hash);
    artifact::read_for_device(&plain, &limited(128, 16_384)).expect("no limits: every device");
}

/// A manifest whose recorded invocations are lowered below a kernel's workgroup and resealed is
/// refused by the plan verifier as the artifact is read (E-ARTIFACT-003), and so is one whose
/// limits are removed.
#[test]
fn a_manifest_whose_limits_do_not_cover_its_kernels_is_refused() {
    let scheduled = compiled(Backend::Vulkan, Some(&rtx3090()));
    for (name, edit) in [
        (
            "lowered",
            serde_json::json!({"invocations": 128, "shared_bytes": 49152}),
        ),
        ("removed", serde_json::Value::Null),
    ] {
        let dir = scratch(name);
        artifact::write(&dir, &scheduled, &modules(&scheduled)).expect("written");
        let path = dir.join(artifact::MANIFEST_FILE);
        let mut json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let object = json.as_object_mut().unwrap();
        if edit.is_null() {
            object.remove("matmul_limits");
        } else {
            object.insert("matmul_limits".to_owned(), edit);
        }
        std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap() + "\n").unwrap();
        artifact::seal(&dir).expect("resealed");
        let refused = artifact::read(&dir).expect_err("a kernel past the recorded invocations");
        assert_eq!(refused.code, E_ARTIFACT_003, "{name}: {refused}");
        assert!(refused.to_string().contains("256"), "{name}: {refused}");
    }
}
