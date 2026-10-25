//! `story:matmul-block-kernel-throughput`, acceptance `profile-wired` (DESIGN.md § 5.2, § 7.1):
//! an artifact compiled for a device profile ([`compile_artifact_for`]) is scheduled within the
//! profile's workgroup invocations and workgroup memory, records them as `matmul_limits`, and
//! the reader and the plan verifier refuse a device that reports less, naming both values; an
//! artifact compiled without a profile is the one [`compile_artifact`] gives, its manifest
//! without `matmul_limits`.

use std::path::{Path, PathBuf};

use modelspec_core::artifact::{
    self, Backend, CompiledArtifact, E_ARTIFACT_002, E_ARTIFACT_003, E_ARTIFACT_005,
    KernelCacheKey, KernelModuleFile,
};
use modelspec_core::compile::{compile_artifact, compile_artifact_for};
use modelspec_core::device::DeviceProfile;
use modelspec_core::diag::Diagnostic;
use modelspec_core::ir::kernel::MatMulLimits;
use modelspec_core::ir::tensor::DType;
use modelspec_core::ir::verify::{
    PlanVerifyError, verify_device_limits, verify_plan, verify_plan_within,
};
use serde_json::Value;

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

/// Writes `compiled`, replaces its manifest's `matmul_limits` with `limits` and reseals it when
/// `reseal`.
fn rewritten(name: &str, compiled: &CompiledArtifact, limits: Value, reseal: bool) -> PathBuf {
    let dir = scratch(name);
    artifact::write(&dir, compiled, &modules(compiled)).expect("written");
    let path = dir.join(artifact::MANIFEST_FILE);
    let mut json: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    json.as_object_mut()
        .unwrap()
        .insert("matmul_limits".to_owned(), limits);
    std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap() + "\n").unwrap();
    if reseal {
        artifact::seal(&dir).expect("resealed");
    }
    dir
}

/// ADV-BLOCKPERF-1: a resealed manifest whose `matmul_limits` break an invariant of
/// `spec/domains/artifact.yaml` `MatMulLimits` (`invocations >= 128`, `shared_bytes >= 16384`)
/// is refused with E-ARTIFACT-003 naming the field and its bound, as the reader refuses the
/// other invariants; limits at exactly the bounds are read.
#[test]
fn the_reader_refuses_limits_below_the_vulkan_minimum_naming_the_field_and_bound() {
    let minimum = compiled(Backend::Vulkan, Some(&limited(128, 16_384)));
    assert_eq!(
        minimum.matmul_limits,
        Some(MatMulLimits {
            invocations: 128,
            shared_bytes: 16_384
        })
    );
    let dir = rewritten(
        "at-the-bounds",
        &minimum,
        serde_json::json!({"invocations": 128, "shared_bytes": 16_384}),
        true,
    );
    artifact::read(&dir).expect("limits at the bounds are read");
    for (name, limits, rule) in [
        (
            "invocations-127",
            serde_json::json!({"invocations": 127, "shared_bytes": 16_384}),
            "matmul_limits.invocations >= 128",
        ),
        (
            "invocations-zero",
            serde_json::json!({"invocations": 0, "shared_bytes": 49_152}),
            "matmul_limits.invocations >= 128",
        ),
        (
            "shared-16383",
            serde_json::json!({"invocations": 1024, "shared_bytes": 16_383}),
            "matmul_limits.shared_bytes >= 16384",
        ),
        (
            "shared-zero",
            serde_json::json!({"invocations": 1024, "shared_bytes": 0}),
            "matmul_limits.shared_bytes >= 16384",
        ),
    ] {
        let dir = rewritten(name, &minimum, limits, true);
        let refused = artifact::read(&dir).expect_err("limits below the invariants");
        assert_eq!(refused.code, E_ARTIFACT_003, "{name}: {refused}");
        assert!(refused.to_string().contains(rule), "{name}: {refused}");
    }
}

/// ADV-BLOCKPERF-2 and ADV-BLOCKPERF-3: a profile whose limits are below the Vulkan minimum the
/// smallest tile needs (`MatMulLimits` `invocations >= 128`, `shared_bytes >= 16384`) is refused
/// by `compile_artifact_for` with one E-ARTIFACT-005, the device cannot run the plan, naming the
/// field, its bound and the reported value; never E-INTERNAL-002, and no artifact records such
/// limits. A profile at exactly the bounds compiles and records them.
#[test]
fn a_profile_below_the_vulkan_minimum_is_refused_with_e_artifact_005_naming_the_field() {
    let mut narrow = rtx3090();
    narrow.max_workgroup_size.x = 64;
    for (name, profile, field, bound, reported) in [
        (
            "invocations-64",
            limited(64, 49_152),
            "invocations",
            "128",
            "64",
        ),
        (
            "invocations-127",
            limited(127, 16_384),
            "invocations",
            "128",
            "127",
        ),
        ("size-x-64", narrow, "invocations", "128", "64"),
        (
            "shared-8192",
            limited(1024, 8192),
            "shared_bytes",
            "16384",
            "8192",
        ),
        (
            "shared-16383",
            limited(128, 16_383),
            "shared_bytes",
            "16384",
            "16383",
        ),
    ] {
        let refused =
            compile_artifact_for("mlp_sweep.mspec", &sweep(), Backend::Vulkan, Some(&profile))
                .expect_err("a profile below the Vulkan minimum");
        let [diagnostic] = &refused[..] else {
            panic!("{name}: one diagnostic: {refused:?}");
        };
        assert_eq!(diagnostic.code, E_ARTIFACT_005, "{name}: {diagnostic}");
        assert_eq!(
            detail(diagnostic, "field"),
            Some(field),
            "{name}: {diagnostic}"
        );
        assert_eq!(
            detail(diagnostic, "bound"),
            Some(bound),
            "{name}: {diagnostic}"
        );
        assert_eq!(
            detail(diagnostic, "reported"),
            Some(reported),
            "{name}: {diagnostic}"
        );
        assert!(
            !diagnostic.to_string().contains("compiler bug"),
            "{name}: {diagnostic}"
        );
    }
    let minimum = compiled(Backend::Vulkan, Some(&limited(128, 16_384)));
    verify_plan_within(&minimum.execution, minimum.matmul_limits).expect("within the minimum");
}

/// The artifact hash covers `matmul_limits` (mutant M4 of the adversary pass leaves them out of
/// it): two artifacts of one plan and one set of modules that differ only in the recorded
/// limits have different hashes, and a manifest whose limits are changed and not resealed is
/// refused with E-ARTIFACT-002.
#[test]
fn the_artifact_hash_covers_matmul_limits() {
    let scheduled = compiled(Backend::Vulkan, Some(&rtx3090()));
    let mut wider = scheduled.clone();
    wider.matmul_limits = Some(MatMulLimits {
        invocations: 1024,
        shared_bytes: 49_153,
    });
    let one =
        artifact::write(&scratch("hash-one"), &scheduled, &modules(&scheduled)).expect("written");
    let other = artifact::write(&scratch("hash-other"), &wider, &modules(&wider)).expect("written");
    assert_ne!(one, other, "the limits are in the artifact hash");

    let dir = rewritten(
        "hash-edited",
        &scheduled,
        serde_json::json!({"invocations": 1024, "shared_bytes": 49_153}),
        false,
    );
    let refused = artifact::read(&dir).expect_err("changed after it was sealed");
    assert_eq!(refused.code, E_ARTIFACT_002, "{refused}");
}

/// The bounds `MatMulLimits::checked` enforces, which the compile and the reader share, are
/// exactly the invariants `spec/domains/artifact.yaml` gives `MatMulLimits`: each is accepted at
/// its bound and refused one below it.
#[test]
fn the_checked_bounds_are_the_specification_invariants_of_matmul_limits() {
    let spec = include_str!("../../../spec/domains/artifact.yaml");
    let start = spec
        .find("- name: modelspec.artifact.MatMulLimits")
        .expect("the specification names MatMulLimits");
    let rest = &spec[start..];
    let block = &rest[..rest[1..]
        .find("\n  - name: ")
        .map_or(rest.len(), |end| end + 1)];
    let invariants: Vec<&str> = block
        .split("invariants:")
        .nth(1)
        .expect("MatMulLimits has invariants")
        .lines()
        .filter_map(|line| line.trim().strip_prefix("- "))
        .collect();
    assert_eq!(
        invariants,
        [
            format!("invocations >= {}", MatMulLimits::MIN_INVOCATIONS),
            format!("shared_bytes >= {}", MatMulLimits::MIN_SHARED_BYTES),
        ],
        "{block}"
    );
    let bounds = MatMulLimits {
        invocations: MatMulLimits::MIN_INVOCATIONS,
        shared_bytes: MatMulLimits::MIN_SHARED_BYTES,
    };
    assert_eq!(bounds.checked(), Ok(bounds));
    for below in [
        MatMulLimits {
            invocations: bounds.invocations - 1,
            ..bounds
        },
        MatMulLimits {
            shared_bytes: bounds.shared_bytes - 1,
            ..bounds
        },
    ] {
        assert!(below.checked().is_err(), "{below:?}");
    }
}
