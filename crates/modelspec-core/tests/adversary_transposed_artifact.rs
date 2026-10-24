//! Adversary cases for story:matmul-reads-transposed-operand, the artifact entry's `transposed`
//! (`WireKernel::transposed`, `#[serde(default)]`, absent when `[false, false]`).
//!
//! The fixture `tests/fixtures/adversary_transposed_base_linear_regression` is the artifact
//! directory `modelspec compile --backend vulkan --device-profile
//! crates/modelspec-runtime/tests/fixtures/rtx3090.device.json examples/linear_regression.mspec`
//! wrote at the unit's base commit `ecabada`, before the flag existed: seven kernels, one of
//! them a `transpose`, and no `transposed` key anywhere.

use std::path::{Path, PathBuf};

use modelspec_core::artifact::{self, LoadedArtifact};
use modelspec_core::ir::kernel::{Kernel, KernelKind};
use modelspec_core::ir::verify::verify_plan;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/adversary_transposed_base_linear_regression")
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary_transposed_artifact")
        .join(name);
    if dir.is_dir() {
        std::fs::remove_dir_all(&dir).expect("the old test dir is removed");
    }
    dir
}

/// The artifact hash `compile` printed for the fixture at the base commit.
const BASE_HASH: &str = "23cb168aea426941cbb843cd379283a76510f4cc29490a98381350fed3d26ca8";

fn base() -> LoadedArtifact {
    artifact::read(&fixture()).unwrap_or_else(|d| panic!("the base artifact reads: {d}"))
}

/// An artifact written before the change reads, with its hash, every kernel reading no operand
/// transposed, and its stored transpose still a kernel the plan dispatches.
#[test]
fn an_artifact_written_before_the_flag_existed_reads_with_no_operand_transposed() {
    let json = std::fs::read_to_string(fixture().join("artifact.json")).expect("reads");
    assert!(
        !json.contains("transposed"),
        "the fixture predates the flag"
    );
    let loaded = base();
    assert_eq!(loaded.artifact_hash.to_string(), BASE_HASH);
    let kernels = &loaded.artifact.execution.kernels;
    assert_eq!(kernels.len(), 7);
    assert!(kernels.iter().all(|k| k.transposed == [false, false]));
    assert_eq!(
        kernels
            .iter()
            .filter(|k| k.kind == KernelKind::Transpose)
            .count(),
        1
    );
    verify_plan(&loaded.artifact.execution).expect("the base plan verifies");
}

/// `Kernel::transposed` is "`[false, false]` on every other kind" than a matmul. An entry that
/// flags an operand of an elementwise kernel, with its module id recomputed from the signature
/// and the artifact resealed, is refused by the reader or by the plan verifier; otherwise it is
/// run, and the emitter, which reads the flag only on a matmul, ignores it.
#[test]
#[ignore = "ADV-T1: a non-matmul kernel flagged transposed is read and verified, not refused"]
fn a_non_matmul_kernel_flagged_transposed_is_refused() {
    for kind in [KernelKind::Elementwise, KernelKind::Transpose] {
        let loaded = base();
        let mut compiled = loaded.artifact.clone();
        let mut modules = loaded.modules.clone();
        let at = compiled
            .execution
            .kernels
            .iter()
            .position(|k| k.kind == kind)
            .expect("the base plan has the kind");
        let old = compiled.execution.kernels[at].id;
        let kernel = &mut compiled.execution.kernels[at];
        kernel.transposed = [true, false];
        kernel.id = Kernel::module_id(&kernel.signature());
        let new = kernel.id;
        assert_ne!(new, old, "the flag is part of the signature on every kind");
        for step in compiled
            .execution
            .step
            .iter_mut()
            .chain(compiled.execution.forward.iter_mut())
        {
            if step.kernel_module_id == Some(old) {
                step.kernel_module_id = Some(new);
            }
        }
        for module in &mut modules {
            if module.kernel_module_id == old {
                module.kernel_module_id = new;
            }
        }
        let dir = scratch(&format!("{kind:?}"));
        artifact::write(&dir, &compiled, &modules).expect("written and sealed");
        let read = artifact::read(&dir);
        let verified = read
            .as_ref()
            .map(|loaded| verify_plan(&loaded.artifact.execution));
        assert!(
            !matches!(verified, Ok(Ok(()))),
            "a {kind:?} kernel flagged transposed [true, false] was read and verified"
        );
    }
}
