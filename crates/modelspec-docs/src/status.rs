//! The status data, `website/data/status.json` (`b10x-status/1`), from [`CAPABILITIES`].
//!
//! `shipped` means code on `main`. A shipped capability names what holds it: a test as
//! `path::function`, or a task of `Taskfile.yml` that `task ci` runs; generation fails when that
//! test or task does not exist, so a claim cannot outlive what it rests on. A capability that is
//! not shipped names the planning artifact that carries it (`story:<id>` or `epic:<id>`), which
//! must exist in `.engineering/planning/`; generation warns about one whose artifact is already
//! `implemented`, because the item is then due to move to `shipped`.

use std::fs;
use std::path::Path;

use serde_json::{Value, json};

/// The status data's file name under [`crate::DATA`].
pub const FILE: &str = "status.json";

/// The date [`CAPABILITIES`] was last checked against `main` and the planning store.
pub const AS_OF: &str = "2026-10-23";

/// What holds a shipped capability.
#[derive(Clone, Copy)]
pub enum Evidence {
    /// A test, written `path::function` relative to the repository root.
    Test(&'static str),
    /// A task of `Taskfile.yml` that `task ci` runs.
    Task(&'static str),
}

/// One capability on the status page.
pub struct Capability {
    pub area: &'static str,
    pub label: &'static str,
    pub detail: &'static str,
    /// `shipped`, `decided` or `planned`.
    pub status: &'static str,
    /// For a shipped capability, what holds it.
    pub evidence: Option<Evidence>,
    /// For a capability that is not shipped, the planning artifact that carries it.
    pub artifact: Option<&'static str>,
    pub href: &'static str,
}

const fn shipped(
    area: &'static str,
    label: &'static str,
    detail: &'static str,
    evidence: Evidence,
    href: &'static str,
) -> Capability {
    Capability {
        area,
        label,
        detail,
        status: "shipped",
        evidence: Some(evidence),
        artifact: None,
        href,
    }
}

/// Every capability the site claims, in the order the status table lists them.
pub const CAPABILITIES: &[Capability] = &[
    shipped(
        "Language",
        "The M0 subset parses and prints canonically",
        "One model and one train block per file; the printer's output parses back to an equal AST.",
        Evidence::Test(
            "crates/modelspec-core/tests/syntax.rs::every_node_kind_round_trips_to_an_equal_ast_up_to_spans",
        ),
        "/docs/concepts/language",
    ),
    shipped(
        "Language",
        "Syntax beyond M0 is refused construct by construct",
        "module, layer, generics, symbolic dimensions and infix operators each have their own E-SYNTAX code.",
        Evidence::Test(
            "crates/modelspec-core/tests/syntax.rs::every_syntax_code_has_exactly_one_fixture",
        ),
        "/docs/examples/beyond-m0",
    ),
    shipped(
        "Checking",
        "Names, types and shapes are checked before anything runs",
        "modelspec check prints every E-NAME and E-TYPE diagnostic with its span; each code has a fixture.",
        Evidence::Test(
            "crates/modelspec-cli/tests/check.rs::every_name_and_type_code_has_a_fixture_named_for_it",
        ),
        "/docs/reference/diagnostics",
    ),
    shipped(
        "Checking",
        "A matmul shape error names both operands",
        "E-TYPE-012 prints the lhs and rhs types and the contracting dimension it expected.",
        Evidence::Test(
            "crates/modelspec-cli/tests/check.rs::matmul_mismatch_prints_the_design_diagnostic",
        ),
        "/docs/examples/shape-error",
    ),
    shipped(
        "Compiler",
        "Tensor IR lowering with a golden print",
        "modelspec ir --stage tensor prints the verified, dead-code-eliminated Tensor IR byte for byte.",
        Evidence::Test(
            "crates/modelspec-cli/tests/ir_tensor.rs::ir_stage_tensor_prints_exactly_the_golden",
        ),
        "/docs/concepts/pipeline",
    ),
    shipped(
        "Compiler",
        "Reverse-mode autodiff for all nine operators",
        "Every gradient rule is checked against central finite differences.",
        Evidence::Test("crates/modelspec-runtime/tests/gradients.rs::two_layer_relu"),
        "/docs/reference/operators",
    ),
    shipped(
        "Compiler",
        "Training IR with an SGD step and versioned parameter effects",
        "modelspec ir --stage training prints forward, backward and optimizer subgraphs and their effects.",
        Evidence::Test(
            "crates/modelspec-cli/tests/train_linear.rs::ir_stage_training_prints_the_golden",
        ),
        "/docs/concepts/pipeline",
    ),
    shipped(
        "Training",
        "Deterministic SGD training on the CPU reference interpreter",
        "Seeded initialization and synthetic data; two runs with one seed print identical bytes.",
        Evidence::Test(
            "crates/modelspec-cli/tests/train_linear.rs::two_runs_with_seed_42_print_identical_bytes",
        ),
        "/docs/concepts/training",
    ),
    shipped(
        "Training",
        "The example trains to its teacher",
        "examples/linear_regression.mspec reaches the teacher weights within the test's tolerance.",
        Evidence::Test(
            "crates/modelspec-cli/tests/train_linear.rs::training_the_example_reaches_the_teacher",
        ),
        "/docs/examples/linear-regression",
    ),
    shipped(
        "Training",
        "A binary built without the vulkan feature refuses the Vulkan backend",
        "modelspec train --backend vulkan exits 1 with E-TARGET-001 before the file is read when the binary has no Vulkan code.",
        Evidence::Test(
            "crates/modelspec-cli/tests/train_linear.rs::a_build_without_the_vulkan_feature_refuses_the_vulkan_backend",
        ),
        "/docs/reference/diagnostics",
    ),
    shipped(
        "Project",
        "The semantic core carries no Vulkan dependency",
        "task core-no-vulkan fails when any crate of modelspec-core's dependency tree is a Vulkan crate.",
        Evidence::Task("core-no-vulkan"),
        "/docs/reference/crates",
    ),
    shipped(
        "Project",
        "A usage error exits 2, a diagnostic 1",
        "The exit status of every modelspec command, held by its own tests.",
        Evidence::Test("crates/modelspec-cli/tests/exit_codes.rs::no_arguments_exits_2"),
        "/docs/reference/cli",
    ),
    shipped(
        "GPU",
        "Vulkan device discovery and feature negotiation",
        "modelspec inspect-device --backend vulkan prints each device's profile as the driver reports it and the negotiation verdict: Vulkan 1.2, no optional feature, no extension.",
        Evidence::Test(
            "crates/modelspec-runtime/src/vulkan/negotiate.rs::negotiate_on_the_fixture_returns_the_required_set",
        ),
        "/docs/reference/cli",
    ),
    shipped(
        "GPU",
        "Headless Vulkan runtime",
        "Allocation, dispatch and synchronization through ash, without a window: a training run uploads once, chains its command buffers and waits on the host once.",
        Evidence::Test(
            "crates/modelspec-runtime/tests/vulkan_train.rs::mnist_mlp_trains_on_the_device_with_one_host_wait",
        ),
        "/docs/reference/cli",
    ),
    shipped(
        "GPU",
        "SPIR-V kernels for the M0 operators and SGD",
        "Emitted with rspirv: elementwise and broadcast kernels, two-stage reductions, the mse loss sum, transpose, matmul and SGD; spirv-val accepts every module.",
        Evidence::Test(
            "crates/modelspec-runtime/tests/spirv_val.rs::spirv_val_accepts_every_form_of_the_catalog_at_every_test_size",
        ),
        "/docs/concepts/pipeline",
    ),
    shipped(
        "GPU",
        "End-to-end f32 training on an RTX 3090",
        "modelspec train --backend vulkan trains both examples on the device; every loss and final parameter lies within the GPU tolerance of the CPU reference run.",
        Evidence::Test(
            "crates/modelspec-runtime/tests/vulkan_train.rs::linear_regression_trains_on_the_device_within_tolerance_of_the_cpu",
        ),
        "/docs/reference/cli",
    ),
    shipped(
        "Command line",
        "Checkpoints and the compile and infer commands",
        "modelspec train writes and resumes .msckpt checkpoints on cpu and vulkan, the device run waiting once per requested checkpoint and once at the end; modelspec infer runs a checkpoint on either backend; a resumed device run lies within the GPU tolerance of the run that never stopped.",
        Evidence::Test(
            "crates/modelspec-runtime/tests/vulkan_checkpoint.rs::a_resumed_device_run_passes_against_the_run_that_never_stopped",
        ),
        "/docs/reference/cli",
    ),
    shipped(
        "GPU",
        "A diverged device run is refused as the CPU run is",
        "On main, not released: modelspec train --backend vulkan checks every loss and the parameters it reads back, after the last step and before each checkpoint, and stops a NaN or ±inf run with the CPU run's E-TRAIN-001, stdout and events.",
        Evidence::Test(
            "crates/modelspec-cli/tests/train_vulkan_non_finite.rs::a_divergent_loss_on_vulkan_prints_what_the_cpu_prints",
        ),
        "/docs/reference/diagnostics",
    ),
    shipped(
        "GPU",
        "A matmul kernel's module id covers its tile",
        "On main, not released: Kernel::signature ends with the MatMulTile, so two matmuls that differ only in tile depth or ahead get different module ids; the SPIR-V and the bits do not change.",
        Evidence::Test(
            "crates/modelspec-core/src/ir/kernel.rs::two_matmuls_differing_only_in_tile_depth_or_ahead_have_different_signatures",
        ),
        "/docs/concepts/pipeline",
    ),
    shipped(
        "Compiler",
        "The plan verifier is bounded on an untrusted artifact",
        "On main, not released: an artifact whose plan has more than 4,096 steps, 65,536 dependencies or 65,536 views is refused with E-ARTIFACT-003 before the verifier runs; a re-sealed 300,000-step chain is refused in under 10 s and 1 GiB.",
        Evidence::Test(
            "crates/modelspec-core/tests/plan_verifier_bounds.rs::a_re_sealed_300000_step_chain_is_refused_in_under_10_s_and_1_gib",
        ),
        "/docs/reference/diagnostics",
    ),
    shipped(
        "Observatory",
        "The observatory UI draws a run's loss in the browser",
        "On main, not released: --ui ui/dist on modelspec train --serve and modelspec observe serves the Vue app at / beside /events; it draws the loss and every metric per step, live or from a recording.",
        Evidence::Test(
            "crates/modelspec-cli/tests/observatory_ui.rs::observatory_ui_live_linear_regression",
        ),
        "/docs/reference/cli",
    ),
    shipped(
        "Observatory",
        "Parameter and gradient statistics on sampled steps",
        "On main, not released: a CPU run's StepCompleted carries, per parameter, the L2 norm and a 32-bin histogram of its values and its gradient at the first step, every --statistics-every steps and the last; a Vulkan run carries its values at the first and the last step.",
        Evidence::Test(
            "crates/modelspec-cli/tests/observatory_parameter_stats.rs::observatory_stats_gradient_at_step",
        ),
        "/docs/reference/cli",
    ),
];

/// Whether `source` defines a function called `name`.
fn defines(source: &str, name: &str) -> bool {
    source.contains(&format!("fn {name}("))
}

fn check_evidence(root: &Path, label: &str, evidence: Evidence) -> Result<(), String> {
    match evidence {
        Evidence::Test(reference) => {
            let (file, function) = reference
                .rsplit_once("::")
                .ok_or_else(|| format!("{label}: evidence {reference} is not path::function"))?;
            let source = fs::read_to_string(root.join(file))
                .map_err(|e| format!("{label}: evidence file {file}: {e}"))?;
            if !defines(&source, function) {
                return Err(format!("{label}: {file} defines no fn {function}"));
            }
        }
        Evidence::Task(task) => {
            let taskfile = fs::read_to_string(root.join("Taskfile.yml"))
                .map_err(|e| format!("{label}: Taskfile.yml: {e}"))?;
            if !taskfile.lines().any(|line| line == format!("  {task}:")) {
                return Err(format!("{label}: Taskfile.yml has no task {task}"));
            }
        }
    }
    Ok(())
}

/// The `status:` of a planning artifact, read from its front matter.
fn artifact_status(root: &Path, label: &str, artifact: &str) -> Result<String, String> {
    let (kind, name) = artifact
        .split_once(':')
        .filter(|(kind, _)| matches!(*kind, "story" | "epic"))
        .ok_or_else(|| format!("{label}: artifact {artifact} is not story:<id> or epic:<id>"))?;
    let path = root.join(format!(".engineering/planning/{kind}/{name}.md"));
    let text = fs::read_to_string(&path)
        .map_err(|e| format!("{label}: planning artifact {artifact}: {e}"))?;
    text.lines()
        .skip(1)
        .take_while(|line| *line != "---")
        .find_map(|line| line.strip_prefix("status:"))
        .map(|status| status.trim().to_owned())
        .ok_or_else(|| format!("{label}: {artifact} has no status"))
}

/// The sentence that names a shipped capability's evidence on the page.
fn held_by(evidence: Evidence) -> String {
    match evidence {
        Evidence::Test(reference) => match reference.rsplit_once("::") {
            Some((file, function)) => format!("Held by test {function} in {file}."),
            None => format!("Held by {reference}."),
        },
        Evidence::Task(task) => format!("Held by task {task}."),
    }
}

/// The status document for `capabilities` in the repository at `root`, and any warnings.
pub fn document_for(
    root: &Path,
    capabilities: &[Capability],
    as_of: &str,
) -> Result<(String, Vec<String>), String> {
    let mut items = Vec::new();
    let mut warnings = Vec::new();
    for capability in capabilities {
        let label = capability.label;
        let detail = match (capability.status, capability.evidence, capability.artifact) {
            ("shipped", Some(evidence), _) => {
                check_evidence(root, label, evidence)?;
                format!("{} {}", capability.detail, held_by(evidence))
            }
            ("shipped", None, _) => {
                return Err(format!("{label}: a shipped capability names its evidence"));
            }
            ("decided" | "planned", Some(_), _) => {
                return Err(format!("{label}: only a shipped capability names evidence"));
            }
            ("decided" | "planned", None, Some(artifact)) => {
                if artifact_status(root, label, artifact)? == "implemented" {
                    warnings.push(format!(
                        "{label}: {artifact} is implemented; move the item to shipped"
                    ));
                }
                format!("{} Planned in {artifact}.", capability.detail)
            }
            ("decided" | "planned", None, None) => {
                return Err(format!(
                    "{label}: a {} capability names its planning artifact",
                    capability.status
                ));
            }
            (other, _, _) => {
                return Err(format!(
                    "{label}: status {other} is not shipped, decided or planned"
                ));
            }
        };
        let mut item = json!({
            "area": capability.area,
            "label": label,
            "detail": detail,
            "status": capability.status,
        });
        if !capability.href.is_empty() {
            item["href"] = Value::from(capability.href);
        }
        items.push(item);
    }
    let document = json!({
        "format": "b10x-status/1",
        "asOf": as_of,
        "source": "generated by modelspec-docs from its capability list, checked against main and the planning store",
        "items": items,
    });
    let text = serde_json::to_string_pretty(&document).map_err(|e| e.to_string())?;
    Ok((text + "\n", warnings))
}

/// The status document of this repository.
pub fn document(root: &Path) -> Result<(String, Vec<String>), String> {
    document_for(root, CAPABILITIES, AS_OF)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repo(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "modelspec-docs-status-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("crates/x/tests")).unwrap();
        fs::create_dir_all(dir.join(".engineering/planning/epic")).unwrap();
        fs::write(
            dir.join("crates/x/tests/cli.rs"),
            "#[test]\nfn no_arguments_exits_2() {}\n",
        )
        .unwrap();
        fs::write(
            dir.join("Taskfile.yml"),
            "version: '3'\ntasks:\n  core-no-vulkan:\n    cmds: []\n",
        )
        .unwrap();
        fs::write(
            dir.join(".engineering/planning/epic/m1.md"),
            "---\nid: epic:m1\nkind: epic\nstatus: draft\n---\n",
        )
        .unwrap();
        fs::write(
            dir.join(".engineering/planning/epic/m0.md"),
            "---\nid: epic:m0\nkind: epic\nstatus: implemented\n---\n",
        )
        .unwrap();
        dir
    }

    fn item(
        status: &'static str,
        evidence: Option<Evidence>,
        artifact: Option<&'static str>,
    ) -> Capability {
        Capability {
            area: "Area",
            label: "Label",
            detail: "Detail.",
            status,
            evidence,
            artifact,
            href: "/docs/x",
        }
    }

    #[test]
    fn shipped_items_need_a_test_or_task_that_exists() {
        let root = repo("shipped");
        let test = Evidence::Test("crates/x/tests/cli.rs::no_arguments_exits_2");
        let (text, _) = document_for(&root, &[item("shipped", Some(test), None)], AS_OF).unwrap();
        let document: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            document["items"][0]["detail"],
            "Detail. Held by test no_arguments_exits_2 in crates/x/tests/cli.rs."
        );
        let task = Evidence::Task("core-no-vulkan");
        let (text, _) = document_for(&root, &[item("shipped", Some(task), None)], AS_OF).unwrap();
        let document: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            document["items"][0]["detail"],
            "Detail. Held by task core-no-vulkan."
        );
        for missing in [
            Evidence::Test("crates/x/tests/cli.rs::gone"),
            Evidence::Test("crates/x/tests/nowhere.rs::no_arguments_exits_2"),
            Evidence::Test("crates/x/tests/cli.rs"),
            Evidence::Task("website"),
        ] {
            assert!(document_for(&root, &[item("shipped", Some(missing), None)], AS_OF).is_err());
        }
        assert!(document_for(&root, &[item("shipped", None, None)], AS_OF).is_err());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn planned_items_need_an_artifact_and_warn_once_it_is_implemented() {
        let root = repo("planned");
        let (text, warnings) =
            document_for(&root, &[item("planned", None, Some("epic:m1"))], AS_OF).unwrap();
        assert!(warnings.is_empty());
        let document: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(document["format"], "b10x-status/1");
        assert_eq!(
            document["items"][0]["detail"],
            "Detail. Planned in epic:m1."
        );
        let (_, warnings) =
            document_for(&root, &[item("planned", None, Some("epic:m0"))], AS_OF).unwrap();
        assert_eq!(warnings.len(), 1);
        for refused in [
            item("planned", None, None),
            item("planned", None, Some("epic:gone")),
            item("planned", None, Some("task:m1")),
            item(
                "planned",
                Some(Evidence::Task("core-no-vulkan")),
                Some("epic:m1"),
            ),
            item("someday", None, Some("epic:m1")),
        ] {
            assert!(document_for(&root, &[refused], AS_OF).is_err());
        }
        let _ = fs::remove_dir_all(&root);
    }
}
