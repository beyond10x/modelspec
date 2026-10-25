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
pub const AS_OF: &str = "2026-10-25";

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

const fn planned(
    area: &'static str,
    label: &'static str,
    detail: &'static str,
    artifact: &'static str,
    href: &'static str,
) -> Capability {
    Capability {
        area,
        label,
        detail,
        status: "planned",
        evidence: None,
        artifact: Some(artifact),
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
        "Reverse-mode autodiff for every operator",
        "The gradient rules of `matmul`, `add`, `sub`, `mul`, `relu`, `transpose`, `sum`, `mean` and `mse` are checked against central finite differences; the rows below hold the rules of the operators added for BERT.",
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
        "SPIR-V kernels for the row-wise BERT operators",
        "embedding, add_rows, gelu, dropout, layer_norm, softmax and their gradients are one per_element, per_row or per_column dispatch each, summed in the CPU interpreter's order, without atomics, the dropout mask computed by Philox4x32-10 on the device and bit-equal to the CPU's; spirv-val accepts every module. A device run of a model with dropout is refused with E-TARGET-011.",
        Evidence::Test(
            "crates/modelspec-runtime/tests/spirv_bert_rowwise.rs::every_rowwise_kernel_on_unaligned_shapes_passes_spirv_val",
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
        "Since 0.7.0: modelspec train --backend vulkan checks every loss and the parameters it reads back, after the last step and before each checkpoint, and stops a NaN or ±inf run with the CPU run's E-TRAIN-001, stdout and events.",
        Evidence::Test(
            "crates/modelspec-cli/tests/train_vulkan_non_finite.rs::a_divergent_loss_on_vulkan_prints_what_the_cpu_prints",
        ),
        "/docs/reference/diagnostics",
    ),
    shipped(
        "GPU",
        "A matmul kernel's module id covers its tile",
        "Since 0.7.0: Kernel::signature ends with the MatMulTile, so two matmuls that differ only in tile depth or ahead get different module ids; the SPIR-V and the bits do not change.",
        Evidence::Test(
            "crates/modelspec-core/src/ir/kernel.rs::two_matmuls_differing_only_in_tile_depth_or_ahead_have_different_signatures",
        ),
        "/docs/concepts/pipeline",
    ),
    shipped(
        "Compiler",
        "The plan verifier is bounded on an untrusted artifact",
        "Since 0.7.0: an artifact whose plan has more than 4,096 steps, 65,536 dependencies or 65,536 views is refused with E-ARTIFACT-003 before the verifier runs; a re-sealed 300,000-step chain is refused in under 10 s and 1 GiB.",
        Evidence::Test(
            "crates/modelspec-core/tests/plan_verifier_bounds.rs::a_re_sealed_300000_step_chain_is_refused_in_under_10_s_and_1_gib",
        ),
        "/docs/reference/diagnostics",
    ),
    shipped(
        "Observatory",
        "The observatory UI draws a run's loss in the browser",
        "Since 0.7.0: --ui ui/dist on modelspec train --serve and modelspec observe serves the Vue app at / beside /events; it draws the loss and every metric per step, live or from a recording.",
        Evidence::Test(
            "crates/modelspec-cli/tests/observatory_ui.rs::observatory_ui_live_linear_regression",
        ),
        "/docs/reference/cli",
    ),
    shipped(
        "Observatory",
        "Parameter and gradient statistics on sampled steps",
        "Since 0.7.0: a CPU run's StepCompleted carries, per parameter, the L2 norm and a 32-bin histogram of its values and its gradient at the first step, every --statistics-every steps and the last; a Vulkan run carries its values at the first and the last step.",
        Evidence::Test(
            "crates/modelspec-cli/tests/observatory_parameter_stats.rs::observatory_stats_gradient_at_step",
        ),
        "/docs/reference/cli",
    ),
    shipped(
        "Performance",
        "A CPU run prepares its step once",
        "Since 0.8.0: the step program is verified once per run instead of once per step, each intermediate is freed after its last consumer, and dense batch tensors are refilled in place.",
        Evidence::Test(
            "crates/modelspec-runtime/tests/prepared_step.rs::a_prepared_run_calls_each_verifier_once",
        ),
        "/docs/concepts/training",
    ),
    shipped(
        "Performance",
        "A diagnostic's position is found by binary search",
        "Since 0.8.0: the parser anchors every lexeme start, so the cost of a span no longer grows with its column.",
        Evidence::Test("crates/modelspec-core/tests/span_lookup.rs::long_line_span_lookup_scans"),
        "/docs/reference/diagnostics",
    ),
    shipped(
        "Observatory",
        "Per-step device counters",
        "Since 0.8.0: modelspec train --backend vulkan --stats and --profile print the pipeline barriers and device allocations of a step and the peak device bytes of the run.",
        Evidence::Test(
            "crates/modelspec-cli/tests/adversary_step_counters.rs::stats_and_profile_print_the_same_three_counters",
        ),
        "/docs/reference/cli",
    ),
    shipped(
        "Project",
        "Every benchmark run on one page",
        "Since 0.8.0: benchmarks/runs.yaml holds every measured run of ModelSpec and its PyTorch opponents, and modelspec-docs draws a chart per workload from it.",
        Evidence::Test(
            "crates/modelspec-docs/src/benchmarks.rs::the_history_holds_every_run_the_story_names",
        ),
        "/docs/benchmarks",
    ),
    shipped(
        "GPU",
        "Register-blocked matmul for large outputs",
        "Since 0.9.0: a matmul of at least 128 blocks of 64 rows accumulates an 8×8 or 8×4 tile per invocation from fragments staged through workgroup memory; the bits do not change.",
        Evidence::Test(
            "crates/modelspec-core/tests/matmul_register_blocked.rs::a_matmul_of_at_least_1024_by_1024_outputs_is_register_blocked_at_least_4_by_4",
        ),
        "/docs/concepts/pipeline",
    ),
    shipped(
        "GPU",
        "A transposed matmul operand is read in place",
        "Since 0.9.0: a transpose that only matmuls read has no dispatch and no buffer; each matmul reads the untransposed view with its strides swapped.",
        Evidence::Test(
            "crates/modelspec-core/tests/matmul_transposed_operand.rs::no_step_plan_of_the_story_dispatches_a_transpose",
        ),
        "/docs/concepts/pipeline",
    ),
    shipped(
        "GPU",
        "Opt-in fast math",
        "Since 0.9.0: compile --fast-math and train --backend vulkan --fast-math drop NoContraction from the kernels, change the artifact hash and compare the run within tolerance; the deterministic run stays the default.",
        Evidence::Test(
            "crates/modelspec-cli/tests/fast_math.rs::compile_fast_math_sets_fast_math_in_the_artifact_and_changes_its_hash",
        ),
        "/docs/reference/cli",
    ),
    shipped(
        "Compiler",
        "Deterministic fused multiply-adds in every reduction",
        "Since 0.10.0: matmul, reductions and the mse loss sum fuse each later product into its sum, on the CPU and on Vulkan, so both give the bits of one fused reference.",
        Evidence::Test(
            "crates/modelspec-runtime/tests/cpu_fma_oracle.rs::every_cpu_matmul_kernel_gives_the_bits_of_the_fused_reference",
        ),
        "/docs/concepts/pipeline",
    ),
    shipped(
        "GPU",
        "Scheduling for a device profile",
        "Since 0.10.0: compile --device-profile and train --backend vulkan schedule the plan within the device's limits, and the artifact records them.",
        Evidence::Test(
            "crates/modelspec-cli/tests/compile_matmul_limits.rs::compile_with_a_profile_schedules_within_its_limits_and_records_them",
        ),
        "/docs/reference/cli",
    ),
    shipped(
        "GPU",
        "Elementwise chains run as one dispatch",
        "Since 0.10.0: a chain of elementwise ops whose values are each read once by the next op is one kernel, so the MNIST loss gradient is one dispatch.",
        Evidence::Test(
            "crates/modelspec-core/tests/fuse_chains.rs::the_mnist_loss_gradient_is_one_sub_mul_dispatch_with_no_buffer_for_the_difference",
        ),
        "/docs/concepts/pipeline",
    ),
    shipped(
        "Observatory",
        "Pipeline statistics per kernel",
        "Since 0.10.0: train --backend vulkan --profile prints the executable statistics the driver reports for each kernel of the plan, such as its register count.",
        Evidence::Test(
            "crates/modelspec-cli/tests/train_pipeline_statistics.rs::the_profile_flag_documents_the_profile_pipeline_lines",
        ),
        "/docs/reference/cli",
    ),
    shipped(
        "Language",
        "`i32` row ids",
        "Since 0.10.0: a tensor may be `i32` where an operator reads row ids (`embedding`) or labels (`cross_entropy`); every other use is a type error.",
        Evidence::Test(
            "crates/modelspec-cli/tests/check_i32.rs::ids_read_only_by_embedding_check_clean",
        ),
        "/docs/concepts/language",
    ),
    shipped(
        "Language",
        "zeros() and ones() initializers",
        "Since 0.10.0: a parameter of any rank from 1, such as a bias, may start at zeros() or ones(); neither draws from the seed.",
        Evidence::Test(
            "crates/modelspec-core/tests/initializers.rs::zeros_and_ones_take_a_parameter_of_rank_1_2_and_3",
        ),
        "/docs/concepts/language",
    ),
    shipped(
        "BERT operators",
        "`embedding`",
        "Since 0.10.0: a row lookup from an `f32[V,H]` table by `i32[N]` ids; the table gradient is a scatter-add checked against central differences.",
        Evidence::Test(
            "crates/modelspec-runtime/tests/gradients_embedding.rs::embedding_table_gradient_matches_central_differences",
        ),
        "/docs/concepts/bert-operators",
    ),
    shipped(
        "BERT operators",
        "`add_rows` and `gelu`",
        "Since 0.10.0: the bias row of a dense layer and the tanh form of GELU, both gradients checked against central differences.",
        Evidence::Test(
            "crates/modelspec-runtime/tests/gradients_dense.rs::add_rows_gradients_of_w_and_b_match_central_differences",
        ),
        "/docs/concepts/bert-operators",
    ),
    shipped(
        "BERT operators",
        "`gelu` gradient across its range",
        "Since 0.10.0: the `gelu` gradient matches central differences from -3 to 3.",
        Evidence::Test(
            "crates/modelspec-runtime/tests/gradients_dense.rs::gelu_gradient_of_w_matches_central_differences_across_minus_3_to_3",
        ),
        "/docs/concepts/bert-operators",
    ),
    shipped(
        "BERT operators",
        "`layer_norm`",
        "Since 0.10.0: row normalization with a gain and a bias, `eps = 1e-12`; the gradients of the input, the gain and the bias are checked against central differences.",
        Evidence::Test(
            "crates/modelspec-runtime/tests/gradients_norm.rs::layer_norm_gradients_of_w_gamma_and_beta_match_central_differences",
        ),
        "/docs/concepts/bert-operators",
    ),
    shipped(
        "BERT operators",
        "`softmax`",
        "Since 0.10.0: a row softmax with the row maximum subtracted before exp; its gradient is checked against central differences.",
        Evidence::Test(
            "crates/modelspec-runtime/tests/gradients_norm.rs::softmax_gradient_of_w_matches_central_differences",
        ),
        "/docs/concepts/bert-operators",
    ),
    shipped(
        "BERT operators",
        "`attention`",
        "Since 0.10.0: multi-head scaled dot-product attention with a key mask and the named argument `heads`; the gradients of q, k and v are checked against central differences.",
        Evidence::Test(
            "crates/modelspec-runtime/tests/gradients_attention.rs::attention_gradients_of_w_q_w_k_and_w_v_match_central_differences",
        ),
        "/docs/concepts/bert-operators",
    ),
    shipped(
        "BERT operators",
        "`cross_entropy`",
        "Since 0.11.0: the weighted masked-language-model loss over `f32[N,V]` logits, `i32[N]` labels and `f32[N]` weights; its gradient is checked against central differences.",
        Evidence::Test(
            "crates/modelspec-runtime/tests/gradients_cross_entropy.rs::cross_entropy_gradient_of_w_matches_central_differences",
        ),
        "/docs/concepts/bert-operators",
    ),
    shipped(
        "BERT operators",
        "`dropout`",
        "Since 0.11.0: `dropout(x, p = 0.1)` draws its mask from the seed, the call site and the step, applies in the training step only and recomputes the mask for its gradient.",
        Evidence::Test(
            "crates/modelspec-runtime/tests/gradients_dropout.rs::gradients_dropout_match_central_differences_at_a_fixed_token",
        ),
        "/docs/concepts/bert-operators",
    ),
    shipped(
        "BERT operators",
        "No Vulkan kernel yet for the BERT operators",
        "Since 0.10.0: every command that needs a Vulkan kernel answers E-TARGET-011 for `attention` at its call, since 0.11.0 for `cross_entropy` at its call and since 0.12.0 for an `adamw` step at its train block, before a device is opened; they train with --backend cpu. The other BERT operators have Vulkan kernels.",
        Evidence::Test(
            "crates/modelspec-cli/tests/adversary_attention_cli.rs::adversary_attention_a8_every_vulkan_command_refuses_attention_at_its_call",
        ),
        "/docs/reference/diagnostics",
    ),
    shipped(
        "Data",
        "Corpus splits",
        "Since 0.10.0: modelspec data prepare copies a WikiText-103 raw release's train and validation files into the Train and HeldOut splits and writes their record.",
        Evidence::Test(
            "crates/modelspec-cli/tests/data_prepare.rs::splits_one_per_role_cover_every_fixture_document_and_validate",
        ),
        "/docs/guides/prepare-training-data",
    ),
    shipped(
        "Data",
        "A WordPiece tokenizer of an exact vocabulary size",
        "Since 0.10.0: modelspec data tokenizer trains on the Train split to the vocabulary size asked for, the five special tokens included.",
        Evidence::Test("crates/modelspec-cli/tests/data_tokenizer.rs::exact_size"),
        "/docs/guides/prepare-training-data",
    ),
    shipped(
        "Data",
        "Packed sequences of an exact length",
        "Since 0.10.0: modelspec data pack encodes both splits into sequences of --seq-len tokens and writes their binding and their record with the held-out unigram baseline.",
        Evidence::Test("crates/modelspec-cli/tests/data_pack.rs::sequences"),
        "/docs/guides/prepare-training-data",
    ),
    shipped(
        "Data",
        "Masked-language-model batches",
        "Since 0.10.0: 15% of the tokens are selected from the seed, and of those 80% are masked, 10% replaced by a random token and 10% kept.",
        Evidence::Test("crates/modelspec-runtime/tests/mlm.rs::proportions"),
        "/docs/guides/prepare-training-data",
    ),
    shipped(
        "Training",
        "Training on a packed directory",
        "Since 0.11.0: modelspec train --backend cpu --data <dir> on a directory modelspec data pack wrote feeds each step its masked-language-model batch, and prints a data line naming the binding.",
        Evidence::Test(
            "crates/modelspec-cli/tests/train_data.rs::trains_mlm_bag_on_packed_batches",
        ),
        "/docs/guides/prepare-training-data",
    ),
    shipped(
        "Training",
        "Dropout applies in the training step only",
        "Since 0.11.0: the final loss line, --eval and infer run the forward program without dropout.",
        Evidence::Test(
            "crates/modelspec-cli/tests/train_dropout.rs::train_dropout_final_loss_is_the_forward_program_without_dropout",
        ),
        "/docs/concepts/training",
    ),
    shipped(
        "Training",
        "Held-out evaluation during training",
        "Since 0.11.0: modelspec train --backend cpu --data <dir> --eval-every <n> evaluates the masked-language-model loss and accuracy on the HeldOut split without dropout at every multiple of n and the last step, prints them and streams them as metrics.",
        Evidence::Test(
            "crates/modelspec-cli/tests/train_eval.rs::values_at_the_evaluated_steps_equal_the_test_computation",
        ),
        "/docs/reference/cli",
    ),
    shipped(
        "Training",
        "AdamW with a warm-up schedule",
        "Since 0.12.0: `optimizer = adamw(...)` trains on the CPU with decoupled weight decay and a learning rate that warms up linearly from 0 and decays linearly to 0 at the last step, compiled into the training step; its first three steps match an `f32` reference bit for bit.",
        Evidence::Test("crates/modelspec-runtime/tests/adamw.rs::one_step"),
        "/docs/concepts/language",
    ),
    shipped(
        "Training",
        "Muon optimizer on the CPU",
        "`optimizer = muon(...)` trains each matrix parameter no `embedding` reads with Muon, Nesterov momentum orthogonalized by Newton-Schulz in the training step, and every other parameter with AdamW, both on the warm-up schedule; one step matches an `f32` reference bit for bit.",
        Evidence::Test("crates/modelspec-runtime/tests/muon.rs::one_step"),
        "/docs/concepts/training",
    ),
    shipped(
        "Training",
        "BERT-tiny checked, and its hidden-32 twin trained on the CPU",
        "`examples/bert_tiny.mspec`, BERT-tiny with 4,367,290 parameters, passes modelspec check; its hidden-32 twin passes a finite-difference check of every parameter in `f64` and, trained twice from one seed with --data and --eval-every, prints the same step and eval lines both times.",
        Evidence::Test("crates/modelspec-cli/tests/bert_cpu_train.rs::reproducible"),
        "/docs/examples/bert-tiny",
    ),
    planned(
        "GPU",
        "Vulkan kernels for attention, cross-entropy and AdamW",
        "A full BERT training step on the GPU.",
        "story:bert-vulkan-kernels-attention-loss-adamw",
        "/docs/roadmap",
    ),
    planned(
        "Training",
        "Pre-train BERT-tiny on the GPU",
        "BERT-tiny pre-trained on the GPU until its held-out loss is below the unigram baseline.",
        "story:bert-tiny-gpu-pretraining",
        "/docs/roadmap",
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
                format!("{} Planned.", capability.detail)
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
        assert_eq!(document["items"][0]["detail"], "Detail. Planned.");
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

    /// The text of every shipped capability, label and detail.
    fn shipped_text() -> Vec<String> {
        CAPABILITIES
            .iter()
            .filter(|capability| capability.status == "shipped")
            .map(|capability| format!("{} {}", capability.label, capability.detail))
            .collect()
    }

    /// The newest release heading of `changelog`, `## [<version>] - <date>`, as (version, date).
    fn latest_release(changelog: &str) -> (String, String) {
        changelog
            .lines()
            .filter_map(|line| line.strip_prefix("## ["))
            .filter_map(|rest| rest.split_once("] - "))
            .map(|(version, date)| (version.to_owned(), date.trim().to_owned()))
            .next()
            .expect("CHANGELOG.md has a release")
    }

    /// The status data the generator writes is as of the newest release of `CHANGELOG.md` or
    /// later, and the status page names that release as the latest, so a release that lands
    /// without the list being checked again fails here.
    #[test]
    fn the_status_page_is_as_of_the_latest_release() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let changelog = fs::read_to_string(root.join("CHANGELOG.md")).unwrap();
        let (version, date) = latest_release(&changelog);
        assert!(
            version.split('.').count() == 3 && date.len() == "2026-10-25".len(),
            "the newest release heading reads [{version}] - {date}"
        );
        let (text, _) = document(&root).unwrap();
        let generated: Value = serde_json::from_str(&text).unwrap();
        let as_of = generated["asOf"].as_str().unwrap();
        assert!(
            as_of >= date.as_str(),
            "the generated status data is as of {as_of}, before the {version} release of {date}"
        );
        let page = fs::read_to_string(root.join("website/docs/status.mdx")).unwrap();
        assert!(
            page.contains(&format!("\n{version} is the latest release;")),
            "the status page does not say {version} is the latest release"
        );
    }

    #[test]
    fn every_registry_operator_has_a_shipped_row() {
        let shipped = shipped_text();
        for (name, _) in &modelspec_core::resolve::OPERATORS {
            assert!(
                shipped
                    .iter()
                    .any(|text| text.contains(&format!("`{name}`"))),
                "no shipped row names the operator `{name}`"
            );
        }
        for capability in CAPABILITIES {
            assert!(
                !capability.label.contains("nine"),
                "{}: the operator count is not nine",
                capability.label
            );
        }
    }

    #[test]
    fn the_data_path_has_shipped_rows() {
        let shipped = shipped_text();
        for command in [
            "modelspec data prepare",
            "modelspec data tokenizer",
            "modelspec data pack",
            "--data",
            "zeros()",
            "`i32`",
            "E-TARGET-011",
        ] {
            assert!(
                shipped.iter().any(|text| text.contains(command)),
                "no shipped row names {command}"
            );
        }
    }

    #[test]
    fn no_page_text_names_a_planning_artifact() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (text, _) = document(&root).unwrap();
        let document: Value = serde_json::from_str(&text).unwrap();
        for item in document["items"].as_array().unwrap() {
            for key in ["label", "detail"] {
                let value = item[key].as_str().unwrap();
                assert!(
                    !value.contains("story:") && !value.contains("epic:"),
                    "{key} names a planning artifact: {value}"
                );
            }
        }
        assert!(
            CAPABILITIES
                .iter()
                .any(|capability| capability.status == "planned"),
            "the planned work of the BERT milestone is on the page"
        );
    }
}
