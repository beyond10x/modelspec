I created 10 draft stories for `epic:m1-vulkan-execution`. Between them they claim all nine M1 checkboxes, and `aep plan artifact validate` exits 0. Nothing was moved or committed. Each draft body is at `~/.local/state/worktree/trees/b10x/modelspec/modelspec-plan-ui/.engineering/drafts/m1-<slug>.md`, which is git-ignored.

**Stories.** Each story also depends on `epic:m1-vulkan-execution` and serves `vision:modelspec`. Paths marked (new) were recorded as inferred; the rest were recorded as cited.

| id | title | depends_on (stories) | scope paths |
|---|---|---|---|
| story:m1-mlp-example | MNIST-shaped MLP example, gradient-checked and trained on the CPU | none | Taskfile.yml, AGENTS.md, CHANGELOG.md, runtime tests/gradients.rs, docs src/example.rs, website/docs/examples; new: examples/mnist_mlp.mspec, cli tests/mnist_mlp.rs, golden/mnist_mlp.training.ir |
| story:m1-inspect-device | Vulkan device discovery with `inspect-device` | none | Cargo.toml, core lib.rs, runtime and cli Cargo.toml/lib.rs, cli main.rs, docs diagnostics.rs and status.rs, website/docs/reference, DESIGN.md, AGENTS.md, CHANGELOG.md; new: core src/device.rs, runtime src/vulkan, tests/vulkan_device.rs, fixtures/rtx3090.device.json, cli src/inspect_device.rs, tests/inspect_device.rs |
| story:m1-kernel-plan | Kernel IR, fixed safe schedule and execution plan, in core | none | spec/domains/artifact.yaml, core ir/{mod,verify,print}.rs, compile.rs, lib.rs, Cargo.toml, cli src/ir.rs, docs src/cli.rs, website/docs/reference, DESIGN.md, CHANGELOG.md; new: ir/kernel.rs, schedule.rs, artifact.rs, tests kernel_plan{,_adversary}.rs, cli tests/ir_kernel.rs, golden/linear_regression.kernel.ir |
| story:m1-vulkan-elementwise | Vulkan memory, pipelines and dispatch; elementwise, broadcast and SGD kernels | inspect-device, kernel-plan | Cargo.toml, runtime Cargo.toml/lib.rs, Taskfile.yml, .github/workflows/check.yml, DESIGN.md, AGENTS.md, CHANGELOG.md; new: runtime src/spirv, src/vulkan, tests/common/mod.rs, tests/spirv_val.rs, tests/vulkan_elementwise.rs |
| story:m1-vulkan-reduce-matmul | Reduction, transpose and naive matmul kernels | vulkan-elementwise | DESIGN.md, CHANGELOG.md; new: spirv/emit.rs, vulkan/dispatch.rs, tests spirv_val.rs and vulkan_reduce_matmul.rs |
| story:m1-compile-artifact | `modelspec compile`: the artifact on disk and a validating reader | reduce-matmul, kernel-plan, mlp-example | core Cargo.toml, cli lib.rs, docs diagnostics.rs and cli.rs, website/docs/reference, DESIGN.md, CHANGELOG.md; new: core artifact.rs, spirv/emit.rs, cli src/compile.rs, tests compile{,_adversary}.rs |
| story:m1-vulkan-train | `train --backend vulkan` with one host wait | compile-artifact, mlp-example | runtime lib.rs and train/runner.rs, cli train.rs, tests/train_linear.rs, docs status.rs and diagnostics.rs, website roadmap.md, getting-started.md, reference, data, DESIGN.md, AGENTS.md, README.md, CHANGELOG.md; new: src/vulkan, tests vulkan_train.rs and no_cuda.rs, cli tests/train_vulkan.rs |
| story:m1-checkpoint-cpu | `.msckpt` save and resume, `infer --backend cpu` | kernel-plan | runtime train/{mod,runner}.rs and Cargo.toml, cli train.rs and lib.rs, docs diagnostics.rs and cli.rs, website/docs/reference, DESIGN.md, CHANGELOG.md; new: train/checkpoint.rs, tests checkpoint{,_adversary}.rs, cli infer.rs, tests/checkpoint_cpu.rs |
| story:m1-vulkan-checkpoint-infer | GPU checkpoints, resume on the GPU, `infer --backend vulkan` | vulkan-train, checkpoint-cpu | cli train.rs, docs status.rs, website roadmap.md and data, AGENTS.md, CHANGELOG.md; new: src/vulkan, train/checkpoint.rs, tests/vulkan_checkpoint.rs, cli infer.rs, tests/checkpoint_vulkan.rs |
| story:m1-vulkan-profile | `train --profile`: per-kernel timestamps and run counters | vulkan-train | cli train.rs, website/sidebars.ts, website/docs/reference, DESIGN.md, CHANGELOG.md; new: src/vulkan, tests/vulkan_profile.rs, cli tests/train_vulkan.rs, website/docs/guides/inspect-a-run.md |

**Who claims each checkbox**

| # | checkbox | claimed by | also contributes |
|---|---|---|---|
| 1, 2, 3 | .mspec MLP; parse to optimizer step; CPU trains and validates gradients | mlp-example | kernel-plan (2) |
| 4 | RTX 3090 found, features printed | inspect-device | |
| 5 | valid SPIR-V for the 5 kernel kinds | compile-artifact | elementwise, reduce-matmul |
| 6, 7 | GPU forward, backward, update without CUDA; GPU/CPU agree | vulkan-train | elementwise sets the tolerances |
| 8 | checkpoint and inference keep parameter identities | checkpoint-cpu | vulkan-checkpoint-infer (the GPU half) |
| 9 | kernels, IR, plan and timings inspectable | vulkan-profile | kernel-plan, compile-artifact |

**One-line acceptance per story**
- mlp-example: `task check-examples` checks the new example; the sampled gradient check passes at 1e-6 (f64) and 1e-3 (f32); a CPU run lowers the loss.
- inspect-device: fails with E-TARGET-002 when no Vulkan loader is found; on the operator run, vendor 0x10de, subgroup size 32 and all 21 profile fields printed.
- kernel-plan: the artifact schema is typed first and passes `ess specify validate`; one dispatch per op; the verifier refuses 6 altered plans; a golden file for `ir --stage kernel`.
- vulkan-elementwise: tolerances written into DESIGN first; `spirv-val` in CI; device run within tolerance with 0 validation messages and 1 host wait per case.
- vulkan-reduce-matmul: DESIGN conflict settled first; MNIST matmul shapes within tolerance; bit-identical across 2 runs.
- compile-artifact: all 5 kernel kinds pass `spirv-val`; two compiles are byte-equal; 7 tampered artifacts refused with E-ARTIFACT codes.
- vulkan-train: 1 host wait, at most 8 submits, 520,864 bytes up and 210,468 down; final parameters within 1e-4 relative L2 of the CPU; no CUDA dependency.
- checkpoint-cpu: 250+250 steps resumed equals 500 steps byte for byte; `infer` loss equals the step-500 loss; 8 malformed checkpoints refused.
- vulkan-checkpoint-infer: 3 host waits with `--checkpoint-every 600`; GPU resume within tolerance; CPU and GPU `infer` agree.
- vulkan-profile: one line per kernel with 1,800 or 1,801 calls; same host waits and same stdout with or without `--profile`.

**Host round trips.** No story needs a host wait per operator or per step. Four acceptance items check the wait count directly: elementwise 5, vulkan-train 2, checkpoint-infer 2 and profile 2. vulkan-train also rewrites DESIGN.md § 6 to block only at the end of a run or at a requested checkpoint.

**UNMAPPED markers.** None of CompiledArtifact, KernelModule, Checkpoint, ParameterRecord or DeviceProfile has one, so none needed settling. Four fields and rules were missing, and each is the first acceptance item of a story:

| what is missing | story |
|---|---|
| buffer views and `forward_plan` in artifact.yaml | kernel-plan |
| the GPU tolerances in DESIGN.md § 9 | elementwise |
| the deterministic reduction rule in DESIGN.md § 3.3 and § 5.2 | reduce-matmul |
| the .msckpt header layout in DESIGN.md § 6.2 | checkpoint-cpu |

Left as they are:
- the `schedule` marker in `training.yaml:53`, which belongs to the BERT work;
- device ownership of `BufferView`, which stays untyped.

**Relation census.** 13 relations: 10 rest on the spec, 3 on code (`training.rs:127`, `data.rs:36`, `compile.rs:37`). None needs stakeholder input, so no blockers were filed.

**Which M1 story each later 10× story builds on**
- Whole-run schedule and gathering minibatches on the GPU: vulkan-train and kernel-plan.
- Fusion: kernel-plan and reduce-matmul.
- Tiled matmul: reduce-matmul and inspect-device (shared-memory limit).
- Reusing buffers once they are free: kernel-plan.
- Benchmark gate and per-run reports: vulkan-profile and mlp-example.
- MNIST data on the GPU: story:bert-train-data-binding and vulkan-train.
- The 10× ADR: no M1 story.

**Decisions not settled by the epic (all mine)**
1. The example uses the MNIST benchmark shapes but trains on synthetic data, because the data interface is still open (DESIGN § 12).
2. Device and CLI setup:
   - `ash` loads the Vulkan library at run time.
   - The CLI gets a `vulkan` feature, on by default.
   - Two new codes: E-TARGET-002 (no loader) and E-TARGET-003 (no device or compute queue).
   - `MODELSPEC_VULKAN_LIBRARY` lets a test point at a loader that does not exist.
   - The DeviceProfile type lives in core as plain data, and `serde` is added.
3. The step plan has no completion point. The runtime adds one at the end of the run and at each checkpoint.
4. Plan rules:
   - Each SGD update is in place, and the verifier proves it runs after every read of the old value.
   - Buffers are not reused in M1.
   - Workgroup limits are the Vulkan minimums (128 invocations, 65,535 groups).
   - `schema_hash` does not depend on the backend or the learning rate.
5. GPU tolerances: absolute 1e-5 and relative 1e-4, taken from story:bert-vulkan-kernels-rowwise. The relative term uses the sum of absolute terms for reductions and matmul. A whole run must end within 1e-4 relative L2 per parameter.
6. CI and test setup:
   - SPIR-V generation sits behind the `vulkan` feature.
   - `task test` and `task lint` add `--features vulkan`.
   - CI installs `spirv-tools`, and the test fails rather than skips without `spirv-val`.
   - GPU tests are `#[ignore]` and agents run them on the local GPU.
   - Nothing in `task ci` creates a Vulkan instance, and there is no software Vulkan driver.
7. **Needs a reviewer:** DESIGN § 3.3 forbids reassociating sums in deterministic mode, but § 5.2 asks for a two-stage reduction, which does exactly that. I settled it as a fixed reduction tree chosen from the shape alone. Matmul keeps the CPU's order (ascending k, no fused multiply-add). This is a design change.
8. Artifact files: `artifact.json` plus `kernels/<id>.spv`. Ids come from content hashes, `deterministic=true` and `fast_math=false`.
9. Up to 256 steps per command buffer, and losses collect in a buffer on the device.
10. Checkpoints:
    - 64-byte alignment, and the payload hash is always written.
    - Periodic checkpoints are named `<stem>-<k>.msckpt`.
    - The RNG counter is 0, since SGD draws no random numbers per step.
    - `infer` runs on the synthetic batch and has no `--input`.
11. compile-artifact also depends on mlp-example, because its acceptance compiles `mnist_mlp.mspec`.

**Not drafted:** M2 work (fusion, tiling, whole-run schedule, pipeline cache), the `bench` verb (in DESIGN § 8 but not among the nine checkboxes), `--data`/`--input`, a TrainingRun record on disk, `tiny_mlp`/modules, a software Vulkan driver.

**Refusals, verbatim.** One from `aep`, when I created vulkan-elementwise before inspect-device existed. It succeeded after I created inspect-device.
```
error: `story:m1-vulkan-elementwise` would point at `story:m1-inspect-device`, which this store does not hold: locator `ep://planning/store/story/m1-inspect-device` does not exist
```
The session's worktree hook also refused two commands. The first was 3 `aep new` calls chained into one Bash call. The second was the inspect-device `new` call, whose summary began with the word "ash"; I reworded the summary. All 10 were created afterwards as single commands.

**`aep plan artifact validate`, verbatim (exit 0)**
```
84 file(s) in ~/.local/state/worktree/trees/b10x/modelspec/modelspec-plan-ui/.engineering/planning: 84 artifact(s)
valid
```
