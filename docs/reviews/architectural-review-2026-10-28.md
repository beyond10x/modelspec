# ModelSpec: repository-wide Rust architectural review

Review date: 2026-10-28  
Repository: `beyond10x/modelspec`  
Reviewed checkout: `<checkout>`  
Reviewed commit: `50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae`  
Disposition: review only; no implementation fixes or tracked repository changes.

This report preserves the findings, reasoning, reproduction evidence, verification results, and limitations of the architectural review. The evidence for the review came from Rust source and Rust tests. Repository Markdown, specifications, planning records, manifests, JavaScript/TypeScript, and website content were not examined as review evidence. References to intended behavior below are grounded in Rust interfaces, Rust comments, tests, or explicitly identified mathematical properties, rather than an audit against excluded design documents.

All source links below are pinned to the reviewed commit. The report is stored outside the main checkout so the requested deliverable does not change the repository. The appendix embeds the temporary Rust probes used during the review; their original temporary locations are not required to retain the source evidence.

## Assessment

The crate separation provides a useful foundation: compiler and IR concerns live in core, runtime execution and data handling live in runtime, CLI orchestration sits above them, observatory owns events and serving, and the docs crate derives documentation through Rust tooling. The most consequential weaknesses occur where one of those boundaries assumes an invariant another component has not established.

The highest-priority findings are an unchecked device-ownership precondition in the safe Vulkan API, differing CPU/GPU admission of invalid model inputs, and a numerically unstable layer-normalization algorithm on ordinary finite values. Other findings affect checkpoint continuation, artifact admission, compiler output liveness, publication of datasets, and agreement between recording failures and completion events.

The numerical findings are known unresolved cases represented by ignored tests. They are not claimed to be new regressions or accidental deviations from the arithmetic recipes currently described in the Rust code. Those recipes themselves fail useful mathematical properties. Backend parity and tests that repeat an implementation's arithmetic are insufficient to catch that class of problem.

Severity labels indicate recommended remediation priority, not evidence of production incidents:

- **P1:** a substantial correctness or unsafe-boundary issue that should be addressed first.
- **P2:** a concrete correctness, lifecycle, or contract issue requiring a planned fix.
- **Architectural concern:** a structural limitation established by inspection, with any unmeasured consequence stated explicitly.

## Findings index

| ID | Priority | Finding | Evidence |
| --- | --- | --- | --- |
| F01 | P1 | Safe Vulkan APIs accept resources from a different logical device | Rust probe and Vulkan validation |
| F02 | P1 | Vulkan accepts out-of-range position IDs that CPU rejects | Integrated CLI reproduction |
| F03 | P1 | Layer normalization is substantially wrong for constant rows | Existing ignored regression rerun |
| F04 | P2 | Resume discards the checkpoint's RNG seed | Integrated CLI reproduction |
| F05 | P2 | Finite cross-entropy weights can silently zero all gradients | Existing ignored regression rerun |
| F06 | P2 | Artifact admission does not establish executable kernel interfaces | Mutated artifact round-trip |
| F07 | P2 | Observable outputs lack consistent liveness and materialization contracts | Two compiler API reproductions |
| F08 | P2 | Dataset publication can partially mutate output before failing | Filesystem failure reproduction |
| F09 | P2 | Failed terminal recording still publishes successful completion | Injected final-write failure |
| F10 | P2 | Vulkan runner selection discards supplied data | Static source finding |
| A01 | Concern | Optional observation work runs without an observation sink | Static source finding; unbenchmarked |
| A02 | Concern | Server lifetime and replay memory are process-scoped/unbounded | Lifetime reproduction; static scaling analysis |
| A03 | Concern | Recordings lack the identity needed to reproduce a run | Static record and writer analysis |
| A04 | Lower priority | Fused-chain names accumulate for the life of the compiler process | Static ownership analysis |

## F01 — Safe Vulkan APIs accept another logical device's resources

**Priority:** P1. **Status:** reproduced without submitting the invalid command buffer.

Primary locations:

- [dispatch.rs:471](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-runtime/src/vulkan/dispatch.rs#L471): copy-region checks and the subsequent unsafe call.
- [dispatch.rs:428](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-runtime/src/vulkan/dispatch.rs#L428): submission accepts a recorded command buffer without checking device identity.

`Recording::copy_region` checks that offsets and sizes fit the buffers, then invokes Vulkan under a safety comment asserting that both buffers belong to the recording's device. The public safe API does not establish that assertion. Its lifetimes ensure the resources remain alive; the same lifetime can describe resources created by different logical devices.

The reproduction creates logical devices A and B on the same physical GPU, creates the source buffer on B and the destination buffer and recording on A, then calls `Recording::copy`. The method returns `Ok(())`. With `MODELSPEC_VK_VALIDATION=1`, validation reports that the source buffer was created from device B while the command buffer belongs to device A, including `VUID-vkCmdCopyBuffer-commonparent`. The recording was dropped without submission.

`Device::submit` has the corresponding ownership problem for `Recorded`: it creates a fence on the receiving device and submits the supplied command buffer without establishing that the command buffer belongs there. This related path was inspected, not independently executed with mismatched resources.

**Impact:** safe Rust callers can violate the ownership preconditions of the unsafe Vulkan layer. This undermines the abstraction's safety boundary even if current CLI paths happen to pass matching resources.

**Recommended ownership:** the Vulkan resource layer should carry and enforce device identity for buffers, recordings, submissions, pipelines, descriptor arenas, and related resources. Runtime identity checks or an appropriate device-branding type design can enforce the invariant. Do not rely on matching borrow lifetimes as proof of matching device ownership.

**Acceptance criteria:** a two-device test must reject cross-device composition before issuing the Vulkan call; valid same-device paths must continue to work under validation. Audit the same assumption at submission and descriptor binding.

## F02 — GPU training silently accepts invalid embedding IDs

**Priority:** P1. **Status:** reproduced end to end on CPU and Vulkan through the freshly built CLI.

Relevant locations:

- [stepwise.rs:842](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-runtime/src/vulkan/stepwise.rs#L842): integer input upload checks only element count.
- [interpreter.rs:665](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-runtime/src/cpu/interpreter.rs#L665): CPU embedding ID validation.
- [rowwise.rs:540](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-runtime/src/spirv/emit/rowwise.rs#L540): GPU handling substitutes zero for an invalid embedding lookup.
- [data.rs:1017](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-runtime/src/train/data.rs#L1017): packed vocabulary validation recognizes token `ids`, not every embedding use such as `positions`.

The reproduction prepares an original synthetic corpus, trains a vocabulary of 300 tokens, and packs sequences of length four. The model calls `embedding(table, positions)` with `table: Tensor<f32, [2, 300]>` and `positions: Tensor<i32, [4]>`, then uses cross-entropy with packed labels and weights.

Observed results:

```text
CPU exit: 1
step 0: input positions holds id 2 at position 2, outside the 2 rows of its table

Vulkan exit: 0
step 0 loss 5.703782
step 1 loss 5.703782
```

The Vulkan run used `MODELSPEC_VK_VALIDATION=1`. Vulkan validation does not diagnose this semantic discrepancy because the emitter guards the lookup rather than making an illegal memory access.

**Impact:** switching backend changes whether an invalid model/data combination is rejected. The GPU silently trains with substituted values and reports success, so a kernel bounds guard is masking a required input diagnostic.

**Recommended ownership:** shared runtime admission should validate semantic inputs for every relevant operator use. Packed-data name conventions cannot substitute for validation of the compiled program's actual reads. GPU bounds guards should remain protective but should not turn invalid inputs into successful computation.

**Acceptance criteria:** this exact model/data combination fails before updates on both backends with a meaningful embedding-bound diagnostic; valid position and token embeddings still execute. Extend the boundary audit to attention masks and cross-entropy labels/weights, without claiming those additional cases were reproduced here.

## F03 — Layer normalization is unstable for constant rows

**Priority:** P1. **Status:** existing ignored regression rerun and still failing.

Primary location: [interpreter.rs:1332](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-runtime/src/cpu/interpreter.rs#L1332). Existing regression: [adversary_norm_runtime.rs:280](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-runtime/tests/adversary_norm_runtime.rs#L280).

The mean is accumulated in the element type and then divided by row width. For a constant `f32` row, summation rounding can make that mean differ from every identical input element. The variance calculation then treats the common rounding deviation as real variance. With epsilon `1e-12`, normalization amplifies it into a substantial output offset.

The test uses ten copies of each input value, gamma one, and beta zero. Representative observations:

| Constant input | Expected output per element | Actual `f32` output per element |
| --- | --- | --- |
| `0.1` | `0` | `-0.007450374` |
| `3.3` | `0` | `0.23191817` |
| `77.7` | `0` | `-0.99151915` |
| `123.456` | `0` | `-0.99151915` |
| `5000.3` | `0` | `-0.999998` |

Identical stored `f32` inputs have exactly zero mathematical variance, regardless of whether their decimal spelling is exactly representable. This is not an ordinary small output-rounding discrepancy.

Reproduction command:

```sh
cargo test -p modelspec-runtime --test adversary_norm_runtime \
  adversary_norm_constant_row_of_an_inexact_value_normalizes_to_beta -- --ignored
```

**Contract qualification:** the current code follows its documented arithmetic recipe. The finding is that the algorithm/recipe fails the mathematical property on valid inputs, not that the implementation accidentally diverges from an independently reviewed design specification. The test is already labeled as the known N2 finding.

**Recommended direction:** adopt a stable deterministic mean/variance calculation and review the corresponding gradient and GPU recipes together. Add active regressions for constant and nearly constant rows in actual `f32` execution. An `f64` oracle and CPU/GPU equality are useful but insufficient by themselves.

## F04 — Resume ignores the stored RNG seed

**Priority:** P2. **Status:** reproduced through the current CLI, including checkpoint contents.

Relevant locations:

- [train.rs:617](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-cli/src/train.rs#L617): retains only the checkpoint step and matched parameters.
- [checkpoint.rs:596](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-runtime/src/train/checkpoint.rs#L596): `resume_into` validates parameter compatibility and optimizer configuration, not the stored seed.
- [runner.rs:392](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-runtime/src/train/runner.rs#L392): run initialization and synthetic data use the current compiled configuration.

Reproduction sequence:

1. Train a small linear model with seed 42 for two updates and write a checkpoint.
2. Resume the same source with zero updates.
3. Change only its source seed to 43 and resume the same checkpoint with zero updates, writing another checkpoint.

Observed output:

```text
initial: step 2 loss 0.5978739
same_seed_resume: step 2 loss 0.5978739
different_seed_resume: step 2 loss 1.0993642
saved_seed=42 resumed_seed=43 same_parameters=true old_step=2 new_step=2
```

The second continuation is accepted without warning. Its parameters and step are unchanged; its synthetic data and persisted seed have changed. Inspection also shows that dropout and packed-data randomness are driven by the current configured seed, although those variants were not separately reproduced for this finding.

**Impact:** a checkpoint continuation can silently become a different stochastic experiment. Existing uninterrupted-versus-resumed tests exercise the same seed and therefore do not establish behavior for this mismatch.

**Recommended ownership:** continuation state should include the checkpoint's RNG settings. Restore the stored seed or explicitly reject an incompatible source seed. If reseeding a continuation is a desired capability, represent it as an explicit operation with corresponding provenance.

**Acceptance criteria:** changing only the source seed cannot silently change a zero-update resume; the selected restore/refuse policy is covered at both the checkpoint boundary and CLI level.

## F05 — Cross-entropy normalization can silently eliminate learning

**Priority:** P2. **Status:** existing ignored regression rerun and still failing.

Primary location: [interpreter.rs:1156](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-runtime/src/cpu/interpreter.rs#L1156). Existing regression: [adversary_cross_entropy_numerics.rs:390](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-runtime/tests/adversary_cross_entropy_numerics.rs#L390).

The forward loss accumulates its weighted numerator and denominator in `f64`. The gradient also sums weights in `f64`, but immediately converts the denominator back to the element type. With equal finite weights `[3e38, 3e38]`, the sum is finite in `f64` and infinite in `f32`. Each gradient scale becomes zero.

The regression uses logits `[0, 2, 0, 2]`, labels `[0, 0]`, and those two weights. The forward loss remains finite, while the observed gradient is:

```text
[-0.0, 0.0, -0.0, 0.0]
```

Multiplying all weights by the same positive factor should leave this normalized loss and its gradient unchanged. The result should therefore match weights `[1, 1]`, which have nonzero gradients for these logits.

Reproduction command:

```sh
cargo test -p modelspec-runtime --test adversary_cross_entropy_numerics adv_f2 -- --ignored
```

That invocation ran three existing ignored tests: the large-single-weight and subnormal-single-weight loss tests now passed, but the overflowing-weight-sum gradient test failed. The ignored annotation's old statement that the loss also becomes NaN is stale for this HEAD; the confirmed current defect is the zero gradient with finite loss.

**Contract qualification:** the narrowing is explicitly described in the Rust arithmetic recipe and repeated in a reference test. Fixing this requires reconsidering that recipe, not merely making the code match it more closely.

**Recommended direction:** preserve enough precision through normalization to avoid an infinite intermediate for finite normalized results. Apply the numerical change consistently across relevant backends and activate a positive-weight-scaling regression in actual `f32` execution.

## F06 — Artifact admission does not establish executable interfaces

**Priority:** P2. **Status:** reproduced using a freshly compiled artifact, mutations, and writer/reader round-trip.

Relevant locations:

- [verify.rs:1729](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-core/src/ir/verify.rs#L1729): dispatch validation establishes module identity and workgroups, but not the complete kernel binding contract.
- [artifact.rs:525](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-core/src/artifact.rs#L525): SPIR-V admission checks word alignment, minimum header length, and magic.

Two related missing obligations were demonstrated:

1. Clearing every dispatch's read list in a standard linear model's step and forward plans still yields `verify_plan: Ok(())`.
2. Replacing the shader bytes with only the five header words, writing the mutated artifact, and reading it back returns `Ok(5)`, admitting five invalid modules alongside missing bindings.

The replacement words were `[0x07230203, 0x00010300, 0, 1, 0]`. They are a header, not an executable shader module. This is a semantic validation gap, not a checksum bypass: the artifact is written and hashed normally.

**Impact:** successful artifact reading/plan verification provides less assurance than downstream execution needs. Different consumers must rediscover missing obligations, and failures occur farther from the trust boundary.

**Qualification:** runtime binding performs additional checks and rejects missing descriptors. This review did not demonstrate unsafe execution of the malformed artifact. Parsing a format and validating it for execution are distinct responsibilities; they should be explicit in types and APIs.

**Recommended ownership:** core should own reusable plan-to-kernel interface validation, including binding arity, dtype, and extent compatibility. The appropriate shader/runtime layer should validate module structure and executable interfaces before pipeline creation. Preserve the core/runtime dependency boundary rather than importing device execution into core merely to validate files.

**Acceptance criteria:** malformed dispatch bindings fail plan verification; header-only shaders cannot become execution-validated modules; a valid generated artifact retains a tested writer-to-reader-to-runtime path.

## F07 — Observable outputs lack explicit liveness and storage rules

**Priority:** P2. **Status:** two independent compiler API reproductions.

### F07a: prediction can disappear during DCE

Primary location: [compile.rs:232](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-core/src/compile.rs#L232). Related admission: [types.rs:975](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-core/src/types.rs#L975).

A source program calls `loss = model.ignore(model.forward(x))`, where `ignore` accepts the forward result but returns a constant. It passes `compile_hir`. DCE is rooted at the loss while `prediction` is retained separately in lowering metadata. The forward value is removed, and `compile_training` reports E-INTERNAL-002, `the lowered Tensor IR lost its prediction`.

The checker establishes a syntactic forward call; the downstream pipeline assumes the resulting value remains live. Those are different invariants.

### F07b: a scalar loss remains immediate when scheduling requires storage

Primary location: [schedule.rs:975](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-core/src/schedule.rs#L975).

A scalar `forward` returning `1.0`, used directly as the loss, passes `compile_training`. Artifact compilation for both CPU and Vulkan fails with E-INTERNAL-002, `%1 is not a value of the step`. Scalar nodes remain immediate values; the final loss lookup accepts only a stored `Place::View`. Prediction has a separate materialization path that does not satisfy the loss lookup.

**Impact:** accepted source reaches internal errors because externally observable values are only partly represented in the contracts between passes.

**Recommended ownership:** explicitly represent every observable root and every output materialization requirement. If a forward result not affecting loss is unsupported, reject it at source admission with a source diagnostic. If supported, preserve it through DCE. Materialize literal losses when the artifact interface requires storage.

**Acceptance criteria:** both cases either compile through the supported pipeline or receive an intentional source-level diagnostic; neither reaches an internal missing-value error. Test shared identity where prediction and loss are the same value.

## F08 — Dataset publication is not transactional

**Priority:** P2. **Status:** failure-path reproduction plus inspection of both publication loops.

Locations:

- [dataset/mod.rs:173](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-runtime/src/dataset/mod.rs#L173): prepared corpus publication.
- [mlm.rs:522](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-runtime/src/dataset/mlm.rs#L522): packed dataset publication.

Both writers stage individual files, then publish them through a sequence of `rename` calls. If a later rename fails, `?` returns directly from the success branch. The cleanup branch handles preparation failures, not failures during publication, and there is no rollback of files already renamed.

The probe initially prepares a dataset, changes the source corpus, and replaces the output `heldout.raw` file with a directory to force the second publication rename to fail. Observed output:

```text
prepare_failed=true
train_replaced_despite_error=true
dataset_record_still_old=true
partial_left_behind=true
```

**Qualification:** the conflicting directory means this particular starting output was no longer a valid prepared dataset. The probe establishes mutation despite refusal and stale metadata; it does not by itself establish destruction of a valid starting dataset. A later filesystem failure during replacement of an otherwise valid generation follows the same non-rollback path. Publishing metadata last and checking hashes can detect inconsistency, but cannot restore the previous generation.

The Rust API comments promise unchanged prior output on refusal, so even the demonstrated conflict violates that stated behavior.

**Recommended direction:** publish complete generations atomically, or implement rollback with explicit recovery state. Individual temporary filenames are not a multi-file transaction.

**Acceptance criteria:** inject a failure at each publication stage and verify the stated guarantee: either the complete previous generation remains usable or a clearly defined recoverable state is exposed, without silently mixing generations.

## F09 — Terminal recording failure is published as successful completion

**Priority:** P2. **Status:** reproduced with a failure imposed specifically on the final event write.

Relevant locations:

- [train.rs:725](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-cli/src/train.rs#L725): event emission records first, remembers a write error, then publishes the same event.
- [train.rs:790](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-cli/src/train.rs#L790): terminal selection and post-emission error handling.

Earlier recording failures can change the selected outcome to `RunFailed`. A failure while writing the already-selected terminal `RunFinished` occurs too late for that conversion. The post-emission check changes only the process exit status.

A Linux-only external Rust probe runs a small CPU job with both recording and SSE enabled. A custom stdout writer imposes `RLIMIT_FSIZE` at the recording's current length immediately before terminal emission. The probe ignores `SIGXFSZ` so the write returns an error, reads the SSE stream, then signals the server to finish.

Observed results:

```text
exit_status=1
terminal_sse=event: RunFinished
stderr: cannot record ... File too large
recording's last event: StepCompleted
```

**Impact:** connected observers receive a successful terminal state for an invocation that reports failure and lacks a complete recording. Existing early-write failure coverage does not establish correct final-write behavior.

**Recommended ownership:** finalization should be failure-aware before terminal publication. Define the recording result and final event together, then publish exactly one consistent terminal outcome.

**Acceptance criteria:** a test failing only the terminal write must observe failure consistently through exit status and SSE. The successful path still records and publishes one terminal event.

## F10 — Vulkan runner selection discards supplied data

**Priority:** P2. **Status:** established from source; no dedicated runtime reproduction.

Primary location: [runner.rs:264](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-runtime/src/vulkan/runner.rs#L264). Input contract: [runner.rs:210](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-runtime/src/vulkan/runner.rs#L210).

`RunOptions.data` supplies a batch and teacher in place of regenerating synthetic data. When dropout or AdamW selects the stepwise implementation, `train_on_device_from` destructures this field as `data: _` and creates `StepwiseOptions { data: None, ... }`. The selected runner regenerates synthetic data.

The stepwise-routing comment describes regeneration, but the caller's explicit override is silently discarded rather than preserved or refused. A cached unmodified batch may happen to reproduce the same values; the option itself still has no effect on this route.

**Impact:** a backend implementation choice changes input ownership and can change the experiment when a caller supplies a different batch. It also prevents callers from relying on the advertised batch-reuse behavior consistently.

**Recommended direction:** propagate the supplied batch through runner selection or explicitly reject an unsupported override. Shared run-input types should make it difficult for an implementation branch to discard an execution input.

**Acceptance criteria:** compare a deliberately distinct supplied batch across both runner-selection paths; it must be honored or rejected explicitly, not silently replaced.

## Architectural concerns

### A01 — Optional observation work runs with no observation output

Locations: [train.rs:1088](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-cli/src/train.rs#L1088), [train.rs:725](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-cli/src/train.rs#L725), and [statistics.rs:34](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-observatory/src/statistics.rs#L34).

The CPU observer computes parameter/gradient statistics on sampled steps regardless of whether `--record` or `--serve` is present. Event creation also occurs with neither sink configured. Histograms allocate an `f64` copy of every sampled tensor and scan those values repeatedly.

This places optional observation costs in the default execution path and can distort comparisons between the CLI and library-only training. The work and allocations are visible in source; their percentage of training time was not benchmarked.

Keep the observation policy separate from the training computation. Determine which information active sinks need, and avoid computing unused statistics. Verify the change with a test or instrumentation that detects statistics work when no sink is present; benchmark only after defining the workload and comparison being claimed.

### A02 — Server lifetime and replay history lack bounded ownership

Locations: [server.rs:119](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-observatory/src/server.rs#L119), [server.rs:154](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-observatory/src/server.rs#L154), and [server.rs:293](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-observatory/src/server.rs#L293).

The listener and shared state move into the accept thread. There is no shutdown coordination or `Drop` implementation to stop it. The public type's Rust comment explicitly describes a process-long lifetime; that is a deliberate constraint, but it is awkward for an embeddable library and for error paths that return after binding.

**Reproduced:** `Server::bind -> drop(server) -> TcpListener::bind(same_address)` returns `Address already in use`. The thread also keeps the shared history and optional UI alive.

Separately, every serialized frame remains in memory. A client concatenates all outstanding history into another allocation while holding the publication mutex. Late or slow readers can retain additional large copies, and copying history blocks publication. Socket writes have no timeout. These scaling consequences were established from source; no load test or peak-memory measurement was performed.

Add explicit shutdown that closes the listener, wakes waiters, and joins relevant workers. Preserve replay semantics with bounded chunks and an appropriate retained-history policy, potentially backed by the recording file. Test rebind after shutdown, early failures after bind, reconnect semantics, and late-reader behavior on a representative long recording.

### A03 — Recordings do not identify the experiment they record

Locations: [event.rs:143](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-observatory/src/event.rs#L143) and [train.rs:758](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-cli/src/train.rs#L758).

`RunStarted` contains invocation steps, start step, and an optional packed-data binding ID. The reviewed writer does not persist a corresponding run manifest associating the randomly minted run ID with source/artifact hashes, backend/device, seed, numerical settings, initialization identity, or resume checkpoint identity. Dense and synthetic data are not identified by the packed binding field.

An event file can replay what was observed, but cannot by itself identify or reconstruct the execution that produced those observations. This is a code-level provenance gap; it is not a claim that an excluded specification was audited and found violated.

A runtime run manifest should own effective execution inputs and be referenced by observatory events. It should record resolved values after resume handling and backend selection, so the manifest describes what actually ran. Verify that a recorded run can be tied to an unambiguous model, backend, random configuration, initialization, and data source.

### A04 — Fused-chain names live forever in the compiler process

Location: [kernel.rs:836](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-core/src/ir/kernel.rs#L836).

The interner stores distinct fused-chain names in a process-global set and obtains their static lifetime through `Box::leak`. Repeated occurrences of the same name are deduplicated, but distinct names are retained for the entire process. Short CLI invocations bound that lifetime naturally; a long-lived compiler or artifact-processing service does not.

This lower-priority observation was established by inspection, without a memory benchmark. Prefer an owned typed representation or interning scoped to a compilation/session if long-lived use is intended. A stress check should vary distinct supported chains rather than repeatedly compiling only the same one.

## Recommended remediation order

1. Enforce Vulkan device ownership and shared semantic input admission (F01–F02).
2. Correct numerical recipes with independent mathematical property checks in `f32` (F03, F05), reviewing CPU and GPU behavior together.
3. Make continuation state and effective run inputs explicit (F04, F10), then persist their identity (A03).
4. Strengthen compiler and artifact boundary contracts (F06–F07).
5. Make dataset publication and terminal finalization robust against late failures (F08–F09).
6. Bound observation costs, replay storage, and server lifetime (A01–A02); address process-global interning if long-lived compiler use is in scope (A04).

These are implementation directions, not completed fixes. Each finding includes a representative acceptance check so remediation can be tied to the demonstrated failure instead of only adding broader component coverage.

## Verification record

| Check | Result | What it establishes |
| --- | --- | --- |
| `cargo test -p modelspec-cli --test train_linear --test checkpoint_cpu` | 19 passed | Normal integrated CPU training, checkpointing, inference, and same-seed continuation paths exercised by those tests |
| `cargo test -p modelspec-core -p modelspec-runtime -p modelspec-observatory --lib` | 190 passed: core 115, runtime 71, observatory 4 | Existing library component tests pass |
| Cross-entropy ignored `adv_f2` tests | 2 passed, 1 failed | Loss-only cases improved; overflowing weight sum still zeroes gradients |
| Ignored constant-row layer-normalization test | 1 failed | Current `f32` constant-row normalization remains incorrect |
| CPU/Vulkan packed-position probe | CPU rejects; Vulkan succeeds | Integrated backend admission discrepancy |
| Two-logical-device recording probe | API accepts; Vulkan validation rejects ownership | Unsafe precondition is not enforced by the safe wrapper |
| Resume seed probe | Mismatch accepted; loss and saved seed change with zero updates | Stored RNG seed is not preserved or validated |
| Malformed artifact probe | `verify_plan` accepts missing reads; reader returns five modules | The admitted artifact lacks required executable interfaces |
| Two compiler-output probes | Earlier stage succeeds; later stage returns E-INTERNAL-002 | Observable-root/storage contract gaps |
| Dataset publication probe | Failure after one replacement; old manifest and partial files remain | Publication failure does not restore previous output |
| Final recording-write probe | Exit 1; SSE `RunFinished`; incomplete recording | Terminal outcome is inconsistent across channels |
| Server-drop probe | Rebind fails with address in use | Dropping the public server does not stop its listener |

The 209 passing normal tests do not negate the targeted failures. Two additional ignored tests passed, while the two numerical regressions described above failed. GPU probes used the local device; no timing benchmarks or concurrent timed GPU workloads were run.

## Coverage and limits

The review covered the Rust architecture of parser/resolver/checker, lowering and DCE, Tensor and training IR, differentiation and optimizer attachment, scheduling and Kernel IR, artifact serialization/admission, CPU execution and tensor storage, initialization/data/checkpoint paths, corpus/tokenizer/packing, Vulkan negotiation/resources/dispatch/synchronization and training runners, relevant shader emission, CLI orchestration, observatory events/recording/server/statistics, and the Rust documentation generator.

Relevant adversarial tests and regression tests were inspected selectively. This was not an assertion that every test file or every possible operator/backend combination had been exhaustively audited. No additional high-priority docs-generator finding was established. The docs generator's generated output and non-Rust inputs were not audited or regenerated.

Full CI, the complete integration suite, the full ignored GPU suite, browser/UI checks, and performance benchmarks were not run. No conclusion about overall release readiness or measured overhead follows from this review. Probe evidence demonstrates the stated scenarios; other paths explicitly labeled static remain unverified at runtime.

The checkout was clean at the start and after the review. Tests generated ordinary build/test outputs under `target`; those are not repository implementation changes. The report and preserved probe text are deliverables of the subsequent request to save the review.

## Reproduction appendix

The following Rust programs are the probes used in the review, preserved as source. Their hard-coded paths reflect the reviewed machine. Before running elsewhere, adjust the checkout and temporary directory paths. Create `/tmp/modelspec-review-cli` before running the CLI probes. The final-write-failure probe uses Linux signal/resource-limit constants and should be run as its own disposable process. The device-ownership probe intentionally records an invalid cross-device operation for validation, then drops it without submission.

The Vulkan-position probe uses the repository's original synthetic-corpus helper. That helper is included at the end of this appendix; the pinned repository source is [markov_corpus.rs](https://github.com/beyond10x/modelspec/blob/50ef1cdee14184c3aa5f1b8b1ba42b2fbcc845ae/crates/modelspec-cli/tests/common/markov_corpus.rs).

The sources were linked with `rustc --edition=2024` against the freshly built workspace libraries, using `-L dependency=target/debug/deps` and the relevant `--extern` arguments. The integrated probes call the freshly built `target/debug/modelspec`. Example compilation pattern:

```sh
rustc --edition=2024 /path/to/probe.rs \
  -L dependency=target/debug/deps \
  --extern modelspec_runtime=target/debug/deps/libmodelspec_runtime-<matching-build-hash>.rlib \
  -o /path/to/probe
```

Use the core, CLI, or observatory library instead for probes importing those crates. Runtime GPU probes require a library built with the Vulkan feature. Build hashes are intentionally not presented as stable identifiers; use artifacts from a matching build of the reviewed revision. Run GPU probes with `MODELSPEC_VK_VALIDATION=1` to collect the validation evidence described above.

### F01: cross-device Vulkan recording

Original location: `/tmp/modelspec-review-vulkan-owner.rs`  
Source SHA-256: `07ecd3183950ec04c7e0048e2bb4934248dd3f0eeaa8b4ffd5cddd44f1c9dfb0`

```rust
use modelspec_runtime::vulkan::{Vulkan,negotiate::negotiate};
fn main(){
 let vk=Vulkan::load().unwrap();
 let candidate=&vk.candidates().unwrap()[0];
 let negotiated=negotiate(&candidate.profile).unwrap();
 let a=vk.create_device(0,&negotiated).unwrap();
 let b=vk.create_device(0,&negotiated).unwrap();
 let source=b.staging_buffer(16).unwrap();
 let target=a.staging_buffer(16).unwrap();
 let mut record=a.record().unwrap();
 println!("cross-device copy accepted: {:?}",record.copy(&source,&target));
 drop(record);
 for message in vk.take_validation_messages(){println!("{message}");}
}
```

### F02: integrated CPU/Vulkan position-ID comparison

Original location: `/tmp/modelspec-review-vulkan-positions.rs`  
Source SHA-256: `008b4d86e79250285ff53a7b1e47efd6396d74f11a696b5c4f9f346d2dce7da3`

```rust
use std::{path::Path,process::{Command,Output}};
#[path="<checkout>/crates/modelspec-cli/tests/common/markov_corpus.rs"] mod markov_corpus;
fn cli(args: &[&str]) -> Output { Command::new("<checkout>/target/debug/modelspec").args(args).output().unwrap() }
fn checked(args: &[&str]) { let out=cli(args); assert!(out.status.success(), "{args:?}: {}",String::from_utf8_lossy(&out.stderr)); }
fn main() {
 let root=Path::new("/tmp/modelspec-review-vulkan-positions"); std::fs::create_dir_all(root).unwrap();
 let corpus=root.join("corpus"); markov_corpus::write(&corpus);
 let dataset=root.join("dataset"); let tok=root.join("tokenizer.json"); let packed=root.join("packed");
 checked(&["data","prepare",corpus.to_str().unwrap(),"--out",dataset.to_str().unwrap()]);
 checked(&["data","tokenizer","--dataset",dataset.to_str().unwrap(),"--vocab-size","300","--out",tok.to_str().unwrap()]);
 checked(&["data","pack","--dataset",dataset.to_str().unwrap(),"--tokenizer",tok.to_str().unwrap(),"--seq-len","4","--out",packed.to_str().unwrap()]);
 let source=r#"model M {
 param table: Tensor<f32, [2, 300]> = xavier_uniform();
 fn forward(positions: Tensor<i32, [4]>) -> Tensor<f32, [4, 300]> { return embedding(table, positions); }
}
train T {
 model = M;
 input positions: Tensor<i32, [4]>;
 target labels: Tensor<i32, [4]>;
 target weights: Tensor<f32, [4]>;
 loss = cross_entropy(model.forward(positions), labels, weights);
 optimizer = sgd(lr = 0.01);
 steps = 1;
 seed = 42;
 device = cpu;
}
"#;
 let file=root.join("positions.mspec");std::fs::write(&file,source).unwrap();
 checked(&["check",file.to_str().unwrap()]);
 for backend in ["cpu","vulkan"] { let out=cli(&["train",file.to_str().unwrap(),"--backend",backend,"--data",packed.to_str().unwrap()]); println!("backend {backend} exit {:?}",out.status.code()); println!("stderr: {}",String::from_utf8_lossy(&out.stderr)); for line in String::from_utf8_lossy(&out.stdout).lines().filter(|l|l.starts_with("step ")||l.starts_with("data ")) {println!("{line}");} }
}
```

### F04: checkpoint resume with a different source seed

Original location: `/tmp/modelspec-review-cli/resume_seed.rs`  
Source SHA-256: `cba73938cc437f0df0ae67b88119e186595c0d4a5f8e5a343785eb0a6da2606e`

```rust
use std::path::{Path,PathBuf};
use std::process::Command;
use modelspec_runtime::train::checkpoint;
fn run(args: &[&str]) -> String {
    let out=Command::new("<checkout>/target/debug/modelspec").args(args).output().unwrap();
    assert!(out.status.success(),"{:?}: {}",args,String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap()
}
fn arg(path: &Path)->&str {path.to_str().unwrap()}
fn main() {
    let dir=PathBuf::from("/tmp/modelspec-review-cli");
    let first=dir.join("seed42.mspec"); let second=dir.join("seed43.mspec");
    let source=r#"model Linear {
        param w: Tensor<f32, [1, 1]> = xavier_uniform();
        fn forward(x: Tensor<f32, [8, 1]>) -> Tensor<f32, [8, 1]> { return matmul(x, w); }
    }
    train Fit {
        model = Linear;
        input x: Tensor<f32, [8, 1]>;
        target y: Tensor<f32, [8, 1]>;
        loss = mse(model.forward(x), y);
        optimizer = sgd(lr = 0.1);
        steps = 2;
        seed = 42;
        device = cpu;
    }"#;
    std::fs::write(&first,source).unwrap();
    std::fs::write(&second,source.replace("seed = 42","seed = 43")).unwrap();
    let original=dir.join("seed-original.msckpt"); let changed=dir.join("seed-changed.msckpt");
    let base=run(&["train",arg(&first),"--backend","cpu","--steps","2","--checkpoint",arg(&original)]);
    let same=run(&["train",arg(&first),"--backend","cpu","--steps","0","--resume",arg(&original)]);
    let different=run(&["train",arg(&second),"--backend","cpu","--steps","0","--resume",arg(&original),"--checkpoint",arg(&changed)]);
    for (name,text) in [("initial",base),("same_seed_resume",same),("different_seed_resume",different)] {
        println!("{name}: {}",text.lines().filter(|l| l.starts_with("step ")).last().unwrap());
    }
    let old=checkpoint::read(&original).unwrap(); let new=checkpoint::read(&changed).unwrap();
    println!("saved_seed={} resumed_seed={} same_parameters={} old_step={} new_step={}",old.rng.seed,new.rng.seed,old.parameters==new.parameters,old.step,new.step);
}
```

### F06: malformed artifact writer/reader round-trip

Original location: `/tmp/modelspec-core-artifact-review.RYeJbA/probe.rs`  
Source SHA-256: `6c52843f6b42b3b3581fe690c2b7d8ad47fe7bb4f452e450e85d8d80cf5ff3ad`

```rust
use modelspec_core::artifact::{self,Backend,KernelModuleFile,KernelCacheKey};
use modelspec_core::{compile,ir::kernel::PlanOp};
use modelspec_core::ir::tensor::DType;
const BASE: &str = r#"model M { param w: Tensor<f32, [2, 1]> = zeros(); fn forward(x: Tensor<f32, [3, 2]>) -> Tensor<f32, [3, 1]> { return matmul(x,w); } } train T { model = M; input x: Tensor<f32, [3, 2]>; target y: Tensor<f32, [3, 1]>; loss = mse(model.forward(x),y); optimizer = sgd(lr = 0.1); steps = 1; seed = 1; device = cpu; }"#;
fn main() {
 let mut a=compile::compile_artifact("p",BASE,Backend::Vulkan).unwrap();
 for s in a.execution.step.iter_mut().chain(a.execution.forward.iter_mut()).filter(|s|s.op==PlanOp::Dispatch) { s.reads.clear(); }
 let modules:Vec<_>=a.execution.kernels.iter().map(|k|KernelModuleFile{kernel_module_id:k.id,entry_point:"main".into(),cache_key:KernelCacheKey{ir_hash:a.ir_hash.0.clone(),shapes:vec![],dtype:DType::F32,fast_math:false,deterministic:true,vendor_id:0,device_id:0,driver_version:0,capabilities:vec![]},spirv:[0x07230203u32,0x00010300,0,1,0].into_iter().flat_map(u32::to_le_bytes).collect()}).collect();
 let dir=std::env::temp_dir().join(format!("modelspec-review-malformed-artifact-{}",std::process::id()));
 artifact::write(&dir,&a,&modules).unwrap();
 println!("artifact reader accepts missing bindings and no-op SPIRV header: {:?}",artifact::read(&dir).map(|x|x.modules.len()));
}
```

### F06 and F07b: missing dispatch reads and literal-loss compilation

Original location: `/tmp/modelspec-core-review.Dj5W2j/probe.rs`  
Source SHA-256: `37e5d3fd55c272b69b9893df3da1037f9254b953d8c5452b37b331305d149211`

```rust
use modelspec_core::artifact::Backend;
use modelspec_core::{compile, ir::kernel::PlanOp, ir::verify::verify_plan};
const S: &str = r#"model M { fn forward(x: Tensor<f32, []>) -> Tensor<f32, []> { return 1.0; } } train T { model = M; input x: Tensor<f32, []>; loss = model.forward(x); optimizer = sgd(lr = 0.1); steps = 1; seed = 1; device = cpu; }"#;
const BASE: &str = r#"model M { param w: Tensor<f32, [2, 1]> = zeros(); fn forward(x: Tensor<f32, [3, 2]>) -> Tensor<f32, [3, 1]> { return matmul(x,w); } } train T { model = M; input x: Tensor<f32, [3, 2]>; target y: Tensor<f32, [3, 1]>; loss = mse(model.forward(x),y); optimizer = sgd(lr = 0.1); steps = 1; seed = 1; device = cpu; }"#;
fn main() {
 println!("compile_training literal loss: {:?}",compile::compile_training("p",S).map(|_|()));
 for backend in [Backend::Cpu,Backend::Vulkan] { println!("compile_artifact {backend:?}: {:?}",compile::compile_artifact("p",S,backend).map(|_|())); }
 let mut a=compile::compile_artifact("p",BASE,Backend::Vulkan).unwrap();
 for s in a.execution.step.iter_mut().chain(a.execution.forward.iter_mut()).filter(|s|s.op==PlanOp::Dispatch) { s.reads.clear(); }
 println!("all Dispatch read bindings removed, verify_plan: {:?}", verify_plan(&a.execution));
}
```

### F07a: prediction discarded by dead-code elimination

Original location: `/tmp/modelspec-core-forward-review.aKPH9K/probe.rs`  
Source SHA-256: `5e7b30f7b3cef0cd9ef1d1d283a584d6eebda893dee9b12b7c5044db38b5c6e5`

```rust
use modelspec_core::compile;
const S: &str = r#"model M { param w: Tensor<f32, [1]> = ones(); fn forward(x: Tensor<f32, [1]>) -> Tensor<f32, [1]> { return mul(x,w); } fn ignore(p: Tensor<f32, [1]>) -> Tensor<f32, []> { return 1.0; } } train T { model = M; input x: Tensor<f32, [1]>; loss = model.ignore(model.forward(x)); optimizer = sgd(lr = 0.1); steps = 1; seed = 1; device = cpu; }"#;
fn main() { println!("check {:?}",compile::compile_hir("p",S).map(|_|())); println!("compile {:?}", compile::compile_training("p",S).map(|_|())); }
```

### F08: failure during dataset publication

Original location: `/tmp/modelspec-review-dataset-commit.rs`  
Source SHA-256: `981cbdee612682a6bb84f28f91c7089058aaafdb341c662abb9b07dc21bb2dd7`

```rust
use modelspec_runtime::dataset;
fn main() {
    let root = std::env::temp_dir().join(format!("modelspec-review-commit-{}", std::process::id()));
    let corpus = root.join("corpus");
    let out = root.join("prepared");
    std::fs::create_dir_all(&corpus).unwrap();
    std::fs::write(corpus.join("wiki.train.raw"), "old training words\n").unwrap();
    std::fs::write(corpus.join("wiki.valid.raw"), "old heldout words\n").unwrap();
    dataset::prepare(&corpus, &out).unwrap();
    let old_train = std::fs::read(out.join("train.raw")).unwrap();
    let old_record = std::fs::read(out.join("dataset.json")).unwrap();
    std::fs::write(corpus.join("wiki.train.raw"), "new training words\n").unwrap();
    std::fs::write(corpus.join("wiki.valid.raw"), "new heldout words\n").unwrap();
    std::fs::remove_file(out.join("heldout.raw")).unwrap();
    std::fs::create_dir(out.join("heldout.raw")).unwrap();
    let result = dataset::prepare(&corpus, &out);
    println!("prepare_failed={}", result.is_err());
    println!("train_replaced_despite_error={}", std::fs::read(out.join("train.raw")).unwrap() != old_train);
    println!("dataset_record_still_old={}", std::fs::read(out.join("dataset.json")).unwrap() == old_record);
    println!("partial_left_behind={}", out.join("heldout.raw.partial").exists());
    std::fs::remove_dir_all(root).unwrap();
}
```

### F09: failure writing the terminal recording event

Original location: `/tmp/modelspec-review-cli/probe.rs`  
Source SHA-256: `371c55f90e6f9c71afac5eba57c0146b820b851efbf2df5eb0288af622ac01c2`

```rust
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::Duration;
use modelspec_cli::train::{self, Args, Backend};

#[repr(C)]
struct Rlimit { soft: u64, hard: u64 }
unsafe extern "C" {
    fn setrlimit(resource: i32, limit: *const Rlimit) -> i32;
    fn signal(sig: i32, handler: usize) -> usize;
    fn kill(pid: i32, sig: i32) -> i32;
    fn getpid() -> i32;
}
struct ArmLimit { path: PathBuf, armed: bool }
impl Write for ArmLimit {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { Ok(bytes.len()) }
    fn flush(&mut self) -> io::Result<()> {
        if !self.armed {
            let size = std::fs::metadata(&self.path)?.len();
            unsafe {
                signal(25, 1);
                assert_eq!(setrlimit(1, &Rlimit { soft: size, hard: u64::MAX }), 0);
            }
            self.armed = true;
        }
        Ok(())
    }
}
fn main() {
    let dir = PathBuf::from("/tmp/modelspec-review-cli");
    let model = dir.join("model.mspec");
    std::fs::write(&model, r#"model Linear {
        param w: Tensor<f32, [1, 1]> = xavier_uniform();
        fn forward(x: Tensor<f32, [8, 1]>) -> Tensor<f32, [8, 1]> { return matmul(x, w); }
    }
    train Fit {
        model = Linear;
        input x: Tensor<f32, [8, 1]>;
        target y: Tensor<f32, [8, 1]>;
        loss = mse(model.forward(x), y);
        optimizer = sgd(lr = 0.1);
        steps = 1;
        seed = 42;
        device = cpu;
    }"#).unwrap();
    let address = { let socket=TcpListener::bind("127.0.0.1:0").unwrap(); socket.local_addr().unwrap() };
    let client = std::thread::spawn(move || {
        let mut stream = (0..100).find_map(|_| {
            match TcpStream::connect(address) { Ok(s) => Some(s), Err(_) => { std::thread::sleep(Duration::from_millis(10)); None } }
        }).expect("listener ready");
        stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        write!(stream, "GET /events HTTP/1.1\r\nHost: {address}\r\n\r\n").unwrap();
        let mut body=String::new(); stream.read_to_string(&mut body).unwrap();
        unsafe { kill(getpid(),15); }
        body
    });
    let record=dir.join("record.jsonl");
    let args=Args { path: model, backend: Backend::Cpu, serve: Some(address), record: Some(record.clone()), ui: None,
        data:None,init:None,eval:None,eval_every:None,steps:None,checkpoint:None,checkpoint_every:None,write_batch:None,
        resume:None,statistics_every:None,device:None,stats:false,profile:false,fast_math:false };
    let mut stdout=ArmLimit {path:record.clone(),armed:false};
    let mut stderr=Vec::new();
    let code=train::run(args,&mut stdout,&mut stderr);
    unsafe { assert_eq!(setrlimit(1,&Rlimit {soft:u64::MAX,hard:u64::MAX}),0); }
    let body=client.join().unwrap();
    println!("exit_status={code}");
    println!("terminal_sse={}",body.lines().filter(|l| l.starts_with("event:")).last().unwrap());
    println!("stderr={}",String::from_utf8(stderr).unwrap());
    println!("recording_ends_with={}",std::fs::read_to_string(record).unwrap().lines().last().unwrap());
}
```

### A02: server listener survives drop

Original location: `/tmp/modelspec-review-cli/server_lifetime.rs`  
Source SHA-256: `8c5133e28e3f7b620dc5717902462ca41d40638155a9c50cd8fa2b95a1054794`

```rust
use modelspec_observatory::{Server, TrainingRunId};
use std::net::TcpListener;
fn main() {
    let server=Server::bind("127.0.0.1:0".parse().unwrap(),TrainingRunId::new()).unwrap();
    let address=server.local_addr();
    drop(server);
    match TcpListener::bind(address) {
       Ok(_) => println!("rebind=ok"),
       Err(e) => println!("rebind_after_drop={e}"),
    }
}
```

### Supporting helper: original synthetic Markov corpus

Original location: `<checkout>/crates/modelspec-cli/tests/common/markov_corpus.rs`  
Source SHA-256: `269fa53a6794e0e29071de4b2045a37fbfb8114f9a8d22187de20ff3651c0e0b`

```rust
//! A deterministic synthetic corpus with structure a small masked-language model can learn, in
//! the layout `modelspec data prepare` reads (coordinator decision D72).
//!
//! The vocabulary is 200 invented two-syllable words. Each word has 2 to 4 fixed successors drawn
//! uniformly, the first of them the most likely ([`WEIGHTS`]), so a word's neighbours narrow it
//! to a few words where its frequency alone leaves about 200: the bigram entropy of the text is
//! far below its unigram entropy. The text is a walk of that chain,
//! cut into sentences of 6 to 12 words that end with ` .`, paragraphs of 4 sentences and articles
//! of 3 paragraphs under a heading ` = <title> = ` with a blank line (one space) before and after
//! it, as the release files of the corpus are laid out. `wiki.train.raw` and `wiki.valid.raw` walk
//! the same chain from other seeds, so the held-out text is not the training text.

use std::path::Path;

/// The words of the chain.
const WORDS: usize = 200;
/// The seed the vocabulary and the successors are drawn with.
const CHAIN_SEED: u64 = 0x6d61_726b_6f76_0001;
const TRAIN_SEED: u64 = 0x6d61_726b_6f76_0002;
const VALID_SEED: u64 = 0x6d61_726b_6f76_0003;
/// The weight of a word's first, second, third and fourth successor: the first is taken 3 times
/// in 4 or more, so a word's left neighbour mostly determines it. With successors of equal
/// weight the twin's held-out loss stayed at its unigram level for 1,000 steps (fix pass 2 of
/// story:bert-tiny-cpu-model).
const WEIGHTS: [usize; 4] = [12, 2, 1, 1];
/// Articles of the train and of the valid file.
const TRAIN_ARTICLES: usize = 40;
const VALID_ARTICLES: usize = 10;

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The invented words and each word's successors.
struct Chain {
    words: Vec<String>,
    successors: Vec<Vec<usize>>,
}

impl Chain {
    fn new() -> Self {
        const ONSETS: [&str; 12] = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t"];
        const VOWELS: [&str; 5] = ["a", "e", "i", "o", "u"];
        let mut rng = Rng(CHAIN_SEED);
        let mut words: Vec<String> = Vec::new();
        while words.len() < WORDS {
            let word: String = (0..2)
                .map(|_| format!("{}{}", ONSETS[rng.below(12)], VOWELS[rng.below(5)]))
                .collect();
            if !words.contains(&word) {
                words.push(word);
            }
        }
        let successors = (0..WORDS)
            .map(|_| (0..2 + rng.below(3)).map(|_| rng.below(WORDS)).collect())
            .collect();
        Chain { words, successors }
    }

    /// The successor of `word` the walk takes next: the `r`-th of its successors with weight
    /// `WEIGHTS[r]`, so the first is the most likely.
    fn next(&self, word: usize, rng: &mut Rng) -> usize {
        let next = &self.successors[word];
        let total: usize = WEIGHTS[..next.len()].iter().sum();
        let mut x = rng.below(total);
        for (&w, &weight) in next.iter().zip(&WEIGHTS) {
            if x < weight {
                return w;
            }
            x -= weight;
        }
        unreachable!("x is below the total weight")
    }

    /// `articles` articles of a walk drawn with `seed`.
    fn text(&self, seed: u64, articles: usize) -> String {
        let mut rng = Rng(seed);
        let mut at = rng.below(WORDS);
        let mut step = |rng: &mut Rng| {
            at = self.next(at, rng);
            self.words[at].as_str()
        };
        let mut out = String::new();
        for _ in 0..articles {
            let mut title: Vec<String> = Vec::new();
            for _ in 0..2 {
                let word = step(&mut rng);
                let mut chars = word.chars();
                let first = chars.next().unwrap().to_ascii_uppercase();
                title.push(format!("{first}{}", chars.as_str()));
            }
            out.push_str(&format!(" \n = {} = \n \n", title.join(" ")));
            for _ in 0..3 {
                let mut line = String::new();
                for _ in 0..4 {
                    for _ in 0..6 + rng.below(7) {
                        line.push(' ');
                        line.push_str(step(&mut rng));
                    }
                    line.push_str(" .");
                }
                out.push_str(&line);
                out.push_str(" \n \n");
            }
        }
        out
    }
}

/// Writes `wiki.train.raw` and `wiki.valid.raw` of the synthetic corpus into `dir`.
pub fn write(dir: &Path) {
    write_with(dir, VALID_ARTICLES);
}

/// [`write`] with `valid_articles` articles in `wiki.valid.raw`, the first of them those of
/// [`write`]'s, for a larger `HeldOut` split over the same `Train` split.
pub fn write_with(dir: &Path, valid_articles: usize) {
    std::fs::create_dir_all(dir).expect("corpus directory");
    let chain = Chain::new();
    for (file, seed, articles) in [
        ("wiki.train.raw", TRAIN_SEED, TRAIN_ARTICLES),
        ("wiki.valid.raw", VALID_SEED, valid_articles),
    ] {
        std::fs::write(dir.join(file), chain.text(seed, articles)).expect("corpus file");
    }
}
```

