# Changelog

All notable changes to ModelSpec are recorded here, in the format of
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). ModelSpec releases from source at
bare-version tags such as `0.1.0`; nothing is on a registry. Each release on GitHub carries the
`modelspec` CLI binary for Linux x86_64 and its SHA-256 checksum. Changes collect under
`[Unreleased]` until the next release; the detail of each change is in its story and commit
history.

## [Unreleased]

### Changed

- `modelspec train --data DIR` on a dense directory binds every `input` and `target` declaration by name, however many there are: `NAME.f32` for an `f32` declaration and `NAME.i32` for an `i32` one, a sample one row along the leading dimension all declarations share, else the whole declared tensor; step k feeds block k of every file, on the CPU and on Vulkan. `--eval DIR` takes the same layout, runs on Vulkan too, and prints `accuracy=` only when the loss compares the model output with the block's one `f32` target. A classifier, a block whose one target is the `i32` labels a `cross_entropy` reads, trains and prints no accuracy. For dense data `E-DATA-001` now means a block with no declaration, `E-DATA-002` applies only to that one compared target and never to `i32` labels, `E-DATA-004` no longer applies, and a missing file is `E-DATA-007` naming it and, when only that exists, the file of the other dtype. The new `E-DATA-011` refuses, once the files of `--data` and `--eval` are read and before the first step on both backends, an `i32` value that is negative or not below the rows of an `embedding` table or the classes of `cross_entropy` logits that read it, naming the file, the element position and the bound (DESIGN.md revision 0.56) (story:decider-dense-named-data).

## [0.21.0] - 2026-10-28

ModelSpec measures its first workload beyond the MLP sweep. `examples/transformer_block.mspec`, the two encoder layers of BERT-tiny, trains on the CPU and through Vulkan, and the bench `transformer_block` times it at batches of 8, 32 and 128 sequences against the same program in PyTorch CUDA, eager and `torch.compile` (`opponents/transformer_block/`), whose `--compare-step` takes one SGD step from a ModelSpec checkpoint and compares every parameter with ModelSpec's checkpoint of the next step. On an RTX 3090 at batch 8, ModelSpec Vulkan takes a median of 93.9 ms per step against 3.9 ms for PyTorch eager and 2.0 ms for `torch.compile`, and the Benchmarks page gains a Transformer block section with the figures of all three batches. `modelspec train --write-batch` writes the synthetic batch a run trains on, synthetic data feeds an input read only as an attention key mask with all ones, and the CPU interpreter accumulates every whole-tensor sum in `f64`, so the losses a CPU run prints can change in their last digits. The specification and CI use ESS 0.72.0.

### Added

- `examples/transformer_block.mspec`: the two encoder layers of BERT-tiny (hidden 128, 2 heads, feed-forward 512, sequences of 128, post layer norm, no dropout) on batches of 8 sequences, an MSE loss against a target of the same shape and SGD at lr 0.01. It passes `modelspec check` (now part of `task check-examples`) and trains on the CPU and through Vulkan, ten Vulkan steps within the run tolerance of the CPU run (story:transformer-block-benchmark).
- The bench `transformer_block` (`cargo bench -p modelspec-runtime --features vulkan --bench transformer_block`): the example at batches of 8, 32 and 128 sequences (`--point B`, repeatable; `--min-seconds`, default 5), with the MLP sweep's warm-up, calibration and three timed runs, one JSON line per point with tokens and matmul-only FLOP per step (339,738,624·B), the median µs per step, tokens per second and GFLOP/s (story:transformer-block-benchmark).
- `opponents/transformer_block/`: the same program in PyTorch CUDA, eager and `torch.compile`, writing the bench's lines, with `--checkpoint` to start from a ModelSpec checkpoint's parameters, `--loss-check` to print the step-0 loss of a checkpoint on the batch ModelSpec wrote as a sanity check, and `--compare-step` to take one SGD step from a ModelSpec checkpoint on that batch and compare every parameter with ModelSpec's checkpoint of the next step, the check that shows both sides compute the same function; its `.msckpt` reader verifies the header checksum, `file_size` and payload hash and refuses a file ModelSpec refuses with `E-CKPT-001`, `E-CKPT-002` or `E-CKPT-004` (story:transformer-block-benchmark).
- `modelspec train --write-batch <DIR>` writes the synthetic batch a run trains on before its first step: one file `DIR/NAME.f32` per `input` and `target` declaration `NAME`, the tensor's elements in row-major order as little-endian `f32` and nothing else, 4 bytes per element of the declared shape (for the example at batch 8, `x.f32` and `y.f32` of 524,288 bytes and `key_mask.f32` of 4,096). It is the opponent's input, not a `--data` directory: `--data` on the example is refused with `E-DATA-001`, the key mask being a second input (story:transformer-block-benchmark).
- Synthetic data feeds an input that the program reads only as the key mask of `attention` all ones, beside the one drawn input, where `train` refused a file with two inputs with `E-DATA-001` (DESIGN.md revision 0.55) (story:transformer-block-benchmark).
- `benchmarks/runs.yaml` takes `tokens_per_step` on a run; a workload whose runs state it gets a section on the Benchmarks page with a chart of µs per step against batch and a table of tokens per second and GFLOP/s per side and batch, and `modelspec-docs generate` checks a `MatmulOnly` count per workload (3196·B·H for `mlp-784-H-10`, 339,738,624·B for `transformer-block`) and refuses `MatmulOnly` on any other workload, and a `transformer-block` run without `batch` or `tokens_per_step`, or with tokens that are not 128·B, at the field (story:transformer-block-benchmark).
- The first transformer-block figures, nine runs in `benchmarks/runs.yaml` and the Transformer block section of the Benchmarks page: on an RTX 3090 at batch 8 (1,024 tokens per step), ModelSpec Vulkan 0.20.0 takes a median of 93,868.4 µs per step, PyTorch CUDA eager 3,937.2 µs and `torch.compile` 1,971.9 µs; PyTorch eager is faster by 23.8×, 46.5× and 53.1× at batches 8, 32 and 128, and `torch.compile` by 47.6×, 97.8× and 124.9×. Three timed runs per point at a123c1a, clocks not locked, raw records in `benchmarks/raw/2026-10-27-transformer-block/` (verification-report:transformer-block-benchmark-2026-10-27) (story:transformer-block-benchmark).

### Changed

- The CPU interpreter accumulates every sum of a whole tensor into one scalar, `reduce_sum`, `mean`, `mse`, the two sums of `cross_entropy` and the held-out loss, in `f64` and rounds it to the element type once; rows, columns and `matmul` elements keep the element type. The step-0 loss of `examples/transformer_block.mspec` at batch 128 is now within the run tolerance of PyTorch's, and the losses a CPU run prints can change in their last digits (DESIGN.md revision 0.55, coordinator decision D84) (story:transformer-block-benchmark).
- The specification and CI use ESS 0.72.0 (was 0.68.0); the JSON Schemas under `crates/modelspec-observatory/schema` and `crates/modelspec-runtime/schema` regenerate unchanged.

## [0.20.0] - 2026-10-26

The Benchmarks page shows the MLP sweep against FLOP per step: one log-log chart of µs per step for ModelSpec Vulkan, PyTorch CUDA eager and PyTorch CUDA `torch.compile`, a GFLOP/s column per side in each batch's table of latest medians, and a "Not measured" section that names the declared workloads without a run yet, the transformer block and BERT-tiny pre-training, with the work that will measure each. `modelspec-docs generate` refuses a run whose `workload` the page does not declare or whose `MatmulOnly` `flop_per_step` is not 3196·B·H, and the pitch deck gains the sweep slide. This release changes no compiler or runtime behaviour and no benchmark number.

### Added

- The Benchmarks page (`website/docs/benchmarks.md`) shows the MLP sweep in one chart of µs per step against FLOP per step, both axes log scale, for ModelSpec Vulkan, PyTorch CUDA eager and PyTorch CUDA `torch.compile` from the runs that state `flop_per_step`, one line per side and batch named in the legend, its caption naming the `flop_rule` the points count by; each batch's table of latest medians gains a GFLOP/s column per side, `flop_per_step` / µs per step / 1000, blank where the run states no `flop_per_step`; and a "Not measured" section lists the declared workloads without a run, the transformer block and BERT-tiny pre-training, each with the work that will measure it, until its first run in `benchmarks/runs.yaml` gives it a section of its own (story:benchmark-results-page).
- `modelspec-docs generate` refuses a run of `benchmarks/runs.yaml` whose `workload` is not one the page declares, and a `MatmulOnly` run whose `flop_per_step` is not 3196·B·H, naming the entry's line and field (story:benchmark-results-page).
- Pitch deck `docs/pitch/`: slide 15, after the MNIST slide, shows the MLP sweep of release 0.10.0 on the RTX 3090, each side's crossover width at batch 1000 (ModelSpec Vulkan H 256, PyTorch eager and `torch.compile` H 4096) and the GFLOP/s of the largest point, batch 8192 and H 4096 (14,834.7, 15,676.1 and 15,812.4); the slides after it move up by one, to 18 (story:benchmark-results-page).

## [0.19.0] - 2026-10-26

BERT-tiny is pre-trained on the GPU below the unigram baseline, and the run is on the documentation site: `examples/bert_tiny.mspec` trained for 10,000 steps on an RTX 3090 through Vulkan over packed WikiText-103 ends at a held-out loss of 6.7674994 nats against the held-out unigram baseline of 6.9265379 (6.9041541 over the evaluated rows) in 45 min. The run record, `docs/runs/bert-tiny-gpu-pretraining` in a new Runs section, holds the commands, the commit, the device, every `eval` line, the training loss every 50 steps, the wall time and the SHA-256 of the recorded event file. This release changes no compiler or runtime behaviour.

### Added

- The record of the first GPU pre-training run of `examples/bert_tiny.mspec`, `website/docs/runs/bert-tiny-gpu-pretraining.md` under a new Runs section of the documentation site: 10,000 steps on an RTX 3090 through Vulkan with the 0.18.0 binary over packed WikiText-103 end at a held-out loss of 6.7674994 nats, 0.159 nats (2.30 %) below the held-out unigram baseline of 6.9265379, with the commands, the commit, the device, every `eval` line, the training loss every 50 steps, the wall time and the SHA-256 of the recorded event file, which stays outside the repository (story:bert-tiny-gpu-pretraining).

## [0.18.0] - 2026-10-26

Held-out evaluation on the GPU works at corpus scale: `train --backend vulkan --eval-every <n>` evaluates each held-out batch in a command buffer of its own through a download buffer of one batch's prediction, so `examples/bert_tiny.mspec` over the packed WikiText-103 corpus trains with `--eval-every` where it was refused before step 0 with `9126805508 bytes exceed maxStorageBufferRange 4294967295`. The `eval` lines and the recorded metrics are unchanged, and `--stats` counts the evaluation's submits and host waits.

### Fixed

- `modelspec train --backend vulkan --eval-every <n>` evaluates each `HeldOut` batch in a command buffer of its own, one host wait per batch, through a download buffer of one batch's prediction (`modelspec_runtime::vulkan::stepwise::download_bytes`), where it downloaded every batch's logits at once: `examples/bert_tiny.mspec` over the packed WikiText-103 corpus at seq-len 128 exited 1 before step 0 with `9126805508 bytes exceed maxStorageBufferRange 4294967295` for any `--eval-every`; the `eval` lines and `StepCompleted` metric samples are reduced as before, in batch order, and `--stats` counts the evaluation's submits and host waits (DESIGN.md revision 0.54) (story:vulkan-eval-download-per-batch)

## [0.17.0] - 2026-10-26

The BERT twin trains on the GPU: Vulkan kernels for `attention` and its gradients, for `cross_entropy` and its gradient, and for the `Sqrt` and `Div` of the AdamW update, each within the GPU tolerance of the CPU interpreter, let `train --backend vulkan` take `--data`, `--eval-every`, `--serve` and `--record` and train the hidden-32 BERT twin with dropout and AdamW, one host wait per step and the held-out evaluation in the step's command buffer, every step and eval line and every final parameter within the run tolerance of the CPU run. `train --backend vulkan` no longer refuses a model with `dropout`; `muon` still answers `E-TARGET-011` on the GPU, and so does `infer --backend vulkan` on a model with dropout.

### Added

- Vulkan kernels for `attention` and its gradients `AttentionGradQ`, `AttentionGradK` and `AttentionGradV`, for `cross_entropy` and its gradient `CrossEntropyGrad`, and for the `Sqrt` and `Div` of the AdamW update, naive, one dispatch per op, every sum in the CPU interpreter's ascending order; on the RTX 3090 each is within the GPU tolerance of the CPU interpreter, and `train --backend vulkan --data <dir> --eval-every <n>` trains the hidden-32 BERT twin with dropout and AdamW, one host wait per step and the evaluation in the step's command buffer, every step and eval line and every final parameter within the new run tolerance of the CPU run, streaming them with `--serve` and `--record`; `muon` still answers E-TARGET-011 (DESIGN.md revision 0.53, story:bert-vulkan-kernels-attention-loss-adamw).

### Changed

- `train --backend vulkan` no longer refuses a model with `dropout`: each step uploads the RNG token of every dropout site, so the device draws the CPU run's masks; `infer --backend vulkan` still refuses one with E-TARGET-011, because inference runs the step plan with its dropouts (story:bert-vulkan-kernels-attention-loss-adamw).

## [0.16.0] - 2026-10-25

BERT-tiny written out in the language: `examples/bert_tiny.mspec` (2 post-layer-norm encoder layers, hidden 128, 2 heads, feed-forward 512, sequences of 128, the 8,192-token vocabulary; 1,486,720 parameters) passes `modelspec check` as part of `task check-examples`, and its hidden-32 twin passes a finite-difference check of every parameter in `f64` and trains reproducibly on the CPU from one seed. `train --data` on a packed directory refuses with `E-DATA-006`, before the first step, an `embedding` table or `cross_entropy` logits whose size differs from the packed vocabulary.

### Added

- `examples/bert_tiny.mspec`, BERT-tiny (2 post-layer-norm encoder layers, hidden 128, 2 heads, feed-forward 512, sequences of 128, the 8,192-token vocabulary of the corpus tokenizer, an MLM head tied to the token table; 1,486,720 parameters, where Turc et al.'s BERT-tiny has 4.4M at 30,522 tokens), joins `task check-examples`, and its hidden-32 twin `crates/modelspec-cli/tests/fixtures/bert_cpu.mspec` passes a finite-difference check of every parameter in `f64` and trains to identical `step` and `eval` lines twice from one seed with `train --backend cpu --data --eval-every` (story:bert-tiny-cpu-model).

### Changed

- `train --data` on a packed directory refuses, with `E-DATA-006` before the first step, an `embedding` table read with the bound `ids` whose rows, or `cross_entropy` logits read with the bound `labels` whose classes, are not the packed `vocabulary_size`, naming the table or logits and both numbers; a wider table trained without a word, a narrower one failed at step 0 (story:bert-tiny-cpu-model).

## [0.15.0] - 2026-10-25

The `muon` optimizer on the CPU reference: each rank-2 parameter that no `embedding` reads as its table trains with Nesterov momentum orthogonalized by quintic Newton–Schulz steps, and every other parameter trains with AdamW. `E-TYPE-025` refuses a `momentum` outside [0, 1) and an `ns_steps` outside [1, 10], `train --checkpoint` and `--resume` refuse a Muon run, and a non-finite Newton-Schulz norm or buffer stops the run with E-TRAIN-001.

### Added

- The `muon` optimizer on the CPU reference: `muon(lr, momentum, weight_decay, ns_steps, warmup_steps, adam_lr, beta1, beta2, eps)` trains each rank-2 parameter that no `embedding` reads as its table with Nesterov momentum orthogonalized by quintic Newton–Schulz steps, and every other parameter with AdamW at `adam_lr`; `E-TYPE-025` refuses a `momentum` outside [0, 1) and an `ns_steps` outside [1, 10], `train --checkpoint` and `--resume` refuse a Muon run, and a non-finite Newton-Schulz norm or buffer is E-TRAIN-001 (DESIGN.md revision 0.51, story:muon-optimizer).

## [0.14.0] - 2026-10-25

Vulkan kernels for the row-wise BERT operators: `embedding`, `add_rows`, `gelu`, `dropout`, `layer_norm`, `softmax` and their gradients run on the device as one dispatch each, every sum in the CPU interpreter's order; `EmbeddingGrad` adds without atomics, so two runs give the same bits; and the `dropout` mask is computed on the device bit-equal to the CPU's. `train --backend vulkan` trains a model of a matmul, `add_rows` and `gelu` within the GPU tolerance of the CPU run; a device run of a model with `dropout` is still refused with E-TARGET-011, because the device run supplies no RNG token per step.

### Added

- Vulkan kernels for `embedding`, `add_rows`, `gelu`, `dropout`, `layer_norm`, `softmax` and their internal gradients (`EmbeddingGrad`, `SumRows`, `GeluGrad`, `DropoutGrad`, `LayerNormGradX`, `LayerNormGradGamma`, `SoftmaxGrad`), one `per_element`, `per_row` or `per_column` dispatch each with workgroups of 64, every sum in the CPU interpreter's ascending order and every product after the first fused by one `Fma` decorated `NoContraction`; `EmbeddingGrad` adds without atomics, so two runs give the same bits, and `Dropout` computes Philox4x32-10 on the device, its mask bit-equal to the CPU's; `train --backend vulkan` trains a matmul, `add_rows` and `gelu` model within the GPU tolerance of the CPU run, and a device run of a model with `dropout` is still refused with E-TARGET-011 because the device run supplies no RNG token per step; DESIGN.md revision 0.50 (story:bert-vulkan-kernels-rowwise).

## [0.13.0] - 2026-10-25

A faster Vulkan training step with the same bits: a `relu_grad` whose gradient operand is a matmul output nothing else reads is now an epilogue of that matmul (`epilogue ReluGrad`), so the MNIST step plan has 9 dispatches, 6 barriers and 12 buffers instead of 10, 7 and 13, and the `mse` loss sum is recorded beside the largest matmul it can share a barrier interval with instead of in an interval of its own. Every bit of the losses, parameters, checkpoints and predictions is unchanged, and at `mlp_sweep` b8192 h4096 on the RTX 3090 the step takes 6,858.9 µs against 6,568.5 µs for PyTorch eager, a ratio of 1.044 (was 1.111).

### Changed

- The Vulkan step plan records the `mse` loss sum right after the largest matmul it can share a barrier interval with, the weight gradient of the first layer in the MLPs, instead of in an interval of its own, and the final loss of `train --backend vulkan` and `infer --backend vulkan` record only the steps the loss is computed from (`ExecutionPlan::loss_steps`, new); every bit of the losses, parameters, checkpoints and predictions is unchanged; at `mlp_sweep` b8192 h4096 on the RTX 3090 the step went from 7,293.8 to 7,050.5 µs in the probe session; DESIGN.md revision 0.45 (story:step-sites-beat-pytorch).
- A `relu_grad` whose gradient operand is a matmul output that nothing else reads is now an epilogue of that matmul in the Vulkan plan (`epilogue ReluGrad`): the matmul reads the pre-activation as a third operand and writes the gated gradient, so the matmul output has no buffer and the separate `relu_grad` dispatch is gone; the MNIST step plan has 9 dispatches, 6 barriers and 12 buffers (was 10, 7 and 13), and `inspect-a-run.md` quotes the new plan; every bit of the losses, parameters, checkpoints and predictions is unchanged; at `mlp_sweep` b8192 h4096 on the RTX 3090 the fused matmul takes 433.9 µs against 805.0 µs for the two dispatches before, and the step with both changes of the story takes 6,858.9 µs against 6,568.5 µs for PyTorch eager (ratio 1.044, was 1.111); DESIGN.md revision 0.49 (story:step-sites-beat-pytorch).

## [0.12.0] - 2026-10-25

AdamW on the CPU interpreter: `optimizer = adamw(...)` trains with decoupled weight decay, per-parameter moments versioned with the parameters, and a learning rate that warms up linearly from 0 and decays linearly to 0 at the run's last step. E-TYPE-023 and E-TYPE-024 refuse out-of-range arguments, E-TRAIN-001 also stops the run on a non-finite moment, a resumed AdamW run is refused while the moments are not checkpointed, and `sgd(lr = F)` keeps its constant rate.

### Added

- `optimizer = adamw(lr = F, weight_decay = F, beta1 = F, beta2 = F, eps = F, warmup_steps = N)` trains with AdamW and decoupled weight decay on the CPU, compiled into the training step's `Optimizer` subgraph with the internal ops `Sqrt` and `Div`, per-parameter moments versioned with the parameters, and a learning rate warming up linearly from 0 and decaying linearly to 0 at the last step; E-TYPE-023 and E-TYPE-024 refuse out-of-range arguments, and `sgd(lr = F)` keeps a constant rate (story:bert-adamw-warmup); E-TRAIN-001 also stops the run when a moment is not finite, and a resumed AdamW run is refused because the moments are not checkpointed yet.

## [0.11.0] - 2026-10-25

Masked-language-model training on the CPU interpreter: the weighted `cross_entropy` loss and `dropout`, `modelspec train --data <packed-dir>` feeding the batches that `modelspec data prepare`, `data tokenizer` and `data pack` write, and `--eval-every <n>` evaluating the held-out split during the run and streaming its loss and accuracy. `task ui-ux` puts the observatory UI iteration loop's hermetic gates and ratchet on every merge, and the documentation site, README and AGENTS.md describe the shipped compiler, with a status row for every capability since 0.8.0 and the test that holds it.

### Added

- `modelspec train --backend cpu --data <packed-dir> --eval-every <n>` evaluates the held-out split with dropout off at every multiple of `n` among the run's steps and at the run's last step, so a resumed run evaluates no step 0 of its own and starts at the first multiple of `n` at or after the step it resumes at, over every full batch of `B` sequences, the last `count mod B` never evaluated, prints `eval <k> loss <v> accuracy <a>` after the `step <k>` line and streams the values as the `StepCompleted` metric samples `heldout_mlm_loss` and `heldout_mlm_accuracy`; without a packed `--data`, with `n = 0` or on `--backend vulkan` it is a usage error; DESIGN.md revision 0.47 (story:bert-heldout-evaluation).
- `task ui-ux` (in `task ci`): the observatory UI loop's hermetic gates on every merge, Playwright specs under `ui/ux/` that replay the three recorded fixtures one event at a time at 1366x768 and 1280x720 in light and dark under one injected font stack (DejaVu Sans and DejaVu Sans Mono): no scroll, always-visible elements, unmissable completion (also on a run ending in `RunFailed`), axe-core accessibility, no animation under reduced motion after any event, and a screen-only user that reads loss, step, gradient norm and end state against each fixture's truth only from text it can see, with `time_to_act` measured on a live replay at one event every 40 ms; a ratchet that fails on any metric worse than `ui/ux/baseline.json` (`layout_shift.per_100_events` among them), on a tolerance edited in that file, and on a baseline made worse than `origin/main`'s without a decision id; a `self-test` project that plants one defect per gate; round R-00's report, contact sheets and findings are in `ui/ux/rounds/R-00/` and `ui/ux/findings/` (story:observatory-ui-iteration-loop).
- `dropout(x, p = <float literal>)` on the CPU interpreter, its mask drawn on Philox stream tag 4 from the seed, the call site and the step and recomputed by the internal `DropoutGrad` from the site's `i32[2]` RNG token; it applies in the training step only, so the final loss line, `--eval` and `infer` run without it; `p = 0.0` lowers to the identity, a `p` that is not a float literal is E-TYPE-019 and one outside [0, 1) the new E-TYPE-022, and `dropout` has no Vulkan kernel yet (E-TARGET-011).
- `modelspec train --backend cpu --data <dir>` on a packed directory (one holding `binding.json`) trains on masked-language-model batches: each `input` and `target` declaration names a batch tensor (`ids`, `positions`, `key_mask`, `labels`, `weights`), step `k` feeds the `Train` batch of step `k` from the run seed, a `data` line names the binding before the first `init` line, `RunStarted` carries `data_binding_id`, and E-DATA-005 and E-DATA-006 refuse a declaration that names no batch tensor or has another dtype or shape (story:bert-train-data-binding); the directory comes from `modelspec data prepare` (story:bert-corpus-splits), `modelspec data tokenizer` (story:bert-tokenizer) and `modelspec data pack` (story:bert-mlm-batches).
- `cross_entropy(logits: f32[N,V], labels: i32[N], weights: f32[N]) -> f32[]`, the weighted masked-language-model loss, every reduction in ascending order and the weighted sum fused after its first product, whose gradient is the internal `CrossEntropyGrad`, which recomputes the row softmax; checked against central differences in `f64` and `f32` for `matmul(x, w)` logits and the tied decoder `matmul(embedding(table, ids), transpose(table))`; a weight sum of 0 gives the loss 0 and a zero gradient; labels or weights of the wrong length are E-TYPE-003, an operand of the wrong rank E-TYPE-004, `f32` labels E-TYPE-018, and labels or weights whose value reads a parameter E-TYPE-021; a label outside the classes or a negative or non-finite weight is a runtime error naming the row before the op runs; no Vulkan kernel yet (E-TARGET-011); DESIGN.md revision 0.44 (story:bert-mlm-cross-entropy).

### Changed

- The documentation site, README and AGENTS.md describe 0.10.0 and `main`: the status page has a row for every capability since 0.8.0 with the test that holds it and planned rows for the BERT work, a new page on the BERT operators and a guide to `modelspec data prepare`, `data tokenizer` and `data pack` (shipped in 0.10.0, whose entry does not name them), the benchmark charts drop their white background, follow the dark theme and keep text at 12 px or more on a 390 px screen, the CLI reference wraps usage lines to its code box, the example pages title each model with its file, and the pitch deck follows the light theme; the `--serve` and `--ui` help of `train` and `observe` says "path / of ADDR" instead of a URL the site linked.

## [0.10.0] - 2026-10-25

Four BERT operators on the CPU interpreter (`embedding`, `add_rows`, `layer_norm` with `softmax`, `attention` with the first named compile-time argument), deterministic fused multiply-adds in every reduction, per-kernel pipeline statistics under `--profile`, device-profile scheduling, elementwise chain fusion, and a sweep harness that keeps the GPU busy between timed runs. At batch 8192 hidden 4096 the back-to-back run of that harness on the unit's tree gives modelspec 6,912.6 µs against PyTorch eager CUDA 6,333.4 µs per step (ratio 1.09; the 0.9.0 harness measured on 74a9520 gave 1.25). The 0.10.0 sweep of all 12 points is recorded in `benchmarks/runs.yaml`.

### Added

- `attention(q, k, v, key_mask, heads = A)`, multi-head scaled dot-product attention over sequences of `S` rows of `q, k, v: f32[N,H]` with a key mask `f32[B,S]`, `N = B * S`, whose gradients are the internal `AttentionGradQ`, `AttentionGradK` and `AttentionGradV`; `heads` is the first named compile-time operator argument, an integer literal, and E-TYPE-019, E-TYPE-020 and E-TYPE-021 refuse a `heads` that is not one, a `heads` that does not divide `H` and a key mask whose value reads a parameter by dataflow, through `let`s, model function bodies and model function arguments, and a named argument before an operand is E-TYPE-002; a mask value other than 0 or 1, or a sequence with every key masked, is a runtime error before the op runs, not NaN; no Vulkan kernel yet (E-TARGET-011); DESIGN.md revision 0.43 (story:bert-attention).
- `layer_norm(x: f32[N,H], gamma: f32[H], beta: f32[H])`, two passes over each row (the mean, then the biased variance), `eps = 1e-12`, whose gradients are the internal `LayerNormGradX`, `LayerNormGradGamma` and `SumRows`, and `softmax(x: f32[N,K])`, the row maximum subtracted before `exp`, whose gradient is the internal `SoftmaxGrad` of its result; every row reduction sums in ascending order; both checked against central differences in `f64` and `f32` on the CPU interpreter; a `gamma` or `beta` of the wrong length is E-TYPE-003 and an operand of the wrong rank E-TYPE-004; no Vulkan kernel yet (E-TARGET-011); DESIGN.md revision 0.41 (story:bert-layer-norm-softmax).
- `add_rows(x: f32[N,H], b: f32[H])`, the bias row of a dense layer, whose bias gradient is the internal `SumRows` (rows summed in ascending order from row 0), and `gelu`, the tanh form `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))` evaluated as the identical `x / (1 + exp(-2u))`, whose gradient is the internal `GeluGrad`; both checked against central differences in `f64` and `f32` on the CPU interpreter, `gelu` within `1e-6` relative of the form in `f64` at every multiple of `1/1024` in `[-3, 3]` and within `1.07e-6` over every `f32` in `[-3, -1]`; a `b` of the wrong length is E-TYPE-003, and an `x` not of rank 2 or a `b` not of rank 1 E-TYPE-004, naming that operand; no Vulkan kernel yet (E-TARGET-011) (story:bert-dense-gelu).
- `zeros()` and `ones()` parameter initializers, for a parameter of any rank from 1 (a dense layer's bias `b: Tensor<f32, [H]>`), beside `xavier_uniform()` at rank 2; they draw nothing, so no other parameter's initial bits change; E-TYPE-013 names the three initializers and E-TYPE-014 the rank each takes (story:bert-dense-gelu).
- `i32` row ids and `embedding(table: f32[V,H], ids: i32[N]) -> f32[N,H]`, whose table gradient is the internal scatter-add `EmbeddingGrad` (rows added in ascending `n`, a row never read exactly 0, ids without a gradient), checked against central differences in `f64` and `f32` for a token table and a positional one; an id outside the table is a `RuntimeError` of the CPU interpreter, any other use of `i32` is E-TYPE-018, and `train` refuses an `i32` input or target with E-DATA-004; `Embedding` and `EmbeddingGrad` have no Vulkan kernel yet, so `ir --stage kernel`, `compile --backend vulkan`, `train --backend vulkan` and `infer --backend vulkan` refuse an `embedding` program with E-TARGET-011 at the call, and `compile --backend cpu` writes its artifact; DESIGN.md revision 0.37 (story:bert-embedding-lookup)
- Changed: deterministic matmul, reduction and mse loss-sum fuse each later product into its sum (GLSL `Fma` on Vulkan, `mul_add` and `_mm256_fmadd_ps` on the CPU), per DESIGN.md § 12 ADR-009. Recorded bits (run records, goldens, same-bits fixtures) change once.
- Added: `train --backend vulkan --profile` prints, per kernel of the plan, the pipeline executable statistics the driver reports through `VK_KHR_pipeline_executable_properties` (`profile pipeline <kernel_module_id> executable <i> statistic "<name>" <value> description "<text>"`; on the RTX 3090 register count, binary size, stack, local and shared memory size), or `profile pipeline statistics: not reported` on a device without the extension, with the reason in parentheses when a query fails, which does not end the run.
- Changed: `compile --device-profile <json>` and `train --backend vulkan` schedule the plan for their device profile, so on a device that allows 256 invocations the large matmuls take the 128×128 block, which now reads workgroup memory in vectors (forward matmul of `mlp_sweep` b8192 h4096 2,600 against 2,854 µs, every output bit-equal); the manifest records the limits as `matmul_limits`, and a device below them is refused with E-ARTIFACT-005, as is a profile below the Vulkan minimum of 128 invocations or 16,384 bytes of workgroup memory at compile (named field and bound, never E-INTERNAL-002); the reader refuses a manifest recording limits below that minimum with E-ARTIFACT-003.

### Changed

- The generated JSON Schemas refuse more records the writers refuse: `ContentHash` admits only lowercase hexadecimal digits (`pattern`), and `seq_len` of `TrainingDataBinding` and `PackedDataset` and `PackedDataset.vocabulary_size` carry `minimum` 1 and 6 through the newtypes `modelspec.data.SequenceLength` and `modelspec.data.VocabularySize`; the 64-digit length of a `ContentHash`, distinct split roles and the `Binary64` bounds are still checked by the writers only, because ESS 0.68.0 does not project them (story:content-hash-schema-constrained).
- A chain of elementwise ops whose every value but the last is read once and by the next op alone runs as one dispatch, in the IR's left-to-right order with `NoContraction` on every op (dropped under `--fast-math`) and at most 3 buffers read: the loss gradient `(p - y) * c` is one kernel, and so is an adjoint's chain of `Add`s, with the `ReluGrad` after it when its reads fit in the 3. `examples/mnist_mlp.mspec` and `mlp_sweep` b8192 h4096 go from 11 to 10 dispatches per step, 10 to 9 barriers per step and 17 to 16 device allocations with the same loss and parameter bits, and `examples/linear_regression.mspec` from 6 to 5 dispatches and 7 to 6 barriers (DESIGN.md 0.42, story:fuse-elementwise-chains-and-gradient-sums).
- `mlp_sweep` runs a point's warm-up, calibration and timed runs as one series on one setup through the new `vulkan::runner::train_repeatedly`, and every Vulkan run reads its parameters and losses back from host-cached memory where the device has it (`Device::download_buffer`). Between two timed runs at batch 8192, hidden 4096 the device now idles 26 ms instead of about 150 ms, which had let the driver drop the SM clock to 780 to 1,155 MHz 0.1 to 0.3 s into every timed run. That point measures 7,084.6 µs per step against 7,830.6 µs on 74a9520 in the same session (PyTorch eager CUDA 6,420.2 µs), with `train --backend vulkan` output and checkpoints bit-equal (story:sweep-run-starts-busy).

## [0.9.0] - 2026-10-24

Three performance stories toward the batch-8192 goal, each verified bit-equal to the plan before it: a register-blocked matmul tile for large outputs (64×128 at 8×8 per invocation, 128 invocations), the opt-in `--fast-math` of ADR-008 (contraction only: the kernels drop `NoContraction`, the run is compared within tolerance, the deterministic figure stays the headline), and matmuls that read a transposed operand in place so the step dispatches no transpose. `mlp_sweep` on the RTX 3090 at batch 8192, hidden 4096: 32,810.6 µs per step in 0.7.1 → 12,008.4 µs deterministic and 9,220.4 µs under `--fast-math` (PyTorch eager CUDA 7,290.5 µs; ratio 4.50 → 1.65 and 1.27). At batch 1000, hidden 1024 ModelSpec runs 483.5 µs against PyTorch's 555.1. The 12 sweep points are re-measured from this release's clean build and recorded in `benchmarks/runs.yaml` (`sweep-b<B>-h<H>-modelspec-vulkan-0.9.0`). ADR-007's MNIST accuracy and final-weight conditions were not measured.

### Added

- `compile --fast-math` and `train --backend vulkan --fast-math` set `fast_math` in the manifest and every kernel cache key (ADR-008): the artifact hash changes, the reader refuses a key whose `fast_math` is not the manifest's with E-ARTIFACT-003, the kernels drop every `NoContraction` and nothing else changes in them, the run's numerics class is `tolerance`, and `--fast-math` with `--backend cpu` is a usage error. `mlp_sweep --fast-math` on the RTX 3090 (medians of 3 runs, µs per step, deterministic figure first): b8192 h4096 13,794.6 → 10,648.6 (PyTorch eager CUDA 7,290.5), b8192 h1024 3,751.8 → 2,747.6 (1,932.1), b1000 h4096 1,882.5 → 1,373.0 (994.7); forward matmul 9,569 → 12,170 GFLOP/s; the tolerance run on `examples/mnist_mlp.mspec` and the b8192 h4096 source stays within 0.2 % of ADR-008's bound and two `fast_math` runs are bit-equal (`benchmarks/raw/2026-10-24-fastmath/`); DESIGN.md revision 0.32 (story:fast-math-mode)
### Changed

- A matmul reads a transposed operand in place (`Kernel::transposed`, printed by `ir --stage kernel` and carried by the artifact entry): a transpose that only matmuls read, that is not the loss or the prediction, and whose operand is not a parameter updated before one of those matmuls, has no dispatch and no buffer, and each matmul stages its tile from the untransposed view with the strides swapped; every output element keeps its order, so the plan-site digests of the MNIST step plan and the sweep plans at b1000 h1024, b8192 h1024 and b8192 h4096 equal the record before the change. The MNIST step plan has 11 kernels (was 14). `mlp_sweep` on the RTX 3090 (medians of 3 runs, µs per step, deterministic first, `--fast-math` second, PyTorch eager CUDA in brackets): b8192 h4096 13,426.7 → 12,008.4 and 10,887.0 → 9,220.4 (7,290.5), b8192 h1024 3,701.2 → 3,380.2 and 2,772.2 → 2,520.1 (1,932.1), b1000 h1024 495.0 → 483.5 and 368.6 → 351.7 (555.1); the two weight-gradient sites read in place 2.4 % and 18 % slower than from a transposed copy while 2,203 µs of transposes per step are gone (`benchmarks/raw/2026-10-24-transposed/`); DESIGN.md revision 0.33 (story:matmul-reads-transposed-operand)

- A matmul whose output covers at least 128 blocks of 64 rows with `n` of at least 64 is register-blocked (`MatMulTile` `cols_per_invocation` and `b_pad`, `MatMulBlock`): each of 128 invocations accumulates an 8×8 (64×128 block) or 8×4 (64×64 block) tile of outputs from fragments staged through workgroup memory, `b` stored k-major, every element still accumulated by one invocation over ascending `k` from the `k = 0` product with `NoContraction` on every multiply and add, so the bits are those of the plan before (22 shapes and 34 edge shapes compared against recorded digests); smaller shapes keep the one-or-two-row tile, so the MNIST kernels change only in their printed tile text and module ids. `mlp_sweep` on the RTX 3090 (medians of 3 runs, µs per step): b8192 h4096 32,810.6 → 13,285.0 (PyTorch eager CUDA 7,290.5), b8192 h1024 6,034.5 → 3,684.7, b1000 h1024 743.5 → 493.8 (PyTorch 555.1); DESIGN.md revision 0.31 states the rule (story:matmul-shape-aware-tiling)

## [0.8.0] - 2026-10-24

Faster CPU steps and per-step device counters, from five performance stories of the 0.5.0 and 0.7.1 reviews: a CPU training run prepares its step once (verifier calls over 5 steps 5 -> 1, peak live values on the MNIST step 23 -> 10, allocations in the second step 170 -> 4), refills its dense batch tensors in place (104.125 allocations per runner step where there were 111.125) and writes matmul outputs without a zero fill first; the parser resolves a diagnostic's span by binary search from lexeme anchors (309,257 characters scanned on a 20,000-`let` one-line program where it scanned 9,087,562,188); and `train --backend vulkan --stats` and `--profile` print the step's pipeline barriers, device allocations and peak device bytes (7 barriers and 12 allocations per step on `examples/mnist_mlp.mspec`). The MNIST CPU bench measured 0.384 ms per step after the prepared step against 0.383 ms before, inside that host's noise. No new timed Vulkan step time was measured for this release: the latest MNIST figure is 0.6.0's 27.14 µs per step and the MLP sweep's figures are 0.7.1's, PyTorch eager CUDA up to 4.26 times faster at batch 8192; ADR-007's accuracy and final-weight conditions were not measured.

### Added

- `benchmarks/runs.yaml` holds every benchmark run measured so far (51: the MNIST 784-64-10 runs of ModelSpec CPU and Vulkan and of the PyTorch CPU and CUDA opponents, and the 36 points of the MLP sweep), typed by the ESS domain `modelspec.benchmark`; `modelspec-docs` checks it and generates the page Benchmarks (`website/docs/benchmarks.md`) with a chart per workload of µs per step over time, ModelSpec beside the opponents.

### Changed

- A CPU training run prepares its step once (`train::optimizer::PreparedStep`, `cpu::interpreter::Prepared`): the step program is verified once per run instead of once per step (5 verifier calls over 5 steps become 1), the `effects_in_order`, writes and in-place-transpose tables are built once, each intermediate is freed after its last consumer (peak live values on the MNIST step 23 of 25 nodes become 10), storage is reused across steps (170 allocations in the second step become 4, the outputs' shape vectors) and outputs are moved out instead of copied; `modelspec train --backend cpu` output on `examples/mnist_mlp.mspec` and `linear_regression.mspec` has the same sha256 as before, and a value listed twice among the step's outputs, as two parameters sharing one gradient are, keeps the loss at position 0 and both gradients; the MNIST CPU bench measured 0.383 ms per step before and 0.384 ms after at the median of 101 interleaved rounds, inside that host's noise, the AVX2 matmul kernel being 75–79 % of a step (`crates/modelspec-runtime/tests/prepared_step.rs`, `prepared_run.rs`) (story:cpu-prepared-training-step)
- The parser places a line-and-column anchor at offset 0, at every lexeme start and at the end of input in one pass, and `span_at` resolves a diagnostic's position from the nearest anchor by binary search instead of counting characters from the line start, so a span's cost no longer grows with its column: on a 20,000-`let` program written on one line the lookup scans 309,257 characters where it scanned 9,087,562,188, the same count as the one-`let`-per-line program, and `modelspec check` on that file takes 0.977 of the per-line file's time in release (`crates/modelspec-core/tests/span_lookup.rs`; `span_lookup_scan_count` is a `doc(hidden)` test hook) (story:parser-span-lookup-not-linear-in-column)
- `modelspec train --backend vulkan --stats` and `--profile` print three per-step counters from a per-thread ledger in `modelspec_runtime::vulkan`: pipeline barriers recorded in the step, device allocations, and the peak device bytes live at once over the run (a buffer freed inside the span counts once, one freed after it is excluded); on `examples/mnist_mlp.mspec` a step records 7 barriers and 12 allocations, `--stats` and `--profile` print the same values in one run, and checkpoints and `--resume` leave the per-step figures unchanged; DESIGN.md § 8 (revision 0.30) lists the lines in the order the code prints them (story:vulkan-step-counters)
- A dense-data training step refills its `x` and `y` batch tensors in place through `HostTensor::refill`, which keeps the shape when it is equal and clears and refills the data, instead of allocating a new key, shape and data per step: a runner step on dense data makes 104.125 allocations where it made 111.125, the same as a synthetic step, and the per-step losses of 1,800 MNIST steps are bit-identical (`crates/modelspec-runtime/tests/dense_batch_reuse.rs`) (story:dense-step-reuses-batch-buffers)
- The CPU `matmul` writes its output without filling it with zeros first: the portable kernel appends rows to the output `Vec`, and the AVX2 f32 and f64 kernels write into reserved spare capacity and set the length to the rows they stored, so no memset of `m·n` elements precedes the kernel; a `k` of 0 still zero-fills, because those zeros are the result, and an element type without vector code in the module runs its own `matmul_avx2` hook on a zero-filled output as before, counted by the `doc(hidden)` test instrument `output_zero_fills` (`crates/modelspec-runtime/tests/matmul_output_fill.rs`, `matmul_output_zero_fills.rs`) (story:cpu-matmul-output-skips-zero-fill)

## [0.7.1] - 2026-10-23

Throughput you can measure, and a Vulkan descriptor pool sized right: the bench `mlp_sweep` trains 784 -> H ReLU -> 10 at 12 sizes on one Vulkan device beside a PyTorch CUDA opponent, every Dispatch step of a plan has a FLOP and a byte count that `ir --stage kernel` and `train --backend vulkan --profile` print, and the descriptor pool counts every view a dispatch reads and writes. On an RTX 3090, `train --backend vulkan --profile` on `examples/mnist_mlp.mspec` reached 447.8 GFLOP/s and 56.6 GB/s over 45.925 µs per profiled step, with 0 uncounted sites; the sweep measured 36 of 36 points, ModelSpec on Vulkan faster than PyTorch eager CUDA at every batch-100 point (28.1 µs against 537.5 µs at hidden 64) and PyTorch eager up to 4.26 times faster at batch 8192 (7290.5 µs against 31067.1 µs at hidden 4096). No new timed MNIST step time was measured for this release; the latest is 0.6.0's 27.14 µs per step, and ADR-007's accuracy and final-weight conditions were not measured.

### Added

- The bench `mlp_sweep` (`cargo bench -p modelspec-runtime --features vulkan --bench mlp_sweep`) trains 784 -> H ReLU -> 10 at hidden 64, 256, 1024, 4096 and batch 100, 1000, 8192 on one Vulkan device, each source formatted and compiled in Rust, and prints one JSON line per point: the matmul-only FLOP per step `3196·B·H`, the µs per step of 3 timed runs of at least 5 s after a 20-step warm-up, their median and GFLOP/s, and the version, git commit and dirty state it was built from, or the refusal of a point that does not compile or run, the teacher-student batch made once per point and handed to every run through the new `RunOptions::data`; `--crossover <file>...` reads its lines and those of the PyTorch CUDA opponent `opponents/mlp_sweep/mlp_sweep.py` (eager and `torch.compile`, TF32 off) and names per side and batch the first width more than 10 % slower than the previous, and `tests/mlp_sweep.rs` checks that all 12 sources compile for the RTX 3090 profile and that their step plans' matmuls count `3196·B·H` (story:mlp-size-sweep-vs-pytorch-cuda)
- Every Dispatch step of an execution plan has a FLOP and a byte count by the counting rule of DESIGN.md § 3.1 (`ExecutionPlan::dispatch_count`, `DispatchCount`, `Kernel::flop`): `2·m·k·n` for a matmul, the arithmetic per output element for an elementwise kernel and an SGD update, one per input element for a reduction or a loss sum, none for a transpose or a broadcast, and the bytes of the views the step reads and writes; a kernel without a rule is `uncounted`, never 0. The counts are computed when the plan is read, so no artifact field, hash or module id changes. `modelspec ir --stage kernel` prints `flop <n> bytes <n>` or `uncounted` on each dispatch line, and `modelspec train --backend vulkan --profile` adds `flop`, `bytes`, `gflop_per_s` and `gb_per_s` to each `profile dispatch` line and ends with `profile step_total`, the step's counted FLOP and bytes over the steady time per step and `uncounted_sites`, the step sites it leaves out (story:profile-achieved-throughput)

### Fixed

- `modelspec train --backend vulkan` and `infer --backend vulkan` size the descriptor pool of `modelspec_runtime::vulkan::runner` at every view each dispatch reads and writes, where it counted one write per dispatch: `examples/mnist_mlp.mspec` binds 39 storage-buffer descriptors from a pool of 39 instead of 38, which a driver that accounts the pool exactly could refuse with `VK_ERROR_OUT_OF_POOL_MEMORY` since 0.4.0; `DescriptorArena::bind`, `Recording::dispatch` and `Recording::dispatch_kernel` admit each descriptor set into the room their pool was created from and refuse one past it with `ExecError::DescriptorPoolFull` before the pool is asked, on every driver (story:vulkan-descriptor-pool-counts-writes)

## [0.7.0] - 2026-10-23

The observatory UI, and Vulkan runs and artifacts that refuse what they cannot trust: `--ui <dir>` on `train --serve` and `observe` serves a Vue app that draws the loss and every metric per step, a CPU run's events carry per-parameter norms and histograms of values and gradients, `train --backend vulkan` stops a diverged run with the CPU run's `E-TRAIN-001`, `artifact::read` refuses a plan past 4,096 steps, 65,536 dependencies or 65,536 views with `E-ARTIFACT-003` and verifies the rest in bounded time, and a matmul's kernel module id covers its `MatMulTile`. No new device benchmark was measured for this release; the latest measurement is 0.6.0's 27.14 µs per step on an RTX 3090.

### Added

- The observatory UI in `ui/` (Vue 3, TypeScript, Vite, pnpm; `task ui` builds `ui/dist`) draws the loss and every `MetricSample` per step from the event stream, and `--ui <dir>` on `modelspec train --serve` and `modelspec observe` serves it at `/` beside `/events` (`modelspec_observatory::Ui`; `spec/domains/observatory.yaml`, `UiDirectory`), refusing a directory over 4096 files or 64 MiB without reading more than 64 MiB of it; a page left open after the server is stopped mid-run connects again every second and shows the run of whatever server then answers at that address; its TypeScript types are generated from the Rust `RunEvent` types by `modelspec-ui-types` and drift-checked in `task ci` (story:observatory-ui-live-loss-chart)
- `modelspec train --backend cpu` fills `StepCompleted.parameters` at the first step, every `--statistics-every N` steps (default 100) and the final step: per parameter the L2 norm and a 32-bin histogram of its values and of its gradient, with NaN and ±inf counted beside the bins (`Histogram.nan_count`, `positive_infinity_count`, `negative_infinity_count`, new required fields), the final step and a parameter the loss does not read having no `Gradient` entry; `sgd_step` returns the step's gradients, and the UI draws a value-norm and a gradient-norm chart per parameter, a metric named `<path> value norm` or `<path> gradient norm` getting ` (metric)` as `loss` does, and the histograms of the latest or a picked sampled step; `--backend vulkan` fills the first and the final step with `Value` entries only, from the values already on the host, sends every other list empty and refuses `--statistics-every` as a usage error (story:observatory-parameter-statistics)

### Fixed

- `modelspec train --backend vulkan` checks every loss and the parameters it reads back, after the last step and before each `--checkpoint-every` checkpoint, and stops a diverged run with the CPU run's `E-TRAIN-001` diagnostic, stdout and `--serve` events instead of printing NaN parameters and exiting 0, with no host wait of its own; a parameter that goes non-finite between read-backs while its step's loss is finite is named at the first later step that shows it, so `sgd(lr = 1e38)` names the loss of step 1 where the CPU names the update of step 0, and `--stats` and `--profile` print the counters of the steps the device ran after the diagnostic (DESIGN.md § 8) (story:train-vulkan-refuses-non-finite-loss)
- `modelspec_core::artifact::read` refuses a plan of more than 4,096 steps (`MAX_PLAN_STEPS`), 65,536 dependencies (`MAX_PLAN_DEPENDENCIES`) or 65,536 views (`MAX_PLAN_VIEWS`) with `E-ARTIFACT-003` before its other invariant checks, and the plan verifier keeps each pass's state and the step orderings as bit sets instead of searching every earlier step, with the same result on every plan; a re-sealed artifact with a 300,000-step dependency chain is refused in 1.0 s to 1.2 s and at most 64 MiB in a debug build, where it ran past 10 s before, and a 4,096-step plan with 16 dependencies per step verifies in 0.09 s to 0.15 s instead of more than 10 s (DESIGN.md § 7.1, § 8.1) (story:plan-verifier-bounded-on-untrusted-artifact)
- `modelspec_core::artifact::read` counts the steps, dependencies and views of each plan as it parses `artifact.json` and refuses a plan past a limit there with `E-ARTIFACT-003`, without holding it and before the module files and hashes are checked, so an unsealed artifact with such a plan gets `E-ARTIFACT-003` instead of `E-ARTIFACT-002`; a sealed 64 MiB manifest of 33.5 million dependencies is refused in 2.4 s to 4.5 s and at most 71 MiB in a debug build, where it passed 1 GiB before; `artifact_hash` is computed as the manifest is serialized, without a copy of it; a manifest with a repeated key is refused with `E-ARTIFACT-001`; and the plan verifier names an unsynced view by looking the earlier step's buffers up in sets, so a pair of Copy steps at the view limit verifies in 0.02 s to 0.05 s instead of 3.4 s (DESIGN.md § 7.1, § 8.1) (story:plan-verifier-bounded-on-untrusted-artifact)
- A matmul kernel's `Kernel::signature`, and so its module id, ends with its `MatMulTile` (`tile workgroup_size … cols … rows … depth … ahead … buffers …`), so two matmuls that differ only in tile depth or ahead no longer print the same id in `modelspec ir --stage kernel` and `train --profile`; every matmul id changes, the SPIR-V and the bits do not (story:kernel-signature-covers-matmul-tile)

## [0.6.0] - 2026-10-23

Checkpoints, inference and artifacts on Vulkan: `train --backend vulkan` writes and resumes checkpoints, `infer --backend vulkan` runs a checkpoint on the device, `compile` writes a sealed artifact directory, and `train --record` with `observe` records and replays a run. The MNIST benchmark example measured 27.14 µs per step on an RTX 3090 (median of 5 timed runs of 144,000 steps), down from 27.282 µs in 0.5.0. ADR-007's accuracy and final-weight conditions were not measured for this release.

### Added

- `modelspec train --backend vulkan` takes `--checkpoint`, `--checkpoint-every` and `--resume`: each requested checkpoint ends a segment of the run with one host wait and a parameter read-back, so 1800 MNIST steps with `--checkpoint-every 600` wait 3 times and write 2 extra files, and a run resumed at step 900 passes against the uninterrupted one within the GPU tolerance; `modelspec infer --backend vulkan [--device <index>]` runs a checkpoint's forward program and loss on the device with one host wait, refusing an unusable checkpoint with the CPU reader's `E-CKPT` code before the loader opens (DESIGN.md § 6.2, § 8) (story:m1-vulkan-checkpoint-infer)
- `modelspec compile <file> --backend cpu|vulkan --output <dir> [--device-profile <json>]` writes an artifact directory, `artifact.json` and for Vulkan one `kernels/<kernel_module_id>.spv` per kernel, sealed by a blake3 `artifact_hash`, byte-equal across two compiles, replacing the `artifact.json` and `kernels/` of an earlier artifact in that directory; `modelspec_core::artifact::read` reads one back and refuses an altered one with `E-ARTIFACT-001` (format version), `E-ARTIFACT-004` (module file missing, not a regular file or too long, or a stray file in `kernels/`), `E-ARTIFACT-002` (hash mismatch) or `E-ARTIFACT-003` (broken invariant or plan, a buffer over 4294967295 bytes, a module that is not SPIR-V), in that order (DESIGN.md § 7.1, § 8) (story:m1-compile-artifact)
- `modelspec train --record <path>` writes the run's events to a recorded run file, one JSON `RunEvent` per line in `sequence` order (`modelspec_observatory::Recorder`), and `modelspec observe <path> [--serve <addr>]` replays it as the `text/event-stream` of `--serve`, the same frames and `id:` lines, refusing a truncated file, a `sequence` gap or two `training_run_id` values, an order `RunEventBody` forbids (a first event other than `RunStarted`, a second `RunStarted`, an event after `RunFinished` or `RunFailed`, a missing end) on one line with exit 1 before serving (`spec/domains/observatory.yaml`, `RecordedRunFile`) (story:observatory-run-recording-replay)

### Changed

- A two-row `MatMulTile` reads tiles of up to 16 values of `k` 2 tiles ahead (`modelspec_core::ir::kernel::MATMUL_TWO_ROW_TILE`) instead of 32 read 6 ahead, with the same bits; the weight-gradient matmul of the MNIST example falls from 4.91 to 4.58 µs and the MNIST step from 27.50 to 27.14 µs on an RTX 3090 (medians of 3 profiled and 5 timed runs of 144,000 steps) (DESIGN.md § 5.2) (story:m1-f-weight-grad-matmul-tiled)

### Fixed

- `modelspec train --backend cpu` stops at the first step whose loss or updated parameter is NaN or ±inf and exits 1 with `E-TRAIN-001` naming the step and the loss or parameter, instead of printing a diverged run and exiting 0; stdout ends with that step's `step <k> loss <v>` line and has no `param` lines, and `--serve` ends with that step's event (its loss `null` when the loss is the non-finite value), then `RunFailed` (DESIGN.md § 8) (story:train-refuses-non-finite-loss)
- A checkpoint is written to `<path>.tmp`, synced and renamed over `<path>`, so a write that fails part-way (a full disk, a file-size limit, a kill) keeps the checkpoint it would replace, including the `--resume` source when `--checkpoint` names the same file.

## [0.5.0] - 2026-10-23

Faster Vulkan training: the `MatMul` kernel stages its operand tiles in workgroup memory, and the MNIST benchmark example measured 27.282 µs per step on an RTX 3090 (median of 5 runs of 144,000 steps, mean P0 clock 1957 to 1965 MHz), down from 40.225 µs in 0.4.0. ADR-007's accuracy and final-weight conditions were not measured for this release.

### Changed

- The Vulkan `MatMul` kernel stages tiles of its operands in workgroup memory (`modelspec_core::ir::kernel::MatMulTile`, 128 invocations per block of 8 rows by 16 columns, or 16 rows at two per invocation when 8 need at least 256 workgroups) with the same bits as before, its tiles of `k` in a loop of 4-tile passes so a module's size does not grow with `k`; the forward matmul of the MNIST example falls from 18.5 to 8.3 µs per step and the MNIST step from 40.2 to 27.3 µs on an RTX 3090 (medians of 3 profiled and 5 timed runs of 144,000 steps), and a tile larger than the device's workgroup memory is refused with `EmitError::SharedMemory` (DESIGN.md § 5.2) (story:m1-f-forward-matmul-tiled)

## [0.4.0] - 2026-10-22

Training on a Vulkan GPU: `train --backend vulkan` runs a whole training run on one device with one host wait. The MNIST benchmark example measured 40.225 µs per step on an RTX 3090 (median of 5 runs of 144,000 steps, mean P0 clock 1935 to 1951 MHz). ADR-007's accuracy and final-weight conditions were not measured for this release.

### Added

- `examples/mnist_mlp.mspec`: the MNIST benchmark shapes (784 → 64 ReLU → 10, no bias, SGD lr 0.1, batch 100, 1,800 steps) on M0 synthetic data, checked by `task check-examples`, gradient-checked at sampled coordinates and with a training-IR golden file (story:m1-mlp-example)
- `modelspec inspect-device --backend vulkan [--device N] [--format text|json]`: the `modelspec_core::device::DeviceProfile` of each Vulkan device as the driver reports it and its negotiation verdict (Vulkan 1.2, numerics class), with codes E-TARGET-002 to E-TARGET-004 (story:m1-inspect-device)
- `modelspec ir --stage kernel` prints the Kernel IR of a training step: its buffer views, kernels with module ids, step plan and forward plan, built by `modelspec_core::schedule::schedule`, checked by `modelspec_core::ir::verify::verify_plan` (and `verify_recorded_back_to_back` for back-to-back recording) and wrapped by `compile::compile_artifact`; a tensor too large to dispatch or bind is refused with `E-TARGET-010` (story:m1-kernel-plan)
- `modelspec train --data <dir>`, `--init <dir>` and `--eval <dir>` train on dense `f32` tensor files with an optional `order.u32`, start parameters from files, and print `eval loss=… accuracy=…`; codes `E-DATA-007`, `E-DATA-008`, `E-DATA-009` (story:train-data-dense-tensors)
- `modelspec_runtime::spirv::emit` builds the SPIR-V module of each elementwise, broadcast and SGD kernel, checked by `spirv-val --target-env vulkan1.2` in `task ci`, and `modelspec_runtime::vulkan` gains device buffers, compute pipelines, one-command-buffer dispatch with a counted host wait, and `MODELSPEC_VK_VALIDATION=1` for the validation layer; GPU results must lie within `GPU_ATOL = 1e-5` plus `GPU_RTOL = 1e-4` times `|c|` (DESIGN.md § 9 gate 5) (story:m1-vulkan-elementwise)
- `modelspec_runtime::spirv::emit::emit_kernels` builds a `KernelModule` with entry point, SPIR-V and `KernelCacheKey` for every kernel of an artifact, now including the two-stage `ReduceSum`/`ReduceMean`, the `mse` loss sum, `Transpose` and `MatMul` in orders fixed by the shape alone, with `NoContraction` and, on a device that reports them, the RTE, denorm-preserve and signed-zero-preserve modes on loss sum and matmul; `Recording::dispatch_kernel` and `Recording::reduce` record them (DESIGN.md § 3.3, § 5.2, § 5.4, § 9 gate 5) (story:m1-vulkan-reduce-matmul)
- `modelspec train --steps`, `--checkpoint`, `--checkpoint-every` and `--resume` write and resume `.msckpt` checkpoints (`modelspec_runtime::train::checkpoint`, a 56-byte header, a JSON manifest and 64-byte aligned tensors under a payload hash), a resumed CPU run writing the same bytes as one that never stopped, and `modelspec infer --backend cpu --checkpoint` runs the forward program with a checkpoint's parameters; codes E-CKPT-001 to E-CKPT-004 (story:m1-checkpoint-cpu)
- `modelspec train --backend vulkan [--device N] [--stats]` trains on one Vulkan device with one host wait per run (`modelspec_runtime::vulkan::runner::train`: chained command buffers of `CHUNK_STEPS` = 256 steps, one upload, one read-back) and prints the lines of the CPU run within the gate 5 tolerance; `--stats` prints the `RunStats` counters on stderr, `train` reports E-TARGET-002 to E-TARGET-004, and only a build without the `vulkan` feature refuses the backend with E-TARGET-001 (story:m1-vulkan-train)
- `modelspec train --backend vulkan --profile` prints on stderr one `profile dispatch <plan> <position> <kernel_module_id> <entry> calls <n> total_us <t> mean_us <m>` line per dispatch site from timestamp queries read back with no extra host wait, then the run counters, pipeline creation and compile time and the steady-state µs per step after a 20-step warm-up (`modelspec_runtime::vulkan::runner::train_profiled`, `modelspec_runtime::vulkan::profile`); the guide `inspect-a-run` shows the IR, plan and timing paths (story:m1-vulkan-profile)

### Changed

- The `mse` loss sum kernel of `modelspec_runtime::spirv::emit` runs one workgroup of 64 invocations that stage the terms in workgroup memory, in chunks of `max_shared_memory_bytes / 4`, and invocation 0 adds them in the same row-major order, so the loss sum and its bits are unchanged; `KernelKind::LossSum.workgroup_size()` is now 64 and its kernel module id changes (story:m1-loss-sum-staged)
- `modelspec_core::schedule::schedule` fuses a `relu` of a `matmul` output into the matmul kernel (`Epilogue::Relu`, writing the pre-activation too when the `relu` gradient reads it), runs each SGD update `p - lr * g` as one `sgd_update` kernel that reads the gradient, and folds a `mul`, `add` or `sub` of two compile-time scalars into its reader's immediate (it stays a kernel when a reader takes no immediate), with the same bits: the MNIST step is 14 dispatches in 8 barrier groups instead of 18 in 11 (story:m1-fuse-elementwise)

## [0.3.0] - 2026-10-22

An AVX2 CPU `matmul`, 2.98× faster on the MNIST shape with the same bits.

### Changed

- CPU reference: `matmul` runs an AVX2 kernel with register blocks where the CPU has AVX2 (`modelspec_runtime::cpu::matmul::Kernel`), and `evaluate` reads inputs, parameters and a transposed left `matmul` operand in place, with the same bits as 0.1.1 (`cpu-matmul-avx2-blocking`)

## [0.2.0] - 2026-10-22

Live training events, input adjoints and a 15× faster CPU `matmul` with the same bits.

### Added

- `modelspec train --serve <addr>` streams the run as typed `RunEvent`s over Server-Sent Events at `http://<addr>/events` on a loopback address, in the new crate `modelspec-observatory` (story:observatory-live-event-stream)
- `modelspec_core::autodiff::differentiate_inputs` builds the gradient of a rank-0 value with respect to named inputs, and `output_element_target` builds the rank-0 target for one element of a model output (story:observatory-input-adjoint)

### Changed

- CPU reference: `matmul` runs in loop order `i`, `k`, `j` with the same bits as 0.1.1, and the `mnist_shape` benchmark (`cargo bench -p modelspec-runtime --bench mnist_shape`) prints images/s for the 784 → 64 → 10 shape (`cpu-matmul-row-order`)

## [0.1.1] - 2026-10-21

Documentation release: the website review's fixes and highlighted `modelspec` code blocks. The
compiler and runtime behave as in 0.1.0.

### Added

- Documentation site: `modelspec` code blocks are syntax-highlighted; the word lists are `website/data/modelspec.grammar.json`, generated from `syntax::lexer::KEYWORDS`, `resolve::OPERATORS` and the parser

### Changed

- Diagnostics reference: every code has its own anchor, such as `reference/diagnostics#e-type-012`
- CLI reference: an argument table without optional arguments drops the Required and Default columns, and each possible value of `--stage` and `--backend` has its own line
- Operators reference: the message of a refused call wraps between its parts
- `website/data/status.json` and the Status page: each shipped capability names the test or task that holds it
- Getting started: the install block uses `install -D` and says `~/.local/bin` must be on `PATH`; the binary links `libc`, `libm` and `libgcc_s`
- Roadmap: planned milestones carry the `planned` badge, and the exclusions list all of DESIGN.md § 1 and § 2.4
- Write a model: step 2 says where `forward` goes, and step 4 shows the `E-TYPE-012` output
- The language: `infer` is refused with `E-SYNTAX-002`; a top-level `target` is later, `target y:` in `train` is M0
- Pitch deck `docs/pitch/`: slides 13 and 15 show the 0.1.0 release, slide 8's loop arrows meet their boxes, tap, swipe and Escape work, and a portrait phone gets a notice

## [0.1.0] - 2026-10-21

The M0 semantic core (`epic:m0-semantic-core`, DESIGN.md revision 0.6). There is no GPU or Vulkan
backend yet: Vulkan execution is milestone M1, and `modelspec train --backend vulkan` is refused
with `E-TARGET-001`.

### Added

- A Cargo workspace of `modelspec-core`, `modelspec-runtime` and `modelspec-cli` (story:m0-workspace-foundation)
- `modelspec-core` has no Vulkan dependency under any feature or target; `task core-no-vulkan` holds it
- `modelspec_core::syntax::parse` and `syntax::print`: lexer and parser for the M0 subset (story:m0-parse-subset)
- Syntax errors `E-SYNTAX-001` to `E-SYNTAX-021`, each with line and column
- `modelspec_core::types::check_source`: name resolution, type and shape checking into `hir::Hir` (story:m0-check-types-shapes)
- Name errors `E-NAME-001` to `E-NAME-008` and type and shape errors `E-TYPE-001` to `E-TYPE-017`
- `modelspec_core::ir::tensor::TensorProgram` and `TensorProgramBuilder`: the tensor SSA IR (story:m0-tensor-ir-core)
- `TensorOp` registry: `Add`, `Sub`, `Mul`, `MatMul`, `Relu`, `Transpose`, `ReduceSum`, `ReduceMean`, `Mse`
- `ir::verify::verify` refuses a use before definition, an undeclared `ParamId` and a shape mismatch
- `ir::print::print`: a deterministic text form of the Tensor IR
- `modelspec_core::lower::lower`: Typed HIR to Tensor IR (story:m0-lower-tensor-ir)
- `modelspec_core::passes::dce::eliminate`: dead-code elimination over the Tensor IR
- `modelspec_core::autodiff::differentiate`: reverse-mode autodiff into a `TrainingProgram` (story:m0-reverse-mode-autodiff)
- `ir::training::attach_sgd` and `ir::verify::verify_step`: the verified SGD training step
- `modelspec_core::compile::compile_training`: source to a verified `CompiledTraining` in one call
- `modelspec_runtime::cpu::evaluate`: the CPU reference interpreter over Tensor IR, in `f32` and `f64` (story:m0-cpu-interpreter)
- `modelspec_runtime::train::runner::run`: seeded SGD training on the CPU interpreter (story:m0-cpu-sgd-training)
- `train::init`: Philox4x32-10 parameter initialization keyed by seed and canonical parameter path (DESIGN.md § 2.3)
- `train::data::synthetic`: M0 synthetic teacher data; data errors `E-DATA-001` to `E-DATA-003`
- `modelspec check <file>`: parse and check; diagnostics on stderr, exit 1
- `modelspec ir <file> --stage tensor|training`: print the verified Tensor IR or training step
- `modelspec train <file> --backend cpu`: train on the CPU reference, printing initial params, losses and final params
- CLI exit codes: 0 success, 1 diagnostics or runtime error, 2 usage error
- `examples/linear_regression.mspec` checks and trains; `task check-examples` runs `modelspec check` on it
- ESS contracts for the compiled artifact, checkpoint, device profile and training run in `spec/`

[Unreleased]: https://github.com/beyond10x/modelspec/compare/0.21.0...HEAD
[0.21.0]: https://github.com/beyond10x/modelspec/compare/0.20.0...0.21.0
[0.20.0]: https://github.com/beyond10x/modelspec/compare/0.19.0...0.20.0
[0.19.0]: https://github.com/beyond10x/modelspec/compare/0.18.0...0.19.0
[0.18.0]: https://github.com/beyond10x/modelspec/compare/0.17.0...0.18.0
[0.17.0]: https://github.com/beyond10x/modelspec/compare/0.16.0...0.17.0
[0.16.0]: https://github.com/beyond10x/modelspec/compare/0.15.0...0.16.0
[0.15.0]: https://github.com/beyond10x/modelspec/compare/0.14.0...0.15.0
[0.14.0]: https://github.com/beyond10x/modelspec/compare/0.13.0...0.14.0
[0.13.0]: https://github.com/beyond10x/modelspec/compare/0.12.0...0.13.0
[0.12.0]: https://github.com/beyond10x/modelspec/compare/0.11.0...0.12.0
[0.11.0]: https://github.com/beyond10x/modelspec/compare/0.10.0...0.11.0
[0.10.0]: https://github.com/beyond10x/modelspec/compare/0.9.0...0.10.0
[0.9.0]: https://github.com/beyond10x/modelspec/compare/0.8.0...0.9.0
[0.8.0]: https://github.com/beyond10x/modelspec/compare/0.7.1...0.8.0
[0.7.1]: https://github.com/beyond10x/modelspec/compare/0.7.0...0.7.1
[0.7.0]: https://github.com/beyond10x/modelspec/compare/0.6.0...0.7.0
[0.6.0]: https://github.com/beyond10x/modelspec/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/beyond10x/modelspec/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/beyond10x/modelspec/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/beyond10x/modelspec/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/beyond10x/modelspec/compare/0.1.1...0.2.0
[0.1.1]: https://github.com/beyond10x/modelspec/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/beyond10x/modelspec/releases/tag/0.1.0
