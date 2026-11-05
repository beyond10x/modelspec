---
format: aep.planning-md/3
id: story:vulkan-adamw-one-dispatch-per-parameter
kind: story
status: implemented
title: One AdamW dispatch per parameter on Vulkan
summary: The schedule gives the sixteen ops of one parameter's AdamW update one sgd_update AdamW kernel that writes the parameter and both moments in place, with the bits of the sixteen-op lowering.
relations:
- decomposes: epic:bert-tiny-beats-torch-compile
- serves: vision:modelspec
revision: 5
transitions:
- {from: "draft", to: "proposed", at: "2026-11-05T08:59:57Z", actor: "human:timo", revision: 3, executor: "agent:modelspec-a4fb0b"}
- {from: "proposed", to: "active", at: "2026-11-05T08:59:57Z", actor: "human:timo", revision: 4, executor: "agent:modelspec-a4fb0b"}
- {from: "active", to: "implemented", at: "2026-11-05T15:49:21Z", actor: "human:timo", revision: 5, decided_on: {"recorded":{"test_result":1}}}
---
# One AdamW dispatch per parameter on Vulkan

## Context

Profile 2026-11-05 at 315a15ef plus story:vulkan-attention-blocked-key-tiles, `train
examples/bert_tiny.mspec` at 200 steps over packed-128, `--backend vulkan --profile` on the RTX
3090: steady 4,966 to 5,026 µs per step, 602 dispatches and 421 barriers per step. The AdamW update
of `ir/training.rs::adamw_update` is sixteen ops per parameter, which the schedule ran as eleven
dispatches (the chains `Mul>Add@0`, `Mul>Mul@0` and `Mul>Add@1>Mul@1`, `Sqrt`, `Div` and the
`sgd_update` `Sub`), each behind a barrier, for 41 parameters: 451 of the step's 602 dispatches.
Removing those dispatches from the command buffer took 330 µs off the step; a step with no
dispatch at all still takes 334 µs for its barriers and timestamps.

## Outcome

The schedule recognises the sixteen ops of one parameter's update (`schedule::adamw_update`) and
gives them one `SgdUpdate` kernel whose op is `ir::kernel::ADAMW_OP`. It reads `p`, `g`, `m`, `v`
and the rank-0 `lr_t`, `c1` and `c2`, takes `beta1`, `1 - beta1`, `beta2`, `1 - beta2`, `eps` and
the weight decay as immediates, and writes three bindings in place: the update into the
parameter's view and the next moments into the views `m` and `v` are read from
(`Kernel::written` is 3). The emitter (`Shell::adamw`) computes the sixteen ops in the rule's
order, each `NoContraction` unless the module is `fast_math`, the root and the quotient by the
instructions of the `Sqrt` and `Div` kernels, so every result has the bits of the sixteen-op
lowering. The thirteen values between have no buffer, and no barrier stands between the updates of
different parameters. An update whose ops are not the rule's, or one of whose inner values has a
second reader or is the loss or the prediction, keeps the dispatches it had. DESIGN.md 0.83
(§ 5.2, § 5.3) states the kernel.

## Acceptance

- `cargo test -p modelspec-core`, `cargo test -p modelspec-runtime --features vulkan` and `cargo
  test -p modelspec-cli` pass; the AdamW module passes `spirv-val` on aligned and unaligned shapes
  and is `NoContraction` except under `fast_math`.
- The plan of `examples/bert_tiny.mspec` holds one `sgd_update AdamW` dispatch per parameter (41),
  no `Sqrt` and no `Div` kernel, and no barrier between two updates.
- `train examples/bert_tiny.mspec --backend vulkan --data <packed-128>` at 200 steps prints the
  losses and parameters it printed before the change, bit for bit.
- `MODELSPEC_VK_VALIDATION=1` device tests `vulkan_bert_attention_loss_adamw` and
  `adversary_gpu_step` with `--ignored`: within tolerance of the CPU, no validation message.
- The 200-step profile above, re-run in one binary with the fusion on and off: at least 300 µs
  less per step with it on.

## Result

Measured 2026-11-05 on the RTX 3090, `train examples/bert_tiny.mspec` at 200 steps over packed-128:

- Without `--profile`, the losses of steps 0, 50, 100, 150 and 200 are 9.035598, 8.717873,
  8.329162, 8.119945 and 8.04707 and the digest of the `param` lines is `a4180f2c012d`, both those
  of the tree before the change; at 2,000 steps the last loss is 7.0493593 with the fusion on and
  off.
- `--stats`: 192 dispatches and 93 barriers per step, against 602 and 421.
- One binary with the fusion switched by a temporary environment variable (removed before the
  commit), three profiled runs each, alternating: off 5,026.4, 4,965.4 and 4,973.2 µs per step,
  on 4,416.3, 4,366.6 and 4,424.8; medians 4,973.2 and 4,416.3, 557 µs less. Host load average 9.7
  to 10.1 during the runs.
- Device tests with `MODELSPEC_VK_VALIDATION=1` and `--ignored`, release: `adversary_gpu_step` (5)
  and `vulkan_bert_attention_loss_adamw` (2) of `modelspec-runtime` and `adversary_gpu_step_cli`
  (9) of `modelspec-cli` pass with no validation message.
- The recorded-bits tests `vulkan_bert_tiny_same_bits`, `bert_tiny_vulkan_bits` and
  `bert_tiny_masked_rows` failed on the tree and were recorded again. The cause is
  story:vulkan-cross-entropy-row-workgroup (tolerance class), committed without re-recording
  them: on the same day and driver (2580660800) the 0.25.0 release binary gives the former
  recording on both tests' data, and this tree prints the same bytes with the fusion on and off.
- The same story moved the twin's key-bias drift (`adversary_gpu_step_cli`, 10 steps, seed 1) from
  0.85 of the element bound on 0.25.0 to 1.23, with the fusion on and off alike; the two drift
  tests now hold a key bias to the rule DESIGN.md § 9 states for it (coordinator decision D74).
