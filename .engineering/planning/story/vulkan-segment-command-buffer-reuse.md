---
format: aep.planning-md/3
id: story:vulkan-segment-command-buffer-reuse
kind: story
status: active
title: A stepwise Vulkan run submits a recorded segment again
summary: A segment of training steps is recorded once per upload buffer and step count and submitted again, so a run makes a fixed number of command pools and the device waits of 107 to 393 ms on a 27 ms segment end.
relations:
- decomposes: epic:bert-tiny-beats-torch-compile
- serves: vision:modelspec
revision: 4
transitions:
- {from: "draft", to: "proposed", at: "2026-11-05T14:34:29Z", actor: "human:timo", revision: 3, executor: "agent:modelspec-a4fb0b"}
- {from: "proposed", to: "active", at: "2026-11-05T14:34:29Z", actor: "human:timo", revision: 4, executor: "agent:modelspec-a4fb0b"}
---
# A stepwise Vulkan run submits a recorded segment again

## Context

In the BERT-tiny series of 2026-11-05 at 1cf9e12f every ModelSpec run had 8 or 9 windows of
100 steps at 4,634 to 17,799 µs per step against a median of 3,678 to 3,730, 3.73 to 4.25 s of
a run's 41 s, where the `torch.compile` runs 2 and 3 at the same load had none
(`benchmarks/raw/2026-11-05-bert-tiny-pretrain-3/`). The windows fell at different steps in each
run, so the time to the target, step 1,100, ranged from 4.06 to 4.66 s.

A trace of the stepwise loop on the RTX 3090 (2026-11-05 13:06Z to 13:12Z, four runs, with and
without `--record`) put the time in the host's wait for the segment in flight: 107 to 393 ms
for a segment of 8 steps that takes 27 ms, several segments in a row, while the host's own work
per segment stayed at 2.9 to 4.4 ms. During those waits `nvidia-smi pmon` showed the process's
memory utilisation at 64 to 88 % instead of 31 to 39 %, and its device memory grew in steps from
1,070 to 1,152 MiB over a run. The runner made a new command pool and command buffer for every
segment and destroyed them after its wait, about 1,300 of each in a run of 10,000 steps.

Probe (2026-11-05 14:02Z to 14:10Z, three runs each way in turn, load 15 to 28): with the
segments of training steps recorded once and submitted again, the longest wait for a segment
was 35, 37 and 34 ms, against 68, 265 and 277 ms with a pool per segment; training seconds
38.76, 39.51 and 39.99 against 40.74, 46.87 and 43.66; the losses were the same bits.

## Outcome

A segment of training steps alone records the same commands as every other segment with its
upload buffer (parity) and its step count: what differs from step to step, the batch, the
dropout tokens and the AdamW scalars, comes in through the upload. `vulkan::stepwise` keeps the
command buffer of each such shape (`dispatch::Reusable`, from `Device::record_reusable` and
`Recorded::reusable`) and submits it again (`Device::submit_again`) for every later segment of
that shape. The run's first command buffer, a segment whose state is read back, the last step's
and every segment of a `--profile` run are recorded as before, for one submission.

`Device::submit_again` refuses a second submission while one is alive
(`ExecError::StillPending`), and `Recorded::reusable` refuses a recording begun for one
submission (`ExecError::RecordedOnce`). `Device::recordings` counts the command buffers
recorded. The losses, parameters, checkpoints, printed lines and `RunStats` of a run are
unchanged. DESIGN.md 0.85 states it (§ 8).

## Acceptance

- `cargo test -p modelspec-runtime --features vulkan` and `cargo clippy -p modelspec-runtime -p
  modelspec-cli --all-targets -- -D warnings` pass.
- `tests/vulkan_segment_reuse.rs`, on the device with synchronization validation on: the
  hidden-32 twin over 41 and 121 steps in segments of 4 records 5 command buffers and submits 12
  and 32, with the bits, in every loss, observed step and final parameter, of the run that
  submits no command buffer twice; a reusable recording runs with what its staging buffer holds
  at each submission, is refused while pending, and a one-submission recording is not kept.
- The device tests of the stepwise runner (`adversary_multistep`, `vulkan_heldout_metrics`,
  `adversary_eval_download`, `vulkan_bert_attention_loss_adamw`, `vulkan_checkpoint`,
  `vulkan_profile`, `vulkan_train`) and `modelspec-cli`'s `train_vulkan`, `bert_cpu_train` and
  `train_data` pass with no validation message.

## Not in this story

- The command buffers of the held-out evaluation, one or more per evaluated step, are still
  recorded each time; their time is outside the training clock.
- The host thread losing the processor between two segments (0.09 to 1.35 s of a run at load 15
  to 28 in the probe) is not addressed.
