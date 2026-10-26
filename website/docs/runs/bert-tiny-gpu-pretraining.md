---
title: "Run: BERT-tiny pre-trained on the GPU"
sidebar_position: 1
description: The record of the first GPU pre-training run of examples/bert_tiny.mspec, 10,000 steps over packed WikiText-103 on an RTX 3090 through Vulkan, ending at a held-out loss of 6.7674994 nats against a unigram baseline of 6.9265379.
lede: "Pre-trained from scratch on one RTX 3090 through Vulkan, BERT-tiny ends 10,000 steps at a held-out masked-language-model loss of 6.7674994 nats, 0.159 nats (2.30 %) below the held-out unigram baseline of its corpus."
source: "The run's stdout and stderr, its recorded event file and the coordinator's start and end timestamps, kept outside the repository; packed.json, binding.json and pack output of the packed corpus; examples/bert_tiny.mspec; the inspect-device --backend vulkan output and the driver's version and model lines of the device; all from 2026-10-26."
---

# Run: BERT-tiny pre-trained on the GPU

On 2026-10-26, `examples/bert_tiny.mspec` was pre-trained from its seeded initialization for
10,000 steps on one NVIDIA GeForce RTX 3090 through Vulkan, over WikiText-103 packed into
sequences of 128 tokens. The held-out masked-language-model loss fell from 9.029848 at step 0 to
**6.7674994** at step 10,000, **0.159 nats (2.30 %) below the held-out unigram baseline of
6.9265379**, the mean loss of predicting each non-special `HeldOut` token from its add-one
frequency in the `Train` split.
The held-out accuracy at step 10,000 is **0.07592517**. The run took 45 min 5 s, evaluations
included.

This is the reference run the BERT milestone asked for: the target was "below the unigram
baseline", because no run existed to set a tighter number from. The figures below are that number's
source.

## Result

| Figure | Value |
|---|---|
| Held-out unigram baseline (`packed.json`) | 6.926537926202018 nats |
| Held-out loss, step 0 | 9.029848 |
| Held-out loss, step 10,000 | 6.7674994 |
| Margin below the baseline | 0.1590385 nats, 2.30 % of the baseline |
| Held-out accuracy, step 10,000 | 0.07592517 |
| First evaluation below the baseline | step 1,500 (6.919595) |
| Unigram baseline over the evaluated rows | 6.904154117063165 nats, 41,857 rows |
| Margin below the baseline over the evaluated rows | 0.1366547 nats, 1.98 % of that baseline |
| First evaluation below the baseline over the evaluated rows | step 2,500 (6.903071) |
| Steps | 0 to 10,000, `steps = 10000` in the `train` block |
| Wall time | 2026-10-26T06:32:19Z to 07:17:24Z, 45 min 5 s (2,705 s) |
| Recorded event file | `full.events.jsonl`, 2,609,690 bytes, 10,003 lines |
| SHA-256 of the event file | `2336f1e1f0a7375a6702a48defba50dbc5b7218fbdf7ffc8853bebc4b12e11bd` |

The margin is the baseline minus the last `eval` loss as printed, 6.926537926202018 − 6.7674994.

The two baselines cover two populations. The baseline of `packed.json`, 6.9265379, is the
add-one unigram loss over every non-special token of the 2,189 `HeldOut` sequences, and it is the
figure the run's acceptance names; against it the margin is 0.1590385 nats (2.30 %) and the
first evaluation below it is step 1,500 (6.919595). Every `eval` loss is measured only over the
41,857 selected masked rows of the 68 full `HeldOut` batches, and the same unigram loss over those
rows' labels is 6.9041541; against it the margin is 0.1366547 nats (1.98 %) and the first
evaluation below it is step 2,500 (6.903071).

## What ran

| Item | Value |
|---|---|
| Commit the binary was built from | `7e371a17f5bf4ef5fe5f5b5872432f415b14708f` |
| `modelspec --version` | `modelspec 0.18.0` |
| Build | `cargo build --release -p modelspec-cli --features vulkan` |
| Device | NVIDIA GeForce RTX 3090, driver 610.57.04 |
| `inspect-device --backend vulkan` | `vendor_id: 4318`, `device_id: 8708`, `driver_version: 2559459584`, `vulkan_api_version: 4211029`, `negotiation: ok` |

`inspect-device` prints the driver version as the Vulkan driver reports it; 2559459584 decodes to
610.57.04 in NVIDIA's encoding, the version the kernel module reports. `train` prints no device
line, so the device is the `inspect-device` output recorded beside the run, whose `device_uuid`
equals the GPU UUID of the RTX 3090.

### The commands

`<data>` is the operator's data directory and `<dir>` the directory the run wrote to; both are
outside the repository. The commands run from the root of a checkout at the commit above, with
`modelspec` the binary of that build.

```sh
modelspec data prepare <data>/wikitext-103-raw/wikitext-103-raw --out <data>/prepared
modelspec data tokenizer --dataset <data>/prepared --vocab-size 8192 --out <data>/tokenizer-8192.json
modelspec data pack --dataset <data>/prepared --tokenizer <data>/tokenizer-8192.json --seq-len 128 --out <data>/packed-128
MODELSPEC_VK_VALIDATION=0 modelspec train examples/bert_tiny.mspec --backend vulkan \
  --data <data>/packed-128 --eval-every 500 \
  --serve 127.0.0.1:7878 --ui ui/dist --record <dir>/full.events.jsonl \
  > <dir>/full.stdout 2> <dir>/full.stderr
```

`data prepare` and `data tokenizer` ran on 2026-10-24; their output was not kept, only their
records. The tokenizer was trained again on 2026-10-26 with the same arguments and another `--out`, and the file came out
byte for byte the same, tokenizer `dfd98df4-cbd0-8076-b713-99ab202efa85` of 8,192 tokens.
`data pack` ran on 2026-10-26 in 18.58 s and printed:

```text
packed 848134f9-e6e7-8c31-9d13-ae6f6a4a1b56 (seq_len 128)
Train 1049703 sequences, blake3 b52330240056d6b0d9a4a3622855e0a24b594daa1be836b41fd40998f963c971
HeldOut 2189 sequences, blake3 80c2af957c4bddb9c7efcd0bf618d7e8d2737571e452f90dbaed59b94d77ec7a
binding 668f3433-dbd7-8871-a5a7-a421c2bd1816 (dataset 0ddcfa78-c328-8cc9-ae3b-490c096d55a5, tokenizer dfd98df4-cbd0-8076-b713-99ab202efa85)
heldout unigram baseline 6.926537926202018 nats
```

`MODELSPEC_VK_VALIDATION=0` runs without the Khronos validation layer. The run served its events
and the observatory UI while it trained; its stderr is the two lines

```text
serving http://127.0.0.1:7878/events
ui http://127.0.0.1:7878/
```

and its stdout begins with the `data` line:

```text
data dataset 0ddcfa78-c328-8cc9-ae3b-490c096d55a5 tokenizer dfd98df4-cbd0-8076-b713-99ab202efa85 seq_len 128 masking 0.15/0.8/0.1/0.1 binding 668f3433-dbd7-8871-a5a7-a421c2bd1816
```

### The corpus

| Item | Value |
|---|---|
| Corpus | WikiText-103 raw: `wiki.train.raw` as `Train` (28,472 documents, 540,568,191 bytes), `wiki.valid.raw` as `HeldOut` (60 documents, 1,146,846 bytes); dataset `0ddcfa78-c328-8cc9-ae3b-490c096d55a5` |
| Tokenizer | WordPiece, 8,192 tokens, trained on `Train` only |
| Packed | `seq_len` 128; 1,049,703 `Train` and 2,189 `HeldOut` sequences; packed `848134f9-e6e7-8c31-9d13-ae6f6a4a1b56` |
| Masking | 15 % of the tokens selected, of those 80 % masked, 10 % random, 10 % kept |
| Baseline | 6.926537926202018 nats |

### The model and its hyper-parameters

From `examples/bert_tiny.mspec`, unchanged by the run:

| Setting | Value |
|---|---|
| Model | 2 post-layer-norm encoder layers, hidden 128, 2 heads, feed-forward 512, learned positions, MLM head tied to the token table; 1,486,720 parameters |
| Batch | 32 sequences of 128 tokens |
| Initialization | `xavier_uniform()` for every matrix, `zeros()` for every bias and layer-norm shift, `ones()` for every layer-norm gain |
| Dropout | 0.1 after the embeddings and each sublayer |
| Optimizer | `adamw(lr = 1e-4, weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-6, warmup_steps = 1000)`: linear warm-up from 0 over 1,000 steps, then linear decay to 0 at the last step |
| Steps | `steps = 10000` |
| Seed | `seed = 42` |
| Evaluation | `--eval-every 500`: every full `HeldOut` batch, without dropout, at step 0, every multiple of 500 and the last step |

## The held-out curve

Every `eval` line of the run, as printed:

| Step | Held-out loss | Held-out accuracy |
|---|---|---|
| 0 | 9.029848 | 0.00019112694 |
| 500 | 7.751668 | 0.03769979 |
| 1000 | 6.928317 | 0.05349165 |
| 1500 | 6.919595 | 0.05349165 |
| 2000 | 6.913906 | 0.05349165 |
| 2500 | 6.903071 | 0.05349165 |
| 3000 | 6.889678 | 0.05349165 |
| 3500 | 6.8805966 | 0.05349165 |
| 4000 | 6.8714867 | 0.05375445 |
| 4500 | 6.85976 | 0.056621354 |
| 5000 | 6.846145 | 0.056645244 |
| 5500 | 6.8329067 | 0.060491674 |
| 6000 | 6.818905 | 0.064624794 |
| 6500 | 6.8074417 | 0.068757914 |
| 7000 | 6.7983656 | 0.07081253 |
| 7500 | 6.789642 | 0.072461 |
| 8000 | 6.7826333 | 0.07305827 |
| 8500 | 6.7763753 | 0.07470674 |
| 9000 | 6.7717495 | 0.07511289 |
| 9500 | 6.7688417 | 0.07556681 |
| 10000 | 6.7674994 | 0.07592517 |

At step 1,000, the end of the learning-rate warm-up, the loss is within 0.002 nats of the
`packed.json` baseline, and the accuracy is 0.05349165 at all six evaluations from step 1,000 to step 3,500. From step 4,000 both move again,
and the last 1,000 steps gain 0.0043 nats.

## The training loss

The loss of the step's training batch, with dropout, from every `step` line the run printed (one
every 50 steps). The row is the step to add the column to:

| Steps | +0 | +50 | +100 | +150 | +200 | +250 | +300 | +350 | +400 | +450 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 9.035597 | 9.027942 | 9.011951 | 8.970035 | 8.88838 | 8.748299 | 8.571261 | 8.404144 | 8.198336 | 7.950393 |
| 500 | 7.785091 | 7.550701 | 7.4505258 | 7.244367 | 7.025592 | 7.125413 | 7.115167 | 7.0855904 | 6.8641562 | 7.1365457 |
| 1000 | 6.8899775 | 6.9689097 | 6.9363556 | 7.0709934 | 6.780761 | 6.989836 | 6.903854 | 6.9725695 | 7.0324545 | 7.0909243 |
| 1500 | 6.95485 | 6.9837427 | 6.863608 | 7.025589 | 6.8241973 | 6.851167 | 7.072536 | 7.0579333 | 6.926041 | 6.8584156 |
| 2000 | 7.0497346 | 6.8190246 | 7.033418 | 6.8819375 | 7.076615 | 7.0276866 | 7.032393 | 6.918658 | 6.8852963 | 6.8998823 |
| 2500 | 6.8869667 | 6.8904223 | 7.037469 | 6.9465747 | 6.8539724 | 7.197012 | 6.9289136 | 6.990609 | 7.1066623 | 6.7433767 |
| 3000 | 6.8695393 | 6.845083 | 7.031476 | 7.0502915 | 6.8496842 | 6.736315 | 6.7675157 | 6.96176 | 7.120426 | 7.065383 |
| 3500 | 6.7505207 | 6.6666365 | 7.0108824 | 6.8758535 | 6.9566207 | 6.96398 | 6.811744 | 6.8505325 | 7.0502462 | 6.934384 |
| 4000 | 7.009967 | 7.0041475 | 6.8930073 | 6.859113 | 6.92094 | 6.927219 | 7.001245 | 6.90847 | 6.901637 | 7.12654 |
| 4500 | 6.9636626 | 6.836495 | 6.8648596 | 6.7916503 | 6.9644675 | 6.9806805 | 7.0620756 | 6.866865 | 6.923736 | 7.0048003 |
| 5000 | 6.9109597 | 6.8250766 | 7.016906 | 6.8660083 | 7.0968914 | 6.9395847 | 6.79049 | 6.8496704 | 6.96941 | 6.74438 |
| 5500 | 6.950887 | 6.880487 | 7.0662704 | 6.8326306 | 6.816191 | 6.908616 | 6.6057186 | 6.9121785 | 7.026356 | 6.822419 |
| 6000 | 6.912746 | 6.79499 | 6.844456 | 6.717765 | 6.9571986 | 6.7928114 | 6.6292334 | 6.897317 | 6.857352 | 6.934957 |
| 6500 | 6.890438 | 6.7544203 | 6.7388115 | 6.9086504 | 7.030515 | 6.9610786 | 6.96801 | 6.797743 | 6.7912774 | 6.980507 |
| 7000 | 6.839261 | 6.801634 | 6.9396605 | 6.83235 | 6.8966413 | 6.759473 | 7.092868 | 6.7232084 | 6.8071465 | 6.7043414 |
| 7500 | 6.829295 | 6.8611875 | 6.6606164 | 7.0465097 | 6.9076962 | 6.9266496 | 6.9225426 | 6.880218 | 6.952533 | 6.7888722 |
| 8000 | 6.748133 | 6.7399645 | 6.918322 | 6.781184 | 6.985034 | 6.837594 | 6.7303915 | 6.953026 | 6.895051 | 6.8712506 |
| 8500 | 6.9588294 | 6.7900424 | 6.9031744 | 6.8392663 | 6.7811856 | 6.7142425 | 6.741662 | 6.903007 | 6.7261434 | 6.9399443 |
| 9000 | 6.768724 | 6.9030156 | 6.695873 | 6.805194 | 6.71568 | 6.9484835 | 6.7462754 | 6.98856 | 6.690157 | 6.836898 |
| 9500 | 6.8189487 | 6.837841 | 6.680764 | 6.752279 | 6.928002 | 6.8201103 | 6.89508 | 6.885496 | 6.776336 | 6.740318 |
| 10000 | 6.8116674 | | | | | | | | | |

The stdout also holds 41 `init` lines, the value of every parameter before step 0, and 41 `param`
lines, its value after the last step.

## The recorded events

`--record` wrote one JSON event per line: one `RunStarted` (`steps` 10000, `start_step` 0, the
binding `668f3433-dbd7-8871-a5a7-a421c2bd1816`), one `StepCompleted` for each step from 0 to
10,000 (10,001 in all), and one `RunFinished`. The 21 evaluated steps carry the metrics
`heldout_mlm_loss` and `heldout_mlm_accuracy`; the last `heldout_mlm_loss` is
6.7674994468688965, which prints as `6.7674994` as an `f32`, the last `eval` loss. Steps 0 and
10,000 also carry the norm and histogram of every parameter's values.

The file stays outside the repository, with the operator's data. Replay it, with the observatory UI,
from the root of a checkout after `task ui`:

```sh
modelspec observe <dir>/full.events.jsonl --serve 127.0.0.1:0 --ui ui/dist
```

Check the copy you replay against the SHA-256 above with `sha256sum`.

## What the record does not hold

- **No warm-up step.** No separate warm-up run preceded it, as the repository's rules ask for a
  GPU run, so the wall time includes the first step's setup on the device.
- **No per-step times.** Neither the stdout nor the events carry a timestamp, so the 2,705 s are
  not split between training and evaluation. On the same device, the step-time measurement before
  the run gave 0.217 to 0.232 s per training step; 10,001 steps at that rate are 2,170 to 2,320 s,
  which leaves an estimated 385 to 535 s, 18 to 25 s for each of the 21 evaluations. Evaluation
  tallies on the host: it runs the 68 full `HeldOut` batches one command buffer and one host wait
  at a time and downloads each batch's prediction, 134,217,728 bytes, to compute the loss and
  accuracy on the CPU. A tally on the device, two floats per batch, is not built.
- **No checkpoint.** The run wrote none; the trained values are the 41 `param` lines of its stdout.
- **No statistics between the first and the last step.** On the GPU the observatory records the
  norm and histogram of each parameter's values at steps 0 and 10,000 only, and no gradient or
  activation statistics.
- **No tighter target.** This run is the reference for one; none is set yet.
