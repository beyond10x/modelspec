# BERT-tiny pre-training: the PyTorch CUDA opponent

The PyTorch side of `story:bert-tiny-vs-pytorch-cuda`. ModelSpec's side is
`modelspec train examples/bert_tiny.mspec --backend vulkan --data <data> --eval-every 100 --clock`.
Both sides pre-train BERT-tiny on the same packed corpus until the held-out loss reaches a
target, and the figure is the training time to it. No ModelSpec build, test, gate or `task`
target runs this directory ([AGENTS.md](../../AGENTS.md), Rules).

## What it measures

The program of `examples/bert_tiny.mspec`, written out with `torch` and `torch.nn.functional` as
ModelSpec defines it (DESIGN.md § 2.1):

- token and learned position embeddings, added, then a layer norm and dropout;
- two post-layer-norm encoder layers: hidden 128, 2 heads of 64 columns, scores scaled by
  `1 / sqrt(64)`, each query attending to the keys of its own sequence whose mask entry is 1, and
  a feed-forward width of 512 with the tanh-form GELU evaluated as `x / (1 + exp(-2u))`;
- the MLM transform `mlm_w` with its GELU and layer norm, then the decoder: the token table
  transposed plus its own bias `mlm_bias`.

Dropout 0.1 follows the embeddings and each attention and feed-forward sublayer. Its masks are
torch's own RNG, not ModelSpec's. `layer_norm` uses `eps` 1e-12. Weights are `[in, out]` and
multiply from the right. The loss is the weighted masked-language-model cross-entropy over 32
sequences of 128 tokens per step, in `f32` on CUDA with TF32 off.

The optimizer is `torch.optim.AdamW`:

- lr 1e-4, weight decay 0.01 on every parameter, betas 0.9 and 0.999, eps 1e-6, `foreach=True`;
- ModelSpec's schedule: with `t` counted from 1, `lr·t/warmup` for `t ≤ warmup` and
  `lr·(steps − t)/(steps − warmup)` after;
- by default 1,000 warm-up steps of 10,000.

The batches are ModelSpec's `mlm_batch`, bit for bit: the Train epoch order and the 80/10/10
masking are drawn with Philox4x32-10 on the run seed (stream tags 5 and 3, DESIGN.md § 2.3), so
both sides see the same sequences, masks and labels at every step. The Philox draw, the `Packed`
reader, the masking and the HeldOut batches are a copy of
[`opponents/bert_tiny/bert_tiny.py`](../bert_tiny/bert_tiny.py), so the two opponents stay
independent. The copy is vectorised over `torch` tensors because the corpus has a million
sequences. At start, the scalar copy checks the vectorised draw: the Philox words of both
streams, the first keys of the epoch order and the first 64 rows of Train and HeldOut batch 0. A
difference refuses the run. The parameters are torch's own Xavier-uniform draw under `--seed`, or
ModelSpec's with `--checkpoint`.

`--mode eager` runs the step as written. `--mode compile` runs the forward and the loss under
`torch.compile(dynamic=False)`, after a dynamo reset. A step is:

- the batch's copy to the device, `non_blocking` from pinned host memory;
- `zero_grad(set_to_none=True)`, the loss and `backward()`;
- `AdamW.step()` and the scheduler's step.

### The training clock

Before the first step, both modes run a warm-up on a copy of the parameters: 3 forward and
backward passes and one `AdamW.step()` with an optimizer of its own. The copy and that optimizer
are then discarded, so the timed run still starts from the parameters it read. In compile mode
the warm-up compiles the step; in both modes it pays the first use of cuBLAS and of AdamW, so
the first timed step carries no first-use cost.

Where each side's clock starts and stops, per training step:

| | starts | stops |
|---|---|---|
| this opponent | a `torch.cuda.synchronize` after the step's batch is staged in pinned host memory | a `torch.cuda.synchronize` after the scheduler's step |
| ModelSpec (`--clock`, stepwise) | recording the step's command buffer, after the host staged the batch | the end of that step's host wait |

Both clocks include the step's copy of its batch to the device, ModelSpec's from its staging
buffer, the opponent's from pinned host memory. These are
outside both:

- the warm-up, the loading, the plan build and pipeline creation;
- the host's draw and staging of the batch;
- the held-out evaluations.

`train_seconds` is the sum of the per-step clock. The figure per run is
`us_per_step = train_seconds · 10⁶ / steps`, the mean over the run's steps, on both sides:
ModelSpec's from the `train_seconds <s> … steps <n>` line of `--clock`, this opponent's from its
result line. The benchmarks page and the report give the median of 3 runs of it.

### The held-out evaluation

At step 0, at every multiple of `--eval-every` (default 100) and at the last step, the opponent
runs the forward without dropout over every full batch of 32 sequences of the HeldOut split. Batch
`b` is drawn as ModelSpec draws it, with the draw id `b + 2^63`. The loss is
`Σ w·nll / Σ w` over the whole split: `nll` in `f32` per row, both sums in `f64`, as ModelSpec's
`HeldOut::evaluate` sums them. The evaluation uses eager PyTorch in both modes.

## Output

One JSON line on stdout per evaluation: `{"step": k, "heldout_loss": v, "train_seconds": s}`, with
`s` the training clock after `k` steps. Then one line in the bench's shape:

| field | meaning |
|---|---|
| `side` | `pytorch-eager` or `pytorch-compile` |
| `workload` | `bert-tiny-pretrain` |
| `batch`, `seq` | 32 sequences of 128 |
| `tokens_per_step` | `batch · seq`, 4096 |
| `flop_per_step`, `flop_rule` | `N·(54·H² + 24·H·F + 6·H·V + 24·S·H)` with `N = B·S`, `H = 128`, `F = 512`, `V = 8,192`, that is 1,170,210,816 per sequence and 37,446,746,112 per step, `matmul-only`. It counts the matrix products of a step, forward and backward, and nothing else; the embedding lookups count 0 (`crates/modelspec-runtime/benches/bert_tiny/flop.rs`) |
| `device` | GPU name, torch and CUDA versions, TF32 off, the mode |
| `build` | `commit`: the git commit of the checkout; `dirty`: whether its tracked files differ from it |
| `status` | `measured` or `refused` |
| `steps` | the training steps |
| `us_per_step` | `train_seconds · 10⁶ / steps`, the mean of the per-step clock over the run, as ModelSpec's `train_seconds` over its steps |
| `tokens_per_s`, `gflops` | `tokens_per_step` and `flop_per_step` over `us_per_step` |
| `train_seconds` | the training clock at the end |
| `eval_every` | `--eval-every` |
| `target_loss` | `--target`, or `null` |
| `steps_to_target`, `seconds_to_target` | the first evaluated step whose held-out loss is at most the target, and the training clock at it; `null` when no evaluation reaches it |
| `final_heldout_loss` | the held-out loss at the last step |
| `refusal` | refused: the exception it raised |

Progress goes to stderr. A failure prints one `bert_tiny_pretrain: refused: …` line on stderr and
the result line with `status` `refused`, then exits 1.

## Running it

It needs Python 3.10 or later and a CUDA build of PyTorch, in a virtual environment outside the
repository, on a GPU with no other compute process. NumPy is not needed.

`--data` is the packed directory `modelspec data pack` wrote, the one ModelSpec's run reads. The
target is its `packed.json` field `heldout_unigram_baseline_nats`:

```console
python3 -m venv ~/.cache/modelspec-opponents-venv
~/.cache/modelspec-opponents-venv/bin/pip install torch
modelspec train examples/bert_tiny.mspec --backend vulkan --data <data> --steps 0 --checkpoint initial.msckpt
~/.cache/modelspec-opponents-venv/bin/python opponents/bert_tiny_pretrain/bert_tiny_pretrain.py \
  --data <data> --mode eager --checkpoint initial.msckpt --target 6.926537926202018 > eager.jsonl
```

These flags change the run:

- `--mode compile` runs the compiled mode.
- `--steps` and `--warmup-steps` change the schedule; the example's are 10,000 and 1,000.
- `--eval-every` changes how often it evaluates.
- `--seed` changes the run seed of the batches and of torch's draws. The default is 42, the
  example's, and it must match ModelSpec's for the batches to be the same.
- `--device` picks the CUDA device.

`train --steps 0 --checkpoint` writes the step-0 parameters of an AdamW run. A checkpoint of a
later AdamW step needs the optimizer moments, which ModelSpec does not save yet.

## The step-0 held-out loss check

`--loss-check PATH.msckpt --data <data>` trains nothing. It prints `eval <step> loss <v>`, the
held-out loss of the checkpoint's parameters without dropout, in the form of ModelSpec's `eval`
line. ModelSpec prints its own with `--eval-every`:

```console
modelspec train examples/bert_tiny.mspec --backend vulkan --data <data> --steps 0 --eval-every 100 --checkpoint initial.msckpt
~/.cache/modelspec-opponents-venv/bin/python opponents/bert_tiny_pretrain/bert_tiny_pretrain.py --data <data> --loss-check initial.msckpt
```

The two losses agree within the run tolerance of DESIGN.md § 9, `|g − c| ≤ 1e-4 + 1e-3·|c|`. This
shows three things: the opponent reads the checkpoint and the HeldOut batches ModelSpec reads, and
it computes the same forward without dropout. On 2026-10-28 on an RTX 3090 both printed
`eval 0 loss 9.029767`.

## No one-step comparison

The transformer-block opponent's `--compare-step` replays one SGD step and compares every
parameter with ModelSpec's. This opponent has none. Dropout draws its masks from each side's own
RNG, so one training step of the two sides differs by design. Also, `train` writes no checkpoint
of an AdamW step after step 0. The held-out loss check above and the loss curves of the two runs
take its place.

## The checkpoint reader

The reader is the one of
[`opponents/transformer_block/transformer_block.py`](../transformer_block/transformer_block.py),
copied. It checks:

- the header's magic, version and endian marker;
- the header checksum (ModelSpec refuses a mismatch with E-CKPT-001);
- the file size the manifest states (E-CKPT-002);
- the BLAKE3 hash of the payload (E-CKPT-004).

A file that fails a check is refused with a line naming the condition and ModelSpec's code. It
reads the parameters by path, `BertTiny.<name>`. The hash is computed in pure Python, a few
seconds per checkpoint of the example.
