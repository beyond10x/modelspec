# Tiny masked language model: the PyTorch Muon and AdamW opponent

The PyTorch side of `story:muon-optimizer`, scenario 7. ModelSpec's side is the test
`muon_heldout_loss_at_step_300_is_at_most_adamw` in `crates/modelspec-cli/tests/train_muon.rs`,
which trains `crates/modelspec-cli/tests/fixtures/mlm_dense.mspec` under `muon` and under `adamw`.
No ModelSpec build, test, gate or `task` target runs this directory ([AGENTS.md](../../AGENTS.md),
Rules).

## What it trains

The model of `mlm_dense.mspec`: a token table `[V, 16]` read by an embedding lookup, one dense
weight `w1: [16, 16]`, the tanh GELU, and the decoder tied to the table through its transpose. The
loss is the weighted masked-language-model cross-entropy over `B` sequences of the packed length
per step (`--batch`, default 4).

Batches are ModelSpec's `mlm_batch`, reproduced bit for bit: the Train epoch order and the 80/10/10
masking are drawn with Philox4x32-10 on `--seed` (stream tags 5 and 3, DESIGN.md § 2.3), so both
sides see the same sequences, masks and labels at every step. The initial parameters are torch's
own Xavier-uniform draw under `--seed`, not ModelSpec's, so the curves agree in shape, not in bits.

| `--optimizer` | `w1` | `table` |
|---|---|---|
| `adamw` | `torch.optim.AdamW` at `--lr` | `torch.optim.AdamW` at `--lr` |
| `muon` | `torch.optim.Muon` at `--lr`, Nesterov, `--ns-steps`, `adjust_lr_fn="match_rms_adamw"` (the Moonshot scale `0.2·sqrt(max(d_out, d_in))`) | `torch.optim.AdamW` at `--adam-lr` |

Both follow ModelSpec's schedule: with `t` counted from 1, `lr·t/warmup` for `t ≤ warmup` and
`lr·(steps − t)/(steps − warmup)` after. The defaults are the hyperparameters of `mlm_dense.mspec`:
`--lr 0.01 --adam-lr 0.01 --momentum 0.95 --ns-steps 5 --warmup-steps 30 --weight-decay 0.01
--beta1 0.9 --beta2 0.999 --eps 1e-8 --steps 300 --eval-every 50 --seed 7`.

torch's Muon differs from ModelSpec's in two places that do not change the update's direction:
it runs Newton–Schulz in bfloat16 where ModelSpec runs `f32`, and its momentum is the
exponential average `b ← m·b + (1 − m)·g` where ModelSpec sums `b ← m·b + g`; the normalisation
before Newton–Schulz removes the constant factor.

## Output

One JSON line on stdout per evaluation, at step 0, every multiple of `--eval-every` and the last
step: `{"step": k, "heldout_loss": v}`. The held-out loss is the weighted mean over every full
batch of the HeldOut split, drawn as ModelSpec draws it, with the parameters after `k` updates.

## Running it

Needs Python 3.10 or later and PyTorch 2.9 or later (for `torch.optim.Muon` with `adjust_lr_fn`),
in a virtual environment outside the repository. It runs on the CPU:

```console
python3 -m venv ~/.cache/modelspec-opponents-venv
~/.cache/modelspec-opponents-venv/bin/pip install torch
~/.cache/modelspec-opponents-venv/bin/python opponents/bert_tiny/bert_tiny.py --data packed --optimizer muon --device cpu
```

`--data` is a directory `modelspec data pack` wrote, made as the test makes it:

```console
modelspec data prepare crates/modelspec-runtime/tests/fixtures/corpus-tokenizer --out dataset
modelspec data tokenizer --dataset dataset --vocab-size 300 --out tokenizer.json
modelspec data pack --dataset dataset --tokenizer tokenizer.json --seq-len 16 --out packed
```
