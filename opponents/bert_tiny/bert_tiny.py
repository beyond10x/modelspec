"""The PyTorch opponent of ModelSpec's small masked-language-model runs.

story:muon-optimizer, scenario 7 (coordinator decision D67 (e)). The model is
crates/modelspec-cli/tests/fixtures/mlm_dense.mspec: a token table
[V, 16] read by an embedding lookup, one dense [16, 16] weight `w1`, the tanh
GELU, and the decoder tied to the table through its transpose, trained on the
weighted masked-language-model cross-entropy

    loss = sum_n w[n] * nll(logits[n], labels[n]) / sum_n w[n]   (0 if sum w = 0)

over B sequences of S packed tokens per step.

The data is the packed directory `modelspec data pack` wrote: binding.json,
packed.json and the little-endian i32 files train.tokens and heldout.tokens.
Each step's batch is ModelSpec's `dataset::mlm::mlm_batch`, reproduced here
bit for bit: the Train epoch order and the 80/10/10 masking are drawn with
Philox4x32-10 on the run seed (stream tags 5 and 3, DESIGN.md § 2.3), so both
sides see the same sequences, masks and labels at every step. The initial
parameters are torch's own Xavier-uniform draw under --seed, not ModelSpec's.

The held-out loss at step k is the weighted mean of the loss above over every
full batch of B sequences of the HeldOut split, each drawn as ModelSpec draws
it, with the parameters after k updates; it is printed at 0, every multiple of
--eval-every and the last step, one JSON line each: {"step": k, "heldout_loss": v}.

--optimizer adamw trains every parameter with torch.optim.AdamW. --optimizer
muon trains `w1` with torch.optim.Muon (Nesterov momentum, `ns_steps`
Newton-Schulz steps, adjust_lr_fn="match_rms_adamw", the Moonshot scale
0.2 * sqrt(max(d_out, d_in))) and the table with torch.optim.AdamW at
--adam-lr. Both follow ModelSpec's schedule (coordinator decision D50): with t
counted from 1, lr * t / warmup for t <= warmup and lr * (steps - t) /
(steps - warmup) after.

ModelSpec never runs this file (AGENTS.md, Rules: Python is for opponents only).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

SPECIALS = 5
MASK_TOKEN_ID = 4
TAG_MASKING = 3
TAG_DATA_ORDER = 5
HIDDEN = 16
MASK32 = 0xFFFFFFFF


def philox4x32_10(counter, key):
    """Philox4x32-10 of four counter words under two key words, as ModelSpec's `train::init`."""
    m0, m1, w0, w1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for round_ in range(10):
        if round_ > 0:
            k0 = (k0 + w0) & MASK32
            k1 = (k1 + w1) & MASK32
        p0 = m0 * c0
        p1 = m1 * c2
        hi0, lo0 = p0 >> 32, p0 & MASK32
        hi1, lo1 = p1 >> 32, p1 & MASK32
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return c0, c1, c2, c3


def block(seed, tag, draw_id, index):
    """The Philox block of element `index` of draw `draw_id` on stream `tag` under `seed`."""
    counter = (index, tag, draw_id & MASK32, (draw_id >> 32) & MASK32)
    return philox4x32_10(counter, (seed & MASK32, (seed >> 32) & MASK32))


def uniform(word):
    return (word + 0.5) / 4294967296.0


class Packed:
    """The packed directory: its record, policy and the ids of both splits."""

    def __init__(self, directory):
        directory = Path(directory)
        binding = json.loads((directory / "binding.json").read_text())
        record = json.loads((directory / "packed.json").read_text())
        self.seq_len = int(record["seq_len"])
        self.vocabulary = int(record["vocabulary_size"])
        self.policy = binding["masking_policy"]
        self.ids = {
            role: self._read(directory / name)
            for role, name in (("Train", "train.tokens"), ("HeldOut", "heldout.tokens"))
        }
        self._orders = {}

    def _read(self, path):
        data = path.read_bytes()
        values = [int.from_bytes(data[i : i + 4], "little", signed=True) for i in range(0, len(data), 4)]
        return values

    def sequences(self, role):
        return len(self.ids[role]) // self.seq_len

    def epoch_order(self, seed, epoch):
        if epoch not in self._orders:
            n = self.sequences("Train")
            keyed = []
            for i in range(n):
                w = block(seed, TAG_DATA_ORDER, epoch, i)
                keyed.append(((w[1] << 32) | w[0], i))
            keyed.sort()
            self._orders[epoch] = [i for _, i in keyed]
        return self._orders[epoch]

    def batch(self, role, seed, step, batch_size):
        """ModelSpec's `mlm_batch`: (ids, labels, weights) of `B * S` rows."""
        per_epoch = self.sequences(role) // batch_size
        index = step % per_epoch
        if role == "Train":
            order = self.epoch_order(seed, step // per_epoch)
            chosen = order[index * batch_size : (index + 1) * batch_size]
            draw_id = step
        else:
            chosen = list(range(index * batch_size, (index + 1) * batch_size))
            draw_id = (step + (1 << 63)) & ((1 << 64) - 1)
        random_tokens = self.vocabulary - SPECIALS
        selection = self.policy["selection_rate"]
        mask = self.policy["mask_fraction"]
        random_fraction = self.policy["random_fraction"]
        ids, labels, weights = [], [], []
        source = self.ids[role]
        for b, sequence in enumerate(chosen):
            start = sequence * self.seq_len
            for p in range(self.seq_len):
                label = source[start + p]
                w0, w1, w2, _ = block(seed, TAG_MASKING, draw_id, b * self.seq_len + p)
                selected = label >= SPECIALS and uniform(w0) < selection
                if not selected:
                    token = label
                elif uniform(w1) < mask:
                    token = MASK_TOKEN_ID
                elif uniform(w1) < mask + random_fraction:
                    token = SPECIALS + ((w2 * random_tokens) >> 32)
                else:
                    token = label
                ids.append(token)
                labels.append(label)
                weights.append(1.0 if selected else 0.0)
        return ids, labels, weights


class MlmDense(torch.nn.Module):
    def __init__(self, vocabulary):
        super().__init__()
        self.table = torch.nn.Parameter(torch.empty(vocabulary, HIDDEN))
        self.w1 = torch.nn.Parameter(torch.empty(HIDDEN, HIDDEN))
        torch.nn.init.xavier_uniform_(self.table)
        torch.nn.init.xavier_uniform_(self.w1)

    def forward(self, ids):
        hidden = torch.nn.functional.embedding(ids, self.table)
        hidden = torch.nn.functional.gelu(hidden @ self.w1, approximate="tanh")
        return hidden @ self.table.T


def weighted_loss(logits, labels, weights):
    """The weighted mean cross-entropy and its weight sum."""
    nll = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    return (weights * nll).sum(), weights.sum()


def tensors(batch, device):
    ids, labels, weights = batch
    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
        torch.tensor(weights, dtype=torch.float32, device=device),
    )


def heldout_loss(model, heldout):
    with torch.no_grad():
        numerator = torch.zeros((), dtype=torch.float32)
        denominator = torch.zeros((), dtype=torch.float32)
        for ids, labels, weights in heldout:
            num, den = weighted_loss(model(ids), labels, weights)
            numerator += num.cpu()
            denominator += den.cpu()
        return 0.0 if denominator.item() == 0.0 else (numerator / denominator).item()


def schedule(warmup, steps):
    """The multiplier of the peak rate at optimizer step e, counted from 0 (t = e + 1)."""

    def factor(e):
        t = e + 1
        if t <= warmup:
            return t / warmup
        return max(steps - t, 0) / (steps - warmup)

    return factor


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="a directory `modelspec data pack` wrote")
    parser.add_argument("--optimizer", choices=("adamw", "muon"), required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch", type=int, default=4, help="sequences per step, B")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--adam-lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 0 <= args.warmup_steps < args.steps:
        sys.exit("--warmup-steps must be below --steps")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    packed = Packed(args.data)
    model = MlmDense(packed.vocabulary).to(device)
    heldout = [
        tensors(packed.batch("HeldOut", args.seed, b, args.batch), device)
        for b in range(packed.sequences("HeldOut") // args.batch)
    ]
    if not heldout:
        sys.exit("the HeldOut split holds no full batch")

    betas = (args.beta1, args.beta2)
    if args.optimizer == "adamw":
        optimizers = [
            torch.optim.AdamW(
                model.parameters(), lr=args.lr, betas=betas, eps=args.eps, weight_decay=args.weight_decay
            )
        ]
    else:
        optimizers = [
            torch.optim.Muon(
                [model.w1],
                lr=args.lr,
                weight_decay=args.weight_decay,
                momentum=args.momentum,
                nesterov=True,
                ns_steps=args.ns_steps,
                adjust_lr_fn="match_rms_adamw",
            ),
            torch.optim.AdamW(
                [model.table], lr=args.adam_lr, betas=betas, eps=args.eps, weight_decay=args.weight_decay
            ),
        ]
    factor = schedule(args.warmup_steps, args.steps)
    schedulers = [torch.optim.lr_scheduler.LambdaLR(o, factor) for o in optimizers]

    def report(step):
        print(json.dumps({"step": step, "heldout_loss": heldout_loss(model, heldout)}), flush=True)

    for step in range(args.steps):
        if step % args.eval_every == 0:
            report(step)
        ids, labels, weights = tensors(packed.batch("Train", args.seed, step, args.batch), device)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        numerator, denominator = weighted_loss(model(ids), labels, weights)
        loss = numerator / denominator if denominator.item() != 0.0 else numerator * 0.0
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        for scheduler in schedulers:
            scheduler.step()
    report(args.steps)


if __name__ == "__main__":
    main()
