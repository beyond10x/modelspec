"""The PyTorch CUDA opponent of ModelSpec's BERT-tiny pre-training, timed to a held-out loss.

story:bert-tiny-vs-pytorch-cuda. The program is examples/bert_tiny.mspec, written out with torch
and torch.nn.functional as ModelSpec defines it (DESIGN.md section 2.1): token and learned
position embeddings, a layer norm, two post-layer-norm encoder layers (hidden 128, 2 heads,
feed-forward 512 with the tanh-form GELU), the MLM transform and its layer norm, and the decoder
tied to the token table through its transpose plus its own bias; dropout 0.1 after the embeddings
and after each attention and feed-forward sublayer, drawn by torch's own RNG; layer norm divides
by sqrt(var + 1e-12); the loss is the weighted masked-language-model cross-entropy

    loss = sum_n w[n] * nll(logits[n], labels[n]) / sum_n w[n]   (0 if sum w = 0)

over B = 32 sequences of S = 128 packed tokens per step, f32 on CUDA with TF32 off. AdamW at lr
1e-4 with weight decay 0.01 on every parameter, betas 0.9 and 0.999, eps 1e-6, on ModelSpec's
schedule: with t counted from 1, lr * t / warmup for t <= warmup and lr * (steps - t) /
(steps - warmup) after (1,000 of 10,000 steps by default).

The batches are ModelSpec's `dataset::mlm::mlm_batch` (the Philox4x32-10 draw, the Packed reader,
the 80/10/10 masking and the HeldOut batches of opponents/bert_tiny/bert_tiny.py, copied here and
vectorised over torch tensors because the corpus has a million sequences; the scalar copy checks
the vectorised draw at start). The parameters are torch's own Xavier-uniform draw under --seed, or
ModelSpec's with --checkpoint.

--mode eager runs the step as written; --mode compile runs the forward and the loss under
torch.compile(dynamic=False). Before the first step both modes run 3 forward and backward passes
on a copy of the parameters, which compiles the compiled mode and initialises cuBLAS in both, and
then forget them. The training clock starts at the first step: per step, host wall time from a
CUDA synchronize before the step's batch is copied to the device to one after the scheduler's
step. The host's draw of the batch, the held-out evaluations, the warm-up and loading are outside
it.

At step 0, every multiple of --eval-every and the last step it evaluates every full HeldOut batch
without dropout, as ModelSpec's `train --eval-every` does, and prints one JSON line
{"step": k, "heldout_loss": v, "train_seconds": s}, s the training clock after k steps. At the end
it prints one JSON line in the bench's shape, with the steps and training seconds to --target.
--loss-check <path.msckpt> prints instead the held-out loss of the checkpoint's parameters as
`eval <step> loss <v>` and trains nothing. Progress goes to stderr; a failure prints one
`bert_tiny_pretrain: refused: ...` line on stderr and exits 1.

The .msckpt reader is opponents/transformer_block/transformer_block.py's, copied: it verifies the
checksums ModelSpec's reader verifies and refuses the files it refuses with E-CKPT-001, E-CKPT-002
and E-CKPT-004.

ModelSpec never runs this file (AGENTS.md, Rules: Python is for opponents only).
"""

import argparse
import array
import json
import math
import os
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

SPECIALS = 5
MASK_TOKEN_ID = 4
TAG_MASKING = 3
TAG_DATA_ORDER = 5
MASK32 = 0xFFFFFFFF
BATCH = 32
SEQ = 128
HIDDEN = 128
HEADS = 2
FFN = 512
VOCAB = 8192
LAYERS = 2
DROPOUT = 0.1
EPS = 1e-12
LR = 1e-4
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
ADAM_EPS = 1e-6
SEED = 42
WARMUP_PASSES = 3
CHECKED_ROWS = 64
FLOP_RULE = "matmul-only"
WORKLOAD = "bert-tiny-pretrain"
MODEL = "BertTiny"
HEADER = 56


def flop_per_step(batch, seq):
    """The matmul FLOP of one step, N (54 H^2 + 24 H F + 6 H V + 24 S H) with N = B S, as
    crates/modelspec-runtime/benches/bert_tiny/flop.rs counts it: the block's
    N (42 H^2 + 24 H F + 24 S H), the input adjoints of the first layer's three projections
    (6 N H^2, the embeddings are trained), the MLM transform (6 N H^2) and the tied decoder
    (6 N H V); embedding lookups count 0. 1,170,210,816 per sequence, 37,446,746,112 at B = 32."""
    n = batch * seq
    return n * (54 * HIDDEN * HIDDEN + 24 * HIDDEN * FFN + 6 * HIDDEN * VOCAB + 24 * seq * HIDDEN)


# The Philox draw, copied from opponents/bert_tiny/bert_tiny.py: the scalar reference.


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


# The same draw over a tensor of element indices, in int64 without overflow.


def mulhilo(multiplier, value):
    """The high and low 32-bit words of `multiplier * value` for a 32-bit constant and a tensor
    of 32-bit words, through 16-bit halves of `value` so that no int64 product overflows."""
    low = multiplier * (value & 0xFFFF)
    upper = multiplier * (value >> 16) + (low >> 16)
    return upper >> 16, ((upper & 0xFFFF) << 16) | (low & 0xFFFF)


def blocks(seed, tag, draw_id, index):
    """[`block`] of every element of the int64 tensor `index`: the four words, each a tensor."""
    m0, m1, w0, w1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
    c0 = index.to(torch.int64)
    c1 = torch.full_like(c0, tag)
    c2 = torch.full_like(c0, draw_id & MASK32)
    c3 = torch.full_like(c0, (draw_id >> 32) & MASK32)
    k0, k1 = seed & MASK32, (seed >> 32) & MASK32
    for round_ in range(10):
        if round_ > 0:
            k0 = (k0 + w0) & MASK32
            k1 = (k1 + w1) & MASK32
        hi0, lo0 = mulhilo(m0, c0)
        hi1, lo1 = mulhilo(m1, c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return c0, c1, c2, c3


def uniforms(words):
    return (words.to(torch.float64) + 0.5) / 4294967296.0


class Packed:
    """The packed directory: its record, policy and the ids of both splits, as int32 tensors.
    The Packed reader of opponents/bert_tiny/bert_tiny.py, reading the token files with
    torch.from_file instead of one Python int per token."""

    def __init__(self, directory):
        directory = Path(directory)
        binding = json.loads((directory / "binding.json").read_text())
        record = json.loads((directory / "packed.json").read_text())
        self.seq_len = int(record["seq_len"])
        self.vocabulary = int(record["vocabulary_size"])
        self.baseline = record.get("heldout_unigram_baseline_nats")
        self.policy = binding["masking_policy"]
        self.ids = {
            role: self._read(directory / name)
            for role, name in (("Train", "train.tokens"), ("HeldOut", "heldout.tokens"))
        }
        self._orders = {}

    def _read(self, path):
        size = os.path.getsize(path)
        if size % 4 != 0:
            raise ValueError(f"{path}: {size} bytes is not a whole number of i32 tokens")
        if sys.byteorder != "little":
            raise ValueError("the token files are little-endian; this host is not")
        return torch.from_file(str(path), shared=False, size=size // 4, dtype=torch.int32)

    def sequences(self, role):
        return self.ids[role].numel() // self.seq_len

    def epoch_order(self, seed, epoch):
        """The Train sequences of `epoch` sorted by their 64-bit Philox key (word 1 high, word 0
        low), ties by index, as the scalar copy sorts (key, index) pairs."""
        if epoch not in self._orders:
            n = self.sequences("Train")
            index = torch.arange(n, dtype=torch.int64)
            w0, w1, _, _ = blocks(seed, TAG_DATA_ORDER, epoch, index)
            order = torch.sort(w0, stable=True).indices
            order = order[torch.sort(w1[order], stable=True).indices]
            self._orders[epoch] = order
        return self._orders[epoch]

    def batch(self, role, seed, step, batch_size):
        """ModelSpec's `mlm_batch`: (ids, labels, weights) of `B * S` rows, CPU tensors."""
        per_epoch = self.sequences(role) // batch_size
        index = step % per_epoch
        if role == "Train":
            order = self.epoch_order(seed, step // per_epoch)
            chosen = order[index * batch_size : (index + 1) * batch_size]
            draw_id = step
        else:
            chosen = torch.arange(index * batch_size, (index + 1) * batch_size, dtype=torch.int64)
            draw_id = (step + (1 << 63)) & ((1 << 64) - 1)
        random_tokens = self.vocabulary - SPECIALS
        rows = (chosen.reshape(-1, 1) * self.seq_len + torch.arange(self.seq_len)).reshape(-1)
        labels = self.ids[role][rows].to(torch.int64)
        w0, w1, w2, _ = blocks(seed, TAG_MASKING, draw_id, torch.arange(rows.numel()))
        selected = (labels >= SPECIALS) & (uniforms(w0) < self.policy["selection_rate"])
        u1 = uniforms(w1)
        mask = self.policy["mask_fraction"]
        random_fraction = self.policy["random_fraction"]
        random_ids = SPECIALS + ((w2 * random_tokens) >> 32)
        ids = torch.where(
            u1 < mask,
            torch.full_like(labels, MASK_TOKEN_ID),
            torch.where(u1 < mask + random_fraction, random_ids, labels),
        )
        ids = torch.where(selected, ids, labels)
        return ids, labels, selected.to(torch.float32)


def scalar_batch(packed, role, seed, step, batch_size, rows):
    """The first `rows` rows of [`Packed.batch`] by the scalar code of
    opponents/bert_tiny/bert_tiny.py, element by element; the Train sequences are those of the
    vectorised epoch order, whose keys [`check_draw`] checks."""
    per_epoch = packed.sequences(role) // batch_size
    index = step % per_epoch
    if role == "Train":
        order = packed.epoch_order(seed, step // per_epoch)
        chosen = [int(i) for i in order[index * batch_size : (index + 1) * batch_size]]
        draw_id = step
    else:
        chosen = list(range(index * batch_size, (index + 1) * batch_size))
        draw_id = (step + (1 << 63)) & ((1 << 64) - 1)
    random_tokens = packed.vocabulary - SPECIALS
    selection = packed.policy["selection_rate"]
    mask = packed.policy["mask_fraction"]
    random_fraction = packed.policy["random_fraction"]
    source = packed.ids[role]
    ids, labels, weights = [], [], []
    for row in range(rows):
        b, p = divmod(row, packed.seq_len)
        label = int(source[chosen[b] * packed.seq_len + p])
        w0, w1, w2, _ = block(seed, TAG_MASKING, draw_id, b * packed.seq_len + p)
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


def check_draw(packed, seed):
    """The vectorised draw against the scalar copy, refusing the run on a difference: the four
    words of the data-order stream at the first and last Train sequences and of the masking
    stream under a Train and a HeldOut draw id; the first epoch's first keys in ascending order;
    and the first rows of Train batch 0 and HeldOut batch 0."""
    n = packed.sequences("Train")
    held_out_id = 1 << 63
    for tag, draw_id, indices in (
        (TAG_DATA_ORDER, 0, list(range(CHECKED_ROWS)) + [n - 1]),
        (TAG_MASKING, 0, list(range(CHECKED_ROWS))),
        (TAG_MASKING, held_out_id, list(range(CHECKED_ROWS))),
    ):
        words = blocks(seed, tag, draw_id, torch.tensor(indices, dtype=torch.int64))
        got = list(zip(*(w.tolist() for w in words)))
        expected = [block(seed, tag, draw_id, i) for i in indices]
        if got != expected:
            raise ValueError(f"the vectorised Philox draw of stream {tag} differs from the "
                             "scalar draw")
    order = [int(i) for i in packed.epoch_order(seed, 0)[:CHECKED_ROWS]]
    keyed = [(((w[1] << 32) | w[0]), i) for i, w in
             ((i, block(seed, TAG_DATA_ORDER, 0, i)) for i in order)]
    if keyed != sorted(keyed):
        raise ValueError("the vectorised epoch order is not ascending in its Philox keys")
    for role in ("Train", "HeldOut"):
        ids, labels, weights = packed.batch(role, seed, 0, BATCH)
        expected = scalar_batch(packed, role, seed, 0, BATCH, CHECKED_ROWS)
        got = (
            ids[:CHECKED_ROWS].tolist(),
            labels[:CHECKED_ROWS].tolist(),
            weights[:CHECKED_ROWS].tolist(),
        )
        if got != expected:
            raise ValueError(f"the vectorised {role} batch 0 differs from the scalar draw")


def shapes():
    """The parameters of examples/bert_tiny.mspec, name and shape, in the file's order."""
    result = [
        ("tok_table", (VOCAB, HIDDEN)),
        ("pos_table", (SEQ, HIDDEN)),
        ("emb_ln_gamma", (HIDDEN,)),
        ("emb_ln_beta", (HIDDEN,)),
    ]
    for layer in range(LAYERS):
        p = f"l{layer}_"
        for name in ("q", "k", "v", "o"):
            result.append((f"{p}w{name}", (HIDDEN, HIDDEN)))
            result.append((f"{p}b{name}", (HIDDEN,)))
        result += [
            (f"{p}ln1_gamma", (HIDDEN,)),
            (f"{p}ln1_beta", (HIDDEN,)),
            (f"{p}w1", (HIDDEN, FFN)),
            (f"{p}b1", (FFN,)),
            (f"{p}w2", (FFN, HIDDEN)),
            (f"{p}b2", (HIDDEN,)),
            (f"{p}ln2_gamma", (HIDDEN,)),
            (f"{p}ln2_beta", (HIDDEN,)),
        ]
    result += [
        ("mlm_w", (HIDDEN, HIDDEN)),
        ("mlm_b", (HIDDEN,)),
        ("mlm_ln_gamma", (HIDDEN,)),
        ("mlm_ln_beta", (HIDDEN,)),
        ("mlm_bias", (VOCAB,)),
    ]
    return result


def initial(device, seed):
    """Torch's own draw of the file's initializers: Xavier-uniform matrices, zero biases and
    betas, unit gammas."""
    generator = torch.Generator(device=device).manual_seed(seed)
    params = {}
    for name, shape in shapes():
        if len(shape) == 2:
            bound = math.sqrt(6.0 / (shape[0] + shape[1]))
            value = (torch.rand(shape, device=device, generator=generator) * 2 - 1) * bound
        elif "gamma" in name:
            value = torch.ones(shape, device=device)
        else:
            value = torch.zeros(shape, device=device)
        params[name] = value.requires_grad_(True)
    return params


# The .msckpt reader, copied from opponents/transformer_block/transformer_block.py.

BLAKE3_IV = (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
             0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19)
BLAKE3_PERMUTATION = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
BLAKE3_ROUND = ((0, 4, 8, 12, 0, 1), (1, 5, 9, 13, 2, 3), (2, 6, 10, 14, 4, 5),
                (3, 7, 11, 15, 6, 7), (0, 5, 10, 15, 8, 9), (1, 6, 11, 12, 10, 11),
                (2, 7, 8, 13, 12, 13), (3, 4, 9, 14, 14, 15))
CHUNK_START, CHUNK_END, PARENT, ROOT = 1, 2, 4, 8
WORD = 0xFFFFFFFF


def blake3_compress(cv, words, counter, length, flags):
    """The BLAKE3 compression function: 7 rounds over the 16-word state, the message permuted
    between rounds; the 16 output words."""
    s = list(cv) + list(BLAKE3_IV[:4]) + [counter & WORD, (counter >> 32) & WORD, length, flags]
    m = list(words)
    for _ in range(7):
        for a, b, c, d, i, j in BLAKE3_ROUND:
            s[a] = (s[a] + s[b] + m[i]) & WORD
            v = s[d] ^ s[a]
            s[d] = (v >> 16) | ((v << 16) & WORD)
            s[c] = (s[c] + s[d]) & WORD
            v = s[b] ^ s[c]
            s[b] = (v >> 12) | ((v << 20) & WORD)
            s[a] = (s[a] + s[b] + m[j]) & WORD
            v = s[d] ^ s[a]
            s[d] = (v >> 8) | ((v << 24) & WORD)
            s[c] = (s[c] + s[d]) & WORD
            v = s[b] ^ s[c]
            s[b] = (v >> 7) | ((v << 25) & WORD)
        m = [m[k] for k in BLAKE3_PERMUTATION]
    for k in range(8):
        s[k] ^= s[k + 8]
        s[k + 8] ^= cv[k]
    return s


def blake3_node(data, first, last):
    """The output node of chunks first .. last - 1 of `data`, 1024 bytes each: a chunk's last
    block, or the parent of the largest power-of-two run of chunks on the left and the rest."""
    count = last - first
    if count == 1:
        chunk = data[first * 1024:last * 1024]
        blocks_ = [chunk[at:at + 64] for at in range(0, len(chunk), 64)] or [b""]
        cv = BLAKE3_IV
        for index, block_ in enumerate(blocks_):
            flags = (CHUNK_START if index == 0 else 0)
            words = struct.unpack("<16I", bytes(block_).ljust(64, b"\0"))
            if index == len(blocks_) - 1:
                return cv, words, first, len(block_), flags | CHUNK_END
            cv = blake3_compress(cv, words, first, 64, flags)[:8]
    left = 1 << ((count - 1).bit_length() - 1)
    words = tuple(blake3_compress(*blake3_node(data, first, first + left))[:8]) + tuple(
        blake3_compress(*blake3_node(data, first + left, last))[:8])
    return BLAKE3_IV, words, 0, 64, PARENT


def blake3_hex(data):
    """BLAKE3 of `data`, the unkeyed 32-byte hash in lower-case hex, as the `blake3` crate
    ModelSpec writes checksums with computes it; written out so the opponent needs only torch."""
    data = memoryview(data)
    cv, words, _, length, flags = blake3_node(data, 0, max(1, -(-len(data) // 1024)))
    return struct.pack("<8I", *blake3_compress(cv, words, 0, length, flags | ROOT)[:8]).hex()


def read_checkpoint(path, device):
    """The parameters and step of a ModelSpec .msckpt (DESIGN.md section 6.2): the 56-byte
    header, the compact JSON manifest at byte 56 and the little-endian f32 payload at the
    records' absolute offsets. It refuses, as ModelSpec's reader does, a wrong magic, version or
    endian marker or header checksum (E-CKPT-001: blake3 over the header, its checksum bytes 24 to
    55 counted as zero, and the manifest), a file that is not the manifest's `file_size`
    (E-CKPT-002), and a payload whose blake3 is not the manifest's `payload_hash` (E-CKPT-004:
    the bytes from the payload start, the end of the manifest rounded up to 64, to the end of
    the file)."""
    with open(path, "rb") as file:
        data = file.read()
    if len(data) < HEADER:
        raise ValueError(f"{path}: {len(data)} bytes, shorter than the {HEADER}-byte header")
    magic, version, endian, _reserved, size = struct.unpack_from("<4sIIIQ", data, 0)
    if magic != b"MSCK" or version != 1 or endian != 0x01020304:
        raise ValueError(f"{path}: not a version-1 little-endian .msckpt")
    if HEADER + size > len(data):
        raise ValueError(f"{path}: the manifest of {size} bytes runs past the end of the file")
    signed = bytearray(data[:HEADER + size])
    signed[24:HEADER] = bytes(HEADER - 24)
    if blake3_hex(signed) != data[24:HEADER].hex():
        raise ValueError(f"{path}: the header checksum differs from the header and manifest "
                         "(ModelSpec refuses this file with E-CKPT-001)")
    manifest = json.loads(data[HEADER:HEADER + size])
    if manifest.get("file_size") != len(data):
        raise ValueError(f"{path}: {len(data)} bytes, not the manifest's file_size "
                         f"{manifest.get('file_size')} (ModelSpec refuses this file with "
                         "E-CKPT-002)")
    payload_start = -(-(HEADER + size) // 64) * 64
    if blake3_hex(data[payload_start:]) != manifest.get("payload_hash"):
        raise ValueError(f"{path}: the payload hash differs from the payload (ModelSpec refuses "
                         "this file with E-CKPT-004)")
    records = {record["path"]: record for record in manifest["parameters"]}
    params = {}
    for name, shape in shapes():
        record = records.get(f"{MODEL}.{name}")
        if record is None:
            raise ValueError(f"{path}: no parameter {MODEL}.{name}")
        if record["dtype"] != "F32" or tuple(record["shape"]) != shape:
            raise ValueError(f"{path}: {MODEL}.{name} is {record['dtype']} {record['shape']}, "
                             f"not F32 {list(shape)}")
        values = array.array("f")
        values.frombytes(data[record["offset"]:record["offset"] + record["length"]])
        if sys.byteorder != "little":
            values.byteswap()
        tensor = torch.frombuffer(bytearray(values.tobytes()), dtype=torch.float32)
        params[name] = tensor.reshape(shape).to(device).requires_grad_(True)
    return params, manifest.get("step")


def gelu(x):
    """The tanh form, evaluated as ModelSpec evaluates it: x / (1 + exp(-2u))."""
    u = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
    return x / (1.0 + torch.exp(-2.0 * u))


def attention(q, k, v, key_mask):
    """ModelSpec's attention: rows b S .. b S + S - 1 are sequence b, head a reads columns
    a H/A .. (a + 1) H/A - 1, scores scaled by 1 / sqrt(H/A), masked keys excluded."""
    batch = key_mask.shape[0]
    d = HIDDEN // HEADS

    def heads(t):
        return t.reshape(batch, SEQ, HEADS, d).transpose(1, 2)

    scores = torch.matmul(heads(q), heads(k).transpose(-1, -2)) * (1.0 / math.sqrt(d))
    keep = (key_mask != 0).reshape(batch, 1, 1, SEQ)
    scores = scores.masked_fill(~keep, float("-inf"))
    probabilities = F.softmax(scores, dim=-1)
    out = torch.matmul(probabilities, heads(v))
    return out.transpose(1, 2).reshape(batch * SEQ, HIDDEN)


def layer(x, key_mask, p, i, training):
    def w(name):
        return p[f"l{i}_{name}"]

    q = torch.matmul(x, w("wq")) + w("bq")
    k = torch.matmul(x, w("wk")) + w("bk")
    v = torch.matmul(x, w("wv")) + w("bv")
    a = torch.matmul(attention(q, k, v, key_mask), w("wo")) + w("bo")
    a = F.dropout(a, DROPOUT, training)
    h = F.layer_norm(x + a, (HIDDEN,), w("ln1_gamma"), w("ln1_beta"), eps=EPS)
    f = torch.matmul(gelu(torch.matmul(h, w("w1")) + w("b1")), w("w2")) + w("b2")
    f = F.dropout(f, DROPOUT, training)
    return F.layer_norm(h + f, (HIDDEN,), w("ln2_gamma"), w("ln2_beta"), eps=EPS)


def forward(p, ids, positions, key_mask, training):
    """The logits [B S, V] of examples/bert_tiny.mspec; dropout only when `training`."""
    e = F.embedding(ids, p["tok_table"]) + F.embedding(positions, p["pos_table"])
    x = F.layer_norm(e, (HIDDEN,), p["emb_ln_gamma"], p["emb_ln_beta"], eps=EPS)
    x = F.dropout(x, DROPOUT, training)
    for i in range(LAYERS):
        x = layer(x, key_mask, p, i, training)
    m = gelu(torch.matmul(x, p["mlm_w"]) + p["mlm_b"])
    m = F.layer_norm(m, (HIDDEN,), p["mlm_ln_gamma"], p["mlm_ln_beta"], eps=EPS)
    return torch.matmul(m, p["tok_table"].t()) + p["mlm_bias"]


def train_loss(p, ids, positions, key_mask, labels, weights):
    """The weighted MLM cross-entropy of a training step, with dropout. The weight sum is a count
    of selected rows, so dividing by it clamped to 1 is the mean, and 0 when no row is selected."""
    nll = F.cross_entropy(forward(p, ids, positions, key_mask, True), labels, reduction="none")
    return (weights * nll).sum() / weights.sum().clamp(min=1.0)


def heldout_loss(params, heldout, positions, key_mask):
    """The weighted mean loss over every HeldOut batch without dropout: per-row nll in f32, both
    sums over the whole split in f64, as ModelSpec's `HeldOut::evaluate` sums them."""
    with torch.no_grad():
        numerator = torch.zeros((), dtype=torch.float64, device=positions.device)
        denominator = torch.zeros((), dtype=torch.float64, device=positions.device)
        for ids, labels, weights in heldout:
            logits = forward(params, ids, positions, key_mask, False)
            nll = F.cross_entropy(logits, labels, reduction="none")
            numerator += (weights.double() * nll.double()).sum()
            denominator += weights.double().sum()
        total = denominator.item()
        return 0.0 if total == 0.0 else float(numerator.item() / total)


def schedule(warmup, steps):
    """The multiplier of the peak rate at optimizer step e, counted from 0 (t = e + 1)."""

    def factor(e):
        t = e + 1
        if t <= warmup:
            return t / warmup
        return max(steps - t, 0) / (steps - warmup)

    return factor


def f32_text(value):
    """The shortest decimal that reads back as the same f32, as ModelSpec prints a loss."""
    bits = struct.pack("<f", value)
    for digits in range(1, 10):
        text = f"{value:.{digits}g}"
        if struct.pack("<f", float(text)) == bits:
            return text
    return repr(value)


def build():
    """The git commit of this checkout and whether its tracked files differ from it; a field git
    does not answer is left out."""
    here = os.path.dirname(os.path.abspath(__file__))

    def git(*args):
        try:
            done = subprocess.run(["git", "-C", here, *args], capture_output=True, text=True,
                                  check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return done.stdout

    result = {}
    head = git("rev-parse", "HEAD")
    if head and head.strip():
        result["commit"] = head.strip()
    status = git("status", "--porcelain", "--untracked-files=no")
    if status is not None:
        result["dirty"] = bool(status.strip())
    return result


def progress(text):
    print(f"bert_tiny_pretrain: {text}", file=sys.stderr, flush=True)


def heldout_batches(packed, seed, device):
    count = packed.sequences("HeldOut") // BATCH
    if count == 0:
        raise ValueError("the HeldOut split holds no full batch of 32 sequences")
    return [
        tuple(t.to(device) for t in packed.batch("HeldOut", seed, b, BATCH))
        for b in range(count)
    ]


def fixed_inputs(device):
    """The positions 0 .. S - 1 of every sequence and the all-ones key mask of a packed batch."""
    positions = torch.arange(SEQ, device=device).repeat(BATCH)
    key_mask = torch.ones(BATCH, SEQ, device=device)
    return positions, key_mask


def loss_check(args, device):
    """Prints `eval <step> loss <v>`: the held-out loss of the checkpoint's parameters."""
    packed = Packed(args.data)
    check_draw(packed, args.seed)
    params, step = read_checkpoint(args.loss_check, device)
    positions, key_mask = fixed_inputs(device)
    loss = heldout_loss(params, heldout_batches(packed, args.seed, device), positions, key_mask)
    print(f"eval {step if step is not None else 0} loss {f32_text(loss)}", flush=True)


def train(args, device, line):
    packed = Packed(args.data)
    if packed.seq_len != SEQ or packed.vocabulary != VOCAB:
        raise ValueError(f"{args.data}: sequences of {packed.seq_len} over {packed.vocabulary} "
                         f"tokens, not the example's {SEQ} over {VOCAB}")
    progress(f"{packed.sequences('Train')} Train and {packed.sequences('HeldOut')} HeldOut "
             "sequences; checking the draw against the scalar copy")
    check_draw(packed, args.seed)
    torch.manual_seed(args.seed)
    if args.checkpoint is None:
        params = initial(device, args.seed)
    else:
        params, step = read_checkpoint(args.checkpoint, device)
        if step not in (None, 0):
            raise ValueError(f"{args.checkpoint}: step {step}, not the initial parameters")
    positions, key_mask = fixed_inputs(device)
    heldout = heldout_batches(packed, args.seed, device)
    loss_fn = train_loss
    if args.mode == "compile":
        torch._dynamo.reset()
        loss_fn = torch.compile(train_loss, dynamic=False)
    leaves = [params[name] for name, _ in shapes()]
    optimizer = torch.optim.AdamW(leaves, lr=LR, betas=BETAS, eps=ADAM_EPS,
                                  weight_decay=WEIGHT_DECAY, foreach=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule(args.warmup_steps,
                                                                      args.steps))

    first = packed.batch("Train", args.seed, 0, BATCH)
    copies = {name: value.detach().clone().requires_grad_(True) for name, value in params.items()}
    ids, labels, weights = (t.to(device) for t in first)
    started = time.perf_counter()
    for _ in range(WARMUP_PASSES):
        loss_fn(copies, ids, positions, key_mask, labels, weights).backward()
    torch.cuda.synchronize(device)
    del copies
    progress(f"warm-up: {WARMUP_PASSES} forward and backward passes on a copy in "
             f"{time.perf_counter() - started:.3f} s ({args.mode})")

    trained = 0.0
    durations = []
    evals = []

    def evaluate(step):
        started = time.perf_counter()
        loss = heldout_loss(params, heldout, positions, key_mask)
        evals.append((step, loss, trained))
        print(json.dumps({"step": step, "heldout_loss": loss, "train_seconds": trained}),
              flush=True)
        progress(f"eval {step} heldout_loss {loss:.6f} train_seconds {trained:.3f} "
                 f"(evaluated in {time.perf_counter() - started:.3f} s)")

    batch = first
    for step in range(args.steps):
        if step % args.eval_every == 0:
            evaluate(step)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        ids, labels, weights = (t.to(device, non_blocking=True) for t in batch)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(params, ids, positions, key_mask, labels, weights)
        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        trained += seconds
        durations.append(seconds)
        if not math.isfinite(loss.item()):
            raise ValueError(f"step {step}: the training loss is {loss.item()}")
        if step + 1 < args.steps:
            batch = packed.batch("Train", args.seed, step + 1, BATCH)
    evaluate(args.steps)

    median = statistics.median(durations) * 1e6
    reached = None
    if args.target is not None:
        reached = next(((k, s) for k, loss, s in evals if loss <= args.target), None)
    line.update({
        "status": "measured",
        "steps": args.steps,
        "median_us_per_step": median,
        "tokens_per_s": BATCH * SEQ * 1e6 / median,
        "gflops": line["flop_per_step"] / (median * 1e3),
        "train_seconds": trained,
        "eval_every": args.eval_every,
        "target_loss": args.target,
        "steps_to_target": reached[0] if reached else None,
        "seconds_to_target": reached[1] if reached else None,
        "final_heldout_loss": evals[-1][1],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True,
                        help="the packed directory `modelspec data pack` wrote (packed.json, "
                             "binding.json, train.tokens, heldout.tokens)")
    parser.add_argument("--mode", choices=("eager", "compile"), default="eager",
                        help="eager PyTorch or torch.compile(dynamic=False) (default eager)")
    parser.add_argument("--device", default="cuda:0", help="the CUDA device (default cuda:0)")
    parser.add_argument("--steps", type=int, default=10000, help="training steps (default 10000)")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="the steps of the linear warm-up (default 1000)")
    parser.add_argument("--eval-every", type=int, default=100,
                        help="evaluate the HeldOut split at every multiple of this step, at "
                             "step 0 and at the last step (default 100)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="the run seed of the batches and of torch's draws (default 42, the "
                             "example's)")
    parser.add_argument("--target", type=float, metavar="LOSS",
                        help="the held-out loss to time: the result gives the first evaluated "
                             "step at or below it and the training seconds to it")
    parser.add_argument("--checkpoint", metavar="PATH.msckpt",
                        help="start from the parameters of this ModelSpec checkpoint of step 0 "
                             "instead of torch's own Xavier-uniform draw")
    parser.add_argument("--loss-check", metavar="PATH.msckpt",
                        help="print `eval <step> loss <v>`, the held-out loss of this "
                             "checkpoint's parameters without dropout, and train nothing")
    args = parser.parse_args()
    if not 0 <= args.warmup_steps < args.steps:
        parser.error("--warmup-steps must be below --steps")
    if args.eval_every < 1:
        parser.error("--eval-every must be at least 1")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)

    if args.loss_check is not None:
        try:
            loss_check(args, device)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            print(f"bert_tiny_pretrain: refused: {type(error).__name__}: {error}",
                  file=sys.stderr)
            sys.exit(1)
        return

    torch.cuda.set_device(device)
    built = build()
    if built.get("dirty") is not False:
        progress(f"warning: not a clean checkout: {built}")
    line = {
        "side": f"pytorch-{args.mode}",
        "workload": WORKLOAD,
        "batch": BATCH,
        "seq": SEQ,
        "tokens_per_step": BATCH * SEQ,
        "flop_per_step": flop_per_step(BATCH, SEQ),
        "flop_rule": FLOP_RULE,
        "device": (f"{torch.cuda.get_device_name(device)}; torch {torch.__version__}; "
                   f"cuda {torch.version.cuda}; tf32 off; mode {args.mode}"),
        "build": built,
    }
    progress(f"{line['side']}: {args.steps} steps of {BATCH * SEQ} tokens, "
             f"{line['flop_per_step']} FLOP per step")
    try:
        train(args, device, line)
    except Exception as error:  # the refusal is the result line
        print(f"bert_tiny_pretrain: refused: {type(error).__name__}: {error}", file=sys.stderr)
        line.update({"status": "refused", "refusal": f"{type(error).__name__}: {error}"})
        print(json.dumps(line), flush=True)
        sys.exit(1)
    print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
