"""The PyTorch CUDA opponent of the ModelSpec transformer-block benchmark.

story:transformer-block-benchmark. The model, sizes and data law match
examples/transformer_block.mspec, which crates/modelspec-runtime/benches/transformer_block.rs
runs: two post-layer-norm encoder layers of BERT-tiny (hidden 128, 2 heads, feed-forward 512 with
the tanh-form GELU, sequences of 128, no dropout), the mean squared error of the second layer's
output against a target of its shape, f32, SGD at lr 0.01, at batches of 8, 32 and 128
sequences. Every operation is written out with torch and torch.nn.functional as ModelSpec
defines it (DESIGN.md section 2.1): a weight is [in, out] and multiplies from the right, layer
norm divides by sqrt(var + 1e-12), the GELU is x / (1 + exp(-2u)) with
u = sqrt(2/pi) (x + 0.044715 x^3), and head a of 2 reads columns 64a to 64a + 63 of q, k and v,
each query attending to the keys of its own sequence whose mask entry is 1.

The batch is synthetic, made on the GPU once per point, as the ModelSpec runner uploads its batch
once: x standard normal, the key mask all ones, and the target the prediction of a teacher of the
same shapes with Xavier-uniform weights. The parameters are torch's own Xavier-uniform draw under
the seed, or the parameters of a ModelSpec checkpoint (--checkpoint).

Each point runs eager and torch.compile (dynamic=False, after a dynamo reset), on CUDA with TF32
off. Per point and mode: a 20-step warm-up (which also compiles), calibration runs until one
lasts at least --min-seconds, then 3 timed runs of that many steps. The timed span is host wall
time from a CUDA synchronize of --device before the first step to one after the last. Each point
and mode prints one JSON line on stdout, in the shape the bench writes; a point that fails prints
its refusal and the run goes on. Progress goes to stderr.

--loss-check <path.msckpt> --batch <dir> prints instead the loss at step 0 of the parameters of a
ModelSpec checkpoint on the batch ModelSpec wrote with `train --write-batch <dir>`, as the line
`step 0 loss <v>` ModelSpec prints, and trains nothing.

ModelSpec never runs this file (AGENTS.md, Rules: Python is for opponents only).
"""

import argparse
import array
import contextlib
import json
import math
import os
import statistics
import struct
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

SEQ = 128
HIDDEN = 128
HEADS = 2
FFN = 512
LAYERS = 2
BATCHES = (8, 32, 128)
LR = 0.01
EPS = 1e-12
SEED = 42
WARMUP_STEPS = 20
RUNS = 3
CALIBRATIONS = 4
FLOP_RULE = "matmul-only"
MODEL = "TransformerBlock"
HEADER = 56


def flop_per_step(batch):
    """The matmul FLOP of one step, N (42 H^2 + 24 H F + 24 S H) with N = B S, as the bench
    counts it (benches/transformer_block/block.rs): 339,738,624 B."""
    n = batch * SEQ
    return n * (42 * HIDDEN * HIDDEN + 24 * HIDDEN * FFN + 24 * SEQ * HIDDEN)


def steps_for(seconds, steps, target):
    """The steps a run needs to last `target` seconds; at least the warm-up."""
    return max(math.ceil(target / (seconds / steps)), WARMUP_STEPS)


def shapes():
    """The parameters of the model, name and shape, in the order of the .mspec file."""
    result = []
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
    return result


def initial(device, seed):
    """Torch's own draw of the file's initializers: Xavier-uniform weights, zero biases and
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


def read_checkpoint(path, device):
    """The parameters of a ModelSpec .msckpt (DESIGN.md section 6.2): the 56-byte header, the
    compact JSON manifest at byte 56 and the little-endian f32 payload at the records' absolute
    offsets. The header's magic, version and endian marker are checked; the checksums are not."""
    with open(path, "rb") as file:
        data = file.read()
    magic, version, endian, _reserved, size = struct.unpack_from("<4sIIIQ", data, 0)
    if magic != b"MSCK" or version != 1 or endian != 0x01020304:
        raise ValueError(f"{path}: not a version-1 little-endian .msckpt")
    manifest = json.loads(data[HEADER:HEADER + size])
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


def read_tensor(path, shape, device):
    """A dense little-endian f32 file of `shape`, as `train --write-batch` writes it."""
    with open(path, "rb") as file:
        data = file.read()
    expected = 4 * math.prod(shape)
    if len(data) != expected:
        raise ValueError(f"{path}: {len(data)} bytes, not {expected} for {list(shape)}")
    values = array.array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    tensor = torch.frombuffer(bytearray(values.tobytes()), dtype=torch.float32)
    return tensor.reshape(shape).to(device)


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


def layer(x, key_mask, p, i):
    def w(name):
        return p[f"l{i}_{name}"]

    q = torch.matmul(x, w("wq")) + w("bq")
    k = torch.matmul(x, w("wk")) + w("bk")
    v = torch.matmul(x, w("wv")) + w("bv")
    a = torch.matmul(attention(q, k, v, key_mask), w("wo")) + w("bo")
    h = F.layer_norm(x + a, (HIDDEN,), w("ln1_gamma"), w("ln1_beta"), eps=EPS)
    f = torch.matmul(gelu(torch.matmul(h, w("w1")) + w("b1")), w("w2")) + w("b2")
    return F.layer_norm(h + f, (HIDDEN,), w("ln2_gamma"), w("ln2_beta"), eps=EPS)


def forward(params, x, key_mask):
    for i in range(LAYERS):
        x = layer(x, key_mask, params, i)
    return x


def loss_of(params, x, key_mask, y):
    """mse(forward(x, key_mask), y): sum((a - b)^2) / element count."""
    return F.mse_loss(forward(params, x, key_mask), y)


def point_data(batch, device):
    generator = torch.Generator(device=device).manual_seed(SEED)
    x = torch.randn(batch * SEQ, HIDDEN, device=device, generator=generator)
    key_mask = torch.ones(batch, SEQ, device=device)
    teacher = initial(device, SEED + 1)
    with torch.no_grad():
        y = forward(teacher, x, key_mask)
    return x, key_mask, y.detach()


def build():
    """The git commit of this checkout and whether its tracked files differ from it, as the
    bench records them; a field git does not answer is left out."""
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


def run(step, steps, device):
    """`steps` steps between two CUDA synchronizes of `device`: seconds and the Unix-ms span."""
    torch.cuda.synchronize(device)
    start_ms = int(time.time() * 1000)
    started = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    end_ms = int(time.time() * 1000)
    return seconds, [start_ms, end_ms]


def measure(batch, mode, compile_mode, device, min_seconds, checkpoint, profile=False):
    if checkpoint is None:
        params = initial(device, SEED)
    else:
        params, _ = read_checkpoint(checkpoint, device)
    names = [name for name, _ in shapes()]
    leaves = [params[name] for name in names]
    loss_fn = loss_of
    if mode == "compile":
        torch._dynamo.reset()
        loss_fn = torch.compile(loss_of, mode=compile_mode, dynamic=False)
    optimizer = torch.optim.SGD(leaves, lr=LR, foreach=True)
    x, key_mask, y = point_data(batch, device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(params, x, key_mask, y)
        loss.backward()
        optimizer.step()

    target = min_seconds * 1.2
    seconds, _ = run(step, WARMUP_STEPS, device)
    print(f"transformer_block:   warm-up: {WARMUP_STEPS} steps in {seconds:.3f} s",
          file=sys.stderr)
    steps = steps_for(seconds, WARMUP_STEPS, target)
    for _ in range(CALIBRATIONS):
        seconds, _ = run(step, steps, device)
        print(f"transformer_block:   calibration: {steps} steps in {seconds:.3f} s",
              file=sys.stderr)
        if seconds >= min_seconds:
            break
        steps = max(steps_for(seconds, steps, target), steps + 1)
    runs = []
    with kernel_profiler(profile) as profiler:
        for _ in range(RUNS):
            seconds, span = run(step, steps, device)
            us = seconds * 1e6 / steps
            print(
                f"transformer_block:   timed: {steps} steps in {seconds:.3f} s, "
                f"{us:.3f} us per step",
                file=sys.stderr,
            )
            if seconds < min_seconds:
                print(f"transformer_block:   warning: the run was shorter than {min_seconds} s",
                      file=sys.stderr)
            runs.append({"us_per_step": us, "seconds": seconds, "span_unix_ms": span})
    if profiler is not None:
        print_kernels(profiler, steps * RUNS)
    return steps, runs


@contextlib.contextmanager
def kernel_profiler(enabled):
    """torch.profiler over the timed runs, CUDA activity only, when `enabled`; else nothing."""
    if not enabled:
        yield None
        return
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        yield profiler


def print_kernels(profiler, steps):
    """One stderr line per CUDA kernel, memset or copy of the timed runs, the most device time
    first: calls, total device us, mean us per call and us per timed step; then their sum."""
    rows = []
    for event in profiler.key_averages():
        total = event.self_device_time_total
        if total <= 0:
            continue
        rows.append((total, event.count, event.key))
    rows.sort(key=lambda row: -row[0])
    for total, count, key in rows:
        print(
            f"transformer_block:   profile kernel calls {count} total_us {total:.3f} "
            f"mean_us {total / count:.3f} us_per_step {total / steps:.3f} name {key}",
            file=sys.stderr,
        )
    summed = sum(row[0] for row in rows)
    print(
        f"transformer_block:   profile steps {steps} device_us_per_step {summed / steps:.3f}",
        file=sys.stderr,
    )


def f32_text(value):
    """The shortest decimal that reads back as the same f32, as ModelSpec prints a loss."""
    bits = struct.pack("<f", value)
    for digits in range(1, 10):
        text = f"{value:.{digits}g}"
        if struct.pack("<f", float(text)) == bits:
            return text
    return repr(value)


def loss_check(checkpoint, batch_dir, device):
    """Prints `step 0 loss <v>`: the loss of the checkpoint's parameters on ModelSpec's batch."""
    params, step = read_checkpoint(checkpoint, device)
    size = os.path.getsize(os.path.join(batch_dir, "x.f32"))
    rows = size // (4 * HIDDEN)
    if rows % SEQ != 0 or rows == 0:
        raise ValueError(f"{batch_dir}/x.f32: {rows} rows is not a whole number of sequences")
    batch = rows // SEQ
    x = read_tensor(os.path.join(batch_dir, "x.f32"), (rows, HIDDEN), device)
    key_mask = read_tensor(os.path.join(batch_dir, "key_mask.f32"), (batch, SEQ), device)
    y = read_tensor(os.path.join(batch_dir, "y.f32"), (rows, HIDDEN), device)
    with torch.no_grad():
        loss = loss_of(params, x, key_mask, y).item()
    print(f"step {step if step is not None else 0} loss {f32_text(loss)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda:0", help="the CUDA device (default cuda:0)")
    parser.add_argument("--min-seconds", type=float, default=5.0,
                        help="the shortest timed run in seconds (default 5)")
    parser.add_argument("--point", action="append", type=int, default=[], metavar="BATCH",
                        help="run only this batch of sequences; repeatable (default 8, 32, 128)")
    parser.add_argument("--modes", default="eager,compile",
                        help="comma-separated: eager, compile (default both)")
    parser.add_argument("--compile-mode", default="default",
                        help="the torch.compile mode (default 'default')")
    parser.add_argument("--checkpoint", metavar="PATH.msckpt",
                        help="start every point from the parameters of this ModelSpec checkpoint "
                             "instead of torch's own Xavier-uniform draw")
    parser.add_argument("--loss-check", metavar="PATH.msckpt",
                        help="print `step 0 loss <v>` of this checkpoint's parameters on the "
                             "batch in --batch and train nothing")
    parser.add_argument("--batch", metavar="DIR",
                        help="with --loss-check: the directory `modelspec train --write-batch` "
                             "wrote (x.f32, key_mask.f32, y.f32)")
    parser.add_argument("--profile", action="store_true",
                        help="run the timed runs under torch.profiler and print the CUDA time of "
                             "each kernel over them to stderr; the timed figures then carry the "
                             "profiler's cost and are not the benchmark's")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)

    if args.loss_check is not None:
        if args.batch is None:
            parser.error("--loss-check needs --batch DIR")
        try:
            loss_check(args.loss_check, args.batch, device)
        except (OSError, ValueError, KeyError) as error:
            print(f"transformer_block: refused: {type(error).__name__}: {error}", file=sys.stderr)
            sys.exit(1)
        return

    torch.cuda.set_device(device)
    built = build()
    if built.get("dirty") is not False:
        print(f"transformer_block: warning: not a clean checkout: {built}", file=sys.stderr)
    named = (
        f"{torch.cuda.get_device_name(device)}; torch {torch.__version__}; "
        f"cuda {torch.version.cuda}; tf32 off; torch.compile mode {args.compile_mode}"
    )
    points = args.point or list(BATCHES)
    modes = [m for m in args.modes.split(",") if m]
    for batch in points:
        for mode in modes:
            side = f"pytorch-{mode}"
            print(f"transformer_block: {side} batch {batch} ({batch * SEQ} tokens): "
                  f"{flop_per_step(batch)} FLOP per step", file=sys.stderr)
            line = {
                "side": side,
                "batch": batch,
                "seq": SEQ,
                "tokens_per_step": batch * SEQ,
                "flop_per_step": flop_per_step(batch),
                "flop_rule": FLOP_RULE,
                "device": named,
                "build": built,
            }
            try:
                if mode not in ("eager", "compile"):
                    raise ValueError(f"unknown mode {mode!r}")
                if batch < 1:
                    raise ValueError(f"batch {batch} is not positive")
                steps, runs = measure(batch, mode, args.compile_mode, device, args.min_seconds,
                                      args.checkpoint, args.profile)
                median = statistics.median(r["us_per_step"] for r in runs)
                line.update({
                    "status": "measured",
                    "steps": steps,
                    "runs": runs,
                    "median_us_per_step": median,
                    "tokens_per_s": batch * SEQ * 1e6 / median,
                    "gflops": line["flop_per_step"] / (median * 1e3),
                })
            except Exception as error:  # a refused point is a result, not the end of the run
                print(f"transformer_block: refused: {error}", file=sys.stderr)
                line.update({"status": "refused", "refusal": f"{type(error).__name__}: {error}"})
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
