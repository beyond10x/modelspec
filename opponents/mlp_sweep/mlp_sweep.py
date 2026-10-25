"""The PyTorch CUDA opponent of the ModelSpec MLP size sweep.

story:mlp-size-sweep-vs-pytorch-cuda. The model, sizes and data law match
crates/modelspec-runtime/benches/mlp_sweep.rs: 784 -> H ReLU -> 10, no bias,
f32, MSE, SGD at lr 0.001, hidden 64, 256, 1024, 4096 and batch 100, 1000,
8192. The batch is synthetic teacher-student data (a standard normal input,
the target the prediction of a teacher of the same shape with Xavier-uniform
weights), made on the GPU once per point, as the ModelSpec runner uploads its
batch once.

Each point runs eager and torch.compile (dynamic=False, after a dynamo reset),
on CUDA with TF32 off. Per point and mode: a 20-step warm-up (which also
compiles), calibration runs until one lasts at least --min-seconds, then 3
timed runs of that many steps. The timed span is host wall time from a CUDA
synchronize of --device before the first step to one after the last: it holds
the steps alone, no setup, data generation or warm-up. Each point and mode
prints one JSON line on stdout, in the shape the bench writes and `--crossover`
reads, naming the git commit of this checkout and whether its tracked files
differ from it; a point that fails prints its refusal and the sweep goes on.
Progress goes to stderr.

ModelSpec never runs this file (AGENTS.md, Rules: Python is for opponents only).
"""

import argparse
import contextlib
import json
import math
import os
import statistics
import subprocess
import sys
import time

import torch

INPUTS = 784
OUTPUTS = 10
HIDDEN = (64, 256, 1024, 4096)
BATCHES = (100, 1000, 8192)
LR = 0.001
SEED = 42
WARMUP_STEPS = 20
RUNS = 3
CALIBRATIONS = 4
FLOP_RULE = "matmul-only"


def flop_per_step(batch, hidden):
    """The matmul FLOP of one step, 3196 * B * H, as the bench counts it."""
    return 2 * batch * hidden * (INPUTS + 3 * OUTPUTS + INPUTS)


def steps_for(seconds, steps, target):
    """The steps a run needs to last `target` seconds; at least the warm-up."""
    return max(math.ceil(target / (seconds / steps)), WARMUP_STEPS)


class Mlp(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.w1 = torch.nn.Linear(INPUTS, hidden, bias=False)
        self.w2 = torch.nn.Linear(hidden, OUTPUTS, bias=False)
        torch.nn.init.xavier_uniform_(self.w1.weight)
        torch.nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x):
        return self.w2(torch.relu(self.w1(x)))


class Loss(torch.nn.Module):
    """The model and its MSE loss: what torch.compile compiles."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, y):
        return torch.nn.functional.mse_loss(self.model(x), y)


def point_data(batch, hidden, device):
    generator = torch.Generator(device=device).manual_seed(SEED)
    x = torch.randn(batch, INPUTS, device=device, generator=generator)
    torch.manual_seed(SEED + 1)
    teacher = Mlp(hidden).to(device)
    with torch.no_grad():
        y = teacher(x)
    return x, y.detach()


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


def measure(batch, hidden, mode, compile_mode, device, min_seconds, profile=False):
    torch.manual_seed(SEED)
    model = Mlp(hidden).to(device)
    loss_fn = Loss(model)
    if mode == "compile":
        torch._dynamo.reset()
        loss_fn = torch.compile(loss_fn, mode=compile_mode, dynamic=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, foreach=True)
    x, y = point_data(batch, hidden, device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(x, y)
        loss.backward()
        optimizer.step()

    target = min_seconds * 1.2
    seconds, _ = run(step, WARMUP_STEPS, device)
    steps = steps_for(seconds, WARMUP_STEPS, target)
    for _ in range(CALIBRATIONS):
        seconds, _ = run(step, steps, device)
        print(f"mlp_sweep:   calibration: {steps} steps in {seconds:.3f} s", file=sys.stderr)
        if seconds >= min_seconds:
            break
        steps = max(steps_for(seconds, steps, target), steps + 1)
    runs = []
    with kernel_profiler(profile) as profiler:
        for _ in range(RUNS):
            seconds, span = run(step, steps, device)
            us = seconds * 1e6 / steps
            print(
                f"mlp_sweep:   timed: {steps} steps in {seconds:.3f} s, {us:.3f} us per step",
                file=sys.stderr,
            )
            if seconds < min_seconds:
                print(f"mlp_sweep:   warning: the run was shorter than {min_seconds} s",
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
            f"mlp_sweep:   profile kernel calls {count} total_us {total:.3f} "
            f"mean_us {total / count:.3f} us_per_step {total / steps:.3f} name {key}",
            file=sys.stderr,
        )
    summed = sum(row[0] for row in rows)
    print(
        f"mlp_sweep:   profile steps {steps} device_us_per_step {summed / steps:.3f}",
        file=sys.stderr,
    )


def parse_point(text):
    batch, hidden = text.split(":")
    return int(batch), int(hidden)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda:0", help="the CUDA device (default cuda:0)")
    parser.add_argument("--min-seconds", type=float, default=5.0,
                        help="the shortest timed run in seconds (default 5)")
    parser.add_argument("--point", action="append", type=parse_point, default=[],
                        help="run only BATCH:HIDDEN; repeatable (default all 12)")
    parser.add_argument("--modes", default="eager,compile",
                        help="comma-separated: eager, compile (default both)")
    parser.add_argument("--compile-mode", default="default",
                        help="the torch.compile mode (default 'default')")
    parser.add_argument("--profile", action="store_true",
                        help="run the timed runs under torch.profiler and print the CUDA time of "
                             "each kernel over them to stderr; the timed figures then carry the "
                             "profiler's cost and are not the sweep's")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    built = build()
    if built.get("dirty") is not False:
        print(f"mlp_sweep: warning: not a clean checkout: {built}", file=sys.stderr)
    named = (
        f"{torch.cuda.get_device_name(device)}; torch {torch.__version__}; "
        f"cuda {torch.version.cuda}; tf32 off; torch.compile mode {args.compile_mode}"
    )
    points = args.point or [(b, h) for b in BATCHES for h in HIDDEN]
    modes = [m for m in args.modes.split(",") if m]
    for batch, hidden in points:
        for mode in modes:
            side = f"pytorch-{mode}"
            print(f"mlp_sweep: {side} batch {batch} hidden {hidden}", file=sys.stderr)
            line = {
                "side": side,
                "batch": batch,
                "hidden": hidden,
                "flop_per_step": flop_per_step(batch, hidden),
                "flop_rule": FLOP_RULE,
                "device": named,
                "build": built,
            }
            try:
                if mode not in ("eager", "compile"):
                    raise ValueError(f"unknown mode {mode!r}")
                steps, runs = measure(batch, hidden, mode, args.compile_mode, device,
                                      args.min_seconds, args.profile)
                median = statistics.median(r["us_per_step"] for r in runs)
                line.update({
                    "status": "measured",
                    "steps": steps,
                    "runs": runs,
                    "median_us_per_step": median,
                    "gflops": line["flop_per_step"] / (median * 1e3),
                })
            except Exception as error:  # a refused point is a result, not the end of the sweep
                print(f"mlp_sweep: refused: {error}", file=sys.stderr)
                line.update({"status": "refused", "refusal": f"{type(error).__name__}: {error}"})
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
