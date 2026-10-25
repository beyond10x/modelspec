# story:sweep-run-starts-busy: diagnosis

RTX 3090, driver 610.57.04, power limit 420 W, clocks never locked. One GPU job at a time; the
GPU had no other compute process before any run (`*.apps-before.txt`, all empty). `<scratch>` is
the unit's scratch directory, `<worktree>` the branch checkout `impl/sweep-run-starts-busy`.

## Verdict

The dip is not introduced by a commit. Every candidate binary from 92ba28b to 74a9520 dips in
one session, and the block-kernel unit's own dip-free binary dipped too, in its third round
(`../2026-10-25-blockperf/sweep/r3-final.*`: 8,540 / 8,632 / 8,493 µs, SM clock down to 1,005 to
1,095 MHz in each run). Its first round was dip-free because the driver held the GPU at P0
between processes then (`# before: … 1920 MHz, P0` in `r1-final.stderr.txt`); before round 3 it
idled at P3 (`# before: … 780 MHz, P3`), as it idles at P3 or P8 today.

The cause is the harness's per-run setup. Between one run's host wait and the next run's first
submit the host spent about 150 ms (b8192 h4096) on the read-back, the release of the run's
buffers and pipelines, the parameter draws, SPIR-V emission, pipeline creation, buffer
allocation and the staging writes, with the GPU idle. In that state the driver drops to P3
(SM 780 to 870 MHz, memory 5,001 MHz) about 0.1 to 0.3 s after the next run's first submit and
needs two to three seconds to climb back to about 1,700 MHz. The GPU is busy through every run:
there is no host wait inside a run.

## The loop (idle-reproduced)

```
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,utilization.gpu,pstate,temperature.gpu --format=csv -lms 100 > <name>.gpu.csv &
mlp_sweep --bench --point 8192:4096 --min-seconds 6 > <name>.jsonl 2> <name>.stderr.txt
```

The bench is `cargo bench -p modelspec-runtime --features vulkan --bench mlp_sweep`, built with
`--no-run` and started directly, cwd `crates/modelspec-runtime` of its tree, so the sampler window
holds no cargo. Each timed span of the JSONL is aligned with the samples by its unix-ms ends. The
driver refreshes the clock reading every 0.5 s.

Binaries (`<scratch>/bin/mlp_sweep-<commit>`): each commit's tree from
`git archive <commit> | tar -x -C <scratch>/tree-<commit>`, built there with
`b10x-build-slot cargo bench -p modelspec-runtime --features vulkan --bench mlp_sweep --no-run`
into its own `target/`. `fix` is `<worktree>` with this unit's change.

Red, 74a9520, run 2 of the final session (`final/s1-74a9520.gpu.csv`; offset from the span's
start, SM clock, memory clock, power, utilization, P-state):

```
run2   -0.06s 1740 MHz | 9751 MHz | 418.38 W | 100 % | P0
run2   +0.04s 1740 MHz | 9751 MHz | 418.38 W | 100 % | P0
run2   +0.14s 780 MHz | 5001 MHz | 363.39 W | 100 % | P3
run2   +0.64s 915 MHz | 5001 MHz | 251.08 W | 100 % | P3
run2   +1.14s 1095 MHz | 5001 MHz | 180.83 W | 100 % | P3
run2   +1.65s 1215 MHz | 5001 MHz | 190.10 W | 100 % | P3
```

Green, the fix, run 2 of the same session (`final/s4-fix.gpu.csv`):

```
run2   -0.03s 1680 MHz | 9751 MHz | 418.01 W | 100 % | P0
run2   +0.37s 1650 MHz | 9751 MHz | 404.83 W | 100 % | P0
run2   +0.87s 1695 MHz | 9751 MHz | 400.24 W | 100 % | P0
```

## Bisect (idle-bisected)

One session, oldest first (`bisect/*-r2.*`), and the first interleaved pair (`bisect/*-r1.*`).
Gap: ms from one timed span's end to the next one's start, about the GPU's idle.

| commit | what | dip | lowest SM clock after 1 s, runs 1/2/3 (MHz) | µs per step, median | gaps (ms) |
|---|---|---|---|---|---|
| 92ba28b | block-kernel unit's parent | yes | 1050 / 975 / 1155 | 9,240.7 | 155 / 152 |
| 238240f | main before the block kernel | yes | 1095 / 975 / 990 | 9,459.4 | 154 / 156 |
| 40dd797 | block-kernel unit | yes | 1005 / 1065 / 1035 | 8,156.1 | 139 / 153 |
| e34687f | its adversary and fix pass | yes | 1110 / 1035 / 1050 | 8,189.0 | 157 / 158 |
| b999009 | its merge | yes | 1065 / 1020 / 1005 | 8,153.0 | 157 / 158 |
| fbadb51 | its plan | yes | 1095 / 1035 / 945 | 8,176.9 | 156 / 153 |
| 1095ecb | layer norm | yes | 1020 / 1020 / 1035 | 8,055.3 | 142 / 138 |
| fca5377 | chain fusion | yes | 930 / 975 / 1095 | 9,368.7 | 136 / 135 |
| bbb8893 | its merge | yes | 990 / 1035 / 1020 | 8,070.2 | 154 / 150 |
| 74a9520 | main | yes | 975 / 975 / 1050 | 8,037.7 | 139 / 140 |
| 40dd797 (r1) | interleaved with main | yes | 1035 / 1095 / 990 | 7,864.5 | 155 / 140 |
| 74a9520 (r1) | interleaved | yes | 1125 / 1035 / 975 | 7,942.0 | 140 / 165 |

No commit separates a dip from no dip, so there is no commit that introduced it; the µs figure
follows the kernels (the block kernel lands at 40dd797, the fusion's parent 238240f lacks it).

## Hypotheses

| hypothesis | probe | observation | verdict |
|---|---|---|---|
| H0: a commit between the block-kernel tree and 74a9520 added the idle | bisect above; the block-kernel unit's own records | every binary dips; its own binary dipped in its round 3 | refuted |
| H1: a fence wait or read-back at the start of a run drains the GPU | timestamps of every host wait and phase in `execute` (`probe/debug-probes.patch`, `probe/phases-r2.stderr.txt`) | one wait per run, at its end; submits 1.6 ms apart, each chunk 256 steps (about 2 s of GPU work) | refuted |
| H2: the warm-up and calibration leave the GPU idle before the timed window | the same timestamps across all runs of a point | every run, timed or not, is preceded by the same setup; the dip follows each of them | refuted as a harness-phase cause; the gap is per run |
| H3: per-run setup on the host with the GPU idle | phase timestamps between `wait done` and the next first submit | 138 to 156 ms: read-back 35 to 53, check 4, release 12, draws 35, SPIR-V 3, pipelines 2, buffers 7, staging alloc 9, staging write 22, descriptors 3.5, first chunk 1.6 | confirmed |
| H4: the upload waits on a staging path | phase timestamps | the upload is a copy in the run's first command buffer, inside the span; only the 22 ms staging write is before it | part of H3 |
| H5: the first submit is small and the GPU outruns the recording | submits per run and recording time | the first chunk records in 1.6 ms and holds 256 steps | refuted |
| H6: the idle before a run, not its content, sets the dip | an added sleep before each run at 8192:256, one variable (`probe/sleep-8192x256-<added ms>`: 0, 25, 50, 100, 200 ms added give the idles below) | idle 40 ms: lowest 1845 MHz, 1,244.6 µs; 66 ms: 1590, 1,268.8; 91 ms: 1320, 1,326.6; 141 ms: 915, 1,434.7; 240 ms: 885, 1,518.6 | confirmed |

## Mechanism and its observation

The runtime observation is the phase trace at b8192 h4096 on 74a9520's code with timestamps
added (`probe/phases-r2.stderr.txt`, `[DEBUG-sweepidle]` lines 98 to 115: the second timed run's
end to the third one's first submit; ms since the previous line):

```
     -    wait done
   0.0 ms stats, profile
  35.2 ms download read
   1.0 ms parameters built
   3.2 ms read back, checked
   0.1 ms run built
  12.3 ms harness: train_with returned
   0.1 ms harness: train_with called
   3.0 ms execute start
  34.7 ms params drawn
   3.1 ms spirv emitted
   1.8 ms pipelines created
   7.3 ms buffers allocated
   9.1 ms staging allocated
  22.2 ms staging written
   3.5 ms descriptors bound, pools made
   1.6 ms submit chunk
```

138.2 ms in all; the two gaps before it are 153.9 and 155.5 ms (the download read 49.1 and 52.5
ms). The GPU idles from `wait done` to the next `submit chunk` before every run; the
sleep probe shows that an idle of that length, and nothing else, puts the SM clock below
1,000 MHz 0.1 to 0.3 s into the following run.

## Fix

- `vulkan::runner::train_repeatedly`: one setup (SPIR-V, pipelines, buffers, descriptor sets,
  parameter draws, upload staging buffer), then run after run, each from step 0 with its own
  step count and the same initial parameters and batch, copied up inside its span. Only the
  losses buffer, the read-back buffer and the timestamp pools are made per run.
- `Device::download_buffer`: the read-back buffer in host-visible, host-coherent, host-cached
  memory where the device has it (the read of 13 MB took 35 to 53 ms from uncached memory).
- The bench runs a point's warm-up, calibration and timed runs as one series
  (`benches/mlp_sweep/runs.rs`), compiled once at the warm-up's step count; a point's plan does
  not depend on its step count (`tests/mlp_sweep.rs`,
  `a_point_s_plan_is_the_same_at_every_step_count`). The measured quantity is unchanged: host
  wall time from the first submit to the end of the one host wait. The PyTorch opponent is not
  changed.

Regression test: `tests/vulkan_mlp_sweep.rs`,
`the_timed_runs_of_the_widest_point_leave_the_device_idle_under_40_ms_between_them`
(`tests/red-74a9520-harness.txt`: gaps 134 and 136 ms, red; `tests/green-fix.txt`: 15 and 16 ms).
`every_run_of_a_series_is_bit_for_bit_the_run_of_the_source_at_its_step_count` checks a series
against `train_with` of the source at each step count.

## Measurement (idle-removed)

Final session, 2026-10-25T04:53:53Z to 04:57:13Z, in this order (`final/`):

| run | binary | lowest SM clock after 1 s, runs 1/2/3 (MHz) | µs per step, runs 1/2/3 | median | gaps (ms) |
|---|---|---|---|---|---|
| s1 | 74a9520 | 1155 / 915 / 1095 | 7,601.6 / 7,922.9 / 7,830.6 | 7,830.6 | 152 / 154 |
| s2 | fix | 1695 / 1695 / 1695 | 6,992.1 / 6,999.1 / 7,011.8 | 6,999.1 | 27 / 26 |
| s3 | 40dd797 | 960 / 1095 / 1155 | 8,074.0 / 7,777.4 / 7,778.9 | 7,778.9 | 152 / 139 |
| s4 | fix | 1665 / 1665 / 1680 | 7,081.6 / 7,085.2 / 7,084.6 | 7,084.6 | 27 / 26 |
| s5 | PyTorch eager CUDA | 1605 / 1605 / 1605 | 6,407.4 / 6,420.2 / 6,426.7 | 6,420.2 | 0 / 0 |

s4 is the acceptance measurement: no sample below 1,500 MHz after the first second of any of
its three runs, median 7,084.6 µs, 1.4 % below 7,188. PyTorch:
`<venv>/bin/python opponents/mlp_sweep/mlp_sweep.py --point 8192:4096 --modes eager`, torch
2.14.1+cu130, cwd `<worktree>`.

## Same bits

`samebits/same-bits-sha256.txt`: `train --backend vulkan` of `examples/mnist_mlp.mspec` and of
the b8192 h4096 source (`--steps 20`), built at 74a9520 and with the fix: losses, stdout and
checkpoint sha256 equal, and equal to the figures of `../2026-10-25-fuse/same-bits-sha256.txt`.
The `--stats` counters are equal but µs per step (`samebits/*-stats.txt`).
