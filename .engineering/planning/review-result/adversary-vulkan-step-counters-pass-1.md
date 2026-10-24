---
format: aep.planning-md/3
id: review-result:adversary-vulkan-step-counters-pass-1
kind: review-result
status: active
title: Adversary pass 1 on vulkan-step-counters
relations:
- reviews: story:vulkan-step-counters
revision: 1
---
That run captured the wrong exit status, so I'm rerunning with full output and real exit codes.
unit: story:vulkan-step-counters, uncommitted working tree on base 93121b3
verdict: nothing found
cases: executed 8→14 (device runs of the selected files), red 0
origin: introduced 0 / pre-existing 0 / undecided 0
wrote-outside-worktree: 3 paths, under `<wave-root>/counters/scratch/adversary/`
needs-coordinator: none

All three counters held up under every attack. The only gap I found was in test coverage, and the case I added closes it.

**1. `git --no-pager diff --stat`**: the same 10 files as the implementor's diff; I changed none of them. My only additions are two untracked test files:
- `crates/modelspec-cli/tests/adversary_step_counters.rs`
- `crates/modelspec-runtime/tests/adversary_step_counters.rs`

**2. Cases added**: all are device tests, run alone on the RTX 3090 with `MODELSPEC_VK_VALIDATION=1`. All are green now and were green on their first run.

| Case | What it asserts | Result |
|---|---|---|
| CLI `the_guides_counter_block_is_what_its_command_prints` | Reads the guide's own command (`--steps 72000 --profile`), runs it, and compares all 8 timing-free `profile` lines with the guide's block | ok, 12.96 s |
| CLI `stats_and_profile_print_the_same_three_counters` | `--stats --profile` in one run, `--stats` alone and `--profile` alone all print the same values; 7 barriers, 12 allocations | ok |
| CLI `checkpoints_and_resume_keep_the_per_step_and_allocation_counts` | `--checkpoint-every 100` gives the same three counters as a plain run, with 5 host waits; `--resume --steps 100` gives the same barriers and allocations | ok |
| runtime `runs_on_one_thread_count_only_their_own_buffers_and_barriers` | Three runs on one thread and device, with a 1 MiB buffer kept across the second run, report identical counters | ok |
| runtime `a_buffer_freed_and_remade_in_a_span_peaks_at_one_buffer` | A buffer freed and re-made inside a span peaks at one buffer, not two; a buffer from an earlier span freed in a later one does not lower the later span's figure | ok, 0.20 s |

**3. Suite runs** (all after the cases existed):

| Command | Exit | Result |
|---|---|---|
| `b10x-build-slot cargo test -p modelspec-runtime --features vulkan` | 0 | 363 passed, 0 failed, 53 ignored |
| `b10x-build-slot cargo test -p modelspec-cli` | 0 | 298 passed, 0 failed, 34 ignored |
| runtime device tests (`vulkan_profile`, `adversary_profile_throughput`, mine) with `--ignored` | 0 | 4 ok |
| CLI device tests (mine, `train_profile_adversary`, `train_vulkan`, `checkpoint_vulkan`) with `--ignored` | 0 | 9 ok |
| docs drift check (`cargo run -p modelspec-docs -- generate --check`) | 0 | all generated files current |
| clippy `-D warnings` on both crates | 0 | clean |
| `cargo fmt --check` | 0 | clean |

My cases are all `#[ignore]`, so the default suite counts don't change. I ran rustfmt on my own files after the suite ran; that was formatting only.

**4. Judgement findings**: one note, about coverage.
- No test that existed before mine drops a real `Buffer` while a span is open. A run frees nothing before its counters are read, and the unit test at `memory.rs:372` calls `freed()` directly, not through the drop.
- So deleting the `freed(...)` call in `Buffer::drop` (`crates/modelspec-runtime/src/vulkan/memory.rs:219`) would leave the unit's suite green. Peak would then become the sum of all allocations rather than the live peak.
- Nothing reaches that path today. story:vulkan-buffer-liveness-suballocation will, because it frees and re-makes buffers within a run.
- My last case closes the gap: with the mutation it would report twice one buffer's size instead of once. I reasoned that out rather than running a mutated copy.

**5. Attacked and could not break**:
- **Second run on the same thread**: counters start fresh for each run and do not accumulate.
- **Buffer dropped after its span closed**: it is excluded from later spans.
- **True peak vs sum**: the ledger tracks the live peak. MNIST peak moves by 561,600 bytes between 1,800 and 72,000 steps, exactly twice the 280,800-byte growth of the loss buffer.
- **Which steps are counted**: barriers are counted from the start to the end of one step's recording. Chunk-boundary submits, the upload barrier and the final loss evaluation fall outside it.
- **Checkpoint and resume**: neither changes the per-step barrier figure or the allocation count.
- **A run that spans two threads**: none exists. `on_device` and `execute` run on one thread, and the CLI starts no threads.
- **Other barrier or allocation sites**: the only call to `cmd_pipeline_barrier` is `sync.rs:88`, and the only call to `allocate_memory` is `memory.rs:308`.
- **The guide's breakdown**: MNIST `plan step` has 8 barrier steps and 0 copies, and the plan has 17 buffers.
- **DESIGN.md § 8**: it lists both the `--stats` lines and the `profile` lines in the order the code prints them (`runner.rs:115-128`, `profile.rs:258-270`).

**6. Paths written outside the worktree** (under `<wave-root>/counters/scratch/adversary/`):
- `mnist-kernel.txt`
- `runtime-suite.log`
- `cli-suite.log`
- `device-runtime.log`
- `device-cli.log`

That is 5 files. The header line says 3; 5 is the correct count. Inside the worktree, the checkpoint test also wrote under `target/tmp/adversary-step-counters/`.

```findings
- file: crates/modelspec-runtime/src/vulkan/memory.rs
  line: 219
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: no test before the adversary's drops a real Buffer inside an open MemorySpan, so removing the freed() call from Buffer::drop leaves the unit's suite green; adversary_step_counters.rs a_buffer_freed_and_remade_in_a_span_peaks_at_one_buffer now covers it
```
