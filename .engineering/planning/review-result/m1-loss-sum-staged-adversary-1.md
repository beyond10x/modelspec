---
format: aep.planning-md/3
id: review-result:m1-loss-sum-staged-adversary-1
kind: review-result
status: active
title: Adversary review of m1-loss-sum-staged at b216b5b
relations:
- reviews: story:m1-loss-sum-staged
revision: 1
---
# Adversary review of m1-loss-sum-staged at b216b5b

Verdict: NEEDS-CHANGE. The kernel keeps the row-major order bit for bit in every case tried on
the RTX 3090 (ranks 0, 1, 3 and 4; chunks of 1, 2, 63, 64, 65, 4,096, 12,287 and 12,288 terms;
78 device runs, 0 differ; 198 modules accepted by `spirv-val` with `NoContraction` on every
float op). The suite cannot see an order change inside a chunk on a tolerance-class device.

Mutants of `emit.rs`:

| mutant | implementor's device test | `task test` | adversary case |
|---|---|---|---|
| swap terms 1 and 2 in each group of 8 (emit.rs:1079) | green | green | red, 3 sums differ |
| no barrier between chunks (emit.rs:1196) | red | not run | red |
| no barrier between staging and summing (emit.rs:1179) | green | red only via the barrier count at spirv_val.rs:503 | red |

Acceptance 3 measured independently: loss-sum `mean_us` 5.682, 5.681 and 5.754 (62 P0 and 64 P3
clock samples, mean 1283 MHz); third-longest dispatch. Against the limit of 5 in story revision 5
it is red; the story's revision 6 restates the limit to at most 6 and not the longest dispatch.

Not broken: chunk boundaries at counts not a multiple of 64 or the chunk; the exact 49,152-byte
limit; overflow in the size calculation (term count capped by `u32_of`); a zero chunk (refused);
the barrier between chunks; DESIGN.md § 3.1, § 5.2, § 5.4 and the CHANGELOG line against the code.

```findings
- file: crates/modelspec-runtime/tests/vulkan_loss_sum.rs
  line: 244
  category: mutant
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: on the RTX 3090 (tolerance class) a mutant that swaps terms 1 and 2 of every unrolled group of 8 passes task test and all acceptance-1 device tests; only adversary_loss_sum's bit check catches it
- file: crates/modelspec-runtime/tests/spirv_val.rs
  line: 503
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: removing the barrier between staging and summing passes the device test and is caught only by a barrier count that a multi-chunk module still satisfies
- file: crates/modelspec-runtime/src/spirv/emit.rs
  line: 1107
  category: acceptance
  severity: blocker
  verdict: CONFIRMED
  origin: introduced
  message: acceptance 3 fails on an independent warm run, loss-sum mean_us 5.682, 5.681 and 5.754 against the limit of 5
- file: website/docs/guides/inspect-a-run.md
  line: 101
  category: contract-drift
  severity: note
  verdict: NEEDS-CHANGE
  origin: introduced
  message: the new profile sample comes from a cold device (120.349 µs per step against 93.205 before and 85-86 µs warm), so the guide shows the change as a slowdown
```
