---
format: aep.planning-md/3
id: review-result:observatory-acceptance-round-2
kind: review-result
status: active
title: Plan critic acceptance, epic:training-observatory, round 2
relations:
- reviews: story:observatory-input-adjoint
- reviews: story:observatory-live-event-stream
- reviews: story:observatory-run-recording-replay
- reviews: story:observatory-ui-live-loss-chart
- reviews: story:observatory-architecture-graph
- reviews: story:observatory-parameter-statistics
- reviews: story:observatory-activation-statistics
- reviews: story:observatory-saliency-gradient-x-input
revision: 1
---
needs-revision

- story:observatory-ui-live-loss-chart — scenario 7 cannot be run on this machine and names no machine where it can. Its PATH is built from `dirname "$(command -v cargo)"`. Here `cargo` is `/usr/bin/cargo` and `node` is `/usr/bin/node`, so `! command -v node` fails. The test is also not run by `task ci`, so it passes only on a machine prepared for it — `.engineering/planning/story/observatory-ui-live-loss-chart.md:86-88`
- story:observatory-input-adjoint — the body requires an all-zero adjoint for an unread input and leaves "output or refused" to be settled in DESIGN.md § 4. No scenario exercises an unread input, so either outcome passes. Scenario 3 covers only the two refusals — `.engineering/planning/story/observatory-input-adjoint.md:37-39` (requirement) and `:76-78` (scenario 3)
- story:observatory-input-adjoint — scenario 1 says "the relative error bound of `gradients.rs`", but that file has two bounds: `TOL_F64 = 1e-6` and `TOL_F32 = 1e-3`. A reader cannot tell which one the CPU f32 adjoint must meet — `.engineering/planning/story/observatory-input-adjoint.md:68-70`; `crates/modelspec-runtime/tests/gradients.rs:29,31`
- story:observatory-input-adjoint — scenario 4 is labelled a command, but the `grep` checks only the revision line. "§ 3.3 pass 4 and § 4 each contains 'input adjoint'" has no command, and DESIGN.md has no "input adjoint" today (`grep -n -i "input adjoint" DESIGN.md` matches nothing) — `.engineering/planning/story/observatory-input-adjoint.md:79-81`
- story:observatory-parameter-statistics — scenarios 2 and 3 quantify over "every event … that carries statistics". The sampling marker is settled in the same story, so a settlement with no sampled step would pass them with zero events checked. Add "at least one event carries statistics" — `.engineering/planning/story/observatory-parameter-statistics.md:89`
- story:observatory-activation-statistics — scenario 2 says "for every activation the settlement names". A settlement that names none passes it vacuously. State a minimum, such as at least the `Relu` output of two-layer-relu — `.engineering/planning/story/observatory-activation-statistics.md:76`

What I read: 8 of 8 stories in full with `aep plan artifact show`. I also read `kinds` and the story lifecycle, ran `ess specify validate --path .` in `spec/`, and used grep on DESIGN.md, `runner.rs`, `train_linear.rs`, `gradients.rs` and the example models.

What I could not establish:
- Whether the ui-live-loss-chart scenario 7 command passes on the intended CI runner. I only know it fails on this machine.
- Whether the saliency "within 1e-2" is absolute or relative. The epic states it the same way, so I left it.
- Out of my lane:
  - The `modelspec-docs/models` path is named for both models, but `examples/` holds no two-layer-relu, which the stories say themselves.
  - The shared-file ordering chain belongs to design and parallel-safety.

Checked and fine:
- The `ESS-ENTITY-019` warnings on `ModelGraph` and `Insight` exist today, so the "gone" scenarios have a real before state.
- The `UNMAPPED` markers exist.
- `step 500` and the named runner and CLI tests exist.
- The DESIGN revision is 0.7 now, so the `0.[89]` grep is a real transition.

```findings
- file: .engineering/planning/story/observatory-ui-live-loss-chart.md
  line: 86
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: undecided
  message: "scenario 7 builds PATH from dirname of command -v cargo, but cargo is /usr/bin/cargo and node is /usr/bin/node here, so the command fails on this machine and the scenario names no machine where it can be run"
- file: .engineering/planning/story/observatory-input-adjoint.md
  line: 76
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "no scenario exercises an input the target does not read, so the all-zero adjoint requirement and the DESIGN.md settlement of zero adjoint versus refusal are unchecked; scenario 3 covers only the two refusals"
- file: .engineering/planning/story/observatory-input-adjoint.md
  line: 68
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scenario 1 cites 'the relative error bound of gradients.rs' but that file has TOL_F64 (1e-6) and TOL_F32 (1e-3), so the bound the adjoint must meet is not named"
- file: .engineering/planning/story/observatory-input-adjoint.md
  line: 79
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "scenario 4 is labelled a command but the grep checks only the revision line; 'each contains \"input adjoint\"' for § 3.3 pass 4 and § 4 has no command"
- file: .engineering/planning/story/observatory-parameter-statistics.md
  line: 89
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: undecided
  message: "scenarios 2 and 3 quantify over 'every event that carries statistics' with the sampling settled in this story, so a settlement with no statistics-bearing step passes with zero events checked; no scenario requires at least one"
- file: .engineering/planning/story/observatory-activation-statistics.md
  line: 76
  category: acceptance
  severity: warning
  verdict: needs-revision
  origin: undecided
  message: "scenario 2 checks 'every activation the settlement names', so a settlement that names none passes vacuously; no minimum set of activations is required"
```
