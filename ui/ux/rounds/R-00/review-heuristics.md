# R-00: heuristic review (L7)

Verdict: 5 pass, 10 partial, 0 fail (1 not observed)

Inputs: report.md, the 11 round shots, the three fixture replays (no rrweb recording; evidence cites
`<fixture> @ event n`, the fixtures under crates/modelspec-cli/tests/fixtures/), the scenario cards,
src/. No operator notes. Extra replays through a pacing proxy, 1366x768 unless named, shots
`shots/review-*.png`. The failed end state is in no R-00 recording; it was replayed from a scratch
recording outside the tree (accuracy.run.jsonl events 0-10 plus a RunFailed).

| # | heuristic | verdict | evidence |
|---|---|---|---|
| 1 | One next thing | pass | read-only page with one control (the histogram step picker); the loss chart is first at every key moment: shots/*-contact.jpg |
| 2 | State in words | partial | state word present in every frame (contact sheets); but an empty or one-point chart says nothing (shots/review-relu-step50-full-light.png, UXF-006) and "step 0 of 21" is shown before any step (shots/accuracy-1366x768-light-worst-shift.jpg, UXF-007) |
| 3 | Nothing required hidden | partial | a parameter's gradient norm needs a PageDown (shots/two-layer-relu-1366x768-light-first-parameter-gradient-found.jpg, UXF-001); bin counts only on hover (UXF-009); the loss trend after the first steps is flattened (shots/linear-regression-contact.jpg, UXF-010) |
| 4 | Stable layout | partial | charts shift when the run id line appears, every scenario @ event 0 (shots/linear-regression-1366x768-light-worst-shift.jpg, UXF-002); nothing else moved in the replays |
| 5 | Provenance of every value | partial | every caption names its step ("step 400: 3.041e-5"); but the gradient norm axis ends at 100/400 under a value norm axis ending at 200/500 (shots/review-relu-end-full-light.png, UXF-008) and a one-point chart labels steps -1 and 1 (shots/review-relu-step0-norm-charts-light.png, UXF-006) |
| 6 | Errors cheap | pass | actions_per_fix.max 1 (report.md); the picker is set back by one key (ArrowDown on the focused select changed it, two_layer_relu.run.jsonl @ event 202) |
| 7 | Words the user can say | pass | labels are plain words and numbers ("histograms at", "value norm", "finished step 200 of 200"): shots/review-relu-end-full-light.png |
| 8 | The counterpart's time respected (read: the reader's) | pass | screen_only.time_to_act_steps.max 0 in all scenarios (report.md): the page never lags the stream |
| 9 | Keyboard and pointer parity | partial | Tab reaches the only control and arrows change it (two_layer_relu.run.jsonl @ event 202, focus ring in shots/review-relu-picked-step-100-light.png); bin counts are pointer-only (UXF-009) |
| 10 | Recognition over recall | partial | "latest sampled step" does not name its step and the norm charts do not mark a picked step (shots/review-relu-picked-step-100-light.png); the caption below names it, so the cost is one glance; no finding |
| 11 | Calm by default | pass | no motion between frames in any contact sheet; gate 5 reduced-motion check passed in all 12 replays (report.md) |
| 12 | One state vocabulary | partial | connecting / streaming / finished / failed used throughout; "step 0 of 21" means both no step and step 0 done (shots/accuracy-1366x768-light-worst-shift.jpg against shots/accuracy-1366x768-dark-first-step.jpg, UXF-007) |
| 13 | A wrong cue recovered in one action | not observed | the app has no guidance cue (report.md, gate 6) |
| 14 | Unmistakable end state | partial | failed is red word plus a red alert line (shots/review-failed-end-light.png, shots/review-failed-end-dark.png); finished differs from streaming by one word in the same style (shots/review-accuracy-streaming-light.png against shots/review-accuracy-finished-light.png), reads "step 20 of 21" (UXF-004) and is no status region (UXF-003) |
| 15 | Readable in every theme; key data at 14 px or more | partial | axe 0 serious/critical in both themes (report.md); caption values 13.5 px (UXF-005); axis labels about 6 px at 390x844 (shots/review-lr-390x844-end-dark.png, UXF-011) |
| 16 | Focus follows the work | partial | the loss chart and a failure alert sit at the top (shots/review-failed-end-dark.png); parameter panels update out of view at the gating viewports (shots/two-layer-relu-contact.jpg, UXF-001, not refiled) |

## Findings filed

| id | title | severity | class | sessions | cost |
|---|---|---|---|---|---|
| UXF-006 | a chart with fewer than two points draws no mark, says nothing, and its step axis reads -1 to 1 | S2 | cue | all three (norm charts for steps 0-99 of two-layer-relu and linear-regression) | S |
| UXF-007 | "step 0 of N" is shown before any step has completed | S3 | copy | all three | S |
| UXF-008 | a gradient norm chart's step axis ends at its last gradient, so the charts of one parameter do not line up | S3 | layout | linear-regression, two-layer-relu | S |
| UXF-009 | a histogram's bin counts show only on pointer hover | S3 | affordance | linear-regression, two-layer-relu | S |
| UXF-010 | the loss chart's linear scale flattens most of the run onto its floor | S3 | cue | linear-regression, two-layer-relu | M |
| UXF-011 | at 390 px wide the chart axis labels shrink to about 6 px | S3 | layout | all three (390x844, measured-only) | S |

Existing UXF-001 to UXF-005 were cited, not refiled. Severity S2 for UXF-006 is my call: it blanks
every norm chart for half of the two-layer-relu run.
