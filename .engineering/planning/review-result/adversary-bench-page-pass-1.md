---
format: aep.planning-md/3
id: review-result:adversary-bench-page-pass-1
kind: review-result
status: active
title: 'Adversary pass 1 on the benchmark results page: the FLOP chart, GFLOP/s, Not measured and the pitch slide'
relations:
- reviews: story:benchmark-results-page
revision: 1
---
unit: story:benchmark-results-page (worktree wt-741409204ee6, the implementor's edits over plan commit ede12dd)
verdict: findings 6 (blocker 0, warning 1, note 5); 4 red cases
cases: executed 74→80 (6 cases in the default suites, 4 red cases ignored by finding id)
origin: introduced 6 / pre-existing 0 / undecided 0
wrote-outside-worktree: `<wave-root>/bench-page/adv-scratch/` (a mutated copy with its target cleaned, mutant logs, a chart render)
needs-coordinator: ADV-BP-5 (page prose names planning ids) and ADV-BP-7 (an undeclared workload is published) are decisions the story does not make

## Cases added

| file | cases | now |
|---|---|---|
| `crates/modelspec-docs/tests/adversary_bench_page.rs` | every GFLOP/s cell equals flop / µs / 1000 of its latest run (36 cells, blank where no count); every `MatmulOnly` run states 3196·B·H (86 runs); the FLOP chart draws 3 sides at each of 12 counts (36 markers, 9 polylines); slide 15 follows the MNIST slide, `data-n` 1..18, counter `/ 18`, crossover widths equal `crossover.txt`; a realistic first transformer run leaves "Not measured" and gets its own section; `generate --check` refuses a hand-edited chart | 6 green |
| the same file | ADV-BP-3 (sides of one point share one x or a wrong count is refused); ADV-BP-5 (page prose names no planning record); ADV-BP-6 (slide 15 GFLOP/s equal the cited runs' medians to ±0.05); ADV-BP-7 (an undeclared workload is refused) | 4 red, ignored |

## Suite run after the cases existed

| command | result | exit |
|---|---|---|
| `cargo test -p modelspec-docs` | 55 + 6 (4 ignored) + 10 + 9 passed, 0 failed | 0 |
| `cargo test -p modelspec-docs --test adversary_bench_page -- --include-ignored` | 6 passed, 4 failed | 101 |
| `task docs-check`, `cargo fmt --check -p modelspec-docs`, `cargo clippy -p modelspec-docs --all-targets -D warnings`, `cargo test -p modelspec-cli --test adversary_gpu_pretrain_cli` | | 0 |

## Findings

| id | file:line | verdict | origin | measured |
|---|---|---|---|---|
| ADV-BP-5 | `benchmarks.rs:919, :924` | NEEDS-CHANGE (warning) | introduced | the page's "Not measured" prose names `story:transformer-block-benchmark` and `story:bert-tiny-vs-pytorch-cuda`; the base page's prose named no planning id |
| ADV-BP-3 | `benchmarks.rs:1336` | INFEASIBLE (note) | introduced | sides of one point under the same rule with different counts are drawn at different x and not refused; `read` does not check `MatmulOnly` against 3196·B·H; every real run fits |
| ADV-BP-6 | `docs/pitch/index.html:634` | CONFIRMED (note) | introduced | slide 15's eager 15,676.1 GFLOP/s divides by the report's rounded 6841.0 µs; the cited run's median 6840.956 gives 15,676.2; the notes say which µs they divide |
| ADV-BP-7 | `benchmarks.rs:929, :1489` | INFEASIBLE (note) | introduced | a run with a misspelt workload is published under its raw id while "Not measured" still lists the declared one |
| ADV-BP-8 | `benchmarks.rs:1405, :1418` | CONFIRMED (note) | introduced | the mutants `group := 0` and `log_x := false` are killed only by the committed-file comparison, not by a fixture test |
| ADV-BP-9 | `benchmarks.rs:1405` | CONFIRMED (note) | introduced | the FLOP chart joins each batch's points but labels no line with its batch; ModelSpec's batch-100 and batch-1000 lines overlap |

## Mutants

| mutant | killed by |
|---|---|
| GFLOP/s divides by 1e6 | unit test, committed-file comparison, check cases |
| earliest run | unit tests, committed-file comparison |
| list every declared workload | unit test, adversary case, committed-file comparison |
| drop the compile side | 4 unit tests |
| no mixed-rule handling | 2 unit tests |
| `group := 0` | committed-file comparison only (ADV-BP-8) |
| `log_x := false` | committed-file comparison only (ADV-BP-8) |

## Attacked and held

All 36 table cells, the 3 largest-point figures and every run's 3196·B·H count; latest date wins, later file order on a tie; `data-n` 1..18 and the counter from `slides.length`; the crossover widths and batch-1000 µs on the slide; every cited run id exists; no `/home/` path; `generate --check` refuses a hand-edited SVG; the MNIST section unchanged.

## Findings block

```findings
- file: crates/modelspec-docs/src/benchmarks.rs
  line: 919
  category: contract-drift
  severity: warning
  verdict: NEEDS-CHANGE
  origin: introduced
  message: The Not measured prose of the public page names planning-store ids story:transformer-block-benchmark and story:bert-tiny-vs-pytorch-cuda, where the base page's prose named none.
- file: crates/modelspec-docs/src/benchmarks.rs
  line: 1336
  category: acceptance
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: Sides of one point that share a flop_rule but state different flop_per_step are drawn at different x and not refused, contrary to the flop_points doc's one-count claim.
- file: docs/pitch/index.html
  line: 634
  category: boundary
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: Slide 15's PyTorch eager GFLOP/s 15,676.1 comes from the report's rounded 6841.0 µs, while the cited run's median 6840.956 µs gives 15,676.2.
- file: crates/modelspec-docs/src/benchmarks.rs
  line: 1489
  category: boundary
  severity: note
  verdict: INFEASIBLE
  origin: introduced
  message: A run with an undeclared workload id is published under its raw id while Not measured goes on listing the declared workload it was meant to be.
- file: crates/modelspec-docs/src/benchmarks.rs
  line: 1418
  category: mutant
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: The log_x := false and group := 0 mutants of the FLOP chart are killed only by the committed-file comparison, not by any fixture test.
- file: crates/modelspec-docs/src/benchmarks.rs
  line: 1405
  category: judgement
  severity: note
  verdict: CONFIRMED
  origin: introduced
  message: The FLOP chart joins each batch's points by a line but labels no line with its batch, so the overlapping ModelSpec batch-100 and batch-1000 lines cannot be told apart.
```
