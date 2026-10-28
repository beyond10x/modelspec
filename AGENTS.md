# ModelSpec

A Rust compiler and runtime for `.mspec` tensor-program specifications. Humans start at
[README.md](README.md); the design is [DESIGN.md](DESIGN.md).

## Serves

- **The north star: `vision:modelspec` § North star (2026-10-28)**; public pages label its speed claim planned.
- **The goal (2026-10-28): `initiative:transformer-within-3x-of-pytorch`.** A transformer trained on
  the GPU through Vulkan within 3× of PyTorch eager's step time; the order of work (B, A, C, D) and
  how to pick the next unit are the repository skill `goal` (`.agents/skills/goal/SKILL.md`,
  `/goal` in Claude Code). Where the skill and the store disagree, the store wins.
- **O6 — self-improvement, built into all of it.** Every compile and training run leaves a typed
  record (artifact and source hashes, the deterministic setting, run steps and resume checkpoint in
  `spec/`), so a change to the compiler can be re-measured against the same run before and after.

## State

Milestone M0 implemented (DESIGN.md revision 0.6, § 1 and § 11, iteration 1; wave `m0` closed);
M1 (`epic:m1-vulkan-execution`) is in progress, latest release 0.21.0. The active milestone is
`epic:bert-from-scratch`, M3's BERT proof model: the operators `embedding` (with `i32` row ids),
`add_rows`, `gelu`, `layer_norm`, `softmax` and `attention` (0.10.0), `cross_entropy` and
`dropout` (0.11.0) run on the CPU interpreter, as do packed `--data` training and
`--eval-every` held-out evaluation (0.11.0) and `optimizer = adamw(...)` (0.12.0) and `optimizer = muon(...)` (0.15.0); since 0.14.0,
`embedding`, `add_rows`, `gelu`, `dropout`, `layer_norm`, `softmax` and their gradients also
have Vulkan kernels checked against the CPU oracle, and since 0.17.0 so do `attention`,
`cross_entropy` and the `adamw` step, and `train --backend vulkan --data` trains the hidden-32 BERT
twin with dropout; a command that needs a Vulkan kernel for `muon` answers E-TARGET-011 at its call.
On 2026-10-26 `examples/bert_tiny.mspec` was pre-trained on the RTX 3090 with the 0.18.0 binary for
10,000 steps over packed WikiText-103 to a held-out loss of 6.7674994 nats, 0.159 below the unigram
baseline 6.9265379 (`website/docs/runs/bert-tiny-gpu-pretraining.md`; the event file stays outside
the repository). A Cargo workspace of five crates: `modelspec-core`
(parser, checker, Tensor IR, autodiff, lowering, DCE; no Vulkan), `modelspec-runtime` (CPU oracle,
SGD training and the corpus preparation, WordPiece tokenizer and MLM batch packing `dataset`; the `vulkan` feature holds Vulkan device discovery, negotiation, SPIR-V
emission and the training runner `vulkan::runner` through `ash` and `rspirv`; the JSON Schema of
the records it writes under `schema/` is `ess generate` output), `modelspec-observatory` (the typed
`RunEvent`s of a run and their Server-Sent Events stream; its JSON Schema under `schema/` is
`ess generate` output), `modelspec-cli` (binary `modelspec`) and `modelspec-docs` (repository
tooling: the derived pages of the documentation site).
`check` resolves names and checks types and shapes;
`ir <file> --stage tensor|training` prints the verified Tensor IR or the verified training step;
`train <file> --backend cpu` trains on the CPU reference (DESIGN.md § 8), and `--backend vulkan`
trains on one Vulkan device with one host wait per run plus one per requested checkpoint
(`--stats` prints its counters); `--checkpoint`, `--checkpoint-every` and `--resume` work on both
backends, and `infer <file> --backend cpu|vulkan --checkpoint <path>` runs a checkpoint's forward
program and loss, on Vulkan with one host wait; a CLI built without the `vulkan` feature refuses
`--backend vulkan` with E-TARGET-001. `inspect-device --backend vulkan`
prints each device's profile and negotiation verdict; the device tests run on a GPU only with
`--ignored`. `train --serve <addr>` streams a run's `RunEvent`s, `--record <path>` writes them to a
file and `observe <path>` replays one; with `--ui ui/dist` either also serves the observatory UI at
`/`, the Vue 3 + TypeScript app in `ui/` (Vite, pnpm) that draws the loss and every metric per step.
`data prepare <corpus-dir> --out <dir>` copies a local WikiText-103 raw release's `wiki.train.raw`
and `wiki.valid.raw` into the `Train` and `HeldOut` splits and writes their `Dataset` record
`dataset.json` (`spec/domains/data.yaml`); nothing is downloaded, and its tests use the original
text of `crates/modelspec-runtime/tests/fixtures/corpus/`, never corpus text.
`data tokenizer --dataset <dir> --vocab-size <n> --out <file>` trains the WordPiece tokenizer on the
`Train` split of such a dataset, never its `HeldOut` split, to exactly `n` tokens and writes its
`Tokenizer` record (`spec/domains/data.yaml`, coordinator decision D23); `dataset::tokenizer`'s
`Codec::encode` and `Codec::decode` are the library calls of later data steps. Its tests also use
`crates/modelspec-runtime/tests/fixtures/corpus-tokenizer/`, original text too.
`data pack --dataset <dir> --tokenizer <file> --seq-len <S> --out <dir>` joins the documents of
both splits with `[SEP]`, cuts them into sequences of exactly `S` tokens and writes
`train.tokens`, `heldout.tokens`, the `TrainingDataBinding` `binding.json` and the `PackedDataset`
`packed.json` with the held-out unigram baseline (`spec/domains/data.yaml`, coordinator decision
D26); `dataset::mlm::mlm_batch` draws the 80/10/10 batch of a step from them, masking on Philox
stream tag 3 and the epoch order on tag 5 (DESIGN.md § 2.3). Its tests are
`cargo test -p modelspec-runtime --test mlm` and `cargo test -p modelspec-cli --test data_pack`,
on the `corpus-tokenizer` fixture.
`train <file> --backend cpu --data <dir>` on a packed directory (one holding `binding.json`) feeds
step `k` the `Train` batch of step `k`, each `input` and `target` naming a batch tensor (`ids`,
`positions`, `key_mask`, `labels`, `weights`), and prints a `data` line first (DESIGN.md 0.46);
any other `--data` directory holds dense `f32` files. `--data`, `--init` and `--eval` are CPU only.
Its tests are `cargo test -p modelspec-cli --test train_data`.
`examples/bert_tiny.mspec` is BERT-tiny written out at the 8,192-token vocabulary of the corpus
tokenizer (1,486,720 parameters; Turc et al.'s BERT-tiny has 4.4M at 30,522 tokens; `cargo test -p
modelspec-cli --test bert_tiny`), and `crates/modelspec-cli/tests/fixtures/bert_cpu.mspec` is the
same program at hidden size 32, whose every parameter `cargo test -p modelspec-runtime --test
gradients_bert_cpu` checks against central differences in `f64` and which `cargo test -p
modelspec-cli --test bert_cpu_train` trains twice from one seed on a synthetic corpus the test
writes (`crates/modelspec-cli/tests/common/markov_corpus.rs`).
Two tests guard the pair, so change both files together:
`adversary_bert_cpu.rs::both_files_are_post_ln_bert_in_the_ir` (modelspec-runtime) walks the
Tensor IR of both as post-layer-norm BERT, and
`adversary_bert_tiny_cli.rs::twin_differs_from_the_example_only_in_shapes_and_run_length`
(modelspec-cli) fails when the twin differs from the example beyond its shapes, `steps`,
`warmup_steps` (10 % of the steps in both) and `lr`.
The documentation site's guide `website/docs/guides/prepare-training-data.md` and concept page
`website/docs/concepts/bert-operators.md` paste output from these commands; rerun them when the
output changes.
CI: `check.yml` runs `task ci`; `shared-gates.yml` runs the
common source gates; `pages.yml` and `b10x-docs-site.yml` build and publish the documentation site.

## Resumed (2026-10-27)

Development paused after release 0.20.0 (Timo, 2026-10-26) and resumed on 2026-10-27 with unit 22,
`story:transformer-block-benchmark` (release 0.21.0). The pause record is
`.engineering/paused-2026-10-26.md`; the work that follows is the `goal` skill's order.

## Commands

Entry points are the `Taskfile.yml` tasks (go-task):

- `task build`: `cargo build --workspace`.
- `task test`: `cargo test --workspace`, then `cargo test -p modelspec-runtime --features vulkan`,
  whose `spirv_val` test runs `spirv-val --target-env vulkan1.2` (SPIRV-Tools) on every emitted
  elementwise, broadcast and SGD module and fails, not skips, without it.
- `task lint`: `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings` and
  the same clippy run of `modelspec-runtime` with `--features vulkan`.
- The device tests need a Vulkan device and are ignored by default:
  `MODELSPEC_VK_VALIDATION=1 cargo test -p modelspec-runtime --features vulkan --test vulkan_elementwise -- --ignored`.
  `MODELSPEC_VK_VALIDATION=1` creates the instance with the Khronos validation layer; each
  message it reports fails the test. The training run on the device is
  `MODELSPEC_VK_VALIDATION=1 cargo test --release -p modelspec-runtime --features vulkan --test vulkan_train -- --ignored`,
  which prints µs per step and dispatches per step; `cargo test -p modelspec-cli --test train_vulkan -- --ignored`
  compares the lines of `train --backend vulkan` with the CPU run.
- `crates/modelspec-runtime/tests/no_cuda.rs` (in `task test`) fails when the runtime's
  dependency graph names a CUDA crate or a built binary links a CUDA library.
- `task core-no-vulkan`: `cargo test -p modelspec-core --test no_vulkan`, which fails when the
  dependency graph of `modelspec-core` names a Vulkan crate on any target with all features on.
- `task cli-without-vulkan`: the `inspect-device`, `train_linear` and `train_linear_adversary`
  tests of a CLI built without the `vulkan` feature, which refuses `--backend vulkan` with
  E-TARGET-001.
- `task check-examples`: `modelspec check` on `examples/linear_regression.mspec`,
  `examples/mnist_mlp.mspec` and `examples/bert_tiny.mspec`; `examples/tiny_mlp.mspec` joins it when modules and generics land (DESIGN.md § 11 iteration 3).
- `task docs-check`: `modelspec-docs generate --check`, which fails when a generated page of the
  documentation site differs from what the code produces now; `task docs-generate` rewrites them.
- `task spec-validate`: `ess specify validate --path .` from inside `spec/`.
- `task schema-check`: `ess generate --kind schema --check` from inside `spec/`, which fails when
  a file the specification generates under `crates/modelspec-observatory/schema/` or
  `crates/modelspec-runtime/schema/` differs from it or is missing; an extra file there passes.
  `task schema-generate` rewrites both. A change to any file under `spec/` changes the schema's
  provenance digest, so run `task schema-generate` in the same change.
- `task ui-types-check`: `modelspec-ui-types --check`, which fails when
  `ui/src/generated/observatory.ts` differs from the TypeScript the Rust types of
  `crates/modelspec-observatory/src/event.rs` generate (`ts-rs`); `task ui-types` rewrites it.
  Never edit `ui/src/generated/` by hand. The test `observatory_ui_types_in_sync` (in `task test`)
  makes the same comparison.
- `task ui`: in `ui/`, `pnpm install --frozen-lockfile`, `pnpm typecheck`, `pnpm test` (the
  `node --test` cases of `src/lib/` and `ux/`) and `pnpm build` into `ui/dist/` (Node 22.18 or later, pnpm
  11; `ui/node_modules/` and `ui/dist/` are never committed).
- `task ui-browser`: `task ui`, then the `#[ignore]`d browser tests of
  `crates/modelspec-cli/tests/observatory_ui.rs` and `observatory_ui_adversary.rs` with
  `--ignored`. They start Chrome or Chromium headless through `headless_chrome`: the executable
  named by `CHROME`, else the first of
  `google-chrome-stable`, `chromium`, `chromium-browser`, `chrome`, ... on `PATH`; `check.yml` sets
  `CHROME`.
- `task ui-ux`: `task ui`, `cargo build -p modelspec-cli`, then `pnpm ux` in `ui/` with the
  browser named by `CHROME` ("Observatory UI loop" below).
- `task ci`: all of the above; the gate. `task ci` needs Node and pnpm for
  `task ui` and a Chrome for `task ui-browser`; `task build` and `task test` need neither, and
  `sh ui/scripts/build-without-node.sh` checks that with a `PATH` that holds no `node` or `pnpm`
  (`task ci` does not run it).
- `task website`: `npm ci` and `npm run build` in `website/` (Node 22). Node is used only here and
  by `task ui`.
- `task release-check TAG=<version>`: refuses unless the tag is a bare version, equals
  `[workspace.package] version` in `Cargo.toml`, and `CHANGELOG.md` has `## [<version>] - <date>`.

CLI exit codes: 0 success, 1 diagnostics or runtime error, 2 usage error.

## Observatory UI

Two build decisions of story:observatory-ui-live-loss-chart:

- The app lives in `ui/` at the repository root (Vue 3, TypeScript, Vite, pnpm; the workspace
  precedent is uilab's `widget/`). `ui/src/generated/` is `modelspec-ui-types` output.
- The server finds the built app at run time, not at compile time: `--ui <dir>` on
  `train --serve` and on `observe` names the directory (`ui/dist` after `task ui`), which is read
  whole into memory before the address is bound and served at `/` beside `/events`
  (`spec/domains/observatory.yaml`, `UiDirectory`). Nothing embeds the app in the binary, so no
  `build.rs` or cargo step calls Node or pnpm, and `task build` and `task test` keep working on a
  machine without them. Without `--ui` only `/events` is served.

Observatory UI loop (story:observatory-ui-iteration-loop, the `ui-iteration-loop` skill):
`task ui-ux`, the last step of `task ci`, builds the app and `modelspec`, then runs `pnpm ux` in
`ui/`, the Playwright specs of `ui/ux/` against
`modelspec observe <fixture> --serve 127.0.0.1:0 --ui ui/dist` with the browser from `CHROME`;
nothing is downloaded. `CHROME` unset: `task ui-ux` refuses before it builds, with one line, and
exits 201, go-task's code for a failed command (`task -x ui-ux` passes the command's 2 through);
`pnpm ux` exits 2. Every page runs under one injected font stack, `DejaVu Sans` for text and
`DejaVu Sans Mono` for code (`FONTS` in `ui/ux/adapter.ts`), so this machine and the CI runner
measure the same words; `pnpm ux` exits 2 with one line when `fc-list` lacks either, and a gate
fails when the browser draws with another font. A pacing proxy replays each fixture of
`ui/ux/scenarios/`. `layout.spec.ts` (L2, through `contract.ts`) sends one event at a time and
checks 3 scenarios at 1366x768 and 1280x720 in light and dark after every event: gate 1 no
scrollable element and no horizontal page scroll, gate 2 the always-visible elements of
`ui/ux/adapter.ts` wholly in view, gate 3 `completion_miss` and `false_complete` 0 (also on the
rotating card ending in `RunFailed`, recorded under `failed_end` in `round.json` and never
ratcheted), gate 5 axe-core without serious or critical violations at every key moment, no
animation under reduced motion after any event, every focusable reached by Tab with a focus
indicator. `screen-only.spec.ts` (L3, gate 4) reads each scenario's `tasks.json` through the
accessibility tree, a value only when its own text is in view, uncovered and not transparent, and
must match `truth.json` in at most 2 key presses a task; the end-state task is answerable only once
the state reads `finished` or `failed`. A second, live replay sends one event every `CADENCE_MS`
(40 ms) without waiting for the screen, and `time_to_act` is the steps sent at least
`RENDER_BUDGET_MS` (250 ms) before a reading minus the steps up to the one the screen shows.
`ratchet.spec.ts` then fails on any metric worse than `ui/ux/baseline.json` beyond its tolerance,
an improvement not taken into it, a missing value, a tolerance in the file that disagrees with
`TOLERANCES` or `KEYS` (the band comes only from the code), or a baseline made worse than
`origin/main`'s without a `decision` id; `origin/main` missing is an error naming the ref, so
`check.yml` fetches `main` before `task ci`. `layout_shift.per_100_events` is a replay's shift sum
per 100 events. The `self-test` project (`self-test.spec.ts`) plants one defect per gate into the
served page and fails when the gate's own code misses it. Run one level alone with `pnpm --dir ui
ux --project gates -g "L2 layout"` (or `-g "L3"`, or `--project self-test`); `UX_UPDATE=1 pnpm
ux` takes improvements into the baseline, `UX_DECISION=<id>` (not empty) also a worse value, and
`pnpm ux --project ratchet --no-deps` reruns the ratchet on the last run. A run writes only to
`ui/node_modules/.cache/ux/`; `UX_ROUND=R-nn pnpm ux` also writes the round's `ui/ux/rounds/R-nn/`
(`report.md` with its caveats first, `round.json` with the measured HEAD, a dirty flag and the
`ui/dist` hash, and `shots/`, replacing only the shots it generates and never `review-*`) and its
row in `rounds/trend.md`, whose columns are named per scenario. Findings are
`ui/ux/findings/UXF-nnn.md`, and `pnpm test` fails on a closed S1 or S2 finding without a
regression asset. `ui/src/**` carries `data-ux-region` marks for the loop; keep them when a
component changes.

## Releases

ModelSpec releases from source at bare-version tags (`0.1.0`, no `v`); every crate inherits the
workspace version and nothing is on a registry.

- `CHANGELOG.md` follows Keep a Changelog 1.1.0. Changes collect under `## [Unreleased]`, one line
  per change a user sees, naming the public item in backticks and its story id where it fits.
- A release commit on `main` sets `[workspace.package] version` in `Cargo.toml` and `Cargo.lock`,
  turns `## [Unreleased]` into `## [<version>] - <date>` under a new empty `## [Unreleased]`, and
  moves the link references at the bottom. `task release-check TAG=<version>` passes on it.
- `check` and `Shared source gates` are green on that commit before it is tagged.
- The tag is the bot's, on that commit:
  `b10x-gates bot --repository beyond10x/modelspec -- tag -a <version> -m "ModelSpec <version>" <commit>`,
  then `b10x-gates bot --repository beyond10x/modelspec -- push origin <version>`.
- `.github/workflows/release.yml` runs on the tag: `release-check`, then `check.yml`, then it builds
  the `modelspec` binary for Linux x86_64 and creates the GitHub Release with the version's
  CHANGELOG section as notes, `modelspec-<version>-x86_64-unknown-linux-gnu` and its `.sha256`.
- A release is finished when the tag, the green `release` run and the Release with both assets are
  verified. Never move or delete a pushed tag; fix forward with the next version.
- Atlas is never on this path. No push, PR landing, release or docs deploy here waits for an Atlas
  pipeline or an Atlas PR; catalog updates in Atlas land on their own schedule afterwards.

## Planning and specification

- Work is planned in the AEP store under `.engineering/` through the `aep` CLI only
  (`aep plan artifact list` to start). Never edit a store file by hand.
- Spec-first: every data contract between compiler, runtime and files on disk (compiled artifact,
  checkpoint, device profile, training-run record) is typed in `spec/` as ESS before code or stories
  use it. Run `ess specify validate --path .` from inside `spec/` after each edit.
  `spec/ess-inputs.yaml` pins the `ess` release; an older installed `ess` delegates to the pin only
  when started inside `spec/`, and refuses `--path spec` from the repository root. A move of that
  pin moves `ESS_PINNED` and `ESS_SHA256` in `.github/workflows/check.yml` in the same commit;
  local `task ci` does not read them, and CI refuses the mismatch (wave m1-h, 2026-10-23).
- The `.mspec` language (grammar, types, shapes, gradient rules) is not an ESS domain; DESIGN.md
  § 2–4 and the tests of DESIGN.md § 9 hold it.
- Generated output (`ess generate`) goes outside `spec/`, beside it, never inside it.

## Rules

- DESIGN.md is the authority. A change to semantics, IR levels, the operator registry or an ADR in
  DESIGN.md § 12 is a design change: edit DESIGN.md in the same change and bump its revision line.
- Keep the seven invariants in DESIGN.md § 0. A change that breaks one is refused, not worked around.
- `modelspec-core` must build and test without Vulkan installed and has no Vulkan dependency under
  any feature or target. Vulkan code sits behind the `vulkan` Cargo feature of `modelspec-runtime`
  (DESIGN.md § 7).
- No Python, PyTorch, CUDA API or CUDA toolkit dependency, in code, tests or tooling.
  Python is allowed only for opponents (the operator, 2026-10-23: "we never want to have any
  1st-level python code in our repo, python is only allowed for opponents"). An opponent is a
  competing implementation, such as a PyTorch CUDA baseline, that a benchmark runs ModelSpec
  against. It lives under `opponents/`, and no build, test, gate or `task` target of ModelSpec
  runs it or depends on it.
- Correctness before speed: the CPU oracle and finite-difference gradient checks (DESIGN.md § 9,
  gates 1–3) exist before any GPU kernel is optimized.
- The files in `examples/` are target syntax. Keep them in step with DESIGN.md § 2 until a parser
  can check them, then make `modelspec check` on every example part of the gate once the parser
  supports its constructs: `linear_regression.mspec` from M0, `mnist_mlp.mspec`, `bert_tiny.mspec`
  since the BERT operators, `transformer_block.mspec` (the benchmark `transformer_block` and its
  opponent `opponents/transformer_block/` run it), `tiny_mlp.mspec` (modules, generics, `vulkan(0)`) from DESIGN.md § 11
  iteration 3.
- Use `Taskfile.yml` (go-task) for build and test entry points once code exists; not a Makefile.
- `website/docs/reference/`, `website/docs/examples/` and `website/data/` are generated by
  `modelspec-docs`: change the code, then `task docs-generate`; never edit them by hand. So are
  `website/docs/benchmarks.md` and `website/static/img/benchmarks/`, from `benchmarks/runs.yaml`: a
  new benchmark run is a new entry there that cites the verification report (and raw records, when
  the report holds only medians) stating its figures, then `task docs-generate`. The other
  pages under `website/docs/` are hand-written; update them in the change that alters what they
  describe. The pitch deck is `docs/pitch/index.html`, which the site build serves at
  `/modelspec/pitch/`; never copy it into `website/`.
- A copy of the tree built outside the worktree (an adversary's mutant or base copy) has its
  `target/` removed with `cargo clean --target-dir <dir>` when its unit merges. Twelve such
  targets of merged waves held 9.6 GB on 2026-10-23.
- Agents run GPU tests and benchmarks on the local GPU themselves, whenever the work needs one:
  `task ci`, `--ignored` device tests, benches and timed runs. There is no separate warm-up run
  before a GPU run; a bench's own warm-up steps outside its timed span stay. A timed run gets the
  GPU alone, and nobody locks its clocks. The operator said so on 2026-10-23
  (approval-record:benchmark-wave-scope D14).
