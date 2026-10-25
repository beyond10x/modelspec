# Site review 2026-10-25: answers

One row per id of the review's fix list. Answers follow coordinator decision D58 (2026-10-25,
`.engineering/waves/docs-1.md`): `fixed` names the change and its file, `template` names the
`@beyond10x/docs-system` 0.12.0 file the defect lives in (not patched here), `not a defect` gives
the reason. The commit column is filled in by the coordinator. Template file paths are relative to
the package root (`website/node_modules/@beyond10x/docs-system/`).

| id | answer | commit |
|---|---|---|
| s5-04 | fixed: the language page states the i32 element type for row ids and labels, its rules and E-TYPE-018, `website/docs/concepts/language.md` | ce3032b, b02cadc |
| s5-03 | fixed: no page counts nine operators; the language page and the writing guide point at the operator reference, the status row is "Reverse-mode autodiff for every operator" and the operators page says "The language has 17 operators", `website/docs/concepts/language.md`, `website/docs/guides/write-a-model.md`, `crates/modelspec-docs/src/status.rs`, `crates/modelspec-docs/src/operators.rs` | ce3032b, b02cadc |
| s5-02 | fixed: the pipeline page drops the built/not-built columns and documents Kernel IR with the output of `modelspec ir --stage kernel`, `website/docs/concepts/pipeline.md` | ce3032b, b02cadc |
| s1-01 | template `dist/scrollable-table.js`, `styles/product-theme.css` (`.markdown table th` at line 203, table code at line 209) |  |
| s5-01 | fixed: the status page names 0.10.0 as the latest release with a release table from 0.7.0 to unreleased, and the status list is checked against 0.10.0 (as of 2026-10-25), `website/docs/status.mdx`, `crates/modelspec-docs/src/status.rs` | ce3032b, b02cadc |
| s2-04 | template `styles/product-theme.css` (`.navbar__title::after` sub-brand, lines 170-171 and 347) |  |
| s2-02 | template `styles/tokens.css` (`:not(pre) > code { overflow-wrap: anywhere }`, line 414) |  |
| s2-07 | fixed: the charts are drawn 440 units wide with text of 15 units or more, so labels render at 12.2 px at 390 px, `crates/modelspec-docs/src/benchmarks.rs`, `website/static/img/benchmarks/*.svg` (shots in `shots/`) | ce3032b |
| s3-01 | fixed: the charts have no background and a `prefers-color-scheme: dark` block remaps their colours, `crates/modelspec-docs/src/benchmarks.rs`, `website/static/img/benchmarks/*.svg` (shots in `shots/`) | ce3032b |
| s5-08 | fixed: the training page documents synthetic, dense and packed data, `--data`, `--init` and `--eval` and E-DATA-001 to E-DATA-009; the diagnostic guide's E-DATA row describes the data sources, `website/docs/concepts/training.md`, `website/docs/guides/read-a-diagnostic.md` | ce3032b, b02cadc |
| s5-06 | fixed: the getting-started transcript is the full output of `modelspec train linear_regression.mspec --backend cpu` run from the current tree, `website/docs/getting-started.md` | ce3032b |
| s1-03 | fixed: the CLI reference wraps usage lines at 64 columns on argument boundaries, `crates/modelspec-docs/src/cli.rs`, `website/docs/reference/cli.md` | ce3032b, b02cadc |
| s4-01 | fixed: the `--serve` and `--ui` help of `train` and `observe` says "path / of ADDR" instead of a URL, `crates/modelspec-cli/src/train.rs`, `crates/modelspec-cli/src/observe.rs`, `website/docs/reference/cli.md` | ce3032b |
| s4-03 | template `styles/product-theme.css` (doc sidebar and footer layout, `.theme-doc-sidebar-container` and `.menu` at lines 283-284, `.footer` at line 301) |  |
| s3-03 | template `styles/product-theme.css` (`a { text-underline-offset: 4px; }` at line 137 with no underline, `--ifm-link-color` at lines 92 and 234) |  |
| s5-10 | fixed: the E-TARGET-011 declaration names every BERT operator without a Vulkan kernel, `crates/modelspec-core/src/compile.rs` and the regenerated `website/docs/reference/diagnostics.md` (the coordinator applied the implementor's patch at commit time) | ce3032b |
| s5-07 | fixed: the roadmap lists 0.4.0 to 0.10.0 under M1, says what of M2 shipped early, marks the BERT operators shipped on the CPU reference and keeps only unbuilt work planned, `website/docs/roadmap.md` | ce3032b |
| s3-02 | fixed: the pitch deck reads the site's stored theme, else `prefers-color-scheme`, and has a light palette, `docs/pitch/index.html` | ce3032b, b02cadc |
| s1-07 | template `dist/theme/ProductLandingPage.js`, `styles/product-components.css` (status section cards) |  |
| s1-08 | template `dist/family.js` (`FAMILY_ORDER` has no `aep` entry, so the AEP card is a custom tool card), `styles/product-components.css` (`.b10x-related__card`, line 417) |  |
| s2-09 | template `styles/product-components.css` (`.b10x-flow__label` 0.65rem at line 367, terminal labels), `styles/product-theme.css` (`.pagination-nav__sublabel` 0.7rem at line 295, `.footer__title` 0.72rem at line 304) |  |
| s2-10 | template `styles/product-components.css` (`.b10x-terminal--tilt { transform: rotate(-1deg) }`, line 182) |  |
| s2-12 | template `dist/product.js` (terminal lines), `styles/product-components.css` (`.b10x-terminal__*`) |  |
| s3-04 | template `styles/product-components.css` (`.b10x-terminal__copy { color: #81928d }`, line 206, and the terminal caption) |  |
| s4-06 | template `dist/theme/NotFound/Content/index.js` (no robots noindex), `dist/product-site.js` (canonical rewrite, line 259) |  |
| s4-07 | template `dist/theme/NotFound/Content/index.js` (no description metadata) |  |
| s4-08 | template `dist/product-site.js` (social card `image` set without an alt, line 367) |  |
| s2-08 | template `styles/product-theme.css` (`.markdown table th` font-size 0.68rem, line 203) |  |
| s4-02 | fixed: the edit URL points at `/edit/main/website/`, `website/docusaurus.config.ts` line 67 (the coordinator applied the implementor's patch at commit time) | ce3032b |
| s5-16 | not a defect: `text` fences hold command output, which has no grammar to colour; leaving it uncoloured keeps it apart from source blocks |  |
| s5-14 | fixed: every example page titles its model fence with the file name, `crates/modelspec-docs/src/example.rs`, `website/docs/examples/*.md` | ce3032b |
| s2-11 | template `styles/tokens.css` (`:not(pre) > code` padding `0.12em 0.35em`, line 414) |  |
| s1-05 | template `styles/tokens.css` (`.theme-code-block` code `overflow-wrap: normal`, line 400) |  |
| s5-15 | fixed: the crates page names the current release, read from the workspace version, `crates/modelspec-docs/src/crates.rs`, `website/docs/reference/crates.md` | ce3032b |
| s5-13 | template `styles/tokens.css` (inline code `font-size: 0.88em`, relative to its context, line 414), `styles/product-theme.css` (table code, line 209) |  |
| s4-04 | template `dist/theme/NotFound/Content/index.js` (no page title) |  |
| s4-05 | template `dist/product-site.js` (canonical link built from the requested path, line 259) |  |
