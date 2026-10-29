//! Adversarial cases for the hand-written documentation pages and the pitch deck.
//!
//! `task docs-check` compares only the generated pages with their generators. Nothing compares a
//! hand-written page with the command line it quotes, so these cases run the commands a page
//! shows through `modelspec_cli::dispatch` and compare the output with the page, and read the
//! pages' claims against the files they describe.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::Parser;
use serde_yaml_ng::Value;

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn page(relative: &str) -> String {
    fs::read_to_string(repository().join("website/docs").join(relative)).unwrap()
}

/// A fresh, empty directory for `case`.
fn scratch(case: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary-docs")
        .join(case);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Runs `modelspec <args>` in-process and returns its exit code, stdout and stderr.
fn modelspec(args: &[&str]) -> (u8, String, String) {
    let argv = std::iter::once("modelspec").chain(args.iter().copied());
    let cli = modelspec_cli::Cli::try_parse_from(argv).unwrap();
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let code = modelspec_cli::dispatch(cli.command, &mut out, &mut err);
    (
        code,
        String::from_utf8(out).unwrap(),
        String::from_utf8(err).unwrap(),
    )
}

/// Every fenced block of `markdown`, as (info string, body).
fn fences(markdown: &str) -> Vec<(String, String)> {
    let mut blocks = Vec::new();
    let mut open: Option<(String, Vec<&str>)> = None;
    for line in markdown.lines() {
        match open.take() {
            None => {
                if let Some(info) = line.strip_prefix("```") {
                    open = Some((info.trim().to_owned(), Vec::new()));
                }
            }
            Some((info, mut body)) => {
                if line.trim_end() == "```" {
                    blocks.push((info, body.join("\n")));
                } else {
                    body.push(line);
                    open = Some((info, body));
                }
            }
        }
    }
    blocks
}

/// The body of the first `text` fence after the `sh` fence whose body is `command`.
fn output_after(markdown: &str, command: &str) -> String {
    let blocks = fences(markdown);
    let at = blocks
        .iter()
        .position(|(info, body)| info == "sh" && body.trim() == command)
        .unwrap_or_else(|| panic!("no sh fence `{command}`"));
    blocks[at + 1..]
        .iter()
        .find(|(info, _)| info == "text")
        .map(|(_, body)| body.clone())
        .unwrap_or_else(|| panic!("no text fence after `{command}`"))
}

/// The front matter of `markdown`, line by line.
fn front_matter(markdown: &str) -> Vec<String> {
    let mut lines = markdown.lines();
    assert_eq!(lines.next(), Some("---"));
    lines
        .take_while(|line| *line != "---")
        .map(str::to_owned)
        .collect()
}

fn trimmed_lines(text: &str) -> Vec<String> {
    text.lines()
        .map(|line| line.trim_end().to_owned())
        .collect()
}

/// The data pipeline of guides/prepare-training-data.md, run on the fixture it names, into
/// `dir`; returns the stdout of each of the four commands.
fn run_the_guide(dir: &Path) -> [String; 4] {
    let root = repository();
    let corpus = root.join("crates/modelspec-runtime/tests/fixtures/corpus-tokenizer");
    let model = root.join("crates/modelspec-cli/tests/fixtures/mlm_bag.mspec");
    let (dataset, tokenizer, packed) = (
        dir.join("dataset"),
        dir.join("tokenizer.json"),
        dir.join("packed"),
    );
    let s = |p: &Path| p.to_str().unwrap().to_owned();
    let (code, prepare, err) = modelspec(&["data", "prepare", &s(&corpus), "--out", &s(&dataset)]);
    assert_eq!(code, 0, "{err}");
    let (code, tok, err) = modelspec(&[
        "data",
        "tokenizer",
        "--dataset",
        &s(&dataset),
        "--vocab-size",
        "300",
        "--out",
        &s(&tokenizer),
    ]);
    assert_eq!(code, 0, "{err}");
    let (code, pack, err) = modelspec(&[
        "data",
        "pack",
        "--dataset",
        &s(&dataset),
        "--tokenizer",
        &s(&tokenizer),
        "--seq-len",
        "16",
        "--out",
        &s(&packed),
    ]);
    assert_eq!(code, 0, "{err}");
    let (code, train, err) = modelspec(&[
        "train",
        &s(&model),
        "--backend",
        "cpu",
        "--data",
        &s(&packed),
    ]);
    assert_eq!(code, 0, "{err}");
    [prepare, tok, pack, train]
}

/// The pasted output of every command of the data guide equals a run of that command in this
/// tree: the ids and digests are content hashes, so a rerun on the fixture gives the same bytes.
#[test]
fn adversary_the_data_guide_transcripts_equal_a_run_in_the_tree() {
    let guide = page("guides/prepare-training-data.md");
    let dir = scratch("data-guide");
    let [prepare, tokenizer, pack, train] = run_the_guide(&dir);
    assert_eq!(
        trimmed_lines(&prepare),
        trimmed_lines(&output_after(
            &guide,
            "modelspec data prepare crates/modelspec-runtime/tests/fixtures/corpus-tokenizer --out dataset"
        ))
    );
    assert_eq!(
        trimmed_lines(&tokenizer),
        trimmed_lines(&output_after(
            &guide,
            "modelspec data tokenizer --dataset dataset --vocab-size 300 --out tokenizer.json"
        ))
    );
    assert_eq!(
        trimmed_lines(&pack),
        trimmed_lines(&output_after(
            &guide,
            "modelspec data pack --dataset dataset --tokenizer tokenizer.json --seq-len 16 --out packed"
        ))
    );
    let cut: Vec<String> = train
        .lines()
        .map(|line| line.chars().take(72).collect::<String>())
        .collect();
    let pasted = output_after(
        &guide,
        "modelspec train mlm_bag.mspec --backend cpu --data packed | cut -c 1-72",
    );
    assert_eq!(cut, pasted.lines().map(str::to_owned).collect::<Vec<_>>());
    let first = train.lines().next().unwrap();
    assert!(
        fences(&guide)
            .iter()
            .any(|(info, body)| info == "text" && body.trim() == first),
        "the guide's full data line is not the run's first line: {first}"
    );
    let binding: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("packed/binding.json")).unwrap())
            .unwrap();
    for (field, value) in [
        (
            "binding_id",
            binding["binding_id"].as_str().unwrap().to_owned(),
        ),
        (
            "dataset_id",
            binding["dataset_id"].as_str().unwrap().to_owned(),
        ),
        (
            "tokenizer_id",
            binding["tokenizer_id"].as_str().unwrap().to_owned(),
        ),
    ] {
        assert!(
            guide.contains(&format!("| `{field}` | `{value}` |")),
            "the guide's binding table has no row `{field}` = {value}"
        );
    }
}

/// The getting-started transcript is the run of `modelspec train linear_regression.mspec
/// --backend cpu` (acceptance 5).
#[test]
fn adversary_the_getting_started_transcript_equals_a_run_in_the_tree() {
    let example = repository().join("examples/linear_regression.mspec");
    let (code, out, err) = modelspec(&["train", example.to_str().unwrap(), "--backend", "cpu"]);
    assert_eq!(code, 0, "{err}");
    let pasted = fences(&page("getting-started.md"))
        .into_iter()
        .find(|(info, body)| info == "text" && body.starts_with("init @LinearReg.w"))
        .expect("the transcript fence")
        .1;
    assert_eq!(trimmed_lines(&out), trimmed_lines(&pasted));
}

/// The E-TARGET-011 output on the BERT operators page, and its exit status 1, for `ir --stage
/// kernel` and for `compile --backend vulkan`, which the page says answers the same.
#[test]
fn adversary_the_bert_page_e_target_011_output_equals_a_run() {
    let fixture = repository().join("crates/modelspec-cli/tests/fixtures");
    let model = fixture.join("linear_regression_muon.mspec");
    let model = model.to_str().unwrap();
    let bert = page("concepts/bert-operators.mdx");
    let pasted = output_after(
        &bert,
        "modelspec ir linear_regression_muon.mspec --stage kernel",
    );
    let relative = |text: &str| text.replace(&format!("{}/", fixture.to_str().unwrap()), "");
    let (code, out, err) = modelspec(&["ir", model, "--stage", "kernel"]);
    assert_eq!(code, 1);
    assert_eq!(
        trimmed_lines(&relative(&format!("{out}{err}"))),
        trimmed_lines(&pasted)
    );
    let dir = scratch("bert-e-target-011");
    let artifact = dir.join("artifact");
    let (code, out, err) = modelspec(&[
        "compile",
        model,
        "--backend",
        "vulkan",
        "--output",
        artifact.to_str().unwrap(),
    ]);
    assert_eq!(code, 1, "compile --backend vulkan: {out}{err}");
    assert_eq!(
        trimmed_lines(&relative(&format!("{out}{err}"))),
        trimmed_lines(&pasted)
    );
}

/// The two IR listings of the pipeline page are `modelspec ir --stage tensor|training` of the
/// example, as the page says.
#[test]
fn adversary_the_pipeline_page_ir_listings_equal_a_run() {
    let example = repository().join("examples/linear_regression.mspec");
    let pipeline = page("concepts/pipeline.md");
    let texts: Vec<String> = fences(&pipeline)
        .into_iter()
        .filter(|(info, _)| info == "text")
        .map(|(_, body)| body)
        .collect();
    for (stage, first) in [
        (
            "tensor",
            "param @LinearReg.w: Tensor<f32, [4, 1]> = xavier_uniform\n",
        ),
        (
            "training",
            "param @LinearReg.w: Tensor<f32, [4, 1]> = xavier_uniform {v0}",
        ),
    ] {
        let (code, out, err) = modelspec(&["ir", example.to_str().unwrap(), "--stage", stage]);
        assert_eq!(code, 0, "{err}");
        let pasted = texts
            .iter()
            .find(|body| body.starts_with(first))
            .unwrap_or_else(|| panic!("no {stage} listing on the page"));
        assert_eq!(
            trimmed_lines(&out),
            trimmed_lines(pasted),
            "--stage {stage}"
        );
    }
}

/// The value of front-matter key `key`, unquoted.
fn field<'a>(fields: &'a [String], key: &str) -> Option<&'a str> {
    fields.iter().find_map(|line| {
        line.strip_prefix(key)
            .and_then(|rest| rest.strip_prefix(':'))
            .map(|value| value.trim().trim_matches('"'))
    })
}

/// The doc ids under `dir/` that `website/sidebars.ts` lists, in its order.
fn sidebar_ids(dir: &str) -> Vec<String> {
    let sidebars = fs::read_to_string(repository().join("website/sidebars.ts")).unwrap();
    sidebars
        .split('\'')
        .skip(1)
        .step_by(2)
        .filter(|id| id.starts_with(&format!("{dir}/")))
        .map(str::to_owned)
        .collect()
}

/// The `docs` skill § 2: every page carries `title`, `sidebar_position` and `description`, and a
/// hand-written concept or guide page also `lede` (one sentence, the page's claim) and `source`
/// (what it was checked against), which is public text: no `/home/` path and no planning id.
/// Every page of `concepts/` and `guides/` is in `sidebars.ts`, and its `sidebar_position` is its
/// place there.
#[test]
fn every_concept_and_guide_page_carries_the_skill_front_matter() {
    let mut wrong = Vec::new();
    let mut pages = 0;
    for dir in ["concepts", "guides"] {
        let listed = sidebar_ids(dir);
        let mut files: Vec<String> = fs::read_dir(repository().join("website/docs").join(dir))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .filter(|name| name.ends_with(".md") || name.ends_with(".mdx"))
            .collect();
        files.sort();
        for name in files {
            pages += 1;
            let relative = format!("{dir}/{name}");
            let id = relative
                .trim_end_matches(".mdx")
                .trim_end_matches(".md")
                .to_owned();
            let fields = front_matter(&page(&relative));
            for key in ["title", "sidebar_position", "description", "lede", "source"] {
                if field(&fields, key).is_none_or(str::is_empty) {
                    wrong.push(format!("{relative}: no {key}"));
                }
            }
            if let Some(lede) = field(&fields, "lede")
                && (!lede.ends_with('.') || lede.trim_end_matches('.').contains(". "))
            {
                wrong.push(format!("{relative}: the lede is not one sentence: {lede}"));
            }
            if let Some(source) = field(&fields, "source") {
                for refused in ["/home/", "story:", "epic:"] {
                    if source.contains(refused) {
                        wrong.push(format!("{relative}: the source names {refused}"));
                    }
                }
            }
            match listed.iter().position(|listed| *listed == id) {
                None => wrong.push(format!("{relative}: not in sidebars.ts")),
                Some(at) => {
                    let position = field(&fields, "sidebar_position").and_then(|p| p.parse().ok());
                    if position != Some(at + 1) {
                        wrong.push(format!(
                            "{relative}: sidebar_position {position:?}, sidebars.ts lists it {}",
                            at + 1
                        ));
                    }
                }
            }
        }
    }
    assert!(pages >= 8, "only {pages} concept and guide pages");
    assert!(wrong.is_empty(), "front matter: {wrong:#?}");
}

/// The overview's MNIST paragraph agrees with `benchmarks/runs.yaml`, charted on the Benchmarks
/// page it links: every timed ModelSpec Vulkan run of the same shapes (784 -> 64 -> 10, batch
/// 100) after the 0.6.0 `train` timing is quoted with its release, as the Benchmarks page rounds
/// it, and the paragraph does not deny them.
#[test]
fn the_overview_mnist_claim_agrees_with_the_benchmark_history() {
    let index = page("index.mdx")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    for denial in [
        "no new timed step time was measured for any later release",
        "not timed again",
    ] {
        assert!(!index.contains(denial), "index.md says \"{denial}\"");
    }
    assert!(
        index.contains("[Benchmarks](/docs/benchmarks)"),
        "index.md does not link the Benchmarks page"
    );
    let runs: Value = serde_yaml_ng::from_str(
        &fs::read_to_string(repository().join("benchmarks/runs.yaml")).unwrap(),
    )
    .unwrap();
    let later: Vec<(String, String)> = runs["runs"]
        .as_sequence()
        .unwrap()
        .iter()
        .filter(|run| {
            run["side"].as_str() == Some("ModelSpecVulkan")
                && run["mode"].as_str() == Some("Timed")
                && run["batch"].as_u64() == Some(100)
                && run["hidden"].as_u64() == Some(64)
                && !matches!(
                    run["modelspec_version"].as_str(),
                    Some("0.4.0" | "0.5.0" | "0.6.0") | None
                )
        })
        .map(|run| {
            (
                run["modelspec_version"].as_str().unwrap().to_owned(),
                format!("{:.3} µs", run["median_us_per_step"].as_f64().unwrap()),
            )
        })
        .collect();
    assert!(
        later.len() >= 2,
        "runs.yaml holds the 0.9.0 and 0.10.0 sweeps"
    );
    for (version, figure) in later {
        assert!(
            index.contains(&figure) && index.contains(&format!("in {version}")),
            "index.md does not quote the {version} run of the MNIST shapes, {figure}"
        );
    }
}

/// The status page says rows that start with "Since" name the release that added them and rows
/// that start with "Not yet released" are on `main` only. Every "Since" names a release of
/// `CHANGELOG.md`. The masked-language-model batches (`crates/modelspec-runtime/tests/mlm.rs`)
/// are first in the 0.10.0 tree, so their row says "Since 0.10.0", and the E-TARGET-011 answer
/// for `muon` (`crates/modelspec-cli/tests/train_muon.rs`) is first in the 0.15.0 tree, so its
/// row says "Since 0.15.0".
#[test]
fn shipped_rows_name_the_release_that_added_them() {
    let root = repository();
    let changelog = fs::read_to_string(root.join("CHANGELOG.md")).unwrap();
    let status: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("website/data/status.json")).unwrap())
            .unwrap();
    let items = status["items"].as_array().unwrap();
    let mut wrong = Vec::new();
    for item in items {
        let detail = item["detail"].as_str().unwrap();
        if let Some(rest) = detail.strip_prefix("Since ") {
            let version = rest.split(':').next().unwrap();
            if !changelog.contains(&format!("\n## [{version}] - ")) {
                wrong.push(format!("{}: no release {version}", item["label"]));
            }
        }
    }
    for (label, release) in [
        ("Masked-language-model batches", "0.10.0"),
        ("No Vulkan kernel yet for muon", "0.15.0"),
    ] {
        let row = items
            .iter()
            .find(|item| item["label"] == label)
            .unwrap_or_else(|| panic!("no row {label}"));
        let detail = row["detail"].as_str().unwrap();
        if !detail.starts_with(&format!("Since {release}: ")) {
            wrong.push(format!("{label} does not name its release: {detail}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// The speaker notes of the pitch deck are served with the site, so they are public text: a
/// `CHANGELOG.md` cite names a release heading (`CHANGELOG.md § 0.7.1`), which does not move when
/// the file grows, never a line; and no planning id (`<kind>:` for every kind of the planning
/// store), planning path or planning command is left.
#[test]
fn the_pitch_notes_cite_changelog_releases_and_name_no_planning_record() {
    let root = repository();
    let changelog = fs::read_to_string(root.join("CHANGELOG.md")).unwrap();
    let deck = fs::read_to_string(root.join("docs/pitch/index.html")).unwrap();
    let mut wrong = Vec::new();
    let mut cites = 0;
    for (at, _) in deck.match_indices("CHANGELOG.md") {
        let rest = &deck[at + "CHANGELOG.md".len()..];
        if rest.starts_with(':') {
            let cite: String = rest.chars().take(12).collect();
            wrong.push(format!("a line cite CHANGELOG.md{cite}"));
        } else if let Some(heading) = rest.strip_prefix(" § ") {
            cites += 1;
            let version: String = heading
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '.')
                .collect();
            let version = version.trim_end_matches('.');
            if !changelog.contains(&format!("\n## [{version}]")) {
                wrong.push(format!("CHANGELOG.md § {version}: no such heading"));
            }
        }
    }
    for (at, _) in deck.match_indices(" :") {
        if deck[at + 2..].starts_with(|c: char| c.is_ascii_digit()) {
            let cite: String = deck[at..].chars().take(12).collect();
            wrong.push(format!("a bare line cite{cite}"));
        }
    }
    let mut refused: Vec<String> = fs::read_dir(root.join(".engineering/planning"))
        .unwrap()
        .map(|entry| entry.unwrap())
        .filter(|entry| entry.file_type().unwrap().is_dir())
        .map(|entry| format!("{}:", entry.file_name().into_string().unwrap()))
        .collect();
    assert!(
        refused.iter().any(|kind| kind == "story:") && refused.len() > 3,
        "the planning kinds: {refused:?}"
    );
    refused.extend([".engineering/".to_owned(), "aep plan".to_owned()]);
    for refused in refused {
        let count = deck.matches(&refused).count();
        if count > 0 {
            wrong.push(format!("{count} × {refused}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
    assert!(cites > 10, "only {cites} CHANGELOG.md § cites");
}

/// A copy of the repository under `CARGO_TARGET_TMPDIR`, without `target`, `.git`,
/// `node_modules` and the built site.
fn sandbox(case: &str) -> PathBuf {
    fn copy(from: &Path, to: &Path) {
        fs::create_dir_all(to).unwrap();
        for entry in fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name();
            if matches!(
                name.to_str(),
                Some("target" | ".git" | "node_modules" | "build" | ".docusaurus")
            ) {
                continue;
            }
            let kind = entry.file_type().unwrap();
            if kind.is_dir() {
                copy(&entry.path(), &to.join(&name));
            } else if kind.is_file() {
                fs::copy(entry.path(), to.join(&name)).unwrap();
            }
        }
    }
    let root = scratch(&format!("sandbox-{case}"));
    copy(&repository(), &root);
    root
}

fn generate_check(root: &Path) -> (Option<i32>, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec-docs"))
        .args(["generate", "--check", "--root"])
        .arg(root)
        .output()
        .unwrap();
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn replace_once(path: &Path, from: &str, to: &str) {
    let text = fs::read_to_string(path).unwrap();
    assert!(text.contains(from), "{} holds no `{from}`", path.display());
    fs::write(path, text.replacen(from, to, 1)).unwrap();
}

/// Mutant (a): a shipped row whose holding test is gone. `task docs-check` must refuse it.
#[test]
fn adversary_docs_check_refuses_a_shipped_row_whose_test_is_gone() {
    let root = sandbox("test-gone");
    replace_once(
        &root.join("crates/modelspec-cli/tests/data_pack.rs"),
        "fn sequences(",
        "fn sequences_renamed(",
    );
    let (code, stderr) = generate_check(&root);
    assert_ne!(code, Some(0), "docs-check passed with the evidence gone");
    assert!(
        stderr.contains("Packed sequences of an exact length")
            && stderr.contains("defines no fn sequences"),
        "{stderr}"
    );
}

/// Mutant (d): a raw admonition title on a hand-written page. `task docs-check` must refuse it.
#[test]
fn adversary_docs_check_refuses_a_raw_admonition_on_a_new_page() {
    let root = sandbox("raw-admonition");
    replace_once(
        &root.join("website/docs/guides/prepare-training-data.md"),
        ":::caution[Planned]",
        ":::caution Planned",
    );
    let (code, stderr) = generate_check(&root);
    assert_ne!(code, Some(0), "docs-check passed a raw admonition");
    assert!(stderr.contains("prepare-training-data.md"), "{stderr}");
}
