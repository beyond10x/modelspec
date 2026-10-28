//! Adversary of story:jepa-sigreg-vulkan: held-out evaluation of a program with `sigreg` on
//! Vulkan. The stepwise run copies each site's projection of step `k` into the command buffer of
//! every `HeldOut` batch it evaluates at step `k` (`vulkan/stepwise.rs`, `projection_regions`);
//! the unit's own tests never evaluate held-out batches with `sigreg`. The hidden-32 BERT twin
//! with `sigreg(key_mask, slices = 4)` added to its loss is trained on packed data with
//! `--eval-every 2` on both backends, and every `eval` loss of the device run is within the run
//! tolerance of the CPU's.
//!
//! Run it with
//! `cargo test -p modelspec-cli --test adversary_sigreg_vulkan_eval -- --ignored --test-threads=1`.
#![cfg(feature = "vulkan")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[allow(dead_code)]
#[path = "../../modelspec-runtime/tests/common/mod.rs"]
mod common;

#[allow(dead_code)]
#[path = "common/markov_corpus.rs"]
mod markov_corpus;

use common::{GPU_RUN_ATOL, GPU_RUN_RTOL, within};

const TWIN: &str = "crates/modelspec-cli/tests/fixtures/bert_cpu.mspec";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("adversary_sigreg_vulkan_eval_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn ok(output: &Output) {
    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout: {}\nstderr: {}",
        text(&output.stdout),
        text(&output.stderr)
    );
}

fn path(p: &Path) -> &str {
    p.to_str().expect("a UTF-8 path")
}

fn packed(case: &Path) -> PathBuf {
    let corpus = case.join("corpus");
    markov_corpus::write_with(&corpus, 12);
    let dataset = case.join("dataset");
    ok(&modelspec(&[
        "data",
        "prepare",
        path(&corpus),
        "--out",
        path(&dataset),
    ]));
    let tokenizer = case.join("tokenizer.json");
    ok(&modelspec(&[
        "data",
        "tokenizer",
        "--dataset",
        path(&dataset),
        "--vocab-size",
        "300",
        "--out",
        path(&tokenizer),
    ]));
    let packed = case.join("packed");
    ok(&modelspec(&[
        "data",
        "pack",
        "--dataset",
        path(&dataset),
        "--tokenizer",
        path(&tokenizer),
        "--seq-len",
        "16",
        "--out",
        path(&packed),
    ]));
    packed
}

/// The twin with `sigreg(key_mask, slices = 4)` added to its loss, 6 steps, warm-up 2.
fn twin_with_sigreg(case: &Path) -> String {
    let mut source = std::fs::read_to_string(repo_root().join(TWIN)).expect("the fixture");
    for (from, to) in [
        ("steps = 800;", "steps = 6;"),
        ("warmup_steps = 80)", "warmup_steps = 2)"),
        (
            "loss = cross_entropy(model.forward(ids, positions, key_mask), labels, weights);",
            "loss = add(cross_entropy(model.forward(ids, positions, key_mask), labels, weights), sigreg(key_mask, slices = 4));",
        ),
    ] {
        assert!(source.contains(from), "the fixture holds {from:?}");
        source = source.replace(from, to);
    }
    let file = case.join("twin_sigreg.mspec");
    std::fs::write(&file, source).expect("written");
    path(&file).to_owned()
}

fn eval_losses(out: &str) -> Vec<(u64, f32)> {
    out.lines()
        .filter_map(|line| line.strip_prefix("eval "))
        .map(|rest| {
            let words: Vec<&str> = rest.split_whitespace().collect();
            (words[0].parse().unwrap(), words[2].parse().unwrap())
        })
        .collect()
}

#[test]
#[ignore = "needs a Vulkan device"]
fn adv_heldout_losses_with_sigreg_on_vulkan_match_the_cpu() {
    let case = scratch("eval");
    let data = packed(&case);
    let file = twin_with_sigreg(&case);
    let run = |backend: &str| {
        let output = modelspec(&[
            "train",
            &file,
            "--backend",
            backend,
            "--data",
            path(&data),
            "--eval-every",
            "2",
        ]);
        ok(&output);
        text(&output.stdout)
    };
    let cpu = run("cpu");
    let gpu = run("vulkan");
    let (c, g) = (eval_losses(&cpu), eval_losses(&gpu));
    assert_eq!(
        c.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        [0, 2, 4, 6],
        "{cpu}"
    );
    assert_eq!(
        g.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        [0, 2, 4, 6],
        "{gpu}"
    );
    let outside: Vec<String> = c
        .iter()
        .zip(&g)
        .filter(|((_, c), (_, g))| !within(*g, *c, c.abs(), GPU_RUN_ATOL, GPU_RUN_RTOL))
        .map(|((k, c), (_, g))| format!("eval {k}: gpu {g} cpu {c}"))
        .collect();
    eprintln!("cpu evals {c:?}\ngpu evals {g:?}");
    assert!(outside.is_empty(), "{outside:#?}");
}
