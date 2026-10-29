//! `modelspec train --backend vulkan` creates its pipelines through a pipeline cache kept on disk
//! (story:vulkan-pipeline-startup): the directory `MODELSPEC_PIPELINE_CACHE` names gets one file
//! for the device, a later run reads it, a corrupt or foreign file is ignored with one warning on
//! stderr and replaced, and stdout, every loss and every parameter, has the same bytes in every
//! case and with the cache `off`. `--stats` names the pipelines made and the time spent making
//! them.
//!
//! Two runs: the hidden-32 twin of BERT-tiny with `steps = 3` and `warmup_steps = 1` (dropout and
//! AdamW, on packed data from the synthetic corpus of `common/markov_corpus.rs`, so the stepwise
//! run of `vulkan::stepwise`) and `examples/mnist_mlp.mspec` for 3 steps
//! on its synthetic data (the chunked run of `vulkan::runner`).
//!
//! The device tests are ignored by default; run them with
//! `cargo test -p modelspec-cli --features vulkan --test pipeline_cache_cli -- --ignored`.
#![cfg(feature = "vulkan")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use modelspec_runtime::vulkan::pipeline_cache::{DIRECTORY_ENV, encode};

#[path = "common/markov_corpus.rs"]
mod markov_corpus;

const TWIN: &str = "crates/modelspec-cli/tests/fixtures/bert_cpu.mspec";
const MNIST_MLP: &str = "examples/mnist_mlp.mspec";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh, empty directory under this crate's build directory, named for the test.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("pipeline_cache_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// `modelspec train <file> --backend vulkan --steps 3 --stats` with the cache directory `cache`.
fn train(file: &str, cache: &str, extra: &[&str]) -> Output {
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args([
            "train",
            file,
            "--backend",
            "vulkan",
            "--steps",
            "3",
            "--stats",
        ])
        .args(extra)
        .env(DIRECTORY_ENV, cache)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{file}: stdout {}\nstderr {}",
        text(&output.stdout),
        text(&output.stderr)
    );
    output
}

/// The `warning:` lines of stderr.
fn warnings(output: &Output) -> Vec<String> {
    text(&output.stderr)
        .lines()
        .filter(|line| line.starts_with("warning:"))
        .map(str::to_owned)
        .collect()
}

/// The one file the cache directory holds.
fn cache_file(dir: &Path) -> PathBuf {
    let files: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("the cache directory exists")
        .map(|entry| entry.expect("an entry").path())
        .collect();
    assert_eq!(files.len(), 1, "one cache file: {files:?}");
    assert!(
        files[0]
            .extension()
            .is_some_and(|extension| extension == "vkpc"),
        "{files:?}"
    );
    files[0].clone()
}

/// The value of the `--stats` line `<name>: <value>`.
fn stat<'o>(stderr: &'o str, name: &str) -> &'o str {
    stderr
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{name}: ")))
        .unwrap_or_else(|| panic!("no `{name}:` line in {stderr}"))
}

/// Cold, warm, corrupt, foreign, short and `off` runs of `file`: the same stdout, a file kept
/// for the device, one warning for each file that is not this device's.
fn cache_cases(file: &str, name: &str, extra: &[&str]) {
    let dir = scratch(name);
    let cache = dir.join("cache");
    let cache_text = cache.to_str().expect("a UTF-8 path");

    let cold = train(file, cache_text, extra);
    assert_eq!(warnings(&cold), Vec::<String>::new());
    let stderr = text(&cold.stderr);
    let pipelines: u64 = stat(&stderr, "pipelines").parse().expect("a count");
    assert!(pipelines > 0, "{stderr}");
    let creation: f64 = stat(&stderr, "pipeline creation us")
        .parse()
        .expect("microseconds");
    assert!(creation > 0.0, "{stderr}");
    let kept = cache_file(&cache);
    let written = std::fs::read(&kept).expect("the cache file reads");

    let warm = train(file, cache_text, extra);
    assert_eq!(warnings(&warm), Vec::<String>::new());
    assert_eq!(text(&warm.stdout), text(&cold.stdout), "{name}: warm run");
    assert_eq!(
        stat(&text(&warm.stderr), "pipelines"),
        pipelines.to_string()
    );

    let mut flipped = written.clone();
    let last = flipped.len() - 1;
    flipped[last] ^= 0x5a;
    let bad: [(&str, Vec<u8>, &str); 3] = [
        (
            "corrupt",
            flipped,
            "its data does not match the hash in its header",
        ),
        (
            "foreign",
            encode("modelspec-pipeline-cache/1 another device", &written[80..]),
            "it was written for another device or driver",
        ),
        (
            "short",
            b"not a cache".to_vec(),
            "shorter than the 80-byte header",
        ),
    ];
    for (case, bytes, reason) in bad {
        std::fs::write(&kept, bytes).expect("the cache file is replaced");
        let run = train(file, cache_text, extra);
        let seen = warnings(&run);
        assert_eq!(seen.len(), 1, "{name} {case}: {seen:?}");
        assert!(
            seen[0].starts_with(&format!(
                "warning: pipeline cache {} ignored: ",
                kept.display()
            )) && seen[0].contains(reason),
            "{name} {case}: {}",
            seen[0]
        );
        assert_eq!(
            text(&run.stdout),
            text(&cold.stdout),
            "{name} {case}: the bits of the cold run"
        );
        let after = train(file, cache_text, extra);
        assert_eq!(
            warnings(&after),
            Vec::<String>::new(),
            "{name} {case}: the ignored file was replaced"
        );
    }

    let none = dir.join("none");
    let off = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(["train", file, "--backend", "vulkan", "--steps", "3"])
        .args(extra)
        .env(DIRECTORY_ENV, "off")
        .env("XDG_CACHE_HOME", &none)
        .env("HOME", &none)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs");
    assert_eq!(off.status.code(), Some(0), "{}", text(&off.stderr));
    assert_eq!(text(&off.stdout), text(&cold.stdout), "{name}: cache off");
    assert!(!none.exists(), "{name}: `off` writes no file");
}

fn ok(args: &[&str]) {
    let output = Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs");
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
}

/// The synthetic corpus of `common/markov_corpus.rs` prepared, a 300-token tokenizer trained on
/// it and the two packed into sequences of 16 under `dir`, as `vulkan_bert_cpu_steps.rs` makes
/// them; returns the packed directory.
fn packed(dir: &Path) -> String {
    let at = |name: &str| dir.join(name).to_str().expect("a UTF-8 path").to_owned();
    markov_corpus::write(&dir.join("corpus"));
    ok(&["data", "prepare", &at("corpus"), "--out", &at("dataset")]);
    ok(&[
        "data",
        "tokenizer",
        "--dataset",
        &at("dataset"),
        "--vocab-size",
        "300",
        "--out",
        &at("tokenizer.json"),
    ]);
    ok(&[
        "data",
        "pack",
        "--dataset",
        &at("dataset"),
        "--tokenizer",
        &at("tokenizer.json"),
        "--seq-len",
        "16",
        "--out",
        &at("packed"),
    ]);
    at("packed")
}

/// The twin with `steps = 3` and `warmup_steps = 1`, written to `<dir>/twin3.mspec`.
fn twin3(dir: &Path) -> String {
    let mut source = std::fs::read_to_string(repo_root().join(TWIN)).expect("the fixture");
    for (from, to) in [
        ("steps = 800;", "steps = 3;"),
        ("warmup_steps = 80)", "warmup_steps = 1)"),
    ] {
        assert!(source.contains(from), "the fixture holds {from:?}");
        source = source.replace(from, to);
    }
    let file = dir.join("twin3.mspec");
    std::fs::write(&file, source).expect("the variant is written");
    file.to_str().expect("a UTF-8 path").to_owned()
}

#[test]
#[ignore = "needs a Vulkan device"]
fn the_stepwise_run_keeps_its_pipelines_on_disk_and_ignores_a_bad_file() {
    let dir = scratch("twin-source");
    let data = packed(&dir);
    cache_cases(&twin3(&dir), "stepwise", &["--data", &data]);
}

#[test]
#[ignore = "needs a Vulkan device"]
fn the_chunked_run_keeps_its_pipelines_on_disk_and_ignores_a_bad_file() {
    cache_cases(MNIST_MLP, "chunked", &[]);
}
