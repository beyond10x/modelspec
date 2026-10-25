//! Adversary pass 1 on story:bert-tiny-cpu-model: whether the twin's training run shows
//! learning, whether its seed is read, whether the twin equals the example beyond its sizes, and
//! whether the example's vocabulary is that of the corpus tokenizer the repository specifies.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const CORPUS: &str = "crates/modelspec-runtime/tests/fixtures/corpus";
const EXAMPLE: &str = "examples/bert_tiny.mspec";
const TWIN: &str = "crates/modelspec-cli/tests/fixtures/bert_cpu.mspec";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("adversary_bert_tiny_{name}"));
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

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn assert_exit(output: &Output, code: i32) {
    assert_eq!(
        output.status.code(),
        Some(code),
        "stdout: {}\nstderr: {}",
        stdout(output),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn path(p: &Path) -> &str {
    p.to_str().expect("a UTF-8 path")
}

/// The fixture corpus with a `vocab`-token tokenizer packed into sequences of 16 under `case`.
fn packed(case: &Path, vocab: u32) -> PathBuf {
    let dataset = case.join("dataset");
    assert_exit(
        &modelspec(&["data", "prepare", CORPUS, "--out", path(&dataset)]),
        0,
    );
    let tokenizer = case.join("tokenizer.json");
    let vocab = vocab.to_string();
    assert_exit(
        &modelspec(&[
            "data",
            "tokenizer",
            "--dataset",
            path(&dataset),
            "--vocab-size",
            &vocab,
            "--out",
            path(&tokenizer),
        ]),
        0,
    );
    let packed = case.join("packed");
    assert_exit(
        &modelspec(&[
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
        ]),
        0,
    );
    packed
}

/// The `(step, loss)` of every `eval <k> loss <v> accuracy <a>` line.
fn evals(stdout: &str) -> Vec<(u64, f64)> {
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix("eval "))
        .map(|rest| {
            let mut words = rest.split_whitespace();
            let step = words.next().unwrap().parse().unwrap();
            assert_eq!(words.next(), Some("loss"), "{rest}");
            (step, words.next().unwrap().parse().unwrap())
        })
        .collect()
}

/// The lines that start with `step `.
fn step_lines(stdout: &str) -> Vec<&str> {
    stdout.lines().filter(|l| l.starts_with("step ")).collect()
}

/// ADV-TINY-1 (H5): the twin's 500 steps on the packed fixture corpus learn something a
/// unigram model knows: the held-out loss of its last `eval` line is below the held-out unigram
/// baseline `data pack` records in `packed.json`. `reproducible` asserts only that the last
/// `step` loss (one batch of 2 sequences) is below the step-0 loss, which a run at
/// `lr = 1e-9` also passes.
#[test]
#[ignore = "ADV-TINY-1: red, the twin's held-out loss stays above the unigram baseline"]
fn adv_tiny_1_twin_heldout_loss_falls_below_the_unigram_baseline() {
    let case = scratch("heldout");
    let packed = packed(&case, 300);
    let record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(packed.join("packed.json")).unwrap())
            .unwrap();
    let baseline = record["heldout_unigram_baseline_nats"].as_f64().unwrap();
    let out = modelspec(&[
        "train",
        TWIN,
        "--backend",
        "cpu",
        "--data",
        path(&packed),
        "--eval-every",
        "100",
    ]);
    assert_exit(&out, 0);
    let out = stdout(&out);
    let evals = evals(&out);
    let (first_step, first) = evals[0];
    let (last_step, last) = *evals.last().unwrap();
    assert!(
        last < baseline,
        "held-out loss {first} at eval {first_step}, {last} at eval {last_step}; the unigram \
         baseline is {baseline}; ln(300) = {}",
        300f64.ln()
    );
}

/// H5: the seed is read: the twin with `seed = 7` instead of 42 prints other `step` lines over
/// the same 60 steps of the same data.
#[test]
fn another_seed_prints_other_step_lines() {
    let case = scratch("seed");
    let packed = packed(&case, 300);
    let source = std::fs::read_to_string(repo_root().join(TWIN)).unwrap();
    assert!(source.contains("seed = 42;"));
    let other = case.join("bert_cpu_seed7.mspec");
    std::fs::write(&other, source.replace("seed = 42;", "seed = 7;")).unwrap();
    let run = |file: &str| {
        let out = modelspec(&[
            "train",
            file,
            "--backend",
            "cpu",
            "--data",
            path(&packed),
            "--steps",
            "60",
        ]);
        assert_exit(&out, 0);
        stdout(&out)
    };
    let a = run(TWIN);
    let b = run(path(&other));
    assert_eq!(step_lines(&a).len(), step_lines(&b).len());
    assert_ne!(
        step_lines(&a),
        step_lines(&b),
        "seed 42 and seed 7 print the same steps"
    );
}

/// The source without its leading comment block.
fn body(file: &str) -> Vec<String> {
    std::fs::read_to_string(repo_root().join(file))
        .unwrap()
        .lines()
        .skip_while(|l| l.trim().is_empty() || l.trim_start().starts_with("//"))
        .map(str::to_owned)
        .collect()
}

/// `line` with the text inside every `[...]` replaced by `#`.
fn without_shapes(line: &str) -> String {
    let mut out = String::new();
    let mut depth = 0;
    for c in line.chars() {
        match c {
            '[' => {
                depth += 1;
                out.push_str("[#");
            }
            ']' => {
                depth -= 1;
                out.push(']');
            }
            _ if depth > 0 => {}
            _ => out.push(c),
        }
    }
    out
}

/// The `name = <value>` of `line`, if it has one.
fn setting<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let at = line.find(&format!("{name} = "))? + name.len() + 3;
    let rest = &line[at..];
    let end = rest.find([',', ')', ';']).unwrap_or(rest.len());
    Some(&rest[..end])
}

/// H3: the twin is the example with other shapes and another run length, and nothing else: line
/// by line, outside `[...]` the two agree except for `steps = ` and the optimizer's
/// `warmup_steps = `, and both warm up over 10 % of their steps (coordinator decision D70 (b)).
/// `twin_is_the_example_with_other_sizes` masks every digit, so `heads`, `p`, `lr`,
/// `weight_decay`, the betas, `eps`, `seed` and the layer index of a parameter may differ there.
#[test]
fn twin_differs_from_the_example_only_in_shapes_and_run_length() {
    let (example, twin) = (body(EXAMPLE), body(TWIN));
    assert_eq!(example.len(), twin.len());
    for (e, t) in example.iter().zip(&twin) {
        let (e0, t0) = (without_shapes(e), without_shapes(t));
        if e0 == t0 {
            continue;
        }
        if setting(e, "steps").is_some() && e.trim_start().starts_with("steps") {
            continue;
        }
        let (Some(we), Some(wt)) = (setting(e, "warmup_steps"), setting(t, "warmup_steps")) else {
            panic!("the twin differs outside its shapes:\n  {e}\n  {t}");
        };
        assert_eq!(
            e0.replace(we, "#"),
            t0.replace(wt, "#"),
            "the optimizer differs beyond warmup_steps"
        );
    }
    let ratio = |lines: &[String]| -> (u64, u64) {
        let steps = lines
            .iter()
            .find(|l| l.trim_start().starts_with("steps = "))
            .and_then(|l| setting(l, "steps"))
            .unwrap()
            .parse()
            .unwrap();
        let warmup = lines
            .iter()
            .find_map(|l| setting(l, "warmup_steps"))
            .unwrap()
            .parse()
            .unwrap();
        (steps, warmup)
    };
    for (file, lines) in [(EXAMPLE, &example), (TWIN, &twin)] {
        let (steps, warmup) = ratio(lines);
        assert_eq!(
            warmup * 10,
            steps,
            "{file}: warmup {warmup} of {steps} steps"
        );
    }
}

/// The comparison above, applied to a mutated twin in memory: it refuses the digit-only
/// mutations that `twin_is_the_example_with_other_sizes` lets through.
#[test]
fn the_stronger_twin_comparison_refuses_digit_mutants() {
    let example = body(EXAMPLE);
    let twin = body(TWIN);
    let differs = |mutant: &[String]| {
        example.iter().zip(mutant).any(|(e, t)| {
            let (e0, t0) = (without_shapes(e), without_shapes(t));
            e0 != t0
                && !e.trim_start().starts_with("steps")
                && match (setting(e, "warmup_steps"), setting(t, "warmup_steps")) {
                    (Some(we), Some(wt)) => e0.replace(we, "#") != t0.replace(wt, "#"),
                    _ => true,
                }
        })
    };
    assert!(!differs(&twin), "the unmutated twin passes");
    for (what, from, to) in [
        ("one head", "heads = 2", "heads = 1"),
        ("dropout 0.5", "p = 0.1", "p = 0.5"),
        ("lr 1e-9", "lr = 1e-4", "lr = 1e-9"),
        (
            "no weight decay",
            "weight_decay = 0.01",
            "weight_decay = 0.0",
        ),
        ("seed 7", "seed = 42", "seed = 7"),
        (
            "layer 1 reads l0_wq",
            "matmul(x0, l1_wq)",
            "matmul(x0, l0_wq)",
        ),
    ] {
        let mutant: Vec<String> = twin.iter().map(|l| l.replacen(from, to, 1)).collect();
        assert_ne!(mutant, twin, "{what}: the mutation applied");
        assert!(
            differs(&mutant),
            "{what}: the comparison accepts the mutant"
        );
    }
}

/// ADV-TINY-2 (H1, H6): the example's token table has the rows of the WordPiece vocabulary the
/// repository trains on the corpus, 8,192 (`spec/domains/data.yaml` `TokenizerKind`,
/// coordinator decisions D19 and D23: "the real run trains at 8,192 (D19), and story bodies that
/// say 30522 are amended when dispatched"). With 30,522 rows, 22,330 rows of `tok_table` and of
/// `mlm_bias` are never read nor a label on that data, and `train` does not refuse it.
#[test]
#[ignore = "ADV-TINY-2: red, the example's vocabulary is 30,522, the corpus tokenizer's 8,192"]
fn adv_tiny_2_example_vocabulary_is_the_corpus_tokenizer_vocabulary() {
    let spec = std::fs::read_to_string(repo_root().join("spec/domains/data.yaml")).unwrap();
    assert!(
        spec.contains("with a vocabulary of 8,192 tokens"),
        "the spec's TokenizerKind vocabulary"
    );
    let source = std::fs::read_to_string(repo_root().join(EXAMPLE)).unwrap();
    let rows = source
        .lines()
        .find(|l| l.contains("param tok_table"))
        .and_then(|l| l.split_once('[').map(|(_, r)| r))
        .and_then(|r| r.split_once(','))
        .map(|(n, _)| n.trim().parse::<u64>().unwrap())
        .unwrap();
    assert_eq!(rows, 8192, "tok_table rows of {EXAMPLE}");
}

/// ADV-TINY-3: `train` on a packed directory whose `vocabulary_size` is below the rows of the
/// program's token table refuses the run naming both, as it refuses a table narrower than the
/// vocabulary at step 0. The twin (300 rows) on the fixture corpus packed at 200 tokens.
#[test]
#[ignore = "ADV-TINY-3: red, a table wider than the packed vocabulary trains without a word"]
fn adv_tiny_3_train_refuses_a_table_wider_than_the_packed_vocabulary() {
    let case = scratch("vocab");
    let packed = packed(&case, 200);
    let out = modelspec(&[
        "train",
        TWIN,
        "--backend",
        "cpu",
        "--data",
        path(&packed),
        "--steps",
        "60",
    ]);
    assert_ne!(
        out.status.code(),
        Some(0),
        "the 300-row twin trained on a 200-token packed directory:\n{}",
        step_lines(&stdout(&out)).join("\n")
    );
}
