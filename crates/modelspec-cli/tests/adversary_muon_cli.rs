//! Adversary cases for `modelspec` on files trained with `muon(...)` (story:muon-optimizer,
//! adversary pass 1): each refusal of the syntax at its boundary and what it accepts there,
//! the Vulkan paths against the story's acceptance 6, `compile` and `--steps`, and the
//! held-out comparison of acceptance 5 under other seeds.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const LINEAR: &str = "crates/modelspec-cli/tests/fixtures/linear_regression_muon.mspec";
const MLM_DENSE: &str = "crates/modelspec-cli/tests/fixtures/mlm_dense.mspec";
const CORPUS: &str = "crates/modelspec-runtime/tests/fixtures/corpus-tokenizer";
const MUON_LINE: &str = "    optimizer = muon(lr = 0.01, momentum = 0.95, weight_decay = 0.01, ns_steps = 5, warmup_steps = 30, adam_lr = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8);";
const ADAMW_LINE: &str = "    optimizer = adamw(lr = 0.01, weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, warmup_steps = 30);";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("adversary_muon_{name}"));
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

fn path(p: &Path) -> &str {
    p.to_str().expect("a UTF-8 path")
}

/// `source_file` with `from` replaced by `to`, written to `<case>/<name>.mspec`; returns the path
/// and the new source.
fn variant(case: &Path, source_file: &str, name: &str, from: &str, to: &str) -> (String, String) {
    let source = std::fs::read_to_string(repo_root().join(source_file)).expect("the fixture");
    assert!(source.contains(from), "the fixture holds {from:?}");
    let source = source.replacen(from, to, 1);
    let file = case.join(format!("{name}.mspec"));
    std::fs::write(&file, &source).expect("the variant is written");
    (path(&file).to_owned(), source)
}

/// The 1-based line and column of the first `needle` in `source`.
fn position(source: &str, needle: &str) -> (usize, usize) {
    source
        .lines()
        .enumerate()
        .find_map(|(i, line)| line.find(needle).map(|c| (i + 1, c + 1)))
        .unwrap_or_else(|| panic!("the source holds `{needle}`"))
}

/// An edit `from` -> `to` and the refusal it must give, `(code, at)`, or `None` when accepted.
type Edit = (
    &'static str,
    &'static str,
    Option<(&'static str, &'static str)>,
);

/// Hypothesis 6. Each edit of the optimizer line at a boundary: accepted (`None`) or refused
/// with exactly one diagnostic, `code`, at the first `at` of the edited source, exit 1.
#[test]
fn adv_muon_check_boundaries() {
    let case = scratch("boundaries");
    let cases: &[Edit] = &[
        ("momentum = 0.95", "momentum = 0.0", None),
        ("momentum = 0.95", "momentum = 0.999999", None),
        ("momentum = 0.95", "momentum = -0.0", None),
        ("ns_steps = 5", "ns_steps = 1", None),
        ("ns_steps = 5", "ns_steps = 10", None),
        (
            "momentum = 0.95",
            "momentum = -0.5",
            Some(("E-TYPE-025", "momentum =")),
        ),
        (
            "momentum = 0.95",
            "momentum = 0.99999999",
            Some(("E-TYPE-025", "momentum =")),
        ),
        (
            "momentum = 0.95",
            "momentum = nan",
            Some(("E-TYPE-019", "momentum =")),
        ),
        (
            "ns_steps = 5",
            "ns_steps = 5.0",
            Some(("E-TYPE-019", "ns_steps =")),
        ),
        (
            "ns_steps = 5",
            "ns_steps = 4294967301",
            Some(("E-TYPE-025", "ns_steps =")),
        ),
        (
            "adam_lr = 0.05",
            "adam_lr = 0.0",
            Some(("E-TYPE-010", "adam_lr =")),
        ),
        (
            "beta1 = 0.9",
            "beta1 = 1.0",
            Some(("E-TYPE-023", "beta1 =")),
        ),
        ("eps = 1e-8", "eps = 0.0", Some(("E-TYPE-023", "eps = 0.0"))),
        (
            "weight_decay = 0.01",
            "weight_decay = -0.1",
            Some(("E-TYPE-023", "weight_decay =")),
        ),
        (
            "warmup_steps = 10",
            "warmup_steps = 200",
            Some(("E-TYPE-024", "warmup_steps =")),
        ),
        (
            "eps = 1e-8)",
            "eps = 1e-8, nesterov = 1.0)",
            Some(("E-TYPE-002", "nesterov =")),
        ),
        ("muon(lr", "muon(0.1, lr", Some(("E-TYPE-002", "0.1, lr"))),
    ];
    for (k, (from, to, expected)) in cases.iter().enumerate() {
        let (file, source) = variant(&case, LINEAR, &format!("case{k}"), from, to);
        let output = modelspec(&["check", &file]);
        let stderr = text(&output.stderr);
        match expected {
            None => {
                assert_eq!(output.status.code(), Some(0), "{to}: {stderr}");
                assert_eq!(stderr, "", "{to}");
            }
            Some((code, at)) => {
                assert_eq!(output.status.code(), Some(1), "{to}: {stderr}");
                let heads: Vec<&str> = stderr.lines().filter(|l| !l.starts_with("  ")).collect();
                assert_eq!(heads.len(), 1, "{to}: one diagnostic:\n{stderr}");
                assert!(heads[0].starts_with(&format!("{code}: ")), "{to}: {stderr}");
                let (line, column) = position(&source, at);
                assert_eq!(
                    stderr.lines().nth(1),
                    Some(format!("  {file}:{line}:{column}").as_str()),
                    "{to}: location\n{stderr}"
                );
            }
        }
    }
}

/// Hypothesis 6. `muon()` is one E-TYPE-001 at `muon` naming all nine arguments; `momentum =
/// 1.0` and `ns_steps = 0` together are both reported, each at its argument.
#[test]
fn adv_muon_check_empty_call_and_two_refusals() {
    let case = scratch("empty");
    let line = "optimizer = muon(lr = 0.05, momentum = 0.95, weight_decay = 0.01, ns_steps = 5, warmup_steps = 10, adam_lr = 0.05, beta1 = 0.9, beta2 = 0.999, eps = 1e-8);";
    let (file, source) = variant(&case, LINEAR, "empty", line, "optimizer = muon();");
    let output = modelspec(&["check", &file]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    let heads: Vec<&str> = stderr.lines().filter(|l| !l.starts_with("  ")).collect();
    assert_eq!(heads.len(), 1, "{stderr}");
    assert!(heads[0].starts_with("E-TYPE-001: "), "{stderr}");
    for name in [
        "lr",
        "momentum",
        "weight_decay",
        "ns_steps",
        "warmup_steps",
        "adam_lr",
        "beta1",
        "beta2",
        "eps",
    ] {
        assert!(heads[0].contains(&format!("`{name}`")), "{name}: {stderr}");
    }
    let (l, c) = position(&source, "muon(");
    assert_eq!(
        stderr.lines().nth(1),
        Some(format!("  {file}:{l}:{c}").as_str())
    );

    let both = line
        .replace("momentum = 0.95", "momentum = 1.0")
        .replace("ns_steps = 5", "ns_steps = 0");
    let (file, source) = variant(&case, LINEAR, "both", line, &both);
    let output = modelspec(&["check", &file]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    let lines: Vec<&str> = stderr.lines().collect();
    let heads: Vec<(usize, &&str)> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| !l.starts_with("  "))
        .collect();
    assert_eq!(heads.len(), 2, "{stderr}");
    for ((i, head), at) in heads.iter().zip(["momentum =", "ns_steps ="]) {
        assert!(head.starts_with("E-TYPE-025: "), "{stderr}");
        let (l, c) = position(&source, at);
        assert_eq!(lines[i + 1], format!("  {file}:{l}:{c}"), "{stderr}");
    }
}

/// Hypothesis 6 and 7. `compile --backend cpu` of a Muon file writes its artifact; `compile
/// --backend vulkan` and `infer --backend vulkan` are E-TARGET-011 at the `train` block, exit
/// 1 as for `adamw`, and write nothing.
#[test]
fn adv_muon_compile_and_infer_paths() {
    let case = scratch("compile");
    let cpu = case.join("cpu");
    let output = modelspec(&[
        "compile",
        LINEAR,
        "--backend",
        "cpu",
        "--output",
        path(&cpu),
    ]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    assert!(cpu.join("artifact.json").is_file());

    let vulkan = case.join("vulkan");
    let missing = case.join("none.msckpt");
    for args in [
        vec![
            "compile",
            LINEAR,
            "--backend",
            "vulkan",
            "--output",
            path(&vulkan),
        ],
        vec![
            "infer",
            LINEAR,
            "--backend",
            "vulkan",
            "--checkpoint",
            path(&missing),
        ],
    ] {
        let output = modelspec(&args);
        let stderr = text(&output.stderr);
        assert_eq!(output.status.code(), Some(1), "{args:?}: {stderr}");
        assert!(output.stdout.is_empty(), "{args:?}");
        assert_eq!(
            stderr.lines().take(2).collect::<Vec<_>>(),
            [
                "E-TARGET-011: optimizer `muon` has no Vulkan kernel yet",
                &format!("  {LINEAR}:10:1"),
            ],
            "{args:?}"
        );
    }
    assert!(!vulkan.exists(), "nothing is written for Vulkan");
}

/// Hypothesis 4. `--steps` at the warm-up of a Muon file is refused before the first step on one
/// line, exit 1; one step more trains.
#[test]
fn adv_muon_steps_override_at_the_warmup() {
    let output = modelspec(&["train", LINEAR, "--backend", "cpu", "--steps", "10"]);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(output.stdout.is_empty(), "{}", text(&output.stdout));
    assert_eq!(stderr.lines().count(), 1, "{stderr}");
    assert!(stderr.contains("warmup of 10 steps"), "{stderr}");
    let output = modelspec(&["train", LINEAR, "--backend", "cpu", "--steps", "11"]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
}

/// Finding ADV-MUON-1 (note). `train --backend vulkan` on a Muon file is E-TARGET-011 at the
/// `train` block and exits 1, as every E-TARGET-011 does: the AdamW case
/// (`adversary_adamw_cli.rs:80`), AGENTS.md's exit codes for a diagnostic and DESIGN.md § 8.
/// The "exit 2" of acceptance 6 and D67 (f) is a wording error (coordinator decision D69).
#[test]
fn adv_muon_1_vulkan_train_exits_1_as_every_e_target_011_does() {
    let output = modelspec(&["train", LINEAR, "--backend", "vulkan"]);
    let stderr = text(&output.stderr);
    assert_eq!(
        stderr.lines().next(),
        Some("E-TARGET-011: optimizer `muon` has no Vulkan kernel yet"),
        "{stderr}"
    );
    assert_eq!(output.status.code(), Some(1), "{stderr}");
}

fn packed(case: &Path) -> PathBuf {
    let dataset = case.join("dataset");
    let tokenizer = case.join("tokenizer.json");
    let packed = case.join("packed");
    for args in [
        vec!["data", "prepare", CORPUS, "--out", path(&dataset)],
        vec![
            "data",
            "tokenizer",
            "--dataset",
            path(&dataset),
            "--vocab-size",
            "300",
            "--out",
            path(&tokenizer),
        ],
        vec![
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
        ],
    ] {
        let output = modelspec(&args);
        assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    }
    packed
}

fn heldout_at_300(file: &str, packed: &Path) -> f32 {
    let output = modelspec(&[
        "train",
        file,
        "--backend",
        "cpu",
        "--data",
        path(packed),
        "--eval-every",
        "50",
    ]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    text(&output.stdout)
        .lines()
        .find_map(|l| l.strip_prefix("eval 300 loss "))
        .and_then(|rest| rest.split(' ').next())
        .and_then(|v| v.parse().ok())
        .expect("an eval line at step 300")
}

/// Finding ADV-MUON-2 (note). Acceptance 5 holds at the fixture's seed 7 by 0.0038 nats
/// (6.7210197 against 6.724814); across seeds 1 to 10 Muon is lower in 9 and higher at seed 2
/// (5.944862 against 5.9250245). The comparison is within seed noise, and the design claims
/// "not worse, not better" (DESIGN.md § 2.1, coordinator decision D69): at seeds 2 and 7 the
/// held-out loss at step 300 under Muon is within 0.05 nats of AdamW's.
#[test]
fn adv_muon_2_heldout_loss_under_muon_is_within_seed_noise_of_adamw() {
    const NOISE: f32 = 0.05;
    let case = scratch("seeds");
    let packed = packed(&case);
    let mut apart = Vec::new();
    for seed in [2u64, 7] {
        let seeded = format!("    seed = {seed};");
        let source = std::fs::read_to_string(repo_root().join(MLM_DENSE)).expect("the fixture");
        let muon = source.replace("    seed = 7;", &seeded);
        let adamw = muon.replace(MUON_LINE, ADAMW_LINE);
        assert_ne!(muon, adamw);
        let (m_file, a_file) = (
            case.join(format!("muon{seed}.mspec")),
            case.join(format!("adamw{seed}.mspec")),
        );
        std::fs::write(&m_file, muon).unwrap();
        std::fs::write(&a_file, adamw).unwrap();
        let (m, a) = (
            heldout_at_300(path(&m_file), &packed),
            heldout_at_300(path(&a_file), &packed),
        );
        eprintln!("seed {seed}: muon {m} adamw {a}");
        if (m - a).abs() > NOISE {
            apart.push((seed, m, a));
        }
    }
    assert!(
        apart.is_empty(),
        "seeds where Muon's held-out loss at 300 is more than {NOISE} nats from AdamW's: \
         {apart:?}"
    );
}
