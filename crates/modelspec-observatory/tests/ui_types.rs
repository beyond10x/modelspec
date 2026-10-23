//! The TypeScript types of the observatory UI, generated from the Rust types of the stream
//! (story:observatory-ui-live-loss-chart).
//!
//! Scenario `observatory.ui.types-in-sync` is [`observatory_ui_types_in_sync`]; `task ci` also runs
//! `modelspec-ui-types --check` (`task ui-types-check`), the same comparison from the command line.

use std::path::{Path, PathBuf};
use std::process::Command;

use modelspec_observatory::{UI_TYPES_PATH, typescript};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh empty directory `name` under the test temp dir.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("ui_types")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    dir
}

/// `observatory.ui.types-in-sync`: the committed `ui/src/generated/observatory.ts` is what the
/// generator writes from the Rust types now, so a field changed in Rust without `task ui-types`
/// fails here.
#[test]
fn observatory_ui_types_in_sync() {
    let path = repo_root().join(UI_TYPES_PATH);
    let committed = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} is committed: {e}", path.display()));
    assert_eq!(
        committed,
        typescript(),
        "{UI_TYPES_PATH} is not what the Rust types generate; run `task ui-types`"
    );
}

/// Every type a `RunEvent` contains is declared, so the UI reads no type the generator did not
/// write.
#[test]
fn every_type_the_stream_sends_is_declared() {
    let ts = typescript();
    for name in [
        "RunEvent",
        "RunEventId",
        "TrainingRunId",
        "RunEventBody",
        "RunEventState",
        "RunStarted",
        "StepCompleted",
        "MetricSample",
        "ParameterStatistics",
        "StatisticsSubject",
        "Histogram",
        "RunFailed",
    ] {
        assert_eq!(
            ts.matches(&format!("\nexport type {name} = ")).count(),
            1,
            "one declaration of {name}:\n{ts}"
        );
    }
}

/// The JSON of the stream holds integers as JSON numbers, which `JSON.parse` reads as `number`,
/// and every float as a number or `null` for NaN and ±inf (`non_finite_null`), so the types say
/// `number` for integers and `number | null` for floats, and an absent `RunFailed.code` is an
/// optional field.
#[test]
fn numbers_are_numbers_and_floats_may_be_null() {
    let ts = typescript();
    assert!(!ts.contains("bigint"), "{ts}");
    for field in [
        "sequence: number,",
        "steps: number,",
        "start_step: number,",
        "step: number,",
        "loss: number | null,",
        "value: number | null,",
        "norm: number | null,",
        "bin_edges: Array<number | null>,",
        "counts: Array<number>,",
        "code?: string,",
        "message: string,",
    ] {
        assert!(ts.contains(field), "{field} in:\n{ts}");
    }
    assert!(
        ts.contains("{ \"kind\": \"RunFinished\" }"),
        "RunFinished has no value:\n{ts}"
    );
}

/// `modelspec-ui-types` writes the file under `--root`, and `--check` exits 0 while the file is
/// current and 1, naming the file, once it differs.
#[test]
fn the_generator_writes_and_checks() {
    let root = scratch("root");
    let generator = || {
        let mut command = Command::new(env!("CARGO_BIN_EXE_modelspec-ui-types"));
        command.arg("--root").arg(&root);
        command
    };
    let missing = generator().arg("--check").output().unwrap();
    assert_eq!(missing.status.code(), Some(1), "{missing:?}");

    let written = generator().output().unwrap();
    assert_eq!(written.status.code(), Some(0), "{written:?}");
    let path = root.join(UI_TYPES_PATH);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), typescript());

    let current = generator().arg("--check").output().unwrap();
    assert_eq!(current.status.code(), Some(0), "{current:?}");

    let stale = typescript().replace("loss: number | null,", "loss: number,");
    std::fs::write(&path, stale).unwrap();
    let drifted = generator().arg("--check").output().unwrap();
    assert_eq!(drifted.status.code(), Some(1), "{drifted:?}");
    let stderr = String::from_utf8(drifted.stderr).unwrap();
    assert!(stderr.contains(UI_TYPES_PATH), "{stderr}");
    assert!(stderr.contains("task ui-types"), "{stderr}");
}
