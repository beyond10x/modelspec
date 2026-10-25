//! The recorded run file (`spec/domains/observatory.yaml`, `RecordedRunFile`), driven without a
//! training run: a [`Recorder`] round trip, and every refusal of [`parse_recording`].

use std::path::Path;

use modelspec_observatory::{
    EventSequence, Recorder, RecordingError, RunEvent, RunEventBody, RunFailed, RunStarted,
    StepCompleted, TrainingRunId, frame, line, parse_recording, read_recording,
};

fn started() -> RunEventBody {
    RunEventBody::RunStarted(RunStarted {
        steps: 1,
        start_step: 0,
        data_binding_id: None,
    })
}

fn step(step: u64) -> RunEventBody {
    RunEventBody::StepCompleted(StepCompleted {
        step,
        loss: 0.5,
        metrics: Vec::new(),
        parameters: Vec::new(),
    })
}

fn run(bodies: Vec<RunEventBody>) -> Vec<RunEvent> {
    let mut sequence = EventSequence::new(TrainingRunId::new());
    bodies.into_iter().map(|b| sequence.event(b)).collect()
}

fn text(events: &[RunEvent]) -> String {
    events.iter().map(line).collect()
}

/// Whether a refusal is the one a damaged recording should get.
type Expected = fn(&RecordingError) -> bool;

#[test]
fn a_recorder_writes_what_read_recording_returns() {
    let events = run(vec![started(), step(0), step(1), RunEventBody::RunFinished]);
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("observatory_record_unit");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("run.jsonl");
    let mut recorder = Recorder::create(&path).unwrap();
    for event in &events {
        recorder.write(event).unwrap();
    }
    drop(recorder);
    assert_eq!(read_recording(&path).unwrap(), events);
    let written = std::fs::read_to_string(&path).unwrap();
    for (event, line) in events.iter().zip(written.lines()) {
        assert!(frame(event).contains(&format!("\ndata: {line}\n\n")));
    }
}

#[test]
fn a_failed_run_is_a_whole_recording() {
    let events = run(vec![
        started(),
        RunEventBody::RunFailed(RunFailed {
            code: Some("E-DATA-003".into()),
            message: "m".into(),
        }),
    ]);
    assert_eq!(parse_recording(&text(&events)).unwrap(), events);
}

#[test]
fn every_damaged_recording_is_refused_on_one_line() {
    let whole = run(vec![started(), step(0), RunEventBody::RunFinished]);
    let not_started = run(vec![step(0), RunEventBody::RunFinished]);
    let mut after_end = run(vec![started(), RunEventBody::RunFinished]);
    let mut extra = after_end[1].clone();
    extra.sequence = 2;
    extra.body = step(0);
    after_end.push(extra);
    let started_again = run(vec![
        started(),
        step(0),
        started(),
        RunEventBody::RunFinished,
    ]);
    let mut repeated = whole.clone();
    repeated[2].sequence = 1;
    let mut mixed = whole.clone();
    mixed[1].training_run_id = TrainingRunId::new();
    let full = text(&whole);
    let cases: Vec<(&str, String, Expected)> = vec![
        ("empty", String::new(), |e| {
            matches!(e, RecordingError::Empty)
        }),
        ("cut", full[..full.len() - 3].to_owned(), |e| {
            matches!(e, RecordingError::Unterminated { line: 3 })
        }),
        ("not json", "{}\n".to_owned(), |e| {
            matches!(e, RecordingError::NotAnEvent { line: 1, .. })
        }),
        ("not started", text(&not_started), |e| {
            matches!(
                e,
                RecordingError::NotStarted {
                    kind: "StepCompleted"
                }
            )
        }),
        ("after end", text(&after_end), |e| {
            matches!(e, RecordingError::AfterEnd { line: 3 })
        }),
        ("started again", text(&started_again), |e| {
            matches!(e, RecordingError::StartedAgain { line: 3 })
        }),
        ("repeated", text(&repeated), |e| {
            matches!(
                e,
                RecordingError::Sequence {
                    line: 3,
                    expected: 2,
                    found: 1
                }
            )
        }),
        ("mixed", text(&mixed), |e| {
            matches!(e, RecordingError::MixedRuns { line: 2, .. })
        }),
        ("unended", text(&whole[..2]), |e| {
            matches!(
                e,
                RecordingError::Unended {
                    kind: "StepCompleted"
                }
            )
        }),
    ];
    for (name, content, expected) in cases {
        let error = parse_recording(&content).expect_err(name);
        assert!(expected(&error), "{name}: {error:?}");
        assert_eq!(error.to_string().lines().count(), 1, "{name}: {error}");
    }
}
