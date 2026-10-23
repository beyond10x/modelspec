//! The recorded run file (`spec/domains/observatory.yaml`, `RecordedRunFile`): JSON Lines, one
//! [`RunEvent`] per line in `sequence` order from 0, each line the JSON of the event's
//! [`frame`](crate::frame) `data:` and ending in a newline, all of one run, the last
//! `RunFinished` or `RunFailed`.
//!
//! A [`Recorder`] writes the file as a run emits its events; [`read_recording`] reads it back
//! and refuses, with a [`RecordingError`] on one line, a file that is not such a recording.

use std::fmt;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use crate::event::{RunEvent, TrainingRunId};

/// The line of `event` in a recorded run file, with its newline: the `data:` of its frame.
pub fn line(event: &RunEvent) -> String {
    let mut line = serde_json::to_string(event).expect("a RunEvent serializes");
    line.push('\n');
    line
}

/// Writes the events of one run to a recorded run file, one line each, as they are emitted.
pub struct Recorder {
    file: File,
}

impl Recorder {
    /// Creates the file at `path`, or truncates the one there.
    pub fn create(path: &Path) -> io::Result<Recorder> {
        Ok(Recorder {
            file: File::create(path)?,
        })
    }

    /// Appends the line of `event` with one write, so the file holds whole lines between events.
    pub fn write(&mut self, event: &RunEvent) -> io::Result<()> {
        self.file.write_all(line(event).as_bytes())
    }
}

/// Why a file is not a recorded run file. Its `Display` is one line.
#[derive(Debug)]
pub enum RecordingError {
    /// The file cannot be read.
    Read(io::Error),
    /// The file has no event.
    Empty,
    /// The last line has no newline: the file was cut off.
    Unterminated { line: usize },
    /// A line is not the JSON of one `RunEvent`.
    NotAnEvent { line: usize, error: String },
    /// A line's `sequence` is not the one after the previous line's (from 0).
    Sequence {
        line: usize,
        expected: u64,
        found: u64,
    },
    /// A line names another run than the first line.
    MixedRuns {
        line: usize,
        first: TrainingRunId,
        found: TrainingRunId,
    },
    /// The first event is not `RunStarted`.
    NotStarted { kind: &'static str },
    /// An event follows the `RunFinished` or `RunFailed` that ends the stream.
    AfterEnd { line: usize },
    /// The last event is not `RunFinished` or `RunFailed`: the run's end is missing.
    Unended { kind: &'static str },
}

impl fmt::Display for RecordingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(error) => write!(f, "cannot read the recording: {error}"),
            Self::Empty => write!(f, "the recording has no event"),
            Self::Unterminated { line } => {
                write!(
                    f,
                    "line {line} has no newline at its end; the recording is cut off"
                )
            }
            Self::NotAnEvent { line, error } => write!(f, "line {line} is not a RunEvent: {error}"),
            Self::Sequence {
                line,
                expected,
                found,
            } => write!(f, "line {line} has sequence {found}, expected {expected}"),
            Self::MixedRuns { line, first, found } => write!(
                f,
                "line {line} is an event of training run {}, line 1 of training run {}",
                found.0, first.0
            ),
            Self::NotStarted { kind } => {
                write!(f, "the first event is {kind}, not RunStarted")
            }
            Self::AfterEnd { line } => {
                write!(f, "line {line} follows the event that ends the stream")
            }
            Self::Unended { kind } => write!(
                f,
                "the last event is {kind}, not RunFinished or RunFailed; the recording is cut off"
            ),
        }
    }
}

impl std::error::Error for RecordingError {}

/// The events of the recorded run file at `path`, refused unless it is a whole recording of one
/// run ([`parse_recording`]).
pub fn read_recording(path: &Path) -> Result<Vec<RunEvent>, RecordingError> {
    let text = std::fs::read_to_string(path).map_err(RecordingError::Read)?;
    parse_recording(&text)
}

/// The events of a recorded run file's content: whole lines of `RunEvent`s, sequences 0, 1, 2,
/// ..., one `training_run_id`, `RunStarted` first and `RunFinished` or `RunFailed` last.
pub fn parse_recording(text: &str) -> Result<Vec<RunEvent>, RecordingError> {
    if text.is_empty() {
        return Err(RecordingError::Empty);
    }
    let lines: Vec<&str> = text.split_terminator('\n').collect();
    if !text.ends_with('\n') {
        return Err(RecordingError::Unterminated { line: lines.len() });
    }
    let mut events: Vec<RunEvent> = Vec::with_capacity(lines.len());
    for (index, text) in lines.into_iter().enumerate() {
        let line = index + 1;
        let event: RunEvent =
            serde_json::from_str(text).map_err(|error| RecordingError::NotAnEvent {
                line,
                error: error.to_string(),
            })?;
        let expected = index as u64;
        if event.sequence != expected {
            return Err(RecordingError::Sequence {
                line,
                expected,
                found: event.sequence,
            });
        }
        if let Some(first) = events.first() {
            if event.training_run_id != first.training_run_id {
                return Err(RecordingError::MixedRuns {
                    line,
                    first: first.training_run_id,
                    found: event.training_run_id,
                });
            }
            if events.last().is_some_and(|last| last.body.is_last()) {
                return Err(RecordingError::AfterEnd { line });
            }
        } else if !matches!(event.body, crate::RunEventBody::RunStarted(_)) {
            return Err(RecordingError::NotStarted {
                kind: event.body.kind(),
            });
        }
        events.push(event);
    }
    match events.last() {
        None => Err(RecordingError::Empty),
        Some(last) if !last.body.is_last() => Err(RecordingError::Unended {
            kind: last.body.kind(),
        }),
        Some(_) => Ok(events),
    }
}
