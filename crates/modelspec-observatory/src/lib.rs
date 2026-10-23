//! The training observatory (epic:training-observatory): the typed events of a training run and
//! the live stream that serves them.
//!
//! The types are those of `spec/domains/observatory.yaml`; their JSON is the one the JSON Schema
//! under `schema/` describes, which `ess generate --kind schema` writes from that specification
//! (`task schema-generate`, checked by `task schema-check`). A run's stream is its [`RunEvent`]s in
//! `sequence` order, numbered by an [`EventSequence`]; a [`Server`] serves it as
//! `text/event-stream` at `GET /events` on a loopback address ([`parse_serve_address`]), one
//! [`frame`] per event. Nothing here knows about a backend: a CPU and a Vulkan run send the same
//! events.
//!
//! A [`Recorder`] writes a run's events to a recorded run file, one [`line`] each, the `data:` of
//! its frame; [`read_recording`] reads one back, refusing a damaged file, and
//! [`Server::replay`] serves it as the live stream was served.

mod event;
mod record;
mod server;

pub use event::{
    EventSequence, Histogram, MetricSample, ParameterStatistics, RunEvent, RunEventBody,
    RunEventId, RunEventState, RunFailed, RunStarted, StatisticsSubject, StepCompleted,
    TrainingRunId,
};
pub use record::{Recorder, RecordingError, line, parse_recording, read_recording};
pub use server::{EVENTS_PATH, Server, frame, parse_serve_address};
