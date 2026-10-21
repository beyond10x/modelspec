//! `modelspec.observatory.RunEvent` and what it contains, as serde types whose JSON is the
//! ESS-generated schema's: a union is `{"kind": <variant>, "value": <payload>}`, `value` left out
//! for a variant without one; an `Optional` field that is absent is left out.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// `modelspec.training.TrainingRunId`: minted per `modelspec train` invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TrainingRunId(pub Uuid);

impl TrainingRunId {
    /// A fresh random (version 4) id.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for TrainingRunId {
    fn default() -> Self {
        Self::new()
    }
}

/// `modelspec.observatory.RunEventId`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RunEventId(pub Uuid);

/// `modelspec.observatory.RunEvent`: one event of a run's stream at its position `sequence`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunEvent {
    pub run_event_id: RunEventId,
    pub training_run_id: TrainingRunId,
    /// The position in the run's stream, from 0.
    pub sequence: u64,
    pub body: RunEventBody,
    /// The entity's lifecycle state; an event has the one state `Emitted`.
    pub state: RunEventState,
}

/// `modelspec.observatory.RunEvent.State`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunEventState {
    Emitted,
}

/// `modelspec.observatory.RunEventBody`: [`RunEventBody::RunStarted`] first, one
/// [`RunEventBody::StepCompleted`] per recorded loss, then exactly one of
/// [`RunEventBody::RunFinished`] and [`RunEventBody::RunFailed`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", deny_unknown_fields)]
pub enum RunEventBody {
    RunStarted(RunStarted),
    StepCompleted(StepCompleted),
    RunFinished,
    RunFailed(RunFailed),
}

impl RunEventBody {
    /// The `kind` tag, which is also the SSE `event:` name (`modelspec.observatory.RunEventKind`).
    pub fn kind(&self) -> &'static str {
        match self {
            Self::RunStarted(_) => "RunStarted",
            Self::StepCompleted(_) => "StepCompleted",
            Self::RunFinished => "RunFinished",
            Self::RunFailed(_) => "RunFailed",
        }
    }

    /// Whether the event ends its stream.
    pub fn is_last(&self) -> bool {
        matches!(self, Self::RunFinished | Self::RunFailed(_))
    }
}

/// `modelspec.observatory.RunStarted`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunStarted {
    /// `TrainingRun.steps`, the steps of this invocation.
    pub steps: u64,
    /// `TrainingRun.start_step`, the step of the first `StepCompleted`.
    pub start_step: u64,
}

/// `modelspec.observatory.StepCompleted`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepCompleted {
    pub step: u64,
    /// The `f32` loss of the step, widened exactly; `loss as f32` is the value the CLI prints.
    pub loss: f64,
    pub metrics: Vec<MetricSample>,
    pub parameters: Vec<ParameterStatistics>,
}

/// `modelspec.observatory.MetricSample`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricSample {
    pub name: String,
    pub value: f64,
}

/// `modelspec.observatory.ParameterStatistics`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParameterStatistics {
    /// A `modelspec.checkpoint.ParameterPath`.
    pub path: String,
    pub subject: StatisticsSubject,
    pub norm: f64,
    pub histogram: Histogram,
}

/// `modelspec.observatory.StatisticsSubject`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatisticsSubject {
    Value,
    Gradient,
}

/// `modelspec.observatory.Histogram`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Histogram {
    pub bin_edges: Vec<f64>,
    pub counts: Vec<u64>,
}

/// `modelspec.observatory.RunFailed`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunFailed {
    /// The diagnostic code that refused the run (DESIGN.md § 8.1); `None` for a runtime error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// The failure on one line.
    pub message: String,
}

/// The events of one run, numbered from 0 under one [`TrainingRunId`].
#[derive(Debug)]
pub struct EventSequence {
    training_run_id: TrainingRunId,
    next: u64,
    ended: bool,
}

impl EventSequence {
    pub fn new(training_run_id: TrainingRunId) -> Self {
        Self {
            training_run_id,
            next: 0,
            ended: false,
        }
    }

    pub fn training_run_id(&self) -> TrainingRunId {
        self.training_run_id
    }

    /// The next event of the run, carrying `body`.
    ///
    /// # Panics
    ///
    /// When an event that ends the stream ([`RunEventBody::is_last`]) was already made: nothing
    /// follows `RunFinished` or `RunFailed`.
    pub fn event(&mut self, body: RunEventBody) -> RunEvent {
        assert!(
            !self.ended,
            "a {} event after the stream ended",
            body.kind()
        );
        self.ended = body.is_last();
        let event = RunEvent {
            run_event_id: RunEventId(Uuid::new_v4()),
            training_run_id: self.training_run_id,
            sequence: self.next,
            body,
            state: RunEventState::Emitted,
        };
        self.next += 1;
        event
    }
}
