//! Adversary cases: the committed `RunEvent` JSON Schema read as the contract it claims to be
//! (`spec/domains/observatory.yaml`, `StreamFrame.data`), against what `frame` serializes.

use std::path::Path;

use modelspec_observatory::{EventSequence, RunEventBody, StepCompleted, TrainingRunId, frame};
use serde_json::{Value, json};

fn validator() -> jsonschema::Validator {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("schema/entities/modelspec.observatory.RunEvent.schema.json");
    let schema: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    jsonschema::validator_for(&schema).unwrap()
}

fn valid_step() -> Value {
    json!({
        "run_event_id": "6f1c2a52-6a3e-4c7e-9a43-1d8d7b0f2e11",
        "training_run_id": "0b9e3d0c-58a4-4b0e-8d1e-3f5b1f0a9c22",
        "sequence": 1,
        "body": {"kind": "StepCompleted",
                 "value": {"step": 0, "loss": 0.5, "metrics": [], "parameters": []}},
        "state": "Emitted"
    })
}

type Mutation = Box<dyn Fn(&mut Value)>;

/// The validator the unit's tests use is live: it refuses what the specification forbids, so a
/// green conformance test is evidence and not a validator that accepts everything.
#[test]
fn the_schema_refuses_what_the_spec_forbids() {
    let v = validator();
    assert!(v.is_valid(&valid_step()));
    let mutations: Vec<(&str, Mutation)> = vec![
        (
            "loss null",
            Box::new(|e| e["body"]["value"]["loss"] = Value::Null),
        ),
        (
            "loss text",
            Box::new(|e| e["body"]["value"]["loss"] = json!("0.5")),
        ),
        (
            "step -1",
            Box::new(|e| e["body"]["value"]["step"] = json!(-1)),
        ),
        (
            "no state",
            Box::new(|e| drop(e.as_object_mut().unwrap().remove("state"))),
        ),
        (
            "state Recorded",
            Box::new(|e| e["state"] = json!("Recorded")),
        ),
        (
            "unknown kind",
            Box::new(|e| e["body"]["kind"] = json!("RunPaused")),
        ),
        ("extra field", Box::new(|e| e["extra"] = json!(1))),
        (
            "run id not uuid",
            Box::new(|e| e["training_run_id"] = json!("run-1")),
        ),
        ("sequence text", Box::new(|e| e["sequence"] = json!("1"))),
        (
            "finished with value",
            Box::new(|e| e["body"] = json!({"kind": "RunFinished", "value": {}})),
        ),
        (
            "failed without message",
            Box::new(|e| e["body"] = json!({"kind": "RunFailed", "value": {"code": "E"}})),
        ),
        (
            "started negative steps",
            Box::new(|e| {
                e["body"] = json!({"kind": "RunStarted", "value": {"steps": -1, "start_step": 0}})
            }),
        ),
    ];
    for (name, mutate) in mutations {
        let mut event = valid_step();
        mutate(&mut event);
        assert!(!v.is_valid(&event), "the schema accepts {name}: {event}");
    }
}

/// `StreamFrame.data` is "valid against the JSON Schema `ess generate --kind schema` writes";
/// `StepCompleted.loss` is `Binary64`, a JSON number. An `f32` loss that is not finite is what a
/// diverging SGD run records.
#[test]
fn a_non_finite_loss_frame_is_schema_valid() {
    let v = validator();
    for loss in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut sequence = EventSequence::new(TrainingRunId::new());
        let event = sequence.event(RunEventBody::StepCompleted(StepCompleted {
            step: 0,
            loss: f64::from(loss),
            metrics: Vec::new(),
            parameters: Vec::new(),
        }));
        let text = frame(&event);
        let data = text.lines().find_map(|l| l.strip_prefix("data: ")).unwrap();
        let json: Value = serde_json::from_str(data).unwrap();
        assert!(v.is_valid(&json), "loss {loss}: {data}");
    }
}
