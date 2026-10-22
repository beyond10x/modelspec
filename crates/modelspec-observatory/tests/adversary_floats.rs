//! Adversary cases, pass 2: the floats of a `RunEvent` after the fix that sends a non-finite one
//! as `null` (`spec/domains/observatory.yaml`, `StepCompleted.loss`, `MetricSample.value`,
//! `ParameterStatistics.norm`, `Histogram.bin_edges`; wave observatory-1 coordinator,
//! 2026-10-21), driven from the committed `RunEvent` JSON Schema and from the bits of the value.

use std::path::Path;

use modelspec_observatory::{
    EventSequence, Histogram, MetricSample, ParameterStatistics, RunEvent, RunEventBody,
    StatisticsSubject, StepCompleted, TrainingRunId, frame,
};
use serde_json::{Value, json};

fn validator() -> jsonschema::Validator {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("schema/entities/modelspec.observatory.RunEvent.schema.json");
    let schema: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    jsonschema::validator_for(&schema).unwrap()
}

/// A fixed-seed xorshift64, so the property runs the same values on every machine.
struct Bits(u64);

impl Bits {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn step(loss: f32, value: f64) -> RunEventBody {
    RunEventBody::StepCompleted(StepCompleted {
        step: 1,
        loss: f64::from(loss),
        metrics: vec![MetricSample {
            name: "m".into(),
            value,
        }],
        parameters: vec![ParameterStatistics {
            path: "LinearReg.w".into(),
            subject: StatisticsSubject::Value,
            norm: value.abs(),
            histogram: Histogram {
                bin_edges: vec![value, value],
                counts: vec![0],
            },
        }],
    })
}

/// The `data:` line of the frame of `body`, parsed, and the event it reads back as.
fn framed(body: RunEventBody) -> (Value, RunEvent) {
    let event = EventSequence::new(TrainingRunId::new()).event(body);
    let text = frame(&event);
    let data = text.lines().find_map(|l| l.strip_prefix("data: ")).unwrap();
    (
        serde_json::from_str(data).unwrap(),
        serde_json::from_str(data).expect("the types read what they write"),
    )
}

/// Property: every finite float an event carries is a JSON number the schema accepts and reads
/// back with the same bits (the sign of -0.0, subnormals, the largest values included); every
/// non-finite one is `null`. `StepCompleted.loss` is the `f32` loss widened exactly, so `loss as
/// f32` must give back the very `f32` the CLI prints (event.rs), and the other floats are
/// `Binary64`, whose value is its bits.
#[test]
fn every_float_round_trips_bit_exact_or_is_null() {
    let v = validator();
    let mut f32s = vec![
        0.0,
        -0.0,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::from_bits(0x007f_ffff),
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        f32::EPSILON,
        1.0e-40,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7f80_0001),
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let mut f64s = vec![
        0.0,
        -0.0,
        f64::from_bits(1),
        f64::from_bits(0x000f_ffff_ffff_ffff),
        f64::MIN_POSITIVE,
        f64::MAX,
        f64::MIN,
        f64::EPSILON,
        0.1,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let mut bits = Bits(0x9e37_79b9_7f4a_7c15);
    for _ in 0..20_000 {
        let word = bits.next();
        f32s.push(f32::from_bits(word as u32));
        f64s.push(f64::from_bits(bits.next()));
    }
    let mut wrong = Vec::new();
    for (k, &loss) in f32s.iter().enumerate() {
        let value = f64s[k % f64s.len()];
        let (json, read) = framed(step(loss, value));
        if !v.is_valid(&json) {
            wrong.push(format!(
                "schema refuses loss {loss:e} value {value:e}: {json}"
            ));
            continue;
        }
        let body = &json["body"]["value"];
        let RunEventBody::StepCompleted(back) = read.body else {
            panic!("{json}")
        };
        if loss.is_finite() {
            if !body["loss"].is_number() || (back.loss as f32).to_bits() != loss.to_bits() {
                wrong.push(format!(
                    "loss {loss:e} ({:#010x}) framed {} reads back {:e} as f32 {:#010x}",
                    loss.to_bits(),
                    body["loss"],
                    back.loss,
                    (back.loss as f32).to_bits()
                ));
            }
        } else if !body["loss"].is_null() || !back.loss.is_nan() {
            wrong.push(format!("loss {loss} framed {}", body["loss"]));
        }
        let floats = [
            ("metric value", value, back.metrics[0].value),
            ("norm", value.abs(), back.parameters[0].norm),
            ("bin edge", value, back.parameters[0].histogram.bin_edges[0]),
        ];
        for (name, sent, got) in floats {
            let same = if sent.is_finite() {
                sent.to_bits() == got.to_bits()
            } else {
                got.is_nan()
            };
            if !same {
                wrong.push(format!(
                    "{name} {sent:e} ({:#018x}) reads back {got:e} ({:#018x})",
                    sent.to_bits(),
                    got.to_bits()
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} events lose a float, first: {:#?}",
        wrong.len(),
        f32s.len(),
        &wrong[..wrong.len().min(4)]
    );
}

fn valid_step() -> Value {
    json!({
        "run_event_id": "6f1c2a52-6a3e-4c7e-9a43-1d8d7b0f2e11",
        "training_run_id": "0b9e3d0c-58a4-4b0e-8d1e-3f5b1f0a9c22",
        "sequence": 1,
        "body": {"kind": "StepCompleted",
                 "value": {"step": 0, "loss": 0.5,
                           "metrics": [{"name": "m", "value": 0.5}],
                           "parameters": [{"path": "LinearReg.w", "subject": "Value",
                                           "norm": 0.5,
                                           "histogram": {"bin_edges": [0.0, 1.0],
                                                         "counts": [1]}}]}},
        "state": "Emitted"
    })
}

type Mutation = Box<dyn Fn(&mut Value)>;

/// `presence: null_when_absent` sends the key with `null`; it does not let the key go. Only the
/// floats became `Optional`: no integer, string or list field of an event admits `null`. Both the
/// schema and the Rust types refuse each of these.
#[test]
fn null_is_admitted_only_where_the_spec_puts_it() {
    let v = validator();
    let base = valid_step();
    assert!(v.is_valid(&base), "{base}");
    assert!(serde_json::from_value::<RunEvent>(base.clone()).is_ok());
    let mut nulls = base.clone();
    let step = &mut nulls["body"]["value"];
    step["loss"] = Value::Null;
    step["metrics"][0]["value"] = Value::Null;
    step["parameters"][0]["norm"] = Value::Null;
    step["parameters"][0]["histogram"]["bin_edges"] = json!([null, null]);
    assert!(v.is_valid(&nulls), "{nulls}");
    assert!(serde_json::from_value::<RunEvent>(nulls).is_ok());
    let remove = |path: &'static [&'static str], key: &'static str| -> Mutation {
        Box::new(move |e: &mut Value| {
            let mut node = e;
            for p in path {
                node = match p.parse::<usize>() {
                    Ok(i) => &mut node[i],
                    Err(_) => &mut node[*p],
                };
            }
            node.as_object_mut().unwrap().remove(key);
        })
    };
    let set = |path: &'static [&'static str], to: Value| -> Mutation {
        Box::new(move |e: &mut Value| {
            let mut node = e;
            for p in path {
                node = match p.parse::<usize>() {
                    Ok(i) => &mut node[i],
                    Err(_) => &mut node[*p],
                };
            }
            *node = to.clone();
        })
    };
    let mutations: Vec<(&str, Mutation)> = vec![
        ("no loss", remove(&["body", "value"], "loss")),
        (
            "no metric value",
            remove(&["body", "value", "metrics", "0"], "value"),
        ),
        (
            "no norm",
            remove(&["body", "value", "parameters", "0"], "norm"),
        ),
        (
            "bin edge text",
            set(
                &[
                    "body",
                    "value",
                    "parameters",
                    "0",
                    "histogram",
                    "bin_edges",
                    "0",
                ],
                json!("0"),
            ),
        ),
        ("step null", set(&["body", "value", "step"], Value::Null)),
        ("sequence null", set(&["sequence"], Value::Null)),
        (
            "count null",
            set(
                &[
                    "body",
                    "value",
                    "parameters",
                    "0",
                    "histogram",
                    "counts",
                    "0",
                ],
                Value::Null,
            ),
        ),
        (
            "bin_edges null",
            set(
                &["body", "value", "parameters", "0", "histogram", "bin_edges"],
                Value::Null,
            ),
        ),
        (
            "metric name null",
            set(&["body", "value", "metrics", "0", "name"], Value::Null),
        ),
        (
            "metrics null",
            set(&["body", "value", "metrics"], Value::Null),
        ),
    ];
    let mut accepted = Vec::new();
    for (name, mutate) in mutations {
        let mut event = valid_step();
        mutate(&mut event);
        if v.is_valid(&event) {
            accepted.push(format!("the schema accepts {name}: {event}"));
        }
        if serde_json::from_value::<RunEvent>(event.clone()).is_ok() {
            accepted.push(format!("RunEvent reads {name}: {event}"));
        }
    }
    assert!(accepted.is_empty(), "{accepted:#?}");
}
