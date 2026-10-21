//! The RunEvent types against the JSON Schema ESS generates for them, and the SSE server of
//! `spec/domains/observatory.yaml` (`ServeAddress`, `StreamFrame`), driven without a training run.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::time::Duration;

use modelspec_observatory::{
    EventSequence, RunEventBody, RunFailed, RunStarted, Server, StepCompleted, TrainingRunId,
    frame, parse_serve_address,
};
use serde_json::Value;

const WAIT: Duration = Duration::from_secs(30);

fn schema() -> jsonschema::Validator {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("schema/entities/modelspec.observatory.RunEvent.schema.json");
    let text = std::fs::read_to_string(path).expect("the schema is committed");
    let schema: Value = serde_json::from_str(&text).expect("the schema is JSON");
    jsonschema::validator_for(&schema).expect("the schema compiles")
}

fn step(step: u64, loss: f32) -> RunEventBody {
    RunEventBody::StepCompleted(StepCompleted {
        step,
        loss: f64::from(loss),
        metrics: Vec::new(),
        parameters: Vec::new(),
    })
}

fn bodies() -> Vec<RunEventBody> {
    vec![
        RunEventBody::RunStarted(RunStarted {
            steps: 1,
            start_step: 0,
        }),
        step(0, 2.5),
        step(1, 0.125),
        RunEventBody::RunFinished,
    ]
}

#[test]
fn every_body_kind_validates_against_the_generated_schema() {
    let validator = schema();
    let mut all = bodies();
    all.push(RunEventBody::RunFailed(RunFailed {
        code: Some("E-DATA-003".into()),
        message: "the input `x` is not read by the loss".into(),
    }));
    all.push(RunEventBody::RunFailed(RunFailed {
        code: None,
        message: "final loss: out of memory".into(),
    }));
    for body in all {
        let event = EventSequence::new(TrainingRunId::new()).event(body);
        let json = serde_json::to_value(&event).unwrap();
        let errors: Vec<String> = validator
            .iter_errors(&json)
            .map(|e| e.to_string())
            .collect();
        assert!(errors.is_empty(), "{json}: {errors:?}");
    }
}

#[test]
fn a_failure_without_a_code_leaves_the_field_out() {
    let mut sequence = EventSequence::new(TrainingRunId::new());
    let event = sequence.event(RunEventBody::RunFailed(RunFailed {
        code: None,
        message: "m".into(),
    }));
    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(
        json["body"],
        serde_json::json!({"kind": "RunFailed", "value": {"message": "m"}})
    );
}

#[test]
fn a_finished_run_body_is_its_tag_alone() {
    let mut sequence = EventSequence::new(TrainingRunId::new());
    let json = serde_json::to_value(sequence.event(RunEventBody::RunFinished)).unwrap();
    assert_eq!(json["body"], serde_json::json!({"kind": "RunFinished"}));
    assert_eq!(json["state"], "Emitted");
}

#[test]
fn a_sequence_numbers_its_events_from_0_under_one_run() {
    let run = TrainingRunId::new();
    let mut sequence = EventSequence::new(run);
    let events: Vec<_> = bodies().into_iter().map(|b| sequence.event(b)).collect();
    for (k, event) in events.iter().enumerate() {
        assert_eq!(event.sequence, k as u64);
        assert_eq!(event.training_run_id, run);
    }
    assert_ne!(events[0].run_event_id, events[1].run_event_id);
}

#[test]
fn an_event_round_trips_through_json() {
    let mut sequence = EventSequence::new(TrainingRunId::new());
    for body in bodies() {
        let event = sequence.event(body);
        let text = serde_json::to_string(&event).unwrap();
        assert_eq!(
            serde_json::from_str::<modelspec_observatory::RunEvent>(&text).unwrap(),
            event
        );
    }
}

#[test]
fn a_frame_is_event_id_and_data_lines_and_a_blank_line() {
    let mut sequence = EventSequence::new(TrainingRunId::new());
    sequence.event(RunEventBody::RunStarted(RunStarted {
        steps: 0,
        start_step: 0,
    }));
    let event = sequence.event(step(0, 1.0));
    let text = frame(&event);
    let json = serde_json::to_string(&event).unwrap();
    assert_eq!(
        text,
        format!("event: StepCompleted\nid: 1\ndata: {json}\n\n")
    );
    assert!(!json.contains('\n'));
}

#[test]
fn a_serve_address_is_a_loopback_socket_address() {
    let accepted = [
        ("127.0.0.1:0", "127.0.0.1:0"),
        ("127.1.2.3:8080", "127.1.2.3:8080"),
        ("[::1]:9000", "[::1]:9000"),
        ("localhost:8080", "127.0.0.1:8080"),
    ];
    for (text, address) in accepted {
        assert_eq!(
            parse_serve_address(text),
            Ok(address.parse::<SocketAddr>().unwrap()),
            "{text}"
        );
    }
    for text in [
        "0.0.0.0:80",
        "192.0.2.1:80",
        "[::]:0",
        "example.org:80",
        "127.0.0.1",
        "",
        "localhost",
    ] {
        assert!(parse_serve_address(text).is_err(), "{text}");
    }
}

#[test]
fn a_server_refuses_a_non_loopback_address() {
    assert!(Server::bind("0.0.0.0:0".parse().unwrap(), TrainingRunId::new()).is_err());
}

/// The status line, headers and body of `GET <path>` with `extra` header lines, read to the end.
fn get(address: SocketAddr, path: &str, extra: &str) -> (String, String, String) {
    let mut stream = TcpStream::connect(address).expect("the server accepts a connection");
    stream.set_read_timeout(Some(WAIT)).unwrap();
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {address}\r\n{extra}\r\n"
    )
    .unwrap();
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .expect("the server ends the response");
    let (head, body) = response.split_once("\r\n\r\n").expect("a response head");
    let (status, headers) = head.split_once("\r\n").unwrap_or((head, ""));
    (status.to_owned(), headers.to_owned(), body.to_owned())
}

fn ended_server() -> (Server, String) {
    let server = Server::bind("127.0.0.1:0".parse().unwrap(), TrainingRunId::new()).unwrap();
    let frames: String = bodies()
        .into_iter()
        .map(|b| frame(&server.emit(b)))
        .collect();
    (server, frames)
}

#[test]
fn a_client_after_the_run_reads_every_frame_and_the_end() {
    let (server, frames) = ended_server();
    let (status, headers, body) = get(server.local_addr(), "/events", "");
    assert_eq!(status, "HTTP/1.1 200 OK");
    assert!(
        headers
            .lines()
            .any(|h| h == "Content-Type: text/event-stream"),
        "{headers}"
    );
    assert_eq!(body, frames);
}

#[test]
fn a_query_string_does_not_change_the_path() {
    let (server, frames) = ended_server();
    assert_eq!(get(server.local_addr(), "/events?from=ui", "").2, frames);
}

#[test]
fn last_event_id_resumes_after_that_sequence() {
    let (server, frames) = ended_server();
    let after_1: String = frames.split_inclusive("\n\n").skip(2).collect();
    assert_eq!(
        get(server.local_addr(), "/events", "Last-Event-ID: 1\r\n").2,
        after_1
    );
    assert_eq!(
        get(server.local_addr(), "/events", "last-event-id: 3\r\n").2,
        ""
    );
    assert_eq!(
        get(server.local_addr(), "/events", "Last-Event-ID: x\r\n").2,
        frames
    );
}

#[test]
fn a_client_connected_during_the_run_reads_events_as_they_are_emitted() {
    let server = Server::bind("127.0.0.1:0".parse().unwrap(), TrainingRunId::new()).unwrap();
    let address = server.local_addr();
    let mut bodies = bodies().into_iter();
    let first = frame(&server.emit(bodies.next().unwrap()));
    let mut stream = TcpStream::connect(address).unwrap();
    stream.set_read_timeout(Some(WAIT)).unwrap();
    write!(stream, "GET /events HTTP/1.1\r\nHost: {address}\r\n\r\n").unwrap();
    let mut seen = Vec::new();
    let mut buffer = [0u8; 4096];
    while !String::from_utf8_lossy(&seen).ends_with(&first) {
        let n = stream.read(&mut buffer).expect("the first frame arrives");
        assert!(n > 0, "the server closed early");
        seen.extend_from_slice(&buffer[..n]);
    }
    let rest: String = bodies.map(|b| frame(&server.emit(b))).collect();
    let mut tail = String::new();
    stream
        .read_to_string(&mut tail)
        .expect("the server ends the response");
    assert_eq!(tail, rest);
}

#[test]
fn another_path_is_not_found_and_another_method_not_allowed() {
    let (server, _) = ended_server();
    assert_eq!(
        get(server.local_addr(), "/", "").0,
        "HTTP/1.1 404 Not Found"
    );
    assert_eq!(
        get(server.local_addr(), "/events/x", "").0,
        "HTTP/1.1 404 Not Found"
    );
    let mut stream = TcpStream::connect(server.local_addr()).unwrap();
    stream.set_read_timeout(Some(WAIT)).unwrap();
    write!(
        stream,
        "POST /events HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n"
    )
    .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    assert!(
        response.starts_with("HTTP/1.1 405 Method Not Allowed\r\n"),
        "{response}"
    );
}

#[test]
#[should_panic(expected = "after the stream ended")]
fn no_event_follows_the_last() {
    let mut sequence = EventSequence::new(TrainingRunId::new());
    sequence.event(RunEventBody::RunFinished);
    sequence.event(step(0, 1.0));
}
