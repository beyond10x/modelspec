//! The built observatory UI served beside the stream (story:observatory-ui-live-loss-chart;
//! `spec/domains/observatory.yaml`, `UiDirectory`, `StreamFrame`), driven without a training run
//! and without Node: the directory here is written by the test.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use modelspec_observatory::{
    EventSequence, RunEventBody, RunStarted, Server, TrainingRunId, Ui, frame,
};

const WAIT: Duration = Duration::from_secs(30);
const INDEX: &str = "<!doctype html><title>t</title><script src=\"./assets/app.js\"></script>";
const SCRIPT: &str = "console.log('app');";

/// A fresh directory `name` under the test temp dir holding a built app: `index.html`,
/// `assets/app.js` and `assets/app.css`.
fn built(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("ui_serve")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("assets")).expect("the temp dir is writable");
    std::fs::write(dir.join("index.html"), INDEX).unwrap();
    std::fs::write(dir.join("assets/app.js"), SCRIPT).unwrap();
    std::fs::write(dir.join("assets/app.css"), "main{}").unwrap();
    dir
}

/// The response to `<method> <path>` at `address`: status line, headers, body.
fn request(address: SocketAddr, method: &str, path: &str) -> (String, String, Vec<u8>) {
    let mut stream = TcpStream::connect(address).expect("the server accepts a connection");
    stream.set_read_timeout(Some(WAIT)).unwrap();
    write!(stream, "{method} {path} HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .expect("the server ends the response");
    let split = response
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .unwrap_or_else(|| panic!("a response head: {response:?}"));
    let head = String::from_utf8(response[..split].to_vec()).unwrap();
    let body = response[split + 4..].to_vec();
    let (status, headers) = head.split_once("\r\n").unwrap_or((&head, ""));
    (status.to_owned(), headers.to_owned(), body)
}

fn header<'a>(headers: &'a str, name: &str) -> Option<&'a str> {
    headers.lines().find_map(|line| {
        let (n, v) = line.split_once(':')?;
        n.trim().eq_ignore_ascii_case(name).then(|| v.trim())
    })
}

fn started() -> RunEventBody {
    RunEventBody::RunStarted(RunStarted {
        steps: 1,
        start_step: 0,
    })
}

#[test]
fn the_root_serves_index_html_and_every_file_beside_it() {
    let ui = Ui::read(&built("root")).expect("the directory is read");
    let server = Server::bind_with_ui(
        "127.0.0.1:0".parse().unwrap(),
        TrainingRunId::new(),
        Some(ui),
    )
    .expect("the server binds");
    let address = server.local_addr();

    let (status, headers, body) = request(address, "GET", "/");
    assert_eq!(status, "HTTP/1.1 200 OK");
    assert_eq!(body, INDEX.as_bytes());
    assert_eq!(
        header(&headers, "content-type"),
        Some("text/html; charset=utf-8")
    );
    assert_eq!(
        header(&headers, "content-length"),
        Some(INDEX.len().to_string().as_str())
    );
    assert_eq!(header(&headers, "x-content-type-options"), Some("nosniff"));

    let (status, headers, body) = request(address, "GET", "/index.html");
    assert_eq!(status, "HTTP/1.1 200 OK");
    assert_eq!(body, INDEX.as_bytes());
    assert_eq!(
        header(&headers, "content-type"),
        Some("text/html; charset=utf-8")
    );

    let (status, headers, body) = request(address, "GET", "/assets/app.js?v=1");
    assert_eq!(status, "HTTP/1.1 200 OK");
    assert_eq!(body, SCRIPT.as_bytes());
    assert_eq!(
        header(&headers, "content-type"),
        Some("text/javascript; charset=utf-8")
    );

    let (status, headers, _) = request(address, "GET", "/assets/app.css");
    assert_eq!(status, "HTTP/1.1 200 OK");
    assert_eq!(
        header(&headers, "content-type"),
        Some("text/css; charset=utf-8")
    );
}

#[test]
fn a_path_outside_the_built_app_is_not_found_and_another_method_not_allowed() {
    let dir = built("outside");
    let ui = Ui::read(&dir).expect("the directory is read");
    let server = Server::bind_with_ui(
        "127.0.0.1:0".parse().unwrap(),
        TrainingRunId::new(),
        Some(ui),
    )
    .expect("the server binds");
    let address = server.local_addr();
    for path in [
        "/missing.js",
        "/assets",
        "/assets/",
        "/../index.html",
        "/assets/../index.html",
        "//index.html",
        "/%2e%2e/Cargo.toml",
    ] {
        assert_eq!(
            request(address, "GET", path).0,
            "HTTP/1.1 404 Not Found",
            "{path}"
        );
    }
    let (status, headers, _) = request(address, "POST", "/");
    assert_eq!(status, "HTTP/1.1 405 Method Not Allowed");
    assert_eq!(header(&headers, "allow"), Some("GET"));
}

/// The directory is read whole before the server binds: a file written into it later is not
/// served.
#[test]
fn the_directory_is_read_once() {
    let dir = built("once");
    let ui = Ui::read(&dir).expect("the directory is read");
    std::fs::write(dir.join("later.js"), "x").unwrap();
    std::fs::write(dir.join("index.html"), "changed").unwrap();
    let server = Server::bind_with_ui(
        "127.0.0.1:0".parse().unwrap(),
        TrainingRunId::new(),
        Some(ui),
    )
    .expect("the server binds");
    assert_eq!(
        request(server.local_addr(), "GET", "/later.js").0,
        "HTTP/1.1 404 Not Found"
    );
    assert_eq!(request(server.local_addr(), "GET", "/").2, INDEX.as_bytes());
}

/// The stream is served beside the app as it is without one.
#[test]
fn the_stream_is_served_beside_the_app() {
    let ui = Ui::read(&built("stream")).expect("the directory is read");
    let mut sequence = EventSequence::new(TrainingRunId::new());
    let events = vec![
        sequence.event(started()),
        sequence.event(RunEventBody::RunFinished),
    ];
    let server = Server::replay_with_ui("127.0.0.1:0".parse().unwrap(), &events, Some(ui))
        .expect("the server binds");
    let (status, headers, body) = request(server.local_addr(), "GET", "/events");
    assert_eq!(status, "HTTP/1.1 200 OK");
    assert_eq!(header(&headers, "content-type"), Some("text/event-stream"));
    let expected: String = events.iter().map(frame).collect();
    assert_eq!(String::from_utf8(body).unwrap(), expected);
    assert_eq!(
        request(server.local_addr(), "GET", "/").0,
        "HTTP/1.1 200 OK"
    );
}

#[test]
fn a_directory_without_index_html_is_refused() {
    let dir = built("no_index");
    std::fs::remove_file(dir.join("index.html")).unwrap();
    let error = Ui::read(&dir).expect_err("no index.html is refused");
    assert!(error.to_string().contains("index.html"), "{error}");
}

#[test]
fn a_missing_directory_is_refused() {
    let dir = built("missing").join("nothing_here");
    assert!(Ui::read(&dir).is_err());
}

#[cfg(unix)]
#[test]
fn a_symlink_in_the_directory_is_not_served() {
    let dir = built("symlink");
    std::os::unix::fs::symlink("/etc/hostname", dir.join("host.txt")).unwrap();
    std::os::unix::fs::symlink(dir.join("assets"), dir.join("linked")).unwrap();
    let ui = Ui::read(&dir).expect("the directory is read");
    let server = Server::bind_with_ui(
        "127.0.0.1:0".parse().unwrap(),
        TrainingRunId::new(),
        Some(ui),
    )
    .expect("the server binds");
    for path in ["/host.txt", "/linked/app.js"] {
        assert_eq!(
            request(server.local_addr(), "GET", path).0,
            "HTTP/1.1 404 Not Found",
            "{path}"
        );
    }
}

/// Without a UI only the stream is served, as before this story.
#[test]
fn without_a_ui_the_root_is_not_found() {
    let server = Server::bind_with_ui("127.0.0.1:0".parse().unwrap(), TrainingRunId::new(), None)
        .expect("the server binds");
    assert_eq!(
        request(server.local_addr(), "GET", "/").0,
        "HTTP/1.1 404 Not Found"
    );
}
