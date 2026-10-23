//! Adversarial cases for the built observatory UI served beside the stream
//! (story:observatory-ui-live-loss-chart; `spec/domains/observatory.yaml`, `UiDirectory`).
//!
//! `UiDirectory` promises a directory of at most 4096 files and 64 MiB, a `Content-Type` from
//! the extension and `GET /<path>` answered with the file at `<path>` under the directory. The
//! cases here drive [`Ui::read`] and the server at those limits and with those paths.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use modelspec_observatory::{Server, TrainingRunId, Ui};

const WAIT: Duration = Duration::from_secs(30);
const INDEX: &str = "<!doctype html><title>t</title>";
const MAX_FILES: usize = 4096;
const MAX_BYTES: u64 = 64 * 1024 * 1024;

/// A fresh directory `name` under the test temp dir holding only `index.html`.
fn built(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("ui_serve_adversary")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    std::fs::write(dir.join("index.html"), INDEX).unwrap();
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

fn serve(dir: &Path) -> Server {
    let ui = Ui::read(dir).expect("the directory is read");
    Server::bind_with_ui(
        "127.0.0.1:0".parse().unwrap(),
        TrainingRunId::new(),
        Some(ui),
    )
    .expect("the server binds")
}

/// `UiDirectory`: "at most 4096 files". `index.html` and 4095 more is the limit and is served; one
/// file more is refused, naming the limit. No other test of the suite reaches the file limit, so
/// `MAX_FILES` could be any number and the suite would stay green.
#[test]
fn a_ui_directory_of_4096_files_is_read_and_one_more_is_refused() {
    let dir = built("files");
    std::fs::create_dir_all(dir.join("assets")).unwrap();
    for i in 0..MAX_FILES - 1 {
        std::fs::write(dir.join(format!("assets/{i:04}.js")), "0").unwrap();
    }
    let ui = Ui::read(&dir).unwrap_or_else(|e| panic!("4096 files are within the limit: {e}"));
    let server = Server::bind_with_ui(
        "127.0.0.1:0".parse().unwrap(),
        TrainingRunId::new(),
        Some(ui),
    )
    .expect("the server binds");
    assert_eq!(
        request(server.local_addr(), "GET", "/assets/4094.js").0,
        "HTTP/1.1 200 OK"
    );

    std::fs::write(dir.join("assets/one-more.js"), "0").unwrap();
    let error = Ui::read(&dir).expect_err("4097 files exceed the limit");
    assert!(error.to_string().contains("4096"), "{error}");
}

/// `UiDirectory`: "at most ... 64 MiB". Files of exactly 64 MiB together are read; one byte more is
/// refused, naming the limit. No other test of the suite reaches the byte limit.
#[test]
fn a_ui_directory_of_64_mib_is_read_and_one_byte_more_is_refused() {
    let dir = built("bytes");
    let filler = dir.join("filler.bin");
    let file = std::fs::File::create(&filler).unwrap();
    file.set_len(MAX_BYTES - INDEX.len() as u64).unwrap();
    drop(file);
    Ui::read(&dir).unwrap_or_else(|e| panic!("64 MiB is within the limit: {e}"));

    std::fs::write(dir.join("z.txt"), "x").unwrap();
    let error = Ui::read(&dir).expect_err("64 MiB and one byte exceed the limit");
    assert!(error.to_string().contains("64 MiB"), "{error}");
}

/// The UI's `Content-Security-Policy` (`server.rs`, `UI_POLICY`) keeps the app to scripts, styles
/// and the stream of this server and out of frames. No test of the suite reads the header, so
/// dropping it from the response leaves every test green, the browser tests included.
#[test]
fn a_ui_file_carries_the_content_security_policy() {
    let server = serve(&built("csp"));
    let (status, headers, _) = request(server.local_addr(), "GET", "/");
    assert_eq!(status, "HTTP/1.1 200 OK");
    let policy = header(&headers, "content-security-policy")
        .unwrap_or_else(|| panic!("a Content-Security-Policy header in:\n{headers}"));
    assert!(policy.contains("default-src 'self'"), "{policy}");
    assert!(policy.contains("frame-ancestors 'none'"), "{policy}");
}

/// `UiDirectory`: "`GET /<path>` [answers] with the file at `<path>` under the directory". A URL
/// path is percent-encoded (RFC 3986 § 2.1), so a browser asks for `assets/chart icon.svg` as
/// `/assets/chart%20icon.svg`; matching the raw request bytes against the file names read means
/// such a file is read and can never be served.
#[test]
fn a_file_whose_name_a_browser_percent_encodes_is_served() {
    let dir = built("encoded");
    std::fs::create_dir_all(dir.join("assets")).unwrap();
    std::fs::write(dir.join("assets/chart icon.svg"), "<svg/>").unwrap();
    let server = serve(&dir);
    let (status, headers, body) = request(server.local_addr(), "GET", "/assets/chart%20icon.svg");
    assert_eq!(
        status, "HTTP/1.1 200 OK",
        "the file `assets/chart icon.svg` at the path a browser requests for it"
    );
    assert_eq!(header(&headers, "content-type"), Some("image/svg+xml"));
    assert_eq!(body, b"<svg/>");
}
