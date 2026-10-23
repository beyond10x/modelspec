//! The live stream of one run (`spec/domains/observatory.yaml`, `ServeAddress` and
//! `StreamFrame`): Server-Sent Events over HTTP/1.1 on a loopback address, on the standard
//! library alone.
//!
//! `GET /events` answers `200` with `Content-Type: text/event-stream` and sends every frame of the
//! run from sequence 0, or from the one after the `Last-Event-ID` request header, then each new
//! frame as it is emitted, and closes the response once the frame that ends the stream is sent.
//! A server given a [`Ui`] (`UiDirectory`) also answers `GET /` with its `index.html` and
//! `GET /<path>` with its file at `<path>`. Every other path answers `404`, every other method
//! `405`. Each connection has its own thread.

use std::io::{self, Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::event::{EventSequence, RunEvent, RunEventBody, TrainingRunId};
use crate::ui::{Ui, UiFile};

/// The path of the stream.
pub const EVENTS_PATH: &str = "/events";

/// The longest request head read; a longer one is dropped unanswered.
const MAX_HEAD: usize = 16 * 1024;

/// How long a connection may take to send its request head.
const HEAD_TIMEOUT: Duration = Duration::from_secs(10);

/// Parses a `--serve` value (`modelspec.observatory.ServeAddress`): `<ip>:<port>` with a loopback
/// IP (`127.0.0.0/8`, `[::1]`) or `localhost:<port>`, which is `127.0.0.1:<port>`.
pub fn parse_serve_address(text: &str) -> Result<SocketAddr, String> {
    let address = match text.strip_prefix("localhost:") {
        Some(port) => port
            .parse::<u16>()
            .map(|port| SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port))
            .map_err(|_| format!("`{port}` is not a port"))?,
        None => text.parse::<SocketAddr>().map_err(|_| {
            format!("`{text}` is not `<ip>:<port>`; use a loopback address such as 127.0.0.1:8080")
        })?,
    };
    if !address.ip().is_loopback() {
        return Err(format!(
            "{address} is not a loopback address; the stream is served on this machine only, \
             such as 127.0.0.1:8080"
        ));
    }
    Ok(address)
}

/// The SSE frame of `event`: its `event:`, `id:` and `data:` lines and a blank line.
pub fn frame(event: &RunEvent) -> String {
    let data = crate::record::line(event);
    format!(
        "event: {}\nid: {}\ndata: {data}\n",
        event.body.kind(),
        event.sequence
    )
}

/// The frames emitted so far, and whether the stream has ended.
#[derive(Default)]
struct Frames {
    frames: Vec<String>,
    ended: bool,
}

#[derive(Default)]
struct Hub {
    frames: Mutex<Frames>,
    grown: Condvar,
    /// The built UI served beside the stream, if any.
    ui: Option<Ui>,
}

impl Hub {
    fn lock(&self) -> MutexGuard<'_, Frames> {
        self.frames
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// The stream of one run, served on a loopback address from [`Server::bind`] until the process
/// ends.
pub struct Server {
    address: SocketAddr,
    hub: Arc<Hub>,
    sequence: Mutex<EventSequence>,
    accept: JoinHandle<()>,
}

impl Server {
    /// Binds `address` and starts accepting connections for the events of run `training_run_id`.
    /// A non-loopback address is refused with [`io::ErrorKind::InvalidInput`].
    pub fn bind(address: SocketAddr, training_run_id: TrainingRunId) -> io::Result<Server> {
        Server::bind_with_ui(address, training_run_id, None)
    }

    /// [`Server::bind`], also serving `ui`, when given, at `/` beside the stream.
    pub fn bind_with_ui(
        address: SocketAddr,
        training_run_id: TrainingRunId,
        ui: Option<Ui>,
    ) -> io::Result<Server> {
        if !address.ip().is_loopback() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{address} is not a loopback address"),
            ));
        }
        let listener = TcpListener::bind(address)?;
        let address = listener.local_addr()?;
        let hub = Arc::new(Hub {
            ui,
            ..Hub::default()
        });
        let accepting = Arc::clone(&hub);
        let accept = std::thread::Builder::new()
            .name("observatory-accept".into())
            .spawn(move || accept(&listener, &accepting))?;
        Ok(Server {
            address,
            hub,
            sequence: Mutex::new(EventSequence::new(training_run_id)),
            accept,
        })
    }

    /// The bound address, with the port the system chose for port 0.
    pub fn local_addr(&self) -> SocketAddr {
        self.address
    }

    /// Makes the next event of the run from `body`, sends it to every connected client and keeps
    /// it for every later one, and returns it.
    ///
    /// # Panics
    ///
    /// After an event that ends the stream, as [`EventSequence::event`] does.
    pub fn emit(&self, body: RunEventBody) -> RunEvent {
        let mut sequence = self
            .sequence
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let event = sequence.event(body);
        self.publish(&event);
        event
    }

    /// Sends `event`, made by the caller's own [`EventSequence`], to every connected client and
    /// keeps it for every later one. A caller that publishes its events does not also
    /// [`emit`](Self::emit), whose numbering is the server's own.
    pub fn publish(&self, event: &RunEvent) {
        let mut frames = self.hub.lock();
        frames.frames.push(frame(event));
        frames.ended = event.body.is_last();
        drop(frames);
        self.hub.grown.notify_all();
    }

    /// Binds `address` and serves `events`, a whole recorded run
    /// ([`read_recording`](crate::read_recording)), as the live stream served them: every frame
    /// at once, then the response ends once the event that ends the stream is sent, so `events`
    /// without one, such as none, are served to a response that stays open. A non-loopback
    /// address is refused as by [`Server::bind`].
    pub fn replay(address: SocketAddr, events: &[RunEvent]) -> io::Result<Server> {
        Server::replay_with_ui(address, events, None)
    }

    /// [`Server::replay`], also serving `ui`, when given, at `/` beside the stream.
    pub fn replay_with_ui(
        address: SocketAddr,
        events: &[RunEvent],
        ui: Option<Ui>,
    ) -> io::Result<Server> {
        let training_run_id = events
            .first()
            .map_or_else(TrainingRunId::new, |event| event.training_run_id);
        let server = Server::bind_with_ui(address, training_run_id, ui)?;
        if let Some(last) = events.last() {
            *server
                .sequence
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = EventSequence::after(last);
        }
        let mut frames = server.hub.lock();
        frames.frames.extend(events.iter().map(frame));
        frames.ended = events.last().is_some_and(|event| event.body.is_last());
        drop(frames);
        server.hub.grown.notify_all();
        Ok(server)
    }

    /// Blocks while the server accepts connections, which is until the process ends.
    pub fn wait(self) {
        let _ = self.accept.join();
    }
}

fn accept(listener: &TcpListener, hub: &Arc<Hub>) {
    for connection in listener.incoming() {
        let Ok(stream) = connection else { continue };
        let hub = Arc::clone(hub);
        let _ = std::thread::Builder::new()
            .name("observatory-client".into())
            .spawn(move || {
                let _ = respond(stream, &hub);
            });
    }
}

/// The method, the path without its query and the `Last-Event-ID` header of a request head.
fn parse_head(head: &str) -> Option<(&str, &str, Option<&str>)> {
    let mut lines = head.split("\r\n");
    let mut request = lines.next()?.split(' ');
    let (method, target) = (request.next()?, request.next()?);
    let path = target.split_once('?').map_or(target, |(path, _)| path);
    let last_event_id = lines.find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.trim()
            .eq_ignore_ascii_case("last-event-id")
            .then(|| value.trim())
    });
    Some((method, path, last_event_id))
}

/// Reads the request head, up to and without its blank line.
fn read_head(stream: &mut TcpStream) -> io::Result<String> {
    stream.set_read_timeout(Some(HEAD_TIMEOUT))?;
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        if head.len() >= MAX_HEAD || stream.read(&mut byte)? == 0 {
            return Err(io::ErrorKind::InvalidData.into());
        }
        head.push(byte[0]);
    }
    head.truncate(head.len() - 4);
    String::from_utf8(head).map_err(|_| io::ErrorKind::InvalidData.into())
}

/// The answer to a method other than `GET` on a path that is served.
const NOT_ALLOWED: &[u8] =
    b"HTTP/1.1 405 Method Not Allowed\r\nAllow: GET\r\nContent-Length: 0\r\n\
      Connection: close\r\n\r\n";

/// The `Content-Security-Policy` of a UI file: the app loads its scripts, styles and stream from
/// this server only.
const UI_POLICY: &str = "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'";

/// Answers with a file of the built UI.
fn serve_file(stream: &mut TcpStream, file: &UiFile) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-cache\r\n\
         X-Content-Type-Options: nosniff\r\nContent-Security-Policy: {UI_POLICY}\r\n\
         Connection: close\r\n\r\n",
        file.content_type,
        file.bytes.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(&file.bytes)?;
    stream.flush()
}

fn respond(mut stream: TcpStream, hub: &Hub) -> io::Result<()> {
    let head = read_head(&mut stream)?;
    let Some((method, path, last_event_id)) = parse_head(&head) else {
        return Ok(());
    };
    if path != EVENTS_PATH {
        let file = hub.ui.as_ref().and_then(|ui| ui.get(path));
        return match file {
            None => stream.write_all(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            ),
            Some(_) if method != "GET" => stream.write_all(NOT_ALLOWED),
            Some(file) => serve_file(&mut stream, file),
        };
    }
    if method != "GET" {
        return stream.write_all(NOT_ALLOWED);
    }
    let mut next = last_event_id
        .and_then(|id| id.parse::<usize>().ok())
        .map_or(0, |id| id.saturating_add(1));
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
          Connection: close\r\n\r\n",
    )?;
    stream.flush()?;
    loop {
        let mut frames = hub.lock();
        while next >= frames.frames.len() && !frames.ended {
            frames = hub
                .grown
                .wait(frames)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        let batch: String = frames.frames.get(next..).unwrap_or_default().concat();
        next = next.max(frames.frames.len());
        let ended = frames.ended;
        drop(frames);
        stream.write_all(batch.as_bytes())?;
        stream.flush()?;
        if ended {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_head_gives_its_method_path_and_last_event_id() {
        assert_eq!(
            parse_head("GET /events?x=1 HTTP/1.1\r\nHost: a\r\nLast-Event-ID:  7 "),
            Some(("GET", "/events", Some("7")))
        );
        assert_eq!(
            parse_head("GET /events HTTP/1.1\r\nHost: a"),
            Some(("GET", "/events", None))
        );
        assert_eq!(parse_head("GET"), None);
    }
}
