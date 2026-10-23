//! `modelspec observe <file> [--serve <addr>]`: replay a recorded run file over the event stream
//! (`spec/domains/observatory.yaml`, `RecordedRunFile`; [`modelspec_observatory`]).
//!
//! The file is what `modelspec train --record <file>` writes: one `RunEvent` per line in
//! `sequence` order. It is read whole before anything is served; a file that cannot be read, a
//! line that is not a whole `RunEvent` (a truncated file), a sequence that skips or repeats a
//! number, two `training_run_id` values, a first event that is not `RunStarted` or a last one
//! that is not `RunFinished` or `RunFailed` is refused on one line on stderr with exit 1. Then
//! the address is bound, `serving http://<bound address>/events` is printed on stderr, and
//! `GET /events` sends every frame of the run at once, the same frames with the same `id:` lines
//! as the live stream sent, and ends the response. The process serves until SIGINT, SIGTERM or
//! SIGHUP and then exits 0; stdout is empty.

use std::io::Write;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::mpsc;

use modelspec_observatory::{EVENTS_PATH, Server, read_recording};

/// Arguments of `modelspec observe`.
#[derive(clap::Args)]
pub struct Args {
    /// The recorded run file to replay, as `modelspec train --record` writes it.
    pub path: PathBuf,
    /// Serve the replay as text/event-stream at path /events of ADDR, a loopback address such as
    /// 127.0.0.1:8080 or localhost:8080 (port 0 picks a free one); the process serves until
    /// SIGINT, SIGTERM or SIGHUP, then exits 0.
    #[arg(
        long,
        value_name = "ADDR",
        default_value = "127.0.0.1:0",
        value_parser = modelspec_observatory::parse_serve_address
    )]
    pub serve: SocketAddr,
}

/// Runs `modelspec observe` and returns the process exit code.
pub fn run(args: Args, _stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    let file = args.path.display().to_string();
    let events = match read_recording(&args.path) {
        Ok(events) => events,
        Err(error) => {
            let _ = writeln!(stderr, "modelspec observe {file}: {error}");
            return crate::EXIT_FAILURE;
        }
    };
    let (send, interrupted) = mpsc::channel();
    let handled = ctrlc::set_handler(move || {
        let _ = send.send(());
    })
    .is_ok();
    let server = match Server::replay(args.serve, &events) {
        Ok(server) => server,
        Err(error) => {
            let _ = writeln!(
                stderr,
                "modelspec observe {file}: cannot serve on {}: {error}",
                args.serve
            );
            return crate::EXIT_FAILURE;
        }
    };
    let _ = writeln!(
        stderr,
        "serving http://{}{EVENTS_PATH}",
        server.local_addr()
    );
    let _ = stderr.flush();
    if handled {
        let _ = interrupted.recv();
    } else {
        server.wait();
    }
    crate::EXIT_SUCCESS
}
