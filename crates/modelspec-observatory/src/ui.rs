//! The built observatory UI served beside the stream (`spec/domains/observatory.yaml`,
//! `UiDirectory`; story:observatory-ui-live-loss-chart).
//!
//! [`Ui::read`] reads a directory whole, `ui/dist` after `task ui`, before the server binds; the
//! server then answers `GET /` with its `index.html` and `GET /<path>` with the file at `<path>`
//! from memory. No request touches the file system, so a path cannot leave the directory, and a
//! file written there later is not served. A request path is matched byte for byte against the
//! relative paths read, without percent-decoding.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

/// The most files a UI directory may hold.
const MAX_FILES: usize = 4096;

/// The most bytes the files of a UI directory may hold together.
const MAX_BYTES: u64 = 64 * 1024 * 1024;

/// The file `GET /` answers with.
const INDEX: &str = "index.html";

/// One file of the built app.
#[derive(Clone, Debug)]
pub(crate) struct UiFile {
    pub(crate) content_type: &'static str,
    pub(crate) bytes: Vec<u8>,
}

/// The files of a built UI directory, by their path relative to it with `/` between components.
#[derive(Clone, Debug)]
pub struct Ui {
    files: BTreeMap<String, UiFile>,
}

impl Ui {
    /// Reads every regular file under `dir`, the directories under it included, without following
    /// a symlink under it. A directory that cannot be read, holds a file whose name is not UTF-8,
    /// has no `index.html` or holds more than 4096 files or 64 MiB is refused.
    pub fn read(dir: &Path) -> io::Result<Ui> {
        let mut ui = Ui {
            files: BTreeMap::new(),
        };
        let mut bytes = 0;
        ui.read_dir(dir, "", &mut bytes)?;
        if !ui.files.contains_key(INDEX) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("it has no {INDEX}; build the UI with `task ui`"),
            ));
        }
        Ok(ui)
    }

    fn read_dir(&mut self, dir: &Path, prefix: &str, bytes: &mut u64) -> io::Result<()> {
        let mut entries = fs::read_dir(dir)?.collect::<io::Result<Vec<_>>>()?;
        entries.sort_by_key(fs::DirEntry::file_name);
        for entry in entries {
            let name = entry.file_name().into_string().map_err(|name| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("the file name {name:?} is not UTF-8"),
                )
            })?;
            let relative = format!("{prefix}{name}");
            let kind = entry.file_type()?;
            if kind.is_dir() {
                self.read_dir(&entry.path(), &format!("{relative}/"), bytes)?;
            } else if kind.is_file() {
                if self.files.len() == MAX_FILES {
                    return Err(io::Error::other(format!(
                        "it holds more than {MAX_FILES} files"
                    )));
                }
                let content = fs::read(entry.path())?;
                *bytes += content.len() as u64;
                if *bytes > MAX_BYTES {
                    return Err(io::Error::other(format!(
                        "it holds more than {} MiB",
                        MAX_BYTES / (1024 * 1024)
                    )));
                }
                let file = UiFile {
                    content_type: content_type(&name),
                    bytes: content,
                };
                self.files.insert(relative, file);
            }
        }
        Ok(())
    }

    /// The file a request for `path`, without its query, is answered with: `index.html` for `/`.
    pub(crate) fn get(&self, path: &str) -> Option<&UiFile> {
        let relative = path.strip_prefix('/')?;
        let relative = if relative.is_empty() { INDEX } else { relative };
        self.files.get(relative)
    }
}

/// The `Content-Type` of a file by the extension of its name.
fn content_type(name: &str) -> &'static str {
    let extension = name.rsplit_once('.').map_or("", |(_, extension)| extension);
    match extension.to_ascii_lowercase().as_str() {
        "html" | "htm" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" | "map" => "application/json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "txt" => "text/plain; charset=utf-8",
        "wasm" => "application/wasm",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_content_type_follows_the_extension() {
        assert_eq!(content_type("index.html"), "text/html; charset=utf-8");
        assert_eq!(content_type("a.B.JS"), "text/javascript; charset=utf-8");
        assert_eq!(content_type("app.css"), "text/css; charset=utf-8");
        assert_eq!(content_type("noext"), "application/octet-stream");
    }

    #[test]
    fn the_root_is_index_html_and_a_path_matches_only_a_file_read() {
        let file = |bytes: &[u8]| UiFile {
            content_type: "text/plain; charset=utf-8",
            bytes: bytes.to_vec(),
        };
        let ui = Ui {
            files: BTreeMap::from([
                (INDEX.to_owned(), file(b"i")),
                ("assets/a.js".to_owned(), file(b"a")),
            ]),
        };
        assert_eq!(ui.get("/").map(|f| f.bytes.as_slice()), Some(&b"i"[..]));
        assert_eq!(
            ui.get("/assets/a.js").map(|f| f.bytes.as_slice()),
            Some(&b"a"[..])
        );
        for path in [
            "",
            "assets/a.js",
            "//index.html",
            "/assets",
            "/assets/",
            "/../index.html",
        ] {
            assert!(ui.get(path).is_none(), "{path}");
        }
    }
}
