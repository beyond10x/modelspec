//! The built observatory UI served beside the stream (`spec/domains/observatory.yaml`,
//! `UiDirectory`; story:observatory-ui-live-loss-chart).
//!
//! [`Ui::read`] reads a directory whole, `ui/dist` after `task ui`, before the server binds; the
//! server then answers `GET /` with its `index.html` and `GET /<path>` with the file at `<path>`
//! from memory. No request touches the file system, so a path cannot leave the directory, and a
//! file written there later is not served. A request path is percent-decoded (RFC 3986 § 2.1)
//! and then matched byte for byte against the relative paths read; a path that does not decode to
//! UTF-8, or decodes to a NUL or a `..` component, names no file.
//!
//! The limits bound what reading holds as well as what is served: a file is measured before it is
//! read and read no further than the bytes left under the limit, so refusing a directory, even
//! one whose file grows while it is read, never holds more than 64 MiB.

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Read};
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
    /// has no `index.html` or holds more than 4096 files or 64 MiB is refused, without reading more
    /// than 64 MiB.
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
                let over = || {
                    io::Error::other(format!(
                        "it holds more than {} MiB",
                        MAX_BYTES / (1024 * 1024)
                    ))
                };
                let left = MAX_BYTES - *bytes;
                if entry.metadata()?.len() > left {
                    return Err(over());
                }
                // One byte past what is left tells a file that grew since it was measured.
                let mut content = Vec::new();
                fs::File::open(entry.path())?
                    .take(left + 1)
                    .read_to_end(&mut content)?;
                if content.len() as u64 > left {
                    return Err(over());
                }
                *bytes += content.len() as u64;
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
        let decoded = percent_decode(path)?;
        if decoded.contains('\0') || decoded.split('/').any(|component| component == "..") {
            return None;
        }
        let relative = decoded.strip_prefix('/')?;
        let relative = if relative.is_empty() { INDEX } else { relative };
        self.files.get(relative)
    }
}

/// `path` with every `%XX` replaced by the byte it encodes; `None` when a `%` is not followed by
/// two hexadecimal digits or the bytes are not UTF-8.
fn percent_decode(path: &str) -> Option<String> {
    let raw = path.as_bytes();
    let mut bytes = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        if raw[i] == b'%' {
            let hex = std::str::from_utf8(raw.get(i + 1..i + 3)?).ok()?;
            if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                return None;
            }
            bytes.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            bytes.push(raw[i]);
            i += 1;
        }
    }
    String::from_utf8(bytes).ok()
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

    #[test]
    fn a_path_is_percent_decoded_before_it_is_matched() {
        let file = |bytes: &[u8]| UiFile {
            content_type: "text/plain; charset=utf-8",
            bytes: bytes.to_vec(),
        };
        let ui = Ui {
            files: BTreeMap::from([
                (INDEX.to_owned(), file(b"i")),
                ("assets/a b.js".to_owned(), file(b"s")),
                ("assets/a%20b.js".to_owned(), file(b"p")),
                ("é.txt".to_owned(), file(b"e")),
            ]),
        };
        let got = |path: &str| ui.get(path).map(|f| f.bytes.clone());
        assert_eq!(got("/assets/a%20b.js"), Some(b"s".to_vec()));
        assert_eq!(got("/%61ssets/a%20b.js"), Some(b"s".to_vec()));
        assert_eq!(got("/assets/a%2520b.js"), Some(b"p".to_vec()));
        assert_eq!(got("/%C3%A9.txt"), Some(b"e".to_vec()));
        assert_eq!(got("/%c3%a9.txt"), Some(b"e".to_vec()));
        for path in [
            "/%",
            "/%2",
            "/%zz",
            "/%+1",
            "/%00",
            "/index.html%00",
            "/%ff",
            "/%C3",
            "/%2e%2e/index.html",
            "/assets/..%2Fassets/a%20b.js",
            "/assets/%2E%2E/index.html",
        ] {
            assert!(got(path).is_none(), "{path}");
        }
    }
}
