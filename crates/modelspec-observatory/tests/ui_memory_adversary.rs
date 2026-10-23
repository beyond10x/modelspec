//! Adversarial case for the memory bound of [`Ui::read`] (story:observatory-ui-live-loss-chart;
//! `spec/domains/observatory.yaml`, `UiDirectory`: the directory "is read whole before the address
//! is bound ... and must hold ... at most 4096 files and 64 MiB").
//!
//! The 64 MiB limit is what bounds the memory the UI holds, because the directory is read whole.
//! A directory that exceeds it is refused, but the refusal must not first hold more than the limit:
//! a mistaken `--ui` (the repository root, a home directory) can hold a file of any size. This file
//! holds this one test so that the process's peak resident set is its own.

#![cfg(target_os = "linux")]

use std::path::Path;

use modelspec_observatory::Ui;

const MIB: u64 = 1024 * 1024;
/// The size of the one large file: eight times the limit.
const LARGE: u64 = 512 * MIB;
/// What reading may hold beyond the resident set before it: the 64 MiB the limit admits and
/// slack for the allocator and the test harness.
const ALLOWED: u64 = 128 * MIB;

/// The `VmRSS` or `VmHWM` line of `/proc/self/status`, in bytes.
fn status_bytes(field: &str) -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("/proc/self/status");
    let kib: u64 = status
        .lines()
        .find_map(|line| line.strip_prefix(field))
        .and_then(|rest| rest.trim().strip_suffix("kB"))
        .and_then(|n| n.trim().parse().ok())
        .unwrap_or_else(|| panic!("{field} in /proc/self/status"));
    kib * 1024
}

#[test]
fn a_file_over_the_limit_is_refused_without_reading_it_whole() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("ui_memory_adversary")
        .join("large");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    std::fs::write(dir.join("index.html"), "<!doctype html>").unwrap();
    // Sparse: no disk is used, and reading it returns zeros.
    let file = std::fs::File::create(dir.join("big.bin")).unwrap();
    file.set_len(LARGE).unwrap();
    drop(file);

    // Writing 5 to clear_refs resets the peak resident set (VmHWM) to the current one.
    std::fs::write("/proc/self/clear_refs", "5").expect("the peak resident set can be reset");
    let before = status_bytes("VmRSS:");
    let refused = Ui::read(&dir);
    let peak = status_bytes("VmHWM:");

    let error = refused.expect_err("a directory of 512 MiB is refused");
    assert!(error.to_string().contains("64 MiB"), "{error}");
    let held = peak.saturating_sub(before);
    assert!(
        held <= ALLOWED,
        "Ui::read held {} MiB at its peak to refuse a directory over the 64 MiB limit (one {} MiB \
         file); it reads a file whole before it checks the limit",
        held / MIB,
        LARGE / MIB
    );
}
