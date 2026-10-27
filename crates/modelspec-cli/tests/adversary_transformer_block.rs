//! Adversary pass 1 of story:transformer-block-benchmark: what `train --steps 0 --checkpoint` and
//! `train --write-batch` export for the opponent's loss check, read against what the help text,
//! the CHANGELOG and DESIGN.md say they are, and the checkpoint read back by `infer`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const EXAMPLE: &str = "examples/transformer_block.mspec";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("adversary_transformer_block_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn path(p: &Path) -> &str {
    p.to_str().expect("a UTF-8 path")
}

fn assert_exit(output: &Output, code: i32) {
    assert_eq!(
        output.status.code(),
        Some(code),
        "stdout: {}\nstderr: {}",
        text(&output.stdout),
        text(&output.stderr)
    );
}

/// The value of the first line of `out` starting with `prefix`.
fn after<'a>(out: &'a str, prefix: &str) -> Option<&'a str> {
    out.lines().find_map(|line| line.strip_prefix(prefix))
}

/// Exports the initial checkpoint and the batch of the example into `case`.
fn export(case: &Path) -> (PathBuf, PathBuf, String) {
    let ckpt = case.join("initial.msckpt");
    let batch = case.join("batch");
    let output = modelspec(&[
        "train",
        EXAMPLE,
        "--backend",
        "cpu",
        "--steps",
        "0",
        "--checkpoint",
        path(&ckpt),
        "--write-batch",
        path(&batch),
    ]);
    assert_exit(&output, 0);
    let loss = after(&text(&output.stdout), "step 0 loss ")
        .expect("a step 0 line")
        .to_owned();
    (ckpt, batch, loss)
}

/// ADV-TB-4, rewritten in fix pass 1 (coordinator decision D84) to what `--write-batch` is
/// documented as writing (its help text, the CHANGELOG, DESIGN.md § 8): one file per input and
/// target declaration of the example and nothing else, each the tensor's row-major
/// little-endian `f32` elements, 4 bytes per element of the declared shape, `x` and `y`
/// `[1024, 128]` and `key_mask` `[8, 128]`, whose values are the synthetic key mask, all ones.
/// It is not a `--data` directory: `--data` on the example is refused with E-DATA-001, because
/// dense data counts the key mask as a second input.
#[test]
fn adv_tb_4_the_written_batch_of_the_example_is_one_f32_file_per_declaration_and_no_data_directory()
{
    let case = scratch("data");
    let (_, batch, _) = export(&case);
    let mut files: Vec<(String, u64)> = std::fs::read_dir(&batch)
        .expect("the batch directory")
        .map(|entry| {
            let entry = entry.expect("an entry");
            let name = entry.file_name().into_string().expect("a UTF-8 name");
            (name, entry.metadata().expect("metadata").len())
        })
        .collect();
    files.sort();
    assert_eq!(
        files,
        [
            ("key_mask.f32".to_owned(), 8 * 128 * 4),
            ("x.f32".to_owned(), 1024 * 128 * 4),
            ("y.f32".to_owned(), 1024 * 128 * 4),
        ]
    );
    let mask = std::fs::read(batch.join("key_mask.f32")).expect("the key mask");
    assert!(
        mask.chunks_exact(4)
            .all(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()) == 1.0),
        "the key mask is all ones"
    );

    let output = modelspec(&[
        "train",
        EXAMPLE,
        "--backend",
        "cpu",
        "--steps",
        "0",
        "--data",
        path(&batch),
    ]);
    assert_exit(&output, 1);
    let stderr = text(&output.stderr);
    assert!(stderr.contains("E-DATA-001"), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
}

/// The checkpoint of `--steps 0` is the initial state: `infer` on it prints the loss the run
/// printed at step 0, and `--write-batch` into a path that is a file is an error, not a panic.
#[test]
fn infer_on_the_step_0_checkpoint_prints_the_step_0_loss() {
    let case = scratch("infer");
    let (ckpt, _, loss) = export(&case);
    let output = modelspec(&[
        "infer",
        EXAMPLE,
        "--backend",
        "cpu",
        "--checkpoint",
        path(&ckpt),
    ]);
    assert_exit(&output, 0);
    assert_eq!(after(&text(&output.stdout), "loss "), Some(loss.as_str()));

    let file = case.join("a-file");
    std::fs::write(&file, b"x").unwrap();
    let output = modelspec(&[
        "train",
        EXAMPLE,
        "--backend",
        "cpu",
        "--steps",
        "0",
        "--write-batch",
        path(&file),
    ]);
    assert_exit(&output, 1);
    let stderr = text(&output.stderr);
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(stderr.contains("a-file"), "{stderr}");
}
