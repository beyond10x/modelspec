//! `modelspec infer <file> --backend cpu|vulkan --checkpoint <path> --data <dir> --function
//! <name> --out <file>` (story:decider-infer-on-data; DESIGN.md § 8, `spec/domains/data.yaml`,
//! `InferenceOutputFile`).
//!
//! The fixture `fixtures/infer_scores.mspec` has two functions over one weight `w: [3, 2]`:
//! `forward`, which returns the loss, and `scores(x: [4, 3]) -> [4, 2]`, the linear map
//! `matmul(x, w)`. Each test starts it from the known weight [`W`] with `--init`, writes that
//! state to a checkpoint with `train --steps 0 --checkpoint`, and runs `infer` on it. The dense
//! directory holds `x.f32`, [`SAMPLES`] samples of three values, three blocks of `B = 4`. Every
//! value of `W` and `x` is a small multiple of 1/4, so every product and partial sum of the
//! matmul is exact in `f32` and the host product below is the CPU's bit for bit, whatever order
//! it adds in.
//!
//! The device test is ignored by default; run it with
//! `cargo test -p modelspec-cli --test infer_data -- --ignored --test-threads=1`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const FIXTURE: &str = "crates/modelspec-cli/tests/fixtures/infer_scores.mspec";

/// The known weight `w: [3, 2]`, row-major.
const W: [f32; 6] = [0.5, -1.25, 2.0, 0.75, -0.5, 1.5];
/// The input width `D` and the score width `C`.
const D: usize = 3;
const C: usize = 2;
/// The leading dimension `B` of `scores`'s argument.
const B: usize = 4;
/// The samples of `x.f32`: three blocks of `B`.
const SAMPLES: usize = 3 * B;

/// What `infer` without `--data` prints today for the fixture from [`W`]: the prediction and
/// the loss of `forward` on the synthetic batch of the train block. `forward` returns the loss,
/// so the prediction is that loss. Recorded from the binary of 3af084f, before `--data` existed.
const WITHOUT_DATA: &str = "\
prediction: Tensor<f32, []> = [3.8793685]
loss 3.8793685
";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh directory for one test.
fn dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("infer_data")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
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

fn ok(args: &[&str]) -> String {
    let output = modelspec(args);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{args:?}\nstdout: {}\nstderr: {}",
        text(&output.stdout),
        text(&output.stderr)
    );
    text(&output.stdout)
}

fn write_f32(path: &Path, values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).expect("the file is writable");
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("the output file exists");
    assert_eq!(bytes.len() % 4, 0, "{} bytes", bytes.len());
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// `samples` samples of `x`, each `D` multiples of 1/4 in `[-2, 2]`.
fn x_values(samples: usize) -> Vec<f32> {
    (0..samples * D)
        .map(|i| ((i * 7) % 17) as f32 * 0.25 - 2.0)
        .collect()
}

/// The checkpoint of the fixture at step 0 with `w` = [`W`], written under `d`.
fn checkpoint(d: &Path) -> PathBuf {
    let init = d.join("init");
    std::fs::create_dir_all(&init).expect("the init dir is writable");
    write_f32(&init.join("w.f32"), &W);
    let ckpt = d.join("scores.msckpt");
    ok(&[
        "train",
        FIXTURE,
        "--backend",
        "cpu",
        "--steps",
        "0",
        "--init",
        path(&init),
        "--checkpoint",
        path(&ckpt),
    ]);
    ckpt
}

/// A dense directory under `d` holding `x.f32` with `samples` samples.
fn data(d: &Path, samples: usize) -> PathBuf {
    let data = d.join("data");
    std::fs::create_dir_all(&data).expect("the data dir is writable");
    write_f32(&data.join("x.f32"), &x_values(samples));
    data
}

/// `matmul(x, W)` on the host, row after row: `samples · C` values.
fn host_scores(samples: usize) -> Vec<f32> {
    let x = x_values(samples);
    let mut out = Vec::with_capacity(samples * C);
    for i in 0..samples {
        for j in 0..C {
            let mut acc = 0.0f32;
            for k in 0..D {
                acc += x[i * D + k] * W[k * C + j];
            }
            out.push(acc);
        }
    }
    out
}

/// `infer --function scores --data <data> --out <out>` on `backend`.
fn infer_scores(backend: &str, ckpt: &Path, data: &Path, out: &Path) -> Output {
    modelspec(&[
        "infer",
        FIXTURE,
        "--backend",
        backend,
        "--checkpoint",
        path(ckpt),
        "--data",
        path(data),
        "--function",
        "scores",
        "--out",
        path(out),
    ])
}

/// infer-function-cpu: `scores` over three blocks writes `3·B·C` values equal, bit for bit, to
/// the host product.
#[test]
fn infer_function_cpu_writes_every_block_bit_for_bit() {
    let d = dir("cpu");
    let ckpt = checkpoint(&d);
    let data = data(&d, SAMPLES);
    let out = d.join("scores.f32");
    let output = infer_scores("cpu", &ckpt, &data, &out);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        text(&output.stderr)
    );
    let written = read_f32(&out);
    let expected = host_scores(SAMPLES);
    assert_eq!(written.len(), 3 * B * C);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&written),
        bits(&expected),
        "{written:?} vs {expected:?}"
    );
}

/// infer-function-stdout: stdout is exactly one line naming the function, the block count, the
/// row count and the file.
#[test]
fn infer_function_stdout_is_one_line_naming_function_blocks_rows_and_file() {
    let d = dir("stdout");
    let ckpt = checkpoint(&d);
    let data = data(&d, SAMPLES);
    let out = d.join("scores.f32");
    let output = infer_scores("cpu", &ckpt, &data, &out);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        text(&output.stderr)
    );
    assert_eq!(
        text(&output.stdout),
        format!(
            "function scores blocks 3 rows {} out {}\n",
            3 * B,
            path(&out)
        )
    );
}

/// infer-without-data-unchanged: without `--data`, `--function` and `--out`, `infer` prints the
/// two lines of `forward` on the synthetic batch it prints today.
#[test]
fn infer_without_data_prints_the_two_lines_of_today() {
    let d = dir("without");
    let ckpt = checkpoint(&d);
    let stdout = ok(&[
        "infer",
        FIXTURE,
        "--backend",
        "cpu",
        "--checkpoint",
        path(&ckpt),
    ]);
    assert_eq!(stdout, WITHOUT_DATA);
}

/// The refusal of `output`: exit `code`, nothing on stdout, and stderr naming `expected`.
fn refused(output: &Output, code: i32, expected: &[&str]) {
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(code), "stderr: {stderr}");
    assert_eq!(text(&output.stdout), "", "stderr: {stderr}");
    for part in expected {
        assert!(stderr.contains(part), "missing {part:?} in: {stderr}");
    }
}

/// infer-function-refusals: an unknown `--function` is E-DATA-010 naming the model's functions;
/// nothing is written to the output file, which an earlier run left in place.
#[test]
fn infer_function_refuses_an_unknown_function_with_e_data_010() {
    let d = dir("unknown");
    let ckpt = checkpoint(&d);
    let data = data(&d, SAMPLES);
    let out = d.join("scores.f32");
    std::fs::write(&out, b"kept").expect("the output file is writable");
    let output = modelspec(&[
        "infer",
        FIXTURE,
        "--backend",
        "cpu",
        "--checkpoint",
        path(&ckpt),
        "--data",
        path(&data),
        "--function",
        "logits",
        "--out",
        path(&out),
    ]);
    refused(&output, 1, &["E-DATA-010", "logits", "forward", "scores"]);
    assert_eq!(std::fs::read(&out).unwrap(), b"kept");
}

/// infer-function-refusals: an argument with no file in the directory is E-DATA-007; nothing
/// is written.
#[test]
fn infer_function_refuses_an_argument_without_a_file_with_e_data_007() {
    let d = dir("missing");
    let ckpt = checkpoint(&d);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    write_f32(&data.join("y.f32"), &x_values(SAMPLES));
    let out = d.join("scores.f32");
    let output = infer_scores("cpu", &ckpt, &data, &out);
    refused(&output, 1, &["E-DATA-007", "x.f32"]);
    assert!(!out.exists(), "nothing is written on a refusal");
}

/// infer-function-refusals: a sample count that is not a multiple of `B` is E-DATA-009;
/// nothing is written.
#[test]
fn infer_function_refuses_a_partial_block_with_e_data_009() {
    let d = dir("partial");
    let ckpt = checkpoint(&d);
    let data = data(&d, SAMPLES - 2);
    let out = d.join("scores.f32");
    let output = infer_scores("cpu", &ckpt, &data, &out);
    refused(
        &output,
        1,
        &[
            "E-DATA-009",
            &format!("samples: {}", SAMPLES - 2),
            "batch: 4",
        ],
    );
    assert!(!out.exists(), "nothing is written on a refusal");
}

/// infer-function-refusals: `--function` or `--out` without `--data` is a usage error, exit 2;
/// nothing is written.
#[test]
fn infer_function_or_out_without_data_is_a_usage_error() {
    let d = dir("usage");
    let ckpt = checkpoint(&d);
    let out = d.join("scores.f32");
    for extra in [
        &["--function", "scores"][..],
        &["--out", path(&out)][..],
        &["--function", "scores", "--out", path(&out)][..],
    ] {
        let mut args = vec![
            "infer",
            FIXTURE,
            "--backend",
            "cpu",
            "--checkpoint",
            path(&ckpt),
        ];
        args.extend_from_slice(extra);
        let output = modelspec(&args);
        assert_eq!(
            output.status.code(),
            Some(2),
            "{extra:?}: stderr: {}",
            text(&output.stderr)
        );
        assert!(
            text(&output.stderr).contains("--data"),
            "{extra:?}: {}",
            text(&output.stderr)
        );
        assert!(!out.exists(), "{extra:?}: nothing is written");
    }
}

/// The names of the entries of `dir`, sorted.
fn entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("the dir is readable")
        .map(|entry| {
            entry
                .expect("the entry is readable")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}

/// infer-function-refusals: an `--out` that resolves to a file the run reads (the checkpoint,
/// the model file, or an argument's file of `--data`, however the path spells it: through `..`
/// or a symbolic link) is E-DATA-010 naming `--out` and that file, before the function runs;
/// every input keeps its bytes and nothing is created beside them.
#[cfg(unix)]
#[test]
fn infer_function_refuses_an_out_that_names_a_file_the_run_reads_with_e_data_010() {
    let d = dir("out-reads");
    let ckpt = checkpoint(&d);
    let data = data(&d, SAMPLES);
    let model = d.join("model.mspec");
    std::fs::copy(repo_root().join(FIXTURE), &model).expect("the fixture copies");
    let link = d.join("link.f32");
    std::os::unix::fs::symlink(data.join("x.f32"), &link).expect("a symlink");
    let inputs = [ckpt.clone(), model.clone(), data.join("x.f32")];
    let before: Vec<Vec<u8>> = inputs.iter().map(|p| std::fs::read(p).unwrap()).collect();
    let (top, inside) = (entries(&d), entries(&data));
    for out in [
        ckpt.clone(),
        model.clone(),
        data.join("x.f32"),
        data.join("..").join("data").join("x.f32"),
        link.clone(),
    ] {
        let output = modelspec(&[
            "infer",
            path(&model),
            "--backend",
            "cpu",
            "--checkpoint",
            path(&ckpt),
            "--data",
            path(&data),
            "--function",
            "scores",
            "--out",
            path(&out),
        ]);
        refused(&output, 1, &["E-DATA-010", "--out", path(&out)]);
        for (input, bytes) in inputs.iter().zip(&before) {
            assert_eq!(
                &std::fs::read(input).unwrap(),
                bytes,
                "--out {}: {} changed",
                out.display(),
                input.display()
            );
        }
        assert_eq!(entries(&d), top, "--out {}", out.display());
        assert_eq!(entries(&data), inside, "--out {}", out.display());
    }
}

/// infer-function-refusals: a function that takes no argument is E-DATA-010, and its refusal
/// names the model's functions, as every E-DATA-010 that refuses a function does
/// (`spec/domains/data.yaml`, `InferenceOutputFile`).
#[test]
fn infer_function_refuses_a_function_without_arguments_naming_the_functions() {
    let d = dir("no-arguments");
    let source = std::fs::read_to_string(repo_root().join(FIXTURE)).expect("the fixture reads");
    let source = source.replacen(
        "    fn scores(",
        "    fn table() -> Tensor<f32, [3, 2]> {\n        return relu(w);\n    }\n\n    fn scores(",
        1,
    );
    let model = d.join("table.mspec");
    std::fs::write(&model, source).expect("the model is writable");
    let ckpt = d.join("table.msckpt");
    ok(&[
        "train",
        path(&model),
        "--backend",
        "cpu",
        "--steps",
        "0",
        "--checkpoint",
        path(&ckpt),
    ]);
    let data = data(&d, SAMPLES);
    let out = d.join("table.f32");
    let output = modelspec(&[
        "infer",
        path(&model),
        "--backend",
        "cpu",
        "--checkpoint",
        path(&ckpt),
        "--data",
        path(&data),
        "--function",
        "table",
        "--out",
        path(&out),
    ]);
    refused(&output, 1, &["E-DATA-010", "forward", "table", "scores"]);
    assert!(!out.exists(), "nothing is written on a refusal");
}

/// infer-function-cpu: FILE is written beside itself and renamed over it, so a run that cannot
/// replace it (here FILE is a directory) exits 1 and leaves FILE and its directory as they
/// were, and a run that succeeds leaves FILE and nothing else beside it.
#[test]
fn infer_function_replaces_out_whole_and_leaves_no_temporary_file() {
    let d = dir("replace");
    let ckpt = checkpoint(&d);
    let data = data(&d, SAMPLES);
    let blocked = d.join("blocked.f32");
    std::fs::create_dir_all(&blocked).unwrap();
    std::fs::write(blocked.join("inside"), b"kept").unwrap();
    let top = entries(&d);
    let output = infer_scores("cpu", &ckpt, &data, &blocked);
    refused(&output, 1, &[path(&blocked)]);
    assert_eq!(
        entries(&d),
        top,
        "a failed write leaves nothing beside FILE"
    );
    assert_eq!(std::fs::read(blocked.join("inside")).unwrap(), b"kept");
    let out = d.join("scores.f32");
    std::fs::write(&out, b"old").unwrap();
    let top = entries(&d);
    let output = infer_scores("cpu", &ckpt, &data, &out);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        text(&output.stderr)
    );
    assert_eq!(entries(&d), top, "a run leaves FILE and nothing else");
    assert_eq!(read_f32(&out).len(), SAMPLES * C);
}

/// infer-function-vulkan:the same run on `--backend vulkan` writes `3·B·C` values, each within
/// the DESIGN.md § 9 gate 5 matmul tolerance of the CPU file: `|g - c| <= 1e-5 + 1e-4 · r`, `r`
/// the sum of the absolute values of the element's `D` products.
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "needs a Vulkan device"]
fn infer_function_vulkan_is_within_tolerance_of_the_cpu_file() {
    let d = dir("vulkan");
    let ckpt = checkpoint(&d);
    let data = data(&d, SAMPLES);
    let cpu_out = d.join("cpu.f32");
    let vulkan_out = d.join("vulkan.f32");
    let cpu = infer_scores("cpu", &ckpt, &data, &cpu_out);
    assert_eq!(cpu.status.code(), Some(0), "{}", text(&cpu.stderr));
    let vulkan = infer_scores("vulkan", &ckpt, &data, &vulkan_out);
    assert_eq!(vulkan.status.code(), Some(0), "{}", text(&vulkan.stderr));
    assert_eq!(
        text(&vulkan.stdout),
        format!(
            "function scores blocks 3 rows {} out {}\n",
            3 * B,
            path(&vulkan_out)
        )
    );
    let c = read_f32(&cpu_out);
    let g = read_f32(&vulkan_out);
    assert_eq!(g.len(), 3 * B * C);
    assert_eq!(c.len(), g.len());
    let x = x_values(SAMPLES);
    for (at, (&g, &c)) in g.iter().zip(&c).enumerate() {
        let (i, j) = (at / C, at % C);
        let r: f32 = (0..D).map(|k| (x[i * D + k] * W[k * C + j]).abs()).sum();
        assert!(
            (g - c).abs() <= 1e-5 + 1e-4 * r,
            "element {at}: vulkan {g}, cpu {c}"
        );
    }
}
