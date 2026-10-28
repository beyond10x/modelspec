//! Adversary cases for `modelspec infer --data DIR --function NAME --out FILE`
//! (story:decider-infer-on-data; `spec/domains/data.yaml`, `InferenceOutputFile`).
//!
//! Each case writes its own model under the test's temp directory: a base model with two
//! parameters, `w: [3, 2]` (read by `forward`) and `emb: [5, 2]` (read by no `forward`), and the
//! model function the case attacks. Every parameter and data value is a small multiple of 1/4,
//! so every product and sum below is exact in `f32` and the host computes the CPU's bits.
//!
//! The device cases are ignored; run them with
//! `cargo test -p modelspec-cli --test adversary_infer_data -- --ignored --test-threads=1`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// The base model; `{EXTRA}` is replaced by the functions a case adds, `{DROPOUT}` by a
/// dropout around `forward`'s product or by nothing.
const BASE: &str = "model Probe {
    param w: Tensor<f32, [3, 2]> = xavier_uniform();
    param emb: Tensor<f32, [5, 2]> = xavier_uniform();

    fn forward(x: Tensor<f32, [4, 3]>, y: Tensor<f32, []>) -> Tensor<f32, []> {
        let d = sub({DROPOUT_OPEN}matmul(x, w){DROPOUT_CLOSE}, y);
        return mean(mul(d, d));
    }

    fn scores(x: Tensor<f32, [4, 3]>) -> Tensor<f32, [4, 2]> {
        return matmul(x, w);
    }
{EXTRA}
}

train Fit {
    model = Probe;
    input x: Tensor<f32, [4, 3]>;
    target y: Tensor<f32, []>;
    loss = model.forward(x, y);
    optimizer = sgd(lr = 0.1);
    steps = 5;
    seed = 7;
    device = cpu;
}
";

const W: [f32; 6] = [0.5, -1.25, 2.0, 0.75, -0.5, 1.5];
const EMB: [f32; 10] = [0.25, -0.5, 1.0, 1.75, -1.5, 0.0, 0.75, -2.0, 1.25, 0.5];
const D: usize = 3;
const C: usize = 2;
const B: usize = 4;
const ROWS_EMB: usize = 5;

fn source(extra: &str) -> String {
    BASE.replace("{EXTRA}", extra)
        .replace("{DROPOUT_OPEN}", "")
        .replace("{DROPOUT_CLOSE}", "")
}

fn source_with_dropout(extra: &str) -> String {
    BASE.replace("{EXTRA}", extra)
        .replace("{DROPOUT_OPEN}", "dropout(")
        .replace("{DROPOUT_CLOSE}", ", p = 0.5)")
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("adversary_infer_data")
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

fn write_i32(path: &Path, values: &[i32]) {
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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The model file of `src` under `d`.
fn model(d: &Path, src: &str) -> PathBuf {
    let file = d.join("probe.mspec");
    std::fs::write(&file, src).expect("the model file is writable");
    file
}

/// The checkpoint of `file` after `steps` steps from `w` = [`W`], `emb` = [`EMB`].
fn checkpoint_after(d: &Path, file: &Path, steps: u32) -> PathBuf {
    let init = d.join("init");
    std::fs::create_dir_all(&init).expect("the init dir is writable");
    write_f32(&init.join("w.f32"), &W);
    write_f32(&init.join("emb.f32"), &EMB);
    let ckpt = d.join(format!("probe-{steps}.msckpt"));
    ok(&[
        "train",
        path(file),
        "--backend",
        "cpu",
        "--steps",
        &steps.to_string(),
        "--init",
        path(&init),
        "--checkpoint",
        path(&ckpt),
    ]);
    ckpt
}

fn checkpoint(d: &Path, file: &Path) -> PathBuf {
    checkpoint_after(d, file, 0)
}

/// `samples` samples of `width` values each, multiples of 1/4 in `[-2, 2]`, with `salt`.
fn quarters(samples: usize, width: usize, salt: usize) -> Vec<f32> {
    (0..samples * width)
        .map(|i| ((i * 7 + salt) % 17) as f32 * 0.25 - 2.0)
        .collect()
}

fn ids(samples: usize) -> Vec<i32> {
    (0..samples)
        .map(|i| ((i * 3 + 1) % ROWS_EMB) as i32)
        .collect()
}

/// `matmul(x, W)` on the host, one row per sample.
fn host_scores(x: &[f32]) -> Vec<f32> {
    let samples = x.len() / D;
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

/// `add(matmul(x, W), embedding(EMB, ids))` on the host.
fn host_mixed(x: &[f32], ids: &[i32]) -> Vec<f32> {
    let scores = host_scores(x);
    scores
        .iter()
        .enumerate()
        .map(|(at, s)| s + EMB[ids[at / C] as usize * C + at % C])
        .collect()
}

fn infer(
    backend: &str,
    file: &Path,
    ckpt: &Path,
    data: &Path,
    function: &str,
    out: &Path,
) -> Output {
    modelspec(&[
        "infer",
        path(file),
        "--backend",
        backend,
        "--checkpoint",
        path(ckpt),
        "--data",
        path(data),
        "--function",
        function,
        "--out",
        path(out),
    ])
}

fn succeeded(output: &Output) {
    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout: {}\nstderr: {}",
        text(&output.stdout),
        text(&output.stderr)
    );
}

/// The first diagnostic code `E-XXX-NNN` on stderr.
fn code(output: &Output) -> Option<String> {
    let stderr = text(&output.stderr);
    stderr
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
        .find(|word| {
            let parts: Vec<&str> = word.split('-').collect();
            parts.len() == 3
                && parts[0] == "E"
                && parts[1].chars().all(|c| c.is_ascii_uppercase())
                && parts[2].len() == 3
                && parts[2].chars().all(|c| c.is_ascii_digit())
        })
        .map(str::to_owned)
}

const MIXED: &str = "
    fn mixed(x: Tensor<f32, [4, 3]>, ids: Tensor<i32, [4]>) -> Tensor<f32, [4, 2]> {
        return add(matmul(x, w), embedding(emb, ids));
    }
";

/// A directory of `blocks` blocks for `mixed`: `x.f32`, `ids.i32`, and a decoy `extra.f32` of
/// another sample count the function does not take.
fn mixed_data(d: &Path, blocks: usize) -> (PathBuf, Vec<f32>, Vec<i32>) {
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let x = quarters(blocks * B, D, 3);
    let ids = ids(blocks * B);
    write_f32(&data.join("x.f32"), &x);
    write_i32(&data.join("ids.i32"), &ids);
    write_f32(&data.join("extra.f32"), &quarters(7, 1, 0));
    (data, x, ids)
}

/// Row order and alignment, parameter sharing: `f32` and `i32` arguments mixed, a parameter
/// `forward` never reads, a directory holding a file the function does not take; five blocks,
/// bit for bit in block order, the file exactly rows x row_elements x 4 bytes.
#[test]
fn mixed_arguments_and_a_parameter_forward_never_reads_are_exact_in_block_order() {
    let d = dir("mixed");
    let file = model(&d, &source(MIXED));
    let ckpt = checkpoint(&d, &file);
    let (data, x, ids) = mixed_data(&d, 5);
    let out = d.join("mixed.f32");
    let output = infer("cpu", &file, &ckpt, &data, "mixed", &out);
    succeeded(&output);
    assert_eq!(
        text(&output.stdout),
        format!("function mixed blocks 5 rows 20 out {}\n", path(&out))
    );
    assert_eq!(std::fs::metadata(&out).unwrap().len(), (20 * C * 4) as u64);
    assert_eq!(bits(&read_f32(&out)), bits(&host_mixed(&x, &ids)));
}

/// A rank-1 result `[B]` is one value per sample.
#[test]
fn a_rank_1_result_is_one_value_per_sample() {
    let d = dir("rank1");
    let file = model(
        &d,
        &source(
            "
    fn r1(v: Tensor<f32, [4]>) -> Tensor<f32, [4]> {
        return relu(v);
    }
",
        ),
    );
    let ckpt = checkpoint(&d, &file);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let v = quarters(12, 1, 5);
    write_f32(&data.join("v.f32"), &v);
    let out = d.join("r1.f32");
    let output = infer("cpu", &file, &ckpt, &data, "r1", &out);
    succeeded(&output);
    assert_eq!(
        text(&output.stdout),
        format!("function r1 blocks 3 rows 12 out {}\n", path(&out))
    );
    assert_eq!(std::fs::metadata(&out).unwrap().len(), 12 * 4);
    let expected: Vec<f32> = v.iter().map(|a| a.max(0.0)).collect();
    assert_eq!(bits(&read_f32(&out)), bits(&expected));
}

/// A rank-3 result `[B, 2, 3]` is one row of six values per sample, the sample's own.
#[test]
fn a_rank_3_result_is_one_row_of_its_trailing_elements_per_sample() {
    let d = dir("rank3");
    let file = model(
        &d,
        &source(
            "
    fn r3(t: Tensor<f32, [4, 2, 3]>) -> Tensor<f32, [4, 2, 3]> {
        return mul(t, t);
    }
",
        ),
    );
    let ckpt = checkpoint(&d, &file);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let t = quarters(8, 6, 1);
    write_f32(&data.join("t.f32"), &t);
    let out = d.join("r3.f32");
    let output = infer("cpu", &file, &ckpt, &data, "r3", &out);
    succeeded(&output);
    assert_eq!(
        text(&output.stdout),
        format!("function r3 blocks 2 rows 8 out {}\n", path(&out))
    );
    assert_eq!(std::fs::metadata(&out).unwrap().len(), 8 * 6 * 4);
    let expected: Vec<f32> = t.iter().map(|a| a * a).collect();
    assert_eq!(bits(&read_f32(&out)), bits(&expected));
}

const IDENT: &str = "
    fn ident(x: Tensor<f32, [4, 3]>) -> Tensor<f32, [4, 3]> {
        return x;
    }
";

/// A function returning its argument writes the argument back, byte for byte.
#[test]
fn a_function_returning_its_argument_writes_the_argument() {
    let d = dir("ident");
    let file = model(&d, &source(IDENT));
    let ckpt = checkpoint(&d, &file);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    write_f32(&data.join("x.f32"), &quarters(8, D, 2));
    let out = d.join("ident.f32");
    let output = infer("cpu", &file, &ckpt, &data, "ident", &out);
    succeeded(&output);
    assert_eq!(
        std::fs::read(&out).unwrap(),
        std::fs::read(data.join("x.f32")).unwrap()
    );
}

const NOISY: &str = "
    fn noisy(x: Tensor<f32, [4, 3]>) -> Tensor<f32, [4, 2]> {
        return dropout(matmul(x, w), p = 0.5);
    }
";

/// Dropout in the function is off: on a checkpoint written after three steps of a run whose
/// `forward` draws dropout masks, `noisy` writes exactly what `scores` writes, twice alike.
#[test]
fn dropout_in_the_function_is_off_on_a_trained_checkpoint() {
    let d = dir("noisy");
    let file = model(&d, &source_with_dropout(NOISY));
    let ckpt = checkpoint_after(&d, &file, 3);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    write_f32(&data.join("x.f32"), &quarters(12, D, 4));
    let (a, b, s) = (d.join("a.f32"), d.join("b.f32"), d.join("s.f32"));
    succeeded(&infer("cpu", &file, &ckpt, &data, "noisy", &a));
    succeeded(&infer("cpu", &file, &ckpt, &data, "noisy", &b));
    succeeded(&infer("cpu", &file, &ckpt, &data, "scores", &s));
    let (a, b, s) = (read_f32(&a), read_f32(&b), read_f32(&s));
    assert_eq!(bits(&a), bits(&b), "two runs differ");
    assert_eq!(bits(&a), bits(&s), "dropout was on: {a:?} vs {s:?}");
}

const PARTIAL: &str = "
    fn partial(x: Tensor<f32, [4, 3]>, z: Tensor<f32, [4, 3]>) -> Tensor<f32, [4, 2]> {
        return matmul(x, w);
    }
";

/// An argument the function does not read is still bound by its file, and does not shift the
/// rows of the one it reads.
#[test]
fn an_argument_the_function_does_not_read_leaves_the_result_exact() {
    let d = dir("partial");
    let file = model(&d, &source(PARTIAL));
    let ckpt = checkpoint(&d, &file);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let x = quarters(12, D, 6);
    write_f32(&data.join("x.f32"), &x);
    write_f32(&data.join("z.f32"), &quarters(12, D, 9));
    let out = d.join("partial.f32");
    let output = infer("cpu", &file, &ckpt, &data, "partial", &out);
    succeeded(&output);
    assert_eq!(bits(&read_f32(&out)), bits(&host_scores(&x)));
}

const NO_ARGS: &str = "
    fn table() -> Tensor<f32, [5, 2]> {
        return relu(emb);
    }
";

/// A function that takes no argument is E-DATA-010 (`spec/domains/data.yaml`, E-DATA-010 in
/// the diagnostics page: "one that takes no argument"); nothing is written.
#[test]
fn a_function_without_arguments_is_e_data_010() {
    let d = dir("noargs");
    let file = model(&d, &source(NO_ARGS));
    let ckpt = checkpoint(&d, &file);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    write_f32(&data.join("x.f32"), &quarters(4, D, 0));
    let out = d.join("table.f32");
    let output = infer("cpu", &file, &ckpt, &data, "table", &out);
    assert_eq!(output.status.code(), Some(1), "{}", text(&output.stderr));
    assert_eq!(
        code(&output).as_deref(),
        Some("E-DATA-010"),
        "{}",
        text(&output.stderr)
    );
    assert!(!out.exists());
}

const TOTAL: &str = "
    fn total(x: Tensor<f32, [4, 3]>) -> Tensor<f32, []> {
        return sum(matmul(x, w));
    }
";

/// A rank-0 result when `B` is 4 is not one row per sample: E-DATA-010, and per
/// `InferenceOutputFile` the refusal names the model's functions.
#[test]
fn a_rank_0_result_with_b_above_1_is_e_data_010_naming_the_functions() {
    let d = dir("total");
    let file = model(&d, &source(TOTAL));
    let ckpt = checkpoint(&d, &file);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    write_f32(&data.join("x.f32"), &quarters(8, D, 0));
    let out = d.join("total.f32");
    let output = infer("cpu", &file, &ckpt, &data, "total", &out);
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert_eq!(code(&output).as_deref(), Some("E-DATA-010"), "{stderr}");
    assert!(!out.exists());
    for function in ["forward", "scores", "total"] {
        assert!(
            stderr.contains(function),
            "the refusal does not name `{function}`: {stderr}"
        );
    }
}

const COLS: &str = "
    fn cols(x: Tensor<f32, [4, 3]>) -> Tensor<f32, [3, 4]> {
        return transpose(x);
    }
";

/// A result whose leading dimension is not `B` is E-DATA-010; nothing is written.
#[test]
fn a_result_whose_leading_dimension_is_not_b_is_e_data_010() {
    let d = dir("cols");
    let file = model(&d, &source(COLS));
    let ckpt = checkpoint(&d, &file);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    write_f32(&data.join("x.f32"), &quarters(8, D, 0));
    let out = d.join("cols.f32");
    let output = infer("cpu", &file, &ckpt, &data, "cols", &out);
    assert_eq!(output.status.code(), Some(1), "{}", text(&output.stderr));
    assert_eq!(
        code(&output).as_deref(),
        Some("E-DATA-010"),
        "{}",
        text(&output.stderr)
    );
    assert!(!out.exists());
}

/// Argument files of different block counts are E-DATA-007; nothing is written.
#[test]
fn argument_files_of_different_block_counts_are_e_data_007() {
    let d = dir("counts");
    let file = model(&d, &source(MIXED));
    let ckpt = checkpoint(&d, &file);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    write_f32(&data.join("x.f32"), &quarters(5 * B, D, 0));
    write_i32(&data.join("ids.i32"), &ids(4 * B));
    let out = d.join("mixed.f32");
    let output = infer("cpu", &file, &ckpt, &data, "mixed", &out);
    assert_eq!(output.status.code(), Some(1), "{}", text(&output.stderr));
    assert_eq!(
        code(&output).as_deref(),
        Some("E-DATA-007"),
        "{}",
        text(&output.stderr)
    );
    assert!(!out.exists());
}

/// `forward` itself, whose arguments share no leading dimension (`y` is a scalar), so `B` is 1
/// and a sample is a whole block: one row, the block's loss, per sample.
#[test]
fn forward_as_the_function_writes_one_loss_per_sample_when_b_is_1() {
    let d = dir("forward");
    let file = model(&d, &source(""));
    let ckpt = checkpoint(&d, &file);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let x = quarters(3 * B, D, 8);
    let y = [0.25f32, -1.0, 1.5];
    write_f32(&data.join("x.f32"), &x);
    write_f32(&data.join("y.f32"), &y);
    let out = d.join("forward.f32");
    let output = infer("cpu", &file, &ckpt, &data, "forward", &out);
    succeeded(&output);
    assert_eq!(
        text(&output.stdout),
        format!("function forward blocks 3 rows 3 out {}\n", path(&out))
    );
    let s = host_scores(&x);
    let expected: Vec<f32> = (0..3)
        .map(|blk| {
            let sum: f64 = s[blk * B * C..(blk + 1) * B * C]
                .iter()
                .map(|&v| {
                    let e = f64::from(v - y[blk]);
                    e * e
                })
                .sum();
            (sum / (B * C) as f64) as f32
        })
        .collect();
    assert_eq!(bits(&read_f32(&out)), bits(&expected));
}

/// `--out` naming an input file of the directory: the run would overwrite the data it read.
/// It is refused and the input is kept.
#[test]
fn out_naming_an_input_file_is_refused_and_the_input_kept() {
    let d = dir("outinput");
    let file = model(&d, &source(""));
    let ckpt = checkpoint(&d, &file);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let x = quarters(3 * B, D, 0);
    write_f32(&data.join("x.f32"), &x);
    let before = std::fs::read(data.join("x.f32")).unwrap();
    let output = infer("cpu", &file, &ckpt, &data, "scores", &data.join("x.f32"));
    assert_eq!(
        std::fs::read(data.join("x.f32")).unwrap(),
        before,
        "the input file was overwritten; exit {:?}, stdout {}",
        output.status.code(),
        text(&output.stdout)
    );
    assert_eq!(output.status.code(), Some(1));
}

/// `--out` in a directory that does not exist: exit 1, nothing on stdout, nothing created.
#[test]
fn out_in_a_missing_directory_exits_1_and_creates_nothing() {
    let d = dir("outmissing");
    let file = model(&d, &source(""));
    let ckpt = checkpoint(&d, &file);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    write_f32(&data.join("x.f32"), &quarters(3 * B, D, 0));
    let out = d.join("absent").join("scores.f32");
    let output = infer("cpu", &file, &ckpt, &data, "scores", &out);
    assert_eq!(output.status.code(), Some(1), "{}", text(&output.stderr));
    assert_eq!(text(&output.stdout), "");
    assert!(!d.join("absent").exists());
}

/// An existing `--out` is replaced only on success: when the write fails part of the way (the
/// file-size limit of the process stands in for a full disk or a quota), the old file is kept.
#[test]
fn an_existing_out_survives_a_write_that_fails_part_of_the_way() {
    let d = dir("outpartial");
    let file = model(&d, &source(""));
    let ckpt = checkpoint(&d, &file);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    write_f32(&data.join("x.f32"), &quarters(400 * B, D, 0));
    let out = d.join("scores.f32");
    std::fs::write(&out, b"kept").unwrap();
    let output = Command::new("sh")
        .arg("-c")
        .arg("ulimit -c 0; ulimit -f 2; exec \"$0\" \"$@\"")
        .arg(env!("CARGO_BIN_EXE_modelspec"))
        .args([
            "infer",
            path(&file),
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
        ])
        .current_dir(repo_root())
        .output()
        .expect("sh runs");
    assert_ne!(output.status.code(), Some(0), "the write was meant to fail");
    let now = std::fs::read(&out).unwrap();
    assert_eq!(
        now,
        b"kept",
        "the old file was replaced by {} bytes of a failed write; status {:?}, stderr {}",
        now.len(),
        output.status,
        text(&output.stderr)
    );
}

/// The checkpoint of another program (one parameter `w`, no `emb`) is refused with
/// E-CKPT-003; nothing is written.
#[test]
fn a_checkpoint_of_another_program_is_e_ckpt_003() {
    let d = dir("otherckpt");
    let other = model(
        &d,
        "model Other {
    param w: Tensor<f32, [3, 2]> = xavier_uniform();

    fn forward(x: Tensor<f32, [4, 3]>, y: Tensor<f32, []>) -> Tensor<f32, []> {
        let d = sub(matmul(x, w), y);
        return mean(mul(d, d));
    }
}

train Fit {
    model = Other;
    input x: Tensor<f32, [4, 3]>;
    target y: Tensor<f32, []>;
    loss = model.forward(x, y);
    optimizer = sgd(lr = 0.1);
    steps = 5;
    seed = 7;
    device = cpu;
}
",
    );
    let ckpt = d.join("other.msckpt");
    ok(&[
        "train",
        path(&other),
        "--backend",
        "cpu",
        "--steps",
        "0",
        "--checkpoint",
        path(&ckpt),
    ]);
    let file = d.join("probe-main.mspec");
    std::fs::write(&file, source("")).unwrap();
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    write_f32(&data.join("x.f32"), &quarters(3 * B, D, 0));
    let out = d.join("scores.f32");
    let output = infer("cpu", &file, &ckpt, &data, "scores", &out);
    assert_eq!(output.status.code(), Some(1), "{}", text(&output.stderr));
    assert_eq!(
        code(&output).as_deref(),
        Some("E-CKPT-003"),
        "{}",
        text(&output.stderr)
    );
    assert!(!out.exists());
}

// Device cases: CPU against Vulkan on the same checkpoint and directory.

#[cfg(feature = "vulkan")]
fn close(g: &[f32], c: &[f32]) {
    assert_eq!(g.len(), c.len(), "row counts differ");
    for (at, (&g, &c)) in g.iter().zip(c).enumerate() {
        assert!(
            (g - c).abs() <= 1e-4 + 1e-4 * c.abs(),
            "element {at}: vulkan {g}, cpu {c}"
        );
    }
}

/// Runs `function` on both backends and checks they agree, the stdout lines differing only in
/// the file name.
#[cfg(feature = "vulkan")]
fn both(d: &Path, file: &Path, ckpt: &Path, data: &Path, function: &str) -> Vec<f32> {
    let (c, g) = (d.join("cpu.f32"), d.join("vulkan.f32"));
    let cpu = infer("cpu", file, ckpt, data, function, &c);
    succeeded(&cpu);
    let vk = infer("vulkan", file, ckpt, data, function, &g);
    succeeded(&vk);
    assert_eq!(
        text(&vk.stdout).replace(path(&g), "FILE"),
        text(&cpu.stdout).replace(path(&c), "FILE")
    );
    let (cv, gv) = (read_f32(&c), read_f32(&g));
    close(&gv, &cv);
    gv
}

/// `mixed` over forty blocks on Vulkan: same rows, in block order, within tolerance of the
/// host and the CPU file.
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_mixed_arguments_forty_blocks_match_the_cpu() {
    let d = dir("vk_mixed");
    let file = model(&d, &source(MIXED));
    let ckpt = checkpoint(&d, &file);
    let (data, x, ids) = mixed_data(&d, 40);
    let g = both(&d, &file, &ckpt, &data, "mixed");
    close(&g, &host_mixed(&x, &ids));
}

/// A function that reads no parameter, rank 1 and rank 3, on Vulkan.
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_functions_without_parameters_rank_1_and_rank_3_match_the_cpu() {
    let d = dir("vk_noparams");
    let file = model(
        &d,
        &source(
            "
    fn r1(v: Tensor<f32, [4]>) -> Tensor<f32, [4]> {
        return relu(v);
    }

    fn r3(t: Tensor<f32, [4, 2, 3]>) -> Tensor<f32, [4, 2, 3]> {
        return mul(t, t);
    }
",
        ),
    );
    let ckpt = checkpoint(&d, &file);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let v = quarters(12, 1, 5);
    let t = quarters(12, 6, 1);
    write_f32(&data.join("v.f32"), &v);
    write_f32(&data.join("t.f32"), &t);
    let g = both(&d, &file, &ckpt, &data, "r1");
    close(&g, &v.iter().map(|a| a.max(0.0)).collect::<Vec<_>>());
    let g = both(&d, &file, &ckpt, &data, "r3");
    close(&g, &t.iter().map(|a| a * a).collect::<Vec<_>>());
}

/// A function returning its argument, on Vulkan.
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_a_function_returning_its_argument_matches_the_cpu() {
    let d = dir("vk_ident");
    let file = model(&d, &source(IDENT));
    let ckpt = checkpoint(&d, &file);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let x = quarters(8, D, 2);
    write_f32(&data.join("x.f32"), &x);
    let g = both(&d, &file, &ckpt, &data, "ident");
    close(&g, &x);
}

/// `forward` with `B` = 1, on Vulkan.
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_forward_with_b_1_matches_the_cpu() {
    let d = dir("vk_forward");
    let file = model(&d, &source(""));
    let ckpt = checkpoint(&d, &file);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    write_f32(&data.join("x.f32"), &quarters(3 * B, D, 8));
    write_f32(&data.join("y.f32"), &[0.25, -1.0, 1.5]);
    both(&d, &file, &ckpt, &data, "forward");
}

/// Dropout off on Vulkan: `noisy` within tolerance of `scores` on a trained checkpoint.
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_dropout_in_the_function_is_off_on_a_trained_checkpoint() {
    let d = dir("vk_noisy");
    let file = model(&d, &source_with_dropout(NOISY));
    let ckpt = checkpoint_after(&d, &file, 3);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    write_f32(&data.join("x.f32"), &quarters(12, D, 4));
    let noisy = both(&d, &file, &ckpt, &data, "noisy");
    let s = d.join("s.f32");
    succeeded(&infer("vulkan", &file, &ckpt, &data, "scores", &s));
    close(&noisy, &read_f32(&s));
}

/// An argument the function does not read, on Vulkan.
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_an_argument_the_function_does_not_read_matches_the_cpu() {
    let d = dir("vk_partial");
    let file = model(&d, &source(PARTIAL));
    let ckpt = checkpoint(&d, &file);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let x = quarters(12, D, 6);
    write_f32(&data.join("x.f32"), &x);
    write_f32(&data.join("z.f32"), &quarters(12, D, 9));
    let g = both(&d, &file, &ckpt, &data, "partial");
    close(&g, &host_scores(&x));
}

/// Every refusal of the CPU run is the Vulkan run's, with the same code and exit status, and
/// neither writes the file.
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_refuses_what_the_cpu_refuses_with_the_same_codes() {
    let d = dir("vk_refusals");
    let all = format!("{MIXED}{NO_ARGS}{TOTAL}{COLS}");
    let file = model(&d, &source(&all));
    let ckpt = checkpoint(&d, &file);
    let good = d.join("good");
    std::fs::create_dir_all(&good).unwrap();
    write_f32(&good.join("x.f32"), &quarters(2 * B, D, 0));
    write_i32(&good.join("ids.i32"), &ids(2 * B));
    let partial_block = d.join("partial_block");
    std::fs::create_dir_all(&partial_block).unwrap();
    write_f32(&partial_block.join("x.f32"), &quarters(2 * B + 1, D, 0));
    write_i32(&partial_block.join("ids.i32"), &ids(2 * B + 1));
    let missing = d.join("missing");
    std::fs::create_dir_all(&missing).unwrap();
    write_f32(&missing.join("x.f32"), &quarters(2 * B, D, 0));
    let cases: [(&str, &Path, &str); 6] = [
        ("logits", &good, "E-DATA-010"),
        ("mixed", &partial_block, "E-DATA-009"),
        ("mixed", &missing, "E-DATA-007"),
        ("table", &good, "E-DATA-010"),
        ("total", &good, "E-DATA-010"),
        ("cols", &good, "E-DATA-010"),
    ];
    let mut differ = Vec::new();
    for (function, data, expected) in cases {
        let (c, g) = (d.join("cpu.f32"), d.join("vulkan.f32"));
        let cpu = infer("cpu", &file, &ckpt, data, function, &c);
        let vk = infer("vulkan", &file, &ckpt, data, function, &g);
        let row = (
            function,
            expected,
            cpu.status.code(),
            code(&cpu),
            vk.status.code(),
            code(&vk),
            c.exists(),
            g.exists(),
        );
        if row.2 != Some(1)
            || row.3.as_deref() != Some(expected)
            || row.4 != Some(1)
            || row.5.as_deref() != Some(expected)
            || row.6
            || row.7
        {
            differ.push(format!("{row:?}\n  vulkan stderr: {}", text(&vk.stderr)));
        }
    }
    assert!(differ.is_empty(), "{}", differ.join("\n"));
}

/// `InferenceOutputFile`: "the checker refuses a function that returns I32, E-TYPE-018".
#[test]
fn a_function_returning_i32_is_refused_with_e_type_018() {
    let d = dir("i32result");
    let file = model(
        &d,
        &source(
            "
    fn passthrough(ids: Tensor<i32, [4]>) -> Tensor<i32, [4]> {
        return ids;
    }
",
        ),
    );
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    write_i32(&data.join("ids.i32"), &ids(8));
    let out = d.join("ids.out");
    let output = infer(
        "cpu",
        &file,
        &d.join("none.msckpt"),
        &data,
        "passthrough",
        &out,
    );
    assert_eq!(output.status.code(), Some(1), "{}", text(&output.stderr));
    assert_eq!(
        code(&output).as_deref(),
        Some("E-TYPE-018"),
        "{}",
        text(&output.stderr)
    );
    assert!(!out.exists());
}

/// `--data` and `--function` without `--out` is a usage error (exit 2), as the module
/// documentation says of `--data` without both.
#[test]
fn data_without_out_is_a_usage_error() {
    let d = dir("noout");
    let file = model(&d, &source(""));
    let ckpt = checkpoint(&d, &file);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    write_f32(&data.join("x.f32"), &quarters(3 * B, D, 0));
    for extra in [&["--function", "scores"][..], &[][..]] {
        let mut args = vec![
            "infer",
            path(&file),
            "--backend",
            "cpu",
            "--checkpoint",
            path(&ckpt),
            "--data",
            path(&data),
        ];
        args.extend_from_slice(extra);
        let output = modelspec(&args);
        assert_eq!(
            output.status.code(),
            Some(2),
            "{extra:?}: {}",
            text(&output.stderr)
        );
        assert_eq!(text(&output.stdout), "");
    }
}

/// A deeper function with its own batch size (`B` = 8 where the train block has 4), mixed
/// arguments, `gelu` and `softmax`, over 25 blocks on Vulkan against the CPU.
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "needs a Vulkan device"]
fn vulkan_a_deep_function_with_its_own_batch_size_matches_the_cpu() {
    let d = dir("vk_deep");
    let file = model(
        &d,
        &source(
            "
    fn deep(x: Tensor<f32, [8, 3]>, ids: Tensor<i32, [8]>) -> Tensor<f32, [8, 2]> {
        let h = add(matmul(x, w), embedding(emb, ids));
        return softmax(gelu(h));
    }
",
        ),
    );
    let ckpt = checkpoint_after(&d, &file, 2);
    let data = d.join("data");
    std::fs::create_dir_all(&data).unwrap();
    write_f32(&data.join("x.f32"), &quarters(25 * 8, D, 11));
    write_i32(&data.join("ids.i32"), &ids(25 * 8));
    let g = both(&d, &file, &ckpt, &data, "deep");
    assert_eq!(g.len(), 25 * 8 * C);
    for row in g.chunks(C) {
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "a softmax row sums to {sum}");
    }
}
