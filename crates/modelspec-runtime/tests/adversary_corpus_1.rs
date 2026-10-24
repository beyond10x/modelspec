//! Adversary cases for `dataset::prepare` (story:bert-corpus-splits; `spec/domains/data.yaml`,
//! `modelspec.data.Dataset`). Every corpus here is original text written for these cases in the
//! line layout of the raw release: a blank line is a single space, headings carry a leading and a
//! trailing space.

use std::fs;
use std::path::{Path, PathBuf};

use modelspec_runtime::dataset::{self, RECORD_FILE, SplitRole};

/// A fresh, empty directory under this crate's build directory, named for the case.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("adversary_corpus_1_{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// A corpus directory `<case>/<dir_name>` holding the two release files.
fn corpus(case: &Path, dir_name: &str, train: &[u8], valid: &[u8]) -> PathBuf {
    let dir = case.join(dir_name);
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("wiki.train.raw"), train).unwrap();
    fs::write(dir.join("wiki.valid.raw"), valid).unwrap();
    dir
}

const VALID: &[u8] = b" \n = Brass Gate = \n \n The brass gate stands at the end of a lane that nobody walks . \n \n";

fn listing(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

/// The UUID the specification states for `dataset_id`, computed here from the bytes of the two
/// files and nothing of the code under test: the leading 16 bytes of blake3 over the 32-byte
/// digests, Train then HeldOut, version nibble 8 and RFC variant.
fn expected_id(train: &[u8], valid: &[u8]) -> String {
    let mut h = blake3::Hasher::new();
    h.update(blake3::hash(train).as_bytes());
    h.update(blake3::hash(valid).as_bytes());
    let d = h.finalize();
    let mut b = [0u8; 16];
    b.copy_from_slice(&d.as_bytes()[..16]);
    b[6] = (b[6] & 0x0f) | 0x80;
    b[8] = (b[8] & 0x3f) | 0x80;
    let x: Vec<String> = b.iter().map(|v| format!("{v:02x}")).collect();
    let x = x.concat();
    format!(
        "{}-{}-{}-{}-{}",
        &x[0..8],
        &x[8..12],
        &x[12..16],
        &x[16..20],
        &x[20..32]
    )
}

// ---- Attack 1: the document rule ----------------------------------------------------------------

/// A key line of a results table, shaped like the ones the release holds inside articles: it
/// starts with `= ` and ends with ` =` once trimmed, with no blank line around it. One article.
#[test]
fn adv_c1_table_key_row_is_not_an_article() {
    let case = scratch("table_key");
    let train = b" \n = Lantern Moor = \n \n Lantern Moor is an invented heath crossed by one road . \n \n = = Results = = \n \n Key : \n = Rounds ; V = \n Victories , D = \n Draws \n \n";
    let dir = corpus(&case, "c", train, VALID);
    let record = dataset::prepare(&dir, &case.join("out")).expect("prepare");
    assert_eq!(
        record.splits[0].document_count, 1,
        "one article; the key row ` = Rounds ; V = ` is not a heading"
    );
}

/// An equation line inside a paragraph, again shaped `= ... =`. One article.
#[test]
fn adv_c1_equation_line_is_not_an_article() {
    let case = scratch("equation");
    let train = b" \n = Quiet Orbit = \n \n The made-up moon keeps its distance , so that r \n = c + k t = \n holds for every hour of the night . \n \n";
    let dir = corpus(&case, "c", train, VALID);
    let record = dataset::prepare(&dir, &case.join("out")).expect("prepare");
    assert_eq!(record.splits[0].document_count, 1);
}

/// The real release, when its directory is named by `MODELSPEC_WIKITEXT103_RAW`: the strict
/// heading rule (D22) counts 28,472 train and 60 validation; the published count is 28,475.
#[test]
#[ignore = "needs the real release: set MODELSPEC_WIKITEXT103_RAW and pass --ignored"]
fn adv_c1_real_release_counts_match_the_strict_heading_rules_counts() {
    let Some(dir) = std::env::var_os("MODELSPEC_WIKITEXT103_RAW") else {
        panic!("set MODELSPEC_WIKITEXT103_RAW to the release directory");
    };
    let out = scratch("real_release");
    let record = dataset::prepare(Path::new(&dir), &out);
    let _ = fs::remove_dir_all(&out);
    let record = record.expect("prepare");
    let counts: Vec<u64> = record.splits.iter().map(|s| s.document_count).collect();
    assert_eq!(counts, [28_472, 60]);
}

/// The heading shapes the release uses, one by one.
#[test]
fn adv_c1_heading_shapes() {
    assert!(dataset::is_article_heading(" = Brass Gate = \n"));
    assert!(dataset::is_article_heading(" = Brass Gate = \r\n"));
    assert!(!dataset::is_article_heading(" = = Repairs = = \n"));
    assert!(!dataset::is_article_heading(
        " = = = Spring repairs = = = \n"
    ));
    assert!(!dataset::is_article_heading(" a = b = \n"));
}

// ---- Attack 2: reproducibility across paths -----------------------------------------------------

#[test]
fn adv_c1_trailing_slash_and_dot_paths_write_identical_records() {
    let case = scratch("paths");
    let train = b" \n = Alder Pond = \n \n A pond . \n";
    let dir = corpus(&case, "corpus", train, VALID);
    let plain = case.join("out_plain");
    dataset::prepare(&dir, &plain).unwrap();
    let slash = PathBuf::from(format!("{}/", dir.display()));
    let dot = dir.join("..").join("corpus").join(".");
    for (i, path) in [slash, dot].iter().enumerate() {
        let out = case.join(format!("out_{i}"));
        dataset::prepare(path, &out).unwrap();
        assert_eq!(
            fs::read(out.join(RECORD_FILE)).unwrap(),
            fs::read(plain.join(RECORD_FILE)).unwrap(),
            "{} writes the record of {}",
            path.display(),
            dir.display()
        );
    }
}

/// `name` is the corpus directory's file name (stated in the specification), so the same bytes
/// under another directory name give the same `dataset_id` and a different `name`.
#[test]
fn adv_c1_directory_name_changes_name_and_not_the_id() {
    let case = scratch("rename");
    let train = b" \n = Alder Pond = \n \n A pond . \n";
    let a = corpus(&case, "first", train, VALID);
    let b = corpus(&case, "second", train, VALID);
    let ra = dataset::prepare(&a, &case.join("oa")).unwrap();
    let rb = dataset::prepare(&b, &case.join("ob")).unwrap();
    assert_eq!(ra.dataset_id, rb.dataset_id);
    assert_eq!((ra.name.as_str(), rb.name.as_str()), ("first", "second"));
}

// ---- Attack 3: line endings and encodings -------------------------------------------------------

fn summary_of(train: &[u8]) -> Result<(u64, u64, String, Vec<u8>), String> {
    let case = scratch(&format!("enc_{}", blake3::hash(train).to_hex()));
    let dir = corpus(&case, "c", train, VALID);
    let out = case.join("out");
    match dataset::prepare(&dir, &out) {
        Ok(r) => {
            let s = &r.splits[0];
            let copy = fs::read(out.join("train.raw")).unwrap();
            Ok((s.document_count, s.byte_count, s.content_hash.clone(), copy))
        }
        Err(e) => {
            if out.exists() {
                assert!(
                    listing(&out).is_empty(),
                    "a refused run leaves {:?}",
                    listing(&out)
                );
            }
            Err(e.to_string())
        }
    }
}

#[test]
fn adv_c1_crlf_bom_and_no_final_newline_keep_bytes_and_counts() {
    let cases: [(&[u8], u64); 4] = [
        (
            b" \r\n = Alder = \r\n \r\n x . \r\n \r\n = Birch = \r\n \r\n y . \r\n",
            2,
        ),
        (
            b"\xef\xbb\xbf = Alder = \n x . \n \n = Birch = \n \n y . \n",
            2,
        ),
        (
            b" = Alder = \n x . \n \n = Birch = \n \n no final newline",
            2,
        ),
        (b" = Alder = ", 1),
    ];
    for (train, documents) in cases {
        let (count, bytes, hash, copy) = summary_of(train).unwrap();
        assert_eq!(count, documents, "{:?}", String::from_utf8_lossy(train));
        assert_eq!(bytes, train.len() as u64);
        assert_eq!(hash, blake3::hash(train).to_hex().to_string());
        assert_eq!(
            blake3::hash(&copy),
            blake3::hash(train),
            "the copy is byte-identical"
        );
    }
}

#[test]
fn adv_c1_empty_blank_and_invalid_files_are_refused_without_leftovers() {
    for (train, reason) in [
        (&b""[..], "holds no document"),
        (&b" \n \n\t\n"[..], "holds no document"),
        (
            &b" = Alder = \n bad \xc3 byte \n"[..],
            "not UTF-8 at byte 17",
        ),
    ] {
        let error = summary_of(train).unwrap_err();
        assert!(
            error.ends_with(&format!("wiki.train.raw: {reason}")),
            "{error}"
        );
    }
}

// ---- Attack 4: the dataset id -------------------------------------------------------------------

#[test]
fn adv_c1_dataset_id_is_blake3_of_train_then_heldout_digests() {
    let case = scratch("id");
    let train = b" \n = Alder Pond = \n \n A pond . \n";
    let dir = corpus(&case, "c", train, VALID);
    let record = dataset::prepare(&dir, &case.join("out")).unwrap();
    assert_eq!(record.dataset_id, expected_id(train, VALID));
    assert_ne!(
        record.dataset_id,
        expected_id(VALID, train),
        "role order matters"
    );
    let id = &record.dataset_id;
    assert_eq!(&id[14..15], "8");
    assert!(matches!(&id[19..20], "8" | "9" | "a" | "b"), "{id}");
}

#[test]
fn adv_c1_valid_split_alone_changes_the_id() {
    let case = scratch("id_valid");
    let train = b" \n = Alder Pond = \n \n A pond . \n";
    let a = corpus(&case, "a", train, VALID);
    let b = corpus(&case, "b", train, b" = Brass Gate = \n Another lane . \n");
    let ra = dataset::prepare(&a, &case.join("oa")).unwrap();
    let rb = dataset::prepare(&b, &case.join("ob")).unwrap();
    assert_ne!(ra.dataset_id, rb.dataset_id);
    assert_eq!(ra.splits[0], rb.splits[0]);
}

// ---- Attack 5: roles are distinct even for identical files --------------------------------------

#[test]
fn adv_c1_identical_files_still_give_one_split_per_role() {
    let case = scratch("same_bytes");
    let dir = corpus(&case, "c", VALID, VALID);
    let record = dataset::prepare(&dir, &case.join("out")).unwrap();
    let roles: Vec<SplitRole> = record.splits.iter().map(|s| s.role).collect();
    assert_eq!(roles, [SplitRole::Train, SplitRole::HeldOut]);
    assert_eq!(record.splits[0].content_hash, record.splits[1].content_hash);
}

// ---- Attack 6: refusal paths --------------------------------------------------------------------

#[test]
fn adv_c1_refused_run_keeps_an_earlier_output_intact() {
    let case = scratch("keep_earlier");
    let train = b" \n = Alder Pond = \n \n A pond . \n";
    let good = corpus(&case, "good", train, VALID);
    let bad = corpus(&case, "bad", b" \n = Birch = \n y . \n", b"\xff\n");
    let out = case.join("out");
    dataset::prepare(&good, &out).unwrap();
    let before: Vec<(String, Vec<u8>)> = listing(&out)
        .into_iter()
        .map(|n| (n.clone(), fs::read(out.join(&n)).unwrap()))
        .collect();
    let error = dataset::prepare(&bad, &out).unwrap_err();
    assert!(error.to_string().contains("wiki.valid.raw"), "{error}");
    let after: Vec<(String, Vec<u8>)> = listing(&out)
        .into_iter()
        .map(|n| (n.clone(), fs::read(out.join(&n)).unwrap()))
        .collect();
    assert_eq!(after, before, "no split, record or partial file changed");
}

#[test]
fn adv_c1_out_that_is_a_file_is_refused_naming_it() {
    let case = scratch("out_file");
    let dir = corpus(&case, "c", VALID, VALID);
    let out = case.join("taken");
    fs::write(&out, b"x").unwrap();
    let error = dataset::prepare(&dir, &out).unwrap_err();
    assert_eq!(error.path, out);
    assert_eq!(fs::read(&out).unwrap(), b"x");
}

#[test]
fn adv_c1_out_equal_to_the_corpus_leaves_the_release_files_alone() {
    let case = scratch("out_is_corpus");
    let train = b" \n = Alder Pond = \n \n A pond . \n";
    let dir = corpus(&case, "c", train, VALID);
    dataset::prepare(&dir, &dir).unwrap();
    assert_eq!(fs::read(dir.join("wiki.train.raw")).unwrap(), train);
    assert_eq!(fs::read(dir.join("wiki.valid.raw")).unwrap(), VALID);
    assert_eq!(
        listing(&dir),
        [
            "dataset.json",
            "heldout.raw",
            "train.raw",
            "wiki.train.raw",
            "wiki.valid.raw"
        ]
    );
}
