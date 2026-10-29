//! `modelspec diagram` (DESIGN.md § 13.7, story:diagram-cli): any level of any view as SVG, as the
//! `DiagramRender` JSON or as the `DiagramGraph` JSON, and every level written to a directory.
//!
//! The `train`, `eval` and `infer` graphs are built by story:diagram-pipelines and composites by
//! story:diagram-zoom; until they land, the cases that need them are ignored and the three views
//! refuse with exit 1.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use modelspec_core::compile;
use modelspec_diagram::layout::{RenderRequest, render};
use modelspec_diagram::model::DiagramGraph;
use modelspec_diagram::{graph, model_graph, svg};

const TWO_LAYER: &str = "crates/modelspec-docs/models/two_layer_relu.mspec";
const LINEAR: &str = "examples/linear_regression.mspec";
const SHAPE_ERROR: &str = "tests/language/check/E-TYPE-003-shape-mismatch.mspec";
const VIEWS: [&str; 4] = ["model", "train", "eval", "infer"];
const PIPELINES: [&str; 3] = ["train", "eval", "infer"];
const FORMATS: [&str; 3] = ["svg", "json", "graph"];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Runs `modelspec <args>` from the repository root.
fn modelspec(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_modelspec"))
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("modelspec runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("output is UTF-8")
}

fn source(file: &str) -> String {
    std::fs::read_to_string(repo_root().join(file)).expect("the file exists")
}

/// A fresh empty directory under the test target's own temporary directory.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("diagram_cli-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the scratch directory is created");
    dir
}

fn validator(entity: &str) -> jsonschema::Validator {
    let path = repo_root().join(format!(
        "crates/modelspec-diagram/schema/entities/modelspec.diagram.{entity}.schema.json"
    ));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let schema: serde_json::Value = serde_json::from_str(&text).unwrap();
    jsonschema::validator_for(&schema).expect("the schema compiles")
}

fn assert_valid(validator: &jsonschema::Validator, json: &str, what: &str) {
    let value: serde_json::Value =
        serde_json::from_str(json).unwrap_or_else(|e| panic!("{what}: not JSON: {e}"));
    let errors: Vec<String> = validator
        .iter_errors(&value)
        .map(|e| format!("{e} at {}", e.instance_path()))
        .collect();
    assert!(errors.is_empty(), "{what}: {errors:#?}");
}

/// Every view in every format exits 0, and the JSON validates against its schema.
fn check_views_and_formats(views: &[&str]) {
    let graphs = validator("DiagramGraph");
    let renders = validator("DiagramRender");
    for view in views {
        for format in FORMATS {
            let out = modelspec(&["diagram", LINEAR, "--view", view, "--format", format]);
            let what = format!("--view {view} --format {format}");
            assert_eq!(out.status.code(), Some(0), "{what}: {}", text(&out.stderr));
            assert!(out.stderr.is_empty(), "{what}: {}", text(&out.stderr));
            let stdout = text(&out.stdout);
            match format {
                "svg" => assert!(stdout.starts_with("<svg"), "{what}: {stdout}"),
                "json" => assert_valid(&renders, &stdout, &what),
                _ => {
                    assert_valid(&graphs, &stdout, &what);
                    let graph: DiagramGraph = serde_json::from_str(&stdout).unwrap();
                    assert_eq!(graph.view.name(), *view, "{what}");
                }
            }
        }
    }
}

/// Writes every level of `view` of `file` into a fresh directory and checks it: one file per scope
/// of the graph, and every composite link naming a file there.
fn check_out_dir(file: &str, view: &str, name: &str) {
    let dir = scratch(name);
    let dir_text = dir.display().to_string();
    let out = modelspec(&["diagram", file, "--view", view, "--out", &dir_text]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let graph_out = modelspec(&["diagram", file, "--view", view, "--format", "graph"]);
    let graph: DiagramGraph = serde_json::from_slice(&graph_out.stdout).expect("graph JSON");
    let expected: BTreeSet<String> = graph
        .scopes
        .iter()
        .map(|scope| format!("{}.svg", scope.key.0))
        .collect();
    let written: BTreeSet<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(written, expected, "one file per scope of the {view} graph");
    for name in &written {
        let svg = std::fs::read_to_string(dir.join(name)).unwrap();
        assert!(svg.starts_with("<svg"), "{name}");
        for href in svg.split("<a href=\"").skip(1) {
            let target = &href[..href.find('"').expect("a closed href")];
            assert!(
                written.contains(target),
                "{name}: the link {target} names no file of {written:?}"
            );
        }
    }
}

/// diagram_cli.svg-default
#[test]
fn svg_default_is_the_model_view_at_depth_1() {
    let out = modelspec(&["diagram", TWO_LAYER]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert!(out.stderr.is_empty(), "{}", text(&out.stderr));
    let source = source(TWO_LAYER);
    let graph = model_graph(TWO_LAYER, &source).expect("two_layer_relu compiles");
    let lines = graph::ir_lines(&compile::compile_training(TWO_LAYER, &source).unwrap());
    let drawn = render(
        &graph,
        &RenderRequest {
            root: None,
            depth: Some(1),
            expanded: Vec::new(),
        },
    );
    assert_eq!(text(&out.stdout), svg::svg(&graph, &drawn, &lines));
}

/// diagram_cli.views-and-formats, the model view.
#[test]
fn model_view_in_every_format() {
    check_views_and_formats(&["model"]);
}

/// diagram_cli.views-and-formats, all four views.
#[test]
#[ignore = "the train, eval and infer graphs arrive with story:diagram-pipelines"]
fn views_and_formats() {
    check_views_and_formats(&VIEWS);
}

/// Until story:diagram-pipelines, a pipeline view is refused on one line, exit 1.
#[test]
fn pipeline_views_refuse_until_they_are_built() {
    for view in PIPELINES {
        let out = modelspec(&["diagram", LINEAR, "--view", view]);
        assert_eq!(out.status.code(), Some(1), "--view {view}");
        assert!(out.stdout.is_empty(), "--view {view}");
        let stderr = text(&out.stderr);
        assert_eq!(stderr.lines().count(), 1, "{stderr}");
        assert!(stderr.contains(&format!("the {view} view")), "{stderr}");
    }
}

/// diagram_cli.out-dir, the model view.
#[test]
fn out_dir_model() {
    check_out_dir(TWO_LAYER, "model", "out-model");
}

/// diagram_cli.out-dir
#[test]
#[ignore = "the train graph arrives with story:diagram-pipelines, its composites with story:diagram-zoom"]
fn out_dir() {
    check_out_dir(LINEAR, "train", "out-train");
}

/// `--out` writes SVG files, so it takes no other format.
#[test]
fn out_dir_takes_only_svg() {
    let dir = scratch("out-json");
    let dir_text = dir.display().to_string();
    for format in ["json", "graph"] {
        let out = modelspec(&["diagram", LINEAR, "--format", format, "--out", &dir_text]);
        assert_eq!(out.status.code(), Some(2), "--format {format}");
        assert!(std::fs::read_dir(&dir).unwrap().next().is_none());
    }
}

/// diagram_cli.refusals: a shape error as `modelspec ir` reports it.
#[test]
fn shape_error_prints_the_diagnostic_of_ir() {
    let diagram = modelspec(&["diagram", SHAPE_ERROR]);
    let ir = modelspec(&["ir", SHAPE_ERROR, "--stage", "tensor"]);
    assert_eq!(diagram.status.code(), Some(1));
    assert_eq!(ir.status.code(), Some(1));
    assert!(diagram.stdout.is_empty());
    assert!(text(&diagram.stderr).contains("E-TYPE-003"));
    assert_eq!(text(&diagram.stderr), text(&ir.stderr));
}

/// A file that cannot be read is one line, exit 1.
#[test]
fn missing_file_is_one_line() {
    let out = modelspec(&["diagram", "no/such/file.mspec"]);
    assert_eq!(out.status.code(), Some(1));
    let stderr = text(&out.stderr);
    assert_eq!(stderr.lines().count(), 1, "{stderr}");
    assert!(
        stderr.starts_with("modelspec diagram no/such/file.mspec: "),
        "{stderr}"
    );
}

/// diagram_cli.refusals: an unknown `--scope` or `--expand` key exits 2 and lists the keys.
#[test]
fn unknown_scope_lists_the_keys() {
    let graph = model_graph(TWO_LAYER, &source(TWO_LAYER)).unwrap();
    let keys: Vec<&str> = graph.scopes.iter().map(|s| s.key.0.as_str()).collect();
    assert!(!keys.is_empty());
    for flag in ["--scope", "--expand"] {
        let out = modelspec(&["diagram", TWO_LAYER, flag, "nope"]);
        assert_eq!(out.status.code(), Some(2), "{flag}");
        assert!(out.stdout.is_empty(), "{flag}");
        let stderr = text(&out.stderr);
        assert!(stderr.contains("`nope`"), "{flag}: {stderr}");
        for key in &keys {
            assert!(
                stderr.lines().any(|line| line.trim() == *key),
                "{flag}: {key} is not listed in {stderr}"
            );
        }
    }
}

/// A known `--scope` and `--expand` key draws; `--depth` reaches the render.
#[test]
fn known_scope_depth_and_expand_draw() {
    let graph = model_graph(TWO_LAYER, &source(TWO_LAYER)).unwrap();
    let root = graph.scopes[0].key.0.clone();
    let out = modelspec(&[
        "diagram", TWO_LAYER, "--scope", &root, "--expand", &root, "--depth", "3", "--format",
        "json",
    ]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let render: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(render["root"], root.as_str());
    assert_eq!(render["depth"], 3);
    assert_eq!(render["expanded"], serde_json::json!([root]));
}
