//! The architecture diagrams of the site's programs (DESIGN.md § 13.6): one standalone SVG per
//! level under [`IMAGES`]`/<program>/<view>/<scope-key>.svg`, as `benchmarks.rs` writes its charts
//! under `website/static/img/benchmarks/`, and one JSON module per program under [`DATA`] holding
//! each level's SVG text by scope key, which `website/src/components/ModelDiagram.tsx` inlines
//! into a page at build time. [`GLYPHS`] holds the glyph sprite for the gallery, and [`CSS`] the
//! site's theme properties of the diagrams, light under `:root` and dark under
//! `[data-theme='dark']`.

use std::path::Path;

use modelspec_core::compile;
use modelspec_diagram::glyph::{Scheme, sprite, theme};
use modelspec_diagram::layout::{RenderRequest, render};
use modelspec_diagram::model::DiagramView;
use modelspec_diagram::{graph, model_graph, svg};
use serde_json::{Map, Value, json};

use crate::{Pages, Result, io};

/// Where the standalone SVGs land, one directory per program and view.
pub const IMAGES: &str = "website/static/img/diagrams";
/// Where the per-program JSON modules land.
pub const DATA: &str = "website/data/diagrams";
/// The glyph sprite as a JSON module, under [`crate::DATA`].
pub const GLYPHS: &str = "glyphs.json";
/// The site's diagram stylesheet, relative to the repository root.
pub const CSS_DIR: &str = "website/src/css";
/// The file name of the stylesheet under [`CSS_DIR`].
pub const CSS: &str = "diagram.css";

/// A program the site draws: its key in file and module names, and its source file.
pub struct Program {
    pub key: &'static str,
    pub file: &'static str,
}

/// Every program the site draws.
pub const PROGRAMS: &[Program] = &[Program {
    key: "two-layer-relu",
    file: "crates/modelspec-docs/models/two_layer_relu.mspec",
}];

/// The drawn levels of one program and view: the directory of its SVGs relative to the repository
/// root, and the files in it.
pub struct Level {
    pub dir: String,
    pub files: Pages,
}

/// What [`files`] generates.
pub struct Generated {
    /// One directory of SVGs per program and view.
    pub images: Vec<Level>,
    /// The JSON module of each program, by file name, for [`DATA`].
    pub data: Pages,
    /// The glyph sprite module, [`GLYPHS`].
    pub glyphs: String,
    /// The stylesheet, [`CSS`].
    pub css: String,
}

/// The diagrams of every program in [`PROGRAMS`], read under `root`.
pub fn files(root: &Path) -> Result<Generated> {
    let mut images = Vec::new();
    let mut data = Pages::new();
    for program in PROGRAMS {
        let source = std::fs::read_to_string(root.join(program.file)).map_err(io(program.file))?;
        let graph = model_graph(program.file, &source).map_err(|diagnostics| {
            format!(
                "{}: does not compile: {}",
                program.file,
                diagnostics
                    .iter()
                    .map(|d| d.code.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        let compiled = compile::compile_training(program.file, &source)
            .map_err(|_| format!("{}: does not compile", program.file))?;
        let lines = graph::ir_lines(&compiled);
        let view = DiagramView::Model;
        let drawn = render(&graph, &RenderRequest::default());
        let root_key = drawn.root.0.clone();
        let text = svg::svg(&graph, &drawn, &lines);
        let mut levels = Map::new();
        levels.insert(root_key.clone(), Value::String(text.clone()));
        images.push(Level {
            dir: format!("{IMAGES}/{}/{}", program.key, view.name()),
            files: Pages::from([(format!("{root_key}.svg"), text)]),
        });
        let module = json!({
            "program": program.key,
            "source": program.file,
            "views": {
                view.name(): {
                    "root": root_key,
                    "levels": levels,
                },
            },
        });
        data.insert(format!("{}.json", program.key), pretty(&module)?);
    }
    Ok(Generated {
        images,
        data,
        glyphs: pretty(&json!({ "sprite": sprite() }))?,
        css: stylesheet(),
    })
}

fn pretty(value: &Value) -> Result<String> {
    serde_json::to_string_pretty(value)
        .map(|text| text + "\n")
        .map_err(|e| e.to_string())
}

/// The theme of the diagrams on the site. Ink, paper and muted follow Infima's colours, so a
/// diagram reads like the page around it; the family and subgraph accents are the glyph set's own
/// light and dark values. Each rule sits on `.ms-glyphs`, where the SVG's own `<style>` sets its
/// defaults with no specificity, so the site's theme toggle wins over the system's colour scheme.
fn stylesheet() -> String {
    let rule = |selector: &str, scheme: Scheme| {
        let mut out = format!("{selector} .ms-glyphs {{\n");
        for (property, value) in theme(scheme) {
            let value = match property.as_str() {
                "--ms-diagram-ink" => "var(--ifm-font-color-base)",
                "--ms-diagram-paper" => "var(--ifm-background-surface-color)",
                "--ms-diagram-muted" => "var(--ifm-color-emphasis-600)",
                _ => value,
            };
            out.push_str(&format!("  {property}: {value};\n"));
        }
        out.push_str("}\n");
        out
    };
    let mut out = format!(
        "/* generated by modelspec-docs from modelspec_diagram::glyph::theme, do not edit */\n\n{}\n{}",
        rule(":root", Scheme::Light),
        rule("[data-theme='dark']", Scheme::Dark),
    );
    out.push_str(FIGURE_CSS);
    out
}

/// The layout of a diagram figure and of the gallery's glyph cells.
const FIGURE_CSS: &str = "
.ms-figure {
  margin: 0 0 var(--ifm-leading) 0;
  padding: 0.75rem;
  border: 1px solid var(--ifm-color-emphasis-300);
  border-radius: var(--ifm-code-border-radius);
  background: var(--ifm-background-surface-color);
}

.ms-figure__canvas {
  overflow-x: auto;
  text-align: center;
}

.ms-figure__canvas svg {
  max-width: 100%;
  height: auto;
}

.ms-figure figcaption {
  margin-top: 0.5rem;
  font-size: 0.875rem;
  color: var(--ifm-color-emphasis-700);
}

.ms-glyph {
  width: 2.5rem;
  height: 2.5rem;
  vertical-align: middle;
}

.ms-glyph--small {
  width: 1.75rem;
  height: 1.75rem;
}

.ms-sprite {
  position: absolute;
  width: 0;
  height: 0;
  overflow: hidden;
}
";

#[cfg(test)]
mod tests {
    use super::*;

    fn repository() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    #[test]
    fn two_layer_relu_is_drawn_once_per_level_with_its_json_module() {
        let generated = files(&repository()).unwrap();
        assert_eq!(generated.images.len(), 1);
        let level = &generated.images[0];
        assert_eq!(
            level.dir,
            "website/static/img/diagrams/two-layer-relu/model"
        );
        let svg = &level.files["FitTwoLayer.svg"];
        assert!(svg.contains(" role=\"img\"") && svg.contains("data-op=\"MatMul\""));
        let module: Value = serde_json::from_str(&generated.data["two-layer-relu.json"]).unwrap();
        assert_eq!(module["views"]["model"]["root"], "FitTwoLayer");
        assert_eq!(
            module["views"]["model"]["levels"]["FitTwoLayer"].as_str(),
            Some(svg.as_str())
        );
    }

    #[test]
    fn the_stylesheet_sets_every_theme_property_light_and_dark() {
        let css = stylesheet();
        let (light, dark) = css.split_once("[data-theme='dark'] .ms-glyphs {").unwrap();
        assert!(light.contains(":root .ms-glyphs {"));
        for (scheme, part) in [(Scheme::Light, light), (Scheme::Dark, dark)] {
            for (property, value) in theme(scheme) {
                let line = part
                    .lines()
                    .find(|l| l.trim_start().starts_with(&format!("{property}:")))
                    .unwrap_or_else(|| panic!("{property} missing"));
                if property.starts_with("--ms-family-") || property.starts_with("--ms-subgraph-") {
                    assert!(line.contains(value), "{line}");
                } else {
                    assert!(line.contains("var(--ifm-"), "{line}");
                }
            }
        }
    }
}
