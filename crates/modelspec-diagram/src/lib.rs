//! Architecture diagrams of ModelSpec programs (DESIGN.md § 13, `spec/domains/diagram.yaml`).
//!
//! [`glyph`] is the glyph set of DESIGN.md § 13.1: one SVG symbol per Tensor IR operation in
//! eleven families, the gradient, scope, stage and endpoint glyphs, and the CSS custom properties
//! that theme them.
//!
//! The model view (DESIGN.md § 13.2, § 13.5): [`graph`] builds the `DiagramGraph` of a verified
//! program, [`layout`] lays out a `DiagramRender` of it, [`svg`] draws a render as a standalone
//! SVG and [`json`] writes both records as JSON; [`model`] holds the records of
//! `spec/domains/diagram.yaml`.

pub mod glyph;
pub mod graph;
pub mod json;
pub mod layout;
pub mod model;
pub mod svg;

use modelspec_core::compile;
use modelspec_core::diag::Diagnostic;

/// The model view of the `.mspec` source `source`, `path` naming it in diagnostics and source
/// spans: [`compile::compile_training`], then [`graph::model`]. A file that does not compile
/// gives the diagnostics `modelspec check` prints.
pub fn model_graph(path: &str, source: &str) -> Result<model::DiagramGraph, Vec<Diagnostic>> {
    let hir = compile::compile_hir(path, source)?;
    let compiled = compile::compile_training(path, source)?;
    Ok(graph::model(&compiled, &hir.train.name, source))
}
