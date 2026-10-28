//! `DiagramGraph` and `DiagramRender` as JSON: the records of `spec/domains/diagram.yaml`, field
//! names as the domain, optional fields absent, pretty-printed with a final newline, so the same
//! graph gives the same bytes.

use crate::model::{DiagramGraph, DiagramRender};

/// The JSON of `graph`.
pub fn graph_json(graph: &DiagramGraph) -> String {
    to_json(graph)
}

/// The JSON of `render`.
pub fn render_json(render: &DiagramRender) -> String {
    to_json(render)
}

fn to_json(value: &impl serde::Serialize) -> String {
    let mut text = serde_json::to_string_pretty(value)
        .expect("a diagram record holds only strings, integers, lists and enums");
    text.push('\n');
    text
}
