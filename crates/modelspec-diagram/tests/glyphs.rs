//! The glyph set of DESIGN.md § 13.1, against the operator registry and the ESS specification
//! (story:diagram-glyph-set, acceptance 1 to 6).

use std::collections::{BTreeMap, BTreeSet};

use modelspec_core::ir::tensor::{Dim, TensorOp};
use modelspec_core::resolve::OPERATORS;
use modelspec_diagram::glyph::{
    EndpointRole, InputRole, OperatorFamily, PipelineStage, family_of, glyph_of, glyphs, operators,
    sprite,
};

/// One start tag of the sprite: its name and attributes in document order.
struct Tag {
    name: String,
    attributes: Vec<(String, String)>,
}

impl Tag {
    fn get(&self, key: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// Every start tag of `svg`, `<style>` text and closing tags skipped. The sprite holds no
/// comments, CDATA or `<` inside text, so a scan for `<` finds every tag.
fn tags(svg: &str) -> Vec<Tag> {
    let mut found = Vec::new();
    let mut rest = svg;
    while let Some(start) = rest.find('<') {
        rest = &rest[start + 1..];
        let end = rest.find('>').expect("a closed tag");
        let body = rest[..end].trim_end_matches('/');
        rest = &rest[end + 1..];
        if body.starts_with('/') || body.starts_with('!') || body.starts_with('?') {
            continue;
        }
        let name_end = body.find(|c: char| c.is_whitespace()).unwrap_or(body.len());
        let name = body[..name_end].to_owned();
        let mut attributes = Vec::new();
        let mut text = body[name_end..].trim_start();
        while !text.is_empty() {
            let eq = text.find('=').expect("an attribute with a value");
            let key = text[..eq].trim().to_owned();
            let after = &text[eq + 1..];
            assert!(after.starts_with('"'), "attribute {key} is quoted");
            let close = after[1..].find('"').expect("a closing quote");
            attributes.push((key, after[1..close + 1].to_owned()));
            text = after[close + 2..].trim_start();
        }
        found.push(Tag { name, attributes });
    }
    found
}

/// The symbols of the sprite, by id without the `ms-` prefix, each with its content; ids in
/// document order with repeats kept.
fn symbols(svg: &str) -> Vec<(String, String)> {
    let mut found = Vec::new();
    let mut rest = svg;
    while let Some(start) = rest.find("<symbol ") {
        rest = &rest[start..];
        let open_end = rest.find('>').expect("a closed tag");
        let open = &rest[..open_end];
        let id = open
            .split("id=\"")
            .nth(1)
            .and_then(|s| s.split('"').next())
            .expect("a symbol id");
        let close = rest.find("</symbol>").expect("a closing tag");
        found.push((
            id.strip_prefix("ms-").expect("an ms- id").to_owned(),
            rest[open_end + 1..close].to_owned(),
        ));
        rest = &rest[close..];
    }
    found
}

fn symbol<'a>(all: &'a [(String, String)], id: &str) -> &'a str {
    let matching: Vec<&(String, String)> = all.iter().filter(|(i, _)| i == id).collect();
    assert_eq!(matching.len(), 1, "exactly one symbol ms-{id}");
    &matching[0].1
}

/// `MatMul` -> `mat-mul`, `LayerNormGradX` -> `layer-norm-grad-x`, `DataIo` -> `data-io`.
fn kebab(name: &str) -> String {
    let mut out = String::new();
    for (i, c) in name.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                out.push('-');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// The `enum` of the `$defs` entry `name` of a schema `ess generate` wrote for the observatory.
fn schema_enum(name: &str) -> Vec<String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../modelspec-observatory/schema/types")
        .join(format!("{name}.schema.json"));
    let text = std::fs::read_to_string(&path).expect("the generated schema exists");
    let document: serde_json::Value = serde_json::from_str(&text).expect("JSON");
    document["$defs"][name]["enum"]
        .as_array()
        .expect("an enum")
        .iter()
        .map(|v| v.as_str().expect("a name").to_owned())
        .collect()
}

/// The `class="ms-outline"` element of a symbol.
fn outline(content: &str) -> Tag {
    let mut outlines: Vec<Tag> = tags(content)
        .into_iter()
        .filter(|t| t.get("class") == Some("ms-outline"))
        .collect();
    assert_eq!(outlines.len(), 1, "one outline in {content}");
    outlines.remove(0)
}

fn has_gradient_mark(content: &str) -> bool {
    tags(content)
        .iter()
        .any(|t| t.get("class") == Some("ms-grad"))
}

/// The `<style>` text of the sprite.
fn style(svg: &str) -> &str {
    let start = svg.find("<style>").expect("a style element") + "<style>".len();
    let end = svg.find("</style>").expect("a closed style element");
    &svg[start..end]
}

/// Acceptance 1, `glyph.every-operator`.
#[test]
fn every_operator_has_exactly_one_symbol() {
    let svg = sprite();
    let all = symbols(&svg);
    for (name, op) in OPERATORS.iter() {
        let id = glyph_of(op, InputRole::Data);
        assert_eq!(id.as_str(), format!("op-{}", kebab(op.name())), "{name}");
        symbol(&all, id.as_str());
    }
    let specified = schema_enum("modelspec.observatory.TensorOperator");
    let mut ids = BTreeSet::new();
    for name in &specified {
        let id = format!("op-{}", kebab(name));
        symbol(&all, &id);
        assert!(ids.insert(id.clone()), "{name} shares {id}");
    }
    let ops = operators();
    let names: Vec<String> = ops.iter().map(|op| op.name().to_owned()).collect();
    assert_eq!(
        names, specified,
        "operators() lists every TensorOp in order"
    );
    let mut given = BTreeMap::new();
    for op in &ops {
        let id = glyph_of(op, InputRole::Data);
        assert_eq!(id.as_str(), format!("op-{}", kebab(op.name())));
        if let Some(other) = given.insert(id.clone(), op.name()) {
            panic!("{} and {other} share {}", op.name(), id.as_str());
        }
    }
}

/// Acceptance 2, `glyph.gradient-mark`.
#[test]
fn gradient_ops_are_their_forward_glyph_with_the_gradient_mark() {
    let pairs = [
        (TensorOp::ReluGrad, TensorOp::Relu),
        (
            TensorOp::EmbeddingGrad {
                rows: Dim::Static(1),
            },
            TensorOp::Embedding,
        ),
        (TensorOp::GeluGrad, TensorOp::Gelu),
        (TensorOp::LayerNormGradX, TensorOp::LayerNorm),
        (TensorOp::LayerNormGradGamma, TensorOp::LayerNorm),
        (TensorOp::SoftmaxGrad, TensorOp::Softmax),
        (
            TensorOp::AttentionGradQ { heads: 1 },
            TensorOp::Attention { heads: 1 },
        ),
        (
            TensorOp::AttentionGradK { heads: 1 },
            TensorOp::Attention { heads: 1 },
        ),
        (
            TensorOp::AttentionGradV { heads: 1 },
            TensorOp::Attention { heads: 1 },
        ),
        (TensorOp::CrossEntropyGrad, TensorOp::CrossEntropy),
        (
            TensorOp::DropoutGrad { p: 0.5 },
            TensorOp::Dropout { p: 0.5 },
        ),
        (TensorOp::SigregColumnsGrad, TensorOp::SigregColumns),
    ];
    let svg = sprite();
    let all = symbols(&svg);
    for (gradient, forward) in &pairs {
        let g = symbol(&all, glyph_of(gradient, InputRole::Data).as_str());
        let f = symbol(&all, glyph_of(forward, InputRole::Data).as_str());
        assert!(
            has_gradient_mark(g),
            "{} has the gradient mark",
            gradient.name()
        );
        assert!(!has_gradient_mark(f), "{} has none", forward.name());
        assert_eq!(
            family_of(gradient),
            family_of(forward),
            "{}",
            gradient.name()
        );
        let (go, fo) = (outline(g), outline(f));
        assert_eq!(go.get("d"), fo.get("d"), "{} outline", gradient.name());
        assert_eq!(
            go.get("stroke"),
            fo.get("stroke"),
            "{} accent",
            gradient.name()
        );
        assert!(
            go.get("stroke-dasharray").is_some()
                && go.get("stroke-dasharray") != fo.get("stroke-dasharray"),
            "{} outline is dashed",
            gradient.name()
        );
    }
    let gradients: BTreeSet<&str> = pairs.iter().map(|(g, _)| g.name()).collect();
    for op in operators() {
        if !gradients.contains(op.name()) {
            let content = symbol(&all, glyph_of(&op, InputRole::Data).as_str());
            assert!(
                !has_gradient_mark(content),
                "{} has no gradient mark",
                op.name()
            );
        }
    }
}

/// Acceptance 3, `glyph.families`.
#[test]
fn every_family_stage_endpoint_and_scope_has_its_glyph() {
    let svg = sprite();
    let all = symbols(&svg);
    let css = style(&svg);
    let catalog = glyphs();
    let mut outlines: BTreeMap<OperatorFamily, (Option<String>, Option<String>)> = BTreeMap::new();
    for family in OperatorFamily::ALL {
        let name = serde_json::to_value(family).expect("JSON");
        let property = format!("--ms-family-{}:", kebab(name.as_str().expect("a name")));
        assert!(css.contains(&property), "{property} in the style");
        let members: Vec<_> = catalog
            .iter()
            .filter(|g| g.family == Some(family) && g.id.as_str().starts_with("op-"))
            .collect();
        assert!(!members.is_empty(), "{family:?} has an operator glyph");
        for glyph in members {
            let content = symbol(&all, glyph.id.as_str());
            if has_gradient_mark(content) {
                continue;
            }
            let o = outline(content);
            let shape = (
                o.get("d").map(str::to_owned),
                o.get("stroke-dasharray").map(str::to_owned),
            );
            let first = outlines.entry(family).or_insert_with(|| shape.clone());
            assert_eq!(
                *first,
                shape,
                "{} has its family's outline",
                glyph.id.as_str()
            );
        }
    }
    let shapes: BTreeSet<_> = outlines.values().collect();
    assert_eq!(
        shapes.len(),
        OperatorFamily::ALL.len(),
        "one outline per family"
    );
    for op in operators() {
        let id = glyph_of(&op, InputRole::Data);
        let entry = catalog.iter().find(|g| g.id == id).expect("in the catalog");
        assert_eq!(entry.family, Some(family_of(&op)), "{}", op.name());
    }
    for stage in PipelineStage::ALL {
        let name = serde_json::to_value(stage).expect("JSON");
        symbol(
            &all,
            &format!("stage-{}", kebab(name.as_str().expect("a name"))),
        );
    }
    for role in EndpointRole::ALL {
        let name = serde_json::to_value(role).expect("JSON");
        symbol(
            &all,
            &format!("endpoint-{}", kebab(name.as_str().expect("a name"))),
        );
    }
    for scope in [
        "scope-function",
        "scope-adjoint",
        "scope-update-sgd",
        "scope-update-adamw",
        "scope-update-muon",
        "scope-module",
        "scope-layer",
    ] {
        symbol(&all, scope);
    }
    let input = TensorOp::Input {
        name: "x".to_owned(),
    };
    assert_eq!(glyph_of(&input, InputRole::Data).as_str(), "op-input");
    assert_eq!(
        glyph_of(&input, InputRole::DropoutToken).as_str(),
        "op-input-token"
    );
    assert_eq!(
        glyph_of(&input, InputRole::HostStep).as_str(),
        "op-input-step"
    );
    for id in ["op-input", "op-input-token", "op-input-step"] {
        symbol(&all, id);
    }
    for op in operators() {
        if !matches!(op, TensorOp::Input { .. }) {
            let data = glyph_of(&op, InputRole::Data);
            assert_eq!(
                glyph_of(&op, InputRole::DropoutToken),
                data,
                "{}",
                op.name()
            );
            assert_eq!(glyph_of(&op, InputRole::HostStep), data, "{}", op.name());
        }
    }
    let ids: BTreeSet<&str> = catalog.iter().map(|g| g.id.as_str()).collect();
    let drawn: BTreeSet<&str> = all.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(ids, drawn, "the catalog lists every symbol");
}

/// The JEPA operators take the families DESIGN.md § 13.1 gives them: `stop_gradient` is Layout,
/// `sigreg` a regularization term of the Loss family, and its column kernel and that kernel's
/// gradient parts of that loss; each has a symbol of its own.
#[test]
fn jepa_operators_take_their_families() {
    let svg = sprite();
    let all = symbols(&svg);
    let cases = [
        (TensorOp::StopGradient, OperatorFamily::Layout),
        (TensorOp::Sigreg { slices: 4 }, OperatorFamily::Loss),
        (TensorOp::SigregColumns, OperatorFamily::Loss),
        (TensorOp::SigregColumnsGrad, OperatorFamily::Loss),
    ];
    let mut marks = BTreeSet::new();
    for (op, family) in &cases {
        assert_eq!(family_of(op), *family, "{}", op.name());
        let content = symbol(&all, glyph_of(op, InputRole::Data).as_str());
        if !has_gradient_mark(content) {
            assert!(
                marks.insert(content.to_owned()),
                "{} is drawn unlike the others",
                op.name()
            );
        }
    }
}

/// Acceptance 4, `glyph.themable`.
#[test]
fn every_paint_is_a_custom_property_with_a_fallback() {
    let svg = sprite();
    let mut paints = 0;
    for tag in tags(&svg) {
        for (key, value) in &tag.attributes {
            if key == "fill" || key == "stroke" || key == "color" || key.ends_with("-color") {
                paints += 1;
                assert!(
                    value.starts_with("var(--ms-") && value.ends_with(')') && value.contains(", "),
                    "<{}> {key}=\"{value}\" is a var(--ms-…, <fallback>)",
                    tag.name
                );
            } else if key != "style" {
                assert!(
                    !value.contains('#'),
                    "<{}> {key}=\"{value}\" holds no colour",
                    tag.name
                );
            } else {
                panic!("<{}> carries a style attribute", tag.name);
            }
        }
    }
    assert!(paints > 0, "the sprite paints something");
    let css = style(&svg);
    let dark = css
        .find("@media (prefers-color-scheme: dark)")
        .expect("a prefers-color-scheme: dark block");
    let (light, dark) = css.split_at(dark);
    let mut properties = vec![
        "--ms-diagram-ink".to_owned(),
        "--ms-diagram-paper".to_owned(),
        "--ms-diagram-muted".to_owned(),
    ];
    for family in OperatorFamily::ALL {
        let name = serde_json::to_value(family).expect("JSON");
        properties.push(format!(
            "--ms-family-{}",
            kebab(name.as_str().expect("a name"))
        ));
    }
    for subgraph in schema_enum("modelspec.observatory.Subgraph") {
        properties.push(format!("--ms-subgraph-{}", kebab(&subgraph)));
    }
    for property in &properties {
        assert!(
            light.contains(&format!("{property}:")),
            "{property} light default"
        );
        assert!(
            dark.contains(&format!("{property}:")),
            "{property} dark default"
        );
    }
}

/// Acceptance 5, `glyph.deterministic`.
#[test]
fn the_sprite_is_the_same_bytes_with_integer_numbers() {
    let svg = sprite();
    assert!(!svg.is_empty(), "the sprite is drawn");
    assert_eq!(svg, sprite());
    for tag in tags(&svg) {
        for (key, value) in &tag.attributes {
            if key != "xmlns" {
                assert!(
                    !value.contains('.'),
                    "<{}> {key}=\"{value}\" has no '.'",
                    tag.name
                );
            }
        }
    }
    let all = symbols(&svg);
    let ids: Vec<&str> = all.iter().map(|(id, _)| id.as_str()).collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(ids, sorted, "symbols in id order, each once");
    for tag in tags(&svg).iter().filter(|t| t.name == "symbol") {
        assert_eq!(tag.get("viewBox"), Some("0 0 24 24"));
    }
}

/// Acceptance 6, `glyph.enums-match-domain`.
#[test]
fn enums_name_the_variants_of_the_domain_in_order() {
    fn names<T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug>(
        all: &[T],
    ) -> Vec<String> {
        all.iter()
            .map(|v| {
                let json = serde_json::to_value(v).expect("JSON");
                let back: T = serde_json::from_value(json.clone()).expect("reads back");
                assert_eq!(&back, v);
                json.as_str().expect("a name").to_owned()
            })
            .collect()
    }
    assert_eq!(
        names(&OperatorFamily::ALL),
        schema_enum("modelspec.diagram.OperatorFamily")
    );
    assert_eq!(
        names(&PipelineStage::ALL),
        schema_enum("modelspec.diagram.PipelineStage")
    );
    assert_eq!(
        names(&EndpointRole::ALL),
        schema_enum("modelspec.diagram.EndpointRole")
    );
    let id = glyph_of(&TensorOp::MatMul, InputRole::Data);
    assert_eq!(
        serde_json::to_value(&id).expect("JSON"),
        serde_json::json!("op-mat-mul")
    );
}
