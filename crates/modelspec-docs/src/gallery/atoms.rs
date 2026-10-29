//! The gallery's atoms page, `website/docs/gallery/atoms.mdx`: one entry per symbol of
//! `modelspec_diagram::glyph::sprite()`, grouped by operator family, then the composite, stage and
//! endpoint glyphs. An operator's entry shows its glyph, the Tensor IR operation, the name a source
//! file calls it by (`modelspec_core::resolve::OPERATORS`, or "internal") and, for a forward
//! operator with a gradient op of its own, that op's glyph. The page inlines the sprite once,
//! from the JSON module [`crate::diagrams::GLYPHS`], and every entry draws its glyph with
//! `<use href="#ms-…">`.

use std::mem::discriminant;

use modelspec_core::ir::tensor::TensorOp;
use modelspec_core::resolve::OPERATORS;
use modelspec_diagram::glyph::{
    EndpointRole, Glyph, GlyphId, InputRole, OperatorFamily, PipelineStage, SCOPE_GLYPHS, glyph_of,
    glyphs, operators,
};

use crate::front_matter;

/// The page's file name under [`super::DIR`].
pub const PAGE: &str = "atoms.mdx";

/// What one operator glyph stands for.
struct Operation {
    glyph: GlyphId,
    /// The Tensor IR operation, `MatMul`, with what an `Input` is fed when the role sets it.
    name: String,
    /// The name a source file calls it by.
    source: Option<&'static str>,
}

/// Every operator glyph, in the declaration order of `TensorOp`, an `Input` once per role.
fn operations() -> Vec<Operation> {
    let mut out = Vec::new();
    for op in operators() {
        let roles: &[(InputRole, &str)] = match op {
            TensorOp::Input { .. } => &[
                (InputRole::Data, "an input, target or batch tensor"),
                (InputRole::DropoutToken, "the RNG token of a dropout site"),
                (InputRole::HostStep, "a host input of an AdamW step"),
            ],
            _ => &[(InputRole::Data, "")],
        };
        for &(role, fed) in roles {
            let name = if fed.is_empty() {
                format!("`{}`", op.name())
            } else {
                format!("`{}`, {fed}", op.name())
            };
            out.push(Operation {
                glyph: glyph_of(&op, role),
                name,
                source: OPERATORS
                    .iter()
                    .find(|(_, registered)| discriminant(registered) == discriminant(&op))
                    .map(|(source, _)| *source),
            });
        }
    }
    out
}

/// A glyph drawn inline, at entry size or, `small`, beside another entry.
fn glyph(id: &GlyphId, label: &str, small: bool) -> String {
    let small = if small { " small" } else { "" };
    format!(
        "<Glyph id=\"{id}\" label=\"{}\"{small} />",
        label.replace(['`', '"'], "")
    )
}

fn heading(family: OperatorFamily) -> &'static str {
    match family {
        OperatorFamily::Elementwise => "Elementwise: circle",
        OperatorFamily::Contraction => "Contraction: square",
        OperatorFamily::Reduction => "Reduction: funnel",
        OperatorFamily::Normalization => "Normalization: rounded square",
        OperatorFamily::Attention => "Attention: hexagon",
        OperatorFamily::Embedding => "Embedding: table",
        OperatorFamily::Layout => "Layout: parallelogram",
        OperatorFamily::Stochastic => "Stochastic: dashed circle",
        OperatorFamily::Loss => "Loss: diamond",
        OperatorFamily::Optimizer => "Optimizer: octagon",
        OperatorFamily::DataIo => "Data in and out: cylinder",
    }
}

/// The page.
pub fn page() -> String {
    let all = glyphs();
    let operations = operations();
    let name_of = |id: &GlyphId| {
        operations
            .iter()
            .find(|o| &o.glyph == id)
            .map_or_else(|| id.to_string(), |o| o.name.clone())
    };
    let mut out = front_matter(
        "Gallery: atoms",
        "Atoms",
        "Every glyph of a ModelSpec diagram: one per Tensor IR operation in eleven families, each operator's source name and gradient op, and the composite, stage and endpoint glyphs.",
        2,
    );
    out.push_str(
        "import {Glyph, GlyphSprite} from '@site/src/components/ModelDiagram';\n\
         import glyphs from '@site/data/glyphs.json';\n\n\
         <GlyphSprite sprite={glyphs.sprite} />\n\n",
    );
    out.push_str(&format!(
        "Each glyph below is one `<symbol>` of the glyph set that `modelspec-diagram` generates \
         (DESIGN.md § 13.1); a diagram draws every node with one of them. A family fixes the \
         outline and the accent colour, the operator the mark inside it. A gradient op is its \
         forward operator's glyph with the outline dashed and the `∇` badge in the top right, and \
         when it is the gradient with respect to one of several operands, that operand's slot is \
         marked along the bottom. {} glyphs in all.\n\n\
         **Source** is the name a `.mspec` file calls the operator by; an operation no source may \
         call is internal. **Gradient op** is the operation the gradient rule of a forward \
         operator emits for it (DESIGN.md § 4); an operator without one takes its gradient \
         through other operators, or takes none.\n",
        all.len()
    ));
    for family in OperatorFamily::ALL {
        let members: Vec<&Operation> = operations
            .iter()
            .filter(|o| family_of_glyph(&all, &o.glyph) == Some(family))
            .collect();
        if members.is_empty() {
            continue;
        }
        out.push_str(&format!("\n## {}\n\n", heading(family)));
        out.push_str("| Glyph | Operation | Source | Gradient op |\n|---|---|---|---|\n");
        for operation in members {
            let entry = all.iter().find(|g| g.id == operation.glyph);
            let source = operation
                .source
                .map_or_else(|| "internal".to_owned(), |s| format!("`{s}`"));
            let gradient = match entry.and_then(|g| g.gradient_of.as_ref()) {
                Some(forward) => format!("gradient of {}", name_of(forward)),
                None => {
                    let ops: Vec<String> = operations
                        .iter()
                        .map(|o| &o.glyph)
                        .filter(|id| {
                            all.iter().any(|g| {
                                &g.id == *id && g.gradient_of.as_ref() == Some(&operation.glyph)
                            })
                        })
                        .map(|id| format!("{} {}", glyph(id, &name_of(id), true), name_of(id)))
                        .collect();
                    if ops.is_empty() {
                        "–".to_owned()
                    } else {
                        ops.join("<br />")
                    }
                }
            };
            out.push_str(&format!(
                "| {} | {} | {source} | {gradient} |\n",
                glyph(&operation.glyph, &name_of(&operation.glyph), false),
                operation.name,
            ));
        }
    }
    let composites: Vec<(GlyphId, String)> = SCOPE_GLYPHS
        .iter()
        .map(|id| (GlyphId((*id).to_owned()), scope_text(id).to_owned()))
        .collect();
    let stages: Vec<(GlyphId, String)> = PipelineStage::ALL
        .iter()
        .map(|stage| {
            (
                stage.glyph(),
                format!("the {stage:?} stage of a pipeline view"),
            )
        })
        .collect();
    let endpoints: Vec<(GlyphId, String)> = EndpointRole::ALL
        .iter()
        .map(|role| (role.glyph(), endpoint_text(*role).to_owned()))
        .collect();
    for (title, intro, list) in [
        (
            "Composites",
            "A collapsed scope is one composite node; its glyph is a stacked card.",
            composites,
        ),
        (
            "Stages",
            "The root of a training, evaluation or inference view is a pipeline of stages; a stage's glyph is a chevron.",
            stages,
        ),
        (
            "Endpoints",
            "Data splits, checkpoints and predictions are endpoints of a pipeline view; an endpoint's glyph is a document.",
            endpoints,
        ),
    ] {
        out.push_str(&format!(
            "\n## {title}\n\n{intro}\n\n| Glyph | Id | Stands for |\n|---|---|---|\n"
        ));
        for (id, text) in list {
            out.push_str(&format!(
                "| {} | `{id}` | {text} |\n",
                glyph(&id, &format!("{id}: {text}"), false)
            ));
        }
    }
    out
}

fn family_of_glyph(all: &[Glyph], id: &GlyphId) -> Option<OperatorFamily> {
    all.iter().find(|g| &g.id == id).and_then(|g| g.family)
}

fn scope_text(id: &str) -> &'static str {
    match id {
        "scope-function" => "an inlined function call",
        "scope-adjoint" => "the backward pass of one forward node",
        "scope-update-sgd" => "an SGD update of one parameter",
        "scope-update-adamw" => "an AdamW update of one parameter",
        "scope-update-muon" => "a Muon update of one parameter",
        "scope-module" => "a module instance (planned, DESIGN.md § 11 iteration 3)",
        "scope-layer" => "a layer instance (planned, DESIGN.md § 11 iteration 3)",
        _ => "a composite",
    }
}

fn endpoint_text(role: EndpointRole) -> &'static str {
    match role {
        EndpointRole::DataSource => "a data split feeding the Data stage",
        EndpointRole::CheckpointSource => "a checkpoint an inference reads",
        EndpointRole::CheckpointSink => "the parameters a training step writes",
        EndpointRole::PredictionSink => "the predictions an inference writes",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use modelspec_diagram::glyph::sprite;

    use super::*;

    /// The ids of the `<Glyph id="…">` elements of `line`, in order.
    fn glyph_ids(line: &str) -> Vec<&str> {
        line.split("<Glyph id=\"")
            .skip(1)
            .map(|rest| &rest[..rest.find('"').unwrap()])
            .collect()
    }

    /// Acceptance 2, `gallery.atoms.every-glyph`: one entry per `<symbol id="ms-…">` of the
    /// sprite, and each forward operator's entry shows the glyph of its gradient op.
    #[test]
    fn gallery_atoms_every_glyph() {
        let sprite = sprite();
        let symbols: Vec<&str> = sprite
            .split("<symbol id=\"ms-")
            .skip(1)
            .map(|rest| &rest[..rest.find('"').unwrap()])
            .collect();
        assert!(symbols.len() > 40, "{}", symbols.len());
        let page = page();
        let mut entries: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for line in page.lines().filter(|l| l.starts_with("| <Glyph id=")) {
            let ids = glyph_ids(line);
            assert!(
                entries.insert(ids[0], ids[1..].to_vec()).is_none(),
                "two entries for {}",
                ids[0]
            );
        }
        let mut expected: Vec<&str> = symbols.clone();
        expected.sort();
        let found: Vec<&str> = entries.keys().copied().collect();
        assert_eq!(found, expected, "one entry per symbol");
        for g in glyphs() {
            if let Some(forward) = &g.gradient_of {
                assert!(
                    entries[forward.as_str()].contains(&g.id.as_str()),
                    "the entry of {forward} shows its gradient op {}",
                    g.id
                );
            }
        }
        for (forward, gradients) in [
            ("op-relu", &["op-relu-grad"][..]),
            (
                "op-attention",
                &[
                    "op-attention-grad-q",
                    "op-attention-grad-k",
                    "op-attention-grad-v",
                ],
            ),
            ("op-mat-mul", &[]),
        ] {
            assert_eq!(entries[forward], gradients, "{forward}");
        }
        assert!(page.contains("| `MatMul` | `matmul` |"), "the source name");
        assert!(page.contains("| `ReluGrad` | internal |"), "an internal op");
    }

    #[test]
    fn every_operator_entry_is_under_its_family() {
        let page = page();
        let mut family = None;
        for line in page.lines() {
            if let Some(title) = line.strip_prefix("## ") {
                family = OperatorFamily::ALL
                    .into_iter()
                    .find(|f| heading(*f) == title);
            } else if line.starts_with("| <Glyph id=")
                && let Some(family) = family
            {
                let id = GlyphId(glyph_ids(line)[0].to_owned());
                assert_eq!(family_of_glyph(&glyphs(), &id), Some(family), "{line}");
            }
        }
    }
}
