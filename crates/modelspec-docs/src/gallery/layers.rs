//! The gallery's layers page, `website/docs/gallery/layers.mdx` (`epic:ir-svg-gallery`, row
//! "layer"; DESIGN.md § 13.3). The language has no `layer` construct yet (DESIGN.md § 11
//! iteration 3), so each entry is a small model whose `forward` is the block, compiled from
//! `crates/modelspec-docs/models/layers/` and drawn in the model view through
//! [`crate::diagrams::draw`] as the program `layer-<entry>`. A model that does not compile stops
//! generation with its file and diagnostic codes. Until the model view draws a function scope as a
//! composite (DESIGN.md § 13.3), each entry shows its block expanded: the block's operators, its
//! arguments as inputs and its parameters as chips.

use std::path::Path;

use crate::diagrams::{self, Drawn, Program};
use crate::{Result, SOURCE, fenced, front_matter};

/// The page's file name under [`super::DIR`].
pub const PAGE: &str = "layers.mdx";
/// The page's title, which the program pages under `blocks::DIR` link back to.
pub const TITLE: &str = "Gallery: layers";

/// One layer of the gallery: its program, its heading and what the block computes.
pub struct Entry {
    pub program: Program,
    pub title: &'static str,
    pub about: &'static str,
}

/// Every layer of the gallery, in page order.
pub const ENTRIES: &[Entry] = &[
    Entry {
        program: Program {
            key: "layer-dense",
            file: "crates/modelspec-docs/models/layers/dense.mspec",
        },
        title: "Dense",
        about: "A linear layer without bias: one `matmul` of the input by a weight.",
    },
    Entry {
        program: Program {
            key: "layer-dense-relu",
            file: "crates/modelspec-docs/models/layers/dense_relu.mspec",
        },
        title: "Dense with relu",
        about: "The dense layer followed by its activation, `relu(matmul(x, w))`.",
    },
    Entry {
        program: Program {
            key: "layer-dense-bias-gelu",
            file: "crates/modelspec-docs/models/layers/dense_bias_gelu.mspec",
        },
        title: "Dense with bias and gelu",
        about: "The feed-forward half of a transformer layer: `add_rows` adds the bias `b` to every \
                row of the product, then `gelu`.",
    },
    Entry {
        program: Program {
            key: "layer-layer-norm-residual",
            file: "crates/modelspec-docs/models/layers/layer_norm_residual.mspec",
        },
        title: "Residual and layer norm",
        about: "A post-layer-norm residual: the sublayer `f` is added back to its input and the \
                sum normalised, `layer_norm(add(x, f), gamma, beta)`.",
    },
    Entry {
        program: Program {
            key: "layer-self-attention",
            file: "crates/modelspec-docs/models/layers/self_attention.mspec",
        },
        title: "Self-attention",
        about: "Three projections of one input, the queries, keys and values, read by one \
                `attention` with two heads and a key mask.",
    },
];

/// What the layers page generates: the page and, per entry, its drawn program, which `main.rs`
/// writes as its SVGs and JSON module and gives a page under `blocks::DIR`.
pub struct Files {
    pub page: String,
    pub drawn: Vec<Drawn>,
}

/// The layers page and the diagrams of [`ENTRIES`], read under `root`.
pub fn files(root: &Path) -> Result<Files> {
    files_of(root, ENTRIES)
}

fn files_of(root: &Path, entries: &[Entry]) -> Result<Files> {
    let mut all = Vec::new();
    let mut imports =
        String::from("import ModelDiagram from '@site/src/components/ModelDiagram';\n");
    let mut body = String::new();
    for entry in entries {
        let drawn = diagrams::draw(root, &entry.program)?;
        let source = std::fs::read_to_string(root.join(entry.program.file))
            .map_err(crate::io(entry.program.file))?;
        let name = identifier(entry.program.key);
        imports.push_str(&format!(
            "import {name} from '@site/data/diagrams/{}.json';\n",
            entry.program.key
        ));
        body.push_str(&format!(
            "\n## {}\n\n{}\n\n[`{file}`]({SOURCE}/{file}):\n\n{}\n<ModelDiagram data={{{name}}} view=\"model\" />\n",
            entry.title,
            entry.about,
            fenced(&format!("modelspec title=\"{}\"", entry.program.file), &source),
            file = entry.program.file,
        ));
        all.push(drawn);
    }
    let mut page = front_matter(
        TITLE,
        "Layers",
        "Composed blocks drawn from their verified Tensor IR: dense layers, a residual with layer norm and self-attention, each a small model whose forward is the block.",
        3,
    );
    page.push_str(&imports);
    page.push_str(
        "\nA layer is a block of operators that a model composes, such as a dense layer with its \
         activation. The language has no `layer` construct yet, and a model function is the unit \
         of composition, inlined into the Tensor IR (DESIGN.md § 0, invariant 2; § 2.1). So each \
         entry below is a small model whose `forward` is the block, checked by `modelspec check`, \
         and its diagram is the model view of that program: the nodes its prediction reaches, \
         without the loss and the target.\n\n\
         Each diagram shows the block expanded: an `Input` per argument of `forward`, one box per \
         operator, and each parameter with one reader drawn as a chip on that reader. Drawing the \
         block collapsed, as one function composite that zooms into this level, arrives with zoom \
         (DESIGN.md § 13.3).\n",
    );
    page.push_str(&body);
    Ok(Files { page, drawn: all })
}

/// The JavaScript name of a program's JSON module: `layer-dense-relu` as `layerDenseRelu`.
fn identifier(key: &str) -> String {
    let mut out = String::new();
    for (i, part) in key.split('-').enumerate() {
        let mut chars = part.chars();
        if let Some(first) = chars.next() {
            if i == 0 {
                out.push(first);
            } else {
                out.extend(first.to_uppercase());
            }
            out.extend(chars);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repository() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    /// The value of every `name="…"` attribute in the tags of `svg` that start with `tag`.
    fn attributes<'a>(svg: &'a str, tag: &str, name: &str) -> Vec<&'a str> {
        svg.split(tag)
            .skip(1)
            .map(|rest| {
                let open = &rest[..rest.find('>').unwrap()];
                let start = open.find(&format!(" {name}=\"")).unwrap() + name.len() + 3;
                &open[start..start + open[start..].find('"').unwrap()]
            })
            .collect()
    }

    /// Acceptance 3, `gallery.layers.block-is-composite`, the expanded level: its operator boxes
    /// are exactly the block's operators and an `Input` per argument, its chips are the
    /// parameters, and nothing of the loss or the target is drawn.
    #[test]
    fn gallery_layers_expanded_level_is_the_block() {
        /// A program key, then the block's operators, its arguments and its parameters.
        type Expected<'a> = (&'a str, &'a [&'a str], &'a [&'a str], &'a [&'a str]);
        let files = files(&repository()).unwrap();
        let expected: [Expected; 5] = [
            ("layer-dense", &["MatMul"], &["x"], &["w"]),
            ("layer-dense-relu", &["MatMul", "Relu"], &["x"], &["w"]),
            (
                "layer-dense-bias-gelu",
                &["MatMul", "AddRows", "Gelu"],
                &["x"],
                &["w", "b"],
            ),
            (
                "layer-layer-norm-residual",
                &["MatMul", "Add", "LayerNorm"],
                &["x"],
                &["w", "gamma", "beta"],
            ),
            (
                "layer-self-attention",
                &["MatMul", "MatMul", "MatMul", "Attention"],
                &["x", "key_mask"],
                &["wq", "wk", "wv"],
            ),
        ];
        assert_eq!(files.drawn.len(), expected.len());
        for ((key, block, inputs, params), drawn) in expected.iter().zip(&files.drawn) {
            let (levels, (module, _)) = diagrams::written(drawn).unwrap();
            let [level] = &levels[..] else {
                panic!("{key}: one view");
            };
            assert_eq!(
                level.dir,
                format!("website/static/img/diagrams/{key}/model")
            );
            let [(_, svg)] = level.files.iter().collect::<Vec<_>>()[..] else {
                panic!("{key}: one level");
            };
            let mut ops = attributes(svg, "<g class=\"node\"", "data-op");
            let input_count = ops.iter().filter(|op| **op == "Input").count();
            ops.retain(|op| *op != "Input");
            let mut want = block.to_vec();
            ops.sort_unstable();
            want.sort_unstable();
            assert_eq!(ops, want, "{key}: the block's operators");
            assert_eq!(input_count, inputs.len(), "{key}: one Input per argument");
            for input in *inputs {
                assert!(
                    svg.contains(&format!("Input \"{input}\"")),
                    "{key}: {input}"
                );
            }
            assert!(!svg.contains("Input \"y\""), "{key}: no target");
            assert!(!svg.contains("data-op=\"Mse\""), "{key}: no loss");
            let chips: Vec<&str> = svg
                .split("<g class=\"chip\"")
                .skip(1)
                .map(|rest| {
                    let text = &rest[rest.find("</title>").unwrap()..];
                    let text = &text[text.find("<text").unwrap()..];
                    &text[text.find('>').unwrap() + 1..text.find("</text>").unwrap()]
                })
                .collect();
            let mut chips = chips;
            let mut params = params.to_vec();
            chips.sort_unstable();
            params.sort_unstable();
            assert_eq!(chips, params, "{key}: the parameters as chips");
            assert_eq!(module, format!("{key}.json"), "{key}");
            assert!(
                files.page.contains(&format!(
                    "import {} from '@site/data/diagrams/{key}.json';",
                    identifier(key)
                )),
                "{key}: imported"
            );
        }
    }

    /// Acceptance 4, `gallery.layers.refuses-broken-model`: a layer model with a shape error
    /// makes generation return an error naming the file and its diagnostic code, not a page.
    #[test]
    fn gallery_layers_refuses_broken_model() {
        let broken = Entry {
            program: Program {
                key: "layer-broken",
                file: "crates/modelspec-docs/models/matmul_mismatch.mspec",
            },
            title: "Broken",
            about: "A shape error.",
        };
        let error = match files_of(&repository(), &[broken]) {
            Ok(_) => panic!("a broken layer model wrote a page"),
            Err(error) => error,
        };
        assert!(
            error
                == "crates/modelspec-docs/models/matmul_mismatch.mspec: does not compile: E-TYPE-012",
            "{error}"
        );
    }

    #[test]
    fn every_entry_shows_its_source_and_its_diagram_in_order() {
        let page = files(&repository()).unwrap().page;
        let mut from = 0;
        for entry in ENTRIES {
            let heading = page[from..]
                .find(&format!("\n## {}\n", entry.title))
                .map(|i| i + from)
                .unwrap_or_else(|| panic!("{}", entry.title));
            let code = page[heading..]
                .find(&format!("```modelspec title=\"{}\"", entry.program.file))
                .map(|i| i + heading)
                .unwrap();
            let diagram = page[code..]
                .find(&format!(
                    "<ModelDiagram data={{{}}} view=\"model\" />",
                    identifier(entry.program.key)
                ))
                .map(|i| i + code)
                .unwrap();
            from = diagram;
        }
        assert_eq!(identifier("layer-dense-bias-gelu"), "layerDenseBiasGelu");
    }
}
