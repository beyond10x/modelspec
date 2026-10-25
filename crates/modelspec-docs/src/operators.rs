//! The operator reference, from the operator registry of `modelspec-core`.
//!
//! The operators a source file may call are `modelspec_core::resolve::OPERATORS`; their operand
//! counts are `ops::arity`, and every example on the page is typed by `ops::infer`, the function
//! the type checker, the IR builder and the verifier all use. An accepted example that the
//! registry refuses, or a refused one it accepts, fails generation. The gradient table is the
//! table in the doc comment of `ops::gradient`.

use std::fs;
use std::path::Path;

use modelspec_core::ir::tensor::{DType, TensorType};
use modelspec_core::ops;
use modelspec_core::resolve::OPERATORS;

use crate::{Result, SOURCE, cell, front_matter, io};

/// The file that holds the registry and the gradient rules, relative to the repository root.
const OPS: &str = "crates/modelspec-core/src/ops.rs";

/// Example operand shapes per source operator: some the rule accepts, then some it refuses.
struct Examples {
    operator: &'static str,
    accepted: &'static [&'static [&'static [u64]]],
    refused: &'static [&'static [&'static [u64]]],
}

const EXAMPLES: &[Examples] = &[
    Examples {
        operator: "matmul",
        accepted: &[&[&[8, 4], &[4, 1]]],
        refused: &[&[&[8, 4], &[3, 1]], &[&[8], &[8, 1]]],
    },
    Examples {
        operator: "add",
        accepted: &[&[&[8, 1], &[8, 1]], &[&[8, 1], &[]]],
        refused: &[&[&[8, 1], &[1, 8]]],
    },
    Examples {
        operator: "sub",
        accepted: &[&[&[8, 1], &[8, 1]], &[&[], &[8, 1]]],
        refused: &[&[&[8, 1], &[8, 4]]],
    },
    Examples {
        operator: "mul",
        accepted: &[&[&[8, 4], &[8, 4]], &[&[8, 4], &[]]],
        refused: &[&[&[8, 4], &[4, 8]]],
    },
    Examples {
        operator: "relu",
        accepted: &[&[&[8, 4]]],
        refused: &[],
    },
    Examples {
        operator: "transpose",
        accepted: &[&[&[8, 4]]],
        refused: &[&[&[8]]],
    },
    Examples {
        operator: "sum",
        accepted: &[&[&[8, 4]]],
        refused: &[],
    },
    Examples {
        operator: "mean",
        accepted: &[&[&[8, 4]]],
        refused: &[],
    },
    Examples {
        operator: "mse",
        accepted: &[&[&[8, 1], &[8, 1]]],
        refused: &[&[&[8, 1], &[1, 8]], &[&[8, 1], &[]]],
    },
    Examples {
        operator: "embedding",
        accepted: &[&[&[7, 3], &[10]]],
        refused: &[&[&[7], &[10]], &[&[7, 3], &[2, 5]]],
    },
    Examples {
        operator: "add_rows",
        accepted: &[&[&[8, 4], &[4]]],
        refused: &[&[&[8, 4], &[5]], &[&[4], &[4]]],
    },
    Examples {
        operator: "gelu",
        accepted: &[&[&[8, 4]], &[&[]]],
        refused: &[],
    },
    Examples {
        operator: "layer_norm",
        accepted: &[&[&[8, 4], &[4], &[4]]],
        refused: &[&[&[8, 4], &[5], &[4]], &[&[4], &[4], &[4]]],
    },
    Examples {
        operator: "softmax",
        accepted: &[&[&[8, 4]]],
        refused: &[&[&[4]]],
    },
];

/// `name(Tensor<...>, ...)`, as a call in source would read with typed operands.
fn call(name: &str, operands: &[TensorType]) -> String {
    let operands: Vec<String> = operands.iter().map(ToString::to_string).collect();
    format!("{name}({})", operands.join(", "))
}

/// `message` as code spans split after each `: ` and `, ` outside `<>`, `[]` and `()`, so a table
/// cell can wrap between the parts of a long refusal.
fn wrappable(message: &str) -> String {
    let mut parts: Vec<(String, &str)> = Vec::new();
    let mut current = String::new();
    let mut depth = 0usize;
    let mut chars = message.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '<' | '[' | '(' => depth += 1,
            '>' | ']' | ')' => depth = depth.saturating_sub(1),
            _ => {}
        }
        if depth == 0 && matches!(c, ':' | ',') && chars.peek() == Some(&' ') {
            chars.next();
            let separator = if c == ':' { ": " } else { ", " };
            parts.push((std::mem::take(&mut current), separator));
        } else {
            current.push(c);
        }
    }
    parts.push((current, ""));
    parts
        .into_iter()
        .filter(|(part, _)| !part.is_empty())
        .map(|(part, separator)| format!("`{part}`{separator}"))
        .collect::<String>()
        .trim_end()
        .to_owned()
}

/// The registry's answer for `op` on operands of `shapes`, each of the dtype the registry gives
/// that operand (`ops::operand_dtype`).
fn typed(
    op: &modelspec_core::ir::tensor::TensorOp,
    shapes: &[&[u64]],
) -> (Vec<TensorType>, std::result::Result<TensorType, String>) {
    let operands: Vec<TensorType> = shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| match ops::operand_dtype(op, index) {
            DType::F32 => TensorType::f32(shape),
            DType::I32 => TensorType::i32(shape),
        })
        .collect();
    let refs: Vec<&TensorType> = operands.iter().collect();
    let result = ops::infer(op, &refs, None).map_err(|e| e.to_string());
    (operands, result)
}

/// The rows of the gradient table in the doc comment of `ops::gradient`.
fn gradient_table(source: &str) -> Result<Vec<String>> {
    let rows: Vec<String> = source
        .lines()
        .map(str::trim)
        .filter_map(|line| line.strip_prefix("/// |"))
        .map(|row| format!("|{row}"))
        .collect();
    if rows.len() < 3 {
        return Err(format!("{OPS} holds no gradient table"));
    }
    Ok(rows)
}

pub fn page(root: &Path) -> Result<String> {
    for examples in EXAMPLES {
        if !OPERATORS.iter().any(|(name, _)| *name == examples.operator) {
            return Err(format!(
                "operator examples name {}, which the registry does not have",
                examples.operator
            ));
        }
    }
    let mut out = front_matter(
        "Operators",
        "Operators",
        "The tensor operators a model may call in M0, their shape rules, and their gradients, from the operator registry.",
        3,
    );
    out.push_str(&format!(
        "M0 has {} operators. A model calls them by name with positional arguments; there are no \
         infix operators yet, so `a + b` is written `add(a, b)`. Every operand is `f32` except \
         the ids of `embedding`, its second operand, which are `i32`.\n\n\
         Generated by `modelspec-docs` from the operator registry in \
         [`{OPS}`]({SOURCE}/{OPS}): the type checker, the IR builder and the verifier all type an \
         operation through the one function `ops::infer`, and every example below is its answer.\n\n\
         ## Summary\n\n| Operator | Tensor IR op | Operands | Example | Result |\n|---|---|---|---|---|\n",
        OPERATORS.len()
    ));
    let mut details = String::new();
    for (name, op) in &OPERATORS {
        let examples = EXAMPLES
            .iter()
            .find(|examples| examples.operator == *name)
            .ok_or_else(|| format!("operator {name} has no example on the operator page"))?;
        let mut first = None;
        details.push_str(&format!(
            "\n## `{name}`\n\nTensor IR op `{}`, {} operand(s).\n\n| Call | Result |\n|---|---|\n",
            op.name(),
            ops::arity(op)
        ));
        for shapes in examples.accepted {
            let (operands, result) = typed(op, shapes);
            let result = result.map_err(|e| {
                format!(
                    "{name}: the page shows {} as accepted, the registry refuses it: {e}",
                    call(name, &operands)
                )
            })?;
            first.get_or_insert_with(|| (call(name, &operands), result.to_string()));
            details.push_str(&format!("| `{}` | `{result}` |\n", call(name, &operands)));
        }
        for shapes in examples.refused {
            let (operands, result) = typed(op, shapes);
            let Err(error) = result else {
                return Err(format!(
                    "{name}: the page shows {} as refused, the registry accepts it",
                    call(name, &operands)
                ));
            };
            details.push_str(&format!(
                "| `{}` | refused: {} |\n",
                call(name, &operands),
                wrappable(&cell(&error))
            ));
        }
        let (example, result) = first.ok_or_else(|| format!("{name}: no accepted example"))?;
        out.push_str(&format!(
            "| [`{name}`](#{name}) | `{}` | {} | `{example}` | `{result}` |\n",
            op.name(),
            ops::arity(op)
        ));
    }
    out.push_str(
        "\nThe type checker reports a refused call as a diagnostic: `E-TYPE-012` for a `matmul` \
         contracting dimension, `E-TYPE-003` for operands of different shapes, `E-TYPE-004` for \
         a rank, `E-TYPE-001` for an argument count, `E-TYPE-018` for an operand of the wrong \
         element type ([Diagnostics](diagnostics.md)).\n",
    );
    out.push_str(&details);
    let source = fs::read_to_string(root.join(OPS)).map_err(io(OPS))?;
    out.push_str(
        "\n## Gradients\n\nThe reverse-mode rule of each op, for `Y = op(A, B, C)` and `G = dL/dY`, \
         from the doc comment of `ops::gradient`. `modelspec-runtime`'s gradient tests check every \
         rule against central finite differences.\n\n",
    );
    for row in gradient_table(&source)? {
        out.push_str(&row);
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registry_operator_is_on_the_page_with_its_rule() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let page = page(&root).unwrap();
        for (name, _) in &OPERATORS {
            assert!(page.contains(&format!("\n## `{name}`\n")), "{name}");
        }
        assert!(page.contains(
            "| `matmul(Tensor<f32, [8, 4]>, Tensor<f32, [4, 1]>)` | `Tensor<f32, [8, 1]>` |"
        ));
        assert!(page.contains(
            "refused: `matmul contracting dimension mismatch`: `lhs Tensor<f32, [8, 4]>`, \
             `rhs Tensor<f32, [3, 1]>`, `expected rhs dimension[0] = 4; got 3` |"
        ));
        assert!(page.contains("| `MatMul` | `G @ transpose(B)` | `transpose(A) @ G` |"));
    }

    /// `embedding` takes `i32` ids: its examples are typed with the dtype the registry gives each
    /// operand, and its gradient row is on the page.
    #[test]
    fn embedding_is_on_the_page_with_i32_ids() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let page = page(&root).unwrap();
        assert!(page.contains("\n## `embedding`\n"));
        assert!(page.contains(
            "| `embedding(Tensor<f32, [7, 3]>, Tensor<i32, [10]>)` | `Tensor<f32, [10, 3]>` |"
        ));
        assert!(page.contains("| `Embedding` |"));
    }

    /// `add_rows` and `gelu` are on the page with an accepted and a refused call each where the
    /// rule refuses one, and the gradient rows of `AddRows` and `Gelu` are in the table.
    #[test]
    fn add_rows_and_gelu_are_on_the_page_with_their_gradients() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let page = page(&root).unwrap();
        assert!(page.contains("\n## `add_rows`\n"));
        assert!(page.contains(
            "| `add_rows(Tensor<f32, [8, 4]>, Tensor<f32, [4]>)` | `Tensor<f32, [8, 4]>` |"
        ));
        assert!(page.contains("| `add_rows(Tensor<f32, [8, 4]>, Tensor<f32, [5]>)` | refused: "));
        assert!(page.contains("\n## `gelu`\n"));
        assert!(page.contains("| `gelu(Tensor<f32, [8, 4]>)` | `Tensor<f32, [8, 4]>` |"));
        assert!(page.contains("| `AddRows` | `G` | `sum_rows(G)`"));
        assert!(page.contains("| `Gelu` | `gelu_grad(A, G)`"));
    }

    /// `layer_norm` and `softmax` are on the page with an accepted and a refused call each, and
    /// the gradient rows of `LayerNorm` and `Softmax` are in the table, `LayerNorm`'s with its
    /// third operand.
    #[test]
    fn layer_norm_and_softmax_are_on_the_page_with_their_gradients() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let page = page(&root).unwrap();
        assert!(page.contains("\n## `layer_norm`\n"));
        assert!(page.contains(
            "| `layer_norm(Tensor<f32, [8, 4]>, Tensor<f32, [4]>, Tensor<f32, [4]>)` | \
             `Tensor<f32, [8, 4]>` |"
        ));
        assert!(page.contains(
            "| `layer_norm(Tensor<f32, [8, 4]>, Tensor<f32, [5]>, Tensor<f32, [4]>)` | refused: "
        ));
        assert!(page.contains("\n## `softmax`\n"));
        assert!(page.contains("| `softmax(Tensor<f32, [8, 4]>)` | `Tensor<f32, [8, 4]>` |"));
        assert!(page.contains("| `softmax(Tensor<f32, [4]>)` | refused: "));
        let row = page
            .lines()
            .find(|l| l.starts_with("| `LayerNorm` | `layer_norm_grad_x(A, B, G)` ("))
            .expect("the LayerNorm gradient row");
        let cells: Vec<&str> = row.split(" | ").collect();
        assert_eq!(cells.len(), 4, "op, dA, dB and dC: {row}");
        assert!(
            cells[2].starts_with("`layer_norm_grad_gamma(A, G)` ("),
            "{row}"
        );
        assert_eq!(cells[3], "`sum_rows(G)` |", "{row}");
        assert!(page.contains("| `Softmax` | `softmax_grad(Y, G)`"));
    }

    #[test]
    fn a_refusal_splits_only_outside_brackets() {
        assert_eq!(
            wrappable("A differ: lhs Tensor<f32, [8, 1]>, rhs X"),
            "`A differ`: `lhs Tensor<f32, [8, 1]>`, `rhs X`"
        );
        assert_eq!(
            wrappable("needs rank 2; got T<f32, [8]>"),
            "`needs rank 2; got T<f32, [8]>`"
        );
    }

    #[test]
    fn the_gradient_table_needs_a_header_a_rule_and_a_row() {
        assert!(gradient_table("/// | op |\n").is_err());
        assert_eq!(
            gradient_table("/// | op |\n/// | --- |\n/// | `Add` |\nfn x() {}\n")
                .unwrap()
                .len(),
            3
        );
    }
}
