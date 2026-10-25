//! Parameter initializers (DESIGN.md § 2.3; coordinator decision D33): `xavier_uniform()` takes
//! a rank-2 parameter, `zeros()` and `ones()` a parameter of any rank from 1, each without
//! arguments. Any other initializer is `E-TYPE-013`, and an initializer on a parameter of a rank
//! it does not take is `E-TYPE-014`, both at the initializer.

use modelspec_core::diag::Diagnostic;
use modelspec_core::ir::tensor::{Initializer, TensorType};
use modelspec_core::types::check_source;

const TRAIN: &str = "
train Fit {
    model = M;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 3]>;
    loss = mse(model.forward(x), y);
    optimizer = sgd(lr = 0.1);
    steps = 3;
    seed = 7;
    device = cpu;
}
";

/// A model `M` whose parameter `w` (line 2) is followed by `params`, from line 3 on.
fn source(params: &str) -> String {
    format!(
        "model M {{\n    param w: Tensor<f32, [4, 3]> = xavier_uniform();\n{params}\n\n    \
         fn forward(x: Tensor<f32, [8, 4]>) -> Tensor<f32, [8, 3]> {{\n        \
         return matmul(x, w);\n    }}\n}}\n{TRAIN}"
    )
}

/// The diagnostics of `params` in [`source`].
fn refused(params: &str) -> Vec<Diagnostic> {
    match check_source("m.mspec", &source(params)) {
        Ok(_) => panic!("expected diagnostics for:\n{params}"),
        Err(diagnostics) => diagnostics,
    }
}

/// `(code, line, column)` of a diagnostic.
fn at(d: &Diagnostic) -> (&str, u32, u32) {
    let span = d.span.as_ref().expect("a source diagnostic has a span");
    (d.code.as_str(), span.line, span.column)
}

/// The column of `word` in the line `line` of `params`, 1-based, for a diagnostic at line
/// `line + 3` of [`source`].
fn column(params: &str, line: usize, word: &str) -> u32 {
    let text = params.lines().nth(line).expect("the line exists");
    text.find(word).expect("the word is on the line") as u32 + 1
}

#[test]
fn zeros_and_ones_take_a_parameter_of_rank_1_2_and_3() {
    let params = "    param b: Tensor<f32, [3]> = zeros();
    param g: Tensor<f32, [8, 3]> = ones();
    param c: Tensor<f32, [2, 2, 2]> = zeros();
    param o: Tensor<f32, [5]> = ones();";
    let hir = check_source("m.mspec", &source(params)).unwrap_or_else(|d| {
        panic!(
            "expected a clean check:\n{}",
            d.iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        )
    });
    let inits: Vec<(&str, &TensorType, Initializer)> = hir
        .model
        .params
        .iter()
        .map(|p| (p.path.as_str(), &p.ty, p.init))
        .collect();
    assert_eq!(
        inits,
        [
            ("M.w", &TensorType::f32(&[4, 3]), Initializer::XavierUniform),
            ("M.b", &TensorType::f32(&[3]), Initializer::Zeros),
            ("M.g", &TensorType::f32(&[8, 3]), Initializer::Ones),
            ("M.c", &TensorType::f32(&[2, 2, 2]), Initializer::Zeros),
            ("M.o", &TensorType::f32(&[5]), Initializer::Ones),
        ]
    );
}

#[test]
fn zeros_and_ones_on_a_rank_0_parameter_are_e_type_014_naming_their_rule() {
    let params = "    param s: Tensor<f32, []> = zeros();
    param t: Tensor<f32, []> = ones();";
    let diagnostics = refused(params);
    let found: Vec<_> = diagnostics.iter().map(at).collect();
    assert_eq!(
        found,
        [
            ("E-TYPE-014", 3, column(params, 0, "zeros")),
            ("E-TYPE-014", 4, column(params, 1, "ones")),
        ]
    );
    assert_eq!(
        diagnostics[0].message,
        "`zeros()` needs a parameter of rank 1 or more; `s` has type Tensor<f32, []>"
    );
    assert_eq!(
        diagnostics[1].message,
        "`ones()` needs a parameter of rank 1 or more; `t` has type Tensor<f32, []>"
    );
    let text = diagnostics[0].to_string();
    assert!(
        text.contains("`xavier_uniform()` needs rank 2"),
        "the note names the other rule:\n{text}"
    );
}

#[test]
fn xavier_uniform_on_rank_1_is_still_e_type_014_and_names_the_rule_of_zeros_and_ones() {
    let params = "    param b: Tensor<f32, [3]> = xavier_uniform();";
    let diagnostics = refused(params);
    let found: Vec<_> = diagnostics.iter().map(at).collect();
    assert_eq!(
        found,
        [("E-TYPE-014", 3, column(params, 0, "xavier_uniform"))]
    );
    assert_eq!(
        diagnostics[0].message,
        "`xavier_uniform()` needs a rank-2 parameter; `b` has type Tensor<f32, [3]>"
    );
    let text = diagnostics[0].to_string();
    assert!(
        text.contains("`zeros()` and `ones()` take a parameter of any rank from 1"),
        "the note names the rule of zeros and ones:\n{text}"
    );
}

#[test]
fn zeros_and_ones_with_arguments_and_an_unknown_initializer_are_e_type_013() {
    let params = "    param b: Tensor<f32, [3]> = zeros(1.0);
    param g: Tensor<f32, [3]> = ones(0.0);
    param n: Tensor<f32, [3]> = normal();";
    let diagnostics = refused(params);
    let found: Vec<_> = diagnostics.iter().map(at).collect();
    assert_eq!(
        found,
        [
            ("E-TYPE-013", 3, column(params, 0, "zeros")),
            ("E-TYPE-013", 4, column(params, 1, "ones")),
            ("E-TYPE-013", 5, column(params, 2, "normal")),
        ]
    );
    assert_eq!(
        diagnostics[2].message,
        "initializer `normal(...)` is not supported; the initializers are `xavier_uniform()`, \
         `zeros()` and `ones()`"
    );
}

#[test]
fn the_initializers_print_by_their_source_names() {
    let names: Vec<String> = Initializer::ALL.iter().map(|i| i.to_string()).collect();
    assert_eq!(names, ["xavier_uniform", "zeros", "ones"]);
    for init in Initializer::ALL {
        assert_eq!(Initializer::from_name(&init.to_string()), Some(init));
    }
    assert_eq!(Initializer::from_name("normal"), None);
}
