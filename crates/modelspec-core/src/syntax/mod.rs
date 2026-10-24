//! Surface syntax of `.mspec` source: tokens, parser and AST (DESIGN.md § 2).
//!
//! [`parse`] is pass 1 of DESIGN.md § 3.3 for the M0 subset; [`print`] is the canonical printer
//! used by the round-trip tests of DESIGN.md § 9 gate 1.

pub mod ast;
pub mod lexer;
pub mod parser;

use std::fmt::Write as _;

use crate::diag::Diagnostic;
use ast::{
    Arg, Ast, Call, DataRole, Device, Expr, ExprKind, Function, ModelItem, Optimizer, Param,
    TensorType,
};

/// Syntax diagnostic codes. A construct outside the M0 subset has its own code.
pub mod codes {
    use crate::diag::Code;

    macro_rules! declare {
        ($($(#[doc = $doc:literal])* $name:ident = $text:literal;)*) => {
            $(
                $(#[doc = $doc])*
                pub const $name: Code = Code::new($text);
            )*

            /// Every syntax code, in numeric order.
            pub const ALL: &[Code] = &[$($name),*];
        };
    }

    declare! {
        /// A character no token starts with.
        UNRECOGNISED_CHARACTER = "E-SYNTAX-001";
        /// A token the grammar does not allow at this position.
        UNEXPECTED_TOKEN = "E-SYNTAX-002";
        /// The source ends inside an item.
        UNEXPECTED_END = "E-SYNTAX-003";
        /// `module`, which arrives with DESIGN.md § 11 iteration 3.
        MODULE = "E-SYNTAX-004";
        /// `layer`, which arrives with DESIGN.md § 11 iteration 3.
        LAYER = "E-SYNTAX-005";
        /// A generic parameter list such as `<B: dim>` after a model, train or function name.
        GENERIC_PARAMETERS = "E-SYNTAX-006";
        /// Generic arguments such as `TinyMLP<64>` in a train block's `model` setting.
        GENERIC_ARGUMENTS = "E-SYNTAX-007";
        /// A named dimension such as `B` in a tensor type; M0 dimensions are integer literals.
        SYMBOLIC_DIMENSION = "E-SYNTAX-008";
        /// An infix or prefix operator; M0 spells arithmetic as calls such as `add(a, b)`.
        OPERATOR = "E-SYNTAX-009";
        /// `const`, which arrives with DESIGN.md § 11 iteration 3.
        CONST = "E-SYNTAX-010";
        /// `repeat`, which arrives with DESIGN.md § 11 iteration 3.
        REPEAT = "E-SYNTAX-011";
        /// An element type other than `f32` or `i32`.
        DTYPE = "E-SYNTAX-012";
        /// An optimizer other than `sgd`.
        OPTIMIZER = "E-SYNTAX-013";
        /// A device other than `cpu` or `vulkan(<index>)`.
        DEVICE = "E-SYNTAX-014";
        /// A numeric literal its position cannot hold.
        LITERAL_OUT_OF_RANGE = "E-SYNTAX-015";
        /// A train setting name the M0 subset does not have.
        UNKNOWN_TRAIN_SETTING = "E-SYNTAX-016";
        /// A train setting given twice.
        DUPLICATE_TRAIN_SETTING = "E-SYNTAX-017";
        /// A train block without one of its required settings.
        MISSING_TRAIN_SETTING = "E-SYNTAX-018";
        /// A second `model` or `train` item in one file.
        SECOND_ITEM = "E-SYNTAX-019";
        /// A file without a `model` or a `train` item.
        MISSING_ITEM = "E-SYNTAX-020";
        /// Argument lists nested more than [`MAX_NESTING`](super::MAX_NESTING) deep.
        NESTING_TOO_DEEP = "E-SYNTAX-021";
    }
}

/// How deep argument lists may nest. Every recursive path of the parser passes through an
/// argument list, so this bounds its recursion. A debug build parses this depth with a quarter of
/// a 2 MiB thread stack, the default for spawned threads; about 250 levels fill it.
pub const MAX_NESTING: usize = 64;

/// Parses one `.mspec` source file of the M0 subset. `file` names the source in diagnostic
/// spans. Each top-level item reports at most its first syntax error.
pub fn parse(file: &str, source: &str) -> Result<Ast, Vec<Diagnostic>> {
    parser::parse(file, source)
}

/// Prints `ast` in canonical form: four-space indentation, one blank line between model items,
/// the model before the train block, and the train settings in a fixed order. Parsing the output
/// yields an AST equal to `ast` up to spans.
pub fn print(ast: &Ast) -> String {
    let mut out = String::new();
    let model = &ast.model;
    writeln!(out, "model {} {{", model.name.name).unwrap();
    for (i, item) in model.items.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        match item {
            ModelItem::Param(param) => print_param(&mut out, param),
            ModelItem::Fn(function) => print_function(&mut out, function),
        }
    }
    out.push_str("}\n\n");

    let train = &ast.train;
    writeln!(out, "train {} {{", train.name.name).unwrap();
    writeln!(out, "    model = {};", train.model.value.name).unwrap();
    for decl in &train.data {
        let role = match decl.role {
            DataRole::Input => "input",
            DataRole::Target => "target",
        };
        writeln!(out, "    {role} {}: {};", decl.name.name, ty(&decl.ty)).unwrap();
    }
    writeln!(out, "    loss = {};", expr(&train.loss.value)).unwrap();
    match &train.optimizer.value {
        Optimizer::Sgd { lr, .. } => writeln!(out, "    optimizer = sgd(lr = {lr:?});").unwrap(),
    }
    writeln!(out, "    steps = {};", train.steps.value).unwrap();
    writeln!(out, "    seed = {};", train.seed.value).unwrap();
    match &train.device.value {
        Device::Cpu => out.push_str("    device = cpu;\n"),
        Device::Vulkan { index, .. } => writeln!(out, "    device = vulkan({index});").unwrap(),
    }
    out.push_str("}\n");
    out
}

fn print_param(out: &mut String, param: &Param) {
    writeln!(
        out,
        "    param {}: {} = {};",
        param.name.name,
        ty(&param.ty),
        call(&param.init)
    )
    .unwrap();
}

fn print_function(out: &mut String, function: &Function) {
    let params: Vec<String> = function
        .params
        .iter()
        .map(|p| format!("{}: {}", p.name.name, ty(&p.ty)))
        .collect();
    writeln!(
        out,
        "    fn {}({}) -> {} {{",
        function.name.name,
        params.join(", "),
        ty(&function.output)
    )
    .unwrap();
    for binding in &function.lets {
        writeln!(
            out,
            "        let {} = {};",
            binding.name.name,
            expr(&binding.value)
        )
        .unwrap();
    }
    writeln!(out, "        return {};", expr(&function.result.value)).unwrap();
    out.push_str("    }\n");
}

fn ty(t: &TensorType) -> String {
    let dtype = match t.dtype.kind {
        ast::DType::F32 => "f32",
        ast::DType::I32 => "i32",
    };
    let dims: Vec<String> = t.dims.iter().map(|d| d.size.to_string()).collect();
    format!("Tensor<{dtype}, [{}]>", dims.join(", "))
}

fn expr(e: &Expr) -> String {
    match &e.kind {
        ExprKind::Var(name) => name.clone(),
        ExprKind::Int(value) => value.to_string(),
        ExprKind::Float(value) => format!("{value:?}"),
        ExprKind::Call(c) => call(c),
        ExprKind::MemberCall(m) => {
            format!("{}.{}({})", m.receiver.name, m.method.name, args(&m.args))
        }
    }
}

fn call(c: &Call) -> String {
    format!("{}({})", c.callee.name, args(&c.args))
}

fn args(list: &[Arg]) -> String {
    list.iter()
        .map(|a| match &a.name {
            Some(name) => format!("{} = {}", name.name, expr(&a.value)),
            None => expr(&a.value),
        })
        .collect::<Vec<_>>()
        .join(", ")
}
