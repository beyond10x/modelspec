//! Adversary cases for `optimizer = adamw(...)` in the checker and the printer
//! (story:bert-adamw-warmup, DESIGN.md § 2.1, coordinator decision D50).
//!
//! Each argument at its boundary, with the code and the span the design names, the forms of
//! argument list the design refuses, and the printer's round trip of an `adamw(...)` call.

use modelspec_core::compile::compile_training;
use modelspec_core::diag::Diagnostic;
use modelspec_core::ir::training::OptimizerRule;
use modelspec_core::syntax::{parse, print};

const FULL: &str =
    "lr = 0.05, weight_decay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, warmup_steps = 10";

/// A linear regression whose optimizer is `adamw(<args>)`, `steps` steps; the optimizer is on
/// line 15, which starts `    optimizer = adamw(`.
fn source(args: &str, steps: u64) -> String {
    format!(
        "model LinearReg {{
    param w: Tensor<f32, [4, 1]> = xavier_uniform();

    fn forward(x: Tensor<f32, [8, 4]>)
        -> Tensor<f32, [8, 1]> {{
        return matmul(x, w);
    }}
}}

train Fit {{
    model = LinearReg;
    input x: Tensor<f32, [8, 4]>;
    target y: Tensor<f32, [8, 1]>;
    loss = mse(model.forward(x), y);
    optimizer = adamw({args});
    steps = {steps};
    seed = 42;
    device = cpu;
}}
"
    )
}

/// `FULL` with the argument `name` given the value `value`.
fn with(name: &str, value: &str) -> String {
    FULL.split(", ")
        .map(|arg| {
            let (n, _) = arg.split_once(" = ").expect("`name = value`");
            if n == name {
                format!("{n} = {value}")
            } else {
                arg.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// The 1-based column on line 15 where `needle` starts.
fn column_of(args: &str, needle: &str) -> u32 {
    let line = format!("    optimizer = adamw({args});");
    line.find(needle)
        .map(|c| c as u32 + 1)
        .unwrap_or_else(|| panic!("{line} holds {needle}"))
}

fn summary(diagnostics: &[Diagnostic]) -> Vec<(String, u32, u32)> {
    diagnostics
        .iter()
        .map(|d| {
            let span = d.span.as_ref().expect("a checker diagnostic has a span");
            (d.code.as_str().to_owned(), span.line, span.column)
        })
        .collect()
}

fn refused(args: &str, steps: u64) -> Vec<(String, u32, u32)> {
    match compile_training("adamw.mspec", &source(args, steps)) {
        Ok(_) => panic!("adamw({args}) with steps = {steps} compiles"),
        Err(diagnostics) => summary(&diagnostics),
    }
}

fn accepted(args: &str, steps: u64) -> modelspec_core::ir::training::AdamW {
    let compiled = compile_training("adamw.mspec", &source(args, steps))
        .unwrap_or_else(|d| panic!("adamw({args}) is refused: {d:?}"));
    match compiled.training.optimizer.rule {
        OptimizerRule::AdamW(adamw) => adamw,
        other => panic!("{other:?}"),
    }
}

/// Each out-of-range argument, alone, is its code at the argument's name (DESIGN.md § 2.1:
/// "each at the argument").
#[test]
fn adamw_out_of_range_arguments_are_refused_at_the_argument() {
    let cases = [
        ("beta1", "1.0", "E-TYPE-023"),
        ("beta1", "1.5", "E-TYPE-023"),
        ("beta1", "-0.1", "E-TYPE-023"),
        ("beta2", "1.0", "E-TYPE-023"),
        ("beta2", "0.99999999", "E-TYPE-023"),
        ("eps", "0.0", "E-TYPE-023"),
        ("eps", "-1e-8", "E-TYPE-023"),
        ("eps", "1e-46", "E-TYPE-023"),
        ("weight_decay", "-0.1", "E-TYPE-023"),
        ("weight_decay", "1e39", "E-TYPE-023"),
        ("lr", "0.0", "E-TYPE-010"),
        ("lr", "-0.05", "E-TYPE-010"),
        ("lr", "1e39", "E-TYPE-010"),
        ("lr", "1", "E-TYPE-019"),
        ("warmup_steps", "1.0", "E-TYPE-019"),
        ("warmup_steps", "x", "E-TYPE-019"),
        ("warmup_steps", "200", "E-TYPE-024"),
        ("warmup_steps", "201", "E-TYPE-024"),
        ("warmup_steps", "18446744073709551615", "E-TYPE-024"),
    ];
    for (name, value, code) in cases {
        let args = with(name, value);
        let column = column_of(&args, &format!("{name} ="));
        assert_eq!(
            refused(&args, 200),
            vec![(code.to_owned(), 15, column)],
            "adamw({args})"
        );
    }
}

/// The boundaries the design admits: a beta of 0, `beta2 = 0.999999`, a subnormal `eps`, no
/// weight decay, no warm-up and a warm-up one step short of the run.
#[test]
fn adamw_in_range_boundaries_are_accepted() {
    assert_eq!(accepted(&with("beta1", "0.0"), 200).beta1, 0.0);
    assert_eq!(accepted(&with("beta2", "0.999999"), 200).beta2, 0.999999f32);
    let eps = accepted(&with("eps", "1e-45"), 200).eps;
    assert!(eps > 0.0 && eps.is_subnormal(), "{eps:e}");
    assert_eq!(
        accepted(&with("weight_decay", "0.0"), 200).weight_decay,
        0.0
    );
    assert_eq!(accepted(&with("warmup_steps", "0"), 200).warmup_steps, 0);
    assert_eq!(
        accepted(&with("warmup_steps", "199"), 200).warmup_steps,
        199
    );
    assert_eq!(accepted(&with("warmup_steps", "0"), 1).warmup_steps, 0);
}

/// The shape of the argument list: one twice, one unknown, one positional, none at all, one
/// missing; each with the code and the place the design names.
#[test]
fn adamw_argument_list_shapes() {
    let twice = format!("{FULL}, lr = 0.05");
    let at = column_of(&twice, ", lr = 0.05") + 2;
    assert_eq!(
        refused(&twice, 200),
        vec![("E-TYPE-002".to_owned(), 15, at)]
    );

    let extra = format!("{FULL}, momentum = 0.9");
    let at = column_of(&extra, "momentum =");
    assert_eq!(
        refused(&extra, 200),
        vec![("E-TYPE-002".to_owned(), 15, at)]
    );

    let call = column_of("", "adamw(");
    assert_eq!(refused("", 200), vec![("E-TYPE-001".to_owned(), 15, call)]);

    let missing = FULL.replace(", beta2 = 0.999", "");
    assert_eq!(
        refused(&missing, 200),
        vec![("E-TYPE-001".to_owned(), 15, call)]
    );

    let positional = FULL.replacen("lr = 0.05", "0.05", 1);
    let got = refused(&positional, 200);
    assert!(
        got.iter().any(|(code, line, column)| code == "E-TYPE-002"
            && *line == 15
            && *column == column_of(&positional, "0.05")),
        "{got:?}"
    );
}

/// `steps = 0` is E-TYPE-011 alone, not E-TYPE-024 as well (the checker's own doc comment).
#[test]
fn adamw_zero_steps_is_e_type_011_alone() {
    let got = refused(&with("warmup_steps", "0"), 0);
    let codes: Vec<&str> = got.iter().map(|(c, _, _)| c.as_str()).collect();
    assert_eq!(codes, ["E-TYPE-011"], "{got:?}");
}

/// The host constants of every step of a long run are D50's: `lr_t`, `c1` and `c2` each computed
/// in `f64` and rounded once to `f32`, for peak rates whose products with `t` are not exact in
/// `f32` (so a rate rounded to `f32` before its division shows).
#[test]
fn adamw_scalars_round_once_over_a_long_run() {
    use modelspec_core::ir::training::AdamW;
    for (lr, warmup_steps, steps) in [(0.1f32, 7u64, 2000u64), (3e-4, 100, 2000), (0.7, 13, 500)] {
        let adamw = AdamW {
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            warmup_steps,
        };
        for t in 1..=steps {
            let peak = f64::from(lr);
            let rate = if t <= warmup_steps {
                peak * t as f64 / warmup_steps as f64
            } else {
                peak * (steps - t) as f64 / (steps - warmup_steps) as f64
            };
            let c = |beta: f32| (1.0 / (1.0 - f64::from(beta).powi(t as i32))) as f32;
            let got = adamw.scalars(lr, t, steps);
            assert_eq!(
                (got.lr_t.to_bits(), got.c1.to_bits(), got.c2.to_bits()),
                (
                    (rate as f32).to_bits(),
                    c(0.9).to_bits(),
                    c(0.999).to_bits()
                ),
                "lr {lr}, warmup {warmup_steps}, steps {steps}, t {t}: {got:?}"
            );
        }
    }
}

/// The printer writes an `adamw(...)` call that parses back to the same tree, for literals at
/// the edges of their kind.
#[test]
fn adamw_printer_round_trips() {
    for args in [
        FULL.to_owned(),
        "warmup_steps = 0, eps = 1e-45, beta2 = 0.999999, beta1 = 0.0, weight_decay = 0.0, lr = 3.0e-4"
            .to_owned(),
        with("warmup_steps", "18446744073709551615"),
    ] {
        let text = source(&args, 200);
        let mut first = parse("a.mspec", &text).unwrap_or_else(|d| panic!("{d:?}"));
        let printed = print(&first);
        assert!(printed.contains("optimizer = adamw("), "{printed}");
        let mut second = parse("a.mspec", &printed).unwrap_or_else(|d| panic!("{printed}\n{d:?}"));
        first.erase_spans();
        second.erase_spans();
        assert_eq!(first, second, "{printed}");
    }
}
