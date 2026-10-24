//! Adversary cases, pass 2, for story:parser-span-lookup-not-linear-in-column: what the gated
//! counter `span_lookup_scan_count` measures. Every lookup the parser makes lands on an anchor, so
//! the count is the anchor pass alone, one per character of the source, whatever the layout and
//! whatever diagnostics are reported. The gate's 1.2x bound therefore compares source lengths.

use modelspec_core::syntax::parser::span_lookup_scan_count;

fn program(lets: usize, separator: &str, bogus_before_return: bool) -> String {
    let mut body = String::new();
    for i in 0..lets {
        body.push_str(&format!("let a{i} = x;"));
        body.push_str(separator);
    }
    if bogus_before_return {
        body.push_str("bogus \u{e9} ");
    }
    format!(
        "model M {{\n    param w: Tensor<f32, [2, 1]> = xavier_uniform();\n    fn forward(x: Tensor<f32, [3, 2]>) -> Tensor<f32, [3, 1]> {{\n{body}return matmul(x, w);\n    }}\n}}\ntrain T {{\n    model = M;\n    input x: Tensor<f32, [3, 2]>;\n    target y: Tensor<f32, [3, 1]>;\n    loss = mse(model.forward(x), y);\n    optimizer = sgd(lr = 0.1);\n    steps = 1;\n    seed = 1;\n    device = cpu;\n}}\n"
    )
}

#[test]
fn scan_count_is_the_anchor_pass_alone() {
    for source in [
        program(2_000, " ", false),
        program(2_000, "\n", false),
        program(2_000, " ", true),
        program(2_000, "\n", true),
        "model M { \u{e9} } train T { \u{6f22} }\r\nmodel N {".to_owned(),
        String::new(),
    ] {
        assert_eq!(
            span_lookup_scan_count("test.mspec", &source),
            source.chars().count(),
            "lookups walked past an anchor for {:?}",
            &source[..source.len().min(60)]
        );
    }
}
