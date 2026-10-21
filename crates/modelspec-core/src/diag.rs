//! User-facing diagnostics and their plain-text rendering (DESIGN.md § 8.1).
//!
//! Codes are declared next to the stage that emits them, not here.

use std::fmt;

/// A stable diagnostic code such as `E-TYPE-012`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Code(&'static str);

impl Code {
    /// Declares a code; `code` is the text users see, such as `E-TYPE-012`.
    pub const fn new(code: &'static str) -> Self {
        Self(code)
    }

    /// The code's text.
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl fmt::Display for Code {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

/// A position in a source file; `line` and `column` count from 1.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SourceSpan {
    pub file: String,
    pub line: u32,
    pub column: u32,
}

impl fmt::Display for SourceSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

/// One detail line under a diagnostic, rendered `<label>: <text>`, or `<text>` without a label.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Detail {
    pub label: Option<String>,
    pub text: String,
}

/// A user-facing diagnostic.
///
/// `span` is the primary source position; a diagnostic about the target rather than the source,
/// such as a missing device capability, has none.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: Code,
    pub message: String,
    pub span: Option<SourceSpan>,
    pub details: Vec<Detail>,
}

impl Diagnostic {
    /// A diagnostic with no span and no detail lines.
    pub fn new(code: Code, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            span: None,
            details: Vec::new(),
        }
    }

    /// Sets the primary source span.
    pub fn at(mut self, span: SourceSpan) -> Self {
        self.span = Some(span);
        self
    }

    /// Appends a labelled detail line.
    pub fn with(mut self, label: impl Into<String>, text: impl Into<String>) -> Self {
        self.details.push(Detail {
            label: Some(label.into()),
            text: text.into(),
        });
        self
    }

    /// Appends an unlabelled detail line.
    pub fn note(mut self, text: impl Into<String>) -> Self {
        self.details.push(Detail {
            label: None,
            text: text.into(),
        });
        self
    }
}

/// Renders the layout of DESIGN.md § 8.1: `<code>: <message>`, then the span and each detail
/// line indented by two spaces, without a trailing newline.
impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)?;
        if let Some(span) = &self.span {
            write!(f, "\n  {span}")?;
        }
        for detail in &self.details {
            match &detail.label {
                Some(label) => write!(f, "\n  {label}: {}", detail.text)?,
                None => write!(f, "\n  {}", detail.text)?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const E_TYPE_012: Code = Code::new("E-TYPE-012");
    const E_TARGET_034: Code = Code::new("E-TARGET-034");

    #[test]
    fn renders_source_diagnostic_in_design_layout() {
        let diagnostic = Diagnostic::new(E_TYPE_012, "matmul contracting dimension mismatch")
            .at(SourceSpan {
                file: "model.mspec".to_owned(),
                line: 17,
                column: 18,
            })
            .with("lhs", "Tensor<f32, [64, 32]>")
            .with("rhs", "Tensor<f32, [64, 128]>")
            .note("expected rhs dimension[0] = 32; got 64");
        assert_eq!(
            diagnostic.to_string(),
            "E-TYPE-012: matmul contracting dimension mismatch\n  \
             model.mspec:17:18\n  \
             lhs: Tensor<f32, [64, 32]>\n  \
             rhs: Tensor<f32, [64, 128]>\n  \
             expected rhs dimension[0] = 32; got 64"
        );
    }

    #[test]
    fn renders_spanless_diagnostic_without_position_line() {
        let diagnostic = Diagnostic::new(
            E_TARGET_034,
            "no supported f16 storage/compute path on selected Vulkan device",
        )
        .with("requested", "f16")
        .with("available", "f32")
        .with("suggestion", "compile with --compute=f32");
        assert_eq!(
            diagnostic.to_string(),
            "E-TARGET-034: no supported f16 storage/compute path on selected Vulkan device\n  \
             requested: f16\n  \
             available: f32\n  \
             suggestion: compile with --compute=f32"
        );
    }
}
