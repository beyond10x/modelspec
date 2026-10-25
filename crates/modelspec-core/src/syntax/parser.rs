//! Hand-written recursive-descent parser from tokens to the AST, for the M0 subset.
//!
//! Grammar (one `model` and one `train` item, in either order):
//!
//! ```text
//! file     := (model | train)*
//! model    := "model" IDENT "{" (param | fn)* "}"
//! param    := "param" IDENT ":" type "=" call ";"
//! fn       := "fn" IDENT "(" [IDENT ":" type ("," IDENT ":" type)*] ")" "->" type
//!             "{" ("let" IDENT "=" expr ";")* "return" expr ";" "}"
//! train    := "train" IDENT "{" setting* "}"
//! setting  := "model" "=" IDENT ";" | ("input" | "target") IDENT ":" type ";"
//!           | "loss" "=" expr ";" | "optimizer" "=" optimizer ";"
//!           | "steps" "=" INT ";" | "seed" "=" INT ";"
//!           | "device" "=" ("cpu" | "vulkan" "(" INT ")") ";"
//! optimizer := "sgd" "(" "lr" "=" FLOAT ")" | ("adamw" | "muon") "(" args ")"
//! type     := "Tensor" "<" ("f32" | "i32") "," "[" [INT ("," INT)*] "]" ">"
//! expr     := INT | FLOAT | IDENT | call | (IDENT | "model") "." IDENT "(" args ")"
//! call     := IDENT "(" args ")"
//! args     := [arg ("," arg)*]
//! arg      := [IDENT "="] expr
//! ```
//!
//! After the first error in an item the parser skips to the end of that item, or to the start of
//! the next item, and goes on. Argument lists nest at most [`MAX_NESTING`] deep.

use std::cell::Cell;

use crate::diag::{Code, Diagnostic, SourceSpan};

use super::ast::{
    Arg, Ast, Call, DType, DataDecl, DataRole, Device, Dim, Dtype, Expr, ExprKind, FnParam,
    Function, Ident, Let, MemberCall, Model, ModelItem, Optimizer, Param, Return, Setting,
    TensorType, Train,
};
use super::lexer::{Lexeme, Token, lex};
use super::{MAX_NESTING, codes};

type PResult<T> = Result<T, Diagnostic>;

/// Train settings written `name = value;`, in canonical order.
const SETTINGS: [&str; 6] = ["model", "loss", "optimizer", "steps", "seed", "device"];

pub(crate) fn parse(file: &str, source: &str) -> Result<Ast, Vec<Diagnostic>> {
    Parser::new(file, source).file()
}

/// Test hook: characters the span lookup scans in one parse (`long_line_span_lookup_scans`).
#[doc(hidden)]
pub fn span_lookup_scan_count(file: &str, source: &str) -> usize {
    let mut parser = Parser::new(file, source);
    let _ = parser.file();
    parser.scanned.get()
}

struct Parser<'s> {
    file: &'s str,
    source: &'s str,
    lexemes: Vec<Lexeme>,
    /// Line and column of offset 0, of every lexeme start and of the end of input, in source
    /// order, computed in one pass over the source.
    anchors: Vec<Anchor>,
    /// Characters scanned so far to place anchors and resolve spans; see
    /// [`span_lookup_scan_count`].
    scanned: Cell<usize>,
    pos: usize,
    /// Argument lists open at the current position.
    nesting: usize,
}

/// A byte offset with its line and column, both counted from 1.
#[derive(Clone, Copy)]
struct Anchor {
    offset: usize,
    line: usize,
    column: usize,
}

impl Anchor {
    /// The anchor of `self.offset + text.len()`, where `text` is the source from `self.offset`.
    fn advance(self, text: &str) -> Self {
        let (mut line, mut column) = (self.line, self.column);
        for c in text.chars() {
            if c == '\n' {
                line += 1;
                column = 1;
            } else {
                column += 1;
            }
        }
        Self {
            offset: self.offset + text.len(),
            line,
            column,
        }
    }
}

impl<'s> Parser<'s> {
    fn new(file: &'s str, source: &'s str) -> Self {
        let lexemes = lex(source);
        let mut anchors = Vec::with_capacity(lexemes.len() + 2);
        let mut last = Anchor {
            offset: 0,
            line: 1,
            column: 1,
        };
        anchors.push(last);
        let mut scanned = 0;
        for offset in lexemes
            .iter()
            .map(|l| l.range.start)
            .chain(std::iter::once(source.len()))
        {
            if offset > last.offset {
                let walked = &source[last.offset..offset];
                scanned += walked.chars().count();
                last = last.advance(walked);
                anchors.push(last);
            }
        }
        let scanned = Cell::new(scanned);
        Self {
            file,
            source,
            lexemes,
            anchors,
            scanned,
            pos: 0,
            nesting: 0,
        }
    }

    // -----------------------------------------------------------------------------------------
    // Positions and tokens.
    // -----------------------------------------------------------------------------------------

    /// Line and column of a byte offset; columns count characters from 1. The nearest anchor at
    /// or before `offset` is found by binary search, so the cost does not grow with the column.
    /// Every character of the slice it walks is added to `scanned`, whatever walks it.
    fn span_at(&self, offset: usize) -> SourceSpan {
        let nearest = self.anchors[self.anchors.partition_point(|a| a.offset <= offset) - 1];
        let walked = &self.source[nearest.offset..offset];
        self.scanned
            .set(self.scanned.get() + walked.chars().count());
        let at = nearest.advance(walked);
        SourceSpan {
            file: self.file.to_owned(),
            line: u32::try_from(at.line).unwrap_or(u32::MAX),
            column: u32::try_from(at.column).unwrap_or(u32::MAX),
        }
    }

    /// The span of the current lexeme, or of the end of input.
    fn span(&self) -> SourceSpan {
        let offset = self
            .lexemes
            .get(self.pos)
            .map_or(self.source.len(), |l| l.range.start);
        self.span_at(offset)
    }

    /// The current token; `None` at the end of input or on an unrecognised character.
    fn peek(&self) -> Option<Token> {
        self.peek_at(0)
    }

    fn peek_at(&self, ahead: usize) -> Option<Token> {
        self.lexemes.get(self.pos + ahead).and_then(|l| l.token)
    }

    /// The source text of the current lexeme; empty at the end of input.
    fn text(&self) -> &'s str {
        self.lexemes
            .get(self.pos)
            .map_or("", |l| &self.source[l.range.clone()])
    }

    fn at_word(&self, word: &str) -> bool {
        self.peek() == Some(Token::Ident) && self.text() == word
    }

    fn bump(&mut self) {
        self.pos += 1;
    }

    fn error(&self, code: Code, message: impl Into<String>) -> Diagnostic {
        Diagnostic::new(code, message).at(self.span())
    }

    /// The diagnostic for a current lexeme the grammar does not allow; `expected` names what it
    /// allows.
    fn unexpected(&self, expected: &str) -> Diagnostic {
        match self.lexemes.get(self.pos) {
            None => self.error(
                codes::UNEXPECTED_END,
                format!("unexpected end of input; expected {expected}"),
            ),
            Some(Lexeme { token: None, range }) => {
                let c = self.source[range.start..].chars().next().unwrap_or('?');
                self.error(
                    codes::UNRECOGNISED_CHARACTER,
                    format!("unrecognised character `{c}`"),
                )
            }
            Some(Lexeme { token: Some(_), .. }) => {
                if let Some(diagnostic) = self.outside_subset() {
                    return diagnostic;
                }
                self.error(
                    codes::UNEXPECTED_TOKEN,
                    format!("expected {expected}, found `{}`", self.text()),
                )
            }
        }
    }

    /// The diagnostic for a current token that starts a construct outside the M0 subset.
    fn outside_subset(&self) -> Option<Diagnostic> {
        let token = self.peek()?;
        if token.is_operator() {
            return Some(self.error(
                codes::OPERATOR,
                format!(
                    "operator `{}` is not in the M0 subset; write arithmetic as a call such as `add(a, b)`",
                    self.text()
                ),
            ));
        }
        if token != Token::Ident {
            return None;
        }
        let code = match self.text() {
            "module" => codes::MODULE,
            "layer" => codes::LAYER,
            "const" => codes::CONST,
            "repeat" => codes::REPEAT,
            _ => return None,
        };
        Some(self.error(
            code,
            format!(
                "`{}` is not in the M0 subset (DESIGN.md § 11, iteration 3)",
                self.text()
            ),
        ))
    }

    fn expect(&mut self, token: Token, expected: &str) -> PResult<()> {
        if self.peek() == Some(token) {
            self.bump();
            Ok(())
        } else {
            Err(self.unexpected(expected))
        }
    }

    fn ident(&mut self, expected: &str) -> PResult<Ident> {
        if self.peek() != Some(Token::Ident) {
            return Err(self.unexpected(expected));
        }
        let ident = Ident {
            name: self.text().to_owned(),
            span: self.span(),
        };
        self.bump();
        Ok(ident)
    }

    fn no_generic_parameters(&self) -> PResult<()> {
        if self.peek() == Some(Token::Lt) {
            return Err(self.error(
                codes::GENERIC_PARAMETERS,
                "generic parameter list `<...>` is not in the M0 subset; dimensions are integer literals",
            ));
        }
        Ok(())
    }

    fn out_of_range(&self, what: &str) -> Diagnostic {
        self.error(
            codes::LITERAL_OUT_OF_RANGE,
            format!("{what} `{}` is out of range", self.text()),
        )
    }

    /// An integer literal that fits `T`.
    fn int<T: TryFrom<u64>>(&mut self, what: &str) -> PResult<T> {
        if self.peek() != Some(Token::Int) {
            return Err(self.unexpected("an integer literal"));
        }
        let value = self
            .text()
            .parse::<u64>()
            .ok()
            .and_then(|v| T::try_from(v).ok())
            .ok_or_else(|| self.out_of_range(what))?;
        self.bump();
        Ok(value)
    }

    // -----------------------------------------------------------------------------------------
    // Items.
    // -----------------------------------------------------------------------------------------

    fn file(&mut self) -> Result<Ast, Vec<Diagnostic>> {
        let mut diagnostics = Vec::new();
        let (mut model, mut train) = (None, None);
        let (mut seen_model, mut seen_train) = (false, false);
        while self.pos < self.lexemes.len() {
            let start = self.pos;
            let result = match self.peek() {
                Some(Token::Model) if seen_model => Err(self.second_item("model")),
                Some(Token::Model) => {
                    seen_model = true;
                    self.model().map(|m| model = Some(m))
                }
                Some(Token::Train) if seen_train => Err(self.second_item("train")),
                Some(Token::Train) => {
                    seen_train = true;
                    self.train().map(|t| train = Some(t))
                }
                _ => Err(self.unexpected("`model` or `train`")),
            };
            if let Err(diagnostic) = result {
                diagnostics.push(diagnostic);
                self.recover(start);
            }
        }
        for (seen, item) in [(seen_model, "model"), (seen_train, "train")] {
            if !seen {
                diagnostics.push(self.error(
                    codes::MISSING_ITEM,
                    format!("the file has no `{item}` item; M0 needs exactly one"),
                ));
            }
        }
        match (model, train) {
            (Some(model), Some(train)) if diagnostics.is_empty() => Ok(Ast {
                model,
                train,
                span: self.span_at(0),
            }),
            _ => Err(diagnostics),
        }
    }

    fn second_item(&self, item: &str) -> Diagnostic {
        self.error(
            codes::SECOND_ITEM,
            format!("second `{item}` item; M0 allows exactly one per file"),
        )
    }

    /// Skips the item that starts at lexeme `start`: through the `}` that closes its first `{`,
    /// or up to the next lexeme that starts an item at any brace depth, or to the end of input.
    /// An unclosed `{` therefore never swallows the item after it.
    fn recover(&mut self, start: usize) {
        let mut depth = 0usize;
        let mut i = start;
        while let Some(lexeme) = self.lexemes.get(i) {
            if i != start && self.starts_item(i) {
                break;
            }
            match lexeme.token {
                Some(Token::LBrace) => depth += 1,
                Some(Token::RBrace) => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        i += 1;
                        break;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        self.pos = i.max(start + 1);
    }

    /// Whether lexeme `i` starts an item: `train`, which only items start with, or `model`
    /// followed by `{`, `NAME {` or `NAME <`. Inside a train item `model` is followed by `=`
    /// (`model = M;`) or `.` (`model.forward(x)`), and a malformed `model M;` is not an item.
    fn starts_item(&self, i: usize) -> bool {
        let token = |at: usize| self.lexemes.get(at).and_then(|l| l.token);
        match token(i) {
            Some(Token::Train) => true,
            Some(Token::Model) => match token(i + 1) {
                Some(Token::LBrace) => true,
                Some(Token::Ident) => matches!(token(i + 2), Some(Token::LBrace | Token::Lt)),
                _ => false,
            },
            _ => false,
        }
    }

    fn model(&mut self) -> PResult<Model> {
        let span = self.span();
        self.bump();
        let name = self.ident("a model name")?;
        self.no_generic_parameters()?;
        self.expect(Token::LBrace, "`{`")?;
        let mut items = Vec::new();
        loop {
            match self.peek() {
                Some(Token::RBrace) => {
                    self.bump();
                    break;
                }
                Some(Token::Param) => items.push(ModelItem::Param(self.param()?)),
                Some(Token::Fn) => items.push(ModelItem::Fn(self.function()?)),
                _ => return Err(self.unexpected("`param`, `fn` or `}`")),
            }
        }
        Ok(Model { name, items, span })
    }

    fn param(&mut self) -> PResult<Param> {
        let span = self.span();
        self.bump();
        let name = self.ident("a parameter name")?;
        self.expect(Token::Colon, "`:`")?;
        let ty = self.tensor_type()?;
        self.expect(Token::Eq, "`=`")?;
        let init = self.call()?;
        self.expect(Token::Semi, "`;`")?;
        Ok(Param {
            name,
            ty,
            init,
            span,
        })
    }

    fn function(&mut self) -> PResult<Function> {
        let span = self.span();
        self.bump();
        let name = self.ident("a function name")?;
        self.no_generic_parameters()?;
        self.expect(Token::LParen, "`(`")?;
        let mut params = Vec::new();
        if self.peek() == Some(Token::RParen) {
            self.bump();
        } else {
            loop {
                let span = self.span();
                let name = self.ident("a parameter name")?;
                self.expect(Token::Colon, "`:`")?;
                let ty = self.tensor_type()?;
                params.push(FnParam { name, ty, span });
                match self.peek() {
                    Some(Token::Comma) => self.bump(),
                    Some(Token::RParen) => {
                        self.bump();
                        break;
                    }
                    _ => return Err(self.unexpected("`,` or `)`")),
                }
            }
        }
        self.expect(Token::Arrow, "`->`")?;
        let output = self.tensor_type()?;
        self.expect(Token::LBrace, "`{`")?;
        let mut lets = Vec::new();
        loop {
            match self.peek() {
                Some(Token::Let) => {
                    let span = self.span();
                    self.bump();
                    let name = self.ident("a binding name")?;
                    self.expect(Token::Eq, "`=`")?;
                    let value = self.expr()?;
                    self.expect(Token::Semi, "`;`")?;
                    lets.push(Let { name, value, span });
                }
                Some(Token::Return) => break,
                _ => return Err(self.unexpected("`let` or `return`")),
            }
        }
        let return_span = self.span();
        self.bump();
        let value = self.expr()?;
        self.expect(Token::Semi, "`;`")?;
        self.expect(Token::RBrace, "`}`")?;
        Ok(Function {
            name,
            params,
            output,
            lets,
            result: Return {
                value,
                span: return_span,
            },
            span,
        })
    }

    fn tensor_type(&mut self) -> PResult<TensorType> {
        let span = self.span();
        if !self.at_word("Tensor") {
            return Err(self.unexpected("`Tensor`"));
        }
        self.bump();
        self.expect(Token::Lt, "`<`")?;
        let dtype_span = self.span();
        if self.peek() != Some(Token::Ident) {
            return Err(self.unexpected("a dtype"));
        }
        let kind = match self.text() {
            "f32" => DType::F32,
            "i32" => DType::I32,
            other => {
                return Err(self.error(
                    codes::DTYPE,
                    format!("dtype `{other}` is not in the M0 subset; only `f32` and `i32` are"),
                ));
            }
        };
        self.bump();
        self.expect(Token::Comma, "`,`")?;
        self.expect(Token::LBracket, "`[`")?;
        let mut dims = Vec::new();
        if self.peek() == Some(Token::RBracket) {
            self.bump();
        } else {
            loop {
                dims.push(self.dim()?);
                match self.peek() {
                    Some(Token::Comma) => self.bump(),
                    Some(Token::RBracket) => {
                        self.bump();
                        break;
                    }
                    _ => return Err(self.unexpected("`,` or `]`")),
                }
            }
        }
        self.expect(Token::Gt, "`>`")?;
        Ok(TensorType {
            dtype: Dtype {
                kind,
                span: dtype_span,
            },
            dims,
            span,
        })
    }

    fn dim(&mut self) -> PResult<Dim> {
        let span = self.span();
        match self.peek() {
            Some(Token::Int) => {
                let size = self.int::<u64>("dimension")?;
                Ok(Dim { size, span })
            }
            Some(Token::Ident) => Err(self.error(
                codes::SYMBOLIC_DIMENSION,
                format!(
                    "symbolic dimension `{}` is not in the M0 subset; dimensions are integer literals",
                    self.text()
                ),
            )),
            _ => Err(self.unexpected("an integer dimension")),
        }
    }

    // -----------------------------------------------------------------------------------------
    // Train block.
    // -----------------------------------------------------------------------------------------

    fn train(&mut self) -> PResult<Train> {
        let span = self.span();
        self.bump();
        let name = self.ident("a train block name")?;
        self.no_generic_parameters()?;
        self.expect(Token::LBrace, "`{`")?;

        let mut model = None;
        let mut data = Vec::new();
        let mut loss = None;
        let mut optimizer = None;
        let mut steps = None;
        let mut seed = None;
        let mut device = None;
        let close = loop {
            let setting_span = self.span();
            let setting = match self.peek() {
                Some(Token::RBrace) => {
                    self.bump();
                    break setting_span;
                }
                Some(Token::Input | Token::Target) => {
                    data.push(self.data_decl()?);
                    continue;
                }
                Some(Token::Model) => "model",
                Some(Token::Ident) => match SETTINGS.iter().copied().find(|s| *s == self.text()) {
                    Some(setting) => setting,
                    None => {
                        if let Some(diagnostic) = self.outside_subset() {
                            return Err(diagnostic);
                        }
                        return Err(self.error(
                            codes::UNKNOWN_TRAIN_SETTING,
                            format!(
                                "unknown train setting `{}`; M0 has `model`, `input`, `target`, \
                                 `loss`, `optimizer`, `steps`, `seed` and `device`",
                                self.text()
                            ),
                        ));
                    }
                },
                _ => return Err(self.unexpected("a train setting or `}`")),
            };
            let given = match setting {
                "model" => model.is_some(),
                "loss" => loss.is_some(),
                "optimizer" => optimizer.is_some(),
                "steps" => steps.is_some(),
                "seed" => seed.is_some(),
                _ => device.is_some(),
            };
            if given {
                return Err(self.error(
                    codes::DUPLICATE_TRAIN_SETTING,
                    format!("train setting `{setting}` is given twice"),
                ));
            }
            self.bump();
            self.expect(Token::Eq, "`=`")?;
            let span = setting_span;
            match setting {
                "model" => {
                    let value = self.ident("a model name")?;
                    if self.peek() == Some(Token::Lt) {
                        return Err(self.error(
                            codes::GENERIC_ARGUMENTS,
                            "generic arguments `<...>` are not in the M0 subset; the model takes none",
                        ));
                    }
                    model = Some(Setting { value, span });
                }
                "loss" => {
                    let value = self.expr()?;
                    loss = Some(Setting { value, span });
                }
                "optimizer" => {
                    let value = self.optimizer()?;
                    optimizer = Some(Setting { value, span });
                }
                "steps" => {
                    let value = self.int::<u64>("step count")?;
                    steps = Some(Setting { value, span });
                }
                "seed" => {
                    let value = self.int::<u64>("seed")?;
                    seed = Some(Setting { value, span });
                }
                _ => {
                    let value = self.device()?;
                    device = Some(Setting { value, span });
                }
            }
            self.expect(Token::Semi, "`;`")?;
        };

        let missing = |setting: &str| {
            Diagnostic::new(
                codes::MISSING_TRAIN_SETTING,
                format!("train block `{}` has no `{setting}` setting", name.name),
            )
            .at(close.clone())
        };
        Ok(Train {
            model: model.ok_or_else(|| missing("model"))?,
            data,
            loss: loss.ok_or_else(|| missing("loss"))?,
            optimizer: optimizer.ok_or_else(|| missing("optimizer"))?,
            steps: steps.ok_or_else(|| missing("steps"))?,
            seed: seed.ok_or_else(|| missing("seed"))?,
            device: device.ok_or_else(|| missing("device"))?,
            name,
            span,
        })
    }

    fn data_decl(&mut self) -> PResult<DataDecl> {
        let span = self.span();
        let role = if self.peek() == Some(Token::Input) {
            DataRole::Input
        } else {
            DataRole::Target
        };
        self.bump();
        let name = self.ident("a data name")?;
        self.expect(Token::Colon, "`:`")?;
        let ty = self.tensor_type()?;
        self.expect(Token::Semi, "`;`")?;
        Ok(DataDecl {
            role,
            name,
            ty,
            span,
        })
    }

    fn optimizer(&mut self) -> PResult<Optimizer> {
        if self.peek() != Some(Token::Ident) {
            return Err(self.unexpected("an optimizer"));
        }
        if self.text() == "adamw" || self.text() == "muon" {
            let muon = self.text() == "muon";
            let span = self.span();
            self.bump();
            self.expect(Token::LParen, "`(`")?;
            let args = self.args()?;
            return Ok(if muon {
                Optimizer::Muon { args, span }
            } else {
                Optimizer::AdamW { args, span }
            });
        }
        if self.text() != "sgd" {
            return Err(self.error(
                codes::OPTIMIZER,
                format!(
                    "optimizer `{}` is not in the M0 subset; only `sgd`, `adamw` and `muon` are",
                    self.text()
                ),
            ));
        }
        self.bump();
        self.expect(Token::LParen, "`(`")?;
        if !self.at_word("lr") {
            return Err(self.unexpected("`lr`"));
        }
        self.bump();
        self.expect(Token::Eq, "`=`")?;
        let lr_span = self.span();
        if self.peek() != Some(Token::Float) {
            return Err(self.unexpected("a float literal"));
        }
        let lr = self.float()?;
        if self.peek() == Some(Token::Comma) && self.peek_at(1) == Some(Token::Ident) {
            // A schedule argument such as `warmup_steps = 10`: `sgd` keeps a constant rate, so
            // the comma before it is refused, naming the argument (coordinator decision D50,
            // settlement 2).
            let comma = self.span();
            self.bump();
            let argument = self.text();
            return Err(Diagnostic::new(
                codes::UNEXPECTED_TOKEN,
                format!(
                    "expected `)`, found `,` before `{argument}`; `sgd` takes `lr` alone, at a \
                     constant rate"
                ),
            )
            .at(comma));
        }
        self.expect(Token::RParen, "`)`")?;
        Ok(Optimizer::Sgd { lr, lr_span })
    }

    fn device(&mut self) -> PResult<Device> {
        if self.peek() != Some(Token::Ident) {
            return Err(self.unexpected("a device"));
        }
        match self.text() {
            "cpu" => {
                self.bump();
                Ok(Device::Cpu)
            }
            "vulkan" => {
                self.bump();
                self.expect(Token::LParen, "`(`")?;
                let index_span = self.span();
                let index = self.int::<u32>("device index")?;
                self.expect(Token::RParen, "`)`")?;
                Ok(Device::Vulkan { index, index_span })
            }
            other => Err(self.error(
                codes::DEVICE,
                format!("device `{other}` is not supported; expected `cpu` or `vulkan(<index>)`"),
            )),
        }
    }

    // -----------------------------------------------------------------------------------------
    // Expressions.
    // -----------------------------------------------------------------------------------------

    /// A float literal at the current position; it must be finite.
    fn float(&mut self) -> PResult<f64> {
        match self.text().parse::<f64>() {
            Ok(value) if value.is_finite() => {
                self.bump();
                Ok(value)
            }
            _ => Err(self.out_of_range("float literal")),
        }
    }

    /// `-` then a float literal, the value of a named argument such as `p = -0.1`, so the checker
    /// can name the value it refuses (`E-TYPE-022`). Nowhere else is `-` a sign: an integer
    /// literal after it, as in `heads = -1`, stays E-SYNTAX-009.
    fn negative_float(&mut self) -> PResult<Expr> {
        let span = self.span();
        self.bump();
        let value = self.float()?;
        Ok(Expr {
            kind: ExprKind::Float(-value),
            span,
        })
    }

    /// An expression. An operator before or after it is reported as E-SYNTAX-009 by
    /// [`Parser::unexpected`] when the caller expects the next token.
    fn expr(&mut self) -> PResult<Expr> {
        let span = self.span();
        let kind = match self.peek() {
            Some(Token::Int) => ExprKind::Int(self.int::<u64>("integer literal")?),
            Some(Token::Float) => ExprKind::Float(self.float()?),
            Some(Token::Ident) if self.peek_at(1) == Some(Token::LParen) => {
                ExprKind::Call(self.call()?)
            }
            Some(Token::Ident | Token::Model) if self.peek_at(1) == Some(Token::Dot) => {
                let receiver = Ident {
                    name: self.text().to_owned(),
                    span: span.clone(),
                };
                self.bump();
                self.bump();
                let method = self.ident("a method name")?;
                self.expect(Token::LParen, "`(`")?;
                let args = self.args()?;
                ExprKind::MemberCall(MemberCall {
                    receiver,
                    method,
                    args,
                    span: span.clone(),
                })
            }
            Some(Token::Ident) => {
                let name = self.text().to_owned();
                self.bump();
                ExprKind::Var(name)
            }
            _ => return Err(self.unexpected("an expression")),
        };
        Ok(Expr { kind, span })
    }

    fn call(&mut self) -> PResult<Call> {
        let span = self.span();
        let callee = self.ident("a function name")?;
        self.expect(Token::LParen, "`(`")?;
        let args = self.args()?;
        Ok(Call { callee, args, span })
    }

    /// Arguments after `(`, through the closing `)`. Past [`MAX_NESTING`] enclosing argument
    /// lists this is E-SYNTAX-021 at the `(`, which bounds the parser's recursion.
    fn args(&mut self) -> PResult<Vec<Arg>> {
        if self.nesting == MAX_NESTING {
            let open = self.lexemes[self.pos - 1].range.start;
            return Err(Diagnostic::new(
                codes::NESTING_TOO_DEEP,
                format!(
                    "argument lists nest more than {MAX_NESTING} deep; the M0 parser allows at most {MAX_NESTING}"
                ),
            )
            .at(self.span_at(open)));
        }
        self.nesting += 1;
        let result = self.args_within();
        self.nesting -= 1;
        result
    }

    fn args_within(&mut self) -> PResult<Vec<Arg>> {
        let mut args = Vec::new();
        if self.peek() == Some(Token::RParen) {
            self.bump();
            return Ok(args);
        }
        loop {
            let span = self.span();
            let name = if self.peek() == Some(Token::Ident) && self.peek_at(1) == Some(Token::Eq) {
                let name = self.ident("an argument name")?;
                self.bump();
                Some(name)
            } else {
                None
            };
            let value = if name.is_some()
                && self.peek() == Some(Token::Minus)
                && self.peek_at(1) == Some(Token::Float)
            {
                self.negative_float()?
            } else {
                self.expr()?
            };
            args.push(Arg { name, value, span });
            match self.peek() {
                Some(Token::Comma) => self.bump(),
                Some(Token::RParen) => {
                    self.bump();
                    return Ok(args);
                }
                _ => return Err(self.unexpected("`,` or `)`")),
            }
        }
    }
}
