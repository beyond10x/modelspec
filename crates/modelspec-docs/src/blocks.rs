//! Every ` ```modelspec ` block of the documentation and its diagram (DESIGN.md § 13.6).
//!
//! [`MANIFEST`] lists each block of `README.md`, `DESIGN.md` and the pages under `website/docs`
//! outside the generated gallery, by file and position: whole (a program of
//! [`crate::diagrams::PROGRAMS`], drawn beside the block), fragment (part of such a program's
//! file, drawn after the last fragment of its set) or skipped (with the codes the compiler
//! answers). [`problems`] fails on a block the manifest does not list, an entry without its
//! block, a whole block or the last fragment of a set without its diagram, a fragment its program
//! does not contain and a skipped block that compiles.
//!
//! [`pages`] writes one page per program under [`DIR`], every level of every view under its own
//! heading, whose anchor [`anchor`] makes from the view and the scope key; a composite of a
//! diagram links there through [`level_href`], and [`unresolved_links`] checks every such link.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use modelspec_core::compile;
use modelspec_diagram::model::DiagramView;

use crate::diagrams::{Drawn, image_path, program};
use crate::{BASE_URL, DOCS, Pages, Result, SOURCE, front_matter, io, sorted_entries};

/// Where the diagram pages land, relative to the repository root.
pub const DIR: &str = "website/docs/diagrams";

/// The files outside [`DOCS`] whose blocks are listed.
pub const TOP_LEVEL: &[&str] = &["README.md", "DESIGN.md"];

/// The generated gallery, whose pages draw their programs themselves.
const GALLERY: &str = "website/docs/gallery";

/// What the documentation does with one block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The block is the whole program `program` of [`crate::diagrams::PROGRAMS`]: its text, whitespace aside, is the
    /// program's file without its leading comment lines, and its diagram follows it.
    Whole { program: &'static str },
    /// The block is part of the file of `program`; the diagram follows the last fragment of the
    /// program's set in its file.
    Fragment { program: &'static str },
    /// The block is not drawn. Compiled, it answers each of `codes`.
    Skipped {
        codes: &'static [&'static str],
        reason: &'static str,
    },
}

/// One block of the documentation, by file and position.
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    /// Relative to the repository root.
    pub file: &'static str,
    /// The position of the block among the file's ` ```modelspec ` blocks, from 1.
    pub ordinal: usize,
    /// The block's first non-blank line, trimmed, which names the block for a reader of this list.
    pub opens: &'static str,
    pub kind: Kind,
}

const TARGET_SYNTAX: &str = "target syntax of DESIGN.md § 2, which the M0 parser refuses until modules land (§ 11, iteration 3)";

/// Every ` ```modelspec ` block of the documentation.
pub const MANIFEST: &[Entry] = &[
    Entry {
        file: "README.md",
        ordinal: 1,
        opens: "model LinearReg {",
        kind: Kind::Whole {
            program: "linear-reg",
        },
    },
    Entry {
        file: "DESIGN.md",
        ordinal: 1,
        opens: "module Linear<In: dim, Out: dim> {",
        kind: Kind::Skipped {
            codes: &["E-SYNTAX-004"],
            reason: TARGET_SYNTAX,
        },
    },
    Entry {
        file: "website/docs/index.mdx",
        ordinal: 1,
        opens: "model LinearReg {",
        kind: Kind::Whole {
            program: "linear-reg",
        },
    },
    Entry {
        file: "website/docs/concepts/language.mdx",
        ordinal: 1,
        opens: "model TwoLayer {",
        kind: Kind::Whole {
            program: "two-layer-relu",
        },
    },
    Entry {
        file: "website/docs/concepts/language.mdx",
        ordinal: 2,
        opens: "module Linear<In: dim, Out: dim> {",
        kind: Kind::Skipped {
            codes: &["E-SYNTAX-004"],
            reason: TARGET_SYNTAX,
        },
    },
    Entry {
        file: "website/docs/concepts/bert-operators.mdx",
        ordinal: 1,
        opens: "model MlmBag {",
        kind: Kind::Whole { program: "mlm-bag" },
    },
    Entry {
        file: "website/docs/guides/write-a-model.mdx",
        ordinal: 1,
        opens: "model TwoLayer {",
        kind: Kind::Fragment {
            program: "two-layer-relu",
        },
    },
    Entry {
        file: "website/docs/guides/write-a-model.mdx",
        ordinal: 2,
        opens: "fn forward(x: Tensor<f32, [16, 4]>)",
        kind: Kind::Fragment {
            program: "two-layer-relu",
        },
    },
    Entry {
        file: "website/docs/guides/write-a-model.mdx",
        ordinal: 3,
        opens: "train FitTwoLayer {",
        kind: Kind::Fragment {
            program: "two-layer-relu",
        },
    },
    Entry {
        file: "website/docs/examples/linear-regression.mdx",
        ordinal: 1,
        opens: "model LinearReg {",
        kind: Kind::Whole {
            program: "linear-reg",
        },
    },
    Entry {
        file: "website/docs/examples/two-layer-relu.mdx",
        ordinal: 1,
        opens: "model TwoLayer {",
        kind: Kind::Whole {
            program: "two-layer-relu",
        },
    },
    Entry {
        file: "website/docs/examples/bert-tiny.mdx",
        ordinal: 1,
        opens: "model BertTiny {",
        kind: Kind::Whole {
            program: "bert-tiny",
        },
    },
    Entry {
        file: "website/docs/examples/beyond-m0.md",
        ordinal: 1,
        opens: "module Linear<In: dim, Out: dim> {",
        kind: Kind::Skipped {
            codes: &["E-SYNTAX-004", "E-SYNTAX-006", "E-SYNTAX-007"],
            reason: TARGET_SYNTAX,
        },
    },
    Entry {
        file: "website/docs/examples/shape-error.md",
        ordinal: 1,
        opens: "model Mismatch {",
        kind: Kind::Skipped {
            codes: &["E-TYPE-012"],
            reason: "the refusal is the page's subject",
        },
    },
];

/// One ` ```modelspec ` block found in a file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    /// Relative to the repository root.
    pub file: String,
    /// The line of the opening fence, from 1.
    pub line: usize,
    /// The line of the closing fence, from 1; the file's last line when the block never closes.
    pub end: usize,
    /// Among the file's ` ```modelspec ` blocks, from 1.
    pub ordinal: usize,
    pub text: String,
}

impl Block {
    /// The block's first non-blank line, trimmed.
    pub fn opens(&self) -> &str {
        self.text
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .unwrap_or("")
    }

    fn at(&self) -> String {
        format!("{}:{}", self.file, self.line)
    }
}

/// The ` ```modelspec ` blocks of `text`, the file `file`: fenced code blocks (backticks or
/// tildes, at most three spaces of indent) whose info string's first word is `modelspec`. A fence
/// inside another fence is text.
pub fn scan_text(file: &str, text: &str) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut open: Option<(char, usize, bool, usize, Vec<&str>)> = None;
    let lines: Vec<&str> = text.lines().collect();
    for (n, line) in lines.iter().enumerate() {
        let indent = line.len() - line.trim_start_matches(' ').len();
        let rest = &line[indent..];
        let fence_char = rest.chars().next().filter(|c| *c == '`' || *c == '~');
        let run = fence_char.map_or(0, |c| rest.chars().take_while(|x| *x == c).count());
        match &mut open {
            None => {
                if indent <= 3
                    && let Some(c) = fence_char
                    && run >= 3
                {
                    let info = rest[run..].trim();
                    let modelspec = info.split_whitespace().next() == Some("modelspec");
                    open = Some((c, run, modelspec, n + 1, Vec::new()));
                }
            }
            Some((c, len, modelspec, start, body)) => {
                let closes = indent <= 3
                    && fence_char == Some(*c)
                    && run >= *len
                    && rest[run..].trim().is_empty();
                if closes {
                    if *modelspec {
                        blocks.push(Block {
                            file: file.to_owned(),
                            line: *start,
                            end: n + 1,
                            ordinal: blocks.len() + 1,
                            text: body.join("\n") + "\n",
                        });
                    }
                    open = None;
                } else {
                    body.push(line);
                }
            }
        }
    }
    if let Some((_, _, true, start, body)) = open {
        blocks.push(Block {
            file: file.to_owned(),
            line: start,
            end: lines.len(),
            ordinal: blocks.len() + 1,
            text: body.join("\n") + "\n",
        });
    }
    blocks
}

/// Every Markdown and MDX page under `dir`, relative to `root`, outside the gallery.
fn pages_under(root: &Path, dir: &str, found: &mut Vec<String>) -> Result<()> {
    if dir == GALLERY {
        return Ok(());
    }
    for entry in sorted_entries(&root.join(dir))? {
        let name = entry.file_name().to_string_lossy().into_owned();
        let relative = format!("{dir}/{name}");
        if entry.path().is_dir() {
            pages_under(root, &relative, found)?;
        } else if name.ends_with(".md") || name.ends_with(".mdx") {
            found.push(relative);
        }
    }
    Ok(())
}

/// Every ` ```modelspec ` block of `README.md`, `DESIGN.md` and the pages under `website/docs`
/// outside the gallery, read under `root`.
pub fn scan(root: &Path) -> Result<Vec<Block>> {
    let mut files: Vec<String> = TOP_LEVEL.iter().map(|f| (*f).to_owned()).collect();
    pages_under(root, DOCS, &mut files)?;
    let mut blocks = Vec::new();
    for file in files {
        let text = fs::read_to_string(root.join(&file)).map_err(io(&file))?;
        blocks.extend(scan_text(&file, &text));
    }
    Ok(blocks)
}

/// The manifest entry of `block`.
fn entry_of<'m>(manifest: &'m [Entry], block: &Block) -> Option<&'m Entry> {
    manifest
        .iter()
        .find(|e| e.file == block.file && e.ordinal == block.ordinal)
}

/// Acceptance `docs.blocks.every-block-listed`: each block without its entry, by file and line;
/// each entry whose block opens with another line; and each entry without its block.
pub fn unlisted(blocks: &[Block], manifest: &[Entry]) -> Vec<String> {
    let mut problems = Vec::new();
    for block in blocks {
        match entry_of(manifest, block) {
            None => problems.push(format!(
                "{}: ```modelspec block {} of the file is not in the manifest (crates/modelspec-docs/src/blocks.rs, MANIFEST)",
                block.at(),
                block.ordinal
            )),
            Some(entry) if entry.opens != block.opens() => problems.push(format!(
                "{}: block {} opens with `{}`, the manifest expects `{}`",
                block.at(),
                block.ordinal,
                block.opens(),
                entry.opens
            )),
            Some(_) => {}
        }
    }
    for entry in manifest {
        if !blocks
            .iter()
            .any(|b| b.file == entry.file && b.ordinal == entry.ordinal)
        {
            problems.push(format!(
                "{}: the manifest lists ```modelspec block {}, which the file does not have",
                entry.file, entry.ordinal
            ));
        }
    }
    problems
}

/// `text` with every run of whitespace one space, and no leading or trailing whitespace.
pub fn normalized(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether `file` contains `fragment`: every non-blank line of the fragment, whitespace-normalized,
/// is a line of the file, in the fragment's order. A guide's step may close a block the next step
/// opens again (`model TwoLayer { param …; param …; }`, then `fn forward` inside it), so a fragment
/// is a run of the file's lines with gaps, not one substring.
pub fn contains_lines(file: &str, fragment: &str) -> bool {
    let mut lines = file.lines().map(normalized).filter(|l| !l.is_empty());
    fragment
        .lines()
        .map(normalized)
        .filter(|l| !l.is_empty())
        .all(|wanted| lines.any(|l| l == wanted))
}

/// `source` without its leading `//` comment lines and blank lines, as the example pages show it.
fn without_leading_comments(source: &str) -> String {
    source
        .lines()
        .skip_while(|l| l.trim().is_empty() || l.trim().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn program_source(root: &Path, key: &str) -> Result<(&'static str, String)> {
    let program =
        program(key).ok_or_else(|| format!("no program `{key}` in diagrams::PROGRAMS"))?;
    let text = fs::read_to_string(root.join(program.file)).map_err(io(program.file))?;
    Ok((program.file, text))
}

/// Acceptance `docs.blocks.fragments-in-fixture`, and its counterpart for a whole block: each
/// fragment its program's file does not contain ([`contains_lines`]), and each whole block that
/// is not its program's file without the file's leading comment lines, whitespace-normalized.
pub fn outside_their_program(
    root: &Path,
    blocks: &[Block],
    manifest: &[Entry],
) -> Result<Vec<String>> {
    let mut problems = Vec::new();
    for block in blocks {
        match entry_of(manifest, block).map(|e| e.kind) {
            Some(Kind::Fragment { program }) => {
                let (file, text) = program_source(root, program)?;
                if !contains_lines(&text, &block.text) {
                    problems.push(format!(
                        "{}: the fragment is not in {file}, the file of `{program}`",
                        block.at()
                    ));
                }
            }
            Some(Kind::Whole { program }) => {
                let (file, text) = program_source(root, program)?;
                if normalized(&without_leading_comments(&text)) != normalized(&block.text) {
                    problems.push(format!(
                        "{}: the block is not {file}, the file of `{program}`, whitespace aside",
                        block.at()
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(problems)
}

/// Acceptance `docs.blocks.skips-still-refused`: each skipped block that compiles, or that does
/// not answer every code its entry names.
pub fn skips_that_compile(blocks: &[Block], manifest: &[Entry]) -> Vec<String> {
    let mut problems = Vec::new();
    for block in blocks {
        let Some(Kind::Skipped { codes, .. }) = entry_of(manifest, block).map(|e| e.kind) else {
            continue;
        };
        match compile::compile_training(&block.at(), &block.text) {
            Ok(_) => problems.push(format!(
                "{}: the skipped block compiles now; make it whole in the manifest so its diagram is drawn",
                block.at()
            )),
            Err(diagnostics) => {
                let answered: BTreeSet<String> =
                    diagnostics.iter().map(|d| d.code.to_string()).collect();
                for code in codes.iter() {
                    if !answered.contains(*code) {
                        problems.push(format!(
                            "{}: the skipped block no longer answers {code} (it answers {})",
                            block.at(),
                            answered.iter().cloned().collect::<Vec<_>>().join(", ")
                        ));
                    }
                }
            }
        }
    }
    problems
}

/// The blocks after which a diagram stands: each whole block, and the last fragment of each
/// program's set in a file; with the program.
fn drawn_after<'b>(blocks: &'b [Block], manifest: &[Entry]) -> Vec<(&'b Block, &'static str)> {
    let mut out = Vec::new();
    for block in blocks {
        match entry_of(manifest, block).map(|e| e.kind) {
            Some(Kind::Whole { program }) => out.push((block, program)),
            Some(Kind::Fragment { program }) => {
                let later = blocks.iter().any(|b| {
                    b.file == block.file
                        && b.ordinal > block.ordinal
                        && entry_of(manifest, b).map(|e| e.kind) == Some(Kind::Fragment { program })
                });
                if !later {
                    out.push((block, program));
                }
            }
            _ => {}
        }
    }
    out
}

/// The identifier a page imports the diagram module of `program` as, from a line
/// `import <name> from '@site/data/diagrams/<program>.json';`.
fn imported_as(page: &str, program: &str) -> Option<String> {
    let module = format!(" from '@site/data/diagrams/{program}.json';");
    page.lines().find_map(|line| {
        let name = line.strip_prefix("import ")?.strip_suffix(&module)?;
        (!name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
            .then(|| name.to_owned())
    })
}

/// Acceptance `docs.blocks.whole-has-diagram`: each whole block and last fragment whose next
/// element is not its program's diagram. On a page of the site that element is
/// `<ModelDiagram data={…} view="model" />` of the program's imported module, on an MDX page;
/// in `README.md` and `DESIGN.md`, which GitHub renders, it is a Markdown image of the program's
/// standalone SVG, a file that exists.
pub fn missing_diagrams(
    root: &Path,
    blocks: &[Block],
    manifest: &[Entry],
    drawn: &[Drawn],
) -> Result<Vec<String>> {
    let mut problems = Vec::new();
    for (block, program) in drawn_after(blocks, manifest) {
        let page = fs::read_to_string(root.join(&block.file)).map_err(io(&block.file))?;
        let next = page
            .lines()
            .skip(block.end)
            .find(|l| !l.trim().is_empty())
            .unwrap_or("")
            .trim();
        if block.file.starts_with(&format!("{DOCS}/")) {
            if !block.file.ends_with(".mdx") {
                problems.push(format!(
                    "{}: the page shows the diagram of `{program}`, so it must be MDX",
                    block.at()
                ));
                continue;
            }
            let Some(name) = imported_as(&page, program) else {
                problems.push(format!(
                    "{}: the page does not import @site/data/diagrams/{program}.json",
                    block.file
                ));
                continue;
            };
            let element = next.starts_with("<ModelDiagram ")
                && next.ends_with("/>")
                && next.contains(&format!(" data={{{name}}} "))
                && next.contains(" view=\"model\" ");
            if !element {
                problems.push(format!(
                    "{}: the next element after the block is not <ModelDiagram data={{{name}}} view=\"model\" />, the diagram of `{program}`: {next}",
                    block.at()
                ));
            }
        } else {
            let Some(view) = drawn
                .iter()
                .find(|d| d.key == program)
                .and_then(|d| d.views.first())
            else {
                problems.push(format!("{}: `{program}` is not drawn", block.at()));
                continue;
            };
            let svg = image_path(program, view.view, &view.root);
            if !(next.contains("![") && next.contains(&format!("]({svg})"))) {
                problems.push(format!(
                    "{}: the next line after the block is not the image {svg}: {next}",
                    block.at()
                ));
            } else if !root.join(&svg).is_file() {
                problems.push(format!("{}: {svg} does not exist", block.at()));
            }
        }
    }
    Ok(problems)
}

/// Every problem of the manifest and the blocks, read under `root`: [`unlisted`],
/// [`outside_their_program`], [`skips_that_compile`] and [`missing_diagrams`].
pub fn problems(root: &Path, drawn: &[Drawn]) -> Result<Vec<String>> {
    let blocks = scan(root)?;
    let mut problems = unlisted(&blocks, MANIFEST);
    problems.extend(outside_their_program(root, &blocks, MANIFEST)?);
    problems.extend(skips_that_compile(&blocks, MANIFEST));
    problems.extend(missing_diagrams(root, &blocks, MANIFEST, drawn)?);
    Ok(problems)
}

/// The heading anchor of one level on its program's diagram page: the view's name and the scope
/// key, every character other than an ASCII letter, digit, `-` or `_` written `-`.
pub fn anchor(view: DiagramView, scope: &str) -> String {
    let scope: String = scope
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    format!("{}-{scope}", view.name())
}

/// The link of a composite to its own level: the level's heading on the program's diagram page.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the link template of the composites story:diagram-zoom draws"
    )
)]
pub fn level_href(program: &str, view: DiagramView, scope: &str) -> String {
    format!("{BASE_URL}docs/diagrams/{program}#{}", anchor(view, scope))
}

/// The file name of a program's diagram page under [`DIR`].
pub fn page_name(program: &str) -> String {
    format!("{program}.mdx")
}

/// The title of the page a block stands on, for a link to it: its front matter's `title`, or its
/// path. A page this run generates is read from `generated`, by path relative to the repository
/// root, so the first generation writes what every later one does.
fn title_of(root: &Path, file: &str, generated: &Pages) -> Result<String> {
    let text = match generated.get(file) {
        Some(text) => text.clone(),
        None => fs::read_to_string(root.join(file)).map_err(io(file))?,
    };
    let title = text
        .strip_prefix("---\n")
        .and_then(|rest| rest.split_once("\n---\n"))
        .and_then(|(front, _)| {
            front
                .lines()
                .find_map(|l| l.strip_prefix("title: "))
                .map(|t| t.trim().trim_matches('"').to_owned())
        });
    Ok(title.unwrap_or_else(|| file.to_owned()))
}

/// The pages that show `program` beside its code, as `(title, link from a page under DIR)`.
fn shown_on(root: &Path, program: &str, generated: &Pages) -> Result<Vec<(String, String)>> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for entry in MANIFEST {
        let named = match entry.kind {
            Kind::Whole { program: p } | Kind::Fragment { program: p } => p == program,
            Kind::Skipped { .. } => false,
        };
        if !named || !seen.insert(entry.file) {
            continue;
        }
        let link = match entry.file.strip_prefix(&format!("{DOCS}/")) {
            Some(page) => format!("../{page}"),
            None => format!("{SOURCE}/{}", entry.file),
        };
        out.push((title_of(root, entry.file, generated)?, link));
    }
    Ok(out)
}

/// What each view shows, for the heading of its section.
fn view_intro(view: DiagramView) -> &'static str {
    match view {
        DiagramView::Model => {
            "The architecture: every operation the prediction reaches, without the loss \
             (DESIGN.md § 13.4)."
        }
        DiagramView::Train => {
            "The training step as a pipeline: data, parameters, forward, loss, backward and the \
             optimizer update (DESIGN.md § 13.4)."
        }
        DiagramView::Eval => "The held-out evaluation: data, forward and loss (DESIGN.md § 13.4).",
        DiagramView::Infer => {
            "Inference from a checkpoint: parameters, data, forward and the prediction \
             (DESIGN.md § 13.4)."
        }
    }
}

fn view_title(view: DiagramView) -> &'static str {
    match view {
        DiagramView::Model => "Model view",
        DiagramView::Train => "Training step",
        DiagramView::Eval => "Evaluation",
        DiagramView::Infer => "Inference",
    }
}

/// The file name of `file`, the program's source, which titles its page.
fn file_name(file: &str) -> &str {
    file.rsplit('/').next().unwrap_or(file)
}

/// One program's diagram page: every level of every view of `drawn` under its own heading,
/// anchored by [`anchor`], with links to `shown_on`, the pages that show the program beside its
/// code. `story:ir-gallery-modules` calls it for the programs it adds.
pub fn diagram_page(drawn: &Drawn, position: u32, shown_on: &[(String, String)]) -> String {
    let name = file_name(drawn.file);
    let mut out = front_matter(
        &format!("Diagrams of {name}"),
        name,
        &format!(
            "Every level of the architecture diagrams of {}, drawn from its verified Tensor IR.",
            drawn.file
        ),
        position,
    );
    out.push_str(&format!(
        "import ModelDiagram from '@site/src/components/ModelDiagram';\n\
         import diagram from '@site/data/diagrams/{key}.json';\n\n\
         Every level of [`{file}`]({SOURCE}/{file}), drawn by `modelspec-diagram` from its \
         verified Tensor IR and regenerated by `modelspec-docs` on every change. Each level has \
         its own heading, and a composite box of a diagram links to the heading of its level.",
        key = drawn.key,
        file = drawn.file,
    ));
    if !shown_on.is_empty() {
        let links: Vec<String> = shown_on
            .iter()
            .map(|(title, link)| format!("[{title}]({link})"))
            .collect();
        out.push_str(&format!(" Shown beside its code on: {}.", links.join(", ")));
    }
    out.push('\n');
    for view in &drawn.views {
        out.push_str(&format!(
            "\n## {}\n\n{}\n",
            view_title(view.view),
            view_intro(view.view)
        ));
        for (scope, _) in &view.levels {
            out.push_str(&format!(
                "\n### `{scope}` \\{{#{}}}\n\n<ModelDiagram data={{diagram}} view=\"{}\" scope=\"{scope}\" pageLink={{false}} />\n",
                anchor(view.view, scope),
                view.view.name(),
            ));
        }
    }
    out
}

/// The section's index page: every program and the train block it draws.
fn index_page(drawn: &[Drawn]) -> String {
    let mut out = front_matter(
        "Diagrams",
        "Overview",
        "One page per program the documentation shows, with every level of its architecture diagrams.",
        1,
    );
    out.push_str(
        "Every program the documentation shows beside its code has a page here with every level \
         of its diagrams, drawn by `modelspec-diagram` from the verified Tensor IR and \
         regenerated by `modelspec-docs` on every change (DESIGN.md § 13.6). A diagram beside a \
         program's code shows its top level and links here.\n\n",
    );
    for program in drawn {
        let root = program
            .views
            .first()
            .map(|v| format!(", train block `{}`", v.root))
            .unwrap_or_default();
        out.push_str(&format!(
            "- [{}](./{}): [`{file}`]({SOURCE}/{file}){root}.\n",
            file_name(program.file),
            page_name(program.key),
            file = program.file,
        ));
    }
    out
}

/// The section's pages, by file name: the index and one page per drawn program, in the order of
/// [`crate::diagrams::PROGRAMS`].
pub fn pages(root: &Path, drawn: &[Drawn], generated: &Pages) -> Result<Pages> {
    let mut pages = Pages::from([("index.md".to_owned(), index_page(drawn))]);
    for (n, program) in drawn.iter().enumerate() {
        let position = u32::try_from(n + 2).map_err(|e| e.to_string())?;
        pages.insert(
            page_name(program.key),
            diagram_page(program, position, &shown_on(root, program.key, generated)?),
        );
    }
    Ok(pages)
}

/// The anchors of the custom heading ids of `page`, written `\{#…}` as MDX needs the brace escaped.
fn anchors_of(page: &str) -> BTreeSet<String> {
    page.lines()
        .filter(|l| l.starts_with('#'))
        .filter_map(|l| {
            let start = l.rfind("\\{#")?;
            l[start + 3..].strip_suffix('}').map(str::to_owned)
        })
        .collect()
}

/// The `href` (or `xlink:href`) of every `<a>` element of `svg`.
fn link_targets(svg: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = svg;
    while let Some(start) = rest.find("<a ") {
        let tag = &rest[start..];
        let end = tag.find('>').unwrap_or(tag.len());
        let tag = &tag[..end];
        for attribute in [" href=\"", " xlink:href=\""] {
            if let Some(at) = tag.find(attribute) {
                let value = &tag[at + attribute.len()..];
                out.push(value[..value.find('"').unwrap_or(value.len())].to_owned());
            }
        }
        rest = &rest[start + 3..];
    }
    out
}

/// Acceptance `docs.diagrams.links-resolve`: every `<a href>` of every SVG of `drawn` that does
/// not name a heading anchor of the diagram page it points at, among `pages`, the diagram pages
/// by file name ([`page_name`]). `story:ir-gallery-modules` calls it for its own programs.
pub fn unresolved_links(drawn: &[Drawn], pages: &Pages) -> Vec<String> {
    let prefix = format!("{BASE_URL}docs/diagrams/");
    let mut problems = Vec::new();
    for program in drawn {
        for view in &program.views {
            for (scope, svg) in &view.levels {
                for href in link_targets(svg) {
                    let resolved = href
                        .strip_prefix(&prefix)
                        .and_then(|rest| rest.split_once('#'))
                        .and_then(|(page, anchor)| {
                            pages
                                .get(&page_name(page))
                                .map(|text| anchors_of(text).contains(anchor))
                        })
                        .unwrap_or(false);
                    if !resolved {
                        problems.push(format!(
                            "{}: level {scope} of the {} view links {href}, which names no heading of a diagram page",
                            program.key,
                            view.view.name()
                        ));
                    }
                }
            }
        }
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagrams::{self, DrawnView};

    fn repository() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn drawn() -> Vec<Drawn> {
        diagrams::files(&repository()).unwrap().drawn
    }

    #[test]
    fn a_fence_inside_a_fence_and_other_languages_are_not_blocks() {
        let text = "a\n````md\n```modelspec\nmodel X {}\n```\n````\n\n```modelspec title=\"x\"\nmodel Y {}\n```\n~~~sh\nls\n~~~\n  ```modelspec\nmodel Z {}\n  ```\n";
        let blocks = scan_text("f.md", text);
        assert_eq!(blocks.len(), 2, "{blocks:?}");
        assert_eq!(
            (blocks[0].line, blocks[0].end, blocks[0].ordinal),
            (8, 10, 1)
        );
        assert_eq!(blocks[0].text, "model Y {}\n");
        assert_eq!(blocks[1].opens(), "model Z {}");
        assert_eq!(blocks[1].ordinal, 2);
    }

    /// Acceptance 1, `docs.blocks.every-block-listed`: the manifest lists exactly the blocks of
    /// the repository, and a block added to `README.md`, `DESIGN.md` or a page fails with its file
    /// and line.
    #[test]
    fn docs_blocks_every_block_listed() {
        let root = repository();
        let blocks = scan(&root).unwrap();
        assert_eq!(blocks.len(), 14);
        assert_eq!(unlisted(&blocks, MANIFEST), Vec::<String>::new());
        assert!(blocks.iter().all(|b| !b.file.starts_with(GALLERY)));
        for file in [
            "README.md",
            "DESIGN.md",
            "website/docs/concepts/language.mdx",
            "website/docs/roadmap.md",
        ] {
            let text = fs::read_to_string(root.join(file)).unwrap();
            let added = format!("{text}\n```modelspec\nmodel Added {{}}\n```\n");
            let line = text.lines().count() + 2;
            let mut changed: Vec<Block> =
                blocks.iter().filter(|b| b.file != file).cloned().collect();
            changed.extend(scan_text(file, &added));
            let problems = unlisted(&changed, MANIFEST);
            assert_eq!(problems.len(), 1, "{problems:?}");
            assert!(
                problems[0].starts_with(&format!("{file}:{line}: ")),
                "{problems:?}"
            );
        }
        let mut missing = blocks.clone();
        missing.retain(|b| b.file != "DESIGN.md");
        assert_eq!(
            unlisted(&missing, MANIFEST),
            ["DESIGN.md: the manifest lists ```modelspec block 1, which the file does not have"]
        );
    }

    /// Acceptance 2, `docs.blocks.whole-has-diagram`.
    #[test]
    fn docs_blocks_whole_has_diagram() {
        let root = repository();
        let blocks = scan(&root).unwrap();
        let drawn = drawn();
        assert_eq!(
            missing_diagrams(&root, &blocks, MANIFEST, &drawn).unwrap(),
            Vec::<String>::new()
        );
        assert_eq!(drawn_after(&blocks, MANIFEST).len(), 8);
        let readme = fs::read_to_string(root.join("README.md")).unwrap();
        assert!(readme.contains("](website/static/img/diagrams/linear-reg/model/Fit.svg)"));
        assert!(
            outside_their_program(&root, &blocks, MANIFEST)
                .unwrap()
                .is_empty()
        );
        let mut wrong = MANIFEST.to_vec();
        for entry in &mut wrong {
            if entry.file == "website/docs/index.mdx" {
                entry.kind = Kind::Whole {
                    program: "two-layer-relu",
                };
            }
        }
        let problems = missing_diagrams(&root, &blocks, &wrong, &drawn).unwrap();
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].contains("two-layer-relu"), "{problems:?}");
        assert_eq!(
            outside_their_program(&root, &blocks, &wrong).unwrap().len(),
            1
        );
    }

    /// Acceptance 3, `docs.blocks.fragments-in-fixture`.
    #[test]
    fn docs_blocks_fragments_in_fixture() {
        let root = repository();
        let blocks = scan(&root).unwrap();
        let fragments: Vec<&Block> = blocks
            .iter()
            .filter(|b| {
                matches!(
                    entry_of(MANIFEST, b).map(|e| e.kind),
                    Some(Kind::Fragment { .. })
                )
            })
            .collect();
        assert_eq!(fragments.len(), 3);
        assert!(contains_lines("a {\n  b;\n  c;\n}\n", "a {\n b;\n}\n"));
        assert!(!contains_lines("a {\n  b;\n  c;\n}\n", "c;\nb;\n"));
        assert!(!contains_lines("a {\n  b;\n}\n", "a {\nb; c;\n"));
        assert!(
            outside_their_program(&root, &blocks, MANIFEST)
                .unwrap()
                .is_empty()
        );
        let mut changed = blocks.clone();
        for block in &mut changed {
            if block.file.ends_with("write-a-model.mdx") && block.ordinal == 2 {
                block.text = block.text.replace("relu(", "gelu(");
            }
        }
        let problems = outside_their_program(&root, &changed, MANIFEST).unwrap();
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].contains("write-a-model.mdx:"), "{problems:?}");
    }

    /// Acceptance 4, `docs.blocks.skips-still-refused`.
    #[test]
    fn docs_blocks_skips_still_refused() {
        let blocks = scan(&repository()).unwrap();
        assert_eq!(skips_that_compile(&blocks, MANIFEST), Vec::<String>::new());
        let mut compiling = blocks.clone();
        let whole = blocks
            .iter()
            .find(|b| b.file == "website/docs/index.mdx")
            .unwrap()
            .text
            .clone();
        for block in &mut compiling {
            if block.file == "DESIGN.md" {
                block.text = whole.clone();
            }
        }
        let problems = skips_that_compile(&compiling, MANIFEST);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(
            problems[0].starts_with("DESIGN.md:") && problems[0].contains("compiles now"),
            "{problems:?}"
        );
    }

    /// Acceptance 5, `docs.diagrams.links-resolve`: every link of every generated SVG resolves,
    /// and a link to a level the page lacks, or to another page, does not.
    #[test]
    fn docs_diagrams_links_resolve() {
        let root = repository();
        let drawn = drawn();
        let pages = pages(&root, &drawn, &Pages::new()).unwrap();
        assert_eq!(unresolved_links(&drawn, &pages), Vec::<String>::new());
        for program in &drawn {
            let page = &pages[&page_name(program.key)];
            for view in &program.views {
                for (scope, _) in &view.levels {
                    assert!(anchors_of(page).contains(&anchor(view.view, scope)));
                }
            }
        }
        let good = level_href("linear-reg", DiagramView::Model, "Fit");
        assert_eq!(good, "/modelspec/docs/diagrams/linear-reg#model-Fit");
        let svg = |href: &str| {
            format!("<svg><a href=\"{href}\"><rect/></a><use href=\"#ms-op-add\"/></svg>")
        };
        let linked = |href: &str| Drawn {
            key: "linear-reg",
            file: "examples/linear_regression.mspec",
            views: vec![DrawnView {
                view: DiagramView::Model,
                root: "Fit".to_owned(),
                levels: vec![("Fit".to_owned(), svg(href))],
            }],
        };
        assert!(unresolved_links(&[linked(&good)], &pages).is_empty());
        for bad in [
            level_href("linear-reg", DiagramView::Model, "Fit/forward@0"),
            level_href("no-such-program", DiagramView::Model, "Fit"),
            "FitTwoLayer.svg".to_owned(),
        ] {
            let problems = unresolved_links(&[linked(&bad)], &pages);
            assert_eq!(problems.len(), 1, "{bad}: {problems:?}");
        }
    }

    #[test]
    fn every_program_of_the_manifest_is_drawn_and_has_a_page() {
        let root = repository();
        let pages = pages(&root, &drawn(), &Pages::new()).unwrap();
        for entry in MANIFEST {
            if let Kind::Whole { program } | Kind::Fragment { program } = entry.kind {
                assert!(diagrams::program(program).is_some(), "{program}");
                assert!(pages.contains_key(&page_name(program)), "{program}");
            }
        }
        assert_eq!(pages.len(), diagrams::PROGRAMS.len() + 1);
        let page = &pages["linear-reg.mdx"];
        assert!(page.contains("### `Fit` \\{#model-Fit}"), "{page}");
        assert!(page.contains("](../index.mdx)"), "{page}");
        assert!(page.contains(&format!("]({SOURCE}/README.md)")), "{page}");
    }

    #[test]
    fn an_anchor_keeps_letters_digits_dashes_and_underscores() {
        assert_eq!(
            anchor(DiagramView::Model, "Fit/encoder@1"),
            "model-Fit-encoder-1"
        );
        assert_eq!(anchor(DiagramView::Train, "a_b-C"), "train-a_b-C");
    }
}
