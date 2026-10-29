//! `modelspec diagram`: draw one level of a view of a `.mspec` file (DESIGN.md § 13.7).
//!
//! The file is checked and compiled as `modelspec ir` does it: a file that does not compile prints
//! its diagnostics to stderr and exits 1, a file that cannot be read is one line and exits 1. A
//! clean file prints the root level of its view to stdout, as SVG (`--format svg`, the default),
//! as the `DiagramRender` JSON (`json`) or as the `DiagramGraph` JSON (`graph`), the records of
//! `spec/domains/diagram.yaml`. `--scope` picks the root, `--depth` how many levels below it are
//! open (the view's default when absent) and `--expand` further scopes to open; a key the graph
//! does not have is a usage error, exit 2, that lists the graph's keys. `--out <dir>` instead
//! writes every scope's level as `<dir>/<scope-key>.svg` and prints each path it wrote.
//!
//! This build draws the `model` view; the `train`, `eval` and `infer` graphs arrive with
//! story:diagram-pipelines, and until then each of them is refused on one line, exit 1.

use std::io::Write;
use std::path::{Path, PathBuf};

use modelspec_core::compile;
use modelspec_diagram::graph::{self, IrLines};
use modelspec_diagram::layout::{RenderRequest, render};
use modelspec_diagram::model::{DiagramGraph, DiagramView, ScopeKey};
use modelspec_diagram::{json, svg};

/// Arguments of `modelspec diagram`.
#[derive(clap::Args)]
pub struct Args {
    /// The .mspec file to draw.
    pub path: PathBuf,
    /// The view to draw.
    #[arg(long, value_enum, default_value_t = View::Model)]
    pub view: View,
    /// The scope drawn as the root, by its key; the view's outermost scope when absent.
    #[arg(long, value_name = "KEY", conflicts_with = "out")]
    pub scope: Option<String>,
    /// How many levels below the root are drawn open; 1 in the model view and 0 in a pipeline view
    /// when absent.
    #[arg(long, value_name = "N")]
    pub depth: Option<u32>,
    /// A scope drawn open beyond the depth, by its key; repeatable.
    #[arg(long, value_name = "KEY")]
    pub expand: Vec<String>,
    /// What to print.
    #[arg(long, value_enum, default_value_t = Format::Svg)]
    pub format: Format,
    /// Write every scope's level as `<DIR>/<scope-key>.svg` instead of printing the root's.
    #[arg(long, value_name = "DIR")]
    pub out: Option<PathBuf>,
}

/// A view `modelspec diagram` draws (DESIGN.md § 13.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum View {
    /// The architecture: the forward program that reaches the prediction.
    Model,
    /// The training step as stages: data, parameters, forward, loss, backward, optimizer.
    Train,
    /// The held-out evaluation: data, forward and loss without dropout.
    Eval,
    /// Inference from a checkpoint to a prediction.
    Infer,
}

impl View {
    fn diagram(self) -> DiagramView {
        match self {
            Self::Model => DiagramView::Model,
            Self::Train => DiagramView::Train,
            Self::Eval => DiagramView::Eval,
            Self::Infer => DiagramView::Infer,
        }
    }
}

/// What `modelspec diagram` prints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Format {
    /// A standalone SVG.
    Svg,
    /// The DiagramRender JSON: the laid-out level.
    Json,
    /// The DiagramGraph JSON: the whole graph of the view.
    Graph,
}

/// Runs `modelspec diagram` and returns the process exit code.
pub fn run(args: Args, stdout: &mut dyn Write, stderr: &mut dyn Write) -> u8 {
    let file = args.path.display().to_string();
    if args.out.is_some() && args.format != Format::Svg {
        let _ = writeln!(
            stderr,
            "modelspec diagram {file}: --out writes SVG files and takes no --format {}",
            match args.format {
                Format::Json => "json",
                _ => "graph",
            }
        );
        return crate::EXIT_USAGE;
    }
    let source = match std::fs::read_to_string(&args.path) {
        Ok(source) => source,
        Err(error) => {
            let _ = writeln!(stderr, "modelspec diagram {file}: {error}");
            return crate::EXIT_FAILURE;
        }
    };
    let compiled = compile::compile_hir(&file, &source).and_then(|hir| {
        compile::compile_training(&file, &source).map(|compiled| (hir.train.name, compiled))
    });
    let (program, compiled) = match compiled {
        Ok(compiled) => compiled,
        Err(diagnostics) => {
            for diagnostic in &diagnostics {
                let _ = writeln!(stderr, "{diagnostic}");
            }
            return crate::EXIT_FAILURE;
        }
    };
    let graph = match args.view {
        View::Model => graph::model(&compiled, &program, &source),
        other => {
            let _ = writeln!(
                stderr,
                "modelspec diagram {file}: the {} view is not drawn by this build; the model view is",
                other.diagram().name()
            );
            return crate::EXIT_FAILURE;
        }
    };
    let lines = graph::ir_lines(&compiled);
    let mut named = args.expand.clone();
    named.extend(args.scope.clone());
    if let Some(unknown) = named
        .iter()
        .find(|key| !graph.scopes.iter().any(|s| &s.key.0 == *key))
    {
        let _ = writeln!(
            stderr,
            "modelspec diagram {file}: the {} view has no scope `{unknown}`; its scope keys are:",
            graph.view.name()
        );
        for scope in &graph.scopes {
            let _ = writeln!(stderr, "  {}", scope.key.0);
        }
        return crate::EXIT_USAGE;
    }
    let expanded: Vec<ScopeKey> = args.expand.into_iter().map(ScopeKey).collect();
    let written = match &args.out {
        Some(dir) => write_levels(&graph, &lines, dir, args.depth, &expanded, stdout),
        None => {
            let request = RenderRequest {
                root: args.scope.map(ScopeKey),
                depth: args.depth,
                expanded,
            };
            let printed = match args.format {
                Format::Svg => svg::svg(&graph, &render(&graph, &request), &lines),
                Format::Json => json::render_json(&render(&graph, &request)),
                Format::Graph => json::graph_json(&graph),
            };
            stdout
                .write_all(printed.as_bytes())
                .and_then(|()| stdout.flush())
                .map_err(|error| format!("cannot write the diagram: {error}"))
        }
    };
    match written {
        Ok(()) => crate::EXIT_SUCCESS,
        Err(error) => {
            let _ = writeln!(stderr, "modelspec diagram {file}: {error}");
            crate::EXIT_FAILURE
        }
    }
}

/// Writes the level of every scope of `graph` as `<dir>/<scope-key>.svg`, each at `depth` (0 when
/// absent, DESIGN.md § 13.3) with `expanded` open, and prints each path.
fn write_levels(
    graph: &DiagramGraph,
    lines: &IrLines,
    dir: &Path,
    depth: Option<u32>,
    expanded: &[ScopeKey],
    stdout: &mut dyn Write,
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    for scope in &graph.scopes {
        let request = RenderRequest {
            root: Some(scope.key.clone()),
            depth: Some(depth.unwrap_or(0)),
            expanded: expanded.to_vec(),
        };
        let path = dir.join(format!("{}.svg", scope.key.0));
        std::fs::write(&path, svg::svg(graph, &render(graph, &request), lines))
            .map_err(|error| format!("{}: {error}", path.display()))?;
        writeln!(stdout, "{}", path.display())
            .map_err(|error| format!("cannot write the path: {error}"))?;
    }
    stdout
        .flush()
        .map_err(|error| format!("cannot write the path: {error}"))
}
