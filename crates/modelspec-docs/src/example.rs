//! The example pages and the landing page's terminal, recorded by running the `modelspec` command
//! line (`modelspec_cli::dispatch`, the code the binary runs) on the example models.
//!
//! The models are `examples/*.mspec` and the ones the site adds under
//! `crates/modelspec-docs/models/`. Every command is parsed by the binary's own clap definition,
//! so a page cannot show an option the binary does not have. Paths are relative to the repository
//! root, which `generate` makes the working directory, so the output names the file as a reader
//! who runs the command from a checkout sees it.

use clap::Parser;
use serde_json::json;

use crate::{Pages, Result, SOURCE, fenced, front_matter, io};

/// The landing page's terminal, a `b10x-terminal/1` document under [`crate::DATA`].
pub const TERMINAL: &str = "modelspec.terminal.json";

/// One run of the command line, as recorded.
pub struct Recorded {
    /// The command line, `modelspec ...`.
    pub command: String,
    pub stdout: String,
    pub stderr: String,
    pub exit: u8,
}

/// Runs `modelspec <args>` in-process and records what it prints and its exit status.
pub fn record(args: &[&str]) -> Result<Recorded> {
    let argv = std::iter::once("modelspec").chain(args.iter().copied());
    let cli = modelspec_cli::Cli::try_parse_from(argv)
        .map_err(|e| format!("modelspec {}: {e}", args.join(" ")))?;
    let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
    let exit = modelspec_cli::dispatch(cli.command, &mut stdout, &mut stderr);
    let utf8 = |bytes: Vec<u8>| {
        String::from_utf8(bytes).map_err(|e| format!("modelspec {}: {e}", args.join(" ")))
    };
    Ok(Recorded {
        command: format!("modelspec {}", args.join(" ")),
        stdout: utf8(stdout)?,
        stderr: utf8(stderr)?,
        exit,
    })
}

/// One example page.
struct Example {
    page: &'static str,
    title: &'static str,
    label: &'static str,
    description: &'static str,
    /// The model, relative to the repository root.
    file: &'static str,
    /// What the page shows, before the model.
    intro: &'static str,
    /// The arguments of each recorded command; `{file}` stands for [`Example::file`].
    runs: &'static [&'static [&'static str]],
    /// The exit status every run must have, so that a model that stops checking fails generation
    /// instead of quietly turning its page into an error page.
    expected: &'static [u8],
}

const EXAMPLES: &[Example] = &[
    Example {
        page: "linear-regression.md",
        title: "Example: linear regression",
        label: "Linear regression",
        description: "The repository's example model, checked, lowered and trained on the CPU reference, with the output each command prints.",
        file: "examples/linear_regression.mspec",
        intro: "The example of the repository: one `[4, 1]` weight, a batch of 8 rows of 4 features, \
                mean squared error and plain SGD. `task check-examples` checks it on every change, and the \
                tests of `modelspec-cli` hold its IR and its training run.",
        runs: &[
            &["check", "{file}"],
            &["ir", "{file}", "--stage", "tensor"],
            &["ir", "{file}", "--stage", "training"],
            &["train", "{file}", "--backend", "cpu"],
        ],
        expected: &[0, 0, 0, 0],
    },
    Example {
        page: "two-layer-relu.md",
        title: "Example: two layers with relu",
        label: "Two layers with relu",
        description: "A two-layer model with a relu between the layers, written for this site, checked and trained on the CPU reference.",
        file: "crates/modelspec-docs/models/two_layer_relu.mspec",
        intro: "Two weights and a `relu` between them, written in the M0 subset: a `let` binding, \
                nested operator calls and two parameters with one canonical path each. \
                [Write a model](../guides/write-a-model.md) builds it step by step.",
        runs: &[
            &["check", "{file}"],
            &["ir", "{file}", "--stage", "tensor"],
            &["train", "{file}", "--backend", "cpu"],
        ],
        expected: &[0, 0, 0],
    },
    Example {
        page: "shape-error.md",
        title: "Example: a shape error",
        label: "A shape error",
        description: "A model whose matmul operands do not fit, and the diagnostic modelspec check prints for it.",
        file: "crates/modelspec-docs/models/matmul_mismatch.mspec",
        intro: "The weight is `[3, 1]` but the input has 4 features, so `matmul(x, w)` has no \
                result type. `modelspec check` refuses the file before anything runs, and `train` \
                refuses it the same way. [Read a diagnostic](../guides/read-a-diagnostic.md) walks \
                through the output.",
        runs: &[
            &["check", "{file}"],
            &["train", "{file}", "--backend", "cpu"],
        ],
        expected: &[1, 1],
    },
    Example {
        page: "beyond-m0.md",
        title: "Example: syntax beyond M0",
        label: "Syntax beyond M0",
        description: "The target syntax of DESIGN.md § 2, which the M0 parser refuses construct by construct.",
        file: "examples/tiny_mlp.mspec",
        intro: "The model of DESIGN.md § 2: a generic `module`, `layer` instances, generic \
                dimensions and a `vulkan(0)` device. None of these is in the M0 subset. The parser \
                names each construct it does not have with its own code instead of a generic syntax \
                error, so the output says what is missing rather than what is wrong.",
        runs: &[&["check", "{file}"]],
        expected: &[1],
    },
];

/// `source` without its leading `//` comment lines and the blank lines after them.
fn without_leading_comments(source: &str) -> &str {
    let mut rest = source;
    while let Some(line) = rest.lines().next() {
        let trimmed = line.trim();
        if !(trimmed.starts_with("//") || trimmed.is_empty()) {
            break;
        }
        rest = &rest[(line.len() + 1).min(rest.len())..];
    }
    rest
}

/// The Markdown of one recorded run.
fn run_section(run: &Recorded) -> String {
    let mut out = format!("\n### `{}`\n\n", run.command);
    out.push_str(&format!("Exit status `{}`.", run.exit));
    if run.stdout.is_empty() && run.stderr.is_empty() {
        out.push_str(" It prints nothing.\n");
        return out;
    }
    out.push('\n');
    if !run.stdout.is_empty() {
        out.push_str("\nstdout:\n\n");
        out.push_str(&fenced("text", &run.stdout));
    }
    if !run.stderr.is_empty() {
        out.push_str("\nstderr:\n\n");
        out.push_str(&fenced("text", &run.stderr));
    }
    out
}

fn page(example: &Example, position: u32, source: &str, runs: &[Recorded]) -> String {
    let mut out = front_matter(example.title, example.label, example.description, position);
    out.push_str(&format!(
        "{}\n\nThe output below was recorded by `modelspec-docs` running the `modelspec` command \
         line on the model; `task docs-check` fails when a run prints anything else.\n\n\
         ## The model\n\n[`{file}`]({SOURCE}/{file}):\n\n",
        example.intro,
        file = example.file,
    ));
    out.push_str(&fenced("modelspec", without_leading_comments(source)));
    out.push_str("\n## Runs\n");
    for run in runs {
        out.push_str(&run_section(run));
    }
    out
}

/// The terminal of the landing page: the linear-regression example checked and trained on the
/// CPU. No page records a Vulkan run, whose output depends on the machine that generates it.
fn terminal(runs: &[Recorded]) -> Result<String> {
    let entries: Vec<serde_json::Value> = runs
        .iter()
        .map(|run| {
            json!({
                "command": run.command,
                "output": format!("{}{}", run.stdout, run.stderr).trim_end().to_owned(),
                "exitCode": run.exit,
            })
        })
        .collect();
    let document = json!({
        "format": "b10x-terminal/1",
        "title": "examples/linear_regression.mspec",
        "recordedWith": format!("modelspec {}, recorded by modelspec-docs", env!("CARGO_PKG_VERSION")),
        "tones": {},
        "entries": entries,
    });
    serde_json::to_string_pretty(&document)
        .map(|text| text + "\n")
        .map_err(|e| e.to_string())
}

/// The generated example pages and the terminal document.
pub struct Generated {
    pub pages: Pages,
    pub terminal: String,
}

/// Records every example. The working directory must be the repository root.
pub fn pages() -> Result<Generated> {
    let mut pages = Pages::new();
    let mut landing = Vec::new();
    for (n, example) in EXAMPLES.iter().enumerate() {
        let source = std::fs::read_to_string(example.file).map_err(io(example.file))?;
        let mut runs = Vec::new();
        for (args, expected) in example.runs.iter().zip(example.expected) {
            let args: Vec<&str> = args
                .iter()
                .map(|arg| if *arg == "{file}" { example.file } else { arg })
                .collect();
            let run = record(&args)?;
            if run.exit != *expected {
                return Err(format!(
                    "{} exited {}, the example page expects {expected}: {}",
                    run.command, run.exit, run.stderr
                ));
            }
            runs.push(run);
        }
        if example.page == "linear-regression.md" {
            for keep in ["check", "train"] {
                landing.extend(
                    runs.iter()
                        .filter(|run| run.command.starts_with(&format!("modelspec {keep} ")))
                        .map(|run| Recorded {
                            command: run.command.clone(),
                            stdout: run.stdout.clone(),
                            stderr: run.stderr.clone(),
                            exit: run.exit,
                        }),
                );
            }
        }
        let position = u32::try_from(n + 1).map_err(|e| e.to_string())?;
        pages.insert(
            example.page.to_owned(),
            page(example, position, &source, &runs),
        );
    }
    Ok(Generated {
        pages,
        terminal: terminal(&landing)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leading_comment_lines_are_dropped_and_the_rest_is_kept() {
        assert_eq!(
            without_leading_comments("// a\n// b\n\nmodel M {\n// kept\n}\n"),
            "model M {\n// kept\n}\n"
        );
        assert_eq!(without_leading_comments("model M {}\n"), "model M {}\n");
    }

    #[test]
    fn a_run_is_parsed_by_the_binary_s_own_definition() {
        assert!(record(&["train", "x.mspec", "--backend", "gpu"]).is_err());
        let missing = record(&["check", "no-such-file.mspec"]).unwrap();
        assert_eq!(missing.exit, modelspec_cli::EXIT_FAILURE);
        assert!(
            missing
                .stderr
                .starts_with("modelspec check no-such-file.mspec: ")
        );
    }

    #[test]
    fn a_silent_run_says_so_and_a_failing_one_shows_its_stderr() {
        let silent = Recorded {
            command: "modelspec check a.mspec".into(),
            stdout: String::new(),
            stderr: String::new(),
            exit: 0,
        };
        assert!(run_section(&silent).contains("Exit status `0`. It prints nothing."));
        let failing = Recorded {
            stderr: "E-TYPE-012: x\n".into(),
            exit: 1,
            ..silent
        };
        let section = run_section(&failing);
        assert!(section.contains("stderr:\n\n```text\nE-TYPE-012: x\n```"));
        assert!(!section.contains("stdout:"));
    }

    #[test]
    fn the_terminal_is_b10x_terminal_with_exit_codes() {
        let text = terminal(&[Recorded {
            command: "modelspec check a.mspec".into(),
            stdout: String::new(),
            stderr: String::new(),
            exit: 0,
        }])
        .unwrap();
        let document: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(document["format"], "b10x-terminal/1");
        assert_eq!(document["entries"][0]["exitCode"], 0);
        assert_eq!(document["entries"][0]["output"], "");
    }
}
