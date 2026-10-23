//! The CLI reference page, walked from the clap definition of `modelspec` (`modelspec_cli::Cli`).

use clap::CommandFactory;

use crate::{SOURCE, cell, front_matter};

/// Every subcommand below `command`, depth first, without clap's `help`.
fn commands(command: &clap::Command, out: &mut Vec<clap::Command>) {
    for sub in command.get_subcommands() {
        if sub.get_name() == "help" {
            continue;
        }
        out.push(sub.clone());
        commands(sub, out);
    }
}

/// The heading anchor Docusaurus gives `## \`modelspec check\``.
fn anchor(command: &clap::Command) -> String {
    command
        .get_bin_name()
        .unwrap_or(command.get_name())
        .replace(' ', "-")
}

fn text(value: Option<&clap::builder::StyledStr>) -> String {
    value.map(ToString::to_string).unwrap_or_default()
}

/// `text` ending in a full stop, so a list of possible values can follow it.
fn sentence(text: String) -> String {
    let text = text.trim_end().to_owned();
    if text.is_empty() || text.ends_with('.') {
        text
    } else {
        text + "."
    }
}

/// What an argument may be: its possible values with their help, one per line of the cell, or
/// nothing.
fn possible_values(arg: &clap::Arg) -> String {
    let values: Vec<String> = arg
        .get_possible_values()
        .iter()
        .filter(|value| !value.is_hide_set())
        .map(|value| match value.get_help() {
            Some(help) => format!("`{}`: {}", value.get_name(), cell(&help.to_string())),
            None => format!("`{}`", value.get_name()),
        })
        .collect();
    if values.is_empty() {
        String::new()
    } else {
        format!(" One of:<br/>{}", values.join("<br/>"))
    }
}

/// One row per argument of `command`: how it is written, whether it is required, its default and
/// its help.
fn arguments(command: &clap::Command) -> Vec<[String; 4]> {
    command
        .get_arguments()
        .filter(|arg| !matches!(arg.get_id().as_str(), "help" | "version"))
        .map(|arg| {
            let value = arg
                .get_value_names()
                .and_then(|names| names.first().map(ToString::to_string))
                .unwrap_or_else(|| arg.get_id().as_str().to_uppercase());
            let written = match arg.get_long() {
                Some(long) if !arg.get_action().takes_values() => format!("--{long}"),
                Some(long) => format!("--{long} <{value}>"),
                None => format!("<{value}>"),
            };
            let defaults: Vec<String> = arg
                .get_default_values()
                .iter()
                .map(|default| default.to_string_lossy().into_owned())
                .collect();
            [
                format!("`{written}`"),
                if arg.is_required_set() { "yes" } else { "no" }.to_owned(),
                if defaults.is_empty() {
                    "none".to_owned()
                } else {
                    format!("`{}`", defaults.join(", "))
                },
                cell(
                    &(sentence(text(arg.get_long_help().or(arg.get_help())))
                        + &possible_values(arg)),
                ),
            ]
        })
        .collect()
}

/// The CLI reference page for the command line `root`.
pub fn page_for(mut root: clap::Command) -> String {
    root.build();
    let name = root.get_name().to_owned();
    let mut all = Vec::new();
    commands(&root, &mut all);

    let mut out = front_matter(
        &format!("{name} CLI reference"),
        "CLI",
        &format!(
            "Every command and option of the {name} command line, generated from its definition."
        ),
        1,
    );
    out.push_str(&format!(
        "Generated from the clap definition of the `{name}` command line in \
         [`crates/modelspec-cli/src/lib.rs`]({SOURCE}/crates/modelspec-cli/src/lib.rs) by \
         `modelspec-docs`. It lists only commands that exist.\n\n> {}\n\n`{name} --version` \
         prints its version, and `--help` after any command prints the text below.\n\n\
         ## Exit status\n\n| Code | Meaning |\n|---|---|\n\
         | `{}` | success |\n| `{}` | diagnostics or a runtime error |\n| `{}` | a usage error |\n\n\
         ## Commands\n\n| Command | What it does |\n|---|---|\n",
        cell(&text(root.get_about())),
        modelspec_cli::EXIT_SUCCESS,
        modelspec_cli::EXIT_FAILURE,
        modelspec_cli::EXIT_USAGE,
    ));
    for command in &all {
        let written = command.get_bin_name().unwrap_or(command.get_name());
        out.push_str(&format!(
            "| [`{written}`](#{}) | {} |\n",
            anchor(command),
            cell(&text(command.get_about()))
        ));
    }
    for command in &mut all {
        let written = command
            .get_bin_name()
            .unwrap_or(command.get_name())
            .to_owned();
        out.push_str(&format!(
            "\n## `{written}`\n\n{}\n\n```text\n{}\n```\n",
            cell(&text(command.get_long_about().or(command.get_about()))),
            command.render_usage().to_string().trim_end()
        ));
        let rows = arguments(command);
        if rows.is_empty() {
            continue;
        }
        if rows.iter().all(|row| row[1] == "yes" && row[2] == "none") {
            out.push_str(
                "\nEvery argument is required and has no default.\n\n\
                 | Argument | Meaning |\n|---|---|\n",
            );
            for row in rows {
                out.push_str(&format!("| {} | {} |\n", row[0], row[3]));
            }
        } else {
            out.push_str("\n| Argument | Required | Default | Meaning |\n|---|---|---|---|\n");
            for row in rows {
                out.push_str(&format!("| {} |\n", row.join(" | ")));
            }
        }
    }
    out
}

/// The CLI reference page for `modelspec`.
pub fn page() -> String {
    page_for(modelspec_cli::Cli::command())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_command_option_and_possible_value_is_listed() {
        let page = page();
        assert!(page.contains("# generated by modelspec-docs, do not edit"));
        assert!(page.contains("custom_edit_url: null"));
        assert!(page.contains("| [`modelspec check`](#modelspec-check) |"));
        assert!(page.contains("## `modelspec train`"));
        assert!(page.contains("| [`modelspec infer`](#modelspec-infer) |"));
        assert!(page.contains("| [`modelspec compile`](#modelspec-compile) |"));
        assert!(page.contains("| `--device-profile <PATH>` | no | none |"));
        assert!(
            page.contains("Usage: modelspec infer --backend <BACKEND> --checkpoint <PATH> <PATH>")
        );
        assert!(page.contains("| `--resume <PATH>` | no | none |"));
        assert!(page.contains("Usage: modelspec train [OPTIONS] --backend <BACKEND> <PATH>"));
        assert!(page.contains(
            "Every argument is required and has no default.\n\n| Argument | Meaning |\n|---|---|\n\
             | `<PATH>` | The .mspec file to check. |"
        ));
        let (check, _) = page.split_once("## `modelspec compile`").unwrap();
        let (_, check) = check.split_once("## `modelspec check`").unwrap();
        assert!(!check.contains("| Required |"));
        let (_, train) = page.split_once("## `modelspec train`").unwrap();
        assert!(train.contains("| Argument | Required | Default | Meaning |"));
        assert!(train.contains("| `--serve <ADDR>` | no | none |"));
        assert!(page.contains("One of:<br/>`cpu`: The CPU interpreter"));
        assert!(page.contains("<br/>`training`: Training IR"));
        assert!(page.contains("<br/>`kernel`: Kernel IR"));
        assert!(page.contains("| `2` | a usage error |"));
        assert!(!page.contains("modelspec help"));
    }
}
