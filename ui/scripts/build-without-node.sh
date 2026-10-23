#!/bin/sh
# observatory.ui.builds-without-node (story:observatory-ui-live-loss-chart, Acceptance 7).
#
# Runs `task build test` from the repository root with a PATH that holds only the tools listed
# below, as symlinks in a fresh directory, so no node, npm, npx or pnpm can be found whatever the
# machine has installed. It guards that no cargo build step and no test of `task test` needs Node:
# the UI is built by `task ui` alone. `task ci` does not run this script, because `task ci` itself
# needs pnpm.
#
# Tools beyond the story's list are the ones the tests of `task test` start: kill, mkfifo,
# prlimit, readelf and spirv-val.
set -eu

# The executable file `$1` on PATH. `command -v` names a shell builtin such as kill by its bare
# name, and the tests start the program, so a name that is not a path is looked up in PATH.
executable() {
    found=$(command -v "$1") || return 1
    case $found in
    /*) printf '%s\n' "$found" && return 0 ;;
    esac
    old_ifs=$IFS
    IFS=:
    for dir in $PATH; do
        if [ -f "$dir/$1" ] && [ -x "$dir/$1" ]; then
            IFS=$old_ifs
            printf '%s\n' "$dir/$1"
            return 0
        fi
    done
    IFS=$old_ifs
    return 1
}

bin=$(mktemp -d)
trap 'rm -rf "$bin"' EXIT
trap 'exit 1' HUP INT TERM

for t in cargo rustc rustdoc cc ld as ar task sh env cat mkdir rm \
    kill mkfifo prlimit readelf spirv-val; do
    path=$(executable "$t") || {
        echo "build-without-node: $t is not on PATH" >&2
        exit 1
    }
    ln -s "$path" "$bin/$t"
done

PATH="$bin" "$bin/sh" -c '! command -v node && ! command -v pnpm && task build test'
