"""`python -m idc` is wired up so that `make run ARGS=...` reaches the package.

Until T10.1 (#41) this file pinned the scaffold's placeholder: every invocation
exited with status 2 and a message naming the ticket that would implement the
commands. The commands are implemented now, so the two assertions moved to the
behaviour that replaced it — the bare program is a usage error because a command
is required, and `--help` works both for the program and for one command. That
is a replacement, not a relaxation: the placeholder's claim is gone from the
code, so a test that still asserted it would pin a message no longer printed.

The subprocess is the point of this file: everything else about the command
surface is tested in-process (`tests/test_cli_parsing.py`,
`tests/test_cli_help.py`, `tests/test_cli_run.py`), and only a real
`python -m idc` shows that the module entry point itself resolves and that it
turns the status `main` returns into the process's.
"""

import subprocess
import sys

from idc.cli import COMMAND_NAMES, EXIT_OK, EXIT_USAGE_ERROR


def run_module(*args):
    return subprocess.run(
        [sys.executable, "-m", "idc", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_bare_program_is_a_usage_error_because_a_command_is_required():
    result = run_module()

    assert result.returncode == EXIT_USAGE_ERROR, result.stderr
    assert result.stderr != ""


def test_the_programs_help_lists_every_command():
    result = run_module("--help")

    assert result.returncode == EXIT_OK, result.stderr
    for name in COMMAND_NAMES:
        assert name in result.stdout


def test_a_commands_help_names_the_program_and_the_command():
    result = run_module("compact", "--help")

    assert result.returncode == EXIT_OK, result.stderr
    assert "idc compact" in result.stdout
