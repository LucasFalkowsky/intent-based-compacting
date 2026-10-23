"""`--help` for the bare program and for every command (T10.1, #41).

A smoke test per command, because a help text is the only documentation a user
of the tool gets at the terminal, and a command that was added to `COMMANDS` but
never given a parser fails exactly here. The usage line is checked for the
program name so that `make run ARGS="compact --help"` prints something that can
be typed back.

The parametrization is over `COMMANDS`, which is why the last test asserts that
the table is not empty: an empty one would turn every case here into a silent
skip rather than a failure.
"""

import pytest

from idc.cli import COMMAND_NAMES, COMMANDS, EXIT_OK, PROGRAM, build_parser, main

SUMMARY_MAX_LENGTH = 60
"""How long a command's one-line summary may be, so the command list stays a list."""


def test_the_bare_program_prints_help_and_succeeds(capsys):
    assert main(["--help"]) == EXIT_OK

    printed = capsys.readouterr().out
    assert PROGRAM in printed
    for name in COMMAND_NAMES:
        assert name in printed


@pytest.mark.parametrize("name", COMMAND_NAMES)
def test_every_command_prints_its_own_help_and_succeeds(name, capsys):
    assert main([name, "--help"]) == EXIT_OK

    assert f"{PROGRAM} {name}" in capsys.readouterr().out


@pytest.mark.parametrize("command", COMMANDS, ids=lambda command: command.name)
def test_every_command_carries_a_one_line_summary_for_the_command_list(command):
    assert command.summary != ""
    assert "\n" not in command.summary
    assert len(command.summary) <= SUMMARY_MAX_LENGTH
    assert command.name in build_parser().format_help()


def test_the_command_table_is_not_empty():
    assert COMMANDS != ()
    assert tuple(command.name for command in COMMANDS) == COMMAND_NAMES
