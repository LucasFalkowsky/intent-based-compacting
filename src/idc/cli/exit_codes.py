"""The statuses the command surface exits with — a contract, not an accident.

Three codes and no more, because a caller of this tool (a shell script, a study
harness, another agent) has exactly three questions to ask about a run:

- `EXIT_OK` — the command did its work.
- `EXIT_INPUT_ERROR` — the command was invoked correctly and something it was
  *handed* could not be used: a transcript that is not there, a malformed
  config file, a classifier adapter this build cannot construct. The reason is
  one line on stderr and nothing is on stdout.
- `EXIT_USAGE_ERROR` — the *invocation* is wrong: an unknown command, a flag a
  command does not have, a path neither the command line nor the config file
  supplied. `2` is `argparse`'s own value for this, and it is kept rather than
  renumbered so that a usage error the parser rejects and one the command
  surface rejects report the same status.

`EXIT_USAGE_ERROR`'s value is also the one the T1 scaffold exited with
(`EXIT_NOT_WIRED_UP = 2`, now gone): a scaffold that refused every invocation
was, in effect, reporting that the invocation could not be served.

`idc.classify.EXIT_INPUT_ERROR` is `2` and deliberately **not** reused here.
That constant belongs to `python -m idc.classify`, a development utility with no
subcommands, where both kinds of mistake report `2` — `argparse`'s own value for
the ones it catches, that constant for the ones the harness catches — so a caller
cannot tell them apart from the status alone. This surface numbers them apart.
"""

__all__ = ["EXIT_INPUT_ERROR", "EXIT_OK", "EXIT_USAGE_ERROR"]

EXIT_OK = 0
"""The command did what it was asked to do."""

EXIT_INPUT_ERROR = 1
"""The invocation was valid and something it was handed could not be used."""

EXIT_USAGE_ERROR = 2
"""The invocation itself is wrong; `argparse`'s own value for this."""
