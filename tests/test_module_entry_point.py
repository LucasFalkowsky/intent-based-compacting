"""`python -m idc` is wired up so that `make run ARGS=...` reaches the package."""

import subprocess
import sys


def run_module(*args):
    return subprocess.run(
        [sys.executable, "-m", "idc", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_module_entry_point_is_reachable():
    result = run_module()

    assert result.returncode == 2, result.stderr


def test_module_entry_point_names_the_ticket_that_implements_it():
    result = run_module("compact", "--agent", "opencode")

    assert "T10.1" in result.stderr
