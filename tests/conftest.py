"""Shared pytest configuration and fixtures for the `idc` test suite."""

from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent.resolve()
"""Absolute path of the `tests/` directory."""

FIXTURES_DIR = TESTS_DIR / "fixtures"
"""Absolute path of the shared transcript fixture corpus (populated in T1.3)."""


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return the directory holding the shared fixture corpus.

    Every test that needs a transcript on disk resolves it through this fixture
    instead of hard-coding a relative path, so the corpus stays discoverable no
    matter which working directory pytest is started from.
    """
    return FIXTURES_DIR
