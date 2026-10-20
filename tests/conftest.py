"""Shared pytest configuration and fixtures for the `idc` test suite."""

from pathlib import Path

import pytest

from fixture_corpus import CORPUS, FIXTURES_DIR, TESTS_DIR, Corpus
from native_transcripts import NATIVE, NATIVE_DIR, NativeTranscripts

__all__ = ["FIXTURES_DIR", "NATIVE_DIR", "TESTS_DIR"]


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return the directory holding the shared fixture corpus.

    Every test that needs a transcript on disk resolves it through this fixture
    instead of hard-coding a relative path, so the corpus stays discoverable no
    matter which working directory pytest is started from.
    """
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    """Return the shared fixture corpus, addressed by fixture name.

    The preferred way into the corpus: `corpus.session("interleaved_intents")`
    loads and validates one transcript, `corpus.names()` discovers all of them.
    See `tests/fixture_corpus.py` and `tests/fixtures/README.md`.
    """
    return CORPUS


@pytest.fixture(scope="session")
def native() -> NativeTranscripts:
    """Return the agent-native transcript fixtures, addressed by agent and name.

    The input side of the importers: `native.path("claude-code", "golden_session")`
    resolves one transcript in an agent's own on-disk format. See
    `tests/native_transcripts.py` and `tests/native/README.md`.
    """
    return NATIVE
