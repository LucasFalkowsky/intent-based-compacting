"""The corpus access layer: load a fixture by name, never by path."""

from pathlib import Path

import pytest
from fixture_corpus import CORPUS, Corpus

from idc.schema import Session

TESTS_DIR = Path(__file__).parent.resolve()
FIXTURES_DIR = TESTS_DIR / "fixtures"


def test_the_singleton_points_at_the_shared_corpus():
    assert CORPUS.directory == FIXTURES_DIR


def test_the_corpus_pytest_fixture_is_the_singleton(corpus):
    assert corpus is CORPUS


def test_names_are_discovered_from_disk_and_sorted():
    discovered = sorted(path.stem for path in FIXTURES_DIR.glob("*.json"))

    assert CORPUS.names() == discovered
    assert CORPUS.names() == sorted(CORPUS.names())


def test_paths_are_absolute_and_exist():
    assert all(path.is_absolute() and path.is_file() for path in CORPUS.paths())


def test_path_resolves_a_name_to_its_file():
    assert CORPUS.path("single_intent") == FIXTURES_DIR / "single_intent.json"


def test_an_unknown_name_is_reported_with_the_known_ones():
    with pytest.raises(KeyError, match="single_intent"):
        CORPUS.path("no_such_fixture")


def test_session_loads_through_the_validating_schema_api():
    session = CORPUS.session("single_intent")

    assert isinstance(session, Session)
    assert session.id == "sess-single-intent"


def test_text_returns_the_file_verbatim():
    assert CORPUS.text("single_intent") == CORPUS.path("single_intent").read_text(encoding="utf-8")


def test_sessions_loads_the_whole_corpus_keyed_by_name():
    sessions = CORPUS.sessions()

    assert sorted(sessions) == CORPUS.names()
    assert all(isinstance(session, Session) for session in sessions.values())


def test_contains_and_len_describe_the_corpus():
    assert "single_intent" in CORPUS
    assert "no_such_fixture" not in CORPUS
    assert len(CORPUS) == len(CORPUS.names())


def test_a_new_fixture_is_picked_up_without_touching_the_access_layer(tmp_path):
    other = Corpus(tmp_path)
    assert other.names() == []

    (tmp_path / "README.md").write_text("not a fixture\n", encoding="utf-8")
    (tmp_path / "later_ticket_case.json").write_text(CORPUS.text("single_intent"), encoding="utf-8")

    assert other.names() == ["later_ticket_case"]
    assert isinstance(other.session("later_ticket_case"), Session)


def test_discovery_is_not_cached_between_calls(tmp_path):
    other = Corpus(tmp_path)
    other.names()

    (tmp_path / "added_afterwards.json").write_text(CORPUS.text("single_intent"), encoding="utf-8")

    assert other.names() == ["added_afterwards"]
