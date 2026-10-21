"""The corpus access layer: load a fixture by name, never by path."""

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

import fixture_corpus
from fixture_corpus import COMMAND_KEYS, CORPUS, Corpus, command_of, exec_runs
from idc.schema import Agent, Message, Role, Session, ToolCall, ToolCategory, ToolResult

TESTS_DIR = Path(__file__).parent.resolve()
FIXTURES_DIR = TESTS_DIR / "fixtures"


def packages_imported_by(module) -> set[str]:
    """Return the `idc.*` modules a module imports, read off its own source."""
    source = Path(module.__file__).read_text(encoding="utf-8")
    return {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("idc.")
    }


def test_the_access_layer_names_every_package_it_depends_on():
    # The layer hands out two artifacts of the package — the canonical sessions
    # and the gold labellings — so it imports two modules of it. A docstring
    # that understates that is how a reader ends up believing the corpus could
    # be read without the tool.
    doc = fixture_corpus.__doc__ or ""

    undeclared = sorted(name for name in packages_imported_by(fixture_corpus) if name not in doc)

    assert undeclared == [], f"the fixture_corpus docstring does not name: {undeclared}"
    assert "no dependency beyond the standard library" not in doc


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


# --- the tolerant argument lookups -----------------------------------------


def exec_session(args: dict[str, object]) -> Session:
    """Return a minimal session whose single `exec` call carries `args`.

    Built in-memory rather than as a fixture: the corpus states what the three
    agents actually send, while these tests state what the readers tolerate.
    """
    return Session(
        id="sess-exec-spelling",
        agent=Agent.OPENCODE,
        created_at=datetime(2026, 9, 1, 9, 0, tzinfo=UTC),
        messages=[
            Message(
                id="m01",
                role=Role.ASSISTANT,
                timestamp=datetime(2026, 9, 1, 9, 0, tzinfo=UTC),
                tool_calls=[ToolCall(id="c01", name="bash", category=ToolCategory.EXEC, args=args)],
            ),
            Message(
                id="m02",
                role=Role.TOOL,
                timestamp=datetime(2026, 9, 1, 9, 1, tzinfo=UTC),
                tool_result=ToolResult(call_id="c01", output="1 passed", exit_code=0),
            ),
        ],
    )


@pytest.mark.parametrize("key", COMMAND_KEYS)
def test_the_command_lookup_accepts_every_spelling_it_documents(key):
    call = ToolCall(id="c01", name="bash", category=ToolCategory.EXEC, args={key: "pytest -q"})

    assert command_of(call) == "pytest -q"


def test_the_command_lookup_returns_none_for_an_unknown_spelling():
    call = ToolCall(id="c01", name="bash", category=ToolCategory.EXEC, args={"cmdline": "pytest"})

    assert command_of(call) is None


@pytest.mark.parametrize("key", COMMAND_KEYS)
def test_exec_runs_reads_the_command_under_every_documented_spelling(key):
    session = exec_session({key: "pytest -q"})

    assert exec_runs(session) == [("pytest -q", session.messages[-1].tool_result)]


def test_exec_runs_names_the_call_when_no_spelling_matches():
    # The point of the ticket's finding: a future fixture that spells the
    # argument differently must say so, not raise a bare KeyError.
    session = exec_session({"cmdline": "pytest -q"})

    with pytest.raises(ValueError, match="c01") as error:
        exec_runs(session)

    assert "command" in str(error.value)
