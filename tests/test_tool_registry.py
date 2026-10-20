"""The tool category registry: per-agent tool knowledge, read from a data file."""

import re
from pathlib import Path

import pytest

from fixture_corpus import CORPUS, tool_calls
from idc.importers import (
    REGISTRY_PATH,
    ToolRegistry,
    ToolRegistryError,
    load_tool_registry,
    tool_registry,
)
from idc.schema import Agent, ToolCategory

MINIMAL_REGISTRY = """\
version = 1
runner_prefixes = ["uv run"]

[agents.claude-code.tools]
Read = { category = "read" }
Task = { category = "other", subagent = true }

[agents.gemini-cli.tools]
read_file = { category = "read" }

[agents.opencode.tools]
read = { category = "read" }

[[verification_patterns]]
id = "pytest"
command = "pytest"
"""


def write_registry(tmp_path: Path, text: str = MINIMAL_REGISTRY) -> Path:
    path = tmp_path / "tool_registry.toml"
    path.write_text(text, encoding="utf-8")
    return path


# --- the packaged data file -------------------------------------------------


def test_the_registry_lives_in_a_data_file_next_to_the_module():
    assert REGISTRY_PATH.is_file()
    assert REGISTRY_PATH.suffix == ".toml"


def test_the_default_registry_is_the_packaged_data_file():
    assert tool_registry().source == REGISTRY_PATH


def test_the_default_registry_is_read_once():
    assert tool_registry() is tool_registry()


def test_the_data_file_declares_every_agent_of_the_schema():
    assert set(tool_registry().agents) == set(Agent)


def test_every_agent_declares_at_least_one_tool():
    registry = tool_registry()

    for agent in Agent:
        assert registry.tool_names(agent), f"{agent} declares no tool"


# --- resolution -------------------------------------------------------------


def test_every_tool_call_in_the_corpus_resolves_to_its_recorded_category():
    registry = tool_registry()
    unresolved = [
        (session.agent, call.name, call.category, registry.category(session.agent, call.name))
        for session in CORPUS.sessions().values()
        for call in tool_calls(session.messages)
        if registry.category(session.agent, call.name) is not call.category
    ]

    assert unresolved == []


def test_every_tool_call_in_the_corpus_is_a_known_tool():
    registry = tool_registry()
    unknown = {
        (session.agent, call.name)
        for session in CORPUS.sessions().values()
        for call in tool_calls(session.messages)
        if not registry.knows(session.agent, call.name)
    }

    assert unknown == set()


def test_every_declared_tool_resolves_to_a_category_and_is_known():
    registry = tool_registry()

    for agent in registry.agents:
        for name in registry.tool_names(agent):
            assert isinstance(registry.category(agent, name), ToolCategory)
            assert registry.knows(agent, name)


def test_an_unknown_tool_resolves_to_other():
    registry = tool_registry()

    assert registry.category(Agent.CLAUDE_CODE, "Teleport") is ToolCategory.OTHER
    assert registry.knows(Agent.CLAUDE_CODE, "Teleport") is False
    assert registry.spec(Agent.CLAUDE_CODE, "Teleport") is None


def test_tool_names_resolve_per_agent_and_case_sensitively():
    registry = tool_registry()

    assert registry.category(Agent.OPENCODE, "read") is ToolCategory.READ
    assert registry.category(Agent.OPENCODE, "Read") is ToolCategory.OTHER
    assert registry.category(Agent.CLAUDE_CODE, "Read") is ToolCategory.READ
    assert registry.category(Agent.CLAUDE_CODE, "read") is ToolCategory.OTHER
    assert registry.category(Agent.GEMINI_CLI, "Bash") is ToolCategory.OTHER


def test_an_agent_may_be_named_by_its_wire_string():
    assert tool_registry().category("claude-code", "Read") is ToolCategory.READ


def test_an_unknown_agent_is_rejected_and_the_known_ones_are_listed():
    with pytest.raises(ToolRegistryError, match="unknown agent 'cursor'") as error:
        tool_registry().category("cursor", "Read")

    message = str(error.value)
    assert all(agent.value in message for agent in Agent)


# --- the subagent marker (consumed by T3.3) ---------------------------------


def test_the_claude_code_subagent_tool_is_marked_as_such():
    registry = tool_registry()

    assert registry.is_subagent_tool(Agent.CLAUDE_CODE, "Task")
    assert registry.subagent_tools(Agent.CLAUDE_CODE) == ("Task",)


def test_an_ordinary_tool_is_not_a_subagent_tool():
    registry = tool_registry()

    assert registry.is_subagent_tool(Agent.CLAUDE_CODE, "Bash") is False
    assert registry.is_subagent_tool(Agent.CLAUDE_CODE, "Teleport") is False


def test_the_subagent_marker_keeps_the_category_the_corpus_records():
    assert tool_registry().category(Agent.CLAUDE_CODE, "Task") is ToolCategory.OTHER


# --- the data file is data: no code change to extend it ---------------------


def test_a_tool_added_to_the_data_file_resolves_without_a_code_change(tmp_path):
    text = MINIMAL_REGISTRY.replace(
        'Read = { category = "read" }\n',
        'Read = { category = "read" }\nTeleport = { category = "exec" }\n',
        1,
    )

    registry = load_tool_registry(write_registry(tmp_path, text))

    assert registry.category(Agent.CLAUDE_CODE, "Teleport") is ToolCategory.EXEC
    assert tool_registry().category(Agent.CLAUDE_CODE, "Teleport") is ToolCategory.OTHER


def test_a_verification_pattern_added_to_the_data_file_matches_without_a_code_change(tmp_path):
    text = MINIMAL_REGISTRY + '\n[[verification_patterns]]\nid = "ward"\ncommand = "ward test"\n'

    registry = load_tool_registry(write_registry(tmp_path, text))

    assert registry.is_verification("ward test --path tests")
    assert tool_registry().is_verification("ward test --path tests") is False


def test_a_data_file_without_a_pattern_matches_no_command(tmp_path):
    text = MINIMAL_REGISTRY.replace('command = "pytest"', 'command = "make test"')

    registry = load_tool_registry(write_registry(tmp_path, text))

    assert registry.is_verification("make test") is True
    assert registry.is_verification("pytest -q") is False


# --- a malformed data file fails loudly -------------------------------------


def test_a_missing_data_file_is_reported_with_its_path(tmp_path):
    missing = tmp_path / "absent.toml"

    with pytest.raises(ToolRegistryError, match=str(missing)):
        load_tool_registry(missing)


def test_a_file_that_is_not_toml_is_reported_as_such(tmp_path):
    path = write_registry(tmp_path, "version = 1\nthis is not toml\n")

    with pytest.raises(ToolRegistryError, match="not valid TOML") as error:
        load_tool_registry(path)

    assert str(path) in str(error.value)


def test_an_unknown_agent_in_the_data_file_is_rejected(tmp_path):
    path = write_registry(
        tmp_path, MINIMAL_REGISTRY + '\n[agents.cursor.tools]\nread = { category = "read" }\n'
    )

    with pytest.raises(ToolRegistryError, match=re.escape("agents.cursor")):
        load_tool_registry(path)


def test_a_missing_agent_in_the_data_file_is_rejected(tmp_path):
    text = MINIMAL_REGISTRY.replace('[agents.opencode.tools]\nread = { category = "read" }\n', "")
    path = write_registry(tmp_path, text)

    with pytest.raises(ToolRegistryError, match="opencode"):
        load_tool_registry(path)


def test_an_unknown_category_in_the_data_file_is_rejected(tmp_path):
    text = MINIMAL_REGISTRY.replace('Read = { category = "read" }', 'Read = { category = "reads" }')
    path = write_registry(tmp_path, text)

    with pytest.raises(
        ToolRegistryError, match=re.escape("agents.claude-code.tools.Read.category")
    ):
        load_tool_registry(path)


def test_an_unknown_field_in_the_data_file_is_rejected(tmp_path):
    text = MINIMAL_REGISTRY.replace(
        'Read = { category = "read" }', 'Read = { category = "read", flavour = "fast" }'
    )
    path = write_registry(tmp_path, text)

    with pytest.raises(ToolRegistryError, match="flavour"):
        load_tool_registry(path)


def test_an_unsupported_version_is_rejected(tmp_path):
    path = write_registry(tmp_path, MINIMAL_REGISTRY.replace("version = 1", "version = 2"))

    with pytest.raises(ToolRegistryError, match="version"):
        load_tool_registry(path)


def test_an_agent_without_tools_is_rejected(tmp_path):
    text = MINIMAL_REGISTRY.replace(
        'read = { category = "read" }\n\n[[verification', "\n[[verification"
    )
    path = write_registry(tmp_path, text)

    with pytest.raises(ToolRegistryError, match=re.escape("agents.opencode.tools")):
        load_tool_registry(path)


def test_a_data_file_without_verification_patterns_is_rejected(tmp_path):
    text = MINIMAL_REGISTRY.split("[[verification_patterns]]")[0]
    path = write_registry(tmp_path, text)

    with pytest.raises(ToolRegistryError, match="verification_patterns"):
        load_tool_registry(path)


def test_duplicate_verification_pattern_ids_are_rejected(tmp_path):
    text = MINIMAL_REGISTRY + '\n[[verification_patterns]]\nid = "pytest"\ncommand = "make test"\n'
    path = write_registry(tmp_path, text)

    with pytest.raises(ToolRegistryError, match="pytest"):
        load_tool_registry(path)


def test_two_patterns_matching_the_same_command_are_rejected(tmp_path):
    text = (
        MINIMAL_REGISTRY
        + '\n[[verification_patterns]]\nid = "pytest-uv"\ncommand = "uv run pytest"\n'
    )
    path = write_registry(tmp_path, text)

    with pytest.raises(ToolRegistryError, match="pytest-uv"):
        load_tool_registry(path)


def test_a_runner_prefix_without_a_command_word_is_rejected(tmp_path):
    # The one malformed entry that would not mis-match but hang: an empty prefix
    # is stripped from every command line, and stripping never terminates.
    text = MINIMAL_REGISTRY.replace(
        'runner_prefixes = ["uv run"]', 'runner_prefixes = ["uv run", ""]'
    )
    path = write_registry(tmp_path, text)

    with pytest.raises(ToolRegistryError, match="runner prefix"):
        load_tool_registry(path)


def test_a_pattern_that_is_only_a_runner_prefix_is_rejected(tmp_path):
    text = MINIMAL_REGISTRY.replace('command = "pytest"', 'command = "uv run"')
    path = write_registry(tmp_path, text)

    with pytest.raises(ToolRegistryError, match="uv run"):
        load_tool_registry(path)


def test_a_registry_can_be_built_from_a_parsed_document_without_a_file():
    registry = ToolRegistry(
        {
            "version": 1,
            "agents": {agent.value: {"tools": {"read": {"category": "read"}}} for agent in Agent},
            "verification_patterns": [{"id": "pytest", "command": "pytest"}],
        }
    )

    assert registry.source is None
    assert registry.category(Agent.OPENCODE, "read") is ToolCategory.READ
    assert registry.is_verification("pytest -q")


def test_a_document_that_is_not_a_mapping_is_rejected():
    with pytest.raises(ToolRegistryError, match="mapping"):
        ToolRegistry([("version", 1)])  # type: ignore[arg-type]
