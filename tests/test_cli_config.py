"""The CLI's configuration file and its validation (T10.1, #41).

The config file is TOML, read with the standard library's `tomllib`, and it is
loaded through `idc.cli.load_config` / `loads_config` / `validate_config` — the
same three-function shape `idc.schema` and `idc.measure.manifest` use. What is
pinned here is the line those three draw: a document the user handed in is a
`ConfigError` with a located message, while a `Config` a caller's own code
builds wrongly is a plain `ValueError`. Neither path may let pydantic's
`ValidationError` or `tomllib`'s `TOMLDecodeError` escape, because either would
put a library traceback on the CLI.
"""

import pydantic
import pytest

from idc.cli import Config, ConfigError, Paths, load_config, loads_config, validate_config
from idc.schema import Agent

FULL = """
agent = "claude-code"
target = "opencode"
adapter = "s1m"
reduce = false

[models]
classifier = "claude-opus-5"
narrative = "claude-opus-5"

[options]
threshold = "0.6"

[paths]
transcript = "session.jsonl"
output = "compacted.jsonl"
compacted = "compacted.jsonl"
manifest = "run.json"
previous_manifest = "previous.json"
"""


def test_an_empty_document_is_a_valid_configuration():
    config = loads_config("")

    assert config.agent is None
    assert config.target is None
    assert config.adapter == "llm"
    assert config.reduce is True
    assert config.models == {}
    assert config.options == {}
    assert config.paths == Paths()


def test_every_documented_key_is_read():
    config = loads_config(FULL)

    assert config.agent is Agent.CLAUDE_CODE
    assert config.target is Agent.OPENCODE
    assert config.adapter == "s1m"
    assert config.reduce is False
    assert config.models == {"classifier": "claude-opus-5", "narrative": "claude-opus-5"}
    assert config.options == {"threshold": "0.6"}
    assert config.paths.transcript == "session.jsonl"
    assert config.paths.output == "compacted.jsonl"
    assert config.paths.compacted == "compacted.jsonl"
    assert config.paths.manifest == "run.json"
    assert config.paths.previous_manifest == "previous.json"


def test_the_target_agent_defaults_to_the_source_agent():
    config = loads_config('agent = "gemini-cli"')

    assert config.target_agent is Agent.GEMINI_CLI


def test_an_explicit_target_agent_wins_over_the_source_agent():
    config = loads_config('agent = "gemini-cli"\ntarget = "opencode"')

    assert config.target_agent is Agent.OPENCODE


def test_text_that_is_not_toml_is_a_config_error_and_not_a_tomllib_error():
    with pytest.raises(ConfigError) as caught:
        loads_config("agent = ", source="idc.toml")

    assert "idc.toml" in str(caught.value)


def test_an_unknown_agent_is_a_config_error_naming_the_field():
    with pytest.raises(ConfigError) as caught:
        loads_config('agent = "emacs"')

    assert "agent" in str(caught.value)


def test_an_unknown_adapter_is_a_config_error_naming_the_known_ones():
    with pytest.raises(ConfigError) as caught:
        loads_config('adapter = "telepathy"')

    message = str(caught.value)
    assert "adapter" in message
    assert "s1m" in message


def test_an_unknown_key_is_refused_rather_than_ignored():
    with pytest.raises(ConfigError) as caught:
        loads_config('adaptor = "s1m"')

    assert "adaptor" in str(caught.value)


def test_an_unknown_key_under_paths_is_refused_too():
    with pytest.raises(ConfigError) as caught:
        loads_config('[paths]\nin = "session.jsonl"')

    assert "paths.in" in str(caught.value)


def test_a_model_id_that_is_not_a_string_is_a_config_error():
    with pytest.raises(ConfigError) as caught:
        loads_config("[models]\nclassifier = 5")

    assert "models.classifier" in str(caught.value)


def test_a_missing_file_is_a_config_error_naming_the_file(tmp_path):
    with pytest.raises(ConfigError) as caught:
        load_config(tmp_path / "absent.toml")

    assert "absent.toml" in str(caught.value)


def test_a_file_is_read_as_utf_8(tmp_path):
    path = tmp_path / "idc.toml"
    path.write_text(FULL, encoding="utf-8")

    assert load_config(path).adapter == "s1m"


def test_an_already_parsed_document_validates_through_validate_config():
    config = validate_config({"agent": "opencode"}, source="<test>")

    assert config.agent is Agent.OPENCODE


def test_a_document_that_is_not_a_table_is_a_config_error():
    with pytest.raises(ConfigError):
        validate_config([1, 2, 3], source="<test>")


def test_a_configuration_a_caller_builds_wrongly_raises_a_plain_value_error():
    with pytest.raises(ValueError, match="adapter") as caught:
        Config(adapter="telepathy")

    assert not isinstance(caught.value, ConfigError)
    assert not isinstance(caught.value, pydantic.ValidationError)


def test_a_configuration_cannot_be_mutated_after_construction():
    config = loads_config(FULL)

    with pytest.raises(ValueError, match="adapter"):
        config.adapter = "llm"


def test_the_run_configuration_recorded_in_a_manifest_names_the_target_agent():
    config = loads_config(FULL)

    recorded = config.run_config("FakeAdapter")

    assert recorded.agent is Agent.OPENCODE
    assert recorded.adapter == "FakeAdapter"
    assert recorded.models == {"classifier": "claude-opus-5", "narrative": "claude-opus-5"}
    assert recorded.options["adapter_requested"] == "s1m"
    assert recorded.options["source_agent"] == "claude-code"
    assert recorded.options["reduce"] == "false"
    assert recorded.options["threshold"] == "0.6"


def test_the_run_configuration_needs_a_source_agent():
    with pytest.raises(ValueError, match="agent"):
        loads_config("").run_config("FakeAdapter")
