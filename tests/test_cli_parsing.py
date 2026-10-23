"""Argument parsing of the eight commands, and how it merges with the config (T10.1, #41).

One command per pipeline stage plus `run`, parsed with the standard library's
`argparse`. Two things are pinned here:

- **What each command accepts.** A flag a command does not have is a usage
  error, not a silently ignored word, and the paths a command needs are
  reported by name when neither the invocation nor the config file supplies
  them.
- **Precedence.** A switch on the command line wins over the config file, and
  the config file wins over the built-in default. `--reduce/--no-reduce` is the
  one tri-state switch, so it is the one that shows the middle case.
- **The two tables the merge runs through.** Every flag below is pinned one by
  one, which says nothing about a *new* field: `configuration` iterates
  `SETTING_KEYS` and `PATH_KEYS`, so a setting or a path the models gain and
  those tuples do not would never be merged, with every one of those per-flag
  tests still green. Both tuples are therefore asserted against the models
  themselves, the way `CLAUDE.md` requires of a list that can drift from its
  source.

`main` is called in-process and returns an exit status; it never raises
`SystemExit`, including for `--help` and for a usage error, which is what makes
these assertions possible at all.
"""

import pytest

from idc.cli import (
    COMMAND_NAMES,
    EXIT_INPUT_ERROR,
    EXIT_OK,
    EXIT_USAGE_ERROR,
    PATH_KEYS,
    SETTING_KEYS,
    Config,
    Paths,
    build_parser,
    configuration,
    main,
)
from idc.schema import Agent

CONFIG = """
agent = "gemini-cli"
target = "opencode"
adapter = "s1m"
reduce = false

[paths]
transcript = "from-config.jsonl"
output = "to-config.jsonl"
"""


def config_file(tmp_path):
    """Write `CONFIG` into `tmp_path` and return its path as a string."""
    path = tmp_path / "idc.toml"
    path.write_text(CONFIG, encoding="utf-8")
    return str(path)


def parse(argv):
    """Parse an invocation and return the configuration it resolves to."""
    return configuration(build_parser().parse_args(argv))


def test_every_pipeline_stage_and_the_end_to_end_run_have_a_command():
    assert COMMAND_NAMES == (
        "import",
        "chunk",
        "classify",
        "resolve",
        "compact",
        "render",
        "measure",
        "run",
    )


def test_the_import_command_takes_the_agent_and_the_two_paths():
    config = parse(["import", "--agent", "opencode", "--in", "a.jsonl", "--out", "b.json"])

    assert config.agent is Agent.OPENCODE
    assert config.paths.transcript == "a.jsonl"
    assert config.paths.output == "b.json"


def test_the_measure_command_takes_a_compacted_transcript_of_its_own():
    config = parse(
        ["measure", "--agent", "claude-code", "--in", "a.jsonl", "--compacted", "b.jsonl"]
    )

    assert config.paths.compacted == "b.jsonl"


def test_the_run_command_takes_the_manifest_paths():
    config = parse(
        [
            "run",
            "--agent",
            "claude-code",
            "--in",
            "a.jsonl",
            "--out",
            "b.jsonl",
            "--manifest",
            "m.json",
            "--previous-manifest",
            "p.json",
        ]
    )

    assert config.paths.manifest == "m.json"
    assert config.paths.previous_manifest == "p.json"


def test_the_compact_command_takes_the_target_agent_the_block_is_built_for():
    config = parse(["compact", "--in", "a.json", "--out", "b.json", "--target", "gemini-cli"])

    assert config.target_agent is Agent.GEMINI_CLI


@pytest.mark.parametrize(
    ("command", "flag"),
    [
        ("chunk", "--agent"),
        ("classify", "--target"),
        ("resolve", "--out"),
        ("compact", "--adapter"),
        ("render", "--agent"),
        ("render", "--target"),
        ("measure", "--out"),
    ],
)
def test_a_flag_a_command_does_not_have_is_a_usage_error(command, flag, capsys):
    assert main([command, flag, "x"]) == EXIT_USAGE_ERROR
    assert flag in capsys.readouterr().err


def test_an_unknown_command_is_a_usage_error(capsys):
    assert main(["summarise"]) == EXIT_USAGE_ERROR
    assert "summarise" in capsys.readouterr().err


def test_no_command_at_all_is_a_usage_error(capsys):
    assert main([]) == EXIT_USAGE_ERROR
    assert capsys.readouterr().err != ""


def test_an_unknown_agent_on_the_command_line_is_a_usage_error(capsys):
    assert main(["import", "--agent", "emacs", "--in", "a", "--out", "b"]) == EXIT_USAGE_ERROR
    assert "emacs" in capsys.readouterr().err


def test_a_missing_input_path_is_a_usage_error_naming_the_option(capsys):
    assert main(["chunk", "--out", "b.json"]) == EXIT_USAGE_ERROR
    assert "--in" in capsys.readouterr().err


def test_a_missing_output_path_is_a_usage_error_naming_the_option(capsys):
    assert main(["chunk", "--in", "a.json"]) == EXIT_USAGE_ERROR
    assert "--out" in capsys.readouterr().err


def test_a_missing_agent_is_a_usage_error_naming_the_option(capsys):
    assert main(["import", "--in", "a.jsonl", "--out", "b.json"]) == EXIT_USAGE_ERROR
    assert "--agent" in capsys.readouterr().err


def test_a_missing_target_agent_is_a_usage_error_naming_the_option(capsys):
    assert main(["compact", "--in", "a.json", "--out", "b.json"]) == EXIT_USAGE_ERROR
    assert "--target" in capsys.readouterr().err


def test_a_missing_compacted_transcript_is_a_usage_error_naming_the_option(capsys):
    assert main(["measure", "--agent", "opencode", "--in", "a.jsonl"]) == EXIT_USAGE_ERROR
    assert "--compacted" in capsys.readouterr().err


def test_the_path_keys_are_exactly_the_fields_of_paths():
    """The `[paths]` merge table against the model it merges into.

    Both halves are asserted on purpose. The equality with `Paths.model_fields`
    is what a sixth path cannot slip past, and it holds by construction today —
    stating it is what makes replacing the derivation with a hand-written list
    a red test rather than a silent return to the drift. The literal spells out
    the five the module documents, so adding a path is a failure here and an
    author who has to look at this line is an author who has to give the new
    path a flag.
    """
    assert tuple(Paths.model_fields) == PATH_KEYS
    assert PATH_KEYS == ("transcript", "output", "compacted", "manifest", "previous_manifest")


def test_the_setting_keys_are_every_config_field_except_the_paths():
    """The same assertion for the settings, and for the same failure.

    `paths` is the one field `configuration` merges under its own name instead
    of through this tuple, so it is the one exclusion; `models` and `options`
    stay in and are skipped at the `getattr`, because neither has a flag.
    """
    assert tuple(name for name in Config.model_fields if name != "paths") == SETTING_KEYS
    assert SETTING_KEYS == ("agent", "target", "adapter", "reduce", "models", "options")
    assert "paths" not in SETTING_KEYS


def test_the_config_file_supplies_the_agent_and_the_paths(tmp_path):
    config = parse(["chunk", "--config", config_file(tmp_path)])

    assert config.agent is Agent.GEMINI_CLI
    assert config.paths.transcript == "from-config.jsonl"
    assert config.paths.output == "to-config.jsonl"


def test_the_command_line_wins_over_the_config_file(tmp_path):
    config = parse(
        [
            "run",
            "--config",
            config_file(tmp_path),
            "--agent",
            "claude-code",
            "--in",
            "from-flag.jsonl",
        ]
    )

    assert config.agent is Agent.CLAUDE_CODE
    assert config.paths.transcript == "from-flag.jsonl"
    assert config.paths.output == "to-config.jsonl"


def test_reduction_is_on_by_default():
    assert parse(["chunk", "--in", "a", "--out", "b"]).reduce is True


def test_the_config_file_can_turn_reduction_off(tmp_path):
    assert parse(["chunk", "--config", config_file(tmp_path)]).reduce is False


def test_the_command_line_can_turn_reduction_back_on(tmp_path):
    config = parse(["chunk", "--config", config_file(tmp_path), "--reduce"])

    assert config.reduce is True


def test_the_command_line_can_turn_reduction_off():
    assert parse(["chunk", "--in", "a", "--out", "b", "--no-reduce"]).reduce is False


def test_a_malformed_config_file_is_an_input_error_and_not_a_usage_error(tmp_path, capsys):
    path = tmp_path / "broken.toml"
    path.write_text("agent = ", encoding="utf-8")

    assert main(["chunk", "--config", str(path)]) == EXIT_INPUT_ERROR
    assert "broken.toml" in capsys.readouterr().err


def test_a_transcript_that_is_not_there_is_an_input_error(tmp_path, capsys):
    absent = tmp_path / "absent.jsonl"

    status = main(["import", "--agent", "claude-code", "--in", str(absent), "--out", "b.json"])

    assert status == EXIT_INPUT_ERROR
    assert "absent.jsonl" in capsys.readouterr().err


def test_the_three_exit_codes_are_distinct():
    assert len({EXIT_OK, EXIT_INPUT_ERROR, EXIT_USAGE_ERROR}) == 3
