"""The commands doing their work, and the seam the fakes reach them through (T10.1, #41).

`idc.cli.Runtime` is the one place a command takes a dependency it cannot
compute from the transcript: the classifier adapter, the narrative client, the
tokenizer registry and the clock. Every test here drives the real command path —
`idc.cli.main` with an argument vector — and injects all four as fakes, so the
suite makes no model call, reads no real clock and counts with no downloaded
vocabulary.

The seam is not decorative. This build ships **no** `ModelClient` and no
`ContinuationScorer`, so `idc.classify.LlmAdapter` and `S1MAdapter` cannot be
constructed here at all, and it ships no tokenizer either (see
`idc.measure.NoTokenizerError`). A `run` with an empty `Runtime` therefore has
to fail with a diagnostic naming what is missing, and the tests below assert
both halves: that the fakes carry a run through, and that their absence is
reported rather than worked around.

`tests/native/claude-code/golden_session.jsonl` is the input throughout: it is
the only hand-written fixture that is a complete native session per agent, and
it holds one user turn, so a script of `NEW` decisions opens exactly one chain.
The scripts are deliberately longer than the run needs — `FakeAdapter` and
`FakeNarrativeClient` refuse to be *over*-drawn, not to be left with entries —
so that a test states what it asserts about and not how many turns the fixture
happens to have.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from idc.classify import NEW, Decision, FakeAdapter
from idc.cli import (
    EXIT_INPUT_ERROR,
    EXIT_OK,
    MANIFEST_SUFFIX,
    Runtime,
    candidate_chains,
    load_artifact,
    main,
)
from idc.compact import DECISIONS_HEADER, REJECTED_HEADER, FakeNarrativeClient
from idc.importers import import_transcript
from idc.measure import (
    FakeClock,
    FakeTokenizer,
    ModelFamily,
    TokenizerRegistry,
    load_manifest,
)
from idc.schema import Agent, Session, load
from native_transcripts import NATIVE

AGENT = Agent.CLAUDE_CODE
"""The agent whose native fixture every run here reads and writes."""

STARTED = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
"""The instant the fake clock hands out at `RunTimer.start`."""

FINISHED = datetime(2026, 3, 1, 9, 0, 30, tzinfo=UTC)
"""The instant it hands out at `RunTimer.finish`."""

SCRIPT_LENGTH = 8
"""How many decisions and narratives the fakes are given; more than any run needs."""

NARRATIVE = (
    "## Intent: Emit UTC timestamps in the invoice exporter\n"
    f"{DECISIONS_HEADER}\n"
    "- The exporter keeps formatting in one helper.\n"
    f"{REJECTED_HEADER}\n"
    "- Converting in the template, dropped.\n"
)
"""A well-formed narrative response, as `idc.compact.as_narrative` accepts one."""


def transcript():
    """Return the native transcript fixture every run here reads."""
    return str(NATIVE.path(AGENT, "golden_session"))


def tokenizers():
    """Return a registry with a `FakeTokenizer` behind every known model family."""
    return TokenizerRegistry({family: FakeTokenizer() for family in ModelFamily})


def runtime(**overrides):
    """Return a `Runtime` whose four seams are the fakes, with overrides applied."""
    seams = {
        "adapter": FakeAdapter([NEW] * SCRIPT_LENGTH),
        "narrative_client": FakeNarrativeClient([NARRATIVE] * SCRIPT_LENGTH),
        "tokenizers": tokenizers(),
        "clock": FakeClock([STARTED, FINISHED]),
    }
    seams.update(overrides)
    return Runtime(**seams)


def run(tmp_path, *extra, name="compacted.jsonl"):
    """Invoke `run` on the fixture, writing into `tmp_path`, and return the status."""
    output = tmp_path / name
    status = main(
        ["run", "--agent", AGENT.value, "--in", transcript(), "--out", str(output), *extra],
        runtime=runtime(),
    )
    return status, output


# --- the end-to-end command through the four fakes -----------------------------


def test_run_writes_the_compacted_transcript(tmp_path):
    status, output = run(tmp_path)

    assert status == EXIT_OK
    assert output.read_text(encoding="utf-8") != ""


def test_the_written_transcript_re_imports_as_this_tools_compaction(tmp_path):
    _, output = run(tmp_path)

    reread = import_transcript(AGENT, output).session
    assert reread.source.value == "intent-compaction"
    assert reread.native_summary_id is not None


def test_run_writes_a_manifest_next_to_its_output_by_default(tmp_path):
    _, output = run(tmp_path)

    manifest = load_manifest(str(output) + MANIFEST_SUFFIX)
    assert manifest.input_file.path == transcript()
    assert manifest.output_file.path == str(output)


def test_the_manifest_goes_where_it_is_asked_to(tmp_path):
    destination = tmp_path / "run.json"

    _, output = run(tmp_path, "--manifest", str(destination))

    assert load_manifest(destination).output_file.path == str(output)


def test_the_manifest_records_the_injected_clocks_two_instants(tmp_path):
    _, output = run(tmp_path)

    manifest = load_manifest(str(output) + MANIFEST_SUFFIX)
    assert manifest.started_at == STARTED
    assert manifest.finished_at == FINISHED


def test_the_clock_is_asked_exactly_twice(tmp_path):
    clock = FakeClock([STARTED, FINISHED])
    output = tmp_path / "compacted.jsonl"

    status = main(
        ["run", "--agent", AGENT.value, "--in", transcript(), "--out", str(output)],
        runtime=runtime(clock=clock),
    )

    assert status == EXIT_OK
    assert clock.remaining == 0


def test_the_manifest_records_the_adapter_and_tokenizer_that_actually_ran(tmp_path):
    _, output = run(tmp_path)

    manifest = load_manifest(str(output) + MANIFEST_SUFFIX)
    assert manifest.config.adapter == "FakeAdapter"
    assert manifest.tokenizer.name == "FakeTokenizer"
    assert manifest.config.options["adapter_requested"] == "llm"
    assert manifest.config.options["source_agent"] == AGENT.value


def test_the_manifest_records_the_measured_numbers(tmp_path):
    _, output = run(tmp_path)

    manifest = load_manifest(str(output) + MANIFEST_SUFFIX)
    assert manifest.metrics.tokens_before > 0
    assert manifest.metrics.tokens_after > 0


def test_a_second_run_chains_to_the_manifest_it_is_given(tmp_path):
    _, first_output = run(tmp_path, name="first.jsonl")
    first = str(first_output) + MANIFEST_SUFFIX

    status = main(
        [
            "run",
            "--agent",
            AGENT.value,
            "--in",
            transcript(),
            "--out",
            str(tmp_path / "second.jsonl"),
            "--previous-manifest",
            first,
        ],
        runtime=runtime(),
    )

    assert status == EXIT_OK
    second = load_manifest(str(tmp_path / "second.jsonl") + MANIFEST_SUFFIX)
    assert second.previous_output_sha256 == load_manifest(first).output_file.sha256


def test_the_chains_of_a_run_are_recoverable_from_its_own_output(tmp_path):
    """The seam a second run needs: the block parses back into the same chains.

    Without it a re-run would offer the classifier no candidate, re-open every
    intent under a new id, and append a second summary next to the first
    instead of extending it.
    """
    _, output = run(tmp_path)

    reread = import_transcript(AGENT, output).session

    recovered = candidate_chains(reread)
    assert [chain.id for chain in recovered] == ["C1"]
    assert all(chain.compacted for chain in recovered)


def test_a_transcript_nothing_compacted_offers_no_candidate_chain():
    assert candidate_chains(import_transcript(AGENT, transcript()).session) == ()


def continued(output, destination):
    """Write `output` with one further user turn appended, as the agent would have.

    A second `run` reads a transcript the user kept working in after the first
    compaction, so the fixture for it is the first run's own output plus a new
    turn. The turn is the golden session's own opening record with another id
    and another question, so nothing here invents a record shape: it is the
    fixture's, read back off disk.

    The new turn matters for more than realism. `idc.render` refuses to write a
    latest chunk that already holds the summary record's identifier
    (`SummaryIdInUseError`), and the first output is a single chunk with the
    summary inside it — so a `run` straight back over it is refused by the
    renderer and never reaches the question this test asks.

    Args:
        output: The compacted transcript the first run wrote.
        destination: Where to write the continued transcript.

    Returns:
        `destination`.
    """
    records = [
        json.loads(line) for line in Path(transcript()).read_text(encoding="utf-8").splitlines()
    ]
    opening = next(record for record in records if record.get("uuid") == "u01")
    turn = {
        **opening,
        "uuid": "u90",
        "promptId": "p90",
        "parentUuid": "a04",
        "message": {"role": "user", "content": "Now document the UTC switch in the README."},
    }
    destination.write_text(
        output.read_text(encoding="utf-8") + json.dumps(turn) + "\n", encoding="utf-8"
    )
    return destination


def test_a_run_over_an_earlier_runs_output_continues_that_runs_chain(tmp_path):
    """The feed itself, pinned on `run`: the earlier chain is offered and reused.

    `test_the_chains_of_a_run_are_recoverable_from_its_own_output` pins
    `candidate_chains`; this pins that `run_command` hands what it returns to
    `idc.classify.classify_chunks`. Both halves are asserted, because the call
    record alone would not say the id survived and the output alone would not
    say where the id came from: the adapter is shown `C1`, and the second output
    recovers one chain still called `C1` instead of a second one beside it.

    The script extends `C1` by id, which the store can only honour while `C1` is
    seeded in it — verified by cutting the feed, which turns the same script
    into an `UnknownChainError` and the run into a diagnostic.
    """
    _, first = run(tmp_path, name="first.jsonl")
    adapter = FakeAdapter([Decision.extend("C1")] * SCRIPT_LENGTH)
    second = tmp_path / "second.jsonl"

    status = main(
        [
            "run",
            "--agent",
            AGENT.value,
            "--in",
            str(continued(first, tmp_path / "continued.jsonl")),
            "--out",
            str(second),
        ],
        runtime=runtime(adapter=adapter),
    )

    assert status == EXIT_OK
    assert adapter.calls[0].candidate_ids == ("C1",)
    recovered = candidate_chains(import_transcript(AGENT, second).session)
    assert [chain.id for chain in recovered] == ["C1"]


def test_classify_offers_an_earlier_runs_chains_to_the_adapter(tmp_path):
    """The same feed on the staged path, which is a second call site.

    `classify_command` reads the chains out of the artifact's session, so a
    `chunk` of a compacted transcript has to carry the summary through for this
    to work at all; the call record states what the adapter saw, and the written
    artifact states that the chain kept its id.
    """
    _, first = run(tmp_path, name="first.jsonl")
    session = tmp_path / "session.json"
    chunked = tmp_path / "chunked.json"
    classified = tmp_path / "classified.json"
    for argv in (
        ["import", "--agent", AGENT.value, "--in", str(first), "--out", str(session)],
        ["chunk", "--in", str(session), "--out", str(chunked)],
    ):
        assert main(argv, runtime=runtime()) == EXIT_OK, argv
    adapter = FakeAdapter([Decision.extend("C1")] * SCRIPT_LENGTH)

    status = main(
        ["classify", "--in", str(chunked), "--out", str(classified)],
        runtime=runtime(adapter=adapter),
    )

    assert status == EXIT_OK
    assert adapter.calls[0].candidate_ids == ("C1",)
    assert [chain.id for chain in load_artifact(classified).chains] == ["C1"]


def test_the_written_transcript_carries_the_narrative_clients_own_wording(tmp_path):
    """The narrative seam, asserted on the file and not on the call record.

    `FakeNarrativeClient` records its calls, but a summary that reached the
    block and not the file would still pass such an assertion.
    """
    _, output = run(tmp_path)

    assert "The exporter keeps formatting in one helper." in output.read_text(encoding="utf-8")


def test_the_run_is_reported_on_stdout(tmp_path, capsys):
    run(tmp_path)

    assert "token" in capsys.readouterr().out


# --- the seam's absence is a diagnostic, not a crash ---------------------------


def test_a_run_without_an_injected_adapter_names_the_adapter_it_cannot_build(tmp_path, capsys):
    status = main(
        [
            "run",
            "--agent",
            AGENT.value,
            "--in",
            transcript(),
            "--out",
            str(tmp_path / "out.jsonl"),
        ],
        runtime=Runtime(tokenizers=tokenizers(), clock=FakeClock([STARTED, FINISHED])),
    )

    assert status == EXIT_INPUT_ERROR
    assert "llm" in capsys.readouterr().err


def test_a_run_without_a_narrative_client_says_so(tmp_path, capsys):
    status = main(
        [
            "run",
            "--agent",
            AGENT.value,
            "--in",
            transcript(),
            "--out",
            str(tmp_path / "out.jsonl"),
        ],
        runtime=runtime(narrative_client=None),
    )

    assert status == EXIT_INPUT_ERROR
    assert "narrative" in capsys.readouterr().err


def test_a_run_without_a_tokenizer_reports_the_measurement_refusal(tmp_path, capsys):
    status = main(
        [
            "run",
            "--agent",
            AGENT.value,
            "--in",
            transcript(),
            "--out",
            str(tmp_path / "out.jsonl"),
        ],
        runtime=runtime(tokenizers=None),
    )

    assert status == EXIT_INPUT_ERROR
    assert "tokenizer" in capsys.readouterr().err


def test_no_output_is_left_behind_when_the_adapter_is_missing(tmp_path):
    output = tmp_path / "out.jsonl"

    main(
        ["run", "--agent", AGENT.value, "--in", transcript(), "--out", str(output)],
        runtime=Runtime(tokenizers=tokenizers(), clock=FakeClock([STARTED, FINISHED])),
    )

    assert not output.exists()


# --- the per-stage commands, and that they compose into the same bytes ---------


def staged(tmp_path):
    """Drive the five file-to-file stages one command at a time; return the output."""
    session = tmp_path / "session.json"
    chunked = tmp_path / "chunked.json"
    classified = tmp_path / "classified.json"
    compacted = tmp_path / "compacted.json"
    output = tmp_path / "staged.jsonl"
    invocations = [
        ["import", "--agent", AGENT.value, "--in", transcript(), "--out", str(session)],
        ["chunk", "--in", str(session), "--out", str(chunked)],
        ["classify", "--in", str(chunked), "--out", str(classified)],
        ["compact", "--in", str(classified), "--out", str(compacted), "--target", AGENT.value],
        ["render", "--in", str(compacted), "--out", str(output)],
    ]
    for argv in invocations:
        assert main(argv, runtime=runtime()) == EXIT_OK, argv
    return session, chunked, classified, compacted, output


def test_import_writes_a_canonical_session_document(tmp_path):
    session, _, _, _, _ = staged(tmp_path)

    assert load(Session, session).agent is AGENT


def test_chunk_writes_the_session_and_its_chunks(tmp_path):
    _, chunked, _, _, _ = staged(tmp_path)

    artifact = load_artifact(chunked)
    assert artifact.chunks != ()
    assert artifact.chains == ()


def test_classify_adds_the_chains_and_keeps_the_chunks(tmp_path):
    _, chunked, classified, _, _ = staged(tmp_path)

    artifact = load_artifact(classified)
    assert [chunk.id for chunk in artifact.chunks] == [
        chunk.id for chunk in load_artifact(chunked).chunks
    ]
    assert artifact.chains != ()


def test_compact_adds_a_block_whose_chains_all_carry_their_summary(tmp_path):
    _, _, _, compacted, _ = staged(tmp_path)

    block = load_artifact(compacted).block
    assert block is not None
    assert block.rendered_for is AGENT
    assert all(chain.compacted and chain.compacted_text for chain in block.chains)


def test_the_staged_pipeline_and_run_write_the_same_bytes(tmp_path):
    _, _, _, _, staged_output = staged(tmp_path)
    _, run_output = run(tmp_path, name="run.jsonl")

    assert run_output.read_bytes() == staged_output.read_bytes()


def test_import_reports_what_it_could_not_represent_on_stderr(tmp_path, capsys):
    status = main(
        [
            "import",
            "--agent",
            AGENT.value,
            "--in",
            transcript(),
            "--out",
            str(tmp_path / "session.json"),
        ],
        runtime=runtime(),
    )

    assert status == EXIT_OK
    assert "skipped-record" in capsys.readouterr().err


def test_resolve_reports_every_chain_on_stdout(tmp_path, capsys):
    _, _, classified, _, _ = staged(tmp_path)
    capsys.readouterr()

    status = main(["resolve", "--in", str(classified)], runtime=runtime())

    assert status == EXIT_OK
    printed = capsys.readouterr().out
    for chain in load_artifact(classified).chains:
        assert chain.id in printed


def test_measure_reports_the_two_counts_on_stdout(tmp_path, capsys):
    _, output = run(tmp_path)
    capsys.readouterr()

    status = main(
        [
            "measure",
            "--agent",
            AGENT.value,
            "--in",
            transcript(),
            "--compacted",
            str(output),
        ],
        runtime=runtime(),
    )

    assert status == EXIT_OK
    assert "token" in capsys.readouterr().out


def test_a_stage_handed_an_artifact_without_what_it_needs_says_which_part_is_missing(
    tmp_path, capsys
):
    session, _, _, _, _ = staged(tmp_path)
    capsys.readouterr()

    status = main(
        ["classify", "--in", str(session), "--out", str(tmp_path / "x.json")],
        runtime=runtime(),
    )

    assert status == EXIT_INPUT_ERROR
    assert "chunks" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("command", "extra"),
    [
        ("chunk", ()),
        ("classify", ()),
        ("compact", ("--target", AGENT.value)),
        ("render", ()),
    ],
)
def test_an_input_file_that_is_not_json_is_an_input_error(command, extra, tmp_path, capsys):
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")

    status = main(
        [command, "--in", str(broken), "--out", str(tmp_path / "out.json"), *extra],
        runtime=runtime(),
    )

    assert status == EXIT_INPUT_ERROR
    assert "broken.json" in capsys.readouterr().err
