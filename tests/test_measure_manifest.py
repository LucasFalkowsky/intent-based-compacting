"""The run manifest: what a run is identified by, and how it is read back (T9.3, #39).

The two acceptance criteria of #39 are the two headline tests here:

- *Manifest validates against its schema* — the manifest's schema is its
  pydantic model, and it is reached only through `loads_manifest` /
  `load_manifest`, which raise this package's own `ManifestError`. A pydantic
  `ValidationError` never escapes, which is asserted and not assumed.
- *Hashes change when input changes* — the same run recorded over two different
  input files differs in `input_file.sha256` and nowhere else that identifies
  the input.

Everything else the manifest records is identity: the instruction digest and
version from `idc.compact`, the tokenizer the numbers were produced with, the
run's configuration, the metrics, the two injected instants, and the predecessor
run's output hash that lets two runs be chained.

The digest tests use `hashlib` as the oracle for the general property and one
published constant — sha256 of the empty byte string — so the test does not
merely repeat the implementation's own call.

The metrics record is pinned the same way. The two one-message sessions below
make no tool call, so over them seven of the record's eleven fields hold
exactly what a manifest that never read them would hold — three of the seven
their field default, the four category counts a plain `0` — and any comparison
against `MetricsRecord.of` would hold for any implementation. Those seven are
therefore pinned over two corpus transcripts that do make tool calls, against
the numbers in `TOOL_FIGURES`, written out by hand here, and against
`tool_statistics` of the measured session as a second oracle.
"""

import hashlib
import json
from datetime import UTC, datetime
from typing import NamedTuple

import pytest
from pydantic import ValidationError

from fixture_corpus import CORPUS
from idc.compact import INSTRUCTION_PATH, INSTRUCTION_VERSION, instruction_digest
from idc.measure import (
    MANIFEST_VERSION,
    FakeTokenizer,
    FileDigest,
    InstructionIdentity,
    ManifestError,
    MeasureError,
    MetricsRecord,
    ModelFamily,
    RunConfig,
    RunManifest,
    RunWindow,
    TokenizerIdentity,
    build_manifest,
    digest_bytes,
    digest_file,
    dump_manifest,
    dumps_manifest,
    load_manifest,
    loads_manifest,
    measure,
    tool_statistics,
)
from idc.schema import Agent, Message, Role, Session, SessionSource

CREATED_AT = datetime(2026, 10, 21, 9, 0, 0, tzinfo=UTC)
"""Fixed creation instant of the sessions measured here."""

WINDOW = RunWindow(
    started_at=datetime(2026, 10, 21, 9, 0, 0, tzinfo=UTC),
    finished_at=datetime(2026, 10, 21, 9, 2, 30, tzinfo=UTC),
)
"""The injected run window; no test here reads a clock."""

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
"""The published sha256 of the empty byte string."""

TOKENIZER_NAME = "FakeTokenizer"
"""What the suite counts with; recorded so a number is attributable."""

TOP_LEVEL_KEYS = {
    "manifest_version",
    "started_at",
    "finished_at",
    "input_file",
    "output_file",
    "config",
    "instruction",
    "tokenizer",
    "metrics",
    "previous_output_sha256",
}
"""Every key the document holds — the manifest's contract with a reader."""


class ToolFigures(NamedTuple):
    """The hand-written tool figures of one corpus transcript.

    Written out here rather than taken from `MetricsRecord.of` or from
    `tool_statistics`: a record asserted only against the code that built it
    holds for any implementation of that code.
    """

    session_id: str
    read_calls: int
    write_calls: int
    exec_calls: int
    other_calls: int
    exploration_rate: float
    files_edited: tuple[str, ...]
    lines_added: int


TOOL_FIGURES = {
    # Eight tool calls: one read, four writes, three execs, nothing else. The
    # added lines are 16 for the test file written whole plus 18 for slug.py
    # over its three writes; `tests/test_measure_statistics.py` arrives at the
    # same 34 from the same transcript, by its own hand count.
    "write_then_two_edits": ToolFigures(
        session_id="sess-write-then-two-edits",
        read_calls=1,
        write_calls=4,
        exec_calls=3,
        other_calls=0,
        exploration_rate=1 / 8,
        files_edited=("/repo/tests/text/test_slug.py", "/repo/src/text/slug.py"),
        lines_added=34,
    ),
    # Seven calls, and the only corpus transcript holding a call of no
    # category — the subagent delegation. It is here for exactly that: with
    # `write_then_two_edits` alone, `other_calls` is 0 in every row and
    # zeroing the field would leave this test green.
    "subagent_result": ToolFigures(
        session_id="sess-subagent-result",
        read_calls=2,
        write_calls=3,
        exec_calls=1,
        other_calls=1,
        exploration_rate=2 / 7,
        files_edited=(
            "src/billing/money.py",
            "src/billing/invoice.py",
            "src/report/totals.py",
        ),
        lines_added=3,
    ),
}
"""Hand-written tool figures per corpus transcript; see the module docstring."""

BLIND_VALUES = {
    "read_calls": 0,
    "write_calls": 0,
    "exec_calls": 0,
    "other_calls": 0,
    "files_edited": (),
    "lines_added": 0,
}
"""What each recorded tool field would hold if the manifest never read it.

**Only three of the six are field defaults**, and the earlier name for this
table claimed all six were. `files_edited`, `lines_added` and
`exploration_rate` have defaults; `read_calls`, `write_calls`, `exec_calls`
and `other_calls` are *required*, so dropping the keyword raises rather than
falling back and the mutation they guard against is a hardcoded `0`, not a
default. Either way the value here is what a row must differ from for that row
to pin the field at all — which is the one property the guard needs, and
`test_every_recorded_tool_field_differs_from_its_blind_value` checks the three
real defaults against `MetricsRecord.model_fields` so a changed default cannot
leave a stale value in this table.

`session_id` is not in here because no value of it is blind: it is required and
a hardcoded one is caught by comparing it to the measured session. Nor is
`exploration_rate`, which is `None` only for a session with no tool call, and
neither transcript in `TOOL_FIGURES` is one.
"""


def session(session_id: str, content: str) -> Session:
    """Return a one-message raw session holding `content`."""
    return Session(
        id=session_id,
        agent=Agent.CLAUDE_CODE,
        model="claude-opus-5",
        source=SessionSource.RAW,
        created_at=CREATED_AT,
        messages=[Message(id="m-1", role=Role.USER, content=content, timestamp=CREATED_AT)],
    )


def run_config() -> RunConfig:
    """Return a configuration naming the adapter and the model ids of a run."""
    return RunConfig(
        agent=Agent.CLAUDE_CODE,
        adapter="fake",
        models={"narrative": "claude-opus-5", "classifier": "claude-sonnet-4-5"},
        options={"max_chunks": "12"},
    )


def metrics():
    """Return the metrics of a run from five tokens to three.

    `please fix the slug helper` is five words and `Summary block here` three,
    counted by `FakeTokenizer`; the manifest has to record those two numbers
    and the fraction between them, not recompute anything of its own.
    """
    return measure(
        session("s-before", "please fix the slug helper"),
        session("s-after", "Summary block here"),
        FakeTokenizer(),
    )


def manifest(
    tmp_path, *, input_bytes: bytes = b"before\n", previous=None, measured=None
) -> RunManifest:
    """Build a manifest over two files written into `tmp_path`.

    Args:
        tmp_path: Directory the two files are written into.
        input_bytes: The bytes of the input file, so a test can vary them.
        previous: The predecessor manifest, if the run continues one.
        measured: The metrics to record; `metrics()` when a test does not care.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "session.jsonl"
    source.write_bytes(input_bytes)
    target = tmp_path / "compacted.jsonl"
    target.write_bytes(b"after\n")
    return build_manifest(
        window=WINDOW,
        input_path=source,
        output_path=target,
        config=run_config(),
        tokenizer_name=TOKENIZER_NAME,
        metrics=metrics() if measured is None else measured,
        previous=previous,
    )


def test_digest_bytes_is_the_sha256_of_the_bytes():
    assert digest_bytes(b"") == EMPTY_SHA256
    assert digest_bytes(b"idc") == hashlib.sha256(b"idc").hexdigest()


def test_digest_file_records_the_path_the_digest_and_the_size(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_bytes(b"idc")

    digested = digest_file(path)

    assert digested == FileDigest(
        path=str(path), sha256=hashlib.sha256(b"idc").hexdigest(), size_bytes=3
    )


def test_the_manifest_digests_a_file_exactly_as_the_instruction_does():
    assert digest_file(INSTRUCTION_PATH).sha256 == instruction_digest()


def test_a_file_that_cannot_be_read_is_a_measure_error(tmp_path):
    with pytest.raises(ManifestError, match=r"missing\.jsonl") as raised:
        digest_file(tmp_path / "missing.jsonl")

    assert isinstance(raised.value, MeasureError)


def test_the_manifest_records_the_instruction_identity(tmp_path):
    recorded = manifest(tmp_path)

    assert recorded.instruction == InstructionIdentity(
        version=INSTRUCTION_VERSION, digest=instruction_digest()
    )


def test_the_manifest_records_the_tokenizer_identity(tmp_path):
    recorded = manifest(tmp_path)

    assert recorded.tokenizer == TokenizerIdentity(name=TOKENIZER_NAME, family=ModelFamily.CLAUDE)


def test_the_manifest_records_the_configuration_and_the_metrics(tmp_path):
    recorded = manifest(tmp_path)

    assert recorded.config == run_config()
    # This pins the pass-through only: both sides are `MetricsRecord.of` of the
    # same metrics. What each field holds is pinned below, over a transcript
    # that makes tool calls.
    assert recorded.metrics == MetricsRecord.of(metrics())
    assert (recorded.metrics.tokens_before, recorded.metrics.tokens_after) == (5, 3)
    assert recorded.metrics.saved_fraction == pytest.approx(0.4)
    assert recorded.metrics.exploration_rate is None


@pytest.mark.parametrize("fixture", sorted(TOOL_FIGURES))
def test_the_manifest_records_the_tool_figures_of_the_measured_session(fixture, tmp_path):
    before = CORPUS.session(fixture)
    expected = TOOL_FIGURES[fixture]
    measured = measure(before, session("s-after", "Summary block here"), FakeTokenizer())

    recorded = manifest(tmp_path, measured=measured).metrics

    assert recorded.session_id == expected.session_id
    assert (
        recorded.read_calls,
        recorded.write_calls,
        recorded.exec_calls,
        recorded.other_calls,
    ) == (
        expected.read_calls,
        expected.write_calls,
        expected.exec_calls,
        expected.other_calls,
    )
    assert recorded.exploration_rate == pytest.approx(expected.exploration_rate)
    assert recorded.files_edited == expected.files_edited
    assert recorded.lines_added == expected.lines_added


@pytest.mark.parametrize("fixture", sorted(TOOL_FIGURES))
def test_the_recorded_tool_figures_are_the_measured_session_s(fixture, tmp_path):
    """The record takes every tool figure from the session, computing none itself."""
    before = CORPUS.session(fixture)
    stats = tool_statistics(before)
    measured = measure(before, session("s-after", "Summary block here"), FakeTokenizer())

    recorded = manifest(tmp_path, measured=measured).metrics

    assert recorded.session_id == stats.session_id
    assert (
        recorded.read_calls,
        recorded.write_calls,
        recorded.exec_calls,
        recorded.other_calls,
    ) == (stats.read_calls, stats.write_calls, stats.exec_calls, stats.other_calls)
    assert recorded.exploration_rate == pytest.approx(stats.exploration_rate)
    assert recorded.files_edited == stats.files_edited
    assert recorded.lines_added == stats.lines_added


def test_every_recorded_tool_field_differs_from_its_blind_value():
    """A field holding the same value in every row is a field no row can pin.

    This is what reopened the hole once: `MetricsRecord.of` could drop
    `files_edited` and `lines_added`, hardcode `session_id` or zero all four
    category counts with the whole suite still green, because every session it
    was measured over had no tool call. The guard fails if `TOOL_FIGURES` ever
    drifts back to such a table.
    """
    assert TOOL_FIGURES, "an empty table would pin nothing at all"
    for field, blind in BLIND_VALUES.items():
        values = {getattr(figures, field) for figures in TOOL_FIGURES.values()}
        assert values - {blind}, f"{field} is {blind!r} in every row of TOOL_FIGURES"

    declared = MetricsRecord.model_fields
    for field, blind in BLIND_VALUES.items():
        if declared[field].is_required():
            continue
        assert declared[field].default == blind, (
            f"{field}'s field default moved to {declared[field].default!r}; "
            f"BLIND_VALUES still says {blind!r}"
        )


def test_the_manifest_records_the_injected_window(tmp_path):
    recorded = manifest(tmp_path)

    assert recorded.started_at == WINDOW.started_at
    assert recorded.finished_at == WINDOW.finished_at
    assert recorded.manifest_version == MANIFEST_VERSION


def test_hashes_change_when_the_input_changes(tmp_path):
    first = manifest(tmp_path, input_bytes=b"before\n")
    second = manifest(tmp_path, input_bytes=b"before, with one more line\n")

    assert first.input_file.sha256 != second.input_file.sha256
    assert first.input_file.size_bytes != second.input_file.size_bytes
    assert first.output_file == second.output_file


def test_the_document_round_trips_through_the_package_api(tmp_path):
    recorded = manifest(tmp_path)

    assert loads_manifest(dumps_manifest(recorded)) == recorded


def test_the_document_round_trips_through_a_file(tmp_path):
    recorded = manifest(tmp_path)
    path = dump_manifest(recorded, tmp_path / "manifest.json")

    assert load_manifest(path) == recorded
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_the_document_holds_exactly_the_documented_keys(tmp_path):
    document = json.loads(dumps_manifest(manifest(tmp_path)))

    assert set(document) == TOP_LEVEL_KEYS
    assert document["previous_output_sha256"] is None


def test_the_dump_is_byte_stable_and_sorts_the_configuration(tmp_path):
    recorded = manifest(tmp_path)

    first = dumps_manifest(recorded)

    assert first == dumps_manifest(loads_manifest(first))
    assert list(json.loads(first)["config"]["models"]) == ["classifier", "narrative"]


def test_the_configuration_mappings_of_a_built_manifest_refuse_a_mutation(tmp_path):
    """`frozen=True` does not freeze a dict a `Mapping` field holds; the type does.

    Sorting the mapping at validation was not enough on its own: with a plain
    dict behind the field, `recorded.config.models["aaa"] = "z"` succeeded on a
    built manifest and the next dump emitted
    `["classifier", "narrative", "aaa"]` — a mutated "frozen" manifest with an
    unsorted configuration, which is the byte stability `dumps_manifest`
    promises gone. Both were run before this test was written.
    """
    recorded = manifest(tmp_path)

    for mapping in (recorded.config.models, recorded.config.options):
        with pytest.raises(TypeError):
            mapping["aaa"] = "z"

    document = json.loads(dumps_manifest(recorded))
    assert list(document["config"]["models"]) == ["classifier", "narrative"]
    assert list(document["config"]["options"]) == ["max_chunks"]


def test_the_configuration_mappings_are_read_only_when_they_are_omitted():
    """A configuration that configured nothing is as read-only as one that did.

    Pydantic does not validate a default, so `default_factory=dict` alone left
    an **omitted** mapping a writable dict while a given one was a
    `mappingproxy` — read-only on every manifest except the ones with no model
    id and no switch recorded. `validate_default=True` is what closes that, and
    without this test the whole suite passed with the keyword removed.

    The call shape is the one the package's own docstring example uses, which
    is how the gap stayed invisible: both mappings omitted.
    """
    config = RunConfig(agent=Agent.CLAUDE_CODE, adapter="s1m")

    for mapping in (config.models, config.options):
        with pytest.raises(TypeError):
            mapping["aaa"] = "z"


def test_the_configuration_sorts_a_mapping_it_is_given_unsorted(tmp_path):
    recorded = build_manifest(
        window=WINDOW,
        input_path=digest_file(INSTRUCTION_PATH).path,
        output_path=digest_file(INSTRUCTION_PATH).path,
        config=RunConfig(agent=Agent.CLAUDE_CODE, adapter="fake", options={"b": "2", "a": "1"}),
        tokenizer_name=TOKENIZER_NAME,
        metrics=metrics(),
    )

    assert list(recorded.config.options) == ["a", "b"]
    assert list(json.loads(dumps_manifest(recorded))["config"]["options"]) == ["a", "b"]


def test_a_first_run_records_no_predecessor(tmp_path):
    recorded = manifest(tmp_path)

    assert recorded.previous_output_sha256 is None


def test_a_second_run_chains_to_its_predecessor_s_output(tmp_path):
    first = manifest(tmp_path / "run-1")
    second = build_manifest(
        window=WINDOW,
        input_path=first.output_file.path,
        output_path=first.output_file.path,
        config=run_config(),
        tokenizer_name=TOKENIZER_NAME,
        metrics=metrics(),
        previous=first,
    )

    assert second.previous_output_sha256 == first.output_file.sha256
    assert loads_manifest(dumps_manifest(second)) == second


def test_a_broken_chain_is_recorded_rather_than_hidden(tmp_path):
    first = manifest(tmp_path / "run-1")
    second = manifest(tmp_path / "run-2", input_bytes=b"edited by hand\n", previous=first)

    assert second.previous_output_sha256 == first.output_file.sha256
    assert second.input_file.sha256 != first.output_file.sha256


MALFORMED = {
    "not json": "{",
    "not an object": "[]",
    "missing field": '{"manifest_version": 1}',
    "unknown field": None,
    "future version": None,
    "short digest": None,
    "naive instant": None,
    "finish before start": None,
    "negative size": None,
}
"""The documents `loads_manifest` has to refuse; filled in by `malformed`."""


def malformed(name: str, tmp_path) -> str:
    """Return the malformed document `name`, as JSON text."""
    document = json.loads(dumps_manifest(manifest(tmp_path)))
    match name:
        case "unknown field":
            document["unexpected"] = "value"
        case "future version":
            document["manifest_version"] = MANIFEST_VERSION + 1
        case "short digest":
            document["input_file"]["sha256"] = "abc123"
        case "naive instant":
            document["started_at"] = "2026-10-21T09:00:00"
        case "finish before start":
            document["finished_at"] = "2026-10-21T08:00:00+00:00"
        case "negative size":
            document["output_file"]["size_bytes"] = -1
        case _:
            return MALFORMED[name]
    return json.dumps(document)


@pytest.mark.parametrize("name", list(MALFORMED))
def test_a_malformed_document_is_refused_as_a_manifest_error(name, tmp_path):
    text = malformed(name, tmp_path)

    with pytest.raises(ManifestError) as raised:
        loads_manifest(text, source="manifest.json")

    assert isinstance(raised.value, MeasureError)
    assert not isinstance(raised.value, ValidationError)
    assert "manifest.json" in str(raised.value)


def test_the_malformed_table_is_not_empty():
    assert len(MALFORMED) == 9


def test_a_file_that_is_not_a_manifest_names_its_path(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ManifestError, match=str(path)):
        load_manifest(path)


def test_a_caller_s_own_bad_construction_is_a_plain_value_error():
    with pytest.raises(ValueError, match="sha256") as raised:
        FileDigest(path="session.jsonl", sha256="not-a-digest", size_bytes=1)

    assert not isinstance(raised.value, MeasureError)
    assert not isinstance(raised.value, ValidationError)


def test_the_manifest_is_frozen(tmp_path):
    recorded = manifest(tmp_path)

    with pytest.raises(ValueError, match="frozen") as raised:
        recorded.manifest_version = MANIFEST_VERSION + 1

    assert not isinstance(raised.value, ValidationError)
