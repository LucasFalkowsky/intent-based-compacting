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
"""

import hashlib
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

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


def manifest(tmp_path, *, input_bytes: bytes = b"before\n", previous=None) -> RunManifest:
    """Build a manifest over two files written into `tmp_path`."""
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
        metrics=metrics(),
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
    assert recorded.metrics == MetricsRecord.of(metrics())
    assert (recorded.metrics.tokens_before, recorded.metrics.tokens_after) == (5, 3)
    assert recorded.metrics.saved_fraction == pytest.approx(0.4)
    assert recorded.metrics.exploration_rate is None


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


def test_a_first_run_records_no_predecessor(tmp_path):
    recorded = manifest(tmp_path)

    assert recorded.previous_output_sha256 is None
    assert recorded.continues_previous is False


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
    assert second.continues_previous is True
    assert loads_manifest(dumps_manifest(second)) == second


def test_a_broken_chain_is_recorded_rather_than_hidden(tmp_path):
    first = manifest(tmp_path / "run-1")
    second = manifest(tmp_path / "run-2", input_bytes=b"edited by hand\n", previous=first)

    assert second.previous_output_sha256 == first.output_file.sha256
    assert second.continues_previous is False


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
