"""The whole pipeline in one `run`, pinned by two golden files (T10.2, #42).

Every other test module takes one stage apart. This one asserts the thing none
of them can: that the stages are **connected** — that `idc.cli.main(["run",
...])` carries one transcript through import, chunking, classification,
resolution, compaction, rendering and measurement and leaves two files behind
whose bytes are exactly the bytes committed under `tests/end_to_end/`. It is
driven through `main` with an argument vector for that reason; calling the stage
functions in sequence would re-implement `run_command` and then assert that the
re-implementation works.

What the run is made of, and why each piece is what it is:

- **The input is the corpus fixture `interleaved_intents`**, the discriminating
  case: three user turns pursuing intents A, B, A, so the run has to open two
  chains and *extend* the first — the only fixture where a chain is rejoined.
  `run` reads an **agent-native** transcript, and the corpus is canonical, so
  the native file is written here by the repository's own Claude Code writer;
  `native_input` says exactly how, and
  `test_the_input_is_the_corpus_fixture_in_claude_codes_own_format` pins that
  what the run imports really is that fixture.
- **The decisions are `idc.classify.gold_script`'s**, read off the fixture's
  gold labelling in `tests/fixtures/gold/interleaved_intents.json`: `NEW`,
  `NEW`, then `C1` for the third chunk. Derived rather than written out here, so
  that a labelling and a golden cannot drift apart silently — a hand-written
  script would keep passing while pinning the wrong chains.
- **The four seams are the fakes of `tests/test_cli_run.py`**, including the
  clock: a run's two instants are injected, which is why the golden manifest can
  hold `started_at` and `finished_at` verbatim instead of blanking them. The
  narrative client answers with two *different* texts, so the golden shows that
  each chain got its own narrative and not merely that one arrived.
- **No model, no clock, no network.** The first two follow from the injected
  fakes. The third is asserted and not assumed: `no_network` replaces the
  socket constructors for the whole module, `test_the_network_guard_bites`
  proves the guard would notice, and
  `test_the_run_opens_no_socket` runs the pipeline under it. The limit of that
  is stated at `no_network`.

**When a golden test fails.** `assert_golden` prints a unified diff and names
the file this run wrote, and the `tests/end_to_end/README.md` recipe is to copy
that file over the golden **after reading the diff**. There is deliberately no
switch that rewrites a golden in place: the one way a golden stops protecting
anything is a regression that was accepted because accepting it was one
keystroke.
"""

import difflib
import hashlib
import json
import socket
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import pytest

from fixture_corpus import CORPUS, TESTS_DIR
from idc.chunk import chunk_session
from idc.classify import NEW, FakeAdapter, gold_script
from idc.cli import EXIT_OK, MANIFEST_SUFFIX, Runtime, candidate_chains, main
from idc.compact import DECISIONS_HEADER, REJECTED_HEADER, FakeNarrativeClient
from idc.importers import import_transcript
from idc.measure import (
    FakeClock,
    FakeTokenizer,
    ModelFamily,
    TokenizerRegistry,
    load_manifest,
)
from idc.render import SUMMARY_MESSAGE_ID, WRITERS, AssembledTranscript
from idc.schema import Agent, SessionSource, dumps

FIXTURE = "interleaved_intents"
"""The corpus fixture the run reads: three chunks, two chains, the third rejoins."""

AGENT = Agent.CLAUDE_CODE
"""The fixture's own agent — the format its native input is written in and rendered for."""

GOLDEN_DIR = TESTS_DIR / "end_to_end"
"""Where the two golden files live; see its `README.md`."""

GOLDEN_TRANSCRIPT = GOLDEN_DIR / f"{FIXTURE}.jsonl"
"""The compacted transcript the run has to write, byte for byte."""

GOLDEN_MANIFEST = GOLDEN_DIR / f"{FIXTURE}.jsonl{MANIFEST_SUFFIX}"
"""Its manifest, named the way `run` names one it was not given a path for."""

INPUT_PLACEHOLDER = "<input>"
"""What stands in the golden manifest for `input_file.path`."""

OUTPUT_PLACEHOLDER = "<output>"
"""What stands in it for `output_file.path`."""

STARTED = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
"""The instant the fake clock hands out at the start of the run."""

FINISHED = datetime(2026, 3, 1, 9, 0, 30, tzinfo=UTC)
"""The instant it hands out at the end of it."""

SUMMARY_PLACEHOLDER = "not part of the input; see native_input"
"""The summary text of the record `native_input` drops again."""


def _narrative(intent: str, decision: str, rejected: str) -> str:
    """Return one well-formed narrative answer, as `idc.compact.as_narrative` takes one."""
    return (
        f"## Intent: {intent}\n{DECISIONS_HEADER}\n- {decision}\n{REJECTED_HEADER}\n- {rejected}\n"
    )


NARRATIVES = (
    _narrative(
        "Make the weighted sampler deterministic and fix its distribution",
        "The sampler takes its generator; the caller owns the seeding.",
        "Seeding the global `random` module from the test, dropped as action at a distance.",
    ),
    _narrative(
        "Rename the config key retry_limit to max_retries",
        "The key is renamed in the settings file and in the loader's default.",
        "Keeping `retry_limit` as an alias, dropped: the rename was asked for everywhere.",
    ),
)
"""One answer per chain, in the order the chains are compacted: C1, then C2.

Two different texts rather than one repeated: a single text would pass whether
the chains got their own narrative or the same one twice.
"""


class NetworkUsedError(AssertionError):
    """Raised when something in a run reaches for a socket."""


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every socket constructor in this module's tests raise.

    What this proves and what it does not: every HTTP client in the Python
    ecosystem ends up at `socket.socket` or `socket.create_connection`, so a
    classifier or narrative backend that tried to call a model would fail here
    rather than reach the network. It is not a sandbox — code that opened a
    descriptor another way, or a C extension that connected without the
    standard library, would not be seen. `test_the_network_guard_bites` pins
    that the guard is installed and raises, so the guarded tests cannot pass
    because the patch silently stopped applying.
    """

    def refuse(*args: object, **kwargs: object) -> None:
        raise NetworkUsedError(f"a socket was opened: {args}, {kwargs}")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def native_input(path: Path) -> Path:
    """Write the corpus fixture to `path` as a Claude Code session file.

    The tool writes agent-native transcripts in exactly one place — the writers
    of `idc.render`, which write a *compacted* run: one summary record, then one
    record per kept message. There is no canonical-to-native exporter, and
    `run` needs a native file, so the fixture is run through the Claude Code
    writer and the summary record — the one record that is the compaction's and
    not the session's — is dropped again, leaving one record per message and a
    file that imports back as a plain `raw` session.

    Writing the records out by hand here instead would put Claude Code's record
    shape in a second place and let it drift from the writer the importer is
    tested against. The two properties this relies on are pinned:
    `test_the_generated_input_is_the_fixtures_messages_and_nothing_else` checks
    that the dropped record is the summary and that nothing else was dropped,
    and `test_the_input_is_the_corpus_fixture_in_claude_codes_own_format` checks
    what the importer reads back out of it.

    Args:
        path: The file to write.

    Returns:
        `path`.
    """
    session = CORPUS.session(FIXTURE)
    written = WRITERS[AGENT](
        AssembledTranscript(summary=SUMMARY_PLACEHOLDER, messages=tuple(session.messages)),
        session=session,
    )
    _summary_record, records = written.split("\n", 1)
    path.write_text(records, encoding="utf-8")
    return path


class Produced(NamedTuple):
    """One finished run and everything a test asserts about it.

    Attributes:
        status: What `main` returned.
        input_path: The native transcript the run read.
        output_path: The compacted transcript it wrote.
        manifest_path: Where it put the manifest, which it was not told.
        adapter: The fake adapter, with its call record.
        client: The fake narrative client, with its call record.
        clock: The fake clock, with what is left of it.
    """

    status: int
    input_path: Path
    output_path: Path
    manifest_path: Path
    adapter: FakeAdapter
    client: FakeNarrativeClient
    clock: FakeClock


@pytest.fixture
def produced(tmp_path: Path) -> Produced:
    """Run the whole pipeline once through `main`, and return what it did.

    Function-scoped on purpose: every test gets its own run, its own temporary
    directory and its own three fakes, so no test can depend on what another one
    drew from them.
    """
    source = native_input(tmp_path / f"{FIXTURE}.jsonl")
    output = tmp_path / "compacted.jsonl"
    adapter = FakeAdapter(gold_script(CORPUS.gold(FIXTURE)))
    client = FakeNarrativeClient(NARRATIVES)
    clock = FakeClock([STARTED, FINISHED])
    status = main(
        ["run", "--agent", AGENT.value, "--in", str(source), "--out", str(output)],
        runtime=Runtime(
            adapter=adapter,
            narrative_client=client,
            tokenizers=TokenizerRegistry({family: FakeTokenizer() for family in ModelFamily}),
            clock=clock,
        ),
    )
    return Produced(
        status=status,
        input_path=source,
        output_path=output,
        manifest_path=Path(str(output) + MANIFEST_SUFFIX),
        adapter=adapter,
        client=client,
        clock=clock,
    )


def readable_records(text: str) -> list[str]:
    r"""Return a JSONL text as diff-readable lines, one JSON field per line.

    For the failure message only — the comparison is always on the bytes. A
    record of this tool's output is a single line of several kilobytes, and a
    diff of two such lines says "line 1 differs" and nothing else. Re-printed
    one field per line, with the `\n` escapes inside a field expanded, a diff
    points at the field and at the line of the summary that moved.

    Args:
        text: The file's text.

    Returns:
        One entry per line of the re-printed form, each prefixed with the
        record it belongs to; a line that is not JSON is returned verbatim, so
        a half-written file still produces a readable diff.
    """
    lines = []
    for index, line in enumerate(text.splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            lines.append(f"record {index}: {line}\n")
            continue
        body = json.dumps(record, indent=2, ensure_ascii=False).replace("\\n", "\n")
        lines.extend(f"record {index}: {part}\n" for part in body.splitlines())
    return lines


def readable_lines(text: str) -> list[str]:
    """Return a text's own lines, for a file that is already readable."""
    return text.splitlines(keepends=True)


def assert_golden(
    actual: str,
    golden: Path,
    written: Path,
    readable: Callable[[str], list[str]] = readable_lines,
) -> None:
    """Fail with a usable diff unless `actual` is the golden file's text.

    Args:
        actual: What the run produced.
        golden: The committed file it has to equal, byte for byte.
        written: Where the run's own file is, for the copy-over recipe.
        readable: How to split both texts for the diff; the comparison itself is
            on the texts and never on what this returns.

    Raises:
        Failed: `pytest.fail`, if the golden file is missing or differs. A
            missing golden fails and does not skip: a golden nobody committed
            protects nothing, and the first run after a rename would otherwise
            read as a pass.
    """
    if not golden.is_file():
        pytest.fail(
            f"the golden file {golden} does not exist. This run wrote {written}; see "
            f"{GOLDEN_DIR / 'README.md'} before copying it over."
        )
    expected = golden.read_text(encoding="utf-8")
    if actual == expected:
        return
    diff = "".join(
        difflib.unified_diff(
            readable(expected),
            readable(actual),
            fromfile=f"{golden.name} (committed)",
            tofile="what this run produced",
            n=2,
        )
    )
    pytest.fail(
        f"{golden.name} is not what this run produced.\n\n{diff}\n"
        f"The run's own file is {written}. If the change is intended, read the diff above, "
        f"then copy that file over {golden} — see {GOLDEN_DIR / 'README.md'}."
    )


def normalized_manifest(text: str, run: Produced) -> str:
    """Return a manifest's text with the two paths of this run masked.

    `input_file.path` and `output_file.path` are the paths the run was handed,
    and those are under pytest's temporary directory: they differ per run and
    per machine, and they are the only two values in the document that do. They
    become `<input>` and `<output>`; everything else — both digests, both sizes,
    the two instants, the configuration, the instruction identity, the tokenizer
    and every measured number — stays as the run wrote it and is pinned.

    Args:
        text: The manifest file's text.
        run: The run it belongs to, for the two paths to mask.

    Returns:
        The text with the two paths replaced.
    """
    return text.replace(str(run.output_path), OUTPUT_PLACEHOLDER).replace(
        str(run.input_path), INPUT_PLACEHOLDER
    )


# --- the two goldens -----------------------------------------------------------


def test_both_golden_files_exist() -> None:
    """Guard the two comparisons below: a missing file must not read as a pass."""
    assert GOLDEN_TRANSCRIPT.is_file()
    assert GOLDEN_MANIFEST.is_file()


def test_the_compacted_transcript_matches_the_golden_file(produced: Produced) -> None:
    assert_golden(
        produced.output_path.read_text(encoding="utf-8"),
        GOLDEN_TRANSCRIPT,
        produced.output_path,
        readable=readable_records,
    )


def test_the_manifest_matches_the_golden_file(produced: Produced) -> None:
    """Compare the masked manifest, and leave the masked copy where it is copyable.

    The masked text is written out under the same name the golden has, so that
    regenerating is a copy and never a hand-edit of two path strings in a file
    somebody is already annoyed at.
    """
    masked = produced.manifest_path.parent / GOLDEN_MANIFEST.name
    masked.write_text(
        normalized_manifest(produced.manifest_path.read_text(encoding="utf-8"), produced),
        encoding="utf-8",
    )

    assert_golden(masked.read_text(encoding="utf-8"), GOLDEN_MANIFEST, masked)


def test_the_masked_manifest_keeps_everything_but_the_two_paths(produced: Produced) -> None:
    """The masking cannot hide a difference it was not meant to hide.

    Both placeholders have to appear — otherwise the comparison above would be
    against a golden holding somebody else's temporary directory — and no part
    of this run's directory may survive anywhere in the text.
    """
    masked = normalized_manifest(produced.manifest_path.read_text(encoding="utf-8"), produced)

    assert masked.count(INPUT_PLACEHOLDER) == 1
    assert masked.count(OUTPUT_PLACEHOLDER) == 1
    assert str(produced.output_path.parent) not in masked


def test_the_two_golden_files_agree_about_the_output_transcript() -> None:
    """The manifest's output digest is the digest of the committed transcript.

    Read off the two committed files alone, with `hashlib` rather than the
    tool's own `digest_bytes`, so a pair that disagrees fails here even if the
    run that produced them agreed with itself.
    """
    data = GOLDEN_TRANSCRIPT.read_bytes()
    manifest = load_manifest(GOLDEN_MANIFEST)

    assert manifest.output_file.sha256 == hashlib.sha256(data).hexdigest()
    assert manifest.output_file.size_bytes == len(data)


# --- what the run was actually run on ------------------------------------------


def test_the_generated_input_is_the_fixtures_messages_and_nothing_else(tmp_path: Path) -> None:
    """`native_input` drops the compaction writer's summary record and no message."""
    session = CORPUS.session(FIXTURE)
    full = WRITERS[AGENT](
        AssembledTranscript(summary=SUMMARY_PLACEHOLDER, messages=tuple(session.messages)),
        session=session,
    ).splitlines()
    kept = native_input(tmp_path / "input.jsonl").read_text(encoding="utf-8").splitlines()

    assert json.loads(full[0])["uuid"] == SUMMARY_MESSAGE_ID
    assert kept == full[1:]
    assert [json.loads(line)["uuid"] for line in kept] == [one.id for one in session.messages]


def test_the_input_is_the_corpus_fixture_in_claude_codes_own_format(produced: Produced) -> None:
    """What the run imports is the fixture, with the two losses Claude Code has.

    The run's own first stage is the oracle here: the file is imported and
    compared with the fixture as `idc.schema.dumps` writes both. Two fields
    differ, and both are fields a real Claude Code transcript does not carry:

    - `ToolResult.exit_code` — `idc.render.native` writes none because the
      format has no field for it, which is why the golden transcript's
      Verification lines read `exit unknown` rather than `exit 0`.
    - `Session.created_at` — the writer puts the session's own start on the
      summary record, which this input does not have, so the importer takes the
      first record's timestamp instead. 20 seconds later in this fixture.

    Everything else is identical, the message identifiers and the tool call
    categories included, which is what makes the gold labelling of
    `interleaved_intents` the right script for this run.
    """
    imported = import_transcript(AGENT, produced.input_path).session
    fixture = CORPUS.session(FIXTURE)
    was = json.loads(dumps(imported))
    expected = json.loads(dumps(fixture))

    codes_before = [one.tool_result.exit_code for one in fixture.messages if one.tool_result]
    codes_after = [one.tool_result.exit_code for one in imported.messages if one.tool_result]

    assert imported.source is SessionSource.RAW
    assert any(code is not None for code in codes_before), "fixture records no exit code"
    assert all(code is None for code in codes_after)
    assert was["created_at"] != expected["created_at"]
    for message in expected["messages"]:
        if message["tool_result"] is not None:
            message["tool_result"]["exit_code"] = None
    expected["created_at"] = was["created_at"]
    assert was == expected


def test_the_scripted_decisions_are_one_per_chunk_of_the_run(produced: Produced) -> None:
    """The gold labelling covers the transcript the run chunked, exactly.

    A script shorter than the run would leave `FakeAdapter` over-drawn and a
    longer one would mean the labelling is of some other transcript; either way
    the golden would pin a run that is not the one this fixture describes.
    """
    imported = import_transcript(AGENT, produced.input_path).session
    chunks = tuple(chunk_session(imported))
    script = gold_script(CORPUS.gold(FIXTURE))

    assert len(chunks) == 3
    assert len(script) == len(chunks)
    assert script == (NEW, NEW, "C1")
    assert produced.adapter.remaining == 0
    assert [call.chunk_id for call in produced.adapter.calls] == [one.id for one in chunks]


def test_the_run_compacts_the_chains_the_gold_labelling_names(produced: Produced) -> None:
    """Each chain of the labelling is asked for its own narrative, in order."""
    labelled = list(dict.fromkeys(chain for _, chain in CORPUS.gold(FIXTURE).labels))

    assert labelled == ["C1", "C2"]
    assert [call.chain_id for call in produced.client.calls] == labelled
    assert [call.response for call in produced.client.calls] == list(NARRATIVES)


def test_the_written_transcript_offers_both_chains_back_to_the_next_run(
    produced: Produced,
) -> None:
    """The loop closes: the output re-imports and parses back into the two chains."""
    reread = import_transcript(AGENT, produced.output_path).session

    assert reread.source is SessionSource.INTENT_COMPACTION
    assert [chain.id for chain in candidate_chains(reread)] == ["C1", "C2"]


# --- no model, no clock, no network --------------------------------------------


def test_the_run_succeeds_and_writes_both_files(produced: Produced) -> None:
    assert produced.status == EXIT_OK
    assert produced.output_path.is_file()
    assert produced.manifest_path.is_file()


def test_the_run_asks_the_clock_exactly_twice(produced: Produced) -> None:
    """The run's two instants are the injected ones, and it reads no third."""
    manifest = load_manifest(produced.manifest_path)

    assert (manifest.started_at, manifest.finished_at) == (STARTED, FINISHED)
    assert produced.clock.remaining == 0


def test_the_network_guard_bites() -> None:
    """The guard `no_network` installs is in place and refuses a connection.

    Without this, a patch that stopped applying — a renamed attribute, a
    fixture that is no longer autouse — would turn every "makes no network
    call" assertion in this module into an assertion about nothing.
    """
    with pytest.raises(NetworkUsedError):
        socket.socket()
    with pytest.raises(NetworkUsedError):
        socket.create_connection(("localhost", 9))


def test_the_run_opens_no_socket(produced: Produced) -> None:
    """Criterion 2 of #40: a whole run, with the fakes, and no network call.

    The run happens under the guard above, so reaching the network would raise
    `NetworkUsedError` out of `main` rather than fail an assertion afterwards.
    """
    assert produced.status == EXIT_OK
