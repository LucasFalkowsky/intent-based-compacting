"""A second run on the first run's own output, plus two new chunks (T10.3, #43).

`tests/test_cli_end_to_end.py` pins **one** whole run. This module pins what the
method is actually for: the *next* one. The first run's output is handed back to
`idc.cli.main` with two further user turns appended to it — the file an agent
leaves behind when the session continues after a compaction — and the two things
the ticket asks are asserted on the result:

- **Only the chain that got new chunks changed.** The chain the new turns
  continue is rewritten; the chain they do not touch comes back out of the
  summary **byte for byte**, and costs no model call.
- **The manifest records the previous run's hash.** `previous_output_sha256` is
  the digest of the file the first run wrote, and
  `Manifest.continues_previous` compares it with this run's own input digest.

Everything here reuses the harness of `tests/test_cli_end_to_end.py` — the four
fakes, `native_input`, `assert_golden`, the manifest masking and the socket
guard — rather than building a second one; `produced` *is* that module's first
run, and this module's own fixture takes it as its starting point.

**What the second run is made of.**

- **The input** is the first run's output file with the records of two new
  chunks appended: `second_run_input`. The transcript therefore opens with the
  summary record of the first run, so the run reads its own earlier work back
  through `idc.cli.candidate_chains` and the classifier is offered `C1` and
  `C2` as candidates. That feed is what this module pins hardest, and the
  pinning was measured rather than hoped for: with `candidates=()` in
  `run_command` instead of `candidates=candidate_chains(...)`, the second run
  exits with `no chain 'C1' in the store; the candidates were: none` and **17
  of the 24 tests here fail**. A review had found that same substitution
  leaving the whole suite green.
- **The decisions** are `("C1", "C1", NEW)`, for the three chunks the second
  run sees: the chunk the first run kept verbatim (the gold labelling's last,
  `C1`), then one new chunk continuing that same chain, then one new chunk on an
  unrelated intent. That pair of decisions is the point of the ticket: `C1` is
  **extended**, `C3` is **opened**, and `C2` is **touched by nothing** — without
  a chain in that last state, "only the extended chain changed" would be a claim
  no assertion could falsify.
- **The two new chunks** are hand-written canonical messages
  (`NEW_MESSAGES`), written into Claude Code's record shape by the repository's
  own writer, exactly as `native_input` does it for the first run's input. They
  are not a corpus fixture: the corpus holds *sessions*, and these eight
  messages are a continuation of one, meaningful only after the first run has
  written its summary.
- **No model, no clock, no network**, as in the module this one builds on: the
  adapter, the narrative client, the tokenizers and the clock are injected, and
  `no_network` is re-exported below so its socket guard covers these tests too.

The two golden files under `tests/end_to_end/` named `*_second_run.*` are what
the second run has to produce, byte for byte; `tests/end_to_end/README.md` says
what they pin and how to regenerate them.
"""

import hashlib
import json
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import pytest

from fixture_corpus import CORPUS
from idc.chunk import chunk_session, is_user_turn
from idc.classify import NEW, FakeAdapter, gold_script
from idc.cli import (
    EXIT_INPUT_ERROR,
    EXIT_OK,
    MANIFEST_SUFFIX,
    Runtime,
    candidate_chains,
    main,
)
from idc.compact import FakeNarrativeClient
from idc.importers import import_transcript
from idc.measure import (
    FakeClock,
    FakeTokenizer,
    ModelFamily,
    TokenizerRegistry,
    load_manifest,
    loads_manifest,
)
from idc.render import WRITERS, AssembledTranscript
from idc.schema import Chunk, Message, Role, Session, ToolCall, ToolCategory, ToolResult
from test_cli_end_to_end import (
    AGENT,
    FIXTURE,
    GOLDEN_DIR,
    GOLDEN_MANIFEST,
    GOLDEN_TRANSCRIPT,
    INPUT_PLACEHOLDER,
    OUTPUT_PLACEHOLDER,
    SUMMARY_PLACEHOLDER,
    NetworkUsedError,
    Produced,
    assert_golden,
    no_network,
    normalized_manifest,
    produced,
    readable_records,
)

__all__ = ["no_network", "produced"]
"""Re-exported from `tests/test_cli_end_to_end.py`, which is where they are defined.

`produced` is the first run this module continues, and `no_network` is the
autouse socket guard. pytest finds a fixture by name in the test module's own
namespace, so importing them is what makes them apply here; `__all__` states
that the two imports are re-exports and not leftovers, the way `conftest.py`
does for the corpus paths.
"""

GOLDEN_SECOND_TRANSCRIPT = GOLDEN_DIR / f"{FIXTURE}_second_run.jsonl"
"""The compacted transcript the second run has to write, byte for byte."""

GOLDEN_SECOND_MANIFEST = GOLDEN_DIR / f"{FIXTURE}_second_run.jsonl{MANIFEST_SUFFIX}"
"""Its manifest, named the way `run` names one it was not given a path for."""

SECOND_STARTED = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
"""The instant the second run's fake clock hands out at its start."""

SECOND_FINISHED = datetime(2026, 3, 1, 10, 0, 45, tzinfo=UTC)
"""The instant it hands out at its end."""

EXTENDED = "C1"
"""The chain both runs touch: the first run's third chunk continued it, and so does a new one."""

UNTOUCHED = "C2"
"""The chain the second run does not touch — the one whose block has to come back unchanged."""

OPENED = "C3"
"""The chain the second run opens for the second new chunk."""

SECOND_SCRIPT = (EXTENDED, EXTENDED, NEW)
"""One decision per chunk the second run sees, in transcript order.

The first is about the chunk the first run left in the transcript verbatim and
is the gold labelling's own last decision —
`test_the_scripts_first_decision_is_the_gold_labellings_last` pins that, so a
changed labelling fails here instead of quietly pinning another chain. The two
after it are this ticket's, and they are what make the run contain an extended
chain, a new one and an untouched one at once.
"""


def _stamp(minute: int, second: int) -> datetime:
    """Return an instant of the fixture's own session day, after its last message."""
    return datetime(2026, 9, 3, 10, minute, second, tzinfo=UTC)


NEW_MESSAGES = (
    Message(
        id="m30",
        role=Role.USER,
        content=(
            "The drift is gone. One more: `pick` raises `IndexError` on an empty weights "
            "dict, and it should return `None`."
        ),
        timestamp=_stamp(20, 0),
    ),
    Message(
        id="m31",
        role=Role.ASSISTANT,
        content="Guarding the empty case in `pick`, before the bisect.",
        timestamp=_stamp(20, 20),
        tool_calls=[
            ToolCall(
                id="c13",
                name="Edit",
                category=ToolCategory.WRITE,
                args={
                    "file_path": "src/rates/sampler.py",
                    "old_string": "        index = bisect.bisect_right(",
                    "new_string": (
                        "        if not self._keys:\n"
                        "            return None\n"
                        "        index = bisect.bisect_right("
                    ),
                },
            )
        ],
    ),
    Message(
        id="m32",
        role=Role.TOOL,
        timestamp=_stamp(20, 40),
        tool_result=ToolResult(
            call_id="c13",
            output="Applied 1 edit to src/rates/sampler.py (+2 lines, -0 lines)",
        ),
    ),
    Message(
        id="m33",
        role=Role.ASSISTANT,
        content="Running the rates suite again.",
        timestamp=_stamp(21, 0),
        tool_calls=[
            ToolCall(
                id="c14",
                name="Bash",
                category=ToolCategory.EXEC,
                args={"command": "uv run pytest tests/rates -q"},
            )
        ],
    ),
    Message(
        id="m34",
        role=Role.TOOL,
        timestamp=_stamp(21, 20),
        tool_result=ToolResult(call_id="c14", output=".....\n5 passed in 0.71s\n", exit_code=0),
    ),
    Message(
        id="m35",
        role=Role.USER,
        content=(
            "The config CLI prints a raw traceback when the settings file is missing. "
            "Print one line and exit 2."
        ),
        timestamp=_stamp(22, 0),
    ),
    Message(
        id="m36",
        role=Role.ASSISTANT,
        content="Catching it at the entry point.",
        timestamp=_stamp(22, 20),
        tool_calls=[
            ToolCall(
                id="c15",
                name="Write",
                category=ToolCategory.WRITE,
                args={
                    "file_path": "src/config/cli.py",
                    "content": (
                        "def main() -> int:\n"
                        "    try:\n"
                        "        load()\n"
                        "    except FileNotFoundError as missing:\n"
                        '        print(f"no config file: {missing.filename}")\n'
                        "        return 2\n"
                        "    return 0\n"
                    ),
                },
            )
        ],
    ),
    Message(
        id="m37",
        role=Role.TOOL,
        timestamp=_stamp(22, 40),
        tool_result=ToolResult(call_id="c15", output="Wrote 7 lines to src/config/cli.py"),
    ),
)
"""The two new chunks, hand-written: two user turns and the work each one caused.

Synthetic throughout, and written to continue the corpus fixture rather than to
repeat it: the identifiers carry on from the fixture's own `m01` to `m29` and
`c01` to `c12` so that nothing collides when the records are appended to the first
run's output, and the timestamps are after its last message.

The first turn continues the sampler chain the first run already summarized, and
it writes a file that chain already wrote — which is what makes the extended
chain's `Solution` section a **merge** of the earlier block's and this run's
(`idc.compact.merged_solution_section`) and not a replacement. The second turn is
an unrelated intent with a write and no verification command, so the chain it
opens shows a block whose `Verification` section is honestly `- (none)`.
"""

SECOND_NARRATIVES = (
    "## Intent: Make the weighted sampler deterministic and fix its distribution\n"
    "### Decisions & constraints\n"
    "- The cumulative weights are precomputed once and `bisect` picks the bucket.\n"
    "- An empty sampler returns `None` rather than raising.\n"
    "### Rejected approaches\n"
    "- Re-accumulating the weights on every draw, dropped: that is where the drift came from.\n",
    "## Intent: Report a missing config file as one line instead of a traceback\n"
    "### Decisions & constraints\n"
    "- The entry point catches `FileNotFoundError` and exits with status 2.\n"
    "### Rejected approaches\n"
    "- Letting the traceback out, dropped: it is not a usable error report.\n",
)
"""One answer per chain the second run compacts: the extended `C1`, then the opened `C3`.

Two answers and not three. The untouched chain is **copied**, so a run that
asked the client about it as well would over-draw this script and raise instead
of passing quietly — which is half of what
`test_the_untouched_chain_costs_no_model_call` relies on.

The first answer is deliberately *not* the first run's answer for `C1`
(`test_cli_end_to_end.NARRATIVES[0]`): the extended chain's block has to change
visibly, and an identical narrative would leave the change to the solution merge
alone.
"""


class SecondRun(NamedTuple):
    """The second run, and the first run it continues.

    Attributes:
        first: The first run, as `tests/test_cli_end_to_end.py` produced it.
        status: What `main` returned for the second run.
        input_path: The first run's output plus the new turns — what it read.
        output_path: The compacted transcript it wrote.
        manifest_path: Where it put the manifest, which it was not told.
        adapter: The fake adapter, with its call record.
        client: The fake narrative client, with its call record.
        clock: The fake clock, with what is left of it.
    """

    first: Produced
    status: int
    input_path: Path
    output_path: Path
    manifest_path: Path
    adapter: FakeAdapter
    client: FakeNarrativeClient
    clock: FakeClock


def second_run_input(path: Path, previous_output: Path) -> Path:
    """Write the session file an agent leaves behind when it continues after a run.

    The first run's output file, byte for byte, with the records of
    `NEW_MESSAGES` appended to it: a transcript that opens with the summary
    record the first run wrote, keeps the chunk it left verbatim, and then holds
    two further user turns and their work.

    The records come from the repository's own Claude Code writer, with its
    summary record dropped again — the trick of
    `tests/test_cli_end_to_end.native_input`, for the same reason: the tool
    writes agent-native records in exactly one place, and spelling Claude Code's
    record shape out a second time here would let it drift from the importer
    that has to read it back. `session=` is the corpus fixture's session because
    these messages continue it, so the records carry its `sessionId`.

    Args:
        path: The file to write.
        previous_output: The transcript the first run wrote.

    Returns:
        `path`.
    """
    written = WRITERS[AGENT](
        AssembledTranscript(summary=SUMMARY_PLACEHOLDER, messages=NEW_MESSAGES),
        session=CORPUS.session(FIXTURE),
    )
    _summary_record, records = written.split("\n", 1)
    path.write_text(
        previous_output.read_text(encoding="utf-8") + records,
        encoding="utf-8",
    )
    return path


def run_again(
    source: Path,
    output: Path,
    *,
    script: tuple[str, ...],
    narratives: tuple[str, ...],
    previous_manifest: Path,
    clock: FakeClock,
) -> tuple[int, FakeAdapter, FakeNarrativeClient]:
    """Run the whole pipeline once more, on a transcript an earlier run wrote.

    Args:
        source: The transcript to read.
        output: Where to write the compacted transcript.
        script: One decision per chunk, for the fake adapter.
        narratives: One answer per chain that is compacted, for the fake client.
        previous_manifest: The manifest of the run this one continues.
        clock: The two instants of this run.

    Returns:
        What `main` returned, and the two fakes with their call records.
    """
    adapter = FakeAdapter(script)
    client = FakeNarrativeClient(narratives)
    status = main(
        [
            "run",
            "--agent",
            AGENT.value,
            "--in",
            str(source),
            "--out",
            str(output),
            "--previous-manifest",
            str(previous_manifest),
        ],
        runtime=Runtime(
            adapter=adapter,
            narrative_client=client,
            tokenizers=TokenizerRegistry({family: FakeTokenizer() for family in ModelFamily}),
            clock=clock,
        ),
    )
    return status, adapter, client


@pytest.fixture
def second(produced: Produced, tmp_path: Path) -> SecondRun:
    """Run the pipeline a second time, on the first run's output plus two new chunks."""
    source = second_run_input(tmp_path / "continued_session.jsonl", produced.output_path)
    output = tmp_path / "compacted_again.jsonl"
    clock = FakeClock([SECOND_STARTED, SECOND_FINISHED])
    status, adapter, client = run_again(
        source,
        output,
        script=SECOND_SCRIPT,
        narratives=SECOND_NARRATIVES,
        previous_manifest=produced.manifest_path,
        clock=clock,
    )
    return SecondRun(
        first=produced,
        status=status,
        input_path=source,
        output_path=output,
        manifest_path=Path(str(output) + MANIFEST_SUFFIX),
        adapter=adapter,
        client=client,
        clock=clock,
    )


def masked_manifest(second: SecondRun) -> str:
    """Return the second run's manifest text with its own two paths masked.

    The masking itself is `tests/test_cli_end_to_end.normalized_manifest`, which
    takes a `Produced`; the second run's paths are put into one with
    `_replace`, so there is exactly one definition of what a committed manifest
    may not contain.

    Args:
        second: The finished second run.

    Returns:
        The manifest's text with `input_file.path` and `output_file.path`
        replaced by the two placeholders.
    """
    return normalized_manifest(
        second.manifest_path.read_text(encoding="utf-8"),
        second.first._replace(input_path=second.input_path, output_path=second.output_path),
    )


def blocks_of(path: Path) -> dict[str, str]:
    """Return the chain blocks a run's output carries, by chain id.

    Read back the way the next run reads them — `idc.cli.candidate_chains` on
    the imported transcript — so that what is compared is what the tool itself
    recovers from the file, and not a second parser written for the comparison.

    Args:
        path: A compacted transcript this tool wrote.

    Returns:
        The id of every chain in the summary, mapped to its block's text.
    """
    session = import_transcript(AGENT, path).session
    return {chain.id: chain.compacted_text for chain in candidate_chains(session)}


def summary_text(path: Path) -> str:
    """Return the text of the summary message a compacted transcript carries.

    Args:
        path: A compacted transcript this tool wrote.

    Returns:
        The summary message's content, marker and all.
    """
    session = import_transcript(AGENT, path).session
    return next(one.content for one in session.messages if one.id == session.native_summary_id)


def opening_turn(session: Session, chunk: Chunk) -> Message:
    """Return the user turn a chunk opens at.

    Args:
        session: The session the chunk ranges over.
        chunk: The chunk to look at.

    Returns:
        The first real user turn in the chunk's span.

    Raises:
        AssertionError: If the chunk holds none. A chunk legitimately can —
            `idc.chunk` makes one per subagent span, and those have no user
            turn — but then there is no turn to compare and the comparison
            this helper exists for would be meaningless rather than false.
            Every chunk of this run's transcripts has `ChunkOrigin.USER`.
    """
    for message in session.messages[chunk.start_index : chunk.end_index]:
        if is_user_turn(message):
            return message
    raise AssertionError(f"chunk {chunk.id} holds no user turn")


# --- the two goldens of the second run -----------------------------------------


def test_both_second_run_golden_files_exist() -> None:
    """Guard the two comparisons below: a missing file must not read as a pass."""
    assert GOLDEN_SECOND_TRANSCRIPT.is_file()
    assert GOLDEN_SECOND_MANIFEST.is_file()


def test_the_second_runs_transcript_matches_the_golden_file(second: SecondRun) -> None:
    assert_golden(
        second.output_path.read_text(encoding="utf-8"),
        GOLDEN_SECOND_TRANSCRIPT,
        second.output_path,
        readable=readable_records,
    )


def test_the_second_runs_manifest_matches_the_golden_file(second: SecondRun) -> None:
    """Compare the masked manifest, and leave the masked copy where it is copyable."""
    copyable = second.manifest_path.parent / GOLDEN_SECOND_MANIFEST.name
    copyable.write_text(masked_manifest(second), encoding="utf-8")

    assert_golden(copyable.read_text(encoding="utf-8"), GOLDEN_SECOND_MANIFEST, copyable)


def test_the_mask_replaces_both_paths_of_the_second_run(second: SecondRun) -> None:
    """No part of this run's temporary directory survives into a committed golden."""
    masked = masked_manifest(second)

    assert masked.count(INPUT_PLACEHOLDER) == 1
    assert masked.count(OUTPUT_PLACEHOLDER) == 1
    assert str(second.output_path.parent) not in masked


def test_the_second_runs_golden_files_agree_about_its_output_transcript() -> None:
    """The manifest's output digest is the digest of the committed transcript."""
    data = GOLDEN_SECOND_TRANSCRIPT.read_bytes()
    manifest = load_manifest(GOLDEN_SECOND_MANIFEST)

    assert manifest.output_file.sha256 == hashlib.sha256(data).hexdigest()
    assert manifest.output_file.size_bytes == len(data)


def test_the_second_runs_golden_manifest_names_the_first_runs_golden_transcript() -> None:
    """The two golden pairs chain: the predecessor digest is the first run's output.

    Read off three committed files and nothing else: the second run's manifest,
    the first run's transcript, and the first run's manifest. This is what keeps
    the two pairs one story — regenerating the first run's transcript without
    regenerating the second run's manifest fails here.
    """
    manifest = load_manifest(GOLDEN_SECOND_MANIFEST)

    assert (
        manifest.previous_output_sha256
        == hashlib.sha256(GOLDEN_TRANSCRIPT.read_bytes()).hexdigest()
    )
    assert load_manifest(GOLDEN_MANIFEST).output_file.sha256 == manifest.previous_output_sha256


# --- only the extended chain changed -------------------------------------------


def test_only_the_chain_that_got_new_chunks_changed(second: SecondRun) -> None:
    """Criterion 1 of #43, chain by chain, between the two runs' output files."""
    before = blocks_of(second.first.output_path)
    after = blocks_of(second.output_path)

    assert list(before) == [EXTENDED, UNTOUCHED]
    assert list(after) == [EXTENDED, UNTOUCHED, OPENED]
    assert after[UNTOUCHED] == before[UNTOUCHED]
    assert after[EXTENDED] != before[EXTENDED]


def test_the_untouched_chains_block_is_byte_identical_in_the_written_transcript(
    second: SecondRun,
) -> None:
    """The copied block is in the second run's summary verbatim, not re-rendered.

    `blocks_of` compares what the parser recovers; this compares the bytes the
    run wrote. The untouched chain's block, as the *first* run wrote it, occurs
    in the second run's summary message exactly once — so a re-render that
    changed a single character would fail here even if it still parsed back
    into an equal chain.
    """
    block = blocks_of(second.first.output_path)[UNTOUCHED]

    assert summary_text(second.output_path).count(block) == 1


def test_the_untouched_chain_costs_no_model_call(second: SecondRun) -> None:
    """What makes the second run cheap: one call per changed chain, and none else.

    Both halves matter. The call record names the extended and the opened chain
    and not the untouched one, and the script is drawn empty — so a run that
    asked about all three would have raised rather than passed.
    """
    assert [call.chain_id for call in second.client.calls] == [EXTENDED, OPENED]
    assert second.client.remaining == 0


def test_the_extended_chains_solution_keeps_what_the_first_run_had(second: SecondRun) -> None:
    """The rewritten block is a merge, so the first run's own solution survives in it.

    Without this, "the extended chain changed" would be satisfied by a block
    that threw the earlier run's work away, which is the one way the change
    must *not* happen.
    """
    before = blocks_of(second.first.output_path)[EXTENDED]
    after = blocks_of(second.output_path)[EXTENDED]
    earlier_solution = '    sampler = WeightedSampler({"a": 0.7, "b": 0.3}, random.Random(1))'

    assert earlier_solution in before
    assert earlier_solution in after
    assert "        if not self._keys:" in after
    assert "        if not self._keys:" not in before


# --- the candidate feed, which is what makes any of it incremental -------------


def test_the_second_run_is_offered_the_first_runs_chains(second: SecondRun) -> None:
    """Every decision of the second run was taken with `C1` and `C2` on offer.

    This is the wire `run_command` makes with
    `candidates=candidate_chains(...)`, and it is the reason the rest of this
    module can hold: a run that was offered nothing could not assign a chunk to
    `C1`, so `idc.classify` would raise and the two goldens would never be
    written.
    """
    assert [call.candidate_ids for call in second.adapter.calls] == [(EXTENDED, UNTOUCHED)] * 3


def test_the_first_runs_output_offers_exactly_the_two_chains_back(second: SecondRun) -> None:
    """The candidates come out of the file, not out of this test's knowledge of it."""
    session = import_transcript(AGENT, second.input_path).session

    assert [chain.id for chain in candidate_chains(session)] == [EXTENDED, UNTOUCHED]


def test_the_second_run_sees_three_chunks_and_decides_each_once(second: SecondRun) -> None:
    """The carried chunk and the two new ones, one decision apiece."""
    session = import_transcript(AGENT, second.input_path).session
    chunks = tuple(chunk_session(session))

    assert [chunk.id for chunk in chunks] == ["K1", "K2", "K3"]
    assert [call.chunk_id for call in second.adapter.calls] == [chunk.id for chunk in chunks]
    assert second.adapter.remaining == 0


def test_the_scripts_first_decision_is_the_gold_labellings_last(second: SecondRun) -> None:
    """The chunk the first run left verbatim is the same turn, and the same chain.

    The second run's first chunk opens at the user turn the first run's last
    chunk opened at, and the gold labelling assigns that chunk to `C1` — which
    is why `SECOND_SCRIPT` begins with `C1` rather than by assumption. The two
    decisions after it are this ticket's own.
    """
    first_session = import_transcript(AGENT, second.first.input_path).session
    second_session = import_transcript(AGENT, second.input_path).session
    last_of_first = tuple(chunk_session(first_session))[-1]
    first_of_second = next(iter(chunk_session(second_session)))

    assert opening_turn(second_session, first_of_second).id == (
        opening_turn(first_session, last_of_first).id
    )
    assert gold_script(CORPUS.gold(FIXTURE))[-1] == SECOND_SCRIPT[0] == EXTENDED
    assert SECOND_SCRIPT[1:] == (EXTENDED, NEW)


def test_the_second_runs_input_is_the_first_runs_output_plus_the_new_turns(
    second: SecondRun,
) -> None:
    """Nothing was rewritten to make the second run work: the file only grew."""
    session = import_transcript(AGENT, second.input_path).session
    carried = import_transcript(AGENT, second.first.output_path).session

    assert second.input_path.read_text(encoding="utf-8").startswith(
        second.first.output_path.read_text(encoding="utf-8")
    )
    assert [one.id for one in session.messages] == [one.id for one in carried.messages] + [
        one.id for one in NEW_MESSAGES
    ]


# --- the manifest half ---------------------------------------------------------


def test_the_second_runs_manifest_records_the_first_runs_output_digest(second: SecondRun) -> None:
    """Criterion 2 of #43: `previous_output_sha256` is the predecessor's output digest."""
    first = load_manifest(second.first.manifest_path)
    again = load_manifest(second.manifest_path)

    assert again.previous_output_sha256 == first.output_file.sha256
    assert (
        again.previous_output_sha256
        == hashlib.sha256(second.first.output_path.read_bytes()).hexdigest()
    )


def test_the_first_run_records_no_predecessor(second: SecondRun) -> None:
    """A first run has nothing to chain to, and says so with `null` and no sentinel."""
    first = load_manifest(second.first.manifest_path)

    assert first.previous_output_sha256 is None
    assert first.continues_previous is False


def test_the_second_run_does_not_claim_its_input_is_the_predecessors_output(
    second: SecondRun,
) -> None:
    """`continues_previous` is about the file, and this run's file has grown.

    The second run was handed the predecessor's output **with two chunks
    appended**, so its `input_file.sha256` is not the predecessor's output
    digest and the comparison is `False` — while the predecessor is still
    recorded. That is the honest reading of the two fields for a continued
    session, and the two tests below say what the `True` case would take and
    why no run of this tool reaches it.
    """
    again = load_manifest(second.manifest_path)

    assert again.previous_output_sha256 is not None
    assert again.input_file.sha256 != again.previous_output_sha256
    assert again.continues_previous is False


def test_the_comparison_holds_when_the_input_digest_is_the_recorded_one(
    second: SecondRun,
) -> None:
    """`continues_previous` is exactly the equality of the two digests, on real ones.

    The second run's own manifest, re-read through `loads_manifest` with its
    `input_file.sha256` set to the digest it recorded for its predecessor's
    output — every other value, both digests' *format* and the predecessor
    digest itself included, is the one a real run wrote. So the property under
    test is the comparison and nothing else: the same document is `False`
    above and `True` here, and the one field that moved is the one
    `continues_previous` reads.

    Why this is not a run: see the test below. A run handed its predecessor's
    output unchanged is refused before it can write a manifest, so there is no
    pair of real runs whose digests meet, and a test that claimed to produce
    one would be claiming something this tool cannot do.
    """
    document = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    document["input_file"]["sha256"] = document["previous_output_sha256"]

    chained = loads_manifest(json.dumps(document))

    assert chained.input_file.sha256 == load_manifest(second.manifest_path).previous_output_sha256
    assert chained.continues_previous is True


def test_a_run_handed_the_untouched_output_is_refused(produced: Produced, tmp_path: Path) -> None:
    """Why no two runs of this tool chain by digest: the unchanged output cannot be run.

    A compacted transcript is a summary record plus the chunk the run kept, and
    that is **one** chunk — which is also the latest one, so rendering it again
    would write a second record under `idc.render.SUMMARY_MESSAGE_ID`.
    `idc.render` refuses that (its `errors` module says why), the command exits
    `EXIT_INPUT_ERROR` and writes no manifest. The input of a successful second
    run therefore always has records the predecessor's output did not, which is
    precisely why `continues_previous` is `False` above.
    """
    output = tmp_path / "unchanged.jsonl"
    status, adapter, client = run_again(
        produced.output_path,
        output,
        script=(EXTENDED,),
        narratives=SECOND_NARRATIVES[:1],
        previous_manifest=produced.manifest_path,
        clock=FakeClock([SECOND_STARTED, SECOND_FINISHED]),
    )

    assert status == EXIT_INPUT_ERROR
    assert not output.exists()
    assert not Path(str(output) + MANIFEST_SUFFIX).exists()
    assert [call.chunk_id for call in adapter.calls] == ["K1"]
    assert [call.chain_id for call in client.calls] == [EXTENDED]


# --- no model, no clock, no network --------------------------------------------


def test_the_second_run_succeeds_and_writes_both_files(second: SecondRun) -> None:
    assert second.status == EXIT_OK
    assert second.output_path.is_file()
    assert second.manifest_path.is_file()


def test_the_second_run_asks_the_clock_exactly_twice(second: SecondRun) -> None:
    """The second run's two instants are the injected ones, and it reads no third."""
    manifest = load_manifest(second.manifest_path)

    assert (manifest.started_at, manifest.finished_at) == (SECOND_STARTED, SECOND_FINISHED)
    assert second.clock.remaining == 0


def test_the_network_guard_covers_this_module(second: SecondRun) -> None:
    """The guard really is applied here, and not only in the module it is defined in.

    An autouse fixture imported into another module's namespace is a pytest
    mechanism, not an obvious one, and the whole "no network" claim of this
    module rests on it: if the import stopped taking effect, every run below
    would be free to reach a backend and nothing would say so. The run in
    `second` happened under this.
    """
    assert second.status == EXIT_OK
    with pytest.raises(NetworkUsedError):
        socket.socket()
    with pytest.raises(NetworkUsedError):
        socket.create_connection(("localhost", 9))


def test_the_second_run_opens_no_socket(second: SecondRun) -> None:
    """Both runs happen under the re-exported guard, so a model call would raise."""
    assert second.first.status == EXIT_OK
    assert second.status == EXIT_OK
