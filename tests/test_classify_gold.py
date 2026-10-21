"""Gold chain labels: the hand-written reference a classifier run is scored against.

The labels are an **artifact**, not a derivation (#19). Three kinds of test live
here:

- **The expectation table.** `GOLD_LABELS` repeats, per fixture, the labelling
  read off `tests/fixtures/` by hand, so the committed documents in
  `tests/fixtures/gold/` have an oracle they cannot agree with by accident. It
  is pinned to the discovered corpus in both directions, like the tables in
  `tests/test_classify_assignment.py` and `tests/test_chunk_subagents.py`: a
  tenth fixture fails here until somebody labels it.
- **The document format**, which is read off disk and therefore refused loudly
  rather than guessed at: a name that does not match the file, a missing or
  unknown key, a repeated chunk id, a chain numbering that is not the one
  `idc.classify` itself assigns.
- **The gold baseline script**, the one adapter script that is right by
  construction. It exists so that the harness can be shown to score a perfect
  run 1.0 — never to produce a label.

Nothing here derives a label from a transcript or from a classifier's output,
and nothing calls a model.
"""

import json
import re
from pathlib import Path

import pytest

from fixture_corpus import COMPACT_DIR, CORPUS, GOLD_DIR
from idc.chunk import chunk_session
from idc.classify import (
    GOLD_CHUNKING,
    NEW,
    ClassifyError,
    GoldChains,
    GoldLabellingError,
    gold_script,
    load_gold,
    load_gold_directory,
)
from idc.schema import Role

FIXTURE_NAMES = CORPUS.names()
"""Every fixture of the shared corpus, discovered at collection time."""

GOLD_LABELS: dict[str, dict[str, str]] = {
    # In corpus order, every labelling read off the fixture by hand. The chunks
    # are those of `idc.chunk.chunk_session` — the chunking the documents
    # declare — and the chains are numbered in the order they open.
    #
    # The summary block of the earlier run is not a user turn, so the new turn
    # is the only chunk and opens C1. A run seeded with the block's own chains
    # (#33) would have to label it C2; that is noted in the document and is not
    # what this harness scores.
    "already_compacted_block": {"K1": "C1"},
    # The failed command and its repetition are path, not a second intent.
    "failed_then_passing_command": {"K1": "C1"},
    # A, B, A — the discriminating case: the third turn rejoins C1.
    "interleaved_intents": {"K1": "C1", "K2": "C2", "K3": "C1"},
    # A file written, patched in vain and written again stays one intent too.
    "rewritten_after_failed_edit": {"K1": "C1"},
    "single_intent": {"K1": "C1"},
    # A slash command is no user turn, so there is one chunk, hence one chain.
    "slash_command": {"K1": "C1"},
    # Under user-boundary chunking the subagent's work is part of the turn that
    # delegated it.
    "subagent_result": {"K1": "C1"},
    # Nor is an injected message in the user role a turn.
    "system_injected_message": {"K1": "C1"},
    # The second turn asks for something else ("Now document the new option in
    # README.md, nothing else."), so it opens a chain of its own.
    "two_sequential_intents": {"K1": "C1", "K2": "C2"},
    # A file written and twice corrected within one turn stays one intent.
    "write_then_two_edits": {"K1": "C1"},
}

VALID_DOCUMENT = {
    "transcript": "a_fixture",
    "chunking": GOLD_CHUNKING,
    "note": "A document the format tests mutate one key at a time.",
    "chains": {"K1": "C1", "K2": "C2", "K3": "C1"},
}
"""The shape every format test starts from, so that each one changes one thing."""


def write_document(directory: Path, name: str, document: object) -> Path:
    """Write `document` as `<name>.json` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def write_text_document(directory: Path, name: str, text: str) -> Path:
    """Write `text` as `<name>.json` verbatim and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(text, encoding="utf-8")
    return path


QUOTED_SPAN = re.compile(r'"([^"]+)"')
"""A quotation inside a gold note; the notes quote the user and nothing else."""

NOTES_THAT_REST_ON_THE_USERS_WORDS = (
    "already_compacted_block",
    "interleaved_intents",
    "two_sequential_intents",
)
"""The labellings that turn on where one intent ends and the next begins.

Their notes have to justify that with what the **user** said. File overlap is a
feature a classifier under test keys on directly, so a note arguing from it
would read as endorsing a file-overlap heuristic — and a heuristic that scores
itself against a reference justified by the same heuristic measures nothing.
"""


def user_turns(name: str) -> list[str]:
    """Return the content of every user-role message of one fixture."""
    return [
        message.content or ""
        for message in CORPUS.session(name).messages
        if message.role is Role.USER
    ]


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_quotation_in_a_gold_note_is_verbatim_from_its_transcript(name):
    turns = user_turns(name)

    invented = [
        span
        for span in QUOTED_SPAN.findall(CORPUS.gold(name).note)
        if not any(span in turn for turn in turns)
    ]

    assert invented == [], f"{name} quotes what nobody said: {invented}"


@pytest.mark.parametrize("name", NOTES_THAT_REST_ON_THE_USERS_WORDS)
def test_a_note_about_an_intent_boundary_quotes_the_user(name):
    assert QUOTED_SPAN.findall(CORPUS.gold(name).note)


def test_the_expectation_table_covers_the_corpus():
    assert sorted(GOLD_LABELS) == sorted(FIXTURE_NAMES)


def test_every_fixture_has_gold_labels():
    assert CORPUS.gold_names() == FIXTURE_NAMES


def test_the_gold_directory_holds_nothing_but_json_documents():
    # Every *file* of `gold/` is a labelling; the one sub-directory it holds is
    # `compact/`, the golden compaction Markdown added in T6.1 (#26), whose own
    # suffix rule is pinned in `tests/test_compact_template.py`.
    unexpected = [
        path.name
        for path in sorted(GOLD_DIR.iterdir())
        if path != COMPACT_DIR and (path.is_dir() or path.suffix != ".json")
    ]

    assert unexpected == []


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_gold_labels_match_the_expectation_table(name):
    gold = CORPUS.gold(name)

    assert gold.transcript == name
    assert gold.as_mapping() == GOLD_LABELS[name]


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_gold_labelling_declares_the_chunking_it_belongs_to(name):
    assert CORPUS.gold(name).chunking == GOLD_CHUNKING


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_gold_labelling_says_why_it_is_what_it_is(name):
    # A label is a judgment; a judgment without its reason is not reviewable.
    assert len(CORPUS.gold(name).note) >= 30


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_gold_labelling_covers_the_chunks_of_its_transcript(name):
    chunk_ids = tuple(chunk.id for chunk in chunk_session(CORPUS.session(name)))

    assert CORPUS.gold(name).chunk_ids == chunk_ids


def test_interleaved_intents_returns_to_the_first_chain():
    gold = CORPUS.gold("interleaved_intents")

    assert gold.as_mapping()["K3"] == gold.as_mapping()["K1"]
    assert gold.chain_ids == ("C1", "C2")


def test_load_gold_directory_returns_the_corpus_in_name_order():
    labellings = load_gold_directory(GOLD_DIR)

    assert [gold.transcript for gold in labellings] == FIXTURE_NAMES


def test_load_gold_directory_rejects_a_directory_holding_no_labelling(tmp_path):
    with pytest.raises(GoldLabellingError, match="no gold labelling"):
        load_gold_directory(tmp_path)


def test_load_gold_directory_rejects_a_missing_directory(tmp_path):
    with pytest.raises(GoldLabellingError, match="is not a directory"):
        load_gold_directory(tmp_path / "nowhere")


def test_load_gold_rejects_a_transcript_name_that_is_not_the_file_name(tmp_path):
    path = write_document(tmp_path, "other_fixture", VALID_DOCUMENT)

    with pytest.raises(GoldLabellingError, match="other_fixture"):
        load_gold(path)


def test_load_gold_rejects_an_unknown_key(tmp_path):
    path = write_document(tmp_path, "a_fixture", VALID_DOCUMENT | {"chunks": {}})

    with pytest.raises(GoldLabellingError, match="chunks"):
        load_gold(path)


@pytest.mark.parametrize("key", ["transcript", "chunking", "chains"])
def test_load_gold_rejects_a_missing_key(tmp_path, key):
    document = {name: value for name, value in VALID_DOCUMENT.items() if name != key}
    path = write_document(tmp_path, "a_fixture", document)

    with pytest.raises(GoldLabellingError, match=key):
        load_gold(path)


def test_load_gold_rejects_malformed_json(tmp_path):
    path = write_text_document(tmp_path, "a_fixture", "{not json\n")

    with pytest.raises(GoldLabellingError, match="a_fixture"):
        load_gold(path)


def test_load_gold_rejects_a_document_that_is_not_an_object(tmp_path):
    path = write_document(tmp_path, "a_fixture", ["K1", "C1"])

    with pytest.raises(GoldLabellingError, match="object"):
        load_gold(path)


def test_load_gold_rejects_a_repeated_chunk_id(tmp_path):
    # A plain `json.loads` keeps the last of two equal keys, so the document
    # would silently lose a label it visibly carries.
    path = write_text_document(
        tmp_path,
        "a_fixture",
        json.dumps(
            {
                "transcript": "a_fixture",
                "chunking": GOLD_CHUNKING,
                "chains": {"K1": "C1"},
            }
        ).replace('{"K1": "C1"}', '{"K1": "C1", "K1": "C2"}'),
    )

    with pytest.raises(GoldLabellingError, match="K1"):
        load_gold(path)


def test_load_gold_rejects_a_chain_id_that_is_not_a_string(tmp_path):
    path = write_document(tmp_path, "a_fixture", VALID_DOCUMENT | {"chains": {"K1": 1}})

    with pytest.raises(GoldLabellingError):
        load_gold(path)


def test_load_gold_reads_the_optional_note_as_empty_when_absent(tmp_path):
    document = {name: value for name, value in VALID_DOCUMENT.items() if name != "note"}
    path = write_document(tmp_path, "a_fixture", document)

    assert load_gold(path).note == ""


def test_gold_chains_rejects_an_empty_labelling():
    with pytest.raises(GoldLabellingError, match="at least one chunk"):
        GoldChains.from_mapping("a_fixture", {}, chunking=GOLD_CHUNKING)


def test_gold_chains_rejects_a_blank_transcript_name():
    with pytest.raises(GoldLabellingError, match="transcript"):
        GoldChains.from_mapping("  ", {"K1": "C1"}, chunking=GOLD_CHUNKING)


@pytest.mark.parametrize(
    "build",
    [
        lambda: GoldChains(transcript="a_fixture", labels=(("K1", "C1"),)),
        lambda: GoldChains.from_mapping("a_fixture", {"K1": "C1"}),
    ],
    ids=["constructor", "from_mapping"],
)
def test_a_gold_labelling_built_in_code_has_to_declare_its_chunking(build):
    # A chunk id is a position, so a labelling means nothing without the
    # chunking it was written against. A document cannot omit it (`chunking` is
    # in `REQUIRED_KEYS`); neither may a labelling built in code, or it claims
    # user boundaries without anybody saying so.
    with pytest.raises(TypeError, match="chunking"):
        build()


def test_gold_chains_rejects_a_blank_chunking():
    with pytest.raises(GoldLabellingError, match="chunking"):
        GoldChains.from_mapping("a_fixture", {"K1": "C1"}, chunking="  ")


@pytest.mark.parametrize("chain_id", ["K1", "c1", "C0", "C01", "chain one", ""])
def test_gold_chains_rejects_a_chain_id_of_the_wrong_shape(chain_id):
    with pytest.raises(GoldLabellingError, match="C<n>"):
        GoldChains.from_mapping("a_fixture", {"K1": chain_id}, chunking=GOLD_CHUNKING)


@pytest.mark.parametrize("chunk_id", ["C1", "k1", "K0", "K01", "chunk one", ""])
def test_gold_chains_rejects_a_chunk_id_of_the_wrong_shape(chunk_id):
    with pytest.raises(GoldLabellingError, match="K<n>"):
        GoldChains.from_mapping("a_fixture", {chunk_id: "C1"}, chunking=GOLD_CHUNKING)


def test_gold_chains_rejects_chunk_ids_that_are_not_in_transcript_order():
    with pytest.raises(GoldLabellingError, match="transcript order"):
        GoldChains.from_mapping("a_fixture", {"K2": "C1", "K1": "C1"}, chunking=GOLD_CHUNKING)


@pytest.mark.parametrize(
    "labels",
    [
        {"K1": "C2"},
        {"K1": "C1", "K2": "C3"},
        {"K1": "C2", "K2": "C1"},
    ],
)
def test_gold_chains_rejects_a_chain_numbering_the_classifier_cannot_produce(labels):
    # A chain id *is* its opening position (`idc.classify.ids`), so the k-th
    # chain to appear is `C<k>`. Without that, per-chunk accuracy would compare
    # two different naming schemes and mean nothing.
    with pytest.raises(GoldLabellingError, match="opening order"):
        GoldChains.from_mapping("a_fixture", labels, chunking=GOLD_CHUNKING)


def test_gold_chains_exposes_its_ids_and_its_mapping():
    gold = GoldChains.from_mapping(
        "a_fixture", {"K1": "C1", "K2": "C2", "K3": "C1"}, chunking=GOLD_CHUNKING
    )

    assert len(gold) == 3
    assert gold.chunk_ids == ("K1", "K2", "K3")
    assert gold.chain_ids == ("C1", "C2")
    assert gold.chain_count == 2
    assert gold.as_mapping() == {"K1": "C1", "K2": "C2", "K3": "C1"}


def test_gold_script_opens_a_chain_once_and_extends_it_afterwards():
    gold = GoldChains.from_mapping(
        "a_fixture", {"K1": "C1", "K2": "C2", "K3": "C1"}, chunking=GOLD_CHUNKING
    )

    assert gold_script(gold) == (NEW, NEW, "C1")


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_gold_script_holds_one_decision_per_labelled_chunk(name):
    gold = CORPUS.gold(name)

    assert len(gold_script(gold)) == len(gold)


def test_a_gold_labelling_error_is_a_classify_error():
    # One `except ClassifyError` covers the stage, the harness included.
    assert issubclass(GoldLabellingError, ClassifyError)
