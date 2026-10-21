"""The evaluation harness: agreement between a classifier run and the gold labels.

The two tests the ticket asks for (#19) are
`test_the_gold_script_scores_a_perfect_run` and
`test_a_wrong_script_scores_below_one`, run over the shared corpus through the
fake adapter — no model, no network, no clock.

Around them, two things are tested the way the ticket requires them to be:

- **The metrics are arithmetic.** `METRIC_CASES` holds hand-worked labellings
  with the accuracy and the Rand index computed on paper, so the formulas have
  an oracle that is independent of any adapter. The table deliberately includes
  the two cases that show the metrics measure different things: a renaming that
  costs all of the accuracy and none of the Rand index, and a single wrong
  decision that costs a quarter of the accuracy and half of the Rand index.
- **The gold labels are input.** Everything scored here is scored against
  `tests/fixtures/gold/`, which no code in this repository writes.
"""

import json
import subprocess
import sys

import pytest

from fixture_corpus import CORPUS, FIXTURES_DIR, GOLD_DIR
from idc.chunk import chunk_session
from idc.classify import (
    EXIT_INPUT_ERROR,
    NEW,
    Agreement,
    CorpusAgreement,
    DecisionRecordError,
    Disagreement,
    EvaluationCase,
    FakeAdapter,
    GoldChains,
    GoldCoverageError,
    GoldLabellingError,
    RunSource,
    UnknownChainError,
    agreement,
    chunks_for,
    classify_chunks,
    evaluate_adapter,
    evaluate_cases,
    format_report,
    gold_script,
    load_decisions,
    per_chunk_accuracy,
    predicted_labels,
    rand_index,
)
from idc.classify.__main__ import main

FIXTURE_NAMES = CORPUS.names()
"""Every fixture of the shared corpus, discovered at collection time."""

MULTI_CHUNK_FIXTURES = ("interleaved_intents", "two_sequential_intents")
"""The fixtures a classifier can get wrong at all.

Seven of the nine transcripts hold a single chunk, where the only decision an
adapter may make is `NEW` — naming a chain is `UnknownChainError`, because none
is on offer yet. A wrong script therefore exists only for these two, which
`test_a_single_chunk_transcript_admits_only_the_gold_decision` pins.
"""

METRIC_CASES = (
    # (id, predicted, gold, accuracy, rand index) — all four numbers by hand.
    (
        "identical",
        {"K1": "C1", "K2": "C1", "K3": "C2", "K4": "C2"},
        {"K1": "C1", "K2": "C1", "K3": "C2", "K4": "C2"},
        1.0,
        1.0,
    ),
    (
        # K2 moved to the other chain: 3 of 4 chunks right, and of the 6 pairs
        # only (K1,K3), (K1,K4) and (K3,K4) agree.
        "one-chunk-misplaced",
        {"K1": "C1", "K2": "C2", "K3": "C2", "K4": "C2"},
        {"K1": "C1", "K2": "C1", "K3": "C2", "K4": "C2"},
        0.75,
        0.5,
    ),
    (
        # Everything merged into one chain: the 2 chunks of C1 keep their label,
        # and of the 6 pairs the 2 that are together in the gold agree.
        "all-merged",
        {"K1": "C1", "K2": "C1", "K3": "C1", "K4": "C1"},
        {"K1": "C1", "K2": "C1", "K3": "C2", "K4": "C2"},
        0.5,
        1 / 3,
    ),
    (
        # One gold chain split into three: only K1 keeps its label and no pair
        # of the 3 is together any more.
        "all-split",
        {"K1": "C1", "K2": "C2", "K3": "C3"},
        {"K1": "C1", "K2": "C1", "K3": "C1"},
        1 / 3,
        0.0,
    ),
    (
        # The partition is right and both names are wrong: the Rand index does
        # not read a label, the accuracy reads nothing else.
        "chains-renamed",
        {"K1": "C2", "K2": "C1"},
        {"K1": "C1", "K2": "C2"},
        0.0,
        1.0,
    ),
    (
        # One chunk, so there is no pair to disagree about.
        "single-chunk-mislabelled",
        {"K1": "C2"},
        {"K1": "C1"},
        0.0,
        1.0,
    ),
    (
        # The third chunk opens a chain instead of rejoining the first — the
        # mistake `interleaved_intents` is in the corpus for.
        "returning-intent-missed",
        {"K1": "C1", "K2": "C2", "K3": "C3"},
        {"K1": "C1", "K2": "C2", "K3": "C1"},
        2 / 3,
        2 / 3,
    ),
)

METRIC_IDS = [entry[0] for entry in METRIC_CASES]
"""Readable ids for the hand-worked metric cases."""


def gold_of(name: str) -> GoldChains:
    """Return one fixture's gold labelling."""
    return CORPUS.gold(name)


def evaluation_case(name: str, script: object) -> EvaluationCase:
    """Return the evaluation case that runs `script` over one fixture."""
    return EvaluationCase(
        gold=gold_of(name),
        session=CORPUS.session(name),
        adapter=FakeAdapter(script),
    )


def write_decisions(directory, name, decisions, **overrides):
    """Write a recorded-run document for one transcript and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    document = {"transcript": name, "decisions": decisions} | overrides
    path = directory / f"{name}.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def recorded(decision: object) -> str:
    """Return a scripted decision as a recorded-run document writes it."""
    return "NEW" if decision is NEW else str(decision)


@pytest.mark.parametrize(
    ("predicted", "gold", "expected"),
    [(entry[1], entry[2], entry[3]) for entry in METRIC_CASES],
    ids=METRIC_IDS,
)
def test_per_chunk_accuracy_is_the_share_of_chunks_labelled_as_in_the_gold(
    predicted, gold, expected
):
    assert per_chunk_accuracy(predicted, gold) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("predicted", "gold", "expected"),
    [(entry[1], entry[2], entry[4]) for entry in METRIC_CASES],
    ids=METRIC_IDS,
)
def test_rand_index_is_the_share_of_chunk_pairs_the_two_partitions_agree_on(
    predicted, gold, expected
):
    assert rand_index(predicted, gold) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("predicted", "gold"),
    [(entry[1], entry[2]) for entry in METRIC_CASES],
    ids=METRIC_IDS,
)
def test_the_rand_index_does_not_depend_on_which_side_is_the_gold(predicted, gold):
    assert rand_index(predicted, gold) == pytest.approx(rand_index(gold, predicted))


@pytest.mark.parametrize("metric", [per_chunk_accuracy, rand_index])
def test_a_metric_refuses_two_labellings_over_different_chunks(metric):
    with pytest.raises(GoldCoverageError, match="K2"):
        metric({"K1": "C1"}, {"K1": "C1", "K2": "C1"})


@pytest.mark.parametrize("metric", [per_chunk_accuracy, rand_index])
def test_a_metric_refuses_an_empty_labelling(metric):
    with pytest.raises(ValueError, match="at least one chunk"):
        metric({}, {})


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_gold_script_scores_a_perfect_run(name):
    # The ticket's first test: the fake adapter playing the gold decisions
    # agrees with the gold labels everywhere, on every fixture.
    result = evaluate_adapter(
        gold_of(name), CORPUS.session(name), FakeAdapter(gold_script(gold_of(name)))
    )

    assert result.accuracy == 1.0
    assert result.rand_index == 1.0
    assert result.disagreements == ()
    assert result.is_perfect


def test_the_gold_script_scores_a_perfect_run_over_the_whole_corpus():
    result = evaluate_cases(
        [evaluation_case(name, gold_script(gold_of(name))) for name in FIXTURE_NAMES],
        RunSource.baseline(),
    )

    assert result.transcript_count == len(FIXTURE_NAMES)
    assert result.accuracy == 1.0
    assert result.rand_index == 1.0
    assert result.is_perfect


@pytest.mark.parametrize(
    ("name", "script", "accuracy", "rand"),
    [
        # Three chains instead of two: the returning intent is missed.
        ("interleaved_intents", [NEW, NEW, NEW], 2 / 3, 2 / 3),
        # One chain instead of two: the unrelated turn is swallowed.
        ("two_sequential_intents", [NEW, "C1"], 0.5, 0.0),
    ],
)
def test_a_wrong_script_scores_below_one(name, script, accuracy, rand):
    # The ticket's second test, with the numbers worked out by hand.
    result = evaluate_adapter(gold_of(name), CORPUS.session(name), FakeAdapter(script))

    assert result.accuracy == pytest.approx(accuracy)
    assert result.rand_index == pytest.approx(rand)
    assert result.accuracy < 1.0
    assert not result.is_perfect


def test_a_wrong_script_names_the_chunks_it_got_wrong():
    result = evaluate_adapter(
        gold_of("interleaved_intents"),
        CORPUS.session("interleaved_intents"),
        FakeAdapter([NEW, NEW, NEW]),
    )

    assert result.disagreements == (Disagreement(chunk_id="K3", predicted="C3", gold="C1"),)
    assert result.predicted_chains == 3
    assert result.gold_chains == 2


@pytest.mark.parametrize(
    "name", [name for name in FIXTURE_NAMES if name not in MULTI_CHUNK_FIXTURES]
)
def test_a_single_chunk_transcript_admits_only_the_gold_decision(name):
    # Why the wrong-script test covers two fixtures and not nine: with no chain
    # open yet, naming one is refused by the classifier before any scoring.
    with pytest.raises(UnknownChainError):
        evaluate_adapter(gold_of(name), CORPUS.session(name), FakeAdapter(["C1"]))


def test_an_agreement_carries_the_counts_the_report_is_built_from():
    result = evaluate_adapter(
        gold_of("interleaved_intents"),
        CORPUS.session("interleaved_intents"),
        FakeAdapter([NEW, NEW, NEW]),
    )

    assert isinstance(result, Agreement)
    assert result.transcript == "interleaved_intents"
    assert result.chunk_count == 3
    assert result.matches == 2
    assert result.pair_count == 3
    assert result.agreeing_pairs == 2


def test_agreement_scores_a_labelling_without_running_an_adapter():
    gold = GoldChains.from_mapping("a_fixture", {"K1": "C1", "K2": "C2", "K3": "C1"})

    result = agreement({"K1": "C1", "K2": "C2", "K3": "C3"}, gold)

    assert result.accuracy == pytest.approx(2 / 3)
    assert result.rand_index == pytest.approx(2 / 3)


def test_agreement_refuses_a_run_that_covers_other_chunks_than_the_gold():
    gold = GoldChains.from_mapping("a_fixture", {"K1": "C1", "K2": "C2"})

    with pytest.raises(GoldCoverageError, match="K2"):
        agreement({"K1": "C1"}, gold)


def test_predicted_labels_reads_one_label_per_assignment():
    session = CORPUS.session("interleaved_intents")
    classification = classify_chunks(session, chunk_session(session), FakeAdapter([NEW, NEW, "C1"]))

    assert predicted_labels(classification) == {"K1": "C1", "K2": "C2", "K3": "C1"}


def test_chunks_for_performs_the_chunking_the_gold_declares():
    session = CORPUS.session("interleaved_intents")

    assert chunks_for(gold_of("interleaved_intents"), session) == chunk_session(session)


def test_chunks_for_refuses_a_chunking_it_cannot_perform():
    gold = GoldChains.from_mapping("a_fixture", {"K1": "C1"}, chunking="subagent-chunks")

    with pytest.raises(GoldLabellingError, match="subagent-chunks"):
        chunks_for(gold, CORPUS.session("single_intent"))


def test_evaluate_cases_aggregates_over_the_chunks_and_the_pairs_of_the_corpus():
    # By hand: interleaved contributes 2 of 3 chunks and 2 of 3 pairs,
    # single_intent 1 of 1 chunk and no pair at all.
    result = evaluate_cases(
        [
            evaluation_case("interleaved_intents", [NEW, NEW, NEW]),
            evaluation_case("single_intent", [NEW]),
        ],
        RunSource.adapter("FakeAdapter"),
    )

    assert isinstance(result, CorpusAgreement)
    assert result.transcript_count == 2
    assert result.chunk_count == 4
    assert result.matches == 3
    assert result.accuracy == pytest.approx(0.75)
    assert result.pair_count == 3
    assert result.agreeing_pairs == 2
    assert result.rand_index == pytest.approx(2 / 3)


def test_evaluate_cases_refuses_an_empty_corpus():
    with pytest.raises(ValueError, match="at least one"):
        evaluate_cases([], RunSource.baseline())


def test_the_report_names_every_transcript_its_numbers_and_its_disagreements():
    result = evaluate_cases(
        [
            evaluation_case("interleaved_intents", [NEW, NEW, NEW]),
            evaluation_case("single_intent", [NEW]),
        ],
        RunSource.adapter("FakeAdapter"),
    )

    report = format_report(result).splitlines()

    assert report[0] == "source: adapter FakeAdapter — run over the labelled transcripts"
    assert report[1].split() == ["transcript", "chunks", "chains", "accuracy", "rand"]
    assert report[2].split() == ["interleaved_intents", "3", "3/2", "0.667", "0.667"]
    assert report[3].split() == ["single_intent", "1", "1/1", "1.000", "1.000"]
    assert report[4] == (
        "corpus [adapter FakeAdapter]: 2 transcripts, 4 chunks, 3 chunk pairs; "
        "accuracy 0.750, rand index 0.667"
    )
    assert report[-2] == "disagreements (predicted / gold):"
    assert report[-1] == "  interleaved_intents K3: C3 / C1"


WORST_LEGAL_SCRIPTS = {
    # The worst a run can do on the two multi-chunk fixtures without being
    # refused: the first chunk has no candidate, so `NEW` is forced there, and
    # every later decision is the one the gold does not name.
    "interleaved_intents": [NEW, "C1", NEW],
    "two_sequential_intents": [NEW, "C1"],
}
"""Per multi-chunk fixture, the legal script that agrees with the gold least."""


def worst_legal_case(name: str) -> EvaluationCase:
    """Return the case that runs the least agreeing legal script over a fixture."""
    return evaluation_case(name, WORST_LEGAL_SCRIPTS.get(name, [NEW]))


def hand_made_agreement(transcript: str, chunk_count: int) -> Agreement:
    """Return an agreement over `chunk_count` chunks, with every decision wrong."""
    pairs = chunk_count * (chunk_count - 1) // 2
    return Agreement(
        transcript=transcript,
        chunk_count=chunk_count,
        matches=0,
        pair_count=pairs,
        agreeing_pairs=0,
        predicted_chains=chunk_count,
        gold_chains=1,
    )


@pytest.mark.parametrize("chunk_count", [1, 2, 5])
def test_an_agreement_knows_which_of_its_chunks_could_not_be_got_wrong(chunk_count):
    # The first chunk of a transcript is offered no candidate, so `NEW` is the
    # only decision the classifier accepts and both sides call that chain C1.
    case = hand_made_agreement("a_fixture", chunk_count)

    assert case.forced_chunks == 1
    assert case.admits_no_wrong_decision == (chunk_count == 1)


@pytest.mark.parametrize(
    ("chunk_counts", "forced_chunks", "forced_transcripts", "floor"),
    [
        ((1,), 1, 1, 1.0),
        ((1, 1, 1, 1, 1, 1, 1, 3, 2), 9, 7, 0.75),
        ((4, 4), 2, 0, 0.25),
        ((1, 1, 3, 2), 4, 2, 4 / 7),
    ],
)
def test_the_corpus_agreement_derives_its_floor_from_its_cases(
    chunk_counts, forced_chunks, forced_transcripts, floor
):
    # Nothing here is a constant of the current corpus: a case added or removed
    # moves all four numbers, which is why they are derived and not written down.
    result = CorpusAgreement(
        cases=tuple(
            hand_made_agreement(f"fixture_{position}", count)
            for position, count in enumerate(chunk_counts, start=1)
        ),
        source=RunSource.adapter("FakeAdapter"),
    )

    assert result.forced_chunks == forced_chunks
    assert result.forced_transcripts == forced_transcripts
    assert result.floor_accuracy == pytest.approx(floor)


def test_no_legal_run_over_the_corpus_scores_below_the_floor():
    # The floor is not a convention but a property of the corpus: the script
    # that agrees least of all the scripts the classifier accepts lands exactly
    # on it, because the forced first chunks are all it gets right.
    worst = evaluate_cases(
        [worst_legal_case(name) for name in FIXTURE_NAMES], RunSource.adapter("FakeAdapter")
    )

    assert worst.chunk_count == 12
    assert worst.forced_chunks == 9
    assert worst.matches == worst.forced_chunks
    assert worst.accuracy == pytest.approx(worst.floor_accuracy)
    assert worst.accuracy == pytest.approx(0.75)


def test_the_report_states_the_floor_it_derived_from_the_corpus():
    result = evaluate_cases(
        [evaluation_case(name, gold_script(gold_of(name))) for name in FIXTURE_NAMES],
        RunSource.baseline(),
    )

    floor_lines = [line for line in format_report(result).splitlines() if line.startswith("floor")]

    assert len(floor_lines) == 2
    assert f"{result.floor_accuracy:.3f}" in floor_lines[0]
    assert f"{result.forced_chunks} of {result.chunk_count} chunks" in floor_lines[0]
    assert f"{result.pair_count} chunk pairs" in floor_lines[1]
    assert f"{result.forced_transcripts} of {result.transcript_count} transcripts" in floor_lines[1]


def test_the_floor_the_report_states_follows_the_corpus_it_was_run_over():
    # Two cases of the same corpus, and the floor moves with them — the report
    # cannot be carrying a number somebody wrote down once.
    one = format_report(
        evaluate_cases([worst_legal_case("interleaved_intents")], RunSource.adapter("FakeAdapter"))
    )
    two = format_report(
        evaluate_cases(
            [worst_legal_case("interleaved_intents"), worst_legal_case("single_intent")],
            RunSource.adapter("FakeAdapter"),
        )
    )

    assert "floor: accuracy 0.333" in one
    assert "1 of 3 chunks" in one
    assert "floor: accuracy 0.500" in two
    assert "2 of 4 chunks" in two


def test_the_report_of_a_perfect_run_lists_no_disagreement():
    report = format_report(
        evaluate_cases([evaluation_case("single_intent", [NEW])], RunSource.baseline())
    )

    assert "disagreements" not in report


@pytest.mark.parametrize(
    ("source", "label", "detail"),
    [
        (RunSource.baseline(), "gold baseline, 1.000 by construction", "gold_script"),
        (RunSource.recorded("runs/s1m-0.5"), "recorded run runs/s1m-0.5", "fake adapter"),
        (RunSource.adapter("S1MAdapter"), "adapter S1MAdapter", "labelled transcripts"),
    ],
    ids=["baseline", "recorded", "adapter"],
)
def test_a_run_source_names_itself_in_its_label_and_in_its_description(source, label, detail):
    assert source.label == label
    assert source.description.startswith(label)
    assert detail in source.description


def test_a_run_source_refuses_to_name_nothing():
    with pytest.raises(ValueError, match="where the decisions came from"):
        RunSource(label="   ")


def test_the_command_scores_the_gold_baseline_over_the_corpus(capsys):
    status = main([str(GOLD_DIR), str(FIXTURES_DIR)])

    out = capsys.readouterr().out
    assert status == 0
    assert all(name in out for name in FIXTURE_NAMES)
    assert "accuracy 1.000, rand index 1.000" in out


def test_the_command_scores_a_recorded_run(tmp_path, capsys):
    decisions_dir = tmp_path / "decisions"
    for name in FIXTURE_NAMES:
        script = [NEW, NEW, NEW] if name == "interleaved_intents" else gold_script(gold_of(name))
        write_decisions(decisions_dir, name, [recorded(entry) for entry in script])

    status = main([str(GOLD_DIR), str(FIXTURES_DIR), "--decisions", str(decisions_dir)])

    out = capsys.readouterr().out
    # 11 of the corpus' 12 chunks and 3 of its 4 chunk pairs, by hand.
    assert status == 0
    assert "accuracy 0.917, rand index 0.750" in out
    assert "interleaved_intents K3: C3 / C1" in out


def test_the_baseline_report_cannot_be_mistaken_for_a_recorded_run(tmp_path, capsys):
    # The baseline scores `gold_script` against the labels it was built from, so
    # it is 1.000 by construction; a recording of those same decisions scores
    # 1.000 too. The numbers are therefore identical and only the provenance
    # tells the two apart — which is why the report has to carry it.
    decisions_dir = tmp_path / "decisions"
    for name in FIXTURE_NAMES:
        write_decisions(
            decisions_dir, name, [recorded(entry) for entry in gold_script(gold_of(name))]
        )

    assert main([str(GOLD_DIR), str(FIXTURES_DIR)]) == 0
    baseline = capsys.readouterr().out
    assert main([str(GOLD_DIR), str(FIXTURES_DIR), "--decisions", str(decisions_dir)]) == 0
    replayed = capsys.readouterr().out

    assert "accuracy 1.000, rand index 1.000" in baseline
    assert "accuracy 1.000, rand index 1.000" in replayed
    assert baseline != replayed
    assert "gold baseline" in baseline
    assert "by construction" in baseline
    assert "gold baseline" not in replayed
    assert str(decisions_dir) in replayed


@pytest.mark.parametrize("decisions", [None, "runs/s1m-0.5"])
def test_the_corpus_line_itself_names_the_source_of_its_numbers(decisions, tmp_path, capsys):
    # The corpus line is the one a reader pastes, so the source has to be on it
    # and not only on a header line the paste leaves behind.
    if decisions is not None:
        decisions = tmp_path / decisions
        for name in FIXTURE_NAMES:
            write_decisions(
                decisions, name, [recorded(entry) for entry in gold_script(gold_of(name))]
            )
    argv = [str(GOLD_DIR), str(FIXTURES_DIR)]
    argv += [] if decisions is None else ["--decisions", str(decisions)]

    assert main(argv) == 0

    corpus = [line for line in capsys.readouterr().out.splitlines() if line.startswith("corpus")]
    expected = "gold baseline" if decisions is None else f"recorded run {decisions}"
    assert len(corpus) == 1
    assert expected in corpus[0]


def test_the_command_reports_a_missing_gold_directory_without_a_traceback(tmp_path, capsys):
    status = main([str(tmp_path / "nowhere"), str(FIXTURES_DIR)])

    captured = capsys.readouterr()
    assert status == EXIT_INPUT_ERROR
    assert "nowhere" in captured.err
    assert captured.out == ""


def test_the_command_reports_a_missing_transcript(tmp_path, capsys):
    status = main([str(GOLD_DIR), str(tmp_path)])

    assert status == EXIT_INPUT_ERROR
    assert "already_compacted_block" in capsys.readouterr().err


def test_the_command_reports_a_missing_recorded_run(tmp_path, capsys):
    decisions_dir = tmp_path / "decisions"
    write_decisions(decisions_dir, "single_intent", ["NEW"])

    status = main([str(GOLD_DIR), str(FIXTURES_DIR), "--decisions", str(decisions_dir)])

    assert status == EXIT_INPUT_ERROR
    assert "already_compacted_block" in capsys.readouterr().err


def test_the_command_reports_a_recorded_run_of_the_wrong_length(tmp_path, capsys):
    decisions_dir = tmp_path / "decisions"
    for name in FIXTURE_NAMES:
        write_decisions(decisions_dir, name, ["NEW"])

    status = main([str(GOLD_DIR), str(FIXTURES_DIR), "--decisions", str(decisions_dir)])

    assert status == EXIT_INPUT_ERROR
    assert "interleaved_intents" in capsys.readouterr().err


def test_load_decisions_reads_new_and_a_chain_id(tmp_path):
    path = write_decisions(tmp_path, "a_run", ["NEW", "C1"])

    assert load_decisions(path) == (NEW, "C1")


def test_load_decisions_rejects_an_entry_that_is_no_decision(tmp_path):
    path = write_decisions(tmp_path, "a_run", ["NEW", "K1"])

    with pytest.raises(DecisionRecordError, match="a_run"):
        load_decisions(path)


def test_load_decisions_rejects_a_transcript_name_that_is_not_the_file_name(tmp_path):
    path = write_decisions(tmp_path, "a_run", ["NEW"], transcript="another_run")

    with pytest.raises(DecisionRecordError, match="another_run"):
        load_decisions(path)


def test_load_decisions_rejects_an_unknown_key(tmp_path):
    path = write_decisions(tmp_path, "a_run", ["NEW"], chains={"K1": "C1"})

    with pytest.raises(DecisionRecordError, match="chains"):
        load_decisions(path)


def test_the_entry_point_runs_the_command_without_a_runpy_warning():
    # `python -m idc.classify` and not `python -m idc.classify.evaluate`: the
    # package imports the harness, so running that module directly would import
    # it twice and runpy would say so on stderr of every evaluation.
    result = subprocess.run(
        [sys.executable, "-m", "idc.classify", str(GOLD_DIR), str(FIXTURES_DIR)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "accuracy 1.000, rand index 1.000" in result.stdout
    assert result.stderr == ""
