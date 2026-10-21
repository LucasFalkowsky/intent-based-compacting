"""The evaluation harness: how far a classifier run agrees with the gold labels.

The stage's measuring instrument, and the source of the study's classifier
numbers (#19). It runs an adapter over labelled transcripts and reports two
numbers per transcript and one pair for the whole corpus:

- **Per-chunk accuracy** — the share of chunks the run puts into the chain the
  gold labels name. It compares chain *ids*, which is meaningful because both
  sides number their chains in opening order: the store assigns `C<n>` by
  position (`idc.classify.ids`) and a gold labelling is refused if it does not
  (`idc.classify.gold`).
- **Chain-level Rand index** — the share of chunk *pairs* the two labellings
  agree about, counting a pair as agreed when both put it into one chain or both
  keep it apart. It reads no chain id, so it answers the other question: whether
  the run found the same *partition*, independently of what it called the parts.

The two differ, and both are reported because of it. A run that opens one chain
too few can score 0.5 accuracy and 0.33 Rand index; a labelling that gets the
partition right and only the names wrong scores 0.0 accuracy and 1.0 Rand index
— that divergence is reachable for two arbitrary labellings scored through
`per_chunk_accuracy` and `rand_index`, but **not for a run of this pipeline**,
because `classify_chunks` numbers chains in opening order and a gold labelling
that does not is refused (`idc.classify.gold`).

Both numbers carry a **floor that is a property of the corpus** rather than of
the backend, and every report states it: the first chunk of a transcript is
offered no candidate, so `NEW` is the only decision the classifier accepts and
that chunk is scored right in every run that completes. A corpus of mostly
one-chunk transcripts therefore cannot score low, and its Rand index rests on
the few pairs the multi-chunk transcripts form. The numbers are derived from the
cases (`CorpusAgreement.floor_accuracy`, `forced_chunks`, `forced_transcripts`,
`pair_count`), never written down, so a fixture added to the corpus moves them.

```python
from idc.classify import FakeAdapter, evaluate_adapter, gold_script, load_gold

gold = load_gold("tests/fixtures/gold/interleaved_intents.json")
result = evaluate_adapter(gold, session, FakeAdapter(gold_script(gold)))
result.accuracy, result.rand_index        # (1.0, 1.0) — the baseline
```

Both metrics are **arithmetic over two labellings** and nothing else: no model,
no clock, no sampling, and they are unit-tested against values worked out by
hand rather than against a classifier's output. The gold labels are read from an
artifact nothing in this repository writes (`idc.classify.gold`), so the harness
cannot score itself.

The command is a development utility, not part of the tool's command surface
(T10.1) — the same arrangement as `python -m idc.schema`:

```
python -m idc.classify tests/fixtures/gold tests/fixtures
python -m idc.classify GOLD_DIR TRANSCRIPT_DIR --decisions RUN_DIR
```

The argument parsing lives in `idc.classify.__main__`, so that running the
command does not import this module twice (`runpy` warns about exactly that when
a package imports the module it is asked to run).

Every report **names the source of its numbers** — a `source:` line and the
label on the corpus line itself, because that is the line a reader pastes. The
baseline and a real run that scored 1.000 print otherwise identical numbers, so
the provenance is the only thing that tells them apart; `RunSource` is therefore
a required part of a `CorpusAgreement` rather than a formatting option.

It pairs a gold labelling with the transcript of the same name and runs one
adapter per transcript. Without `--decisions` that adapter plays the **gold
baseline** (`gold_script`), which is right by construction and therefore prints
1.000 — the harness's own check. With `--decisions` it replays a **recorded
run**: the decision list a real backend produced, written to a file so that it
can be scored offline. That indirection is not a convenience, it is the only
honest option: the LLM and S1M adapters take their model as an injected
dependency, so no command of this tool can construct one, and nothing here ever
calls a model.

What a recording can and cannot answer is settled on #18 and #19: a decision is
exact, a whole-transcript threshold comparison is not, because the candidate set
itself depends on the threshold. A sweep therefore means re-running the
classifier with another adapter and scoring again, which is what this harness is
for.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Self

from idc.chunk import chunk_session
from idc.classify.adapters import NEW, DecisionLike, IntentAdapter
from idc.classify.classifier import Classification, classify_chunks
from idc.classify.errors import (
    DecisionRecordError,
    GoldCoverageError,
    GoldLabellingError,
)
from idc.classify.fake import FakeAdapter
from idc.classify.gold import (
    GOLD_CHUNKING,
    GoldChains,
    gold_script,
    load_gold_directory,
    read_json_object,
)
from idc.classify.ids import is_chain_id
from idc.schema import Chunk, Session, load

CHUNKINGS: Mapping[str, Callable[[Session], list[Chunk]]] = {
    GOLD_CHUNKING: chunk_session,
}
"""Which chunking a gold labelling may declare, and what performs it.

One entry, deliberately: the shared corpus is labelled for the user-boundary
chunking (T3.1), and a labelling for the subagent chunking (T3.3) is a second
artifact that does not exist yet. A gold document naming anything else is
refused rather than scored against the wrong chunks, and adding a chunking is a
change here *and* in the documents — which is the point of declaring it.
"""

DOCUMENT_SUFFIX = ".json"
"""Every document a case is built from is one `<transcript name>.json` file.

The transcript, its gold labelling and a recorded run of it are three files of
the same name in three directories, which is what lets the command pair them
without a path inside any of the documents.
"""

NEW_DECISION = "NEW"
"""How a recorded run spells the decision to open a chain."""

DECISION_KEYS = ("transcript", "decisions")
"""The keys a recorded-run document has to carry."""

DECISION_OPTIONAL_KEYS = ("note",)
"""The keys it may carry."""

BASELINE_LABEL = "gold baseline, 1.000 by construction"
"""How the report names the one run that is right before it is performed.

The baseline plays `gold_script` back against the very labels it was derived
from, so it reproduces them exactly — its 1.000 is a self-check of the harness
and no statement about any classifier. The label says so on the line a reader
would paste, because a real run that happened to score 1.000 prints a line of
the same shape.
"""

BASELINE_DETAIL = (
    "the gold labels played back through gold_script, so it scores itself and "
    "measures no classifier"
)
"""The sentence the baseline's `source:` line carries beyond its label."""

EXIT_OK = 0
"""The command's exit status when it printed a report."""

EXIT_INPUT_ERROR = 2
"""Its exit status when an input could not be used; the message goes to stderr."""


def _checked_chunk_ids(predicted: Mapping[str, str], gold: Mapping[str, str]) -> tuple[str, ...]:
    """Return the chunk ids both labellings cover, in the gold's order.

    Args:
        predicted: The run's `chunk id -> chain id` labelling.
        gold: The reference labelling.

    Returns:
        The chunk ids, in the order the gold lists them.

    Raises:
        ValueError: If both labellings are empty — a share of nothing is not a
            number, and an empty labelling is the caller's own mistake rather
            than something a document could say (`GoldChains` refuses one).
        GoldCoverageError: If the two cover different chunks.
    """
    missing = [chunk for chunk in gold if chunk not in predicted]
    unexpected = [chunk for chunk in predicted if chunk not in gold]
    if missing or unexpected:
        raise GoldCoverageError(missing, unexpected)
    if not gold:
        raise ValueError("a labelling is scored over at least one chunk")
    return tuple(gold)


def per_chunk_accuracy(predicted: Mapping[str, str], gold: Mapping[str, str]) -> float:
    """Return the share of chunks the run labels as the gold does.

    Args:
        predicted: The run's `chunk id -> chain id` labelling.
        gold: The reference labelling.

    Returns:
        A number in `[0, 1]`: the number of chunks whose chain id is the gold's,
        divided by the number of chunks.

    Raises:
        ValueError: If the labellings are empty.
        GoldCoverageError: If they cover different chunks.
    """
    chunk_ids = _checked_chunk_ids(predicted, gold)
    matches = sum(1 for chunk in chunk_ids if predicted[chunk] == gold[chunk])
    return matches / len(chunk_ids)


def _pair_agreement(
    predicted: Mapping[str, str], gold: Mapping[str, str], chunk_ids: Sequence[str]
) -> tuple[int, int]:
    """Return how many chunk pairs the two labellings agree on, and how many there are.

    Args:
        predicted: The run's labelling.
        gold: The reference labelling.
        chunk_ids: The chunks to pair up.

    Returns:
        `(agreeing pairs, pairs)`. A pair agrees when both labellings put its
        two chunks into one chain, or both keep them apart.
    """
    pairs = 0
    agreeing = 0
    for first, second in combinations(chunk_ids, 2):
        pairs += 1
        if (predicted[first] == predicted[second]) == (gold[first] == gold[second]):
            agreeing += 1
    return agreeing, pairs


def rand_index(predicted: Mapping[str, str], gold: Mapping[str, str]) -> float:
    """Return the Rand index of the run's chains against the gold's.

    The share of chunk pairs the two labellings agree about — `(a + b) / C(n, 2)`
    with `a` the pairs both keep together and `b` the pairs both keep apart. It
    reads no chain id, so renaming the chains does not change it, and it is
    symmetric in its two arguments.

    Args:
        predicted: The run's `chunk id -> chain id` labelling.
        gold: The reference labelling.

    Returns:
        A number in `[0, 1]`; `1.0` for a single chunk, which forms no pair that
        could disagree.

    Raises:
        ValueError: If the labellings are empty.
        GoldCoverageError: If they cover different chunks.
    """
    chunk_ids = _checked_chunk_ids(predicted, gold)
    agreeing, pairs = _pair_agreement(predicted, gold, chunk_ids)
    return agreeing / pairs if pairs else 1.0


@dataclass(frozen=True, slots=True)
class Disagreement:
    """One chunk a run put into another chain than the gold labels name.

    Attributes:
        chunk_id: The chunk.
        predicted: The chain the run assigned it to.
        gold: The chain the gold labels name.
    """

    chunk_id: str
    predicted: str
    gold: str


@dataclass(frozen=True, slots=True)
class Agreement:
    """How far one run over one transcript agrees with its gold labels.

    The counts are kept next to the two shares, because the corpus-wide numbers
    are built from them — summing shares would weight a one-chunk transcript
    like a twenty-chunk one.

    Attributes:
        transcript: Name of the transcript.
        chunk_count: How many chunks were labelled on both sides.
        matches: How many of them the run labelled as the gold does.
        pair_count: How many chunk pairs the transcript has, `C(n, 2)`.
        agreeing_pairs: How many of those pairs the two labellings agree on.
        predicted_chains: How many chains the run opened.
        gold_chains: How many chains the gold labels hold.
        disagreements: The chunks that went elsewhere, in transcript order.
    """

    transcript: str
    chunk_count: int
    matches: int
    pair_count: int
    agreeing_pairs: int
    predicted_chains: int
    gold_chains: int
    disagreements: tuple[Disagreement, ...] = ()

    @property
    def accuracy(self) -> float:
        """Return the per-chunk accuracy."""
        return self.matches / self.chunk_count

    @property
    def rand_index(self) -> float:
        """Return the chain-level Rand index; `1.0` when there is no pair."""
        return self.agreeing_pairs / self.pair_count if self.pair_count else 1.0

    @property
    def forced_chunks(self) -> int:
        """Return how many of the chunks could not be labelled wrongly.

        A transcript's **first** chunk is offered no candidate — there is no
        chain yet — so `NEW` is the only decision `classify_chunks` accepts, and
        both sides name the chain it opens `C1`: the store numbers chains in
        opening order (`idc.classify.ids`) and a gold labelling is refused
        unless it does too (`idc.classify.gold`). Every run that completes
        therefore scores that one chunk right, in every transcript. It holds for
        the runs this harness performs, which start from no chain; a store
        seeded with chains parsed back out of a summary block (#33) is a
        different situation.
        """
        return min(self.chunk_count, 1)

    @property
    def admits_no_wrong_decision(self) -> bool:
        """Return whether every decision of this transcript was forced.

        True for a one-chunk transcript: its only decision is the forced one, so
        the transcript contributes a guaranteed match and no chunk pair at all.
        """
        return self.chunk_count == self.forced_chunks

    @property
    def is_perfect(self) -> bool:
        """Return whether the run agrees with the gold labels everywhere."""
        return self.matches == self.chunk_count and self.agreeing_pairs == self.pair_count


@dataclass(frozen=True, slots=True)
class RunSource:
    """Where the decisions a report scores came from.

    A `CorpusAgreement` carries one, so that a number cannot travel without the
    thing that produced it: the baseline is right by construction, a recorded
    run is a real backend's decisions read from a directory, and a live adapter
    is neither. Two numbers of the same value mean entirely different things
    depending on which of the three it was, and the report prints the label on
    the corpus line itself.

    Attributes:
        label: The short name of the source, for the corpus line.
        detail: One clause saying what that source is, for the `source:` line.
            Empty when the label says everything.
    """

    label: str
    detail: str = ""

    def __post_init__(self) -> None:
        """Reject a source that names nothing.

        Raises:
            ValueError: If the label holds no word. A report without a source is
                the mistake this type exists against, and an unnamed source is
                the caller's own construction error rather than something a
                document could say.
        """
        if not self.label.strip():
            raise ValueError("a run source names where the decisions came from")

    @property
    def description(self) -> str:
        """Return the label with its detail, as the `source:` line reads it."""
        return f"{self.label} — {self.detail}" if self.detail else self.label

    @classmethod
    def baseline(cls) -> Self:
        """Return the source of a run of the gold baseline (`gold_script`)."""
        return cls(label=BASELINE_LABEL, detail=BASELINE_DETAIL)

    @classmethod
    def recorded(cls, directory: str | Path) -> Self:
        """Return the source of a recorded run replayed from `directory`.

        Args:
            directory: Directory the recorded runs were read from.

        Returns:
            The source, naming the directory — the one thing that says which
            backend's decisions these were.
        """
        return cls(
            label=f"recorded run {Path(directory)}",
            detail="decisions recorded from a real backend, replayed through the fake adapter",
        )

    @classmethod
    def adapter(cls, name: str) -> Self:
        """Return the source of a run performed live by an adapter.

        Args:
            name: How the adapter is named in the report — a class name, or a
                backend and its settings.

        Returns:
            The source; what a study script that runs an adapter itself passes
            to `evaluate_cases`.
        """
        return cls(label=f"adapter {name}", detail="run over the labelled transcripts")


@dataclass(frozen=True, slots=True)
class CorpusAgreement:
    """The agreement over several labelled transcripts.

    Both corpus numbers are **micro** averages — accuracy over all chunks, Rand
    index over all chunk pairs — so a transcript counts as much as it carries.
    Pairs are never formed across transcripts: two chunks of different
    transcripts are not in one chain in any labelling, and counting them would
    inflate both sides with agreements nobody decided.

    Both numbers therefore have a **structural floor** that says more about the
    corpus than about the classifier, and it is derived here rather than written
    down anywhere: `floor_accuracy` is the share of chunks whose decision was
    forced, and `pair_count` is how little the Rand index rests on. A corpus of
    mostly one-chunk transcripts scores high whatever the backend does.

    Attributes:
        cases: One agreement per transcript, in the order they were evaluated.
        source: Where the decisions came from. Required, because the numbers
            below mean different things depending on it — the baseline's 1.000
            is arithmetic, a real run's 1.000 is a result.
    """

    cases: tuple[Agreement, ...]
    source: RunSource

    def __post_init__(self) -> None:
        """Reject an empty corpus.

        Raises:
            ValueError: If no case was evaluated; there is nothing to average.
        """
        if not self.cases:
            raise ValueError("a corpus agreement covers at least one transcript")

    @property
    def transcript_count(self) -> int:
        """Return how many transcripts were evaluated."""
        return len(self.cases)

    @property
    def chunk_count(self) -> int:
        """Return how many chunks were scored in total."""
        return sum(case.chunk_count for case in self.cases)

    @property
    def matches(self) -> int:
        """Return how many chunks were labelled as the gold does."""
        return sum(case.matches for case in self.cases)

    @property
    def pair_count(self) -> int:
        """Return how many chunk pairs were scored, summed per transcript."""
        return sum(case.pair_count for case in self.cases)

    @property
    def agreeing_pairs(self) -> int:
        """Return how many of those pairs the labellings agree on."""
        return sum(case.agreeing_pairs for case in self.cases)

    @property
    def accuracy(self) -> float:
        """Return the per-chunk accuracy over all chunks of the corpus."""
        return self.matches / self.chunk_count

    @property
    def rand_index(self) -> float:
        """Return the Rand index over all chunk pairs; `1.0` when there is none."""
        return self.agreeing_pairs / self.pair_count if self.pair_count else 1.0

    @property
    def forced_chunks(self) -> int:
        """Return how many chunks of the corpus could not be labelled wrongly.

        One per transcript — its first; see `Agreement.forced_chunks`.
        """
        return sum(case.forced_chunks for case in self.cases)

    @property
    def forced_transcripts(self) -> int:
        """Return how many transcripts admit no wrong decision at all.

        The one-chunk ones: they contribute a guaranteed match to the accuracy
        and no pair to the Rand index.
        """
        return sum(1 for case in self.cases if case.admits_no_wrong_decision)

    @property
    def floor_accuracy(self) -> float:
        """Return the accuracy no run that completes can fall below.

        The share of chunks whose decision was forced. It is a property of the
        corpus, not of a backend: a run can only lose the chunks above it, which
        is what makes the distance between `accuracy` and this number the part
        that measures anything.
        """
        return self.forced_chunks / self.chunk_count

    @property
    def is_perfect(self) -> bool:
        """Return whether every case agrees with its gold labels everywhere."""
        return all(case.is_perfect for case in self.cases)


def predicted_labels(classification: Classification) -> dict[str, str]:
    """Return a run's `chunk id -> chain id` labelling.

    Args:
        classification: What `classify_chunks` returned.

    Returns:
        One chain id per classified chunk, in transcript order.
    """
    return {assignment.chunk_id: assignment.chain_id for assignment in classification.assignments}


def agreement(predicted: Mapping[str, str], gold: GoldChains) -> Agreement:
    """Return how far one labelling agrees with the gold labels.

    The scoring step on its own, for a run that was produced elsewhere.

    Args:
        predicted: The run's `chunk id -> chain id` labelling.
        gold: The reference labelling.

    Returns:
        The agreement, with the counts both metrics are built from.

    Raises:
        GoldCoverageError: If the run and the gold cover different chunks.
    """
    reference = gold.as_mapping()
    chunk_ids = _checked_chunk_ids(predicted, reference)
    agreeing, pairs = _pair_agreement(predicted, reference, chunk_ids)
    disagreements = tuple(
        Disagreement(chunk_id=chunk, predicted=predicted[chunk], gold=reference[chunk])
        for chunk in chunk_ids
        if predicted[chunk] != reference[chunk]
    )
    return Agreement(
        transcript=gold.transcript,
        chunk_count=len(chunk_ids),
        matches=len(chunk_ids) - len(disagreements),
        pair_count=pairs,
        agreeing_pairs=agreeing,
        predicted_chains=len({predicted[chunk] for chunk in chunk_ids}),
        gold_chains=gold.chain_count,
        disagreements=disagreements,
    )


def chunks_for(gold: GoldChains, session: Session) -> list[Chunk]:
    """Return the chunks of `session` under the chunking the gold declares.

    Args:
        gold: The labelling, which names its chunking.
        session: The transcript to chunk.

    Returns:
        The chunks, in transcript order.

    Raises:
        GoldLabellingError: If the labelling declares a chunking this harness
            cannot perform; see `CHUNKINGS`.
    """
    chunker = CHUNKINGS.get(gold.chunking)
    if chunker is None:
        known = ", ".join(CHUNKINGS)
        raise GoldLabellingError(
            f"the gold labelling of {gold.transcript!r} is written for the chunking "
            f"{gold.chunking!r}, which this harness cannot perform; it knows: {known}"
        )
    return chunker(session)


def evaluate_adapter(gold: GoldChains, session: Session, adapter: IntentAdapter) -> Agreement:
    """Run one adapter over one labelled transcript and score the result.

    Args:
        gold: The transcript's gold labels.
        session: The transcript itself.
        adapter: The decision backend to measure.

    Returns:
        The agreement between the run and the gold labels.

    Raises:
        GoldLabellingError: If the gold declares a chunking that cannot be
            performed.
        GoldCoverageError: If the run's chunks are not the labelled ones.
        ClassifyError: If the run itself fails — an adapter naming a chain that
            was not a candidate, a script that ran out; see `classify_chunks`.
    """
    chunks = chunks_for(gold, session)
    classification = classify_chunks(session, chunks, adapter)
    return agreement(predicted_labels(classification), gold)


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One transcript, its gold labels and the adapter to measure on it.

    The adapter belongs to the case rather than to the run: every backend worth
    measuring carries state over its decisions — a script position, a client, a
    cache — so one adapter per transcript is what keeps the cases independent.

    Attributes:
        gold: The transcript's gold labels.
        session: The transcript.
        adapter: The decision backend for this transcript.
    """

    gold: GoldChains
    session: Session
    adapter: IntentAdapter


def evaluate_cases(cases: Sequence[EvaluationCase], source: RunSource) -> CorpusAgreement:
    """Evaluate every case and return the corpus-wide agreement.

    Args:
        cases: The transcripts to evaluate, in report order.
        source: Where the decisions the cases' adapters make come from; see
            `RunSource`. It is an argument rather than something derived from
            the adapters, because an adapter cannot say whether it is replaying
            a recording or deciding for itself.

    Returns:
        The agreement per case and the micro averages over all of them, under
        the source that produced them.

    Raises:
        ValueError: If `cases` is empty.
        ClassifyError: If a case cannot be evaluated; see `evaluate_adapter`.
    """
    return CorpusAgreement(
        cases=tuple(evaluate_adapter(case.gold, case.session, case.adapter) for case in cases),
        source=source,
    )


def _plural(count: int, noun: str) -> str:
    """Return `count` and `noun`, pluralized with a trailing `s`."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def format_report(result: CorpusAgreement) -> str:
    """Return the agreement as the report the command prints.

    A `source:` line, one row per transcript, the corpus line — which repeats
    the source, because that is the line a reader pastes — the two `floor:`
    lines that say how much of the corpus number is structural, and, only when
    there are any, the chunks that went into another chain than the gold labels
    name.

    Args:
        result: What `evaluate_cases` returned, source included.

    Returns:
        The report, without a trailing newline.
    """
    rows = [("transcript", "chunks", "chains", "accuracy", "rand")]
    rows += [
        (
            case.transcript,
            str(case.chunk_count),
            f"{case.predicted_chains}/{case.gold_chains}",
            f"{case.accuracy:.3f}",
            f"{case.rand_index:.3f}",
        )
        for case in result.cases
    ]
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    lines = [f"source: {result.source.description}"]
    lines.append(
        "  ".join(
            [rows[0][0].ljust(widths[0])]
            + [cell.rjust(width) for cell, width in zip(rows[0][1:], widths[1:], strict=True)]
        )
    )
    lines += [
        "  ".join(
            [row[0].ljust(widths[0])]
            + [cell.rjust(width) for cell, width in zip(row[1:], widths[1:], strict=True)]
        )
        for row in rows[1:]
    ]
    lines.append(
        f"corpus [{result.source.label}]: {_plural(result.transcript_count, 'transcript')}, "
        f"{_plural(result.chunk_count, 'chunk')}, "
        f"{_plural(result.pair_count, 'chunk pair')}; "
        f"accuracy {result.accuracy:.3f}, rand index {result.rand_index:.3f}"
    )
    lines.append(
        f"floor: accuracy {result.floor_accuracy:.3f} — {result.forced_chunks} of "
        f"{_plural(result.chunk_count, 'chunk')} are a transcript's first chunk, where NEW is "
        f"the only decision the classifier accepts, so no run that completes scores below it"
    )
    lines.append(
        f"floor: the rand index rests on {_plural(result.pair_count, 'chunk pair')}; "
        f"{result.forced_transcripts} of {_plural(result.transcript_count, 'transcript')} hold "
        f"a single chunk and form no pair at all"
    )
    disagreements = [
        f"  {case.transcript} {entry.chunk_id}: {entry.predicted} / {entry.gold}"
        for case in result.cases
        for entry in case.disagreements
    ]
    if disagreements:
        lines.append("disagreements (predicted / gold):")
        lines += disagreements
    return "\n".join(lines)


def load_decisions(path: str | Path) -> tuple[DecisionLike, ...]:
    """Return the decisions one recorded-run document holds.

    The document is a JSON object naming its transcript and listing one decision
    per chunk, in transcript order — `"NEW"` or a chain id:

    ```json
    {"transcript": "interleaved_intents", "decisions": ["NEW", "NEW", "C1"]}
    ```

    A confidence is deliberately not part of it: agreement is computed from the
    decisions alone, and a recorded confidence answers a different question
    (#18).

    Args:
        path: Path of a `<transcript name>.json` recorded-run document.

    Returns:
        The decisions, in the order they were made, ready for `FakeAdapter`.

    Raises:
        DecisionRecordError: If the document cannot be read, does not have the
            documented shape, names a transcript other than its own file, or
            holds an entry that is no decision.
    """
    path = Path(path)
    try:
        document = read_json_object(path, DECISION_KEYS, DECISION_OPTIONAL_KEYS)
    except GoldLabellingError as error:
        raise DecisionRecordError(str(error)) from error
    transcript = document["transcript"]
    if transcript != path.stem:
        raise DecisionRecordError(
            f"{path.name} records the transcript {transcript!r}; a recorded run is named after "
            f"the transcript it was produced on"
        )
    entries = document["decisions"]
    if not isinstance(entries, list):
        raise DecisionRecordError(
            f"{path.name} must list its decisions under 'decisions'; got {type(entries).__name__}"
        )
    decisions: list[DecisionLike] = []
    for position, entry in enumerate(entries, start=1):
        if entry == NEW_DECISION:
            decisions.append(NEW)
        elif is_chain_id(entry):
            decisions.append(entry)
        else:
            raise DecisionRecordError(
                f"{path.name} records {entry!r} as decision {position}; a recorded decision is "
                f"{NEW_DECISION!r} or a chain id of the form C<n>"
            )
    return tuple(decisions)


def load_cases(
    gold_directory: str | Path,
    transcript_directory: str | Path,
    decision_directory: str | Path | None = None,
) -> list[EvaluationCase]:
    """Return one case per gold labelling, paired by transcript name.

    The reading half of the command, and what a study script calls to evaluate
    a corpus: a gold labelling, the transcript of the same name, and the adapter
    to measure — the gold baseline, or a recorded run when one is given.

    Args:
        gold_directory: Directory holding the gold labellings.
        transcript_directory: Directory holding `<transcript name>.json` for
            each of them.
        decision_directory: Directory holding a recorded run per transcript, or
            `None` to run the gold baseline.

    Returns:
        The cases, in the order of the gold labellings.

    Raises:
        GoldLabellingError: If a labelling or a transcript is missing or
            unusable.
        DecisionRecordError: If a recorded run is missing or unusable, or does
            not hold one decision per chunk.
        SchemaValidationError: If a transcript is not a canonical session.
    """
    transcript_directory = Path(transcript_directory)
    decision_directory = None if decision_directory is None else Path(decision_directory)
    cases: list[EvaluationCase] = []
    for gold in load_gold_directory(gold_directory):
        transcript_path = transcript_directory / f"{gold.transcript}{DOCUMENT_SUFFIX}"
        if not transcript_path.is_file():
            raise GoldLabellingError(
                f"no transcript {transcript_path.name} in {transcript_directory} for the gold "
                f"labelling of {gold.transcript!r}"
            )
        session = load(Session, transcript_path)
        if decision_directory is None:
            adapter: IntentAdapter = FakeAdapter(gold_script(gold))
        else:
            decision_path = decision_directory / f"{gold.transcript}{DOCUMENT_SUFFIX}"
            if not decision_path.is_file():
                raise DecisionRecordError(
                    f"no recorded run {decision_path.name} in {decision_directory} for the gold "
                    f"labelling of {gold.transcript!r}"
                )
            decisions = load_decisions(decision_path)
            if len(decisions) != len(gold):
                raise DecisionRecordError(
                    f"{decision_path.name} records {len(decisions)} decision(s) for "
                    f"{gold.transcript!r}, which is labelled over {len(gold)} chunk(s)"
                )
            adapter = FakeAdapter(decisions)
        cases.append(EvaluationCase(gold=gold, session=session, adapter=adapter))
    return cases
