"""Gold chain labels: the hand-written reference a classifier run is scored against.

A gold labelling says, for one transcript, which intent chain each of its chunks
belongs to. It is **read**, never derived: the evaluation harness
(`idc.classify.evaluate`) scores a run against it, so a labelling this package
could compute would make the harness score itself.

```python
from idc.classify import gold_script, load_gold

gold = load_gold("tests/fixtures/gold/interleaved_intents.json")
gold.as_mapping()          # {"K1": "C1", "K2": "C2", "K3": "C1"}
gold_script(gold)          # (NEW, NEW, "C1") — the baseline, see below
```

The document is a JSON object with three required keys and one optional one:

```json
{
  "transcript": "interleaved_intents",
  "chunking": "user-boundaries",
  "note": "why the labelling is what it is",
  "chains": {"K1": "C1", "K2": "C2", "K3": "C1"}
}
```

Four things about that format are decisions, not incidentals:

- **`transcript` has to equal the file name**, so a labelling cannot be scored
  against a transcript it was not written for after a rename.
- **`chunking` names the chunking the chunk ids belong to**, because a chunk id
  is a position (`K<n>`) and the same transcript yields different chunks under
  `chunk_session` and under `chunk_with_subagents`. The harness refuses a
  chunking it cannot perform instead of scoring against the wrong chunks.
- **The chains are numbered `C1…Ck` in the order they open**, which is the
  numbering `idc.classify.ids` itself assigns. Per-chunk accuracy compares chain
  *ids*, so a labelling that named its chains anything else would be compared
  against a different naming scheme and the number would mean nothing. The
  constraint is checked, not assumed.
- **A repeated chunk id is refused.** `json.loads` keeps the last of two equal
  keys, so a hand-written document holding `"K1"` twice would silently lose a
  label it visibly carries.

Every refusal is a `GoldLabellingError` and therefore a `ClassifyError`: a gold
document is input read from a file, like a summary block parsed back in (#33),
and a hand-edited one must not reach a caller as a traceback.

`gold_script` is the one place where labels turn into decisions: it returns the
adapter script a *perfect* backend would have produced — `NEW` the first time a
chain appears, the chain's id afterwards. It exists so that the harness can be
shown to score a perfect run 1.0 (and as the baseline of its command), and it
never runs in the other direction: no function here reads a transcript, a
`Classification` or a model.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from idc.chunk import CHUNK_ID_PREFIX, chunk_id
from idc.classify.adapters import NEW, DecisionLike
from idc.classify.errors import GoldLabellingError
from idc.classify.ids import require_chain_id

GOLD_CHUNKING = "user-boundaries"
"""The chunking the labels of the shared corpus are written against.

`idc.chunk.chunk_session`: one chunk per user turn. The name is data — it is
what a gold document declares — and `idc.classify.evaluate` maps it to the
function that performs it, so a labelling for another chunking is refused there
rather than scored against the wrong chunks.
"""

GOLD_SUFFIX = ".json"
"""One gold labelling per `<transcript name>.json` file."""

REQUIRED_KEYS = ("transcript", "chunking", "chains")
"""The keys a gold document has to carry."""

OPTIONAL_KEYS = ("note",)
"""The keys it may carry: the human reason for the labelling."""


def _chunk_position(value: object) -> int | None:
    """Return the position a chunk id encodes, or `None` if it is not one.

    The id is checked by rebuilding it with `idc.chunk.chunk_id`, so this holds
    no second copy of the `K<n>` form and cannot drift from the chunker's.

    Args:
        value: Anything; a non-string is not a chunk id.

    Returns:
        The number `n` of `K<n>`, or `None`.
    """
    if not isinstance(value, str) or not value.startswith(CHUNK_ID_PREFIX):
        return None
    digits = value[len(CHUNK_ID_PREFIX) :]
    if not digits.isascii() or not digits.isdigit():
        return None
    position = int(digits)
    if position < 1 or chunk_id(position) != value:
        return None
    return position


@dataclass(frozen=True, slots=True)
class GoldChains:
    """One transcript's gold chain labels, in transcript order.

    Attributes:
        transcript: Name of the transcript the labels belong to — the fixture
            name, and the file name of the document they were read from.
        labels: One `(chunk id, chain id)` pair per chunk, in transcript order.
            A tuple rather than a mapping, because the order is part of the
            labelling: it is what `gold_script` plays and what the chain
            numbering is checked against.
        chunking: Which chunking the chunk ids are those of; see
            `GOLD_CHUNKING`. Required, and deliberately without a default: a
            chunk id is a *position*, so a labelling whose chunking nobody
            stated is not a reference at all, and a document cannot leave it
            out either (`REQUIRED_KEYS`).
        note: Why the labelling is what it is, in the labeller's own words.
            Empty when the document carries none.
    """

    transcript: str
    labels: tuple[tuple[str, str], ...]
    chunking: str
    note: str = ""

    def __post_init__(self) -> None:
        """Reject everything that would make the labelling unusable as a reference.

        Raises:
            GoldLabellingError: If the transcript name or the chunking holds no
                word, if the labelling covers no chunk, if a chunk id is not of
                the form `K<n>` or the ids are not in transcript order, if a
                chain id is not of the form `C<n>`, or if the chains are not
                numbered `C1…Ck` in the order they first appear.
        """
        if not self.transcript.strip():
            raise GoldLabellingError("a gold labelling must name the transcript it belongs to")
        if not self.chunking.strip():
            raise GoldLabellingError(
                f"the gold labelling of {self.transcript!r} must name the chunking its chunk "
                f"ids belong to"
            )
        if not self.labels:
            raise GoldLabellingError(
                f"the gold labelling of {self.transcript!r} covers at least one chunk; it is "
                f"the reference a run is scored against"
            )
        previous_position = 0
        opened: list[str] = []
        for chunk, chain in self.labels:
            position = _chunk_position(chunk)
            if position is None:
                raise GoldLabellingError(
                    f"{self.transcript!r} labels {chunk!r}, which is not a chunk id; a chunk "
                    f"id is of the form K<n> with n counted from 1"
                )
            if position <= previous_position:
                raise GoldLabellingError(
                    f"{self.transcript!r} labels {chunk!r} behind chunk {previous_position}; "
                    f"a gold labelling lists its chunks in transcript order"
                )
            previous_position = position
            require_chain_id(
                chain,
                what=f"the chain {self.transcript!r} labels {chunk!r} with",
                error=GoldLabellingError,
            )
            if chain not in opened:
                opened.append(chain)
                expected = f"C{len(opened)}"
                if chain != expected:
                    raise GoldLabellingError(
                        f"{self.transcript!r} opens chain {chain!r} as its chain number "
                        f"{len(opened)}, which the classifier would number {expected!r}; a "
                        f"gold labelling numbers its chains in opening order"
                    )

    @classmethod
    def from_mapping(
        cls,
        transcript: str,
        labels: Mapping[str, str] | Iterable[tuple[str, str]],
        *,
        chunking: str,
        note: str = "",
    ) -> Self:
        """Return the labelling for a `chunk id -> chain id` mapping.

        Args:
            transcript: Name of the transcript the labels belong to.
            labels: The labels, in transcript order. A mapping is read in its
                own order, which for a document parsed from JSON is the order
                the keys were written in.
            chunking: Which chunking the chunk ids are those of; required,
                like on the dataclass itself.
            note: Why the labelling is what it is.

        Returns:
            The validated labelling.

        Raises:
            GoldLabellingError: If the labelling is not usable as a reference;
                see `__post_init__`.
        """
        pairs = labels.items() if isinstance(labels, Mapping) else labels
        return cls(
            transcript=transcript,
            labels=tuple((chunk, chain) for chunk, chain in pairs),
            chunking=chunking,
            note=note,
        )

    def __len__(self) -> int:
        """Return how many chunks the labelling covers."""
        return len(self.labels)

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """Return the labelled chunk ids, in transcript order."""
        return tuple(chunk for chunk, _ in self.labels)

    @property
    def chain_ids(self) -> tuple[str, ...]:
        """Return the chain ids, each once, in the order they open."""
        opened: list[str] = []
        for _, chain in self.labels:
            if chain not in opened:
                opened.append(chain)
        return tuple(opened)

    @property
    def chain_count(self) -> int:
        """Return how many chains the labelling holds."""
        return len(self.chain_ids)

    def as_mapping(self) -> dict[str, str]:
        """Return the labels as a `chunk id -> chain id` mapping, in order."""
        return dict(self.labels)


def gold_script(gold: GoldChains) -> tuple[DecisionLike, ...]:
    """Return the adapter script a perfect backend would have produced.

    The baseline of the harness: `NEW` the first time a chain appears, that
    chain's id for every later chunk of it. Played through
    `idc.classify.FakeAdapter` it reproduces the gold chains exactly, which is
    what shows that a perfect run scores 1.0.

    Args:
        gold: The labelling to read.

    Returns:
        One decision per labelled chunk, in transcript order.
    """
    script: list[DecisionLike] = []
    opened: set[str] = set()
    for _, chain in gold.labels:
        if chain in opened:
            script.append(chain)
        else:
            opened.add(chain)
            script.append(NEW)
    return tuple(script)


def _no_duplicate_keys(pairs: Sequence[tuple[str, object]], source: str) -> dict[str, object]:
    """Return the pairs as a dict, refusing a key that appears twice.

    Args:
        pairs: The key/value pairs of one JSON object, in document order.
        source: How the document is named in the error message.

    Returns:
        The object as a dict.

    Raises:
        GoldLabellingError: If a key appears twice — `json.loads` would keep the
            last one and the document would lose a label it visibly carries.
    """
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise GoldLabellingError(f"{source} holds the key {key!r} twice")
        document[key] = value
    return document


def read_json_object(
    path: Path, required: Sequence[str], optional: Sequence[str] = ()
) -> dict[str, object]:
    """Return the JSON object a document holds, with its keys checked.

    The reader both hand-written artifacts of the harness go through — the gold
    labellings here and the recorded runs of `idc.classify.evaluate` — so that
    the two refuse the same mistakes with the same message. It always raises
    `GoldLabellingError`; the recorded-run side re-raises that as a
    `DecisionRecordError`, because there the document is a run and not a
    labelling.

    Args:
        path: Path of the document.
        required: Keys it has to carry.
        optional: Keys it may carry.

    Returns:
        The object, with its keys in document order.

    Raises:
        GoldLabellingError: If the file cannot be read, is not JSON, is not a
            JSON object, repeats a key, or carries an unknown or missing one.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise GoldLabellingError(f"cannot read the gold labelling {path}: {error}") from error
    try:
        parsed = json.loads(
            text, object_pairs_hook=lambda pairs: _no_duplicate_keys(pairs, f"{path.name}")
        )
    except json.JSONDecodeError as error:
        raise GoldLabellingError(f"{path.name} is not a JSON document: {error}") from error
    if not isinstance(parsed, dict):
        raise GoldLabellingError(
            f"{path.name} must hold a JSON object with the keys "
            f"{', '.join(required)}; got {type(parsed).__name__}"
        )
    missing = [key for key in required if key not in parsed]
    if missing:
        raise GoldLabellingError(f"{path.name} is missing the key(s) {', '.join(missing)}")
    allowed = tuple(required) + tuple(optional)
    unknown = [key for key in parsed if key not in allowed]
    if unknown:
        raise GoldLabellingError(
            f"{path.name} holds the unknown key(s) {', '.join(unknown)}; "
            f"allowed: {', '.join(allowed)}"
        )
    return parsed


def load_gold(path: str | Path) -> GoldChains:
    """Return the gold labelling one document holds.

    Args:
        path: Path of a `<transcript name>.json` gold document.

    Returns:
        The validated labelling.

    Raises:
        GoldLabellingError: If the document cannot be read, does not have the
            documented shape, names a transcript other than its own file, or is
            not a usable labelling (see `GoldChains.__post_init__`).
    """
    path = Path(path)
    document = read_json_object(path, REQUIRED_KEYS, OPTIONAL_KEYS)
    transcript = document["transcript"]
    if transcript != path.stem:
        raise GoldLabellingError(
            f"{path.name} names the transcript {transcript!r}; a gold labelling is named after "
            f"the transcript it labels, so this document would have to be {transcript!r}.json"
        )
    chains = document["chains"]
    if not isinstance(chains, dict):
        raise GoldLabellingError(
            f"{path.name} must map every chunk id to a chain id under 'chains'; got "
            f"{type(chains).__name__}"
        )
    for chunk, chain in chains.items():
        if not isinstance(chain, str):
            raise GoldLabellingError(
                f"{path.name} labels {chunk!r} with {chain!r}, which is not a chain id"
            )
    note = document.get("note", "")
    chunking = document["chunking"]
    if not isinstance(chunking, str) or not isinstance(note, str):
        raise GoldLabellingError(f"{path.name} must carry 'chunking' and 'note' as strings")
    return GoldChains.from_mapping(str(transcript), dict(chains), chunking=chunking, note=note)


def load_gold_directory(directory: str | Path) -> tuple[GoldChains, ...]:
    """Return every gold labelling of a directory, sorted by transcript name.

    Args:
        directory: Directory holding one `<transcript name>.json` per labelled
            transcript — `tests/fixtures/gold/` for the shared corpus.

    Returns:
        The labellings, in the order of their file names.

    Raises:
        GoldLabellingError: If the directory does not exist, holds no gold
            document, or holds one that is not a usable labelling.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise GoldLabellingError(f"{directory} is not a directory of gold labellings")
    paths = sorted(directory.glob(f"*{GOLD_SUFFIX}"))
    if not paths:
        raise GoldLabellingError(f"no gold labelling (*{GOLD_SUFFIX}) in {directory}")
    return tuple(load_gold(path) for path in paths)
