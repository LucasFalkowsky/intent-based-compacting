"""What one stage hands the next on disk, when the stages are separate commands.

The pipeline is one command per stage with files in between, and only three of
the things a stage produces are canonical documents on their own: a `Session`, a
`Chunk` and a `Chain`. A chunking is a *list* of chunks, a classification is a
list of chains plus the session they range over, and `idc.schema` has no type
for either — adding one would be a change to the canonical schema, which
`CLAUDE.md` makes a ticket of its own.

So the intermediate file is an envelope this package defines, holding one
canonical document per part:

```json
{
  "session": { ... },          // always
  "chunks":  [ { ... } ],      // written by `chunk`
  "chains":  [ { ... } ],      // written by `classify`
  "block":   { ... }           // written by `compact`
}
```

Each part is validated through `idc.schema.validate_document`, never through
pydantic's `model_validate`, so a hand-edited envelope fails as one
`SchemaValidationError` naming the type and the field path. The envelope itself
— the four keys and their shapes — is this package's, so a problem with it is an
`ArtifactError`.

A **bare `Session` document is accepted too**, and read as an envelope with no
chunks. That is not a convenience: `idc import` writes a plain canonical session
(the format `README.md` documents and `idc.schema.load` reads), and `idc chunk`
has to be able to read exactly that file. The two cases are told apart by the
`session` key, which a `Session` cannot have — it forbids unknown fields — so
the discrimination is decidable and not a guess.

What a stage then *needs* is asked for explicitly: `require_chunks`,
`require_chains` and `require_block` each name the command that would have
produced the missing part. A stage that went ahead without them would write an
empty summary with a green exit status, which is the failure mode this module
exists to turn into a diagnostic.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from idc.cli.errors import ArtifactError
from idc.schema import Chain, Chunk, CompactedBlock, Session, dumps, validate_document

__all__ = [
    "ARTIFACT_KEYS",
    "BLOCK_KEY",
    "CHAINS_KEY",
    "CHUNKS_KEY",
    "SESSION_KEY",
    "Artifact",
    "dump_artifact",
    "dumps_artifact",
    "load_artifact",
    "loads_artifact",
]

SESSION_KEY = "session"
"""Envelope key holding the canonical `Session`; also what identifies an envelope."""

CHUNKS_KEY = "chunks"
"""Envelope key holding the `Chunk` list `idc chunk` wrote."""

CHAINS_KEY = "chains"
"""Envelope key holding the `Chain` list `idc classify` wrote."""

BLOCK_KEY = "block"
"""Envelope key holding the `CompactedBlock` `idc compact` wrote."""

ARTIFACT_KEYS = (SESSION_KEY, CHUNKS_KEY, CHAINS_KEY, BLOCK_KEY)
"""Every key an envelope may hold, in pipeline order. Anything else is refused."""

INDENT = 2
"""Indentation of a written envelope, so a diff between two runs is readable."""


@dataclass(frozen=True, slots=True)
class Artifact:
    """One pipeline file, as the stage that reads it sees it.

    Attributes:
        session: The transcript every other part ranges over. After `idc chunk`
            with reduction on, this is the *reduced* session and not the one
            `idc import` wrote — the chunks' ranges are indices into it, so the
            two travel together or neither is readable.
        chunks: The session's chunks, in transcript order; empty before
            `idc chunk`.
        chains: The intent chains, in the order the classifier opened them;
            empty before `idc classify`.
        block: The compacted block; `None` before `idc compact`.
        source: The file this was read from, for an error message; `None` when
            it was built in memory.
    """

    session: Session
    chunks: tuple[Chunk, ...] = ()
    chains: tuple[Chain, ...] = ()
    block: CompactedBlock | None = None
    source: str | None = None

    def require_chunks(self) -> tuple[Chunk, ...]:
        """Return the chunks, refusing an artifact that holds none.

        Returns:
            The chunks, in transcript order.

        Raises:
            ArtifactError: If there are none. Both the empty list and the
                absent key report here: a stage asked to work on no chunk has
                nothing to summarise, and the only honest answer is to say
                which command was skipped.
        """
        if not self.chunks:
            raise ArtifactError("holds no chunks; run `idc chunk` on it first", source=self.source)
        return self.chunks

    def require_chains(self) -> tuple[Chain, ...]:
        """Return the chains, refusing an artifact that holds none.

        Returns:
            The chains, in the order they were opened.

        Raises:
            ArtifactError: If there are none; the message names `idc classify`.
        """
        if not self.chains:
            raise ArtifactError(
                "holds no chains; run `idc classify` on it first", source=self.source
            )
        return self.chains

    def require_block(self) -> CompactedBlock:
        """Return the compacted block, refusing an artifact that holds none.

        Returns:
            The block, whose `rendered_for` is the agent the output is written
            for.

        Raises:
            ArtifactError: If there is none; the message names `idc compact`.
        """
        if self.block is None:
            raise ArtifactError(
                "holds no compacted block; run `idc compact` on it first", source=self.source
            )
        return self.block


def _document_of(obj: Session | Chunk | Chain | CompactedBlock) -> Any:
    """Return a canonical object as parsed JSON data, serialized by the package.

    Args:
        obj: The canonical object.

    Returns:
        Its JSON representation as Python data, so that several objects can be
        put into one envelope. Serialized with `idc.schema.dumps` rather than
        by reaching into pydantic, so the field order and the enum spellings
        are the ones the canonical documents use everywhere else.
    """
    return json.loads(dumps(obj))


def dumps_artifact(artifact: Artifact) -> str:
    """Serialize a pipeline artifact to JSON text.

    Args:
        artifact: The artifact to write.

    Returns:
        The envelope as indented JSON without a trailing newline. A part that
        is empty is left out, so the file says which stages have run.
    """
    document: dict[str, Any] = {SESSION_KEY: _document_of(artifact.session)}
    if artifact.chunks:
        document[CHUNKS_KEY] = [_document_of(chunk) for chunk in artifact.chunks]
    if artifact.chains:
        document[CHAINS_KEY] = [_document_of(chain) for chain in artifact.chains]
    if artifact.block is not None:
        document[BLOCK_KEY] = _document_of(artifact.block)
    return json.dumps(document, indent=INDENT, ensure_ascii=False)


def dump_artifact(artifact: Artifact, path: str | Path) -> Path:
    """Write a pipeline artifact to a file as JSON.

    Args:
        artifact: The artifact to write.
        path: Destination; written as UTF-8 with a trailing newline.

    Returns:
        The path written to.
    """
    destination = Path(path)
    destination.write_text(dumps_artifact(artifact) + "\n", encoding="utf-8")
    return destination


def _sequence(document: dict[str, Any], key: str, source: str | None) -> Sequence[Any]:
    """Return one list-valued key of an envelope, refusing anything else.

    Args:
        document: The envelope.
        key: The key to read.
        source: Where the envelope came from.

    Returns:
        The list, empty when the key is absent.

    Raises:
        ArtifactError: If the key holds something that is not a list. Checked
            rather than iterated: iterating a mapping would yield its keys and
            report a schema error about a string, which names the wrong
            problem.
    """
    value = document.get(key, [])
    if not isinstance(value, list):
        raise ArtifactError(
            f"{key!r} holds {type(value).__name__}, not a list of documents", source=source
        )
    return value


def loads_artifact(data: str | bytes, *, source: str | None = None) -> Artifact:
    """Parse and validate a pipeline artifact given as JSON text.

    Accepts both shapes the pipeline writes: the envelope, and a bare canonical
    `Session` document as `idc import` writes it. See the module docstring for
    why both are read here and how they are told apart.

    Args:
        data: The JSON document as text or UTF-8 bytes.
        source: Where it came from, for the error messages.

    Returns:
        The artifact, with every part validated.

    Raises:
        ArtifactError: If the text is not JSON, is not a JSON object, or holds
            a key the envelope does not define.
        SchemaValidationError: If one of its parts does not satisfy the
            canonical schema.
    """
    try:
        document = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ArtifactError(
            f"not valid JSON ({exc.msg} at line {exc.lineno}, column {exc.colno})", source=source
        ) from None
    if not isinstance(document, dict):
        raise ArtifactError(
            f"is a JSON {type(document).__name__}, not an object", source=source
        ) from None
    if SESSION_KEY not in document:
        return Artifact(session=validate_document(Session, document, source=source), source=source)
    unknown = sorted(set(document) - set(ARTIFACT_KEYS))
    if unknown:
        known = ", ".join(ARTIFACT_KEYS)
        raise ArtifactError(
            f"holds unknown key(s) {', '.join(repr(key) for key in unknown)}; known keys: {known}",
            source=source,
        )
    block = document.get(BLOCK_KEY)
    return Artifact(
        session=validate_document(Session, document[SESSION_KEY], source=source),
        chunks=tuple(
            validate_document(Chunk, item, source=source)
            for item in _sequence(document, CHUNKS_KEY, source)
        ),
        chains=tuple(
            validate_document(Chain, item, source=source)
            for item in _sequence(document, CHAINS_KEY, source)
        ),
        block=None if block is None else validate_document(CompactedBlock, block, source=source),
        source=source,
    )


def load_artifact(path: str | Path) -> Artifact:
    """Read and validate a pipeline artifact from a file.

    Args:
        path: Path of the JSON document, read as UTF-8.

    Returns:
        The artifact, with every part validated.

    Raises:
        ArtifactError: If the file cannot be read, is not UTF-8, is not JSON,
            or holds a key the envelope does not define. The message names the
            file.
        SchemaValidationError: If one of its parts does not satisfy the
            canonical schema.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"cannot be read ({exc.strerror or exc})", source=str(source)) from None
    except UnicodeDecodeError:
        raise ArtifactError("is not valid UTF-8", source=str(source)) from None
    return loads_artifact(text, source=str(source))
