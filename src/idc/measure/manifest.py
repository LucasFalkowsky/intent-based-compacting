"""The run manifest: the one file that says what a run's numbers mean.

A results table of token counts is not evidence on its own — the same
transcript measured with a different tokenizer, compacted with a different
instruction or classified by a different adapter produces different numbers,
and nothing in the numbers says which. The manifest is the record that makes a
run attributable and repeatable: the digests of the file that went in and the
file that came out, the configuration (target agent, classifier adapter, model
ids, the run's options), the instruction's digest and version, the **tokenizer's
identity**, the metrics, the two instants of the run, and the output digest of
the run this one continued.

```python
from idc.measure import RunConfig, RunTimer, SystemClock, build_manifest, dump_manifest

timer = RunTimer(SystemClock())
timer.start()
...                                     # import, chunk, classify, resolve, compact
manifest = build_manifest(
    window=timer.finish(),
    input_path=source,
    output_path=target,
    config=RunConfig(agent=agent, adapter="s1m", models={"narrative": model_id}),
    tokenizer_name="o200k_base",
    metrics=metrics,
    previous=previous_manifest,         # None on a first run
)
dump_manifest(manifest, target.with_suffix(".manifest.json"))
```

The decisions this module makes, each of which a reader of the study may want
to check:

- **A digest is of a file's bytes.** `digest_file` is sha256 over the bytes,
  the same function `idc.compact.instruction_digest` is, so an input hash and
  an instruction hash mean the same thing and a test can compare them
  directly. Nothing normalises JSON before hashing: the artifact that was read
  is the artifact that is identified.
- **The tokenizer is named, not inferred.** `ContextCount` records the
  transcript's `model` — what a tokenizer is resolved *from* — and not the
  tokenizer that was injected, so a token count on its own is unattributable.
  `TokenizerIdentity` therefore carries two fields: the `family` the model was
  placed in, which this module computes with `model_family`, and a `name` the
  caller supplies for the tokenizer build itself (`FakeTokenizer`,
  `o200k_base`, a vocabulary's own version). Two fields rather than one joined
  identifier, because the two say different things: `family` is what this build
  resolves from the transcript, `name` is what the caller states was actually
  used, and the two **can** disagree — a `claude` session counted with another
  family's vocabulary is exactly the mistake a manifest should expose. Joined
  into one string, that disagreement would be unrepresentable rather than
  visible.
- **A predecessor is identified by its output digest.** `previous_output_sha256`
  is the sha256 of the file the previous run wrote — the file an incremental
  run is handed as its input — so two manifests chain by comparing this run's
  predecessor digest with the previous run's `output_file.sha256`, and
  `continues_previous` says whether this run's own input *is* that file. A
  digest of the previous manifest document would name the run but could not be
  checked against anything in this run's own data. A first run has no
  predecessor and records `null`; there is no sentinel digest, because a
  sentinel in a hash field would read as a hash.
- **The document is read back through this module only.** `loads_manifest` and
  `load_manifest` are the only way in, and they raise `ManifestError` — a
  `MeasureError`, catchable with the rest of the stage. Construction and
  assignment by a caller's own code raise a plain `ValueError` instead, which
  is the line `idc.measure.errors` draws; either way pydantic's
  `ValidationError` does not escape, so no pydantic traceback can reach the
  CLI.
- **The manifest is its own artifact.** It is not part of the canonical
  transcript schema (`src/idc/schema/`), which is the contract for
  transcripts; its schema is the model in this module and the loader that
  enforces it.
"""

import hashlib
import json
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Final, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidationError,
    model_validator,
)

from idc.compact import INSTRUCTION_VERSION, instruction_digest
from idc.measure.clock import RunWindow
from idc.measure.errors import ManifestError
from idc.measure.metrics import Metrics
from idc.measure.tokenizers import ModelFamily, model_family
from idc.schema import Agent

__all__ = [
    "MANIFEST_VERSION",
    "FileDigest",
    "InstructionIdentity",
    "MetricsRecord",
    "RunConfig",
    "RunManifest",
    "TokenizerIdentity",
    "build_manifest",
    "digest_bytes",
    "digest_file",
    "dump_manifest",
    "dumps_manifest",
    "load_manifest",
    "loads_manifest",
]

MANIFEST_VERSION: Final = 1
"""Version of the manifest document this build writes and reads.

A document declaring any other version is refused rather than read with the
fields this build happens to know: a manifest is study material, and silently
reading a newer one would attribute a run's numbers to a configuration the
document may describe differently.
"""

INDENT: Final = 2
"""Indentation of the dumped JSON; fixed so that two dumps are byte-comparable."""

SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"
"""What a digest field accepts: 64 lower-case hex characters, and nothing else."""

Sha256Hex = Annotated[str, Field(pattern=SHA256_PATTERN)]
"""A hex sha256 digest, as `digest_bytes` renders one."""


def _sorted_read_only(given: Mapping[str, str]) -> Mapping[str, str]:
    """Return `given` sorted by key and read-only."""
    return MappingProxyType(dict(sorted(given.items())))


SortedMapping = Annotated[
    Mapping[str, str],
    AfterValidator(_sorted_read_only),
    PlainSerializer(dict, return_type=dict[str, str]),
]
"""A string mapping that is sorted by key once and cannot be changed after.

Both halves are load-bearing, and `frozen=True` on the model provides neither:
it refuses an *attribute* assignment and does not touch the object the
attribute holds. Sorting at validation alone was not enough either —
`config.models["aaa"] = "z"` on a built manifest then succeeded and the next
`dumps_manifest` emitted `["classifier", "narrative", "aaa"]`, which is the
byte stability of the dump gone. The validator sorts, `MappingProxyType`
refuses the insertion, and the serializer renders the proxy as the JSON object
pydantic cannot serialize it as by itself — in the validator's order, so the
dumped keys are sorted too.
"""


def digest_bytes(data: bytes) -> str:
    """Return the hex sha256 of `data`.

    Args:
        data: The bytes to hash.

    Returns:
        64 lower-case hex characters.
    """
    return hashlib.sha256(data).hexdigest()


def _as_manifest_error(exc: ValidationError, source: str | None) -> ManifestError:
    """Translate a pydantic error into one located `ManifestError`."""
    problems = tuple(
        (
            ".".join(str(part) for part in error["loc"]) or "<document>",
            error["msg"].removeprefix("Value error, "),
        )
        for error in exc.errors()
    )
    count = f"{len(problems)} problem{'s' if len(problems) != 1 else ''}"
    return ManifestError(count, source=source, errors=problems)


class _ManifestModel(BaseModel):
    """Base of the manifest's data types: no unknown fields, no mutation.

    `frozen=True` is half of "no mutation": it refuses an attribute assignment
    and does nothing about the object an attribute holds, so a mapping field
    needs a read-only type of its own — see `SortedMapping`, which is what
    `RunConfig`'s two mappings are. Every other field here is a string, a
    number, a datetime, a tuple or another `_ManifestModel`.

    Invalid input raises a plain `ValueError` whichever way a caller's own code
    produces it — construction or assignment — and never pydantic's
    `ValidationError`. The load path is the other half: it raises
    `ManifestError`, because a document is input the stage was handed. See
    `idc.measure.errors` for the line between the two.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    def __init__(self, **data: Any) -> None:
        """Build the object, reporting invalid fields as a plain `ValueError`."""
        try:
            super().__init__(**data)
        except ValidationError as exc:
            raise ValueError(str(_as_manifest_error(exc, None))) from None

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse an assignment, reporting it as a plain `ValueError`."""
        try:
            super().__setattr__(name, value)
        except ValidationError as exc:
            raise ValueError(str(_as_manifest_error(exc, None))) from None


class FileDigest(_ManifestModel):
    """What identifies one file a run read or wrote.

    Attributes:
        path: The path as the run was given it, verbatim — a label, not an
            identity. A file that moved is still the same file to this record;
            the digest is what identifies it.
        sha256: Hex sha256 of the file's bytes.
        size_bytes: How many bytes were hashed, so an accidental truncation is
            visible without rehashing.
    """

    path: str = Field(min_length=1)
    sha256: Sha256Hex
    size_bytes: int = Field(ge=0)

    def __str__(self) -> str:
        """Return the digest as one diagnostic line."""
        return f"{self.path} ({self.size_bytes} byte(s), sha256 {self.sha256[:12]}…)"


def digest_file(path: str | Path) -> FileDigest:
    """Return the digest of a file the run was handed.

    Args:
        path: The file to hash; read as bytes, never decoded.

    Returns:
        Its path, the hex sha256 of its bytes and its size.

    Raises:
        ManifestError: If the file cannot be read. A `MeasureError`: the file
            is input the stage was handed, and "this file is not there" is a
            diagnostic for the CLI user rather than a traceback.
    """
    source = Path(path)
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise ManifestError(f"cannot hash {source}: {exc.strerror or exc}") from None
    return FileDigest(path=str(source), sha256=digest_bytes(data), size_bytes=len(data))


class InstructionIdentity(_ManifestModel):
    """Which compaction instruction a run used.

    The wording of the instruction decides what the narrative sections say, so
    a run's output is only reproducible against the exact prompt it used. The
    digest identifies those bytes; the version says which contract they follow.

    Attributes:
        version: The `version` the instruction file declares.
        digest: Hex sha256 of the instruction file's bytes, as
            `idc.compact.instruction_digest` returns it.
    """

    version: int = Field(ge=1)
    digest: Sha256Hex

    def __str__(self) -> str:
        """Return the identity as one diagnostic line."""
        return f"instruction v{self.version} (sha256 {self.digest[:12]}…)"


class TokenizerIdentity(_ManifestModel):
    """Which tokenizer a run's token counts were produced with.

    Attributes:
        name: The tokenizer build, as the caller names it — `FakeTokenizer` for
            the suite, a vocabulary name or version for a real one. Supplied
            rather than read off the object, because `Tokenizer` is a
            one-method protocol and a count carries no backend identity.
        family: The model family the transcript's model was placed in, computed
            with `model_family`. It says which tokenizer *should* have been
            used; `name` says which one was.
    """

    name: str = Field(min_length=1)
    family: ModelFamily

    def __str__(self) -> str:
        """Return the identity as one diagnostic line."""
        return f"{self.name} for the {self.family.value} family"


class RunConfig(_ManifestModel):
    """How a run was configured.

    Attributes:
        agent: The agent whose native tags the output was rendered in.
        adapter: The classifier adapter that produced the intent chains, by the
            name the run invoked it under.
        models: The model id per role the run asked a model for — the
            classifier's and the narrative writer's are different ids in
            general. A `SortedMapping`: sorted by key and read-only, so two
            runs with the same configuration dump the same bytes and no later
            insertion can change that.
        options: The rest of the run's switches as strings, for the same
            reason: whatever a later CLI ticket adds is recordable here without
            a schema change. A `SortedMapping` too.
    """

    agent: Agent
    adapter: str = Field(min_length=1)
    # `validate_default` so that an omitted mapping is the same read-only type
    # as a given one: a default is not validated otherwise, and `{}` would be a
    # writable dict on exactly the manifests that configured nothing.
    models: SortedMapping = Field(default_factory=dict, validate_default=True)
    options: SortedMapping = Field(default_factory=dict, validate_default=True)

    def __str__(self) -> str:
        """Return the configuration as one diagnostic line."""
        models = ", ".join(f"{role}={model}" for role, model in self.models.items()) or "no model"
        return f"{self.agent.value} via the {self.adapter} adapter ({models})"


class MetricsRecord(_ManifestModel):
    """The measured numbers of a run, flattened for the manifest.

    `Metrics` keeps the per-message breakdown and the per-file tallies, which
    are how a doubted number is traced back to the message that produced it.
    The manifest records the summary instead: the figures the study reports,
    plus the files that were edited by name. A reader who needs the breakdown
    measures the transcript again — the input digest says which transcript that
    is.

    Attributes:
        session_id: `Session.id` of the measured "before" transcript.
        tokens_before: Tokens of the visible context before compaction.
        tokens_after: Tokens of the visible context after it.
        saved_fraction: `(before - after) / before`; negative if the context
            grew, which is reported and not clamped.
        exploration_rate: `read_calls / total_calls` of the before session, or
            `null` when it made no tool call — see `Metrics.exploration_rate`.
        read_calls: Read-category calls of the before session.
        write_calls: Its write-category calls, failed ones included.
        exec_calls: Its exec-category calls.
        other_calls: Its calls of no other category.
        files_edited: Every distinct path an accepted write named, as the agent
            spelled it, in first-write order.
        lines_added: The added lines of those writes, summed; writes whose
            count the transcript does not state are left out of it.
    """

    session_id: str = Field(min_length=1)
    tokens_before: int = Field(ge=0)
    tokens_after: int = Field(ge=0)
    saved_fraction: float
    exploration_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    read_calls: int = Field(ge=0)
    write_calls: int = Field(ge=0)
    exec_calls: int = Field(ge=0)
    other_calls: int = Field(ge=0)
    files_edited: tuple[str, ...] = ()
    lines_added: int = Field(default=0, ge=0)

    def __str__(self) -> str:
        """Return the record as one diagnostic line."""
        return (
            f"{self.session_id}: {self.tokens_before} -> {self.tokens_after} tokens, "
            f"{self.saved_fraction * 100:.1f}% saved, {len(self.files_edited)} file(s)"
        )

    @classmethod
    def of(cls, metrics: Metrics) -> Self:
        """Return the record of one run's metrics.

        Args:
            metrics: The metrics `measure` produced.

        Returns:
            The flattened record, taking every value from `metrics` and
            computing none of its own.
        """
        return cls(
            session_id=metrics.tools.session_id,
            tokens_before=metrics.before.tokens,
            tokens_after=metrics.after.tokens,
            saved_fraction=metrics.saved_fraction,
            exploration_rate=metrics.exploration_rate,
            read_calls=metrics.tools.read_calls,
            write_calls=metrics.tools.write_calls,
            exec_calls=metrics.tools.exec_calls,
            other_calls=metrics.tools.other_calls,
            files_edited=metrics.tools.files_edited,
            lines_added=metrics.tools.lines_added,
        )


class RunManifest(_ManifestModel):
    """Everything that identifies one compaction run.

    Attributes:
        manifest_version: `MANIFEST_VERSION`; any other value is refused.
        started_at: When the run began, as an aware instant.
        finished_at: When it ended; never before `started_at`.
        input_file: Digest of the transcript the run read.
        output_file: Digest of the transcript it wrote.
        config: How it was configured.
        instruction: Which compaction instruction it used.
        tokenizer: Which tokenizer its token counts were produced with.
        metrics: What it measured.
        previous_output_sha256: `output_file.sha256` of the run this one
            continued, or `null` on a first run. See the module docstring for
            why the predecessor is identified by its output and not by a digest
            of its manifest.
    """

    manifest_version: int = Field(default=MANIFEST_VERSION, ge=1)
    started_at: AwareDatetime
    finished_at: AwareDatetime
    input_file: FileDigest
    output_file: FileDigest
    config: RunConfig
    instruction: InstructionIdentity
    tokenizer: TokenizerIdentity
    metrics: MetricsRecord
    previous_output_sha256: Sha256Hex | None = None

    @model_validator(mode="after")
    def _check_version_and_order(self) -> Self:
        """Refuse another build's version and a finish that precedes the start.

        A validator and not a check in `__init__`, because the load path goes
        through `model_validate` and would skip the latter: both invariants
        have to hold for a document that was read as much as for one that was
        built.

        Returns:
            The manifest itself, unchanged.

        Raises:
            ValueError: If the version is not `MANIFEST_VERSION`, or if
                `finished_at` is before `started_at`. Reported as a
                `ManifestError` on the load path and as a plain `ValueError` on
                construction; see `_ManifestModel`.
        """
        if self.manifest_version != MANIFEST_VERSION:
            raise ValueError(
                f"this build reads manifest version {MANIFEST_VERSION}, the document declares "
                f"{self.manifest_version}"
            )
        if self.finished_at < self.started_at:
            raise ValueError(
                f"a run cannot finish before it started; {self.finished_at.isoformat()} is "
                f"before {self.started_at.isoformat()}"
            )
        return self

    def __str__(self) -> str:
        """Return the manifest as one diagnostic line."""
        chained = (
            "first run"
            if self.previous_output_sha256 is None
            else f"after {self.previous_output_sha256[:12]}…"
        )
        return f"{self.input_file.path} -> {self.output_file.path} ({chained}): {self.metrics}"

    @property
    def duration(self) -> timedelta:
        """Return how long the run took; never negative."""
        return self.finished_at - self.started_at

    @property
    def continues_previous(self) -> bool:
        """Return whether this run's input has exactly the predecessor's bytes.

        **This is `False` for every successful run of this tool, and the
        docstring used to claim the opposite** — that an equal pair is "the
        chain a second run is supposed to form". A second run'''s input is the
        agent'''s *continued* transcript: it begins from what the previous run
        wrote and has grown by the turns that came after, so it is never byte-
        identical to it. Handing the tool the previous output unchanged does
        not produce the `True` case either, because such a transcript is a
        single chunk holding the summary record and `idc.render` refuses to
        write a second record under `SUMMARY_MESSAGE_ID`. Both halves are
        pinned by `tests/test_cli_second_run.py`, which is where the false
        claim was found.

        What to read instead is `previous_output_sha256`, which records *which*
        run this one continues whether or not any predicate holds. Whether this
        comparison should be redefined — the predecessor'''s summary block
        occurring in this run'''s input is the question a reader actually has —
        or dropped, is an open decision and has a ticket of its own (#112).

        Returns:
            `True` when a predecessor is recorded and this run'''s input has
            exactly its bytes. `False` on a first run, and `False` when a
            predecessor is recorded whose output is not what this run read:
            the hash stays in the document either way, so the recorded
            predecessor is visible rather than erased.
        """
        return (
            self.previous_output_sha256 is not None
            and self.previous_output_sha256 == self.input_file.sha256
        )


def build_manifest(
    *,
    window: RunWindow,
    input_path: str | Path,
    output_path: str | Path,
    config: RunConfig,
    tokenizer_name: str,
    metrics: Metrics,
    instruction_path: str | Path | None = None,
    previous: RunManifest | None = None,
) -> RunManifest:
    """Record one run, hashing the two files it was handed.

    No git call and no agent call: the digests are of the bytes of
    `input_path` and `output_path`, which the caller already read and wrote.

    Args:
        window: The run's two instants, taken from an injected `Clock`.
        input_path: The transcript the run read.
        output_path: The transcript it wrote.
        config: How the run was configured.
        tokenizer_name: What to record as the tokenizer's identity; its family
            is computed from the measured transcript's model.
        metrics: What `measure` reported for the run.
        instruction_path: The instruction file to digest; the packaged one when
            omitted, which is what a run uses.
        previous: The manifest of the run this one continued, or `None` on a
            first run. Only its `output_file.sha256` is recorded.

    Returns:
        The manifest, ready to be dumped.

    Raises:
        ManifestError: If either file cannot be read.
        InstructionError: If the instruction file cannot be read.
        UnknownModelError: If the measured transcript's model belongs to no
            known family, so the tokenizer's family cannot be named. A
            `MeasureError`: a number counted for an unplaceable model is not
            attributable, which is what this manifest exists to prevent.
    """
    return RunManifest(
        manifest_version=MANIFEST_VERSION,
        started_at=window.started_at,
        finished_at=window.finished_at,
        input_file=digest_file(input_path),
        output_file=digest_file(output_path),
        config=config,
        instruction=InstructionIdentity(
            version=INSTRUCTION_VERSION, digest=instruction_digest(instruction_path)
        ),
        tokenizer=TokenizerIdentity(name=tokenizer_name, family=model_family(metrics.before.model)),
        metrics=MetricsRecord.of(metrics),
        previous_output_sha256=None if previous is None else previous.output_file.sha256,
    )


def dumps_manifest(manifest: RunManifest) -> str:
    """Serialize a manifest to JSON text.

    Args:
        manifest: The manifest to dump.

    Returns:
        Indented JSON without a trailing newline. The same manifest always
        dumps the same bytes — the field order is the model's, and
        `RunConfig`'s two mappings are `SortedMapping`s, sorted by key at
        validation and read-only afterwards — so two runs' manifests are
        diffable.
    """
    return manifest.model_dump_json(indent=INDENT)


def dump_manifest(manifest: RunManifest, path: str | Path) -> Path:
    """Write a manifest to a file as JSON.

    Args:
        manifest: The manifest to dump.
        path: Destination; written as UTF-8 with a trailing newline.

    Returns:
        The path written to.
    """
    destination = Path(path)
    destination.write_text(dumps_manifest(manifest) + "\n", encoding="utf-8")
    return destination


def loads_manifest(data: str | bytes, *, source: str | None = None) -> RunManifest:
    """Parse and validate a manifest given as JSON text.

    The only way into a manifest document, together with `load_manifest`:
    nothing outside this module calls pydantic's `model_validate`, so a
    malformed manifest cannot put a pydantic traceback on the CLI.

    Args:
        data: The JSON document as text or UTF-8 bytes.
        source: Where it came from, for the error message.

    Returns:
        The validated manifest.

    Raises:
        ManifestError: If the text is not JSON, or the document does not
            satisfy the manifest's shape — a wrong version, a digest that is
            not 64 hex characters, a naive instant, an unknown field and a
            missing one all report here, one located problem per line.
    """
    try:
        document = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ManifestError(
            f"not valid JSON ({exc.msg} at line {exc.lineno}, column {exc.colno})", source=source
        ) from None
    try:
        return RunManifest.model_validate(document)
    except ValidationError as exc:
        raise _as_manifest_error(exc, source) from None
    except ValueError as exc:
        raise ManifestError(str(exc), source=source) from None


def load_manifest(path: str | Path) -> RunManifest:
    """Read and validate a manifest from a file.

    Args:
        path: Path of the JSON document, read as UTF-8.

    Returns:
        The validated manifest.

    Raises:
        ManifestError: If the file cannot be read, is not JSON, or does not
            satisfy the manifest's shape. The message names the file.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"cannot be read ({exc.strerror or exc})", source=str(source)) from None
    except UnicodeDecodeError:
        raise ManifestError("not valid UTF-8", source=str(source)) from None
    return loads_manifest(text, source=str(source))
