"""The narrative instruction: one fixed prompt, shipped as a versioned data file.

Every chain is compacted with the **same** instruction, and the study has to be
able to say which one that was for a given run. That is the whole reason this
module exists instead of a string literal in `idc.compact.narrative`:

- The text lives in `narrative_instruction.toml` next to this module — package
  data, loaded at runtime, the way `idc.importers.tool_registry` loads
  `tool_registry.toml`. It is inside `src/idc/`, so the wheel ships it: the
  build includes the package directory (`[tool.hatch.build.targets.wheel]`),
  so a built wheel holds `idc/compact/narrative_instruction.toml` beside
  `idc/importers/tool_registry.toml` — read off the wheel, not assumed.
- It declares a `version`, and `INSTRUCTION_VERSION` is the only one this build
  reads. A text and a build cannot silently disagree about what was asked.
- `instruction_digest` is the sha256 of the file's **bytes** — the value the run
  manifest (T9.3, #39) records. `idc.measure.manifest.build_manifest` is its
  one consumer: it stores the digest next to `INSTRUCTION_VERSION`, which is
  what ties a run's narrative sections to the exact prompt that produced them.

```python
from idc.compact import instruction, instruction_digest

instruction().text        # the prompt, as the file holds it
instruction_digest()      # 64 hex characters, stable for the same bytes
```

Nothing here calls a model, and nothing here asks a clock or the network: an
instruction is read from one file and nothing else.
"""

import hashlib
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from idc.compact.errors import InstructionError

__all__ = [
    "INSTRUCTION_PATH",
    "INSTRUCTION_VERSION",
    "Instruction",
    "instruction",
    "instruction_digest",
    "load_instruction",
]

INSTRUCTION_PATH = Path(__file__).with_name("narrative_instruction.toml")
"""The packaged instruction data file — the default source of the instruction."""

INSTRUCTION_VERSION = 1
"""The only `version` this build of the tool reads.

A build reads one instruction, so that the blocks of one run were all written
with the same prompt. A file declaring anything else is refused rather than
read, because a run that silently used another instruction cannot be compared
with the runs of the study.
"""

_VERSION_KEY = "version"
"""Key of the instruction's version in the data file."""

_TEXT_KEY = "instruction"
"""Key of the instruction's text in the data file."""


@dataclass(frozen=True, slots=True)
class Instruction:
    """One instruction as it was read: its text, its version and its digest.

    A value object rather than a bare string, because what identifies an
    instruction is not its text alone: the manifest records the digest, the
    build checks the version, and a report names the file. Carrying the three
    together means a caller cannot pass the text on and lose which instruction
    it was.

    Attributes:
        version: The `version` the file declares — `INSTRUCTION_VERSION` for
            every instruction `load_instruction` returns.
        text: The prompt as the file holds it, with the whitespace around it
            stripped — TOML has already dropped the newline after the opening
            `'''`, and the one before the closing `'''` goes the same way.
        digest: Hex sha256 of the file's bytes; see `instruction_digest`.
        path: The file it was read from.
    """

    version: int
    text: str
    digest: str
    path: Path


def _digest_of(data: bytes) -> str:
    """Return the hex sha256 of `data`."""
    return hashlib.sha256(data).hexdigest()


def _read(source: Path) -> bytes:
    """Return the bytes of an instruction file, or raise `InstructionError`.

    Args:
        source: The file to read.

    Returns:
        Its bytes, unchanged — the digest is taken over exactly these.

    Raises:
        InstructionError: If the file cannot be read. The message names it, so
            a file missing from a built wheel is readable from the error.
    """
    try:
        return source.read_bytes()
    except OSError as exc:
        raise InstructionError(
            f"cannot read the narrative instruction {source}: {exc.strerror or exc}"
        ) from None


def _document(data: bytes, source: Path) -> dict[str, object]:
    """Return an instruction file's TOML document, or raise `InstructionError`.

    Args:
        data: The file's bytes.
        source: The file they were read from, for the message.

    Returns:
        The parsed document.

    Raises:
        InstructionError: If the bytes are not UTF-8 or not valid TOML.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise InstructionError(f"invalid narrative instruction {source}: not valid UTF-8") from None
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise InstructionError(
            f"invalid narrative instruction {source}: not valid TOML ({exc})"
        ) from None


def load_instruction(path: str | Path | None = None) -> Instruction:
    """Read an instruction data file, validate it, and take its digest.

    Args:
        path: The data file; the packaged `INSTRUCTION_PATH` when omitted. A
            path of its own is what a test uses to pin that the text really
            comes off the disk, and what a future study run would use to
            compare two instructions.

    Returns:
        The instruction, carrying the digest of the bytes that were read.

    Raises:
        InstructionError: If the file cannot be read, is not UTF-8, is not
            valid TOML, declares a `version` other than `INSTRUCTION_VERSION`,
            or holds an `instruction` that is absent, not a string, or without a
            word in it. The message names the file in every case.
    """
    source = INSTRUCTION_PATH if path is None else Path(path)
    data = _read(source)
    document = _document(data, source)
    version = document.get(_VERSION_KEY)
    if version != INSTRUCTION_VERSION:
        raise InstructionError(
            f"invalid narrative instruction {source}: this build reads "
            f"{_VERSION_KEY} = {INSTRUCTION_VERSION}, the file declares {version!r}"
        )
    text = document.get(_TEXT_KEY)
    if not isinstance(text, str) or not text.strip():
        raise InstructionError(
            f"invalid narrative instruction {source}: {_TEXT_KEY!r} must be a string "
            f"that holds a word; got {text!r}"
        )
    return Instruction(
        version=INSTRUCTION_VERSION, text=text.strip(), digest=_digest_of(data), path=source
    )


@lru_cache(maxsize=1)
def instruction() -> Instruction:
    """Return the packaged instruction, read once per process.

    Returns:
        The instruction of `INSTRUCTION_PATH`. The value is immutable and
        shared, so every chain of a run is compacted with the same prompt and
        the manifest's digest is the digest of what was actually used.

    Raises:
        InstructionError: If the packaged data file is broken or missing.
    """
    return load_instruction()


def instruction_digest(path: str | Path | None = None) -> str:
    """Return the hex sha256 of an instruction file's bytes.

    This is the value T9.3's run manifest (#39) records, and the reason the
    instruction is a file: a digest identifies the exact prompt a run used,
    which a wording nobody wrote down cannot.

    Args:
        path: The data file; the packaged `INSTRUCTION_PATH` when omitted.

    Returns:
        64 lower-case hex characters. The same bytes always give the same
        digest, and any change to the file — the text, the version, a comment —
        gives a different one.

    Raises:
        InstructionError: If the file cannot be read. The digest is taken of a
            file that was read, not of one that might be there.
    """
    source = INSTRUCTION_PATH if path is None else Path(path)
    return _digest_of(_read(source))
