"""Reading a line-delimited JSON transcript, the shape two of the agents use.

One place reads a `*.jsonl` file, so that every importer reports a broken
transcript the same way and none of them has to think about blank lines or
encodings. The reader does exactly two things: it hands back one mapping per
line together with that line's number, and it refuses a file it cannot read.

The line number is the point. A transcript is thousands of lines of one-line
JSON, so "the tenth record" is useless to the person holding the file while
"line 412" is the line their editor jumps to; every error an importer raises
about a single record carries it.

A malformed line is **fatal**, not skipped. A transcript is the study's data: a
line that is not JSON means the file was damaged, and quietly dropping it would
silently change what the measurement counts. Trimming such a file is the
researcher's decision, not this tool's.

That is also why a line ends at a newline and nowhere else. `str.splitlines()`
breaks on U+2028, U+2029 and U+0085 as well, and those three are legal
*unescaped* inside a JSON string — neither `JSON.stringify` nor `json.dumps`
escapes them. A message holding one (minified JavaScript, pasted text) would be
split mid-string, and the fatal-line rule would then reject the whole session
over a file that is perfectly valid. A carriage return before the newline is
dropped, so a transcript written on Windows reads the same.

A file whose records are separated by carriage returns alone — an old Mac
editor — is therefore one single line here, and that is fatal: such a file is
not JSONL. It is reported *as* that, by the carriage return, rather than by the
JSON error of a line this reader invented by not splitting; a transcript written
that way has to be converted, not parsed harder. The same message answers for a
lone literal carriage return inside a record, which JSON does not allow
unescaped either — the escaped spelling is what a writer emits and it survives.

A leading byte-order mark is dropped rather than treated as part of the first
record: an editor on Windows writes one, it carries no information a transcript
needs, and leaving it in front of line 1 would make a valid file fatally
unreadable.

```python
for line in read_json_lines("session.jsonl"):
    line.number      # 1, 2, 3, …
    line.data["type"]
```
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from idc.importers.base import TranscriptImportError

__all__ = ["JsonLine", "read_json_lines"]


@dataclass(frozen=True, slots=True)
class JsonLine:
    """One record of a JSONL transcript, with the line it was read from.

    Attributes:
        number: The 1-based line number in the file, as an editor counts.
        data: The parsed JSON object.
    """

    number: int
    data: dict[str, Any]


def read_json_lines(path: str | Path) -> list[JsonLine]:
    """Read a JSONL transcript into one mapping per line.

    Args:
        path: The transcript file, read as UTF-8, with a leading byte-order
            mark dropped if the editor that wrote it left one. A line ends at a
            newline — a
            carriage return before it is dropped — and not at every character
            Unicode calls a line break, because U+2028, U+2029 and U+0085 occur
            unescaped inside JSON strings. Blank lines are skipped; the line
            numbers of the remaining records are unaffected by them.

    Returns:
        One `JsonLine` per non-blank line, in file order.

    Raises:
        TranscriptImportError: If the file cannot be read, is not valid UTF-8,
            holds a line that is not JSON or a line whose JSON is not an object,
            or holds no record at all. Every message names the file, and the
            per-line ones name the line. A line holding an unescaped carriage
            return is reported by that carriage return — the file is
            separated by carriage returns and is not JSONL — rather than by the
            JSON error it causes.
    """
    source = Path(path)
    try:
        text = source.read_bytes().decode("utf-8-sig")
    except OSError as exc:
        raise TranscriptImportError(
            f"cannot read the transcript {source}: {exc.strerror or exc}"
        ) from None
    except UnicodeDecodeError:
        raise TranscriptImportError(f"invalid transcript {source}: not valid UTF-8") from None
    lines: list[JsonLine] = []
    for number, raw in enumerate(text.split("\n"), start=1):
        line = raw.rstrip("\r")
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if "\r" in line:
                raise TranscriptImportError(
                    f"invalid transcript {source}: line {number} holds an unescaped carriage "
                    f"return, which JSON does not allow inside a string; a transcript whose "
                    f"records are separated by carriage returns instead of newlines reads as "
                    f"one line and is not JSONL"
                ) from None
            raise TranscriptImportError(
                f"invalid transcript {source}: line {number} is not valid JSON ({exc.msg})"
            ) from None
        if not isinstance(record, dict):
            raise TranscriptImportError(
                f"invalid transcript {source}: line {number} is not a JSON object "
                f"but a {type(record).__name__}"
            )
        lines.append(JsonLine(number, record))
    if not lines:
        raise TranscriptImportError(f"invalid transcript {source}: no records")
    return lines
