"""Loading and dumping canonical objects, with validation on every load.

Nothing in the pipeline parses a canonical document by hand: a stage calls
`load`/`loads` and either gets a valid object or a `SchemaValidationError` that
names the type, the offending field path and what was wrong. The error type and
its formatting live in `idc.schema.errors`, because construction and assignment
report the same way (see `idc.schema.models`); the only thing the load path adds
is the source file.

The dumped form is plain, indented JSON with a trailing newline: it is
committed as study material and read in diffs, so stability of the text matters
as much as the round trip itself.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from idc.schema.errors import SchemaValidationError, as_schema_error
from idc.schema.models import CanonicalModel

INDENT = 2
"""Indentation of the dumped JSON; fixed so that dumps are byte-comparable."""


def validate_document[M: CanonicalModel](
    model: type[M], data: Mapping[str, Any] | Any, *, source: str | None = None
) -> M:
    """Validate already-parsed JSON data against a canonical type.

    Args:
        model: The canonical type to validate against.
        data: The parsed document, normally a mapping.
        source: Where the document came from, for the error message.

    Returns:
        The validated instance of `model`.

    Raises:
        SchemaValidationError: If the document does not satisfy the schema.
    """
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise as_schema_error(exc, model, source) from None


def loads[M: CanonicalModel](model: type[M], data: str | bytes, *, source: str | None = None) -> M:
    """Parse and validate a canonical document given as JSON text.

    Args:
        model: The canonical type to validate against.
        data: The JSON document as text or UTF-8 bytes.
        source: Where the document came from, for the error message.

    Returns:
        The validated instance of `model`.

    Raises:
        SchemaValidationError: If the text is not JSON or does not satisfy the
            schema.
    """
    try:
        document = json.loads(data)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(
            model.__name__,
            f"not valid JSON ({exc.msg} at line {exc.lineno}, column {exc.colno})",
            source=source,
        ) from None
    return validate_document(model, document, source=source)


def load[M: CanonicalModel](model: type[M], path: str | Path) -> M:
    """Read and validate a canonical document from a file.

    Args:
        model: The canonical type to validate against.
        path: Path of the JSON document, read as UTF-8.

    Returns:
        The validated instance of `model`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        SchemaValidationError: If the file is not JSON or does not satisfy the
            schema.
    """
    path = Path(path)
    return loads(model, path.read_text(encoding="utf-8"), source=str(path))


def dumps(obj: CanonicalModel) -> str:
    """Serialize a canonical object to JSON text.

    Args:
        obj: Any canonical object.

    Returns:
        Indented JSON without a trailing newline. Non-ASCII characters are kept
        as characters, so transcripts stay readable in a diff.
    """
    return obj.model_dump_json(indent=INDENT)


def dump(obj: CanonicalModel, path: str | Path) -> Path:
    """Write a canonical object to a file as JSON.

    Args:
        obj: Any canonical object.
        path: Destination path; the file is written as UTF-8 with a trailing
            newline.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.write_text(dumps(obj) + "\n", encoding="utf-8")
    return path
