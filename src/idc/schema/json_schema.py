"""Generation of the committed JSON Schema document for the canonical types.

The document in `canonical.schema.json` is **generated**, never hand-written: it
is derived from the models in `idc.schema.models`, committed next to them, and a
test fails if regenerating it would change a byte. That makes the schema usable
as the published contract of the study's data — it can be referenced, diffed and
validated against by tools outside this repository — while keeping the models
the only place a field is ever defined.

Regenerate it with `make schema` (`python -m idc.schema`) after changing a
model, which — per `CLAUDE.md` — is a ticket of its own.
"""

import json
from pathlib import Path
from typing import Any

from pydantic.json_schema import models_json_schema

from idc.schema.models import CANONICAL_MODELS

SCHEMA_PATH = Path(__file__).with_name("canonical.schema.json")
"""Location of the committed JSON Schema document, shipped inside the package."""

SCHEMA_TITLE = "idc canonical transcript schema"
"""Title of the generated document."""

SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
"""JSON Schema dialect pydantic v2 emits."""


def build_json_schema() -> dict[str, Any]:
    """Generate the JSON Schema document for every canonical type.

    Returns:
        A document whose `$defs` holds one definition per canonical type and per
        enum they reference. There is no root type: the document describes all
        seven types, and a consumer references the one it needs as
        `#/$defs/Session`.
    """
    _, schema = models_json_schema(
        [(model, "validation") for model in CANONICAL_MODELS],
        title=SCHEMA_TITLE,
        ref_template="#/$defs/{model}",
    )
    return {"$schema": SCHEMA_DIALECT, **schema}


def schema_document_text() -> str:
    """Render the generated schema exactly as it is committed.

    Returns:
        The document as UTF-8 JSON text with sorted keys, two-space indentation
        and a trailing newline — deterministic, so the file is diffable and the
        sync test can compare bytes.
    """
    return json.dumps(build_json_schema(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_json_schema(path: str | Path = SCHEMA_PATH) -> Path:
    """Write the generated schema document to `path`.

    Args:
        path: Destination; defaults to the committed location.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.write_text(schema_document_text(), encoding="utf-8")
    return path


def load_json_schema() -> dict[str, Any]:
    """Read the committed schema document.

    Returns:
        The parsed document. Use this instead of `build_json_schema` when the
        published contract is what matters; the two agree by test.
    """
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
