"""Canonical transcript schema — the contract between all pipeline stages.

Every stage consumes and produces canonical objects defined here, so that the
agent-specific shapes exist only inside `idc.importers` and `idc.render`.
Changing this schema is a ticket of its own, never a side effect of another
change.

The seven types are, from the innermost outwards: `ToolCall`, `ToolResult`,
`Message` and `Session` (the imported transcript), `Chunk` (the chunked
transcript), `Chain` (the classified transcript) and `CompactedBlock` (the
summary that is rendered and parsed back). Their closed value sets live in
`idc.schema.enums`.

Load and dump canonical documents through this package rather than with `json`
directly — every load validates, and every failure is one `SchemaValidationError`
that names the type and the offending field path:

```python
from idc.schema import Session, dump, load

session = load(Session, "session.json")
dump(session, "copy.json")       # byte-identical round trip
```

`canonical.schema.json` next to this module is the same contract as a JSON
Schema document, generated from the models and kept in sync by a test; regenerate
it with `make schema`.
"""

from idc.schema.enums import Agent, ChunkOrigin, Role, SessionSource, ToolCategory
from idc.schema.json_schema import (
    SCHEMA_DIALECT,
    SCHEMA_PATH,
    SCHEMA_TITLE,
    build_json_schema,
    load_json_schema,
    schema_document_text,
    write_json_schema,
)
from idc.schema.models import (
    CANONICAL_MODELS,
    CanonicalModel,
    Chain,
    Chunk,
    CompactedBlock,
    Message,
    Session,
    ToolCall,
    ToolResult,
)
from idc.schema.serialization import (
    SchemaValidationError,
    dump,
    dumps,
    load,
    loads,
    validate_document,
)

__all__ = [
    "CANONICAL_MODELS",
    "SCHEMA_DIALECT",
    "SCHEMA_PATH",
    "SCHEMA_TITLE",
    "Agent",
    "CanonicalModel",
    "Chain",
    "Chunk",
    "ChunkOrigin",
    "CompactedBlock",
    "Message",
    "Role",
    "SchemaValidationError",
    "Session",
    "SessionSource",
    "ToolCall",
    "ToolCategory",
    "ToolResult",
    "build_json_schema",
    "dump",
    "dumps",
    "load",
    "load_json_schema",
    "loads",
    "schema_document_text",
    "validate_document",
    "write_json_schema",
]
