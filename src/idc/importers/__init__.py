"""Importers that read agent transcripts into the canonical format.

One importer per supported agent (Claude Code, Gemini CLI, OpenCode). An
importer is responsible for the agent's on-disk shape — JSON or JSONL, tool-call
encoding, role names — and for nothing else: everything downstream sees only
canonical objects.

The agents' tool knowledge is shared and lives in a data file, not in code; read
it through this package:

```python
from idc.importers import tool_registry

tool_registry().category("claude-code", "Edit")     # ToolCategory.WRITE
tool_registry().is_verification("uv run pytest -q") # True
```

See `idc.importers.tool_registry` for the lookup rules and for how a command
line is matched against the verification patterns. The importers themselves are
added in T2.2 to T2.4; this package carries no implementations yet.
"""

from idc.importers.tool_registry import (
    REGISTRY_PATH,
    REGISTRY_VERSION,
    ToolRegistry,
    ToolRegistryError,
    ToolSpec,
    VerificationMatch,
    VerificationPattern,
    load_tool_registry,
    tool_registry,
)

__all__ = [
    "REGISTRY_PATH",
    "REGISTRY_VERSION",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolSpec",
    "VerificationMatch",
    "VerificationPattern",
    "load_tool_registry",
    "tool_registry",
]
