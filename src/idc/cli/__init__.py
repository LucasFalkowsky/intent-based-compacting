"""Command surface of the tool: one command per pipeline stage, plus `run`.

```
idc import   --agent claude-code --in session.jsonl --out session.json
idc chunk    --in session.json   --out chunked.json
idc classify --in chunked.json   --out classified.json
idc resolve  --in classified.json
idc compact  --in classified.json --out compacted.json --target claude-code
idc render   --in compacted.json  --out compacted.jsonl
idc measure  --agent claude-code --in session.jsonl --compacted compacted.jsonl
idc run      --agent claude-code --in session.jsonl --out compacted.jsonl
```

`run` is the end-to-end command the study uses: it imports, chunks, classifies,
compacts, renders, measures and writes a run manifest in one pass. The seven
before it are the same pipeline with files in between, which is how a stage's
output is inspected and how a disputed number is traced back to the stage that
produced it. Either way this package is the only place that reads configuration
or touches the filesystem on the tool's behalf; the stages themselves take
objects and return objects.

Reading the package:

- `idc.cli.main` — the parser, the `COMMANDS` table and `main`, the one entry
  point. `python -m idc` is a two-line wrapper around it.
- `idc.cli.commands` — what each command does, and the three decisions about how
  the stages are wired together.
- `idc.cli.config` — the TOML configuration file and its validation.
- `idc.cli.artifacts` — the envelope one stage hands the next on disk.
- `idc.cli.runtime` — the four dependencies a command cannot compute, and the
  seam they arrive through.
- `idc.cli.exit_codes` and `idc.cli.errors` — the exit statuses, and the error
  family `main` turns into them.

No agent integration and no git calls anywhere in here: a command reads
transcript files and writes transcript files.
"""

from idc.cli.artifacts import (
    ARTIFACT_KEYS,
    BLOCK_KEY,
    CHAINS_KEY,
    CHUNKS_KEY,
    SESSION_KEY,
    Artifact,
    dump_artifact,
    dumps_artifact,
    load_artifact,
    loads_artifact,
)
from idc.cli.commands import (
    MANIFEST_SUFFIX,
    candidate_chains,
    chunk_command,
    classify_command,
    compact_command,
    compacted_block,
    import_command,
    manifest_path,
    measure_command,
    render_command,
    resolve_command,
    resolve_report,
    run_command,
    summary_block_text,
)
from idc.cli.config import (
    ADAPTER_NAMES,
    DEFAULT_ADAPTER,
    PATH_KEYS,
    Config,
    Paths,
    load_config,
    loads_config,
    read_config_document,
    validate_config,
)
from idc.cli.errors import (
    ArtifactError,
    BackendUnavailableError,
    CliError,
    ConfigError,
    UsageError,
)
from idc.cli.exit_codes import EXIT_INPUT_ERROR, EXIT_OK, EXIT_USAGE_ERROR
from idc.cli.main import (
    COMMAND_NAMES,
    COMMANDS,
    DIAGNOSTIC_ERRORS,
    PROGRAM,
    Command,
    build_parser,
    configuration,
    main,
)
from idc.cli.runtime import Runtime

__all__ = [
    "ADAPTER_NAMES",
    "ARTIFACT_KEYS",
    "BLOCK_KEY",
    "CHAINS_KEY",
    "CHUNKS_KEY",
    "COMMANDS",
    "COMMAND_NAMES",
    "DEFAULT_ADAPTER",
    "DIAGNOSTIC_ERRORS",
    "EXIT_INPUT_ERROR",
    "EXIT_OK",
    "EXIT_USAGE_ERROR",
    "MANIFEST_SUFFIX",
    "PATH_KEYS",
    "PROGRAM",
    "SESSION_KEY",
    "Artifact",
    "ArtifactError",
    "BackendUnavailableError",
    "CliError",
    "Command",
    "Config",
    "ConfigError",
    "Paths",
    "Runtime",
    "UsageError",
    "build_parser",
    "candidate_chains",
    "chunk_command",
    "classify_command",
    "compact_command",
    "compacted_block",
    "configuration",
    "dump_artifact",
    "dumps_artifact",
    "import_command",
    "load_artifact",
    "load_config",
    "loads_artifact",
    "loads_config",
    "main",
    "manifest_path",
    "measure_command",
    "read_config_document",
    "render_command",
    "resolve_command",
    "resolve_report",
    "run_command",
    "summary_block_text",
    "validate_config",
]
