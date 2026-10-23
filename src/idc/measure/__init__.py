"""What a run measured, and what makes those numbers mean something.

Measurement is deterministic: the numbers reported by the study are computed
from the transcripts alone, never estimated by a model. Six decisions carry the
stage, and each is stated where it is implemented:

- **What is counted** — `idc.measure.tokens`: a message's `content`, every tool
  call's name and its arguments as JSON, and its tool result's output, over the
  **visible context** that `idc.importers.visible_context` resolves and never
  over the raw session. That module's docstring names the bytes exactly, and
  says why the figure is a lower bound on a provider's own count.
- **What counts with what** — `idc.measure.tokenizers`: a `Tokenizer` protocol
  with one method, injected the way `idc.classify`'s adapters are, and a
  registry that places a transcript's `model` in a `ModelFamily`. A model of no
  known family is refused rather than given a default, so two runs can never
  look comparable while having been counted differently. This build ships no
  real tokenizer — every one of them needs a downloaded vocabulary or a network
  call — so `idc.measure.fake.FakeTokenizer` (one token per word) is the only
  one here, and it is what the whole suite counts with.
- **What is reported** — `idc.measure.compression`: `saved_fraction =
  (before - after) / before`, with a zero baseline refused, a context that grew
  reported as a negative fraction, and neither clamped.
- **What else the transcript says about the work** —
  `idc.measure.statistics`: the exploration rate, the writes per file, the
  lines they added and the exit-code histogram, over the same visible context
  the token counts are over. A write whose added lines the transcript does not
  state is reported as uncounted rather than given a guess.
- **What one call answers** — `idc.measure.metrics`: `measure(before, after,
  tokenizer)` assembles the four above for one pair of transcripts and computes
  nothing of its own. One tokenizer counts both sides, the tool statistics are
  the *before* session's, and an exploration rate with no tool call behind it
  is `None` instead of a refusal.
- **What makes a number attributable** — `idc.measure.manifest`: the run
  manifest, with the digests of the input and output bytes, the configuration,
  the compaction instruction's digest and version, the **tokenizer's identity**
  (a count does not carry it), the metrics, the run's two instants — injected
  through `idc.measure.clock`, never read from a clock inside the stage — and
  the predecessor run's output digest, so two runs chain. It is its own
  artifact, read back only through `load_manifest`/`loads_manifest`.

```python
from idc.measure import FakeTokenizer, RunConfig, RunTimer, SystemClock
from idc.measure import build_manifest, dump_manifest, measure

timer = RunTimer(SystemClock())
timer.start()
...                                        # the pipeline
metrics = measure(session, compacted, FakeTokenizer())
metrics.saved_fraction                     # what the study reports
dump_manifest(
    build_manifest(
        window=timer.finish(),
        input_path=source,
        output_path=target,
        config=RunConfig(agent=session.agent, adapter="s1m"),
        tokenizer_name="FakeTokenizer",
        metrics=metrics,
    ),
    target.with_suffix(".manifest.json"),
)
```
"""

from idc.measure.compression import CompressionRate
from idc.measure.errors import (
    EmptyBaselineError,
    MeasureError,
    NoTokenizerError,
    UnknownModelError,
)
from idc.measure.fake import FakeTokenizer
from idc.measure.tokenizers import (
    FAMILY_SEPARATOR,
    ModelFamily,
    Tokenizer,
    TokenizerRegistry,
    model_family,
)
from idc.measure.tokens import (
    ContextCount,
    MessageCount,
    count_context,
    count_message,
    count_messages,
    message_parts,
)

# One self-contained block, and the sub-tickets of T9 (#36) **append** to it
# with `__all__ += [...]` of their own rather than editing this one: two
# sub-tickets of one parent once collided in a package `__init__.py` because
# both rewrote the single list.
__all__ = [
    "FAMILY_SEPARATOR",
    "CompressionRate",
    "ContextCount",
    "EmptyBaselineError",
    "FakeTokenizer",
    "MeasureError",
    "MessageCount",
    "ModelFamily",
    "NoTokenizerError",
    "Tokenizer",
    "TokenizerRegistry",
    "UnknownModelError",
    "count_context",
    "count_message",
    "count_messages",
    "message_parts",
    "model_family",
]

# T9.2 (#38) — exploration rate, write counts and the exit-code histogram.
# Appended as its own block, with its own imports and its own `__all__ +=`,
# the way the comment above asks: nothing here edits the list T9.1 wrote.
#
# `idc.measure.statistics` carries this part's decisions in its docstring —
# the denominator is the visible context (the same message set the token
# counts are over), a session with no tool call refuses a rate instead of
# answering `0.0`, two spellings of one path are two files, an edit's added
# lines are the `+` lines of a diff of the strings the agent sent, and the
# histogram is keyed by exit code over every answered call with `None` as a
# bucket of its own. A write whose added lines the transcript does not state —
# an edit applied to every occurrence, above all — is reported as uncounted
# rather than given one application's worth.
from idc.measure.errors import NoToolCallsError
from idc.measure.statistics import (
    ExitCodeCount,
    FileEdits,
    ToolStatistics,
    lines_added_of,
    tool_statistics,
)

__all__ += [
    "ExitCodeCount",
    "FileEdits",
    "NoToolCallsError",
    "ToolStatistics",
    "lines_added_of",
    "tool_statistics",
]

# T9.3 (#39) — the metrics aggregate and the run manifest. Its own block with
# its own imports and its own `__all__ +=`, as the comment above the first
# block asks.
#
# The two modules carry this part's decisions in their docstrings.
# `idc.measure.metrics`: one injected tokenizer counts both sides of a run, the
# tool statistics are the *before* session's because the after session's
# history is this tool's own summary block, and an exploration rate with no
# tool call behind it is `None` rather than a refusal. `idc.measure.manifest`:
# a digest is sha256 over a file's bytes (the same function the instruction
# digest is), the tokenizer is named by the caller because a count carries no
# backend identity, a predecessor run is identified by its *output* digest so
# that two manifests chain, and the document is read back only through
# `loads_manifest` / `load_manifest`, which raise `ManifestError`.
# `idc.measure.clock` holds the one seam measurement cannot compute: the two
# instants of a run, injected, never read from a clock inside the stage.
#
# `noqa: E402` on each import below: ruff permits an import after a dunder
# *assignment* (which is why T9.1's block needs none) but not after the
# augmented `__all__ += [...]` that T9.2's block ends with. Keeping the
# per-ticket blocks is worth five suppressions; moving these imports above
# that line would mean editing another sub-ticket's block, which the comment
# above the first `__all__` exists to prevent.
from idc.measure.clock import Clock, RunTimer, RunWindow, SystemClock  # noqa: E402
from idc.measure.errors import ManifestError  # noqa: E402
from idc.measure.fake import FakeClock  # noqa: E402
from idc.measure.manifest import (  # noqa: E402
    MANIFEST_VERSION,
    FileDigest,
    InstructionIdentity,
    MetricsRecord,
    RunConfig,
    RunManifest,
    TokenizerIdentity,
    build_manifest,
    digest_bytes,
    digest_file,
    dump_manifest,
    dumps_manifest,
    load_manifest,
    loads_manifest,
)
from idc.measure.metrics import Metrics, measure  # noqa: E402

__all__ += [
    "MANIFEST_VERSION",
    "Clock",
    "FakeClock",
    "FileDigest",
    "InstructionIdentity",
    "ManifestError",
    "Metrics",
    "MetricsRecord",
    "RunConfig",
    "RunManifest",
    "RunTimer",
    "RunWindow",
    "SystemClock",
    "TokenizerIdentity",
    "build_manifest",
    "digest_bytes",
    "digest_file",
    "dump_manifest",
    "dumps_manifest",
    "load_manifest",
    "loads_manifest",
    "measure",
]
