"""One call that produces every number the study reports for a run.

`count_context`, `CompressionRate` and `tool_statistics` each answer one
question, and the study needs all of them for the same pair of transcripts.
`measure` is the single entry point that assembles them, and `Metrics` is what a
run manifest records. Nothing here computes a number of its own: every field is
the answer of the part that owns it, which is why the tests for this module
assert composition rather than arithmetic.

```python
from idc.measure import FakeTokenizer, measure

metrics = measure(session, compacted, FakeTokenizer())
metrics.saved_fraction      # from CompressionRate
metrics.exploration_rate    # from ToolStatistics, or None
metrics.tools.files_edited  # from ToolStatistics
```

Three decisions are worth reading before the code:

- **One tokenizer counts both sides.** `measure` takes a single `Tokenizer` and
  uses it for the "before" and the "after" context, so a compression rate can
  never be the difference between two tokenizers. A caller resolves it once with
  `TokenizerRegistry.tokenizer_for`; the two sessions of one run are the same
  model's.
- **The tool statistics are the "before" session's.** The study counts what the
  agent did while solving the task, and the "after" session is this tool's own
  output: its history has been replaced by a summary block, so its tool calls
  are a property of the compaction and not of the work. Counting both would
  invite reading two numbers as comparable when only one of them measures an
  agent at all. The statistics carry their `session_id`, so which transcript
  they are over is readable off the object.
- **An undefined exploration rate is `None`, not a refusal.**
  `ToolStatistics.exploration_rate` raises `NoToolCallsError` for a session that
  used no tool, which is right for a caller asking that one question. A session
  that made no tool call is a legitimate run to measure, though — every other
  metric is defined for it — so `Metrics.exploration_rate` reports `None` and
  `measure` does not raise. The refusal stays available on `metrics.tools`.

What `measure` does **not** swallow: a baseline that counts no token is still
`EmptyBaselineError`, and a model of no known family is still
`UnknownModelError` when the tokenizer is resolved. Both are refusals about the
input, and turning them into a `None` would put an unmeasurable run in a results
table.
"""

from dataclasses import dataclass

from idc.importers import ImportedSession
from idc.measure.compression import CompressionRate
from idc.measure.errors import NoToolCallsError
from idc.measure.statistics import ToolStatistics, tool_statistics
from idc.measure.tokenizers import Tokenizer
from idc.measure.tokens import ContextCount, count_context
from idc.schema import Session

__all__ = ["Metrics", "measure"]


@dataclass(frozen=True, slots=True)
class Metrics:
    """Every measured number of one run, assembled from the parts that own it.

    Attributes:
        before: The token count of the visible context as it stood.
        after: The token count of the compacted context.
        compression: The rate between the two counts.
        tools: The tool statistics of the **before** session; see the module
            docstring for why only that one.
    """

    before: ContextCount
    after: ContextCount
    compression: CompressionRate
    tools: ToolStatistics

    def __str__(self) -> str:
        """Return the metrics as one diagnostic line."""
        rate = self.exploration_rate
        explored = "no tool call" if rate is None else f"{rate:.2f} exploration"
        return (
            f"{self.before.tokens} -> {self.after.tokens} tokens "
            f"({self.compression.percent_saved:.1f}% saved), "
            f"{self.tools.total_calls} tool call(s), {explored}, "
            f"{self.tools.file_count} file(s), {self.tools.lines_added} line(s) added"
        )

    @property
    def saved_fraction(self) -> float:
        """Return `(before - after) / before` — the figure the study reports."""
        return self.compression.saved_fraction

    @property
    def exploration_rate(self) -> float | None:
        """Return the before session's exploration rate, or `None` with no tool call.

        Returns:
            `read_calls / total_calls` of the before session, or `None` when it
            made no tool call at all — the one metric that has no value then.
            See the module docstring for why this is `None` here and a refusal
            on `tools`.
        """
        try:
            return self.tools.exploration_rate
        except NoToolCallsError:
            return None


def measure(
    before: ImportedSession | Session,
    after: ImportedSession | Session,
    tokenizer: Tokenizer,
) -> Metrics:
    """Return every measured number of one compaction run.

    Args:
        before: The transcript as it stood — an `ImportedSession` or a
            canonical `Session`, including one loaded back from a file.
        after: The compacted transcript, in the same two shapes.
        tokenizer: The tokenizer **both** sides are counted with;
            `TokenizerRegistry.tokenizer_for` picks the one a session's model
            calls for.

    Returns:
        The metrics: both counts, the compression rate between them, and the
        tool statistics of `before`.

    Raises:
        EmptyBaselineError: If the before context counts no token, which leaves
            the compression rate undefined.
        TranscriptImportError: If either session was compacted but does not
            locate its summary; `idc.importers.visible_context` refuses it and
            this function lets the refusal through.
        SchemaValidationError: If either session's visible messages do not
            satisfy the canonical schema.
    """
    counted_before = count_context(before, tokenizer)
    counted_after = count_context(after, tokenizer)
    return Metrics(
        before=counted_before,
        after=counted_after,
        compression=CompressionRate.between(counted_before, counted_after),
        tools=tool_statistics(before),
    )
