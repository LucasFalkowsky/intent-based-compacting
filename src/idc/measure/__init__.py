"""Token measurement before and after compaction.

Measurement is deterministic: the numbers reported by the study are computed
from the transcripts alone, never estimated by a model. Three decisions carry
the stage, and each is stated where it is implemented:

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

```python
from idc.measure import CompressionRate, FakeTokenizer, count_context

before = count_context(session, FakeTokenizer())
after = count_context(compacted, FakeTokenizer())
CompressionRate.between(before, after).saved_fraction
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
