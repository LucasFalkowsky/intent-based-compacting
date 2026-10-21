"""Counting tokens over the visible context, and nothing else.

```python
from idc.measure import FakeTokenizer, count_context

counted = count_context(session, FakeTokenizer())
counted.tokens          # the whole visible context
counted.per_message     # one entry per visible message, in transcript order
counted.hidden_ids      # what was in the file and not in the count
```

**What is counted — exactly.** For every message, in this order:

1. `Message.content`,
2. per tool call, in the order the message carries them: `ToolCall.name`, then
   `ToolCall.args` as JSON (`json.dumps(..., sort_keys=True,
   ensure_ascii=False)`, so the bytes do not depend on dictionary order),
3. `ToolResult.output`, when the message carries a result.

Empty pieces are not handed to the tokenizer at all. `message_parts` is that
list, and it is public so a caller can see what a number was produced from.

Counting `content` alone would under-report precisely the thing this study is
about: a tool result is what fills a context window, and an agent is sent all
three of these. The arguments are included for the same reason — a `Write` call
carries the whole new file in them.

**What is *not* counted, and why.** `Message.id`, `ToolCall.id`,
`ToolCall.category`, `ToolResult.call_id`, `ToolResult.exit_code`,
`ToolResult.is_error`, `Message.timestamp` and the two `is_*` flags are this
tool's own canonical bookkeeping: the identifiers are assigned by the importers,
the category is this tool's classification of a tool name, and the exit code and
error flag are derived by an importer from the output. None of them is text an
agent was sent, so counting them would add bytes the measured conversation never
had.

**The count is a lower bound, deliberately.** An agent's wire format wraps every
message in framing a transcript does not record — a role marker, a JSON
envelope, the tool-call scaffolding a provider's API adds — and this ticket does
not model it: nothing in the canonical schema says what any of the three agents
sends, and inventing a per-message constant would put a made-up number into the
study's results. So the reported figure is the tokens of the *text* of a
conversation, under-counting the provider's own bill by an amount that grows
with the number of messages.

That is acceptable for the before/after comparison this stage exists for,
because both sides are counted by the one same definition and the comparison is
between them, not against a provider's invoice. What is **not** claimed is that
the omitted framing cancels in the rate: it does not. A compacted context holds
fewer messages, so it carries less framing, and whether that pushes the true
saving above or below the one reported here depends on how the text is
distributed over the messages on either side.

**What the output does *not* carry is which tokenizer produced the number.**
The tokenizer is injected and `count_context` never checks it against the
document's `model`, so a session recorded as one model can be counted with any
tokenizer a caller hands over and nothing in the `ContextCount` would say so.
That is deliberate — the registry is the caller's, by the decision above — but
it means a count is reproducible only together with the tokenizer set the run
used, which is the run manifest's job to record (#39) and not this value
object's to guess at. An earlier version of this paragraph claimed the output
named its own definition; it does not, and a review disproved it.

**The refusal that is passed through.** `count_context` resolves what is visible
with `idc.importers.visible_context` and does not re-derive it. That function
raises `TranscriptImportError` when a session the agent compacted itself does
not locate its summary, and nothing here catches it: the refusal exists so that
measurement cannot count history the agent has already dropped, and wrapping it
in a `MeasureError` would blur what a reader has to do about it.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass

from idc.importers import ImportedSession, visible_context
from idc.measure.tokenizers import Tokenizer
from idc.schema import Message, Session

__all__ = [
    "ContextCount",
    "MessageCount",
    "count_context",
    "count_message",
    "count_messages",
    "message_parts",
]


def message_parts(message: Message) -> tuple[str, ...]:
    """Return the texts of a message that are counted, in the counted order.

    See this module's docstring for the whole rule and for what is left out.
    Empty pieces are dropped, so a `tool` message that carries only a result
    answers with one element, a tool call without arguments contributes its
    name alone, and a message that carries nothing readable answers with none.
    A tool call's name is always there — the schema refuses an empty one — so
    it is never dropped.

    Args:
        message: The message to read.

    Returns:
        The content, then every tool call's name and its arguments as canonical
        JSON, then the tool result's output; the content, the arguments and the
        output only if they hold a character.
    """
    parts: list[str] = []
    if message.content:
        parts.append(message.content)
    for call in message.tool_calls:
        parts.append(call.name)
        if call.args:
            parts.append(json.dumps(call.args, sort_keys=True, ensure_ascii=False))
    if message.tool_result is not None and message.tool_result.output:
        parts.append(message.tool_result.output)
    return tuple(parts)


def count_message(message: Message, tokenizer: Tokenizer) -> int:
    """Return how many tokens one message is.

    Args:
        message: The message to count.
        tokenizer: The tokenizer to count its parts with.

    Returns:
        The sum over `message_parts(message)`.
    """
    return sum(tokenizer.count(part) for part in message_parts(message))


def count_messages(messages: Sequence[Message], tokenizer: Tokenizer) -> int:
    """Return how many tokens a sequence of messages is.

    The plain sum over `count_message`, offered so that a caller can count a
    selection — the messages a context left out, say — without re-deriving the
    rule. It counts whatever it is handed and asks no questions about
    visibility; `count_context` is the one that decides that.

    Args:
        messages: The messages to count, in any order.
        tokenizer: The tokenizer to count them with.

    Returns:
        The total, zero for an empty sequence.
    """
    return sum(count_message(message, tokenizer) for message in messages)


@dataclass(frozen=True, slots=True)
class MessageCount:
    """What one visible message contributed to a context's count.

    Attributes:
        message_id: `Message.id` of the message.
        tokens: How many tokens its counted parts are.
    """

    message_id: str
    tokens: int


@dataclass(frozen=True, slots=True)
class ContextCount:
    """The token count of one transcript's visible context.

    Carries the per-message breakdown and what was left out, so that a result a
    reader doubts can be traced to the message that produced it without
    counting the transcript again.

    Attributes:
        session_id: `Session.id` of the transcript that was counted.
        model: The model the document recorded, or `None`. Kept because it is
            what a reader resolves a tokenizer *from* — **not** a record of
            which tokenizer was actually used: the tokenizer is injected and is
            never checked against this field, so the two can disagree. See the
            module docstring.
        tokens: The whole visible context.
        per_message: One entry per visible message, in transcript order.
        hidden_ids: Identifiers of the messages that are in the file and not in
            the count, in transcript order — see
            `idc.importers.visible_context` for the two reasons.
    """

    session_id: str
    model: str | None
    tokens: int
    per_message: tuple[MessageCount, ...]
    hidden_ids: tuple[str, ...]

    @property
    def message_ids(self) -> tuple[str, ...]:
        """Return the identifiers of the counted messages, in transcript order."""
        return tuple(entry.message_id for entry in self.per_message)

    def __str__(self) -> str:
        """Return the count as one diagnostic line."""
        return (
            f"{self.session_id}: {self.tokens} token(s) over "
            f"{len(self.per_message)} visible message(s), {len(self.hidden_ids)} hidden"
        )


def count_context(session: ImportedSession | Session, tokenizer: Tokenizer) -> ContextCount:
    """Return the token count of the part of a transcript the agent is sent.

    Counting is over the **visible context** and never over the raw session:
    `idc.importers.visible_context` is the one place that decides what counts as
    context, so that a measurement counts the same thing whichever agent
    recorded the transcript, and this function re-derives none of it.

    Args:
        session: An `ImportedSession`, or a canonical `Session` on its own —
            including one loaded back from a file.
        tokenizer: The tokenizer to count with.
            `TokenizerRegistry.tokenizer_for` picks the one a session's model
            calls for.

    Returns:
        The count, with the per-message breakdown and the identifiers of the
        messages that were left out.

    Raises:
        TranscriptImportError: If the session was compacted by the agent itself
            but does not locate its summary. Raised by `visible_context` and
            passed through unchanged — measuring a transcript whose dropped
            history cannot be located is the one thing this must not do.
        SchemaValidationError: If the visible messages do not satisfy the
            canonical schema, likewise from `visible_context`.
    """
    context = visible_context(session)
    per_message = tuple(
        MessageCount(message_id=message.id, tokens=count_message(message, tokenizer))
        for message in context.messages
    )
    return ContextCount(
        session_id=context.session.id,
        model=context.session.model,
        tokens=sum(entry.tokens for entry in per_message),
        per_message=per_message,
        hidden_ids=context.hidden_ids,
    )
