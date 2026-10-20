"""The LLM backend: a fixed prompt, a validated answer, a bounded retry.

One of the two real decision backends of the stage (#17; the other is #18). It
is an `IntentAdapter` like every other: it is asked, per chunk, whether that
chunk continues one of the open chains or starts a new one, and it answers with
a `Decision`. What it adds is the one thing a model needs — a prompt — and the
three things a model's answer needs: a shape to be read in, a bound on how often
it may be asked again, and a refusal that says where the answer went wrong.

```python
from idc.classify import LlmAdapter, classify

chains = classify(session, chunks, LlmAdapter(my_client))
```

Four decisions make this module what it is:

- **The model client is injected, and this module ships none.** A
  `ModelClient` is one method, text in and text out, so the backend holds no
  transport, no credential, no retry policy of the network and no process: the
  caller's client owns all of that, and a test passes a stub that answers from a
  list. Nothing here opens a connection, which is why no test of this
  repository calls a model — it is a property of the module, not a habit of the
  tests, and `tests/test_classify_llm_adapter.py` asserts it over this file's
  own imports.
- **The prompt is fixed.** `build_prompt` is a pure function of the chunk under
  decision and the candidates, and its text is pinned character for character by
  a test. The study compares runs of the method; a prompt that drifted between
  two of them would compare two methods instead of two transcripts. The model
  gets each candidate's id, label and last user message and the new turn's user
  message, verbatim — the same two kinds of text, so it compares like with like.
- **An answer is read, or it is refused with a reason.** The answer is one JSON
  object, `{"decision": "<C-id>|NEW", "label": "<new label if NEW>"}`. Anything
  that cannot be read as that — prose, a JSON array, a missing field, a
  `"decision"` that is neither `NEW` nor a `C<n>` — is unreadable, and *that*
  is what the retry is for: the next prompt names the reason, and when the bound
  is used up the adapter raises `UnreadableAnswerError`, which is an
  `AdapterContractError` and so a `ClassifyError` (T10's CLI catches the stage
  with one `except`). An answer naming a chain that was **not** a candidate is
  not retried but rejected on the spot with the store's own
  `UnknownChainError`: the model understood the format and decided something
  that does not exist, and the error belongs where the answer arrived rather
  than one stage further down.
- **Nothing a model says becomes data.** The decision is built through
  `as_decision`, never through `Decision.extend` on an unchecked string, so a
  wrong-shaped id stays a failure of the stage instead of becoming a plain
  `ValueError` (see `Decision`'s own docstring). The `label` the model suggests
  is recorded on the call and goes no further: the store names a new chain
  itself, deterministically, and T6 writes the real label. A label that holds no
  word is read as no label at all, because the store refuses such a label and an
  advisory field may not take a classification down.

The adapter is deterministic for a deterministic client: no clock, no random
number, no sleep between attempts. A backoff, like everything else about the
transport, belongs to the injected client.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from idc.classify.adapters import ChainCandidate, ChunkUnderDecision, Decision, as_decision
from idc.classify.errors import AdapterContractError, UnknownChainError
from idc.classify.ids import is_chain_id

DEFAULT_MAX_ATTEMPTS: Final[int] = 2
"""How often one chunk is asked about at most: the first attempt plus one retry.

The retry exists for a formatting slip — a fenced answer, a missing field, prose
around the JSON — and one is enough for that, because the second prompt names
what was wrong. A backend that answers unreadably twice is misconfigured rather
than unlucky, and a study run that silently burned three calls per chunk on it
would be measuring the wrong thing. A caller that knows better raises the bound
per adapter.
"""

_NEW_ANSWER: Final[str] = "NEW"
"""The one answer that is not a chain id, as the prompt asks for it."""

_MESSAGE_OPEN = "<<<"
"""Opens a block of verbatim transcript text in the prompt."""

_MESSAGE_CLOSE = ">>>"
"""Closes a block of verbatim transcript text in the prompt."""

_NO_MESSAGE = "(no user message)"
"""What stands in a message block that has no text.

A chunk of subagent work carries no user turn, and a user turn can carry no text
(#16) — both arrive as the empty string. An empty block would leave the model
guessing what the blank means.
"""

_PREAMBLE = """\
You classify one turn of a coding-agent transcript into a user intent.

An intent is what the user is trying to get done. The new turn continues an open
intent when it works towards the same goal — a follow-up, a correction, another
attempt at the same task — and it starts a new intent when it does not."""
"""What the model is asked to do, and what the question means."""

_CANDIDATES_HEADING = "Open intents, each with its id, its label and its last user message:"
"""Introduces the candidate blocks."""

_NO_CANDIDATES = "There is no open intent yet, so the new turn starts one."
"""Replaces the candidate section for the first chunk of a transcript."""

_SUBJECT_HEADING = "The new turn [{chunk_id}], and its user message:"
"""Introduces the chunk under decision."""

_ANSWER_HEADING = "Answer with one JSON object and nothing else, in exactly this shape:"
"""Introduces the answer shape."""

_ANSWER_SHAPE = (
    '{"decision": "<the id of the intent the new turn continues, or NEW>",\n'
    ' "label": "<a short label for the new intent, only when the decision is NEW>"}'
)
"""The shape `_read` parses, written out for the model.

Over two lines so that both fields stay readable; JSON does not care, and the
answer is parsed rather than compared.
"""

_RETRY_NOTICE = (
    "Your previous answer could not be read: {reason}. Answer with one JSON object\n"
    "and nothing else."
)
"""What is appended to the prompt when the previous answer was unreadable.

The reason is named: a model that is told *what* was wrong can fix it, and a
retry that repeated the identical prompt would only be a second sample.
"""

_SECTION_SEPARATOR = "\n\n"
"""What the prompt's sections are joined with."""

_CODE_FENCE = re.compile(r"\A```[A-Za-z0-9_+-]*\n(?P<body>.*?)\n?```\Z", re.DOTALL)
"""A fenced code block around the whole answer, which is stripped before parsing.

The one tolerated deviation from "JSON and nothing else", because it is both the
most frequent one and unambiguous. Prose *around* the JSON is not searched for an
object: guessing which brace the model meant would be a decision nobody made.
"""

_ANSWER_EXCERPT_MAX_LENGTH = 200
"""How much of an unreadable answer the error message shows."""

_EXCERPT_ELLIPSIS = "…"
"""What marks an answer the error message cut short."""


@runtime_checkable
class ModelClient(Protocol):
    """The injected dependency: something that answers a prompt with text.

    Deliberately the smallest possible hole. One method, a string in and a
    string out — no model name, no temperature, no token budget, no retry, no
    credential. Everything a real backend needs lives in the caller's client,
    so this module needs neither a transport nor a configuration, a study runner
    can plug in whatever it already has, and a test plugs in a canned list.

    A `Protocol` rather than a base class, for the reason `IntentAdapter` is
    one: a client inherits nothing. It is `runtime_checkable`, which compares
    the presence of `complete` and nothing about its signature.
    """

    def complete(self, prompt: str) -> str:
        """Return the model's answer to `prompt`.

        Args:
            prompt: The whole prompt, as `build_prompt` wrote it.

        Returns:
            The model's answer, verbatim. The adapter reads it; an answer that
            cannot be read is retried and then refused, so a client need not
            validate anything — but it must not swallow its own failures: an
            exception it raises reaches the caller of `classify` unwrapped,
            which is what tells a timeout apart from a classification problem.
        """
        ...


class UnreadableAnswerError(AdapterContractError):
    """The model did not answer with a readable decision within the bound.

    The loud end of the bounded retry: every attempt was spent and none of the
    answers could be read as a decision. An `AdapterContractError` — the answer
    is a bad value *and* a failure of the stage — so `except ClassifyError`
    catches it like every other refusal of the stage, and the attributes carry
    what the message is built from, so a test asserts on them instead of on
    prose.

    Attributes:
        chunk_id: The chunk the decision was wanted for.
        answers: What the client returned, per attempt, in order. Typed as
            objects rather than strings because an answer that is no string at
            all is exactly one of the cases that lands here.
        reasons: Why each of those answers could not be read, in the same order.
    """

    def __init__(self, chunk_id: str, answers: Sequence[object], reasons: Sequence[str]) -> None:
        """Build the error from the chunk, the answers and their reasons.

        Args:
            chunk_id: The chunk the decision was wanted for.
            answers: What the client returned, per attempt.
            reasons: Why each answer could not be read.
        """
        self.chunk_id = chunk_id
        self.answers = tuple(answers)
        self.reasons = tuple(reasons)
        reason = self.reasons[-1] if self.reasons else "no answer was recorded"
        shown = _excerpt(self.answers[-1]) if self.answers else ""
        super().__init__(
            f"the model did not answer with a readable decision for chunk {chunk_id!r} in "
            f"{self.attempts} attempt(s); last reason: {reason}; last answer: {shown!r}"
        )

    @property
    def attempts(self) -> int:
        """Return how many answers were asked for and refused."""
        return len(self.answers)


@dataclass(frozen=True, slots=True)
class LlmCall:
    """One question the adapter asked the model, and the decision it read.

    The record `FakeCall` is for the scripted adapter: what was asked, what came
    back and what it was read as — the hook the evaluation harness (#19) reads a
    run back from, and what a test asserts the prompt and the retry on. Only a
    call that produced a decision is recorded; a call that failed raises, and the
    error carries the answers instead.

    Attributes:
        chunk_id: `Chunk.id` the decision was made for.
        candidates: The candidates the model was shown, in the order they were
            offered.
        prompts: The prompt of every attempt, in order — the first one and, when
            an answer had to be retried, the one naming what was wrong.
        answers: What the client returned per attempt, in order. The last one is
            the answer that was read; the ones before it were unreadable, and
            an answer that is no string is one of the ways to be unreadable.
        decision: The decision that was read out of the last answer.
        label: The label the model suggested for a new chain, whitespace
            collapsed, or `None` when it suggested none or none that holds a
            word. **Advisory only**: the store names a new chain itself and T6
            writes the real label, so this is kept for the harness and goes
            nowhere else.
    """

    chunk_id: str
    candidates: tuple[ChainCandidate, ...]
    prompts: tuple[str, ...]
    answers: tuple[object, ...]
    decision: Decision
    label: str | None = None

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Return the ids of the candidates that were on offer."""
        return tuple(candidate.id for candidate in self.candidates)

    @property
    def attempts(self) -> int:
        """Return how many answers it took to reach the decision."""
        return len(self.answers)


def build_prompt(
    chunk: ChunkUnderDecision,
    candidates: Sequence[ChainCandidate],
    *,
    retry_reason: str | None = None,
) -> str:
    """Return the prompt that asks for one classification decision.

    The fixed prompt of the method, and a pure function of its two arguments, so
    the same question produces the same prompt in every run of the study. It
    shows the model each candidate's id, label and last user message and the new
    turn's user message, all verbatim: both halves of the question are the same
    kind of text, which is what the adapter interface hands over (#16) and all
    it hands over.

    Transcript text is placed between `<<<` and `>>>` on lines of their own and
    is **not** escaped — a transcript is the user's own prose, and rewriting it
    would classify something the user never wrote.

    Args:
        chunk: The chunk under decision; its id names the turn and its user
            message is the text to classify. An empty user message is marked as
            absent rather than left blank.
        candidates: The open chains, in the order they were opened. Empty for
            the first chunk of a transcript, where the prompt says so instead of
            offering an empty list.
        retry_reason: Why the previous answer could not be read, when this is a
            retry. It is appended as a notice, so the model is asked the same
            question and told what was wrong with its last answer. A reason that
            holds no word adds no notice.

    Returns:
        The prompt, without a trailing newline.
    """
    sections = [_PREAMBLE]
    if candidates:
        sections.append(_CANDIDATES_HEADING)
        sections.extend(
            f"[{candidate.id}] {candidate.label}\n{_message_block(candidate.last_user_message)}"
            for candidate in candidates
        )
    else:
        sections.append(_NO_CANDIDATES)
    sections.append(
        f"{_SUBJECT_HEADING.format(chunk_id=chunk.id)}\n{_message_block(chunk.user_message)}"
    )
    sections.append(_ANSWER_HEADING)
    sections.append(_ANSWER_SHAPE)
    if retry_reason and retry_reason.strip():
        sections.append(_RETRY_NOTICE.format(reason=retry_reason.strip()))
    return _SECTION_SEPARATOR.join(sections)


def _message_block(text: str) -> str:
    """Return transcript text as a delimited block, marking an absent message.

    Args:
        text: The message's content, verbatim; may be empty.

    Returns:
        The text between the delimiters, or the absence marker between them.
    """
    return f"{_MESSAGE_OPEN}\n{text or _NO_MESSAGE}\n{_MESSAGE_CLOSE}"


def _excerpt(answer: object) -> str:
    """Return an answer as text, cut short for an error message.

    Args:
        answer: What the client returned.

    Returns:
        The answer itself when it is a short string, its representation when it
        is no string, cut at `_ANSWER_EXCERPT_MAX_LENGTH` characters.
    """
    text = answer if isinstance(answer, str) else repr(answer)
    if len(text) <= _ANSWER_EXCERPT_MAX_LENGTH:
        return text
    return text[: _ANSWER_EXCERPT_MAX_LENGTH - len(_EXCERPT_ELLIPSIS)] + _EXCERPT_ELLIPSIS


class _RetryableAnswerError(Exception):
    """An answer that is no decision, carried out of the reading step.

    Module-private and never raised past `decide`: it says "ask again", and the
    retry loop turns the last one into `UnreadableAnswerError`. A sentinel
    return value would have to be told apart from a decision at every step of
    the reading.

    Attributes:
        reason: Why the answer could not be read, phrased to be read inside
            `_RETRY_NOTICE`.
    """

    def __init__(self, reason: str) -> None:
        """Build the signal from the reason the answer could not be read.

        Args:
            reason: Why the answer could not be read.
        """
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class _ReadAnswer:
    """What one readable answer said.

    Attributes:
        chain_id: The chain the model named, or `None` when it answered `NEW`.
            Of the form `C<n>`; whether such a chain exists is decided against
            the candidates, not here.
        label: The label it suggested, whitespace collapsed, or `None`.
    """

    chain_id: str | None
    label: str | None


def _read(answer: object) -> _ReadAnswer:
    """Read one model answer, or say why it cannot be read.

    Args:
        answer: What the client returned.

    Returns:
        What the answer said.

    Raises:
        _RetryableAnswerError: If the answer is not one JSON object with a usable
            `"decision"` field, or carries a `"label"` that is no string. Fields
            the shape does not mention are ignored — a model that adds its
            reasoning has still answered.
    """
    if not isinstance(answer, str):
        raise _RetryableAnswerError(f"the answer is a {type(answer).__name__} and not a string")
    payload = _load_json(answer)
    if not isinstance(payload, dict):
        raise _RetryableAnswerError(
            f"the answer is a {type(payload).__name__} and not a JSON object"
        )
    if "decision" not in payload:
        raise _RetryableAnswerError('the answer has no "decision" field')
    written = payload["decision"]
    if not isinstance(written, str):
        raise _RetryableAnswerError(
            f'the "decision" field is a {type(written).__name__} and not a string'
        )
    return _ReadAnswer(chain_id=_read_decision(written), label=_read_label(payload.get("label")))


def _load_json(answer: str) -> object:
    """Return the JSON value an answer holds.

    Args:
        answer: The answer as the client returned it.

    Returns:
        The parsed JSON value, whatever its type.

    Raises:
        _RetryableAnswerError: If the answer is not JSON. Surrounding whitespace and
            a code fence around the whole answer are stripped first.
    """
    text = answer.strip()
    fenced = _CODE_FENCE.match(text)
    if fenced:
        text = fenced.group("body")
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise _RetryableAnswerError(f"the answer is not JSON ({error.msg})") from error


def _read_decision(written: str) -> str | None:
    """Return the chain id the `"decision"` field names, or `None` for `NEW`.

    Surrounding whitespace and the case of the answer are tolerated: a model
    that writes `" new "` or `"c1"` has answered unambiguously, and a retry
    would only cost a call. Everything else is unreadable — `is_chain_id` is the
    canonical schema's own check (#16), so the adapter cannot invent an id the
    schema would reject.

    Args:
        written: The field's value.

    Returns:
        The chain id in its canonical spelling, or `None` when the answer is
        `NEW`.

    Raises:
        _RetryableAnswerError: If the field is blank, or names neither `NEW` nor a
            chain id of the form `C<n>`.
    """
    stripped = written.strip()
    if not stripped:
        raise _RetryableAnswerError('the "decision" field is empty')
    normalized = stripped.upper()
    if normalized == _NEW_ANSWER:
        return None
    if is_chain_id(normalized):
        return normalized
    raise _RetryableAnswerError(
        f'the "decision" field is neither "{_NEW_ANSWER}" nor a chain id of the form '
        f"C<n>; got {stripped!r}"
    )


def _read_label(label: object) -> str | None:
    """Return the suggested label, or `None` when none was suggested.

    A label that holds no word — absent, `null`, empty, whitespace — is read as
    no label. The store refuses such a label (`BlankChainLabelError`), so
    passing it on would take a run down over a field that decides nothing; and
    refusing the model's *decision* over it would be worse, because the decision
    itself was readable. Whitespace is collapsed, so a label stays one line.

    Args:
        label: The `"label"` field's value, if any.

    Returns:
        The collapsed label, or `None`.

    Raises:
        _RetryableAnswerError: If the field is present and is no string. The shape
            asks for a string; an object or a number there means the model did
            not follow it, and that is what the retry is for.
    """
    if label is None:
        return None
    if not isinstance(label, str):
        raise _RetryableAnswerError(
            f'the "label" field is a {type(label).__name__} and not a string'
        )
    return " ".join(label.split()) or None


class LlmAdapter:
    """An `IntentAdapter` that asks a model, through an injected client.

    Args:
        client: The model client. Anything with `complete(prompt) -> str`; this
            module ships none, opens no connection and reads no credential.
        max_attempts: How often one chunk may be asked about, the first attempt
            included. Only an answer that cannot be *read* costs a further
            attempt; an answer naming a chain that does not exist is rejected
            without one.

    Raises:
        ValueError: If `max_attempts` is smaller than 1 — a caller's own bad
            literal, which the package deliberately leaves outside
            `ClassifyError`.
    """

    def __init__(self, client: ModelClient, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> None:
        """Build the adapter from its client and its attempt bound."""
        if max_attempts < 1:
            raise ValueError(f"an adapter needs at least one attempt per chunk; got {max_attempts}")
        self._client = client
        self._max_attempts = max_attempts
        self._calls: list[LlmCall] = []

    def __repr__(self) -> str:
        """Return a representation naming the client and the attempt bound."""
        return f"{type(self).__name__}(client={self._client!r}, max_attempts={self._max_attempts})"

    @property
    def client(self) -> ModelClient:
        """Return the injected model client."""
        return self._client

    @property
    def max_attempts(self) -> int:
        """Return how often one chunk may be asked about at most."""
        return self._max_attempts

    @property
    def calls(self) -> tuple[LlmCall, ...]:
        """Return every call that produced a decision, in order."""
        return tuple(self._calls)

    def decide(self, chunk: ChunkUnderDecision, candidates: Sequence[ChainCandidate]) -> Decision:
        """Ask the model whether `chunk` continues one of `candidates`.

        Args:
            chunk: The chunk to classify — its id and its own user message.
            candidates: The open chains, in the order they were opened.

        Returns:
            The decision, as a `Decision`: the normal form, because it is the
            only one that can carry a confidence for the harness (#19). The
            fixed prompt of this ticket does not ask the model for one, so the
            confidence is `None` and is never invented here.

        Raises:
            UnreadableAnswerError: If no answer within the bound could be read
                as a decision.
            UnknownChainError: If an answer named a chain that was not among the
                candidates. Not retried: the model answered in the shape it was
                asked for and decided something that does not exist, and the
                refusal belongs where the answer arrived.
            Exception: Whatever the client raises, unwrapped — only the backend
                knows what its own timeout or refusal means.
        """
        offered = tuple(candidates)
        prompts: list[str] = []
        answers: list[object] = []
        reasons: list[str] = []
        reason: str | None = None
        for _attempt in range(self._max_attempts):
            prompt = build_prompt(chunk, offered, retry_reason=reason)
            prompts.append(prompt)
            answer = self._client.complete(prompt)
            answers.append(answer)
            try:
                read = _read(answer)
            except _RetryableAnswerError as unreadable:
                reason = unreadable.reason
                reasons.append(reason)
                continue
            decision = _decide(read, offered)
            self._calls.append(
                LlmCall(
                    chunk_id=chunk.id,
                    candidates=offered,
                    prompts=tuple(prompts),
                    answers=tuple(answers),
                    decision=decision,
                    label=read.label,
                )
            )
            return decision
        raise UnreadableAnswerError(chunk.id, answers, reasons)


def _decide(read: _ReadAnswer, candidates: Sequence[ChainCandidate]) -> Decision:
    """Turn a readable answer into a decision, against the chains on offer.

    Args:
        read: What the answer said.
        candidates: The chains that were offered.

    Returns:
        The decision.

    Raises:
        UnknownChainError: If the answer named a chain the candidates do not
            hold — including any chain at all when none was offered.
    """
    if read.chain_id is None:
        return Decision.new()
    known = tuple(candidate.id for candidate in candidates)
    if read.chain_id not in known:
        raise UnknownChainError(read.chain_id, known)
    # Through `as_decision` and not `Decision.extend`, although `_read_decision`
    # already checked the shape: a model's answer must never be able to leave
    # this stage as a plain `ValueError`, and that is exactly what constructing
    # a `Decision` from a wrong-shaped id would be (see `Decision`'s docstring).
    # The check above and this one are the same guard twice, on purpose — one of
    # them may be refactored, and the family must not depend on which.
    return as_decision(read.chain_id)
