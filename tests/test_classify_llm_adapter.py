"""The LLM backend: the fixed prompt, the validated answer, the bounded retry.

The ticket's own tests (#17) plus what has to hold once a real model sits behind
the client. Four kinds of test live here:

- **The prompt.** `EXPECTED_PROMPT` and `EXPECTED_FIRST_TURN_PROMPT` pin it
  character for character, because the prompt is *fixed* by the scope: the study
  compares runs, and a prompt that drifts between them compares two methods
  instead of two transcripts. The structural tests next to them say what the
  golden string is an instance of.
- **The answers.** One table of unreadable answers and one of tolerated
  spellings, both parametrized, so a new case is a row rather than a function.
- **The loud failures.** An exhausted retry bound, a chain the candidates do not
  hold, a chain id of the wrong shape, and an error the client itself raised —
  each has to fail as the stage's own family or, for the last one, deliberately
  not.
- **The classification.** The adapter behind `classify`, against the scripted
  fake as its oracle: the same canned answers must produce the same chains the
  fake produces, twice, which is T4's first acceptance criterion.

No test here calls a model: the client is a stub that answers from a canned
list, and two tests state that as a property of the module rather than as a
habit — its imports and a run that would fail if a socket were opened.
"""

import ast
import socket
from pathlib import Path

import pytest

from idc.chunk import chunk_session
from idc.classify import (
    DEFAULT_MAX_ATTEMPTS,
    NEW,
    AdapterContractError,
    ChainCandidate,
    ChunkUnderDecision,
    ClassifyError,
    Decision,
    FakeAdapter,
    IntentAdapter,
    LlmAdapter,
    LlmCall,
    ModelClient,
    UnknownChainError,
    UnreadableAnswerError,
    build_prompt,
    classify,
)
from idc.classify import llm as llm_module

SUBJECT = ChunkUnderDecision(id="K3", user_message="Back to the sampler: the weights are off")
"""The chunk under decision the golden prompts are built from."""

CANDIDATES = (
    ChainCandidate(
        id="C1",
        label="fix the failing sampler test",
        last_user_message="`test_weighted_pick` fails",
    ),
    ChainCandidate(
        id="C2",
        label="rename the config key",
        last_user_message="Unrelated: rename `retry_limit`",
    ),
)
"""The open chains the golden prompt offers, in the order they were opened."""

EXPECTED_PROMPT = """\
You classify one turn of a coding-agent transcript into a user intent.

An intent is what the user is trying to get done. The new turn continues an open
intent when it works towards the same goal — a follow-up, a correction, another
attempt at the same task — and it starts a new intent when it does not.

Open intents, each with its id, its label and its last user message:

[C1] fix the failing sampler test
<<<
`test_weighted_pick` fails
>>>

[C2] rename the config key
<<<
Unrelated: rename `retry_limit`
>>>

The new turn [K3], and its user message:
<<<
Back to the sampler: the weights are off
>>>

Answer with one JSON object and nothing else, in exactly this shape:

{"decision": "<the id of the intent the new turn continues, or NEW>",
 "label": "<a short label for the new intent, only when the decision is NEW>"}"""
"""The prompt, in full, for two open chains — the shape the study runs with."""

EXPECTED_FIRST_TURN_PROMPT = """\
You classify one turn of a coding-agent transcript into a user intent.

An intent is what the user is trying to get done. The new turn continues an open
intent when it works towards the same goal — a follow-up, a correction, another
attempt at the same task — and it starts a new intent when it does not.

There is no open intent yet, so the new turn starts one.

The new turn [K1], and its user message:
<<<
Make the retry backoff configurable
>>>

Answer with one JSON object and nothing else, in exactly this shape:

{"decision": "<the id of the intent the new turn continues, or NEW>",
 "label": "<a short label for the new intent, only when the decision is NEW>"}"""
"""The same prompt for the first chunk of a transcript, where nothing is open."""

DIGITS_OVER_THE_LIMIT = "1" * 6000
"""An integer literal longer than CPython's `sys.int_max_str_digits` (4300).

`json.loads` refuses such a literal with a **plain `ValueError`** and not a
`JSONDecodeError`, so a reader that caught only the latter let it out — and a
repetition loop writing one digit too many is a real failure mode of a model,
not a contrived one.
"""

NESTED_DEEPER_THAN_IT_PARSES = "[" * 200_000
"""An answer of nothing but open brackets.

CPython's `json` puts no limit on nesting, so this exhausts the parser's stack
and raises a `RecursionError` — which is no `ValueError` at all and so needs its
own arm in the reader.
"""

UNREADABLE_ANSWERS = (
    pytest.param("not json at all", "not JSON", id="prose"),
    pytest.param("", "not JSON", id="empty"),
    pytest.param("[1, 2]", "and not a JSON object", id="json-array"),
    pytest.param('"NEW"', "and not a JSON object", id="json-string"),
    pytest.param("42", "and not a JSON object", id="json-number"),
    pytest.param("{}", 'no "decision" field', id="no-decision-field"),
    pytest.param('{"label": "x"}', 'no "decision" field', id="only-a-label"),
    pytest.param('{"decision": null}', "and not a string", id="null-decision"),
    pytest.param('{"decision": 1}', "and not a string", id="numeric-decision"),
    pytest.param('{"decision": ["C1"]}', "and not a string", id="list-decision"),
    pytest.param('{"decision": "  "}', 'the "decision" field is empty', id="blank-decision"),
    pytest.param('{"decision": "K1"}', "neither", id="chunk-id"),
    pytest.param('{"decision": "C0"}', "neither", id="chain-zero"),
    pytest.param('{"decision": "chain one"}', "neither", id="prose-decision"),
    pytest.param('{"decision": "NEW", "label": 5}', "and not a string", id="numeric-label"),
    pytest.param(None, "and not a string", id="no-string-at-all"),
    pytest.param(DIGITS_OVER_THE_LIMIT, "not readable as JSON", id="digits-over-the-limit"),
    pytest.param(
        '{"decision": ' + DIGITS_OVER_THE_LIMIT + "}",
        "not readable as JSON",
        id="a-decision-of-digits-over-the-limit",
    ),
    pytest.param(NESTED_DEEPER_THAN_IT_PARSES, "not readable as JSON", id="nested-too-deeply"),
)
"""Every answer the adapter must refuse to read, with a fragment of its reason.

The reason is what the retry prompt names, so it is asserted on: a model that is
told *what* was wrong with its last answer can fix it, and a test that only
asserted "it raised" would not notice the reason going empty.
"""

TOLERATED_NEW_ANSWERS = (
    pytest.param('{"decision": "NEW"}', id="plain"),
    pytest.param('{"decision": " NEW "}', id="padded"),
    pytest.param('{"decision": "new"}', id="lowercase"),
    pytest.param('{"decision": "New"}', id="capitalized"),
    pytest.param('{"decision": "NEW", "reason": "a new topic"}', id="extra-field"),
    pytest.param('```json\n{"decision": "NEW"}\n```', id="fenced"),
    pytest.param('```\n{"decision": "NEW"}\n```', id="fenced-without-a-language"),
    pytest.param('```json\n{"decision": "NEW"}```', id="fenced-without-a-closing-newline"),
    pytest.param('\n\n{"decision": "NEW"}\n', id="surrounded-by-blank-lines"),
)
"""Spellings of `NEW` the adapter reads rather than refuses.

A real model pads, lower-cases and fences its JSON; each of those costs a retry
if it is not tolerated, and none of them is ambiguous.
"""

TOLERATED_EXTEND_ANSWERS = (
    pytest.param('{"decision": "C2"}', id="plain"),
    pytest.param('{"decision": " C2 "}', id="padded"),
    pytest.param('{"decision": "c2"}', id="lowercase"),
    pytest.param('{"decision": "C2", "label": "ignored here"}', id="with-a-label"),
    pytest.param('```json\n{"decision": "C2"}\n```', id="fenced"),
)
"""Spellings of a chain id the adapter reads as extending that chain."""

BLANK_LABELS = (
    pytest.param('{"decision": "NEW"}', id="absent"),
    pytest.param('{"decision": "NEW", "label": null}', id="null"),
    pytest.param('{"decision": "NEW", "label": ""}', id="empty"),
    pytest.param('{"decision": "NEW", "label": "   "}', id="spaces"),
    pytest.param('{"decision": "NEW", "label": "\\n\\t"}', id="whitespace"),
)
"""Labels that hold no word. The store refuses such a label (#16), so the
adapter may not pass one on — and may not refuse the decision over it either."""

ALLOWED_IMPORTS = {"collections", "dataclasses", "idc", "json", "re", "typing"}
"""Every top-level module `idc.classify.llm` may import.

A whitelist rather than a blacklist: the ticket's hard constraint is that the
backend reaches nothing by itself, and an allowed list cannot be circumvented by
a package nobody thought to forbid.
"""

FORBIDDEN_IMPORTS = {
    "aiohttp",
    "anthropic",
    "boto3",
    "google",
    "http",
    "httpx",
    "openai",
    "os",
    "requests",
    "socket",
    "ssl",
    "subprocess",
    "urllib",
    "urllib3",
}
"""The imports that would make the module reach out on its own.

Redundant against `ALLOWED_IMPORTS` on purpose: this set is what names the risk,
so the test that fails says *why* the import may not be there.
"""

FORBIDDEN_FRAGMENTS = ("os.environ", "getenv(", "urlopen", "api_key", "Bearer ")
"""Source fragments that would mean the module handles a credential or a
connection itself. The client is injected; this module ships none."""


class StubClient:
    """A `ModelClient` that answers from a canned list, recording the prompts.

    The whole decision backend of this test module. It is exhausted loudly
    rather than repeating its last answer, for the reason `FakeAdapter` is: an
    adapter that asked one time more than the test wrote down would otherwise
    pass for a reason the test never stated.

    Args:
        answers: What the model is to answer, in order.
    """

    def __init__(self, *answers: object) -> None:
        """Keep the canned answers and start with an empty prompt record."""
        self.answers = list(answers)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        """Record the prompt and return the next canned answer."""
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(
                f"the stub client was asked for answer {len(self.prompts)} and holds none"
            )
        return self.answers.pop(0)  # type: ignore[return-value]


class ExplodingClient:
    """A client whose own call fails — a backend error, not a stage error."""

    def complete(self, prompt: str) -> str:
        """Raise the way a real client's transport would."""
        raise RuntimeError("the model did not answer")


def adapter_for(*answers: object, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> LlmAdapter:
    """Return an adapter whose client answers with `answers`, in order."""
    return LlmAdapter(StubClient(*answers), max_attempts=max_attempts)


def decide(*answers: object, candidates: tuple[ChainCandidate, ...] = CANDIDATES) -> Decision:
    """Return the decision the adapter reads out of `answers`."""
    return adapter_for(*answers, max_attempts=len(answers)).decide(SUBJECT, candidates)


def test_the_prompt_is_the_one_the_study_runs_with():
    # Pinned in full, not by fragments: "fixed prompt" is the scope's own word,
    # and a run of the study that silently reworded it is not comparable with
    # the run before it.
    assert build_prompt(SUBJECT, CANDIDATES) == EXPECTED_PROMPT


def test_the_prompt_of_the_first_turn_says_that_nothing_is_open():
    subject = ChunkUnderDecision(id="K1", user_message="Make the retry backoff configurable")

    assert build_prompt(subject, ()) == EXPECTED_FIRST_TURN_PROMPT


def test_the_prompt_carries_every_candidate_in_the_order_it_was_offered():
    # The structural statement the two golden strings are instances of, so a
    # third candidate needs no third golden string.
    prompt = build_prompt(SUBJECT, CANDIDATES)

    positions = [prompt.index(f"[{candidate.id}]") for candidate in CANDIDATES]

    assert positions == sorted(positions)
    for candidate in CANDIDATES:
        assert candidate.label in prompt
        assert candidate.last_user_message in prompt


def test_the_prompt_carries_the_user_message_of_the_chunk_under_decision():
    assert SUBJECT.user_message in build_prompt(SUBJECT, CANDIDATES)
    assert f"[{SUBJECT.id}]" in build_prompt(SUBJECT, CANDIDATES)


def test_a_multi_line_user_message_reaches_the_prompt_verbatim():
    subject = ChunkUnderDecision(id="K2", user_message="first line\n\n  second line")

    assert "first line\n\n  second line" in build_prompt(subject, CANDIDATES)


def test_a_turn_without_a_user_message_is_marked_as_having_none():
    # `ChunkUnderDecision.user_message` is empty for a chunk of subagent work
    # and for a user turn that carried no text (#16). The prompt may not then
    # show the model an empty block it has to guess the meaning of.
    subject = ChunkUnderDecision(id="K2")

    prompt = build_prompt(subject, ())

    assert "(no user message)" in prompt


def test_a_candidate_without_a_last_user_message_is_marked_the_same_way():
    candidate = ChainCandidate(id="C1", label="carried over from an earlier run")

    prompt = build_prompt(SUBJECT, (candidate,))

    assert "(no user message)" in prompt


def test_the_prompt_asks_for_the_json_the_adapter_parses():
    # The one place prompt and parser have to agree; asserted so that renaming a
    # field in one of them fails here instead of in production.
    prompt = build_prompt(SUBJECT, CANDIDATES)

    assert '"decision"' in prompt
    assert '"label"' in prompt
    assert "NEW" in prompt


def test_the_same_question_builds_the_same_prompt():
    assert build_prompt(SUBJECT, CANDIDATES) == build_prompt(SUBJECT, CANDIDATES)


def test_the_retry_prompt_repeats_the_question_and_names_what_was_wrong():
    prompt = build_prompt(SUBJECT, CANDIDATES, retry_reason="the answer is not JSON")

    assert prompt.startswith(EXPECTED_PROMPT)
    assert "the answer is not JSON" in prompt[len(EXPECTED_PROMPT) :]


def test_a_blank_retry_reason_adds_no_notice():
    assert build_prompt(SUBJECT, CANDIDATES, retry_reason="   ") == EXPECTED_PROMPT


@pytest.mark.parametrize("answer", TOLERATED_NEW_ANSWERS)
def test_an_answer_of_new_opens_a_chain(answer):
    assert decide(answer) == Decision.new()


@pytest.mark.parametrize("answer", TOLERATED_EXTEND_ANSWERS)
def test_an_answer_naming_a_candidate_extends_that_chain(answer):
    assert decide(answer) == Decision.extend("C2")


def test_the_decision_is_the_normal_form_and_invents_no_confidence():
    # A `Decision` rather than a bare id, because that is the form that can
    # carry a confidence (#19) — and `None`, because the fixed prompt of this
    # ticket does not ask the model for one.
    decision = decide('{"decision": "C1"}')

    assert isinstance(decision, Decision)
    assert decision.confidence is None


@pytest.mark.parametrize(("answer", "reason"), UNREADABLE_ANSWERS)
def test_an_unusable_answer_is_refused_with_the_reason_it_is_unusable(answer, reason):
    adapter = adapter_for(answer, max_attempts=1)

    with pytest.raises(UnreadableAnswerError) as caught:
        adapter.decide(SUBJECT, CANDIDATES)

    assert reason in caught.value.reasons[0]
    assert caught.value.attempts == 1


@pytest.mark.parametrize(("answer", "reason"), UNREADABLE_ANSWERS)
def test_an_unusable_answer_is_a_failure_of_the_stage(answer, reason):
    # One `except ClassifyError` is what T10's CLI is written against, and the
    # answer is a bad value as well, exactly like `AdapterContractError` (#16).
    adapter = adapter_for(answer, max_attempts=1)

    with pytest.raises(ClassifyError) as caught:
        adapter.decide(SUBJECT, CANDIDATES)

    assert isinstance(caught.value, AdapterContractError)
    assert isinstance(caught.value, ValueError)


def test_an_invalid_json_answer_is_retried_and_the_second_answer_is_used():
    client = StubClient("not json at all", '{"decision": "C1"}')
    adapter = LlmAdapter(client)

    decision = adapter.decide(SUBJECT, CANDIDATES)

    assert decision == Decision.extend("C1")
    assert len(client.prompts) == 2
    assert client.prompts[0] == EXPECTED_PROMPT
    assert "could not be read" in client.prompts[1]
    assert "not JSON" in client.prompts[1]


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(DIGITS_OVER_THE_LIMIT, id="digits-over-the-limit"),
        pytest.param(NESTED_DEEPER_THAN_IT_PARSES, id="nested-too-deeply"),
    ],
)
def test_a_degenerate_answer_is_retried_like_any_other_unreadable_answer(answer):
    # Deliberate, and the reason it is a test: a degenerate answer is
    # *unreadable*, and unreadable is exactly what the retry is for. Such an
    # answer may well repeat, but then it costs the one extra call the bound
    # allows and ends in `UnreadableAnswerError` like every other refusal —
    # whereas a special case would split one documented rule into two for no
    # gain the study can measure.
    client = StubClient(answer, '{"decision": "C1"}')
    adapter = LlmAdapter(client)

    assert adapter.decide(SUBJECT, CANDIDATES) == Decision.extend("C1")
    assert len(client.prompts) == 2
    assert "could not be read" in client.prompts[1]


def test_the_retried_call_keeps_both_answers_in_its_record():
    adapter = adapter_for("not json at all", '{"decision": "NEW"}')

    adapter.decide(SUBJECT, CANDIDATES)

    call = adapter.calls[0]
    assert call.answers == ("not json at all", '{"decision": "NEW"}')
    assert call.attempts == 2
    assert len(call.prompts) == 2


def test_the_retry_is_bounded_and_the_exhausted_bound_is_loud():
    client = StubClient("first rubbish", "second rubbish")
    adapter = LlmAdapter(client, max_attempts=2)

    with pytest.raises(UnreadableAnswerError) as caught:
        adapter.decide(SUBJECT, CANDIDATES)

    error = caught.value
    assert error.chunk_id == "K3"
    assert error.attempts == 2
    assert error.answers == ("first rubbish", "second rubbish")
    assert len(error.reasons) == 2
    assert "2 attempt" in str(error)
    assert len(client.prompts) == 2, "the bound is the number of calls, not of parses"
    assert adapter.calls == (), "a decide that failed records no decision"


def test_a_long_unreadable_answer_is_cut_short_in_the_message_but_not_in_the_record():
    # A model that answers with a page of prose must not put that page into a
    # traceback — while the harness and a human debugging the run still need the
    # answer in full.
    rubbish = "no JSON here, " * 100
    adapter = adapter_for(rubbish, max_attempts=1)

    with pytest.raises(UnreadableAnswerError) as caught:
        adapter.decide(SUBJECT, CANDIDATES)

    assert caught.value.answers == (rubbish,)
    assert len(str(caught.value)) < len(rubbish)
    assert "…" in str(caught.value)


def test_an_error_that_recorded_no_answer_still_says_so():
    # The bound is at least one attempt, so this cannot arise from `decide`; the
    # guard is what keeps the error itself from failing while it is being built.
    error = UnreadableAnswerError("K1", (), ())

    assert error.attempts == 0
    assert "no answer was recorded" in str(error)


def test_the_default_bound_is_one_retry():
    assert DEFAULT_MAX_ATTEMPTS == 2


def test_the_bound_is_the_adapters_own():
    client = StubClient("one", "two", '{"decision": "NEW"}')
    adapter = LlmAdapter(client, max_attempts=3)

    assert adapter.decide(SUBJECT, CANDIDATES) == Decision.new()
    assert adapter.max_attempts == 3
    assert len(client.prompts) == 3


def test_a_single_attempt_is_not_retried():
    client = StubClient("rubbish")
    adapter = LlmAdapter(client, max_attempts=1)

    with pytest.raises(UnreadableAnswerError):
        adapter.decide(SUBJECT, CANDIDATES)

    assert len(client.prompts) == 1


@pytest.mark.parametrize("attempts", [0, -1])
def test_an_adapter_without_an_attempt_is_a_caller_mistake(attempts):
    # The carve-out the package documents: a value the caller wrote wrongly is a
    # plain `ValueError`, not a failure of the stage.
    with pytest.raises(ValueError, match="at least one attempt") as caught:
        LlmAdapter(StubClient(), max_attempts=attempts)

    assert not isinstance(caught.value, ClassifyError)


def test_a_chain_the_candidates_do_not_hold_is_rejected_where_it_was_answered():
    # The store would catch it too, but then the traceback would name the store:
    # the adapter is where the model's answer arrived, so it is where the answer
    # is refused.
    client = StubClient('{"decision": "C9"}')
    adapter = LlmAdapter(client)

    with pytest.raises(UnknownChainError) as caught:
        adapter.decide(SUBJECT, CANDIDATES)

    assert caught.value.chain_id == "C9"
    assert caught.value.known == ("C1", "C2")
    assert "C1, C2" in str(caught.value)
    assert len(client.prompts) == 1, "a chain that does not exist is no formatting slip"
    assert adapter.calls == ()


def test_a_chain_answered_where_none_is_open_is_rejected():
    adapter = adapter_for('{"decision": "C1"}')

    with pytest.raises(UnknownChainError) as caught:
        adapter.decide(ChunkUnderDecision(id="K1", user_message="first turn"), ())

    assert caught.value.known == ()


def test_a_wrong_shaped_chain_id_never_escapes_as_a_bare_value_error():
    # The trap `Decision`'s docstring marks: a model asked for `C<n>` writes
    # `K1`, and `Decision.extend("K1")` would be a plain `ValueError` instead of
    # a failure of the stage. The adapter reads its answers through the
    # contract's own helpers, so this stays catchable.
    adapter = adapter_for('{"decision": "K1"}', max_attempts=1)

    with pytest.raises(ClassifyError):
        adapter.decide(SUBJECT, CANDIDATES)


def test_an_error_in_the_client_reaches_the_caller_unwrapped():
    # The second documented carve-out: only the backend knows what its own
    # timeout means, and a transport failure is no classification problem.
    adapter = LlmAdapter(ExplodingClient())

    with pytest.raises(RuntimeError) as caught:
        adapter.decide(SUBJECT, CANDIDATES)

    assert not isinstance(caught.value, ClassifyError)


def test_the_label_of_a_new_chain_is_recorded_for_the_evaluation_harness():
    adapter = adapter_for('{"decision": "NEW", "label": "add a retry backoff"}')

    adapter.decide(SUBJECT, CANDIDATES)

    assert adapter.calls[0].label == "add a retry backoff"


def test_a_label_is_collapsed_to_one_line():
    adapter = adapter_for('{"decision": "NEW", "label": "  add a\\n  retry backoff  "}')

    adapter.decide(SUBJECT, CANDIDATES)

    assert adapter.calls[0].label == "add a retry backoff"


@pytest.mark.parametrize("answer", BLANK_LABELS)
def test_a_label_that_holds_no_word_is_read_as_no_label(answer):
    adapter = adapter_for(answer)

    decision = adapter.decide(SUBJECT, CANDIDATES)

    assert decision == Decision.new()
    assert adapter.calls[0].label is None


def test_a_label_on_an_extending_answer_is_recorded_and_changes_nothing():
    adapter = adapter_for('{"decision": "C1", "label": "a label nobody asked for"}')

    decision = adapter.decide(SUBJECT, CANDIDATES)

    assert decision == Decision.extend("C1")
    assert adapter.calls[0].label == "a label nobody asked for"


def test_the_call_record_keeps_what_the_adapter_was_asked():
    adapter = adapter_for('{"decision": "C2"}')

    adapter.decide(SUBJECT, CANDIDATES)

    call = adapter.calls[0]
    assert isinstance(call, LlmCall)
    assert call.chunk_id == "K3"
    assert call.candidate_ids == ("C1", "C2")
    assert call.candidates == CANDIDATES
    assert call.decision == Decision.extend("C2")


def test_the_adapter_satisfies_the_adapter_protocol():
    assert isinstance(LlmAdapter(StubClient()), IntentAdapter)


def test_a_stub_client_satisfies_the_model_client_protocol():
    # The injected dependency is a protocol, so a study runner can pass whatever
    # it has — and a test can pass a list of strings.
    assert isinstance(StubClient(), ModelClient)
    assert not isinstance(object(), ModelClient)


def test_the_adapter_reads_its_client_and_bound_back():
    client = StubClient()

    adapter = LlmAdapter(client)

    assert adapter.client is client
    assert adapter.max_attempts == DEFAULT_MAX_ATTEMPTS
    assert "max_attempts" in repr(adapter)


def test_canned_answers_classify_a_transcript_like_the_scripted_fake(corpus):
    # `interleaved_intents` is the discriminating fixture: its turns are A, B, A,
    # so the third chunk has to rejoin C1. The scripted fake is the oracle — the
    # adapter may not produce different chains for the same decisions.
    session = corpus.session("interleaved_intents")
    chunks = chunk_session(session)
    answers = [
        '{"decision": "NEW", "label": "the sampler"}',
        '{"decision": "NEW"}',
        ' {"decision": "C1"} ',
    ]

    chains = classify(session, chunks, LlmAdapter(StubClient(*answers)))

    assert chains == classify(session, chunks, FakeAdapter([NEW, NEW, "C1"]))
    assert [chain.chunk_ids for chain in chains] == [["K1", "K3"], ["K2"]]


def test_the_same_canned_answers_give_the_same_chains(corpus):
    # T4's first acceptance criterion, for this backend: deterministic in,
    # deterministic out. Nothing here reads a clock or a random number.
    session = corpus.session("interleaved_intents")
    chunks = chunk_session(session)
    answers = ['{"decision": "NEW"}', '{"decision": "NEW"}', '{"decision": "C1"}']

    first = LlmAdapter(StubClient(*answers))
    second = LlmAdapter(StubClient(*answers))

    assert classify(session, chunks, first) == classify(session, chunks, second)
    assert [call.prompts for call in first.calls] == [call.prompts for call in second.calls]


def test_a_blank_label_never_reaches_the_store(corpus):
    # The store refuses a label that holds no word, so an adapter that passed
    # the model's `"   "` on would take the whole run down over an advisory
    # field. The chain keeps its provisional label instead.
    session = corpus.session("single_intent")
    chunks = chunk_session(session)

    chains = classify(session, chunks, LlmAdapter(StubClient('{"decision": "NEW", "label": " "}')))

    assert chains == classify(session, chunks, FakeAdapter([NEW]))
    assert chains[0].label.strip()


def test_the_classifier_offers_the_candidates_the_prompt_then_names(corpus):
    session = corpus.session("interleaved_intents")
    chunks = chunk_session(session)
    client = StubClient('{"decision": "NEW"}', '{"decision": "NEW"}', '{"decision": "C1"}')

    classify(session, chunks, LlmAdapter(client))

    assert "[C1]" not in client.prompts[0]
    assert "[C1]" in client.prompts[1]
    assert "[C2]" not in client.prompts[1]
    assert "[C1]" in client.prompts[2]
    assert "[C2]" in client.prompts[2]


def module_imports() -> set[str]:
    """Return the top-level modules `idc.classify.llm` imports."""
    source = Path(llm_module.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def test_the_backend_imports_nothing_that_could_reach_the_network():
    # The ticket's hard constraint, stated about the module instead of about the
    # test suite: the client is injected, so this module needs no transport, no
    # process and no credential — and a dependency that implied one would show
    # up here before it showed up in a study run.
    imported = module_imports()

    assert imported, "the imports could not be read; the test would pass vacuously"
    assert imported & FORBIDDEN_IMPORTS == set()
    assert imported <= ALLOWED_IMPORTS, f"unexpected imports: {sorted(imported - ALLOWED_IMPORTS)}"


@pytest.mark.parametrize("fragment", FORBIDDEN_FRAGMENTS)
def test_the_backend_handles_no_credential_and_no_connection(fragment):
    source = Path(llm_module.__file__).read_text(encoding="utf-8")

    assert fragment not in source


def test_a_whole_classification_opens_no_socket(corpus, monkeypatch):
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the LLM adapter must not open a connection")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    session = corpus.session("two_sequential_intents")
    chunks = chunk_session(session)
    client = StubClient('{"decision": "NEW"}', '{"decision": "NEW"}')

    chains = classify(session, chunks, LlmAdapter(client))

    assert len(chains) == 2
