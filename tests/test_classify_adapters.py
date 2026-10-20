"""The adapter interface and the fake adapter that plays a scripted decision list.

The interface is the deliverable of T4.1 (#16), not an implementation detail:
the LLM adapter (#17), the S1M adapter (#18) and the end-to-end test (#42)
build against it, and the fake is what keeps that test network-free. These
tests therefore pin the *shape* of the contract — what a decision may say, what
a candidate carries, what makes an object an adapter — and the three loud
failures the stage must not paper over: a script that ran out, a value that is
not a decision, and a decision that names a chain nobody offered (the last one
in `tests/test_classify_store.py`, where the guard lives).

Two properties are asserted here on purpose because no later test can:

- a hand-written adapter that knows nothing about the fake satisfies
  `IntentAdapter`, so a real adapter cannot come to depend on anything only the
  fake provides, and it decides from the chunk's **own text** — which is what
  #17 and #18 do, and what `decide` therefore has to hand over;
- the fake does **not** check its scripted chain ids against the candidates it
  was given. A model can return a chain that is not on offer, and the pipeline
  has to fail on it; a fake that silently could not produce that case would
  hide the guard instead of exercising it.
"""

from collections.abc import Sequence
from dataclasses import FrozenInstanceError

import pytest

from idc.classify import (
    NEW,
    AdapterContractError,
    ChainCandidate,
    ChunkUnderDecision,
    Decision,
    FakeAdapter,
    FakeCall,
    IntentAdapter,
    NewChain,
    ScriptExhaustedError,
    as_decision,
)
from idc.schema import Chain, Chunk


def chunk(identifier: str = "K1", start: int = 0, end: int = 1) -> Chunk:
    return Chunk(id=identifier, start_index=start, end_index=end)


def subject(identifier: str = "K1", user_message: str = "do the thing") -> ChunkUnderDecision:
    return ChunkUnderDecision(id=identifier, user_message=user_message)


def candidate(identifier: str = "C1", label: str = "an intent") -> ChainCandidate:
    return ChainCandidate(id=identifier, label=label, last_user_message="do the thing")


class MinimalAdapter:
    """An adapter written without knowing the fake: the real adapters' shape.

    It uses nothing but the two arguments of `decide` and returns a `Decision`,
    which is exactly what #17 and #18 have to be able to do — and it decides
    from the **content** it was handed, not from the order the candidates
    happen to be in, because that is the only thing a real backend can do.
    The keyword rule is deliberately crude: what it pins is that the chunk's
    own text reaches the adapter at all.
    """

    def decide(self, chunk: ChunkUnderDecision, candidates: Sequence[ChainCandidate]) -> Decision:
        """Extend the candidate that shares a content word, else open a chain."""
        keywords = self._keywords(chunk.user_message)
        for offered in candidates:
            if keywords & self._keywords(offered.last_user_message):
                return Decision.extend(offered.id, confidence=0.5)
        return Decision.new(confidence=0.5)

    @staticmethod
    def _keywords(text: str) -> set[str]:
        """Return the words of `text` long enough not to be a stop word."""
        return {word for word in text.lower().split() if len(word) > 4}


class NotAnAdapter:
    """An object that has no `decide` method at all."""

    def classify(self) -> None:
        """Do nothing; the name is deliberately the wrong one."""


class WrongShapeAdapter:
    """An object with a `decide` of the wrong shape — but with one.

    `runtime_checkable` compares attribute names and not signatures, so this
    object passes `isinstance`. It exists to pin that limit, because a docstring
    that claimed more would be claiming a guard nobody has.
    """

    def decide(self) -> None:
        """Take no arguments at all, which no adapter may do."""


def test_new_is_one_named_value_that_reads_as_new():
    assert repr(NEW) == "NEW"
    assert isinstance(NEW, NewChain)
    # A freshly built sentinel compares and hashes equal to the exported one, so
    # an adapter that constructs its own is not a second, different `NEW`.
    assert NewChain() == NEW
    assert len({NEW, NewChain()}) == 1


def test_a_decision_either_names_a_chain_or_opens_one():
    extend = Decision.extend("C2")
    opening = Decision.new()

    assert (extend.target, extend.is_new, extend.confidence) == ("C2", False, None)
    assert (opening.target, opening.is_new, opening.confidence) == (NEW, True, None)


def test_a_decision_carries_an_optional_confidence():
    assert Decision.extend("C1", confidence=0.25).confidence == 0.25
    assert Decision.new(confidence=1.0).confidence == 1.0
    assert Decision.extend("C1").confidence is None


def test_a_decision_is_frozen():
    decision = Decision.extend("C1")

    with pytest.raises(FrozenInstanceError):
        decision.target = "C2"


@pytest.mark.parametrize("target", ["", "K1", "C0", "C01", "c1", "C1 ", "chain-1", 1, None])
def test_a_decision_rejects_a_target_that_is_not_a_chain_id(target):
    with pytest.raises(ValueError, match="C<n>"):
        Decision(target=target)


@pytest.mark.parametrize("confidence", [-0.1, 1.5, "high", True])
def test_a_decision_rejects_a_confidence_that_is_not_a_probability(confidence):
    with pytest.raises(ValueError, match="confidence"):
        Decision.extend("C1", confidence=confidence)


def test_a_candidate_carries_the_id_the_label_and_the_last_user_message():
    offered = ChainCandidate(id="C3", label="fix the flaky test", last_user_message="make it pass")

    assert (offered.id, offered.label, offered.last_user_message) == (
        "C3",
        "fix the flaky test",
        "make it pass",
    )
    assert ChainCandidate(id="C3", label="fix it").last_user_message == ""


def test_a_candidate_is_built_from_a_chain_without_restating_its_fields():
    chain = Chain(id="C2", label="rename the config key", chunk_ids=["K2"])

    offered = ChainCandidate.from_chain(chain, last_user_message="rename it everywhere")

    assert offered == ChainCandidate(
        id="C2", label="rename the config key", last_user_message="rename it everywhere"
    )


@pytest.mark.parametrize("identifier", ["", "K2", "C0", "c2"])
def test_a_candidate_rejects_an_id_that_is_not_a_chain_id(identifier):
    with pytest.raises(ValueError, match="C<n>"):
        ChainCandidate(id=identifier, label="an intent")


def test_a_candidate_rejects_an_empty_label():
    with pytest.raises(ValueError, match="label"):
        ChainCandidate(id="C1", label="")


def test_the_chunk_under_decision_carries_its_id_and_its_own_user_message():
    under_decision = ChunkUnderDecision(id="K3", user_message="back to the sampler")

    assert (under_decision.id, under_decision.user_message) == ("K3", "back to the sampler")
    assert ChunkUnderDecision(id="K3").user_message == ""


def test_the_chunk_under_decision_is_built_from_a_chunk_without_restating_its_fields():
    # The counterpart of `ChainCandidate.from_chain`. The text is resolved by
    # the store, the only side of the stage that may read the session.
    under_decision = ChunkUnderDecision.from_chunk(chunk("K2"), "rename the key")

    assert under_decision == ChunkUnderDecision(id="K2", user_message="rename the key")
    # A chunk without a user turn resolves to `None`, which is the absence
    # `ChainCandidate.last_user_message` already spells as the empty string.
    assert ChunkUnderDecision.from_chunk(chunk("K2"), None).user_message == ""


def test_the_chunk_under_decision_needs_an_id():
    with pytest.raises(ValueError, match="id"):
        ChunkUnderDecision(id="", user_message="do the thing")


def test_as_decision_reads_the_three_forms_the_interface_allows():
    decision = Decision.extend("C1", confidence=0.75)

    assert as_decision(decision) is decision
    assert as_decision(NEW) == Decision.new()
    assert as_decision(NewChain()) == Decision.new()
    assert as_decision("C4") == Decision.extend("C4")


@pytest.mark.parametrize("value", [None, 1, 0.5, object(), ["C1"], {"target": "C1"}])
def test_as_decision_rejects_anything_that_is_not_a_decision(value):
    with pytest.raises(AdapterContractError, match="Decision"):
        as_decision(value)


def test_as_decision_reports_a_string_that_is_not_a_chain_id_as_a_bad_chain_id():
    # A string is read as a chain id, so the message has to name the id form
    # rather than the adapter contract — that is the mistake the caller made.
    with pytest.raises(ValueError, match="C<n>"):
        as_decision("K1")


def test_the_fake_and_a_hand_written_adapter_both_satisfy_the_interface():
    assert isinstance(FakeAdapter([NEW]), IntentAdapter)
    assert isinstance(MinimalAdapter(), IntentAdapter)
    assert not isinstance(NotAnAdapter(), IntentAdapter)


def test_the_protocol_can_only_check_that_a_decide_attribute_exists():
    # `runtime_checkable` compares attribute names, not signatures, so this is
    # the whole of what `isinstance` is able to say. The classifier does not use
    # it as a guard, and no docstring may claim that it does.
    assert isinstance(WrongShapeAdapter(), IntentAdapter)


def test_a_hand_written_adapter_decides_from_the_chunks_own_text():
    # The property the interface exists for: #17 prompts with the new chunk's
    # user message and #18 scores it against each candidate, so `decide` has to
    # hand that text over. An adapter that could only count candidates would
    # pick `C1` here; this one picks the chain the text belongs to.
    adapter = MinimalAdapter()
    candidates = [
        ChainCandidate(
            id="C1", label="rename the key", last_user_message="rename the config key everywhere"
        ),
        ChainCandidate(
            id="C2", label="the flaky sampler", last_user_message="the weighted pick drifts"
        ),
    ]

    assert adapter.decide(
        subject("K3", "the weighted pick still drifts"), candidates
    ) == Decision.extend("C2", confidence=0.5)
    assert adapter.decide(subject("K4", "document the new option"), candidates) == Decision.new(
        confidence=0.5
    )
    assert adapter.decide(subject("K5", "rename the config key"), []) == Decision.new(
        confidence=0.5
    )


def test_the_fake_plays_its_script_in_order():
    adapter = FakeAdapter([NEW, "C1", Decision.extend("C1", confidence=0.4)])

    first = adapter.decide(subject("K1"), [])
    second = adapter.decide(subject("K2"), [candidate()])
    third = adapter.decide(subject("K3"), [candidate()])

    assert [first, second, third] == [
        Decision.new(),
        Decision.extend("C1"),
        Decision.extend("C1", confidence=0.4),
    ]


def test_the_fake_normalizes_its_script_when_it_is_built():
    adapter = FakeAdapter(iter([NEW, "C1"]))

    assert adapter.script == (Decision.new(), Decision.extend("C1"))
    assert adapter.remaining == 2


@pytest.mark.parametrize("entry", [None, 1, object()])
def test_the_fake_rejects_a_script_entry_that_is_not_a_decision_when_it_is_built(entry):
    with pytest.raises(AdapterContractError, match="Decision"):
        FakeAdapter([NEW, entry])


def test_the_fake_records_every_call_it_answered():
    adapter = FakeAdapter([NEW, "C1"])
    offered = candidate("C1", "the first intent")

    adapter.decide(subject("K1"), [])
    adapter.decide(subject("K2"), [offered])

    assert adapter.calls == (
        FakeCall(chunk_id="K1", candidates=(), decision=Decision.new()),
        FakeCall(chunk_id="K2", candidates=(offered,), decision=Decision.extend("C1")),
    )
    assert [call.candidate_ids for call in adapter.calls] == [(), ("C1",)]


def test_the_fake_fails_loudly_when_its_script_runs_out():
    adapter = FakeAdapter([NEW])
    adapter.decide(subject("K1"), [])

    with pytest.raises(ScriptExhaustedError, match="K2"):
        adapter.decide(subject("K2"), [candidate()])

    assert adapter.remaining == 0


def test_the_fake_does_not_check_its_decisions_against_the_candidates():
    # Documented on purpose: a model can name a chain that is not on offer, and
    # the pipeline's guard is what has to catch it (see the store's tests). A
    # fake that refused here could not produce that case at all.
    adapter = FakeAdapter(["C9"])

    assert adapter.decide(subject("K1"), []) == Decision.extend("C9")
