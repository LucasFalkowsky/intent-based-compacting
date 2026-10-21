"""The tokenizer seam: the protocol, the fake, and which model gets which one."""

from datetime import UTC, datetime

import pytest

from fixture_corpus import CORPUS
from idc.measure import (
    FakeTokenizer,
    ModelFamily,
    NoTokenizerError,
    Tokenizer,
    TokenizerRegistry,
    UnknownModelError,
    model_family,
)
from idc.schema import Agent, Message, Role, Session, SessionSource

CREATED_AT = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
"""Fixed instant; no test in this module reads the clock."""


def session(model: str | None) -> Session:
    """Return a one-message session recorded for `model`."""
    return Session(
        id="sess-tok",
        agent=Agent.CLAUDE_CODE,
        model=model,
        source=SessionSource.RAW,
        created_at=CREATED_AT,
        messages=[Message(id="m01", role=Role.USER, content="go")],
    )


# --- the fake tokenizer: one token per word ----------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0),
        ("   ", 0),
        ("\n\t ", 0),
        ("one", 1),
        ("two words", 2),
        ("  leading and trailing  ", 3),
        ("collapsed   runs\tof\nwhitespace", 4),
        ("punctuation, like this.", 3),
        ('{"file_path": "src/billing/export.py"}', 2),
    ],
)
def test_the_fake_tokenizer_counts_one_token_per_whitespace_separated_word(text, expected):
    # Hand-counted: a token is a run of non-whitespace, which is what makes the
    # expectations of every other measurement test readable.
    assert FakeTokenizer().count(text) == expected


def test_the_fake_tokenizer_records_every_text_it_was_asked_about():
    tokenizer = FakeTokenizer()

    tokenizer.count("two words")
    tokenizer.count("")

    assert tokenizer.calls == ("two words", "")
    assert tokenizer.total == 2


def test_the_fake_tokenizer_is_a_tokenizer():
    # `runtime_checkable` checks the presence of `count`, not its signature, so
    # this pins the name the protocol fixes and nothing more.
    assert isinstance(FakeTokenizer(), Tokenizer)


def test_two_fake_tokenizers_do_not_share_their_record():
    first, second = FakeTokenizer(), FakeTokenizer()

    first.count("a b c")

    assert second.calls == ()


# --- which family a model belongs to -----------------------------------------


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("claude-opus-5", ModelFamily.CLAUDE),
        ("claude-sonnet-4-5", ModelFamily.CLAUDE),
        ("claude", ModelFamily.CLAUDE),
        ("gemini-2.5-pro", ModelFamily.GEMINI),
        ("gemini", ModelFamily.GEMINI),
        ("  CLAUDE-Opus-5  ", ModelFamily.CLAUDE),
    ],
)
def test_the_family_is_the_model_name_up_to_the_first_dash(model, family):
    assert model_family(model) is family


def test_every_model_the_corpus_records_resolves_to_a_family():
    # Fixture-derived, so the table itself is asserted on: an empty corpus would
    # otherwise make this pass without resolving a single model.
    models = {CORPUS.session(name).model for name in CORPUS.names()}

    assert models, "the corpus records no model at all"
    assert {model_family(model) for model in models} == {ModelFamily.CLAUDE, ModelFamily.GEMINI}


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5",
        "llama-3",
        "anthropic/claude-opus-5",
        "",
        "   ",
        None,
    ],
)
def test_a_model_of_no_known_family_is_refused_rather_than_given_a_default(model):
    # The refusal is the decision of this ticket: a silent fallback would make
    # two runs comparable that were counted with different tokenizers.
    with pytest.raises(UnknownModelError) as refusal:
        model_family(model)

    assert "claude" in str(refusal.value)
    assert refusal.value.model == model


def test_the_unknown_model_refusal_names_every_family_it_knows():
    with pytest.raises(UnknownModelError) as refusal:
        model_family("gpt-5")

    assert refusal.value.known == tuple(family.value for family in ModelFamily)


# --- the registry: picking a tokenizer for a session -------------------------


def test_a_registry_hands_out_the_tokenizer_registered_for_the_model_s_family():
    tokenizer = FakeTokenizer()
    registry = TokenizerRegistry({ModelFamily.CLAUDE: tokenizer})

    assert registry.tokenizer_for(session("claude-opus-5")) is tokenizer
    assert registry.tokenizer_for_model("claude-sonnet-4-5") is tokenizer


def test_a_family_with_no_tokenizer_registered_is_refused_and_says_what_to_do():
    registry = TokenizerRegistry({ModelFamily.CLAUDE: FakeTokenizer()})

    with pytest.raises(NoTokenizerError) as refusal:
        registry.tokenizer_for(session("gemini-2.5-pro"))

    assert refusal.value.family == ModelFamily.GEMINI.value
    assert "register" in str(refusal.value)


def test_an_unknown_model_is_refused_by_the_registry_too():
    registry = TokenizerRegistry({ModelFamily.CLAUDE: FakeTokenizer()})

    with pytest.raises(UnknownModelError):
        registry.tokenizer_for(session("gpt-5"))


def test_registering_a_tokenizer_changes_only_the_registry_it_was_registered_on():
    # There is no module-level default registry on purpose: a run's tokenizer
    # set is passed in, so two runs cannot differ by import order.
    first = TokenizerRegistry()
    second = TokenizerRegistry()

    first.register(ModelFamily.CLAUDE, FakeTokenizer())

    assert first.families == (ModelFamily.CLAUDE,)
    assert second.families == ()


def test_a_registry_copies_the_mapping_it_was_built_from():
    mapping = {ModelFamily.CLAUDE: FakeTokenizer()}
    registry = TokenizerRegistry(mapping)

    mapping.clear()

    assert registry.families == (ModelFamily.CLAUDE,)


def test_an_empty_registry_ships_no_tokenizer_at_all():
    # This build contains no real tokenizer: every one of them needs a
    # downloaded vocabulary or a network call, and no test here may do either.
    assert TokenizerRegistry().families == ()
