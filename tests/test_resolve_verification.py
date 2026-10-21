"""The verification run of a chain (T5.2, #22).

The oracle is the corpus and the tool registry: a chain is built the way the
pipeline builds it — `chunk_session`, then `classify` with the fixture's own
gold script — and what counts as a verification command is the registry's
answer, not a second pattern list written here.

Three fixtures carry the ticket's three cases. `failed_then_passing_command`
runs the same command twice, red then green, which is the "keep the passing
run" case. `rewritten_after_failed_edit` makes no `exec` call at all, which is
the "no verification" case — as does the second chain of
`two_sequential_intents`. The third case, an `exec` call with exit code 0 whose
command runs no verification, is the one the corpus cannot hold: every `exec`
call of every fixture runs pytest. It is built by rewriting one fixture's own
command argument and loading that document through `loads`, the way
`tests/test_resolve_writes.py` builds the argument spellings the corpus does
not record.

The corpus-wide tests at the end are the invariants: the accepted run is the
last passing one or, failing that, the last run at all; every verification run
of a chain is either the accepted one or discarded, exactly once; and resolving
twice gives the same answer.
"""

import json

import pytest

from fixture_corpus import COMMAND_KEYS as CORPUS_COMMAND_KEYS
from fixture_corpus import CORPUS
from fixture_corpus import command_of as corpus_command_of
from idc.chunk import chunk_session
from idc.classify import FakeAdapter, classify, gold_script
from idc.importers import VerificationMatch, tool_registry
from idc.resolve import (
    COMMAND_KEYS,
    LocatedCall,
    SolutionVerification,
    VerificationRun,
    VerificationStatus,
    chain_calls,
    command_of,
    exec_calls,
    solution_verification,
    verification_runs,
)
from idc.schema import Chain, Chunk, Session, ToolCall, ToolCategory, ToolResult, loads

FIXTURE_NAMES = CORPUS.names()
"""Every fixture of the shared corpus, discovered at collection time."""

FAILED_THEN_PASSING = "failed_then_passing_command"
"""The fixture of the first acceptance criterion: the same command red, then green."""

NO_COMMAND = "rewritten_after_failed_edit"
"""The fixture of the second one: a chain that runs no command at all."""

THREE_RUNS = "write_then_two_edits"
"""Two failing runs and a passing one — the longest run sequence of the corpus."""

SEQUENTIAL = "two_sequential_intents"
"""Two chains, only the first of which verifies anything."""

ONE_RUN = "single_intent"
"""One chain with exactly one passing run — the document the rewrites start from."""

NON_VERIFICATION_COMMANDS = (
    "git status --short",
    "ruff check src",
    "mypy src",
    'git commit -m "add pytest"',
    'echo "go test"',
    "uv run pytest tests/billing -q --collect-only",
)
"""Commands that run no verification, each with a reason of its own.

`git status` and `echo` do nothing a chain could be proved by; `ruff` and `mypy`
are the deliberate omission of the registry's pattern list (a linter passing is
not this ticket's verification); `git commit -m "add pytest"` and
`echo "go test"` mention a runner in an argument; and `--collect-only` is a
pytest run that exits 0 without running a test. All six are the registry's
answers, asserted as such in `test_the_table_of_non_verification_commands_runs_nothing`.
"""


def classified(name: str) -> tuple[Session, list[Chunk], tuple[Chain, ...]]:
    """Return a fixture's session, chunks and chains, built by the pipeline."""
    session = CORPUS.session(name)
    return session_chains(session, name)


def session_chains(session: Session, name: str) -> tuple[Session, list[Chunk], tuple[Chain, ...]]:
    """Return chunks and chains for a session, labelled by the fixture's gold script.

    Split out of `classified` so a rewritten document — same messages, one
    argument or one result changed — is chunked and classified exactly like the
    fixture it came from.
    """
    chunks = chunk_session(session)
    chains = tuple(classify(session, chunks, FakeAdapter(gold_script(CORPUS.gold(name)))))
    return session, chunks, chains


def only_chain(name: str) -> tuple[Session, list[Chunk], Chain]:
    """Return a single-chain fixture's session, chunks and that one chain."""
    session, chunks, chains = classified(name)
    assert len(chains) == 1, f"{name} is used as a single-chain fixture"
    return session, chunks, chains[0]


def ids(runs: tuple[VerificationRun, ...]) -> list[str]:
    """Return the tool-call ids of `runs`, in the order they are held."""
    return [run.id for run in runs]


def session_with_command(name: str, call_id: str, command: str) -> Session:
    """Return a fixture with one `exec` call's command rewritten, via the load path.

    Every `exec` call of the corpus runs pytest (`tests/fixtures/README.md`
    describes the corpus, and `exec_runs` in `tests/fixture_corpus.py` reads
    them), so the case of a command that runs no verification is made by
    rewriting the fixture's own document and loading it through `loads` — not
    by writing a transcript into this file.
    """
    document = json.loads(CORPUS.text(name))
    rewritten = 0
    for message in document["messages"]:
        for call in message["tool_calls"]:
            if call["id"] == call_id:
                for key in CORPUS_COMMAND_KEYS:
                    if key in call["args"]:
                        call["args"][key] = command
                        rewritten += 1
    assert rewritten == 1, f"{call_id} must name its command exactly once in {name}"
    return loads(Session, json.dumps(document))


def session_with_renamed_argument(name: str, call_id: str, old: str, new: str) -> Session:
    """Return a fixture with one call's argument key renamed, via the load path."""
    document = json.loads(CORPUS.text(name))
    renamed = 0
    for message in document["messages"]:
        for call in message["tool_calls"]:
            if call["id"] == call_id:
                call["args"][new] = call["args"].pop(old)
                renamed += 1
    assert renamed == 1, f"{call_id} must occur exactly once in {name}"
    return loads(Session, json.dumps(document))


def session_with_extra_argument(name: str, call_id: str, key: str, value: str) -> Session:
    """Return a fixture with one argument added to one call, via the load path."""
    document = json.loads(CORPUS.text(name))
    extended = 0
    for message in document["messages"]:
        for call in message["tool_calls"]:
            if call["id"] == call_id:
                call["args"][key] = value
                extended += 1
    assert extended == 1, f"{call_id} must occur exactly once in {name}"
    return loads(Session, json.dumps(document))


def rewrite_result(
    document: dict,
    call_id: str,
    *,
    exit_code: int | None,
    is_error: bool,
    output: str | None = None,
) -> None:
    """Rewrite one call's result inside a parsed fixture document, in place.

    The corpus has no chain whose every verification run failed — in all three
    fixtures that run a command twice, the last run is the passing one — and
    none whose runs go green before they go red. Both shapes are made by
    rewriting the result of `failed_then_passing_command` in the fixture's own
    document; the messages, the chunks and the gold labelling are untouched by
    that.
    """
    rewritten = 0
    for message in document["messages"]:
        result = message.get("tool_result")
        if result is not None and result["call_id"] == call_id:
            result["exit_code"] = exit_code
            result["is_error"] = is_error
            if output is not None:
                result["output"] = output
            rewritten += 1
    assert rewritten == 1, f"{call_id} must be answered exactly once"


def session_with_result(
    name: str, call_id: str, *, exit_code: int | None, is_error: bool, output: str | None = None
) -> Session:
    """Return a fixture with one call's result rewritten, via the load path."""
    document = json.loads(CORPUS.text(name))
    rewrite_result(document, call_id, exit_code=exit_code, is_error=is_error, output=output)
    return loads(Session, json.dumps(document))


# --- the three acceptance criteria -----------------------------------------


def test_a_failing_then_passing_run_keeps_the_passing_one():
    session, chunks, chain = only_chain(FAILED_THEN_PASSING)
    results = {
        message.tool_result.call_id: message.tool_result
        for message in session.messages
        if message.tool_result is not None
    }

    verification = solution_verification(session, chunks, chain)

    assert verification is not None
    assert verification.run.id == "c06", "the passing run is the chain's proof"
    assert verification.status is VerificationStatus.PASSED
    assert verification.passed is True
    assert ids(verification.discarded) == ["c03"], "the failing run before it is path"
    assert verification.command == "uv run pytest tests/net/test_backoff.py -q", (
        "the command line is kept as the agent sent it, prefix and all"
    )
    assert verification.run.match.pattern.id == "pytest", (
        "the runner prefix is stripped by the registry, not by this stage"
    )
    assert verification.run.call.result is results["c06"], (
        "the run points at the session's own result object, so its output is verbatim"
    )
    assert verification.output == results["c06"].output
    assert verification.run.exit_code == 0


def test_a_chain_that_runs_no_command_has_no_verification():
    session, chunks, chain = only_chain(NO_COMMAND)

    assert exec_calls(session, chunks, chain) == (), f"{NO_COMMAND} runs no command at all"
    assert verification_runs(session, chunks, chain) == ()
    assert solution_verification(session, chunks, chain) is None


def test_a_chain_that_runs_no_command_has_no_verification_next_to_one_that_does():
    # The same state inside a transcript whose *other* chain verifies: nothing
    # of the first chain's run may reach the second one.
    session, chunks, chains = classified(SEQUENTIAL)
    verifying, documenting = chains

    assert solution_verification(session, chunks, verifying) is not None
    assert exec_calls(session, chunks, documenting) == ()
    assert solution_verification(session, chunks, documenting) is None


@pytest.mark.parametrize("command", NON_VERIFICATION_COMMANDS)
def test_a_command_that_runs_no_verification_is_not_the_chains_proof(command):
    session = session_with_command(ONE_RUN, "c04", command)
    session, chunks, chains = session_chains(session, ONE_RUN)
    (chain,) = chains
    ran = exec_calls(session, chunks, chain)

    assert ids(verification_runs(session, chunks, chain)) == []
    assert solution_verification(session, chunks, chain) is None
    assert [located.id for located in ran] == ["c04"], "the call itself is still reported"
    assert ran[0].result is not None
    assert ran[0].result.exit_code == 0, "and it is not the exit code that keeps it out"


def test_the_table_of_non_verification_commands_runs_nothing():
    # The table above is an expectation about the registry's pattern list, and
    # a command that silently started matching would turn every case of the
    # parametrization into a tautology.
    assert NON_VERIFICATION_COMMANDS, "the table of non-verification commands is not empty"
    registry = tool_registry()

    matching = [
        command for command in NON_VERIFICATION_COMMANDS if registry.is_verification(command)
    ]

    assert matching == []


# --- what a chain whose verification failed looks like ---------------------


def test_a_chain_whose_every_run_failed_records_the_last_failing_one():
    session = session_with_result(
        FAILED_THEN_PASSING,
        "c06",
        exit_code=1,
        is_error=True,
        output="F....\n1 failed, 4 passed in 0.22s\n",
    )
    session, chunks, chains = session_chains(session, FAILED_THEN_PASSING)
    (chain,) = chains

    verification = solution_verification(session, chunks, chain)

    assert verification is not None
    assert verification.status is VerificationStatus.FAILED
    assert verification.status.value == "verification_failed", (
        "the ticket's own name for the state is the value a later stage writes out"
    )
    assert verification.passed is False
    assert verification.run.id == "c06", "the last failing run, not the first"
    assert ids(verification.discarded) == ["c03"]
    assert verification.output == "F....\n1 failed, 4 passed in 0.22s\n", (
        "the failing run is kept with its output verbatim, like a passing one"
    )


def test_a_run_the_agent_saw_fail_did_not_pass_however_it_exited():
    # `exit_code` and `is_error` are independent fields (#3), so a result that
    # exited 0 and that the agent nevertheless saw as a failure is possible —
    # and must not become a proof. No corpus transcript holds that combination,
    # which is why it is built here; dropping the `is_error` half of the rule
    # passes every other test of this module.
    session = session_with_result(FAILED_THEN_PASSING, "c06", exit_code=0, is_error=True)
    session, chunks, chains = session_chains(session, FAILED_THEN_PASSING)
    (chain,) = chains
    runs = verification_runs(session, chunks, chain)

    verification = solution_verification(session, chunks, chain)

    assert [run.exit_code for run in runs] == [1, 0]
    assert [run.passed for run in runs] == [False, False]
    assert verification is not None
    assert verification.status is VerificationStatus.FAILED
    assert verification.run.id == "c06"


def test_a_run_whose_exit_code_the_transcript_does_not_hold_did_not_pass():
    # A tool that reports no exit status leaves `exit_code` null (#3). Reading
    # that as a pass would make a chain claim a verification it cannot show, so
    # the run counts as not passing and the chain is `verification_failed`.
    session = session_with_result(FAILED_THEN_PASSING, "c06", exit_code=None, is_error=False)
    session, chunks, chains = session_chains(session, FAILED_THEN_PASSING)
    (chain,) = chains

    verification = solution_verification(session, chunks, chain)

    assert verification is not None
    assert verification.run.id == "c06"
    assert verification.run.exit_code is None
    assert verification.run.passed is False
    assert verification.status is VerificationStatus.FAILED


def test_a_run_that_failed_after_a_passing_one_does_not_unseat_it():
    # The rule is "the last run that passed" (#20, #22), so a passing run stays
    # the chain's proof even when a later run of the same chain went red. The
    # later run is not dropped: it is in `discarded`, with its output, so a
    # consumer sees that the chain did not end green. No transcript of the
    # corpus has this shape, which is why it is built here — and why the
    # reading is pinned rather than left to the next edit of the rule.
    document = json.loads(CORPUS.text(FAILED_THEN_PASSING))
    rewrite_result(document, "c03", exit_code=0, is_error=False, output="5 passed in 0.20s\n")
    rewrite_result(
        document, "c06", exit_code=1, is_error=True, output="1 failed, 4 passed in 0.22s\n"
    )
    session, chunks, (chain,) = session_chains(
        loads(Session, json.dumps(document)), FAILED_THEN_PASSING
    )

    verification = solution_verification(session, chunks, chain)

    assert verification is not None
    assert [run.passed for run in verification_runs(session, chunks, chain)] == [True, False]
    assert verification.run.id == "c03"
    assert verification.status is VerificationStatus.PASSED
    assert ids(verification.discarded) == ["c06"], "the later failure is kept as path"


def test_the_two_states_the_ticket_separates_are_two_values():
    # "No verification" and "verification failed" are different answers: the
    # first is the absence of a run, the second a run that is reported with its
    # output. A single falsy value for both would collapse them.
    none_session, none_chunks, none_chain = only_chain(NO_COMMAND)
    failed = session_with_result(FAILED_THEN_PASSING, "c06", exit_code=2, is_error=True)
    failed_session, failed_chunks, (failed_chain,) = session_chains(failed, FAILED_THEN_PASSING)

    assert solution_verification(none_session, none_chunks, none_chain) is None
    assert isinstance(
        solution_verification(failed_session, failed_chunks, failed_chain), SolutionVerification
    )


# --- the runs of a chain ---------------------------------------------------


def test_every_matching_run_of_a_chain_is_reported_in_transcript_order():
    session, chunks, chain = only_chain(THREE_RUNS)

    runs = verification_runs(session, chunks, chain)
    verification = solution_verification(session, chunks, chain)

    assert ids(runs) == ["c04", "c06", "c08"]
    assert [run.passed for run in runs] == [False, False, True]
    assert verification is not None
    assert verification.run.id == "c08"
    assert ids(verification.discarded) == ["c04", "c06"]
    assert ids(verification.runs) == ["c04", "c06", "c08"], "the runs stay in transcript order"


def test_a_chain_resolves_only_the_runs_of_its_own_chunks():
    session, chunks, chains = classified("interleaved_intents")
    first, second = chains

    assert ids(verification_runs(session, chunks, first)) == ["c05", "c12"]
    assert ids(verification_runs(session, chunks, second)) == ["c09"]


def test_the_exec_calls_of_a_chain_are_its_exec_calls_and_nothing_else():
    session, chunks, chain = only_chain(FAILED_THEN_PASSING)

    assert [located.id for located in exec_calls(session, chunks, chain)] == ["c03", "c06"]
    assert all(
        located.category is ToolCategory.EXEC for located in exec_calls(session, chunks, chain)
    )


# --- reading the command off a call ----------------------------------------


def test_the_stage_and_the_corpus_agree_on_the_command_argument_spellings():
    # One list exists twice: the stage reads a call's command with its own
    # `COMMAND_KEYS`, and `tests/fixture_corpus.py` holds an independent copy so
    # that the stage is not its own oracle. Nothing else compares them, so a
    # spelling added on one side only would leave both sides agreeing with
    # themselves and the oracle silently weaker.
    assert COMMAND_KEYS == CORPUS_COMMAND_KEYS


@pytest.mark.parametrize("key", COMMAND_KEYS)
def test_every_argument_spelling_the_agents_use_names_the_command(key):
    call = ToolCall(id="c01", name="Bash", category=ToolCategory.EXEC, args={key: "make test"})

    assert command_of(call) == "make test"
    assert corpus_command_of(call) == "make test"


def test_a_call_whose_arguments_name_no_command_runs_nothing():
    call = ToolCall(id="c01", name="Bash", category=ToolCategory.EXEC, args={"cmd": "make test"})

    assert command_of(call) is None


def test_a_command_argument_that_holds_no_word_names_no_command():
    blank = ToolCall(id="c01", name="Bash", category=ToolCategory.EXEC, args={"command": "   "})
    empty = ToolCall(id="c02", name="Bash", category=ToolCategory.EXEC, args={"command": ""})

    assert command_of(blank) is None
    assert command_of(empty) is None
    # The oracle has to fold the same input into `None`, or it would endorse
    # the reading the stage deliberately rejects. T5.1 (#21) had to fix exactly
    # this in the production `file_of`, where a blank path escaped as a plain
    # `ValueError` outside the `ResolveError` family.
    assert corpus_command_of(blank) is None
    assert corpus_command_of(empty) is None


def test_an_exec_call_whose_command_is_spelled_unknown_is_no_verification():
    session = session_with_renamed_argument(ONE_RUN, "c04", "command", "cmd")
    session, chunks, chains = session_chains(session, ONE_RUN)
    (chain,) = chains

    assert [located.id for located in exec_calls(session, chunks, chain)] == ["c04"]
    assert verification_runs(session, chunks, chain) == (), (
        "a command this build cannot read is no proof; the call is reported, not refused"
    )
    assert solution_verification(session, chunks, chain) is None


def test_a_write_call_is_never_a_verification_however_it_is_spelled():
    # Only `exec` calls run commands. `c03` of `single_intent` is a write; given
    # a `command` argument that matches a pattern, it must still not become the
    # chain's proof — the category decides, not the argument.
    session = session_with_extra_argument(ONE_RUN, "c03", "command", "make test")
    session, chunks, chains = session_chains(session, ONE_RUN)
    (chain,) = chains
    write = next(
        call for message in session.messages for call in message.tool_calls if call.id == "c03"
    )
    verification = solution_verification(session, chunks, chain)

    assert command_of(write) == "make test", "the lookup itself is category-blind"
    assert ids(verification_runs(session, chunks, chain)) == ["c04"]
    assert verification is not None
    assert verification.run.id == "c04"


# --- the value objects -----------------------------------------------------


def test_a_verification_run_reports_what_the_agent_saw():
    result = ToolResult(call_id="c01", output="5 passed", exit_code=0, is_error=False)
    call = ToolCall(
        id="c01", name="Bash", category=ToolCategory.EXEC, args={"command": "make test"}
    )
    located = LocatedCall(
        call=call, chunk_id="K1", message_id="m01", message_index=0, result=result
    )
    match = tool_registry().match_verification("make test")
    assert isinstance(match, VerificationMatch)

    run = VerificationRun(call=located, match=match)

    assert run.id == "c01"
    assert run.command == "make test"
    assert run.pattern_id == "make-test"
    assert run.exit_code == 0
    assert run.output == "5 passed"
    assert run.passed is True
    assert run.status is VerificationStatus.PASSED


def test_a_verification_run_without_a_result_did_not_pass():
    call = ToolCall(
        id="c01", name="Bash", category=ToolCategory.EXEC, args={"command": "make test"}
    )
    located = LocatedCall(call=call, chunk_id="K1", message_id="m01", message_index=0, result=None)
    match = tool_registry().match_verification("make test")
    assert match is not None

    run = VerificationRun(call=located, match=match)

    assert run.output is None
    assert run.exit_code is None
    assert run.passed is False
    assert run.status is VerificationStatus.FAILED


def test_a_verification_that_counts_its_run_twice_is_a_programming_error():
    session, chunks, chain = only_chain(FAILED_THEN_PASSING)
    runs = verification_runs(session, chunks, chain)

    with pytest.raises(ValueError, match="c06"):
        SolutionVerification(run=runs[1], discarded=runs)


# --- corpus-wide invariants ------------------------------------------------


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_runs_of_a_chain_are_the_matching_exec_calls_in_order(name):
    session, chunks, chains = classified(name)
    registry = tool_registry()

    for chain in chains:
        runs = verification_runs(session, chunks, chain)

        assert ids(runs) == [
            located.id
            for located in exec_calls(session, chunks, chain)
            if (command := command_of(located.call)) is not None
            and registry.is_verification(command)
        ]
        assert [run.call.message_index for run in runs] == sorted(
            run.call.message_index for run in runs
        )


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_exec_calls_of_a_chain_are_the_exec_calls_of_its_chunks_in_order(name):
    session, chunks, chains = classified(name)

    for chain in chains:
        assert [located.id for located in exec_calls(session, chunks, chain)] == [
            located.id
            for located in chain_calls(session, chunks, chain)
            if located.category is ToolCategory.EXEC
        ]


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_accepted_run_is_the_last_passing_one_or_the_last_run(name):
    session, chunks, chains = classified(name)

    for chain in chains:
        runs = verification_runs(session, chunks, chain)
        verification = solution_verification(session, chunks, chain)

        if not runs:
            assert verification is None
            continue
        assert verification is not None
        passing = [run for run in runs if run.passed]
        expected = passing[-1] if passing else runs[-1]
        assert verification.run.id == expected.id
        assert verification.passed == bool(passing)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_run_of_a_chain_is_accepted_or_discarded_exactly_once(name):
    session, chunks, chains = classified(name)

    for chain in chains:
        runs = verification_runs(session, chunks, chain)
        verification = solution_verification(session, chunks, chain)
        if verification is None:
            assert runs == ()
            continue
        accounted = [verification.run.id, *ids(verification.discarded)]

        assert sorted(accounted) == sorted(ids(runs))
        assert len(set(accounted)) == len(accounted), "a run is accepted or discarded, not both"
        assert ids(verification.runs) == ids(runs)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_resolving_the_same_chain_twice_gives_the_same_answer(name):
    session, chunks, chains = classified(name)

    for chain in chains:
        assert solution_verification(session, chunks, chain) == solution_verification(
            session, chunks, chain
        )


def test_the_corpus_exercises_both_states():
    # The invariants above are parametrized over the corpus, and both of their
    # branches have to be reached by it: a corpus of transcripts that all verify
    # (or none of which do) would leave one branch unexercised while every test
    # still passed.
    verifying: list[str] = []
    silent: list[str] = []
    for name in FIXTURE_NAMES:
        session, chunks, chains = classified(name)
        for chain in chains:
            target = verifying if solution_verification(session, chunks, chain) else silent
            target.append(f"{name}:{chain.id}")

    assert verifying, "no chain of the corpus runs a verification"
    assert silent, "every chain of the corpus runs a verification"
