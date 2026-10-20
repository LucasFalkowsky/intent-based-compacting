"""Verification-pattern matching: which commands count as a verification run.

Both directions are tested on purpose. A missed verification costs one summary
line; a false positive makes the resolver (T5.2) keep an unrelated command as
the chain's proof that the solution works, so the commands that must *not*
match are the expensive half of this file.
"""

import pytest

from fixture_corpus import CORPUS, exec_runs
from idc.importers import tool_registry

MATCHING = [
    # Python
    "pytest",
    "pytest -q tests/api",
    "uv run pytest tests/api -q",
    "python -m pytest",
    "python3 -m pytest -x",
    "python -m unittest discover",
    "tox",
    "tox -e py313",
    "nox -s tests",
    "./.venv/bin/pytest -q",
    "/usr/local/bin/pytest",
    "CI=1 PYTHONPATH=src pytest -q",
    # Make
    "make test",
    "make check",
    # JavaScript
    "npm test",
    "npm test -- --runInBand",
    "npm run test",
    "yarn test",
    "yarn run test",
    "pnpm test",
    "pnpm run test",
    "bun test",
    "bun run test",
    "npx jest",
    "vitest run",
    "mocha tests",
    # other ecosystems
    "cargo test",
    "cargo test --all-features",
    "cargo nextest run",
    "go test ./...",
    "dotnet test",
    "mvn test",
    "gradle test",
    "./gradlew test",
    "ctest --output-on-failure",
    "rspec",
    "bundle exec rspec spec/models",
    "rake test",
    "vendor/bin/phpunit",
    "swift test",
    "mix test",
    "deno test -A",
    "dart test",
    "flutter test",
    # composed commands: one segment is enough
    "cd services/api && uv run pytest -q",
    "ruff check . && pytest -q",
    "pytest -q || echo failed",
    "make lint; make test",
    "pytest -q 2>&1 | tee pytest.log",
    # a comment ends its line, not the scan: the run on the next line counts
    "# run the suite\nmake test",
    "make build # make test\npytest -q",
    # a flag that disqualifies one runner is not a flag of every runner: `-n` is
    # make's dry run but pytest-xdist's parallelism, `-l` is tox's env listing
    # but pytest's --showlocals
    "pytest -n 4",
    "pytest -l --tb=short",
    "nox -s tests",
]

NOT_MATCHING = [
    # the words appear, but not as the command that is run
    'git commit -m "add pytest"',
    'git commit -m "make test green"',
    'echo "go test"',
    "echo pytest > notes.txt",
    'grep -rn "make test" docs',
    'rg "cargo test" --glob "*.md"',
    # paths that merely contain the word
    "cat tests/test_sampler.py",
    "ls tests",
    "cd /repo/pytest-plugins && ls",
    "./scripts/pytest-wrapper.sh",
    "rm -rf .pytest_cache",
    "cat go-test-notes.md",
    # neighbouring commands of the same tool that do not run tests
    "cargo build",
    "cargo fmt",
    "go build ./...",
    "npm run build",
    "npm install pytest-cov",
    "make build",
    "make lint",
    "python -m pip install pytest",
    "uv run ruff check .",
    "uv sync --all-groups",
    "mypy src",
    # a here-document body is data the command writes, not a command
    "python - <<'EOF'\npytest\nEOF",
    'cat > Makefile <<"EOF"\ntest:\n\tmake test\nEOF',
    # an escaped quote does not end the string it is in, so the separator and
    # the runner behind it stay inside an argument of `echo`, `git commit -m`,
    # `python -c` — the spellings a recorded agent command really uses
    'git commit -m "rename \\"test\\" target; make test still works"',
    'echo "Run \\"make build\\"; make test is next" >> README.md',
    'echo "quote \\" then | jest --ci"',
    'echo "print(\\"hi\\"); pytest -q" > x.py',
    'python -c "print(1)" && echo "a\\"b; pytest -q x"',
    # an unquoted `#` comments out the rest of its line, so nothing behind it runs
    "echo done # note; make test",
    "echo done #&& make test",
    "ls # ; pytest",
    "# step 1; make test\nmake build",
    # `#` only starts a comment at the beginning of a word, so neither of these
    # is a run of anything the registry knows either
    "echo a#b",
    "make test#x",
    # a real runner asked what it would do, which proves nothing: T5.2 sees only
    # the command and a zero exit code, so it cannot tell these from a real run
    "pytest --version",
    "pytest --help",
    "pytest -h",
    "pytest --collect-only",
    "tox -l",
    "nox --list",
    "go test -list .",
    "npm test --help",
    "make test -n",
    "make check --dry-run",
    "cargo test --list",
    "uv run pytest --version",
    # not a test command at all
    "git status",
    "ls -la",
    "",
    "   ",
]


@pytest.mark.parametrize("command", MATCHING)
def test_a_verification_command_matches(command):
    assert tool_registry().is_verification(command), command


@pytest.mark.parametrize("command", NOT_MATCHING)
def test_a_command_that_only_mentions_tests_does_not_match(command):
    match = tool_registry().match_verification(command)

    assert match is None, f"{command!r} matched {match.pattern.id if match else None!r}"


def test_every_exec_call_of_the_corpus_is_a_verification_run():
    # Every exec call the corpus records is a pytest run; the registry must see
    # all of them, since T5.2 resolves the chain's verification from them.
    commands = [
        command for session in CORPUS.sessions().values() for command, _ in exec_runs(session)
    ]
    registry = tool_registry()

    assert commands != []
    assert [command for command in commands if not registry.is_verification(command)] == []


def test_the_match_names_the_pattern_and_the_command_it_matched():
    match = tool_registry().match_verification("cd src && uv run pytest tests/api -q")

    assert match is not None
    assert match.pattern.id == "pytest"
    assert match.command == "uv run pytest tests/api -q"
    assert match.tokens == ("pytest", "tests/api", "-q")


def test_the_first_matching_segment_wins():
    match = tool_registry().match_verification("make test && cargo test")

    assert match is not None
    assert match.command == "make test"


def test_matching_is_quote_aware():
    registry = tool_registry()

    assert registry.is_verification('python -c "import sys" && go test ./...')
    assert registry.match_verification('echo "pytest -q && make test"') is None


def test_an_escaped_quote_does_not_end_the_string_it_is_in():
    # Inside double quotes `\"` is a literal quote, not the closing one. Reading
    # it as the closing quote scans the rest of the argument as unquoted shell,
    # where a `;` opens a segment and the next word becomes a command.
    registry = tool_registry()

    assert (
        registry.match_verification('git commit -m "rename \\"test\\" target; make test too"')
        is None
    )
    # a backslash is literal inside single quotes, so there the quote does close
    assert registry.is_verification("echo 'a\\' && pytest -q")


def test_a_comment_ends_its_line_and_not_the_scan():
    registry = tool_registry()

    assert registry.match_verification("echo done # note; make test") is None
    assert registry.is_verification("# run the suite\nmake test")


def test_a_runner_that_is_only_asked_what_it_would_do_is_no_verification():
    # A zero exit code is all T5.2 has besides the command line, and these exit
    # zero without running a single test.
    registry = tool_registry()

    assert registry.match_verification("pytest --collect-only") is None
    assert registry.match_verification("make test -n") is None
    assert registry.is_verification("pytest -q")
    assert registry.is_verification("make test")


def test_an_unbalanced_quote_does_not_raise():
    registry = tool_registry()

    assert registry.is_verification('pytest -k "slow')
    assert registry.match_verification('git commit -m "add pytest') is None


def test_a_here_document_does_not_hide_a_command_in_its_body():
    # `python - <<EOF` runs Python, not the lines it is fed; a body line that
    # happens to read like a test run must not become the chain's verification.
    registry = tool_registry()

    assert registry.match_verification("python - <<'EOF'\nmake test\nEOF") is None
    assert registry.is_verification("pytest -q <<'EOF'\nEOF")


def test_the_patterns_are_exposed_with_stable_ids():
    patterns = tool_registry().verification_patterns

    assert patterns != ()
    assert len({pattern.id for pattern in patterns}) == len(patterns)
    assert "pytest" in {pattern.id for pattern in patterns}
