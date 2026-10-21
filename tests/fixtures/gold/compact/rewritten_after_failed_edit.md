## [C1] Intent: `docs/install.md` still tells people to use pip and a…
### Decisions & constraints
- (none)
### Rejected approaches
- (none)
### Solution
- `docs/install.md`
````
# Installing

The project is managed with [uv](https://docs.astral.sh/uv/).

```
uv sync --all-groups
uv run pytest
```

`uv sync --all-groups` creates `.venv/` next to the lock file and installs the
pinned versions of the runtime *and* the development dependencies; `uv run`
executes a command inside it without activating anything.
````
### Verification
- (none)
