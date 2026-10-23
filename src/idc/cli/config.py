"""The run configuration: the TOML file, its validation, and what it records.

A compaction run has four kinds of setting that are tedious to retype and that
the study has to be able to read back off a result: which **agent** the
transcript comes from and which agent the summary is **rendered for**, which
classifier **adapter** decided the chains and with which **models**, whether the
harness noise inside a chunk is **reduced**, and the **paths** the files live at.
They are a file so that a run is one command, and TOML so that it is the format
`src/idc/compact/narrative_instruction.toml` already is — read with the standard
library's `tomllib`, which is why the config file costs no dependency.

```toml
agent  = "claude-code"      # the agent the input transcript comes from
target = "opencode"         # the agent the summary is rendered for; defaults to `agent`
adapter = "s1m"             # which classifier adapter the run asks for
reduce  = true              # reduce the harness noise inside a chunk

[models]                    # recorded in the run manifest, one id per role
classifier = "claude-opus-5"
narrative  = "claude-opus-5"

[options]                   # anything else worth recording, as strings
threshold = "0.6"

[paths]
transcript        = "session.jsonl"    # what the command reads   (--in)
output            = "compacted.jsonl"  # what it writes           (--out)
compacted         = "compacted.jsonl"  # `measure`'s second file  (--compacted)
manifest          = "run.json"         # where `run` puts it      (--manifest)
previous_manifest = "previous.json"    # the run this one follows  (--previous-manifest)
```

Every key is optional, including `[paths]`: a command line can supply any of
them, and `idc.cli.main.configuration` merges the two with the command line
winning. What neither supplies is a `UsageError` naming the flag, raised by the
command surface and not here — this module's job is to say what a *valid*
configuration is.

**Validation goes through `load_config` / `loads_config` / `validate_config`
and nowhere else.** That is the rule `idc.schema` and `idc.measure.manifest`
follow, for the reason `CLAUDE.md` gives: only these three raise the CLI's own
error, and `Config.model_validate` would put a pydantic traceback on the
terminal. The line between a `ConfigError` and a plain `ValueError` is where the
data came from — a document a user handed in against a `Config` a caller's own
code built wrongly — and `_ConfigModel` is the half of that which keeps
pydantic's `ValidationError` from escaping either way.

One thing `RunConfig` has and this model does not: read-only mappings.
`Config.models` and `Config.options` are plain dicts, so `frozen=True` stops an
attribute being replaced and not an entry being added to the mapping it holds.
That is deliberate — a configuration is read once at the start of a command,
whereas a `RunManifest` is study material whose bytes have to be reproducible —
and `run_config` copies both mappings on the way into the manifest, where
`SortedMapping` does make them read-only.
"""

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from idc.cli.errors import ConfigError
from idc.measure import RunConfig
from idc.schema import Agent

__all__ = [
    "ADAPTER_NAMES",
    "DEFAULT_ADAPTER",
    "PATH_KEYS",
    "Config",
    "Paths",
    "load_config",
    "loads_config",
    "read_config_document",
    "validate_config",
]

ADAPTER_NAMES = ("llm", "s1m")
"""The classifier adapters a run may ask for — `idc.classify`'s two real ones.

`FakeAdapter` is deliberately absent: it plays a script a test wrote and has no
name a config file could select it by. A test reaches the command surface
through `idc.cli.Runtime`, which takes the adapter *object*.
"""

DEFAULT_ADAPTER = "llm"
"""The adapter a configuration asks for when it names none: the study's primary."""

PATH_KEYS = ("transcript", "output", "compacted", "manifest", "previous_manifest")
"""The keys of `[paths]`, in the order the documentation above lists them."""


def _located(exc: ValidationError) -> list[tuple[str, str]]:
    """Return one `(dotted location, message)` pair per problem pydantic found.

    Args:
        exc: The validation error to flatten.

    Returns:
        The problems, in pydantic's order, with the location rendered as a
        dotted field path — `paths.transcript`, `models.classifier` — and
        `<document>` for a problem that belongs to no field.
    """
    problems = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<document>"
        problems.append((location, error["msg"]))
    return problems


def _as_config_error(exc: ValidationError, source: str | None) -> ConfigError:
    """Translate pydantic's error into this package's, keeping every problem.

    Args:
        exc: The validation error.
        source: Where the document came from, or `None`.

    Returns:
        The `ConfigError` to raise, with one located problem per line.
    """
    problems = _located(exc)
    count = f"{len(problems)} problem{'s' if len(problems) != 1 else ''}"
    return ConfigError(count, source=source, errors=problems)


class _ConfigModel(BaseModel):
    """Base of the configuration's data types: no unknown fields, no mutation.

    `extra="forbid"` so that a misspelled key is a diagnostic rather than a
    setting that silently did nothing — `adaptor = "s1m"` is the whole reason.
    `frozen=True` so that a command cannot rewrite the configuration it was
    handed; see the module docstring for what that does and does not cover.

    Invalid input raises a plain `ValueError` whichever way a caller's own code
    produces it — construction or assignment — and never pydantic's
    `ValidationError`. The load path is the other half: it raises `ConfigError`,
    because a document is input the surface was handed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    def __init__(self, **data: Any) -> None:
        """Build the object, reporting invalid fields as a plain `ValueError`."""
        try:
            super().__init__(**data)
        except ValidationError as exc:
            raise ValueError(str(_as_config_error(exc, None))) from None

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse an assignment, reporting it as a plain `ValueError`."""
        try:
            super().__setattr__(name, value)
        except ValidationError as exc:
            raise ValueError(str(_as_config_error(exc, None))) from None


class Paths(_ConfigModel):
    """Where a command's files are, as the configuration spells them.

    Every field is optional because every one of them has a flag as well, and a
    command needs only some of them. Held as strings and not as `Path`s: the
    path is also what the run manifest records verbatim as the label of a file
    it hashed, and a `Path` would normalise the spelling the user typed.

    Attributes:
        transcript: What the command reads — `--in`.
        output: What it writes — `--out`.
        compacted: The compacted transcript `measure` compares the input
            against — `--compacted`.
        manifest: Where `run` writes its manifest — `--manifest`. When unset,
            `run` derives it from `output`; see `idc.cli.commands`.
        previous_manifest: The manifest of the run this one continues —
            `--previous-manifest`. Only its output digest is recorded.
    """

    transcript: str | None = None
    output: str | None = None
    compacted: str | None = None
    manifest: str | None = None
    previous_manifest: str | None = None


class Config(_ConfigModel):
    """One run's configuration, however it was assembled.

    Attributes:
        agent: The agent the input transcript comes from, or `None` when
            nothing named it. `None` is valid here and refused later by the
            commands that need it, so that `idc chunk` needs no agent at all.
        target: The agent the summary is rendered for, when it differs from
            `agent`. Read through `target_agent`, never directly.
        adapter: The classifier adapter the run asks for, one of
            `ADAPTER_NAMES`.
        reduce: Whether the chunk stage reduces the harness noise inside a
            chunk; on by default, because that is the pipeline `README.md`
            describes.
        models: The model id per role, recorded in the run manifest. Not used
            to construct anything: this build constructs no model backend (see
            `idc.cli.Runtime`), so these ids are provenance, and a run whose
            adapter was injected records what it actually ran next to them.
        options: Anything else worth recording, as strings. Merged into the
            manifest's own options by `run_config`.
        paths: Where the files are.
    """

    agent: Agent | None = None
    target: Agent | None = None
    adapter: str = DEFAULT_ADAPTER
    reduce: bool = True
    models: dict[str, str] = Field(default_factory=dict)
    options: dict[str, str] = Field(default_factory=dict)
    paths: Paths = Field(default_factory=Paths)

    @field_validator("adapter")
    @classmethod
    def _check_adapter(cls, value: str) -> str:
        """Refuse an adapter this build has no implementation for.

        Args:
            value: The adapter name the document gave.

        Returns:
            The name, unchanged.

        Raises:
            ValueError: If it is not one of `ADAPTER_NAMES`. The message lists
                them, so the repair is readable from the error alone.
        """
        if value not in ADAPTER_NAMES:
            known = ", ".join(repr(name) for name in ADAPTER_NAMES)
            raise ValueError(f"unknown classifier adapter {value!r}; known adapters: {known}")
        return value

    @property
    def target_agent(self) -> Agent | None:
        """Return the agent the summary is rendered for.

        Returns:
            `target` when the configuration names one, `agent` otherwise, and
            `None` when it names neither. Defaulting to the source agent is
            what makes a run re-readable by the agent it came from, which is
            the normal case; rendering for another agent is the exception and
            has to be said.
        """
        return self.target if self.target is not None else self.agent

    def run_config(self, adapter_used: str) -> RunConfig:
        """Return what a run manifest records about this configuration.

        Args:
            adapter_used: The name of the adapter that actually decided the
                chains — `type(adapter).__name__` of the object the command
                ran with. Recorded instead of `adapter`, because `adapter` says
                what the run *asked* for and this build constructs none of
                them; the requested name is kept alongside, under
                `options["adapter_requested"]`, so the manifest states both.

        Returns:
            The `RunConfig` to hand to `idc.measure.build_manifest`. Its
            `agent` is `target_agent` — the manifest's `agent` is the agent the
            output was rendered for — and the source agent is recorded under
            `options["source_agent"]`, which is why both are readable from one
            manifest.

        Raises:
            ValueError: If no source agent is configured. A caller's own
                mistake: every command that builds a manifest requires
                `--agent` before it gets here.
        """
        if self.agent is None:
            raise ValueError("a run manifest records the source agent; this config names no agent")
        options = dict(self.options)
        options["adapter_requested"] = self.adapter
        options["reduce"] = "true" if self.reduce else "false"
        options["source_agent"] = self.agent.value
        return RunConfig(
            agent=self.target_agent,
            adapter=adapter_used,
            models=dict(self.models),
            options=options,
        )

    def with_overrides(self, overrides: Mapping[str, Any], *, source: str | None = None) -> Self:
        """Return this configuration with the given values replaced.

        The merge goes through validation rather than through `model_copy`, so
        an override is checked exactly as a key in the file is.

        Args:
            overrides: The values to replace, by field name. A `paths` entry is
                merged key by key, so overriding `--out` does not drop the
                `transcript` the file supplied.
            source: Where the file half came from, for the error message.

        Returns:
            The merged configuration.

        Raises:
            ConfigError: If the merged document is not a valid configuration.
        """
        document = self.model_dump(mode="json")
        paths = {**document.get("paths", {}), **dict(overrides.get("paths") or {})}
        document.update({key: value for key, value in overrides.items() if key != "paths"})
        document["paths"] = paths
        return validate_config(document, source=source)


def validate_config(document: Mapping[str, Any] | Any, *, source: str | None = None) -> Config:
    """Validate an already-parsed configuration document.

    Args:
        document: The parsed document, normally a mapping.
        source: Where it came from, for the error message.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the document is not a table, or does not satisfy the
            configuration's shape — an unknown key, an unknown agent, an
            unknown adapter and a model id that is not a string all report
            here, one located problem per line.
    """
    if not isinstance(document, Mapping):
        raise ConfigError(
            f"a configuration is a table of settings, not {type(document).__name__}", source=source
        )
    try:
        return Config.model_validate(document)
    except ValidationError as exc:
        raise _as_config_error(exc, source) from None
    except ValueError as exc:
        raise ConfigError(str(exc), source=source) from None


def loads_config(data: str, *, source: str | None = None) -> Config:
    """Parse and validate a configuration given as TOML text.

    Args:
        data: The TOML document as text.
        source: Where it came from, for the error message.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the text is not TOML, or does not satisfy the
            configuration's shape. `tomllib.TOMLDecodeError` never escapes.
    """
    try:
        document = tomllib.loads(data)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"not valid TOML ({exc})", source=source) from None
    return validate_config(document, source=source)


def read_config_document(path: str | Path) -> dict[str, Any]:
    """Read a configuration file into a parsed document, without validating it.

    The merge in `idc.cli.main.configuration` needs the document before the
    command line is applied to it, so that the two halves are validated
    together and a flag can repair what the file left out.

    Args:
        path: The TOML file, read as bytes by `tomllib`.

    Returns:
        The parsed document.

    Raises:
        ConfigError: If the file cannot be read or is not TOML. The message
            names the file.
    """
    source = Path(path)
    try:
        with source.open("rb") as handle:
            return tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot be read ({exc.strerror or exc})", source=str(source)) from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"not valid TOML ({exc})", source=str(source)) from None


def load_config(path: str | Path) -> Config:
    """Read and validate a configuration file.

    Args:
        path: The TOML file.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the file cannot be read, is not TOML, or does not
            satisfy the configuration's shape. The message names the file.
    """
    return validate_config(read_config_document(path), source=str(Path(path)))
