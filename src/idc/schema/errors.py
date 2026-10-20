"""The one error type the canonical schema raises, and its formatting.

Pydantic's `ValidationError` is deliberately not allowed to escape the package:
stages (and the CLI) should handle one error type, and its message should read
like a tool's diagnostic rather than like a library traceback. Every entry point
that can fail validation — `load`/`loads`/`validate_document`, direct
construction and attribute assignment — translates pydantic's error here, so all
of them report the same located message.

This module imports nothing from the rest of the package: both
`idc.schema.models` and `idc.schema.serialization` depend on it.
"""

from typing import Any

from pydantic import BaseModel, ValidationError

__all__ = ["SchemaValidationError", "as_schema_error"]


class SchemaValidationError(ValueError):
    """A document or object does not satisfy the canonical schema.

    Attributes:
        model: Name of the canonical type the document was validated against.
        source: Where the document came from (a path, or `None` for text and for
            an object built in memory).
        errors: One `(location, message)` pair per problem found, with the
            location given as a dotted field path such as `messages.0.role`.
    """

    def __init__(
        self,
        model: str,
        reason: str,
        *,
        source: str | None = None,
        errors: tuple[tuple[str, str], ...] = (),
    ) -> None:
        """Build the error and render its message.

        Args:
            model: Name of the canonical type.
            reason: Summary line, e.g. `2 problems` or `not valid JSON`.
            source: Origin of the document, included in the message when given.
            errors: The located problems, listed one per line after the summary.
        """
        self.model = model
        self.source = source
        self.errors = errors
        origin = f" from {source}" if source else ""
        lines = [f"invalid {model} document{origin}: {reason}"]
        lines += [f"  {location}: {message}" for location, message in errors]
        super().__init__("\n".join(lines))


def _location(loc: tuple[Any, ...]) -> str:
    """Render a pydantic error location as a dotted field path."""
    return ".".join(str(part) for part in loc) or "<document>"


def as_schema_error(
    exc: ValidationError, model: type[BaseModel], source: str | None = None
) -> SchemaValidationError:
    """Translate a pydantic `ValidationError` into a `SchemaValidationError`.

    Args:
        exc: The pydantic error to translate; none of its internals (its type,
            its text or its documentation URL) survive the translation.
        model: The canonical type that was being built or validated.
        source: Where the document came from, or `None` when there is no source
            file — which is the case for construction and assignment.

    Returns:
        The equivalent `SchemaValidationError`, with one dotted field path per
        problem.
    """
    errors = tuple(
        (_location(error["loc"]), error["msg"].removeprefix("Value error, "))
        for error in exc.errors()
    )
    problems = f"{len(errors)} problem{'s' if len(errors) != 1 else ''}"
    return SchemaValidationError(model.__name__, problems, source=source, errors=errors)
