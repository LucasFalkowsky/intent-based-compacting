"""Chain identifiers: `C<n>`, counted from 1 by position.

A chain id is assigned by **position** and deliberately not derived from the
chain's content, exactly like a chunk id (`idc.chunk.boundaries.chunk_id`): a
re-run over a transcript that only grew has to keep the ids of the chains that
were already there, because the compactor writes them into the summary block
(T6.3) and the parser reads them back and offers those chains as candidates
again (T8). A content hash would change an id whenever a chain is extended and
break both.

The pattern is not restated here: `CHAIN_ID_PATTERN` is the canonical schema's
own constant (`Chain.id` validates against it), so this module imports it
instead of keeping a second copy that could drift from the contract.
"""

import re
from typing import TypeIs

from idc.schema.models import CHAIN_ID_PATTERN

CHAIN_ID_PREFIX = "C"
"""Chain ids are `C<n>`, counted from 1 — the `K<n>` of a chunk, for a chain."""

_CHAIN_ID = re.compile(CHAIN_ID_PATTERN)
"""The canonical schema's own chain-id pattern, compiled once."""


def chain_id(position: int) -> str:
    """Return the id of the chain at `position`.

    Args:
        position: The chain's position among the chains of the transcript,
            counted from 1.

    Returns:
        The id `C<position>`.

    Raises:
        ValueError: If `position` is smaller than 1; chains are numbered from 1,
            and `C0` is not a chain id the schema accepts.
    """
    if position < 1:
        raise ValueError(f"a chain position is counted from 1; got {position}")
    return f"{CHAIN_ID_PREFIX}{position}"


def chain_position(identifier: str) -> int:
    """Return the position encoded in a chain id.

    Args:
        identifier: A chain id of the form `C<n>`.

    Returns:
        The number `n` the id carries.

    Raises:
        ValueError: If `identifier` is not a chain id.
    """
    return int(require_chain_id(identifier)[len(CHAIN_ID_PREFIX) :])


def is_chain_id(value: object) -> TypeIs[str]:
    """Return whether `value` is a chain id of the form `C<n>`.

    Args:
        value: Anything; a non-string is simply not a chain id.

    Returns:
        `True` if the value is a string the canonical schema accepts as
        `Chain.id`. Narrowing to `str` rather than to a type of its own: a chain
        id *is* a string, and `require_chain_id` returns it as one.
    """
    return isinstance(value, str) and _CHAIN_ID.fullmatch(value) is not None


def require_chain_id(
    value: object, *, what: str = "a chain id", error: type[Exception] = ValueError
) -> str:
    """Return `value` as a chain id, or raise naming the form it must have.

    Args:
        value: The value to check.
        what: How the value is named in the error message, so that the caller's
            vocabulary (a decision's target, a candidate's id) survives into the
            message.
        error: The exception to raise. `ValueError` for a value object built
            wrongly by the caller; `AdapterContractError` where the id came from
            a backend's answer, so that the stage keeps one catchable family
            without a second copy of this message.

    Returns:
        The value unchanged, as a string.

    Raises:
        Exception: Of type `error`, if the value is not a chain id.
    """
    if is_chain_id(value):
        return value
    raise error(f"{what} must be of the form C<n> with n counted from 1; got {value!r}")
