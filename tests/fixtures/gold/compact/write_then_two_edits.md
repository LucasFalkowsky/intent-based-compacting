## [C1] Intent: Add a `slugify` helper in `src/text/slug.py` with tests…
### Decisions & constraints
- (none)
### Rejected approaches
- (none)
### Solution
- `/repo/tests/text/test_slug.py`
```
import pytest

from src.text.slug import slugify


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hello World", "hello-world"),
        ("Grüße, Welt!", "gruesse-welt"),
        ("a   b", "a-b"),
        ("Füße --- weg", "fuesse-weg"),
    ],
)
def test_slugify(raw, expected):
    assert slugify(raw) == expected
```
- `/repo/src/text/slug.py`
```
"""Turn arbitrary text into a URL slug."""

import re
import unicodedata

_PUNCTUATION = re.compile(r"[^a-z0-9\s-]")
_WHITESPACE = re.compile(r"[\s-]+")


def slugify(raw: str) -> str:
    """Return `raw` as a lowercase, hyphen-separated slug."""
    folded = unicodedata.normalize("NFKD", raw)
    folded = folded.encode("ascii", "ignore").decode()
    stripped = _PUNCTUATION.sub("", folded.lower())
    return _WHITESPACE.sub("-", stripped).strip("-")
```
- `/repo/src/text/slug.py`
```
    for umlaut, replacement in _GERMAN.items():
        raw = raw.replace(umlaut, replacement)
    folded = unicodedata.normalize("NFKD", raw)
```
- `/repo/src/text/slug.py`
```
    stripped = _PUNCTUATION.sub(" ", folded.lower())
    return _WHITESPACE.sub("-", stripped).strip("-")
```
### Verification
- `uv run pytest tests/text -q` → exit 0
```
....
4 passed in 0.12s
```
