## [C1] Intent: The CSV export of the billing report writes naive…
### Decisions & constraints
- (none)
### Rejected approaches
- (none)
### Solution
- `src/billing/export.py`
```
        "booked_at": entry.booked_at.astimezone(UTC).isoformat(),
```
### Verification
- `uv run pytest tests/billing/test_export.py -q` → exit 0
```
..
2 passed in 0.14s
```
