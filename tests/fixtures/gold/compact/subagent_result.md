## [C1] Intent: We lose cents somewhere in the invoice totals. Find every…
### Decisions & constraints
- (none)
### Rejected approaches
- (none)
### Solution
- `src/billing/money.py`
```
    return Decimal(str(amount))
```
- `src/billing/invoice.py`
```
    total = Decimal("0")
```
- `src/report/totals.py`
```
    return sum((Decimal(str(value)) for value in values), Decimal("0"))
```
### Verification
- `uv run pytest tests/billing tests/report -q` → exit 0
```
............
12 passed in 0.41s
```
