## [C1] Intent: Make the retry backoff configurable: read…
### Decisions & constraints
- (none)
### Rejected approaches
- (none)
### Solution
- `src/net/backoff.py`
```
def delay_for(attempt: int) -> float:
    return settings.retry_base_delay * 2**attempt
```
- `src/net/config.py`
```
        return float(os.environ.get("RETRY_BASE_DELAY", 0.5))
```
### Verification
- `uv run pytest tests/net/test_backoff.py -q` → exit 0
```
.....
5 passed in 0.19s
```
