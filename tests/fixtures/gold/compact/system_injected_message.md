## [C1] Intent: The nightly job swallows every exception from…
### Decisions & constraints
- (none)
### Rejected approaches
- (none)
### Solution
- `/repo/src/jobs/nightly.py`
```
        except Exception:
            logger.exception("nightly sync failed for tenant %s", tenant.id)
            raise
```
### Verification
- `uv run pytest tests/jobs -q` → exit 0
```
...
3 passed in 0.22s
```
