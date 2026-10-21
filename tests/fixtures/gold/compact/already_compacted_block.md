## [C1] Intent: The rate limiter from before returns 429 without a…
### Decisions & constraints
- (none)
### Rejected approaches
- (none)
### Solution
- `src/api/limiter.py`
```
    @property
    def retry_after(self) -> int:
        return max(1, math.ceil(1 / self._rate))

    def take(self) -> bool:
```
- `src/api/middleware.py`
```
    if request.url.path in EXEMPT_PATHS:
        return await call_next(request)
    bucket = buckets[token_of(request)]
    if not bucket.take():
        return Response(
            status_code=429,
            headers={"Retry-After": str(bucket.retry_after)},
        )
```
### Verification
- `uv run pytest tests/api -q` → exit 0
```
.........
9 passed in 0.44s
```
