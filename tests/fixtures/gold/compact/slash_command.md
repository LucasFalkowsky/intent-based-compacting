## [C1] Intent: Add request-size validation to `src/api/orders.py`: a…
### Decisions & constraints
- (none)
### Rejected approaches
- (none)
### Solution
- `src/api/orders.py`
```
    payload = await request.json()
    if len(payload) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413)
```
- `src/api/orders.py`
```
    if int(request.headers.get("content-length", 0)) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413)
    payload = await request.json()
```
### Verification
- `uv run pytest tests/api -q` → exit 0
```
.......
7 passed in 0.26s
```
