## [C1] Intent: `tests/rates/test_sampler.py::test_weighted_pick` fails…
### Decisions & constraints
- (none)
### Rejected approaches
- (none)
### Solution
- `src/rates/sampler.py`
```
    def __init__(
        self, weights: dict[str, float], rng: random.Random | None = None
    ) -> None:
        self._weights = weights
        self._rng = rng or random.Random()
```
- `src/rates/sampler.py`
```
        index = bisect.bisect_right(self._cumulative, self._rng.random())
        return self._keys[min(index, len(self._keys) - 1)]
```
- `tests/rates/test_sampler.py`
```
    sampler = WeightedSampler({"a": 0.7, "b": 0.3}, random.Random(1))
```
### Verification
- `uv run pytest tests/rates -q` → exit 0
```
....
4 passed in 0.63s
```

## [C2] Intent: Unrelated: rename the config key `retry_limit` to…
### Decisions & constraints
- (none)
### Rejected approaches
- (none)
### Solution
- `config/settings.toml`
```
max_retries = 5
```
- `src/config/loader.py`
```
    max_retries: int = 3
```
### Verification
- `uv run pytest tests/config -q` → exit 0
```
....
4 passed in 0.18s
```
