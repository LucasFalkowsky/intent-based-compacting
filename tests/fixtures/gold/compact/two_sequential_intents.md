## [C1] Intent: Add a `--format json` option to the `report` command. The…
### Decisions & constraints
- (none)
### Rejected approaches
- (none)
### Solution
- `src/billing/cli.py`
```
@click.option("--out", type=click.Path(), required=True)
@click.option("--format", "fmt", type=click.Choice(["csv", "json"]), default="csv")
```
- `src/billing/export.py`
```
def export_json(entries: list[Entry], target: Path) -> None:
    target.write_text(json.dumps([_format_row(e) for e in entries], indent=2))


def export_report(entries: list[Entry], target: Path) -> None:
```
### Verification
- `uv run pytest tests/billing -q` → exit 0
```
.......
7 passed in 0.31s
```

## [C2] Intent: Now document the new option in README.md, nothing else.
### Decisions & constraints
- (none)
### Rejected approaches
- (none)
### Solution
- `README.md`
```
| `--out` | — | file the report is written to |
| `--format` | `csv` | output format, `csv` or `json` |
```
### Verification
- (none)
