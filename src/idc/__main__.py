"""Module entry point: `python -m idc`, which `make run ARGS="…"` forwards to.

Two lines of its own and nothing more. The whole command surface — the parser,
the command table and `main` — lives in `idc.cli`, so that it can be driven
in-process by a test or by another program (`main(argv, runtime=…)` returns a
status and never exits the interpreter), and this module exists only to turn
that status into the process's.

```
python -m idc --help
python -m idc run --agent claude-code --in session.jsonl --out compacted.jsonl
make run ARGS="compact --help"
```

`main` is re-exported here because `python -m idc` is the documented invocation
and a reader who opens this file expects to find the function it calls.
"""

import sys

from idc.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
