"""Regenerate the committed JSON Schema document: `python -m idc.schema`.

This is a development utility, not part of the tool's command surface (T10.1):
`make schema` calls it after a canonical model changed, and the sync test in
`tests/test_schema_json_schema.py` fails until it has been run.
"""

import sys

from idc.schema.json_schema import write_json_schema


def main() -> int:
    """Write the schema document and report where it went.

    Returns:
        The process exit status: always 0.
    """
    path = write_json_schema()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
