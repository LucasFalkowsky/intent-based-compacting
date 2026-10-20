"""Command surface of the tool.

The commands mirror the pipeline stages (`import`, `chunk`, `classify`,
`resolve`, `compact`, `render`, `measure`) plus the end-to-end `compact` run
used by the study, and they are the only place that reads configuration or
touches the filesystem on the user's behalf.

The commands are added in ticket T10.1; this package carries no implementations
yet. `python -m idc` therefore reports that the surface is not wired up — see
`idc.__main__`.
"""
