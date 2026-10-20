"""Canonical transcript schema — the contract between all pipeline stages.

Every stage consumes and produces canonical objects defined here, so that the
agent-specific shapes exist only inside `idc.importers` and `idc.render`.
Changing this schema is a ticket of its own, never a side effect of another
change.

The models are added in ticket T1.2; this package carries no definitions yet.
"""
