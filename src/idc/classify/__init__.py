"""Classification of chunks into intent chains.

A chain is a sequence of chunks that pursue the same user intent. Classification
runs through an adapter so that the LLM-backed and the S1M-backed strategies are
interchangeable, and so that tests can use a fake adapter instead of a model.
Chains parsed back out of an already-compacted transcript are offered as
candidates on the next run.

The classifier is added in ticket T4; this package carries no implementations
yet.
"""
