"""What `idc.measure` exports, pinned name by name (T9.3, #39).

The package's `__all__` is built in three blocks, one per sub-ticket of T9
(#36), each splicing the names before it with `__all__ = [*__all__, ...]`. That
splice is the only thing standing between the three blocks and a silently
shortened export list, and nothing pinned it: the list is assembled at import
time, so a block that overwrote the names before it instead of extending them
would still import cleanly and export a subset.

This file is that pin, and follows `test_chunk_api` / `test_resolve_api`: the
expected set is written out by hand, every name in it has to be reachable from
the package, and no name may appear twice.
"""

import importlib

EXPECTED_EXPORTS = {
    "FAMILY_SEPARATOR",
    "CompressionRate",
    "ContextCount",
    "EmptyBaselineError",
    "FakeTokenizer",
    "MeasureError",
    "MessageCount",
    "ModelFamily",
    "NoTokenizerError",
    "Tokenizer",
    "TokenizerRegistry",
    "UnknownModelError",
    "count_context",
    "count_message",
    "count_messages",
    "message_parts",
    "model_family",
    "ExitCodeCount",
    "FileEdits",
    "NoToolCallsError",
    "ToolStatistics",
    "lines_added_of",
    "tool_statistics",
    "MANIFEST_VERSION",
    "Clock",
    "FakeClock",
    "FileDigest",
    "InstructionIdentity",
    "ManifestError",
    "Metrics",
    "MetricsRecord",
    "RunConfig",
    "RunManifest",
    "RunTimer",
    "RunWindow",
    "SystemClock",
    "TokenizerIdentity",
    "build_manifest",
    "digest_bytes",
    "digest_file",
    "dump_manifest",
    "dumps_manifest",
    "load_manifest",
    "loads_manifest",
    "measure",
}
"""Every name `idc.measure` exports, across the three sub-ticket blocks."""


def test_public_exports_are_declared_and_importable():
    module = importlib.import_module("idc.measure")

    assert set(module.__all__) == EXPECTED_EXPORTS
    for name in module.__all__:
        assert hasattr(module, name), name


def test_no_name_is_exported_twice():
    """Three blocks splicing one list is how a name gets in there twice."""
    module = importlib.import_module("idc.measure")

    assert len(module.__all__) == len(set(module.__all__))
