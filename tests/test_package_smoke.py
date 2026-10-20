"""Smoke tests: the package and all of its sub-packages import cleanly."""

import importlib

import pytest

SUBPACKAGES = [
    "schema",
    "importers",
    "chunk",
    "classify",
    "resolve",
    "compact",
    "render",
    "measure",
    "cli",
]


def test_root_package_imports():
    module = importlib.import_module("idc")

    assert module.__doc__, "the root package must carry a module docstring"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    module = importlib.import_module(f"idc.{name}")

    assert module.__name__ == f"idc.{name}"
    assert module.__doc__, f"idc.{name} must carry a module docstring"


def test_package_lives_in_src_layout():
    module = importlib.import_module("idc")
    package_dir = next(iter(module.__path__))

    assert package_dir.replace("\\", "/").endswith("src/idc")
