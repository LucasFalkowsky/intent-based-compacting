"""The shared fixture corpus directory exists and is discoverable."""

from pathlib import Path

from conftest import FIXTURES_DIR


def test_fixtures_dir_exists_and_is_a_directory():
    assert FIXTURES_DIR.is_dir()


def test_fixtures_dir_lives_inside_tests():
    assert FIXTURES_DIR.parent == Path(__file__).parent.resolve()
    assert FIXTURES_DIR.name == "fixtures"


def test_fixtures_dir_fixture_resolves_to_the_same_directory(fixtures_dir):
    assert fixtures_dir == FIXTURES_DIR
    assert fixtures_dir.is_dir()


def test_fixtures_dir_is_discoverable_from_any_working_directory(fixtures_dir, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert fixtures_dir.is_absolute()
    assert fixtures_dir.is_dir()
