"""The shared fixture corpus directory exists and is discoverable."""

from pathlib import Path

TESTS_DIR = Path(__file__).parent.resolve()


def test_fixtures_dir_exists_and_is_a_directory():
    assert (TESTS_DIR / "fixtures").is_dir()


def test_fixtures_dir_fixture_points_at_the_corpus(fixtures_dir):
    assert fixtures_dir == TESTS_DIR / "fixtures"
    assert fixtures_dir.is_dir()


def test_fixtures_dir_is_absolute_and_survives_a_chdir(fixtures_dir, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert fixtures_dir.is_absolute()
    assert fixtures_dir.is_dir()
