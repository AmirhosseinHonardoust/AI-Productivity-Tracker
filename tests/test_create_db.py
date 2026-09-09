from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import create_db

FIXTURES = Path(__file__).parent / "fixtures"


def test_validate_columns_passes_on_full_frame() -> None:
    df = pd.read_csv(FIXTURES / "events_train_small.csv")
    create_db._validate_columns(df, create_db.REQUIRED_TRAIN, "events_train")  # no raise


def test_validate_columns_raises_on_missing() -> None:
    df = pd.read_csv(FIXTURES / "events_train_small.csv").drop(columns=["sleep_hours"])
    with pytest.raises(ValueError, match="sleep_hours"):
        create_db._validate_columns(df, create_db.REQUIRED_TRAIN, "events_train")


def test_warn_on_unexpected_ranges_flags_bad_chronotype_and_stress(caplog) -> None:
    df = pd.read_csv(FIXTURES / "events_train_small.csv")
    with caplog.at_level("WARNING"):
        create_db._warn_on_unexpected_ranges(df, "events_train")
    text = caplog.text
    assert "neutral" in text
    assert "stress_level" in text


def test_warn_on_unexpected_ranges_silent_on_clean_data(caplog) -> None:
    df = pd.DataFrame(
        {
            "chronotype": ["morning", "evening"],
            "stress_level": [2, 4],
        }
    )
    with caplog.at_level("WARNING"):
        create_db._warn_on_unexpected_ranges(df, "events_train")
    assert caplog.text == ""


def test_load_to_db_creates_both_tables(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    create_db.load_to_db(
        FIXTURES / "events_train_small.csv",
        FIXTURES / "events_candidates_small.csv",
        db_path,
    )
    with sqlite3.connect(db_path) as con:
        train_count = con.execute("SELECT COUNT(*) FROM events_train").fetchone()[0]
        cand_count = con.execute("SELECT COUNT(*) FROM events_candidates").fetchone()[0]
    assert train_count == 20
    assert cand_count == 3


def test_read_csv_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        create_db._read_csv(tmp_path / "does_not_exist.csv")
