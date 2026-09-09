from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

SQL_PATH = Path(__file__).resolve().parents[1] / "src" / "queries.sql"


@pytest.fixture
def con():
    con = sqlite3.connect(":memory:")
    con.execute("""
        CREATE TABLE events_train (
            user_id TEXT, date TEXT, sleep_hours REAL, chronotype TEXT,
            focus_start_hour INTEGER, deep_work_minutes INTEGER, meetings_minutes INTEGER,
            late_meetings_minutes INTEGER, breaks_count INTEGER, avg_break_minutes REAL,
            context_switches INTEGER, notifications INTEGER, steps INTEGER,
            stress_level INTEGER, mood REAL, caffeine_mg INTEGER, hydration_glasses INTEGER,
            productivity_score REAL
        );
        """)
    con.execute("""
        INSERT INTO events_train VALUES
        ('U1','2024-01-01', 6.0, 'morning', 8, 60, 90, 10, 3, 5.0, 12, 40, 8000, 4, 3.5,
         120, 5, 6.5)
        """)
    con.execute("""
        INSERT INTO events_train VALUES
        ('U2','2024-01-01', 9.0, 'evening', 20, 30, 40, 5, 1, 10.0, 3, 5, 2000, 2, 4.0, 0, 2, 3.0)
        """)
    con.executescript(SQL_PATH.read_text(encoding="utf-8"))
    yield con
    con.close()


def test_features_train_row_count(con) -> None:
    df = pd.read_sql_query("SELECT * FROM features_train;", con)
    assert len(df) == 2


def test_sleep_deficit(con) -> None:
    df = pd.read_sql_query("SELECT user_id, sleep_deficit FROM features_train;", con)
    row = df.set_index("user_id")
    assert row.loc["U1", "sleep_deficit"] == pytest.approx(abs(6.0 - 8.0))
    assert row.loc["U2", "sleep_deficit"] == pytest.approx(abs(9.0 - 8.0))


def test_circadian_alignment_morning_and_evening(con) -> None:
    df = pd.read_sql_query("SELECT user_id, circadian_alignment FROM features_train;", con)
    row = df.set_index("user_id")
    # morning: MAX(0, 10 - focus_start_hour) => MAX(0, 10-8) = 2
    assert row.loc["U1", "circadian_alignment"] == pytest.approx(2)
    # evening: MAX(0, focus_start_hour - 13) => MAX(0, 20-13) = 7
    assert row.loc["U2", "circadian_alignment"] == pytest.approx(7)


def test_yerkes_arousal(con) -> None:
    df = pd.read_sql_query("SELECT user_id, yerkes_arousal FROM features_train;", con)
    row = df.set_index("user_id")
    # stress * (1 - |stress-3|/2)
    assert row.loc["U1", "yerkes_arousal"] == pytest.approx(4 * (1 - abs(4 - 3) / 2))
    assert row.loc["U2", "yerkes_arousal"] == pytest.approx(2 * (1 - abs(2 - 3) / 2))


def test_break_quality_meeting_load_context_penalty(con) -> None:
    df = pd.read_sql_query(
        "SELECT user_id, break_quality, meeting_load, context_penalty FROM features_train;", con
    )
    row = df.set_index("user_id")
    assert row.loc["U1", "break_quality"] == pytest.approx(3 * 5.0)
    assert row.loc["U1", "meeting_load"] == pytest.approx(90 + 10 * 1.5)
    assert row.loc["U1", "context_penalty"] == pytest.approx(40 + 12)


def test_health_score_caffeine_band(con) -> None:
    df = pd.read_sql_query("SELECT user_id, health_score FROM features_train;", con)
    row = df.set_index("user_id")
    # U1: caffeine_mg=120 is within [50,200] -> +0.2 bonus
    assert row.loc["U1", "health_score"] == pytest.approx(8000 / 10000.0 + 0.2)
    # U2: caffeine_mg=0 is outside [50,200] -> no bonus
    assert row.loc["U2", "health_score"] == pytest.approx(2000 / 10000.0)
