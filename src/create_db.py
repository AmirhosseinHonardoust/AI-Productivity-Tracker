#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Final

import pandas as pd

LOGGER_NAME: Final[str] = "ai_productivity.create_db"
REQUIRED_TRAIN: Final[set[str]] = {
    "user_id", "date", "sleep_hours", "chronotype", "focus_start_hour",
    "deep_work_minutes", "meetings_minutes", "late_meetings_minutes",
    "breaks_count", "avg_break_minutes", "context_switches", "notifications",
    "steps", "stress_level", "mood", "caffeine_mg", "hydration_glasses",
    "productivity_score",
}
REQUIRED_CAND: Final[set[str]] = REQUIRED_TRAIN - {"productivity_score"}

SCHEMA_TRAIN: Final[str] = """
CREATE TABLE IF NOT EXISTS events_train (
    user_id TEXT, date TEXT, sleep_hours REAL, chronotype TEXT, focus_start_hour INTEGER,
    deep_work_minutes INTEGER, meetings_minutes INTEGER, late_meetings_minutes INTEGER,
    breaks_count INTEGER, avg_break_minutes REAL, context_switches INTEGER, notifications INTEGER,
    steps INTEGER, stress_level INTEGER, mood REAL, caffeine_mg INTEGER, hydration_glasses INTEGER,
    productivity_score REAL
);
"""
SCHEMA_CAND: Final[str] = """
CREATE TABLE IF NOT EXISTS events_candidates (
    user_id TEXT, date TEXT, sleep_hours REAL, chronotype TEXT, focus_start_hour INTEGER,
    deep_work_minutes INTEGER, meetings_minutes INTEGER, late_meetings_minutes INTEGER,
    breaks_count INTEGER, avg_break_minutes REAL, context_switches INTEGER, notifications INTEGER,
    steps INTEGER, stress_level INTEGER, mood REAL, caffeine_mg INTEGER, hydration_glasses INTEGER
);
"""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _validate_columns(df: pd.DataFrame, required: set[str], table: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for {table}: {sorted(missing)}")


def load_to_db(train_csv: Path, cand_csv: Path, db_path: Path) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    with sqlite3.connect(db_path) as con:
        con.execute(SCHEMA_TRAIN)
        con.execute(SCHEMA_CAND)

        train_df = _read_csv(train_csv)
        _validate_columns(train_df, REQUIRED_TRAIN, "events_train")
        train_df.to_sql("events_train", con, if_exists="replace", index=False)

        cand_df = _read_csv(cand_csv)
        _validate_columns(cand_df, REQUIRED_CAND, "events_candidates")
        cand_df.to_sql("events_candidates", con, if_exists="replace", index=False)

    logger.info("Loaded CSVs into %s: [events_train, events_candidates]", db_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load productivity CSVs into SQLite.")
    p.add_argument("--train", required=True, help="Path to events_train.csv")
    p.add_argument("--candidates", required=True, help="Path to events_candidates.csv")
    p.add_argument("--db", default="productivity.db", help="SQLite DB output path")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    load_to_db(Path(args.train), Path(args.candidates), Path(args.db))


if __name__ == "__main__":
    main()
