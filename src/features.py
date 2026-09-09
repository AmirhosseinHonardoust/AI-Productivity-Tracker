#!/usr/bin/env python3
"""Shared feature definitions for training and scoring.

Single source of truth for the numeric/categorical/target columns consumed by
`train_regression.py` and `score_new_days.py`, so the two stay in sync with
each other and with the columns produced by `queries.sql`.
"""

from __future__ import annotations

from typing import Final

NUMERIC: Final[list[str]] = [
    "sleep_hours",
    "focus_start_hour",
    "deep_work_minutes",
    "meetings_minutes",
    "late_meetings_minutes",
    "breaks_count",
    "avg_break_minutes",
    "context_switches",
    "notifications",
    "steps",
    "stress_level",
    "mood",
    "caffeine_mg",
    "hydration_glasses",
    "sleep_deficit",
    "circadian_alignment",
    "yerkes_arousal",
    "break_quality",
    "meeting_load",
    "context_penalty",
    "health_score",
]
CATEGORICAL: Final[list[str]] = ["chronotype"]
TARGET: Final[str] = "productivity_score"
