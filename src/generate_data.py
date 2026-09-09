#!/usr/bin/env python3
"""Generate synthetic events_train / events_candidates CSVs.

The bundled `data/events_train.csv` and `data/events_candidates.csv` are
synthetic, but no generator was checked into the repo, so they couldn't be
regenerated or resized. This script fills that gap: it produces data with the
same schema (see `create_db.REQUIRED_TRAIN` / `REQUIRED_CAND`) and roughly the
same value ranges as the bundled sample, with `productivity_score` built from
the same psychological drivers `queries.sql` derives features from (sleep
deficit, circadian alignment, stress, meeting load, distractions, health),
plus noise -- so a freshly generated dataset behaves like a real one when run
through the existing pipeline.

This does NOT reproduce the bundled `data/*.csv` byte-for-byte; it produces a
fresh synthetic sample of the same shape and style, given a seed.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

LOGGER_NAME: Final[str] = "ai_productivity.generate_data"


def _generate_rows(n_users: int, n_days: int, rng: np.random.Generator) -> pd.DataFrame:
    n = n_users * n_days
    user_ids = [f"U{u:03d}" for u in range(n_users) for _ in range(n_days)]
    dates = [
        (pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)).date().isoformat()
        for _ in range(n_users)
        for d in range(n_days)
    ]

    chronotype = rng.choice(["morning", "evening"], size=n)
    sleep_hours = np.clip(rng.normal(7.0, 1.1, n), 3.5, 10.5)
    focus_start_hour = rng.integers(5, 23, n)
    deep_work_minutes = rng.integers(0, 301, n)
    meetings_minutes = rng.integers(0, 301, n)
    late_meetings_minutes = rng.integers(0, 61, n)
    breaks_count = rng.integers(0, 9, n)
    avg_break_minutes = np.clip(rng.normal(6.0, 3.0, n), 1.0, 20.0)
    context_switches = rng.integers(0, 51, n)
    notifications = rng.integers(0, 301, n)
    steps = rng.integers(0, 15001, n)
    stress_level = rng.integers(1, 6, n)
    mood = np.clip(rng.normal(3.5, 0.9, n), 1.0, 5.0)
    caffeine_mg = rng.integers(0, 301, n)
    hydration_glasses = rng.integers(0, 13, n)

    # Same psychological drivers as queries.sql, so the synthetic target
    # correlates with the features the model is trained on.
    sleep_deficit = np.abs(sleep_hours - 8.0)
    circadian_alignment = np.where(
        chronotype == "morning",
        np.maximum(0, 10 - focus_start_hour),
        np.maximum(0, focus_start_hour - 13),
    )
    yerkes_arousal = stress_level * (1.0 - np.abs(stress_level - 3) / 2.0)
    meeting_load = meetings_minutes + late_meetings_minutes * 1.5
    context_penalty = notifications + context_switches
    health_score = steps / 10000.0 + np.where((caffeine_mg >= 50) & (caffeine_mg <= 200), 0.2, 0.0)

    noise = rng.normal(0, 1.5, n)
    productivity_score = np.clip(
        6.0
        + 0.02 * deep_work_minutes
        + 0.3 * circadian_alignment
        + 0.4 * yerkes_arousal
        + 0.6 * health_score
        - 0.5 * sleep_deficit
        - 0.01 * meeting_load
        - 0.01 * context_penalty
        + noise,
        0.0,
        17.0,
    )

    return pd.DataFrame(
        {
            "user_id": user_ids,
            "date": dates,
            "sleep_hours": sleep_hours,
            "chronotype": chronotype,
            "focus_start_hour": focus_start_hour,
            "deep_work_minutes": deep_work_minutes,
            "meetings_minutes": meetings_minutes,
            "late_meetings_minutes": late_meetings_minutes,
            "breaks_count": breaks_count,
            "avg_break_minutes": avg_break_minutes,
            "context_switches": context_switches,
            "notifications": notifications,
            "steps": steps,
            "stress_level": stress_level,
            "mood": mood,
            "caffeine_mg": caffeine_mg,
            "hydration_glasses": hydration_glasses,
            "productivity_score": productivity_score,
        }
    )


def generate(
    n_train_users: int,
    n_train_days: int,
    n_candidate_rows: int,
    seed: int,
    train_out: Path,
    candidates_out: Path,
) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    rng = np.random.default_rng(seed)

    train_df = _generate_rows(n_train_users, n_train_days, rng)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_out, index=False)
    logger.info("Wrote %d training rows to %s", len(train_df), train_out)

    # Candidates: same schema minus the target, drawn independently.
    cand_users = max(1, -(-n_candidate_rows // max(1, n_train_days)))  # ceil division
    cand_df = _generate_rows(cand_users, n_train_days, rng)
    cand_df = cand_df.drop(columns=["productivity_score"]).head(n_candidate_rows)
    candidates_out.parent.mkdir(parents=True, exist_ok=True)
    cand_df.to_csv(candidates_out, index=False)
    logger.info("Wrote %d candidate rows to %s", len(cand_df), candidates_out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic events_train/events_candidates CSVs."
    )
    p.add_argument("--users", type=int, default=100, help="Number of distinct users")
    p.add_argument("--days", type=int, default=54, help="Days of history per user")
    p.add_argument("--candidate-rows", type=int, default=350, help="Number of candidate rows")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument(
        "--train-out", default="data/events_train.csv", help="Output path for training CSV"
    )
    p.add_argument(
        "--candidates-out",
        default="data/events_candidates.csv",
        help="Output path for candidates CSV",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    generate(
        n_train_users=args.users,
        n_train_days=args.days,
        n_candidate_rows=args.candidate_rows,
        seed=args.seed,
        train_out=Path(args.train_out),
        candidates_out=Path(args.candidates_out),
    )


if __name__ == "__main__":
    main()
