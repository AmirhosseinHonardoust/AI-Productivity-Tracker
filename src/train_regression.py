#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import (
    ensure_outdir,
    plot_bar,
    plot_hist,
    plot_scatter_actual_vs_pred,
    save_csv,
    save_json,
)

LOGGER_NAME: Final[str] = "ai_productivity.train"

NUMERIC: Final[list[str]] = [
    "sleep_hours", "focus_start_hour", "deep_work_minutes", "meetings_minutes",
    "late_meetings_minutes", "breaks_count", "avg_break_minutes",
    "context_switches", "notifications", "steps", "stress_level", "mood",
    "caffeine_mg", "hydration_glasses", "sleep_deficit", "circadian_alignment",
    "yerkes_arousal", "break_quality", "meeting_load", "context_penalty", "health_score",
]
CATEGORICAL: Final[list[str]] = ["chronotype"]
TARGET: Final[str] = "productivity_score"


def _load_features(db_path: Path, sql_path: Path) -> pd.DataFrame:
    sql_text = Path(sql_path).read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as con:
        con.executescript(sql_text)
        df = pd.read_sql_query("SELECT * FROM features_train;", con)
    if df.empty:
        raise RuntimeError("features_train is empty. Check your data and queries.sql.")
    return df


def run_training(db_path: Path, sql_path: Path, outdir: Path) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    outdir = ensure_outdir(outdir)
    charts = ensure_outdir(outdir / "charts")

    df = _load_features(db_path, sql_path)
    X = df[NUMERIC + CATEGORICAL].copy()
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ],
        remainder="drop",
    )
    model = ElasticNet(alpha=0.06, l1_ratio=0.25, random_state=42)
    pipe = Pipeline([("pre", pre), ("model", model)])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = {"r2": float(r2_score(y_test, y_pred)), "mae": float(mean_absolute_error(y_test, y_pred))}
    save_json(metrics, outdir / "metrics.json")
    logger.info("Metrics: %s", metrics)

    # Diagnostics on full set
    y_all = pipe.predict(X)
    preds = df[["user_id", "date"]].copy()
    preds["actual_productivity"] = y.values
    preds["predicted_productivity"] = y_all
    save_csv(preds, outdir / "predictions_train.csv")

    plot_scatter_actual_vs_pred(
        y_true=y, y_pred=y_all, out_path=charts / "actual_vs_predicted.png",
        title="Actual vs Predicted Productivity",
    )
    plot_hist(y - y_all, charts / "residuals_hist.png", title="Residuals (Actual − Predicted)")

    # Coefficients (std. scale + one-hot cats)
    coefs = pipe.named_steps["model"].coef_
    cat_names = list(pipe.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
    feature_names = NUMERIC + cat_names
    importance = pd.DataFrame({"feature": feature_names, "importance": coefs})
    importance = importance.reindex(np.abs(importance["importance"]).sort_values(ascending=False).index).reset_index(drop=True)
    save_csv(importance, outdir / "feature_importance.csv")
    plot_bar(importance["feature"][:20], importance["importance"][:20], charts / "feature_importance.png",
             title="Feature Importance (Std. Coefficients)")

    joblib.dump(pipe, outdir / "model.joblib")
    logger.info("Artifacts saved to: %s", outdir.resolve())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train productivity regression model (ElasticNet).")
    p.add_argument("--db", default="productivity.db", help="SQLite DB path")
    p.add_argument("--sql", default="src/queries.sql", help="Path to queries.sql")
    p.add_argument("--outdir", default="outputs", help="Output directory")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    run_training(Path(args.db), Path(args.sql), Path(args.outdir))


if __name__ == "__main__":
    main()
