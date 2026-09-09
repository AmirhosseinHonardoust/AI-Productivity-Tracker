from __future__ import annotations

from pathlib import Path

import create_db
import score_new_days
import train_regression

FIXTURES = Path(__file__).parent / "fixtures"


def test_full_pipeline_smoke(tmp_path) -> None:
    """create_db -> train_regression -> score_new_days on tiny fixture data."""
    db_path = tmp_path / "test.db"
    sql_path = Path(__file__).resolve().parents[1] / "src" / "queries.sql"
    outdir = tmp_path / "outputs"

    create_db.load_to_db(
        FIXTURES / "events_train_small.csv",
        FIXTURES / "events_candidates_small.csv",
        db_path,
    )
    train_regression.run_training(db_path, sql_path, outdir)

    metrics_path = outdir / "metrics.json"
    model_path = outdir / "model.joblib"
    assert metrics_path.exists()
    assert model_path.exists()
    assert (outdir / "predictions_train.csv").exists()
    assert (outdir / "charts" / "actual_vs_predicted.png").exists()
    assert (outdir / "charts" / "residuals_hist.png").exists()
    assert (outdir / "charts" / "feature_importance.png").exists()

    import json

    metrics = json.loads(metrics_path.read_text())
    assert set(metrics) == {"r2", "mae"}
    assert isinstance(metrics["r2"], float)
    assert isinstance(metrics["mae"], float)

    score_new_days.score(db_path, sql_path, model_path, outdir)
    scored_path = outdir / "scored_candidates.csv"
    assert scored_path.exists()

    import pandas as pd

    scored = pd.read_csv(scored_path)
    assert len(scored) == 3
    assert "predicted_productivity" in scored.columns


def test_train_and_score_use_the_same_feature_lists() -> None:
    """Regression guard: train_regression and score_new_days must read features
    from the single shared `features` module, not independently duplicated
    lists that could drift apart.
    """
    assert train_regression.NUMERIC is score_new_days.NUMERIC
    assert train_regression.CATEGORICAL is score_new_days.CATEGORICAL
