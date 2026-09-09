from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import create_db
import score_new_days
import train_regression

FIXTURES = Path(__file__).parent / "fixtures"
SQL_PATH = Path(__file__).resolve().parents[1] / "src" / "queries.sql"


# --- parse_args defaults -----------------------------------------------------


def test_create_db_parse_args_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["create_db.py", "--train", "t.csv", "--candidates", "c.csv"]
    )
    args = create_db.parse_args()
    assert args.db == "productivity.db"
    assert args.log_level == "INFO"


def test_train_regression_parse_args_defaults(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["train_regression.py"])
    args = train_regression.parse_args()
    assert args.db == "productivity.db"
    assert args.sql == "src/queries.sql"
    assert args.outdir == "outputs"


def test_score_new_days_parse_args_defaults(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["score_new_days.py"])
    args = score_new_days.parse_args()
    assert args.model == "outputs/model.joblib"
    assert args.outdir == "outputs"


# --- main() end-to-end wiring (exercises the CLI entrypoints, not just the
# library functions the other tests call directly) ---------------------------


def test_create_db_main(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "cli.db"
    monkeypatch.setattr(
        "sys.argv",
        [
            "create_db.py",
            "--train",
            str(FIXTURES / "events_train_small.csv"),
            "--candidates",
            str(FIXTURES / "events_candidates_small.csv"),
            "--db",
            str(db_path),
        ],
    )
    create_db.main()
    assert db_path.exists()


def test_train_regression_main(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "cli.db"
    create_db.load_to_db(
        FIXTURES / "events_train_small.csv",
        FIXTURES / "events_candidates_small.csv",
        db_path,
    )
    outdir = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_regression.py",
            "--db",
            str(db_path),
            "--sql",
            str(SQL_PATH),
            "--outdir",
            str(outdir),
        ],
    )
    train_regression.main()
    assert (outdir / "model.joblib").exists()


def test_score_new_days_main(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "cli.db"
    create_db.load_to_db(
        FIXTURES / "events_train_small.csv",
        FIXTURES / "events_candidates_small.csv",
        db_path,
    )
    outdir = tmp_path / "out"
    train_regression.run_training(db_path, SQL_PATH, outdir)
    monkeypatch.setattr(
        "sys.argv",
        [
            "score_new_days.py",
            "--db",
            str(db_path),
            "--sql",
            str(SQL_PATH),
            "--model",
            str(outdir / "model.joblib"),
            "--outdir",
            str(outdir),
        ],
    )
    score_new_days.main()
    assert (outdir / "scored_candidates.csv").exists()


# --- empty feature view edge cases -------------------------------------------


def test_train_regression_raises_on_empty_features(tmp_path) -> None:
    db_path = tmp_path / "empty.db"
    # Valid schema, zero rows -> features_train view will also be empty.
    empty_train = tmp_path / "events_train_empty.csv"
    header = pd.read_csv(FIXTURES / "events_train_small.csv", nrows=0)
    header.to_csv(empty_train, index=False)
    empty_cand = tmp_path / "events_candidates_empty.csv"
    header.drop(columns=["productivity_score"]).to_csv(empty_cand, index=False)

    create_db.load_to_db(empty_train, empty_cand, db_path)
    with pytest.raises(RuntimeError, match="features_train is empty"):
        train_regression._load_features(db_path, SQL_PATH)


def test_score_new_days_raises_on_empty_features(tmp_path) -> None:
    db_path = tmp_path / "empty.db"
    header = pd.read_csv(FIXTURES / "events_train_small.csv", nrows=0)
    empty_train = tmp_path / "events_train_empty.csv"
    header.to_csv(empty_train, index=False)
    empty_cand = tmp_path / "events_candidates_empty.csv"
    header.drop(columns=["productivity_score"]).to_csv(empty_cand, index=False)

    create_db.load_to_db(empty_train, empty_cand, db_path)
    with pytest.raises(RuntimeError, match="features_candidates is empty"):
        score_new_days._load_features(db_path, SQL_PATH)
