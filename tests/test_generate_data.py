from __future__ import annotations

from pathlib import Path

import pandas as pd

import generate_data
from create_db import REQUIRED_CAND, REQUIRED_TRAIN


def test_generate_writes_expected_row_counts(tmp_path: Path) -> None:
    train_out = tmp_path / "train.csv"
    cand_out = tmp_path / "cand.csv"
    generate_data.generate(
        n_train_users=5,
        n_train_days=10,
        n_candidate_rows=17,
        seed=1,
        train_out=train_out,
        candidates_out=cand_out,
    )
    train_df = pd.read_csv(train_out)
    cand_df = pd.read_csv(cand_out)
    assert len(train_df) == 5 * 10
    assert len(cand_df) == 17


def test_generate_matches_required_schema(tmp_path: Path) -> None:
    train_out = tmp_path / "train.csv"
    cand_out = tmp_path / "cand.csv"
    generate_data.generate(
        n_train_users=3,
        n_train_days=5,
        n_candidate_rows=4,
        seed=2,
        train_out=train_out,
        candidates_out=cand_out,
    )
    train_df = pd.read_csv(train_out)
    cand_df = pd.read_csv(cand_out)
    assert set(train_df.columns) >= REQUIRED_TRAIN
    assert set(cand_df.columns) >= REQUIRED_CAND
    assert "productivity_score" not in cand_df.columns


def test_generate_is_deterministic_for_a_given_seed(tmp_path: Path) -> None:
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    generate_data.generate(
        n_train_users=4,
        n_train_days=6,
        n_candidate_rows=5,
        seed=7,
        train_out=out_a,
        candidates_out=tmp_path / "a_cand.csv",
    )
    generate_data.generate(
        n_train_users=4,
        n_train_days=6,
        n_candidate_rows=5,
        seed=7,
        train_out=out_b,
        candidates_out=tmp_path / "b_cand.csv",
    )
    pd.testing.assert_frame_equal(pd.read_csv(out_a), pd.read_csv(out_b))


def test_generate_values_within_expected_ranges(tmp_path: Path) -> None:
    train_out = tmp_path / "train.csv"
    generate_data.generate(
        n_train_users=10,
        n_train_days=20,
        n_candidate_rows=5,
        seed=3,
        train_out=train_out,
        candidates_out=tmp_path / "cand.csv",
    )
    df = pd.read_csv(train_out)
    assert set(df["chronotype"].unique()) <= {"morning", "evening"}
    assert df["stress_level"].between(1, 5).all()
    assert df["productivity_score"].between(0, 17).all()


def test_parse_args_defaults(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["generate_data.py"])
    args = generate_data.parse_args()
    assert args.users == 100
    assert args.days == 54
    assert args.candidate_rows == 350
    assert args.train_out == "data/events_train.csv"
    assert args.candidates_out == "data/events_candidates.csv"


def test_main_writes_files(monkeypatch, tmp_path: Path) -> None:
    train_out = tmp_path / "train.csv"
    cand_out = tmp_path / "cand.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_data.py",
            "--users",
            "3",
            "--days",
            "4",
            "--candidate-rows",
            "6",
            "--train-out",
            str(train_out),
            "--candidates-out",
            str(cand_out),
        ],
    )
    generate_data.main()
    assert train_out.exists()
    assert cand_out.exists()
