from __future__ import annotations

from pathlib import Path

import create_db
import score_new_days
import train_regression
import utils

FIXTURES = Path(__file__).parent / "fixtures"
SQL_PATH = Path(__file__).resolve().parents[1] / "src" / "queries.sql"


def test_write_and_verify_sha256_roundtrip(tmp_path) -> None:
    f = tmp_path / "model.joblib"
    f.write_bytes(b"pretend model bytes")
    sidecar = utils.write_sha256_sidecar(f)
    assert sidecar.exists()
    assert sidecar.name == "model.joblib.sha256"
    assert utils.verify_sha256_sidecar(f) is True


def test_verify_sha256_detects_tampering(tmp_path) -> None:
    f = tmp_path / "model.joblib"
    f.write_bytes(b"original bytes")
    utils.write_sha256_sidecar(f)
    f.write_bytes(b"tampered bytes!!")
    assert utils.verify_sha256_sidecar(f) is False


def test_verify_sha256_none_when_no_sidecar(tmp_path) -> None:
    f = tmp_path / "model.joblib"
    f.write_bytes(b"no sidecar for this one")
    assert utils.verify_sha256_sidecar(f) is None


def test_train_regression_writes_checksum_sidecar(tmp_path) -> None:
    db_path = tmp_path / "cli.db"
    create_db.load_to_db(
        FIXTURES / "events_train_small.csv",
        FIXTURES / "events_candidates_small.csv",
        db_path,
    )
    outdir = tmp_path / "out"
    train_regression.run_training(db_path, SQL_PATH, outdir)
    sidecar = outdir / "model.joblib.sha256"
    assert sidecar.exists()
    assert utils.verify_sha256_sidecar(outdir / "model.joblib") is True


def test_score_new_days_runs_with_valid_checksum(tmp_path, caplog) -> None:
    db_path = tmp_path / "cli.db"
    create_db.load_to_db(
        FIXTURES / "events_train_small.csv",
        FIXTURES / "events_candidates_small.csv",
        db_path,
    )
    outdir = tmp_path / "out"
    train_regression.run_training(db_path, SQL_PATH, outdir)
    with caplog.at_level("WARNING"):
        score_new_days.score(db_path, SQL_PATH, outdir / "model.joblib", outdir)
    assert "does not match" not in caplog.text
    assert (outdir / "scored_candidates.csv").exists()


def test_score_new_days_warns_on_tampered_model(tmp_path, caplog) -> None:
    db_path = tmp_path / "cli.db"
    create_db.load_to_db(
        FIXTURES / "events_train_small.csv",
        FIXTURES / "events_candidates_small.csv",
        db_path,
    )
    outdir = tmp_path / "out"
    train_regression.run_training(db_path, SQL_PATH, outdir)

    model_path = outdir / "model.joblib"
    original_sidecar = (outdir / "model.joblib.sha256").read_text()
    (outdir / "model.joblib.sha256").write_text("0" * 64 + "\n")

    with caplog.at_level("WARNING"):
        score_new_days.score(db_path, SQL_PATH, model_path, outdir)
    assert "does not match" in caplog.text

    # restore, otherwise unrelated -- not strictly needed since tmp_path is per-test
    (outdir / "model.joblib.sha256").write_text(original_sidecar)


def test_score_new_days_info_logs_when_sidecar_missing(tmp_path, caplog) -> None:
    db_path = tmp_path / "cli.db"
    create_db.load_to_db(
        FIXTURES / "events_train_small.csv",
        FIXTURES / "events_candidates_small.csv",
        db_path,
    )
    outdir = tmp_path / "out"
    train_regression.run_training(db_path, SQL_PATH, outdir)

    model_path = outdir / "model.joblib"
    (outdir / "model.joblib.sha256").unlink()

    with caplog.at_level("INFO"):
        score_new_days.score(db_path, SQL_PATH, model_path, outdir)
    assert "No .sha256 sidecar found" in caplog.text
    assert (outdir / "scored_candidates.csv").exists()
