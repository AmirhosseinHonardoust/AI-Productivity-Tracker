#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Final

import joblib
import pandas as pd

from features import CATEGORICAL, NUMERIC
from utils import ensure_outdir, save_csv, verify_sha256_sidecar

LOGGER_NAME: Final[str] = "ai_productivity.score"


def _load_features(db_path: Path, sql_path: Path) -> pd.DataFrame:
    sql_text = Path(sql_path).read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as con:
        con.executescript(sql_text)
        df = pd.read_sql_query("SELECT * FROM features_candidates;", con)
    if df.empty:
        raise RuntimeError("features_candidates is empty. Check your data and queries.sql.")
    return df


def score(db_path: Path, sql_path: Path, model_path: Path, outdir: Path) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    outdir = ensure_outdir(outdir)

    checksum_ok = verify_sha256_sidecar(model_path)
    if checksum_ok is False:
        logger.warning(
            "%s does not match its .sha256 sidecar. The file may be corrupted or "
            "tampered with. `joblib.load` executes arbitrary code for untrusted "
            "pickle files -- only proceed if you trust this file's origin.",
            model_path,
        )
    elif checksum_ok is None:
        logger.info("No .sha256 sidecar found for %s; skipping integrity check.", model_path)

    df = _load_features(db_path, sql_path)
    pipe = joblib.load(model_path)

    X = df[NUMERIC + CATEGORICAL]
    preds = pipe.predict(X)

    out = df[["user_id", "date"]].copy()
    out["predicted_productivity"] = preds
    save_csv(out, outdir / "scored_candidates.csv")
    logger.info("Scored %d candidate days -> %s", len(out), outdir.resolve())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score candidate days using trained productivity model."
    )
    p.add_argument("--db", default="productivity.db", help="SQLite DB path")
    p.add_argument("--sql", default="src/queries.sql", help="Path to queries.sql")
    p.add_argument("--model", default="outputs/model.joblib", help="Fitted model path")
    p.add_argument("--outdir", default="outputs", help="Output directory")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    score(Path(args.db), Path(args.sql), Path(args.model), Path(args.outdir))


if __name__ == "__main__":
    main()
