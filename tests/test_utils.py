from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import utils


def test_ensure_outdir_creates_nested(tmp_path) -> None:
    target = tmp_path / "a" / "b" / "c"
    result = utils.ensure_outdir(target)
    assert result == target
    assert target.is_dir()


def test_save_csv_roundtrip(tmp_path) -> None:
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    out = utils.save_csv(df, tmp_path / "sub" / "data.csv")
    assert out.exists()
    loaded = pd.read_csv(out)
    pd.testing.assert_frame_equal(loaded, df)


def test_save_json_roundtrip(tmp_path) -> None:
    obj = {"r2": 0.5, "mae": 1.2}
    out = utils.save_json(obj, tmp_path / "metrics.json")
    assert json.loads(out.read_text()) == obj


def test_plot_scatter_actual_vs_pred_writes_file(tmp_path) -> None:
    out = utils.plot_scatter_actual_vs_pred([1, 2, 3], [1.1, 1.9, 3.2], tmp_path / "s.png")
    assert Path(out).stat().st_size > 0


def test_plot_hist_writes_file(tmp_path) -> None:
    out = utils.plot_hist([0.1, -0.2, 0.3, 0.0], tmp_path / "h.png", title="Residuals")
    assert Path(out).stat().st_size > 0


def test_plot_bar_writes_file(tmp_path) -> None:
    out = utils.plot_bar(["a", "b"], [0.5, -0.3], tmp_path / "b.png")
    assert Path(out).stat().st_size > 0
