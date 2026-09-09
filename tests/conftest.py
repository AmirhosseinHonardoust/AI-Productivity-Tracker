from __future__ import annotations

import os
import sys
from pathlib import Path

# Headless backend so chart-generating tests don't need a display (esp. in CI).
os.environ.setdefault("MPLBACKEND", "Agg")

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
