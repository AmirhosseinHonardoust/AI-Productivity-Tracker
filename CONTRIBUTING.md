# Contributing

Thanks for considering a contribution to AI Productivity Tracker.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pre-commit install  # optional, but runs ruff/black before each commit
```

## Quality gate

Every PR must pass:

```bash
ruff check src tests
black --check src tests
mypy --ignore-missing-imports src/*.py
pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=95
```

These are the exact checks CI (`.github/workflows/ci.yml`) runs, across
Python 3.10, 3.11, and 3.12, followed by a full pipeline smoke run
(`create_db.py` → `train_regression.py` → `score_new_days.py`) on the bundled
sample data.

## Conventions

- **Typing**: all functions have type hints; `mypy` is configured with
  `disallow_untyped_defs` and `disallow_incomplete_defs`, so new code must be
  fully typed too.
- **Imports**: heavy or rarely-needed imports (e.g. `hashlib`, `json` in
  `utils.py`) are done lazily inside the function that uses them, matching
  the existing style. Keep this pattern for similar cases.
- **Feature lists**: `NUMERIC`/`CATEGORICAL`/`TARGET` in `src/features.py`
  are the single source of truth for both training and scoring — don't
  duplicate these lists elsewhere.
- **Minimal churn**: prefer the smallest change that achieves the goal; avoid
  renaming or moving files unless necessary.
- **Behavior-preserving refactors**: if you refactor logic that affects
  model output, prove equivalence (e.g. diff `outputs/metrics.json` before
  and after) and include that in the PR description.

## Tests

- New behavior needs a test. Coverage must stay at or above 95% (currently
  100%).
- Use `tests/fixtures/events_train_small.csv` / `events_candidates_small.csv`
  for fast unit tests instead of the full `data/*.csv`.
- `tests/conftest.py` puts `src/` on `sys.path` and sets a headless
  matplotlib backend — no extra setup needed in individual test files.

## Reporting issues

Open a GitHub issue with steps to reproduce. For anything security-related,
see [SECURITY.md](SECURITY.md).
