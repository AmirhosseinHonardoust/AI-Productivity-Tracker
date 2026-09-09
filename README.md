<div align="center">

# AI Productivity Tracker
<img width="1672" height="941" alt="AI-Productivity-Tracker" src="https://github.com/user-attachments/assets/b7442878-8286-45c5-9b2a-1071635a96b1" />

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![SQL](https://img.shields.io/badge/SQL-SQLite%20Feature%20Engineering-informational)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ElasticNet%20Regression-orange)
![Honest ML](https://img.shields.io/badge/Honest%20ML-Model%20Comparison-green)
![Status](https://img.shields.io/badge/Status-Educational%20ML%20Project-purple)
[![CI](https://github.com/AmirhosseinHonardoust/AI-Productivity-Tracker/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AmirhosseinHonardoust/AI-Productivity-Tracker/actions/workflows/ci.yml)

</div>

A responsible machine learning project that turns daily behavioral logs into a **productivity signal**, using **SQL feature engineering** grounded in psychological theory, an **ElasticNet regression** pipeline with **honest model-comparison evaluation**, **checksum-verified model loading**, a **synthetic data generator**, and **command-line training and scoring**.

> **Important:** This project is an **educational productivity-modeling demo**, not a validated psychological or occupational-health instrument.
>
> The model, features, and reports are designed to demonstrate a professional, honest regression workflow. They estimate a productivity score from self-reported daily behavior in a small synthetic dataset; they do not diagnose burnout, stress disorders, or any health condition, and should not be used to evaluate, rank, or monitor real people.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [What This Project Does Not Do](#what-this-project-does-not-do)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training and Evaluation](#training-and-evaluation)
- [Model Selection and Honest Evaluation](#model-selection-and-honest-evaluation)
- [Model Output](#model-output)
- [Model Artifacts and Loading Safety](#model-artifacts-and-loading-safety)
- [Evaluation Metrics](#evaluation-metrics)
- [Visual Reports](#visual-reports)
- [Testing and CI](#testing-and-ci)
- [Code Quality](#code-quality)
- [Limitations](#limitations)
- [Responsible Use](#responsible-use)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

Productivity modeling is often presented as if a handful of behavioral metrics can score how well someone is working. In reality, a regression model can only quantify correlation within its training data, not causally explain a person's day. A model score is only useful if it can support a defensible action:

- surface which behavioral factors move the score, and in which direction
- disclose how much variance the model actually explains, honestly
- compare against alternative model families instead of assuming the chosen one is best
- keep training and scoring feature lists in sync so they can't silently drift apart

This project demonstrates an end-to-end, honest regression workflow on a labeled behavioral dataset. It includes SQL-based feature engineering informed by behavioral science, model training with diagnostic charts, a documented model comparison, checksum-verified artifacts, and a synthetic data generator for reproducing or resizing the sample data.

The goal is to show how a regression model can be turned into an **interpretable, honestly-evaluated analysis tool**, not just a single R² number.

---

## What This Project Does

This project can:

- Load daily behavioral CSVs into SQLite with column and range validation
- Derive psychology-informed features via SQL views (`queries.sql`)
- Train an interpretable ElasticNet regression model
- Report R² and MAE on a held-out test split, honestly
- Compare against a tuned ElasticNet, a Random Forest, and a Gradient Boosting model to justify the choice
- Generate diagnostic charts: actual vs. predicted, residuals, and feature importance
- Score new, unlabeled "candidate" days with the trained model
- Save model artifacts with SHA-256 checksum sidecars
- Verify artifact integrity before unpickling on load
- Generate a fresh synthetic dataset of any size with `generate_data.py`
- Run automated tests and CI across Python 3.10, 3.11, and 3.12
- Enforce a strict quality gate (ruff, black, mypy, 95%+ coverage)

---

## What This Project Does Not Do

This project does **not**:

- Diagnose burnout, stress disorders, or any medical or psychological condition
- Track real biometric or workplace data, the bundled data is synthetic
- Guarantee its psychology-based feature formulas are clinically validated
- Replace a wearable device, therapist, manager, or productivity coach
- Achieve strong explanatory power out of the box, R² ≈ 0.32 is a moderate, disclosed ceiling
- Personalize per user, it is a single pooled model across all users, not a per-person model
- Monitor, rank, or evaluate real employees

A real workplace analytics system would need consented real-world data, fairness review, per-user baselines, and human oversight.

---

## Key Features

- **SQL feature engineering** deriving psychology-informed signals (circadian alignment, Yerkes–Dodson arousal, break quality, meeting load, distraction penalty, health score)
- **ElasticNet regression** baseline with L1/L2 regularization for interpretable coefficients
- **Shared `features.py` module** so training and scoring read the same feature list (no train/serve skew)
- **Documented model comparison** against ElasticNetCV, Random Forest, and Gradient Boosting
- **Checksum-verified model loading** with SHA-256 sidecars
- **Synthetic data generator** (`generate_data.py`) to regenerate or resize the sample dataset
- **Command-line training and scoring** with logged, reproducible outputs
- **Unit tests and GitHub Actions CI** across three Python versions
- **Strict typing** (`mypy` with `disallow_untyped_defs`) and 100% test coverage
- **Pre-commit hooks, Makefile, and contributor/security documentation**

---

## System Workflow

```text
Raw daily behavioral logs (CSV)
        ↓
SQLite load + column/range validation (create_db.py)
        ↓
SQL feature engineering (queries.sql)
        ↓
Standardization + one-hot encoding
        ↓
ElasticNet regression (train_regression.py)
        ↓
Metrics, diagnostics, and charts
        ↓
Checksum-verified model artifact
        ↓
Scoring new/candidate days (score_new_days.py)
```

---

## Project Structure

```text
AI-Productivity-Tracker/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── events_train.csv
│   └── events_candidates.csv
│
├── outputs/
│   ├── charts/
│   │   ├── actual_vs_predicted.png
│   │   ├── residuals_hist.png
│   │   └── feature_importance.png
│   ├── metrics.json
│   ├── feature_importance.csv
│   ├── predictions_train.csv
│   ├── scored_candidates.csv
│   ├── model.joblib
│   └── model.joblib.sha256
│
├── src/
│   ├── create_db.py
│   ├── queries.sql
│   ├── features.py
│   ├── generate_data.py
│   ├── train_regression.py
│   ├── score_new_days.py
│   └── utils.py
│
├── tests/
│   ├── fixtures/
│   ├── conftest.py
│   ├── test_checksum.py
│   ├── test_cli.py
│   ├── test_create_db.py
│   ├── test_generate_data.py
│   ├── test_pipeline_smoke.py
│   ├── test_queries_sql.py
│   └── test_utils.py
│
├── README.md
├── CONTRIBUTING.md
├── SECURITY.md
├── Makefile
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirhosseinHonardoust/AI-Productivity-Tracker.git
cd AI-Productivity-Tracker
```

### 2. Create a Virtual Environment

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

For development tools (pytest, Ruff, Black, mypy):

```bash
pip install -r requirements-dev.txt
```

---

## Quick Start

Load data into SQLite:

```bash
python src/create_db.py --train data/events_train.csv --candidates data/events_candidates.csv --db productivity.db
```

Train and evaluate the model:

```bash
python src/train_regression.py --db productivity.db --sql src/queries.sql --outdir outputs
```

Score new (unlabeled) days:

```bash
python src/score_new_days.py --db productivity.db --sql src/queries.sql --model outputs/model.joblib --outdir outputs
```

All three steps can also be run with `make pipeline`.

---

## Training and Evaluation

The training script loads engineered features from SQLite, splits the data, trains the ElasticNet pipeline, evaluates on a held-out test set, writes diagnostic charts, and saves a checksum-verified model artifact.

```bash
python src/train_regression.py \
  --db productivity.db \
  --sql src/queries.sql \
  --outdir outputs
```

Generated outputs include:

```text
outputs/metrics.json
outputs/feature_importance.csv
outputs/predictions_train.csv
outputs/model.joblib
outputs/model.joblib.sha256
outputs/charts/
```

---

## Model Selection and Honest Evaluation

`productivity_score` is a continuous target with real noise, so the honest question is not "can we predict it perfectly" but "how much of the variance can these features actually explain." Rather than assume the chosen model is best, the project compares it against three alternatives on the same train/test split:

<div align="center">

| Model | R² | MAE |
|---|---|---|
| ElasticNet (alpha=0.06, l1_ratio=0.25), current | 0.318 | 2.17 |
| ElasticNetCV (tuned alpha/l1_ratio) | 0.318 | 2.16 |
| RandomForestRegressor (300 trees) | 0.282 | 2.24 |
| GradientBoostingRegressor | 0.286 | 2.23 |

</div>

That similarity across a linear model and two non-linear ensembles suggests **R² ≈ 0.32 is close to the ceiling** for what these particular features predict, rather than a modeling choice leaving easy accuracy on the table. ElasticNet is kept because it ties for best and its coefficients are directly interpretable (`outputs/feature_importance.csv`), which the tree ensembles don't offer as cleanly.

> A moderate R² here is expected and disclosed, not a bug. Improving on it would mean better features (multi-day rolling trends, per-user baselines), not a different algorithm.

---

## Model Output

Scoring writes one row per candidate day:

<div align="center">

| Column | Meaning |
|---|---|
| `user_id` | Identifier of the person the day belongs to |
| `date` | The calendar date being scored |
| `predicted_productivity` | The model's continuous productivity estimate |

</div>

Example row from `outputs/scored_candidates.csv`:

```text
user_id,date,predicted_productivity
U000,2024-01-01,5.933999657857651
```

There is no classification threshold or label, the pipeline reports a continuous estimate, not a decision.

---

## Model Artifacts and Loading Safety

The trained model is stored as a joblib/pickle file. **Deserializing a pickle executes arbitrary code**, so only load artifacts you produced yourself or fully trust, never one downloaded from an untrusted source.

As an integrity safeguard, training writes a checksum sidecar next to the model, and loading verifies it:

```text
outputs/model.joblib
outputs/model.joblib.sha256
```

`utils.verify_sha256_sidecar()` checks the sidecar before `score_new_days.py` unpickles the model and logs a warning on mismatch.

> The checksum guards against accidental corruption or a casual file swap, not a determined attacker. See `SECURITY.md` for the full threat model.

---

## Evaluation Metrics

Evaluation uses a fixed train/test split (`random_state=42`, reproducible).

<div align="center">

| Metric | Why it matters |
|---|---|
| R² | Proportion of variance in `productivity_score` explained by the model |
| MAE | Mean absolute error, in the same units as `productivity_score` |
| Coefficient magnitude | Which features drive the prediction, and in which direction |

</div>

Actual numbers from the bundled sample data:

<div align="center">

| Metric | Value (`outputs/metrics.json`) |
|---|---|
| R² | 0.318 |
| MAE | 2.17 |

</div>

> R² of ~0.32 means the model explains roughly a third of the variance in `productivity_score`, a moderate result, not a strong one. Run the pipeline yourself to reproduce these values, or regenerate the dataset with `generate_data.py` to see how they shift.

---

## Visual Reports

### Model evaluation charts

<div align="center">

| Actual vs. Predicted | Feature Importance |
|---|---|
| <img width="420" alt="Actual vs predicted" src="https://github.com/user-attachments/assets/f945a765-53d2-4461-8dde-e0827ff57291" /> | <img width="420" alt="Feature importance" src="https://github.com/user-attachments/assets/6a5741f6-3074-40c4-a4d7-79bad5f1a718" /> |
| **Analysis:** Points close to the diagonal show accurate predictions. The spread near extreme productivity values is expected given the moderate R², the model captures the broad trend, not every day. | **Analysis:** `yerkes_arousal` and `deep_work_minutes` dominate, both increasing predicted productivity, while `sleep_deficit` and `meeting_load` pull it down, consistent with the behavioral-science framing in `queries.sql`. |

</div>

<details>
<summary>Additional residuals chart</summary>

<div align="center">
        
| Residuals Chart |
|---|
| <img width="420" alt="Residuals distribution" src="https://github.com/user-attachments/assets/01e682a9-fd91-4833-bc94-0f9365e3a40c" /> |
| The residuals histogram is bell-shaped and centered near zero, indicating no systematic over- or under-prediction. Small tails suggest a handful of outlier days the linear model doesn't capture well. |

</div>

</details>

---

## Testing and CI

Run unit tests locally:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=95
```

Lint, format-check, and type-check:

```bash
ruff check src tests
black --check src tests
mypy --ignore-missing-imports src/*.py
```

All three can also be run with `make check`.

The GitHub Actions workflow checks, across Python 3.10, 3.11, and 3.12:

- dependency installation
- linting with Ruff (`src` and `tests`)
- formatting with Black (`src` and `tests`)
- type checking with mypy
- unit tests with a 95% coverage gate (currently at 100%)
- a full pipeline smoke run: `create_db.py` → `train_regression.py` → `score_new_days.py`

CI is defined in:

```text
.github/workflows/ci.yml
```

---

## Code Quality

The project separates responsibilities across modules:

<div align="center">

| Module | Purpose |
|---|---|
| `src/create_db.py` | Loads and validates CSVs into SQLite, with non-fatal range warnings |
| `src/queries.sql` | Derives psychology-informed features as SQLite views |
| `src/features.py` | Single source of truth for the feature list shared by train and score |
| `src/generate_data.py` | Generates a fresh synthetic dataset matching the bundled schema |
| `src/train_regression.py` | Trains the ElasticNet pipeline, evaluates, writes charts and artifacts |
| `src/score_new_days.py` | Verifies artifact integrity and scores candidate days |
| `src/utils.py` | Shared I/O, plotting, and checksum helpers |

</div>

Tooling is configured through `pyproject.toml` (Ruff, Black, mypy, pytest) and `requirements-dev.txt`, with `.pre-commit-config.yaml` and a `Makefile` for local convenience.

---

## Limitations

This project has important limitations:

- The dataset is a small, synthetic sample, not real workplace telemetry
- The model does not understand causality, it reports correlation only
- R² ≈ 0.32 is a genuine ceiling for these features, not a tuning gap
- The model is pooled across users, not personalized to any individual
- The psychology-informed feature formulas are illustrative, not clinically validated
- The bundled data cannot demonstrate real-world generalization to unseen users

The project is strongest as a portfolio demonstration of honest, well-tested regression workflow design.

---

## Responsible Use

This repository is intended for:

- machine learning and data engineering education
- demonstrating honest, comparison-based model evaluation
- practicing SQL-driven feature engineering
- exploring interpretable linear models
- responsible-ML documentation practice
- portfolio demonstration

It should not be used as-is for:

- monitoring, ranking, or scoring real employees or students
- making hiring, performance, or disciplinary decisions
- diagnosing burnout, stress, or any health condition
- workplace surveillance of any kind
- any high-stakes decision about a real person

Any real deployment would require consented data, fairness review, per-user modeling, and human oversight.

---

## Future Improvements

Potential next improvements:

- Add multi-day rolling trends and per-user baselines as features
- Add calibration/uncertainty estimates around each prediction
- Add SHAP or permutation-importance explanations
- Explore per-user or hierarchical models instead of a single pooled model
- Add a lightweight dashboard for interactive score review
- Add Docker support for one-command reproduction
- Validate the psychology-informed feature formulas against real behavioral research

---

## Tech Stack

- Python
- pandas
- SQLite
- scikit-learn
- matplotlib
- joblib
- pytest
- Ruff
- Black
- mypy
- pre-commit
- GitHub Actions

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is licensed under the [MIT License](LICENSE).

It is intended for educational, research, and portfolio purposes. If you use or modify this project, please keep the responsible-use notes and limitations clear.
