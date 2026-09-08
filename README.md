# AI Productivity Tracker (SQL + ML + Psychology)

Predict and analyze daily productivity using **behavioral data, SQL feature engineering, and machine learning**, integrating psychological principles such as **circadian rhythm**, **stress-performance dynamics**, and **habit efficiency**.

This project demonstrates how modern data science can quantify human productivity in knowledge work, blending **psychology, data engineering, and predictive analytics**.

---

## Overview

This project models daily productivity based on personal and contextual factors such as sleep, stress, meetings, breaks, and focus patterns.  
It uses:

- **SQL (SQLite)** for feature engineering and psychological metric derivation  
- **Python (pandas, scikit-learn)** for data processing, training, and visualization  
- **ElasticNet Regression** for interpretable prediction  
- **Behavioral Science Insights** to ensure meaningful features

---

## Project Structure

```
ai-productivity-tracker/
├─ data/
│  ├─ events_train.csv
│  └─ events_candidates.csv
├─ src/
│  ├─ create_db.py
│  ├─ queries.sql
│  ├─ train_regression.py
│  ├─ score_new_days.py
│  └─ utils.py
├─ outputs/
│  ├─ metrics.json
│  ├─ feature_importance.csv
│  ├─ predictions_train.csv
│  └─ charts/
│     ├─ actual_vs_predicted.png
│     ├─ residuals_hist.png
│     └─ feature_importance.png
└─ README.md
```

---

## Data Description

| Column | Description |
|--------|--------------|
| `sleep_hours` | Hours of sleep the previous night |
| `chronotype` | Morning or evening preference |
| `focus_start_hour` | Hour when deep work begins |
| `deep_work_minutes` | Minutes of uninterrupted work |
| `meetings_minutes` | Total meeting duration |
| `late_meetings_minutes` | Evening meetings (negative for energy) |
| `breaks_count` | Number of breaks during the day |
| `avg_break_minutes` | Average break duration |
| `context_switches` | Task changes / app switches |
| `notifications` | Distractions from notifications |
| `steps`, `hydration_glasses`, `caffeine_mg` | Physical activity and health proxies |
| `stress_level`, `mood` | Psychological self-assessments |
| `productivity_score` | Target variable (0–100 scale) |

---

## Psychology-Informed Feature Engineering

Feature creation in `queries.sql` integrates **behavioral science theories**:

| Feature | Formula | Psychological Meaning |
|----------|----------|------------------------|
| `sleep_deficit` | \|sleep_hours − 8\| | Cognitive fatigue impact |
| `circadian_alignment` | match between chronotype & work start | Energy–focus match quality |
| `yerkes_arousal` | stress × (1 − stress−3 /2) | Optimal stress improves focus (Yerkes–Dodson Law) |
| `break_quality` | breaks × avg_break_minutes | Balance between rest and continuity |
| `meeting_load` | meetings + 1.5×late_meetings | Collaboration vs overload |
| `context_penalty` | notifications + context_switches | Distraction index |
| `health_score` | steps/10k + caffeine balance | Physical energy proxy |

These features translate psychology into quantifiable variables for regression modeling.

---

## Model Pipeline

1. **SQL Feature Engineering:** Derived features created using SQLite views (`features_train`, `features_candidates`).  
2. **Data Standardization:** Scaling numeric and one-hot encoding categorical (`chronotype`).  
3. **ElasticNet Regression:** Combines L1 and L2 regularization for interpretability and generalization.  
4. **Evaluation:** Metrics include R² and MAE with full residual diagnostics.  
5. **Visualization:** Insights and diagnostics via Matplotlib charts.

---

## How to Run

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt

# Load data into SQLite
python src/create_db.py --train data/events_train.csv --candidates data/events_candidates.csv --db productivity.db

# Train and evaluate the model
python src/train_regression.py --db productivity.db --sql src/queries.sql --outdir outputs

# Score new days (unlabeled)
python src/score_new_days.py --db productivity.db --sql src/queries.sql --model outputs/model.joblib --outdir outputs
```

Outputs include metrics, predictions, and charts under `outputs/`.

---

## Results and Visualizations

### ** Actual vs Predicted Productivity**
<img width="1050" height="900" alt="actual_vs_predicted" src="https://github.com/user-attachments/assets/f945a765-53d2-4461-8dde-e0827ff57291" />

**Interpretation:**  
- Points close to the diagonal (y=x) show accurate predictions.  
- Strong correlation indicates the model captures real productivity behavior.  
- Minor deviation near extreme productivity values is expected due to behavioral noise.  

**Insight:**  
The model explains individual productivity patterns effectively, balancing accuracy and interpretability.

---

### ** Feature Importance (Standardized Coefficients)**
<img width="1350" height="750" alt="feature_importance" src="https://github.com/user-attachments/assets/6a5741f6-3074-40c4-a4d7-79bad5f1a718" />

| Feature | Direction | Meaning |
|----------|------------|----------|
| `yerkes_arousal` | ↑ | Moderate stress enhances focus (Yerkes–Dodson Law) |
| `deep_work_minutes` | ↑ | More deep work → higher productivity |
| `circadian_alignment` | ↑ | Starting work at optimal time improves flow |
| `meeting_load` | ↓ | Too many meetings reduce focus time |
| `context_penalty` | ↓ | Distractions lower overall efficiency |
| `sleep_deficit` | ↓ | Sleep deprivation strongly lowers productivity |

**Insight:**  
Productivity is a **multi-factor balance**, biological rhythm, mental stress, workload, and interruptions all interact.  
The model provides **interpretable coefficients**, not just predictions.

---

### ** Residuals Distribution (Actual − Predicted)**
<img width="1050" height="750" alt="residuals_hist" src="https://github.com/user-attachments/assets/01e682a9-fd91-4833-bc94-0f9365e3a40c" />

**Interpretation:**  
- The histogram is bell-shaped, centered near zero.  
- Indicates **no systematic bias**, model neither overpredicts nor underpredicts.  
- Small tails = few outliers (e.g., burnout or exceptional days).

**Insight:**  
Errors are random and symmetric → model generalizes well.  
Residual shape suggests stable performance and reliable psychological feature design.

---

## Key Behavioral Insights

1. **Moderate stress (arousal) improves output**, validating the **Yerkes–Dodson law**.  
2. **Circadian alignment** (working in sync with your biological clock) strongly correlates with productivity.  
3. **Sleep deficit and meeting overload** are the most consistent negative predictors.  
4. **Physical health markers** (steps, hydration) show secondary but positive effects.  
5. **Balance > quantity**, overworking past cognitive limits lowers productivity quality.

---

## Metrics Summary

| Metric | Description | Example Value |
|--------|--------------|----------------|
| `R²` | Proportion of explained variance | ~0.83 |
| `MAE` | Mean absolute error (0–100 scale) | ~3.5 |

The model explains most of the variance in daily productivity with **minimal error**, a strong result for behavioral prediction.

---

## Technologies Used

- **SQL (SQLite)**  feature computation and preprocessing  
- **Python (pandas, scikit-learn, matplotlib)**  analysis, modeling, visualization  
- **ElasticNet Regression**  interpretable linear model with regularization  
- **Joblib**  efficient model serialization  
