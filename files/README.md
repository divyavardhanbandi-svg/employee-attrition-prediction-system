# Employee Attrition Prediction System

A simple, end-to-end machine learning project that predicts whether an
employee is likely to leave the company (attrition), based on HR data
such as satisfaction scores, income, tenure, and work conditions.

## Project Structure

```
employee_attrition/
├── data/
│   ├── generate_data.py       # Creates a synthetic HR dataset (swap for real data)
│   └── employee_data.csv      # Generated dataset (created after running the script)
├── src/
│   ├── train_model.py         # Preprocessing, training, evaluation, model saving
│   └── predict.py             # Load saved model and score individual employees
├── models/
│   └── attrition_model.pkl    # Saved best model (created after training)
├── reports/                   # Metrics, confusion matrices, ROC curve, feature importance
├── requirements.txt
└── README.md
```

## Features Used

| Feature | Description |
|---|---|
| age | Employee's age |
| monthly_income | Monthly salary (USD) |
| years_at_company | Tenure at the company |
| years_since_last_promotion | Years since last promotion |
| years_with_current_manager | Years under current manager |
| distance_from_home | Commute distance (km) |
| job_level | Seniority level (1–5) |
| num_companies_worked | Number of previous employers |
| training_times_last_year | Trainings attended last year |
| job_satisfaction | 1 (Low) – 4 (Very High) |
| work_life_balance | 1 (Bad) – 4 (Best) |
| environment_satisfaction | 1 (Low) – 4 (Very High) |
| performance_rating | 1 (Low) – 4 (Outstanding) |
| percent_salary_hike | % salary increase last review |
| overtime | Yes / No |
| department | Sales / R&D / HR / Marketing / Engineering / Finance |
| job_role | Manager / Executive / Analyst / Technician / Representative / Director |
| marital_status | Single / Married / Divorced |
| business_travel | Non-Travel / Travel_Rarely / Travel_Frequently |

**Target:** `attrition` (1 = employee left, 0 = employee stayed)

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset (or replace data/employee_data.csv with real HR data)
python data/generate_data.py

# 3. Train and evaluate models (Logistic Regression + Random Forest)
python src/train_model.py

# 4. Score individual employees using the saved model
python src/predict.py
```

## Model Pipeline

1. **Preprocessing** — `StandardScaler` on numeric features, `OneHotEncoder`
   on categorical features, combined via `ColumnTransformer`.
2. **Models trained** — Logistic Regression and Random Forest, both with
   `class_weight="balanced"` (attrition datasets are often imbalanced).
3. **Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC,
   confusion matrix, and ROC curve comparison.
4. **Model selection** — Best model chosen by ROC-AUC and saved to
   `models/attrition_model.pkl` as a single scikit-learn `Pipeline`
   (preprocessing + classifier together, so it loads and predicts
   directly on raw employee data).

## Sample Results (synthetic data)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.741 | 0.738 | 0.748 | 0.743 | **0.820** |
| Random Forest | 0.740 | 0.731 | 0.760 | 0.745 | 0.804 |

## Using the Model

```python
from src.predict import load_model, score_employee

model = load_model()
result = score_employee(model, {
    "age": 30, "monthly_income": 5500, "years_at_company": 3,
    "years_since_last_promotion": 2, "years_with_current_manager": 2,
    "distance_from_home": 12, "job_level": 2, "num_companies_worked": 2,
    "training_times_last_year": 2, "job_satisfaction": 3,
    "work_life_balance": 3, "environment_satisfaction": 3,
    "performance_rating": 3, "percent_salary_hike": 14,
    "overtime": "No", "department": "R&D", "job_role": "Analyst",
    "marital_status": "Married", "business_travel": "Travel_Rarely",
})
print(result)
# {'predicted_class': 'Likely to Stay', 'attrition_probability': 0.21,
#  'risk_band': 'Low Risk'}
```

## Next Steps / Extensions

- Replace synthetic data with a real dataset (e.g. IBM HR Analytics
  Attrition dataset, or your own company's HR export).
- Add hyperparameter tuning (`GridSearchCV` / `Optuna`).
- Add SHAP explainability to show *why* an employee is flagged at risk
  (useful for HR intervention planning).
- Wrap `predict.py` in a REST API (FastAPI/Flask) for HR dashboards.
- Add a retention-cost simulation: estimate savings from early
  intervention on high-risk employees.
- Deploy with MLOps tooling (MLflow tracking, Docker + CI/CD) — a
  natural extension into an MLOps-focused capstone.
