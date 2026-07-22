# Autoresearch: Fraud Detection Model Optimization

## Objective
Build the best possible fraud detection model for the credit card fraud dataset. The data is highly imbalanced (0.17% fraud, 473 fraud cases out of 283,726 transactions). The goal is to maximize F1-score for the fraud class while keeping precision and recall balanced.

## Metrics
- **Primary**: f1_fraud (unitless, higher is better) — F1-score for the fraud class only
- **Secondary**: precision_fraud, recall_fraud, roc_auc, pr_auc, accuracy, train_time_s, infer_time_s, true_positives, false_positives, false_negatives, true_negatives

## How to Run
`./autoresearch.sh` — runs `uv run python train_model.py` which outputs `METRIC name=value` lines.

## Files in Scope
- `train_model.py` — Main training script. Contains data loading, preprocessing, model training, and evaluation.
- `autoresearch.sh` — Benchmark runner wrapper.

## Off Limits
- `data/` — Do not modify data files.
- `notebooks/` — Reference only, do not modify.

## Constraints
- Use data from `data/input/creditcard_clean.csv`
- Train/test split must use `test_size=0.2, random_state=42, stratify=y` for reproducibility
- All code changes go in `train_model.py`
- No new dependencies beyond what's in `pyproject.toml`

## Data Summary
- 283,726 rows, 30 features (V1-V28 PCA components + Time + Amount), binary target (Class)
- 473 fraud cases (0.17%) — extreme class imbalance
- Features V1-V28 are already PCA-transformed and anonymized
- Time and Amount are the only non-anonymized features

## What's Been Tried
### Baseline
- XGBoost with scale_pos_weight, StandardScaler, default hyperparams (n_estimators=200, max_depth=4, learning_rate=0.1)
- Results to be established as baseline
