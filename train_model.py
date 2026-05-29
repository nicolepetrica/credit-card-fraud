"""
Fraud detection model training and evaluation.

Outputs structured METRIC lines for autoresearch.
Usage: python train_model.py
"""
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")


def load_data():
    """Load and prepare the dataset."""
    df = pd.read_csv("data/input/creditcard_clean.csv")
    y = df["Class"]
    X = df.drop("Class", axis=1)
    return X, y


def engineer_features(X):
    """Add interaction and derived features."""
    X = X.copy()
    v_cols = [c for c in X.columns if c.startswith("V")]

    # Amount interactions with V features
    for v in v_cols:
        X[f"Amount_x_{v}"] = X["Amount"] * X[v]

    # Amount transformations
    X["Amount_log"] = np.log1p(X["Amount"])
    X["Amount_sq"] = X["Amount"] ** 2

    # Time-based cyclical features
    X["Time_hour"] = (X["Time"] / 3600) % 24
    X["Time_sin"] = np.sin(2 * np.pi * X["Time_hour"] / 24)
    X["Time_cos"] = np.cos(2 * np.pi * X["Time_hour"] / 24)
    X = X.drop("Time_hour", axis=1)

    # Amount quantile bucket
    X["Amount_q10"] = pd.qcut(X["Amount"], q=10, labels=False, duplicates="drop")

    return X


def train_and_evaluate(X, y):
    """Train XGBoost model and evaluate. Returns metrics dict."""
    # Train/test split (stratified to preserve fraud ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature engineering
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Further split train into train+val for threshold tuning (stratified)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Compute class imbalance ratio for scale_pos_weight
    ratio = (y_tr == 0).sum() / (y_tr == 1).sum()

    # Train XGBoost
    t0 = time.perf_counter()
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0

    # Find optimal threshold on validation set (maximize F1)
    from sklearn.metrics import f1_score
    val_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_f1 = -1
    for thresh in thresholds:
        y_val_pred = (val_proba >= thresh).astype(int)
        f1 = f1_score(y_val, y_val_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    # Predict on test set using optimal threshold
    t0 = time.perf_counter()
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= best_thresh).astype(int)
    infer_time = time.perf_counter() - t0

    # Metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "f1_fraud": report["1"]["f1-score"],
        "precision_fraud": report["1"]["precision"],
        "recall_fraud": report["1"]["recall"],
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "accuracy": report["accuracy"],
        "train_time_s": train_time,
        "infer_time_s": infer_time,
    }

    # Confusion matrix details
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics["true_positives"] = int(tp)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_negatives"] = int(tn)
    metrics["optimal_threshold"] = best_thresh

    return metrics


def main():
    X, y = load_data()

    total_start = time.perf_counter()
    metrics = train_and_evaluate(X, y)
    total_time = time.perf_counter() - total_start

    # Output structured metrics for autoresearch
    print(f"total_time_s={total_time:.3f}")

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"METRIC {name}={value:.6f}")
        else:
            print(f"METRIC {name}={value}")

    # Primary metric is F1 for fraud class
    print(f"\nF1-Score (fraud): {metrics['f1_fraud']:.4f}")
    print(f"Precision (fraud): {metrics['precision_fraud']:.4f}")
    print(f"Recall (fraud): {metrics['recall_fraud']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")


if __name__ == "__main__":
    main()
