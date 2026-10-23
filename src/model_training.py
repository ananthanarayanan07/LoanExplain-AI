"""Random Forest training, evaluation, saving and loading.

The whole workflow (imputation -> encoding -> Random Forest) lives in ONE
scikit-learn Pipeline. The pipeline is fitted on the training split only, so
no information from the test split leaks into preprocessing.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import data_processing as dp

REQUIRED_BUNDLE_KEYS = [
    "pipeline", "feature_columns", "numeric_columns", "binary_columns", "categorical_columns",
    "categorical_choices", "numeric_ranges", "feature_defaults", "class_names", "target_column",
    "metrics", "X_test", "y_test",
]


class ModelNotFoundError(FileNotFoundError):
    """Raised when the saved model file does not exist."""


class ModelLoadError(RuntimeError):
    """Raised when the saved model file is corrupt or incompatible."""


class TrainingError(RuntimeError):
    """Raised when training fails for a reason the user can act on."""


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def build_pipeline(numeric_cols, binary_cols, categorical_cols, n_estimators=200,
                   max_depth=None, min_samples_leaf=1, random_state=dp.RANDOM_SEED) -> Pipeline:
    """Preprocessing (ColumnTransformer) + RandomForestClassifier in one Pipeline."""
    transformers = []
    if numeric_cols:   # numbers: fill gaps with the median (trees do not need scaling)
        transformers.append(("num", SimpleImputer(strategy="median"), list(numeric_cols)))
    if binary_cols:    # 0/1 flags such as Credit_History: fill gaps with the most common value
        transformers.append(("bin", SimpleImputer(strategy="most_frequent"), list(binary_cols)))
    if categorical_cols:  # text categories: fill gaps, then one-hot encode (unknown values -> all zeros)
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), list(categorical_cols)))
    if not transformers:
        raise TrainingError("No usable feature columns were found.")

    preprocessor = ColumnTransformer(transformers, verbose_feature_names_out=False)
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        class_weight="balanced",   # handles class imbalance
        random_state=random_state, n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


# --------------------------------------------------------------------------- #
# Training + evaluation
# --------------------------------------------------------------------------- #
def train_and_evaluate(df: pd.DataFrame, target_col: str, positive_label: str | None = None,
                       test_size: float = 0.2, random_state: int = dp.RANDOM_SEED,
                       n_estimators: int = 200, max_depth: int | None = None,
                       min_samples_leaf: int = 1) -> dict:
    """Validate, clean, split, train and evaluate. Returns a 'bundle' dictionary."""
    report = dp.validate_dataset(df, target_col)
    if not report.is_valid:
        raise dp.DataValidationError(" ".join(report.errors))

    clean_df, notes = dp.clean_dataframe(df, target_col)
    y, target_info = dp.prepare_target(clean_df[target_col], positive_label)
    X_raw = clean_df.drop(columns=[target_col])
    types = dp.infer_column_types(X_raw)
    X = dp.prepare_features(X_raw, types["numeric"], types["binary"], types["categorical"])

    # Split BEFORE fitting any preprocessing -> no data leakage. Stratify keeps class ratios.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError as exc:
        raise TrainingError(f"Could not split the data: {exc}") from exc

    pipeline = build_pipeline(types["numeric"], types["binary"], types["categorical"],
                              n_estimators, max_depth, min_samples_leaf, random_state)
    try:
        pipeline.fit(X_train, y_train)
    except Exception as exc:
        raise TrainingError(f"Model training failed: {exc}") from exc

    # ---- Evaluate ONLY on the held-out test set ----
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    names = ["Not approved", "Approved"]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
    }
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report_text = classification_report(y_test, y_pred, labels=[0, 1], target_names=names, zero_division=0)
    report_dict = classification_report(y_test, y_pred, labels=[0, 1], target_names=names,
                                        zero_division=0, output_dict=True)

    # Cross-validation on the TRAINING split only (extra stability check)
    cv_scores = None
    try:
        n_splits = int(min(5, y_train.value_counts().min()))
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
    except Exception:
        cv_scores = None

    feature_names_out = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    importances = pd.DataFrame({
        "feature": feature_names_out,
        "importance": pipeline.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # ---- Limitations that depend on the actual data ----
    limitations = []
    test_counts = y_test.value_counts().to_dict()
    for cls, label in ((0, names[0]), (1, names[1])):
        if test_counts.get(cls, 0) < 10:
            limitations.append(f"The test set holds only {test_counts.get(cls, 0)} '{label}' example(s); "
                               "metrics for that class are unreliable.")
    if len(X) < 1000:
        limitations.append(f"The dataset is small ({len(X)} rows); results can change a lot with another split.")
    if y.mean() > 0.6 or y.mean() < 0.4:
        limitations.append(f"Classes are imbalanced (approved share = {y.mean():.1%}); "
                           "accuracy alone can look good even for a weak model.")
    limitations.append("Metrics describe this one dataset and one random split; they do not guarantee "
                       "performance on other applicants or other banks.")

    numeric_ranges = {}
    for col in types["numeric"]:
        s = X_train[col].dropna()
        numeric_ranges[col] = {"min": float(s.min()), "max": float(s.max()), "median": float(s.median()),
                               "p05": float(s.quantile(0.05)), "p95": float(s.quantile(0.95))}
    categorical_choices = {c: sorted(X_train[c].dropna().unique().tolist()) for c in types["categorical"]}
    for col in types["binary"]:
        categorical_choices[col] = sorted(float(v) for v in X_train[col].dropna().unique())
    feature_defaults = {c: numeric_ranges[c]["median"] for c in types["numeric"]}
    for col in types["categorical"] + types["binary"]:
        mode = X_train[col].mode(dropna=True)
        feature_defaults[col] = mode.iloc[0] if len(mode) else categorical_choices[col][0]

    return {
        "pipeline": pipeline,
        "feature_columns": list(X.columns),
        "numeric_columns": types["numeric"],
        "binary_columns": types["binary"],
        "categorical_columns": types["categorical"],
        "categorical_choices": categorical_choices,
        "numeric_ranges": numeric_ranges,
        "feature_defaults": feature_defaults,
        "class_names": names,
        "target_column": target_col,
        "target_info": target_info,
        "metrics": metrics,
        "confusion_matrix": cm,
        "report_text": report_text,
        "report_dict": report_dict,
        "cv_f1_scores": None if cv_scores is None else [float(s) for s in cv_scores],
        "feature_names_out": feature_names_out,
        "feature_importances": importances,
        "X_train_sample": X_train.sample(min(200, len(X_train)), random_state=random_state),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": pd.Series(y_pred, index=y_test.index),
        "y_proba": pd.Series(y_proba, index=y_test.index),
        "n_rows": int(len(X)), "n_features": int(X.shape[1]),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "class_counts_all": {names[i]: int((y == i).sum()) for i in (0, 1)},
        "random_state": int(random_state), "test_size": float(test_size),
        "params": {"n_estimators": n_estimators, "max_depth": max_depth,
                   "min_samples_leaf": min_samples_leaf, "class_weight": "balanced"},
        "limitations": limitations,
        "cleaning_notes": notes,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------- #
# Saving and loading
# --------------------------------------------------------------------------- #
def save_model_bundle(bundle: dict, path: str | Path) -> Path:
    """Save the complete pipeline + metadata with Joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_model_bundle(path: str | Path) -> dict:
    """Load a saved bundle and check that it has everything the app needs."""
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(f"No trained model found at {path}. Train the model first.")
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(f"The model file could not be loaded ({exc}). Retrain the model.") from exc
    if not isinstance(bundle, dict):
        raise ModelLoadError("The model file has an unexpected format. Retrain the model.")
    missing = [k for k in REQUIRED_BUNDLE_KEYS if k not in bundle]
    if missing:
        raise ModelLoadError(f"The model file is missing information ({', '.join(missing)}). Retrain the model.")
    return bundle
