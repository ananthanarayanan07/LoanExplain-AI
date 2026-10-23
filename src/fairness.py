"""Simple group-wise fairness diagnostics on the held-out test set.

These are DESCRIPTIVE checks, not proof that a model is fair or unfair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SENSITIVE_HINTS = ["gender", "sex", "married", "marital", "age", "race", "religion",
                   "nationality", "dependents", "education", "property_area", "area"]
MIN_GROUP_SIZE = 10

DEFINITIONS = {
    "Actual approval rate": "Share of applicants in the group whose label in the dataset is 'approved'.",
    "Predicted approval rate (selection rate)": "Share of the group that the model predicts as 'approved'. "
        "Comparing this across groups is called demographic parity.",
    "True positive rate": "Among applicants who were really approved in the dataset, the share the model also "
        "predicts as approved (equal opportunity compares this across groups).",
}


def candidate_columns(feature_columns: list[str]) -> list[str]:
    """Input columns that are sensitive attributes or may act as proxies for them."""
    return [c for c in feature_columns if any(h in c.lower() for h in SENSITIVE_HINTS)]


def group_report(X_test: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series, column: str,
                 min_group_size: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    """Per-group counts and rates for one categorical column of the test set."""
    groups = X_test[column].astype(object).where(X_test[column].notna(), "Missing").astype(str)
    rows = []
    for group in sorted(groups.unique()):
        mask = (groups == group).to_numpy()
        yt, yp = y_true.to_numpy()[mask], y_pred.to_numpy()[mask]
        rows.append({
            "group": group,
            "n": int(mask.sum()),
            "actual_approval_rate": float(yt.mean()),
            "predicted_approval_rate": float(yp.mean()),
            "true_positive_rate": float(yp[yt == 1].mean()) if (yt == 1).any() else np.nan,
            "small_group": bool(mask.sum() < min_group_size),
        })
    return pd.DataFrame(rows)


def selection_rate_gap(report: pd.DataFrame, min_group_size: int = MIN_GROUP_SIZE) -> float | None:
    """Largest difference in predicted approval rate between groups with enough rows."""
    usable = report[report["n"] >= min_group_size]
    if len(usable) < 2:
        return None
    return float(usable["predicted_approval_rate"].max() - usable["predicted_approval_rate"].min())
