"""SHAP explanations for the trained Random Forest pipeline.

SHAP values are computed on the *preprocessed* features (after imputation and
one-hot encoding), using the same fitted preprocessor that was used for
training. For easier reading, local explanations are summed back to the
original input columns (SHAP values are additive, so this is valid).
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier


class ExplanationError(RuntimeError):
    """Raised when SHAP cannot explain the current model configuration."""


@dataclass
class ShapResult:
    values: np.ndarray          # (n_rows, n_features) SHAP values for ONE class
    base_value: float           # average model output for that class
    data: pd.DataFrame          # preprocessed feature values the SHAP values refer to
    feature_names: list[str]
    class_index: int            # which class the values explain (0 / 1)


def _forest(bundle: dict) -> RandomForestClassifier:
    model = bundle["pipeline"].named_steps.get("model")
    if not isinstance(model, RandomForestClassifier):
        raise ExplanationError("SHAP TreeExplainer is only set up for the Random Forest model in this app.")
    return model


def transform_features(bundle: dict, X_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the FITTED preprocessor (never re-fit) and keep readable column names."""
    try:
        pre = bundle["pipeline"].named_steps["preprocess"]
        arr = pre.transform(X_raw[bundle["feature_columns"]])
        arr = arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)
        return pd.DataFrame(arr.astype(float), columns=list(pre.get_feature_names_out()), index=X_raw.index)
    except Exception as exc:
        raise ExplanationError(f"Could not preprocess the data for SHAP: {exc}") from exc


def get_explainer(bundle: dict) -> shap.TreeExplainer:
    """TreeExplainer is the exact, fast SHAP method for tree ensembles."""
    try:
        return shap.TreeExplainer(_forest(bundle))
    except ExplanationError:
        raise
    except Exception as exc:
        raise ExplanationError(f"Could not create the SHAP explainer: {exc}") from exc


def extract_class_values(raw, class_index: int) -> np.ndarray:
    """Handle the different output shapes returned by different SHAP versions.

    * old SHAP: a list with one (n_rows, n_features) array per class
    * new SHAP: one array of shape (n_rows, n_features, n_classes)
    * some binary setups: one 2-D array that already belongs to the positive class
    """
    if isinstance(raw, list):
        if class_index >= len(raw):
            raise ExplanationError(f"SHAP returned {len(raw)} class(es); class {class_index} is unavailable.")
        return np.asarray(raw[class_index], dtype=float)
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 3:
        if class_index >= arr.shape[2]:
            raise ExplanationError(f"SHAP returned {arr.shape[2]} class(es); class {class_index} is unavailable.")
        return arr[:, :, class_index]
    if arr.ndim == 2:
        return arr if class_index == 1 else -arr
    raise ExplanationError(f"Unexpected SHAP output shape {arr.shape}.")


def extract_base_value(expected, class_index: int) -> float:
    """Expected value may be one value per class, or a single value for the positive class."""
    arr = np.atleast_1d(np.asarray(expected, dtype=float))
    if arr.size > class_index:
        return float(arr[class_index])
    return float(arr[0]) if class_index == 1 else 1.0 - float(arr[0])


def compute_shap(bundle: dict, X_raw: pd.DataFrame, class_index: int = 1,
                 explainer: shap.TreeExplainer | None = None) -> ShapResult:
    """SHAP values (for the chosen class) for the given RAW rows."""
    if X_raw is None or len(X_raw) == 0:
        raise ExplanationError("There are no rows to explain.")
    data = transform_features(bundle, X_raw)
    try:
        explainer = explainer or get_explainer(bundle)
        raw = explainer.shap_values(data.values, check_additivity=False)
        values = extract_class_values(raw, class_index)
        base = extract_base_value(explainer.expected_value, class_index)
    except ExplanationError:
        raise
    except Exception as exc:
        raise ExplanationError(f"SHAP could not explain this model: {exc}") from exc
    if values.shape != data.shape:
        raise ExplanationError(f"SHAP output shape {values.shape} does not match the data shape {data.shape}.")
    return ShapResult(values, base, data, list(data.columns), class_index)


# --------------------------------------------------------------------------- #
# Global importance
# --------------------------------------------------------------------------- #
def global_importance(result: ShapResult) -> pd.DataFrame:
    """Mean absolute SHAP value per preprocessed feature."""
    return (pd.DataFrame({"feature": result.feature_names,
                          "mean_abs_shap": np.abs(result.values).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def map_to_original_column(feature_name: str, original_columns: list[str]) -> str:
    """'Property_Area_Urban' -> 'Property_Area'; numeric names map to themselves."""
    matches = [c for c in original_columns if feature_name == c or feature_name.startswith(c + "_")]
    return max(matches, key=len) if matches else feature_name


def global_importance_by_input(result: ShapResult, original_columns: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per ORIGINAL input column (one-hot columns are summed per row first)."""
    mapping = [map_to_original_column(n, original_columns) for n in result.feature_names]
    per_row = pd.DataFrame(result.values, columns=mapping).T.groupby(level=0).sum().T
    return (per_row.abs().mean().rename("mean_abs_shap").reset_index()
            .rename(columns={"index": "feature"}).sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Local explanation
# --------------------------------------------------------------------------- #
def local_contributions(bundle: dict, result: ShapResult, row_position: int, raw_row: pd.Series) -> pd.DataFrame:
    """Contribution of every ORIGINAL input column for one applicant."""
    cols = bundle["feature_columns"]
    mapping = [map_to_original_column(n, cols) for n in result.feature_names]
    summed = pd.Series(result.values[row_position], index=mapping).groupby(level=0).sum()
    table = pd.DataFrame({
        "feature": cols,
        "value": [raw_row[c] for c in cols],
        "shap_value": [float(summed.get(c, 0.0)) for c in cols],
    })
    table["abs_shap"] = table["shap_value"].abs()
    return table.sort_values("abs_shap", ascending=False).reset_index(drop=True)


def plain_language_explanation(contribs: pd.DataFrame, class_names: list[str], class_index: int,
                               model_output: float, base_value: float, top_n: int = 3) -> list[str]:
    """Turn the largest SHAP values into sentences. Only describes what SHAP computed."""
    cls = class_names[class_index]
    lines = [f"The model's average output for **{cls}** is {base_value:.2f}; for this applicant it is "
             f"{model_output:.2f}."]
    for _, row in contribs.head(top_n).iterrows():
        if abs(row["shap_value"]) < 1e-9:
            continue
        direction = "raised" if row["shap_value"] > 0 else "lowered"
        value = f"{row['value']:g}" if isinstance(row["value"], (int, float, np.floating)) else row["value"]
        lines.append(f"**{row['feature']} = {value}** {direction} the model's output for {cls} "
                     f"by {abs(row['shap_value']):.3f}.")
    lines.append("This describes how the *model* used each input. It does not prove that the input "
                 "causes approval in the real world, and correlated inputs can share credit between them.")
    return lines


# --------------------------------------------------------------------------- #
# Plots (Matplotlib, so they work with every SHAP version)
# --------------------------------------------------------------------------- #
def plot_summary(result: ShapResult, max_display: int = 15):
    """SHAP beeswarm summary plot."""
    plt.close("all")
    shap.summary_plot(result.values, result.data, feature_names=result.feature_names,
                      max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 5.5)
    fig.tight_layout()
    return fig


def plot_importance_bar(importance: pd.DataFrame, top_n: int = 15, value_col: str = "mean_abs_shap",
                        title: str = "Mean |SHAP value| (average impact on model output)"):
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.38 * len(top) + 1)))
    ax.barh(top["feature"], top[value_col], color="#2563eb")
    ax.set_xlabel(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_local(contribs: pd.DataFrame, class_name: str, top_n: int = 10):
    top = contribs.head(top_n).iloc[::-1]
    colors = ["#16a34a" if v > 0 else "#dc2626" for v in top["shap_value"]]
    labels = [f"{f} = {v:g}" if isinstance(v, (int, float, np.floating)) else f"{f} = {v}"
              for f, v in zip(top["feature"], top["value"])]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(top) + 1)))
    ax.barh(labels, top["shap_value"], color=colors)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel(f"SHAP value (green raises, red lowers the model output for '{class_name}')")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
