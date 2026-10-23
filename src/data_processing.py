"""Data loading, validation and cleaning helpers for LoanExplain AI.

Everything in this file works on plain pandas DataFrames, so it is easy to test
and does not depend on Streamlit.
"""
from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# Public copy of the well-known "Loan Prediction" dataset (Analytics Vidhya).
DATASET_URL = "https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv"
DEFAULT_DATASET_NAME = "loan_prediction.csv"
UPLOADED_DATASET_NAME = "uploaded_dataset.csv"

RANDOM_SEED = 42          # reproducible split / model
MIN_ROWS = 50             # refuse to train on fewer rows than this
MIN_CLASS_COUNT = 10      # each class needs at least this many rows

TARGET_CANDIDATES = [
    "Loan_Status", "LoanStatus", "Loan_Approved", "Loan_Approval", "Approved",
    "Approval_Status", "Status", "Target", "Label",
]
POSITIVE_TOKENS = {"y", "yes", "1", "true", "approved", "approve", "accepted", "t", "good"}
NEGATIVE_TOKENS = {"n", "no", "0", "false", "rejected", "denied", "declined", "f", "bad"}
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class DataValidationError(ValueError):
    """Raised when a dataset is missing, unreadable or unusable."""


class DatasetDownloadError(RuntimeError):
    """Raised when the public dataset cannot be downloaded automatically."""


class TargetEncodingError(ValueError):
    """Raised when the target column cannot be turned into two classes automatically."""

    def __init__(self, message: str, values: list[str] | None = None):
        super().__init__(message)
        self.values = values or []


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def find_default_dataset(data_dir: str | Path) -> Path | None:
    """Return the first dataset found in the data folder (or None)."""
    data_dir = Path(data_dir)
    for name in (DEFAULT_DATASET_NAME, UPLOADED_DATASET_NAME):
        if (data_dir / name).exists():
            return data_dir / name
    csv_files = sorted(data_dir.glob("*.csv")) if data_dir.exists() else []
    return csv_files[0] if csv_files else None


def load_csv(source) -> pd.DataFrame:
    """Read a CSV from a path or an uploaded file object, with friendly errors."""
    try:
        if hasattr(source, "read"):
            raw = source.read()
        else:
            path = Path(source)
            if not path.exists():
                raise DataValidationError(f"Dataset file not found: {path}")
            raw = path.read_bytes()
        if isinstance(raw, bytes):
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
        else:
            text = raw
        if not text.strip():
            raise DataValidationError("The CSV file is empty.")
        df = pd.read_csv(io.StringIO(text))
    except DataValidationError:
        raise
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("The CSV file contains no data.") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"The file is not a valid CSV: {exc}") from exc
    except Exception as exc:  # any other read problem
        raise DataValidationError(f"Could not read the CSV file: {exc}") from exc

    df.columns = [str(c).strip() for c in df.columns]
    if df.empty or df.shape[1] == 0:
        raise DataValidationError("The dataset has no rows or no columns.")
    return df


def download_dataset(dest: str | Path, url: str = DATASET_URL, timeout: int = 30) -> Path:
    """Download the public dataset, check that it parses, then save it."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            content = response.read()
    except Exception as exc:
        raise DatasetDownloadError(
            f"Automatic download failed ({exc}). Download the CSV manually "
            f"(see data/README.md) or upload it in the Dataset Explorer page."
        ) from exc
    load_csv(io.BytesIO(content))  # raises DataValidationError if it is not a usable CSV
    dest.write_bytes(content)
    return dest


# --------------------------------------------------------------------------- #
# Target detection and validation
# --------------------------------------------------------------------------- #
def detect_target_column(df: pd.DataFrame) -> str | None:
    """Guess the target column from common names (case-insensitive)."""
    lowered = {c.lower(): c for c in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    status_cols = [c for c in df.columns if "status" in c.lower()]
    return status_cols[0] if len(status_cols) == 1 else None


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_dataset(df: pd.DataFrame | None, target_col: str | None) -> ValidationReport:
    """Run automated quality checks BEFORE training."""
    report = ValidationReport()
    if df is None or df.empty:
        report.errors.append("The dataset is empty.")
        return report
    report.info["rows"], report.info["columns"] = int(df.shape[0]), int(df.shape[1])

    if not target_col or target_col not in df.columns:
        report.errors.append(
            "Target column not found. Select the column that holds the loan outcome "
            f"(for example 'Loan_Status'). Available columns: {', '.join(map(str, df.columns))}."
        )
        return report

    target = df[target_col]
    n_missing_target = int(target.isna().sum())
    if n_missing_target:
        report.warnings.append(f"{n_missing_target} row(s) have no target value and will be dropped.")
    target = target.dropna().astype(str).str.strip()
    counts = target.value_counts()
    report.info["class_counts"] = counts.to_dict()

    if len(df) < MIN_ROWS:
        report.errors.append(f"Only {len(df)} rows - at least {MIN_ROWS} are needed to train and test a model.")
    if len(counts) < 2:
        report.errors.append("The target column has fewer than two classes, so a classifier cannot be trained.")
    elif len(counts) > 2:
        report.errors.append(
            f"The target column has {len(counts)} distinct values; this app supports binary targets "
            "(approved / not approved) only."
        )
    elif counts.min() < MIN_CLASS_COUNT:
        report.errors.append(
            f"The smallest class has only {int(counts.min())} rows (minimum {MIN_CLASS_COUNT}); "
            "the train/test split would not contain enough examples of it."
        )
    elif counts.max() / counts.min() > 1.5:
        report.warnings.append(
            f"Class imbalance: {counts.idxmax()} = {int(counts.max())} rows vs {counts.idxmin()} = "
            f"{int(counts.min())} rows. The model uses balanced class weights, but accuracy alone can mislead."
        )

    if df.shape[1] < 2:
        report.errors.append("The dataset needs at least one feature column besides the target.")
    n_dup = int(df.duplicated().sum())
    report.info["duplicates"] = n_dup
    if n_dup:
        report.warnings.append(f"{n_dup} duplicate row(s) found. Duplicates can leak between train and test sets.")
    high_missing = [c for c in df.columns if c != target_col and df[c].isna().mean() > 0.5]
    if high_missing:
        report.warnings.append(f"More than 50% missing values in: {', '.join(high_missing)}.")
    constant = [c for c in df.columns if c != target_col and df[c].nunique(dropna=True) <= 1]
    if constant:
        report.warnings.append(f"Constant (uninformative) column(s): {', '.join(constant)}.")
    return report


# --------------------------------------------------------------------------- #
# Cleaning and feature preparation
# --------------------------------------------------------------------------- #
def identify_id_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """Columns that look like row identifiers (e.g. Loan_ID) carry no signal and are dropped."""
    ids = []
    for col in df.columns:
        name = col.lower()
        if col != target_col and (name == "id" or name.endswith("_id")):
            if df[col].nunique(dropna=True) == df[col].notna().sum():
                ids.append(col)
    return ids


def clean_dataframe(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, list[str]]:
    """Strip text, turn blanks into NaN, drop rows without target / ID / empty columns."""
    notes: list[str] = []
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        series = out[col].astype("object").map(lambda v: v.strip() if isinstance(v, str) else v)
        is_missing = series.map(lambda v: isinstance(v, str) and v.lower() in MISSING_TOKENS)
        series = series.where(~is_missing, np.nan)
        numeric_version = pd.to_numeric(series, errors="coerce")
        if series.notna().any() and numeric_version.notna().sum() == series.notna().sum():
            series = numeric_version  # a text column that only holds numbers
        out[col] = series

    n_before = len(out)
    out = out.dropna(subset=[target_col])
    if len(out) < n_before:
        notes.append(f"Dropped {n_before - len(out)} row(s) with a missing target value.")
    id_cols = identify_id_columns(out, target_col)
    if id_cols:
        out = out.drop(columns=id_cols)
        notes.append(f"Dropped identifier column(s) with no predictive meaning: {', '.join(id_cols)}.")
    empty_cols = [c for c in out.columns if out[c].isna().all()]
    if empty_cols:
        out = out.drop(columns=empty_cols)
        notes.append(f"Dropped completely empty column(s): {', '.join(empty_cols)}.")
    return out.reset_index(drop=True), notes


def _clean_value(value):
    """Convert one categorical value to a tidy string (NaN stays NaN)."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def prepare_target(series: pd.Series, positive_label: str | None = None) -> tuple[pd.Series, dict]:
    """Turn a two-class target into 0/1, where 1 = approved (positive class)."""
    cleaned = series.map(_clean_value)
    uniques = sorted(cleaned.dropna().unique(), key=str)
    if len(uniques) != 2:
        raise TargetEncodingError(
            f"The target must have exactly two classes, but found {len(uniques)}: {uniques[:10]}.", uniques
        )
    if positive_label is not None:
        positive = str(positive_label)
        if positive not in uniques:
            raise TargetEncodingError(f"Positive label '{positive}' is not one of {uniques}.", uniques)
    else:
        pos = [u for u in uniques if u.lower() in POSITIVE_TOKENS]
        neg = [u for u in uniques if u.lower() in NEGATIVE_TOKENS]
        if len(pos) == 1 and len(neg) == 1:
            positive = pos[0]
        else:
            raise TargetEncodingError(
                f"Could not tell which of {uniques} means 'approved'. Choose the positive label manually.",
                uniques,
            )
    negative = next(u for u in uniques if u != positive)
    y = (cleaned == positive).astype(int)
    info = {"positive_label": positive, "negative_label": negative,
            "class_names": ["Not approved", "Approved"]}
    return y, info


def infer_column_types(X: pd.DataFrame) -> dict[str, list[str]]:
    """Split feature columns into numeric / binary-numeric / categorical groups."""
    numeric, binary, categorical = [], [], []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
            (binary if X[col].nunique(dropna=True) <= 2 else numeric).append(col)
        else:
            categorical.append(col)
    return {"numeric": numeric, "binary": binary, "categorical": categorical}


def prepare_features(X: pd.DataFrame, numeric_cols, binary_cols, categorical_cols) -> pd.DataFrame:
    """Give every column a consistent type. Used for training AND prediction."""
    out = X.copy()
    for col in list(numeric_cols) + list(binary_cols):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    for col in categorical_cols:
        out[col] = out[col].map(_clean_value).astype(object)
    return out
