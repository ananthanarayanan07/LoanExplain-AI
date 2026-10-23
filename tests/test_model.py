"""Basic automated tests. Run with:  python -m pytest -v

The data below is a small SYNTHETIC fixture used only to test the code. It is not real loan data
and no results from it are reported anywhere in the project.
"""
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src import data_processing as dp
from src import explainability as ex
from src import model_training as mt
from src import prediction as pr


def make_fixture_df(n=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Loan_ID": [f"LP{i:04d}" for i in range(n)],
        "Gender": rng.choice(["Male", "Female"], n),
        "Married": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["0", "1", "2", "3+"], n),
        "Education": rng.choice(["Graduate", "Not Graduate"], n),
        "ApplicantIncome": rng.integers(1500, 12000, n),
        "CoapplicantIncome": rng.integers(0, 5000, n).astype(float),
        "LoanAmount": rng.integers(50, 400, n).astype(float),
        "Loan_Amount_Term": rng.choice([180.0, 360.0], n),
        "Credit_History": rng.choice([0.0, 1.0], n, p=[0.2, 0.8]),
        "Property_Area": rng.choice(["Urban", "Rural", "Semiurban"], n),
    })
    score = df["Credit_History"] * 2 + df["ApplicantIncome"] / 12000 - df["LoanAmount"] / 400 + rng.normal(0, .4, n)
    df["Loan_Status"] = np.where(score > 1.2, "Y", "N")
    df.loc[rng.choice(n, 15, replace=False), "LoanAmount"] = np.nan      # missing numbers
    df.loc[rng.choice(n, 10, replace=False), "Gender"] = np.nan          # missing categories
    df.loc[rng.choice(n, 10, replace=False), "Credit_History"] = np.nan
    return df


@pytest.fixture(scope="module")
def df():
    return make_fixture_df()


@pytest.fixture(scope="module")
def bundle(df):
    return mt.train_and_evaluate(df, "Loan_Status", n_estimators=50)


# ---------------------------- dataset validation ---------------------------- #
def test_target_detection(df):
    assert dp.detect_target_column(df) == "Loan_Status"
    assert dp.detect_target_column(df.drop(columns=["Loan_Status"])) is None


def test_valid_dataset_passes(df):
    assert dp.validate_dataset(df, "Loan_Status").is_valid


def test_missing_target_column_is_reported(df):
    report = dp.validate_dataset(df, "Not_A_Column")
    assert not report.is_valid and "Target column not found" in report.errors[0]


def test_empty_and_tiny_datasets_rejected(df):
    assert not dp.validate_dataset(pd.DataFrame(), "Loan_Status").is_valid
    assert not dp.validate_dataset(df.head(20), "Loan_Status").is_valid


def test_single_class_target_rejected(df):
    one_class = df.assign(Loan_Status="Y")
    assert not dp.validate_dataset(one_class, "Loan_Status").is_valid


def test_invalid_csv_handled():
    with pytest.raises(dp.DataValidationError):
        dp.load_csv(io.BytesIO(b""))
    with pytest.raises(dp.DataValidationError):
        dp.load_csv("does_not_exist.csv")


def test_target_encoding():
    y, info = dp.prepare_target(pd.Series(["Y", "N", "Y"]))
    assert list(y) == [1, 0, 1] and info["positive_label"] == "Y"
    with pytest.raises(dp.TargetEncodingError):
        dp.prepare_target(pd.Series(["a", "b", "a"]))
    y, _ = dp.prepare_target(pd.Series(["a", "b", "a"]), positive_label="a")
    assert list(y) == [1, 0, 1]


# ------------------------------ model training ------------------------------ #
def test_training_produces_metrics_on_test_set(bundle, df):
    assert set(["accuracy", "precision", "recall", "f1"]) <= set(bundle["metrics"])
    assert bundle["n_train"] + bundle["n_test"] == len(df)
    assert bundle["confusion_matrix"].sum() == bundle["n_test"]      # evaluated on the test rows only
    assert "Loan_ID" not in bundle["feature_columns"]                  # identifier dropped
    assert "Loan_Status" not in bundle["feature_columns"]              # target never an input


def test_training_is_reproducible(df):
    a = mt.train_and_evaluate(df, "Loan_Status", n_estimators=30)
    b = mt.train_and_evaluate(df, "Loan_Status", n_estimators=30)
    assert a["metrics"] == b["metrics"]


def test_training_rejects_invalid_data(df):
    with pytest.raises(dp.DataValidationError):
        mt.train_and_evaluate(df, "Wrong_Target")


# ------------------------------ prediction output --------------------------- #
def test_prediction_output(bundle):
    result = pr.predict_one(bundle, dict(bundle["feature_defaults"]))
    assert result["label"] in bundle["class_names"]
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6


def test_unknown_category_is_handled_safely(bundle):
    inputs = {**bundle["feature_defaults"], "Property_Area": "Moon Base"}
    result = pr.predict_one(bundle, inputs)
    assert result["warnings"] and result["label"] in bundle["class_names"]


def test_missing_numeric_in_pipeline_is_imputed(bundle):
    row = bundle["X_test"].head(1).copy()
    row["LoanAmount"] = np.nan
    assert bundle["pipeline"].predict(row).shape == (1,)


# ---------------------------- invalid input handling ------------------------ #
@pytest.mark.parametrize("field,value", [
    ("ApplicantIncome", -100), ("LoanAmount", -1), ("ApplicantIncome", "abc"),
    ("ApplicantIncome", float("nan")), ("Credit_History", 5), ("Education", None),
])
def test_invalid_inputs_rejected(bundle, field, value):
    inputs = {**bundle["feature_defaults"], field: value}
    with pytest.raises(pr.InputValidationError):
        pr.predict_one(bundle, inputs)


def test_target_column_cannot_be_input(bundle):
    inputs = {**bundle["feature_defaults"], "Loan_Status": "Y"}
    with pytest.raises(pr.InputValidationError):
        pr.predict_one(bundle, inputs)


def test_missing_field_rejected(bundle):
    inputs = dict(bundle["feature_defaults"])
    inputs.pop("Married")
    with pytest.raises(pr.InputValidationError):
        pr.predict_one(bundle, inputs)


# --------------------------- saving and loading ----------------------------- #
def test_save_and_load_roundtrip(bundle, tmp_path):
    path = mt.save_model_bundle(bundle, tmp_path / "m.joblib")
    loaded = mt.load_model_bundle(path)
    inputs = dict(bundle["feature_defaults"])
    assert pr.predict_one(bundle, inputs)["probabilities"] == pr.predict_one(loaded, inputs)["probabilities"]


def test_missing_and_corrupt_model_files(tmp_path):
    with pytest.raises(mt.ModelNotFoundError):
        mt.load_model_bundle(tmp_path / "nope.joblib")
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a model")
    with pytest.raises(mt.ModelLoadError):
        mt.load_model_bundle(bad)


# ------------------------------------ SHAP ---------------------------------- #
def test_shap_values_are_consistent(bundle):
    row = pr.make_input_frame(bundle, dict(bundle["feature_defaults"]))
    result = ex.compute_shap(bundle, row, class_index=1)
    assert result.values.shape == (1, len(result.feature_names))
    prob = bundle["pipeline"].predict_proba(row)[0, 1]
    assert abs(result.base_value + result.values.sum() - prob) < 1e-2     # SHAP additivity
    contribs = ex.local_contributions(bundle, result, 0, row.iloc[0])
    assert set(contribs["feature"]) == set(bundle["feature_columns"])


def test_shap_output_shape_handling():
    two_class = np.zeros((4, 3, 2))
    assert ex.extract_class_values(two_class, 1).shape == (4, 3)
    assert ex.extract_class_values([np.ones((4, 3)), np.zeros((4, 3))], 0).sum() == 12
    with pytest.raises(ex.ExplanationError):
        ex.extract_class_values(np.zeros(5), 1)
