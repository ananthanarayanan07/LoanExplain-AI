"""Input validation and prediction using the saved pipeline.

The same trained pipeline is used for every prediction - nothing is re-fitted here.
"""
from __future__ import annotations

import math

import pandas as pd

from src import data_processing as dp


class InputValidationError(ValueError):
    """Raised when the applicant inputs are invalid. `.errors` lists every problem."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def validate_inputs(bundle: dict, inputs: dict) -> tuple[dict, list[str], list[str]]:
    """Check applicant inputs. Returns (clean_inputs, errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    clean: dict = {}
    if not isinstance(inputs, dict):
        return clean, ["Inputs must be a dictionary of feature values."], warnings

    target = bundle["target_column"]
    if target in inputs:
        errors.append(f"'{target}' is the target column and must not be used as an input.")
    extra = [k for k in inputs if k not in bundle["feature_columns"] and k != target]
    if extra:
        warnings.append(f"Ignored unknown input(s): {', '.join(map(str, extra))}.")

    for col in bundle["feature_columns"]:
        value = inputs.get(col)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"'{col}' is required.")
            continue

        if col in bundle["numeric_columns"]:
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors.append(f"'{col}' must be a number (got {value!r}).")
                continue
            if not math.isfinite(number):
                errors.append(f"'{col}' must be a finite number.")
                continue
            rng = bundle["numeric_ranges"][col]
            if rng["min"] >= 0 and number < 0:
                errors.append(f"'{col}' cannot be negative.")
                continue
            if number > rng["max"] * 10 + 1 or (rng["min"] > 0 and number < rng["min"] / 10):
                warnings.append(f"'{col}' = {number:g} is far outside the training range "
                                f"({rng['min']:g} to {rng['max']:g}); the prediction is an extrapolation.")
            clean[col] = number

        elif col in bundle["binary_columns"]:
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors.append(f"'{col}' must be one of {bundle['categorical_choices'][col]} (got {value!r}).")
                continue
            if number not in bundle["categorical_choices"][col]:
                errors.append(f"'{col}' must be one of {bundle['categorical_choices'][col]} (got {value!r}).")
                continue
            clean[col] = number

        else:  # categorical text column
            text = str(value).strip()
            if text not in bundle["categorical_choices"][col]:
                warnings.append(f"'{col}' = '{text}' was not seen during training; the model treats it as "
                                "an unknown category (all one-hot columns set to 0).")
            clean[col] = text
    return clean, errors, warnings


def make_input_frame(bundle: dict, clean_inputs: dict) -> pd.DataFrame:
    """One-row DataFrame with the exact columns and types the pipeline expects."""
    frame = pd.DataFrame([{c: clean_inputs[c] for c in bundle["feature_columns"]}])
    return dp.prepare_features(frame, bundle["numeric_columns"], bundle["binary_columns"],
                               bundle["categorical_columns"])


def predict_one(bundle: dict, inputs: dict) -> dict:
    """Validate the inputs, then predict with the trained pipeline."""
    clean, errors, warnings = validate_inputs(bundle, inputs)
    if errors:
        raise InputValidationError(errors)
    frame = make_input_frame(bundle, clean)
    pipeline = bundle["pipeline"]
    proba = pipeline.predict_proba(frame)[0]
    classes = list(pipeline.named_steps["model"].classes_)
    class_index = int(pipeline.predict(frame)[0])
    names = bundle["class_names"]
    return {
        "class_index": class_index,
        "label": names[class_index],
        "probabilities": {names[int(c)]: float(p) for c, p in zip(classes, proba)},
        "frame": frame,
        "inputs": clean,
        "warnings": warnings,
    }


def compare_predictions(bundle: dict, original: dict, modified: dict) -> dict:
    """Predict twice with the SAME model and describe what changed."""
    before = predict_one(bundle, original)
    after = predict_one(bundle, modified)
    changes = [{"feature": c, "original": before["inputs"][c], "modified": after["inputs"][c]}
               for c in bundle["feature_columns"] if before["inputs"][c] != after["inputs"][c]]
    approved = bundle["class_names"][1]
    return {
        "original": before,
        "modified": after,
        "changes": changes,
        "prediction_changed": before["class_index"] != after["class_index"],
        "approved_probability_change": after["probabilities"][approved] - before["probabilities"][approved],
    }
