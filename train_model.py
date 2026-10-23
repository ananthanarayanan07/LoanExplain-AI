"""Command-line training script for LoanExplain AI.

Usage (from the project folder, with the virtual environment active):
    python train_model.py
    python train_model.py --data data/my_file.csv --target Loan_Status
"""
import argparse
import json
import sys
from pathlib import Path

from src import data_processing as dp
from src import model_training as mt

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "loan_model.joblib"
RESULTS_PATH = BASE_DIR / "reports" / "results.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the LoanExplain AI Random Forest model.")
    parser.add_argument("--data", type=Path, default=None, help="Path to a CSV file (default: data/ folder).")
    parser.add_argument("--target", default=None, help="Target column name (auto-detected if omitted).")
    parser.add_argument("--positive-label", default=None, help="Value that means 'approved' (auto if omitted).")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=dp.RANDOM_SEED)
    parser.add_argument("--trees", type=int, default=200)
    args = parser.parse_args()

    data_path = args.data or dp.find_default_dataset(BASE_DIR / "data")
    try:
        if data_path is None:
            print("No dataset found in data/. Trying to download the public Loan Prediction dataset...")
            data_path = dp.download_dataset(BASE_DIR / "data" / dp.DEFAULT_DATASET_NAME)
        df = dp.load_csv(data_path)
        target = args.target or dp.detect_target_column(df)
        if target is None:
            print(f"ERROR: could not detect the target column. Use --target. Columns: {list(df.columns)}")
            return 1
        print(f"Dataset: {data_path} ({df.shape[0]} rows, {df.shape[1]} columns) | target: {target}")
        bundle = mt.train_and_evaluate(df, target, args.positive_label, args.test_size, args.seed, args.trees)
    except (dp.DatasetDownloadError, dp.DataValidationError, dp.TargetEncodingError, mt.TrainingError) as exc:
        print(f"ERROR: {exc}")
        return 1

    mt.save_model_bundle(bundle, MODEL_PATH)
    print(f"\nTest-set results ({bundle['n_test']} held-out rows):")
    for name, value in bundle["metrics"].items():
        print(f"  {name:28s} {value:.4f}")
    print("\nConfusion matrix [rows = actual, cols = predicted] (Not approved, Approved):")
    print(bundle["confusion_matrix"])
    print("\n" + bundle["report_text"])
    for item in bundle["limitations"]:
        print("Limitation:", item)

    results = {k: bundle[k] for k in ("metrics", "n_rows", "n_features", "n_train", "n_test", "random_state",
                                      "test_size", "params", "class_counts_all", "cv_f1_scores", "trained_at",
                                      "sklearn_version")}
    results["confusion_matrix"] = bundle["confusion_matrix"].tolist()
    results["classification_report"] = bundle["report_dict"]
    results["top_features"] = bundle["feature_importances"].head(10).round(4).to_dict("records")
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nModel saved to {MODEL_PATH}\nResults saved to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
