"""LoanExplain AI - Streamlit dashboard.

Run with:  streamlit run app.py

EDUCATIONAL DEMONSTRATION ONLY. This app does not make real lending decisions.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src import data_processing as dp
from src import explainability as ex
from src import fairness as fa
from src import model_training as mt
from src import prediction as pr

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "loan_model.joblib"
DISCLAIMER = ("**Educational use only.** LoanExplain AI is an academic demonstration. It does not make real "
              "banking decisions and nothing here guarantees loan approval.")

st.set_page_config(page_title="LoanExplain AI", page_icon="🏦", layout="wide")
sns.set_theme(style="whitegrid")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.hero {background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%); color: white; padding: 26px 30px;
       border-radius: 14px; margin-bottom: 18px;}
.hero h1 {color: white; margin: 0 0 4px 0; font-size: 2rem;}
.hero p {margin: 0; opacity: .92; font-size: 1.02rem;}
.metric-card {border: 1px solid rgba(128,128,128,.28); border-radius: 12px; padding: 14px 18px;
              background: rgba(128,128,128,.07); height: 100%;}
.metric-card .label {font-size: .74rem; text-transform: uppercase; letter-spacing: .05em; opacity: .7;}
.metric-card .value {font-size: 1.65rem; font-weight: 700; line-height: 1.25;}
.metric-card .sub {font-size: .78rem; opacity: .65;}
.result-card {border-radius: 12px; padding: 18px 22px; font-size: 1.25rem; font-weight: 700; margin: 8px 0;}
.result-yes {background: rgba(22,163,74,.14); border: 1px solid rgba(22,163,74,.5);}
.result-no {background: rgba(220,38,38,.12); border: 1px solid rgba(220,38,38,.45);}
</style>
""", unsafe_allow_html=True)

FIELD_HELP = {
    "Gender": "Applicant gender as recorded in the dataset (a sensitive attribute - see Fairness page).",
    "Married": "Marital status of the applicant.",
    "Dependents": "Number of people financially dependent on the applicant ('3+' means three or more).",
    "Education": "Graduate or Not Graduate.",
    "Self_Employed": "Whether the applicant is self-employed.",
    "ApplicantIncome": "Applicant's monthly income, in the currency units used by the dataset.",
    "CoapplicantIncome": "Co-applicant's monthly income (0 if there is no co-applicant).",
    "LoanAmount": "Requested loan amount. In the public Loan Prediction dataset this is in thousands.",
    "Loan_Amount_Term": "Loan repayment term in months (360 = 30 years).",
    "Credit_History": "1 = credit history meets the dataset's guidelines, 0 = it does not.",
    "Property_Area": "Urban, Semiurban or Rural.",
}
VALUE_LABELS = {"Credit_History": {1.0: "1 - meets guidelines", 0.0: "0 - does not meet guidelines"}}


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def metric_card(label: str, value: str, sub: str = ""):
    st.markdown(f'<div class="metric-card"><div class="label">{label}</div>'
                f'<div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def show_fig(fig):
    """Draw a Matplotlib figure full-width (works on old and new Streamlit versions)."""
    try:
        st.pyplot(fig, width="stretch")
    except TypeError:
        st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def show_df(data, **kwargs):
    """Show a DataFrame full-width (works on old and new Streamlit versions)."""
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kwargs)


def pretty(col: str) -> str:
    return col.replace("_", " ")


@st.cache_data(show_spinner=False)
def _read_dataset(path_str: str, mtime: float) -> pd.DataFrame:
    """Cached CSV read - Streamlit reruns do not re-read the file unless it changed."""
    return dp.load_csv(path_str)


def get_active_dataset():
    """Returns (DataFrame | None, Path | None, error message | None)."""
    path_str = st.session_state.get("dataset_path")
    if not path_str:
        found = dp.find_default_dataset(DATA_DIR)
        path_str = str(found) if found else ""
        st.session_state["dataset_path"] = path_str
    if not path_str or not Path(path_str).exists():
        return None, None, "No dataset found."
    try:
        return _read_dataset(path_str, Path(path_str).stat().st_mtime), Path(path_str), None
    except dp.DataValidationError as exc:
        return None, Path(path_str), str(exc)


@st.cache_resource(show_spinner=False)
def _load_bundle_cached(path_str: str, mtime: float) -> dict:
    """Cached model load - the model is NOT retrained on reruns."""
    return mt.load_model_bundle(path_str)


def get_bundle(silent: bool = False):
    if not MODEL_PATH.exists():
        return None
    try:
        return _load_bundle_cached(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    except (mt.ModelLoadError, mt.ModelNotFoundError) as exc:
        if not silent:
            st.error(f"Could not load the saved model: {exc}")
        return None


def require_model():
    bundle = get_bundle()
    if bundle is None:
        st.warning("No trained model is available yet. Open **Training & Evaluation** and train the model first "
                   "(or run `python train_model.py`).")
    return bundle


def target_selector(df: pd.DataFrame, key: str) -> str | None:
    """Detect the target column; let the user choose another one if needed."""
    detected = dp.detect_target_column(df)
    current = st.session_state.get("target_col")
    default = current if current in df.columns else detected
    options = list(df.columns)
    if default is None:
        st.warning("The target column could not be detected automatically. Please choose it below "
                   "(the column holding the loan outcome, e.g. Loan_Status).")
    index = options.index(default) if default in options else 0
    choice = st.selectbox("Target column (loan outcome)", options, index=index, key=f"target_{key}",
                          help="The column the model learns to predict. It is never used as an input feature.")
    if default is None and choice not in options:
        return None
    st.session_state["target_col"] = choice
    return choice


def positive_label_selector(series: pd.Series, key: str) -> str | None:
    """Return the positive label if the user must choose it manually, else None (auto-detected)."""
    try:
        dp.prepare_target(series.dropna())
        return None
    except dp.TargetEncodingError as exc:
        if len(exc.values) != 2:
            st.error(str(exc))
            return None
        st.warning(str(exc))
        return st.selectbox("Which value means 'approved'?", exc.values, key=f"pos_{key}")


# --------------------------------------------------------------------------- #
# Input form (shared by Prediction and What-If pages)
# --------------------------------------------------------------------------- #
def input_widget(bundle: dict, col: str, key: str, default=None, container=st):
    help_text = FIELD_HELP.get(col, f"Input feature '{col}' from the training data.")
    label = pretty(col)
    default = bundle["feature_defaults"][col] if default is None else default
    if col in bundle["numeric_columns"]:
        rng = bundle["numeric_ranges"][col]
        min_value = 0.0 if rng["min"] >= 0 else None
        return container.number_input(label, min_value=min_value, value=float(default), step=1.0, key=key,
                                      help=f"{help_text} Training range: {rng['min']:g} to {rng['max']:g}.")
    choices = bundle["categorical_choices"][col]
    labels = VALUE_LABELS.get(col, {})
    index = choices.index(default) if default in choices else 0
    return container.selectbox(label, choices, index=index, key=key, help=help_text,
                               format_func=lambda v: labels.get(v, str(int(v)) if isinstance(v, float) else str(v)))


def render_inputs(bundle: dict, prefix: str, defaults: dict | None = None, columns: int = 3) -> dict:
    cols = st.columns(columns)
    values = {}
    for i, col in enumerate(bundle["feature_columns"]):
        d = (defaults or {}).get(col)
        values[col] = input_widget(bundle, col, f"{prefix}_{col}", d, container=cols[i % columns])
    return values


def probability_chart(probs_a: dict, probs_b: dict | None = None, labels=("Prediction", "Modified")):
    names = list(probs_a.keys())
    fig, ax = plt.subplots(figsize=(6, 2.6))
    x = np.arange(len(names))
    if probs_b is None:
        bars = ax.bar(x, [probs_a[n] for n in names], color=["#dc2626", "#16a34a"][: len(names)])
        ax.bar_label(bars, fmt="%.2f")
    else:
        w = 0.38
        b1 = ax.bar(x - w / 2, [probs_a[n] for n in names], w, label=labels[0], color="#64748b")
        b2 = ax.bar(x + w / 2, [probs_b[n] for n in names], w, label=labels[1], color="#2563eb")
        ax.bar_label(b1, fmt="%.2f")
        ax.bar_label(b2, fmt="%.2f")
        ax.legend(frameon=False)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Model-estimated share")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# A. Home
# --------------------------------------------------------------------------- #
def page_home():
    st.markdown('<div class="hero"><h1>🏦 LoanExplain AI</h1>'
                '<p>Transparent loan approval prediction using Explainable Artificial Intelligence</p></div>',
                unsafe_allow_html=True)
    st.info(DISCLAIMER)
    st.markdown("""
**What this project does.** It trains a Random Forest classifier on historical loan applications, predicts
whether an application would be approved, and uses **SHAP** to show *which inputs pushed the model's output
up or down* - so the prediction is not a black box.

**Objectives**
1. Prepare real loan data safely (validation, missing values, no data leakage).
2. Train and honestly evaluate a Random Forest on a held-out test set.
3. Explain predictions globally and for single applicants with SHAP.
4. Let users explore *what-if* changes and understand fairness limits.
""")
    df, path, err = get_active_dataset()
    bundle = get_bundle(silent=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Dataset status", "Loaded ✅" if df is not None else "Missing ⚠️",
                    path.name if path else "Go to Dataset Explorer")
    with c2:
        metric_card("Dataset rows", f"{len(df):,}" if df is not None else "-", "records in the CSV")
    with c3:
        n_feat = bundle["n_features"] if bundle else (df.shape[1] - 1 if df is not None else None)
        metric_card("Number of features", str(n_feat) if n_feat is not None else "-",
                    "input columns used by the model" if bundle else "columns minus target (before cleaning)")
    with c4:
        metric_card("Model status", "Trained ✅" if bundle else "Not trained ⚠️",
                    bundle["trained_at"] if bundle else "Go to Training & Evaluation")
    if err and df is None and path is not None:
        st.error(err)
    if df is None:
        st.warning("No dataset is loaded. Open **Dataset Explorer** to download the public dataset or upload a CSV.")

    st.subheader("Test-set evaluation summary")
    if bundle:
        m = bundle["metrics"]
        cols = st.columns(5)
        for c, (label, key) in zip(cols, [("Accuracy", "accuracy"), ("Precision", "precision"),
                                          ("Recall", "recall"), ("F1-score", "f1"),
                                          ("Majority baseline", "majority_baseline_accuracy")]):
            with c:
                metric_card(label, f"{m[key]:.3f}", "held-out test set" if key != "majority_baseline_accuracy"
                            else "accuracy of always guessing the majority class")
        st.caption(f"Computed on {bundle['n_test']} held-out test rows (seed {bundle['random_state']}). "
                   "Precision, recall and F1 are for the 'Approved' class.")
    else:
        st.warning("The model has not been trained yet, so there are no evaluation results to show.")


# --------------------------------------------------------------------------- #
# B. Dataset Explorer
# --------------------------------------------------------------------------- #
def page_explorer():
    st.header("📂 Dataset Explorer")
    st.caption("Everything on this page is computed from the **actual dataset values** loaded below.")

    with st.expander("Load or replace the dataset", expanded=get_active_dataset()[0] is None):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Option 1 - download the public Loan Prediction dataset**")
            if st.button("⬇️ Download dataset", help=f"Downloads from {dp.DATASET_URL}"):
                try:
                    with st.spinner("Downloading..."):
                        saved = dp.download_dataset(DATA_DIR / dp.DEFAULT_DATASET_NAME)
                    st.session_state["dataset_path"] = str(saved)
                    st.success(f"Saved to {saved.name}.")
                    st.rerun()
                except (dp.DatasetDownloadError, dp.DataValidationError) as exc:
                    st.error(str(exc))
        with c2:
            st.markdown("**Option 2 - upload your own CSV**")
            uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
            if uploaded is not None and st.session_state.get("last_upload_id") != uploaded.file_id:
                try:
                    df_up = dp.load_csv(uploaded)
                    dest = DATA_DIR / dp.UPLOADED_DATASET_NAME
                    DATA_DIR.mkdir(exist_ok=True)
                    df_up.to_csv(dest, index=False)
                    st.session_state["dataset_path"] = str(dest)
                    st.session_state["last_upload_id"] = uploaded.file_id
                    st.session_state.pop("target_col", None)
                    st.success(f"Loaded {len(df_up)} rows from {uploaded.name}.")
                    st.rerun()
                except dp.DataValidationError as exc:
                    st.session_state["last_upload_id"] = uploaded.file_id
                    st.error(f"Invalid CSV: {exc}")

    df, path, err = get_active_dataset()
    if df is None:
        st.error(err or "No dataset loaded.")
        st.info("Download the dataset above or see `data/README.md` for manual instructions.")
        return
    st.success(f"Using **{path.name}** - {df.shape[0]:,} rows x {df.shape[1]} columns.")
    target = target_selector(df, "explorer")
    report = dp.validate_dataset(df, target)

    tabs = st.tabs(["Preview", "Quality checks", "Columns & statistics", "Distributions",
                    "Loan approval", "Correlation"])

    with tabs[0]:
        n = st.slider("Rows to preview", 5, min(100, len(df)), 10)
        show_df(df.head(n))
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])

    with tabs[1]:
        st.subheader("Automated dataset quality checks")
        for msg in report.errors:
            st.error(msg)
        for msg in report.warnings:
            st.warning(msg)
        if report.is_valid and not report.warnings:
            st.success("All automated checks passed.")
        elif report.is_valid:
            st.success("No blocking problems - the dataset can be used for training (see warnings above).")
        missing = df.isna().sum().rename("missing").to_frame()
        missing["missing_%"] = (missing["missing"] / len(df) * 100).round(2)
        c1, c2 = st.columns([1, 1])
        c1.markdown("**Missing values per column**")
        with c1:
            show_df(missing)
        has_missing = missing[missing["missing"] > 0]
        if len(has_missing):
            fig, ax = plt.subplots(figsize=(5, max(2.5, 0.4 * len(has_missing) + 1)))
            ax.barh(has_missing.index[::-1], has_missing["missing_%"][::-1], color="#f59e0b")
            ax.set_xlabel("% missing")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            with c2:
                show_fig(fig)
        else:
            c2.success("No missing values.")
        st.markdown(f"**Duplicate rows:** {int(df.duplicated().sum())}")
        num = df.select_dtypes(include="number")
        negatives = {c: int((num[c] < 0).sum()) for c in num.columns if (num[c] < 0).any()}
        if negatives:
            st.warning(f"Negative values found (unexpected for income / amount columns): {negatives}")
        else:
            st.markdown("**Negative numeric values:** none")

    with tabs[2]:
        info = pd.DataFrame({"dtype": df.dtypes.astype(str), "unique values": df.nunique(),
                             "missing": df.isna().sum()})
        st.markdown("**Column names and data types**")
        show_df(info)
        num = df.select_dtypes(include="number")
        if num.shape[1]:
            st.markdown("**Numerical summary statistics**")
            show_df(num.describe().T)
        else:
            st.info("No numerical columns found.")

    with tabs[3]:
        num_cols = list(df.select_dtypes(include="number").columns)
        cat_cols = [c for c in df.columns if c not in num_cols]
        c1, c2 = st.columns(2)
        with c1:
            if cat_cols:
                col = st.selectbox("Categorical column", cat_cols, key="cat_dist")
                counts = df[col].fillna("Missing").astype(str).value_counts()
                fig, ax = plt.subplots(figsize=(5, 3.2))
                sns.barplot(x=counts.index, y=counts.values, ax=ax, color="#2563eb")
                ax.set_ylabel("Count")
                ax.set_xlabel(col)
                fig.tight_layout()
                show_fig(fig)
            else:
                st.info("No categorical columns.")
        with c2:
            if num_cols:
                col = st.selectbox("Numerical column", num_cols, key="num_dist")
                bins = st.slider("Histogram bins", 5, 60, 25)
                fig, ax = plt.subplots(figsize=(5, 3.2))
                sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=ax, color="#0ea5e9")
                fig.tight_layout()
                show_fig(fig)
                skew = df[col].skew()
                if abs(skew) > 1:
                    st.caption(f"Skewness = {skew:.2f}: the distribution has a long tail (a few very large values). "
                               "Random Forests are fairly robust to this.")
            else:
                st.info("No numerical columns.")

    with tabs[4]:
        if report.errors and (target is None or target not in df.columns):
            st.info("Select a valid target column to see the approval distribution.")
        else:
            pos = positive_label_selector(df[target].dropna(), "explorer")
            try:
                y, tinfo = dp.prepare_target(df[target].dropna(), pos)
            except dp.TargetEncodingError as exc:
                st.error(str(exc))
                y = None
            if y is not None:
                counts = y.map({1: f"Approved ({tinfo['positive_label']})",
                                0: f"Not approved ({tinfo['negative_label']})"}).value_counts()
                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots(figsize=(4.5, 3.2))
                    ax.bar(counts.index, counts.values, color=["#16a34a", "#dc2626"][: len(counts)])
                    ax.bar_label(ax.containers[0])
                    ax.set_ylabel("Applications")
                    ax.spines[["top", "right"]].set_visible(False)
                    fig.tight_layout()
                    show_fig(fig)
                with c2:
                    rate = y.mean()
                    st.metric("Approval rate in the dataset", f"{rate:.1%}")
                    if rate > 0.6 or rate < 0.4:
                        st.warning("The classes are imbalanced, so accuracy alone can be misleading. "
                                   "The model uses balanced class weights and we also report precision, recall and F1.")
                cat_cols = [c for c in df.columns if c != target and not pd.api.types.is_numeric_dtype(df[c])]
                cat_cols += [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])
                             and df[c].nunique() <= 5]
                if cat_cols:
                    col = st.selectbox("Approval rate by group", cat_cols, key="rate_by")
                    grouped = (pd.DataFrame({"group": df.loc[y.index, col].fillna("Missing").astype(str), "y": y})
                               .groupby("group")["y"].agg(["mean", "count"]).reset_index())
                    fig, ax = plt.subplots(figsize=(5.5, 3.2))
                    sns.barplot(data=grouped, x="group", y="mean", ax=ax, color="#2563eb")
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Approval rate")
                    ax.set_xlabel(col)
                    fig.tight_layout()
                    show_fig(fig)
                    gap = grouped["mean"].max() - grouped["mean"].min()
                    st.caption(f"Observation: approval rates in this dataset differ by {gap:.1%} between the groups "
                               f"of '{col}' (group sizes: {dict(zip(grouped['group'], grouped['count']))}). "
                               "A gap shows association in the data, not a cause.")

    with tabs[5]:
        num = df.select_dtypes(include="number")
        num = num.loc[:, num.nunique() > 1]
        if num.shape[1] < 2:
            st.info("A correlation heatmap needs at least two non-constant numerical columns, so none is shown.")
        else:
            cols = st.multiselect("Columns to include", list(num.columns), default=list(num.columns))
            if len(cols) >= 2:
                corr = num[cols].corr()
                fig, ax = plt.subplots(figsize=(6.5, 5))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
                fig.tight_layout()
                show_fig(fig)
                pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
                if len(pairs):
                    (a, b), v = pairs.abs().idxmax(), pairs.loc[pairs.abs().idxmax()]
                    st.caption(f"Observation: the strongest pairwise correlation is between **{a}** and **{b}** "
                               f"(r = {v:.2f}). Strongly correlated inputs can share importance in the model.")
            else:
                st.info("Select at least two columns.")


# --------------------------------------------------------------------------- #
# C. Training & Evaluation
# --------------------------------------------------------------------------- #
def page_training():
    st.header("🤖 Model Training & Evaluation")
    df, path, err = get_active_dataset()
    if df is None:
        st.error(err or "No dataset loaded.")
        st.info("Load a dataset in **Dataset Explorer** first.")
        return

    st.markdown("A scikit-learn **Pipeline** (imputation → one-hot encoding → Random Forest) is fitted on the "
                "**training split only**. All metrics below come from the **held-out test split**.")
    target = target_selector(df, "train")
    report = dp.validate_dataset(df, target)
    for msg in report.errors:
        st.error(msg)
    for msg in report.warnings:
        st.warning(msg)
    positive = positive_label_selector(df[target].dropna(), "train") if target in df.columns else None

    with st.expander("Model settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        test_size = c1.slider("Test set share", 0.10, 0.40, 0.20, 0.05, help="Share of rows held out for testing.")
        n_est = c2.slider("Trees", 50, 500, 200, 50)
        depth = c3.selectbox("Max depth", [None, 4, 6, 8, 12], help="None = trees grow until pure.")
        seed = c4.number_input("Random seed", 0, 10_000, dp.RANDOM_SEED, help="Same seed = same split and model.")

    if st.button("🚀 Train model", type="primary", disabled=not report.is_valid):
        try:
            with st.spinner("Training and evaluating..."):
                bundle = mt.train_and_evaluate(df, target, positive, test_size, int(seed), n_est, depth)
                mt.save_model_bundle(bundle, MODEL_PATH)
            st.cache_resource.clear()          # make the app load the freshly saved model
            st.session_state.pop("shap_cache", None)
            st.session_state.pop("applicant", None)
            st.session_state.pop("prediction", None)
            st.success(f"Model trained and saved to `models/{MODEL_PATH.name}`.")
        except (dp.DataValidationError, dp.TargetEncodingError, mt.TrainingError) as exc:
            st.error(f"Training stopped: {exc}")
        except Exception as exc:
            st.error(f"Unexpected training error: {exc}")

    bundle = get_bundle()
    if bundle is None:
        st.warning("The model has not been trained yet.")
        return

    st.subheader("Test-set results")
    st.caption(f"Trained {bundle['trained_at']} on {bundle['n_train']} rows; evaluated on {bundle['n_test']} unseen rows.")
    m = bundle["metrics"]
    cols = st.columns(5)
    for c, (label, key) in zip(cols, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                                      ("F1-score", "f1"), ("Balanced accuracy", "balanced_accuracy")]):
        with c:
            metric_card(label, f"{m[key]:.3f}")
    st.caption(f"Always predicting the majority class would score {m['majority_baseline_accuracy']:.3f} accuracy - "
               "the model should be compared against this baseline, not against 100%.")
    if bundle.get("cv_f1_scores"):
        s = np.array(bundle["cv_f1_scores"])
        st.caption(f"5-fold cross-validation F1 on the training split: {s.mean():.3f} ± {s.std():.3f} "
                   "(shows how much the score varies between splits).")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confusion matrix (test set)**")
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        sns.heatmap(bundle["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=bundle["class_names"], yticklabels=bundle["class_names"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        show_fig(fig)
    with c2:
        st.markdown("**Classification report (test set)**")
        show_df(pd.DataFrame(bundle["report_dict"]).T.round(3))
        tn, fp, fn, tp = bundle["confusion_matrix"].ravel()
        st.caption(f"Reading the matrix: {tp} approved applicants were correctly predicted, {tn} rejected were "
                   f"correctly predicted, {fp} rejected applicants were wrongly predicted as approved and {fn} approved "
                   "applicants were wrongly predicted as rejected.")

    st.markdown("**Feature importance (Random Forest impurity-based)**")
    top_n = st.slider("Features to show", 3, min(25, len(bundle["feature_importances"])), 10)
    fig = ex.plot_importance_bar(bundle["feature_importances"], top_n, "importance",
                                 "Random Forest feature importance")
    show_fig(fig)
    st.caption("Impurity-based importance describes how often the trees used a feature for splitting. It can favour "
               "features with many distinct values and does not show the direction of the effect - see the SHAP page.")

    st.subheader("Limitations of these results")
    for item in bundle["limitations"] + [
        "Possible bias: the data reflects past decisions and may contain historical discrimination (see Fairness page).",
    ]:
        st.markdown(f"- {item}")
    for note in bundle.get("cleaning_notes", []):
        st.caption(f"Preprocessing note: {note}")


# --------------------------------------------------------------------------- #
# D. Loan Prediction
# --------------------------------------------------------------------------- #
def page_prediction():
    st.header("🔮 Loan Prediction")
    st.info(DISCLAIMER)
    bundle = require_model()
    if bundle is None:
        return
    st.caption("This applicant is **illustrative** - values you type are not records from the dataset.")
    defaults = st.session_state.get("applicant")
    with st.form("prediction_form"):
        values = render_inputs(bundle, "pred", defaults)
        submitted = st.form_submit_button("Predict", type="primary")
    if submitted:
        try:
            result = pr.predict_one(bundle, values)
            st.session_state["applicant"] = result["inputs"]
            st.session_state["prediction"] = result
        except pr.InputValidationError as exc:
            for e in exc.errors:
                st.error(e)
            st.session_state.pop("prediction", None)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    result = st.session_state.get("prediction")
    if result:
        for w in result["warnings"]:
            st.warning(w)
        approved = result["class_index"] == 1
        st.markdown(f'<div class="result-card {"result-yes" if approved else "result-no"}">'
                    f'Model prediction: {result["label"]}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            show_fig(probability_chart(result["probabilities"]))
        with c2:
            st.markdown("**Model-estimated class probabilities**")
            for k, v in result["probabilities"].items():
                st.write(f"{k}: **{v:.1%}**")
            st.caption("For a Random Forest this is the share of trees voting for each class (with balanced class "
                       "weights). It is **not** a calibrated, real-world approval probability.")
        st.success("Next: open **SHAP Explanations** to see why the model produced this result, or **What-If "
                   "Analysis** to explore changes.")


# --------------------------------------------------------------------------- #
# E. SHAP
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _explainer(_bundle: dict, key: str):
    return ex.get_explainer(_bundle)


def _global_shap(bundle: dict):
    """Compute global SHAP values once per trained model and keep them in session state."""
    cache = st.session_state.setdefault("shap_cache", {})
    key = bundle["trained_at"]
    if key not in cache:
        X = bundle["X_test"].head(500)
        cache[key] = ex.compute_shap(bundle, X, class_index=1, explainer=_explainer(bundle, key))
    return cache[key]


def page_shap():
    st.header("🔍 Explainable AI - SHAP")
    bundle = require_model()
    if bundle is None:
        return
    st.markdown("**SHAP** assigns every input a value showing how much it moved the model's output for a class, "
                "relative to the average. It uses `TreeExplainer`, the exact method for Random Forests. "
                "*Feature importance describes the model's behaviour, not causes in the real world.*")
    tab_global, tab_local = st.tabs(["Global explanation", "Individual explanation"])

    with tab_global:
        try:
            with st.spinner("Computing SHAP values on the held-out test set..."):
                result = _global_shap(bundle)
        except ex.ExplanationError as exc:
            st.error(f"SHAP could not explain this model: {exc}")
            result = None
        if result is not None:
            st.caption(f"SHAP values for the **{bundle['class_names'][1]}** class, computed on {len(result.data)} "
                       "held-out test applicants (actual dataset records).")
            by_input = ex.global_importance_by_input(result, bundle["feature_columns"])
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**SHAP bar plot (per input column)**")
                show_fig(ex.plot_importance_bar(by_input, 15))
            with c2:
                st.markdown("**SHAP summary plot (per encoded feature)**")
                st.caption("Each dot is an applicant. Colour = feature value (red high, blue low); "
                           "position = push towards (right) or away from (left) 'Approved'.")
                try:
                    show_fig(ex.plot_summary(result))
                except Exception as exc:
                    st.error(f"Could not draw the SHAP summary plot: {exc}")
            top = by_input.head(3)
            st.markdown("**In plain language**")
            st.write("On average, the inputs that moved the model's output the most were "
                     + ", ".join(f"**{r.feature}** ({r.mean_abs_shap:.3f})" for r in top.itertuples())
                     + ". Correlated inputs (for example applicant and co-applicant income) can split credit between "
                       "them, so a low value does not mean an input is irrelevant in reality.")

    with tab_local:
        source = st.radio("Which applicant should be explained?",
                          ["Last applicant from the Loan Prediction page (illustrative)",
                           "A record from the held-out test set (actual dataset record)"], horizontal=True)
        raw_row, actual_label = None, None
        if source.startswith("Last"):
            applicant = st.session_state.get("applicant")
            if not applicant:
                st.info("No applicant yet. Make a prediction on the **Loan Prediction** page first.")
            else:
                raw_row = pr.make_input_frame(bundle, applicant)
        else:
            idx = st.selectbox("Test-set record", list(bundle["X_test"].index),
                               format_func=lambda i: f"Row {i}")
            raw_row = bundle["X_test"].loc[[idx]]
            actual_label = bundle["class_names"][int(bundle["y_test"].loc[idx])]
        if raw_row is not None:
            try:
                pred_idx = int(bundle["pipeline"].predict(raw_row)[0])
                proba = bundle["pipeline"].predict_proba(raw_row)[0]
                res = ex.compute_shap(bundle, raw_row, class_index=pred_idx,
                                      explainer=_explainer(bundle, bundle["trained_at"]))
                contribs = ex.local_contributions(bundle, res, 0, raw_row.iloc[0])
            except ex.ExplanationError as exc:
                st.error(f"SHAP could not explain this prediction: {exc}")
                return
            except Exception as exc:
                st.error(f"Explanation failed: {exc}")
                return
            name = bundle["class_names"][pred_idx]
            msg = f"Model prediction: **{name}** ({proba[pred_idx]:.1%} of trees)."
            if actual_label:
                msg += f" Label recorded in the dataset: **{actual_label}**."
            st.markdown(msg)
            show_fig(ex.plot_local(contribs, name))
            st.caption(f"Base value ({res.base_value:.2f}) + sum of all contributions ({res.values[0].sum():.2f}) = "
                       f"model output for '{name}' ({res.base_value + res.values[0].sum():.2f}). The base value is "
                       "not 50% by design: it is the average output under balanced class weights.")
            st.markdown("**In plain language**")
            for line in ex.plain_language_explanation(contribs, bundle["class_names"], pred_idx,
                                                      float(res.base_value + res.values[0].sum()), res.base_value):
                st.markdown(f"- {line}")
            with st.expander("Contribution table"):
                show_df(contribs.drop(columns="abs_shap").astype({"value": str}))


# --------------------------------------------------------------------------- #
# F. What-if
# --------------------------------------------------------------------------- #
def page_whatif():
    st.header("🧪 What-If Analysis")
    st.info(DISCLAIMER)
    bundle = require_model()
    if bundle is None:
        return
    st.markdown("Change a few inputs and compare the model's predictions. Both predictions come from the **same "
                "trained model**. A change in the prediction shows how the *model* reacts - it does **not** mean "
                "that changing this value would cause a real bank to approve a loan.")
    last = st.session_state.get("applicant")
    base_choice = st.radio("Starting applicant",
                           ["Typical applicant (training medians / most common values)",
                            "Last applicant from the Loan Prediction page"] if last else
                           ["Typical applicant (training medians / most common values)"], horizontal=True)
    original = dict(last) if (last and base_choice.startswith("Last")) else dict(bundle["feature_defaults"])
    original = {c: original[c] for c in bundle["feature_columns"]}

    changed = st.multiselect("Features to change", bundle["feature_columns"],
                             format_func=pretty, help="Pick one or more inputs to modify.")
    modified = dict(original)
    if changed:
        cols = st.columns(min(3, len(changed)))
        for i, col in enumerate(changed):
            modified[col] = input_widget(bundle, col, f"whatif_{col}", original[col], container=cols[i % len(cols)])
    else:
        st.info("Select at least one feature to change.")
        return

    try:
        cmp = pr.compare_predictions(bundle, original, modified)
    except pr.InputValidationError as exc:
        for e in exc.errors:
            st.error(e)
        return
    names = bundle["class_names"]
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Original prediction", cmp["original"]["label"],
                    f"P(Approved) = {cmp['original']['probabilities'][names[1]]:.1%}")
    with c2:
        metric_card("Modified prediction", cmp["modified"]["label"],
                    f"P(Approved) = {cmp['modified']['probabilities'][names[1]]:.1%}")
    with c3:
        metric_card("Change in P(Approved)", f"{cmp['approved_probability_change']:+.1%}",
                    "prediction changed" if cmp["prediction_changed"] else "same predicted class")
    st.markdown("")
    left, right = st.columns([1, 1])
    with left:
        show_fig(probability_chart(cmp["original"]["probabilities"], cmp["modified"]["probabilities"],
                                   ("Original", "Modified")))
    with right:
        st.markdown("**What changed**")
        if cmp["changes"]:
            show_df(pd.DataFrame(cmp["changes"]).astype(str), hide_index=True)
        else:
            st.info("The modified values are identical to the original ones.")
    if cmp["prediction_changed"]:
        st.warning("The predicted class changed. This shows the model is sensitive to these inputs, not that the "
                   "change would cause approval in reality.")
    for w in cmp["modified"]["warnings"]:
        st.warning(w)

    st.subheader("Sweep one numeric feature")
    sweep_cols = bundle["numeric_columns"]
    if sweep_cols:
        col = st.selectbox("Feature to sweep", sweep_cols, format_func=pretty)
        rng = bundle["numeric_ranges"][col]
        grid = np.linspace(rng["p05"], rng["p95"], 25)
        probs = [pr.predict_one(bundle, {**original, col: float(v)})["probabilities"][names[1]] for v in grid]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(grid, probs, marker="o", color="#2563eb")
        ax.axvline(original[col], color="grey", linestyle="--", label="original value")
        ax.set_xlabel(pretty(col))
        ax.set_ylabel("Model-estimated P(Approved)")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        show_fig(fig)
        st.caption(f"Other inputs are held at their original values; {pretty(col)} varies between the 5th and 95th "
                   "percentile of the training data. The curve shows model behaviour, not a causal effect.")


# --------------------------------------------------------------------------- #
# G. Fairness & limitations
# --------------------------------------------------------------------------- #
def page_fairness():
    st.header("⚖️ Fairness & Limitations")
    st.warning("High accuracy does **not** mean the model is fair. This page explains the risks and shows simple, "
               "descriptive group checks.")
    st.markdown("""
**Dataset bias.** The model learns from past decisions. If those decisions were biased, the model can reproduce the bias.

**Historical discrimination.** Lending has a history of unequal treatment by gender, marital status, location and
more. Patterns learned from such data can carry that history forward.

**Sensitive attributes and proxies.** Columns such as *Gender* or *Married* are sensitive. Removing them does not
guarantee fairness because other columns (for example *Property_Area* or *Dependents*) can act as proxies.

**Class imbalance.** More approved than rejected examples make accuracy look better than the model really is, and
the minority class is learned less well.

**Limits of feature importance.** SHAP and impurity importance describe the *model's behaviour*. Correlated features
share importance, so rankings can be unstable.

**Correlation vs causation.** A feature that helps prediction is not necessarily a cause of approval.
What-if results show model sensitivity, not real-world effects.

**Generalisation.** The dataset is small and from one source. Performance on other populations is unknown.

**Human review.** Real lending decisions need accountable human review, legal compliance, adverse-action
explanations and monitoring. This model must never be the sole decision-maker.
""")
    bundle = get_bundle(silent=True)
    if bundle is None:
        st.info("Train a model to see group-wise checks on the held-out test set.")
        return
    st.subheader("Group-wise check on the held-out test set")
    with st.expander("Definitions and limitations of these metrics"):
        for k, v in fa.DEFINITIONS.items():
            st.markdown(f"- **{k}:** {v}")
        st.markdown(f"- **Limitations:** groups with fewer than {fa.MIN_GROUP_SIZE} test rows are flagged and "
                    "ignored in the gap; small groups give very noisy rates; the metrics use a single split, a "
                    "single threshold and only the groups present in this dataset. They cannot prove fairness.")
    cats = [c for c in bundle["categorical_columns"] + bundle["binary_columns"]]
    sensitive = fa.candidate_columns(cats) or cats
    if not sensitive:
        st.info("No categorical columns are available for a group check.")
        return
    col = st.selectbox("Group by", sensitive, format_func=pretty)
    rep = fa.group_report(bundle["X_test"], bundle["y_test"], bundle["y_pred"], col)
    show = rep.rename(columns={"actual_approval_rate": "actual approval rate",
                               "predicted_approval_rate": "predicted approval rate",
                               "true_positive_rate": "true positive rate"})
    show_df(show.round(3), hide_index=True)
    fig, ax = plt.subplots(figsize=(6.5, 3))
    x = np.arange(len(rep))
    ax.bar(x - 0.2, rep["actual_approval_rate"], 0.4, label="Actual", color="#64748b")
    ax.bar(x + 0.2, rep["predicted_approval_rate"], 0.4, label="Predicted", color="#2563eb")
    ax.set_xticks(x, rep["group"])
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    show_fig(fig)
    gap = fa.selection_rate_gap(rep)
    if gap is None:
        st.info("Fewer than two groups have enough test rows for a reliable gap estimate.")
    else:
        st.caption(f"Largest difference in predicted approval rate between groups: {gap:.1%}. A difference is a signal "
                   "to investigate, not a verdict - it can come from real differences in the data, small samples or bias.")
    if rep["small_group"].any():
        st.warning("Some groups are very small; treat their rates with caution.")


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
PAGES = {
    "🏠 Home Dashboard": page_home,
    "📂 Dataset Explorer": page_explorer,
    "🤖 Training & Evaluation": page_training,
    "🔮 Loan Prediction": page_prediction,
    "🔍 SHAP Explanations": page_shap,
    "🧪 What-If Analysis": page_whatif,
    "⚖️ Fairness & Limitations": page_fairness,
}


def main():
    with st.sidebar:
        st.markdown("## 🏦 LoanExplain AI")
        st.caption("Explainable loan approval prediction")
        choice = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        df, _, _ = get_active_dataset()
        st.markdown(f"Dataset: {'✅ loaded' if df is not None else '⚠️ missing'}")
        st.markdown(f"Model: {'✅ trained' if get_bundle(silent=True) else '⚠️ not trained'}")
        st.divider()
        st.caption("Educational demonstration only. Not a real lending system.")
    try:
        PAGES[choice]()
    except Exception as exc:  # last-resort safety net so the app never shows a raw traceback
        st.error(f"Something went wrong on this page: {exc}")
        st.caption("Try reloading the page. If the problem continues, retrain the model or reload the dataset.")


main()
