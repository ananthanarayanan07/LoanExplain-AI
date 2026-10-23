# LoanExplain AI

**Transparent Loan Approval Prediction Using Explainable Artificial Intelligence**

An interactive Streamlit dashboard that trains a Random Forest on the public *Loan Prediction* dataset, predicts
loan approval, and explains every prediction with **SHAP**.

> ⚠️ **Educational use only.** This is an academic demonstration. It does not make real banking decisions and
> nothing it shows guarantees loan approval.

---

## 1. Project structure

```
LoanExplain-AI/
├── app.py                  # Streamlit dashboard (7 pages)
├── train_model.py          # Command-line training script
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── README.md           # How to get the dataset (the CSV is downloaded, not bundled)
├── models/
│   └── .gitkeep            # loan_model.joblib is created here after training
├── src/
│   ├── data_processing.py  # load / validate / clean data, detect target
│   ├── model_training.py   # Pipeline + Random Forest, evaluation, save / load
│   ├── prediction.py       # input validation, prediction, what-if comparison
│   ├── explainability.py   # SHAP (TreeExplainer), plots, plain-language text
│   └── fairness.py         # simple group-wise checks on the test set
├── reports/
│   ├── project_report.md   # Academic report (real results from an actual run)
│   ├── results.json        # Metrics written by train_model.py
│   └── viva_and_demo_guide.md
└── tests/
    └── test_model.py       # 25 automated tests
```

I added `src/fairness.py` and `reports/viva_and_demo_guide.md` to your suggested structure; everything else matches it.

## 2. Dataset

The app uses the public **Loan Prediction** dataset (614 applications; columns: Loan_ID, Gender, Married,
Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area, Loan_Status). It is downloaded from a public GitHub copy of the data:
`https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv`.
No records are invented. If the download fails, see `data/README.md` for manual download or upload a CSV in the app.

## 3. Setup on Windows (step by step)

**Step 1 - Install Python (skip if `py --version` already shows 3.11 or 3.12).**
Download Python 3.12 from <https://www.python.org/downloads/windows/>. In the installer, **tick "Add python.exe to PATH"**,
then click *Install Now*. Open a new PowerShell window and check:

```powershell
py --version
```

**Step 2 - Open the project folder in VS Code.**
Unzip the project, open VS Code, choose *File → Open Folder…* and select `LoanExplain-AI`.
Open a terminal with *Terminal → New Terminal* (it should be PowerShell). Or from PowerShell:

```powershell
cd path\to\LoanExplain-AI
code .
```

**Step 3 - Create a virtual environment.**

```powershell
py -m venv .venv
```

**Step 4 - Activate it (PowerShell).**

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the start of the prompt. If PowerShell says scripts are disabled, run this once and retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Step 5 - Install dependencies.**

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Step 6 - Get the dataset.** Choose one:
- Do nothing: `python train_model.py` and the app's **Download dataset** button fetch it automatically; or
- Save the CSV manually as `data\loan_prediction.csv` (see `data/README.md`); or
- Upload any compatible CSV on the **Dataset Explorer** page.

**Step 7 - Train the model.**

```powershell
python train_model.py
```

(You can also click **Train model** inside the app.) This saves `models\loan_model.joblib`.

**Step 8 - Launch the app.**

```powershell
streamlit run app.py
```

Your browser opens at `http://localhost:8501`.

**Step 9 - Run the tests.**

```powershell
python -m pytest -v
```

**Step 10 - Troubleshooting.**

| Problem | Fix |
|---|---|
| `py` / `python` not recognised | Reinstall Python and tick *Add python.exe to PATH*; open a **new** terminal. |
| `Activate.ps1 cannot be loaded` | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, then activate again. |
| `streamlit` is not recognised | The venv is not active. Activate it, or run `python -m streamlit run app.py`. |
| `ModuleNotFoundError: No module named 'src'` | Run commands from inside the `LoanExplain-AI` folder, not a parent or sub-folder. |
| `pip install` fails for `shap` / `numba` / `llvmlite` | Use Python 3.11 or 3.12 (very new Python versions may lack wheels). Recreate the venv with `py -3.12 -m venv .venv`. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| "Could not load the saved model" after upgrading packages | Delete `models\loan_model.joblib` and train again. |
| Dataset download blocked (college network) | Download the CSV in a browser and save as `data\loan_prediction.csv`. |
| App shows old data after replacing the CSV | Press **R** in the browser to rerun, or restart Streamlit. |

## 4. How the app works

| Page | What it does |
|---|---|
| 🏠 Home Dashboard | Intro, objectives, dataset / model status, test-set summary (or a clear "not trained" notice). |
| 📂 Dataset Explorer | Download/upload CSV, preview, dtypes, missing values, duplicates, statistics, distributions, approval rate by group, correlation heatmap (hidden when unsuitable), auto-generated observations. |
| 🤖 Training & Evaluation | Validation, stratified train/test split, Random Forest, accuracy/precision/recall/F1, confusion matrix, classification report, cross-validation, feature importance, limitations, save model. |
| 🔮 Loan Prediction | Form generated from the trained pipeline's columns; validation; predicted class; model-estimated probabilities; disclaimer. |
| 🔍 SHAP Explanations | Global bar + summary plots; local explanation (illustrative applicant or real test record); plain-language text. |
| 🧪 What-If Analysis | Change chosen inputs, compare original vs modified prediction/probabilities, plus a one-feature sweep curve. |
| ⚖️ Fairness & Limitations | Bias, imbalance, causation, human review, plus group-wise rates on the test set with definitions. |

**Design decisions (assumptions I made)**
- The target is converted to 0/1 (1 = approved). Y/Yes/1/Approved are recognised automatically; otherwise the app asks which value means "approved".
- ID-like columns (e.g. `Loan_ID`) are dropped. `Dependents` is treated as a category (it contains "3+").
- Numeric gaps → median; binary numeric (`Credit_History`) and category gaps → most frequent value; categories are one-hot encoded with `handle_unknown="ignore"`. All of this lives inside one Pipeline fitted on the training split only (no leakage).
- Class imbalance is handled with `class_weight="balanced"` and a stratified split.
- The model is trained once, saved with Joblib, and loaded through `st.cache_resource`; Streamlit reruns never retrain it. Training happens only when you press the button or run `train_model.py`.
- Random Forest "probabilities" are the share of trees voting for each class; they are **not** calibrated real-world probabilities, and the app says so.
- SHAP values are computed with `TreeExplainer` on the preprocessed features and explain the **predicted class**. Local results are summed back to the original input columns for readability.

## 5. Verification status (what was actually run)

Run in a Linux sandbox with Python 3.12, streamlit 1.66, pandas 3.0, numpy 2.4, scikit-learn 1.8, shap 0.54:
- `python -m pytest` → **25 passed**.
- `python train_model.py` (including automatic dataset download) completed; results are in `reports/results.json` and the report.
- Every Streamlit page was loaded headlessly with Streamlit's `AppTest` (with and without a dataset/model), and a prediction, SHAP explanation and what-if change were exercised without errors.

**Not verified:** visual appearance in a real browser, and the Windows/PowerShell steps (they are standard commands but were not run on Windows). Check the screenshots yourself before submitting.

## 6. How the project addresses the six IA3 criteria (5 marks each)

| Criterion | Where it is demonstrated |
|---|---|
| **1. Concept Understanding** | Random Forest, pipelines, train/test split, leakage prevention, class imbalance, SHAP and the correlation-vs-causation caveat are implemented *and explained in the app* (Training, SHAP, Fairness pages) and in report sections 13-14. Baseline accuracy and cross-validation show understanding of why accuracy alone misleads. |
| **2. Applying to Data** | Real public dataset; automated validation; handling of missing values, categories and unexpected inputs; stratified split with fixed seed; evaluation only on the held-out test set; saved/loaded Joblib pipeline (`src/data_processing.py`, `src/model_training.py`). |
| **3. Visualization & Interpretation** | Missing-value, distribution, approval-rate and correlation charts; confusion matrix; feature importance; SHAP summary/bar/local plots; what-if bars and sweep curve. Each chart has a data-driven caption interpreting it (generated from the actual values, never hard-coded). |
| **4. Presentation** | Sidebar navigation, metric cards, tooltips, consistent layout, clear success/warning/error messages, graceful "not trained / no dataset" states, step-by-step Windows guide, demo script in `reports/viva_and_demo_guide.md`. |
| **5. Innovation** | SHAP individual explanations; interactive what-if analysis and sweep; global importance; evaluation dashboard; automated dataset quality checks; plain-language explanations; fairness section with group-wise metrics and stated limitations. |
| **6. Documentation** | This README, `data/README.md`, a 23-section academic report with real results, commented code, 25 tests, and a viva/demo guide. |

## 7. Adding screenshots to the report

1. Run the app and visit each page after training.
2. Press `Win + Shift + S`, capture the page, and save PNGs in `reports\screenshots\` (e.g. `01_home.png`, `02_explorer.png`, `03_training.png`, `04_prediction.png`, `05_shap_global.png`, `06_shap_local.png`, `07_whatif.png`, `08_fairness.png`).
3. In `reports/project_report.md`, section 16, replace each placeholder line with `![Home dashboard](screenshots/01_home.png)`.

Screenshots must be of **your own running application**.
