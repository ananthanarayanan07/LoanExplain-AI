# LoanExplain AI: Transparent Loan Approval Prediction Using Explainable Artificial Intelligence

**Student name:** `[YOUR NAME]`  **Roll number:** `[YOUR ROLL NO]`  **Course / IA3:** `[COURSE NAME]`  **Date:** `[DATE]`

> All numerical results in section 17 come from an actual run of `python train_model.py` (seed 42, 80/20
> stratified split, 200 trees) on the public Loan Prediction dataset, stored in `reports/results.json`. Re-run the
> script and update the numbers if you change the dataset, seed or library versions. Items in `[BRACKETS]` are
> for you to fill in.

---

## 1. Title
LoanExplain AI: Transparent Loan Approval Prediction Using Explainable Artificial Intelligence.

## 2. Abstract
Machine-learning models are increasingly used to support lending, but their predictions are often opaque.
This project builds an interactive Streamlit application that trains a Random Forest classifier on the public
Loan Prediction dataset (614 applications, 11 usable input features) and explains each prediction with SHAP
(SHapley Additive exPlanations). The application validates and cleans the data, evaluates the model on a held-out
test set, offers individual and global explanations, supports what-if analysis, and discusses fairness and
limitations. On the held-out test set (123 applications) the model reached an accuracy of 0.821 and an F1-score of
0.876 for the "Approved" class, compared with 0.691 accuracy for always predicting the majority class. SHAP
identified credit history as the most influential input. The system is an educational demonstration and not a
real lending tool.

## 3. Introduction
Lenders must decide whether an applicant is likely to repay a loan. Data-driven models can support this, but
people affected by the decision, and the people who build the system, need to understand *why* a model produced
a result. Explainable AI (XAI) provides tools for this. This project combines a standard classifier with SHAP to
make a loan-approval model transparent.

## 4. Problem Statement
Given historical loan applications with applicant and loan details, predict whether an application was approved,
and explain the model's output so that users can see which inputs influenced it and understand the limits of
that explanation.

## 5. Objectives
1. Prepare real loan data safely (validation, missing values, categorical encoding, no data leakage).
2. Train a Random Forest classifier and evaluate it honestly on unseen data.
3. Explain predictions globally and individually using SHAP.
4. Provide an interactive what-if tool to explore model sensitivity.
5. Present the work in a clear dashboard with a fairness and limitations section.

## 6. Scope of the Project
**In scope:** binary classification (approved / not approved) on tabular data; a single Random Forest model;
TreeExplainer-based SHAP; a local web dashboard; basic group-wise fairness diagnostics.
**Out of scope:** real banking decisions, credit scoring regulation, model calibration, other algorithms,
deployment to production, legal compliance.

## 7. Existing System and Its Limitations
Traditional loan assessment uses manual review or rule-based scorecards. Manual review is slow and inconsistent;
scorecards are transparent but less flexible. Many machine-learning demos report only accuracy, give a single
label with no explanation, retrain the model on every interaction, or ignore fairness. Black-box predictions
provide no insight into which factors mattered, which makes errors and bias hard to detect.

## 8. Proposed System
LoanExplain AI addresses these limitations by (a) validating data automatically, (b) using a leakage-free
scikit-learn pipeline, (c) reporting several metrics computed only on a held-out test set, (d) explaining every
prediction with SHAP, (e) supporting what-if exploration, and (f) documenting fairness risks. The model is trained
once, saved with Joblib and loaded for prediction, so interactions never retrain it.

## 9. System Architecture
```
 CSV dataset ──► data_processing.py ──► validation report / cleaned data
                                              │ (stratified split, seed 42)
                                              ▼
                              model_training.py: Pipeline
                  [Impute → One-Hot (categoricals)] → RandomForest(balanced)
                         │ fitted on TRAIN only        │
                         ▼                             ▼
              metrics on TEST set         models/loan_model.joblib  (pipeline + metadata)
                                                      │
              ┌───────────────────────────────────────┼─────────────────────────┐
              ▼                                       ▼                         ▼
        prediction.py                         explainability.py            fairness.py
   (validate inputs, predict,             (SHAP TreeExplainer, plots,     (group-wise rates
    what-if comparison)                    plain-language text)            on test set)
              └───────────────────────────────────────┴─────────────────────────┘
                                              ▼
                                   app.py (Streamlit, 7 pages)
```

## 10. Technologies Used
Python 3; Streamlit (GUI); Pandas and NumPy (data); scikit-learn (Pipeline, ColumnTransformer, Random Forest,
metrics); SHAP (explanations); Matplotlib and Seaborn (visualisation); Joblib (model persistence); pytest (tests).
Tested versions are listed in `requirements.txt`.

## 11. Dataset Description
The public **Loan Prediction** dataset (originating from an Analytics Vidhya practice contest) was downloaded from
a public GitHub copy. It has 614 rows and 13 columns: Loan_ID, Gender, Married, Dependents, Education,
Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount (thousands), Loan_Amount_Term (months),
Credit_History (1/0), Property_Area and the target Loan_Status (Y/N). In this data 422 applications (68.7%) are
approved and 192 (31.3%) are not. Several columns contain missing values (for example LoanAmount and
Credit_History); see the Dataset Explorer page for exact counts. `[Add: source URL / date accessed]`.

## 12. Data Preprocessing Methodology
1. **Validation:** the dataset must be non-empty, contain a binary target and enough rows per class.
2. **Cleaning:** whitespace stripped, blanks converted to missing, rows without a target dropped, identifier column
   `Loan_ID` removed (11 features remain).
3. **Column types:** numeric (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term), binary numeric
   (Credit_History) and categorical (Gender, Married, Dependents, Education, Self_Employed, Property_Area).
4. **Split first:** a stratified 80/20 train/test split (seed 42) is made *before* fitting any preprocessing.
5. **Pipeline (fitted on training data only):** numeric → median imputation; binary → most-frequent imputation;
   categorical → most-frequent imputation then one-hot encoding with `handle_unknown="ignore"`.
6. **Imbalance:** `class_weight="balanced"` and stratification. Scaling is not applied because trees are not
   sensitive to feature scale.

## 13. Random Forest Algorithm Explanation
A Random Forest is an ensemble of decision trees. Each tree is trained on a bootstrap sample of the training rows
and considers a random subset of features at every split, which decorrelates the trees. For classification,
the forest aggregates the trees' votes; the fraction of trees voting for a class is used here as a model-estimated
class probability (it is not guaranteed to be calibrated). Random Forests handle mixed feature types, capture
non-linear interactions and are less prone to overfitting than a single tree (Breiman, 2001). Settings used: 200
trees, unlimited depth, balanced class weights, `random_state=42`.

## 14. Explainable AI and SHAP Explanation
SHAP is based on Shapley values from cooperative game theory. For a prediction, each feature receives a value
equal to its average marginal contribution to the model output across all feature orderings. Contributions add up:
*base value + sum of SHAP values = model output* (verified in the application and in the automated tests).
`TreeExplainer` computes these values exactly and efficiently for tree ensembles (Lundberg et al., 2020).
In this project SHAP values are computed on the preprocessed features, explain the predicted class, and are
summed back to the original input columns for local explanations.
**Caveats:** SHAP describes the *model's* behaviour, not causes in the real world; correlated features share
credit; one-hot features of a category must be read together.

## 15. System Implementation
| Module | Responsibility |
|---|---|
| `src/data_processing.py` | CSV loading, download, target detection, validation, cleaning, target encoding |
| `src/model_training.py` | Pipeline construction, training, test-set evaluation, cross-validation, Joblib save/load |
| `src/prediction.py` | Input validation, prediction, original-vs-modified comparison |
| `src/explainability.py` | SHAP computation (handles different SHAP output shapes), plots, plain-language text |
| `src/fairness.py` | Group-wise actual/predicted approval rates and true-positive rate |
| `train_model.py` | Command-line training, writes `reports/results.json` |
| `app.py` | Streamlit interface; caches the dataset and the loaded model so reruns do not retrain |
| `tests/test_model.py` | 25 automated tests |

## 16. GUI Module Descriptions
- **Home Dashboard** - introduction, objectives, dataset/model status, row and feature counts, test summary.
- **Dataset Explorer** - upload/download, preview, dtypes, missing values, duplicates, statistics, distributions, approval distribution, correlation heatmap.
- **Training & Evaluation** - settings, training, metrics, confusion matrix, classification report, feature importance, limitations.
- **Loan Prediction** - dynamic validated form, predicted class, class probabilities, disclaimer.
- **SHAP Explanations** - global bar and summary plots; local explanation for one applicant.
- **What-If Analysis** - modify inputs, compare predictions, sweep one feature.
- **Fairness & Limitations** - risks and group-wise test-set checks.

**Screenshots** (replace each placeholder with your own captures; see README section 7):

`[SCREENSHOT 1: Home dashboard  → ![Home](screenshots/01_home.png)]`
`[SCREENSHOT 2: Dataset Explorer → ![Explorer](screenshots/02_explorer.png)]`
`[SCREENSHOT 3: Training & Evaluation → ![Training](screenshots/03_training.png)]`
`[SCREENSHOT 4: Loan Prediction → ![Prediction](screenshots/04_prediction.png)]`
`[SCREENSHOT 5: SHAP global → ![SHAP global](screenshots/05_shap_global.png)]`
`[SCREENSHOT 6: SHAP local → ![SHAP local](screenshots/06_shap_local.png)]`
`[SCREENSHOT 7: What-If → ![What-if](screenshots/07_whatif.png)]`
`[SCREENSHOT 8: Fairness → ![Fairness](screenshots/08_fairness.png)]`

## 17. Results and Evaluation Metrics
Source: `reports/results.json`, produced by `python train_model.py`. Training rows: 491; test rows: 123.

| Metric (test set) | Value |
|---|---|
| Accuracy | 0.821 |
| Precision (Approved) | 0.839 |
| Recall (Approved) | 0.918 |
| F1-score (Approved) | 0.876 |
| Balanced accuracy | 0.762 |
| Majority-class baseline accuracy | 0.691 |
| 5-fold CV F1 on training split | 0.848 ± 0.035 |

**Confusion matrix** (rows = actual, columns = predicted):

| | Predicted Not approved | Predicted Approved |
|---|---|---|
| **Actual Not approved** | 23 | 15 |
| **Actual Approved** | 7 | 78 |

**Per-class report:** Not approved - precision 0.77, recall 0.61, F1 0.68 (38 test rows); Approved - precision 0.84,
recall 0.92, F1 0.88 (85 test rows).

## 18. Visualization Interpretation
- **Confusion matrix:** the model finds most approved applications (78 of 85) but misses many rejected ones
  (only 23 of 38 identified); 15 rejected applicants were predicted "Approved". Because the cost of a wrongly
  approved loan can be high in practice, this weakness matters more than the headline accuracy.
- **Accuracy vs baseline:** 0.821 against 0.691 shows the model learns something beyond the class ratio, but the
  gain is moderate. Balanced accuracy (0.762) is lower because the minority class is predicted less well.
- **Random Forest feature importance (top):** ApplicantIncome 0.204, LoanAmount 0.192, Credit_History 0.181,
  CoapplicantIncome 0.118. Impurity importance favours numeric features with many distinct values.
- **SHAP global importance (mean |SHAP| per input, 'Approved' class, test set):** Credit_History 0.159,
  Property_Area 0.051, ApplicantIncome 0.047, LoanAmount 0.043, CoapplicantIncome 0.032. SHAP ranks credit history
  far above the others, which differs from the impurity ranking; this is why several importance views are shown.
- **Group-wise check (test set, descriptive only):** predicted approval rate was 0.68 for Female (n=25) and 0.77 for
  Male (n=96); the actual rates in the data were 0.56 and 0.73. Groups are small, so these gaps are not conclusive.
  Predicted rates were higher than actual rates in every group shown, which reflects the model's tendency to
  predict "Approved".
`[Add your own observations after viewing the charts in your running app.]`

## 19. Innovation and Contributions
SHAP-based individual explanations with plain-language text; interactive what-if analysis and one-feature sweeps;
global importance from two methods; evaluation dashboard with baseline and cross-validation; automated dataset
quality checks; a fairness and limitations section with group-wise metrics; dynamic input forms generated from the
trained pipeline; leakage-free, cached, reproducible training.

## 20. Ethical Considerations and Limitations
- The data is small (614 rows) and from one source; metrics vary with the split and may not generalise.
- Classes are imbalanced (68.7% approved); accuracy alone can mislead.
- Historical data can encode discrimination; sensitive attributes (e.g. Gender, Married) and proxies
  (e.g. Property_Area) are present. High accuracy does not imply fairness.
- Fairness checks here are descriptive, use small groups and a single threshold.
- Feature importance and SHAP describe model behaviour, not causation; correlated features share importance.
- What-if results show model sensitivity, not that changing a value would cause real approval.
- Random Forest probabilities are vote fractions, not calibrated probabilities.
- Real lending requires human review, legal compliance and monitoring. This system is for education only.

## 21. Future Enhancements
Probability calibration; comparison with logistic regression and gradient boosting; threshold tuning based on the
cost of errors; larger, more recent datasets; richer fairness metrics and mitigation methods; counterfactual
explanations; user authentication and logging; deployment to a cloud service.

## 22. Conclusion
The project delivers a working, reproducible application that trains and honestly evaluates a Random Forest on real
loan data and explains its predictions with SHAP. The model beats the majority-class baseline (accuracy 0.821 vs
0.691) but misclassifies many rejected applications, which shows why transparent evaluation and human oversight are
necessary. The explanation, what-if and fairness tools help users understand both the model and its limits.

## 23. References
> Verify details (volume, pages, URLs) against the original sources before submitting.

1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
2. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems 30*.
3. Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56-67.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
5. Hardt, M., Price, E., & Srebro, N. (2016). Equality of Opportunity in Supervised Learning. *Advances in Neural Information Processing Systems 29*.
6. Analytics Vidhya. *Loan Prediction Problem* practice dataset. Copy used: https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv `[date accessed]`.
7. Streamlit documentation: https://docs.streamlit.io  ·  SHAP documentation: https://shap.readthedocs.io
