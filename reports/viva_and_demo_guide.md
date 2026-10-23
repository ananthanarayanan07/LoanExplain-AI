# IA3 Demonstration Guide, Viva Q&A and Final Checklist

## A. Expected workflow
1. Start the app → **Home** shows "Model not trained" (if it is the first run).
2. **Dataset Explorer** → download the dataset → review quality checks and charts.
3. **Training & Evaluation** → *Train model* → read metrics, confusion matrix, limitations.
4. **Loan Prediction** → enter an illustrative applicant → predict.
5. **SHAP Explanations** → global plots, then the individual explanation.
6. **What-If Analysis** → change Credit History / income → compare.
7. **Fairness & Limitations** → group-wise table and discussion.

## B. Suggested 8-minute demonstration
| Time | Do | Say |
|---|---|---|
| 0:00 | Home page | Purpose, "educational only", objectives. |
| 1:00 | Explorer → Quality checks, approval distribution, correlation | "68.7% approved, so classes are imbalanced; some columns have missing values." |
| 2:30 | Training page → Train | "Split first, pipeline fitted on training data only, metrics on the held-out test set." |
| 3:30 | Metrics + confusion matrix | "Accuracy 0.82 vs baseline 0.69; weaker on rejected applicants (recall 0.61)." |
| 4:30 | Prediction → SHAP local | "Credit history pushed the output up; this describes the model, not causation." |
| 6:00 | What-if (Credit_History 1 → 0) | "Same model, only one input changed; shows sensitivity, not a real-world effect." |
| 7:00 | Fairness page | "High accuracy ≠ fair; sensitive attributes, proxies, small groups; human review needed." |

Tip: run `python train_model.py` before the demo so the model exists even if the venue network blocks the download.

## C. Viva questions and answers
**1. Why a Random Forest?** It handles mixed numeric/categorical data, captures non-linear interactions, needs little scaling, is robust to overfitting compared with one tree, and works with SHAP's exact `TreeExplainer`.

**2. How does a Random Forest work?** Many decision trees are trained on bootstrap samples with random feature subsets at each split; their votes are aggregated. Randomness decorrelates the trees, reducing variance.

**3. What is data leakage and how did you avoid it?** Leakage is information from the test set influencing training. I split the data first, and imputers/encoders are inside a Pipeline fitted only on the training split. The target is never an input feature, and the ID column is dropped.

**4. Why did you not just report accuracy?** The data is imbalanced (68.7% approved). Always predicting "approved" gives 0.691 accuracy. So I also report precision, recall, F1, balanced accuracy, the confusion matrix and the baseline.

**5. How do you interpret your confusion matrix?** 78 approved applicants were found, 7 missed; 23 rejected applicants were found, but 15 were wrongly predicted as approved. The model is better at recognising approvals than rejections.

**6. How did you handle missing values and unseen categories?** Median imputation for numbers, most-frequent for binary and categorical columns, all inside the pipeline. `OneHotEncoder(handle_unknown="ignore")` turns an unseen category into all zeros; the app also warns the user.

**7. What is SHAP?** A method based on Shapley values that assigns each feature a contribution to a prediction. Base value + sum of contributions = model output.

**8. Why `TreeExplainer`?** It is the exact and fast SHAP algorithm for tree ensembles; no sampling approximation is needed.

**9. Why do you convert SHAP output by class?** For a classifier SHAP returns values per class (a list or a 3-D array depending on version). I select the array for the predicted class so contributions match the shown prediction.

**10. Difference between global and local explanations?** Global: which features matter on average across many applicants (bar/summary plots). Local: why this one applicant got this output (local bar plot).

**11. Does SHAP prove that credit history causes approval?** No. It explains the model's behaviour. Correlation is not causation, and correlated features share credit.

**12. Why does SHAP importance differ from Random Forest importance?** Impurity importance counts how useful splits were during training and favours features with many distinct values; SHAP measures the average effect on predictions. Showing both avoids over-trusting one view.

**13. What are the model's "probabilities"?** The share of trees voting for each class (with balanced class weights). They are not calibrated real-world approval probabilities.

**14. What does your what-if analysis show?** How the same trained model's prediction changes when selected inputs change. It does not mean changing that input would cause a real bank to approve a loan.

**15. Is the model fair?** I cannot claim that. High accuracy does not imply fairness. The data may reflect historical bias; Gender and Married are sensitive and Property_Area may be a proxy. My group-wise rates are descriptive, use small groups and one threshold.

**16. What are the main limitations?** Small dataset (614 rows), one random split, imbalance, possible bias, uncalibrated probabilities, single source, no real-world validation.

**17. Why is the model not retrained on every click?** Streamlit reruns the script on every interaction. The model is trained once, saved with Joblib and loaded through `st.cache_resource`; the dataset read is cached with `st.cache_data`.

**18. How would you improve the project?** Calibrate probabilities, compare other models, tune the decision threshold by error cost, use more data, add stronger fairness analysis and counterfactual explanations.

**19. How did you test it?** 25 pytest tests cover dataset validation, training, prediction output, invalid inputs, save/load, and SHAP consistency (additivity and output-shape handling). The tests use a small synthetic fixture only to test code; the reported results come from the real dataset.

**20. Where did the data come from?** The public Loan Prediction dataset (Analytics Vidhya practice contest), downloaded from a public GitHub copy. No records were invented.

## D. Final checklist mapped to the six criteria
**1. Concept Understanding**
- [ ] I can explain Random Forest, pipelines, train/test split and leakage.
- [ ] I can explain SHAP (base value + contributions) and its limits.
- [ ] I can explain precision, recall, F1 and why accuracy alone misleads.

**2. Applying to Data**
- [ ] Dataset downloaded or uploaded; quality checks reviewed.
- [ ] Model trained; metrics come from the held-out test set.
- [ ] Model saved to `models/loan_model.joblib` and loaded by the app.

**3. Visualization & Interpretation**
- [ ] Charts viewed and interpreted: missing values, distributions, approval rate, correlation, confusion matrix, importance, SHAP summary/bar/local, what-if.
- [ ] I can state one real observation from each page.

**4. Presentation**
- [ ] App runs on my laptop (`streamlit run app.py`); demo rehearsed (section B).
- [ ] Screenshots taken from my own app and inserted in the report.

**5. Innovation**
- [ ] SHAP individual explanations, what-if, global importance, evaluation dashboard, data quality checks, plain-language text, fairness section: all demonstrated.

**6. Documentation**
- [ ] README, report (name/date/screenshots filled), `results.json` match my own run.
- [ ] `python -m pytest -v` passes on my machine.
- [ ] References checked; nothing claimed that the app does not do.
