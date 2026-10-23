# Dataset

LoanExplain AI uses the public **Loan Prediction** dataset (originally from Analytics Vidhya's
"Loan Prediction Problem" practice contest; 614 applications, 12 input columns + `Loan_Status`).
The records are real contest data - nothing in this project is invented or simulated.

## Option 1 - automatic (recommended)
Open the app -> **Dataset Explorer** -> **Download dataset**, or run `python train_model.py`.
Both fetch `loan_prediction.csv` from:
https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv

## Option 2 - manual
1. Open the URL above (or search Kaggle for "Loan Prediction Problem Dataset", file `train_u6lujuX_CVtuZ9i.csv`).
2. Save the file as `data/loan_prediction.csv`.

## Option 3 - upload your own CSV
Use **Dataset Explorer -> Upload**. The file needs a binary target column (e.g. `Loan_Status` with Y/N).
If your target has another name, select it from the drop-down. If the two values are not Y/N, Yes/No or 0/1,
the app asks which value means "approved".

Column meanings: Loan_ID (dropped, identifier), Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount (thousands), Loan_Amount_Term (months),
Credit_History (1/0), Property_Area, Loan_Status (Y/N, target).
