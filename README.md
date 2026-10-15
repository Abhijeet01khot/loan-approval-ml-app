
# 🏦 Loan Approval ML App

A machine learning web application that predicts loan approval outcomes using applicant information and a trained Random Forest classifier.

## Overview

This project demonstrates an end-to-end machine learning workflow, from data preprocessing and model evaluation to interactive predictions through Streamlit.

The model was trained using the Kaggle Loan Prediction Problem dataset.

## Features

- Applicant information input form
- Missing-value imputation and categorical encoding
- Random Forest classification
- Validation accuracy, balanced accuracy, and ROC-AUC evaluation
- Model-estimated approval probability
- Interactive Streamlit interface

## Technology Stack

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

## Dataset

The dataset contains 614 labelled loan applications with applicant information such as income, education, employment status, credit history, loan amount, and property area.

The target column is `Loan_Status`:
- `Y`: Loan approved
- `N`: Loan not approved

Dataset source: [Kaggle Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)

## Model Evaluation

The current 80/20 stratified validation split produced these results:

| Metric | Score |
|---|---:|
| Accuracy | 78.86% |
| Balanced Accuracy | 75.25% |
| ROC-AUC | 0.772 |

These results are from one validation split and may vary on other data.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Abhijeet01khot/loan-approval-ml-app.git
cd loan-approval-ml-app
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Train the model

Ensure `loan_prediction_train.csv` is present, then run:

```bash
python train_model.py
```

This creates `loan_approval_model.joblib`.

### 4. Launch the application

```bash
python -m streamlit run app.py
```

## Project Structure

```text
loan-approval-ml-app/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── loan_prediction_train.csv
└── loan_approval_model.joblib
```

## Important Disclaimer

This project is intended for educational and demonstration purposes only. Its predictions are not financial advice and should not be used to make real lending decisions. The dataset is relatively small and the model has not been validated for real-world lending.
