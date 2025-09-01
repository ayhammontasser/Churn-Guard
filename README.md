 Churn-Guard
 Introduction

Customer churn is one of the biggest challenges for businesses.
Churn-Guard is a machine learning project that uses the IBM Telco Customer Churn Dataset to predict which customers are most likely to leave.
The project provides actionable insights to support marketing retention strategies and improve Customer Lifetime Value (CLV).

 Project Objectives

Perform EDA to understand churn patterns.

Engineer relevant features (contracts, payment methods, tenure buckets, charges).

Build and compare ML models (Logistic Regression, Random Forest, XGBoost).

Evaluate performance using Recall, Precision, F1-score, ROC-AUC.

Provide explainability (SHAP values, feature importance).

Deploy an interactive dashboard and API for predictions.

 Dataset

Source: IBM Telco Customer Churn Dataset (Kaggle)

Size: 7043 customers × 21 features

Target Variable: Churn (Yes/No)

 Repository Structure
Churn-Guard/
│
├─ data/
│   ├─ raw/               # raw dataset from Kaggle
│   └─ processed/         # cleaned/encoded data
│
├─ notebooks/
│   ├─ 01_eda.ipynb       # exploratory data analysis
│   ├─ 02_features.ipynb  # feature engineering
│   ├─ 03_modeling.ipynb  # model training & evaluation
│   └─ 04_explain.ipynb   # SHAP, insights
│
├─ src/
│   ├─ data/              # scripts for data loading/cleaning
│   ├─ features/          # feature engineering pipeline
│   ├─ models/            # training, prediction, evaluation
│   └─ utils/             # metrics, plots
│
├─ app/
│   ├─ streamlit_app.py   # interactive churn dashboard
│   └─ api.py             # FastAPI endpoint
│
├─ models/                # saved ML pipelines
├─ requirements.txt
├─ README.md
└─ .gitignore

 Tech Stack

Python (Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn)

Visualization: Matplotlib, Seaborn, Plotly

Explainability: SHAP

Deployment: Streamlit, FastAPI, Docker (optional)

Version Control: Git + GitHub

 Roadmap

 Data ingestion & cleaning

 Exploratory Data Analysis (EDA)

 Feature engineering & preprocessing pipeline

 Baseline model (Logistic Regression)

 Advanced models (Random Forest, XGBoost)

 Model explainability with SHAP

 Build Streamlit dashboard

 Deploy FastAPI endpoint

 Add monitoring & retraining pipeline

 Outcomes

Predict churn probabilities for each customer.

Insights into top churn drivers (contract type, monthly charges, tenure, payment method).

Dashboard for marketing teams to test scenarios and plan retention campaigns.

 Team

This repository is collaboratively developed by the Churn-Guard Team (Data & Marketing).
