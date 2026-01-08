**HR Analytics – Employee Attrition Prediction**

A production-ready machine learning pipeline for predicting employee attrition using the IBM HR Analytics dataset.
The project covers data ingestion, preprocessing, model training, evaluation, and deployment via Streamlit.

**Project Overview**

Employee attrition is costly and difficult to predict without data-driven insights.
This project builds an end-to-end ML system that:

- Processes raw HR data

- Trains and evaluates multiple classification models

- Selects the best-performing model

- Deploys predictions via a Streamlit web application

**Dataset**

Source: IBM HR Analytics Employee Attrition Dataset

Target Variable: Attrition (Yes / No)

Features: Demographics, job role, compensation, satisfaction metrics, and work history

**Machine Learning Pipeline**
-- Data Ingestion

Reads raw CSV data

Saves raw, train, and test datasets

Uses stratified sampling to preserve class balance

-- Preprocessing

Numerical features:

Median imputation

Standard scaling

Categorical features:

Most-frequent imputation

One-hot encoding

Implemented using ColumnTransformer

Saved as a reusable pipeline (preprocessor.pkl)

-- Model Training

Models evaluated:

Logistic Regression

Random Forest

Decision Tree

Cross-validation and GridSearchCV applied

Best model selected using F1-score

Final model saved as model.pkl

-- Evaluation Metrics

F1 Score

Recall

ROC-AUC

**Streamlit Application
**
The Streamlit app allows users to:

-- Input employee attributes

-- Predict attrition risk

-- Display prediction probabilities

**Best Practices Applied**

-- No data leakage between train and test sets

-- Saved preprocessing + model artifacts

-- Clean modular codebase

-- Production-compatible inference

-- Git-safe (no venv/ committed)

Author

John Emmanuel Durosimi Terry

Data Scientist | Machine Learning Engineer
