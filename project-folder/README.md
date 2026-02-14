
# ML Assignment 2 – Classification Models on UCI Breast Cancer (Diagnostic)

## Problem Statement
Build and evaluate multiple classification models to predict whether a breast tumor is **malignant** or **benign** using a public dataset. Implement the following models on the **same dataset** and report the metrics: Accuracy, AUC, Precision, Recall, F1, and MCC.

## Dataset Description
- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository (available via `sklearn.datasets.load_breast_cancer`)
- **Instances:** 569
- **Features:** 30 numeric features (see UCI description for details)
- **Target:** `malignant (0)` or `benign (1)`

> Positive class is **malignant (0)**.

## Models Used
Logistic Regression, Decision Tree, kNN, Naive Bayes (Gaussian), Random Forest, XGBoost

## Evaluation Metrics (Test Split = 20%, stratified)
| ML Model Name            |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-------------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression      |     0.9825 | 0.9954 |      0.9762 |   0.9762 | 0.9762 | 0.9623 |
| Decision Tree            |     0.9123 | 0.9157 |      0.8478 |   0.9286 | 0.8864 | 0.8174 |
| kNN                      |     0.9561 | 0.9788 |      0.9512 |   0.9286 | 0.9398 | 0.9054 |
| Naive Bayes (Gaussian)   |     0.9386 | 0.9878 |      0.9268 |   0.9048 | 0.9157 | 0.8676 |
| Random Forest (Ensemble) |     0.9474 | 0.9937 |      0.9286 |   0.9286 | 0.9286 | 0.8869 |
| XGBoost (Ensemble)       |     0.9561 | 0.995  |      0.9744 |   0.9048 | 0.9383 | 0.9058 |

## Project Structure
```
project-folder/
├── app.py
├── requirements.txt
├── README.md
├── metrics.csv
├── metrics.md
├── reports.md
└── model/
    ├── *.pkl
    └── train.py
```
