# model/train.py
"""
Training script for ML Assignment 2 on the UCI Breast Cancer (Diagnostic) dataset.
- Trains 6 classifiers
- Evaluates on a stratified 80/20 hold-out split
- Saves trained models to ./model/*.pkl
- Saves metrics to ./metrics.csv
- Saves classification reports and confusion matrices to ./reports.md

Positive class is **malignant (label=0)** for clinical relevance.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    roc_auc_score, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Output dirs
os.makedirs('model', exist_ok=True)

# Load dataset
bunch = load_breast_cancer(as_frame=True)
df = bunch.frame.copy()
X = df.drop(columns=['target'])
y = df['target']
feature_names = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessors
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
preprocess = ColumnTransformer(transformers=[('num', numeric_transformer, feature_names)], remainder='passthrough')

# Models
models = {
    'Logistic Regression': Pipeline([('preprocess', preprocess), ('clf', LogisticRegression(max_iter=5000))]),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'kNN': Pipeline([('preprocess', preprocess), ('clf', KNeighborsClassifier(n_neighbors=5))]),
    'Naive Bayes (Gaussian)': GaussianNB(),
    'Random Forest (Ensemble)': RandomForestClassifier(n_estimators=300, random_state=42),
}
if HAS_XGB:
    models['XGBoost (Ensemble)'] = XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, eval_metric='logloss'
    )

# Train & evaluate
rows = []
reports_text = []
# For AUC: remap malignant(0)->1 for convenience
y_test_auc = (y_test == 0).astype(int)

for name, model in models.items():
    model.fit(X_train, y_train)

    # Save model
    fname = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')
    with open(f'model/{fname}.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics (positive label = malignant = 0)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=0)
    rec = recall_score(y_test, y_pred, pos_label=0)
    f1 = f1_score(y_test, y_pred, pos_label=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        pos_index = list(model.classes_).index(0)
        y_score = proba[:, pos_index]
        auc = roc_auc_score(y_test_auc, y_score)
    else:
        auc = float('nan')

    rows.append({'ML Model Name': name, 'Accuracy': round(acc,4), 'AUC': round(auc,4) if not np.isnan(auc) else np.nan,
                 'Precision': round(prec,4), 'Recall': round(rec,4), 'F1': round(f1,4), 'MCC': round(mcc,4)})

    # Reports
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    rep = classification_report(y_test, y_pred, target_names=list(bunch.target_names))
    reports_text.append(f"## {name}\n\n````\n{rep}\n````\n\nConfusion Matrix (rows=true [malignant=0, benign=1], cols=pred):\n\n{cm}\n\n")

# Save metrics
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv('metrics.csv', index=False)

with open('reports.md', 'w') as f:
    f.write('\n\n'.join(reports_text))

print(metrics_df)
# See earlier provided training script in the chat; this is a minimal placeholder to keep structure.
# You can overwrite this with the full version if needed.
