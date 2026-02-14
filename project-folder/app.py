
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title='ML Assignment 2 - Classification Models', layout='wide')

st.title('ML Assignment 2 – Classification Models on UCI Breast Cancer (Diagnostic)')
st.write('Upload **test CSV** (columns must match the original feature names) or use the demo test split.')

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
models = {}
for mf in os.listdir(MODEL_DIR):
    if mf.endswith('.pkl'):
        with open(os.path.join(MODEL_DIR, mf), 'rb') as f:
            models[mf.replace('.pkl','').replace('_',' ').title()] = pickle.load(f)

feature_names = ['mean radius','mean texture','mean perimeter','mean area','mean smoothness','mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension','radius error','texture error','perimeter error','area error','smoothness error','compactness error','concavity error','concave points error','symmetry error','fractal dimension error','worst radius','worst texture','worst perimeter','worst area','worst smoothness','worst compactness','worst concavity','worst concave points','worst symmetry','worst fractal dimension']

model_name = st.sidebar.selectbox('Select Model', sorted(models.keys()))
uploaded = st.file_uploader('Upload test CSV (X_test) with the 30 features as columns (no target column).', type=['csv'])

bunch = load_breast_cancer(as_frame=True)
X_default = bunch.frame.drop(columns=['target'])
y_default = bunch.frame['target']

if uploaded is not None:
    X_test = pd.read_csv(uploaded)
    missing = [c for c in feature_names if c not in X_test.columns]
    if missing:
        st.error(f'Missing columns: {missing}')
        st.stop()
    X_test = X_test[feature_names]
    y_test = None
else:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_default, y_default, test_size=0.2, random_state=42, stratify=y_default)

model = models[model_name]
if hasattr(model, 'predict_proba'):
    proba = model.predict_proba(X_test)
    pos_index = list(model.classes_).index(0)
    y_score = proba[:, pos_index]
    y_pred = (y_score >= 0.5).astype(int)
    y_pred = np.where(y_score >= 0.5, 0, 1)
else:
    y_pred = model.predict(X_test)
    y_score = None

if 'y_test' in locals() and y_test is not None:
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=0)
    rec = recall_score(y_test, y_pred, pos_label=0)
    f1 = f1_score(y_test, y_pred, pos_label=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    if y_score is not None:
        auc = roc_auc_score((y_test==0).astype(int), y_score)
    else:
        auc = float('nan')
    st.metric('Accuracy', f"{acc:.4f}")
    st.metric('AUC', f"{auc:.4f}" if auc==auc else 'NA')
    st.metric('MCC', f"{mcc:.4f}")
    st.code(classification_report(y_test, y_pred, target_names=list(bunch.target_names)))
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    st.write(pd.DataFrame(cm, index=['true_malignant','true_benign'], columns=['pred_malignant','pred_benign']))
else:
    st.info('Using uploaded data without labels. For full metrics, use demo split by not uploading any file.')
