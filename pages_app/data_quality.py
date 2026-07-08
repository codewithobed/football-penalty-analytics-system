import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc

from src.preprocessing import find_target_column, add_outcome_label
from src.eda import safe_col, football_chart
from src.training import train_models
from src.evaluation import get_feature_importance
from src.prediction import build_prediction_form, local_explanation, downloadable_prediction_report

def render_data_quality(df):
    st.title("🧹 Data Quality Report")
    if df is None:
        st.warning("Upload your dataset first.")
    else:
        target_col = find_target_column(df)
        temp = add_outcome_label(df, target_col) if target_col else df.copy()
        missing = temp.isna().sum().reset_index()
        missing.columns = ["Column", "Missing Values"]
        missing["Missing Percentage"] = (missing["Missing Values"] / len(temp) * 100).round(2)
        st.dataframe(missing.sort_values("Missing Values", ascending=False), use_container_width=True)
        st.metric("Duplicate Rows", int(temp.duplicated().sum()))
        if target_col:
            st.subheader("Outcome Cleaning Check")
            st.write(temp["Outcome_Label"].value_counts(dropna=False))
