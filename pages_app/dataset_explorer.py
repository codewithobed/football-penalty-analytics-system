import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc

from src.preprocessing import find_target_column, add_outcome_label
from src.eda import safe_col, football_chart
from src.training import train_models
from src.evaluation import get_feature_importance
from src.prediction import build_prediction_form, local_explanation, downloadable_prediction_report

def render_dataset_explorer(df):
    st.title("📊 Dataset Explorer")
    if df is None:
        st.warning("Upload your dataset first.")
    else:
        target_col = find_target_column(df)
        labelled_df = add_outcome_label(df, target_col) if target_col else df.copy()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isna().sum().sum()))
        c4.metric("Target Column", target_col if target_col else "Not detected")

        if target_col:
            st.subheader("Cleaned Outcome Distribution")
            st.write(labelled_df["Outcome_Label"].value_counts(dropna=False))

        st.subheader("Preview of Cleaned Dataset")
        st.dataframe(labelled_df.head(20), use_container_width=True)

        st.subheader("Data Types and Missing Values")
        info = pd.DataFrame({
            "Column": labelled_df.columns,
            "Data Type": [str(labelled_df[col].dtype) for col in labelled_df.columns],
            "Missing Values": [int(labelled_df[col].isna().sum()) for col in labelled_df.columns],
            "Unique Values": [int(labelled_df[col].nunique(dropna=True)) for col in labelled_df.columns]
        })
        st.dataframe(info, use_container_width=True)
