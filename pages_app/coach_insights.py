import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc

from src.preprocessing import find_target_column, add_outcome_label
from src.eda import safe_col, football_chart
from src.training import train_models
from src.evaluation import get_feature_importance
from src.prediction import build_prediction_form, local_explanation, downloadable_prediction_report

def render_coach_insights(df):
    st.title("📋 Coach Insight Report")
    if df is None:
        st.warning("Upload your dataset first.")
    else:
        target_col = find_target_column(df)
        if target_col:
            temp = add_outcome_label(df, target_col)
            goal_rate = (temp["Outcome_Label"] == "Goal").mean() * 100
            st.write(f"The dataset contains **{len(temp)} penalties** with an overall conversion rate of **{goal_rate:.1f}%**.")
            for col in ["Kicker_Foot", "Kicker_Side", "Goalie_Side", "Team_Type", "Country"]:
                if col in temp.columns:
                    rates = temp.groupby(col)["Outcome_Label"].apply(lambda x: (x == "Goal").mean() * 100).sort_values(ascending=False)
                    if len(rates):
                        st.info(f"Highest observed goal rate by **{col}**: **{rates.index[0]}** ({rates.iloc[0]:.1f}%).")
            if "results_df" in st.session_state:
                best = st.session_state["results_df"].sort_values(["F1-score", "ROC-AUC", "Accuracy"], ascending=False).iloc[0]
                st.success(f"Best trained model: **{best['Model']}** based on F1-score, ROC-AUC and Accuracy.")
