import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc

from src.preprocessing import find_target_column, add_outcome_label
from src.eda import safe_col, football_chart
from src.training import train_models
from src.evaluation import get_feature_importance
from src.prediction import build_prediction_form, local_explanation, downloadable_prediction_report

def render_home(df):
    st.markdown("""
    <style>
    .main-title {
        font-size: 44px;
        font-weight: 800;
        color: #0B6623;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 20px;
        color: #555;
        margin-bottom: 25px;
    }
    .info-box {
        background-color: #F3F8F4;
        padding: 22px;
        border-radius: 14px;
        border-left: 7px solid #0B6623;
        margin-bottom: 25px;
        font-size: 16px;
        line-height: 1.6;
    }
    .section-card {
        background-color: #FFFFFF;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #E8E8E8;
        margin-bottom: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">⚽ Football Penalty Analytics System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Machine learning decision-support dashboard for football coaches and analysts</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">
    This system allows coaches and sports analysts to upload football penalty data, explore performance patterns,
    train machine learning models, predict penalty outcomes, and interpret model decisions using explainability tools.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    if df is not None:
        target_col = find_target_column(df)

        if target_col:
            temp = add_outcome_label(df, target_col)
            total_penalties = int(temp["Outcome_Label"].notna().sum())
            goals = int((temp["Outcome_Label"] == "Goal").sum())
            misses = int((temp["Outcome_Label"] == "Miss").sum())
            conversion_rate = (goals / total_penalties * 100) if total_penalties else 0
        else:
            total_penalties = len(df)
            goals = 0
            misses = 0
            conversion_rate = 0

        col1.metric("Total Penalties", total_penalties)
        col2.metric("Goals", goals)
        col3.metric("Misses", misses)
        col4.metric("Conversion Rate", f"{conversion_rate:.1f}%")
    else:
        col1.metric("Total Penalties", "Upload data")
        col2.metric("Goals", "-")
        col3.metric("Misses", "-")
        col4.metric("Conversion Rate", "-")

    st.markdown("---")

    st.subheader("System Workflow")
    w1, w2, w3, w4 = st.columns(4)
    w1.info("1️⃣ Upload football penalty dataset")
    w2.info("2️⃣ Explore data and visualisations")
    w3.info("3️⃣ Train and compare ML models")
    w4.info("4️⃣ Predict and explain outcomes")

    st.subheader("Core Capabilities")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.success("📊 Dataset exploration")
        st.success("📈 Football-specific EDA")
        st.success("🧹 Data quality checks")

    with c2:
        st.success("🤖 ML model training")
        st.success("📉 Model evaluation")
        st.success("🏆 Best model selection")

    with c3:
        st.success("🎯 Outcome prediction")
        st.success("🧠 Explainability")
        st.success("📋 Coach insights")
