import streamlit as st
import pandas as pd
import plotly.express as px

from src.preprocessing import find_target_column, add_outcome_label


def render_home(df):
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0B6623 0%, #102A43 100%);
            padding: 34px;
            border-radius: 20px;
            color: white;
            margin-bottom: 28px;
        ">
            <h1 style="color:white; margin-bottom: 8px;">⚽ Football Penalty Analytics System</h1>
            <p style="font-size:19px; margin-bottom: 0;">
                A machine learning decision-support platform for football coaches, analysts and performance teams.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df is not None:
        target_col = find_target_column(df)

        if target_col:
            temp = add_outcome_label(df, target_col)
            total_penalties = int(temp["Outcome_Label"].notna().sum())
            goals = int((temp["Outcome_Label"] == "Goal").sum())
            misses = int((temp["Outcome_Label"] == "Miss").sum())
            conversion_rate = (goals / total_penalties * 100) if total_penalties else 0
        else:
            temp = df.copy()
            total_penalties = len(df)
            goals = 0
            misses = 0
            conversion_rate = 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📊 Total Penalties", total_penalties)
        k2.metric("🥅 Goals", goals)
        k3.metric("❌ Misses", misses)
        k4.metric("📈 Conversion Rate", f"{conversion_rate:.1f}%")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📊 Total Penalties", "Upload data")
        k2.metric("🥅 Goals", "-")
        k3.metric("❌ Misses", "-")
        k4.metric("📈 Conversion Rate", "-")

    st.markdown("---")

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("🎯 Project Purpose")
        st.markdown(
            """
            <div class="info-box">
            This software artefact supports football penalty analysis by combining dataset exploration,
            machine learning model training, prediction, explainability and coach-focused reporting.
            It is designed to help users understand not only whether a penalty is likely to be scored,
            but also which match and player factors contribute to that prediction.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("🧭 System Workflow")
        w1, w2 = st.columns(2)
        w1.success("1️⃣ Upload or select penalty dataset")
        w2.success("2️⃣ Explore football-specific patterns")
        w3, w4 = st.columns(2)
        w3.success("3️⃣ Train and compare ML models")
        w4.success("4️⃣ Predict, explain and report outcomes")

    with right:
        st.subheader("✅ Supervisor Requirements Covered")
        st.info("Upload/select football penalty data")
        st.info("View dataset summaries and EDA")
        st.info("Run predictions with confidence")
        st.info("Evaluate models beyond accuracy")
        st.info("Explain important prediction factors")
        st.info("Generate coach-focused PDF reports")

    if df is not None and find_target_column(df):
        st.markdown("---")
        st.subheader("📊 Quick Dataset Snapshot")

        temp = add_outcome_label(df, find_target_column(df))
        outcome_counts = temp["Outcome_Label"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome", "Count"]

        c1, c2 = st.columns(2)

        with c1:
            fig = px.pie(
                outcome_counts,
                names="Outcome",
                values="Count",
                title="Goal vs Miss Distribution",
                hole=0.35,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "Country" in temp.columns:
                country_counts = temp["Country"].astype(str).value_counts().head(8).reset_index()
                country_counts.columns = ["Country", "Penalties"]
                fig = px.bar(
                    country_counts,
                    x="Country",
                    y="Penalties",
                    title="Top Countries in Dataset",
                    text="Penalties",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Country column not available for country-level overview.")

    st.markdown("---")
    st.subheader("🚀 Application Modules")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.success("📊 Dataset Explorer")
        st.write("Inspect rows, columns, missing values and target labels.")

        st.success("📈 Football EDA")
        st.write("Analyse penalty outcomes, kicker foot, goalkeeper side and match context.")

    with m2:
        st.success("🤖 Model Training")
        st.write("Train and compare Logistic Regression, Decision Tree, Random Forest and Gradient Boosting.")

        st.success("📉 Advanced Analytics")
        st.write("Review ROC curves, precision–recall curves and model ranking.")

    with m3:
        st.success("🎯 Prediction")
        st.write("Predict penalty outcome and confidence score.")

        st.success("📄 Coach Report")
        st.write("Generate downloadable PDF and text reports for coaching use.")
