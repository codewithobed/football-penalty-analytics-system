import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc

from src.data_loader import load_data
from src.preprocessing import clean_dataset, find_target_column, add_outcome_label
from src.eda import safe_col, football_chart
from src.training import train_models
from src.evaluation import get_feature_importance
from src.prediction import build_prediction_form, local_explanation, downloadable_prediction_report

st.set_page_config(page_title="Football Penalty Analytics System", page_icon="⚽", layout="wide")

st.sidebar.title("📁 Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your football penalty dataset", type=["csv", "xlsx", "xls"])

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "Home", "Executive Dashboard", "Dataset Explorer", "Football EDA", "Model Training", "Model Evaluation",
    "Advanced Analytics",
    "Prediction", "Explainability", "Coach Report Generator", "Data Quality Report", "Coach Insights", "Next Development Steps"
])

df = None
if uploaded_file is not None:
    try:
        df = clean_dataset(load_data(uploaded_file))
        st.sidebar.success("Dataset loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Could not load dataset: {e}")

if page == "Home":
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

elif page == "Executive Dashboard":
    st.title("🏟️ Executive Coach Dashboard")
    st.write("A coach-focused summary of penalty data, machine learning performance, and prediction outputs.")

    if df is None:
        st.warning("Upload your football penalty dataset from the sidebar to activate the executive dashboard.")
    else:
        target_col = find_target_column(df)

        if not target_col:
            st.error("No target column detected. The dashboard requires an Outcome column.")
        else:
            temp = add_outcome_label(df, target_col)

            total_penalties = int(temp["Outcome_Label"].notna().sum())
            goals = int((temp["Outcome_Label"] == "Goal").sum())
            misses = int((temp["Outcome_Label"] == "Miss").sum())
            conversion_rate = (goals / total_penalties * 100) if total_penalties else 0

            best_model_name = "Train models first"
            best_accuracy = "-"
            best_f1 = "-"

            if "results_df" in st.session_state:
                results_df = st.session_state["results_df"]
                best_row = results_df.sort_values(["F1-score", "ROC-AUC", "Accuracy"], ascending=False).iloc[0]
                best_model_name = best_row["Model"]
                best_accuracy = f"{best_row['Accuracy']:.3f}"
                best_f1 = f"{best_row['F1-score']:.3f}"

            latest_prediction = "No prediction yet"
            latest_confidence = "-"

            if "last_prediction" in st.session_state:
                latest_prediction = st.session_state["last_prediction"]["label"]
                latest_confidence = f"{st.session_state['last_prediction']['probability_goal'] * 100:.1f}%"

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("📊 Total Penalties", total_penalties)
            k2.metric("🥅 Conversion Rate", f"{conversion_rate:.1f}%")
            k3.metric("🤖 Best Model", best_model_name)
            k4.metric("🎯 Latest Prediction", latest_prediction)

            k5, k6, k7, k8 = st.columns(4)
            k5.metric("✅ Goals", goals)
            k6.metric("❌ Misses", misses)
            k7.metric("📈 Best Accuracy", best_accuracy)
            k8.metric("🔥 Best F1-score", best_f1)

            st.markdown("---")

            left_col, right_col = st.columns(2)

            with left_col:
                st.subheader("Goal vs Miss Distribution")
                outcome_counts = temp["Outcome_Label"].value_counts().reset_index()
                outcome_counts.columns = ["Outcome", "Count"]
                fig = px.pie(
                    outcome_counts,
                    names="Outcome",
                    values="Count",
                    title="Penalty Outcome Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

            with right_col:
                st.subheader("Model Performance Summary")
                if "results_df" in st.session_state:
                    chart_df = st.session_state["results_df"].copy()
                    chart_df = chart_df.melt(
                        id_vars="Model",
                        value_vars=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
                        var_name="Metric",
                        value_name="Score"
                    )
                    fig = px.bar(
                        chart_df,
                        x="Model",
                        y="Score",
                        color="Metric",
                        barmode="group",
                        title="Model Comparison Across Metrics"
                    )
                    fig.update_layout(yaxis_range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Train models on the Model Training page to display model comparison.")

            st.markdown("---")

            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Top Coach Insights")
                for col in ["Kicker_Foot", "Kicker_Side", "Goalie_Side", "Team_Type", "Country"]:
                    if col in temp.columns:
                        rates = (
                            temp.groupby(col)["Outcome_Label"]
                            .apply(lambda x: (x == "Goal").mean() * 100)
                            .sort_values(ascending=False)
                        )
                        if len(rates):
                            st.info(
                                f"Highest observed goal rate by **{col}**: "
                                f"**{rates.index[0]}** ({rates.iloc[0]:.1f}%)."
                            )

            with c2:
                st.subheader("Latest Prediction Summary")
                if "last_prediction" in st.session_state:
                    lp = st.session_state["last_prediction"]
                    st.success(f"Prediction: **{lp['label']}**")
                    st.metric("Probability of Goal", f"{lp['probability_goal'] * 100:.1f}%")

                    if lp["explanation"] is not None and not lp["explanation"].empty:
                        local_exp = lp["explanation"].copy()
                        local_exp["Current Value"] = local_exp["Current Value"].astype(str)
                        st.dataframe(
                            local_exp[["Feature", "Current Value", "Baseline Comparison Impact"]].head(5),
                            use_container_width=True
                        )
                else:
                    st.info("Make a prediction on the Prediction page to display latest prediction details.")

            st.markdown("---")
            st.subheader("Coach Recommendation")
            if "results_df" in st.session_state:
                st.success(
                    f"The current best-performing model is **{best_model_name}**. "
                    "This recommendation is based on F1-score first, followed by ROC-AUC and Accuracy, "
                    "which is more reliable than selecting a model using accuracy alone."
                )
            else:
                st.warning(
                    "Train the machine learning models to generate a coach-ready model recommendation."
                )

elif page == "Dataset Explorer":
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

elif page == "Football EDA":
    st.title("📈 Football-Specific Exploratory Data Analysis")
    if df is None:
        st.warning("Upload your dataset first.")
    else:
        target_col = find_target_column(df)
        if not target_col:
            st.error("Outcome column not found.")
        else:
            temp = add_outcome_label(df, target_col)
            total = int(temp["Outcome_Label"].notna().sum())
            goals = int((temp["Outcome_Label"] == "Goal").sum())
            misses = int((temp["Outcome_Label"] == "Miss").sum())
            conversion_rate = goals / total * 100 if total else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Penalties", total)
            c2.metric("Goals", goals)
            c3.metric("Misses", misses)
            c4.metric("Conversion Rate", f"{conversion_rate:.1f}%")

            tabs = st.tabs(["Outcome", "Kicker/Goalkeeper", "Match Context", "Relationships"])

            with tabs[0]:
                counts = temp["Outcome_Label"].value_counts().reset_index()
                counts.columns = ["Outcome", "Count"]
                st.plotly_chart(px.pie(counts, names="Outcome", values="Count", title="Penalty Outcome Distribution"), use_container_width=True)
                football_chart(temp, safe_col(temp, ["Kicker_Side"]), "Kicker Side by Outcome", "Outcome_Label")
                football_chart(temp, safe_col(temp, ["Goalie_Side"]), "Goalkeeper Side by Outcome", "Outcome_Label")

            with tabs[1]:
                football_chart(temp, safe_col(temp, ["Kicker_Foot"]), "Kicker Foot Distribution")
                football_chart(temp, safe_col(temp, ["Kicker_Foot"]), "Kicker Foot by Outcome", "Outcome_Label")

                player_col = safe_col(temp, ["player_name", "Player_Name", "Player", "player"])
                if player_col:
                    top_players = temp[player_col].astype(str).value_counts().head(15).reset_index()
                    top_players.columns = ["Player", "Penalties"]
                    fig = px.bar(top_players, x="Player", y="Penalties", text="Penalties", title="Top 15 Penalty Takers")
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

            with tabs[2]:
                football_chart(temp, safe_col(temp, ["Country"]), "Penalty Records by Country")
                football_chart(temp, safe_col(temp, ["Team_Type"]), "Team Type Distribution")
                if "Minute" in temp.columns:
                    st.plotly_chart(px.histogram(temp, x="Minute", nbins=20, title="Penalty Minute Distribution"), use_container_width=True)
                    st.plotly_chart(px.box(temp, x="Outcome_Label", y="Minute", title="Penalty Minute by Outcome"), use_container_width=True)

            with tabs[3]:
                numeric_df = temp.select_dtypes(include="number")
                if numeric_df.shape[1] >= 2:
                    st.plotly_chart(px.imshow(numeric_df.corr(), text_auto=True, aspect="auto", title="Correlation Heatmap"), use_container_width=True)

elif page == "Model Training":
    st.title("🤖 Machine Learning Model Training")
    if df is None:
        st.warning("Upload your dataset first.")
    else:
        target_col = find_target_column(df)
        if not target_col:
            st.error("No target column detected.")
        else:
            test_size = st.slider("Test set size", 0.15, 0.40, 0.25, 0.05)
            random_state = st.number_input("Random state", value=42, step=1)

            if st.button("Train and Compare Models"):
                with st.spinner("Training models..."):
                    results_df, trained_models = train_models(df, target_col, test_size, int(random_state))
                    st.session_state["results_df"] = results_df
                    st.session_state["trained_models"] = trained_models
                st.success("Models trained successfully.")

            if "results_df" in st.session_state:
                results_df = st.session_state["results_df"]
                display_df = results_df.copy()
                for col in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
                st.dataframe(display_df, use_container_width=True)
                best = results_df.sort_values(["F1-score", "ROC-AUC", "Accuracy"], ascending=False).iloc[0]
                st.success(f"Recommended model: **{best['Model']}** (F1-score: {best['F1-score']:.3f}, ROC-AUC: {best['ROC-AUC']:.3f}, Accuracy: {best['Accuracy']:.3f})")
                st.download_button("Download model comparison CSV", results_df.to_csv(index=False).encode("utf-8"), "model_comparison.csv", "text/csv")

elif page == "Model Evaluation":
    st.title("📉 Model Evaluation")
    if "trained_models" not in st.session_state:
        st.warning("Train models first.")
    else:
        trained_models = st.session_state["trained_models"]
        model_name = st.selectbox("Select model", list(trained_models.keys()))
        item = trained_models[model_name]

        cm = confusion_matrix(item["y_test"], item["y_pred"])
        cm_df = pd.DataFrame(cm, index=["Actual Miss", "Actual Goal"], columns=["Predicted Miss", "Predicted Goal"])
        st.dataframe(cm_df, use_container_width=True)
        st.plotly_chart(px.imshow(cm_df, text_auto=True, aspect="auto", title=f"Confusion Matrix - {model_name}"), use_container_width=True)

        report = pd.DataFrame(classification_report(item["y_test"], item["y_pred"], target_names=["Miss", "Goal"], output_dict=True, zero_division=0)).T
        st.dataframe(report, use_container_width=True)

        importance = get_feature_importance(item)
        if importance is not None:
            fig = px.bar(importance, x="Importance", y="Feature", orientation="h", title=f"Top Feature Importance - {model_name}")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

elif page == "Advanced Analytics":
    st.title("📈 Advanced Model Analytics")
    st.write(
        "This page provides deeper model evaluation using ROC curves, Precision–Recall curves, "
        "model ranking and coach-friendly interpretation."
    )

    if "trained_models" not in st.session_state or "results_df" not in st.session_state:
        st.warning("Train the models first on the Model Training page.")
    else:
        results_df = st.session_state["results_df"].copy()
        trained_models = st.session_state["trained_models"]

        st.subheader("🏆 Model Ranking")
        ranking_df = results_df.sort_values(
            ["F1-score", "ROC-AUC", "Accuracy"],
            ascending=False
        ).reset_index(drop=True)
        ranking_df.insert(0, "Rank", ranking_df.index + 1)

        display_ranking = ranking_df.copy()
        for metric in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]:
            display_ranking[metric] = display_ranking[metric].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
            )
        st.dataframe(display_ranking, use_container_width=True)

        best = ranking_df.iloc[0]
        st.success(
            f"Best overall model: **{best['Model']}**. "
            "The ranking prioritises F1-score, then ROC-AUC, then accuracy."
        )

        st.markdown("---")
        st.subheader("📊 Metric Comparison")

        metric_chart_df = results_df.melt(
            id_vars="Model",
            value_vars=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
            var_name="Metric",
            value_name="Score"
        )

        fig = px.bar(
            metric_chart_df,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Model Performance Across Evaluation Metrics"
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        selected_model = st.selectbox(
            "Select model for advanced curve analysis",
            list(trained_models.keys()),
            index=list(trained_models.keys()).index(best["Model"])
        )

        item = trained_models[selected_model]
        y_test = item["y_test"]
        y_proba = item["y_proba"]

        if y_proba is None:
            st.warning("Probability scores are not available for this model.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                roc_df = pd.DataFrame({
                    "False Positive Rate": fpr,
                    "True Positive Rate": tpr
                })

                fig = px.line(
                    roc_df,
                    x="False Positive Rate",
                    y="True Positive Rate",
                    title=f"ROC Curve - {selected_model} (AUC = {roc_auc:.3f})"
                )
                fig.add_shape(
                    type="line",
                    x0=0, y0=0, x1=1, y1=1,
                    line=dict(dash="dash")
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Precision–Recall Curve")
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                pr_auc = auc(recall, precision)
                pr_df = pd.DataFrame({
                    "Recall": recall,
                    "Precision": precision
                })

                fig = px.line(
                    pr_df,
                    x="Recall",
                    y="Precision",
                    title=f"Precision–Recall Curve - {selected_model} (AUC = {pr_auc:.3f})"
                )
                fig.update_layout(yaxis_range=[0, 1], xaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Coach-Friendly Interpretation")

            if roc_auc >= 0.75:
                st.success(
                    f"The ROC-AUC of {roc_auc:.3f} suggests that the model has strong ability "
                    "to distinguish between goals and misses."
                )
            elif roc_auc >= 0.60:
                st.warning(
                    f"The ROC-AUC of {roc_auc:.3f} suggests moderate discrimination. "
                    "The model may be useful as decision support but should not be used alone."
                )
            else:
                st.error(
                    f"The ROC-AUC of {roc_auc:.3f} suggests weak discrimination. "
                    "More data, better features or model tuning may be needed."
                )

            if best["Model"] == selected_model:
                st.info(
                    "This model is currently recommended because it performs best using the combined "
                    "ranking logic: F1-score first, then ROC-AUC and Accuracy."
                )

        st.download_button(
            "Download advanced model ranking CSV",
            ranking_df.to_csv(index=False).encode("utf-8"),
            "advanced_model_ranking.csv",
            "text/csv"
        )


elif page == "Prediction":
    st.title("🎯 Penalty Outcome Prediction")
    if df is None:
        st.warning("Upload your dataset first.")
    elif "trained_models" not in st.session_state:
        st.warning("Train models first.")
    else:
        results_df = st.session_state["results_df"]
        trained_models = st.session_state["trained_models"]
        best_model = results_df.sort_values(["F1-score", "ROC-AUC", "Accuracy"], ascending=False).iloc[0]["Model"]
        selected_model = st.selectbox("Choose trained model", list(trained_models.keys()), index=list(trained_models.keys()).index(best_model))
        st.info(f"Recommended model: **{best_model}**")

        item = trained_models[selected_model]
        input_df = build_prediction_form(df, item)
        st.dataframe(input_df, use_container_width=True)

        if st.button("Predict Penalty Outcome"):
            prediction = int(item["pipeline"].predict(input_df)[0])
            probability_goal = float(item["pipeline"].predict_proba(input_df)[0][1])
            label = "Goal" if prediction == 1 else "Miss"
            explanation_df, _ = local_explanation(input_df, item)
            st.session_state["last_prediction"] = {
                "model": selected_model,
                "label": label,
                "probability_goal": probability_goal,
                "input": input_df,
                "explanation": explanation_df
            }

            if "prediction_history" not in st.session_state:
                st.session_state["prediction_history"] = []

            st.session_state["prediction_history"].append({
                "Model": selected_model,
                "Predicted Outcome": label,
                "Probability of Goal": probability_goal,
                "Confidence (%)": round(probability_goal * 100, 1)
            })

            if label == "Goal":
                st.success("Predicted Outcome: GOAL")
            else:
                st.error("Predicted Outcome: MISS")

            st.metric("Confidence: Probability of Goal", f"{probability_goal * 100:.1f}%")

            if probability_goal >= 0.75:
                st.info("Coach Interpretation: The model considers this a high-probability scoring situation.")
            elif probability_goal >= 0.50:
                st.warning("Coach Interpretation: The model considers this a moderate-probability scoring situation.")
            else:
                st.error("Coach Interpretation: The model considers this a low-probability scoring situation.")
            if not explanation_df.empty:
                st.subheader("Local Explanation")

                display_explanation = explanation_df[
                    ["Feature", "Current Value", "Baseline Comparison Impact"]
                ].copy()

                display_explanation["Current Value"] = (
                    display_explanation["Current Value"].astype(str)
                )

                st.dataframe(display_explanation, use_container_width=True)

                fig = px.bar(
                    explanation_df,
                    x="Baseline Comparison Impact",
                    y="Feature",
                    orientation="h",
                    title="Local Feature Impact on Probability of Goal"
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

            report = downloadable_prediction_report(selected_model, label, probability_goal, input_df, explanation_df)
            st.download_button("Download prediction report", report, "prediction_report.txt", "text/plain")

            if "prediction_history" in st.session_state and st.session_state["prediction_history"]:
                st.subheader("Prediction History")
                history_df = pd.DataFrame(st.session_state["prediction_history"])
                st.dataframe(history_df, use_container_width=True)
                st.download_button(
                    "Download prediction history CSV",
                    history_df.to_csv(index=False).encode("utf-8"),
                    "prediction_history.csv",
                    "text/csv"
                )

elif page == "Explainability":
    st.title("🧠 Explainability")
    if "trained_models" not in st.session_state:
        st.warning("Train models first.")
    else:
        trained_models = st.session_state["trained_models"]
        model_name = st.selectbox("Select model for explainability", list(trained_models.keys()))
        item = trained_models[model_name]

        importance = get_feature_importance(item)
        if importance is not None:
            fig = px.bar(importance, x="Importance", y="Feature", orientation="h", title=f"Global Feature Importance - {model_name}")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(importance, use_container_width=True)
            st.download_button("Download feature importance CSV", importance.to_csv(index=False).encode("utf-8"), "feature_importance.csv", "text/csv")

        st.subheader("Local Prediction Explanation")
        if "last_prediction" in st.session_state:
            lp = st.session_state["last_prediction"]
            st.write(f"Last prediction: **{lp['label']}**, Probability of Goal: **{lp['probability_goal'] * 100:.1f}%**")
            if lp["explanation"] is not None and not lp["explanation"].empty:
                st.dataframe(lp["explanation"], use_container_width=True)
        else:
            st.info("Make a prediction first to see local explanation details.")

elif page == "Coach Report Generator":
    st.title("📄 Coach Report Generator")
    st.write(
        "Generate a downloadable coach-focused report containing dataset summary, model performance, "
        "latest prediction, explainability summary, and practical recommendations."
    )

    if df is None:
        st.warning("Upload your dataset first.")
    else:
        target_col = find_target_column(df)

        if not target_col:
            st.error("No target column detected. The report requires an Outcome column.")
        else:
            temp = add_outcome_label(df, target_col)

            total_penalties = int(temp["Outcome_Label"].notna().sum())
            goals = int((temp["Outcome_Label"] == "Goal").sum())
            misses = int((temp["Outcome_Label"] == "Miss").sum())
            conversion_rate = (goals / total_penalties * 100) if total_penalties else 0

            report_lines = []
            report_lines.append("FOOTBALL PENALTY ANALYTICS SYSTEM")
            report_lines.append("Coach Decision-Support Report")
            report_lines.append("=" * 55)
            report_lines.append("")
            report_lines.append("1. DATASET SUMMARY")
            report_lines.append(f"Total penalties analysed: {total_penalties}")
            report_lines.append(f"Goals scored: {goals}")
            report_lines.append(f"Misses / saves: {misses}")
            report_lines.append(f"Goal conversion rate: {conversion_rate:.1f}%")
            report_lines.append("")

            st.subheader("Dataset Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Penalties", total_penalties)
            c2.metric("Goals", goals)
            c3.metric("Misses", misses)
            c4.metric("Conversion Rate", f"{conversion_rate:.1f}%")

            st.markdown("---")
            st.subheader("Model Performance Summary")

            if "results_df" in st.session_state:
                results_df = st.session_state["results_df"].copy()
                best = results_df.sort_values(["F1-score", "ROC-AUC", "Accuracy"], ascending=False).iloc[0]

                st.success(
                    f"Best model: **{best['Model']}** "
                    f"(F1-score: {best['F1-score']:.3f}, "
                    f"ROC-AUC: {best['ROC-AUC']:.3f}, "
                    f"Accuracy: {best['Accuracy']:.3f})"
                )

                display_results = results_df.copy()
                for metric in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]:
                    display_results[metric] = display_results[metric].apply(
                        lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
                    )
                st.dataframe(display_results, use_container_width=True)

                report_lines.append("2. MACHINE LEARNING MODEL PERFORMANCE")
                report_lines.append(f"Recommended model: {best['Model']}")
                report_lines.append(f"Accuracy: {best['Accuracy']:.3f}")
                report_lines.append(f"Precision: {best['Precision']:.3f}")
                report_lines.append(f"Recall: {best['Recall']:.3f}")
                report_lines.append(f"F1-score: {best['F1-score']:.3f}")
                report_lines.append(f"ROC-AUC: {best['ROC-AUC']:.3f}")
                report_lines.append("")
                report_lines.append("Model comparison table:")
                report_lines.append(results_df.to_string(index=False))
                report_lines.append("")
            else:
                st.warning("Train the models first to include model performance in the report.")
                report_lines.append("2. MACHINE LEARNING MODEL PERFORMANCE")
                report_lines.append("Models have not yet been trained.")
                report_lines.append("")

            st.markdown("---")
            st.subheader("Latest Prediction Summary")

            if "last_prediction" in st.session_state:
                lp = st.session_state["last_prediction"]
                st.success(f"Latest prediction: **{lp['label']}**")
                st.metric("Probability of Goal", f"{lp['probability_goal'] * 100:.1f}%")

                report_lines.append("3. LATEST PREDICTION")
                report_lines.append(f"Predicted outcome: {lp['label']}")
                report_lines.append(f"Probability of Goal: {lp['probability_goal'] * 100:.1f}%")
                report_lines.append("")

                if lp["explanation"] is not None and not lp["explanation"].empty:
                    local_exp = lp["explanation"].copy()
                    local_exp["Current Value"] = local_exp["Current Value"].astype(str)
                    st.write("Top local explanation factors:")
                    st.dataframe(
                        local_exp[["Feature", "Current Value", "Baseline Comparison Impact"]].head(10),
                        use_container_width=True
                    )

                    report_lines.append("4. LOCAL EXPLAINABILITY SUMMARY")
                    report_lines.append(
                        local_exp[
                            ["Feature", "Current Value", "Baseline Comparison Impact"]
                        ].head(10).to_string(index=False)
                    )
                    report_lines.append("")
            else:
                st.info("Make a prediction first to include prediction details in the report.")
                report_lines.append("3. LATEST PREDICTION")
                report_lines.append("No prediction has been generated yet.")
                report_lines.append("")

            st.markdown("---")
            st.subheader("Coach-Focused Recommendations")

            recommendations = []

            if conversion_rate >= 80:
                recommendations.append(
                    "The dataset shows a high overall conversion rate, suggesting that successful penalties dominate the sample."
                )
            elif conversion_rate >= 60:
                recommendations.append(
                    "The dataset shows a moderate conversion rate, so coaches should examine contextual factors linked to missed penalties."
                )
            else:
                recommendations.append(
                    "The dataset shows a relatively low conversion rate, suggesting a need for technical and tactical review."
                )

            for col in ["Kicker_Foot", "Kicker_Side", "Goalie_Side", "Team_Type", "Country"]:
                if col in temp.columns:
                    rates = (
                        temp.groupby(col)["Outcome_Label"]
                        .apply(lambda x: (x == "Goal").mean() * 100)
                        .sort_values(ascending=False)
                    )
                    if len(rates):
                        recommendations.append(
                            f"The strongest observed goal rate by {col} is {rates.index[0]} ({rates.iloc[0]:.1f}%)."
                        )

            if "results_df" in st.session_state:
                recommendations.append(
                    "Model selection should be justified using F1-score, ROC-AUC and accuracy rather than accuracy alone."
                )

            for rec in recommendations:
                st.info(rec)

            report_lines.append("5. COACH RECOMMENDATIONS")
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")

            report_lines.append("")
            report_lines.append("Note: This report is intended to support coaching and analysis decisions, not replace expert judgement.")

            report_text = "\n".join(report_lines)

            st.download_button(
                "Download Coach Report (.txt)",
                report_text.encode("utf-8"),
                "coach_report.txt",
                "text/plain"
            )


elif page == "Data Quality Report":
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

elif page == "Coach Insights":
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

elif page == "Next Development Steps":
    st.title("🚀 Next Development Steps")
    st.markdown("""
    ### Refactored version completed
    - `app.py` controls the Streamlit interface.
    - `src/data_loader.py` handles data loading.
    - `src/preprocessing.py` handles cleaning and outcome processing.
    - `src/eda.py` handles EDA helpers.
    - `src/training.py` handles model training.
    - `src/evaluation.py` handles feature importance.
    - `src/prediction.py` handles prediction and local explanation.

    ### Next steps
    - Add screenshots to README.
    - Add system design diagrams.
    - Write methodology and implementation chapters.
    """)
