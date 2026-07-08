import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc

from src.preprocessing import find_target_column, add_outcome_label
from src.eda import safe_col, football_chart
from src.training import train_models
from src.evaluation import get_feature_importance
from src.prediction import build_prediction_form, local_explanation, downloadable_prediction_report

def render_executive_dashboard(df):
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
