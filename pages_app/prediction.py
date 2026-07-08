import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc

from src.preprocessing import find_target_column, add_outcome_label
from src.eda import safe_col, football_chart
from src.training import train_models
from src.evaluation import get_feature_importance
from src.prediction import build_prediction_form, local_explanation, downloadable_prediction_report

def render_prediction(df):
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
