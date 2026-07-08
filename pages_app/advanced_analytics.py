import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc

from src.preprocessing import find_target_column, add_outcome_label
from src.eda import safe_col, football_chart
from src.training import train_models
from src.evaluation import get_feature_importance
from src.prediction import build_prediction_form, local_explanation, downloadable_prediction_report

def render_advanced_analytics():
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
