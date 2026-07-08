import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc

from src.preprocessing import find_target_column, add_outcome_label
from src.eda import safe_col, football_chart
from src.training import train_models
from src.evaluation import get_feature_importance
from src.prediction import build_prediction_form, local_explanation, downloadable_prediction_report

def render_coach_report(df):
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
