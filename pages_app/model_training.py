import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc

from src.preprocessing import find_target_column, add_outcome_label
from src.eda import safe_col, football_chart
from src.training import train_models
from src.evaluation import get_feature_importance
from src.prediction import build_prediction_form, local_explanation, downloadable_prediction_report

def render_model_training(df):
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
