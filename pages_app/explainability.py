import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc

from src.preprocessing import find_target_column, add_outcome_label
from src.eda import safe_col, football_chart
from src.training import train_models
from src.evaluation import get_feature_importance
from src.prediction import build_prediction_form, local_explanation, downloadable_prediction_report

def render_explainability():
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
