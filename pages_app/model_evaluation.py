import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc

from src.preprocessing import find_target_column, add_outcome_label
from src.eda import safe_col, football_chart
from src.training import train_models
from src.evaluation import get_feature_importance
from src.prediction import build_prediction_form, local_explanation, downloadable_prediction_report

def render_model_evaluation():
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
