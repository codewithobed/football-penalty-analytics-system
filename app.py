import streamlit as st

from src.data_loader import load_data
from src.preprocessing import clean_dataset
from src.ui_styles import apply_global_styles, footer
from pages_app.home import render_home
from pages_app.executive_dashboard import render_executive_dashboard
from pages_app.dataset_explorer import render_dataset_explorer
from pages_app.football_eda import render_football_eda
from pages_app.model_training import render_model_training
from pages_app.model_evaluation import render_model_evaluation
from pages_app.advanced_analytics import render_advanced_analytics
from pages_app.prediction import render_prediction
from pages_app.explainability import render_explainability
from pages_app.coach_report import render_coach_report
from pages_app.data_quality import render_data_quality
from pages_app.coach_insights import render_coach_insights
from pages_app.next_steps import render_next_steps

st.set_page_config(page_title="Football Penalty Analytics System", page_icon="⚽", layout="wide")
apply_global_styles()

st.sidebar.title("📁 Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your football penalty dataset", type=["csv", "xlsx", "xls"])

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "Home",
    "Executive Dashboard",
    "Dataset Explorer",
    "Football EDA",
    "Model Training",
    "Model Evaluation",
    "Advanced Analytics",
    "Prediction",
    "Explainability",
    "Coach Report Generator",
    "Data Quality Report",
    "Coach Insights",
    "Next Development Steps"
])

df = None
if uploaded_file is not None:
    try:
        df = clean_dataset(load_data(uploaded_file))
        st.sidebar.success("Dataset loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Could not load dataset: {e}")

PAGE_RENDERERS = {
    "Home": lambda: render_home(df),
    "Executive Dashboard": lambda: render_executive_dashboard(df),
    "Dataset Explorer": lambda: render_dataset_explorer(df),
    "Football EDA": lambda: render_football_eda(df),
    "Model Training": lambda: render_model_training(df),
    "Model Evaluation": render_model_evaluation,
    "Advanced Analytics": render_advanced_analytics,
    "Prediction": lambda: render_prediction(df),
    "Explainability": render_explainability,
    "Coach Report Generator": lambda: render_coach_report(df),
    "Data Quality Report": lambda: render_data_quality(df),
    "Coach Insights": lambda: render_coach_insights(df),
    "Next Development Steps": lambda: render_next_steps(df),
}

PAGE_RENDERERS[page]()
footer()