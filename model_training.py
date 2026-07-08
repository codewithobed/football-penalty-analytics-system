import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from datetime import datetime
import joblib

from src.preprocessing import find_target_column
from src.training import train_models

SAVED_MODELS_DIR = Path("saved_models")
SAVED_MODELS_DIR.mkdir(exist_ok=True)


def _format_results_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Format model metrics for display."""
    display_df = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    return display_df


def _get_best_model(results_df: pd.DataFrame) -> pd.Series:
    """Select best model using F1-score, then ROC-AUC, then Accuracy."""
    return results_df.sort_values(["F1-score", "ROC-AUC", "Accuracy"], ascending=False).iloc[0]


def _available_model_files():
    """Return saved model bundle files."""
    return sorted(SAVED_MODELS_DIR.glob("*.joblib"), reverse=True)


def render_model_training(df):
    st.title("🤖 Machine Learning Model Training")
    st.write(
        "Train, compare, save and reload machine learning models for football penalty outcome prediction."
    )

    if df is None:
        st.warning("Upload your dataset first.")
        return

    target_col = find_target_column(df)
    if not target_col:
        st.error("No target column detected.")
        return

    tab_train, tab_saved = st.tabs(["Train Models", "Load Saved Model"])

    with tab_train:
        test_size = st.slider("Test set size", 0.15, 0.40, 0.25, 0.05)
        random_state = st.number_input("Random state", value=42, step=1)

        if st.button("Train and Compare Models"):
            with st.spinner("Training models..."):
                results_df, trained_models = train_models(df, target_col, test_size, int(random_state))
                st.session_state["results_df"] = results_df
                st.session_state["trained_models"] = trained_models
                st.session_state["target_col"] = target_col

            st.success("Models trained successfully.")

        if "results_df" in st.session_state:
            results_df = st.session_state["results_df"]
            st.subheader("Model Comparison")
            st.dataframe(_format_results_table(results_df), use_container_width=True)

            best = _get_best_model(results_df)
            st.success(
                f"Recommended model: **{best['Model']}** "
                f"(F1-score: {best['F1-score']:.3f}, "
                f"ROC-AUC: {best['ROC-AUC']:.3f}, "
                f"Accuracy: {best['Accuracy']:.3f})"
            )

            chart_df = results_df.melt(
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
                title="Model Performance Comparison"
            )
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "Download model comparison CSV",
                results_df.to_csv(index=False).encode("utf-8"),
                "model_comparison.csv",
                "text/csv"
            )

            st.markdown("---")
            st.subheader("Save Trained Models")

            default_name = f"penalty_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            save_name = st.text_input("Saved model filename", value=default_name)

            if st.button("Save Current Trained Models"):
                if not save_name.endswith(".joblib"):
                    save_name = save_name + ".joblib"

                save_path = SAVED_MODELS_DIR / save_name

                bundle = {
                    "results_df": st.session_state["results_df"],
                    "trained_models": st.session_state["trained_models"],
                    "target_col": st.session_state.get("target_col", target_col),
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "best_model": best["Model"],
                    "notes": "Football Penalty Analytics System trained model bundle"
                }

                joblib.dump(bundle, save_path)
                st.success(f"Model bundle saved successfully: `{save_path}`")

    with tab_saved:
        st.subheader("Load Saved Model Bundle")

        model_files = _available_model_files()

        if not model_files:
            st.info("No saved model bundles found yet. Train and save a model first.")
        else:
            selected_file = st.selectbox(
                "Select saved model bundle",
                [str(path) for path in model_files]
            )

            if st.button("Load Selected Model Bundle"):
                bundle = joblib.load(selected_file)

                st.session_state["results_df"] = bundle["results_df"]
                st.session_state["trained_models"] = bundle["trained_models"]
                st.session_state["target_col"] = bundle.get("target_col", target_col)

                st.success(f"Loaded saved model bundle: `{selected_file}`")

                if "created_at" in bundle:
                    st.info(f"Created at: {bundle['created_at']}")

                if "best_model" in bundle:
                    st.info(f"Saved best model: {bundle['best_model']}")

            if "results_df" in st.session_state:
                st.subheader("Loaded Model Performance")
                st.dataframe(_format_results_table(st.session_state["results_df"]), use_container_width=True)

                best = _get_best_model(st.session_state["results_df"])
                st.success(
                    f"Current recommended model: **{best['Model']}** "
                    f"(F1-score: {best['F1-score']:.3f}, "
                    f"ROC-AUC: {best['ROC-AUC']:.3f}, "
                    f"Accuracy: {best['Accuracy']:.3f})"
                )
