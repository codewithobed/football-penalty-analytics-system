import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report

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
    "Home", "Dataset Explorer", "Football EDA", "Model Training", "Model Evaluation",
    "Prediction", "Explainability", "Data Quality Report", "Coach Insights", "Next Development Steps"
])

df = None
if uploaded_file is not None:
    try:
        df = clean_dataset(load_data(uploaded_file))
        st.sidebar.success("Dataset loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Could not load dataset: {e}")

if page == "Home":
    st.title("⚽ Football Penalty Analytics System")
    st.write("A decision-support dashboard for football coaches and analysts.")
    st.info("Workflow: upload dataset → explore data → train models → evaluate → predict → explain.")

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
