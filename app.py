
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

st.set_page_config(page_title="Football Penalty Analytics System", page_icon="⚽", layout="wide")

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data
def load_data(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload CSV or Excel.")

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", case=False, regex=True)]
    df = df.loc[:, ~df.columns.astype(str).str.endswith(".1")]
    df.columns = (df.columns.astype(str).str.strip()
                  .str.replace(" ", "_", regex=False)
                  .str.replace("-", "_", regex=False)
                  .str.replace("/", "_", regex=False))
    df = df.loc[:, df.isna().mean() < 0.95]

    for col in df.columns:
        if col.lower() in ["minute", "home_goals", "away_goals", "player_id", "goalkeeper_id", "goalkeer_id"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return df

def find_target_column(df):
    for col in ["Outcome", "outcome", "Result", "result", "Target", "target"]:
        if col in df.columns:
            return col
    return None

def normalise_outcome(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in ["goal", "scored", "score", "success", "successful", "1", "1.0"]:
        return 1
    if text in ["miss", "missed", "saved", "save", "failed", "unsuccessful", "0", "0.0"]:
        return 0
    try:
        numeric = float(text)
        return 1 if numeric >= 0.5 else 0
    except Exception:
        return np.nan

def add_outcome_label(df, target_col):
    df = df.copy()
    df["Outcome_Binary"] = df[target_col].apply(normalise_outcome)
    df["Outcome_Label"] = df["Outcome_Binary"].map({1: "Goal", 0: "Miss"})
    return df

def safe_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None

def football_chart(df, x_col, title, color_col=None):
    if x_col is None:
        st.info("Required column not found for this chart.")
        return
    temp = df.copy()
    if color_col and color_col in temp.columns:
        fig = px.histogram(temp.dropna(subset=[x_col, color_col]), x=x_col, color=color_col,
                           barmode="group", title=title, text_auto=True)
    else:
        counts = temp[x_col].astype(str).value_counts().reset_index()
        counts.columns = [x_col, "Count"]
        fig = px.bar(counts, x=x_col, y="Count", title=title, text="Count")
    fig.update_layout(xaxis_title=x_col, yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

def build_ml_dataset(df, target_col):
    df = add_outcome_label(df, target_col).dropna(subset=["Outcome_Binary"])
    drop_cols = [target_col, "Outcome_Binary", "Outcome_Label", "player_id", "goalkeeper_id", "goalkeer_id"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df["Outcome_Binary"].astype(int)

    dropped = []
    for col in X.select_dtypes(include=["object", "category"]).columns:
        if X[col].nunique(dropna=True) > 60:
            dropped.append(col)
    X = X.drop(columns=dropped, errors="ignore")
    return X, y, dropped

def train_models(df, target_col, test_size=0.25, random_state=42):
    X, y, dropped_cols = build_ml_dataset(df, target_col)
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                     ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", numeric_pipeline, numeric_features),
                                      ("cat", categorical_pipeline, categorical_features)], remainder="drop")

    stratify_y = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=random_state, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state)
    }

    results, trained = [], {}
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            y_proba, auc = None, np.nan

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": auc
        })
        trained[name] = {
            "pipeline": pipe, "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba,
            "X_train": X_train, "X_test": X_test, "feature_columns": X.columns.tolist(),
            "numeric_features": numeric_features, "categorical_features": categorical_features,
            "dropped_cols": dropped_cols
        }
    return pd.DataFrame(results), trained

def get_feature_importance(item):
    pipe, model = item["pipeline"], item["pipeline"].named_steps["model"]
    try:
        names = pipe.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        return None
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return None
    return pd.DataFrame({"Feature": names, "Importance": values}).sort_values("Importance", ascending=False).head(20)

def build_prediction_form(df, item):
    cols, nums, cats = item["feature_columns"], item["numeric_features"], item["categorical_features"]
    input_data = {}
    left, right = st.columns(2)
    for i, col in enumerate(cols):
        with (left if i % 2 == 0 else right):
            if col in nums:
                vals = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([0])
                default, minv, maxv = vals.median(), vals.min(), vals.max()
                default = float(default) if np.isfinite(default) else 0.0
                minv = float(minv) if np.isfinite(minv) else 0.0
                maxv = float(maxv) if np.isfinite(maxv) and maxv != minv else minv + 100.0
                input_data[col] = st.number_input(col, min_value=minv, max_value=maxv, value=default, step=1.0)
            elif col in cats:
                values = sorted([str(v) for v in df[col].dropna().unique()]) if col in df.columns else ["Unknown"]
                input_data[col] = st.selectbox(col, values if values else ["Unknown"])
            else:
                input_data[col] = st.text_input(col, "")
    return pd.DataFrame([input_data], columns=cols)

def local_explanation(input_df, item, top_n=10):
    """Model-agnostic local explanation: changes each feature to baseline and measures probability impact."""
    pipe = item["pipeline"]
    X_train = item["X_train"]
    base_prob = float(pipe.predict_proba(input_df)[0][1])
    impacts = []

    for col in input_df.columns:
        altered = input_df.copy()
        if col in item["numeric_features"]:
            baseline = pd.to_numeric(X_train[col], errors="coerce").median()
            if pd.isna(baseline):
                continue
            altered[col] = baseline
        elif col in item["categorical_features"]:
            mode = X_train[col].mode(dropna=True)
            if mode.empty:
                continue
            altered[col] = mode.iloc[0]
        else:
            continue
        try:
            new_prob = float(pipe.predict_proba(altered)[0][1])
            impact = base_prob - new_prob
            impacts.append({"Feature": col, "Current Value": input_df[col].iloc[0],
                            "Baseline Comparison Impact": impact})
        except Exception:
            pass

    out = pd.DataFrame(impacts)
    if out.empty:
        return out, base_prob
    out["Absolute Impact"] = out["Baseline Comparison Impact"].abs()
    return out.sort_values("Absolute Impact", ascending=False).head(top_n), base_prob

def downloadable_prediction_report(model_name, label, prob, input_df, explanation_df):
    lines = [
        "Football Penalty Analytics System - Prediction Report",
        f"Selected model: {model_name}",
        f"Predicted outcome: {label}",
        f"Probability of Goal: {prob*100:.1f}%" if prob is not None else "Probability of Goal: N/A",
        "",
        "Input values:",
        input_df.to_string(index=False),
        "",
        "Top local explanation factors:",
        explanation_df.to_string(index=False) if explanation_df is not None and not explanation_df.empty else "No explanation available.",
        "",
        "Note: This prediction supports, but does not replace, expert judgement."
    ]
    return "\n".join(lines).encode("utf-8")

# Sidebar
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

# Pages
if page == "Home":
    st.title("⚽ Football Penalty Analytics System")
    st.write("A decision-support dashboard for football coaches and analysts.")
    st.subheader("Version 5.0")
    st.success("Adds local explainability, downloadable reports, and a dedicated Explainability page.")
    st.info("Workflow: upload dataset → train models → evaluate → predict → explain.")

elif page == "Dataset Explorer":
    st.title("📊 Dataset Explorer")
    if df is None:
        st.warning("Upload your dataset first.")
    else:
        target = find_target_column(df)
        labelled = add_outcome_label(df, target) if target else df
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0]); c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isna().sum().sum()))
        c4.metric("Target Column", target if target else "Not detected")
        if target:
            st.subheader("Cleaned Outcome Distribution")
            st.write(labelled["Outcome_Label"].value_counts(dropna=False))
        st.dataframe(labelled.head(20), use_container_width=True)
        info = pd.DataFrame({"Column": labelled.columns,
                             "Data Type": [str(labelled[c].dtype) for c in labelled.columns],
                             "Missing Values": [int(labelled[c].isna().sum()) for c in labelled.columns],
                             "Unique Values": [int(labelled[c].nunique(dropna=True)) for c in labelled.columns]})
        st.dataframe(info, use_container_width=True)

elif page == "Football EDA":
    st.title("📈 Football-Specific Exploratory Data Analysis")
    if df is None:
        st.warning("Upload your dataset first.")
    else:
        target = find_target_column(df)
        if not target:
            st.error("Outcome column not found.")
        else:
            temp = add_outcome_label(df, target)
            total = temp["Outcome_Label"].notna().sum()
            goals = int((temp["Outcome_Label"] == "Goal").sum())
            misses = int((temp["Outcome_Label"] == "Miss").sum())
            rate = goals / total * 100 if total else 0
            a,b,c,d = st.columns(4)
            a.metric("Total Penalties", total); b.metric("Goals", goals); c.metric("Misses", misses); d.metric("Conversion Rate", f"{rate:.1f}%")
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
            with tabs[2]:
                football_chart(temp, safe_col(temp, ["Country"]), "Penalty Records by Country")
                football_chart(temp, safe_col(temp, ["Team_Type"]), "Team Type Distribution")
                if "Minute" in temp.columns:
                    st.plotly_chart(px.histogram(temp, x="Minute", nbins=20, title="Penalty Minute Distribution"), use_container_width=True)
            with tabs[3]:
                nums = temp.select_dtypes(include=[np.number])
                if nums.shape[1] >= 2:
                    st.plotly_chart(px.imshow(nums.corr(), text_auto=True, aspect="auto", title="Correlation Heatmap"), use_container_width=True)

elif page == "Model Training":
    st.title("🤖 Machine Learning Model Training")
    if df is None:
        st.warning("Upload your dataset first.")
    else:
        target = find_target_column(df)
        if not target:
            st.error("No target column detected.")
        else:
            test_size = st.slider("Test set size", 0.15, 0.40, 0.25, 0.05)
            random_state = st.number_input("Random state", value=42, step=1)
            if st.button("Train and Compare Models"):
                with st.spinner("Training models..."):
                    results, trained = train_models(df, target, test_size, int(random_state))
                    st.session_state["results_df"] = results
                    st.session_state["trained_models"] = trained
                st.success("Models trained successfully.")
            if "results_df" in st.session_state:
                results = st.session_state["results_df"]
                display = results.copy()
                for col in ["Accuracy","Precision","Recall","F1-score","ROC-AUC"]:
                    display[col] = display[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
                st.dataframe(display, use_container_width=True)
                best = results.sort_values(["F1-score","ROC-AUC","Accuracy"], ascending=False).iloc[0]
                st.success(f"Recommended model: **{best['Model']}** (F1-score: {best['F1-score']:.3f}, ROC-AUC: {best['ROC-AUC']:.3f}, Accuracy: {best['Accuracy']:.3f})")
                st.download_button("Download model comparison CSV", results.to_csv(index=False).encode("utf-8"), "model_comparison.csv", "text/csv")

elif page == "Model Evaluation":
    st.title("📉 Model Evaluation")
    if "trained_models" not in st.session_state:
        st.warning("Train models first.")
    else:
        trained = st.session_state["trained_models"]
        name = st.selectbox("Select model", list(trained.keys()))
        item = trained[name]
        cm = confusion_matrix(item["y_test"], item["y_pred"])
        cm_df = pd.DataFrame(cm, index=["Actual Miss","Actual Goal"], columns=["Predicted Miss","Predicted Goal"])
        st.dataframe(cm_df, use_container_width=True)
        st.plotly_chart(px.imshow(cm_df, text_auto=True, aspect="auto", title=f"Confusion Matrix - {name}"), use_container_width=True)
        report = pd.DataFrame(classification_report(item["y_test"], item["y_pred"], target_names=["Miss","Goal"], output_dict=True, zero_division=0)).T
        st.dataframe(report, use_container_width=True)
        imp = get_feature_importance(item)
        if imp is not None:
            st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h", title=f"Top Feature Importance - {name}").update_layout(yaxis={"categoryorder":"total ascending"}), use_container_width=True)
            st.dataframe(imp, use_container_width=True)

elif page == "Prediction":
    st.title("🎯 Penalty Outcome Prediction")
    if df is None:
        st.warning("Upload your dataset first.")
    elif "trained_models" not in st.session_state:
        st.warning("Train models first.")
    else:
        results = st.session_state["results_df"]
        trained = st.session_state["trained_models"]
        best_name = results.sort_values(["F1-score","ROC-AUC","Accuracy"], ascending=False).iloc[0]["Model"]
        selected = st.selectbox("Choose trained model", list(trained.keys()), index=list(trained.keys()).index(best_name))
        st.info(f"Recommended model: **{best_name}**")
        item = trained[selected]
        input_df = build_prediction_form(df, item)
        st.subheader("Prediction Input Preview")
        st.dataframe(input_df, use_container_width=True)
        if st.button("Predict Penalty Outcome"):
            pred = int(item["pipeline"].predict(input_df)[0])
            prob = float(item["pipeline"].predict_proba(input_df)[0][1])
            label = "Goal" if pred == 1 else "Miss"
            explanation, base_prob = local_explanation(input_df, item)
            st.session_state["last_prediction"] = {"model": selected, "label": label, "prob": prob, "input": input_df, "explanation": explanation}
            st.markdown("---")
            st.subheader("Prediction Result")
            st.success("Predicted Outcome: GOAL") if label == "Goal" else st.error("Predicted Outcome: MISS")
            st.metric("Confidence: Probability of Goal", f"{prob*100:.1f}%")
            if prob >= .75:
                st.info("Coach Interpretation: The model considers this a high-probability scoring situation.")
            elif prob >= .50:
                st.warning("Coach Interpretation: The model considers this a moderate-probability scoring situation.")
            else:
                st.error("Coach Interpretation: The model considers this a low-probability scoring situation.")
            if not explanation.empty:
                st.subheader("Local Explanation")
                st.write("These features changed the predicted goal probability the most compared with a baseline value.")
                st.dataframe(explanation[["Feature","Current Value","Baseline Comparison Impact"]], use_container_width=True)
                fig = px.bar(explanation, x="Baseline Comparison Impact", y="Feature", orientation="h",
                             title="Local Feature Impact on Probability of Goal")
                fig.update_layout(yaxis={"categoryorder":"total ascending"})
                st.plotly_chart(fig, use_container_width=True)
            report = downloadable_prediction_report(selected, label, prob, input_df, explanation)
            st.download_button("Download prediction report", report, "prediction_report.txt", "text/plain")

elif page == "Explainability":
    st.title("🧠 Explainability")
    if "trained_models" not in st.session_state:
        st.warning("Train models first.")
    else:
        trained = st.session_state["trained_models"]
        name = st.selectbox("Select model for explainability", list(trained.keys()))
        item = trained[name]
        imp = get_feature_importance(item)
        st.subheader("Global Feature Importance")
        if imp is not None:
            st.write("This shows which variables are most influential across the model.")
            st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h", title=f"Global Feature Importance - {name}").update_layout(yaxis={"categoryorder":"total ascending"}), use_container_width=True)
            st.dataframe(imp, use_container_width=True)
            st.download_button("Download feature importance CSV", imp.to_csv(index=False).encode("utf-8"), "feature_importance.csv", "text/csv")
        else:
            st.info("Feature importance is not available for this model.")
        st.subheader("Local Prediction Explanation")
        if "last_prediction" in st.session_state:
            lp = st.session_state["last_prediction"]
            st.write(f"Last prediction: **{lp['label']}**, Probability of Goal: **{lp['prob']*100:.1f}%**")
            if lp["explanation"] is not None and not lp["explanation"].empty:
                st.dataframe(lp["explanation"], use_container_width=True)
        else:
            st.info("Make a prediction first to see local explanation details.")

elif page == "Data Quality Report":
    st.title("🧹 Data Quality Report")
    if df is None:
        st.warning("Upload your dataset first.")
    else:
        target = find_target_column(df)
        temp = add_outcome_label(df, target) if target else df
        missing = temp.isna().sum().reset_index()
        missing.columns = ["Column","Missing Values"]
        missing["Missing Percentage"] = (missing["Missing Values"]/len(temp)*100).round(2)
        st.dataframe(missing.sort_values("Missing Values", ascending=False), use_container_width=True)
        st.metric("Duplicate Rows", int(temp.duplicated().sum()))
        if target:
            st.subheader("Outcome Cleaning Check")
            st.write(temp["Outcome_Label"].value_counts(dropna=False))

elif page == "Coach Insights":
    st.title("📋 Coach Insight Report")
    if df is None:
        st.warning("Upload your dataset first.")
    else:
        target = find_target_column(df)
        if target:
            temp = add_outcome_label(df, target)
            rate = (temp["Outcome_Label"]=="Goal").mean()*100
            st.write(f"The dataset contains **{len(temp)} penalties** with an overall conversion rate of **{rate:.1f}%**.")
            for col in ["Kicker_Foot","Kicker_Side","Goalie_Side","Team_Type","Country"]:
                if col in temp.columns:
                    rates = temp.groupby(col)["Outcome_Label"].apply(lambda x: (x=="Goal").mean()*100).sort_values(ascending=False)
                    if len(rates):
                        st.info(f"Highest observed goal rate by **{col}**: **{rates.index[0]}** ({rates.iloc[0]:.1f}%).")
            if "results_df" in st.session_state:
                best = st.session_state["results_df"].sort_values(["F1-score","ROC-AUC","Accuracy"], ascending=False).iloc[0]
                st.success(f"Best trained model: **{best['Model']}** based on F1-score, ROC-AUC and Accuracy.")

elif page == "Next Development Steps":
    st.title("🚀 Next Development Steps")
    st.markdown("""
    ### Version 5.0 Completed
    - Dedicated Explainability page
    - Global feature importance
    - Local prediction explanation
    - Downloadable model comparison
    - Downloadable feature importance
    - Downloadable prediction report

    ### Final polish
    - Add screenshots to dissertation
    - Add system architecture diagram
    - Add use case diagram
    - Add implementation chapter
    - Create GitHub repository
    """)
