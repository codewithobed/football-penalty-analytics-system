import pandas as pd
import numpy as np
import streamlit as st

def build_prediction_form(df, trained_item):
    """Build a prediction form from model features."""
    feature_columns = trained_item["feature_columns"]
    numeric_features = trained_item["numeric_features"]
    categorical_features = trained_item["categorical_features"]

    input_data = {}
    left, right = st.columns(2)

    for i, col in enumerate(feature_columns):
        container = left if i % 2 == 0 else right

        with container:
            if col in numeric_features:
                values = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([0])
                default = values.median()
                min_value = values.min()
                max_value = values.max()

                default = float(default) if np.isfinite(default) else 0.0
                min_value = float(min_value) if np.isfinite(min_value) else 0.0
                max_value = float(max_value) if np.isfinite(max_value) and max_value != min_value else min_value + 100.0

                input_data[col] = st.number_input(
                    col,
                    min_value=min_value,
                    max_value=max_value,
                    value=default,
                    step=1.0
                )
            elif col in categorical_features:
                options = sorted([str(v) for v in df[col].dropna().unique()]) if col in df.columns else ["Unknown"]
                if not options:
                    options = ["Unknown"]
                input_data[col] = st.selectbox(col, options)
            else:
                input_data[col] = st.text_input(col, "")

    return pd.DataFrame([input_data], columns=feature_columns)

def local_explanation(input_df, trained_item, top_n=10):
    """Model-agnostic local explanation using baseline replacement."""
    pipeline = trained_item["pipeline"]
    X_train = trained_item["X_train"]

    base_probability = float(pipeline.predict_proba(input_df)[0][1])
    impacts = []

    for col in input_df.columns:
        altered = input_df.copy()

        if col in trained_item["numeric_features"]:
            baseline = pd.to_numeric(X_train[col], errors="coerce").median()
            if pd.isna(baseline):
                continue
            altered[col] = baseline
        elif col in trained_item["categorical_features"]:
            mode = X_train[col].mode(dropna=True)
            if mode.empty:
                continue
            altered[col] = mode.iloc[0]
        else:
            continue

        try:
            new_probability = float(pipeline.predict_proba(altered)[0][1])
            impact = base_probability - new_probability
            impacts.append({
                "Feature": col,
                "Current Value": input_df[col].iloc[0],
                "Baseline Comparison Impact": impact
            })
        except Exception:
            continue

    explanation = pd.DataFrame(impacts)

    if explanation.empty:
        return explanation, base_probability

    explanation["Absolute Impact"] = explanation["Baseline Comparison Impact"].abs()
    return explanation.sort_values("Absolute Impact", ascending=False).head(top_n), base_probability

def downloadable_prediction_report(model_name, label, probability_goal, input_df, explanation_df):
    """Create a text prediction report."""
    lines = [
        "Football Penalty Analytics System - Prediction Report",
        f"Selected model: {model_name}",
        f"Predicted outcome: {label}",
        f"Probability of Goal: {probability_goal * 100:.1f}%",
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
