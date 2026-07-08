import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc

from src.preprocessing import find_target_column, add_outcome_label
from src.eda import safe_col, football_chart
from src.training import train_models
from src.evaluation import get_feature_importance
from src.prediction import build_prediction_form, local_explanation, downloadable_prediction_report

def render_football_eda(df):
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
