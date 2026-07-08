import streamlit as st
import plotly.express as px

def safe_col(df, names):
    """Return the first matching column name."""
    for name in names:
        if name in df.columns:
            return name
    return None

def football_chart(df, x_col, title, color_col=None):
    """Render a football-specific chart."""
    if x_col is None:
        st.info("Required column not found for this chart.")
        return

    temp = df.copy()

    if color_col and color_col in temp.columns:
        fig = px.histogram(
            temp.dropna(subset=[x_col, color_col]),
            x=x_col,
            color=color_col,
            barmode="group",
            title=title,
            text_auto=True
        )
    else:
        counts = temp[x_col].astype(str).value_counts().reset_index()
        counts.columns = [x_col, "Count"]
        fig = px.bar(counts, x=x_col, y="Count", title=title, text="Count")

    fig.update_layout(xaxis_title=x_col, yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
