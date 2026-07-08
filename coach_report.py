import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from src.preprocessing import find_target_column, add_outcome_label


def _build_pdf_report(report_data: dict) -> bytes:
    """Generate a professional PDF coach report."""
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#0B6623"),
        fontSize=20,
        leading=24,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#0B6623"),
        fontSize=14,
        leading=18,
        spaceBefore=10,
        spaceAfter=8,
    )
    normal = styles["BodyText"]

    elements = []

    elements.append(Paragraph("Football Penalty Analytics System", title_style))
    elements.append(Paragraph("Coach Decision-Support Report", styles["Heading3"]))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("1. Dataset Summary", heading_style))
    summary_table = Table([
        ["Metric", "Value"],
        ["Total penalties analysed", str(report_data["total_penalties"])],
        ["Goals scored", str(report_data["goals"])],
        ["Misses / saves", str(report_data["misses"])],
        ["Goal conversion rate", f"{report_data['conversion_rate']:.1f}%"],
    ], colWidths=[8 * cm, 7 * cm])

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B6623")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F3F8F4")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("2. Machine Learning Model Performance", heading_style))
    if report_data["model_rows"]:
        model_table = Table(report_data["model_rows"], repeatRows=1)
        model_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B6623")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(model_table)
    else:
        elements.append(Paragraph("Models have not yet been trained.", normal))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("3. Latest Prediction", heading_style))
    if report_data["prediction"]:
        pred = report_data["prediction"]
        elements.append(Paragraph(f"Predicted outcome: <b>{pred['label']}</b>", normal))
        elements.append(Paragraph(f"Probability of Goal: <b>{pred['probability_goal'] * 100:.1f}%</b>", normal))
    else:
        elements.append(Paragraph("No prediction has been generated yet.", normal))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("4. Explainability Summary", heading_style))
    if report_data["explanation_rows"]:
        exp_table = Table(report_data["explanation_rows"], repeatRows=1)
        exp_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B6623")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(exp_table)
    else:
        elements.append(Paragraph("No local explanation is available yet.", normal))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("5. Coach Recommendations", heading_style))
    for i, rec in enumerate(report_data["recommendations"], 1):
        elements.append(Paragraph(f"{i}. {rec}", normal))
        elements.append(Spacer(1, 5))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph(
        "Note: This report supports coaching and analysis decisions but should not replace expert judgement.",
        styles["Italic"]
    ))

    doc.build(elements)
    return buffer.getvalue()


def render_coach_report(df):
    st.title("📄 Coach Report Generator")
    st.write(
        "Generate a downloadable coach-focused report containing dataset summary, model performance, "
        "latest prediction, explainability summary, and practical recommendations."
    )

    if df is None:
        st.warning("Upload your dataset first.")
        return

    target_col = find_target_column(df)

    if not target_col:
        st.error("No target column detected. The report requires an Outcome column.")
        return

    temp = add_outcome_label(df, target_col)

    total_penalties = int(temp["Outcome_Label"].notna().sum())
    goals = int((temp["Outcome_Label"] == "Goal").sum())
    misses = int((temp["Outcome_Label"] == "Miss").sum())
    conversion_rate = (goals / total_penalties * 100) if total_penalties else 0

    st.subheader("Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Penalties", total_penalties)
    c2.metric("Goals", goals)
    c3.metric("Misses", misses)
    c4.metric("Conversion Rate", f"{conversion_rate:.1f}%")

    st.markdown("---")
    st.subheader("Model Performance Summary")

    model_rows = []
    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"].copy()
        best = results_df.sort_values(["F1-score", "ROC-AUC", "Accuracy"], ascending=False).iloc[0]

        st.success(
            f"Best model: **{best['Model']}** "
            f"(F1-score: {best['F1-score']:.3f}, "
            f"ROC-AUC: {best['ROC-AUC']:.3f}, "
            f"Accuracy: {best['Accuracy']:.3f})"
        )

        display_results = results_df.copy()
        for metric in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]:
            display_results[metric] = display_results[metric].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
            )
        st.dataframe(display_results, use_container_width=True)

        model_rows = [["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]]
        for _, row in results_df.iterrows():
            model_rows.append([
                str(row["Model"]),
                f"{row['Accuracy']:.3f}" if pd.notna(row["Accuracy"]) else "N/A",
                f"{row['Precision']:.3f}" if pd.notna(row["Precision"]) else "N/A",
                f"{row['Recall']:.3f}" if pd.notna(row["Recall"]) else "N/A",
                f"{row['F1-score']:.3f}" if pd.notna(row["F1-score"]) else "N/A",
                f"{row['ROC-AUC']:.3f}" if pd.notna(row["ROC-AUC"]) else "N/A",
            ])
    else:
        st.warning("Train the models first to include model performance in the report.")

    st.markdown("---")
    st.subheader("Latest Prediction Summary")

    prediction = None
    explanation_rows = []

    if "last_prediction" in st.session_state:
        lp = st.session_state["last_prediction"]
        prediction = {
            "label": lp["label"],
            "probability_goal": lp["probability_goal"]
        }

        st.success(f"Latest prediction: **{lp['label']}**")
        st.metric("Probability of Goal", f"{lp['probability_goal'] * 100:.1f}%")

        if lp["explanation"] is not None and not lp["explanation"].empty:
            local_exp = lp["explanation"].copy()
            local_exp["Current Value"] = local_exp["Current Value"].astype(str)

            st.write("Top local explanation factors:")
            st.dataframe(
                local_exp[["Feature", "Current Value", "Baseline Comparison Impact"]].head(10),
                use_container_width=True
            )

            explanation_rows = [["Feature", "Current Value", "Impact"]]
            for _, row in local_exp[["Feature", "Current Value", "Baseline Comparison Impact"]].head(10).iterrows():
                explanation_rows.append([
                    str(row["Feature"]),
                    str(row["Current Value"]),
                    f"{row['Baseline Comparison Impact']:.4f}",
                ])
    else:
        st.info("Make a prediction first to include prediction details in the report.")

    st.markdown("---")
    st.subheader("Coach-Focused Recommendations")

    recommendations = []

    if conversion_rate >= 80:
        recommendations.append(
            "The dataset shows a high overall conversion rate, suggesting that successful penalties dominate the sample."
        )
    elif conversion_rate >= 60:
        recommendations.append(
            "The dataset shows a moderate conversion rate, so coaches should examine contextual factors linked to missed penalties."
        )
    else:
        recommendations.append(
            "The dataset shows a relatively low conversion rate, suggesting a need for technical and tactical review."
        )

    for col in ["Kicker_Foot", "Kicker_Side", "Goalie_Side", "Team_Type", "Country"]:
        if col in temp.columns:
            rates = (
                temp.groupby(col)["Outcome_Label"]
                .apply(lambda x: (x == "Goal").mean() * 100)
                .sort_values(ascending=False)
            )
            if len(rates):
                recommendations.append(
                    f"The strongest observed goal rate by {col} is {rates.index[0]} ({rates.iloc[0]:.1f}%)."
                )

    if "results_df" in st.session_state:
        recommendations.append(
            "Model selection should be justified using F1-score, ROC-AUC and accuracy rather than accuracy alone."
        )

    for rec in recommendations:
        st.info(rec)

    report_data = {
        "total_penalties": total_penalties,
        "goals": goals,
        "misses": misses,
        "conversion_rate": conversion_rate,
        "model_rows": model_rows,
        "prediction": prediction,
        "explanation_rows": explanation_rows,
        "recommendations": recommendations,
    }

    pdf_bytes = _build_pdf_report(report_data)

    st.download_button(
        "Download Coach Report (.pdf)",
        pdf_bytes,
        "coach_report.pdf",
        "application/pdf"
    )

    # Keep text report option for accessibility and quick inspection
    report_text = "\n".join([
        "FOOTBALL PENALTY ANALYTICS SYSTEM",
        "Coach Decision-Support Report",
        "=" * 55,
        "",
        f"Total penalties analysed: {total_penalties}",
        f"Goals scored: {goals}",
        f"Misses / saves: {misses}",
        f"Goal conversion rate: {conversion_rate:.1f}%",
        "",
        "Coach Recommendations:",
        *[f"{i}. {rec}" for i, rec in enumerate(recommendations, 1)]
    ])

    st.download_button(
        "Download Coach Report (.txt)",
        report_text.encode("utf-8"),
        "coach_report.txt",
        "text/plain"
    )
