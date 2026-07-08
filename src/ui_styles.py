import streamlit as st


def apply_global_styles():
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3 {
            color: #102A43;
            font-weight: 800;
        }

        .stMetric {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            padding: 14px;
            border-radius: 14px;
            box-shadow: 0 2px 6px rgba(15,23,42,.05);
        }

        .info-box {
            background-color: #F3F8F4;
            padding: 20px;
            border-radius: 14px;
            border-left: 7px solid #0B6623;
            margin-bottom: 20px;
        }

        .app-footer {
            margin-top: 50px;
            padding-top: 15px;
            border-top: 1px solid #E2E8F0;
            text-align: center;
            color: #64748B;
            font-size: 13px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        """
        <div class="app-footer">
        Football Penalty Analytics System | MSc Data Science Project | University of Roehampton
        </div>
        """,
        unsafe_allow_html=True,
    )