"""Explanations Page: SHAP feature attributions and TCN temporal importance.

Disclaimer: Feature attributions reflect model mechanisms, not physical causality.
"""

import streamlit as st
from app.components.warnings import render_operational_disclaimer


def render() -> None:
    render_operational_disclaimer()
    st.title("Model Explanations")
    st.info("Attributions indicate model behavior, not physical causality.")
