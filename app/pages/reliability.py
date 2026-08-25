"""Reliability Page: Probability calibration curves, Brier scores, and ECE analysis."""

import streamlit as st
from app.components.warnings import render_operational_disclaimer


def render() -> None:
    render_operational_disclaimer()
    st.title("Probability Calibration and Reliability")
    st.info("Calibration metrics view active. Calibration artifacts pending.")
