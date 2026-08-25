"""Ranked Alerts Page: Interactive table of prioritized conjunction events under alert budgets."""

import streamlit as st
from app.components.warnings import render_operational_disclaimer


def render() -> None:
    render_operational_disclaimer()
    st.title("Ranked Conjunction Alerts")
    st.info("Ranked alerts view active. Precomputed prediction artifacts pending.")
