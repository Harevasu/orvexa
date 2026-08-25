"""Horizon Comparison Page: Multi-horizon (2, 3, 5, 7 day) performance and usable event coverage."""

import streamlit as st
from app.components.warnings import render_operational_disclaimer


def render() -> None:
    render_operational_disclaimer()
    st.title("Warning Horizon Comparison")
    st.info("Multi-horizon sweep view active. Horizon evaluation artifacts pending.")
