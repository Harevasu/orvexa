"""Overview Page: Dataset summary, project architecture, and KPI headline metrics."""

import streamlit as st
from app.components.warnings import render_operational_disclaimer


def render() -> None:
    render_operational_disclaimer()
    st.title("Project Overview")
    st.info("Overview page active. Data loading pending.")
