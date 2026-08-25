"""Mandatory research limitation and non-operational disclaimers."""

import streamlit as st

DISCLAIMER_TEXT = "Research estimate only. ORVEXA is not an operational collision-avoidance authority."


def render_operational_disclaimer() -> None:
    """Render mandatory operational disclaimer across all dashboard pages."""
    st.warning(f"⚠️ **{DISCLAIMER_TEXT}**")
