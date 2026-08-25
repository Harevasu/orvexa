"""ORVEXA Streamlit Research Dashboard.

DISCLAIMER: Research estimate only. ORVEXA is not an operational collision-avoidance authority.
"""

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="ORVEXA | Conjunction Risk Prioritization",
        page_icon="🛰️",
        layout="wide",
    )
    st.warning("⚠️ Research estimate only. ORVEXA is not an operational collision-avoidance authority.")
    st.title("🛰️ ORVEXA Research Decision-Support Dashboard")
    st.info("Dashboard scaffolding active. Model artifacts and precomputed predictions pending pipeline execution.")


if __name__ == "__main__":
    main()
