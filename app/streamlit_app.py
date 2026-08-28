"""ORVEXA Streamlit Research Dashboard.

DISCLAIMER: Research estimate only. ORVEXA is not an operational collision-avoidance authority.
"""

import sys
from pathlib import Path

# Make the project root importable when Streamlit executes multipage files.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="ORVEXA | Conjunction Risk Prioritization",
        page_icon="🛰️",
        layout="wide",
    )

    st.warning(
        "⚠️ Research estimate only. ORVEXA is not an operational collision-avoidance authority."
    )

    st.title("🛰️ ORVEXA Research Decision-Support Dashboard")
    st.info(
        "Dashboard scaffolding active. Model artifacts and precomputed predictions pending pipeline execution."
    )


if __name__ == "__main__":
    main()