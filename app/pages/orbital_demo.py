"""Orbital Demonstration Page: SGP4 propagation and TLE orbital trajectory visualization.

IMPORTANT: Orbital tools in this module are strictly for ephemeris visualization and demonstration.
Arbitrary TLE input is completely decoupled from ESA ML risk scoring.
"""

import streamlit as st
from app.components.warnings import render_operational_disclaimer


def render() -> None:
    render_operational_disclaimer()
    st.title("Auxiliary Orbital Propagation Demonstration")
    st.info("TLE/SGP4 orbit view active. Standalone propagation decoupled from ML risk scoring.")
