"""Pytest configuration and shared fixtures for ORVEXA testing."""

from pathlib import Path
import sys
import os

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_esa_path(fixtures_dir: Path) -> Path:
    """Return path to mini ESA fixture dataset."""
    return fixtures_dir / "mini_esa.csv"
