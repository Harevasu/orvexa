"""Unit test suite for ORVEXA package."""

import os
import sys

# Ensure src directory is always available on sys.path for test discovery and direct execution
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
