"""Data ingestion, serialization, manifest, and checksum utilities."""

from pathlib import Path
from typing import Any, Dict


def load_raw_data(data_path: str | Path) -> Any:
    """Load raw dataset file (CSV/Parquet).

    Parameters
    ----------
    data_path : str or Path
        Path to the raw dataset file.
    """
    raise NotImplementedError("Data I/O not yet implemented.")


def save_manifest(manifest_path: str | Path, manifest_data: Dict[str, Any]) -> None:
    """Write data or artifact manifest with SHA-256 checksums.

    Parameters
    ----------
    manifest_path : str or Path
        Output path for manifest JSON/YAML.
    manifest_data : dict
        Manifest metadata dictionary.
    """
    raise NotImplementedError("Manifest saving not yet implemented.")
