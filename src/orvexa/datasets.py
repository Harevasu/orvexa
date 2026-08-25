"""PyTorch dataset and DataLoader builders for left-padded CDM sequence tensors and masks."""

from typing import Any, Tuple


class CDMSequenceDataset:
    """Dataset producing (N, T, F) tensors and (N, T) boolean validity masks with left padding."""

    def __init__(self, sequences: Any, targets: Any, max_seq_len: int = 23) -> None:
        self.sequences = sequences
        self.targets = targets
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        raise NotImplementedError("Dataset length not yet implemented.")

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        """Returns (sequence_tensor, mask_tensor, target)."""
        raise NotImplementedError("Dataset indexing not yet implemented.")
