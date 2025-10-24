import json
from typing import List, Dict, Any
from torch.utils.data import Dataset


class FineTuningDataset(Dataset):
    """Dataset wrapper for JSONL formatted fine-tuning data."""

    def __init__(self, file_path: str):
        self.data: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]