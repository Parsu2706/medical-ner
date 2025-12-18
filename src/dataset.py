from typing import Dict, List

import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, encodings: Dict[str, List], labels: List[List[int]]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index: int):
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)
