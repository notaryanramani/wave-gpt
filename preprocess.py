from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from typing import Tuple, List


@dataclass
class PreprocessParams:
    block_size:int = 128
    batch_size:int = 32
    test_split:float = 0.2


params = PreprocessParams()


def train_val_split(tokens, val_split = params.split) -> Tuple[List[int], List[int]]:
    split = int(len(tokens) * (1. - val_split))
    train_set = tokens[:split]
    val_set = tokens[split:]
    return train_set, val_set


class OpenWebText(Dataset):
    def __init__(self, tokens,  block_size = 128):
        self.block_size = block_size
        self.tokens = tokens

    def __len__(self) -> int:
        return len(self.tokens) // self.block_size

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.tokens[idx: idx+self.block_size])
        y = torch.tensor(self.tokens[idx+1: idx+self.block_size+1])
        return x, y

