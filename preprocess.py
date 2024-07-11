from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from typing import Tuple, List
import os
import tiktoken
import mmap


@dataclass
class PreprocessParams:
    block_size:int = 128
    batch_size:int = 32
    split:float = 0.2


params = PreprocessParams()


def train_val_split(tokens, val_split:float = params.split) -> Tuple[List[int], List[int]]:
    split = int(len(tokens) * (1. - val_split))
    train_set = tokens[:split]
    val_set = tokens[split:]
    return train_set, val_set


class OpenWebText(Dataset):
    def __init__(self, text_file, split, block_size = 128, chunk_size = 50000):
        self.block_size = block_size
        self.file_path = text_file
        self.split = split
        self.tokenizer = tiktoken.get_encoding('r50k_base')
        self.file_size = os.path.getsize(text_file)
        self.chunk_size = chunk_size
        if self.split == 'train':
            self.random_start_chunk = torch.randint(0, int(0.8 * self.file_size) - self.chunk_size, (1,)).item()
        else:
            self.random_start_chunk = torch.randint(int(0.8 * self.file_size) , self.file_size - self.chunk_size, (1,)).item()

    def _get_chuck(self):
        with open(self.file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                mm.seek(self.random_start_chunk)
                block = mm.read(self.chunk_size)
                self.text = block.decode('utf-8', errors='ignore').replace('\r', '')
        return self.text
            

    def __len__(self) -> int:
        return self.file_size // self.chunk_size

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx % 1000 == 0:
            self.text = self._get_chuck()
            self.tokens = self.tokenizer.encode(self.text)
        r_idx = torch.randint(0, len(self.tokens) - self.block_size, (32, ))
        x = torch.stack([torch.tensor(self.tokens[i: i+self.block_size]) for i in r_idx])
        y = torch.stack([torch.tensor(self.tokens[i+1: i+self.block_size+1]) for i in r_idx])
        return x, y

