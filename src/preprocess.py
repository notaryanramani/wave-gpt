from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from typing import Tuple
import os
import tiktoken
import mmap


@dataclass
class PreprocessParams:
    block_size:int = 128
    batch_size:int = 32
    split:float = 0.2


params = PreprocessParams()


class OpenWebText(Dataset):
    def __init__(self, text_file, split, block_size = 128, chunk_size = 25000):
        self.block_size = block_size
        self.file_path = text_file
        self.split = split
        self.tokenizer = tiktoken.get_encoding('r50k_base')
        self.file_size = os.path.getsize(text_file)
        self.chunk_size = chunk_size
        if self.split == 'train':
            self.start_chunk = 0
        else:
            self.start_chunk = int(0.8 * self.file_size)
        
    def _get_chuck(self):
        with open(self.file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                mm.seek(self.start_chunk)
                block = mm.read(self.chunk_size)
                text = block.decode('utf-8', errors='ignore').replace('\r', '')
                self.start_chunk += self.chunk_size
        return text
            

    def __len__(self) -> int:
        if self.split == 'train':
            return int(self.file_size // self.chunk_size * (1 - params.split))
        else:
            return int(self.file_size // self.chunk_size * params.split)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self._get_chuck()
        tokens = self.tokenizer.encode(text)
        r_idx = torch.randint(0, len(tokens) - self.block_size, (32, ))
        x = torch.stack([torch.tensor(tokens[i: i+self.block_size]) for i in r_idx])
        y = torch.stack([torch.tensor(tokens[i+1: i+self.block_size+1]) for i in r_idx])
        return x, y

