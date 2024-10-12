from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from typing import Tuple
import os
import tiktoken
import mmap
import numpy as np

from .transformer import ModelHyperParams


params = ModelHyperParams()


class OpenWebText(Dataset):
    def __init__(self, text_file, block_size = params.block_size, batch_size = params.batch_size):
        self.block_size = block_size
        self.batch_size = batch_size
        self.file_path = text_file
        self.tokenizer = tiktoken.get_encoding('r50k_base')
        self.file_size = os.path.getsize(text_file)
        self.chunk_size = block_size * batch_size * 8 
        self.start_chunk = 0

        
    def _get_chuck(self):
        with open(self.file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                if self.start_chunk > self.file_size - self.chunk_size:
                    self.start_chunk = 0
                mm.seek(self.start_chunk)
                block = mm.read(self.chunk_size)
                text = block.decode('utf-8', errors='ignore').replace('\r', '')
                self.start_chunk += self.chunk_size
        return text
            

    def __len__(self) -> int:
        return int(self.file_size // self.chunk_size)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text = self._get_chuck()
        tokens = self.tokenizer.encode(text)
        r_idx = torch.randint(self.block_size, len(tokens) - self.block_size, (self.batch_size, ))
        x = torch.stack([torch.tensor(tokens[i: i+self.block_size]) for i in r_idx])
        prev_x = torch.stack([torch.tensor(tokens[i-self.block_size: i]) for i in r_idx])
        y = torch.stack([torch.tensor(tokens[i+1: i+self.block_size+1]) for i in r_idx])
        return x, prev_x, y
    

class Shakespeare(Dataset):
    def __init__(self, filePath, block_size = params.block_size):
        self.block_size = block_size
        with open(filePath, 'r') as f:
            self.text = f.read()

    def __len__(self):
        return len(self.text) // self.block_size
    
    def __getitem__(self, index):
        index = min(index, len(self.text) - self.block_size - 1)
        index = max(index, self.block_size)
        x = self.text[index   : index+self.block_size]
        x_prev = self.text[index-self.block_size : index]
        y = self.text[index+1 : index+1+self.block_size]
        return list(x), list(x_prev), list(y)
    

class ShardsLoader:
    def __init__(
        self, 
        process_rank,
        num_processes,
        path = 'data/', 
        split = 'train', 
        block_size = params.block_size, 
        batch_size = params.batch_size
    ):
        self.path = path
        self.T = block_size
        self.B = batch_size
        self.process_rank = process_rank
        self.num_processes = num_processes

        shards = os.listdir(path)
        shards = sorted([os.path.join(path, shard) for shard in shards])
        shards = [s for s in shards if split in s]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split}"
        self.reset()
    
    def _load_tokens(self, file):
        tokens = np.load(file)
        tokens = tokens.astype(np.int64)
        tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens

    def reset(self):
        self.current_shard = 0
        self.tokens = self._load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def get_batch(self):
        B, T = self.B, self.T
        toks = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = toks[:-1].view(B, T)
        y = toks[1:].view(B, T)

        self.current_position += B * T * self.num_processes

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self._load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

        
