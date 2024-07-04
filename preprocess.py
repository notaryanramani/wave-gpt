import tiktoken
from dataclasses import dataclass
import torch
from typing import Tuple, List


@dataclass
class PreprocessParams:
    block_size:int = 128
    batch_size:int = 32
    test_split:float = 0.2


params = PreprocessParams()


class DataPreprocess:
    def __init__(self,
        text:str,
        tokenizer:str,
        block_size:int = params.block_size,
        batch_size:int = params.batch_size,
        test_split:float = params.test_split
    ):
        self.text = text
        self.batch_size = self.batch_size
        self.block_size = self.block_size

        tok = tiktoken.get_encoding(tokenizer)
        self.train_data, self.test_data = self.train_test_split(tok, test_split)

    def train_test_split(self, tok, test_split)-> Tuple[List[int], ...]:
        tokens = tok.encode(self.text)
        split = int((1 - test_split) * len(tokens))
        train_data = tokens[:split]
        test_data = tokens[split:]
        return train_data, test_data

    def get_data(self, split = "train") -> Tuple[torch.Tensor, ...]:
        data = self.train_data if split == "train" else self.test_data
        idx = torch.randint(len(data) - self.block_size, (self.batch_size, ))
        x = torch.stack([torch.tensor(data[i.item():i.item()+self.block_size]) for i in idx])
        y = torch.stack([torch.tensor(data[i.item()+1:i.item()+self.block_size+1]) for i in idx])
        return x, y
