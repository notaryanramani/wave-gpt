import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelHyperParams:
    device:str = "cuda" if torch.cuda.is_available() else "cpu"
    n_embd:int = 128
    n_heads:int = 4
    n_layers:int = 4
    wavenet_layers:int = 3
    reshape_factor:int = 4
    block_size:int = 64
    batch_size:int = 32
    dropout:float = 0.2


params = ModelHyperParams()


class FeedForward(nn.Module):
    def __init__(self, n_embd:int = params.n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(params.dropout)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out


class Decoder(nn.Module):
    def __init__(self, n_heads:int = params.n_heads, n_embd:int = params.n_embd, dropout:float = params.dropout):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.tril(torch.ones((params.block_size, params.block_size), dtype=torch.float32)))


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        x = self.ln1(x)
        att_output, _ = self.mha(x, x, x, is_causal=True, need_weights=False , attn_mask=self.mask[:T, :T])
        x = x + att_output
        x = self.ln2(x)
        x = x + self.ffn(x)
        out = self.dropout(x)
        return out
