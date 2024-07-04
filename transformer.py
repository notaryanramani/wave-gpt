import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelParams:
    device:str = "cuda" if torch.cuda.is_available() else "cpu"
    n_embd:int = 256
    dropout:float = 0.2


params = ModelParams()


class Head(nn.Module):
    def __init__(self, head_size:int, n_embd:int = params.n_embd, dropout:float = params.dropout):
        super().__init__()
        self.q = nn.Linear(n_embd, head_size)
        self.k = nn.Linear(n_embd, head_size)
        self.v = nn.Linear(n_embd, head_size)
        self.dropout = dropout

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads:int, n_embd:int = params.n_embd, dropout:float = params.dropout):
        super().__init__()
        assert n_embd % n_heads == 0, "n_heads should be divisible by n_embd"
        head_size = n_embd // n_heads
        self.heads = nn.ModuleList([Head(head_size, n_embd, dropout) for _ in range(n_heads)])
        self.proj(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x


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
    def __init__(self, n_heads:int, n_embd:int = params.n_embd, dropout:float = params.dropout):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, n_embd, dropout)
        self.ffn = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.ln1(x)
        x = x + self.mha(x)
        x = self.ln2(x)
        x = x + self.ffn(x)
        out = self.dropout(x)
        return out
