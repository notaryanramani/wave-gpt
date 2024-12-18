from src.transformer import Decoder, ModelHyperParams
from src.wavenet import WaveNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from huggingface_hub import PyTorchModelHubMixin


params = ModelHyperParams()


class WaveGPT(
    nn.Module,
    PyTorchModelHubMixin,
):
    def __init__(
        self,
        vocab_size:int,
        n_embd:int = params.n_embd,
        n_heads:int = params.n_heads,
        n_layers:int = params.n_layers,
        block_size:int = params.block_size,
        wavenet_layers:int = params.wavenet_layers,
        reshape_factor:int = params.reshape_factor,
        dropout = params.dropout
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        self.decoders = nn.Sequential(*[Decoder(n_heads, n_embd, dropout) for _ in range(n_layers)])
        self.wavenet = WaveNet(block_size, n_embd, wavenet_layers, reshape_factor)
        self.beta = nn.Parameter(torch.zeros(1))

        self.ln = nn.LayerNorm(n_embd)
        self.linear = nn.Linear(n_embd, vocab_size)

    def forward(
        self, 
        x:torch.Tensor, 
        x_prev:Optional[torch.Tensor] = None, 
        y:Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = x.size()
        x = self.token_emb(x)

        pos_emb = self.pos_emb(torch.arange(T).to(params.device))
        x = x + pos_emb
        
        gpt_x = self.decoders(x)

        if x_prev is not None:
            x_prev = self.token_emb(x_prev)
            x_prev = x_prev + pos_emb
            wave_x = self.wavenet(x)
            x = (gpt_x * (1. - self.beta)) + (wave_x * self.beta)

        x = self.ln(x)
        logits = self.linear(x)

        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), y.view(B*T))

        return logits, loss

    def generate(self, x:torch.Tensor) -> torch.Tensor:
        logits, _ = self(x)
        return logits


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size:int,
        n_embd:int = params.n_embd,
        n_heads:int = params.n_heads,
        n_layers:int = params.n_layers,
        block_size:int = params.block_size,
        dropout = params.dropout
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        self.decoders = nn.Sequential(*[Decoder(n_heads, n_embd, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embd)
        self.linear = nn.Linear(n_embd, vocab_size)

    def forward(self, x:torch.Tensor, y:Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = x.size()
        x = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(T).to(params.device))
        x = x + pos_emb
        x = self.decoders(x)
        x = self.ln(x)
        logits = self.linear(x)

        if y is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), y.view(B*T))
        else:
            loss = None

        return logits, loss

    def generate(self, x:torch.Tensor) -> torch.Tensor:
        logits, _ = self(x)
        return logits
