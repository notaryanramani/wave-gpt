import torch
import torch.nn as nn
from src.transformer import ModelHyperParams


params = ModelHyperParams()


class WaveNetLayer(nn.Module):
    def __init__(self, n_embd:int, reshape_factor: int):
        super().__init__()
        fan_in = n_embd * reshape_factor
        self.layernorm = nn.LayerNorm(fan_in)
        self.linear = nn.Linear(fan_in, n_embd)
        self.relu = nn.ReLU()
        self.rf = reshape_factor

    def forward(self, x):
        B, T, C = x.size()
        x = x.view(B, T // self.rf, C * self.rf)
        x = self.layernorm(x)
        x = self.linear(x)
        out = self.relu(x)
        return out


class WaveNet(nn.Module):
    def __init__(self, block_size:int = params.block_size, n_embd:int = params.n_embd, n_layers:int = params.wavenet_layers, reshape_factor:int = params.reshape_factor):
        super().__init__()
        self.need_output_pooling = False
        self.wavenet = nn.Sequential(*[WaveNetLayer(n_embd, reshape_factor) for _ in range(n_layers)])
        assert block_size // (reshape_factor ** n_layers) > 0, "The reshape_factor ** n_layers  shouldn't greater than block_size to carry out convolutions"
        output_pooling_factor = block_size // (reshape_factor ** n_layers)
        if output_pooling_factor > 1:
            print(f"requires output pooling of {output_pooling_factor}")
            self.need_output_pooling = True
            self.output_linear = WaveNetLayer(n_embd, output_pooling_factor)

    def forward(self, x):
        x = self.wavenet(x)
        if self.need_output_pooling:
            x = self.output_linear(x)
        return x
