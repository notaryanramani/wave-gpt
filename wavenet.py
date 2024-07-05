import torch
import torch.nn as nn


class WaveNetBlock(nn.Module):
    def __init__(self, n_embd, reshape_factor: int = 4):
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
    def __init__(self, block_size:int, n_embd:int, n_layers:int = 7, reshape_factor:int = 2):
        super().__init__()
        self.need_output_pooling = False
        self.wavenet = nn.Sequential(*[WaveNetBlock(n_embd, reshape_factor) for _ in range(n_layers)])
        assert block_size // (reshape_factor ** n_layers) > 0, "The reshape_factor ** n_layers  shouldn't greater than block_size to carry out convolutions"
        output_pooling_factor = block_size // (reshape_factor ** n_layers)
        print(output_pooling_factor)
        if output_pooling_factor > 1:
            self.need_output_pooling = True
            self.output_linear = WaveNetBlock(n_embd, output_pooling_factor)

    def forward(self, x):
        x = self.wavenet(x)
        print(x.shape)
        if self.need_output_pooling:
            x = self.output_linear(x)
        return x