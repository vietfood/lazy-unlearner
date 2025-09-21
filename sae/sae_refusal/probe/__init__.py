import torch
from jaxtyping import Float
from torch import nn


class LinearProbe(nn.Module):
    def __init__(self, d_input: int):
        super().__init__()

        self.linear = nn.Linear(d_input, 1, bias=True)  # binary output

    def forward(self, input_acts: Float[torch.Tensor, "batch d_input"]):
        return self.linear(input_acts).squeeze(-1)  # Squeeze to shape ["batch"]