import torch
import torch.nn as nn
import torch.nn.functional as F

class SiLUFFN(nn.Module):
    def __init__(self, hidden_dim, dim):
        super().__init__()
        self.up_proj = nn.Linear(hidden_dim, dim)
        self.gate_proj = nn.Linear(hidden_dim, dim)
        self.down_proj = nn.Linear(dim, hidden_dim)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        return self.down_proj(self.up_proj(x)*gate)
    

x = torch.randn(2,4,8)
ffn = SiLUFFN(8,16)
print(ffn(x))
        