import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base

        # 只对一半维度生成频率（因为两两成对旋转）
        half_dim = dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """
        x: (batch, seq_len, dim)
        """
        b, seq_len, d = x.shape
        assert d == self.dim

        # 位置
        pos = torch.arange(seq_len, device=x.device).float()

        # 计算角度 (seq_len, half_dim)
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)

        # 拼成 (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos()[None, :, :]  # (1, seq, dim)
        sin = emb.sin()[None, :, :]  # (1, seq, dim)

        # 偶数奇数维交错旋转
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        # 旋转
        x_rotated = torch.stack([-x2, x1], dim=-1).reshape_as(x)

        return x * cos + x_rotated * sin
    

if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn(2, 4, 8)  # batch=2, seq=4, dim=8
    print("input:\n", x)

    rope = RoPE(dim=8)

    y = rope(x)

    print("\noutput:\n", y)
    print("\nshape:", y.shape)