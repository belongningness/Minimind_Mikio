import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """
    基于 SiLU(Swish) + Gate 的 FFN
    对应现代 LLM 中常见的 SwiGLU 结构
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()

        # gate 分支
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)

        # value/content 分支
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

        # 输出投影
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        """
        x shape:
        [batch_size, seq_len, dim]
        """
        # gate 分支：SiLU 激活
        ### 别忘了self.gate_proj(x)中要加x
        gate = F.silu(self.gate_proj(x))

        # value 分支
        up = self.up_proj(x)

        # gating
        hidden = gate * up

        # 投影回原维度
        out = self.down_proj(hidden)

        return out


# =========================
# 可运行示例
# =========================
if __name__ == "__main__":

    # 假设：
    # batch_size = 2
    # seq_len = 4
    # hidden_size = 8

    x = torch.randn(2, 4, 8)

    print("输入 shape:")
    print(x.shape)

    # 创建 FFN
    ffn = SwiGLUFFN(
        dim=8,
        hidden_dim=16
    )

    # 前向传播
    y = ffn(x)

    print("\n输出 shape:")
    print(y.shape)

    print("\n输出 tensor:")
    print(y)