import torch
import torch.nn as nn

class RoPE(nn.Module):
    # dim: 输入向量维度，base：控制频率衰减
    def __init__(self, dim, base=10000):
        ### 别忘了继承父类……
        super().__init__()
        self.dim = dim
        self.base = base

        # 只对一半维度生成频率（因为两两成对旋转）
        half_dim = dim // 2
        # theta_i=base^(-i/(d/2)), 计算每一组的旋转频率theta
        ### 别忘了1.0/
        self.inv_freq = 1.0 / (base ** (torch.arange(0, half_dim).float() / half_dim))
        # 存为常量不参与更新, 随着模型走，比直接定义self.inv_freq更快
        # self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """
        x: (batch, seq_len, dim)
        """
        ### 注意是x.shape不是x.shape()
        b, seq_len, d = x.shape
        # 检查，确保输入维度一致
        # assert d == self.dim

        # 每个 token 的位置[0, 1, 2, 3, ..., seq_len-1]
        pos = torch.arange(0, seq_len, device=x.device).float()

        # 计算角度 (seq_len, half_dim)
        # 外积freqs[i, j] = pos[i] * inv_freq[j]，每个位置 * 每个维度 → 一个角度
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)

        # 扩展到完整维度，把 (seq_len, half_dim) 变成 (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)

        # 计算sin(theta) cos(theta)在旋转公式里要用，还要拓展一维1，便于维度对齐
        cos = emb.cos()[None, :, :]  # (1, seq, dim)
        sin = emb.sin()[None, :, :]  # (1, seq, dim)

        # 划分奇数偶数维度
        # x1 = [x0, x2, x4……]
        # x2 = [x1, x3, x5……]
        # ...表示前几维不切片，之后最后一维才切   x[..., ::2]等价于x[:, :, ::2]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        # torch.stack([-x2, x1], dim=-1): (x1, x2) ->(-x2, x1)
        # stack后维度为(batch, seq_len, dim/2, 2)，.reshape_as(x)修改维度为(batch, seq_len, dim)
        # dim=-1表示-x2 x1俩堆叠在一起，形成新的维度，放在最后一维(B, T, D/2, 2)，也就是最后多了2
        ### 注意这里是reshape_as不是shape_as
        x_rotated = torch.stack([-x2, x1], dim=-1).reshape_as(x)
        
        # 第一项x * cos：x * cosθ = (x1 cosθ, x2 cosθ)
        # 第二项x_rotated * sin： (-x2, x1) = (-x2 sinθ, x1 sinθ)
        # 两项相加得x' = (x1 cosθ - x2 sinθ, x2 cosθ + x1 sinθ)，符合上面的旋转公式
        return x * cos + x_rotated * sin
    
### 注意是 "__main__"不是 "__init__"
if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn(2, 4, 8)  # batch=2, seq=4, dim=8
    print("input:\n", x)

    rope = RoPE(dim=8)

    y = rope(x)

    print("\noutput:\n", y)
    print("\nshape:", y.shape)