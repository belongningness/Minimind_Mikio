import torch
import torch.nn as nn
import math

class MQA(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads

        """MHA的qkv维度均为[d_model, self.head_dim], MQA只生成一份KV,
           所以维度为[d_model, self.head_dim]"""
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, self.head_dim)
        self.v_linear = nn.Linear(d_model, self.head_dim)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)

        """相比MHA, MQA不需要计算多个头的KV,所以无循环"""
        # Linear projections, all heads share the same K, V
        q = self.q_linear(x)    # (b, len, d_model)
        k = self.k_linear(x)    # (b, len, head_dim)
        v = self.v_linear(x)    # (b, len, head_dim)

        """需要整理Q的维度:[b, len, d_model]->[b, len, num_heads, head_dim]->[b, num_heads, len, head_dim], 
           d_model=num_heads*head_dim. kv加一维[b, len, head_dim]->[b, 1, len, head_dim]"""
        # q.view(B, L, H, d)：把 D 拆成 H×d，
        # 注意交换LH维变成(b, head, len, head_dim)，便于后续乘积
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) 
        # 在索引1处加一维
        k = torch.unsqueeze(k, 1)  # (b, 1, len, head_dim)
        v = torch.unsqueeze(v, 1)  # (b, 1, len, head_dim)

        # Attention 
        # (b, head, len, len)
        attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(attention, dim=-1)

        # Output
        output = torch.matmul(attention, v) # (b, head, len, head_dim)
        
        """复原output原始维度:[b, num_heads, len, head_dim]->[b, len, num_heads, head_dim]->[b, len, d_model]"""
        # 换维度顺序 → 保证内存连续 → 把 heads 和 head_dim 拼回 d_model
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # Linear projection
        output = self.out_linear(output)

        return output

# Example usage
d_model = 512
num_heads = 8
mqa = MQA(d_model, num_heads)
x = torch.randn(10, 20, d_model)  # (batch_size, sequence_length, d_model)
output = mqa(x)
print(output.shape)  # Should be (10, 20, 512)