import torch
import torch.nn as nn
import math

class MHA(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model

        """与自注意力SA的输入[B,L,D]相比, MHA存在H个头且需要输入头的维度d, 且D=H*d"""
        self.num_heads = num_heads
        assert d_model % num_heads == 0  # 确保嵌入维度可以被头的数量整除
        self.head_dim = d_model // num_heads  # 每个头的维度, //整除向下取整

        """SA只有一个head对应一套QKV, MHA每个head单独一套QKV线性层, 输入为嵌入维度D, 输出为头维度d
           MHA还需要FC层, 输入输出都是嵌入维度D"""
        self.q_linear = nn.ModuleList([nn.Linear(d_model, self.head_dim, bias=False) for _ in range(num_heads)])
        self.k_linear = nn.ModuleList([nn.Linear(d_model, self.head_dim, bias=False) for _ in range(num_heads)])
        self.v_linear = nn.ModuleList([nn.Linear(d_model, self.head_dim, bias=False) for _ in range(num_heads)])
        self.out_linear = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # 获取x的三个维度[B,L,D]
        batch_size, seq_len, _ = x.shape
        # outputs 用来收集每个 head 的输出（每个 head 输出 [B, L, d]）
        outputs = []

        """循环执行每个头对应的QKV映射"""
        # Parallel
        for s in range(self.num_heads):
            q = self.q_linear[s](x) # (batch_size, seq_len, head_dim)
            k = self.k_linear[s](x)
            v = self.v_linear[s](x) # (batch_size, seq_len, head_dim)

            # Attention
            # (batch_size, seq_len, seq_len)
            attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            """与SA不同, MHA需要对下三角做掩码 """
            # torch.ones(seq_len, seq_len)：先造一个全 1 的方阵，形状 [L, L]
            # torch.triu(..., diagonal=1): 取上三角(upper triangle)
            # .bool()：把 0/1 变成 False/True 的布尔 mask 
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            # unsqueeze(0)：在最前面加一个维度，把 [L, L] 变成 [1, L, L]
            mask = mask.unsqueeze(0).to(x.device)   # (1, seq_len, seq_len)
            # 对所有 mask==True 的位置，把对应的 score 替换成 -inf负无穷
            attention = attention.masked_fill(mask, float('-inf'))

            attention = torch.softmax(attention, dim=-1)
            
            # Output
            output = torch.matmul(attention, v) # (batch_size, seq_len, head_dim)
            outputs.append(output)

        """output存储了H个head的结果, 维度为[B,L,d,H]dim=-1 表示沿着最后一维拼接(也就是H维度)
            形状变成[B,L,D]"""
        # Linear projection
        output = torch.cat(outputs, dim=-1) # (batch_size, seq_len, d_model)
        output = self.out_linear(output)
        
        return output

# Example usage
d_model = 512
num_heads = 8
mha = MHA(d_model, num_heads)
x = torch.randn(10, 20, d_model)  # (batch_size, sequence_length, d_model)
output = mha(x)
print(output.shape)  # Should be (10, 20, 512)