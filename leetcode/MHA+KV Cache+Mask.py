import torch
import torch.nn as nn
import math

class MHA(nn.Module):
    def __init__(self, d_model, heads_num):
        super().__init__()
        self.d_model = d_model
        self.heads_num = heads_num
        assert d_model % heads_num == 0
        self.head_dim = d_model // heads_num

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # 初始化cache
        self.cache_k = None
        self.cache_v = None

    def forward(self, x):
        bs, seql, _ = x.shape

        # 1. 计算Q、K、V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 重塑为多头格式 [B, num_heads, L, d_k]
        q = q.view(bs, seql, self.heads_num, self.head_dim).transpose(1,2)
        k = k.view(bs, seql, self.heads_num, self.head_dim).transpose(1,2)
        v = v.view(bs, seql, self.heads_num, self.head_dim).transpose(1,2)

        # 3. KV Cache 更新: 先判断 cache 是否存在，再决定拼接还是初始化
        if self.cache_k is not None:
            # cache 已有历史 KV, 拼接当前步的 K/V
            # 在L维度拼接，因为不能改变每个头维度(self.d_k),
            # 头数目(self.num_heads),批次数(B)，所以只能在L维度拼接
            ### 注意cat的写法，[self.cache_k, K]别落了[]
            self.cache_k = torch.cat([self.cache_k, k], dim=-2)
            self.cache_v = torch.cat([self.cache_v, v], dim=-2)
        else:
            # 第一次，直接初始化 cache
            self.cache_k = k
            self.cache_v = v
        # Attention 使用完整的历史 KV（而非仅当前步的 K/V）
        k = self.cache_k  # [bs, heads, total_len, head_dim]
        v = self.cache_v  # [bs, heads, total_len, head_dim]

        # 4. 计算 attention score (Q 对完整 K 做 attention)
        # 因为使用了KVcache完整KV，所以之后的mask必须与完整KV的维度一致
        # 所以需要得到完整长度total_len
        total_len = k.shape[-2]
        attention = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)  # [bs, heads, seql, total_len]

        # 5. mask
        # 这里ones初始化的形状是[seql, total_len]
        mask = torch.triu(torch.ones(seql, total_len), diagonal=1).bool()  # [seql, total_len]
        ### 注意这里连续使用两次.unsqueeze(0)，而不是.unsqueeze(0,1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seql, total_len]
        # 应用mask
        attention = attention.masked_fill(mask, float('-inf'))
        attention = torch.softmax(attention, dim=-1)
        output = attention @ v

        # 6. 合并多头并输出，还原初始维度
        output = output.transpose(-1,-2).view(bs, seql, -1) # [bs, seql, d_model]
        return self.o_proj(output)
    
    # 重置KV cache
    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None
    
if __name__ == "__main__":
    print("="*60)
    print("测试带KV Cache的MultiHeadAttention")
    print("="*60)
    
    # 初始化模型
    d_model = 512
    num_heads = 8
    mha = MHA(d_model, num_heads)

    # 测试2: 使用KV Cache（自回归生成模拟）
    print("\n2. 使用KV Cache - 模拟自回归生成")
    mha.reset_cache()  # 重置缓存
    
    # 第一步：输入第一个token
    x1 = torch.randn(2, 1, d_model)
    out1 = mha(x1)
    print(f"第1步 - 输入: {x1.shape}, 输出: {out1.shape}")
    print(f"  缓存形状: K={mha.cache_k.shape}, V={mha.cache_v.shape}")
    
    # 第二步：输入第二个token
    x2 = torch.randn(2, 1, d_model)
    out2 = mha(x2)
    print(f"第2步 - 输入: {x2.shape}, 输出: {out2.shape}")
    print(f"  缓存形状: K={mha.cache_k.shape}, V={mha.cache_v.shape}")
    
    # 第三步：输入第三个token
    x3 = torch.randn(2, 1, d_model)
    out3 = mha(x3)
    print(f"第3步 - 输入: {x3.shape}, 输出: {out3.shape}")
    print(f"  缓存形状: K={mha.cache_k.shape}, V={mha.cache_v.shape}")