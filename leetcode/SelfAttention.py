import torch
import torch.nn as nn
import math
# 定义自注意力模块类，继承nn.Module
class self_attention(nn.Module):
    """__init__初始化对象属性, a = self_attention(512)创建对象, 并调用__init__, embedding_dim=512
       将512赋予self.embedding_dim"""
    def __init__(self,embed_size):
        # 初始化父类nn.Module，搭建好父类中重要参数
        super().__init__()
        # 嵌入向量维度
        self.embed_size = embed_size
        # 定义三个线性层QKV，维度为512
        # 线性层实现y=XW^T+b, W∈embedding_dim*embedding_dim，b∈embedding_dim
        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)

    # 前向传播函数
    def forward(self, x):
        # 对x做线性变换q=XW^T+b
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # qk的维度均为[B,L,D]，在每个batch中，要计算第i个token和所有L个token的得分，
        # 所以将k第1，2维转置，k维度变成[B,D,L], qk乘积维度[B,L,L]
        # 第二维的索引i：第i个位置在提问（Query） 
        # 第三维的索引j：第i个位置在看第j个位置（Key/Value）值多少
        attention_weight = torch.matmul(q, k.transpose(-2, -1))
        # print(attention_weight)
        # 为了避免维度大时点积值过大,权重除以根号D
        # 沿着最后一维(dim=-1)进行softmax归一化，对k的打分归一化
        attention_weight = torch.softmax(attention_weight/math.sqrt(self.embed_size), dim=-1)
        # print(attention_weight)
        # 注意力权重对V做加权求和
        attention_output = torch.matmul(attention_weight, v)
        return attention_output

if __name__ == "__main__":
    # batch(B): 一次前向/反向传播同时处理多少条样本; 
    # seq_len(L):每条样本里 token 的个数;
    # embedding_dim(D):每个 token 用多少维向量表示
    # batch=1，seq_len=2, embedding_dim=512
    x = torch.randn(1, 2, 512)
    # 创建self_attention对象，维度512
    attention = self_attention(512)
    # nn.Module自动调用forward
    output = attention(x)
    print(output.shape)
