import torch
import torch.nn as nn
import math

class LoRA(nn.Module):
    def __init__(self, dim_in, dim_out, alpha, r):
        super().__init__()
        self.alpha = alpha
        self.r = r
        self.scale = self.alpha / self.r

        # LoRA参数
        ### 注意A是随机初始化，B初始化为0
        self.A = nn.Parameter(torch.randn(dim_in, r))
        self.B = nn.Parameter(torch.zeros(r, dim_out))
        
        # 原始权重（冻结）
        ### 一定注意requires_grad=False冻结参数，也是随机初始化
        self.weight = nn.Parameter(torch.randn(dim_in, dim_out), requires_grad=False)

    def forward(self, x):
        # 原始输出 + LoRA输出
        # x @ weight: [batch, dim_in] @ [dim_in, dim_out] = [batch, dim_out]
        # x @ A @ B: [batch, dim_in] @ [dim_in, r] @ [r, dim_out] = [batch, dim_out]
        return x @ self.weight + x @ self.A @ self.B * self.scale


# 完整的应用实例
if __name__ == '__main__':
    print("="*60)
    print("LoRA 完整实例演示")
    print("="*60)
    
    # 基本参数
    dim_in = 512
    dim_out = 512
    r = 4
    alpha = 32
    batch_size = 64
    
    print(f"\n配置参数:")
    print(f"  输入维度: {dim_in}")
    print(f"  输出维度: {dim_out}")
    print(f"  LoRA秩 r: {r}")
    print(f"  LoRA缩放系数 α: {alpha}")
    print(f"  缩放因子 scale = α/r = {alpha}/{r} = {alpha/r}")
    
    # 创建输入
    x = torch.randn(batch_size, dim_in)
    print(f"\n输入形状: {x.shape}")
    
    # 创建LoRA层
    lora = LoRA(dim_in, dim_out, alpha, r)
    
    # 前向传播
    output = lora(x)
    print(f"输出形状: {output.shape}")