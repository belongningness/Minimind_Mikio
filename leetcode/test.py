import torch
import torch.nn as nn
import math

class GQA(nn.Module):
    def __init__(self, d_model, head_nums, group_nums):
        super().__init__()
        self.d_model = d_model
        self.head_nums = head_nums
        self.group_nums = group_nums
        self.head_dim = d_model // head_nums
        self.group_h = head_nums // group_nums
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, group_nums*self.head_dim)
        self.v_proj = nn.Linear(d_model, group_nums*self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        bs, seql, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(bs, seql, self.group_nums, self.group_h, self.head_dim).permute(0,2,3,1,4)
        k = k.view(bs, seql, self.group_nums, self.head_dim).transpose(1,2)
        v = v.view(bs, seql, self.group_nums, self.head_dim).transpose(1,2)
        k = torch.unsqueeze(k, 2)
        v = torch.unsqueeze(v, 2)
        attention = torch.softmax(q@k.transpose(-1,-2)/math.sqrt(self.head_dim), dim=-1)@v
        output = attention.permute(0,3,1,2,4).contiguous().view(bs, seql, self.d_model)
        return self.out_proj(output)
    
# Example usage
d_model = 512
num_heads = 8
num_groups = 4

gqa = GQA(d_model, num_heads, num_groups)
input_tensor = torch.randn(32, 10, d_model)  # (batch_size, sequence_length, d_model)
output_tensor = gqa(input_tensor)
print(output_tensor.shape)  # Output: torch.Size([32, 10, 512])