import torch
import torch.nn as nn
import math

class GQA(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        """与MHA相比, GQA存在g个组"""
        super(GQA, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups

        self.group_heads = num_heads // num_groups # 每一组有多少个head
        self.head_dim = d_model // num_heads

        """一共g组K和V, 每组d维"""
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.head_dim * num_groups)
        self.W_v = nn.Linear(d_model, self.head_dim * num_groups)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections, all heads share the same K, V
        q = self.W_q(x)    # (b, len, d_model)
        k = self.W_k(x)    # (b, len, head_dim * num_groups)
        v = self.W_v(x)    # (b, len, head_dim * num_groups)
        
        # (b, len, num_groups, group_heads, head_dim) -> (b, num_groups, group_heads, len, head_dim)
        q = q.view(batch_size, seq_len, self.num_groups, self.group_heads, 
                   self.head_dim).permute(0, 2, 3, 1, 4) 
        # (b, num_groups, len, head_dim)
        k = k.view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)  
        # (b, num_groups, len, head_dim)  
        v = v.view(batch_size, seq_len, self.num_groups, self.head_dim).transpose(1, 2)     
        
        k = torch.unsqueeze(k, 2) # (b, num_groups, 1, len, head_dim)  
        v = torch.unsqueeze(v, 2) # (b, num_groups, 1, len, head_dim)

        # Attention 
        # (b, num_groups, group_heads, len, len)
        attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(attention, dim=-1)
        
        """复原output原始维度:(b, num_groups, group_heads, len, head_dim) -> 
        (b, len, num_groups, group_heads, head_dim) -> (b, len, d_model)"""
        output = torch.matmul(attention, v) # (b, group, g_heads, len, head_dim)
        output = output.permute(0, 3, 1, 2, 4).contiguous().view(batch_size, -1, self.d_model)

        # Linear projection
        output = self.out_linear(output)
        return output


# Example usage
d_model = 512
num_heads = 8
num_groups = 4

gqa = GQA(d_model, num_heads, num_groups)
input_tensor = torch.randn(32, 10, d_model)  # (batch_size, sequence_length, d_model)
output_tensor = gqa(input_tensor)
print(output_tensor.shape)  # Output: torch.Size([32, 10, 512])
