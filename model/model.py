import torch
import torch.nn.functional as F

# 1. Safe Softmax (防止数值溢出)
def safe_softmax(logits, dim=-1):
    """
    Safe softmax implementation that prevents overflow by subtracting max value.
    
    Args:
        logits: Input tensor
        dim: Dimension along which to compute softmax
    Returns:
        Softmax probabilities
    """
    # Subtract max for numerical stability
    logits_max = torch.max(logits, dim=dim, keepdim=True)[0]
    shifted_logits = logits - logits_max
    exp_logits = torch.exp(shifted_logits)
    sum_exp = torch.sum(exp_logits, dim=dim, keepdim=True)
    return exp_logits / sum_exp

# 2. 带因果Mask的Softmax (用于自回归模型，如Transformer解码器)
def causal_mask_softmax(logits, dim=-1):
    """
    Softmax with causal masking for autoregressive generation.
    Sets future positions to -inf before applying softmax.
    
    Args:
        logits: Input tensor of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len)
        dim: Dimension along which to compute softmax (typically last dimension)
    Returns:
        Masked softmax probabilities
    """
    seq_len = logits.shape[-1]
    
    # Create causal mask (upper triangular matrix)
    # 1 = mask out (future positions), 0 = keep
    mask = torch.triu(torch.ones(seq_len, seq_len, device=logits.device), diagonal=1).bool()
    
    # Apply mask: set masked positions to -inf
    # We need to broadcast mask to match logits dimensions
    if logits.dim() == 3:
        # Shape: (batch_size, seq_len, seq_len)
        mask = mask.unsqueeze(0)  # Add batch dimension
        masked_logits = logits.masked_fill(mask, float('-inf'))
    else:
        # Shape: (seq_len, seq_len)
        masked_logits = logits.masked_fill(mask, float('-inf'))
    
    # Apply safe softmax
    return safe_softmax(masked_logits, dim=dim)

# 演示示例
if __name__ == "__main__":
    print("=" * 50)
    print("示例 1: Safe Softmax")
    print("=" * 50)
    
    # 创建包含大数值的logits
    logits_large = torch.tensor([[1000.0, 1001.0, 1002.0], 
                                  [1.0, 2.0, 3.0]])
    print(f"输入 logits:\n{logits_large}")
    
    # 标准softmax (可能溢出)
    try:
        standard_softmax = F.softmax(logits_large, dim=-1)
        print(f"\n标准 Softmax:\n{standard_softmax}")
    except:
        print("\n标准 Softmax 发生数值溢出!")
    
    # Safe softmax
    safe_result = safe_softmax(logits_large, dim=-1)
    print(f"\nSafe Softmax:\n{safe_result}")
    print(f"概率和: {safe_result.sum(dim=-1)}")
    
    print("\n" + "=" * 50)
    print("示例 2: 因果Mask Softmax")
    print("=" * 50)
    
    # 创建注意力分数矩阵 (模拟4个token的序列)
    seq_len = 4
    attn_scores = torch.randn(1, seq_len, seq_len)  # batch_size=1
    print(f"原始注意力分数:\n{attn_scores.squeeze()}\n")
    
    # 应用因果mask softmax
    causal_probs = causal_mask_softmax(attn_scores, dim=-1)
    print(f"因果Mask Softmax 结果:\n{causal_probs.squeeze()}\n")
    
    # 显示mask模式
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    print(f"因果Mask模式 (True表示被mask的位置):\n{mask}")
    
    print("\n解释: 每个token只能看到当前及之前的token")
    for i in range(seq_len):
        print(f"Token {i} 对之前的概率分布: {causal_probs[0, i, :i+1].tolist()}")