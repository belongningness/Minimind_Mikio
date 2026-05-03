import torch
import torch.nn.functional as F

def safe_softmax(logits, dim=-1):
    """
    Safe Softmax实现,防止数值溢出
    Args:
        logits: 输入张量
        dim: 计算的维度, 沿着最后一维计算max和sum
    Returns:
        softmax概率分布
    """
    # 减去最大值防止exp溢出
    # torch.max(logits, dim=dim, keepdim=True) 返回两个值：
    # [0]: 最大值本身
    # [1]: 最大值所在的索引
    logits_max = torch.max(logits, dim=dim, keepdim=True)[0]
    shifted_logits = logits - logits_max
    
    # 计算exp
    ### 注意exp的写法
    exp_logits = torch.exp(shifted_logits)
    
    # 计算sum
    sum_exp = torch.sum(exp_logits, dim=dim, keepdim=True)
    
    # 返回概率分布
    return exp_logits / sum_exp

def causal_mask_softmax(logits, dim=-1):
    """
    带因果Mask的Softmax实现
    Args:
        logits: 输入张量 [seq_len, seq_len]
        dim: 计算的维度（通常是最后一维）
    Returns:
        带因果mask的softmax概率分布
    """
    seq_len = logits.shape[0]
    # 创建因果mask，因为输入logits和torch.ones(seq_len, seq_len)构造的全1矩阵，
    # 它们维度均为[seq_len, seq_len]，所以不需要像MHA一样unsqueeze拓展维度
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    # mask = mask.unsqueeze(0)
    # 将mask中为0的位置设为负无穷
    masked_logits = logits.masked_fill(mask, float('-inf'))
    
    # 应用safe softmax
    return safe_softmax(masked_logits, dim=dim)

# 测试代码
if __name__ == "__main__":
    # 测试1: Safe Softmax
    print("="*50)
    print("测试 Safe Softmax")
    print("="*50)
    
    # 创建包含较大值的logits
    logits = torch.tensor([[100.0, 101.0, 102.0],
                           [1.0, 2.0, 3.0]])
    
    print("输入logits:")
    print(logits)
    
    # 使用我们的实现
    our_softmax = safe_softmax(logits, dim=-1)
    print("\n我们的Safe Softmax实现:")
    print(our_softmax)
    
    # 测试2: 因果Mask Softmax
    print("\n" + "="*50)
    print("测试 因果Mask Softmax")
    print("="*50)
    
    # 创建序列
    seq_len = 4
    # 模拟注意力分数 [seq_len, seq_len]
    attention_scores = torch.randn(seq_len, seq_len)
    
    print("原始注意力分数:")
    print(attention_scores)
    
    causal_probs = causal_mask_softmax(attention_scores, dim=-1)
    print("\n因果Mask Softmax后的概率分布:")
    print(causal_probs)