# 更完整的DPO实现
import torch

def dpo_loss_complete(policy_chosen_logps, policy_rejected_logps,
                      reference_chosen_logps, reference_rejected_logps,
                      beta=0.1):
    """
    完整的DPO损失,包含参考模型
    """
    # 计算策略模型的差异
    policy_ratio = policy_chosen_logps - policy_rejected_logps
    
    # 计算参考模型的差异
    reference_ratio = reference_chosen_logps - reference_rejected_logps
    
    # DPO损失
    log_ratio = policy_ratio - reference_ratio
    loss = -torch.log(torch.sigmoid(beta * log_ratio))
    
    return loss.mean()

# 实例
# 模拟一批数据，每个样本有chosen和rejected的log概率
batch_size = 4

# 策略模型输出的log概率
policy_chosen = torch.tensor([-0.5, -0.3, -0.8, -0.2])   # chosen响应的log概率
policy_rejected = torch.tensor([-1.2, -1.5, -0.9, -1.8])  # rejected响应的log概率

# 参考模型输出的log概率（通常是冻结的初始模型）
ref_chosen = torch.tensor([-0.6, -0.4, -0.7, -0.3])
ref_rejected = torch.tensor([-1.1, -1.4, -1.0, -1.7])

beta = 0.1  # 温度参数

loss_full = dpo_loss_complete(policy_chosen, policy_rejected,
                              ref_chosen, ref_rejected, beta)

print("DPO Loss实例:")
print(f"  Beta参数: {beta}")
print(f"\n策略模型输出:")
print(f"    chosen log概率: {policy_chosen.tolist()}")
print(f"    rejected log概率: {policy_rejected.tolist()}")
print(f"DPO损失: {loss_full.item():.4f}")

# 解释DPO损失的含义
print(f"\nDPO损失解释:")
print(f"  当chosen概率远大于rejected时，损失接近0")
print(f"  当chosen概率小于rejected时，损失较大")