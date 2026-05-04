import numpy as np

def binary_cross_entropy_loss(predictions, targets):
    """
    二分类交叉熵损失
    BCE = -[y*log(p) + (1-y)*log(1-p)]
    """
    # 步骤1: 防止log(0)出现无穷大
    eps = 1e-7  # 很小的正数
    predictions = np.clip(predictions, eps, 1 - eps)
    # np.clip 将值限制在 [eps, 1-eps] 范围内
    
    # 步骤2: 计算每个样本的损失
    # 公式: -[y*log(p) + (1-y)*log(1-p)]
    # 1 - targets是标签翻转，0变1,1变0
    ### 注意两个分式之间是+不是*
    losses = -(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    
    # 步骤3: 返回平均损失
    return np.mean(losses)

# 实例
### 注意这里和CE不同，一个元素代表一个样本，而CE是一行代表一个
y_pred = np.array([0.9, 0.3, 0.8, 0.2, 0.7])
y_true = np.array([1, 0, 1, 0, 1])

loss_manual = binary_cross_entropy_loss(y_pred, y_true)

print("Binary Cross-Entropy Loss实例:")
print(f"  预测概率: {y_pred}")
print(f"  真实标签: {y_true}")
print(f"  手动实现BCE: {loss_manual:.4f}")