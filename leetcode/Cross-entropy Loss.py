import torch
import torch.nn.functional as F
import numpy as np

def cross_entropy_loss(predictions, targets):
    """
    多分类交叉熵损失
    CE = -Σ y_true * log(y_pred)
    """
    # softmax，现将logits转化为概率
    ### 注意np.sum计算的是np.exp(predictions)，别把exp丢了；
    ### np.sum和torch.sum的参数也有点不同 dim/axis
    probs = np.exp(predictions) / np.sum(np.exp(predictions), axis=-1, keepdims=True)
    
    # 计算CE loss
    n_samples = len(targets)
    loss = 0
    for i in range(n_samples):
        # probs 是一个二维数组 [样本数 × 类别数]
        # i选择第几行（第几个样本）
        # targets[i]取i样本的真实类别对应的概率
        loss += -np.log(probs[i, targets[i]])
    
    return loss / n_samples

# 实例
# 3个样本，3个类别
# 每一行表示当前样本处于0 1 2三类的概率
predictions = np.random.rand(3, 4)
targets = np.array([0, 1, 2])  # 真实类别

loss_manual = cross_entropy_loss(predictions, targets)
loss_torch = F.cross_entropy(torch.tensor(predictions), torch.tensor(targets))

print("Cross-Entropy Loss实例:")
print(f"  预测logits:\n{predictions}")
print(f"  真实标签: {targets}")
print(f"  手动实现损失: {loss_manual:.4f}")
print(f"  PyTorch损失: {loss_torch.item():.4f}\n")