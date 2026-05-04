import numpy as np

def auc_score(y_true, y_pred):
    """
    简单实现AUC
    AUC = P(正样本分数 > 负样本分数)
    """
    # 概率法（直接计算正>负的概率）
    # y_true表示原始正负标签, y_pred表示预测概率，我们先根据y_true划分正负样本，
    # 再从y_pred中两两比较所有样本的概率，记录正样本概率>负样本概率的次数/总比较次数

    # np.where返回矩阵，[0]直接看元素索引
    pos_indices = np.where(y_true == 1)[0]
    print(pos_indices)
    neg_indices = np.where(y_true == 0)[0]
    
    correct = 0  # 记录正样本概率>负样本概率的次数
    total = 0  # 全部比较次数
    
    for pos_idx in pos_indices:
        for neg_idx in neg_indices:
            # 因为pos_indices neg_indices是索引，所以直接可以提取y_pred的对应索引概率
            # 正样本概率>负样本概率 correct+1
            if y_pred[pos_idx] > y_pred[neg_idx]:
                correct += 1
            elif y_pred[pos_idx] == y_pred[neg_idx]:
                correct += 0.5  # 平局算0.5
            total += 1
    
    auc = correct / total if total > 0 else 0.5
    return auc

### 注意一定是array
y_true = np.array([1, 0, 1, 0, 1, 0])
y_pred = np.random.rand(6)
# y_pred = np.array([0.9, 0.4, 0.8, 0.3, 0.7, 0.2])

auc_manual = auc_score(y_true, y_pred)

print("AUC实例:")
print(f"  预测值: {y_pred}")
print(f"  真实值: {y_true}")
print(f"  手动实现AUC: {auc_manual:.4f}")