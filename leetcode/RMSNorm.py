import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-5):
        # 注意一定要继承父类
        super().__init__()
        # 输入x是多维张量比如(样本数batch, token数seq_len, 每个token的特征向量dim)
        self.dim = dim
        # eps在分母中防止RMS=0
        self.eps = eps
        # nn.Parameter表示weight是一个可学习变量，创建一个维度为(dim,1)的weight
        # 因为最后x*weight(逐元素相乘，而不是矩阵相乘)，x的维度是(2,3,4)
        # 所以weight的维度必须与x的最后一维dim相同
        self.weight = nn.Parameter(torch.ones(dim))

    # _表示仅在类内使用的函数
    def _norm(self, x):
        # mean(-1)表示对x的最后一维dim取均值。因为要计算单个token内的特征均值，所以只能对dim做均值
        # keepdim=True让最终均值的维度与x相同(batch, seq_len,1)，不加这一项就变成(batch, seq_len)
        return torch.rsqrt(x.pow(2).mean(-1, keepdim = True)+self.eps)*x
    
    def forward(self, x):
        # x.float()是为了提高计算精度(float32)
        # 本来weight的维度是(4,)，这里相乘出现了boardcast机制，维度变成(1,1,4)和x的(2,3,4)匹配
        return self.weight*self._norm(x.float()).type_as(x)



if __name__ == "__main__":
    # 固定随机化种子
    torch.manual_seed(0)

    x = torch.randn(2, 3, 4)  # batch=2, seq=3, dim=4
    print("input:\n", x)

    norm = RMSNorm(dim=4, eps=1e-5)

    y = norm(x)

    print("\noutput:\n", y)
    print("\nshape:", y.shape)