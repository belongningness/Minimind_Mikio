import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        # 和BatchNorm只有此处不同，-1表示根据特征做均值；而BatchNorm是0
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x-mean)*torch.rsqrt(var.pow(2)+self.eps)
    
    def forward(self, x):
        return self.gamma*self._norm(x.float()).type_as(x)+self.beta
    
if __name__ == "__main__":
    torch.manual_seed(0)
    # LayerNorm和RMSNorm同样适用于NLP，所以输入格式一样
    x = torch.randn(2,3,4)
    ln = LayerNorm(dim=4)
    print("input\n", x)
    
    output = ln(x)
    print("output\n", output)