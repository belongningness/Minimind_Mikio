import torch

def softmax(logits):
    max_val = torch.max(logits, dim=-1, keepdim=True).values
    exp_logits = torch.exp(logits - max_val)
    return exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
                                  
def kl(p_logits, q_logits):
    eps = 1e-9
    p_prob = softmax(p_logits)
    q_prob = softmax(q_logits)
    # 添加的eps = 1e-9 是为了防止log(0)
    P_log_prob = torch.log(p_prob + eps)
    q_log_prob = torch.log(q_prob + eps)
    kl_div = p_prob * (P_log_prob - q_log_prob)
    return torch.sum(kl_div, dim=-1)

x = torch.randn(2,3)
y = torch.randn(2,3)
print("x:\n",x, "\ny:\n",y)
print(kl(x,y))