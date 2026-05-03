import torch

def softmax(logits, dim=-1):
    logits_max = torch.max(logits, dim=dim, keepdim=True).values
    logits_safe = logits - logits_max
    logits_exp = torch.exp(logits_safe)
    return logits_exp / torch.sum(logits_exp, dim=dim, keepdim=True)

def kl(p, q, dim=-1):
    p_prob = softmax(p)
    q_prob = softmax(q)
    log_p = torch.log(p_prob)
    log_q = torch.log(q_prob)
    kl_div = p_prob * (log_p - log_q)
    return torch.sum(kl_div, dim=dim, keepdim=True)

p = torch.randn(2,3)
q = torch.randn(2,3)
print(p, "\n", q)
print("\n", kl(p,q))