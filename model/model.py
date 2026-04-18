from transformers import PretrainedConfig


# huggingface的一个类，继承模型参数，可以传入HuggingFace
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

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
        # 因为最后x*weight，所以weight的维度必须与x的最后一维dim相同
        self.weight = nn.Parameter(torch.ones(dim))

    # _表示仅在类内使用的函数
    def _norm(self, x):
        # mean(-1)表示对x的最后一维dim取均值。因为要计算单个token内的特征均值，所以只能对dim做均值
        # keepdim=True让最终均值的维度与x相同(batch, seq_len,1)，不加这一项就变成(batch, seq_len)
        return torch.rsqrt(x.pow(2).mean(-1, keepdim = True)+self.eps)*x
    
    def forward(self, x):
        # x.float()是为了提高计算精度(float32)
        return self.weight*self._norm(x.float()).type_as(x)