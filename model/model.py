import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


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


# YaRN 
# end表示测试时的token长度上限
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    # 初始化RoPE频率(旋转角度theta)freqs    公式：freqs=1/base^(2i/d)
    # torch.arange(0, dim, 2) 从 0 到 dim，每隔 2 取一个，因为rope是两两一对计算角度
    # [: dim // 2]只是确保长度刚好是dim / 2，其实不写也行
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0

    # rope_scaling判断是否进入Yarn
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        # 从rope_scaling字典里读取YaRN的超参数. 
        # orig_max训练是最长上下文长度，模型见过的最大长度
        # factor 拓展倍数，比如训练时2048，测试时4096，拓展倍数就是2
        # beta_fast, beta_slow 划分高频/低频区
        # attn_factor 控制 attention 强度（softmax 尺度修正）
        # .get("factor", 16) 字典里有"factor"就用字典的值，没有就用默认值16
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )

        # 推理长度> 训练长度，需要外推
        if end / orig_max > 1.0:
            # 给定一个波长 b，需要反推出它对应 RoPE 的维度位置i。因为Yarn按波长划分高频低频，划分结束后计算要落实到维度上，所以波长->维度
            # i小就是高频区变化快；i大就是低频区
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))

            # 划分高低纬度
            # low 不需要缩放的高频部分的终点
            # high 需要缩放的低频部分的起点
            # 整个[0,dim//2]被划分为三段: [0, low)不需要缩放的高频, [low, high]中频, (high, dim//2]需要缩放的低频
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)

            # 计算缩放因子ramp
            # [0, low)中ramp=0(不缩放)，[low, high]过渡，(high, dim//2]中ramp=1(缩放)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)

            # 当ramp=0时（高频）：系数为1，保持原频率不变
            # 当ramp=1时（低频）：系数为1/factor，factor是训练长度/测试长度，实现线性插值缩放
            # ramp在0~1之间，平滑过渡
            freqs = freqs * (1 - ramp + ramp / factor)

    # 根据end生成位置索引t [0,1,2,……]
    t = torch.arange(end, device=freqs.device)
    # 计算外积，将t和频率(旋转的单位角度)相乘，得到每个位置的旋转角度
    freqs = torch.outer(t, freqs).float()
    # 因为torch.cos(freqs)后维度为(seq_len, dim/2)，而Q K的维度是(..., dim)，维度不匹配
    # 又因为RoPE是两两一对共用一组相对位置，所以把torch.cos(freqs)复制两份即可
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

