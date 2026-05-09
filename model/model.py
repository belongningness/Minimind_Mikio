from transformers import PretrainedConfig
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

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    # 初始化rope的频率freqs，公式为\theta_i = 1/base^(2i/d)
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0

    # 需要用到YARN
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        # 需要从MokioMindConfig类中提取self.rope_scaling超参数
        # orig_max：模型训练时的最大上下文长度，作为判断当前训练长度是否大于最大可承受长度的临界值
        # factor：上下文拓展倍数，压缩倍数
        # beta_fast, beta_slow：高频低频不同的缩放速度
        # attn_factor：温度，用于对冲缩放倍数
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )

        # 当前prompt长度>最大训练长度，需要缩放
        if end / orig_max > 1.0:
            # YaRN用训练长度L和波长lambda之比b来区分高低频边界，划分结束后计算要落实到维度上，所以波长->维度
            # 转换计算公式见笔记，这里inv_dim表示高低频的维度边界i
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            # 划分高低维，
            # low表示不需要缩放的低频，<low完全不缩放；freq′=freq
            # high是需要缩放的高频，>high完全缩放；freq′=freq/factor
            # low high之间平滑过渡. freq′=freq⋅((1−α)+α​/factor)
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # 平滑过渡的权重α
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    # 为每个位置生成id
    t = torch.arange(end, device=freqs.device)
    # 旋转角度 = 位置 × 频率，rope定义
    freqs = torch.outer(t, freqs).float()
    # cat两次：两维一组旋转
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

# 开始rope旋转
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # rotate_half函数将(x1, x2) ->(-x2, x1)
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    # 第一项q * cos.unsqueeze(unsqueeze_dim)：q * cosθ = (q1 cosθ, q2 cosθ)
    # 第二项rotate_half(q) * sin.unsqueeze(unsqueeze_dim)： (-q2, q1) = (-q2 sinθ, q1 sinθ)
    # 两项相加得q' = (q1 cosθ - q2 sinθ, q2 cosθ + q1 sinθ)，符合上面的旋转公式
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

# n_rep：每个KV头需要重复的次数
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    # 重复次数为1，不需要操作
    if n_rep == 1: return x
    # x[:, :, :, None, :]   形状从 (bs, slen, num_kv_heads, head_dim) 变为：(bs, slen, num_kv_heads, 1, head_dim)
    # .expand(bs, slen, num_key_value_heads, n_rep, head_dim)  扩展第 3 维（原来是 1）到 n_rep 长度，不会复制数据。形状变为：(bs, slen, num_kv_heads, n_rep, head_dim)
    # .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)  将后两维压平，最终形状为：(bs, slen, num_kv_heads × n_rep, head_dim)
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    # config是之前定义过的配置对象，包含一些参数
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        # 定义KV头数，如果配置中没有单独指定 KV 头数，则默认等于 Q 头数
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        # Q头数，也是注意力头数
        self.n_local_heads = config.num_attention_heads
        # KV头数
        self.n_local_kv_heads = self.num_key_value_heads
        # 重复次数，即一对KV服务多少Q
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 头维度
        self.head_dim = config.head_dim
        self.is_causal = True

        # 投影层
        # QKV线性层属于输入Linear，输入维度都是config.hidden_size=512
        # q_proj 输出：num_attention_heads × head_dim
        # k_proj 输出：num_key_value_heads × head_dim
        # v_proj 输出：num_key_value_heads × head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 输出Linear（在softmax之后的那层Linear），所以输入输出维度和前面相反
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        # QK归一化，V不需要
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # 注意力矩阵的dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        # 输出后的dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        # 保存 dropout 概率值
        self.dropout = config.dropout
        # 磁盘IO的计算，算attention更快
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    # 投影计算QKV，通过view把输入拆分为多个头，计算时QK需要用RoPE
    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # bsz：batchsize; seq_len：序列长度; _ 隐藏层维度
        bsz, seq_len, _ = x.shape
        # 线性投影，x输入(bsz, seq_len, hidden_size)，输出xq：(bsz, seq_len, num_attention_heads × head_dim)
        # 输出 xk/xv：(bsz, seq_len, num_key_value_heads × head_dim)
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 重塑为多头格式，将最后一维拆分为 (n_heads, head_dim)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # QK归一化
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        # QK应用RoPE
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # KV cache
        # past_key_value 存储了之前所有 token 的 K 和 V，将历史 KV 与当前新的 KV 拼接
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        # 如果 use_cache=True，返回更新后的 KV 供下一轮使用
        past_kv = (xk, xv) if use_cache else None

        # transpose(1, 2): 转置：从 (bs, seq, heads, dim) → (bs, heads, seq, dim)
        # repeat_kv: 将 KV 头数拓展到与Q头数一致
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))

        # 优化计算
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        # 标准注意力写法
        else:
            # 计算注意力分数 QK^T/根号d
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 因果掩码，让每个位置只能看到当前及之前的位置，不能看到未来
            # torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1) 创建上三角矩阵（不含对角线），上三角用-inf填充
            # scores[:, :, :, -seq_len:] += 只修改最后 seq_len 个位置，因为可能拼接了历史 KV cache，历史部分已经处理过，不需要再次掩码
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            
            # 自定义掩码，屏蔽特定位置，一般是padding token（填充令牌） ，为了让同一批次中不同长度的序列能够对齐而添加的特殊占位符
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            
            # dim=-1: 在最后一个维度（key 维度）上做 softmax，将注意力转为概率分布
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        # 形状变化：(bsz, heads, seq_len, dim) → (bsz, seq_len, heads, dim)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # 输出层的linear + dropout正则化
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
    
class FeedForward(nn.Module):
    def __init__(self, config: MokioMindConfig, intermediate_size: int = None):
        super().__init__()
        # 定义升维的维度
        if config.intermediate_size is None:
            # config.hidden_size*8/3是经验值
            intermediate_size = int(config.hidden_size*8/3)
            # 最终升维的维度，向上对齐到64的倍数
            config.intermediate_size = 64*((intermediate_size+64-1)//64)

        # 门控，和up_proj一样，两者并行的
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # 降维，从config.intermediate_size降到config.hidden_size
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        # 升维，从原本的config.hidden_size升到config.intermediate_size
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # dropout
        self.dropout = nn.Dropout(config.dropout)
        # SiLU激活函数
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 根据FFN的流程图看
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))