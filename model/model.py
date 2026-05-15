from transformers import PretrainedConfig
import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


# huggingface的一个类，继承模型参数，可以传入HuggingFace
class MiniMindConfig(PretrainedConfig):
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

# GQA
class Attention(nn.Module):
    # config是之前定义过的配置对象，包含一些参数
    def __init__(self, config: MiniMindConfig):
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
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
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

# MOE专家
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 输入维度：token数目，输出维度：专家个数
        # 给各token打分，token 对各 expert 的偏好
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # 建立多个独立的SiLUFFN作为专家
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        # 把[batch_size, seq_len, hidden_dim]拉平为[batch_size*seq_len, hidden_dim]，这样一行代表一个token便于打分
        x_flat = x.view(-1, hidden_dim)
        # 计算各token对各专家的打分，维度[num_tokens, num_experts]
        scores = F.softmax(self.gate(x_flat), dim=-1)
        # 对于每个token，只激活前config.num_experts_per_tok个专家
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        # 因为只选了部分expert，所以被选择的experts的概率之和不为1.因此归一化top-k概率，令他们和为1
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # y用于累计expert输出，初始化为0张量
        y = torch.zeros_like(x_flat)
        # 遍历每一个expert，i是编号，expert是一个FFN网络
        for i, expert in enumerate(self.experts):
            # 把妹选的expert mask掉
            mask = (topk_idx == i)
            if mask.any():
                # 筛选出需要当前expert_i的token索引
                token_idx = mask.any(dim=-1).nonzero().flatten()
                # 提取当前expert_i对应的权重(一列，包含所有用到expert_i的token的权重)
                weight = topk_weight[mask].view(-1, 1)
                # x_flat[token_idx]：对于当前的expert_i，提取所有用到它的token
                # expert(x_flat[token_idx])：用expert(FFN)网络去计算筛选的token
                # 因为最终token_j的输出=它所激活的各expert输出的带权平均值，所以expert处理完要*weight
                # 并且最后的乘积还要加回原来的token中，等所有的experts全部处理完了，自然每个token的加权结果也出来了
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())


        if self.training and self.config.router_aux_loss_coef > 0:
            # 强制负载均衡，MOE最怕所有的token只用expert0，其他expert闲置，白白浪费参数
            # load统计每个expert的负载率，expert_i利用率=expert_i被激活次数/token数量
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            # scores.mean(0)：根据各token对experts的打分，按列平均，得到每个expert在所有token中的平均得分，也就是expert偏好
            # load * scores.mean(0)：实际负载率和偏好不能同时集中在一个expert上，否则aux_loss将会很大
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

# 各模块衔接起来，组成完整的架构图
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states # 残差保存
        # GQA部分：先RMSNorm再GQA
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        # GQA还需要加上原始输入token，记为residual（残差连接）
        hidden_states += residual
        # FFN部分，直接调用FFN类即可，无需拆分步骤
        # 注意这里也要加未经FFN处理的数据，但是不再是原始数据residual了，而是GQA输出的结果
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

# MiniMindBlock是单层网络，这里组装多层网络
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # vocab_size：词表大小; num_hidden_layers：Transformer层数
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # 把 token id 映射成向量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # 训练时随机置零，防止过拟合
        self.dropout = nn.Dropout(config.dropout)
        # N层Transformer
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # RoPE预计算，计算cos(theta)  sin(theta)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        # 把计算好的freqs_cos, freqs_sin保存，但是不作为参数更新，值也不改变
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        # 当前 token 在整个序列中的起始位置
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        # (batch, seq, hidden_size)
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # buffer初始化丢失时，重新计算freqs_cos, freqs_sin
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        # [start_pos:start_pos + seq_length]只取当前处理片段的freqs_cos, freqs_sin，计算RoPE,避免浪费
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []

        # 遍历每层Transformer，past_key_values是每一层的KV cache（各层都有自己的kv cache）
        # zip后格式为[(Block0, (k0,v0)), (Block1, (k1,v1)), (Block2, (k2,v2)]
        for layer, past_key_value in zip(self.layers, past_key_values):
            # 开始执行当下层的MiniMindBlock，也就是当下层级的GQA
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            # past_key_values是旧的kvcache，而每层更新后的KV要存入present（追溯到Attention中的past_kv）
            presents.append(present)
        # 输出前最后的RMSnorm
        hidden_states = self.norm(hidden_states)
        # 执行MOE层，aux_loss总和
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss   

# 把自编写函数与huggingface结合，形成标准化模型
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    # 输入 embedding 和 输出 lm_head 共享权重
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        # 调用自写的model
        self.model = MiniMindModel(self.config)
        # 把hidden_states 映射成token logits
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        # 训练时需要所有 token logits， 但生成时只需要最后一个 token logits 例如"Hello world" 只需要最后的 "world" → next token
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        # 训练时计算loss
        if labels is not None:
            # logits表示可用于预测的token，label表示标准答案
            # logits.shape = (batch, seq_len, vocab_size)
            # logits[..., :-1, :]可用于预测的token序列，第t个token预测第t+1个token
            #           因为最后一个token是[EOS]不需要预测，所以把seq_len的最后一个元素裁掉
            # labels[..., 1:]：第一个token是已知的不需要被预测，所以裁掉第一个元素
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            # 把预测的结果x和标答y做交叉熵损失
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
            # 把模型输出的所有结果打包成一个“结构化对象”返回
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    # LLM模型推理文本生成，不断预测下一个token → 拼接到输入后面 → 再预测
    # 推理模式，关闭梯度、反向传播、autograd graph
    @torch.inference_mode()
    # max_new_tokens：最多生成多少 token
    # temperature：控制随机性，temperature小，更确定更保守；temperature 大，更随机更发散
    # top_k=50：只保留概率最大的50个token，其他直接禁掉
    # top_p=0.85：只保留累计概率达到85% 的 token
    # do_sample=True：随即采样
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        # .repeat(num_return_sequences, 1)复制num_return_sequences遍输入，便于生成多份答案
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        # 同样复制mask，生成多份答案的时候要用
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        # 从 kwargs 里取出 "past_key_values"，如果没有就返回 None
        past_key_values = kwargs.pop("past_key_values", None)
        # finished记录哪些样本已经生成结束，碰到EOS，此处在初始化
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        # 流式输出，边输出边打印
        if streamer: streamer.put(input_ids.cpu())

        # 每轮生成一个token
        for _ in range(max_new_tokens):
            # 已经缓存了多少token
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            # 因为KVcache已经记录了，所以每次只需要输入新增token，即input_ids[:, past_len:]，此处得到logits和new KV cache
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            # 每生成一个新 token，就把 attention_mask 也扩展一位；因为序列长度变长了mask 也必须同步变长
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            # 取最后一个 token logits，最后一个位置的 logits 恰好就是"下一个 token"的预测
            logits = outputs.logits[:, -1, :] / temperature
            # repetition_penalty避免重复生成
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    # seen：找到已经出现过的token，然后降低其概率（重复惩罚）
                    # score：取出“已经出现过token”的分数
                    # logits[i, seen]：如果 score > 0 则score /= penalty（正数更小）; 如果 score < 0 则score *= penalty（负数更负）
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            
            if top_k > 0:
                # torch.topk(logits, top_k)[0][..., -1, None]]取第k大的值
                # 整句用inf mask掉小于第k大的值的所有值
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')

            # 累计概率小于1
            if top_p < 1.0:
                # 根据概率降序排序
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                # 计算累计概率>top_p
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                # mask[..., 1:]=mask[..., :-1]: mask右移，相当于F F T T-> F F F T
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                # 前面sort()改变了token顺序,现在需要映射回原词表位置
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')

            # 采样next_token
            # torch.multinomial按概率随机采样。例如66%选第一个 24%选第二个 10%选第三个
            # 如果 do_sample=False，永远选概率最大
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            
            # 生成结束符EOS
            # 同时计算很多个batch，结束时间不一样，先结束的batch一直在后面补EOS，直到所有batch结束为止————为了保持batch间维度一致
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            # 把新 token 接到句子后面
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            # 保存最新的cache
            past_key_values = outputs.past_key_values if use_cache else None
            # 打字机效果输出token
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                # 识别EOS
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                # 结束
                if finished.all(): break
        # 如果全部结束
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids