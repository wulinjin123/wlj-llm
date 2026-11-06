"""从零手敲的 MiniMind 同构模型（63.91M）。

这个文件是**逐步长出来的**，按 S1 -> S8 的顺序一节一节填：
    S1 RMSNorm      -> S2 RoPE        -> S3 GQA 注意力
    S5 SwiGLU FFN   -> S6 Block       -> S7 Model   -> S8 CausalLM

S1~S8 八节都已经填完，每节的验收方式：
    python -m pytest tests -q

============================ 硬性约束 ============================
1. 参数名必须和官方实现完全一致，否则 S8 加载权重会报 key 不匹配。
   已固定的名字（不要改名）：
     embed_tokens / layers / norm / lm_head
     self_attn.{q,k,v,o}_proj / self_attn.{q,k}_norm
     input_layernorm / post_attention_layernorm
     mlp.{gate,up,down}_proj
2. 不许把 `freqs_cos` / `freqs_sin` 存进 state_dict（用 persistent=False）。
3. 前向签名不要改：Model.forward / CausalLM.forward 的参数顺序被测试和训练脚本依赖。
=================================================================
"""

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch  # type: ignore[import-not-found]
import torch.nn.functional as F  # type: ignore[import-not-found]
import torch.nn as nn
from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.init as init

# ==============================================================
# S0  超参配置（已给好，理解即可，不用改）
# ==============================================================
@dataclass
class ModelConfig(PretrainedConfig):
    """63.91M 版本的默认值。改任何一个都会导致参数量对不上官方权重。

    继承 `PretrainedConfig` 而不是普通 dataclass：`CausalLM` 继承自
    `PreTrainedModel`，它的 `__init__` 要求 `isinstance(config, PretrainedConfig)`，
    否则直接抛 TypeError。

    `@dataclass` 保留着，但**真正生效的是下面手写的 __init__**（dataclass 的注解
    只当作字段文档）。原因是 PretrainedConfig 的 __init__ 是 keyword-only，
    并且会覆盖 vocab_size 这类标准字段，所以这些值必须通过 super() 传进去。
    """

    model_type = "minimind"

    hidden_size: int = 768  # 模型宽度 d
    num_hidden_layers: int = 8  # Block 层数
    num_attention_heads: int = 8  # Q 头数
    num_key_value_heads: int = 4  # K/V 头数（GQA：8 个 Q 头共用 4 组 K/V）
    head_dim: int = 96  # 每个头的维度（注意 96*8=768，正好等于 hidden）
    intermediate_size: int = 2432  # SwiGLU 中间维
    vocab_size: int = 6400
    max_position_embeddings: int = 32768
    rope_theta: float = 1e6  # 位置编码基频，越大越能外推长文本
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True  # 输入 embedding 和输出 lm_head 共享权重
    rope_scaling: dict | None = None  # 传 {"type":"yarn", "factor":16, ...} 就启用 YaRN
    flash_attn: bool = True  # 是否允许走 SDPA 快路径
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    use_moe:bool=False
    num_experts:int=4
    num_experts_per_tok:int=1#top-k激活专家数单token
    moe_intermediate_size:int=2432#单个专家中间维（和ffn层一样大）
    norm_topk_prob:bool=True#topk权重是否归一化到一
    seq_aux:bool=False#aux_loss是按batch算还是按序列算
    router_aux_loss_coef:float=5e-4#负载均衡损失系数alpha
    n_shared_experts:int=0#共享专家个数，0=不用（minimind 官方不带共享专家）

    


    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        head_dim: int = 96,
        intermediate_size: int = 2432,
        vocab_size: int = 6400,
        max_position_embeddings: int = 32768,
        rope_theta: float = 1e6,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        hidden_act: str = "silu",
        tie_word_embeddings: bool = True,
        rope_scaling: dict | None = None,
        flash_attn: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        use_moe:bool=False,
        num_experts:int=4,
        num_experts_per_tok:int=1,#top-k激活专家数单token
        moe_intermediate_size:int=2432,#单个专家中间维（和ffn层一样大）
        norm_topk_prob:bool=True,#topk权重是否归一化到一
        seq_aux:bool=False,#aux_loss是按batch算还是按序列算
        router_aux_loss_coef:float=5e-4,#负载均衡损失系数alpha
        n_shared_experts:int=0,#共享专家个数，0=不用
    **kwargs,
    ):
        # ---- 非标准字段：PretrainedConfig 不认识，自己挂上 ----
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.rope_scaling = rope_scaling
        self.flash_attn = flash_attn
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.seq_aux = seq_aux
        self.router_aux_loss_coef = router_aux_loss_coef
        self.n_shared_experts = n_shared_experts

        # ---- 标准字段：必须交给父类，否则会被它覆盖成 None ----
        super().__init__(
            vocab_size=vocab_size,
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )


# ==============================================================
# S1  RMSNorm
# ==============================================================
class RMSNorm(nn.Module):
    """对最后一维做归一化，但**不减均值**，只除以均方根，再接一个可学习缩放。

        y = x / sqrt(mean(x^2) + eps) * weight

    和 LayerNorm 的差别：省掉减均值那一步，也就省掉一份 bias，
    在 LLM 里效果一样但更快。

    实现要点：
      - 先转 float32 算，算完再转回原 dtype。bf16 直接算会掉精度甚至出 nan。
      - 用 torch.rsqrt 而不是 1/torch.sqrt。
      - weight 初始化为全 1（是 Parameter，不是 buffer）。

    验收：python -m pytest tests/test_norm.py -q
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * y).type_as(x)


# ==============================================================
# S2  RoPE（旋转位置编码）
# ==============================================================


def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    # 48 个频率：rope_base^(-2i/dim)。i 小 -> 频率高 -> 转得快 -> 只管近距离
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0  # 幅度补偿系数，只有 YaRN 会改它

    # ---- YaRN 分支：可选，测试传的是 rope_scaling=None，会整段跳过 ----
    if rope_scaling is not None:
        orig_max = rope_scaling.get(
            "original_max_position_embeddings", 2048
        )  # 训练时的原始窗口
        factor = rope_scaling.get("factor", 16)  # 外推倍数
        beta_fast = rope_scaling.get("beta_fast", 32.0)  # 高频侧边界系数
        beta_slow = rope_scaling.get("beta_slow", 1.0)  # 低频侧边界系数
        attn_factor = rope_scaling.get("attention_factor", 1.0)  # 整体幅度补偿

        if end / orig_max > 1.0:  # 只有真的超出训练长度才需要外推
            inv_dim = lambda b: (
                (dim * math.log(orig_max / (b * 2 * math.pi)))
                / (2 * math.log(rope_base))
            )
            low = max(math.floor(inv_dim(beta_fast)), 0)  # ramp 起点（高频不压）
            high = min(
                math.ceil(inv_dim(beta_slow)), dim // 2 - 1
            )  # ramp 终点（低频全压）
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )  # 0~1 线性过渡，防除零
            freqs = freqs * (1 - ramp + ramp / factor)  # 低频压缩、高频保留

    t = torch.arange(end, device=freqs.device)  # 位置序列 0,1,2,...,end-1
    freqs = torch.outer(
        t, freqs
    ).float()  # 外积 -> (end, dim/2)，每行是一个位置的角度表
    # cat 两遍的原因：half-split 布局下第 i 维配第 i+dim/2 维，48 个频率要铺成 96 维才能逐维相乘
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def rotate_half(x):
    d = x.shape[-1] // 2  # 最后一维的一半：96 -> 48
    return torch.cat(
        (-x[..., d:], x[..., :d]), dim=-1
    )  # [x1,x2] -> [-x2,x1]，负号在前半段


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos, sin = (
        cos.unsqueeze(unsqueeze_dim),
        sin.unsqueeze(unsqueeze_dim),
    )  # (t,d) -> (t,1,d) 才能广播
    q_embed = (q * cos) + (
        rotate_half(q) * sin
    )  # 二维旋转公式：x' = x·cos + rot(x)·sin
    k_embed = (k * cos) + (rotate_half(k) * sin)  # k 同样处理
    return q_embed, k_embed


# ==============================================================
# S3  GQA 注意力（含 KV Cache）
# ==============================================================
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把 K/V 的头数从 n_kv_heads 复制到 n_heads，好和 Q 做点积。

    输入 (b, t, h_kv, d) -> 输出 (b, t, h_kv*n_rep, d)

    不能用 `repeat`（会真的复制内存），要用 `expand` + `reshape`：
    expand 只改 stride 不占内存，reshape 才在需要时真正展开。
    """

    bsz, seq_len, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bsz, seq_len, num_kv_heads, n_rep, head_dim)
        .reshape(bsz, seq_len, num_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """多头注意力 + GQA + KV Cache + QK-Norm。

    结构顺序（一步都不能换）：
        x -> q/k/v 投影 -> 拆头 -> QK-Norm -> RoPE -> 拼 cache
          -> 重复 K/V 到 Q 的头数 -> 注意力 -> 合并头 -> o_proj

    几个容易错的地方：
      1. q 的输出维度是 n_heads*head_dim，k/v 是 n_kv_heads*head_dim —— 两者不等，这是 GQA 的全部含义。
      2. QK-Norm 作用在 head_dim 上（96 维），不是 hidden_size（768 维），而且在 RoPE **之前**。
      3. 有 cache 时 seq_len == 1，此时**不能**用 is_causal=True（会变成只看自己），
         要改走手写 score 的分支，并且只能给最后 seq_len 列加因果 mask。
      4. present 返回的是**拼接后**的完整 K/V，不是本步新增的那一小段。

    参数名（必须一致）：q_proj k_proj v_proj o_proj q_norm k_norm

    验收：python -m pytest tests/test_attention.py -q
        其中"逐步解码 == 整段前向"是本节最硬的判据。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()  # 768 -> 96*8
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = config.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim  # 用配置里的值，不要用 hidden//heads 二次推导
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, self.n_local_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_local_heads * self.head_dim, config.hidden_size, bias=False
        )  # 768 -> 768
        self.q_norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # 作用在 96 维，不是 768
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = (
            hasattr(F, "scaled_dot_product_attention")
            and getattr(config, "flash_attn", False)
        )  # 能不能走快路径

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # 投影，计算qkv
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 投影完之后输入拆分成多个头，用view
        # 注意 v 必须用 kv 头数：v_proj 的输出宽度是 4*96=384，按 8 头 view 会直接报错
        q = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        k = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        v = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # q、k 先做 QK-Norm（作用在 head_dim=96 上，且必须在 RoPE 之前）
        q = self.q_norm(q)
        k = self.k_norm(k)
        # view 出来是**新张量**，而后面拼 cache / repeat_kv 用的都是 xv，
        # 所以必须把它接回去。少了这一行，xv 还是 (b,t,384) 的 3 维投影结果。
        xv = v

        # k,q使用RoPE，
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(q, k, cos[:seq_len], sin[:seq_len])
        # k和v使用kv Cache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 换布局 + 把 K/V 头复制到和 Q 一样多。
        # 这一步**不能**放进上面的 if 里：训练时永远没有 cache，
        # 少了 transpose 就会拿 (b,t,h,d) 去做 q@k.T，算出 (b,t,h,h)，静默全错。
        kv_len = xk.size(1)
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        if self.flash and seq_len > 1 and past_key_value is None:
            if attention_mask is None:
                output = F.scaled_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                )
            else:
                # 训练时 batch 里有 padding：把因果 mask 和 padding mask 合成一个 bool 矩阵。
                # 不能同时给 attn_mask 和 is_causal（两者语义互斥，会 RuntimeError）。
                keep = attention_mask.view(bsz, 1, 1, kv_len).bool()
                causal = torch.ones(
                    seq_len, kv_len, dtype=torch.bool, device=xq.device
                ).tril()
                output = F.scaled_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    attn_mask=keep & causal,
                    dropout_p=self.dropout if self.training else 0.0,
                )

        # 两个 mask 是**叠加在一起**加到 scores 上的，两个约束同时生效：
        # 不能看未来（因果）
        # 不能看 padding（pad 屏蔽）
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal:
                # 只屏蔽 j > (kv_len-seq_len)+i 的位置。有 cache 时 kv_len 比 seq_len 长，
                # 所以 mask 必须按 (seq_len, kv_len) 造，不能写成 (seq_len, seq_len)
                scores = scores + torch.triu(
                    torch.full(
                        (seq_len, kv_len), float("-inf"), device=scores.device
                    ),
                    diagonal=kv_len - seq_len + 1,
                ).unsqueeze(0).unsqueeze(0)
                if attention_mask is not None:
                    am = attention_mask
                    if am.size(1) != kv_len:
                        am = am[:, -kv_len:]
                    scores += (1.0 - am.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        # 进行attention计算，然后输出投影
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))  # 代表这个分支应用在残差网络
        return output, past_kv


# ==============================================================
# S5  SwiGLU 前馈网络
# ==============================================================
class FeedForward(nn.Module):
    """down( silu(gate(x)) * up(x) )

    为什么是三矩阵而不是"升维-GELU-降维"两矩阵：
      SwiGLU 多一路门控，表达力更强，代价是多一份参数。
      为了参数量不失控，中间维用 2432 而不是 4*768=3072。

    参数名（必须一致）：gate_proj up_proj down_proj
    """

    # 初始化
    # 升维
    # 降维
    # 门控
    # dropout
    # 激活函数

    def __init__(self, args: ModelConfig, intermediate_size: int | None = None):
        super().__init__()
        # 专家可以传更小的中间维；不传就用全局那份（默认两者都是 2432）
        mid = intermediate_size if intermediate_size is not None else args.intermediate_size
        self.up_proj = nn.Linear(
            args.hidden_size, mid, bias=False
        )  # 升
        self.down_proj = nn.Linear(
            mid, args.hidden_size, bias=False
        )  # 降维
        self.gate_proj = nn.Linear(
            args.hidden_size, mid, bias=False
        )  # 门控
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self, x):
        # 先各自投影 -> 只对 gate 激活 -> 逐元素相乘 -> 降维。顺序错了不报错但效果差
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
# ==============================================================
# 新增 ：MOE
# ==============================================================
class MOE(nn.Module):
    """路由层：给每个 token 对每个专家打分，选出 top-k。

    三个输出（注意顺序是 weight 在前，docstring 之前写反了）：
        topk_weight  (B*S, k)  对应的权重，加权求和时用
        topk_idx     (B*S, k)  每个 token 选中哪几个专家（编号）
        aux_loss     ()        负载均衡损失，训练时加进总 loss
    """

    def __init__(self, args: ModelConfig):
        super().__init__()
        self.top_k = args.num_experts_per_tok        # k：每个 token 用几个专家
        self.n_routed_experts = args.num_experts     # E：专家总数
        self.alpha = args.router_aux_loss_coef       # aux_loss 的系数
        self.seq_aux = args.seq_aux                  # True=按单条序列算，False=按整个 batch 算
        self.norm_topk_prob = args.norm_topk_prob    # k>1 时把权重归一化
        self.scoring_func = "softmax"                # 打分函数，目前只有 softmax
        # 路由矩阵 (E, H)。等价于 nn.Linear(H, E, bias=False)，但写成 Parameter 方便手写 F.linear
        self.weight = nn.Parameter(
            torch.empty(self.n_routed_experts, args.hidden_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


    def reset_parameters(self):
        # 初始化权重
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        # moe 时候只关心 token 值，不关心具体位置，所以我们合并 batch 和 seq_len 的维度
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        # linear 映射计算每个 token 对于各个 expert 的 logits(打分)
        logits = F.linear(hidden_states, self.weight, None)
        # 合并之后，对权重进行线性映射，计算出每个 token 对于各个 experts 的打分，使用 softmax 函数对打分进行归一化
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f"scoring_func not implemented: {self.scoring_func}")

        # 第一种方法，序列级别的 aux_loss
        # 第二种方法，batch 级别的 aux_loss
        topk_weight, topk_idx = torch.topk(
            scores, self.top_k, dim=-1, sorted=False
        )  # [bsz*seq_len, top_k]
        # 权重归一化：对选择的专家权重进行标准化，目的：确保每个 token 对多个专家的权重之和为一，避免权重累计过大
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 第二步计算辅助损失，确保负载均衡
        # 判断是在训练模式下并且 alpha 大于零，才计算
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores  # 保存原始分数用于计算辅助损失
            aux_topk = self.top_k
            # 将 topk_idx 从 [bsz*seq_len, top_k] 恢复回 [bsz, seq_len, top_k]
            topk_idx_for_aux_loss = topk_idx.view(bsz, seq_len, -1)

            # 序列级别辅助损失 seq_aux
            if self.seq_aux:
                # 回复 scores 维度: bsz*seq_len, n_experts -> bsz, seq_len, n_experts
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # 统计每个 batch 中每个专家被选中的次数，ce; {bsz, n_routed_experts}: 表示每个专家被选中的次数
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                # scatter_add：根据 topk_idx_for_aux_loss 中的值，将 1 累加到 ce 中对应的位置
                # 例如：如果 topk_idx_for_aux_loss[0] = [1, 3], 则 ce[0][1] = 1, ce[0][3] = 1
                ce.scatter_add_(
                    1,#dim维度，在专家e上累加
                    topk_idx_for_aux_loss.reshape(bsz, -1),#被选择的专家下标
                    torch.ones(
                        bsz, seq_len * aux_topk, device=hidden_states.device#负责计数加一
                    ),
                )   
                # 计算每个专家的平均使用率
                # ce 除以总的选择数，得到每个专家的使用比例
                ce = ce.div(seq_len * aux_topk / self.n_routed_experts)
                # 计算辅助损失，ce * scores_for_seq_aux.mean(dim=1) = 专家使用率乘以专家的平均分数（偏好），目的是让所有专家获得相似的使用率和分数
                aux_loss = (
                    (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean()
                    * self.alpha
                )
            # 第二种方法，batch 级别的 aux_loss
            else:
                # 将 top_idx 展平: [bsz, seq_len, top_k] -> [bsz*seq_len*top_k]
                # 转化为 one-hot 形式，每列上只有 0,1 得到 [bsz*seq_len*top_k, n_routed_experts]
                # 例如 n_route_experts = 4, top_idx_for_aux_loss 包含值 0-4，则 one-hot 形式会将对应的值转成 1，其余位置为 0
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                # ce: [n_routed_experts] 表示整个 batch 中每个专家的平均选择率
                # 比如 [0.2, 0.3, 0.2, 0.5] 表示每个专家被选中的概率
                ce = mask_ce.float().mean(0)
                # Pi: [n_routed_experts], 表示整个 batch 中每个专家的平均使用率
                Pi = scores_for_aux.mean(dim=0)
                # fi 标准化的专家选择次数（乘以 n_routed_experts 确保平衡）
                fi = ce * self.n_routed_experts
                # 辅助损失，Pi*fi 越接近于 1，说明专家分数和选择率越平衡
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0.0#推理时候或者alpha=0.0时，aux_loss=0.0，不关心
        return topk_weight, topk_idx, aux_loss

class MoeFeedForward(nn.Module):
    """E 个专家 + 1 个路由器。每个 token 只走 top-k 个专家，结果按权重相加。"""
    def __init__(self,config:ModelConfig):
        super().__init__()
        self.config=config
        self.top_k=config.num_experts_per_tok          # 原为 num_expeerts_per_tok（拼错）
        #专家层：每个专家是独立的一份 SwiGLU，中间维用 moe_intermediate_size
        self.experts=nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)          # 原为 config.n_routed_experts（不存在）
        ])
        #门控层：类名是 MOE，不是 MoEGate
        self.gate = MOE(config)
        #aux_loss 先给初值，推理时 Model 那边 getattr 就不会炸
        self.aux_loss = 0.0
        #共享专家
        if config.n_shared_experts>0:
            self.shared_experts= nn.ModuleList([
                FeedForward(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_shared_experts)
            ])


    def forward(self,x):
        #复制输入，便于共享专家使用
        identity=x
        #保持原始x的形状
        orig_shape=x.shape
        bsz,seq_len,h=orig_shape
        #使用门控机制选择专家，注意返回顺序是 (weight, idx, aux_loss)
        topk_weight,topk_idx,aux_loss=self.gate(x)
        #展开x以便处理
        x=x.view(-1,x.shape[-1])
        flat_topk_idx=topk_idx.view(-1)
        if self.training:
            #每个 token 复制 top_k 份，分别交给它选中的 k 个专家处理
            x=x.repeat_interleave(self.top_k,dim=0)
            y=torch.empty_like(x,dtype=torch.float32)
            #遍历所有专家，把指向自己的 token 一次算完
            for i,expert in enumerate(self.experts):
                y[flat_topk_idx==i]=expert(x[flat_topk_idx==i]).to(y.dtype)
            # ---- 加权求和：必须在 for 循环外做，且是「逐 token 乘各自权重再求和」 ----
            # y 现在按 [t0_e0, t0_e1, t1_e0, ...] 排好，view 成 (N, k, h)
            # topk_weight 是 (N, k)，unsqueeze 成 (N, k, 1) 后逐元素相乘，再对 k 维求和
            n_tok = topk_weight.shape[0]
            y = (y.view(n_tok, self.top_k, h) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape).to(identity.dtype)
        #如果是推理阶段
        else:
            y=self.moe_infer(x,flat_topk_idx,topk_weight.view(-1,1)).view(
                *orig_shape
            ).to(identity.dtype)
        if self.config.n_shared_experts>0:
            for expert in self.shared_experts:
                y=y+expert(identity)
        self.aux_loss=aux_loss
        return y
    @torch.no_grad()
    #MoE推理方法
    def moe_infer(self,x,flat_expert_indices,flat_expert_weights):
        #使用cache，创建一个和x形状相同的零张量（原为 torch.zero_like，不存在）
        expert_cache=torch.zeros_like(x)
        #对专家索引进行排序，最后是[0,0,0,0,1,1,1,2,2,2,...]，分拣打包

        idx=flat_expert_indices.argsort()

        tokens_per_expert=flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        #计算每个token对应的专家索引
        tokens_idxs=idx//self.config.num_experts_per_tok
        #对每个打包好的包进行处理
        for i,end_idx in enumerate(tokens_per_expert):
            #计算当前包的起始位置
            start_idx=0 if i==0 else tokens_per_expert[i-1]
            if start_idx==end_idx:
                continue
            #取出当前包对应的专家  
            expert=self.experts[i]
            #取出当前token对应的原始id
            exp_token_idx=tokens_idxs[start_idx:end_idx]
            #取出当前token对应的数据
            expert_tokens=x[exp_token_idx]
            #计算专家输出，一次性处理当前包的所有token
            #.to() 要传 dtype，传张量本身在旧版 torch 会报错/行为不一致
            expert_out=expert(expert_tokens).to(expert_cache.dtype)
            #加权
            expert_out.mul_(flat_expert_weights[idx[start_idx:end_idx]])
            #将结果散点加权到缓存中对应位置
            expert_cache.scatter_add_(
                0,exp_token_idx.view(-1,1).repeat(1,x.shape[-1]),expert_out
            )
        return expert_cache


# ==============================================================
# S6  Transformer Block（Pre-Norm 残差结构）
# ==============================================================
class Block(nn.Module):
    """x = x + attn( norm1(x) )
        x = x + mlp ( norm2(x) )

    Pre-Norm 的含义：norm 放在子层**之前**，残差通路上没有任何归一化。
    这是深层网络能训起来的关键——梯度可以沿残差直通到底。

    参数名（必须一致）：self_attn / input_layernorm / post_attention_layernorm / mlp
    """

    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.self_attn = Attention(config)
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp =MoeFeedForward(config) if config.use_moe else FeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value


# 输入的hidden_state，然后进行GQA attention计算然后残差处理再进过FFN层计算再残差处理


# ==============================================================
# S7  Model（embedding + N 层 Block + 输出 norm）
# ==============================================================
class Model(nn.Module):
    """参数名（必须一致）：embed_tokens / layers / norm

    两处细节：
      1. freqs_cos / freqs_sin 用 register_buffer(..., persistent=False) 注册，
         这样 `.to(device)` 会跟着走，但不会进 state_dict（否则加载官方权重会报 unexpected key）。
      2. 带 cache 时输入只有 1 个 token，但它的位置编码要取第 start 个位置，
         start = 已有 cache 的长度。取错位置会让生成结果完全不对，而且不报错。

    验收：python -m pytest tests/test_model.py -q
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.start_pos = 0
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_base = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [Block(i, config) for i in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # RoPE预计算
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,  # 必须是 head_dim=96，传 hidden_size 会生成 (32768,768) 的表
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        bsz, seq_len = input_ids.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None

        layer_past_key_values = (
            past_key_values
            if isinstance(past_key_values, (list, tuple))
            else [None] * len(self.layers)
        )

        start_pos = (
            layer_past_key_values[0][0].shape[1]
            if (layer_past_key_values and layer_past_key_values[0] is not None) 
            else 0
        )

        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_len],
            self.freqs_sin[start_pos : start_pos + seq_len],
        )
        present = []
        aux_loss = 0.0
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, layer_past_key_values)
        ):
            hidden_states, present_key_value = layer(
                hidden_states,
                position_embeddings,
                past_key_value,
                use_cache,
                attention_mask,
            )
            present.append(present_key_value)
            # MoE 每一层的负载均衡损失要汇总起来，否则路由器没有梯度约束、专家会塌缩到少数几个
            if self.config.use_moe:
                aux_loss = aux_loss + layer.mlp.aux_loss

        # 下面两行必须在 for 之外：norm 只对"全部层叠加完"的 hidden 做一次，
        # return 也是整个 forward 结束才返回。写在循环里等于只跑第 0 层就退出。
        hidden_states = self.norm(hidden_states)
        self.aux_loss = aux_loss
        return hidden_states, present


# ==============================================================
# S8  CausalLM（加 lm_head + loss + 采样生成）
# ==============================================================
class CausalLM(PreTrainedModel, GenerationMixin):
    """最外层封装。

    三件事：
      1. lm_head 把 hidden 映射到 vocab；开权重绑定后要和 embed_tokens **共用同一个
         Parameter 对象**（`a.weight = b.weight`，不是 copy），这样优化器只更新一份。
      2. loss：天然是 next-token prediction，所以 logits 要去掉最后一个位置、
         labels 要去掉第一个位置，错一位对齐；忽略 -100（padding / 非回答段）。
      3. generate：每步只喂新 token（配合 KV Cache），位置由 past_len 决定。

    验收：python -m pytest tests/test_model.py -q 且参数量必须 = 63.91M
    """

    config_class = ModelConfig

    def __init__(self, config: ModelConfig):
        self.config = config
        super().__init__(config)
        self.model = Model(config)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        # 权重共享，输出层和嵌入层权重共享
        self.model.embed_tokens.weight = self.lm_head.weight
        # 权重初始化：不加这一步，初始 loss 会是 600 多而不是 ln(6400)≈8.76
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """Linear / Embedding 统一初始化成 N(0, 0.02)，和官方 MiniMind 一致。

        nn.Embedding 默认是 N(0,1)，768 维点积后 hidden 幅度 ~√768≈27；
        lm_head 又和 embedding 共享权重，logits 再乘一次，标准差也到了 27 量级，
        交叉熵自然在几百。改成 0.02 之后 logits 标准差 ~0.55，loss≈8.9。
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self,input_ids:Optional[torch.Tensor]=None,
                attention_mask:Optional[torch.Tensor]=None,
                past_key_values: Optional[tuple[tuple[torch.Tensor, ...], ...]] = None,
                use_cache:bool=False,logits_to_keep:Union[int,torch.Tensor]=None,
                labels:Optional[torch.Tensor]=None,
                **kwargs):
       hidden_states,past_key_value=self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        **kwargs
       ) 
       logits=self.lm_head(hidden_states)

       # ---- loss：next-token prediction，所以 logits 去掉最后一位、labels 去掉第一位 ----
       loss=None
       if labels is not None:
           shift_logits=logits[:,:-1,:].contiguous()   # 位置 i 的输出
           shift_labels=labels[:,1:].contiguous()      # 去预测位置 i+1 的 token
           if (shift_labels!=-100).any():
               loss=F.cross_entropy(
                   shift_logits.view(-1,shift_logits.size(-1)),
                   shift_labels.view(-1),
                   ignore_index=-100,
               )
           else:
               # 整条样本都被忽略时 cross_entropy 会返回 nan，给一个恒为 0 的标量
               loss=logits.sum()*0.0
           # MoE 的负载均衡损失并进总 loss（只有训练时 hidden 的 aux_loss 才是张量）
           aux = getattr(self.model,"aux_loss",0.0)
           if loss is not None and isinstance(aux, torch.Tensor):
               loss = loss + aux.to(loss.device, loss.dtype)

       #logits to keep是整数，那就保留最后n个位置，生成时候只需要最后一个logit来预测下一个token
       #注意：else 分支必须给 slice(None)。原来的写法在这里是 None，
       #而 hidden_states[:, None, :] 在 PyTorch 里是"插入一维新维度"，不是"全选"
       slice_indices=(
           slice(-logits_to_keep,None)
           if isinstance(logits_to_keep,int) and logits_to_keep>0
           else slice(None)
       )
       logits=logits[:,slice_indices,:]
       return logits,loss,past_key_value

    @torch.no_grad()
    def generate(self,input_ids:torch.Tensor,max_new_tokens:int=32,
                 temperature:float=1.0,top_k:Optional[int]=None,
                 top_p:Optional[float]=None,eos_token_id:Optional[int]=None,
                 use_cache:bool=True,**kwargs)->torch.Tensor:
        """自回归生成（自带 KV Cache，覆盖 GenerationMixin.generate）。

        为什么自己写：GenerationMixin 的 generate 会去读 self.generation_config，
        传 eos_token_id=None 时并不代表"不停"，而是"用配置里的 eos"，
        会提前中断、返回长度不确定。这里规定 eos_token_id=None 就**不提前停止**，
        max_new_tokens 精确可控。
        """
        was_training=self.training
        self.eval()
        try:
            out=input_ids
            present=None
            for _ in range(max_new_tokens):
                # 第一步喂整段 prompt，之后每步只喂上一步新生成的 1 个 token
                cur=out if present is None else out[:,-1:]
                logits,_,present=self(cur,past_key_values=present,use_cache=use_cache)
                next_logits=logits[:,-1,:].float()
                next_id=self._sample(next_logits,temperature,top_k,top_p)
                out=torch.cat([out,next_id],dim=1)
                if eos_token_id is not None and bool((next_id==eos_token_id).all()):
                    break
            return out
        finally:
            if was_training:
                self.train()

    @staticmethod
    def _sample(logits,temperature=1.0,top_k=None,top_p=None):
        if temperature is None or temperature<=0:
            return logits.argmax(dim=-1,keepdim=True)
        logits=logits/max(temperature,1e-5)
        if top_k is not None and top_k>0:
            k=min(int(top_k),logits.size(-1))
            kth=torch.topk(logits,k,dim=-1).values[...,-1,None]
            logits=logits.masked_fill(logits<kth,float("-inf"))
        if top_p is not None and 0.0<top_p<1.0:
            sorted_logits,sorted_idx=torch.sort(logits,descending=True,dim=-1)
            probs=F.softmax(sorted_logits,dim=-1)
            cumulative=probs.cumsum(dim=-1)
            drop=(cumulative-probs)>=top_p      # 累积已超过 top_p 的尾部丢掉
            sorted_logits=sorted_logits.masked_fill(drop,float("-inf"))
            logits=torch.full_like(logits,float("-inf")).scatter(-1,sorted_idx,sorted_logits)
        probs=F.softmax(logits,dim=-1)
        return torch.multinomial(probs,num_samples=1)