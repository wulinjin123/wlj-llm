import torch

from model.model import Attention, ModelConfig, precompute_freqs_cis


def _cfg():
    c = ModelConfig()
    c.num_hidden_layers = 2
    return c


def test_kv_cache_equals_full_forward():
    """逐步解码（带 cache）必须等于整段一次前向 —— 本节最硬的判据"""
    cfg = _cfg()
    attn = Attention(cfg).eval()
    x = torch.randn(1, 8, cfg.hidden_size)
    cos, sin = precompute_freqs_cis(cfg.head_dim, 32, cfg.rope_theta)

    with torch.no_grad():
        full, _ = attn(x, (cos[:8], sin[:8]))
        outs, pkv = [], None
        for t in range(8):
            o, pkv = attn(x[:, t:t + 1], (cos[t:t + 1], sin[t:t + 1]), pkv, use_cache=True)
            outs.append(o)
        step = torch.cat(outs, dim=1)

    assert (full - step).abs().max().item() < 1e-5


def test_gqa_shapes():
    """输出回 hidden_size；present 里的 K/V 头数是 n_kv_heads 不是 n_heads"""
    cfg = _cfg()
    attn = Attention(cfg)
    x = torch.randn(2, 5, cfg.hidden_size)
    cos, sin = precompute_freqs_cis(cfg.head_dim, 32, cfg.rope_theta)
    with torch.no_grad():
        out, present = attn(x, (cos[:5], sin[:5]), use_cache=True)
    assert out.shape == (2, 5, cfg.hidden_size)
    assert present[0].shape == (2, 5, cfg.num_key_value_heads, cfg.head_dim)


def test_qkv_param_count():
    """精确核对参数量：多一个 bias 或少一个 q_norm 都会被这条抓出来"""
    cfg = _cfg()
    attn = Attention(cfg)
    n = sum(p.numel() for p in attn.parameters())
    h, hd = cfg.hidden_size, cfg.head_dim
    expected = (h * cfg.num_attention_heads * hd
                + 2 * h * cfg.num_key_value_heads * hd
                + cfg.num_attention_heads * hd * h
                + 2 * hd)                        # 末项 = q_norm.weight + k_norm.weight
    assert n == expected


def test_no_attention_to_future():
    """因果性：改动第 5 个 token，前 5 个位置的输出不能变"""
    cfg = _cfg()
    attn = Attention(cfg).eval()
    x = torch.randn(1, 8, cfg.hidden_size)
    cos, sin = precompute_freqs_cis(cfg.head_dim, 32, cfg.rope_theta)
    with torch.no_grad():
        a, _ = attn(x, (cos[:8], sin[:8]))
        x2 = x.clone()
        x2[:, 5:] += 1.0
        b, _ = attn(x2, (cos[:8], sin[:8]))
    assert (a[:, :5] - b[:, :5]).abs().max().item() < 1e-5
