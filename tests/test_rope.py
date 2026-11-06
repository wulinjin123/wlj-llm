import torch

from model.model import apply_rotary_pos_emb, precompute_freqs_cis


def _score(q_pos, k_pos, dim=96):
    """把 q 放在 q_pos 位置、k 放在 k_pos 位置，返回它们旋转后的内积。"""
    cos, sin = precompute_freqs_cis(dim, 256, 1e6)
    torch.manual_seed(0)
    q = torch.randn(1, 4, 1, dim)
    k = torch.randn(1, 4, 1, dim)
    qr, _ = apply_rotary_pos_emb(q, q, cos[q_pos:q_pos + 1], sin[q_pos:q_pos + 1])
    _, kr = apply_rotary_pos_emb(k, k, cos[k_pos:k_pos + 1], sin[k_pos:k_pos + 1])
    return (qr * kr).sum().item()


def test_relative_position_invariance():
    """整体平移、只保留相对距离 3 —— 内积必须几乎不变。这是 RoPE 存在的全部意义"""
    a, b = _score(0, 3), _score(100, 103)
    assert abs(a - b) / max(abs(a), 1e-6) < 1e-5


def test_different_distance_changes_score():
    """相对距离不同，内积应该明显不同（否则说明压根没转，只是原样乘了 cos/sin）"""
    a, b = _score(0, 3), _score(0, 30)
    assert abs(a - b) / max(abs(a), 1e-6) > 1e-2


def test_norm_preserved():
    """旋转是正交变换，向量长度不变。长度变了说明 cos/sin 拼错了"""
    cos, sin = precompute_freqs_cis(96, 64, 1e6)
    x = torch.randn(1, 2, 1, 96)
    xr, _ = apply_rotary_pos_emb(x, x, cos[5:6], sin[5:6])
    assert abs(xr.norm().item() - x.norm().item()) / x.norm().item() < 1e-5


def test_shape_and_dtype():
    """cos/sin 必须是 (end, dim) 而不是 (end, dim/2)，否则后面广播不上"""
    cos, sin = precompute_freqs_cis(96, 32, 1e6)
    assert cos.shape == (32, 96) and sin.shape == (32, 96)
