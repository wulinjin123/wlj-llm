import os
import sys

import pytest
import torch

from model.model import CausalLM, ModelConfig


def test_param_count_is_63_91m():
    """参数量必须精确对上 63.91M，否则说明某个维度或权重绑定错了"""
    m = CausalLM(ModelConfig())
    n = sum(p.numel() for p in m.parameters()) / 1e6
    assert abs(n - 63.91) < 0.05, f"实际 {n:.2f}M"


def test_tied_embedding_shares_one_tensor():
    """权重绑定必须是同一个对象的两个引用，不能是两份拷贝"""
    m = CausalLM(ModelConfig())
    assert m.lm_head.weight is m.model.embed_tokens.weight


def _tiny_cfg():
    return ModelConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=2,
                       num_key_value_heads=1, head_dim=32, intermediate_size=128,
                       vocab_size=128)


def test_loss_skips_ignore_index():
    """全是 -100 的样本 loss 也得是有限值，不能是 nan"""
    cfg = _tiny_cfg()
    m = CausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    labels = ids.clone()
    labels[:, :6] = -100
    with torch.no_grad():
        _, loss, _ = m(ids, labels=labels)
    assert loss is not None and torch.isfinite(loss)


def test_forward_shapes():
    cfg = _tiny_cfg()
    m = CausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 5))
    with torch.no_grad():
        logits, _, presents = m(ids, use_cache=True)
    assert logits.shape == (2, 5, cfg.vocab_size)
    assert presents[0][0].shape == (2, 5, cfg.num_key_value_heads, cfg.head_dim)


def test_generate_extends_sequence():
    """生成至少能多吐 token，且不崩"""
    cfg = _tiny_cfg()
    m = CausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    out = m.generate(ids, max_new_tokens=5, top_k=10, top_p=0.9, eos_token_id=None)
    assert out.shape[1] == 9


# ---------------------------------------------------------------
# 与官方实现的键名对齐检查（需要本机有 minimind 仓库，没有就跳过）
# ---------------------------------------------------------------
MINIMIND = os.environ.get("MINIMIND_REPO", r"D:\Projects\minimind")


@pytest.mark.skipif(not os.path.isdir(MINIMIND), reason="本机没有 minimind 仓库")
def test_state_dict_keys_match_official():
    """我的 state_dict 键名必须和官方 MiniMindForCausalLM 完全一致。

    这条比"logits 对齐"更容易跑（不需要权重文件），
    而且它直接决定了以后能不能 strict=True 加载官方权重。

    注意：本项目也有个 model 包，和 minimind 的 model 包重名，
    所以不能用 sys.path + import，必须按文件路径单独加载。
    """
    import importlib.util

    path = os.path.join(MINIMIND, "model", "model_minimind.py")
    spec = importlib.util.spec_from_file_location("mm_official", path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)

    mine = set(CausalLM(ModelConfig()).state_dict().keys())
    ref = set(official.MiniMindForCausalLM(official.MiniMindConfig(
        hidden_size=768, num_hidden_layers=8, num_attention_heads=8,
        num_key_value_heads=4, head_dim=96, intermediate_size=2432,
        vocab_size=6400, max_position_embeddings=32768,
    )).state_dict().keys())

    missing = sorted(ref - mine)
    unexpected = sorted(mine - ref)
    assert not missing, f"缺这些 key（加载权重会报 missing）: {missing[:10]}"
    assert not unexpected, f"多了这些 key（会报 unexpected）: {unexpected[:10]}"
