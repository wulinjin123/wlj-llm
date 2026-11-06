"""从零实现的小型 LLM。所有组件都在 model.py 里，按 S1->S8 逐步填。"""

from model.model import (
    Attention,
    Block,
    CausalLM,
    FeedForward,
    Model,
    ModelConfig,
    RMSNorm,
    apply_rotary_pos_emb,
    precompute_freqs_cis,
    repeat_kv,
    rotate_half,
)

__all__ = [
    "Attention", "Block", "CausalLM", "FeedForward", "Model", "ModelConfig",
    "RMSNorm", "apply_rotary_pos_emb", "precompute_freqs_cis", "repeat_kv", "rotate_half",
]
