import torch
import torch.nn.functional as F

from model.model import RMSNorm


def test_matches_torch_rms_norm():
    """和 PyTorch 原生 F.rms_norm 对拍，误差必须 < 1e-6"""
    x = torch.randn(2, 4, 16)
    mine = RMSNorm(16)(x)
    ref = F.rms_norm(x, (16,), torch.ones(16), 1e-6)
    assert (mine - ref).abs().max().item() < 1e-6


def test_bf16_keeps_dtype_and_finite():
    """bf16 输入必须回 bf16，且不能出 nan（这就是"先转 float32 算"的原因）"""
    x = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    out = RMSNorm(16)(x)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


def test_zero_input_no_nan():
    """全零输入不能除零出 nan，eps 就是干这个的"""
    x = torch.zeros(1, 1, 16)
    out = RMSNorm(16)(x)
    assert torch.isfinite(out).all()
