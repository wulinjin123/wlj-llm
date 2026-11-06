"""项目入口 / 进度自检。

随时跑这个文件，它会告诉你现在走廊到了哪一步：

    python main.py

每一步都是"能跑就往下走，跑不了就报具体缺哪一节"，
所以它也可以当作你每写完一节 model/model.py 之后的验收开关。
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS, FAIL = "  [OK]  ", "  [--]  "


def step(num: str, desc: str, fn):
    """跑一个检查项，失败就打印原因并返回 False。"""
    try:
        info = fn()
        print(f"{PASS}S{num} {desc}" + (f"  ->  {info}" if info else ""))
        return True
    except NotImplementedError as exc:
        print(f"{FAIL}S{num} {desc}  ->  未实现：{exc}")
        return False
    except Exception as exc:
        print(f"{FAIL}S{num} {desc}  ->  {type(exc).__name__}: {exc}")
        return False


def main():
    print("=" * 66)
    print("tiny-llm 自检")
    print("=" * 66)
    print(f"  torch      {torch.__version__}")
    print(f"  cuda       {torch.cuda.is_available()}"
          + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    if torch.cuda.is_available():
        print(f"  bf16       {torch.cuda.is_bf16_supported()}")
    print("-" * 66)

    from model.model import ModelConfig, RMSNorm

    def s1():
        x = torch.randn(2, 4, 16)
        out = RMSNorm(16)(x)
        ref = torch.nn.functional.rms_norm(x, (16,), torch.ones(16), 1e-6)
        return f"与 F.rms_norm 最大误差 {(out - ref).abs().max().item():.2e}"

    def s2():
        from model.model import precompute_freqs_cis
        cos, sin = precompute_freqs_cis(96, 64, 1e6)
        return f"cos/sin 形状 {tuple(cos.shape)} {tuple(sin.shape)}"

    def s3():
        from model.model import Attention
        cfg = ModelConfig(num_hidden_layers=1)
        attn = Attention(cfg).eval()
        from model.model import precompute_freqs_cis
        cos, sin = precompute_freqs_cis(cfg.head_dim, 32, cfg.rope_theta)
        x = torch.randn(1, 8, cfg.hidden_size)
        with torch.no_grad():
            full, _ = attn(x, (cos[:8], sin[:8]))
            outs, pkv = [], None
            for t in range(8):
                o, pkv = attn(x[:, t:t + 1], (cos[t:t + 1], sin[t:t + 1]), pkv, use_cache=True)
                outs.append(o)
            err = (full - torch.cat(outs, dim=1)).abs().max().item()
        return f"逐步解码 vs 整段前向 误差 {err:.2e}"

    def s7():
        from model.model import Model
        m = Model(ModelConfig(num_hidden_layers=1)).eval()
        with torch.no_grad():
            h, _ = m(torch.randint(0, 6400, (1, 5)))
        return f"hidden 形状 {tuple(h.shape)}"

    def s8():
        from model.model import CausalLM
        m = CausalLM(ModelConfig()).eval()
        n = sum(p.numel() for p in m.parameters()) / 1e6
        ids = torch.randint(0, 6400, (1, 16))
        with torch.no_grad():
            logits, loss, _ = m(ids, labels=ids)
        t0 = time.time()
        with torch.no_grad():
            out = m.generate(ids[:, :4], max_new_tokens=16, use_cache=True)
        dt = time.time() - t0
        return (f"参数量 {n:.2f}M  loss {loss.item():.4f}  "
                f"生成 {out.shape[1] - 4} tok 用时 {dt:.2f}s")

    ok = [step("1", "RMSNorm", s1)]
    ok.append(step("2", "RoPE 位置表", s2))
    ok.append(step("3", "GQA + KV Cache", s3))
    ok.append(step("7", "Model 前向", s7))
    ok.append(step("8", "CausalLM 全链路", s8))

    print("-" * 66)
    if all(ok):
        print("全部通过。可以开始训练了：")
        print("  python trainer/train_pretrain.py --data_path "
              r"D:\Projects\minimind\dataset\pretrain_20k.jsonl --epochs 1 --batch_size 8")
    else:
        print(f"还有 {ok.count(False)} 项未完成，打开 model/model.py 从第一个 [--] 开始填。")


if __name__ == "__main__":
    main()
