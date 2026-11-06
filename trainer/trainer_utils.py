"""训练公用工具：路径、随机种子、学习率调度、日志、检查点。

这一份是"工具代码"，不需要手写，理解每个函数干什么就行。
唯一要注意的是最上面那段 sys.path 注入 —— 不加它，`python trainer/train_pretrain.py`
会因为找不到 model 包而报 ModuleNotFoundError（Python 只把脚本所在目录加进 sys.path）。
"""

import math
import os
import sys
import time
from pathlib import Path

import torch

# ---- 让子目录里的脚本也能 import 到项目根下的 model / dataset 包 ----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- 数据与分词器路径（本机已有的现成资源，不用再下载 3GB）----
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", r"D:\Projects\minimind\model")
DATA_DIR = os.environ.get("MINIMIND_DATA_DIR", r"D:\Projects\minimind\dataset")
OUT_DIR = os.environ.get("TINYLLM_OUT", str(ROOT / "out"))


def set_seed(seed: int = 42):
    """固定随机种子。多卡时 torch.manual_seed 会内部按 rank 偏移，不需要自己加 rank。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(step: int, total_steps: int, warmup_steps: int,
           base_lr: float, min_lr: float = 1e-5) -> float:
    """学习率调度：前 warmup_steps 线性升到 base_lr，之后余弦退到 min_lr。

    为什么要 warmup：训练初期权重随机、梯度方向噪声大，直接用大学习率
    容易把 embedding 打坏（loss 先降后暴涨）。warmup 让模型先"站稳"再加速。
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + (base_lr - min_lr) * cosine


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


class TrainLogger:
    """控制台打印 + （可选）swanlab 曲线。

    没装 swanlab 或不想联网时，传 use_swanlab=False 即可，不影响训练。
    """

    def __init__(self, out_dir: str, project: str = "tiny-llm", run_name: str = "pretrain",
                 use_swanlab: bool = True):
        self.start = time.time()
        self.swan = None
        if use_swanlab:
            try:
                import swanlab
                self.swan = swanlab.init(project=project, experiment_name=run_name,
                                         logdir=out_dir)
            except Exception as exc:                      # 联网失败不该中断训练
                print(f"[warn] swanlab 未启用: {exc}")

    def log(self, step: int, total: int, loss: float, lr: float, tokens_per_s: float):
        elapsed = time.time() - self.start
        print(f"step {step:>6}/{total}  loss {loss:.4f}  lr {lr:.2e}  "
              f"{tokens_per_s:>7.0f} tok/s  {elapsed / 60:>5.1f} min", flush=True)
        if self.swan is not None:
            self.swan.log({"loss": loss, "lr": lr, "tokens_per_s": tokens_per_s}, step=step)

    def finish(self):
        if self.swan is not None:
            try:
                import swanlab
                swanlab.finish()
            except Exception:
                pass


def save_ckpt(path: str, model, optimizer, epoch: int, step: int):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
    }, path)


def try_resume(path: str, model, optimizer):
    """断点续训。返回 (start_epoch, start_step)，没有检查点就返回 (0, 0)。"""
    if not path or not os.path.exists(path):
        return 0, 0
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[resume] 从 {path} 恢复：epoch={ckpt.get('epoch')} step={ckpt.get('step')}")
    return ckpt.get("epoch", 0), ckpt.get("step", 0)


def load_weights(model, path: str, strict: bool = True):
    """加载纯权重（.pth 里只存 state_dict 的情况）。"""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    return model.load_state_dict(state, strict=strict)
