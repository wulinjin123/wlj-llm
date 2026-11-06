"""多阶段模型统一评测：把 baseline / SFT / DPO / LoRA 放到同一把尺子上比。

为什么需要这个脚本（而不是直接用 eval_pretrain.py）：
eval_pretrain.py 只用 PretrainDataset 在**全 token**上算 loss，对 SFT 之后的模型
不公平 —— 它在 prompt 段背了不该背的交叉熵，而 SFT 训练时 prompt 段的 label 是 -100。
拿它比 SFT 前后的模型，会得出「SFT 让模型变差」的错误结论（实测确实会）。

这里给三把互不替代的尺子：
  lm        pretrain_20k 尾100，seq340，全 token 算 loss       → 通用语言建模
  holdout   dpo.jsonl 的 chosen 造的 SFT 格式留出集，seq768    → 干净泛化（从没见过）
  indomain  sft_20k 尾100，seq768，只算 assistant 段           → 训练集拟合程度

用法：
    # 单个权重
    python trainer/eval_stages.py --ckpt out/sft_epoch1.pth

    # 多个权重对比 + 生成样例
    python trainer/eval_stages.py --ckpt out_quick/pretrain_epoch1.pth ^
        --ckpt out/sft_epoch1.pth --ckpt out/dpo_768.pth --gen ^
        --save_json D:\\WorkBuddy-Output\\eval.json

判读标准（本次实测的参考值，稠密 63.9M）：
    baseline  lm 4.48 / holdout 6.92 / indomain 5.68
    SFT       lm 4.68 / holdout 7.16 / indomain 2.96   ← indomain 大降才算学到
    训崩的DPO lm 6.60 / holdout 12.7 / indomain 15.9   ← 三项同时暴涨就是烧了
"""

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.lm_dataset import PretrainDataset, SFTDataset  # noqa: E402
from model.model import CausalLM, ModelConfig               # noqa: E402
from trainer.trainer_utils import DATA_DIR, load_tokenizer  # noqa: E402

DEFAULT_PROMPTS = ["人工智能是", "中国的首都是", "我喜欢吃", "今天天气"]
HOLDOUT_NAME = "sft_holdout_from_dpo.jsonl"


def build_holdout(src_path: str, dest_path: str, n: int = 100) -> str:
    """从 dpo.jsonl 的 chosen 造一份 SFT 格式留出集。

    为什么用 chosen：它是「理想回答」，且 SFT 训练时从没见过这份数据
    （SFT 用的是 sft_20k.jsonl），所以拿它评 assistant loss 是真正的域外泛化。

    注意格式转换：dpo.jsonl 每行是 {"chosen": [...], "rejected": [...]}，
    而 SFTDataset 要的是 {"conversations": [...]}，不转换会 KeyError: 'conversations'。
    """
    if os.path.exists(dest_path):
        return dest_path
    rows = []
    with open(src_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            chosen = d.get("chosen")
            if not chosen:
                continue
            rows.append(json.dumps({"conversations": chosen}, ensure_ascii=False))
    tail = rows[-n:] if len(rows) > n else rows
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tail))
    print(f"[data] 已从 {Path(src_path).name} 末尾造出留出集：{dest_path}（{len(tail)} 条）")
    return dest_path


def tail_loader(cls, path, tok, max_len, limit, batch_size):
    ds = cls(path, tok, max_length=max_len)
    n = len(ds)
    ds.samples = ds.samples.select(range(max(n - limit, 0), n))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), len(ds)


@torch.no_grad()
def run_pretrain(model, dl, device):
    tot, ntok = 0.0, 0
    for b in dl:
        ids = b["input_ids"].to(device)
        attn = b["attention_mask"].to(device)
        labels = b["labels"].to(device)
        _, loss, _ = model(input_ids=ids, attention_mask=attn, labels=labels)
        v = int((labels[:, 1:] != -100).sum())
        tot += loss.item() * v
        ntok += v
    avg = tot / max(ntok, 1)
    return avg, math.exp(avg), ntok


@torch.no_grad()
def run_sft(model, dl, tok, device):
    tot, ntok = 0.0, 0
    for ids, labels in dl:
        ids, labels = ids.to(device), labels.to(device)
        attn = (ids != tok.pad_token_id).long()
        _, loss, _ = model(input_ids=ids, attention_mask=attn, labels=labels)
        v = int((labels[:, 1:] != -100).sum())
        tot += loss.item() * v
        ntok += v
    avg = tot / max(ntok, 1)
    return avg, math.exp(avg), ntok


@torch.no_grad()
def run_gen(model, tok, prompts, device, max_new_tokens, seed=42):
    out = []
    torch.manual_seed(seed)  # 固定种子：多个模型必须用同一串随机数才可比
    for text in prompts:
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if ids[0, 0].item() != tok.bos_token_id:
            ids = torch.cat(
                [torch.tensor([[tok.bos_token_id]], device=device), ids], dim=1
            )
        g = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.8,
                           top_p=0.9, eos_token_id=tok.eos_token_id, use_cache=True)
        out.append(tok.decode(g[0], skip_special_tokens=True))
    return out


def load_state(ckpt, lora_path=None, lora_rank=16):
    """加载权重。传了 --lora 就先把旁路合进基座，返回不带 lora key 的纯权重。"""
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    if not lora_path:
        return state
    from model.lora import apply_lora, merge_lora
    base = CausalLM(ModelConfig())
    base.load_state_dict(state, strict=True)
    n_hit = apply_lora(base, rank=lora_rank)
    n_merge, merged = merge_lora(base, lora_path)
    print(f"    [LoRA] 挂载 {n_hit} 层 / 合并 {n_merge} 层")
    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True,
                   help="权重路径，可重复传多个做对比")
    p.add_argument("--lora", type=str, default="",
                   help="LoRA 旁路权重；配合 --ckpt 指基座使用（会先合并再评）")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--limit", type=int, default=100, help="每份数据取末尾多少条")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lm_seq_len", type=int, default=340)
    p.add_argument("--sft_seq_len", type=int, default=768)
    p.add_argument("--gen", action="store_true", help="跑固定 prompt 生成样例")
    p.add_argument("--max_new_tokens", type=int, default=60)
    p.add_argument("--prompts", type=str, default="", help="逗号分隔，覆盖默认 4 条")
    p.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_json", type=str, default="")
    a = p.parse_args()

    device = a.device
    tok = load_tokenizer()

    holdout = build_holdout(
        os.path.join(DATA_DIR, "dpo.jsonl"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", HOLDOUT_NAME),
    )
    dl_lm, n_lm = tail_loader(PretrainDataset,
                              os.path.join(DATA_DIR, "pretrain_20k.jsonl"),
                              tok, a.lm_seq_len, a.limit, a.batch_size)
    dl_ho, n_ho = tail_loader(SFTDataset, holdout, tok, a.sft_seq_len,
                              a.limit, a.batch_size)
    dl_id, n_id = tail_loader(SFTDataset,
                              os.path.join(DATA_DIR, "sft_20k.jsonl"),
                              tok, a.sft_seq_len, a.limit, a.batch_size)
    print(f"[data] lm({n_lm}条) / holdout({n_ho}条) / indomain({n_id}条)\n")

    prompts = [x.strip() for x in a.prompts.split(",") if x.strip()] or DEFAULT_PROMPTS
    results = {}

    for ckpt in a.ckpt:
        print(f"=== {ckpt} ===", flush=True)
        state = load_state(ckpt, a.lora or None, a.lora_rank)
        model = CausalLM(ModelConfig(use_moe=bool(a.use_moe),
                                     num_experts=a.num_experts))
        model.load_state_dict(state, strict=True)   # strict：缺 key 直接报，不静默跳过
        model = model.to(device).eval()

        lm = run_pretrain(model, dl_lm, device)
        ho = run_sft(model, dl_ho, tok, device)
        idm = run_sft(model, dl_id, tok, device)
        r = {
            "ckpt": ckpt,
            "lm_loss": lm[0], "lm_ppl": lm[1], "lm_tokens": lm[2],
            "holdout_loss": ho[0], "holdout_ppl": ho[1], "holdout_tokens": ho[2],
            "indomain_loss": idm[0], "indomain_ppl": idm[1], "indomain_tokens": idm[2],
        }
        print(f"  lm {lm[0]:.4f}/{lm[1]:.1f}   holdout {ho[0]:.4f}/{ho[1]:.1f}"
              f"   indomain {idm[0]:.4f}/{idm[1]:.1f}")
        if a.gen:
            r["gens"] = run_gen(model, tok, prompts, device, a.max_new_tokens)
            for i, t in enumerate(r["gens"]):
                print(f"  [{i}] {t[:90]!r}")
        print(flush=True)
        results[ckpt] = r

        del model, state
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if a.save_json:
        Path(a.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(a.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("结果已写入", a.save_json)


if __name__ == "__main__":
    main()
