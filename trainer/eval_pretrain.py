"""预训练评估脚本：算 loss / perplexity + 看生成效果。

用法：
    # 1) 只算 loss（快，30 秒）
    python trainer/eval_pretrain.py --ckpt out/pretrain_epoch1.pth --limit 100

    # 2) loss + 生成样例
    python trainer/eval_pretrain.py --ckpt out/pretrain_epoch1.pth --limit 100 --gen

    # 3) 自定义 prompt（逗号分隔）
    python trainer/eval_pretrain.py --ckpt out/pretrain_epoch1.pth --gen ^
        --prompts "人工智能是,中国的首都是,今天天气"

三个指标怎么看：
    loss  交叉熵，越低头越准。随机初始化 ≈ ln(6400) = 8.76，训练到 6 以下算"学到了词序"。
    ppl   = exp(loss)，可以理解成"每一步平均在多少个候选词里犹豫"。
    生成  只是定性看流畅度，63M 模型 + 20k 样本别期待成句，重点是"像中文、不是乱码"。

注意：这里默认从数据**末尾**抽样本（--limit 条），如果你训练时用了全量数据，
它是"训练集抽样"，只能反映拟合程度，不是严格意义的验证集。
想要干净 holdout，就在训练命令里加 --val_ratio 0.02（若脚本支持），或用
--val_file 指向另一份 jsonl。
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.lm_dataset import PretrainDataset          # noqa: E402
from model.model import CausalLM, ModelConfig          # noqa: E402
from trainer.trainer_utils import (                    # noqa: E402
    DATA_DIR, OUT_DIR, load_tokenizer,
)

DEFAULT_PROMPTS = ["人工智能是", "中国的首都是", "我喜欢吃", "今天天气"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=os.path.join(OUT_DIR, "pretrain_epoch1.pth"),
                   help="权重路径，支持 {'model':...} 或纯 state_dict 两种格式")
    p.add_argument("--data_path", type=str,
                   default=os.path.join(DATA_DIR, "pretrain_20k.jsonl"))
    p.add_argument("--limit", type=int, default=100, help="从数据末尾取多少条做评估")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=340)
    p.add_argument("--max_batches", type=int, default=0, help="0=不限制")
    p.add_argument("--gen", action="store_true", help="跑生成样例")
    p.add_argument("--prompts", type=str, default="", help="逗号分隔的 prompt")
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_moe", type=int, default=0, choices=[0, 1],
                   help="评估 MoE 权重时必须传 1，否则报 missing keys: ...experts.0...")
    p.add_argument("--num_experts", type=int, default=4, help="专家个数")
    p.add_argument("--save_json", type=str, default="", help="把结果写成 json，便于前后对比")
    return p.parse_args()


def load_model(ckpt_path: str, device: str, use_moe: bool = False, num_experts: int = 4):
    """加载权重。strict=True：宁可报错也不静默跳过缺的 key（跳过=白训）。

    注意 use_moe 必须和训练时一致：结构不同 strict 会直接拦住，这是好事。
    """
    model = CausalLM(ModelConfig(use_moe=use_moe, num_experts=num_experts)).to(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=True)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[eval] 权重={ckpt_path}  参数量={n:.2f}M  missing={len(missing)} unexpected={len(unexpected)}")
    return model.eval()


@torch.no_grad()
def eval_loss(model, dl, device, max_batches: int):
    """按**有效 token 数**加权平均，而不是简单平均每个 batch 的 loss。

    因为最后几个 batch 可能 PAD 更多（有效 token 少），简单平均会让它们权重过大。
    """
    total_loss, total_tokens, n_batch = 0.0, 0, 0
    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        _, loss, _ = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=labels)
        valid = int((labels[:, 1:] != -100).sum())
        total_loss += loss.item() * valid
        total_tokens += valid
        n_batch += 1
        if max_batches and n_batch >= max_batches:
            break

    avg = total_loss / max(total_tokens, 1)
    return avg, math.exp(avg), total_tokens, n_batch


@torch.no_grad()
def eval_gen(model, tokenizer, prompts, device, args):
    for text in prompts:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        # 没加 BOS 的话补一个，和训练时的样本格式对齐
        if ids[0, 0].item() != tokenizer.bos_token_id:
            ids = torch.cat([torch.tensor([[tokenizer.bos_token_id]], device=device), ids], dim=1)
        t0 = time.time()
        out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_p=args.top_p,
                             eos_token_id=tokenizer.eos_token_id, use_cache=True)
        dt = time.time() - t0
        new_tokens = out.shape[1] - ids.shape[1]
        print(f"\n  prompt : {text!r}")
        print(f"  生成   : {tokenizer.decode(out[0], skip_special_tokens=True)!r}")
        print(f"  速度   : {new_tokens} tok / {dt:.2f}s = {new_tokens / max(dt, 1e-6):.1f} tok/s")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer()
    ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 从末尾取样，避开训练时最开头那批（若你只训了前 N 条，这里就没重叠）
    n = len(ds)
    start = max(n - args.limit, 0)
    ds.samples = ds.samples.select(range(start, n))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, pin_memory=(device == "cuda"), drop_last=False)
    print(f"[eval] 评估样本={len(ds)}（第 {start}~{n - 1} 条）  batch={args.batch_size}")

    model = load_model(args.ckpt, device, bool(args.use_moe), args.num_experts)
    loss, ppl, tokens, n_batch = eval_loss(model, dl, device, args.max_batches)
    print(f"\n  loss = {loss:.4f}    ppl = {ppl:.2f}    有效token = {tokens}    batch数 = {n_batch}")

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()] or DEFAULT_PROMPTS
    if args.gen:
        print("\n------------- 生成样例 -------------")
        eval_gen(model, tokenizer, prompts, device, args)

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump({"ckpt": args.ckpt, "loss": loss, "ppl": ppl,
                       "tokens": tokens, "samples": len(ds),
                       "batch_size": args.batch_size, "max_seq_len": args.max_seq_len},
                      f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入 {args.save_json}")


if __name__ == "__main__":
    main()
