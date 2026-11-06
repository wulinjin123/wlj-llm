"""LoRA 微调脚本。与 train_sft.py 的差别只有 5 处：
  1. 加载基座后挂 LoRA 旁路（apply_lora）
  2. 冻结非 LoRA 参数（param.requires_grad = "lora" in name）
  3. 优化器只收 LoRA 参数
  4. 梯度裁剪也只裁 LoRA 参数
  5. 只存 A/B（0.76MB），不存整个 767MB 基座

冒烟（约 1 分钟，确认挂载和保存链路通）：
    python trainer/train_lora.py --data_path D:\\Projects\\minimind\\dataset\\sft_20k.jsonl ^
        --from_weight out\\sft_epoch1.pth --out_dir out --lora_name lora_demo ^
        --batch_size 8 --max_seq_len 768 --max_steps 30 --no_swanlab

正式（3 epoch）：
    python trainer/train_lora.py --data_path D:\\Projects\\minimind\\dataset\\sft_20k.jsonl ^
        --from_weight out\\sft_epoch1.pth --out_dir out --lora_name lora_demo ^
        --batch_size 8 --max_seq_len 768 --epochs 3 --no_swanlab
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.lm_dataset import SFTDataset
from model.lora import apply_lora, save_lora
from model.model import CausalLM, ModelConfig
from trainer.trainer_utils import (OUT_DIR, TrainLogger, get_lr, load_tokenizer,
                                   load_weights, set_seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--from_weight", type=str, required=True,
                   help="基座权重 .pth 绝对路径（一般用 SFT 之后的 sft_epoch1.pth）")
    p.add_argument("--out_dir", type=str, default=OUT_DIR)
    p.add_argument("--lora_name", type=str, default="lora_demo", help="输出文件名（不含 .pth）")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=768,
                   help="SFT 数据 p50=684，用 340 只有 4%% 的样本完整，768 能保留 76%%")
    p.add_argument("--learning_rate", type=float, default=1e-4,
                   help="LoRA 只训 A/B，学习率可以比全参微调大一个量级")
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0,
                   help="Windows 上必须 0，否则 DataLoader 多进程会卡死")
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=0,
                   help="最多训练多少步就收工（0=不限），用于几分钟的快速验证")
    p.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_swanlab", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    print(f"device={device} dtype={dtype}")

    tokenizer = load_tokenizer()
    ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=(device == "cuda"), drop_last=True)
    print(f"样本数={len(ds)} 每 epoch 步数={len(dl)}")

    model = CausalLM(ModelConfig(use_moe=bool(args.use_moe), num_experts=args.num_experts)).to(device)
    # 顺序不能反：必须先灌基座权重再挂 LoRA。
    # 反过来 load_weights(strict=True) 会因为模型多了 lora.* 的 key 而报 missing keys
    load_weights(model, args.from_weight)
    print(f"已加载基座权重 {args.from_weight}")

    n_hit = apply_lora(model, rank=args.lora_rank)

    # 只留 LoRA 参数可训练 —— 忘了这一步等于在做全参微调，显存和耗时都会翻几倍，
    # 而且最后只存 A/B，训出来的东西根本对不上
    lora_params = []
    for name, param in model.named_parameters():
        param.requires_grad = "lora" in name
        if param.requires_grad:
            lora_params.append(param)
    total = sum(p.numel() for p in model.parameters())
    n_lora = sum(p.numel() for p in lora_params)
    print(f"挂载层数={n_hit} 总参数={total / 1e6:.3f}M "
          f"LoRA={n_lora / 1e6:.4f}M ({n_lora / total * 100:.3f}%)")
    if n_hit == 0 or n_lora == 0:
        raise RuntimeError("LoRA 没挂上，检查 model/lora.py 的 apply_lora")

    # 优化器只收 LoRA 参数：收 model.parameters() 就把 63.91M 全训了
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    total_steps = len(dl) * args.epochs
    logger = TrainLogger(args.out_dir, run_name="lora", use_swanlab=not args.no_swanlab)

    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for input_ids, labels in dl:
            if args.max_steps and step >= args.max_steps:
                print(f"[max_steps] 已跑满 {args.max_steps} 步，提前收工", flush=True)
                break

            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # 和 train_sft.py 一致：SFTDataset 只返回 (input_ids, labels)，mask 自己造
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            lr = get_lr(step, total_steps, args.warmup, args.learning_rate, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.autocast("cuda", dtype=dtype, enabled=(device == "cuda")):
                _, loss, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                                   labels=labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # 裁剪也只裁 LoRA 参数，传 model.parameters() 在冻结场景下会刷一堆"无梯度"警告
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if step % args.log_interval == 0:
                used = time.time() - t0
                tok_s = (step + 1) * args.batch_size * args.max_seq_len / max(used, 1e-6)
                logger.log(step, len(dl), loss.item(), lr, tok_s)
            step += 1
        if args.max_steps:
            break

    os.makedirs(args.out_dir, exist_ok=True)   # torch.save 不会替你建目录
    path = os.path.join(args.out_dir, f"{args.lora_name}.pth")
    n_tensor = save_lora(model, path)
    print(f"LoRA 权重已存 {path}（{n_tensor} 个张量，{os.path.getsize(path) / 2**20:.2f}MB）")
    logger.finish()
    print("LoRA 训练结束")


if __name__ == "__main__":
    main()
