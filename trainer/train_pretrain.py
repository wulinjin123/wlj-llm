"""预训练脚本：整段文本自监督，目标是 loss 从 ~8.9 降到 6 以下。

跑之前必须先完成 model/model.py 的 S1~S8，否则会直接抛 NotImplementedError。

冒烟（1 分钟，确认链路通）：
    python trainer/train_pretrain.py --data_path D:\\Projects\\minimind\\dataset\\pretrain_20k.jsonl ^
        --epochs 1 --batch_size 8 --max_seq_len 340 --log_interval 10

正式（约 4.6 小时 / 1 epoch，挂机跑）：
    python trainer/train_pretrain.py --data_path D:\\Projects\\minimind\\dataset\\pretrain_t2t_mini.jsonl ^
        --epochs 1 --batch_size 24 --max_seq_len 340 --log_interval 50

断了就加 --from_resume 1 接着跑。
"""

import argparse
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
    OUT_DIR, TrainLogger, get_lr, load_tokenizer, save_ckpt, set_seed, try_resume,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=OUT_DIR)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--max_seq_len", type=int, default=340)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=100, help="warmup 步数")
    p.add_argument("--accumulation_steps", type=int, default=4,
                   help="梯度累积。等效 batch = batch_size * accumulation_steps")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0,
                   help="Windows 上必须 0，否则 DataLoader 多进程会卡死")
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--save_interval", type=int, default=500, help="每多少步存一次断点")
    p.add_argument("--max_steps", type=int, default=0,
                   help="最多训练多少步就收工（0=不限）。用于 3 分钟快速验证，跑满会自动存权重")
    p.add_argument("--from_resume", type=int, default=0)
    p.add_argument("--use_moe", type=int, default=0, choices=[0, 1],
                   help="1=每层 FFN 换成 4 专家 MoE（参数 63.91M -> 198.42M）")
    p.add_argument("--num_experts", type=int, default=4, help="专家个数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_swanlab", action="store_true")
    return p.parse_args()


def train_epoch(
    model, dl, optimizer, args, device, dtype, logger, scaler,
    epoch, start_epoch, start_step, global_step, total_steps,
):
    start_time = time.time()
    for step, batch in enumerate(dl):
        if epoch == start_epoch and step < start_step:
            continue
        # 快速验证用：跑满 max_steps 就退出本 epoch，主循环照常存权重
        if args.max_steps > 0 and global_step >= args.max_steps:
            print(f"[max_steps] 已跑满 {args.max_steps} 步，提前收工", flush=True)
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        lr = get_lr(global_step, total_steps, args.warmup,
                    args.learning_rate, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with torch.autocast(device_type="cuda", dtype=dtype,
                            enabled=(device == "cuda")):
            # CausalLM.forward 返回三元组 (logits, loss, present)，不是 HF 那种输出对象
            logits, loss, _ = model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
            # aux_loss 已经在 CausalLM.forward 内部加进 loss 了（labels is not None 那段），
            # 这里**只读取来打日志，不能再加一次**，否则负载均衡项被放大 2 倍。
            # 属性挂在 model.model 上，不是 CausalLM 本身。
            aux_loss = getattr(model.model, "aux_loss", None) if args.use_moe else None
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        global_step += 1
        if step % args.log_interval == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            tokens_per_second = (step + 1) * args.batch_size * args.max_seq_len / elapsed
            logger.log(step, len(dl), loss.item() * args.accumulation_steps,
                       lr, tokens_per_second)
            if aux_loss is not None:
                print(f"        aux_loss {float(aux_loss):.6f}", flush=True)
        if args.save_interval > 0 and global_step % args.save_interval == 0:
            save_ckpt(os.path.join(args.out_dir, "pretrain_resume.pth"),
                      model, optimizer, epoch, step)

    return global_step

def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 4060 原生支持 bf16：不用 GradScaler、数值范围比 fp16 大得多，不会出现 loss 变 nan
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    print(f"device={device}  dtype={dtype}  args={vars(args)}")

    tokenizer = load_tokenizer()
    ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=(device == "cuda"),
                    drop_last=True)
    print(f"样本数={len(ds)}  每 epoch 步数={len(dl)}")

    model = CausalLM(ModelConfig(use_moe=bool(args.use_moe),
                                 num_experts=args.num_experts)).to(device)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数量={n_param:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  betas=(0.9, 0.95), weight_decay=0.01)
    # bf16 不需要梯度缩放；只有 fp16 才需要。scaler 关掉后 scale/unscale 都变成 no-op
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    steps_per_epoch = len(dl)
    total_steps = steps_per_epoch * args.epochs
    start_epoch, start_step = 0, 0
    if args.from_resume:
        start_epoch, start_step = try_resume(
            os.path.join(args.out_dir, "pretrain_resume.pth"), model, optimizer
        )

    logger = TrainLogger(args.out_dir, run_name="pretrain", use_swanlab=not args.no_swanlab)
    model.train()

    global_step = start_epoch * steps_per_epoch + start_step
    for epoch in range(start_epoch, args.epochs):
        global_step = train_epoch(
            model, dl, optimizer, args, device, dtype, logger, scaler,
            epoch, start_epoch, start_step, global_step, total_steps,
        )

        ckpt = os.path.join(args.out_dir, f"pretrain_epoch{epoch + 1}.pth")
        save_ckpt(ckpt, model, optimizer, epoch + 1, 0)
        print(f"[epoch {epoch + 1}] 完成，权重已存 {ckpt}", flush=True)
        # 每个 epoch 结束重置 resume 起点，避免下一 epoch 又从中间跳过
        start_step = 0

    logger.finish()
    print("训练结束")


if __name__ == "__main__":
    main()
