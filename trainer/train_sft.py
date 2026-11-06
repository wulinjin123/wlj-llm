"""SFT 训练脚本。与 train_pretrain.py 的差别：
  1. 数据集换成 SFTDataset（只有 assistant 回复段算 loss）
  2. 权重从预训练 ckpt 起步，不是随机初始化
  3. 学习率小一个量级（1e-5），步数少
  4. 默认切 2% 做验证集、每 500 步评一次、连续 2 次不改善就早停

📚 为什么要加第 4 条（实测记录）
不加早停跑满 2500 步：训练域 assistant ppl 从 292.6 → 19.2（看着很好），
但域外留出集从 1015 → 1288，通用 LM ppl 从 87.9 → 107.5 —— 后 1000 步在拟合噪声。
早停会把最好的那份单独存成 sft_best.pth。

⚠️ 验证集是从训练数据尾部切的，和训练集**同分布**，所以它只能防「同分布过拟合」。
想看真正的泛化退化（域外），得用 _tmp/eval_sft_fmt.py 在 dpo chosen 那种留出集上评。

冒烟（约 1 分钟，确认链路通）：
    python trainer/train_sft.py --data_path D:\\Projects\\minimind\\dataset\\sft_20k.jsonl ^
        --from_weight out_quick\\pretrain_epoch1.pth --batch_size 8 --max_seq_len 768 ^
        --max_steps 30 --log_interval 10 --no_swanlab

正式（1 epoch = 2450 步，实测 210 ms/步，约 9 分钟，挂机跑）：
    python trainer/train_sft.py --data_path D:\\Projects\\minimind\\dataset\\sft_20k.jsonl ^
        --from_weight out_quick\\pretrain_epoch1.pth --batch_size 8 --max_seq_len 768 ^
        --val_ratio 0.02 --eval_interval 500 --patience 2 --no_swanlab

断了就加 --from_resume 1 接着跑（接的是 out_dir 下的 sft_resume.pth）。
不想切验证集（行为和旧版一致）就加 --val_ratio 0。
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.lm_dataset import SFTDataset
from model.model import CausalLM, ModelConfig
from trainer.trainer_utils import (OUT_DIR, TrainLogger, get_lr, load_tokenizer,
                                   load_weights, save_ckpt, set_seed, try_resume)


@torch.no_grad()
def evaluate(model, dl, device, dtype, tokenizer):
    """在验证集上算 **token 加权** 的 assistant loss。

    为什么必须 token 加权而不是 batch 平均：最后一个 batch 可能 pad 得多、
    有效 token 少，直接平均会让它权重过大，验证曲线会抖。
    返回值 (loss, ppl, 有效 token 数)。
    """
    model.eval()
    tot, ntok = 0.0, 0
    for ids, labels in dl:
        ids = ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attn = (ids != tokenizer.pad_token_id).long()
        with torch.autocast("cuda", dtype=dtype, enabled=(device == "cuda")):
            _, loss, _ = model(input_ids=ids, attention_mask=attn, labels=labels)
        valid = int((labels[:, 1:] != -100).sum())
        tot += loss.item() * valid
        ntok += valid
    model.train()
    avg = tot / max(ntok, 1)
    return avg, math.exp(avg), ntok


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--from_weight", type=str, default="", help="预训练权重 .pth 绝对路径")
    p.add_argument("--out_dir", type=str, default=OUT_DIR)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=768,
                   help="SFT 数据 p50=684，用 340 只有 4%% 的样本完整，768 能保留 76%%")
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0,
                   help="Windows 上必须 0，否则 DataLoader 多进程会卡死")
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--save_interval", type=int, default=500)
    # ===== 验证集与早停（治小数据过拟合）=====
    # 实测：2500 步训下来训练域 assistant ppl 从 292.6 降到 19.2，
    # 但干净留出集从 1015 涨到 1288 —— 后 1000 步是在拟合噪声。
    # 切 2% 出来盯着，拐头就停，并把最好那份单独存成 sft_best.pth。
    p.add_argument("--val_ratio", type=float, default=0.02,
                   help="从数据尾部切多少比例做验证集（0=不切，行为与旧版一致）")
    p.add_argument("--holdout_path", type=str, default="",
                   help="域外留出集（SFT 格式 jsonl）。给了就用它做早停依据，"
                        "比同分布验证集更能反映真实泛化；"
                        "可用 trainer/eval_stages.py 自动生成 out/sft_holdout_from_dpo.jsonl")
    p.add_argument("--eval_interval", type=int, default=500,
                   help="每 N 步评一次验证集（0=不评）")
    p.add_argument("--patience", type=int, default=2,
                   help="验证集连续 N 次不改善就早停（0=不早停）")
    p.add_argument("--max_steps", type=int, default=0,
                   help="最多训练多少步就收工（0=不限），用于几分钟的快速验证")
    p.add_argument("--from_resume", type=int, default=0)
    p.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_swanlab", action="store_true")
    return p.parse_args()


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
    print(f"device={device} dtype={dtype}")

    tokenizer = load_tokenizer()
    # SFTDataset 返回的是 tuple (input_ids, labels)，不是 dict，循环里必须解包
    ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # ===== 切验证集 =====
    # 两条路：
    #   ① --holdout_path 给了域外数据 → 用它（真泛化信号，推荐）
    #   ② 否则从训练数据尾部切 --val_ratio → 同分布验证集（只能防同分布过拟合）
    # 实测记录：同分布验证集在 900 步内一路从 4.44 降到 3.34，**看不出过拟合**；
    # 而同期域外留出集已经在涨。前者永远"看起来很健康"，这就是要用 ① 的原因。
    val_dl, val_tag = None, ""
    if args.holdout_path:
        ds_val = SFTDataset(args.holdout_path, tokenizer, max_length=args.max_seq_len)
        val_dl = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device == "cuda"), drop_last=False)
        val_tag = f"域外({len(ds_val)}条 {os.path.basename(args.holdout_path)})"
        print(f"早停依据：{val_tag}")
    elif args.val_ratio > 0:
        n_all = len(ds)
        n_val = max(int(n_all * args.val_ratio), 8)
        if n_val >= n_all:
            raise SystemExit(
                f"[val_ratio] {args.val_ratio} 太大：要切 {n_val} 条验证集，"
                f"总共才 {n_all} 条"
            )
        # 用同一个底层 HF Dataset 切两个互不重叠的视图，不额外占内存
        ds_val = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
        ds_val.samples = ds.samples.select(range(n_all - n_val, n_all))
        ds.samples = ds.samples.select(range(n_all - n_val))
        val_dl = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device == "cuda"), drop_last=False)
        val_tag = f"同分布(尾部{len(ds_val)}条)"
        print(f"验证集切分：训练 {len(ds)} 条 / 验证 {len(ds_val)} 条（取数据尾部）")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=(device == "cuda"), drop_last=True)
    print(f"样本数={len(ds)} 每 epoch 步数={len(dl)}")

    model = CausalLM(ModelConfig(use_moe=bool(args.use_moe), num_experts=args.num_experts)).to(device)
    if args.from_weight:
        load_weights(model, args.from_weight)
        print(f"已加载预训练权重 {args.from_weight}")
    print(f"参数量={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    total_steps = len(dl) * args.epochs
    start_epoch, start_step = 0, 0
    if args.from_resume:
        start_epoch, start_step = try_resume(os.path.join(args.out_dir, "sft_resume.pth"),
                                            model, optimizer)

    logger = TrainLogger(args.out_dir, run_name="sft", use_swanlab=not args.no_swanlab)
    model.train()
    # resume 时必须把已跑过的步数补回来，否则 lr 会从 warmup 起点重新爬（等于二次 warmup）
    global_step = start_epoch * len(dl) + start_step
    # 早停状态：best_val 是历史最好的验证 loss，bad_evals 是连续没改善的次数
    best_val, bad_evals, stopped = float("inf"), 0, False
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        for step, (input_ids, labels) in enumerate(dl):
            if epoch == start_epoch and step < start_step:
                continue
            if args.max_steps and global_step >= args.max_steps:
                print(f"[max_steps] 已跑满 {args.max_steps} 步，提前收工", flush=True)
                break

            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # SFTDataset 不返回 attention_mask，这里自己按 pad 补齐。
            # 注意这条数据是**右填充**，pad 全在末尾，因果 mask 已经挡掉了，
            # 实测传与不传 loss 完全一致（5.5675 vs 5.5675），代价是每步慢约 11%。
            # 保留它是为了换左填充/样本拼接时不出错；想提速可以整行删掉。
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            lr = get_lr(global_step, total_steps, args.warmup, args.learning_rate, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.autocast("cuda", dtype=dtype, enabled=(device == "cuda")):
                # 本项目 CausalLM 返回元组 (logits, loss, present)，不是 HF 的 outputs 对象
                _, loss, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                                   labels=labels)
                # aux_loss 已经在 CausalLM.forward 内部加进 loss 了（见 model.py 里
                # labels is not None 那段），这里**只取出来打日志，不能再加一次**，
                # 否则路由器的负载均衡项被放大 2 倍。
                # 路径是 model.model.aux_loss：CausalLM 包的 Model 才持有它
                aux_loss = getattr(model.model, "aux_loss", None) if args.use_moe else None
                total = loss / args.accumulation_steps

            scaler.scale(total).backward()
            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if step % args.log_interval == 0:
                # 这里打的是不含 aux 的原始 loss，方便和 eval 的数对上
                used = time.time() - t0
                tok_s = (step + 1) * args.batch_size * args.max_seq_len / max(used, 1e-6)
                logger.log(step, len(dl), loss.item(), lr, tok_s)
                if aux_loss is not None:
                    print(f"        aux_loss {float(aux_loss):.6f}", flush=True)
            # 📚 验证集评估 + 早停
            # 只有当验证 loss 真的比历史最好还低 1e-4 以上才算改善，
            # 平着不动也算「没改善」——这正是过拟合期的特征
            if val_dl is not None and args.eval_interval and global_step % args.eval_interval == 0:
                val_loss, val_ppl, _ = evaluate(model, val_dl, device, dtype, tokenizer)
                improved = val_loss < best_val - 1e-4
                print(f"[val] step {global_step}  train(batch均) {loss.item():.4f}  "
                      f"val[{val_tag}](token均) {val_loss:.4f}  ppl {val_ppl:.2f}"
                      f"{'  *best' if improved else ''}", flush=True)
                if improved:
                    best_val, bad_evals = val_loss, 0
                    save_ckpt(os.path.join(args.out_dir, "sft_best.pth"),
                              model, optimizer, epoch, step)
                else:
                    bad_evals += 1
                    if args.patience and bad_evals >= args.patience:
                        print(f"[early-stop] 验证集连续 {bad_evals} 次未改善"
                              f"（best {best_val:.4f}），停止训练", flush=True)
                        stopped = True
                        break

            if args.save_interval and global_step % args.save_interval == 0:
                save_ckpt(os.path.join(args.out_dir, "sft_resume.pth"), model, optimizer, epoch, step)

        # 文件名带 epoch，多 epoch 时不会互相覆盖
        ckpt = os.path.join(args.out_dir, f"sft_epoch{epoch + 1}.pth")
        save_ckpt(ckpt, model, optimizer, epoch + 1, 0)
        print(f"[epoch {epoch + 1}] 权重已存 {ckpt}")
        start_step = 0
        if stopped:
            print(f"[early-stop] 已按验证集停下，最好权重 "
                  f"{os.path.join(args.out_dir, 'sft_best.pth')}（val_loss {best_val:.4f}）",
                  flush=True)
            break
        if args.max_steps:
            break

    logger.finish()
    print("SFT 结束")


if __name__ == "__main__":
    main()
