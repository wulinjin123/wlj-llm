"""DPO 训练脚本（风格对齐 Wood-Q/MokioMind 的 trainer/train_dpo.py）。

和参考仓库的差异：前三条是硬件/接口决定的，后两条是**实测出来的修复**。

  1. max_seq_len 默认 1024（参考 4096）：本地 dpo.jsonl 实测 token p50=660 /
     p90=931 / max=1217，1024 能保留 95.8%，4096 要多花约 4 倍显存只多覆盖 4.2%。
  2. 模型换成本项目的 CausalLM / ModelConfig，工具函数换成本项目的 trainer_utils
     （参考仓库的 init_model / lm_checkpoint 依赖 ../checkpoints 那套目录约定）。
  3. 补了参考仓库没有的 acc 指标（chosen 隐含奖励 > rejected 的比例），
     这是判断 DPO 到底有没有学到东西的唯一直接指标。
  4. --nll_coef：chosen 上的 NLL 正则（RPO 做法），治 likelihood displacement。
     默认 0 = 纯 DPO，与参考仓库完全一致。
  5. --gen_guard_interval：每 N 步真采样一次，输出太短就判定崩坏并停。
     默认 0 = 关闭。

📚 为什么需要 4 和 5（实测记录）
纯 DPO + lr=5e-6 跑 1000 步会**把模型烧掉**：生成退化成「人工智能是AI」几个字就 EOS，
而训练 loss 全程稳在 0.69 附近、acc 还有 100% —— 从 loss 完全看不出崩了。
根因是 DPO 只优化 chosen/rejected 的**差值**，不管绝对概率，于是优化器可以
把两者的 logp 一起压低（likelihood displacement），小模型上就表现为「不会说话了」。
4 用 NLL 把 chosen 的概率钉住；5 用生成长度做早期哨兵，比 loss 敏感得多。

用法（推荐配置）：
    python trainer/train_dpo.py --data_path ...\\dpo.jsonl ^
        --from_weight out\\sft_epoch1.pth --learning_rate 1e-6 ^
        --nll_coef 0.2 --gen_guard_interval 200 --max_steps 800
"""

import os
import sys

# 📚 Python模块系统
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse  # 命令行参数解析
import time  # 时间统计
import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # 神经网络函数
from contextlib import nullcontext  # 上下文管理器（CPU 上替代 autocast）
from torch import optim  # 优化器
from torch.utils.data import DataLoader  # 数据加载

# tiny-llm相关组件
from model.model import CausalLM, ModelConfig  # 模型与配置
from dataset.lm_dataset import DPODataset  # DPO数据集
from trainer.trainer_utils import (  # 训练工具函数
    OUT_DIR,
    get_lr,
    load_tokenizer,
    load_weights,
    save_ckpt,
    set_seed,
    try_resume,
)


def Logger(content):
    # 参考仓库的 Logger 只在主进程打印；本项目单卡训练，直接 print
    print(content, flush=True)


def is_main_process():
    # 参考仓库用 dist 判断；本项目单卡，永远是主进程
    return True


class GenGuardAbort(Exception):
    """生成守卫判定模型已崩，主动中止训练。

    注意：这不是代码错误，是「训练成功停下来」的一种形式。
    __main__ 里会捕获它并打印回滚建议，退出码保持 0。
    """


def logits_to_log_probs(logits, labels):
    # 词表logits转换为log概率
    # ！比参考仓库多一个 .float()：bf16 下 log_softmax 有效位数不足，
    #   同一份权重算出的 logp 会偏 1e-2 量级，足以淹没 DPO 的隐含奖励差。
    log_probs = F.log_softmax(logits.float(), dim=2)
    # 从log词表概率里选出label对应的log概率
    # 也就是拿到token在其对应位置的概率
    log_probs_per_token = torch.gather(
        log_probs, dim=2, index=labels.unsqueeze(2)
    ).squeeze(-1)  # gather 出来是 [B,S,1]，必须 squeeze(-1) 降到 [B,S]
    return log_probs_per_token


def seq_log_probs(log_probs, mask, logp_reduce="mean"):
    # 把逐token的log概率压成每条序列一个标量
    #   mean：除以mask的token数（参考仓库的做法，数值稳定）
    #   sum ：直接求和（DPO原论文的做法，长度敏感，beta 要调小到 1e-2 量级）
    if logp_reduce == "sum":
        return (log_probs * mask).sum(dim=1)
    seq_lengths = mask.sum(dim=1).clamp_min(
        1e-8
    )  # 防止整条 mask 全 0 时除零
    return (log_probs * mask).sum(dim=1) / seq_lengths


# DPO的loss计算
# 公式：L = -log(σ(β * (π(y_w) - π(y_l) - (π_ref(y_w) - π_ref(y_l)))))
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta, logp_reduce="mean"):
    # 计算ref和policy的序列log概率
    ref_seq = seq_log_probs(ref_log_probs, mask, logp_reduce)
    policy_seq = seq_log_probs(policy_log_probs, mask, logp_reduce)

    # 分别获取chosen和rejected的ref和policy的log概率
    batch_size = ref_seq.shape[0]
    chosen_ref_log_probs = ref_seq[: batch_size // 2]
    reject_ref_log_probs = ref_seq[batch_size // 2 :]
    chosen_policy_log_probs = policy_seq[: batch_size // 2]
    reject_policy_log_probs = policy_seq[batch_size // 2 :]
    # 计算策略模型的log概率差异
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    # 参考模型的log概率差异
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    # DPO损失计算
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


# DPO的隐含奖励：r = β·(logπ - logπ_ref)。
# 训练好之后直接把 r 当 reward model 用，不需要单独训一个打分器。
def dpo_rewards(ref_log_probs, policy_log_probs, mask, beta, logp_reduce="mean"):
    ref_seq = seq_log_probs(ref_log_probs, mask, logp_reduce)
    policy_seq = seq_log_probs(policy_log_probs, mask, logp_reduce)
    batch_size = ref_seq.shape[0]
    chosen_reward = beta * (
        policy_seq[: batch_size // 2] - ref_seq[: batch_size // 2]
    )
    reject_reward = beta * (
        policy_seq[batch_size // 2 :] - ref_seq[batch_size // 2 :]
    )
    accuracy = (chosen_reward > reject_reward).float().mean()
    return chosen_reward.detach(), reject_reward.detach(), accuracy


# ========== NLL 正则（RPO 做法） ==========
# 📚 为什么单纯 DPO 会训崩
# DPO 的梯度只关心「chosen 比 rejected 高多少」这个差值，**不管两者的绝对概率**。
# 于是优化器有一条捷径：把 rejected 压下去的同时，把 chosen 也一起压下去，
# 只要差值变大就行。这就是 likelihood displacement。
# 小模型上它的表现非常直观：输出越来越短，最后学会一开口就吐 EOS。
# 加一项「只在 chosen 上算 -log p(y_w)」，等于把 chosen 的绝对概率钉住，
# 只允许 rejected 往下掉——差值照样变大，但模型不会塌。
def nll_on_chosen(policy_log_probs, mask):
    half = policy_log_probs.shape[0] // 2
    log_probs = policy_log_probs[:half]  # 前一半是 chosen
    m = mask[:half]
    # 按有效 token 数平均，和 seq_log_probs 的 mean 口径一致
    return -(log_probs * m).sum() / m.sum().clamp_min(1e-8)


# ========== 生成守卫 ==========
# 📚 为什么用「输出长度」而不是 loss 来判断崩没崩
# 本次实测：lr=5e-6 跑 1000 步时 loss 一直停在 0.69 附近（看起来完全正常），
# 但生成已经退化成「人工智能是AI」——几个字就 EOS。loss 对这件事**完全不敏感**，
# 因为它只衡量 chosen/rejected 的相对差，不衡量模型还会不会说话。
# 真采样一次看长度，是唯一能提前发现的判据。
def gen_guard(model, tokenizer, prompt, device, max_new_tokens=32):
    was_training = model.training
    model.eval()
    # 守卫的采样不能污染训练的随机流，用完原样还回去
    rng = torch.get_rng_state()
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    # 和 eval_pretrain.py 对齐：没 BOS 就补一个，否则分布和训练时不一致
    if ids[0, 0].item() != tokenizer.bos_token_id:
        ids = torch.cat(
            [torch.tensor([[tokenizer.bos_token_id]], device=device), ids], dim=1
        )
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, temperature=0.8, top_p=0.9,
            eos_token_id=tokenizer.eos_token_id, use_cache=True,
        )
    torch.set_rng_state(rng)
    if was_training:
        model.train()
    new_tokens = out[0, ids.shape[1] :].tolist()
    # 只数到第一个 eos 为止：后面即使还有 token 也是无效续写
    n_valid = len(new_tokens)
    for i, t in enumerate(new_tokens):
        if t == tokenizer.eos_token_id:
            n_valid = i
            break
    return n_valid


def train_epoch(
    epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None,
    beta=0.1, tokenizer=None,
):
    start_time = time.time()
    for step, batch in enumerate(loader, start=1):
        # 📚 断点续训：跳过前 start_step 个 batch
        # 参考仓库靠 SkipBatchSampler 在 DataLoader 层面跳过，这里在循环里跳过，
        # 效果一样，代价是会白读一遍前 start_step 个 batch
        if step <= start_step:
            continue
        # 📚 max_steps：快速验证用，跑满就收工
        if args.max_steps and step > args.max_steps:
            break

        x_chosen = batch["x_chosen"].to(args.device)
        x_rejected = batch["x_rejected"].to(args.device)
        y_chosen = batch["y_chosen"].to(args.device)
        y_rejected = batch["y_rejected"].to(args.device)
        mask_chosen = batch["mask_chosen"].to(args.device).float()
        mask_rejected = batch["mask_rejected"].to(args.device).float()
        attention_mask_chosen = batch["attention_mask_chosen"].to(args.device)
        attention_mask_rejected = batch["attention_mask_rejected"].to(args.device)

        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)
        attention_mask = torch.cat(
            [attention_mask_chosen, attention_mask_rejected], dim=0
        )

        # 📚 学习率调度
        lr = get_lr(
            epoch * iters + step,
            args.epochs * iters,
            args.warmup,
            args.learning_rate,
            args.min_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            # 📚 参考模型前向传播
            # 参考模型冻结，只用于计算baseline概率
            with torch.no_grad():
                ref_outputs = ref_model(input_ids=x, attention_mask=attention_mask)
                ref_logits = ref_outputs[0]
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            # 📚 策略模型前向传播
            # 策略模型是需要优化的主要模型
            outputs = model(input_ids=x, attention_mask=attention_mask)
            logits = outputs[0]
            policy_log_probs = logits_to_log_probs(logits, y)

            # 📚 DPO损失计算
            dpo_loss_val = dpo_loss(
                ref_log_probs, policy_log_probs, mask, beta=beta,
                logp_reduce=args.logp_reduce,
            )
            # 📚 NLL 正则：把 chosen 的绝对概率钉住，压住 likelihood displacement
            # nll_coef=0 时这一项恒为 0，行为退回参考仓库的纯 DPO
            nll_val = nll_on_chosen(policy_log_probs, mask)
            # ！参考仓库是 outputs.aux_loss；本项目 CausalLM 不返回它，
            #   负载均衡损失挂在 model.model.aux_loss 上，不开MoE 时补个 0
            aux_loss = getattr(model.model, "aux_loss", None)
            if not torch.is_tensor(aux_loss):
                aux_loss = torch.zeros((), device=args.device)
            loss = dpo_loss_val + args.nll_coef * nll_val + aux_loss
            loss = loss / args.accumulation_steps

        # 📚 反向传播
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 📚 训练日志
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            with torch.no_grad():
                chosen_r, reject_r, acc = dpo_rewards(
                    ref_log_probs, policy_log_probs, mask, beta=beta,
                    logp_reduce=args.logp_reduce,
                )

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) "
                f"loss:{current_loss:.6f} dpo:{dpo_loss_val.item():.6f} "
                f"nll:{nll_val.item():.4f} acc:{acc.item() * 100:.1f}% "
                f"chosen_r:{chosen_r.mean().item():+.4f} "
                f"reject_r:{reject_r.mean().item():+.4f} "
                f"lr:{current_lr:.12f} epoch_Time:{eta_min}min:"
            )

            if wandb:
                wandb.log(
                    {
                        "loss": current_loss,
                        "dpo": dpo_loss_val.item(),
                        "nll": nll_val.item(),
                        "acc": acc.item(),
                        "chosen_r": chosen_r.mean().item(),
                        "reject_r": reject_r.mean().item(),
                        "lr": current_lr,
                        "epoch_Time": eta_min,
                    }
                )

        # 📚 生成守卫：每 N 步真采样一次，看模型还会不会说话
        # 放在保存之前：先判断崩没崩，崩了就存一份坏权重当证据再停，
        # 不要让坏权重顶掉正常保存点的名字
        if (
            args.gen_guard_interval
            and step % args.gen_guard_interval == 0
            and tokenizer is not None
        ):
            n_valid = gen_guard(model, tokenizer, args.gen_guard_prompt, args.device)
            Logger(
                f"[guard] step={step} prompt={args.gen_guard_prompt!r} "
                f"有效输出={n_valid} token（阈值 {args.gen_guard_min}）"
            )
            if n_valid < args.gen_guard_min:
                moe_suffix = "_moe" if lm_config.use_moe else ""
                bad_ckp = (
                    f"{args.save_dir}/{args.save_weight}"
                    f"_broken_step{step}{moe_suffix}.pth"
                )
                torch.save(
                    {k: v.half() for k, v in model.state_dict().items()}, bad_ckp
                )
                raise GenGuardAbort(bad_ckp)

        # 📚 模型保存
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = (
                f"{args.save_dir}/{args.save_weight}"
                f"_{lm_config.hidden_size}{moe_suffix}.pth"
            )
            state_dict = {k: v.half() for k, v in model.state_dict().items()}  # 半精度保存
            torch.save(state_dict, ckp)
            # 参考仓库这里调 lm_checkpoint 存续训状态，本项目用 save_ckpt 顶替
            save_ckpt(resume_path, model, optimizer, epoch, step)
            model.train()


if __name__ == "__main__":
    """
    DPO主函数：直接偏好优化脚本的入口点

    📚 DPO训练流程：
    1. 准备策略模型和参考模型（同一份权重起步，参考模型冻结）
    2. 加载偏好数据（chosen vs rejected）
    3. 同时前向传播计算两种模型的概率
    4. 计算DPO损失并优化策略模型
    5. 迭代直到收敛

    📚 和SFT的关系：
    SFT 教模型"怎么答"，DPO 教模型"哪个答得更好"。
    DPO 不需要 reward model，也不需要在线生成，因为它在数学上把
    「先训奖励模型、再用PPO优化」两步解析地化简成了一步监督学习。
    所以后训练链路里它比 PPO 便宜得多：只跑两个模型，没有采样。

    📚 关键自检（最重要的一条）：
    第一步的 loss 必须约等于 ln2 = 0.693147。
    因为 policy 和 ref 从同一份权重起步，logits 恒为 0，而 -logsigmoid(0) = ln2。
    偏离这个值说明两件事之一：policy/ref 没有从同一份权重起步，或者 mask 算错了。
    """

    # 📚 命令行参数解析
    parser = argparse.ArgumentParser(
        description="tiny-llm DPO (Direct Preference Optimization)"
    )

    # ========== 基础训练参数 ==========
    parser.add_argument(
        "--save_dir",
        "--out_dir",  # 兼容本项目其它脚本的叫法
        dest="save_dir",
        type=str,
        default=OUT_DIR,
        help="模型保存目录",
    )
    parser.add_argument(
        "--save_weight", default="dpo", type=str, help="保存权重的前缀名"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="训练轮数（DPO通常1-2轮）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="batch size；一对 chosen/rejected 算 2 条序列，"
        "bs=2 就是 4 条一起前向。实测 bs4+seq1024 峰值 6578MB，"
        "8GB 卡上余量太少，故默认 2",
    )

    # 📚 DPO学习率知识点
    # DPO 只做偏好微调，学习率要远小于 SFT（1e-5），否则会把 SFT 学到的东西冲掉
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-7,
        help="初始学习率（DPO要远小于SFT，避免遗忘）",
    )
    parser.add_argument("--min_lr", type=float, default=5e-8, help="余弦退火的下限")
    parser.add_argument("--warmup", type=int, default=20, help="学习率预热步数")

    # ========== 硬件配置 ==========
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数（Windows 上必须 0）")

    # ========== 训练策略 ==========
    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="最多跑多少个batch step（0=不限），快速验证链路用",
    )

    # ========== 模型架构参数 ==========
    parser.add_argument("--hidden_size", default=768, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--max_seq_len",
        default=1024,
        type=int,
        help="训练的最大截断长度；实测 token p99=1113，1024 保留 95.8%，"
        "参考仓库的 4096 要多花 4 倍显存",
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )
    parser.add_argument("--num_experts", default=4, type=int, help="MoE专家个数")

    # ========== DPO数据和模型参数 ==========
    # 📚 DPO数据格式知识点
    # 数据包含chosen（偏好）和rejected（不偏好）回答配对
    parser.add_argument(
        "--data_path",
        type=str,
        default=r"D:\Projects\minimind\dataset\dpo.jsonl",
        help="DPO训练数据路径",
    )

    # 📚 DPO权重继承知识点
    # DPO通常基于SFT模型进行对齐优化；用pretrain权重做DPO没有意义，
    # 基座对 chosen/rejected 本来就没有偏好，梯度信号会被噪声淹没
    parser.add_argument(
        "--from_weight",
        type=str,
        required=True,
        help="基于哪个权重训练（通常是SFT模型 out/sft_epoch1.pth）",
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测&续训（0=否，1=是）",
    )

    # 📚 DPO beta参数知识点
    # beta控制KL约束强度，越大越不敢偏离ref，0.1-0.5是常见范围
    # 注意 beta 和 logp_reduce 必须配套：mean 配 0.1；sum 要配 1e-2 量级，
    # 否则 chosen-rejected 的隐含奖励差会到几十，sigmoid 直接饱和、梯度消失
    parser.add_argument(
        "--beta", default=0.1, type=float, help="DPO中的beta参数（控制优化强度）"
    )
    parser.add_argument(
        "--logp_reduce",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="mean=跟参考仓库（按token平均）；sum=DPO原论文（求和，需调小beta）",
    )

    # ========== 治 likelihood displacement 的两个开关（默认关闭） ==========
    # 默认 0 = 行为和参考仓库完全一致，不影响你照着视频对照；
    # 实测「纯 DPO + lr 5e-6」会把模型训崩（生成立刻 EOS），这两个开关就是解药。
    parser.add_argument(
        "--nll_coef",
        type=float,
        default=0.0,
        help="chosen 上的 NLL 正则系数（RPO 做法）。0=纯DPO；"
        "建议 0.1~0.5：把 chosen 的绝对概率钉住，只让 rejected 往下掉",
    )
    parser.add_argument(
        "--gen_guard_interval",
        type=int,
        default=0,
        help="每 N 步真采样一次检查模型是否被训崩（0=关闭）。"
        "崩坏时 loss 可能完全正常，只有生成长度能提前发现",
    )
    parser.add_argument(
        "--gen_guard_min",
        type=int,
        default=8,
        help="有效输出（截到 eos）短于这个 token 数就判定崩坏",
    )
    parser.add_argument(
        "--gen_guard_prompt", type=str, default="人工智能是", help="守卫用的固定 prompt"
    )

    # ========== 实验跟踪 ==========
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb（本机用 swanlab 顶替）")
    parser.add_argument(
        "--wandb_project", type=str, default="tiny-llm-DPO", help="wandb项目名"
    )
    # 兼容项：旧的 launch.json 会传 --no_swanlab，而这里默认就不联网
    parser.add_argument("--no_swanlab", action="store_true", help="（兼容项）关闭实验跟踪")

    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    set_seed(42)

    # 📚 混合精度：4060 是 Ada 架构，支持 bf16，优先用它（不用 GradScaler）
    device_type = "cuda" if "cuda" in args.device else "cpu"
    if device_type == "cuda" and not torch.cuda.is_bf16_supported():
        Logger("[warn] 当前卡不支持 bf16，退回 fp16")
        args.dtype = "float16"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.autocast(device_type, dtype=dtype)
    )
    # 参考仓库是 torch.cuda.amp.GradScaler(...)，这里用新版写法避免 FutureWarning；
    # bf16 下 enabled=False，等价于不做梯度缩放
    scaler = torch.amp.GradScaler(
        device_type, enabled=(args.dtype == "float16" and device_type == "cuda")
    )

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = ModelConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        num_experts=args.num_experts,
    )
    moe_suffix = "_moe" if lm_config.use_moe else ""
    resume_path = (
        f"{args.save_dir}/{args.save_weight}{moe_suffix}_resume.pth"
    )

    # ========== 3. 配置wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb  # 本机没有 wandb，用 swanlab 顶替（接口一致）

        wandb_run_name = (
            f"tiny-llm-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}"
            f"-LR-{args.learning_rate}"
        )
        wandb.init(project=args.wandb_project, name=wandb_run_name)

    # ========== 4. 定义模型和参考模型 ==========
    # 📚 DPO双模型架构
    # 策略模型：需要优化的模型
    # 参考模型：冻结的baseline模型
    tokenizer = load_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token

    model = CausalLM(lm_config)
    load_weights(model, args.from_weight)
    model = model.to(args.device)
    Logger(
        f"策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M"
    )

    # 📚 参考模型初始化
    # 参考模型与策略模型初始权重相同，但完全冻结。
    # 两者必须从同一份权重起步，否则初始 loss 不是 ln2≈0.693，
    # 而且 DPO 的 KL 约束失去了参照系。
    ref_model = CausalLM(lm_config)
    load_weights(ref_model, args.from_weight)
    ref_model = ref_model.to(args.device)
    ref_model.eval()  # 设为评估模式
    ref_model.requires_grad_(False)  # 冻结所有参数
    Logger(
        f"参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M"
    )

    # 📚 DPO数据集
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    Logger(
        f"偏好对数量：{len(train_ds)}  beta={args.beta}  "
        f"logp_reduce={args.logp_reduce}  max_seq_len={args.max_seq_len}"
    )

    # ========== 5. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if args.from_resume == 1:
        start_epoch, start_step = try_resume(resume_path, model, optimizer)

    # ========== 6. 开始训练 ==========
    try:
        for epoch in range(start_epoch, args.epochs):
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=True,  # 丢掉凑不满的尾巴，避免最后一个 batch 只有 chosen 没有 rejected
                pin_memory=True,
            )
            iters = len(loader) if not args.max_steps else min(len(loader), args.max_steps)

            if epoch == start_epoch and start_step > 0:
                # 第一个epoch且存在检查点：从该位置继续
                Logger(
                    f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，"
                    f"从step {start_step + 1}开始"
                )
                train_epoch(
                    epoch,
                    loader,
                    iters,
                    ref_model,
                    lm_config,
                    start_step,
                    wandb,
                    args.beta,
                    tokenizer,
                )
            else:
                train_epoch(
                    epoch, loader, iters, ref_model, lm_config, 0, wandb,
                    args.beta, tokenizer,
                )

            # max_steps 是"总共跑多少步"的意思，跑满就不要再进下一个 epoch
            if args.max_steps:
                break
    except GenGuardAbort as e:
        # 这不是报错，是守卫主动拦停：坏权重已经单独存好，不会覆盖正常检查点
        Logger(
            f"[guard] 判定模型已崩坏，训练已中止。坏权重：{e}\n"
            f"[guard] 建议：回滚到 {args.save_dir} 下最近一份正常权重，"
            f"然后加 --nll_coef 0.2 或把 --learning_rate 降到 1e-6 重跑"
        )
        # 模块级不能用 return，用干净退出（退出码 0：这是「成功拦停」不是「失败」）
        sys.exit(0)

    if args.max_steps:
        Logger(f"[max_steps] 已跑满 {args.max_steps} 步，提前收工")
