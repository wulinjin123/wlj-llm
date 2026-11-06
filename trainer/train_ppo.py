"""PPO 训练脚本（对齐 MokioMind / minimind 官方流程，按本项目接口适配）。

和参考仓库（MokioMind trainer/train_ppo.py）逐条对齐了什么、哪儿必须改：

  对齐：
    - 数据用 dataset.lm_dataset.RLAIFDataset，返回 {"prompt", "answer"} 字符串，
      不在数据集里 tokenize（PPO 要先在线生成，之后才知道长度）。
    - 四个模型的角色完全一致：
        actor      当前策略，要训练
        old_actor  采样时的旧策略，算 ratio 用，定期从 actor 同步
        ref        参考策略（训练前的 SFT 权重），算 KL 惩罚，冻结
        critic     价值网络，在 CausalLM 上挂一个 value_head，要训练
    - 损失结构和参考一致：
        ratio 裁剪的 PPO policy loss + vf_coef * value MSE + kl_coef * KL(ref)

  必须改（照抄会算错或跑不动）：
    1. 不拼 batch，逐条生成 + 逐条前向。
       本项目的 Model.forward 收了 attention_mask 但并没有把它传进 Attention，
       实测带不带 mask 的 loss 完全相同。参考仓库靠 HF 的 attention_mask 做左填充，
       在这里 padding 会被当成真 token 参与注意力，结果全错。
       逐条处理天然没有 padding，代价是 batch_size 只表示"累积几条再更新"。
    2. 没有 reward model。参考仓库要加载 internlm2-1_8b-reward（1.8B），
       8GB 显存放不下。这里默认走 rule_reward 规则打分，
       如果传了 --reward_model_path 才去加载。
    3. max_gen_len 默认 256 而不是 1536。实测 4060 8GB 上单条生成 256 token
       已占约 5GB，1536 会直接 OOM。
    4. 参考仓库默认 hidden_size=512，本项目是 768（MiniMind-3 稠密 63.91M）。

冒烟（约 3 分钟，确认链路通）：
    python trainer/train_ppo.py --data_path D:\\Projects\\minimind\\dataset\\rlaif.jsonl ^
        --from_weight out_quick\\pretrain_epoch1.pth --max_steps 4 --max_gen_len 64 ^
        --log_interval 1 --no_swanlab

正式（1 epoch = 19502 步，太长了，先 --max_steps 200 看看奖励有没有涨）：
    python trainer/train_ppo.py --data_path D:\\Projects\\minimind\\dataset\\rlaif.jsonl ^
        --from_weight out\\sft_epoch1.pth --max_gen_len 256 --no_swanlab

说明：本地 rlaif.jsonl 19502 条的末轮 assistant 内容**全部为空**（这是 RLAIF 数据的
标准结构，表示"此处待模型生成"），所以 answer 字段恒为空、不参与奖励计算，
奖励完全由 rule_reward 对生成文本打分。
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.lm_dataset import RLAIFDataset
from model.model import CausalLM, ModelConfig
from trainer.trainer_utils import (OUT_DIR, TrainLogger, load_tokenizer,
                                   load_weights, save_ckpt, set_seed)


# ==============================================================
# Critic：在 CausalLM 上加一个标量价值头
# ==============================================================
class CriticModel(CausalLM):
    """价值网络。结构和 actor 一样，只多一个 value_head。

    为什么要单独一个模型而不是和 actor 共享：actor 要最大化奖励，critic 要拟合
    回报，两者目标相反，共享主干会互相拉扯。代价是显存翻倍，所以 PPO 比 SFT 吃卡。

    注意与参考仓库的差异：那边写的是
        outputs = self.model(...); hidden = self.model.norm(outputs[0])
    多 norm 了一次 —— 因为它的 Model 返回的是未归一化的 hidden。
    本项目的 Model.forward 内部已经做过 self.norm，这里直接用 outputs[0]。
    """
    def __init__(self,params):
        super().__init__(params)
        self.value_head=nn.Linear(params.hidden_size,1)

    def forward(self,input_ids=None,attention_mask=None,**kwargs):
        outputs=self.model(
            input_ids=input_ids,attention_mask=attention_mask,**kwargs
        )
        hidden_states=self.model.norm(outputs[0])

        value=self.value_head(hidden_states).squeeze(-1)
        return value    
    
    
        # (B, L)


# ==============================================================
# 奖励函数
# ==============================================================
def token_logprobs(logits, labels):
    """取出 labels 每个位置在 logits 里的 log 概率。

    logits (B, L, V)，labels (B, L) 或 (B, L-1)。
    labels 比 logits 少一位时自动切掉 logits 最后一位（next-token 对齐）。
    返回 (B, L')。

    GRPO 脚本也 import 这个函数，改签名要两边一起看。
    """
    if logits.size(1) == labels.size(1) + 1:
        logits = logits[:, :-1]
    return (
        F.log_softmax(logits.float(), dim=-1)
        .gather(2, labels.unsqueeze(-1))
        .squeeze(-1)
    )


def rule_reward(text: str, reasoning: bool = False) -> float:
    """纯规则打分，不需要 reward model。

    参考仓库的奖励 = 0.5*(格式) + 0.25*4*(标签数) + reward_model 打分。
    这里去掉最后一项，同时补两项更实用的：
      1. 结构分：reasoning 模式要求 <think>…</think><answer>…</answer> 齐全
      2. 长度分：太短说明模型在敷衍，太长说明在啰嗦
      3. 重复惩罚：模型退化时会陷入复读，用 4-gram 重复率兜住
    """
    t = text.strip()
    if not t:
        return -1.0                     # 一个字都没生成，明确惩罚

    r = 0.0

    # ---- 1. 结构分 ----
    if reasoning:
        if re.match(r"^<think>.*?</think>\s*<answer>.*?</answer>$", t, re.S):
            r += 0.5
        # 每个标签恰好出现一次再加 0.125，重复出现说明结构崩了
        for tag in ("<think>", "</think>", "<answer>", "</answer>"):
            if t.count(tag) == 1:
                r += 0.125

    # ---- 2. 长度分 ----
    n = len(t)
    if n < 10:
        r -= 1.0
    elif n < 20:
        r -= 0.3
    elif n <= 300:
        r += 0.3
    else:
        r -= 0.001 * (n - 300)          # 超长缓慢扣，不硬砍

    # ---- 3. 重复惩罚 ----
    # 小模型最常见的退化就是复读（"红色的颜色有许多红色和白色的颜色"）。
    # 实测 4-gram 重复率区分度很干净：正常回答 0.000，整段复读 0.973，轻度复读 0.66。
    # 系数用 2.0：整段复读会被扣到 -1.6，明确低于正常回答的 +0.3。
    if n >= 8:
        grams = [t[i:i + 4] for i in range(n - 3)]
        rep = 1.0 - len(set(grams)) / len(grams)
        r -= 2.0 * rep

    return r


def calculate_rewards(prompts, responses, args, reward_model=None,
                      reward_tokenizer=None):
    """先算规则分，有 reward model 再叠加模型分。

    参考仓库只有 reward model 一条路；这里规则分永远在，
    模型分是可选加成，没装模型也能训。
    """
    rewards = torch.tensor(
        [rule_reward(t, reasoning=bool(args.reasoning)) for t in responses],
        dtype=torch.float32,
        device=args.device,
    )
    if reward_model is None:
        return rewards

    with torch.no_grad():
        scores = []
        for prompt, response in zip(prompts, responses):
            # 把 prompt 还原成 messages，再接上模型生成的回复
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": c.strip()} for role, c in matches]
            chat = messages + [{"role": "assistant", "content": response}]
            try:
                score = float(reward_model.get_score(reward_tokenizer, chat))
            except Exception:
                score = 0.0             # 奖励模型抽风就退化成纯规则分
            score = max(min(score, 3.0), -3.0)
            if args.reasoning:
                m = re.search(r"<answer>(.*?)</answer>", response, re.S)
                if m:
                    chat2 = messages + [{"role": "assistant",
                                         "content": m.group(1).strip()}]
                    try:
                        s2 = float(reward_model.get_score(reward_tokenizer, chat2))
                    except Exception:
                        s2 = 0.0
                    score = score * 0.4 + max(min(s2, 3.0), -3.0) * 0.6
            scores.append(score)
        rewards = rewards + torch.tensor(scores, dtype=torch.float32,
                                         device=args.device)
    return rewards


# ==============================================================
# 单条样本的 PPO 更新
# ==============================================================
def ppo_one_sample(actor, old_actor, ref, critic, tokenizer, prompt, args):
    """对一条 prompt 做：生成 → 打分 → 算四个模型的 logp/value → 反传。

    返回一个 dict 供日志用。loss 已经 backward，但还没 step（由调用方累积后 step）。
    """
    device = args.device

    # ---------- 1. 生成 ----------
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_seq_len,
    ).input_ids.to(device)
    prompt_len = enc.shape[1]

    actor.eval()
    with torch.no_grad():
        gen = actor.generate(
            enc,
            max_new_tokens=args.max_gen_len,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=tokenizer.eos_token_id,
        )
    resp_len = gen.shape[1] - prompt_len
    text = tokenizer.decode(gen[0, prompt_len:], skip_special_tokens=True)

    # ---------- 2. 打分 ----------
    reward = calculate_rewards([prompt], [text], args,
                               args._reward_model, args._reward_tok)[0]

    # ---------- 3. response 位置的 mask ----------
    # gen 形状 (1, P+R)，logp 形状 (1, P+R-1)：位置 i 的 logp 预测的是 gen[i+1]。
    # response 的第一个 token 在 gen 的下标 P，对应 logp 的下标 P-1。
    L = gen.shape[1] - 1
    resp_mask = torch.zeros(1, L, dtype=torch.bool, device=device)
    resp_mask[:, prompt_len - 1:prompt_len - 1 + resp_len] = True

    # ---------- 4. critic 估价值 ----------
    with torch.autocast("cuda", dtype=args.dtype, enabled=(device == "cuda")):
        values = critic(input_ids=gen)[0]              # (L_all,)
        loss_logits = actor(input_ids=gen)[0]          # (1, L_all, V)
    v_last = values[-1]                                # 末位价值 = 整条状态价值

    labels = gen[:, 1:]                                # (1, L_all-1)
    logp = token_logprobs(loss_logits, labels)         # (1, L_all-1)
    actor_logp = (logp * resp_mask).sum()

    # ---------- 5. old / ref 的 logp（冻结，不用梯度） ----------
    with torch.no_grad():
        with torch.autocast("cuda", dtype=args.dtype, enabled=(device == "cuda")):
            old_logits = old_actor(input_ids=gen)[0]
            ref_logits = ref(input_ids=gen)[0]
        old_logp = (token_logprobs(old_logits, labels) * resp_mask).sum()
        ref_logp = (token_logprobs(ref_logits, labels) * resp_mask).sum()

    # ---------- 6. PPO 损失 ----------
    advantage = (reward - v_last.detach()).float()     # 标量，单条就是这么算
    ratio = torch.exp(actor_logp - old_logp)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon,
                        1.0 + args.clip_epsilon) * advantage
    policy_loss = -torch.min(surr1, surr2)

    value_loss = F.mse_loss(v_last.float(), reward.float())
    kl_ref = actor_logp - ref_logp                     # 与 ref 的 KL 近似
    loss = policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref

    (loss / args.accumulation_steps).backward()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "reward": reward.item(),
        "kl_ref": kl_ref.item(),
        "ratio": ratio.item(),
        "gen_len": resp_len,
        "text": text,
    }


def parse_args():
    p = argparse.ArgumentParser()

    # ---- 数据与权重 ----
    p.add_argument("--data_path", type=str, required=True,
                   help="rlaif.jsonl 路径")
    p.add_argument("--from_weight", type=str, required=True,
                   help="actor/old/ref/critic 的初始权重，一般用 SFT 后的 .pth")
    p.add_argument("--out_dir", type=str, default=OUT_DIR)
    p.add_argument("--save_name", type=str, default="ppo_epoch1.pth")

    # ---- 训练规模 ----
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4,
                   help="按条累积的样本数，等价于梯度累积；不是真正拼 batch")
    p.add_argument("--accumulation_steps", type=int, default=4,
                   help="每多少条样本 step 一次优化器")
    p.add_argument("--max_steps", type=int, default=0,
                   help="最多跑多少条样本就收工（0=不限），用于快速验证")
    p.add_argument("--learning_rate", type=float, default=1e-6,
                   help="PPO 学习率要比 SFT 小 1~2 个量级，否则策略一步跨太远")
    p.add_argument("--critic_learning_rate", type=float, default=1e-5,
                   help="critic 是回归任务，可以比 actor 大一点")
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0,
                   help="Windows 上必须 0")

    # ---- 生成 ----
    p.add_argument("--max_seq_len", type=int, default=640,
                   help="prompt 上限；实测 p99=698，640 能保留 96.9%%")
    p.add_argument("--max_gen_len", type=int, default=256,
                   help="生成长度；1536 在 8GB 上必 OOM，256 实测约 5GB")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)

    # ---- PPO 超参 ----
    p.add_argument("--clip_epsilon", type=float, default=0.2,
                   help="参考仓库用 0.1，但那是配 reward model 的；纯规则分噪声大，0.2 更稳")
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--kl_coef", type=float, default=0.02)
    p.add_argument("--update_old_actor_freq", type=int, default=4,
                   help="每多少条样本把 actor 同步给 old_actor")
    p.add_argument("--reasoning", type=int, default=0, choices=[0, 1],
                   help="1=按 <think>/<answer> 结构给格式分；SFT 基座没训过该格式就保持 0")

    # ---- 可选 reward model ----
    p.add_argument("--reward_model_path", type=str, default="",
                   help="留空=纯规则奖励；填了才加载（1.8B 模型，8GB 显存放不下）")

    # ---- 记录 ----
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--save_interval", type=int, default=0,
                   help="每多少条存一次；0=只在结束时存")
    p.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_swanlab", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    dtype = (torch.bfloat16
             if device == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float16)
    args.dtype = dtype

    # ---------------- tokenizer ----------------
    tokenizer = load_tokenizer()
    # 本项目 generate 不处理 attention_mask，所以绝对不能左填充拼 batch，
    # 这里保持默认右填充且逐条处理，padding_side 设不设都不影响结果。
    tokenizer.pad_token = tokenizer.eos_token

    # ---------------- 配置 ----------------
    config = ModelConfig(use_moe=bool(args.use_moe), num_experts=args.num_experts)

    def build(need_grad: bool, name: str):
        m = CausalLM(config)
        load_weights(m, args.from_weight)      # 四个模型都从同一份 SFT 权重起步
        m = m.to(device)
        if not need_grad:
            m = m.eval().requires_grad_(False)
        print(f"[build] {name:<10} 参数量 {sum(p.numel() for p in m.parameters())/1e6:.2f}M "
              f"requires_grad={need_grad}", flush=True)
        return m

    print("[build] 四个模型都要单独占一份显存，这是 PPO 比 SFT 吃卡的根本原因", flush=True)
    actor = build(True, "actor")
    old_actor = build(False, "old_actor")
    ref = build(False, "ref")

    critic = CriticModel(config)
    load_weights(critic, args.from_weight, strict=False)   # value_head 是新加的，允许缺
    critic = critic.to(device)
    missing = [k for k in critic.state_dict() if k.startswith("value_head")]
    print(f"[build] critic     value_head 随机初始化（{len(missing)} 个张量），"
          f"其余从 SFT 权重继承", flush=True)

    # ---------------- 可选 reward model ----------------
    args._reward_model = None
    args._reward_tok = None
    if args.reward_model_path:
        from transformers import AutoModel, AutoTokenizer
        print(f"[reward] 加载 {args.reward_model_path}", flush=True)
        args._reward_model = (AutoModel.from_pretrained(
            args.reward_model_path, torch_dtype=torch.float16,
            trust_remote_code=True).to(device).eval().requires_grad_(False))
        args._reward_tok = AutoTokenizer.from_pretrained(
            args.reward_model_path, trust_remote_code=True)
    else:
        print("[reward] 未指定 reward model，使用规则奖励 rule_reward", flush=True)

    # ---------------- 优化器 ----------------
    actor_opt = optim.AdamW(actor.parameters(), lr=args.learning_rate)
    critic_opt = optim.AdamW(critic.parameters(), lr=args.critic_learning_rate)

    # ---------------- 数据 ----------------
    ds = RLAIFDataset(args.data_path, tokenizer,
                      max_length=args.max_seq_len + args.max_gen_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True)
    total = args.max_steps or (len(dl) * args.batch_size * args.epochs)
    print(f"[data] 样本数={len(ds)}  batch={args.batch_size}  "
          f"计划处理 {total} 条", flush=True)

    logger = TrainLogger(args.out_dir, run_name="ppo",
                         use_swanlab=not args.no_swanlab)

    # ---------------- 训练 ----------------
    step = 0
    t0 = time.time()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        for batch in dl:
            for prompt in batch["prompt"]:
                if args.max_steps and step >= args.max_steps:
                    break

                # 每 4 条把 actor 的权重同步给 old_actor（参考仓库 update_old_actor_freq）
                if step % args.update_old_actor_freq == 0:
                    old_actor.load_state_dict(actor.state_dict())
                    old_actor.eval().requires_grad_(False)

                actor.train()
                critic.train()
                info = ppo_one_sample(actor, old_actor, ref, critic,
                                      tokenizer, prompt, args)

                # 累积够 accumulation_steps 条才真正更新参数
                if (step + 1) % args.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
                    actor_opt.step()
                    critic_opt.step()
                    actor_opt.zero_grad(set_to_none=True)
                    critic_opt.zero_grad(set_to_none=True)

                step += 1

                if step % args.log_interval == 0:
                    lr = actor_opt.param_groups[0]["lr"]
                    print(f"step {step:>5}/{total}  "
                          f"loss {info['loss']:+.4f}  "
                          f"policy {info['policy_loss']:+.4f}  "
                          f"value {info['value_loss']:.4f}  "
                          f"reward {info['reward']:+.3f}  "
                          f"kl_ref {info['kl_ref']:+.5f}  "
                          f"ratio {info['ratio']:.3f}  "
                          f"len {info['gen_len']:>3}  "
                          f"lr {lr:.1e}  "
                          f"{time.time() - t0:.0f}s", flush=True)
                    if step <= 2:
                        # 前两步把生成文本打出来，肉眼确认模型在说人话
                        print(f"      ↳ 生成: {info['text'][:120]!r}", flush=True)

                if args.save_interval and step % args.save_interval == 0:
                    ckpt = os.path.join(args.out_dir, f"ppo_step{step}.pth")
                    save_ckpt(ckpt, actor, actor_opt, epoch, step)
                    print(f"[save] {ckpt}", flush=True)

            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            print(f"[max_steps] 已跑满 {args.max_steps} 条，提前收工", flush=True)
            break

    # ---------------- 收尾 ----------------
    ckpt = os.path.join(args.out_dir, args.save_name)
    save_ckpt(ckpt, actor, actor_opt, args.epochs, step)
    # 存一份干净的纯权重，方便 eval / 后续 DPO 直接用
    pure = os.path.join(args.out_dir, args.save_name.replace(".pth", "_weight.pth"))
    torch.save({k: v.half() for k, v in actor.state_dict().items()}, pure)
    print(f"[save] {ckpt}")
    print(f"[save] {pure}（纯权重 fp16）")
    logger.finish()


if __name__ == "__main__":
    main()
