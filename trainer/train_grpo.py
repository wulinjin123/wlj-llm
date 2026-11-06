"""GRPO 训练脚本（风格对齐 Wood-Q/MokioMind 的 trainer/train_grpo.py）。

GRPO = PPO 去掉 critic，用「同一 prompt 的 G 条回答互相比较」来代替价值网络估基线。
所以它既没有 reward model 的训练，也没有 value head，显存比 PPO 省一半。

和参考仓库的差异（5 处，都是本项目接口/硬件决定的）：
  1. 逐 prompt 生成，不用 num_return_sequences / attention_mask。
     本项目的 CausalLM.generate 是手写的（覆盖 GenerationMixin），签名只有
     input_ids，不收 attention_mask 也不收 num_return_sequences。
     而且 Model.forward 收了 attention_mask 却没传进 Attention（实测对 loss 零影响），
     一旦左填充拼 batch，padding 会被当成真 token 参与注意力，结果全错且不报错。
     做法：一条 prompt 复制 G 份再生成，同组长度天然一致，根本不需要 padding。
  2. 奖励默认走规则分（train_ppo.rule_reward），--reward_model_path 留空即不加载。
     internlm2-1_8b-reward 本地没有，8GB 显存也塞不下 reward model + policy。
  3. max_gen_len 默认 256（参考 1536），4060 上 1536 必 OOM。
  4. max_seq_len 默认 512（参考 66）：本地 rlaif prompt 实测 cap 512 保留 83.5%、
     640 保留 96.9%。
  5. num_generations 默认 4（参考 8）：G8+gen256 在 8GB 上余量太少。
"""

import os
import sys
import gc

# 📚 Python模块系统
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse  # 命令行参数解析
import torch  # PyTorch深度学习框架
from torch import optim  # 优化器
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火
from torch.utils.data import DataLoader  # 数据加载

# tiny-llm相关组件
from model.model import CausalLM, ModelConfig  # 模型与配置
from dataset.lm_dataset import RLAIFDataset  # RLAIF数据集（只返回 prompt 字符串）
from trainer.trainer_utils import (  # 训练工具函数
    OUT_DIR,
    load_tokenizer,
    load_weights,
    save_ckpt,
    set_seed,
    try_resume,
)
# 奖励函数和 logp 提取都复用 train_ppo.py 里的实现，不重复写一份
from trainer.train_ppo import calculate_rewards, token_logprobs


def Logger(content):
    # 参考仓库的 Logger 只在主进程打印；本项目单卡训练，直接 print
    print(content, flush=True)


def is_main_process():
    # 参考仓库用 dist 判断；本项目单卡，永远是主进程
    return True


def get_per_token_logps(mdl, input_ids, n_keep):
    """只算最后 n_keep 个位置的 log 概率。

    参考仓库写的是 mdl(input_ids=..., logits_to_keep=n_keep+1).logits[:, :-1, :]，
    本项目 forward 返回的是元组 (logits, loss, past_key_value)，所以取 [0]。

    注意一点：本项目的 CausalLM.forward 是先对**全序列**算完 lm_head 再切片，
    所以 logits_to_keep 并不会省掉 lm_head 的算力（这一点和 HF 官方实现不同），
    它省下的是返回张量和后面 gather 的范围。保留这个参数是为了和参考仓库语义一致。

    切片对齐：取末尾 n_keep+1 个位置的 logits 再去掉最后一个，
    剩下的位置 t 预测 t+1 的 token，正好对得上 input_ids[:, -n_keep:]。
    """
    logits = mdl(input_ids=input_ids, logits_to_keep=n_keep + 1)[0][:, :-1, :]
    return token_logprobs(logits, input_ids[:, -n_keep:])


def grpo_train_epoch(
    epoch,
    loader,
    iters,
    ref_model,
    reward_model,
    reward_tokenizer,
    start_step=0,
    wandb=None,
):
    for step, batch in enumerate(loader, start=1):
        # 📚 断点续训：跳过前 start_step 个 batch
        if step <= start_step:
            continue
        # 📚 max_steps：快速验证用，跑满就收工
        if args.max_steps and step > args.max_steps:
            break

        prompts = batch["prompt"]

        # 📚 逐 prompt 生成（差异点 1，见文件头）
        # step_loss_sum 累计本 step 内各 prompt 的 loss，日志时取平均
        step_loss_sum = 0.0
        step_reward_sum = 0.0
        step_len_sum = 0.0
        step_adv_sum = 0.0

        for prompt in prompts:
            prompt_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                return_token_type_ids=False,
                add_special_tokens=False,
            ).to(args.device)

            if args.max_seq_len:
                # 超长 prompt 保留末尾（保留最近的对话，开头的历史可以丢）
                prompt_inputs["input_ids"] = prompt_inputs["input_ids"][
                    :, -args.max_seq_len :
                ]
                prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][
                    :, -args.max_seq_len :
                ]

            prompt_len = prompt_inputs["input_ids"].size(1)

            with torch.no_grad():
                model_for_gen = model
                # 同一条 prompt 复制 G 份，等价于 HF 的 num_return_sequences，
                # 但长度完全一致，不会引入 padding
                gen_ids = prompt_inputs["input_ids"].repeat_interleave(
                    args.num_generations, dim=0
                )
                outputs = model_for_gen.generate(
                    gen_ids,
                    max_new_tokens=args.max_gen_len,
                    temperature=0.8,
                    eos_token_id=tokenizer.eos_token_id,
                )

            completion_ids = outputs[:, prompt_len:]
            n_keep = completion_ids.size(1)

            per_token_logps = get_per_token_logps(model, outputs, n_keep)
            with torch.no_grad():
                ref_per_token_logps = get_per_token_logps(ref_model, outputs, n_keep)

            completions = tokenizer.batch_decode(
                completion_ids, skip_special_tokens=True
            )
            # 📚 奖励：同一条 prompt 要重复 G 次传给打分函数，和 responses 一一对应
            rewards = calculate_rewards(
                [prompt] * args.num_generations,
                completions,
                args,
                reward_model,
                reward_tokenizer,
            ).to(args.device)

            # 📚 组内归一化得到优势（GRPO 的核心：用同组均值当 baseline，不要 critic）
            # G=1 时组内没有差异，而且 std() 对单元素返回 nan（无偏估计除以 n-1=0），
            # 所以单独兜一下：优势直接给 0，此时 loss 只剩 KL 项
            grouped_rewards = rewards.view(-1, args.num_generations)
            if args.num_generations > 1:
                mean_r = grouped_rewards.mean(dim=1).repeat_interleave(
                    args.num_generations
                )
                std_r = grouped_rewards.std(dim=1).repeat_interleave(
                    args.num_generations
                )
                advantages = torch.clamp(
                    (rewards - mean_r) / (std_r + 1e-4), -10, 10
                )
                # 参考仓库在这里又做了一次全局归一化。G 条一组时它基本是恒等变换，
                # 但保留下来是为了和视频里的数值完全对得上
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
            else:
                mean_r = std_r = rewards
                advantages = torch.zeros_like(rewards)

            # 📚 完播掩码：eos 之后的 token 都是「为了凑齐长度而继续解码」的垃圾，不计入 loss
            is_eos = completion_ids == tokenizer.eos_token_id
            eos_idx = torch.full(
                (is_eos.size(0),),
                is_eos.size(1),
                dtype=torch.long,
                device=args.device,
            )
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            completion_mask = (
                torch.arange(is_eos.size(1), device=args.device).expand(
                    is_eos.size(0), -1
                )
                <= eos_idx.unsqueeze(1)
            ).int()

            # 📚 KL 惩罚：k3 估计器 exp(kl) - kl - 1，恒正、在 kl=0 处为 0
            kl_div = ref_per_token_logps - per_token_logps
            per_token_kl = torch.exp(kl_div) - kl_div - 1
            # 参考仓库的写法：exp(logp - logp.detach()) 恒等于 1（GRPO 每条样本只更新一次，
            # 没有 old policy，重要性采样比就是 1），所以这里实际就是 -(优势 - β·KL)。
            # 保留原式是为了和视频代码逐行对应
            per_token_loss = -(
                torch.exp(per_token_logps - per_token_logps.detach())
                * advantages.unsqueeze(1)
                - args.beta * per_token_kl
            )

            loss = (
                (per_token_loss * completion_mask).sum(dim=1)
                / completion_mask.sum(dim=1)
            ).mean()
            # 一个 step 里有多条 prompt，逐条反传、梯度累加，等价于拼成一个大 batch。
            # 除以 len(prompts) 是为了让梯度量级和参考仓库的 batch 均值一致
            step_loss_sum += loss.item()
            step_reward_sum += rewards.mean().item()
            step_len_sum += completion_mask.sum(dim=1).float().mean().item()
            step_adv_sum += advantages.mean().item()

            loss = loss / (len(prompts) * args.accumulation_steps)
            loss.backward()

            del (
                prompt_inputs,
                gen_ids,
                outputs,
                completion_ids,
                per_token_logps,
                ref_per_token_logps,
                completions,
                rewards,
                grouped_rewards,
                mean_r,
                std_r,
                advantages,
                completion_mask,
                per_token_kl,
                per_token_loss,
            )

        n_prompt = max(len(prompts), 1)

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # 📚 训练日志
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = step_loss_sum / n_prompt
            avg_reward_val = step_reward_sum / n_prompt
            avg_len_val = step_len_sum / n_prompt
            avg_adv_val = step_adv_sum / n_prompt
            current_lr = optimizer.param_groups[0]["lr"]

            Logger(
                f"Epoch: {epoch + 1}, Step: {step}/{iters}, "
                f"Actor Loss: {policy_loss_val:.6f}, Reward: {avg_reward_val:.6f}, "
                f"Avg Response Len: {avg_len_val:.2f}, LR: {current_lr:.2e}"
            )

            if wandb and is_main_process():
                wandb.log(
                    {
                        "policy_loss": policy_loss_val,
                        "reward": avg_reward_val,
                        "avg_response_len": avg_len_val,
                        "advantages_mean": avg_adv_val,
                        "learning_rate": current_lr,
                    }
                )

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

        # 📚 释放显存：生成 + 前向 + 反向三份激活叠在一起，8GB 卡上不主动清会越跑越紧
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    """
    GRPO主函数：组相对策略优化脚本的入口点

    📚 GRPO训练流程：
    1. 准备策略模型和参考模型（同一份权重起步，参考模型冻结）
    2. 一条 prompt 让策略模型生成 G 条回答
    3. 给这 G 条回答各自打分
    4. 组内减均值除标准差得到优势 advantages
    5. 用 -(优势 - β·KL) 当 loss 反传，不需要 critic

    📚 和PPO的关系：
    PPO 要再挂一个和策略模型同样大的 critic 来估「这条回答有多好」的基线；
    GRPO 直接用同组 G 条回答的平均分当基线，省掉一整个模型。
    代价是每条 prompt 要生成 G 条（PPO 只生成 1 条），所以算力换显存。

    📚 关键自检：
    同一组 G 条回答如果奖励完全相同，std≈0，advantages 会全为 0，loss 退化成
    只有 KL 项 —— 这是 GRPO 的正常行为（没有组内差异就没有学习信号），
    不代表代码写错了。看 Reward 有没有在动、Avg Response Len 是否合理更靠谱。
    """

    # 📚 命令行参数解析
    parser = argparse.ArgumentParser(
        description="tiny-llm GRPO (Group Relative Policy Optimization)"
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
        "--save_weight", default="grpo", type=str, help="保存权重的前缀名"
    )
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size（每条 prompt 会生成 G 条）")
    # 📚 GRPO 学习率知识点
    # RL 阶段学习率必须远小于 SFT，否则一次更新就把 SFT 学到的东西冲掉；
    # 参考仓库用 8e-8，跑不动可以提到 1e-6 量级
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="初始学习率")

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
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
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
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )
    parser.add_argument("--num_experts", default=4, type=int, help="MoE专家个数")

    parser.add_argument(
        "--max_seq_len",
        default=512,
        type=int,
        help="Prompt最大长度（超长保留末尾）；实测 cap512 保留 83.5%%、640 保留 96.9%%",
    )
    parser.add_argument("--max_gen_len", type=int, default=256, help="生成的最大长度")

    # ========== GRPO数据与采样参数 ==========
    parser.add_argument(
        "--data_path",
        type=str,
        default=r"D:\Projects\minimind\dataset\rlaif.jsonl",
        help="RLAIF数据路径",
    )
    # 📚 GRPO 的 G 知识点
    # G 就是「一条 prompt 生成几条回答来互相比」。G=1 时组内没有差异、
    # 优势恒为 0，GRPO 完全失效；G 越大信号越稳，但显存和耗时线性增长
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="每个prompt生成的样本数（G）；G=1 时优势恒为0，训不动",
    )
    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    parser.add_argument(
        "--reasoning",
        type=int,
        default=0,
        choices=[0, 1],
        help="推理模型类型（0=普通模型，1=推理模型：额外要求 think/answer 标签格式）",
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default="",
        help="Reward模型路径；留空则只用规则奖励（internlm2-1_8b-reward 8GB 放不下）",
    )

    parser.add_argument(
        "--from_weight",
        type=str,
        required=True,
        help="基于哪个权重训练（通常是 SFT/DPO 之后的 out/xxx.pth）",
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测&续训（0=否，1=是）",
    )

    # ========== 实验跟踪 ==========
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb（本机用 swanlab 顶替）")
    parser.add_argument(
        "--wandb_project", type=str, default="tiny-llm-GRPO", help="wandb项目名"
    )
    # 兼容项：旧的启动配置会传 --no_swanlab，而这里默认就不联网
    parser.add_argument("--no_swanlab", action="store_true", help="（兼容项）关闭实验跟踪")

    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    set_seed(42)

    # 📚 注意：GRPO 这里**不从参考仓库那样开 autocast**，全程 fp32。
    # 原因是 KL 项要用 policy 和 ref 两份 logp 相减，两边精度必须一致，
    # 一开就得两边都开；而 64M 模型 fp32 才 256MB，混合精度的收益几乎为 0，
    # 真正吃显存的是「生成 + 反传」两段激活。--dtype 参数保留只为和参考仓库对齐。

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 注意：这里没有像参考仓库那样把 max_position_embeddings 设成
    # max_seq_len + max_gen_len。本项目 freqs_cos 虽然 persistent=False 不进
    # state_dict，但把上限压到刚好等于生成长度容易在边界上越界，默认 32768 够用
    lm_config = ModelConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        num_experts=args.num_experts,
    )
    moe_suffix = "_moe" if lm_config.use_moe else ""
    resume_path = f"{args.save_dir}/{args.save_weight}{moe_suffix}_resume.pth"

    # ========== 3. 配置wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb  # 本机没有 wandb，用 swanlab 顶替（接口一致）

        wandb_run_name = (
            f"tiny-llm-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}"
            f"-LR-{args.learning_rate}"
        )
        wandb.init(project=args.wandb_project, name=wandb_run_name)

    # ========== 4. 定义模型和参考模型 ==========
    # 📚 GRPO双模型架构：只有策略模型和参考模型，没有 critic
    tokenizer = load_tokenizer()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = CausalLM(lm_config)
    load_weights(model, args.from_weight)
    model = model.to(args.device)
    Logger(f"策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")

    # 📚 参考模型初始化
    # 参考模型与策略模型初始权重相同，但完全冻结，只用来算 KL
    ref_model = CausalLM(lm_config)
    load_weights(ref_model, args.from_weight)
    ref_model = ref_model.to(args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    Logger(f"参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M")

    # ========== 5. 奖励模型（可选） ==========
    # 📚 参考仓库这里是必填的 internlm2-1_8b-reward；
    # 本项目默认走规则奖励，只有显式传了路径才加载
    reward_model = None
    reward_tokenizer = None
    if args.reward_model_path and os.path.exists(args.reward_model_path):
        from transformers import AutoModel, AutoTokenizer

        reward_model = AutoModel.from_pretrained(
            args.reward_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        reward_model = reward_model.to(args.device).eval().requires_grad_(False)
        reward_tokenizer = AutoTokenizer.from_pretrained(
            args.reward_model_path, trust_remote_code=True
        )
        Logger(f"奖励模型已加载：{args.reward_model_path}")
    else:
        Logger("未加载奖励模型，使用规则奖励（格式+长度+重复惩罚）")

    # ========== 6. 数据集与优化器 ==========
    # 📚 RLAIF 数据集只返回 {"prompt", "answer"} 字符串，不做离线 tokenize，
    # 因为 GRPO 要先把 prompt 喂给模型在线生成，长度每次都不同
    train_ds = RLAIFDataset(args.data_path, tokenizer)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size)
    iters = len(loader_for_count) if not args.max_steps else min(
        len(loader_for_count), args.max_steps
    )
    total_optimizer_steps = max(1, (iters // args.accumulation_steps) * args.epochs)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10
    )
    Logger(
        f"prompt 数量：{len(train_ds)}  G={args.num_generations}  beta={args.beta}  "
        f"max_seq_len={args.max_seq_len}  max_gen_len={args.max_gen_len}"
    )

    # ========== 7. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if args.from_resume == 1:
        # 注意：save_ckpt 只存 model/optimizer，不存 scheduler，续训后 LR 会从头走
        start_epoch, start_step = try_resume(resume_path, model, optimizer)

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False,  # GRPO 逐 prompt 处理，尾巴短一点无所谓
            pin_memory=True,
        )
        iters = len(loader) if not args.max_steps else min(len(loader), args.max_steps)

        if epoch == start_epoch and start_step > 0:
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，"
                f"从step {start_step + 1}开始"
            )
            grpo_train_epoch(
                epoch,
                loader,
                iters,
                ref_model,
                reward_model,
                reward_tokenizer,
                start_step,
                wandb,
            )
        else:
            grpo_train_epoch(
                epoch, loader, iters, ref_model, reward_model, reward_tokenizer, 0, wandb
            )

        # max_steps 是"总共跑多少步"的意思，跑满就不要再进下一个 epoch
        if args.max_steps:
            break

    if args.max_steps:
        Logger(f"[max_steps] 已跑满 {args.max_steps} 步，提前收工")
