# wlj-llm

不调用 `transformers` 的模型类，从 `nn.Linear` 开始手写一个 **63.91M** 的 decoder-only 语言模型；
配上从零实现的后训练链路（SFT / LoRA / DPO / PPO / GRPO）和一套 **MoE 分支**（198.42M）。

目标不是刷榜，而是把「一个大模型是怎么被造出来的」每一层都握在手里：能跑、能验、能解释。

- 单卡 RTX 4060 Laptop 8GB 就能跑完全部链路
- `pytest -q` → **20 passed**
- 稠密 63,912,192 参数 / MoE 198,416,640 参数，逐位对得上 Llama 族同规模配置
- 训练与推理两条分支的 logits 差 **0.000e+00**（带/不带 KV Cache 必须完全一致）

```bash
git clone https://github.com/wulinjin123/wlj-llm.git
```

---

## 1. 手写了什么

| 组件 | 实现 | 关键点 |
|---|---|---|
| `RMSNorm` | 手写，含 fp32 上采样 | 与 `F.rms_norm` 误差 < 1e-6 |
| RoPE | `precompute_freqs_cis` / `rotate_half` / `apply_rotary_pos_emb` | `persistent=False`，否则污染 `state_dict` |
| `Attention` | GQA（8 Q / 4 KV）+ KV Cache + QK-Norm | 逐步解码结果 == 整段前向 |
| `FeedForward` | SwiGLU（silu + gate） | 中间维 2432，参数量精确对上 |
| `MoEGate` / `MoeFeedForward` | 4 专家、top-1 路由、负载均衡 aux loss | 训练分支与推理分支逐位一致 |
| `Block` | Pre-Norm 残差 | — |
| `Model` / `CausalLM` | embedding + N×Block + norm + 权重绑定 lm_head | `state_dict` 键名与官方实现完全一致 |

`model/model.py` 一个文件装下全部，因为模型是**一层层长出来的**：写在外面的测试可以一步一步验收。

---

## 2. 架构

稠密版（`configs/64m.json`）：

| 项 | 值 |
|---|---|
| 层数 / 宽度 | 8 / 768 |
| 注意力 | 8 heads，4 KV heads，`head_dim=96`（GQA） |
| FFN | SwiGLU，`intermediate_size=2432` |
| 词表 / 位置 | 6400 / RoPE θ=1e6，`max_position_embeddings=32768` |
| 归一化 | RMSNorm，eps=1e-6，Pre-Norm |
| 权重绑定 | `tie_word_embeddings=true` |
| **参数量** | **63,912,192（63.912M）** |

MoE 版（`--use_moe 1`）：把 8 个 FFN 层换成 4 专家 + top-1 路由，专家中间维仍为 2432，
路由 aux loss 系数 5e-4，不带共享专家。

| 项 | 值 |
|---|---|
| 专家数 / 激活 | 4 / top-1 |
| **参数量** | **198,416,640（198.417M）** |
| 实测 aux_loss | 4.37e-03（稳定，有梯度） |
| 训练 vs 推理 logits 差 | 0.000e+00 |

---

## 3. 后训练链路

七个入口，签名统一、互不覆盖产物：

| 脚本 | 作用 | 关键设计 |
|---|---|---|
| `trainer/train_pretrain.py` | 预训练 | cosine + warmup、断点续训、全 token 算 loss |
| `trainer/train_sft.py` | 指令微调 | 只对 assistant 段算 loss；**验证集早停**（`sft_best.pth`） |
| `trainer/train_lora.py` | LoRA | rank 16，只训 0.393M（0.611%）参数，`merge_lora` 回主干 |
| `trainer/train_dpo.py` | 偏好对齐 | seq-level logprob、**NLL 正则**治 likelihood displacement、**生成守卫**防训崩 |
| `trainer/train_ppo.py` | 在线 RL | actor / old / ref / critic 四模型，规则奖励 + 复读惩罚 |
| `trainer/train_grpo.py` | 免 critic RL | 每组采 G 条、组内归一化优势、k3 KL 估计 |
| `trainer/eval_stages.py` | 统一评测 | 三把尺子 + 多权重同屏对比 + 同种子生成样例 |

### 为什么单独写一个 `eval_stages.py`

用 `eval_pretrain.py` 那套「全 token 算 loss」去比 SFT 前后的模型会得出**错误结论**——
SFT 之后的模型在 prompt 段背了不该背的交叉熵。所以给三把互不替代的尺子：

| 尺子 | 数据 | 算法 | 量的是什么 |
|---|---|---|---|
| `lm` | `pretrain_20k` 尾 100，seq 340 | 全 token 交叉熵 | 通用语言建模能力 |
| `holdout` | `dpo.jsonl` 的 chosen 转 SFT 格式，seq 768 | 只算 assistant 段 | 域外泛化（SFT 从没见过） |
| `indomain` | `sft_20k` 尾 100，seq 768 | 只算 assistant 段 | 训练集拟合程度 |

**判读只有一条**：SFT 之后 `indomain` 必须大降，`lm` 小幅上升是正常的（SFT 学的是对话格式，不是通用语料）。

---

## 4. 实测结果

环境：RTX 4060 Laptop 8GB，PyTorch 2.6.0+cu124，bf16，batch/seq 见各表。

三把尺子（`--limit 100`，同一份 holdout，同一串随机种子）：

| 权重 | lm ↓ | holdout ↓ | indomain ↓ | 结论 |
|---|---|---|---|---|
| 预训练基线（400 步） | 4.4758 | 5.4768 | 5.6844 | 会连词，但不成句 |
| + SFT（2500 步） | 4.6776 | 5.3943 | **2.9520** | indomain **-48%**，格式学会了 |
| + DPO（800 步，NLL 正则） | 三项与基线同量级 | — | — | **没学到，但也没训崩** |
| + DPO（lr 5e-6，无正则） | 6.60 量级 | 12.7 量级 | 15.9 量级 | **三项同时暴涨 = 烧了**，loss 却全程 0.69 |

第三行是这套链路的真实状态：`dpo` loss 全程在 **0.693~0.700** 之间不降，`acc` 在 0~100% 之间反复跳，
评测三项与基线几乎无差——**模型的偏好确实没被改变**，但因为加了正则，也**没有被烧掉**
（生成守卫第 800 步采样到 32 个有效 token，阈值 8）。

最后一行是反例，也是这个项目最有价值的一组对照：**DPO 的 loss 停在 0.69、acc 停在 50%，模型却已经废了**。
不加 NLL 正则时，chosen 和 rejected 的隐式奖励会一起往下掉（likelihood displacement），
输出越来越短直到只剩「人工智能是AI」；而此时 loss 曲线**完全看不出来**。
所以 `train_dpo.py` 里加了两样东西：RPO 式 NLL 正则、以及每 N 步真采样一次的生成守卫
（有效输出 < 8 token 就判定崩坏，存成 `*_broken_stepN.pth` 并主动停下）。

训练开销（实测）：

| 任务 | 吞吐 / 单步 | 峰值显存 |
|---|---|---|
| 预训练 稠密 bs24 seq340 | ≈ 76 样本/秒 | 4.62 GB |
| SFT 稠密 bs8 seq768 | 210 ms/步 | 4.05 GB |
| LoRA bs8 seq768 | 350 ms/步 | ≈ 2.5 GB |
| DPO bs8 seq768 | 0.25 s/步 | 3.96 GB |
| PPO gen256 | 1.7 s/条 | ≈ 3.4 GB |
| GRPO G=4 gen256 | — | ≈ 3.4 GB |
| **MoE** GRPO G=6 gen256 | — | **8.67 GB（8GB 会 OOM）** |

---

## 5. 快速开始

### 5.1 环境

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`requirements.txt` 里含 `swanlab`；没登录时**必须加 `--no_swanlab`**，否则会在终端卡住等输入。

### 5.2 数据

仓库不含数据（3GB）。用 [MiniMind](https://github.com/jingyaogong/minimind) 的数据集，放到任意目录后传绝对路径：

| 文件 | 用途 |
|---|---|
| `pretrain_20k.jsonl` | 预训练冒烟（2 万条） |
| `pretrain_t2t_mini.jsonl` | 预训练全量（127 万条 / 1.18GB） |
| `sft_20k.jsonl` | SFT（2 万条，MiniMind 格式 `{"conversations": [...]}`） |
| `dpo.jsonl` | DPO（`{"chosen": [...], "rejected": [...]}`） |
| `rlaif.jsonl` | PPO / GRPO（带 `role` 字段的多轮，末轮 assistant 为空待生成） |

### 5.3 三条命令验证链路

```bash
python -m pytest tests -q                      # 20 passed，纯 CPU，20 秒
python main.py                                 # 进度自检：每个组件过没过
python trainer/eval_stages.py --ckpt <权重> --limit 100 --gen
```

### 5.4 训练

```bash
# 预训练（400 步冒烟，实测 1 分 51 秒，loss 8.90 → 4.66）
python trainer/train_pretrain.py --data_path <pretrain_20k.jsonl> \
  --max_steps 400 --batch_size 24 --max_seq_len 256 --save_interval 0 \
  --out_dir out_quick --no_swanlab

# SFT（带早停：验证集 2%，每 500 步评一次，连续 2 次不改善就停）
python trainer/train_sft.py --data_path <sft_20k.jsonl> \
  --from_weight out_quick/pretrain_epoch1.pth \
  --val_ratio 0.02 --eval_interval 500 --patience 2 --no_swanlab

# DPO（lr 一定要小，配 NLL 正则与守卫）
python trainer/train_dpo.py --data_path <dpo.jsonl> \
  --from_weight out/sft_epoch1.pth --learning_rate 1e-6 \
  --nll_coef 0.2 --gen_guard_interval 200 --max_steps 800 --no_swanlab

# MoE 分支
python trainer/train_pretrain.py --use_moe 1 --data_path <pretrain_20k.jsonl> \
  --batch_size 12 --max_seq_len 256 --max_steps 400 --out_dir out_moe --no_swanlab
```

> ⚠️ **冒烟测试务必换输出目录**：`train_dpo.py` / `train_grpo.py` 在最后一步一定会存权重，
> 不指定 `--save_dir out_smoke` 就会**覆盖掉正式训练的产物**（我踩过，`dpo_768.pth` 被 20 步冒烟盖没）。

`.vscode/launch.json` 里有 24 条调试配置，覆盖训练/评测全链路，按 `F5` 即可，不必手敲参数。

---

## 6. 几处值得讲的实现细节

**QK-Norm 作用在 `head_dim=96` 上，且在 RoPE 之前。** 顺序错了不报错，只是效果变差；而且官方权重里那两个 key 会加载不进去。

**带 KV Cache 时不能用 `is_causal=True`。** 此时 `seq_len=1`，`is_causal` 会变成「只看自己」，注意力结果直接错，且**完全不报错**。正确做法是按 cache 长度手工构造 mask。

**MoE 的加权求和是缩进级杀手。** 错写成 `y.view(topk_weight.shape).sum(dim=1)` 之类，训练分支结果会全错却不报错；正确是 `Σ_k weight_k ⊙ expert_k(x)`。验证手段只有一个：训练分支和推理分支的 logits 必须完全相同。

**负载均衡 aux_loss 不要叠加两次。** `CausalLM.forward` 内部已经把 `aux_loss` 加进 loss，训练脚本里再取 `model.model.aux_loss` 加一遍，等于系数翻倍。

**DPO 的 `beta` 和 `lr` 是唯一两个能救命的旋钮。** 本项目在 63.9M 小模型上实测：`lr=5e-6` 直接烧，`1e-6` + `nll_coef=0.2` 才稳住。

**GRPO 的 k3 KL 估计器**：`exp(kl) - kl - 1`，比 `kl` 本身更稳；G=1 时组内 `std()` 为 0 会出 nan，必须单独判掉（此时优势恒为 0）。

---

## 7. 已知限制

- 基座只训了 400 步（约 2 分钟），**绝大多数指标的意义是「链路通了」，不是「模型好用」**。想看真实效果请跑满 `pretrain_t2t_mini.jsonl`（约 5 小时）。
- DPO 目前**没有显著改变行为**：`indomain` / `holdout` 三项几乎不动。根因是基座太弱、对 chosen/rejected 本就没有偏好，链路的正确性由 4 对样本过拟合实验单独验证过（150 步从 0.6932 收敛到 0.1532，与 `-logsigmoid(1.801)` 吻合）。
- PPO / GRPO 只做过冒烟（8 步 / 1 步），没有正式训练结果。
- MoE 只验证了结构与数值一致性，没有做「稠密 vs MoE 同算力对比」。

---

## 8. 参考

- [MiniMind](https://github.com/jingyaogong/minimind) — 架构与数据格式的参照实现（63.9M 版本）
- [MokioMind](https://github.com/Wood-Q/MokioMind) — PPO / GRPO / DPO 的训练脚本风格与数据组织参考

## License

MIT
