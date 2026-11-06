# tiny-llm-from-scratch · 从零手写一个 63.91M 的 LLM

> 不调用 `transformers` 的模型类，从 `nn.Linear` 开始手写一个和 Llama / MiniMind 同架构族的
> decoder-only 语言模型。目标：一份能扛住复试 20 分钟追问的工程 + 一条自己跑出来的 loss 曲线。

---

## 0. 目录结构

```
tiny-llm/
├── .vscode/                 解释器锁定 + 调试入口
├── configs/                 64m.json（63.91M）/ 30m.json（调试用，更小更快）
├── dataset/
│   └── lm_dataset.py        PretrainDataset / SFTDataset        【已给，不用写】
├── model/
│   └── model.py             ★ 全部手写内容都在这一个文件里
├── trainer/
│   ├── train_pretrain.py    预训练入口                          【已给】
│   └── trainer_utils.py     学习率调度 / 日志 / 检查点          【已给】
├── tests/                   20 个断言，验收标准就在这里
├── main.py                  进度自检：现在做到第几步，一跑就知道
├── .python-version          3.11
├── pyproject.toml
└── README.md
```

**只有 `model/model.py` 需要你写。** 里面 18 处 `raise NotImplementedError` 就是待填的坑，
其余文件都是"工具"——能读懂就行，抄了不亏。

为什么全部塞一个 `model.py`：模型是**一层层长出来的**（RMSNorm → RoPE → 注意力 → FFN → Block → 整模型），
写在外面的测试可以一步步验收，比一开始就拆成 7 个文件更符合"从零"的过程。

---

## 1. 环境（已就绪，不需要再装）

| 项 | 值 |
|---|---|
| 解释器 | `D:\Anaconda\envs\minimind\python.exe`（Python 3.11.18） |
| PyTorch | 2.6.0+cu124，CUDA 可用，bf16 支持 |
| GPU | RTX 4060 Laptop **8GB**（安全线：显存峰值 ≤ 5GB） |
| 数据 | `D:\Projects\minimind\dataset\`（**不复制**，直接引用绝对路径） |
| 参考实现 | `D:\Projects\minimind\`（只在"卡住"和"对齐验证"时打开） |

> ⚠️ 别在 Git Bash 里敲 `python`。你机器上 PATH 第一位是 WorkBuddy 自带的沙箱 Python，
> `pip install` 会装错地方。用 Anaconda Prompt，或者在 VSCode 终端里写全路径。

---

## 2. 从现在开始，每天的第一条命令

```
python main.py
```

它会告诉你每一节过没过。全部通过后就可以训练。

```
python -m pytest tests -q
```

跑全部断言。**红了不是坏事，是地图**——报错信息里写着是 S 几没实现。

---

## 3. 推进顺序（8 节，全部在 `model/model.py` 里）

| 节 | 内容 | 完成标志 | 预计 |
|---|---|---|---|
| S1 | `RMSNorm` | 与 `F.rms_norm` 误差 < 1e-6 | 0.5h |
| S2 | `precompute_freqs_cis` / `rotate_half` / `apply_rotary_pos_emb` | 整体平移后内积不变 < 1e-5 | 3h |
| S3 | `repeat_kv` / `Attention`（GQA + KV Cache + QK-Norm） | **逐步解码 == 整段前向** | 3h |
| S5 | `FeedForward`（SwiGLU） | 参数量精确对上 | 1h |
| S6 | `Block`（Pre-Norm 残差） | 形状不变、梯度可通 | 1h |
| S7 | `Model`（embedding + N×Block + norm） | hidden 形状 (b,t,768) | 2h |
| S8 | `CausalLM`（lm_head + loss + generate） | **参数 63.91M；state_dict 键名与官方完全一致** | 4h |

### 一条纪律

```
1. 读 model/model.py 里这一节的 docstring（目标 + 接口 + 陷阱）
2. 关掉参考实现，凭理解自己写
3. python -m pytest tests -q
4. 绿了 → 下一节；红了 → 先自己调，卡满 40 分钟再看 D:\Projects\minimind
5. 绿了之后回头对照参考实现，把差异记进下面第 6 节的表
```

第 5 步最重要。**能说出"我写的和他写的哪里不一样、为什么"，这个组件才算真的会了。**

---

## 4. 训练

S1~S8 全绿之后：

```
:: 冒烟（1 分钟，先确认链路通）
python trainer/train_pretrain.py --data_path D:\Projects\minimind\dataset\pretrain_20k.jsonl --epochs 1 --batch_size 8 --log_interval 20

:: 正式（全量 127 万条，约 4.6 小时，晚上挂机）
python trainer/train_pretrain.py --data_path D:\Projects\minimind\dataset\pretrain_t2t_mini.jsonl --epochs 1 --batch_size 24 --log_interval 50

:: 断点续训
python trainer/train_pretrain.py ... --from_resume 1
```

实测参考值（本机 4060 8GB，batch 24 / seq 340）：显存 ≈ 4.6GB，吞吐 ≈ 76 样本/秒。

**别把 batch 调到 32 以上。** 实测 768/32/512 会报 8.38GB —— 超过物理显存后不报 OOM，
而是偷偷用内存当显存，单步耗时涨 3-5 倍，你会以为是代码有问题。

---

## 5. 数据说明

| 文件 | 用途 | 规模 |
|---|---|---|
| `pretrain_t2t_mini.jsonl` | 预训练全量 | 127 万条 / 1.18GB |
| `pretrain_20k.jsonl` | 冒烟用子集 | 2 万条 |
| `sft_t2t_mini.jsonl` | SFT 全量 | 90 万条 / 1.62GB |
| `sft_20k.jsonl` | SFT 冒烟 | 2 万条 |

两种数据集的区别只在**哪些 token 算 loss**：
- 预训练：整条文本每个 token 都算，只跳过 pad（label = -100）
- SFT：只有 assistant 回复段算，提问段和特殊 token 全 -100

验证方法：`python -m dataset.lm_dataset` 会打印一条样本，肉眼确认"被监督的部分"是不是回答。

---

## 6. 差异记录（每完成一节就填）

| 组件 | 我的实现 | 参考实现 | 差异原因 |
|---|---|---|---|
| S1 RMSNorm | | | |
| S2 RoPE | | | |
| S3 Attention | | | |
| S3 KV Cache | | | |
| S5 SwiGLU | | | |
| S6 Block | | | |
| S8 权重绑定 | | | |

---

## 7. 常见坑

1. **`seq_len` 是显存主因，不是模型宽度。** 340 → 512 显存涨 40%、时间涨 60%。
2. **显存不报 OOM 也会变慢。** 超过 8GB 会掉进共享内存，单步耗时涨 3-5 倍。开 `nvidia-smi -l 2` 盯着。
3. **`num_workers` 在 Windows 上设 0。** 多进程 DataLoader 会静默卡死（有人卡了 37 分钟）。
4. **`freqs_cos` / `freqs_sin` 必须 `persistent=False`。** 否则它们会进 state_dict，
   加载官方权重时报 unexpected key。
5. **QK-Norm 别漏。** 它作用在 `head_dim=96` 上、在 RoPE **之前**。少了它，
   官方权重里那两个 key 加载不进去。
6. **带 cache 时不能用 `is_causal=True`。** 此时 seq_len=1，`is_causal` 会变成"只看自己"，
   注意力结果直接错，而且不报错。
7. **别用 `strict=False` 掩盖问题。** 正确判据是 `missing == [] and unexpected == []`。

---

## 8. 相关内容

- 详细计划表（含每节的理论要点、面试话术、完整源码附录）：
  `D:\WorkBuddy-Output\2026-11-02\手敲大模型_从零实现计划表_2026-11-02.md`
- 参考实现：`D:\Projects\minimind`（MiniMind，63.9M 版本）
