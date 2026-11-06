"""数据集。

两种格式，差别全在**哪些 token 算 loss**：

  预训练 PretrainDataset：一行一个 {"text": ...}，整条文本每个 token 都算 loss。
      —— 学的是"接下去该说什么字"，所以输入和输出错一位，全部位置都有监督。

  SFT      SFTDataset：   一行一个 {"conversations": [{role, content}, ...]}，
      只有 assistant 回复那一段算 loss，user 提问和特殊 token 全部设为 -100。
      —— 学的是"被问到该怎么答"，如果连提问也算 loss，模型会学着去模仿用户说话。

-100 是 PyTorch cross_entropy 约定的 ignore_index，会自动跳过、不产生梯度。

自检（需要真实的 tokenizer 和一份数据）：
    python -m dataset.lm_dataset
"""

import json
import os
import random
from tracemalloc import start

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

os.environ["TOKENIZER_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    """整段文本算 loss。

    一条样本的构造：
        [BOS] + 正文（截断到 max_length-2） + [EOS] + [PAD] * 补齐
    然后 labels = input_ids，但 PAD 位置置 -100。

    两个容易漏的点：
      - 截断长度要留 2 个位置给 BOS/EOS，否则补齐时会超过 max_length。
      - PAD 必须置 -100。不置的话模型会被训练成"预测出 PAD"，生成时容易直接吐 PAD。
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 340):
        super().__init__()
        self.tokenizer=tokenizer
        self.max_length=max_length#输入给GPU的最大长度
        #使用huggingface的惰性加载，避免一次性读入大文件
        self.samples=load_dataset("json",data_files=data_path,split="train")
    def __len__(self):
        return len(self.samples)
    #getitem函数：我们拿到的是每一行的jsonl，要输出的是Input_ids,attention_mask,labels
    
    def __getitem__(self,index):
        sample=self.samples[index]
        #tokenizer把文本转化成input_ids
        tokens=self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length-2,#流出位置给BOS和EOS
            truncation=True,#超出自动截断
        
        ).input_ids
        #需要加上EOS,BOS和PAD填充
        tokens=[self.tokenizer.bos_token_id]+tokens+[self.tokenizer.eos_token_id]
        input_ids=tokens+[self.tokenizer.pad_token_id]*(self.max_length-len(tokens))
        #转换成张量
        input_ids=torch.tensor(input_ids,dtype=torch.long)
        #需要编写labels，防止PAD参与loss计算
        labels=input_ids.clone()
        labels[labels==self.tokenizer.pad_token_id]=-100#PAD位置置-100,防止PAD参与loss计算
        #需要编写attention_massk,告诉模型哪些位置有效，哪些位置是PAD,非PAD位置是1,PAD位置是0
        attention_mask=(input_ids!=self.tokenizer.pad_token_id).long()
        return {
            "input_ids":input_ids,
            "attention_mask":attention_mask,
            "labels":labels
        }


def pre_processing_chat(conversations, add_system_ratio=0.2):
    """按概率给样本插一条 system prompt（数据增强，让模型见过带/不带 system 两种输入）。"""
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]
    if conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [{"role": "system", "content": random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2, remove_empty_think=None):
    """以 80% 概率删掉空 think 块，避免模型学成"每次都先输出一对空 think"。"""
    if "<think>\n\n</think>\n\n" in prompt_content:
        if remove_empty_think is None:
            remove_empty_think = random.random() > empty_think_ratio
        if remove_empty_think:
            prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content


class SFTDataset(Dataset):
    """只有 assistant 回复段算 loss。

    做法：先用 tokenizer 的 chat template 把整段对话拼成一串字符串再分词，
    然后**扫描**这串 id，找出每一处 `assistant\\n` 标记，从标记后开始、
    到下一个 `<eos>\\n` 结束，这段范围内的 label 才填真实 id，其余全 -100。

    为什么不用"按角色逐段拼"的写法：chat template 会插入各种特殊 token，
    手拼很容易和模板不一致，导致训练/推理时 prompt 格式对不上。
    """

    def __init__(self,jsonl_path,tokenizer,max_length=1024):
        super().__init__()
        self.tokenizer=tokenizer
        self.max_length=max_length
        # load_dataset 是模块级函数，不是 Dataset 的方法
        self.samples=load_dataset("json",data_files=jsonl_path,split="train")
        # 必须取 .input_ids：漏了拿到的是 BatchEncoding，后面"list 切片 == 对象"恒为 False，
        # labels 会全 -100，loss 直接变 nan
        self.bos_id=tokenizer(f"{tokenizer.bos_token}assistant\n",add_special_tokens=False).input_ids
        # eos 只拼 "\n"。写成 "assistant\n" 这个 5-token 序列在正文里根本不出现，扫描扫不到边界
        self.eos_id=tokenizer(f"{tokenizer.eos_token}\n",add_special_tokens=False).input_ids
        

    def __len__(self):
        return len(self.samples)
    def craeate_chat_prompt(self,conversations):
        # 原写法 conversations[0]["function"] 会 KeyError（数据里只有 role/content/reasoning_content）；
        # apply_chat_template 的参数是 tokenize 不是 tokenizer，写错会被 **kwargs 吞掉不报错
        messages=[]
        tools=None
        for message in conversations:
            message=dict(message)
            if message.get("role")=="system" and message.get("tools"):
                tools=(
                    json.loads(message["tools"])
                    if isinstance(message["tools"],str)
                    else message["tools"]
                )
            if message.get("tool_calls") and isinstance(message["tool_calls"],str):
                message["tool_calls"]=json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,tokenize=False,add_generation_prompt=False,tools=tools
        )


    def generate_labels(self,input_ids):
        #让所有的inputid都为-100，-100是交叉熵损失默认忽视的值
        labels=[-100]*len(input_ids)
        i=0
        while i<len(input_ids):
            if input_ids[i:i+len(self.bos_id)]==self.bos_id:
                start=i+len(self.bos_id)
                end=start
                while end<len(input_ids):
                    if input_ids[end:end+len(self.eos_id)]==self.eos_id:
                        break
                    end+=1
                # 注意这两段在外层 while 里，不能缩进到 while end 内部：
                # 缩进去会每前进一个 token 就重标一遍，而且 i 被提前设成末尾导致直接退出
                for j in range(start,min(end+len(self.eos_id),len(input_ids))):
                    labels[j]=input_ids[j]
                i=end+len(self.eos_id) if end<len(input_ids) else len(input_ids)
            else:
                # 少了这个分支，没命中 bos 时 i 不增加，会死循环
                i+=1
        return labels


    def __getitem__(self, index):
        #是否要随机插入system_prompt
        sample=self.samples[index]
        conversations=pre_processing_chat(sample["conversations"])
        
        #用chat_template把对话转化成文本
        prompt=self.craeate_chat_prompt(conversations)
        #清理空think块
        prompt=post_processing_chat(prompt)
        
        #tokenizer截断，并且补足pad
        input_ids=self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids+=[self.tokenizer.pad_token_id]*(self.max_length-len(input_ids))


        #生成标签，只让assitant参与loss计算
        labels=self.generate_labels(input_ids)
        return torch.tensor(input_ids,dtype=torch.long),torch.tensor(labels,dtype=torch.long)


class DPODataset(Dataset):
    """偏好对数据集：同一问题给一条「好回答」和一条「差回答」，让模型学会排序。

    数据格式（本地 dpo.jsonl 实测 17166 条，全部是单轮 user/assistant）：
        {"chosen":   [{"role":"user",...},{"role":"assistant",...}],
         "rejected": [{"role":"user",...},{"role":"assistant",...}]}

    和 SFTDataset 的关键差别：
      SFT 要的是「具体 token 的 label」（喂给 cross_entropy）；
      DPO 要的是「0/1 掩码」，因为 DPO 的损失是把整段回答的 log 概率
      求和（或平均）后做对比，不需要逐 token 的 label。

    返回 8 个张量，chosen / rejected 各 4 个：
        x_chosen / x_rejected           输入（去掉最后一个 token）
        y_chosen / y_rejected           目标（去掉第一个 token，错一位对齐）
        mask_chosen / mask_rejected     只保留 assistant 回答段，其余 0
        attention_mask_chosen / _rejected   padding 位置为 0

    ⚠️ 关于 attention_mask：本项目 Model.forward 收了它但并没有传进 Attention
    （实测带不带它 loss 完全相同）。**这里之所以仍然安全，是因为 DPO 用右填充**：
    padding 全在序列末尾，而 attention 是 causal 的，末尾的 token 不会被前面的
    位置"看到"，所以不会污染 assistant 段的 log 概率。
    换成左填充（PPO 生成时必须左填充）就会污染，那正是 train_ppo.py 必须
    逐条处理、不能拼 batch 的原因。
    """

    def __init__(self, file_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # pad 回退到 0，保证补齐不会因为 pad_token_id=None 崩掉
        self.padding = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )
        # 和 SFTDataset 完全一致：先 tokenize「assistant\n」这个起始标记，
        # 后面靠扫描它来定位每段回答的起点
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids
        self.samples = load_dataset("json", data_files=file_path, split="train")

    def __len__(self):
        return len(self.samples)

    def generate_loss_mask(self, input_ids):
        """扫描出每段 assistant 回答，返回 0/1 掩码（1=要计入 log 概率）。

        算法和 SFTDataset.generate_labels 一模一样，只是填 1 而不是填 token id：
        找到 bos_id → 从它后面开始 → 找到 eos_id 结束 → 这段标 1（含 eos）。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), len(input_ids))):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def _encode(self, conversations):
        """一段对话 -> (x, y, mask, attention_mask)，四者长度都是 max_length-1。"""
        text = self.tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False
        )
        text = post_processing_chat(text, remove_empty_think=True)   # 固定行为，可复现
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",          # 右填充，原因见类 docstring
        )
        ids = enc["input_ids"]
        loss_mask = self.generate_loss_mask(ids)

        # 自回归错一位：输入 [:-1]，目标是 [1:]，mask 也跟着 [1:]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        m = torch.tensor(loss_mask[1:], dtype=torch.long)
        am = torch.tensor(ids[:-1], dtype=torch.long).ne(self.padding).long()
        return x, y, m, am

    def __getitem__(self, index):
        sample = self.samples[index]
        xc, yc, mc, amc = self._encode(sample["chosen"])
        xr, yr, mr, amr = self._encode(sample["rejected"])
        return {
            "x_chosen": xc, "y_chosen": yc,
            "mask_chosen": mc, "attention_mask_chosen": amc,
            "x_rejected": xr, "y_rejected": yr,
            "mask_rejected": mr, "attention_mask_rejected": amr,
        }


class RLAIFDataset(Dataset):
    """PPO / RL 用的数据集：返回**字符串**，不返回定长张量。

    和 SFTDataset 的根本差别：
      SFT 的答案数据里是固定的，可以离线 tokenize 成定长张量；
      PPO 要先把 prompt 喂给 actor **在线生成**回复，回复每轮都不一样、长度不定，
      只能等生成完再 tokenize，所以数据集只负责把「上文」和「参考答案」分开。

    返回 {"prompt": str, "answer": str}
        prompt  已带 add_generation_prompt=True 的结尾标记，模型接着往下写就行
        answer  末轮参考答案，只用来算奖励，不参与 loss
    """

    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        # 保留 bos/eos，方便以后做 mask 或截断时用；本类当前不参与计算
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}", add_special_tokens=False
        ).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """把一段多轮对话拆成 (上文 prompt, 末轮参考答案 answer)。

        与参考实现的差异：那边是按奇偶下标硬猜 user/assistant，
        本地 rlaif.jsonl 19502 条 **100% 自带 role 字段**，所以优先读 role，
        只有在老数据缺 role 时才退回奇偶规则。
        """
        messages = []
        for i, turn in enumerate(conversations):
            # 优先读数据自带的 role；老数据没 role 才退回奇偶规则
            role = turn.get("role") or ("user" if i % 2 == 0 else "assistant")
            messages.append({"role": role, "content": turn["content"]})

        # 末轮必须是 assistant 才有参考答案。如果数据没答完（末轮是 user），
        # 就把整段当成 prompt、answer 留空，不要在 prompt 里塞一个假答案。
        if messages and messages[-1]["role"] == "assistant":
            answer = messages[-1]["content"]
            context = messages[:-1]
        else:
            answer = ""
            context = messages

        # add_generation_prompt=True 让模板在末尾补上「assistant 开始回复」的引导标记
        prompt = self.tokenizer.apply_chat_template(
            context, tokenize=False, add_generation_prompt=True
        )
        # 传 remove_empty_think=True 而不是默认的随机删：RL 要可复现，
        # 同一个 prompt 会被 actor / old_actor / ref 三个模型各前向一次，格式必须稳定
        prompt = post_processing_chat(prompt, remove_empty_think=True)
        return prompt, answer

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt, answer = self.create_chat_prompt(sample["conversations"])
        return {"prompt": prompt, "answer": answer}



if __name__ == "__main__":
    # 自检：打印一条样本，肉眼确认"该算 loss 的位置"对不对
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from transformers import AutoTokenizer

    tok_path = os.environ.get("TOKENIZER_PATH", r"D:\Projects\minimind\model")
    data_dir = os.environ.get("MINIMIND_DATA_DIR", r"D:\Projects\minimind\dataset")
    tok = AutoTokenizer.from_pretrained(tok_path)

    ds = PretrainDataset(os.path.join(data_dir, "pretrain_20k.jsonl"), tok, max_length=64)
    sample = ds[0]                       # PretrainDataset 返回 dict，SFTDataset 返回元组
    ids, labels = sample["input_ids"], sample["labels"]
    print(f"[Pretrain] 样本数={len(ds)}  shape={tuple(ids.shape)}")
    print("  有效 label 数 =", int((labels != -100).sum()))
    print("  解码 =", tok.decode(ids[labels != -100][:40], skip_special_tokens=True))

    ds2 = SFTDataset(os.path.join(data_dir, "sft_20k.jsonl"), tok, max_length=256)
    ids2, labels2 = ds2[0]
    print(f"\n[SFT] 样本数={len(ds2)}  shape={tuple(ids2.shape)}")
    print("  有效 label 数 =", int((labels2 != -100).sum()),
          "/ 全量 =", int((ids2 != tok.pad_token_id).sum()))
    print("  只算 loss 的部分 =", repr(tok.decode(ids2[labels2 != -100][:60])))

    ds3 = RLAIFDataset(os.path.join(data_dir, "rlaif.jsonl"), tok)
    s3 = ds3[0]
    print(f"\n[RLAIF] 样本数={len(ds3)}  prompt 字符数={len(s3['prompt'])}")
    print("  prompt 结尾 =", repr(s3["prompt"][-60:]))
    print("  answer 前 60 字 =", repr(s3["answer"][:60]))
    p_ids = tok(s3["prompt"], add_special_tokens=False).input_ids
    print(f"  prompt token 数={len(p_ids)}  末 token={p_ids[-1]} "
          f"(应为 assistant 引导标记)")

    ds4 = DPODataset(os.path.join(data_dir, "dpo.jsonl"), tok, max_length=1024)
    s4 = ds4[0]
    print(f"\n[DPO] 样本数={len(ds4)}  每项 shape={tuple(s4['x_chosen'].shape)}")
    for side in ("chosen", "rejected"):
        y = s4[f"y_{side}"]
        m = s4[f"mask_{side}"]
        am = s4[f"attention_mask_{side}"]
        n_mask = int(m.sum())
        # mask 为 1 的位置，取对应的目标 token 解码，应当正好是 assistant 回答
        print(f"  {side:<8} 有效 mask={n_mask:>4}  attention_mask 非零={int(am.sum()):>4}")
        print(f"           回答内容 = {tok.decode(y[m == 1][:50], skip_special_tokens=True)!r}")
    # 关键不变式：mask 里为 1 的位置必须都落在非 padding 区
    bad = 0
    for side in ("chosen", "rejected"):
        m, am = s4[f"mask_{side}"], s4[f"attention_mask_{side}"]
        bad += int((m.bool() & ~am.bool()).sum())
    print(f"  mask 落在 padding 区的 token 数 = {bad}  (必须为 0)")
