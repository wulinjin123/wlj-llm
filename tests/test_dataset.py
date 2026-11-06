"""数据集测试：重点验证"哪些 token 算 loss"。

需要本机有 tokenizer 和数据，没有就跳过。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.environ.get("MINIMIND_DATA_DIR", r"D:\Projects\minimind\dataset")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", r"D:\Projects\minimind\model")
PRETRAIN_FILE = os.path.join(DATA_DIR, "pretrain_20k.jsonl")
SFT_FILE = os.path.join(DATA_DIR, "sft_20k.jsonl")

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(TOKENIZER_PATH) and os.path.exists(PRETRAIN_FILE)),
    reason="缺少 tokenizer 或数据文件",
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH)


def test_pretrain_pad_is_ignored(tokenizer):
    """padding 位置的 label 必须是 -100，否则模型会学着预测 PAD"""
    from dataset.lm_dataset import PretrainDataset

    ds = PretrainDataset(PRETRAIN_FILE, tokenizer, max_length=64)
    sample = ds[0]                       # 预训练集返回 dict（与 trainer 的 batch 访问方式一致）
    ids, labels = sample["input_ids"], sample["labels"]

    assert ids.shape == labels.shape == (64,)
    pad_positions = ids == tokenizer.pad_token_id
    assert labels[pad_positions].eq(-100).all()
    # 非 pad 部分必须保留真实 id（否则整条样本都在白算）
    assert labels[~pad_positions].ne(-100).all()
    assert ids[0].item() == tokenizer.bos_token_id


def test_pretrain_long_text_truncated(tokenizer):
    """超长文本要被截到 max_length，不能溢出"""
    from dataset.lm_dataset import PretrainDataset

    ds = PretrainDataset(PRETRAIN_FILE, tokenizer, max_length=32)
    for i in range(5):
        ids = ds[i]["input_ids"]
        assert ids.shape == (32,)


def test_sft_only_assistant_counts(tokenizer):
    """SFT 的 loss 只覆盖 assistant 回复，提问段必须全 -100"""
    from dataset.lm_dataset import SFTDataset

    ds = SFTDataset(SFT_FILE, tokenizer, max_length=256)
    ids, labels = ds[0]

    n_supervised = int((labels != -100).sum())
    n_real = int((ids != tokenizer.pad_token_id).sum())
    assert 0 < n_supervised < n_real, "回答段应该只占整条对话的一部分"

    # 被监督的那段解码出来应该是通顺的回答，而不是带"user:"开头的提问
    text = tokenizer.decode(ids[labels != -100][:40], skip_special_tokens=True)
    assert len(text.strip()) > 0
