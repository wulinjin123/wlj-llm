"""手动试生成：在终端里输入一句话，看模型往下续写什么。

用法（VS Code）：
    打开本文件 → 运行和调试面板选「运行当前文件」→ Ctrl+F5
    然后在下方 TERMINAL 里输入 prompt，回车。
    输入空行（直接回车）退出。

改权重路径：把下面 CKPT 换成 out/pretrain_epoch1.pth 就是正式训练的模型。
注意：pretrain 阶段只学「续写」，不是对话模型，它不会回答你的问题，只会接着往下编。
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.model import CausalLM, ModelConfig     # noqa: E402
from trainer.trainer_utils import load_tokenizer  # noqa: E402

CKPT = "out_quick/pretrain_epoch1.pth"   # 超快版权重；正式训练产物是 out/pretrain_epoch1.pth
# 权重是 MoE 训出来的就加 --use_moe（或直接把这里改成 True），否则 strict 加载会报 missing keys
USE_MOE = "--use_moe" in sys.argv


def main():
    tok = load_tokenizer()

    model = CausalLM(ModelConfig(use_moe=USE_MOE)).cuda()
    state = torch.load(ROOT / CKPT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=True)
    model.eval()
    print(f"已加载 {CKPT}  missing={len(missing)}  unexpected={len(unexpected)}")
    print("输入一句话回车，看模型续写；直接回车（空行）退出。\n")

    while True:
        try:
            q = input("你: ")
        except (EOFError, KeyboardInterrupt):
            break
        if not q.strip():
            break

        ids = torch.tensor(
            [[tok.bos_token_id] + tok(q, add_special_tokens=False).input_ids],
            device="cuda",
        )
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=48, temperature=0.8, top_p=0.9,
                                 eos_token_id=tok.eos_token_id, use_cache=True)
        print("续写:", tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
