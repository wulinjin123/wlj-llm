"""LoRA：低秩旁路 W + B@A。"""
import torch
from torch import nn


class LoRA(nn.Module):
    """一个旁路：y = W x + B(A x)。

    A 用正态初始化、B 全零 —— 这样挂上去的瞬间 B(A x) 恒为 0，
    模型输出和挂之前完全一样，不会破坏已经训好的基座。
    （如果 B 也随机初始化，一挂上去就把 SFT 学到的东西打乱了。）
    """

    def __init__(self, in_features, out_features, rank):
        super(LoRA, self).__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        self.A.weight.data.normal_(mean=0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=8):
    """给每个「方阵 Linear」挂一个旁路，返回命中的层数。

    为什么只挑方阵（in==out）：
      Attention 里的 q_proj / o_proj 是 768x768 方阵；
      MLP 的 gate/up/down_proj 不是方阵，lm_head（6400x768）也不是。
    只挂 q/o 两个，8 层共 16 个点，rank16 时参数 393,216（0.61%），
    权重文件才 0.76MB —— 这就是 LoRA 号称「省」的地方。

    每一层必须单独挂实例、单独包 forward：
    不能在循环外面创建一个 lora 反复复用，那样所有层会共享同一份 A/B。
    """
    n_hit = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.weight.shape[0] != module.weight.shape[1]:
            continue                        # 非方阵不挂

        # module.lora = ... 会被 nn.Module 注册成子模块，
        # 于是 state_dict 里自动出现 "{name}.lora.A.weight" 这样的 key
        # 必须跟随宿主层所在的 device/dtype：调用方通常先把 model.to("cuda") 再挂旁路，
        # 这里若留在 CPU，第一次前向就会报 "two devices, cuda:0 and cpu"
        module.lora = LoRA(
            module.weight.shape[0], module.weight.shape[1], rank
        ).to(device=module.weight.device, dtype=module.weight.dtype)

        original_forward = module.forward

        # 默认参数把「原 forward」和「本层的 lora」绑死在闭包里，
        # 否则循环结束后所有层都会指向最后一层
        def forward_with_lora(x, layer1=original_forward, layer2=module.lora):
            return layer1(x) + layer2(x)

        module.forward = forward_with_lora
        n_hit += 1
    return n_hit


def save_lora(model, path):
    """只存旁路参数（不带基座），返回存了多少个张量。"""
    state = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            prefix = f"{name}.lora." if name else "lora."
            state.update(
                {
                    f"{prefix}{k}": v.detach().cpu().half()
                    for k, v in module.lora.state_dict().items()
                }
            )
    torch.save(state, path)
    return len(state)


def load_lora(model, path):
    """把旁路参数灌回一个已经 apply_lora 过的模型，返回命中的层数。"""
    state = torch.load(path, map_location="cpu", weights_only=False)
    n_load = 0
    for name, module in model.named_modules():
        if not hasattr(module, "lora"):
            continue
        prefix = f"{name}.lora." if name else "lora."
        part = {
            k[len(prefix):]: v
            for k, v in state.items()
            if k.startswith(prefix)
        }
        if part:
            module.lora.load_state_dict(part)
            n_load += 1
    return n_load


def merge_lora(model, lora_path, save_path=None):
    """把 B@A 加回 W，得到一份不带旁路的普通权重。

    返回 (合并层数, 合并后的 state_dict)。

    ⚠️ 必须走 state_dict() 而不是 named_parameters()：
    tie_word_embeddings=True 时 lm_head.weight 和 embed_tokens.weight
    是**同一个张量对象**，named_parameters() 会去重只返回一个 key，
    照它写回就会漏掉另一个，load_weights(strict=True) 直接报 missing key。
    """
    state = torch.load(lora_path, map_location="cpu", weights_only=False)
    sd = model.state_dict()
    n_merge = 0

    for name, module in model.named_modules():
        if not hasattr(module, "lora"):
            continue
        prefix = f"{name}.lora." if name else "lora."
        key_w = f"{name}.weight"
        if key_w not in sd:
            continue
        A = state.get(f"{prefix}A.weight")
        B = state.get(f"{prefix}B.weight")
        if A is None or B is None:
            continue
        # W' = W + B @ A（用 fp32 算，避免 fp16 相乘丢精度）
        delta = (B.float() @ A.float()).to(sd[key_w].dtype)
        sd[key_w] = sd[key_w] + delta
        n_merge += 1

    # 合并完就没有旁路了，把 lora.* 的 key 去掉，否则加载时报 unexpected keys
    sd = {k: v for k, v in sd.items() if ".lora." not in k}

    if save_path:
        torch.save({k: v.half() for k, v in sd.items()}, save_path)
    return n_merge, sd
