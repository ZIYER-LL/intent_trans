import torch
from transformers import AutoModelForCausalLM

MODEL_DIR = "/work/2024/zhulei/models/qwen3-4b"

# 你想优先注入的“候选后缀”（按常见 Llama/Qwen 系列）
CANDIDATE_SUFFIXES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    # 下面是一些可能的变体（不同版本/实现会出现）
    "wq", "wk", "wv", "wo",
    "w1", "w2", "w3",
    "c_attn", "c_proj", "dense", "fc1", "fc2"
]

def collect_linear_module_names(model):
    linear_types = (torch.nn.Linear,)
    names = []
    for name, module in model.named_modules():
        if isinstance(module, linear_types):
            names.append(name)
    return names

def recommend_target_suffixes(linear_names, candidates):
    suffix_hit = []
    for sfx in candidates:
        if any(n.endswith("." + sfx) or n.endswith(sfx) for n in linear_names):
            suffix_hit.append(sfx)
    return suffix_hit

def main():
    # 用 CPU 加载，避免占用你训练 GPU（4B CPU 加载可以接受）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="cpu",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    linear_names = collect_linear_module_names(model)
    print(f"Total Linear modules: {len(linear_names)}\n")

    # 打印一些典型路径，方便你肉眼确认结构
    print("Examples of Linear module names (first 40):")
    for n in linear_names[:40]:
        print("  ", n)

    # 推荐 target_modules（后缀命中）
    targets = recommend_target_suffixes(linear_names, CANDIDATE_SUFFIXES)
    print("\nRecommended TARGET_MODULES (suffixes that actually exist):")
    print(targets)

    # 额外：强制检查 attention / mlp 是否都覆盖到（粗略）
    attn = [n for n in linear_names if "attn" in n.lower() or "self_attn" in n.lower()]
    mlp  = [n for n in linear_names if "mlp"  in n.lower() or "ffn" in n.lower()]

    print(f"\nHeuristic counts: attn_linear={len(attn)}, mlp_linear={len(mlp)}")
    if len(attn) > 0:
        print("attn example:", attn[0])
    if len(mlp) > 0:
        print("mlp  example:", mlp[0])

if __name__ == "__main__":
    main()
