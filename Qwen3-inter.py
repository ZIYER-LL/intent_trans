import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================
# 配置部分
# =====================
GPU_IDX = 3  
DEFAULT_MODEL_DIR = "/work/2024/zhulei/intent-driven"  # 本地模型路径
USE_8BIT = False          # True 可节省显存（需 bitsandbytes）
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
# =====================

def load_model(model_dir=DEFAULT_MODEL_DIR, use_8bit=USE_8BIT):
    """加载本地模型和 tokenizer，返回 (tokenizer, model, device, dtype)"""
    print(f"Loading model from: {model_dir}")

    # 选择 device
    device = torch.device(f"cuda:{GPU_IDX}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 目标 dtype（仅在非 8bit 且 gpu 可用时使用 fp16）
    if torch.cuda.is_available() and not use_8bit:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # 本地加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    # 本地加载模型
    if use_8bit:
        # 8-bit 加载通常需要 device_map="auto"（bitsandbytes）
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            load_in_8bit=True,
            local_files_only=True
        )
        # 当使用 device_map="auto" 时，模型可能已经被分配到 GPU 上的多个设备
        # 这里不强制 .to(device)
    else:
        # 以指定 dtype 加载，然后把模型整体搬到 device
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            local_files_only=True
        )
        model.to(device)

    model.eval()
    return tokenizer, model, device, dtype

def infer(model, tokenizer, prompt, device, dtype):
    """生成文本。确保 inputs 被搬到 device，使用混合精度（GPU + fp16）"""
    # tokenizer 返回的 tensor 放到指定 device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 在 GPU 上并开启混合精度（如果适用）
    use_autocast = (device.type == "cuda" and dtype == torch.float16)
    with torch.inference_mode():
        if use_autocast:
            # 在 CUDA + fp16 情况下使用 autocast 提升性能
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    use_cache=True
                )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                use_cache=True
            )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    print("=== Qwen3-4B 本地交互推理（GPU 优化版） ===")
    tokenizer, model, device, dtype = load_model(USE_8BIT and DEFAULT_MODEL_DIR or DEFAULT_MODEL_DIR, USE_8BIT)

    while True:
        try:
            try:
                prompt = input("\n请输入指令（输入 exit 退出）：\n> ")
                # 自动清理非法 UTF-8 字符，避免报错
                prompt = prompt.encode("utf-8", errors="ignore").decode("utf-8")
            except UnicodeDecodeError:
                print("输入含有非法字符，请重新输入。")
                continue

            if prompt.strip().lower() in ["exit", "quit"]:
                print("退出推理.")
                break

            output = infer(model, tokenizer, prompt, device, dtype)
            print("\n>>> 模型输出:\n", output)
        except KeyboardInterrupt:
            print("\n退出推理.")
            break

if __name__ == "__main__":
    main()

