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
JSON_MODE = True          # True 自动生成结构化 JSON 输出
# =====================

def load_model(model_dir=DEFAULT_MODEL_DIR, use_8bit=USE_8BIT):
    """加载本地模型和 tokenizer，返回 (tokenizer, model, device, dtype)"""
    print(f"Loading model from: {model_dir}")

    # 选择 device
    device = torch.device(f"cuda:{GPU_IDX}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 目标 dtype
    dtype = torch.float16 if torch.cuda.is_available() and not use_8bit else torch.float32

    # 本地加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    # 本地加载模型
    if use_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            load_in_8bit=True,
            local_files_only=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            local_files_only=True
        )
        model.to(device)

    model.eval()
    return tokenizer, model, device, dtype

def infer(model, tokenizer, prompt, device, dtype, history=None):
    """生成文本。history 为多轮上下文，可选"""
    # 拼接多轮上下文
    if history:
        full_prompt = "\n".join(history + [prompt])
    else:
        full_prompt = prompt

    # 如果开启 JSON 模式，包装 prompt
    if JSON_MODE:
        full_prompt = (
            f"你是一个网络配置工程师，请根据以下指令生成结构化 JSON，"
            f"字段包括 bandwidth(Mbps), latency(ms), qos, optimization。"
            f"不要输出多余文字，只输出 JSON。\n指令：{full_prompt}"
        )

    # tokenizer 返回的 tensor 放到 device
    inputs = tokenizer(full_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # GPU + fp16 时开启 autocast
    use_autocast = (device.type == "cuda" and dtype == torch.float16)
    with torch.inference_mode():
        if use_autocast:
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
    print("=== Qwen3-4B 本地交互推理（GPU + JSON模式） ===")
    tokenizer, model, device, dtype = load_model(USE_8BIT and DEFAULT_MODEL_DIR or DEFAULT_MODEL_DIR, USE_8BIT)

    history = []  # 多轮历史上下文

    while True:
        try:
            try:
                prompt = input("\n请输入指令（输入 exit 退出）：\n> ")
                # 自动清理非法 UTF-8 字符，去掉全角空格
                prompt = prompt.encode("utf-8", errors="ignore").decode("utf-8").replace("　", " ")
            except UnicodeDecodeError:
                print("输入含有非法字符，请重新输入。")
                continue

            if prompt.strip().lower() in ["exit", "quit"]:
                print("退出推理.")
                break

            output = infer(model, tokenizer, prompt, device, dtype, history)
            print("\n>>> 模型输出:\n", output)

            # 保存多轮历史
            history.append(prompt)
            history.append(output)
        except KeyboardInterrupt:
            print("\n退出推理.")
            break

if __name__ == "__main__":
    main()

