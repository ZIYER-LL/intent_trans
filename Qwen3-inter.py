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

    device = torch.device(f"cuda:{GPU_IDX}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dtype = torch.float16 if torch.cuda.is_available() and not use_8bit else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

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

def clean_input(text: str) -> str:
    """清理中文输入，替换全角符号，去掉不可打印字符"""
    replacements = {
        "，": ",",
        "。": ".",
        "：": ":",
        "；": ";",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "　": " ",  # 全角空格
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # 去掉不可打印字符
    text = "".join(c for c in text if c.isprintable())
    # 确保 UTF-8 编码安全
    return text.encode("utf-8", errors="ignore").decode("utf-8")

def infer(model, tokenizer, prompt, device, dtype, history=None):
    """生成文本。history 为多轮上下文，可选"""
    if history:
        full_prompt = "\n".join(history + [prompt])
    else:
        full_prompt = prompt

    if JSON_MODE:
        full_prompt = (
            f"你是一个网络配置工程师，请根据以下指令生成结构化 JSON，"
            f"字段包括 bandwidth(Mbps), latency(ms), qos, optimization。"
            f"不要输出多余文字，只输出 JSON。\n指令：{full_prompt}"
        )

    inputs = tokenizer(full_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

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

    history = []

    while True:
        try:
            prompt = input("\n请输入指令（输入 exit 退出）：\n> ")
            prompt = clean_input(prompt)

            if prompt.strip().lower() in ["exit", "quit"]:
                print("退出推理.")
                break

            output = infer(model, tokenizer, prompt, device, dtype, history)
            print("\n>>> 模型输出:\n", output)

            history.append(prompt)
            history.append(output)
        except KeyboardInterrupt:
            print("\n退出推理.")
            break
        except Exception as e:
            print(f"推理出现异常: {e}")

if __name__ == "__main__":
    main()


