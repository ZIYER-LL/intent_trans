import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================
# 配置部分
# =====================
DEFAULT_MODEL_DIR = "/work/2024/zhulei/intent-driven"  # 本地模型路径
USE_8BIT = False          # True 可节省显存
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
# =====================

def load_model(model_dir=DEFAULT_MODEL_DIR, use_8bit=USE_8BIT):
    """加载本地模型和 tokenizer"""
    print(f"Loading model from: {model_dir}")
    
    if torch.cuda.is_available():
        device_map = "auto"
        dtype = torch.float16
    else:
        device_map = {"": "cpu"}
        dtype = torch.float32
    
    # 本地加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    
    # 本地加载模型
    if use_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map=device_map,
            load_in_8bit=True,
            local_files_only=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map=device_map,
            torch_dtype=dtype,
            local_files_only=True
        )
    return tokenizer, model

def infer(model, tokenizer, prompt):
    """生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    print("=== Qwen3-4B 本地交互推理 ===")
    tokenizer, model = load_model()
    
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
            
            output = infer(model, tokenizer, prompt)
            print("\n>>> 模型输出:\n", output)
        except KeyboardInterrupt:
            print("\n退出推理.")
            break

if __name__ == "__main__":
    main()

