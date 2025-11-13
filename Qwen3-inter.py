import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_dir: str, use_8bit: bool = False):
    print(f"Loading model from: {model_dir}")
    # 选择设备与 dtype
    if torch.cuda.is_available():
        device_map = "auto"
        dtype = torch.float16
    else:
        device_map = {"": "cpu"}
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # model 加载
    if use_8bit:
        print("Loading in 8-bit mode (bitsandbytes required).")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, device_map=device_map, load_in_8bit=True, trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, device_map=device_map, torch_dtype=dtype, trust_remote_code=True
        )
    model.eval()
    return tokenizer, model

def build_prompt(history):
    """把历史消息拼成单个 prompt。简单拼接，按需修改格式/指令模板。"""
    if not history:
        return ""
    # 使用双换行分隔轮次，格式：User: ... \n Assistant: ...
    parts = []
    for role, text in history:
        if role == "user":
            parts.append(f"User: {text}")
        else:
            parts.append(f"Assistant: {text}")
    # 最后加上 Assistant: 让模型继续答
    parts.append("Assistant:")
    return "\n".join(parts)

def generate_reply(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    # 输出可能包含输入，取最后生成部分进行解码更安全，但不同 tokenizer 行为不一。
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # 如果模型重复展示整个 prompt，尝试从 prompt 末尾截取生成（保守策略）
    if prompt.strip() and text.startswith(prompt.strip()):
        reply = text[len(prompt.strip()):].strip()
        if not reply:
            reply = text  # 回退到完整输出
    else:
        reply = text
    return reply

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="本地模型目录路径")
    parser.add_argument("--use_8bit", action="store_true", help="是否使用 bitsandbytes 的 8bit 模式")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_dir, args.use_8bit)
    print("Model loaded. Enter text (or 'reset' to clear, 'exit' to quit).\n")

    history = []  # history: list of (role, text) role in {"user","assistant"}
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Bye.")
                break
            if user_input.lower() == "reset":
                history = []
                print("[history cleared]")
                continue

            # append user message
            history.append(("user", user_input))
            prompt = build_prompt(history)

            # 若 prompt 太长，可能超上下文：可只保留最近几轮
            # 这里简单策略：若 token 数过多，保留最后 6 条轮次
            try:
                enc_len = len(tokenizer(prompt)["input_ids"])
                if enc_len > 3000:  # 视模型上下文限制调整
                    # 保留最后 N 轮
                    keep = 6
                    short_hist = history[-keep*2:]  # user+assistant counts
                    prompt = build_prompt(short_hist)
            except Exception:
                pass

            reply = generate_reply(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )

            # append assistant reply to history and print
            history.append(("assistant", reply))
            print("Assistant:", reply)
    except KeyboardInterrupt:
        print("\nInterrupted. Bye.")

if __name__ == "__main__":
    main()