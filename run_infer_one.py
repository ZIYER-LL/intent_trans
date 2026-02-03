import json
import re
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ====== 路径配置：按你自己的实际路径改 ======
BASE_MODEL = "/work/2024/zhulei/models/qwen3-4b"
LORA_DIR   = "/work/2024/zhulei/intent-driven/outputs/qwen3-4b-lora"

# ====== 测试输入 ======
USER_INPUT = "删除上海嘉定的数据上报/日志回传子网：至少支撑100000 台设备、上行 ≥ 1 Gbps。"

# ====== 强约束输出格式的指令（你训练集风格） ======
INSTRUCTION = """你是5G网络切片/子网编排助手。请从用户输入中抽取意图与参数，并严格按以下 JSON 结构输出，不要输出任何解释性文字、不要输出 markdown。

输出必须是一个 JSON 对象，格式如下：
{
  "intent": "CREATE_SUBNET|DELETE_SUBNET|MODIFY_SUBNET|QUERY_SUBNET",
  "confidence": 0.0,
  "parameters": {
    "location": { "city": null, "district": null },
    "service_type": "INTERNET_ACCESS|REALTIME_XR_GAMING|REALTIME_VIDEO|URLLC_CONTROL|FILE_TRANSFER|IOT_SENSOR|REALTIME_VOICE_CALL|STREAMING_LIVE|STREAMING_VIDEO|unkown",
    "sla_requirements": {
      "bandwidth_down": { "value": null, "unit": null, "operator": null },
      "bandwidth_up":   { "value": null, "unit": null, "operator": null },
      "max_latency":    { "value": null, "unit": null, "operator": null },
      "max_jitter":     { "value": null, "unit": null, "operator": null },
      "min_reliability":{ "value": null, "unit": null, "operator": null },
      "connected_devices": { "value": null, "unit": null, "operator": null }
    },
    "network_config_hints": {
      "suggested_upf_type": null,
      "qos_class": null,
      "isolation_level": null
    }
  },
  "errors": []
}

规则：
1) 如果某字段未提及，填 null（字符串也用 null）。
2) operator 只能是: ">=", "<=", "=", ">", "<" 之一。
3) unit 只能是: "Mbps" "Gbps" "ms" "count" 或 null。
4) service_type 按语义选择最匹配类别；无法判断则用 "unkown"。
5) 只输出 JSON。
"""

def extract_json_object(text: str):
    """
    从模型输出里提取最外层 JSON 对象（尽量鲁棒）。
    返回 (json_str, obj)；失败则 (None, None)
    """
    # 先找第一个 '{' 到最后一个 '}' 的大块
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, None

    candidate = text[start:end+1].strip()

    # 有时输出里会夹带多个 JSON/或多余内容，尝试用栈匹配最外层
    # 从 start 起逐字符扫描，找到第一个“完整闭合”的 JSON 对象
    brace = 0
    in_str = False
    esc = False
    first_obj_end = None

    for i, ch in enumerate(text[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                brace += 1
            elif ch == "}":
                brace -= 1
                if brace == 0:
                    first_obj_end = i
                    break

    if first_obj_end is not None:
        candidate = text[start:first_obj_end+1].strip()

    # 清理可能的 ```json ``` 包裹
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```$", "", candidate)

    try:
        obj = json.loads(candidate)
        return candidate, obj
    except Exception:
        return None, None


def main():
    print("== Loading tokenizer/model ==")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # pad token 兜底
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, LORA_DIR)
    model.eval()

    # 构造 messages（更贴近 qwen chat template）
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": USER_INPUT},
    ]

    prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tok(prompt, return_tensors="pt").to(model.device)

    print("== Generating ==")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )

    decoded = tok.decode(out[0], skip_special_tokens=True)

    print("\n===== RAW OUTPUT =====")
    print(decoded)

    js, obj = extract_json_object(decoded)
    if obj is None:
        print("\n[WARN] 未能从输出中解析出合法 JSON。")
        sys.exit(2)

    print("\n===== PARSED JSON =====")
    print(json.dumps(obj, ensure_ascii=False, indent=2))

    # 额外：简单检查字段
    required_top = ["intent", "confidence", "parameters", "errors"]
    missing = [k for k in required_top if k not in obj]
    if missing:
        print(f"\n[WARN] JSON 缺少顶层字段: {missing}")
    else:
        print("\n[OK] JSON 顶层字段齐全。")


if __name__ == "__main__":
    main()
