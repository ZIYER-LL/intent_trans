import argparse
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# tqdm：没有也能跑（自动降级）
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# =========================
# ✅ 默认配置：直接 python eval_parse_rate.py 就跑这些
# =========================
DEFAULT_BASE_MODEL = "/work/2024/zhulei/models/qwen3-4b"
DEFAULT_LORA_DIR   = "/work/2024/zhulei/intent-driven/outputs/qwen3-4b-lora"
DEFAULT_VAL_PATH   = "/work/2024/zhulei/intent-driven/val_qwen3.jsonl"
DEFAULT_BATCH_SIZE = 4
DEFAULT_SAVE_PRED  = "/work/2024/zhulei/intent-driven/val_preds.jsonl"
DEFAULT_MAX_NEW_TOKENS = 512

# ====== 你的任务枚举（按你数据定义） ======
VALID_INTENTS = {"CREATE_SUBNET", "DELETE_SUBNET", "MODIFY_SUBNET", "QUERY_SUBNET"}
VALID_SERVICE_TYPES = {
    "INTERNET_ACCESS",
    "REALTIME_XR_GAMING",
    "REALTIME_VIDEO",
    "URLLC_CONTROL",
    "FILE_TRANSFER",
    "IOT_SENSOR",
    "REALTIME_VOICE_CALL",
    "STREAMING_LIVE",
    "STREAMING_VIDEO",
    "unkown",  # 注意你数据里是 unkown（拼写）
}
VALID_OPERATORS = {">=", "<=", "=", ">", "<"}
VALID_UNITS = {"Mbps", "Gbps", "ms", "count", None}

# ====== 强约束指令（system prompt） ======
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

# -------------------------
# I/O & parsing helpers
# -------------------------
def load_samples(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        return samples
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        raise ValueError("JSON 文件需为 list 格式（或改用 jsonl）。")

def extract_user_text(sample: Dict[str, Any]) -> str:
    """
    支持两类：
      1) {"messages":[{"role":"user","content":"..."}, ...]}
      2) {"user":"..."} 或 {"input":"..."}
    """
    if "messages" in sample and isinstance(sample["messages"], list):
        for msg in reversed(sample["messages"]):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", "")).strip()
        if sample["messages"]:
            return str(sample["messages"][0].get("content", "")).strip()
        return ""
    for k in ("user", "input", "query", "text"):
        if k in sample:
            return str(sample[k]).strip()
    return ""

def build_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def try_parse_strict(text: str) -> Tuple[bool, Optional[Any]]:
    s = text.strip()
    try:
        obj = json.loads(s)
        return True, obj
    except Exception:
        return False, None

def extract_json_object_lenient(text: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return False, None, None

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

    if first_obj_end is None:
        return False, None, None

    candidate = text[start:first_obj_end + 1].strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```$", "", candidate)

    try:
        obj = json.loads(candidate)
        return True, obj, candidate
    except Exception:
        return False, None, candidate

def validate_obj(obj: Any) -> Dict[str, bool]:
    ok = {
        "is_dict": isinstance(obj, dict),
        "has_top_fields": False,
        "intent_valid": False,
        "service_type_valid": False,
        "operators_valid": True,
        "units_valid": True,
    }
    if not isinstance(obj, dict):
        return ok

    required_top = ["intent", "confidence", "parameters", "errors"]
    ok["has_top_fields"] = all(k in obj for k in required_top)

    intent = obj.get("intent", None)
    ok["intent_valid"] = (intent in VALID_INTENTS)

    params = obj.get("parameters", {})
    service_type = params.get("service_type", None) if isinstance(params, dict) else None
    ok["service_type_valid"] = (service_type in VALID_SERVICE_TYPES)

    def walk(x: Any):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "operator" and v is not None:
                    ok["operators_valid"] = ok["operators_valid"] and (v in VALID_OPERATORS)
                if k == "unit":
                    ok["units_valid"] = ok["units_valid"] and (v in VALID_UNITS)
                walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    walk(obj)
    return ok

# -------------------------
# generation (GPU-optimized)
# -------------------------
@torch.inference_mode()
def generate_batch(model, tokenizer, prompts: List[str], max_new_tokens: int) -> Tuple[List[str], int, int]:
    """
    return: decoded_texts, input_tokens, generated_tokens
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    input_tokens = int(inputs["input_ids"].numel())

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    # out.shape: [bs, in_len + gen_len]
    generated_tokens = int(out.numel() - inputs["input_ids"].numel())
    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    return texts, input_tokens, generated_tokens

def gpu_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)

def main():
    ap = argparse.ArgumentParser(description="Fast GPU eval: strict/lenient JSON parse rate + contract checks.")
    ap.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    ap.add_argument("--lora_dir", type=str, default=DEFAULT_LORA_DIR)
    ap.add_argument("--val_path", type=str, default=DEFAULT_VAL_PATH)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max_samples", type=int, default=0, help="0=全量；否则取前N条")
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--save_pred", type=str, default=DEFAULT_SAVE_PRED)
    ap.add_argument("--gpu", type=int, default=-1, help="-1=auto；否则指定 cuda:{id}")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--compile", action="store_true", help="torch.compile(model) 可能更快（也可能更慢，看环境）")
    ap.add_argument("--log_every", type=int, default=20, help="每N个batch打印一次速度/显存日志")
    args = ap.parse_args()

    # ===== data =====
    samples = load_samples(args.val_path)
    if args.max_samples and args.max_samples > 0:
        samples = samples[:args.max_samples]
    total = len(samples)
    print(f"[INFO] Loaded {total} samples from: {args.val_path}")

    # ===== device =====
    if torch.cuda.is_available():
        if args.gpu >= 0:
            torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        props = torch.cuda.get_device_properties(device)
        print(f"[INFO] Using GPU: {device} | {props.name} | VRAM={props.total_memory/1024**3:.1f}GB")
    else:
        device = torch.device("cpu")
        print("[WARN] CUDA not available, running on CPU (will be slow).")

    # ===== tokenizer =====
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"  # ✅ decoder-only + batch generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ===== dtype =====
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # ===== model =====
    print("[INFO] Loading base model ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    print(f"[INFO] Base model loaded in {time.time()-t0:.1f}s")

    print("[INFO] Loading LoRA adapter ...")
    t1 = time.time()
    model = PeftModel.from_pretrained(model, args.lora_dir)
    model.eval()
    print(f"[INFO] LoRA loaded in {time.time()-t1:.1f}s")

    # 可选：开启一些推理加速
    try:
        model.config.use_cache = True
    except Exception:
        pass

    # 可选：torch.compile（不一定每个环境都快）
    if args.compile:
        if device.type == "cuda":
            print("[INFO] torch.compile enabled (may take time on first run)...")
            model = torch.compile(model)

    # ===== metrics =====
    strict_ok = 0
    lenient_ok = 0
    has_top_fields_ok = 0
    intent_valid_ok = 0
    service_type_valid_ok = 0
    operators_valid_ok = 0
    units_valid_ok = 0
    failures = []

    preds_f = open(args.save_pred, "w", encoding="utf-8") if args.save_pred else None

    # ===== loop =====
    bs = max(1, args.batch_size)
    n_batches = (total + bs - 1) // bs
    iter_range = range(n_batches)
    if tqdm is not None:
        iter_range = tqdm(iter_range, total=n_batches, desc="eval", dynamic_ncols=True)

    total_in_tok = 0
    total_gen_tok = 0
    total_time = 0.0

    for b in iter_range:
        s = b * bs
        e = min((b + 1) * bs, total)
        batch = samples[s:e]

        prompts = []
        meta = []
        for i, sample in enumerate(batch, start=s):
            user_text = extract_user_text(sample)
            prompts.append(build_prompt(tokenizer, user_text))
            meta.append({"id": i, "user_text": user_text})

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t_batch0 = time.time()
        outs, in_tok, gen_tok = generate_batch(model, tokenizer, prompts, args.max_new_tokens)
        dt = time.time() - t_batch0

        total_in_tok += in_tok
        total_gen_tok += gen_tok
        total_time += dt

        # 统计
        for out_text, m in zip(outs, meta):
            ok_s, _ = try_parse_strict(out_text)
            if ok_s:
                strict_ok += 1

            ok_l, obj_l, extracted = extract_json_object_lenient(out_text)
            if ok_l:
                lenient_ok += 1
                checks = validate_obj(obj_l)
                has_top_fields_ok += int(checks["has_top_fields"])
                intent_valid_ok += int(checks["intent_valid"])
                service_type_valid_ok += int(checks["service_type_valid"])
                operators_valid_ok += int(checks["operators_valid"])
                units_valid_ok += int(checks["units_valid"])
            else:
                if len(failures) < 5:
                    failures.append({
                        "id": m["id"],
                        "user_text": m["user_text"],
                        "raw_output": out_text[:1200],
                        "extracted_candidate": extracted,
                    })

            if preds_f:
                rec = {
                    "id": m["id"],
                    "user_text": m["user_text"],
                    "raw_output": out_text,
                    "strict_parse_ok": ok_s,
                    "lenient_parse_ok": ok_l,
                    "parsed_json": obj_l if ok_l else None,
                }
                preds_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # 日志：速度/显存
        if (b + 1) % max(1, args.log_every) == 0 or (b + 1) == n_batches:
            tps_in = in_tok / max(dt, 1e-8)
            tps_gen = gen_tok / max(dt, 1e-8)
            if device.type == "cuda":
                mem = gpu_mem_gb()
                print(f"[LOG] batch {b+1}/{n_batches} | bs={e-s} | time={dt:.2f}s"
                      f" | in_tok/s={tps_in:.0f} | gen_tok/s={tps_gen:.0f} | peak_mem={mem:.2f}GB")
            else:
                print(f"[LOG] batch {b+1}/{n_batches} | bs={e-s} | time={dt:.2f}s"
                      f" | in_tok/s={tps_in:.0f} | gen_tok/s={tps_gen:.0f}")

    if preds_f:
        preds_f.close()

    def pct(x: int, denom: int) -> float:
        return 100.0 * x / max(denom, 1)

    print("\n========== SPEED ==========")
    print(f"Total time: {total_time:.1f}s | avg {(total_time/max(n_batches,1)):.2f}s/batch")
    print(f"Total input tokens: {total_in_tok} | Total generated tokens: {total_gen_tok}")
    if total_time > 0:
        print(f"Overall in_tok/s={total_in_tok/total_time:.0f} | gen_tok/s={total_gen_tok/total_time:.0f}")

    print("\n========== PARSE METRICS ==========")
    print(f"Total samples: {total}")
    print(f"Strict parse ok : {strict_ok}/{total} = {pct(strict_ok, total):.2f}%   (整段必须是 JSON)")
    print(f"Lenient parse ok: {lenient_ok}/{total} = {pct(lenient_ok, total):.2f}%  (允许抽取最外层 JSON)")

    print("\n========== CONTRACT METRICS (on lenient-parsed) ==========")
    if lenient_ok > 0:
        print(f"Top fields complete : {has_top_fields_ok}/{lenient_ok} = {pct(has_top_fields_ok, lenient_ok):.2f}%")
        print(f"Intent enum valid   : {intent_valid_ok}/{lenient_ok} = {pct(intent_valid_ok, lenient_ok):.2f}%")
        print(f"ServiceType valid   : {service_type_valid_ok}/{lenient_ok} = {pct(service_type_valid_ok, lenient_ok):.2f}%")
        print(f"Operators valid     : {operators_valid_ok}/{lenient_ok} = {pct(operators_valid_ok, lenient_ok):.2f}%")
        print(f"Units valid         : {units_valid_ok}/{lenient_ok} = {pct(units_valid_ok, lenient_ok):.2f}%")
    else:
        print("No lenient-parsed samples, skip contract metrics.")

    if failures:
        print("\n========== EXAMPLES (first 5 lenient-parse failures) ==========")
        for f in failures:
            print(f"\n--- id={f['id']} ---")
            print("USER:", f["user_text"])
            print("RAW :", f["raw_output"][:800])
            if f["extracted_candidate"]:
                print("CAND:", f["extracted_candidate"][:800])

    if args.save_pred:
        print(f"\n[INFO] Predictions saved to: {args.save_pred}")

    print("\n[INFO] Done.")

if __name__ == "__main__":
    main()
