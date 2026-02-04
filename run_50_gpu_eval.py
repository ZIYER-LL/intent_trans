import argparse
import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_BASE_MODEL = "/work/2024/zhulei/intent-driven/qwen3-4b"
DEFAULT_LORA_DIR   = "/work/2024/zhulei/intent-driven/outputs/qwen3-4b-lora"
DEFAULT_VAL_PATH   = "/work/2024/zhulei/intent-driven/val_qwen3.jsonl"
DEFAULT_BATCH_SIZE = 4
DEFAULT_SAVE_PRED  = "/work/2024/zhulei/intent-driven/val_preds.jsonl"
DEFAULT_MAX_NEW_TOKENS = 512

INTENTS = ["QUERY_SUBNET", "CREATE_SUBNET", "MODIFY_SUBNET", "DELETE_SUBNET"]

# 你现在先看效果，不强制校验 service_type 是否在 9 类里（避免你没填全时误判）
# 真正做完整评测时，可以把你的 9 类填到这里并开启严格模式
STRICT_SERVICE_TYPE = False
SERVICE_TYPES_9 = [
    "IOT_SENSOR",
    "REALTIME_VIDEO",
    "STREAMING_VIDEO",
    "STREAMING_LIVE",
    # TODO: 其余 5 类可后续补齐
]

SLA_FIELDS = [
    "bandwidth_down",
    "bandwidth_up",
    "max_latency",
    "max_jitter",
    "min_reliability",
    "connected_devices",
]
HINT_FIELDS = ["suggested_upf_type", "qos_class", "isolation_level"]

EPS = 1e-9

SYSTEM_PROMPT = """你是网络意图转译器。请把用户请求转译为严格 JSON。
要求：
1) 只输出 JSON（不要解释、不要 markdown、不要多余字符）。
2) 顶层必须包含：intent, confidence, parameters, errors。
3) parameters 必须包含：location{city,district}, service_type, sla_requirements(包含所有字段，未提及写 null), network_config_hints。
4) operator 只能用 >= <= =，unit 使用标准单位（Mbps/Gbps/ms/%/count 等）。
"""

# ========== JSON 抽取 ==========
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_first_json_obj(s: Any) -> Optional[Dict[str, Any]]:
    if s is None:
        return None
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        return None
    m = _JSON_OBJ_RE.search(s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

# ========== 读 jsonl ==========
def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows

def extract_user_and_gold(sample: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    msgs = sample.get("messages", [])
    user_text, assistant_text = None, None
    for m in msgs:
        if m.get("role") == "user":
            user_text = m.get("content")
        elif m.get("role") == "assistant":
            assistant_text = m.get("content")

    gold_obj = extract_first_json_obj(assistant_text)
    if gold_obj is None:
        raise ValueError("gold assistant.content 不是合法 JSON，或无法抽取 JSON 对象。")
    return user_text or "", gold_obj

# ========== 规范化 ==========
OP_MAP = {"≥": ">=", "≤": "<=", ">=": ">=", "<=": "<=", "=": "="}

def norm_operator(op: Any) -> Optional[str]:
    if op is None:
        return None
    op = str(op).strip()
    return OP_MAP.get(op, op)

def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).strip())
    except Exception:
        return None

def norm_unit(u: Any) -> Optional[str]:
    if u is None:
        return None
    u = str(u).strip()
    u = u.replace("mbps", "Mbps").replace("gbps", "Gbps").replace("MS", "ms").replace("Ms", "ms")
    u = u.replace("％", "%")
    return u

def convert_bandwidth_to_mbps(value: Optional[float], unit: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    if value is None or unit is None:
        return value, unit
    unit = norm_unit(unit)
    if unit == "Mbps":
        return value, "Mbps"
    if unit == "Gbps":
        return value * 1000.0, "Mbps"
    return value, unit

def convert_time_to_ms(value: Optional[float], unit: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    if value is None or unit is None:
        return value, unit
    unit = norm_unit(unit)
    if unit == "ms":
        return value, "ms"
    if unit == "s":
        return value * 1000.0, "ms"
    return value, unit

def convert_reliability_to_percent(value: Optional[float], unit: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    if value is None or unit is None:
        return value, unit
    unit = norm_unit(unit)
    if unit == "%":
        return value, "%"
    if unit in ("ratio", "prob"):
        return value * 100.0, "%"
    return value, unit

def normalize_sla_field(field: str, triple: Dict[str, Any]) -> Dict[str, Any]:
    op = norm_operator(triple.get("operator"))
    val = to_float(triple.get("value"))
    unit = norm_unit(triple.get("unit"))

    if field in ("bandwidth_down", "bandwidth_up"):
        val, unit = convert_bandwidth_to_mbps(val, unit)
    elif field in ("max_latency", "max_jitter"):
        val, unit = convert_time_to_ms(val, unit)
    elif field == "min_reliability":
        val, unit = convert_reliability_to_percent(val, unit)
    elif field == "connected_devices":
        if unit is None and val is not None:
            unit = "count"

    return {"operator": op, "value": val, "unit": unit}

def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

# ========== 轻量 schema 校验 + canonicalize ==========
def validate_and_canonicalize(obj: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
    issues = []
    if not isinstance(obj, dict):
        return False, {}, ["pred_not_dict"]

    # 顶层必备字段
    for k in ("intent", "confidence", "parameters", "errors"):
        if k not in obj:
            issues.append(f"missing_{k}")

    intent = obj.get("intent")
    if intent not in INTENTS:
        issues.append("bad_intent")

    params = obj.get("parameters", {})
    if not isinstance(params, dict):
        issues.append("bad_parameters")
        params = {}

    loc = params.get("location", {})
    if not isinstance(loc, dict):
        issues.append("bad_location")
        loc = {}
    city = loc.get("city", None)
    district = loc.get("district", None)
    city = str(city).strip() if city is not None else None
    district = str(district).strip() if district is not None else None

    st = params.get("service_type", None)
    st = str(st).strip() if st is not None else None
    if st is None:
        issues.append("missing_service_type")
    elif STRICT_SERVICE_TYPE and (st not in SERVICE_TYPES_9):
        issues.append("bad_service_type")

    sla = params.get("sla_requirements", {})
    if not isinstance(sla, dict):
        issues.append("bad_sla_requirements")
        sla = {}

    canon_sla = {}
    for f in SLA_FIELDS:
        t = sla.get(f, {"value": None, "unit": None, "operator": None})
        if not isinstance(t, dict):
            t = {"value": None, "unit": None, "operator": None}
        canon_sla[f] = normalize_sla_field(f, t)

    hints = params.get("network_config_hints", {})
    if not isinstance(hints, dict):
        issues.append("bad_network_config_hints")
        hints = {}
    canon_hints = {}
    for k in HINT_FIELDS:
        v = hints.get(k, None)
        canon_hints[k] = str(v).strip() if v is not None else None

    errors = obj.get("errors", [])
    if errors is None:
        errors = []
    if not isinstance(errors, list):
        issues.append("bad_errors")
        errors = []

    canonical = {
        "intent": intent,
        "confidence": to_float(obj.get("confidence")),
        "parameters": {
            "location": {"city": city, "district": district},
            "service_type": st,
            "sla_requirements": canon_sla,
            "network_config_hints": canon_hints,
        },
        "errors": errors,
    }

    schema_ok = len([x for x in issues if x.startswith("missing_") or x.startswith("bad_")]) == 0 and ("bad_intent" not in issues)
    return schema_ok, canonical, issues

# ========== 指标 ==========
def macro_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0

def triple_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if (a.get("operator") or None) != (b.get("operator") or None):
        return False
    if (a.get("unit") or None) != (b.get("unit") or None):
        return False
    va, vb = a.get("value"), b.get("value")
    if va is None and vb is None:
        return True
    if va is None or vb is None:
        return False
    return abs(float(va) - float(vb)) <= EPS

def triple_to_tuple(field: str, triple: Dict[str, Any]) -> Optional[Tuple[str, str, float, str]]:
    op = triple.get("operator")
    val = triple.get("value")
    unit = triple.get("unit")
    if op is None and val is None and unit is None:
        return None
    if val is None or op is None:
        return ("INVALID", field, float("nan"), unit or "NA")
    return (field, op, float(val), unit or "NA")

def constraint_set(canon_obj: Dict[str, Any]) -> set:
    s = set()
    sla = safe_get(canon_obj, ["parameters", "sla_requirements"], {})
    for f in SLA_FIELDS:
        tup = triple_to_tuple(f, sla.get(f, {}))
        if tup is not None:
            s.add(tup)
    return s

def set_f1(gold_set: set, pred_set: set) -> Tuple[float, float, float]:
    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0
    tp = len(gold_set & pred_set)
    prec = tp / len(pred_set) if pred_set else 0.0
    rec = tp / len(gold_set) if gold_set else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def end2end_em(gold: Dict[str, Any], pred: Dict[str, Any]) -> bool:
    if gold.get("intent") != pred.get("intent"):
        return False
    if safe_get(gold, ["parameters", "service_type"]) != safe_get(pred, ["parameters", "service_type"]):
        return False
    g_loc = safe_get(gold, ["parameters", "location"], {})
    p_loc = safe_get(pred, ["parameters", "location"], {})
    if (g_loc.get("city") or None) != (p_loc.get("city") or None):
        return False
    if (g_loc.get("district") or None) != (p_loc.get("district") or None):
        return False
    g_sla = safe_get(gold, ["parameters", "sla_requirements"], {})
    p_sla = safe_get(pred, ["parameters", "sla_requirements"], {})
    for f in SLA_FIELDS:
        if not triple_equal(g_sla.get(f, {}), p_sla.get(f, {})):
            return False
    g_h = safe_get(gold, ["parameters", "network_config_hints"], {})
    p_h = safe_get(pred, ["parameters", "network_config_hints"], {})
    for k in HINT_FIELDS:
        if (g_h.get(k) or None) != (p_h.get(k) or None):
            return False
    return True

def expected_calibration_error(confs: List[Optional[float]], correct: List[int], n_bins: int = 10) -> float:
    pairs = []
    for c, y in zip(confs, correct):
        if c is None or (isinstance(c, float) and math.isnan(c)):
            continue
        if c > 1.0:
            c = c / 100.0
        c = max(0.0, min(1.0, float(c)))
        pairs.append((c, y))
    if not pairs:
        return float("nan")
    bins = [[] for _ in range(n_bins)]
    for c, y in pairs:
        idx = min(n_bins - 1, int(c * n_bins))
        bins[idx].append((c, y))
    total = sum(len(b) for b in bins)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(x for x, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        ece += (len(b) / total) * abs(acc - avg_conf)
    return ece

# ========== 推理（GPU 加速） ==========
def load_model(base_model: str, lora_dir: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True  # 提速（对大模型推理常见有效）
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = None

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()

    return tok, model

def build_prompt(tokenizer, user_text: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def batched_generate(tokenizer, model, user_texts: List[str], batch_size: int, max_new_tokens: int) -> List[str]:
    import torch

    outs: List[str] = []
    for i in range(0, len(user_texts), batch_size):
        batch = user_texts[i:i+batch_size]
        prompts = [build_prompt(tokenizer, t) for t in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

        if hasattr(model, "device"):
            enc = {k: v.to(model.device) for k, v in enc.items()}

        # 每条样本的真实 prompt 长度（避免 padding 影响切片）
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()

        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,      # 评测建议先关采样，稳定
                temperature=0.0,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for j in range(gen.size(0)):
            new_tokens = gen[j, int(prompt_lens[j]):]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            outs.append(text)

    return outs

# ========== 主流程 ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--lora_dir", default=DEFAULT_LORA_DIR)
    ap.add_argument("--val_path", default=DEFAULT_VAL_PATH)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--save_pred", default=DEFAULT_SAVE_PRED)
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--print_examples", type=int, default=3)
    args = ap.parse_args()

    rows = load_jsonl(args.val_path, limit=args.limit)
    user_texts, gold_raws = [], []
    for r in rows:
        u, g = extract_user_and_gold(r)
        user_texts.append(u)
        gold_raws.append(g)

    tok, model = load_model(args.base_model, args.lora_dir)
    pred_raw_texts = batched_generate(tok, model, user_texts, args.batch_size, args.max_new_tokens)

    # 评测 + 保存
    os.makedirs(os.path.dirname(args.save_pred) or ".", exist_ok=True)
    parse_ok = 0
    schema_ok = 0
    issues = Counter()

    y_int_true, y_int_pred = [], []
    y_srv_true, y_srv_pred = [], []

    triple_hits = Counter()
    triple_total = Counter()

    set_f1_list = []
    em_list = []
    confs = []
    em_correct = []

    with open(args.save_pred, "w", encoding="utf-8") as f:
        for idx, (text, gold_raw, pred_raw) in enumerate(zip(user_texts, gold_raws, pred_raw_texts)):
            _, gold, gold_issues = validate_and_canonicalize(gold_raw)
            for it in gold_issues:
                issues[f"gold_{it}"] += 1

            pred_obj = extract_first_json_obj(pred_raw)
            if pred_obj is None:
                issues["parse_fail"] += 1
                y_int_true.append(gold.get("intent"))
                y_int_pred.append("UNPARSEABLE")
                y_srv_true.append(safe_get(gold, ["parameters", "service_type"]))
                y_srv_pred.append("UNPARSEABLE")
                em_list.append(False)
                confs.append(None)
                em_correct.append(0)

                f.write(json.dumps({
                    "id": idx,
                    "text": text,
                    "gold": gold,
                    "pred_raw": pred_raw,
                    "pred_json": None,
                    "pred_schema_ok": False,
                    "pred_issues": ["parse_fail"],
                    "em": False,
                }, ensure_ascii=False) + "\n")
                continue

            parse_ok += 1
            pred_schema_ok, pred, pred_issues = validate_and_canonicalize(pred_obj)
            if pred_schema_ok:
                schema_ok += 1
            for it in pred_issues:
                issues[it] += 1

            # intent/service
            y_int_true.append(gold.get("intent"))
            y_int_pred.append(pred.get("intent") if pred.get("intent") in INTENTS else "OTHER")

            g_st = safe_get(gold, ["parameters", "service_type"])
            p_st = safe_get(pred, ["parameters", "service_type"])
            y_srv_true.append(g_st)
            y_srv_pred.append(p_st if p_st is not None else "OTHER")

            # Triple-Exact：只在 gold 非空字段上算
            g_sla = safe_get(gold, ["parameters", "sla_requirements"], {})
            p_sla = safe_get(pred, ["parameters", "sla_requirements"], {})
            for fld in SLA_FIELDS:
                if triple_to_tuple(fld, g_sla.get(fld, {})) is None:
                    continue
                triple_total[fld] += 1
                if triple_equal(g_sla.get(fld, {}), p_sla.get(fld, {})):
                    triple_hits[fld] += 1

            # 约束集合 F1
            set_f1_list.append(set_f1(constraint_set(gold), constraint_set(pred))[2])

            # EM
            em = end2end_em(gold, pred)
            em_list.append(em)

            confs.append(pred.get("confidence"))
            em_correct.append(1 if em else 0)

            f.write(json.dumps({
                "id": idx,
                "text": text,
                "gold": gold,
                "pred_raw": pred_raw,
                "pred_json": pred,
                "pred_schema_ok": pred_schema_ok,
                "pred_issues": pred_issues,
                "em": em,
            }, ensure_ascii=False) + "\n")

    total = len(em_list)
    parse_rate = parse_ok / total if total else 0.0
    schema_rate = schema_ok / total if total else 0.0
    intent_labels = INTENTS + ["OTHER", "UNPARSEABLE"]
    intent_mf1 = macro_f1(y_int_true, y_int_pred, intent_labels)

    # 预览阶段：service labels 用 gold 里出现过的即可（避免你 9 类没填全导致宏平均不稳）
    uniq_gold_srv = sorted(set([x for x in y_srv_true if x is not None]))
    srv_labels = uniq_gold_srv + ["OTHER", "UNPARSEABLE"]
    srv_mf1 = macro_f1(y_srv_true, y_srv_pred, srv_labels) if uniq_gold_srv else float("nan")

    total_triples = sum(triple_total.values())
    hit_triples = sum(triple_hits.values())
    triple_exact_overall = hit_triples / total_triples if total_triples else 0.0

    em_rate = sum(1 for x in em_list if x) / total if total else 0.0
    avg_set_f1 = sum(set_f1_list) / len(set_f1_list) if set_f1_list else 0.0
    ece = expected_calibration_error(confs, em_correct, n_bins=10)

    print("========== QUICK RUN (first {} samples) ==========".format(args.limit))
    print(f"Saved predictions to: {args.save_pred}")
    print(f"JSON Parse Rate:   {parse_rate:.4f}")
    print(f"Schema Valid Rate: {schema_rate:.4f}")
    print(f"Intent Macro-F1:   {intent_mf1:.4f}")
    print(f"Service Macro-F1:  {srv_mf1:.4f}  (labels from gold subset)")
    print(f"Triple-Exact@GoldNonNull (overall): {triple_exact_overall:.4f}")
    print(f"Constraint Set F1 (avg):           {avg_set_f1:.4f}")
    print(f"End-to-End EM:     {em_rate:.4f}")
    print(f"ECE (EM as correct): {ece:.4f}")

    print("\n--- Top issues (debug) ---")
    for k, v in issues.most_common(12):
        print(f"{k:24s}: {v}")

    # 打印几个样例方便你肉眼看输出是否“纯 JSON”
    n_show = min(args.print_examples, args.limit)
    print("\n--- {} examples ---".format(n_show))
    # 从保存文件里读回头几条（避免内存里状态不一致）
    shown = 0
    with open(args.save_pred, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            print("\n[{}] {}".format(row["id"], row["text"]))
            print("pred_raw:", row["pred_raw"][:400].replace("\n", "\\n"))
            print("em:", row["em"], "schema_ok:", row["pred_schema_ok"])
            shown += 1
            if shown >= n_show:
                break

if __name__ == "__main__":
    main()
