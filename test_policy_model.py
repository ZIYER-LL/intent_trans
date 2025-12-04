#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Policy模型测试脚本
计算 policy 参数的误差指标：MAE, MSE, RMSE, 相对误差等
"""

import os
import json
import re
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =====================
# 配置参数
# =====================
BASE_MODEL_DIR = "/work/2024/zhulei/intent-driven/qwen3-4b"  # 基础模型路径
LORA_MODEL_DIR = "/work/2024/zhulei/intent-driven/outputs/qwen3-4b-lora-policy"  # LoRA模型路径
TEST_DATA_PATH = "/work/2024/zhulei/intent-driven/test_policy.json"  # 测试数据路径
GPU_ID = 7  # 使用的GPU ID

# 推理参数
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1  # 降低温度以获得更确定性的输出
TOP_P = 0.9
DO_SAMPLE = True

# Policy参数名称
POLICY_PARAMS = [
    "latency_ms",
    "jitter_ms", 
    "packet_loss_rate",
    "bandwidth_kbps",
    "reliability",
    "priority"
]

# =====================
# 3GPP标准规则书 - 各服务类型的Policy参数有效范围
# =====================

# 基于 3GPP TS 26.114、TS 22.261、实时视频典型服务要求
REALTIME_VIDEO_RULEBOOK = {
    "latency_ms": [40, 120],
    "jitter_ms": [1, 20],
    "packet_loss_rate": [0.001, 0.02],
    "bandwidth_kbps": [1500, 6000],
    "reliability": [0.999, 0.9999],
    "priority": 2
}

# 基于3GPP TS 26.114（VoIP/IMS）、TS 22.105、TS 22.261汇总
REALTIME_VOICE_CALL_RULEBOOK = {
    "latency_ms": [50, 150],
    "jitter_ms": [1, 30],
    "packet_loss_rate": [0.001, 0.03],
    "bandwidth_kbps": [20, 100],
    "reliability": [0.98, 0.999],
    "priority": 2
}

# 基于3GPP TS 22.261（XR Requirements）、3GPP TS 26.118（XR streaming）ITU-T G.1035（Cloud Gaming QoE）
REALTIME_XR_GAMING_RULEBOOK = {
    "latency_ms": [5, 25],
    "jitter_ms": [1, 10],
    "packet_loss_rate": [0.0005, 0.01],
    "bandwidth_kbps": [50000, 150000],  # 50Mbps~150Mbps
    "reliability": [0.999, 0.9999],
    "priority": 2
}

# 基于3GPP TS 26.234（Progressive/Adaptive Streaming）、DASH（ISO/IEC 23009-1）视频流标准
STREAMING_VIDEO_RULEBOOK = {
    "latency_ms": [100, 300],
    "jitter_ms": [5, 50],
    "packet_loss_rate": [0.001, 0.05],
    "bandwidth_kbps": [3000, 12000],   # 3Mbps~12Mbps
    "reliability": [0.99, 0.999],
    "priority": 4
}

# 基于3GPP TS 26.235（Adaptive Streaming）、TS 22.261（Media services requirements）
STREAMING_LIVE_RULEBOOK = {
    "latency_ms": [80, 200],
    "jitter_ms": [5, 20],
    "packet_loss_rate": [0.001, 0.03],
    "bandwidth_kbps": [8000, 20000],  # 8Mbps~20Mbps
    "reliability": [0.99, 0.999],
    "priority": 3
}

# 基于3GPP TS 22.261（Data-centric services）
FILE_TRANSFER_RULEBOOK = {
    "latency_ms": [80, 300],
    "jitter_ms": [10, 100],
    "packet_loss_rate": [0.001, 0.05],
    "bandwidth_kbps": [5000, 20000],
    "reliability": [0.999, 0.99999],
    "priority": 5
}

# 基于3GPP TS 22.104（工业 IoT）、TS 22.261（5G/6G 服务需求）、TS 23.501（QoS Framework）
IOT_SENSOR_RULEBOOK = {
    "latency_ms": [10, 80],
    "jitter_ms": [1, 20],
    "packet_loss_rate": [0.0001, 0.01],
    "bandwidth_kbps": [50, 500],
    "reliability": [0.999, 0.99999],
    "priority": 3
}

# 基于3GPP TS 22.261（eMBB service requirements）、3GPP TS 23.501（QoS Framework）
INTERNET_ACCESS_RULEBOOK = {
    "latency_ms": [50, 300],
    "jitter_ms": [10, 80],
    "packet_loss_rate": [0.001, 0.05],
    "bandwidth_kbps": [10000, 50000],
    "reliability": [0.99, 0.999],
    "priority": 4
}

# 基于3GPP TS 22.104（Mission Critical services）、3GPP TS 22.261（Service requirements for 5G/6G）、3GPP TS 23.501（QoS Framework）
URLLC_CONTROL_RULEBOOK = {
    "latency_ms": [1, 10],
    "jitter_ms": [0.1, 2],
    "packet_loss_rate": [0.00001, 0.001],
    "bandwidth_kbps": [100, 1000],
    "reliability": [0.99999, 0.999999],
    "priority": 1
}

# 服务类型到规则书的映射
SERVICE_RULEBOOK_MAP = {
    "realtime_video": REALTIME_VIDEO_RULEBOOK,
    "realtime_voice_call": REALTIME_VOICE_CALL_RULEBOOK,
    "realtime_xr_gaming": REALTIME_XR_GAMING_RULEBOOK,
    "streaming_video": STREAMING_VIDEO_RULEBOOK,
    "streaming_live": STREAMING_LIVE_RULEBOOK,
    "file_transfer": FILE_TRANSFER_RULEBOOK,
    "iot_sensor": IOT_SENSOR_RULEBOOK,
    "internet_access": INTERNET_ACCESS_RULEBOOK,
    "urllc_control": URLLC_CONTROL_RULEBOOK,
}

# =====================
# 工具函数
# =====================

def load_model_with_lora(base_model_dir, lora_model_dir, gpu_id=0):
    """加载基础模型和LoRA权重"""
    print(f"正在加载基础模型: {base_model_dir}")
    
    if torch.cuda.is_available():
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        dtype = torch.float16
        device_map = {"": device}
    else:
        device = "cpu"
        dtype = torch.float32
        device_map = "cpu"
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_dir,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载基础模型
    print("加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    
    # 加载LoRA权重
    if os.path.exists(lora_model_dir):
        print(f"加载LoRA权重: {lora_model_dir}")
        model = PeftModel.from_pretrained(base_model, lora_model_dir)
        print("✅ LoRA模型加载完成")
    else:
        print(f"⚠️  LoRA路径不存在，使用基础模型: {lora_model_dir}")
        model = base_model
    
    model.eval()  # 设置为评估模式
    return tokenizer, model

def create_prompt(user_intent):
    """根据用户意图构造prompt（使用messages格式）"""
    messages = [
        {
            "role": "user",
            "content": user_intent
        }
    ]
    return messages

def parse_policy_output(output_text):
    """
    从模型输出中解析policy JSON
    支持多种输出格式：
    1. 纯JSON格式: {"latency_ms": 70, "jitter_ms": 10, ...}
    2. 带说明的JSON: 根据用户需求，policy为: {"latency_ms": 70, ...}
    3. 代码块格式: ```json\n{...}\n```
    """
    policy = None
    
    # 方法1: 尝试提取完整的JSON对象
    # 匹配 { ... "latency_ms": xxx ... } 格式
    json_pattern = r'\{[^{}]*"latency_ms"[^{}]*\}'
    
    # 尝试匹配更复杂的嵌套JSON（可能包含多行）
    json_patterns = [
        r'\{[^{}]*(?:"latency_ms"|"jitter_ms"|"packet_loss_rate"|"bandwidth_kbps"|"reliability"|"priority")[^{}]*\}',
        r'\{.*?"latency_ms".*?\}',
    ]
    
    for pattern in json_patterns:
        matches = re.finditer(pattern, output_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                candidate = json.loads(match.group(0))
                # 验证是否包含policy参数
                if any(key in candidate for key in POLICY_PARAMS):
                    policy = candidate
                    break
            except json.JSONDecodeError:
                continue
        if policy:
            break
    
    # 方法2: 尝试提取代码块中的JSON
    if policy is None:
        code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.finditer(code_block_pattern, output_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                candidate = json.loads(match.group(1))
                if any(key in candidate for key in POLICY_PARAMS):
                    policy = candidate
                    break
            except json.JSONDecodeError:
                continue
    
    # 方法3: 尝试直接解析整个输出
    if policy is None:
        try:
            candidate = json.loads(output_text.strip())
            if isinstance(candidate, dict) and any(key in candidate for key in POLICY_PARAMS):
                policy = candidate
        except json.JSONDecodeError:
            pass
    
    # 方法4: 尝试提取所有可能的JSON对象
    if policy is None:
        # 查找所有可能的JSON对象
        json_objects = re.findall(r'\{[^{}]*(?:"latency_ms"|"jitter_ms"|"packet_loss_rate"|"bandwidth_kbps"|"reliability"|"priority")[^{}]*\}', 
                                  output_text, re.DOTALL | re.IGNORECASE)
        for obj_str in json_objects:
            try:
                candidate = json.loads(obj_str)
                if isinstance(candidate, dict) and any(key in candidate for key in POLICY_PARAMS):
                    policy = candidate
                    break
            except json.JSONDecodeError:
                continue
    
    return policy

def check_param_in_range(param_name, param_value, rulebook):
    """
    检查参数值是否在规则书定义的有效范围内
    
    Args:
        param_name: 参数名称
        param_value: 参数值
        rulebook: 规则书字典
    
    Returns:
        bool: True表示在范围内，False表示不在范围内
    """
    if param_name not in rulebook:
        return None  # 规则书中没有定义该参数
    
    rule = rulebook[param_name]
    
    # 如果规则是单个值（如priority），则必须完全匹配
    if not isinstance(rule, list):
        return param_value == rule
    
    # 如果规则是范围 [min, max]
    if isinstance(rule, list) and len(rule) == 2:
        min_val, max_val = rule[0], rule[1]
        return min_val <= param_value <= max_val
    
    return None

def check_policy_compliance(policy, service_type):
    """
    检查policy是否符合对应服务类型的规则书要求
    
    Args:
        policy: policy字典
        service_type: 服务类型
    
    Returns:
        dict: 包含每个参数的合规性检查结果
    """
    compliance = {}
    
    # 获取对应的规则书
    rulebook = SERVICE_RULEBOOK_MAP.get(service_type)
    if rulebook is None:
        # 如果没有找到对应的规则书，返回None表示无法检查
        return None
    
    # 检查每个参数
    for param in POLICY_PARAMS:
        if param in policy:
            param_value = float(policy[param])
            is_compliant = check_param_in_range(param, param_value, rulebook)
            compliance[param] = {
                'value': param_value,
                'in_range': is_compliant,
                'rule': rulebook.get(param)
            }
        else:
            compliance[param] = {
                'value': None,
                'in_range': False,  # 缺少参数视为不合规
                'rule': rulebook.get(param)
            }
    
    # 计算整体合规率
    total_params = len([p for p in POLICY_PARAMS if p in policy])
    compliant_params = sum(1 for p in POLICY_PARAMS if p in policy and compliance.get(p, {}).get('in_range') == True)
    compliance['overall_compliance_rate'] = compliant_params / total_params if total_params > 0 else 0
    compliance['all_compliant'] = all(compliance.get(p, {}).get('in_range') == True for p in POLICY_PARAMS if p in policy)
    
    return compliance

def infer(model, tokenizer, user_intent):
    """对输入进行推理"""
    # 构造messages
    messages = create_prompt(user_intent)
    
    # 使用tokenizer的apply_chat_template格式化
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 推理
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出（只取新生成的部分）
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return output_text

def load_test_data(test_data_path):
    """加载测试数据"""
    print(f"正在加载测试数据: {test_data_path}")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"测试集大小: {len(data)} 条")
    return data

def extract_policy_from_data(item):
    """从数据项中提取用户意图、服务类型和期望的policy"""
    service_type = None
    policy = {}
    
    if "instruction" in item:
        # 格式: {instruction: {user_intent: ..., service_type: ..., intent_type: ...}}
        user_intent = item["instruction"].get("user_intent", "")
        service_type = item["instruction"].get("service_type", None)
        intent_type = item["instruction"].get("intent_type", None)
        
        # 如果有output字段，提取期望的policy（用于有标准答案的情况）
        if "output" in item:
            policy = item["output"].get("policy", {})
    elif "messages" in item:
        # 格式: {messages: [{role: "user", content: ...}, {role: "assistant", content: ...}]}
        user_intent = ""
        for msg in item["messages"]:
            if msg["role"] == "user":
                user_intent = msg["content"]
            elif msg["role"] == "assistant":
                try:
                    content = msg["content"]
                    # 尝试解析JSON
                    if isinstance(content, str):
                        policy = json.loads(content)
                    elif isinstance(content, dict):
                        policy = content
                except (json.JSONDecodeError, TypeError):
                    pass
    else:
        # 其他可能的格式
        user_intent = item.get("user_intent", item.get("input", ""))
        service_type = item.get("service_type", None)
        policy = item.get("policy", item.get("output", {}))
        # 如果output是字符串，尝试解析
        if isinstance(policy, str):
            try:
                policy = json.loads(policy)
            except json.JSONDecodeError:
                policy = {}
    
    return user_intent, service_type, policy

def calculate_metrics(results):
    """计算policy参数的误差指标和合规性指标"""
    total = len(results)
    
    # 统计解析成功的数量
    parse_success = sum(1 for r in results if r['policy_pred'] is not None)
    parse_rate = parse_success / total if total > 0 else 0
    
    # 初始化每个参数的误差列表（仅当有真实值时计算）
    param_errors = {param: {'mae': [], 'mse': [], 'relative_error': []} for param in POLICY_PARAMS}
    
    # 初始化合规性统计
    compliance_stats = {
        'total_with_service_type': 0,
        'total_compliant_policies': 0,
        'param_compliance': {param: {'compliant': 0, 'total': 0} for param in POLICY_PARAMS},
        'service_type_stats': {},
        'intent_type_stats': {}
    }
    
    # 计算每个样本的误差和合规性
    for result in results:
        if result['policy_pred'] is None:
            continue
        
        policy_true = result.get('policy_true', {})
        policy_pred = result['policy_pred']
        service_type = result.get('service_type')
        intent_type = result.get('intent_type')
        
        # 检查预测policy的合规性
        if service_type:
            compliance_stats['total_with_service_type'] += 1
            pred_compliance = check_policy_compliance(policy_pred, service_type)
            
            if pred_compliance:
                # 统计整体合规性
                if pred_compliance.get('all_compliant', False):
                    compliance_stats['total_compliant_policies'] += 1
                
                # 统计每个参数的合规性
                for param in POLICY_PARAMS:
                    if param in policy_pred:
                        compliance_stats['param_compliance'][param]['total'] += 1
                        if pred_compliance.get(param, {}).get('in_range') == True:
                            compliance_stats['param_compliance'][param]['compliant'] += 1
                
                # 按服务类型统计
                if service_type not in compliance_stats['service_type_stats']:
                    compliance_stats['service_type_stats'][service_type] = {
                        'total': 0,
                        'compliant': 0
                    }
                compliance_stats['service_type_stats'][service_type]['total'] += 1
                if pred_compliance.get('all_compliant', False):
                    compliance_stats['service_type_stats'][service_type]['compliant'] += 1
                
                # 按意图类型统计
                if intent_type:
                    if intent_type not in compliance_stats['intent_type_stats']:
                        compliance_stats['intent_type_stats'][intent_type] = {
                            'total': 0,
                            'compliant': 0
                        }
                    compliance_stats['intent_type_stats'][intent_type]['total'] += 1
                    if pred_compliance.get('all_compliant', False):
                        compliance_stats['intent_type_stats'][intent_type]['compliant'] += 1
        
        # 计算误差（仅当有真实值时）
        if policy_true and len(policy_true) > 0:
            for param in POLICY_PARAMS:
                if param in policy_true and param in policy_pred:
                    true_val = float(policy_true[param])
                    pred_val = float(policy_pred[param])
                    
                    # 计算绝对误差
                    abs_error = abs(pred_val - true_val)
                    param_errors[param]['mae'].append(abs_error)
                    param_errors[param]['mse'].append(abs_error ** 2)
                    
                    # 计算相对误差（避免除零）
                    if true_val != 0:
                        rel_error = abs_error / abs(true_val)
                        param_errors[param]['relative_error'].append(rel_error)
    
    # 计算每个参数的平均指标
    metrics = {
        'parse_rate': parse_rate,
        'parse_success': parse_success,
        'total': total,
        'param_metrics': {}
    }
    
    for param in POLICY_PARAMS:
        if len(param_errors[param]['mae']) > 0:
            metrics['param_metrics'][param] = {
                'mae': np.mean(param_errors[param]['mae']),
                'mse': np.mean(param_errors[param]['mse']),
                'rmse': np.sqrt(np.mean(param_errors[param]['mse'])),
                'mean_relative_error': np.mean(param_errors[param]['relative_error']) if len(param_errors[param]['relative_error']) > 0 else None,
                'valid_samples': len(param_errors[param]['mae'])
            }
        else:
            metrics['param_metrics'][param] = {
                'mae': None,
                'mse': None,
                'rmse': None,
                'mean_relative_error': None,
                'valid_samples': 0
            }
    
    # 计算整体policy的相似度（使用余弦相似度或欧氏距离，仅当有真实值时）
    policy_similarities = []
    has_ground_truth = any(r.get('policy_true') and len(r.get('policy_true', {})) > 0 for r in results)
    
    if has_ground_truth:
        for result in results:
            if result['policy_pred'] is None:
                continue
            
            policy_true = result.get('policy_true', {})
            policy_pred = result['policy_pred']
            
            if not policy_true or len(policy_true) == 0:
                continue
            
            # 提取所有参数的向量
            true_vec = []
            pred_vec = []
            for param in POLICY_PARAMS:
                if param in policy_true and param in policy_pred:
                    true_vec.append(float(policy_true[param]))
                    pred_vec.append(float(policy_pred[param]))
            
            if len(true_vec) > 0:
                # 计算余弦相似度
                true_vec = np.array(true_vec)
                pred_vec = np.array(pred_vec)
                
                # 归一化
                true_norm = np.linalg.norm(true_vec)
                pred_norm = np.linalg.norm(pred_vec)
                
                if true_norm > 0 and pred_norm > 0:
                    cosine_sim = np.dot(true_vec, pred_vec) / (true_norm * pred_norm)
                    policy_similarities.append(cosine_sim)
                
                # 计算归一化欧氏距离
                # 将距离转换为相似度 (0-1之间，1表示完全相同)
                euclidean_dist = np.linalg.norm(true_vec - pred_vec)
                # 使用最大可能距离进行归一化（这里使用经验值）
                max_dist = np.linalg.norm(true_vec) + np.linalg.norm(pred_vec)
                if max_dist > 0:
                    normalized_sim = 1 - min(euclidean_dist / max_dist, 1.0)
                    # 也可以直接使用欧氏距离
                    result['euclidean_distance'] = euclidean_dist
                    result['cosine_similarity'] = cosine_sim if true_norm > 0 and pred_norm > 0 else None
        
        if len(policy_similarities) > 0:
            metrics['mean_cosine_similarity'] = np.mean(policy_similarities)
        else:
            metrics['mean_cosine_similarity'] = None
    else:
        metrics['mean_cosine_similarity'] = None
    
    # 计算合规性指标
    if compliance_stats['total_with_service_type'] > 0:
        metrics['overall_compliance_rate'] = compliance_stats['total_compliant_policies'] / compliance_stats['total_with_service_type']
        metrics['param_compliance_rates'] = {}
        for param in POLICY_PARAMS:
            total = compliance_stats['param_compliance'][param]['total']
            compliant = compliance_stats['param_compliance'][param]['compliant']
            if total > 0:
                metrics['param_compliance_rates'][param] = compliant / total
            else:
                metrics['param_compliance_rates'][param] = None
        
        # 按服务类型的合规率
        metrics['service_type_compliance'] = {}
        for service_type, stats in compliance_stats['service_type_stats'].items():
            if stats['total'] > 0:
                metrics['service_type_compliance'][service_type] = stats['compliant'] / stats['total']
        
        # 按意图类型的合规率
        metrics['intent_type_compliance'] = {}
        for intent_type, stats in compliance_stats['intent_type_stats'].items():
            if stats['total'] > 0:
                metrics['intent_type_compliance'][intent_type] = stats['compliant'] / stats['total']
    else:
        metrics['overall_compliance_rate'] = None
        metrics['param_compliance_rates'] = {}
        metrics['service_type_compliance'] = {}
        metrics['intent_type_compliance'] = {}
    
    metrics['compliance_stats'] = compliance_stats
    metrics['has_ground_truth'] = has_ground_truth
    
    return metrics

def main():
    print("=" * 60)
    print("🚀 Policy模型测试开始")
    print("=" * 60)
    
    # 检查路径
    if not os.path.exists(BASE_MODEL_DIR):
        raise FileNotFoundError(f"基础模型路径不存在: {BASE_MODEL_DIR}")
    
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"测试数据路径不存在: {TEST_DATA_PATH}")
    
    # 加载模型
    tokenizer, model = load_model_with_lora(BASE_MODEL_DIR, LORA_MODEL_DIR, GPU_ID)
    print(f"模型设备: {next(model.parameters()).device}\n")
    
    # 加载测试数据
    test_data = load_test_data(TEST_DATA_PATH)
    
    # 进行测试
    print("开始测试...")
    print("-" * 60)
    
    results = []
    for idx, item in enumerate(test_data, 1):
        # 提取用户意图、服务类型和期望的policy
        user_intent, service_type, policy_true = extract_policy_from_data(item)
        
        # 提取意图类型
        intent_type = None
        if "instruction" in item:
            intent_type = item["instruction"].get("intent_type", None)
        
        if not user_intent:
            print(f"⚠️  第 {idx} 条测试数据缺少用户意图，已跳过")
            continue
        
        # 推理
        try:
            output_text = infer(model, tokenizer, user_intent)
            policy_pred = parse_policy_output(output_text)
        except Exception as e:
            print(f"⚠️  第 {idx} 条测试出错: {e}")
            policy_pred = None
            output_text = ""
        
        # 检查预测policy的合规性
        pred_compliance = None
        if policy_pred is not None and service_type:
            pred_compliance = check_policy_compliance(policy_pred, service_type)
        
        # 记录结果
        result = {
            'idx': idx,
            'user_intent': user_intent,
            'service_type': service_type,
            'intent_type': intent_type,
            'policy_true': policy_true,
            'policy_pred': policy_pred,
            'output': output_text,
            'parse_success': policy_pred is not None,
            'pred_compliance': pred_compliance
        }
        results.append(result)
        
        # 显示进度
        if idx % 10 == 0:
            print(f"已测试 {idx}/{len(test_data)} 条...")
    
    print("-" * 60)
    print("测试完成！\n")
    
    # 计算指标
    metrics = calculate_metrics(results)
    
    # 显示结果
    print("=" * 60)
    print("📊 测试结果")
    print("=" * 60)
    print(f"总测试样本数: {metrics['total']}")
    print(f"解析成功率: {metrics['parse_rate']:.4f} ({metrics['parse_success']}/{metrics['total']})")
    
    if metrics['mean_cosine_similarity'] is not None:
        print(f"平均余弦相似度: {metrics['mean_cosine_similarity']:.4f}")
    
    # 显示合规性指标
    if metrics.get('overall_compliance_rate') is not None:
        print(f"\n📋 3GPP标准合规性指标:")
        print("-" * 60)
        print(f"整体合规率: {metrics['overall_compliance_rate']:.4f} ({metrics['compliance_stats']['total_compliant_policies']}/{metrics['compliance_stats']['total_with_service_type']})")
        
        print(f"\n各参数合规率:")
        for param in POLICY_PARAMS:
            compliance_rate = metrics['param_compliance_rates'].get(param)
            if compliance_rate is not None:
                stats = metrics['compliance_stats']['param_compliance'][param]
                print(f"  {param}: {compliance_rate:.4f} ({stats['compliant']}/{stats['total']})")
        
        if metrics.get('service_type_compliance'):
            print(f"\n各服务类型合规率:")
            for service_type, rate in sorted(metrics['service_type_compliance'].items()):
                stats = metrics['compliance_stats']['service_type_stats'][service_type]
                print(f"  {service_type}: {rate:.4f} ({stats['compliant']}/{stats['total']})")
        
        if metrics.get('intent_type_compliance'):
            print(f"\n各意图类型合规率:")
            for intent_type, rate in sorted(metrics['intent_type_compliance'].items()):
                stats = metrics['compliance_stats']['intent_type_stats'][intent_type]
                print(f"  {intent_type}: {rate:.4f} ({stats['compliant']}/{stats['total']})")
    
    # 仅当有真实值时才显示误差指标
    if metrics.get('has_ground_truth'):
        print("\n各Policy参数误差指标:")
        print("-" * 60)
        for param in POLICY_PARAMS:
            param_metric = metrics['param_metrics'][param]
            if param_metric['valid_samples'] > 0:
                print(f"\n{param}:")
                print(f"  有效样本数: {param_metric['valid_samples']}")
                print(f"  MAE (平均绝对误差): {param_metric['mae']:.6f}")
                print(f"  MSE (均方误差): {param_metric['mse']:.6f}")
                print(f"  RMSE (均方根误差): {param_metric['rmse']:.6f}")
                if param_metric['mean_relative_error'] is not None:
                    print(f"  平均相对误差: {param_metric['mean_relative_error']:.4%}")
                # 显示合规率（如果有）
                if metrics.get('param_compliance_rates', {}).get(param) is not None:
                    print(f"  3GPP合规率: {metrics['param_compliance_rates'][param]:.4f}")
            else:
                print(f"\n{param}: 无有效样本")
    else:
        print("\n⚠️  注意: 测试数据中没有标准答案，无法计算误差指标")
    
    print("=" * 60)
    
    # 显示错误样本
    print("\n❌ 解析失败样本分析:")
    print("-" * 60)
    
    parse_failures = [r for r in results if not r['parse_success']]
    print(f"\n解析失败数量: {len(parse_failures)}")
    for r in parse_failures[:5]:  # 只显示前5个
        print(f"\n  样本 {r['idx']}:")
        print(f"  用户意图: {r['user_intent'][:80]}...")
        print(f"  模型输出: {r['output'][:200]}...")
    
    # 显示预测误差较大的样本（仅当有真实值时）
    if metrics.get('has_ground_truth'):
        print("\n⚠️  预测误差较大样本 (前5个):")
        print("-" * 60)
        
        # 计算每个样本的总误差
        for result in results:
            if result['policy_pred'] is None:
                result['total_error'] = float('inf')
                result['euclidean_distance'] = float('inf')
                continue
            
            total_error = 0
            policy_true = result.get('policy_true', {})
            policy_pred = result['policy_pred']
            
            if not policy_true or len(policy_true) == 0:
                result['total_error'] = None
                result['euclidean_distance'] = None
                continue
            
            # 计算所有参数的误差
            true_vec = []
            pred_vec = []
            for param in POLICY_PARAMS:
                if param in policy_true and param in policy_pred:
                    true_val = float(policy_true[param])
                    pred_val = float(policy_pred[param])
                    true_vec.append(true_val)
                    pred_vec.append(pred_val)
                    total_error += abs(pred_val - true_val) ** 2
            
            result['total_error'] = np.sqrt(total_error) if total_error > 0 else 0
            
            # 计算欧氏距离
            if len(true_vec) > 0:
                result['euclidean_distance'] = np.linalg.norm(np.array(true_vec) - np.array(pred_vec))
            else:
                result['euclidean_distance'] = float('inf')
        
        # 按误差排序
        error_samples = sorted([r for r in results if r.get('policy_pred') is not None and r.get('total_error') is not None], 
                              key=lambda x: x.get('total_error', float('inf')), reverse=True)
        
        for r in error_samples[:5]:
            print(f"\n  样本 {r['idx']} (总误差: {r.get('total_error', 0):.4f}):")
            print(f"  服务类型: {r.get('service_type', '未知')}")
            print(f"  用户意图: {r['user_intent'][:80]}...")
            if r.get('policy_true'):
                print(f"  真实Policy: {json.dumps(r['policy_true'], ensure_ascii=False, indent=2)}")
            print(f"  预测Policy: {json.dumps(r['policy_pred'], ensure_ascii=False, indent=2)}")
            if r.get('pred_compliance'):
                all_compliant = r['pred_compliance'].get('all_compliant', False)
                print(f"  合规性: {'✅ 完全合规' if all_compliant else '❌ 部分不合规'}")
                print(f"  合规率: {r['pred_compliance'].get('overall_compliance_rate', 0):.2%}")
    
    # 显示不合规样本
    print("\n❌ 不合规样本分析 (前5个):")
    print("-" * 60)
    non_compliant_samples = [r for r in results if r.get('pred_compliance') and not r['pred_compliance'].get('all_compliant', True)]
    for r in non_compliant_samples[:5]:
        print(f"\n  样本 {r['idx']}:")
        print(f"  服务类型: {r.get('service_type', '未知')}")
        print(f"  用户意图: {r['user_intent'][:80]}...")
        print(f"  预测Policy: {json.dumps(r['policy_pred'], ensure_ascii=False, indent=2)}")
        print(f"  不合规参数:")
        for param in POLICY_PARAMS:
            if param in r.get('pred_compliance', {}):
                param_info = r['pred_compliance'][param]
                if param_info.get('in_range') == False:
                    rule = param_info.get('rule')
                    value = param_info.get('value')
                    if isinstance(rule, list) and len(rule) == 2:
                        print(f"    {param}: {value} (范围: [{rule[0]}, {rule[1]}])")
                    else:
                        print(f"    {param}: {value} (要求: {rule})")
    
    # 保存详细结果
    output_file = "test_policy_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'results': results
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✅ 详细结果已保存到: {output_file}")

if __name__ == "__main__":
    main()

