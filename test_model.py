#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型测试脚本
计算 intent_type accuracy, service_type accuracy, joint accuracy
"""

import os
import json
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =====================
# 配置参数
# =====================
BASE_MODEL_DIR = "/work/2024/zhulei/intent-driven/qwen3-4b"  # 基础模型路径
LORA_MODEL_DIR = "/work/2024/zhulei/intent-driven/outputs/qwen3-4b-lora-intent"  # LoRA模型路径
TEST_DATA_PATH = "/work/2024/zhulei/intent-driven/test_intent.json"  # 测试数据路径（相对于脚本运行目录）
GPU_ID = 2  # 使用的GPU ID

# 推理参数
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.1  # 降低温度以获得更确定性的输出
TOP_P = 0.9
DO_SAMPLE = True

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

def create_prompt(input_text):
    """根据输入文本构造prompt（使用messages格式）"""
    messages = [
        {
            "role": "user",
            "content": input_text
        }
    ]
    return messages

def parse_model_output(output_text, input_text):
    """
    从模型输出中解析intent_type和service_type
    支持多种输出格式：
    1. JSON格式: {"intent_type": "...", "service_type": "..."}
    2. 自然语言格式: intent_type: xxx, service_type: xxx
    3. 其他格式
    """
    intent_type = None
    service_type = None
    
    # 已知的所有可能值
    known_intents = ["slice_create", "route_preference", "slice_qos_modify", "access_control"]
    known_services = [
        "realtime_video", "realtime_voice_call", "realtime_xr_gaming",
        "streaming_live", "streaming_video", "iot_sensor", 
        "urllc_control", "internet_access"
    ]
    
    # 方法1: 尝试提取完整的JSON格式
    # 匹配 { ... "intent_type": "xxx" ... "service_type": "xxx" ... }
    json_patterns = [
        r'\{[^{}]*"intent_type"\s*:\s*"([^"]+)"[^{}]*"service_type"\s*:\s*"([^"]+)"[^{}]*\}',
        r'\{[^{}]*"service_type"\s*:\s*"([^"]+)"[^{}]*"intent_type"\s*:\s*"([^"]+)"[^{}]*\}',
    ]
    
    for pattern in json_patterns:
        json_match = re.search(pattern, output_text, re.IGNORECASE | re.DOTALL)
        if json_match:
            if "intent_type" in pattern:
                intent_type = json_match.group(1).strip()
                service_type = json_match.group(2).strip()
            else:
                service_type = json_match.group(1).strip()
                intent_type = json_match.group(2).strip()
            break
    
    # 方法2: 尝试提取单独的JSON字段
    if intent_type is None:
        intent_json_pattern = r'"intent_type"\s*:\s*"([^"]+)"'
        intent_match = re.search(intent_json_pattern, output_text, re.IGNORECASE)
        if intent_match:
            intent_type = intent_match.group(1).strip()
    
    if service_type is None:
        service_json_pattern = r'"service_type"\s*:\s*"([^"]+)"'
        service_match = re.search(service_json_pattern, output_text, re.IGNORECASE)
        if service_match:
            service_type = service_match.group(1).strip()
    
    # 方法3: 尝试提取键值对格式 (intent_type: xxx 或 intent_type=xxx)
    if intent_type is None:
        intent_patterns = [
            r'intent_type["\s:：=]+\s*([a-z_]+)',
            r'intent["\s:：=]+\s*([a-z_]+)',
        ]
        for pattern in intent_patterns:
            intent_match = re.search(pattern, output_text, re.IGNORECASE)
            if intent_match:
                candidate = intent_match.group(1).strip()
                if candidate in known_intents:
                    intent_type = candidate
                    break
    
    if service_type is None:
        service_patterns = [
            r'service_type["\s:：=]+\s*([a-z_]+)',
            r'service["\s:：=]+\s*([a-z_]+)',
        ]
        for pattern in service_patterns:
            service_match = re.search(pattern, output_text, re.IGNORECASE)
            if service_match:
                candidate = service_match.group(1).strip()
                if candidate in known_services:
                    service_type = candidate
                    break
    
    # 方法4: 在整个输出中搜索已知的值（作为最后手段）
    if intent_type is None:
        for intent in known_intents:
            # 使用单词边界确保完整匹配
            pattern = r'\b' + re.escape(intent) + r'\b'
            if re.search(pattern, output_text, re.IGNORECASE):
                intent_type = intent
                break
    
    if service_type is None:
        for service in known_services:
            # 使用单词边界确保完整匹配
            pattern = r'\b' + re.escape(service) + r'\b'
            if re.search(pattern, output_text, re.IGNORECASE):
                service_type = service
                break
    
    return intent_type, service_type

def infer(model, tokenizer, input_text):
    """对输入进行推理"""
    # 构造messages
    messages = create_prompt(input_text)
    
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

def calculate_metrics(results):
    """计算准确率指标"""
    total = len(results)
    intent_correct = 0
    service_correct = 0
    joint_correct = 0
    
    for result in results:
        if result['intent_pred'] == result['intent_true']:
            intent_correct += 1
        if result['service_pred'] == result['service_true']:
            service_correct += 1
        if (result['intent_pred'] == result['intent_true'] and 
            result['service_pred'] == result['service_true']):
            joint_correct += 1
    
    intent_acc = intent_correct / total if total > 0 else 0
    service_acc = service_correct / total if total > 0 else 0
    joint_acc = joint_correct / total if total > 0 else 0
    
    return {
        'intent_accuracy': intent_acc,
        'service_accuracy': service_acc,
        'joint_accuracy': joint_acc,
        'intent_correct': intent_correct,
        'service_correct': service_correct,
        'joint_correct': joint_correct,
        'total': total
    }

def main():
    print("=" * 60)
    print("🚀 模型测试开始")
    print("=" * 60)
    
    # 检查路径
    if not os.path.exists(BASE_MODEL_DIR):
        raise FileNotFoundError(f"基础模型路径不存在: {BASE_MODEL_DIR}")
    
    # 尝试多个可能的测试数据路径
    test_paths = [
        TEST_DATA_PATH,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), TEST_DATA_PATH),
        os.path.join(os.getcwd(), TEST_DATA_PATH),
    ]
    test_data_path = None
    for path in test_paths:
        if os.path.exists(path):
            test_data_path = path
            break
    
    if test_data_path is None:
        raise FileNotFoundError(f"测试数据路径不存在，尝试过的路径: {test_paths}")
    
    TEST_DATA_PATH = test_data_path
    
    # 加载模型
    tokenizer, model = load_model_with_lora(BASE_MODEL_DIR, LORA_MODEL_DIR, GPU_ID)
    print(f"模型设备: {next(model.parameters()).device}\n")
    
    # 加载测试数据
    test_data = load_test_data(test_data_path)
    
    # 进行测试
    print("开始测试...")
    print("-" * 60)
    
    results = []
    for idx, item in enumerate(test_data, 1):
        input_text = item['input']
        intent_true = item['intent_type']
        service_true = item['service_type']
        
        # 推理
        try:
            output_text = infer(model, tokenizer, input_text)
            intent_pred, service_pred = parse_model_output(output_text, input_text)
        except Exception as e:
            print(f"⚠️  第 {idx} 条测试出错: {e}")
            intent_pred = None
            service_pred = None
            output_text = ""
        
        # 记录结果
        result = {
            'idx': idx,
            'input': input_text,
            'intent_true': intent_true,
            'intent_pred': intent_pred,
            'service_true': service_true,
            'service_pred': service_pred,
            'output': output_text,
            'intent_correct': intent_pred == intent_true,
            'service_correct': service_pred == service_true,
            'joint_correct': (intent_pred == intent_true and service_pred == service_true)
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
    print(f"\nIntent Type Accuracy: {metrics['intent_accuracy']:.4f} ({metrics['intent_correct']}/{metrics['total']})")
    print(f"Service Type Accuracy: {metrics['service_accuracy']:.4f} ({metrics['service_correct']}/{metrics['total']})")
    print(f"Joint Accuracy: {metrics['joint_accuracy']:.4f} ({metrics['joint_correct']}/{metrics['total']})")
    print("=" * 60)
    
    # 显示错误样本
    print("\n❌ 错误样本分析:")
    print("-" * 60)
    
    intent_errors = [r for r in results if not r['intent_correct']]
    service_errors = [r for r in results if not r['service_correct']]
    joint_errors = [r for r in results if not r['joint_correct']]
    
    print(f"\nIntent错误 ({len(intent_errors)} 个):")
    for r in intent_errors[:5]:  # 只显示前5个
        print(f"  输入: {r['input'][:50]}...")
        print(f"  真实: {r['intent_true']}, 预测: {r['intent_pred']}")
        print(f"  输出: {r['output'][:100]}...")
        print()
    
    print(f"\nService错误 ({len(service_errors)} 个):")
    for r in service_errors[:5]:  # 只显示前5个
        print(f"  输入: {r['input'][:50]}...")
        print(f"  真实: {r['service_true']}, 预测: {r['service_pred']}")
        print(f"  输出: {r['output'][:100]}...")
        print()
    
    # 保存详细结果
    output_file = "test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'results': results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 详细结果已保存到: {output_file}")

if __name__ == "__main__":
    main()


