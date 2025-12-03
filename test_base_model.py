#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础模型测试脚本
计算 intent_type accuracy, service_type accuracy, joint accuracy
不加载LoRA权重，仅测试基础模型性能
"""

import os
import json
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================
# 配置参数
# =====================
BASE_MODEL_DIR = "/work/2024/zhulei/intent-driven/qwen3-4b"  # 基础模型路径
TEST_DATA_PATH = "/work/2024/zhulei/intent-driven/test_intent.json"  # 测试数据路径
GPU_ID = 2  # 使用的GPU ID

# 推理参数
MAX_NEW_TOKENS = 100  # 减少token数量，只需要JSON输出
DO_SAMPLE = False  # 使用贪婪解码，更确定

# =====================
# 工具函数
# =====================

def load_base_model(base_model_dir, gpu_id=0):
    """加载基础模型（不加载LoRA权重）"""
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
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    
    model.eval()  # 设置为评估模式
    print("✅ 基础模型加载完成")
    return tokenizer, model

def create_prompt(input_text):
    """根据输入文本构造prompt"""
    # 使用详细的文本prompt，明确列出所有可选值，要求只输出JSON
    prompt = f"""任务：从用户输入中识别intent_type和service_type。

可选值：
intent_type: slice_create, slice_qos_modify, route_preference, access_control
service_type: realtime_video, realtime_voice_call, realtime_xr_gaming, streaming_video, streaming_live, file_transfer, iot_sensor, internet_access, urllc_control

用户输入：{input_text}

只输出JSON，不要任何其他文字："""
    return prompt

def parse_model_output(output_text, input_text):
    """
    从模型输出中解析intent_type和service_type
    采用多级fallback策略：先尝试直接解析JSON，再提取JSON对象，最后使用正则提取字段
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
    
    # 方法1: 尝试直接解析JSON
    try:
        parsed = json.loads(output_text)
        intent_type = parsed.get("intent_type")
        service_type = parsed.get("service_type")
        if intent_type and service_type:
            return intent_type, service_type
    except:
        pass
    
    # 方法2: 尝试提取第一个 { ... } 之间的内容
    try:
        start = output_text.find("{")
        end = output_text.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = output_text[start:end]
            parsed = json.loads(json_str)
            intent_type = parsed.get("intent_type")
            service_type = parsed.get("service_type")
            if intent_type and service_type:
                return intent_type, service_type
    except:
        pass
    
    # 方法3: 使用正则提取JSON字段
    intent_json_pattern = r'"intent_type"\s*:\s*"([^"]+)"'
    intent_match = re.search(intent_json_pattern, output_text, re.IGNORECASE)
    if intent_match:
        intent_type = intent_match.group(1).strip()
    
    service_json_pattern = r'"service_type"\s*:\s*"([^"]+)"'
    service_match = re.search(service_json_pattern, output_text, re.IGNORECASE)
    if service_match:
        service_type = service_match.group(1).strip()
    
    # 方法4: 验证提取的值是否在已知列表中
    if intent_type and intent_type not in known_intents:
        intent_type = None
    if service_type and service_type not in known_services:
        service_type = None
    
    return intent_type, service_type

def infer(model, tokenizer, input_text):
    """对输入进行推理"""
    # 构造文本prompt（不使用messages格式）
    prompt_text = create_prompt(input_text)
    
    # Tokenize
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    # 推理 - 使用贪婪解码（do_sample=False）以获得更确定性的输出
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,  # 减少token数量，只需要JSON输出
            do_sample=False,  # 使用贪婪解码，更确定
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码完整输出（包含输入和生成的部分）
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取新生成的部分（去掉输入prompt）
    # 简单方法：找到prompt的结尾，取后面的部分
    if prompt_text in full_output:
        output_text = full_output.split(prompt_text, 1)[1]
    else:
        # 如果找不到prompt，取新生成的tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 简化截断：提取第一个 { 到最后一个 } 之间的内容
    start = output_text.find("{")
    end = output_text.rfind("}")
    if start >= 0 and end > start:
        output_text = output_text[start:end+1]
    else:
        # 如果没有找到JSON，截断到第一个换行或合理长度
        for stop_char in ["\n\n", "\n", "。", "，"]:
            if stop_char in output_text:
                output_text = output_text.split(stop_char)[0]
                break
        if len(output_text) > 200:
            output_text = output_text[:200]
    
    return output_text.strip()

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
    print("🚀 基础模型测试开始")
    print("=" * 60)
    
    # 检查路径
    if not os.path.exists(BASE_MODEL_DIR):
        raise FileNotFoundError(f"基础模型路径不存在: {BASE_MODEL_DIR}")
    
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"测试数据路径不存在: {TEST_DATA_PATH}")
    
    # 加载模型
    tokenizer, model = load_base_model(BASE_MODEL_DIR, GPU_ID)
    print(f"模型设备: {next(model.parameters()).device}\n")
    
    # 加载测试数据
    test_data = load_test_data(TEST_DATA_PATH)
    
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
    print("📊 测试结果（基础模型）")
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
    output_file = "test_base_model_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model_type': 'base_model',
            'base_model_dir': BASE_MODEL_DIR,
            'metrics': metrics,
            'results': results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 详细结果已保存到: {output_file}")

if __name__ == "__main__":
    main()










