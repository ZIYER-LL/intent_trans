#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-4B LoRA微调脚本
支持多GPU训练、梯度检查点、混合精度等优化
"""

import os
import json
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset

# =====================
# 配置参数
# =====================
MODEL_DIR = "/work/2024/zhulei/models/qwen3-4b"  # 模型路径
TRAIN_DATA_PATH = "/work/2024/zhulei/intent-driven/train_qwen.json"  # 训练数据路径
VAL_DATA_PATH = "/work/2024/zhulei/intent-driven/val_qwen.json"  # 验证数据路径（可选）
OUTPUT_DIR = "./outputs/qwen3-4b-lora"  # 输出目录
LORA_R = 8  # LoRA rank
LORA_ALPHA = 32  # LoRA alpha
LORA_DROPOUT = 0.1  # LoRA dropout
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # 目标模块

# 训练参数
BATCH_SIZE = 4  # 批次大小（根据显存调整）
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积步数
LEARNING_RATE = 2e-4  # 学习率
NUM_EPOCHS = 3  # 训练轮数
MAX_LENGTH = 1024  # 最大序列长度
SAVE_STEPS = 500  # 每多少步保存一次
EVAL_STEPS = 500  # 每多少步评估一次
LOGGING_STEPS = 50  # 每多少步记录一次日志
WARMUP_STEPS = 100  # 预热步数
FP16 = True  # 是否使用混合精度训练
GRADIENT_CHECKPOINTING = True  # 是否使用梯度检查点（节省显存）
DDP_FIND_UNUSED_PARAMETERS = False  # DDP模式下的参数

# 其他参数
SEED = 42  # 随机种子
RESUME_FROM_CHECKPOINT = None  # 从检查点恢复训练（如："./outputs/qwen3-4b-lora/checkpoint-1000"）

# =====================
# 工具函数
# =====================

def preprocess_function(examples, tokenizer, max_length=1024):
    """数据预处理函数 - 正确处理Qwen对话格式"""
    # 使用apply_chat_template格式化对话
    texts = []
    assistant_contents = []
    
    for messages in examples["messages"]:
        # 格式化完整对话（包括user和assistant）
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
        
        # 提取assistant内容
        assistant_content = None
        for msg in messages:
            if msg["role"] == "assistant":
                assistant_content = msg["content"]
                break
        assistant_contents.append(assistant_content or "")
    
    # Tokenize完整对话
    model_inputs = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None
    )
    
    # 创建labels，只对assistant部分计算loss
    labels = []
    for i, (text, assistant_content) in enumerate(zip(texts, assistant_contents)):
        if not assistant_content:
            # 如果没有assistant内容，所有位置都不计算loss
            labels.append([-100] * len(model_inputs["input_ids"][i]))
            continue
        
        # 找到assistant内容在文本中的位置
        # 在Qwen的chat template中，assistant内容通常在特定标记之后
        # 简化方法：找到assistant内容在完整文本中的起始位置
        assistant_start = text.find(assistant_content)
        
        if assistant_start == -1:
            # 如果找不到，尝试tokenize assistant内容来匹配
            assistant_tokens = tokenizer(assistant_content, add_special_tokens=False)["input_ids"]
            full_tokens = model_inputs["input_ids"][i]
            
            # 在完整序列中查找assistant tokens
            label_ids = [-100] * len(full_tokens)
            for j in range(len(full_tokens) - len(assistant_tokens) + 1):
                if full_tokens[j:j+len(assistant_tokens)] == assistant_tokens:
                    # 找到匹配位置，设置labels
                    for k in range(len(assistant_tokens)):
                        label_ids[j + k] = full_tokens[j + k]
                    break
            labels.append(label_ids)
        else:
            # 基于字符位置计算token位置（近似方法）
            # Tokenize到assistant开始位置之前的部分
            prefix_text = text[:assistant_start]
            prefix_tokens = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
            
            label_ids = [-100] * len(model_inputs["input_ids"][i])
            # 从prefix长度之后开始计算loss
            start_idx = len(prefix_tokens)
            end_idx = min(start_idx + len(tokenizer(assistant_content, add_special_tokens=False)["input_ids"]), 
                         len(model_inputs["input_ids"][i]))
            
            if start_idx < len(label_ids):
                for j in range(start_idx, end_idx):
                    label_ids[j] = model_inputs["input_ids"][i][j]
            
            labels.append(label_ids)
    
    model_inputs["labels"] = labels
    
    return model_inputs

def load_dataset(data_path, tokenizer, max_length=1024):
    """加载数据集"""
    print(f"正在加载数据集: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"数据集大小: {len(data)} 条")
    
    # 转换为Dataset格式
    dataset_dict = {"messages": [item["messages"] for item in data]}
    dataset = Dataset.from_dict(dataset_dict)
    
    # 预处理
    dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return dataset

# =====================
# 主函数
# =====================

def main():
    # 设置随机种子
    set_seed(SEED)
    
    # 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载tokenizer
    print(f"正在加载tokenizer: {MODEL_DIR}")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型
    print(f"正在加载模型: {MODEL_DIR}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16 if FP16 else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 启用梯度检查点
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        print("已启用梯度检查点")
    
    # 准备模型用于k-bit训练（如果需要）
    # model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    print("配置LoRA参数...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 加载训练数据
    train_dataset = load_dataset(TRAIN_DATA_PATH, tokenizer, MAX_LENGTH)
    
    # 加载验证数据（如果存在）
    eval_dataset = None
    if VAL_DATA_PATH and os.path.exists(VAL_DATA_PATH):
        eval_dataset = load_dataset(VAL_DATA_PATH, tokenizer, MAX_LENGTH)
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=FP16,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS if eval_dataset else None,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,  # 只保留最近3个检查点
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
        warmup_steps=WARMUP_STEPS,
        report_to="tensorboard" if os.path.exists("tensorboard") else None,
        ddp_find_unused_parameters=DDP_FIND_UNUSED_PARAMETERS,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        save_safetensors=True,
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 从检查点恢复（如果指定）
    if RESUME_FROM_CHECKPOINT:
        print(f"从检查点恢复训练: {RESUME_FROM_CHECKPOINT}")
        trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    else:
        # 开始训练
        print("开始训练...")
        trainer.train()
    
    # 保存最终模型
    print("保存最终模型...")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    
    # 保存训练配置
    config_to_save = {
        "model_dir": MODEL_DIR,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES,
        "max_length": MAX_LENGTH,
        "training_args": training_args.to_dict(),
        "train_time": datetime.now().isoformat()
    }
    
    with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, ensure_ascii=False, indent=2)
    
    print(f"\n训练完成！模型保存在: {output_dir}")
    print(f"LoRA权重保存在: {output_dir / 'adapter_model.bin'}")
    print("\n使用方法:")
    print(f"  from peft import PeftModel")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{MODEL_DIR}')")
    print(f"  model = PeftModel.from_pretrained(model, '{output_dir}')")

if __name__ == "__main__":
    main()

