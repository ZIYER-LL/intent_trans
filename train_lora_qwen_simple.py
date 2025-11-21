#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-4B LoRA微调脚本（简化版 - 使用SFTTrainer）
推荐使用此版本，更简单可靠
"""

import os
import json
import torch
from datetime import datetime
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer
from datasets import Dataset

# =====================
# 配置参数
# =====================
MODEL_DIR = "/work/2024/zhulei/models/qwen3-4b"  # 模型路径
TRAIN_DATA_PATH = "/work/2024/zhulei/intent-driven/train_qwen.json"  # 训练数据路径
OUTPUT_DIR = "/work/2024/zhulei/intent-driven/outputs/qwen3-4b-lora"  # 输出目录

# LoRA参数
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
LOGGING_STEPS = 50  # 每多少步记录一次日志
WARMUP_STEPS = 100  # 预热步数
FP16 = True  # 是否使用混合精度训练
GRADIENT_CHECKPOINTING = True  # 是否使用梯度检查点（节省显存）

# 其他参数
SEED = 42  # 随机种子
RESUME_FROM_CHECKPOINT = None  # 从检查点恢复训练

# =====================
# 工具函数
# =====================

def load_dataset(data_path, tokenizer):
    """加载数据集并进行tokenization"""
    print(f"正在加载数据集: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"数据集大小: {len(data)} 条")
    
    # 使用tokenizer的apply_chat_template格式化对话
    texts = []
    for item in data:
        text = tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    
    # 进行tokenization
    def tokenize_function(texts):
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            return_tensors=None
        )
        return tokenized
    
    # Tokenize所有文本
    tokenized_data = tokenize_function(texts)
    
    # 转换为Dataset格式
    dataset = Dataset.from_dict(tokenized_data)
    
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
    train_dataset = load_dataset(TRAIN_DATA_PATH, tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=FP16,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        evaluation_strategy="no",  # 不进行评估
        save_total_limit=3,  # 只保留最近3个检查点
        warmup_steps=WARMUP_STEPS,
        report_to="tensorboard" if os.path.exists("tensorboard") else None,
        dataloader_pin_memory=True,
        save_safetensors=True,
    )
    
    # 创建SFTTrainer（使用最简参数）
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
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






