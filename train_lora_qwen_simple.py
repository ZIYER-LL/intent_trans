#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
from datetime import datetime
from pathlib import Path
import time
import math
import re
from collections import defaultdict

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
MODEL_DIR = "/work/2024/zhulei/intent-driven/qwen3-4b"  # 模型路径
TRAIN_DATA_PATH = "/work/2024/zhulei/intent-driven/train_qwen3.jsonl"  # 训练数据路径
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
GPU_ID = 2  # 指定使用的GPU ID（根据nvidia-smi选择空闲的GPU，GPU 2/4/5/6/7都可用）

# =====================
# 工具函数
# =====================

def load_dataset(data_path, tokenizer):
    print(f"正在加载数据集: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"训练数据文件不存在: {data_path}")

    data_path_lower = data_path.lower()
    data = []

    # ✅ 兼容 jsonl：一行一个 JSON
    if data_path_lower.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"第 {line_no} 行不是合法 JSON：{e}") from e
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    print(f"数据集大小: {len(data)} 条")

    if not isinstance(data, list):
        raise ValueError("数据格式错误：应该是列表格式（或 jsonl 每行一个对象）")

    if len(data) > 0 and "messages" not in data[0]:
        raise ValueError("数据格式错误：每个条目应包含 'messages' 字段")

    formatted_data = []
    for idx, item in enumerate(data):
        try:
            text = tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
            formatted_data.append({"text": text})
        except Exception as e:
            print(f"警告：处理第 {idx+1} 条数据时出错: {e}，已跳过")
            continue

    print(f"成功处理 {len(formatted_data)} 条数据")
    return Dataset.from_list(formatted_data)

def _grad_norm(model, only_lora: bool = False) -> float:
    total_sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if only_lora and ("lora" not in name.lower()):
            continue
        g = p.grad.detach()
        if g.is_sparse:
            g = g.coalesce().values()
        gn = g.float().norm(2).item()
        total_sq += gn * gn
    return math.sqrt(total_sq)


def _lora_param_norms_by_group(model, top_k: int = 12):
    """
    统计 LoRA 参数范数，按 layer_id + module_type 分组取 TopK，
    避免 TensorBoard 曲线太多。
    """
    groups_sq = defaultdict(float)

    # 兼容常见结构：model.layers.N 或 transformer.h.N
    layer_pat = re.compile(r"(?:layers|h)\.(\d+)")
    module_keys = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj",
                   "w1", "w2", "w3", "fc1", "fc2"]

    for name, p in model.named_parameters():
        lname = name.lower()
        if "lora" not in lname:
            continue

        m = layer_pat.search(name)
        layer_id = m.group(1) if m else "misc"

        mod = "misc"
        for k in module_keys:
            if k in lname:
                mod = k
                break

        gname = f"layer{layer_id}/{mod}"
        pn = p.detach().float().norm(2).item()
        groups_sq[gname] += pn * pn

    groups = {k: math.sqrt(v) for k, v in groups_sq.items()}
    top = dict(sorted(groups.items(), key=lambda x: x[1], reverse=True)[:top_k])
    return top


class LoRAMonitorSFTTrainer(SFTTrainer):
    """
    在训练过程中写 TensorBoard 标量：
    - train loss + loss_ema
    - lr
    - grad_norm_all / grad_norm_lora
    - lora_param_norm (TopK grouped)
    - tokens/sec
    - step_time + gpu_mem
    """
    def __init__(self, *args, ema_beta: float = 0.98, lora_top_k: int = 12, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_beta = ema_beta
        self.loss_ema = None
        self.lora_top_k = lora_top_k

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
    
        step_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
    
        if self.args.n_gpu > 1:
            loss = loss.mean()
    
        self.accelerator.backward(loss)
    
        if self.is_world_process_zero() and (self.state.global_step % self.args.logging_steps == 0):
            loss_val = float(loss.detach().float().item())
            if self.loss_ema is None:
                self.loss_ema = loss_val
            else:
                b = self.ema_beta
                self.loss_ema = b * self.loss_ema + (1 - b) * loss_val
    
            lr = 0.0
            if self.optimizer is not None and len(self.optimizer.param_groups) > 0:
                lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
    
            gn_all = _grad_norm(model, only_lora=False)
            gn_lora = _grad_norm(model, only_lora=True)
    
            if "attention_mask" in inputs:
                tokens = int(inputs["attention_mask"].detach().sum().item())
            else:
                tokens = int(inputs["input_ids"].detach().numel())
    
            step_time = time.perf_counter() - step_start
            tps = tokens / max(step_time, 1e-8)
    
            mem_gb = 0.0
            if torch.cuda.is_available():
                mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
            logs = {
                "train/loss": loss_val,
                "train/loss_ema": float(self.loss_ema),
                "train/lr": lr,
                "train/grad_norm_all": float(gn_all),
                "train/grad_norm_lora": float(gn_lora),
                "train/tokens_per_sec": float(tps),
                "train/step_time_sec": float(step_time),
                "train/gpu_mem_gb": float(mem_gb),
            }
    
            for k, v in _lora_param_norms_by_group(model, top_k=self.lora_top_k).items():
                logs[f"train/lora_param_norm/{k}"] = float(v)
    
            self.log(logs)
    
        return loss.detach() / self.args.gradient_accumulation_steps

# =====================
# 主函数
# =====================

def main():
    # 设置随机种子
    set_seed(SEED)
    
    # 检查GPU可用性并指定使用指定的GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ GPU可用！设备数量: {gpu_count}")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_total = props.total_memory / 1024**3
            print(f"  GPU {i}: {props.name}")
            print(f"    总显存: {memory_total:.2f} GB")
            if i == GPU_ID:
                print(f"    ⭐ 已选择此GPU")
        
        # 检查指定的GPU ID是否有效
        if GPU_ID >= gpu_count:
            print(f"⚠️  警告：指定的GPU {GPU_ID}不存在，只有{gpu_count}块GPU，将使用GPU 0")
            selected_gpu = 0
        else:
            selected_gpu = GPU_ID
        
        # 指定使用选定的GPU
        torch.cuda.set_device(selected_gpu)
        print(f"\n🎯 指定使用GPU {selected_gpu}: cuda:{selected_gpu} ({torch.cuda.get_device_name(selected_gpu)})")
        sys.stdout.flush()
    else:
        print("⚠️  警告：未检测到GPU，将使用CPU训练（速度会很慢）")
        print("   建议使用GPU进行训练")
        selected_gpu = None
        sys.stdout.flush()
    
    # 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载tokenizer
    print(f"正在加载tokenizer: {MODEL_DIR}")
    sys.stdout.flush()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True
    )
    print("✅ Tokenizer加载完成")
    sys.stdout.flush()
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型
    print(f"正在加载模型: {MODEL_DIR}")
    print("⚠️  模型加载可能需要几分钟，请耐心等待...")
    sys.stdout.flush()
    
    # 指定使用选定的GPU
    if torch.cuda.is_available() and selected_gpu is not None:
        device = f"cuda:{selected_gpu}"
        print(f"指定使用GPU: {device} ({torch.cuda.get_device_name(selected_gpu)})")
        sys.stdout.flush()
        print("开始加载模型权重...")
        sys.stdout.flush()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float16 if FP16 else torch.float32,
            device_map={"": device},  # 使用字典格式指定设备，修复device_map参数问题
            trust_remote_code=True
        )
    else:
        device = "cpu"
        print("⚠️  未检测到GPU，使用CPU")
        sys.stdout.flush()
        print("开始加载模型权重...")
        sys.stdout.flush()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float32,  # CPU不支持float16
            device_map="cpu",
            trust_remote_code=True
        )
    
    print("✅ 模型加载完成")
    # 打印模型所在的设备
    print(f"模型已加载到: {next(model.parameters()).device}")
    sys.stdout.flush()
    
    # 启用梯度检查点
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print("已启用梯度检查点 + enable_input_require_grads")
    
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
    print("\n开始加载训练数据...")
    sys.stdout.flush()
    train_dataset = load_dataset(TRAIN_DATA_PATH, tokenizer)
    print("✅ 训练数据加载完成\n")
    sys.stdout.flush()
    
    # 训练参数
    # 兼容不同版本的transformers：4.21.0+使用eval_strategy，旧版本使用evaluation_strategy
    training_args_dict = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "fp16": FP16,
        "logging_steps": LOGGING_STEPS,
        "logging_strategy": "steps",
        "save_steps": SAVE_STEPS,
        "save_total_limit": 3,
        "warmup_steps": WARMUP_STEPS,
    
        # ✅ TensorBoard
        "report_to": ["tensorboard"],
        "logging_dir": str(output_dir / "tb"),
    
        "dataloader_pin_memory": True,
        "save_safetensors": True,
    }
    
    # 根据transformers版本选择正确的参数名
    try:
        # 尝试使用新版本的参数名（4.21.0+）
        training_args = TrainingArguments(**training_args_dict, eval_strategy="no")
    except TypeError:
        # 如果失败，使用旧版本的参数名
        training_args = TrainingArguments(**training_args_dict, evaluation_strategy="no")
    
    # 创建SFTTrainer
    # 不同版本的trl库可能有不同的参数，这里使用兼容的方式
    # 尝试使用新版本的参数（包含tokenizer）
    try:
        trainer = LoRAMonitorSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            max_seq_length=MAX_LENGTH,
            dataset_text_field="text",
        )
    except TypeError:
        # 如果失败，尝试不使用tokenizer参数（某些版本会自动从model获取）
        try:
            trainer = LoRAMonitorSFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                max_seq_length=MAX_LENGTH,
                dataset_text_field="text",
            )
        except TypeError:
            # 如果还是失败，使用最简参数
            trainer = LoRAMonitorSFTTrainer(
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










