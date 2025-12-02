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
MODEL_DIR = "/work/2024/zhulei/intent-driven/qwen3-4b"  # 模型路径
TRAIN_DATA_PATH = "/work/2024/zhulei/intent-driven/train_intent_qwen.json"  # 训练数据路径
OUTPUT_DIR = "/work/2024/zhulei/intent-driven/outputs/qwen3-4b-lora-intent"  # 输出目录

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
    """加载数据集并格式化为文本（让SFTTrainer处理tokenization）"""
    print(f"正在加载数据集: {data_path}")
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"训练数据文件不存在: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"数据集大小: {len(data)} 条")
    
    # 验证数据格式
    if not isinstance(data, list):
        raise ValueError("数据格式错误：应该是列表格式")
    
    if len(data) > 0 and "messages" not in data[0]:
        raise ValueError("数据格式错误：每个条目应包含'messages'字段")
    
    # 使用tokenizer的apply_chat_template格式化对话为文本
    # SFTTrainer会自动处理tokenization，所以这里只格式化，不tokenize
    formatted_data = []
    for idx, item in enumerate(data):
        try:
            if "messages" not in item:
                print(f"警告：第 {idx+1} 条数据缺少'messages'字段，已跳过")
                continue
            
            # 使用apply_chat_template将messages格式化为文本
            # tokenize=False 表示只格式化，不进行tokenization
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
    
    # 转换为Dataset格式，字段名为"text"
    # SFTTrainer会读取这个"text"字段并进行tokenization
    dataset = Dataset.from_list(formatted_data)
    
    return dataset

# =====================
# 主函数
# =====================

def main():
    # 设置随机种子
    set_seed(SEED)
    
    # 检查GPU可用性并指定使用第一块GPU
    if torch.cuda.is_available():
        print(f"✅ GPU可用！设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        # 指定使用第一块GPU (cuda:0)
        torch.cuda.set_device(0)
        print(f"\n🎯 指定使用第一块GPU: cuda:0 ({torch.cuda.get_device_name(0)})")
    else:
        print("⚠️  警告：未检测到GPU，将使用CPU训练（速度会很慢）")
        print("   建议使用GPU进行训练")
    
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
    # 指定使用第一块GPU (cuda:0)
    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"指定使用GPU: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        print("⚠️  未检测到GPU，使用CPU")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16 if FP16 else torch.float32,
        device_map=device,  # 明确指定使用第一块GPU
        trust_remote_code=True
    )
    
    # 打印模型所在的设备
    print(f"模型已加载到: {next(model.parameters()).device}")
    
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
        "save_steps": SAVE_STEPS,
        "save_total_limit": 3,  # 只保留最近3个检查点
        "warmup_steps": WARMUP_STEPS,
        "report_to": "tensorboard" if os.path.exists("tensorboard") else None,
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
        trainer = SFTTrainer(
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
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                max_seq_length=MAX_LENGTH,
                dataset_text_field="text",
            )
        except TypeError:
            # 如果还是失败，使用最简参数
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

