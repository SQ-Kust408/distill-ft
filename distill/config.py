# ======================
# config.py
# ======================
from transformers import TrainingArguments

# 训练参数配置
distill_args = TrainingArguments(
    output_dir="./distilled_model",
    per_device_train_batch_size=1,  # 根据显存调整
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=50,
    fp16=False,  # 禁用混合精度训练
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=2000,  # 修改为 200 的整数倍
    load_best_model_at_end=True,
    report_to="none"
)

# 模型配置
teacher_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
student_model_name = "Qwen/Qwen3-0.6B"

# 数据配置
max_seq_length = 128
temperature = 1.0
distill_loss_weight = 0.7