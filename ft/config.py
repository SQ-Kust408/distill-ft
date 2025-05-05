# config.py
from transformers import TrainingArguments

# 训练参数配置
training_args = TrainingArguments(
    output_dir="deepseek-r1-1.5b-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    warmup_steps=100,
    num_train_epochs=3,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    learning_rate=2e-5
)
model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
# 数据处理参数
max_length = 512
