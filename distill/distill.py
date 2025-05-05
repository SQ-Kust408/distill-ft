# ======================
# distill.py
# ======================
import torch
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForSeq2Seq,
    logging
)
from data_preprocessing import load_and_preprocess_data
from config import (
    distill_args,
    teacher_model_name,
    student_model_name,
    max_seq_length,
    temperature,
    distill_loss_weight
)

# 禁用冗余日志
logging.set_verbosity_error()


class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        student_logits = outputs.logits  # [batch_size, seq_length, vocab_size]

        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            teacher_logits = teacher_outputs.logits

        # 硬标签损失（核心修正：对齐序列长度）
        student_logits = student_logits[:, :-1, :]  # 去掉最后一个时间步，形状 [batch_size, seq_length-1, vocab_size]
        shift_labels = inputs["labels"][:, 1:]  # 去掉第一个时间步，形状 [batch_size, seq_length-1]
        ce_loss = torch.nn.functional.cross_entropy(
            # 使用 reshape 替代 view
            student_logits.reshape(-1, student_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100
        )

        # 软标签损失（沿用原逻辑，但确保维度对齐）
        mask = (shift_labels != -100).unsqueeze(-1)
        valid_teacher_logits = teacher_logits[:, :-1, :].masked_select(mask).reshape(-1, teacher_logits.size(-1))
        valid_student_logits = student_logits.masked_select(mask).reshape(-1, student_logits.size(-1))

        teacher_probs = torch.nn.functional.softmax(valid_teacher_logits / temperature, dim=-1)
        student_probs = torch.nn.functional.softmax(valid_student_logits / temperature, dim=-1)
        kl_loss = torch.nn.functional.kl_div(
            student_probs.log(),
            teacher_probs,
            reduction="batchmean"
        )
        # 组合损失
        total_loss = ce_loss + distill_loss_weight * kl_loss

        return (total_loss, outputs) if return_outputs else total_loss


def main():
    # 加载数据和分词器
    tokenized_dataset, tokenizer = load_and_preprocess_data()

    # 初始化模型
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        device_map="auto",
        torch_dtype=torch.float16
    ).eval()

    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 数据整理器
    # distill.py（数据整理器配置）
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        max_length=max_seq_length,
        pad_to_multiple_of=None,  # 暂时关闭倍数填充，调试阶段使用
        label_pad_token_id=-100
    )

    # 初始化Trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,  # 添加教师模型引用
        model=student_model,
        args=distill_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # 启动训练
    trainer.train()


if __name__ == "__main__":
    main()