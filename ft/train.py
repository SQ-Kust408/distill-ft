# train.py（保持文件调用）
from transformers import AutoModelForCausalLM, Trainer, DataCollatorForSeq2Seq
from data_preprocessing import load_and_preprocess_data  # 明确调用预处理模块
from config import training_args, model_name, max_length


def main():
    # 加载预处理后的数据
    tokenized_dataset, tokenizer = load_and_preprocess_data()

    # 验证数据格式
    sample = tokenized_dataset["train"][0]
    print(f"验证输入长度: {len(sample['input_ids'])}, 标签长度: {len(sample['labels'])}")

    # 初始化模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_cache=False  # 必须关闭缓存
    )

    # 配置动态填充collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding="longest",  # 动态填充到批次内最大长度
        max_length=max_length,
        pad_to_multiple_of=8  # 优化GPU显存
    )

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator
    )

    # 启动训练
    trainer.train()


if __name__ == "__main__":
    main()