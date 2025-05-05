# ======================
# data_preprocessing.py
# ======================
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from config import max_seq_length, teacher_model_name


def load_and_preprocess_data():
    # 加载数据集
    dataset = load_dataset("json", data_files={
        "train": "train.json",
        "valid": "valid.json",
        "test": "test.json"
    })

    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(
        teacher_model_name,
        use_fast=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 预处理函数
    def process_examples(examples):
        # 直接使用 examples["input"] 中的字符串构建提示
        prompts = [f"### 输入：{input_text}\n### 故事：" for input_text in examples["input"]]
        full_texts = [p + s for p, s in zip(prompts, examples["story"])]

        # 动态填充至最长序列，确保输入和标签长度一致
        tokenized = tokenizer(
            full_texts,
            max_length=max_seq_length,
            truncation=True,
            padding="longest",
            return_attention_mask=True,
            add_special_tokens=True
        )

        # 通过 prompt 结束位置生成标签（假设 prompt 以固定格式结尾）
        labels = []
        prompt_end_token = tokenizer.encode("### 故事：", add_special_tokens=False)[-1]
        for input_ids in tokenized["input_ids"]:
            if prompt_end_token in input_ids:
                prompt_end = input_ids.index(prompt_end_token) + 1  # +1 跳过分隔符
                label = [-100] * prompt_end + input_ids[prompt_end:]
            else:
                # 如果 prompt_end_token 不存在，将整个 input_ids 作为标签
                label = input_ids
            labels.append(label)

        tokenized["labels"] = labels
        for label in labels:
            if all(l == -100 for l in label):
                print("Warning: Found label all -100!")
        return tokenized

    # 处理数据集
    tokenized_dataset = dataset.map(
        process_examples,
        batched=True,
        batch_size=256,
        remove_columns=dataset["train"].column_names
    )
    # 在 data_preprocessing.py 中添加
    sample = tokenized_dataset["train"][0]
    print(f"Input length: {len(sample['input_ids'])}, Label length: {len(sample['labels'])}")

    return tokenized_dataset, tokenizer