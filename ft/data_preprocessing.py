# data_preprocessing.py（保持文件独立）
from datasets import load_dataset
from transformers import AutoTokenizer
from config import max_length, model_name


def load_and_preprocess_data():
    # 加载原始数据
    dataset = load_dataset("json", data_files={
        "train": "train.json",
        "validation": "valid.json",
        "test": "test.json"
    })

    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 统一处理函数
    def process(examples):
        # 构造prompt
        prompts = [f"根据以下信息生成故事：{input_}" for input_ in examples["input"]]
        labels = examples["story"]

        # 联合编码
        model_inputs = tokenizer(
            text=prompts,
            text_target=labels,  # 关键参数
            truncation=True,
            max_length=max_length,
            padding="max_length"  # 固定长度填充
        )

        # 创建因果掩码
        input_ids = model_inputs["input_ids"]
        labels = model_inputs["labels"]
        for i in range(len(labels)):
            # 找到输入部分的结束位置（第一个pad_token）
            context_len = input_ids[i].index(tokenizer.pad_token) if tokenizer.pad_token in input_ids[i] else len(
                input_ids[i])
            # 将输入部分的label设为-100（忽略loss计算）
            labels[i] = [-100] * context_len + labels[i][context_len:]

        return model_inputs

    # 应用处理
    tokenized_dataset = dataset.map(process, batched=True, batch_size=256)
    return tokenized_dataset, tokenizer