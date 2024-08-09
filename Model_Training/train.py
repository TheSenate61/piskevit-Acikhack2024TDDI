import pandas as pd
import json
from datasets import Dataset
import os

alpaca_prompt = """Aşağıda, bir görevi tanımlayan bir talimat ile daha fazla bağlam sağlayan bir girdi bulunmaktadır. İsteği uygun şekilde tamamlayan bir yanıt yazın.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = "<|endoftext|>"  # Adjust if your tokenizer uses a different EOS token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


def process_csv(file_path):
    df = pd.read_csv(file_path)
    records = []
    for _, row in df.iterrows():
        instruction = row['prompt']
        if row['is_response_0_safe'] and not row['is_response_1_safe']:
            response = row['response_0']
        elif row['is_response_1_safe'] and not row['is_response_0_safe']:
            response = row['response_1']
        else:
            response = row['response_1'] if row['safer_response_id'] == 1 else row['response_0']
        records.append({"instruction": instruction, "input": "", "output": response})
    return records

def process_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = []
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                for item in value:
                    instruction_key, response_key = item.keys()
                    instruction = item[instruction_key]
                    response = item[response_key]
                    records.append({"instruction": instruction, "input": "", "output": response})
    elif isinstance(data, list):
        for item in data:
            instruction_key, response_key = item.keys()
            instruction = item[instruction_key]
            response = item[response_key]
            records.append({"instruction": instruction, "input": "", "output": response})
    return records

def load_and_process_data(folder_path):
    all_records = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.csv'):
            all_records.extend(process_csv(file_path))
        elif file_name.endswith('.json'):
            all_records.extend(process_json(file_path))
    return all_records

data_folder = "/content/drive/MyDrive/eğitimdata"


all_records = load_and_process_data(data_folder)

dataset = Dataset.from_pandas(pd.DataFrame(all_records))
dataset = dataset.map(formatting_prompts_func, batched=True)

from datasets import Dataset
import pandas as pd
import os
import json
from datasets import Dataset, DatasetDict
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
from transformers import TrainingArguments
from trl import DPOTrainer
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "Koç",
    ),
)
trainer.train()
