import torch
import torch.nn.functional as F
import pandas as pd
from datasets import Dataset, DatasetDict
import numpy as np
from transformers import (
    BartTokenizer, BartForConditionalGeneration,
	T5Tokenizer, T5ForConditionalGeneration,
    DistilBertTokenizer, DistilBertForSequenceClassification
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer

# Load dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

df1 = pd.read_csv("massive_emotion_dataset.csv").sample(250, replace=False)
df2 = pd.read_csv("emotion_paraphrase_dataset_1.csv")
df3 = pd.read_csv("emotion_paraphrase_dataset_2.csv")
df4 = pd.read_csv("emotion_paraphrase_dataset_3.csv")

df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# # train_df = pd.read_csv("training.csv")
# val_df = pd.read_csv("seq2seq.csv")
# test_df = pd.read_csv("seq2seq.csv")

data = Dataset.from_pandas(df)
# train_data = Dataset.from_pandas(train_df)
# val_data = Dataset.from_pandas(val_df)
# test_data = Dataset.from_pandas(test_df)

dataset = Dataset.from_pandas(df)
train_test_split = dataset.train_test_split(test_size=0.4, seed=42)
val_test_split = train_test_split["test"].train_test_split(test_size=0.5, seed=42)

text_dataset = DatasetDict({
    "train": dataset,
    # "train": train_test_split["train"],
    "validation": val_test_split["train"],
    "test": val_test_split["test"]
})

# s2s_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-small")
s2s_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

def preprocess(example):
    model_inputs = s2s_tokenizer(["paraphrase: " + x for x in example["original"]], truncation=True, padding="max_length", max_length=128)
    # model_inputs = s2s_tokenizer(example["original"], truncation=True, padding="max_length", max_length=128)
    # print(model_inputs)
    # model_inputs["labels"] = np.random.choice([1, 2], size=len(example["label"]))
    model_inputs["labels"] = s2s_tokenizer(example["paraphrased"], truncation=True, padding="max_length", max_length=128)['input_ids']

    return model_inputs

tokenized = text_dataset.map(preprocess, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# seq2seq_model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-small")
seq2seq_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
# seq2seq_model = BartForConditionalGeneration.from_pretrained("./results_seq2seq_BART_def_0622/checkpoint-100")

seq2seq_model = seq2seq_model.to(device)

training_args = TrainingArguments(
    output_dir="./results_seq2seq_BART_def_0623",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="steps",
    save_steps=100,
    learning_rate=5e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    # fp16=True,
    logging_steps=10,
)

trainer = Trainer(
    model=seq2seq_model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=s2s_tokenizer,
)

trainer.train()
