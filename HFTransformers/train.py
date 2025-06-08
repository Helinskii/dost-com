import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import DistilBertTokenizerFast, MobileBertTokenizerFast
from transformers import DistilBertForSequenceClassification, MobileBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import torch
# import os
# os.environ["WANDB_DISABLED"] = "true"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_df = pd.read_csv("training.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

train_data = Dataset.from_pandas(train_df)
val_data = Dataset.from_pandas(val_df)
test_data = Dataset.from_pandas(test_df)

text_dataset = DatasetDict({
    "train": train_data,
    "validation": val_data,
    "test": test_data
})
print(text_dataset)

tokenizer = MobileBertTokenizerFast.from_pretrained("google/mobilebert-uncased")
# tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
labels = sorted(train_df["label"].unique())
label2id = {label: i for i, label in enumerate(labels)}

def tokenize_and_encode(batch):
    tokens = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = label2id[batch["label"]]
    return tokens

dataset = text_dataset.map(tokenize_and_encode)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
model = MobileBertForSequenceClassification.from_pretrained(
    "google/mobilebert-uncased", num_labels=len(labels)
)
# model = DistilBertForSequenceClassification.from_pretrained(
#     "distilbert-base-uncased", num_labels=len(labels)
# )

def compute_metrics(pred):
    logits, labels = pred
    preds = torch.argmax(torch.tensor(logits), axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

training_args = TrainingArguments(
    output_dir="mobilebert_emotion_token128",
    eval_strategy ="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs_mobilebert_token128",
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics
)

trainer.train()
