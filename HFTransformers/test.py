import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, MobileBertTokenizerFast, MobileBertForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_path = "mobilebert_emotion_token128/checkpoint-3000"
# model_path = "distilbert_emotion_token64/checkpoint-3000"

tokenizer = MobileBertTokenizerFast.from_pretrained("google/mobilebert-uncased")
# tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

model = MobileBertForSequenceClassification.from_pretrained(model_path)
# model = DistilBertForSequenceClassification.from_pretrained(model_path)

model.eval()

test_df = pd.read_csv("test.csv")
texts = test_df["text"].tolist()
true_labels = test_df["label"].tolist()

label_list = ['sadness', 'joy', 'love', 'anger', 'fear', 'unknown']
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {v: k for k, v in label2id.items()}

predicted_labels = []
predicted_id = []

for text in tqdm(texts, desc="Evaluating"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
        predicted_id.append(pred)
        predicted_labels.append(id2label[pred])

print("Accuracy:", accuracy_score(true_labels, predicted_id))
print("F1-score:", f1_score(true_labels, predicted_id, average="weighted"))
print("\nClassification Report:\n", classification_report(true_labels, predicted_id))
