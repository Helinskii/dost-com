import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, MobileBertTokenizerFast, MobileBertForSequenceClassification

model_path = "mobilebert_emotion_token128/checkpoint-3000"
# model_path = "distilbert_emotion_token64/checkpoint-3000"

tokenizer = MobileBertTokenizerFast.from_pretrained("google/mobilebert-uncased")
# tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

model = MobileBertForSequenceClassification.from_pretrained(model_path)
# model = DistilBertForSequenceClassification.from_pretrained(model_path)

model.eval()

label_list = ['sadness', 'joy', 'love', 'anger', 'fear', 'unknown']

test_df = pd.read_csv("test.csv")
texts = test_df["text"].tolist()
true_labels = test_df["label"].tolist()

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    # print(text)
    # print(inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return label_list[prediction]

# for text, label in zip(texts, true_labels):
#     print(text)
#     print('label: ', label_list[label], 'pred: ', predict_emotion(text))

print(predict_emotion("I can't believe how amazing this is!"))
print(predict_emotion("I'm feeling really down today."))
print(predict_emotion("This is okay, nothing special."))
