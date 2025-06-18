import pandas as pd
import torch
from transformers import MobileBertTokenizerFast, MobileBertForSequenceClassification, AutoModelForSequenceClassification, DistilBertForSequenceClassification
from collections import Counter

import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# model_path = "TinyBERT_General_4L_312D_full_128/checkpoint-3000"
model_path = "mobilebert_emotion_token128"
# model_path = "distilbert-base-uncased_full_128/checkpoint-2000"

tokenizer = MobileBertTokenizerFast.from_pretrained("google/mobilebert-uncased")

model = MobileBertForSequenceClassification.from_pretrained(model_path, num_labels=6)

model.eval()

label_list = ['sadness', 'joy', 'love', 'anger', 'fear', 'unknown']

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return label_list[int(prediction)]

def predict_compl_emotion(messages):
    
    final_res = []
    text_emotion = []
    emotion_score = []
    for msg in messages:
        msg_emotion = predict_emotion(msg)
        final_res.append(msg_emotion)
        text_emotion.append({msg: msg_emotion})

    emotion_counts = Counter(final_res)
    all_emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'unknown']
    emotion_counts = {e: emotion_counts.get(e, 0) for e in all_emotions}
    total = sum(emotion_counts.values())
    emotion_score = {e: round((count / total), 2) for e, count in emotion_counts.items()}

    return emotion_score, text_emotion