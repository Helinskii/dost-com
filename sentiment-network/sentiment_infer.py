import json, torch
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TextClassificationPipeline,
)
import numpy as np

STAGE2_DIR = Path("mobilebert-uncased_full_128_stage2")
CKPT = max(STAGE2_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
MODEL_ID = "google/mobilebert-uncased"

model     = AutoModelForSequenceClassification.from_pretrained(CKPT)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
if list(model.config.id2label.values())[0].startswith("LABEL"):
    model.config.id2label = {i: lab for i, lab in enumerate(EMOTIONS)}
    model.config.label2id = {lab: i for i, lab in enumerate(EMOTIONS)}

pipe = TextClassificationPipeline(
    model=model.to("cuda" if torch.cuda.is_available() else "cpu"),
    tokenizer=tokenizer,
    function_to_apply="softmax",
    return_all_scores=True,
    batch_size=32,
)

def classify(text: str) -> dict:
    return {r["label"]: round(r["score"], 4) for r in pipe(text)[0]}

def predict_emotion(text: str):
    # DEBUG
    # value = pipe(text)
    # print(value)

    probs = np.array([round(r["score"], 4) for r in pipe(text)[0]])
    # print(probs)
    prediction = np.argmax(probs)
    # print(EMOTIONS[int(prediction)])
    return EMOTIONS[int(prediction)], classify(text)

## DEPRECATED AS THIS IS BEING ACHIEVED IN SENTIMENT_ANALYSIS.PY
# def predict_compl_emotion(messages):
    
#     final_res = []
#     text_emotion = []
#     emotion_score = []
#     for msg in messages:
#         msg_emotion, _ = predict_emotion(msg)
#         final_res.append(msg_emotion)
#         text_emotion.append({msg: msg_emotion})

#     emotion_counts = Counter(final_res)
#     all_emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'unknown']
#     emotion_counts = {e: emotion_counts.get(e, 0) for e in all_emotions}
#     total = sum(emotion_counts.values())
#     emotion_score = {e: round((count / total), 2) for e, count in emotion_counts.items()}

#     return emotion_score, text_emotion

if __name__ == "__main__":
    prompt = "Ugh, the app keeps crashing, I'm furious."
    # print(json.dumps({"query": prompt, "response": classify(prompt)}, indent=2))
    emo, probs = predict_emotion(prompt)
    print(f"Emotion = {emo}\tProbabilities = {probs}")


#### DEPRECATED INFERENCING MECHANISM
# import pandas as pd
# import torch
# from transformers import MobileBertTokenizerFast, MobileBertForSequenceClassification, AutoModelForSequenceClassification, DistilBertForSequenceClassification
# from collections import Counter

# import warnings
# from urllib3.exceptions import NotOpenSSLWarning
# warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# import numpy as np

# # model_path = "TinyBERT_General_4L_312D_full_128/checkpoint-3000"
# model_path = "mobilebert_emotion_token128"
# # model_path = "distilbert-base-uncased_full_128/checkpoint-2000"

# tokenizer = MobileBertTokenizerFast.from_pretrained("google/mobilebert-uncased")

# model = MobileBertForSequenceClassification.from_pretrained(model_path, num_labels=6)

# model.eval()

# label_list = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# def predict_emotion(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         # DEBUG
#         print(f"Logits = {logits}")
#         logits_array = logits[0]
#         sig_logits = torch.sigmoid(logits_array)
#         # DEBUG
#         print(f"Sigmoid(logits) = {sig_logits}")
#         prediction = torch.argmax(logits, dim=1).item()
#     return label_list[int(prediction)]

# # TEST DATA
# text = "Ugh, the app keeps crashing, I'm furious."
# emo = predict_emotion(text)
# print(f"Emotion = {emo}")