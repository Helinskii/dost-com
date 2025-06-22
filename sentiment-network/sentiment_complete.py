'''
Script to distill sentiments detected for each message in the chat
to multiple analytics - overall emotion/sentiment, per user, contextual
'''

import pandas as pd

df = pd.read_csv("morrie_mitch_emotions.csv")
emotion_counts = df['emotion'].value_counts().to_dict()

print(emotion_counts)

all_emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'unknown']
emotion_counts = {e: emotion_counts.get(e, 0) for e in all_emotions}

total = sum(emotion_counts.values())

emotion_score = {e: count / total for e, count in emotion_counts.items()}

print("Emotion scores:")
for emotion, score in emotion_score.items():
    print(f"{emotion}: {score:.3f}")