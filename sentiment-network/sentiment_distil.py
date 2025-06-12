import re
from sentiment_infer import predict_emotion
from tqdm import tqdm
import pandas as pd

results = []

with open("morrie_mitch_conversation.md", "r") as f:
    for line in tqdm(f):
        match = re.match(r"\*\*(.+?):\*\* (.+)", line.strip())
        if match:
            speaker = match.group(1)
            text = match.group(2)
            emotion = predict_emotion(text)
            results.append({"speaker": speaker, "text": text, "emotion": emotion})

df = pd.DataFrame(results)
df.to_csv("morrie_mitch_emotions.csv", index = False)

